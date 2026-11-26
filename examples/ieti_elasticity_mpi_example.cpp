/** @file ieti_elasticity_mpi_example.cpp

    @brief MPI-parallel IETI-DP solver for linear elasticity on a multipatch domain

    The example solves linear elasticity (Hooke's law, Young's modulus and
    Poisson ratio as options) on a box that is split into Nx x Ny (x Nz)
    patches of tensor B-splines, or on a multipatch read from an xml file.
    The face with minimal x-coordinate is clamped and a constant traction acts
    on the face with maximal x-coordinate.

    The discretized problem is solved with a dual-primal tearing and
    interconnecting method (IETI-DP) that is built from the classes of
    gsIeti: gsIetiMapper provides jump matrices and primal constraints
    (vertex values and, optionally, edge and face averages),
    gsPrimalSystem provides the primal basis, gsScaledDirichletPrec provides
    the scaled Dirichlet preconditioner and gsConjugateGradient solves the
    Schur complement system in the Lagrange multipliers.

    Parallelization. The patches are distributed over the MPI ranks. Each rank
    assembles and factorizes only its own patches. The vector of Lagrange
    multipliers is replicated on all ranks. Every application of the Schur
    complement or of the preconditioner evaluates the contributions of the
    local patches and sums them over the ranks with MPI_Allreduce. The primal
    problem is replicated: the contributions to the coarse matrix are gathered
    on all ranks and factorized there. Because the conjugate gradient
    iteration only acts on replicated vectors, gsConjugateGradient is used
    unchanged.

    The program checks its own result and exits with a non-zero code if a
    check fails (option --check compares with the serial IETI-DP implementation
    of the library classes, option --check-direct with a direct solve of the
    assembled global system by gsElasticityAssembler).

    The result is printed as one line of key=value fields.

    This file is part of the G+Smo library.

    This Source Code Form is subject to the terms of the Mozilla Public
    License, v. 2.0. If a copy of the MPL was not distributed with this
    file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#include <gismo.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <numeric>
#include <sstream>

#ifdef _OPENMP
#include <omp.h>
#endif

#if defined(__has_include)
#  if __has_include(<gsElasticity/src/gsElasticityAssembler.h>)
#    include <gsElasticity/src/gsElasticityAssembler.h>
#    define IETI_HAVE_ELASTICITY 1
#  endif
#endif

using namespace gismo;

namespace {

typedef gsSparseMatrix<real_t,RowMajor> JumpMatrix;
typedef gsSparseMatrix<real_t>          SparseMatrix;
typedef gsSparseVector<real_t>          SparseVector;
typedef gsMatrix<real_t>                Matrix;
typedef gsLinearOperator<real_t>        Op;
typedef Op::Ptr                         OpPtr;

double wallTime()
{
#ifdef GISMO_WITH_MPI
    return MPI_Wtime();
#else
    return std::chrono::duration<double>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
#endif
}

/// Communicator wrapper. The reductions are timed; the time includes the
/// waiting for the slowest rank.
struct Par
{
    int    rank;
    int    size;
    double commTime;
    index_t commCalls;
    double commBytes;
#ifdef GISMO_WITH_MPI
    MPI_Comm comm;
    explicit Par(MPI_Comm c) : comm(c), commTime(0), commCalls(0), commBytes(0)
    { MPI_Comm_rank(c,&rank); MPI_Comm_size(c,&size); }

    void sum(Matrix& m)
    {
        if (size==1) return;
        const double t = MPI_Wtime();
        MPI_Allreduce(MPI_IN_PLACE, m.data(), (int)m.size(), MPI_DOUBLE, MPI_SUM, comm);
        commTime  += MPI_Wtime() - t;
        commCalls += 1;
        commBytes += 8.0*m.size();
    }
    double reduce(double v, MPI_Op op) const
    {
        double r = v;
        if (size>1) MPI_Allreduce(&v, &r, 1, MPI_DOUBLE, op, comm);
        return r;
    }
    double maxOf(double v) const { return reduce(v, MPI_MAX); }
    double sumOf(double v) const { return reduce(v, MPI_SUM); }
    index_t minOf(index_t v) const
    {
        long long a = v, r = v;
        if (size>1) MPI_Allreduce(&a, &r, 1, MPI_LONG_LONG, MPI_MIN, comm);
        return (index_t)r;
    }
    index_t maxOf(index_t v) const
    {
        long long a = v, r = v;
        if (size>1) MPI_Allreduce(&a, &r, 1, MPI_LONG_LONG, MPI_MAX, comm);
        return (index_t)r;
    }
    template<class T> std::vector<T> allgatherv(const std::vector<T>& in, MPI_Datatype dt) const
    {
        if (size==1) return in;
        std::vector<int> cnt(size), dsp(size);
        int n = (int)in.size();
        MPI_Allgather(&n, 1, MPI_INT, cnt.data(), 1, MPI_INT, comm);
        int tot = 0;
        for (int i=0; i<size; ++i) { dsp[i] = tot; tot += cnt[i]; }
        std::vector<T> out(tot);
        MPI_Allgatherv(in.data(), n, dt, out.data(), cnt.data(), dsp.data(), dt, comm);
        return out;
    }
    static MPI_Datatype dtype(const int*)    { return MPI_INT;    }
    static MPI_Datatype dtype(const double*) { return MPI_DOUBLE; }
#else
    Par() : rank(0), size(1), commTime(0), commCalls(0), commBytes(0) {}
    void   sum(Matrix&) {}
    double maxOf(double v) const { return v; }
    double sumOf(double v) const { return v; }
    index_t minOf(index_t v) const { return v; }
    index_t maxOf(index_t v) const { return v; }
    template<class T> std::vector<T> allgatherv(const std::vector<T>& in, int) const { return in; }
    static int dtype(const int*)    { return 0; }
    static int dtype(const double*) { return 0; }
#endif
    template<class T> std::vector<T> gather(const std::vector<T>& in) const
    { return allgatherv(in, dtype((const T*)NULL)); }
};

//---------------------------------------------------------------------------
// Problem definition (replicated on all ranks)
//---------------------------------------------------------------------------

struct Options
{
    index_t dim;
    std::string patches;
    std::string xml;
    index_t degree, refine;
    real_t  young, poisson;
    std::string traction;
    std::string primal;        // v, ve, vef
    bool    eliminateVertices;
    std::string localSolver;   // lu, ldlt
    std::string partition;     // lex, morton
    real_t  tol;
    index_t maxIt;
    bool    check, checkDirect, fastDiag, quiet;
    std::string tag;
    real_t  gateMult, gateL2, gateComp, gateDirectL2, gateDirectComp;
};

struct Problem
{
    index_t dim;
    gsMultiPatch<>                   mp;
    gsMultiBasis<>                   mb;
    std::vector<gsBoundaryConditions<> > bc;
    std::vector<gsIetiMapper<> >     mapper;
    index_t nPatches, nLambda, nPrimal, nUnknowns;
    index_t grid[3];                    // patches per direction, 0 if unknown
    real_t  lambda, mu;
    std::vector<real_t> traction;
    std::vector<patchSide> clampSides, tractionSides;
    gsConstantFunction<> zero;
};

std::vector<index_t> parseInts(const std::string& s)
{
    std::vector<index_t> r;
    std::stringstream ss(s);
    std::string tok;
    while (std::getline(ss, tok, ',')) r.push_back(atoi(tok.c_str()));
    return r;
}

std::vector<real_t> parseReals(const std::string& s)
{
    std::vector<real_t> r;
    std::stringstream ss(s);
    std::string tok;
    while (std::getline(ss, tok, ',')) r.push_back(atof(tok.c_str()));
    return r;
}

/// Physical position of the midpoint of a boundary side
gsMatrix<> sideMidpoint(const gsMultiPatch<>& mp, const patchSide& ps)
{
    const index_t d = mp.parDim();
    gsMatrix<> supp = mp[ps.patch].support();
    gsMatrix<> pt(d,1), x;
    for (index_t i=0; i<d; ++i)
        pt(i,0) = 0.5*(supp(i,0)+supp(i,1));
    pt(ps.direction(),0) = ps.parameter() ? supp(ps.direction(),1) : supp(ps.direction(),0);
    mp[ps.patch].eval_into(pt, x);
    return x;
}

template<typename Iterator>
typename std::iterator_traits<Iterator>::value_type blockDiagonal( Iterator begin, Iterator end )
{
    typedef typename std::iterator_traits<Iterator>::value_type sparseMatrix;
    typedef typename sparseMatrix::Scalar T;
    typedef typename sparseMatrix::Index Index;
    Index nz = 0;
    for (Iterator it=begin; it!=end; ++it) nz += it->nonZeros();
    gsSparseEntries<T> se;
    se.reserve(nz);
    Index rows = 0, cols = 0;
    for (Iterator sm_it=begin; sm_it!=end; ++sm_it)
    {
        const sparseMatrix& sm = *sm_it;
        for(Index j=0; j<sm.outerSize(); ++j)
            for(typename sparseMatrix::InnerIterator it(sm,j); it; ++it)
                se.add(rows+it.row(), cols+it.col(), it.value());
        rows += sm.rows();
        cols += sm.cols();
    }
    sparseMatrix result(rows, cols);
    result.setFrom(se);
    return result;
}

void buildProblem(const Options& o, Problem& P)
{
    P.grid[0] = P.grid[1] = P.grid[2] = 0;
    if (!o.xml.empty())
    {
        gsMultiPatch<>::uPtr in = gsReadFile<>(o.xml);
        GISMO_ENSURE(in, "Cannot read multipatch from " << o.xml);
        P.mp = *in;
        P.dim = P.mp.geoDim();
    }
    else
    {
        P.dim = o.dim;
        std::vector<index_t> n = parseInts(o.patches);
        n.resize(3, 1);
        if (P.dim==2) n[2] = 1;
        for (index_t i=0; i<3; ++i) P.grid[i] = n[i];
        for (index_t k=0; k<n[2]; ++k)
            for (index_t j=0; j<n[1]; ++j)
                for (index_t i=0; i<n[0]; ++i)
                {
                    if (P.dim==3)
                        P.mp.addPatch( *gsNurbsCreator<>::BSplineCube(1.0, i, j, k) );
                    else
                        P.mp.addPatch( *gsNurbsCreator<>::BSplineSquare(1.0, i, j) );
                }
        P.mp.computeTopology();
    }
    const index_t dim = P.dim;
    P.nPatches = P.mp.nPatches();

    P.lambda = o.young*o.poisson / ((1+o.poisson)*(1-2*o.poisson));
    P.mu     = o.young / (2*(1+o.poisson));
    P.traction = parseReals(o.traction);
    P.traction.resize(dim, 0.0);
    P.zero = gsConstantFunction<>(0.0, dim);

    // Bases: degree o.degree, o.refine uniform refinements
    P.mb = gsMultiBasis<>(P.mp, true);
    for (index_t k=0; k<P.nPatches; ++k)
        P.mb[k].setDegreePreservingMultiplicity(o.degree);
    for (index_t i=0; i<o.refine; ++i)
        P.mb.uniformRefine();

    // Boundary conditions
    real_t xmin = 1e300, xmax = -1e300;
    for (index_t k=0; k<P.nPatches; ++k)
    {
        xmin = std::min(xmin, P.mp[k].coefs().col(0).minCoeff());
        xmax = std::max(xmax, P.mp[k].coefs().col(0).maxCoeff());
    }
    const real_t eps = 1e-8*(xmax-xmin);
    P.bc.assign(dim, gsBoundaryConditions<>());
    for (gsMultiPatch<>::const_biterator it = P.mp.bBegin(); it < P.mp.bEnd(); ++it)
    {
        const real_t x = sideMidpoint(P.mp, *it)(0,0);
        if (x < xmin + eps)
        {
            P.clampSides.push_back(*it);
            for (index_t r=0; r<dim; ++r)
                P.bc[r].addCondition(*it, condition_type::dirichlet, &P.zero);
        }
        else if (x > xmax - eps)
        {
            P.tractionSides.push_back(*it);
            for (index_t r=0; r<dim; ++r)
                P.bc[r].addCondition(*it, condition_type::neumann, &P.zero);
        }
    }
    GISMO_ENSURE(!P.clampSides.empty(), "No face to clamp found.");
    for (index_t r=0; r<dim; ++r)
        P.bc[r].setGeoMap(P.mp);

    // IETI mappers, one per displacement component
    P.mapper.resize(dim);
    for (index_t r=0; r<dim; ++r)
    {
        gsExprAssembler<> assembler;
        gsExprAssembler<>::space u = assembler.getSpace(P.mb);
        u.setup(P.bc[r], dirichlet::interpolation, 0);
        P.mapper[r].init(P.mb, u.mapper(), u.fixedPart());
    }
    for (index_t r=0; r<dim; ++r)
        P.mapper[r].computeJumpMatrices(true, true);
    for (index_t r=0; r<dim; ++r)
    {
        P.mapper[r].cornersAsPrimals();
        if (o.primal != "v")
            P.mapper[r].interfaceAveragesAsPrimals(P.mp, 1);
        if (o.primal == "vef" && dim==3)
            P.mapper[r].interfaceAveragesAsPrimals(P.mp, 2);
    }
    P.nLambda = P.nPrimal = P.nUnknowns = 0;
    for (index_t r=0; r<dim; ++r)
    {
        P.nLambda   += P.mapper[r].nLagrangeMultipliers();
        P.nPrimal   += P.mapper[r].nPrimalDofs();
        P.nUnknowns += P.mapper[r].dofMapperGlobal().freeSize();
    }
}

//---------------------------------------------------------------------------
// Patch-local data
//---------------------------------------------------------------------------

struct Raw
{
    SparseMatrix              A;
    Matrix                    f;
    JumpMatrix                B;
    std::vector<index_t>      skeleton;
    std::vector<SparseVector> constraints;
    std::vector<index_t>      primalIdx;
};

index_t localSize(const Problem& P, index_t k)
{
    index_t n = 0;
    for (index_t r=0; r<P.dim; ++r) n += P.mapper[r].jumpMatrix(k).cols();
    return n;
}

void assembleRaw(const Problem& P, index_t k, Raw& out)
{
    const index_t dim = P.dim;

    gsMultiPatch<>  mpl = P.mp[k];
    gsMultiBasis<>  mbl(P.mb[k]);
    std::vector<gsBoundaryConditions<> > bcl(dim);
    for (index_t r=0; r<dim; ++r)
    {
        P.bc[r].getConditionsForPatch(k, bcl[r]);
        bcl[r].setGeoMap(mpl);
    }

    typedef gsExprAssembler<>::geometryMap geometryMap;
    typedef gsExprAssembler<>::space       space;

    gsExprAssembler<> A(dim, dim);
    A.setIntegrationElements(mbl);
    geometryMap G = A.getMap(mpl);

    std::vector<std::remove_cv<space>::type> v;
    for (index_t r=0; r<dim; ++r)
    {
        v.push_back(A.getSpace(mbl, 1, r));
        v[r].setup(bcl[r], dirichlet::interpolation, 0);
        P.mapper[r].initFeSpace(v[r], k);
    }
    A.initSystem();

    const real_t lam = P.lambda, mu = P.mu;
    for (index_t r=0; r<dim; ++r)
        for (index_t s=0; s<dim; ++s)
        {
            if (r==s)
            {
                auto e = ( lam * igrad(v[r],G)[r] * igrad(v[r],G)[r].tr()
                         + mu  * ( igrad(v[r],G) * igrad(v[r],G).tr()
                                 + igrad(v[r],G)[r] * igrad(v[r],G)[r].tr() ) ) * meas(G);
                A.assemble(e);
            }
            else
            {
                auto e = ( lam * igrad(v[r],G)[r] * igrad(v[s],G)[s].tr()
                         + mu  * igrad(v[r],G)[s] * igrad(v[s],G)[r].tr() ) * meas(G);
                A.assemble(e);
            }
        }
    for (index_t r=0; r<dim; ++r)
    {
        const real_t t = P.traction[r];
        if (t != 0)
        {
            auto e = t * ( v[r] * nv(G).norm() );
            A.assembleBdr(bcl[r].get("Neumann"), e);
        }
    }

    out.A = A.matrix();
    out.f = A.rhs();

    std::vector<JumpMatrix> jm;
    for (index_t r=0; r<dim; ++r) jm.push_back(P.mapper[r].jumpMatrix(k));
    out.B = blockDiagonal(jm.begin(), jm.end());

    out.skeleton.clear();
    out.constraints.clear();
    out.primalIdx.clear();
    index_t offset = 0, poffset = 0;
    const index_t n = out.A.rows();
    for (index_t r=0; r<dim; ++r)
    {
        const std::vector<index_t> sd = P.mapper[r].skeletonDofs(k);
        for (size_t j=0; j<sd.size(); ++j) out.skeleton.push_back(sd[j]+offset);
        const std::vector<SparseVector>& cond = P.mapper[r].primalConstraints(k);
        for (size_t j=0; j<cond.size(); ++j)
        {
            SparseVector tmp(n);
            for (SparseVector::InnerIterator it(cond[j]); it; ++it)
                tmp[offset + it.row()] = it.value();
            out.constraints.push_back(give(tmp));
        }
        const std::vector<index_t>& ind = P.mapper[r].primalDofIndices(k);
        for (size_t j=0; j<ind.size(); ++j) out.primalIdx.push_back(ind[j]+poffset);
        offset  += P.mapper[r].jumpMatrix(k).cols();
        poffset += P.mapper[r].nPrimalDofs();
    }
    GISMO_ASSERT(offset == n, "Local dimension mismatch on patch " << k);
}

/// Mass matrix of the scalar space on a patch, including all dofs
SparseMatrix assembleMass(const Problem& P, index_t k)
{
    gsMultiPatch<> mpl = P.mp[k];
    gsMultiBasis<> mbl(P.mb[k]);
    gsBoundaryConditions<> nobc;
    nobc.setGeoMap(mpl);
    gsExprAssembler<> A(1,1);
    A.setIntegrationElements(mbl);
    gsExprAssembler<>::geometryMap G = A.getMap(mpl);
    gsExprAssembler<>::space w = A.getSpace(mbl);
    w.setup(nobc, dirichlet::interpolation, 0);
    A.initSystem();
    A.assemble( w * w.tr() * meas(G) );
    return A.matrix();
}

/// Coefficients (patch-local basis index, one column per component) of the
/// displacement from the vector of free local dofs
Matrix localCoefficients(const Problem& P, index_t k, const Matrix& u)
{
    Matrix c;
    index_t offset = 0;
    for (index_t r=0; r<P.dim; ++r)
    {
        const index_t sz = P.mapper[r].jumpMatrix(k).cols();
        Matrix cr = P.mapper[r].incorporateFixedPart(k, u.middleRows(offset, sz));
        if (r==0) c.resize(cr.rows(), P.dim);
        c.col(r) = cr.col(0);
        offset += sz;
    }
    return c;
}

/// Squared L2 norm of the displacement given by the coefficient matrix
real_t squaredL2(const SparseMatrix& M, const Matrix& c)
{
    real_t s = 0;
    for (index_t r=0; r<c.cols(); ++r)
        s += c.col(r).dot( M * c.col(r) );
    return s;
}

//---------------------------------------------------------------------------
// Distribution of the patches
//---------------------------------------------------------------------------

unsigned long mortonKey(unsigned long x, unsigned long y, unsigned long z)
{
    unsigned long key = 0;
    for (int b=0; b<20; ++b)
    {
        key |= ((x>>b)&1ul) << (3*b);
        key |= ((y>>b)&1ul) << (3*b+1);
        key |= ((z>>b)&1ul) << (3*b+2);
    }
    return key;
}

std::vector<index_t> distribute(const Problem& P, const std::string& scheme, int rank, int size)
{
    const index_t N = P.nPatches;
    std::vector<index_t> order(N);
    std::iota(order.begin(), order.end(), 0);
    if (scheme == "morton" && P.grid[0] > 0)
    {
        std::vector<std::pair<unsigned long,index_t> > key(N);
        for (index_t q=0; q<N; ++q)
        {
            const index_t i = q % P.grid[0];
            const index_t j = (q / P.grid[0]) % P.grid[1];
            const index_t k = q / (P.grid[0]*P.grid[1]);
            key[q] = std::make_pair(mortonKey(i,j,k), q);
        }
        std::sort(key.begin(), key.end());
        for (index_t q=0; q<N; ++q) order[q] = key[q].second;
    }
    const index_t lo = (index_t)(((long long)rank*N)/size);
    const index_t hi = (index_t)(((long long)(rank+1)*N)/size);
    return std::vector<index_t>(order.begin()+lo, order.begin()+hi);
}

//---------------------------------------------------------------------------
// Distributed IETI-DP
//---------------------------------------------------------------------------

struct Timers
{
    double mapper, assembly, factor, coarse, rhs, solve, total;
    Timers() : mapper(0), assembly(0), factor(0), coarse(0), rhs(0), solve(0), total(0) {}
};

struct OwnedPatch
{
    index_t k, n;
    JumpMatrix       B;        ///< jump matrix, nLambda x n
    SparseMatrix     E;        ///< embedding of the local unknowns into the saddle point system
    Matrix           Psi;      ///< local primal basis, n x m
    std::vector<index_t> pdof; ///< global primal index of each column of Psi
    Matrix           f;        ///< right-hand side
    OpPtr            solver;   ///< solver for the local saddle point system
};

class IetiMpi
{
public:
    IetiMpi(const Problem& P, Par& par, const Options& o, std::vector<index_t> mine)
    : m_P(P), m_par(par), m_opt(o), m_mine(give(mine)), m_schurOp(*this), m_precOp(*this) {}

    void setup(Timers& t);

    /// Distributed Schur complement of the IETI-DP system
    class Schur : public Op
    {
    public:
        explicit Schur(const IetiMpi& s) : m_s(s) {}
        void apply(const Matrix& x, Matrix& y) const { m_s.applySchur(x, y); }
        index_t rows() const { return m_s.m_P.nLambda; }
        index_t cols() const { return m_s.m_P.nLambda; }
    private:
        const IetiMpi& m_s;
    };

    /// Distributed scaled Dirichlet preconditioner
    class Prec : public Op
    {
    public:
        explicit Prec(const IetiMpi& s) : m_s(s) {}
        void apply(const Matrix& x, Matrix& y) const { m_s.applyPrec(x, y); }
        index_t rows() const { return m_s.m_P.nLambda; }
        index_t cols() const { return m_s.m_P.nLambda; }
    private:
        const IetiMpi& m_s;
    };

    const Matrix& rhs() const { return m_rhs; }
    Schur& schur() { return m_schurOp; }
    Prec&  prec()  { return m_precOp; }
    const std::vector<OwnedPatch>& patches() const { return m_patches; }

    /// Local solutions (free local dofs) of the owned patches
    std::vector<Matrix> reconstruct(const Matrix& lambda) const;

    /// Compliance f^T u summed over all ranks
    real_t compliance(const std::vector<Matrix>& u) const
    {
        Matrix c(1,1);
        c(0,0) = 0;
        for (size_t i=0; i<m_patches.size(); ++i)
            c(0,0) += m_patches[i].f.col(0).dot(u[i].col(0));
        m_par.sum(c);
        return c(0,0);
    }

    index_t nnzFactors() const { return 0; }

private:
    void applySchur(const Matrix& x, Matrix& y) const;
    void applyPrec(const Matrix& x, Matrix& y) const;
    Matrix coarseSolve(const Matrix& g) const
    {
        Matrix out;
        if (m_P.nPrimal > 0) m_coarse->apply(g, out);
        return out;
    }

    const Problem&      m_P;
    Par&                m_par;
    Options             m_opt;
    std::vector<index_t> m_mine;
    std::vector<OwnedPatch> m_patches;
    OpPtr               m_coarse;
    Matrix              m_fc;
    Matrix              m_rhs;
    gsScaledDirichletPrec<> m_precLocal;
    OpPtr               m_precLocalOp;
    Schur               m_schurOp;
    Prec                m_precOp;
    friend class Schur;
    friend class Prec;
};

void IetiMpi::setup(Timers& t)
{
    const index_t nPrimal = m_P.nPrimal;
    std::vector<int>    ti, tj;
    std::vector<double> tv;
    Matrix fc;
    fc.setZero(nPrimal, 1);

    m_patches.clear();
    m_patches.reserve(m_mine.size());
    m_precLocal.reserve(m_mine.size());

    for (size_t q=0; q<m_mine.size(); ++q)
    {
        const index_t k = m_mine[q];
        Raw raw;

        double t0 = wallTime();
        assembleRaw(m_P, k, raw);
        t.assembly += wallTime() - t0;

        t0 = wallTime();
        OwnedPatch p;
        p.k = k;
        p.n = raw.A.rows();
        p.f = raw.f;

        // Scaled Dirichlet preconditioner: Schur complement on the skeleton
        {
            gsScaledDirichletPrec<>::Blocks blocks =
                gsScaledDirichletPrec<>::matrixBlocks(raw.A, raw.skeleton);
            m_precLocal.addSubdomain(
                gsScaledDirichletPrec<>::restrictJumpMatrix(raw.B, raw.skeleton).moveToPtr(),
                gsScaledDirichletPrec<>::schurComplement(blocks, makeSparseCholeskySolver(blocks.A11))
            );
        }

        // Primal constraints, local saddle point system, primal basis
        SparseMatrix modA, embBasis;
        Matrix rhsBasis;
        gsPrimalSystem<>::incorporateConstraints(raw.constraints, m_opt.eliminateVertices,
            raw.A, modA, p.E, embBasis, rhsBasis);
        if (m_opt.localSolver == "ldlt")
            p.solver = makeSparseCholeskySolver(modA);
        else
            p.solver = makeSparseLUSolver(modA);
        SparseMatrix Psi = gsPrimalSystem<>::primalBasis(
            p.solver, embBasis, rhsBasis, raw.primalIdx, nPrimal);

        const index_t m = raw.primalIdx.size();
        p.pdof = raw.primalIdx;
        p.Psi.setZero(p.n, m);
        for (index_t j=0; j<m; ++j)
            for (SparseMatrix::InnerIterator it(Psi, p.pdof[j]); it; ++it)
                p.Psi(it.row(), j) = it.value();

        // Contribution to the coarse problem: Psi^T A Psi and Psi^T f
        {
            Matrix APsi = raw.A * p.Psi;
            Matrix Ac = p.Psi.transpose() * APsi;
            Matrix fl = p.Psi.transpose() * raw.f;
            for (index_t a=0; a<m; ++a)
            {
                fc(p.pdof[a],0) += fl(a,0);
                for (index_t b=0; b<m; ++b)
                {
                    ti.push_back((int)p.pdof[a]);
                    tj.push_back((int)p.pdof[b]);
                    tv.push_back(Ac(a,b));
                }
            }
        }
        p.B = give(raw.B);
        m_patches.push_back(give(p));
        t.factor += wallTime() - t0;
    }

    // Coarse problem, replicated on all ranks
    double t0 = wallTime();
    if (nPrimal > 0)
    {
        std::vector<int>    gi = m_par.gather(ti);
        std::vector<int>    gj = m_par.gather(tj);
        std::vector<double> gv = m_par.gather(tv);
        gsSparseEntries<real_t> se;
        se.reserve(gv.size());
        for (size_t i=0; i<gv.size(); ++i)
            se.add(gi[i], gj[i], gv[i]);
        SparseMatrix Ac(nPrimal, nPrimal);
        Ac.setFrom(se);
        Ac.makeCompressed();
        m_coarse = makeSparseLUSolver(Ac);
        m_par.sum(fc);
    }
    m_fc = fc;
    t.coarse += wallTime() - t0;

    // Right-hand side of the Schur complement system
    t0 = wallTime();
    m_precLocal.setupMultiplicityScaling();
    m_precLocalOp = m_precLocal.preconditioner();

    Matrix d;
    d.setZero(m_P.nLambda, 1);
    for (size_t i=0; i<m_patches.size(); ++i)
    {
        const OwnedPatch& p = m_patches[i];
        Matrix s;
        p.solver->apply(p.E * p.f, s);
        d += p.B * (p.E.transpose() * s);
    }
    if (nPrimal > 0)
    {
        Matrix yf = coarseSolve(m_fc);
        for (size_t i=0; i<m_patches.size(); ++i)
        {
            const OwnedPatch& p = m_patches[i];
            Matrix yl(p.pdof.size(), 1);
            for (size_t a=0; a<p.pdof.size(); ++a) yl(a,0) = yf(p.pdof[a],0);
            d += p.B * (p.Psi * yl);
        }
    }
    m_par.sum(d);
    m_rhs = d;
    t.rhs += wallTime() - t0;
}

void IetiMpi::applySchur(const Matrix& x, Matrix& y) const
{
    const index_t nPrimal = m_P.nPrimal;
    y.setZero(m_P.nLambda, 1);
    Matrix g;
    g.setZero(nPrimal, 1);

    for (size_t i=0; i<m_patches.size(); ++i)
    {
        const OwnedPatch& p = m_patches[i];
        Matrix w = p.B.transpose() * x;
        Matrix s;
        p.solver->apply(p.E * w, s);
        y += p.B * (p.E.transpose() * s);
        if (nPrimal > 0)
        {
            Matrix gl = p.Psi.transpose() * w;
            for (size_t a=0; a<p.pdof.size(); ++a) g(p.pdof[a],0) += gl(a,0);
        }
    }
    if (nPrimal > 0)
    {
        m_par.sum(g);
        Matrix yc = coarseSolve(g);
        for (size_t i=0; i<m_patches.size(); ++i)
        {
            const OwnedPatch& p = m_patches[i];
            Matrix yl(p.pdof.size(), 1);
            for (size_t a=0; a<p.pdof.size(); ++a) yl(a,0) = yc(p.pdof[a],0);
            y += p.B * (p.Psi * yl);
        }
    }
    m_par.sum(y);
}

void IetiMpi::applyPrec(const Matrix& x, Matrix& y) const
{
    m_precLocalOp->apply(x, y);
    m_par.sum(y);
}

std::vector<Matrix> IetiMpi::reconstruct(const Matrix& lambda) const
{
    const index_t nPrimal = m_P.nPrimal;
    Matrix g;
    g.setZero(nPrimal, 1);
    for (size_t i=0; i<m_patches.size(); ++i)
    {
        const OwnedPatch& p = m_patches[i];
        Matrix w = p.B.transpose() * lambda;
        Matrix gl = p.Psi.transpose() * w;
        for (size_t a=0; a<p.pdof.size(); ++a) g(p.pdof[a],0) += gl(a,0);
    }
    Matrix yc;
    if (nPrimal > 0)
    {
        m_par.sum(g);
        yc = coarseSolve(m_fc - g);
    }
    std::vector<Matrix> u(m_patches.size());
    for (size_t i=0; i<m_patches.size(); ++i)
    {
        const OwnedPatch& p = m_patches[i];
        Matrix s;
        p.solver->apply(p.E * (p.f - p.B.transpose() * lambda), s);
        u[i] = p.E.transpose() * s;
        if (nPrimal > 0)
        {
            Matrix yl(p.pdof.size(), 1);
            for (size_t a=0; a<p.pdof.size(); ++a) yl(a,0) = yc(p.pdof[a],0);
            u[i] += p.Psi * yl;
        }
    }
    return u;
}

//---------------------------------------------------------------------------
// Reference: the serial IETI-DP of the library classes (all patches)
//---------------------------------------------------------------------------

struct Reference
{
    Matrix lambda;
    OpPtr  schur;
    Matrix rhs;
    std::vector<Matrix> u;       // local solutions of all patches
    index_t iterations;
};

void libraryReference(const Problem& P, const Options& o, Reference& ref)
{
    gsIetiSystem<> ieti;
    ieti.reserve(P.nPatches+1);
    gsScaledDirichletPrec<> prec;
    prec.reserve(P.nPatches);
    gsPrimalSystem<> primal(P.nPrimal);
    primal.setEliminatePointwiseConstraints(o.eliminateVertices);

    for (index_t k=0; k<P.nPatches; ++k)
    {
        Raw raw;
        assembleRaw(P, k, raw);
        {
            gsScaledDirichletPrec<>::Blocks blocks =
                gsScaledDirichletPrec<>::matrixBlocks(raw.A, raw.skeleton);
            prec.addSubdomain(
                gsScaledDirichletPrec<>::restrictJumpMatrix(raw.B, raw.skeleton).moveToPtr(),
                gsScaledDirichletPrec<>::schurComplement(blocks, makeSparseCholeskySolver(blocks.A11))
            );
        }
        primal.handleConstraints(raw.constraints, raw.primalIdx, raw.B, raw.A, raw.f);
        ieti.addSubdomain(raw.B.moveToPtr(), makeMatrixOp(raw.A), give(raw.f),
                          makeSparseLUSolver(raw.A));
    }
    {
        OpPtr localSolver = makeSparseLUSolver(primal.localMatrix());
        ieti.addSubdomain(primal.jumpMatrix().moveToPtr(),
                          makeMatrixOp(primal.localMatrix().moveToPtr()),
                          give(primal.localRhs()), localSolver);
    }
    prec.setupMultiplicityScaling();

    ref.schur = ieti.schurComplement();
    ref.rhs   = ieti.rhsForSchurComplement();
    ref.lambda.setZero(ieti.nLagrangeMultipliers(), 1);
    gsConjugateGradient<> PCG(ref.schur, prec.preconditioner());
    PCG.setTolerance(o.tol);
    PCG.setMaxIterations(o.maxIt);
    Matrix hist;
    PCG.solveDetailed(ref.rhs, ref.lambda, hist);
    ref.iterations = hist.rows()-1;
    ref.u = primal.distributePrimalSolution(
        ieti.constructSolutionFromLagrangeMultipliers(ref.lambda));
}

} // namespace

//---------------------------------------------------------------------------
// main
//---------------------------------------------------------------------------

int main(int argc, char* argv[])
{
#ifdef GISMO_WITH_MPI
    gsMpi::init(argc, argv);
    Par par(gsMpi::worldComm());
#else
    Par par;
#endif

    Options o;
    o.dim = 3; o.patches = "2,2,2"; o.degree = 2; o.refine = 2;
    o.young = 1.0; o.poisson = 0.3;
    o.primal = "ve"; o.eliminateVertices = false;
    o.localSolver = "lu"; o.partition = "lex";
    o.tol = 1e-12; o.maxIt = 500;
    o.check = o.checkDirect = o.fastDiag = o.quiet = false;
    o.gateMult = 1e-10; o.gateL2 = 1e-8; o.gateComp = 1e-8;
    o.gateDirectL2 = 1e-8; o.gateDirectComp = 1e-8;

    gsCmdLine cmd("MPI-parallel IETI-DP solver for linear elasticity on a multipatch box.");
    cmd.addInt   ("d", "dim",       "Dimension of the generated box (2 or 3)", o.dim);
    cmd.addString("n", "patches",   "Patches per direction, e.g. 4,2,2", o.patches);
    cmd.addString("",  "xml",       "Read the multipatch from this xml file instead of generating a box", o.xml);
    cmd.addInt   ("p", "degree",    "Spline degree", o.degree);
    cmd.addInt   ("r", "refine",    "Number of uniform refinements per patch", o.refine);
    cmd.addReal  ("E", "young",     "Young's modulus", o.young);
    cmd.addReal  ("",  "poisson",   "Poisson ratio", o.poisson);
    cmd.addString("",  "traction",  "Traction vector on the face of maximal x", o.traction);
    cmd.addString("",  "primal",    "Primal space: v (vertices), ve (+ edge averages), vef (+ face averages)", o.primal);
    cmd.addSwitch("",  "eliminate-vertices", "Eliminate vertex constraints in the local systems", o.eliminateVertices);
    cmd.addString("",  "local-solver", "Local solver: lu or ldlt (ldlt only with --eliminate-vertices and primal v)", o.localSolver);
    cmd.addString("",  "partition", "Patch distribution: lex or morton", o.partition);
    cmd.addReal  ("t", "tol",       "Relative residual tolerance of the CG solver", o.tol);
    cmd.addInt   ("",  "maxit",     "Maximum number of CG iterations", o.maxIt);
    cmd.addSwitch("",  "check",     "Compare with the serial IETI-DP of the library classes", o.check);
    cmd.addSwitch("",  "check-direct", "Compare with a direct solve of the global system (gsElasticityAssembler)", o.checkDirect);
    cmd.addSwitch("q", "quiet",     "Print only the result line", o.quiet);
    cmd.addString("",  "tag",       "Label copied to the result line", o.tag);
    try { cmd.getValues(argc, argv); } catch (int rv) { return rv; }
    if (o.traction.empty()) o.traction = (o.dim==3) ? "0,0,-1" : "0,-1";

    const double tStart = wallTime();
    Timers tm;
    int status = 0;
    std::string failures;

    // Problem setup (replicated)
    Problem P;
    double t0 = wallTime();
    buildProblem(o, P);
    tm.mapper = wallTime() - t0;

    GISMO_ENSURE(par.size <= P.nPatches, "More ranks than patches.");
    if (o.localSolver == "ldlt")
        GISMO_ENSURE(o.eliminateVertices && o.primal == "v",
                     "--local-solver=ldlt requires --eliminate-vertices and --primal=v.");

    std::vector<index_t> mine = distribute(P, o.partition, par.rank, par.size);

    IetiMpi ieti(P, par, o, mine);
    ieti.setup(tm);

    // Solve
    Matrix lambda;
    lambda.setZero(P.nLambda, 1);
    gsConjugateGradient<> PCG(memory::make_shared_not_owned(static_cast<Op*>(&ieti.schur())),
                              memory::make_shared_not_owned(static_cast<Op*>(&ieti.prec())));
    PCG.setCalcEigenvalues(true);
    PCG.setTolerance(o.tol);
    PCG.setMaxIterations(o.maxIt);
    Matrix hist;
    par.commTime = 0; par.commCalls = 0; par.commBytes = 0;
    t0 = wallTime();
    PCG.solveDetailed(ieti.rhs(), lambda, hist);
    const double tSolve = wallTime() - t0;
    tm.solve = tSolve;
    const double commSolve = par.commTime;
    const index_t iterations = hist.rows()-1;
    const bool converged = hist(iterations,0) < o.tol;
    const real_t cond = PCG.getConditionNumber();

    // True relative residual of the multiplier system
    Matrix Fl;
    ieti.schur().apply(lambda, Fl);
    const real_t relRes = (ieti.rhs() - Fl).norm() / ieti.rhs().norm();

    std::vector<Matrix> u = ieti.reconstruct(lambda);
    const real_t compliance = ieti.compliance(u);

    // Consistency of the replicated iteration
    if (par.minOf(iterations) != par.maxOf(iterations))
    { status = 1; failures += " iteration-counts-differ"; }
    if (!converged) { status = 1; failures += " no-convergence"; }
    if (relRes > 10*o.tol && relRes > 1e-10) { status = 1; failures += " residual"; }

    // Checks
    real_t chkMult = -1, chkL2 = -1, chkComp = -1, chkDirL2 = -1, chkDirComp = -1;
    real_t l2norm = -1;
    const index_t dim = P.dim;
    std::vector<SparseMatrix> mass;
    if (o.check || o.checkDirect)
    {
        for (size_t i=0; i<ieti.patches().size(); ++i)
            mass.push_back(assembleMass(P, ieti.patches()[i].k));
        Matrix l2(1,1);
        l2(0,0) = 0;
        for (size_t i=0; i<mass.size(); ++i)
            l2(0,0) += squaredL2(mass[i], localCoefficients(P, ieti.patches()[i].k, u[i]));
        par.sum(l2);
        l2norm = std::sqrt(l2(0,0));
    }

    if (o.check)
    {
        Reference ref;
        libraryReference(P, o, ref);
        Matrix Fref;
        ref.schur->apply(lambda, Fref);
        chkMult = (Fref - ref.rhs).norm() / ref.rhs.norm();

        Matrix part(2,1);
        part.setZero();
        Matrix cref(1,1);
        cref(0,0) = 0;
        Matrix lref(1,1);
        lref(0,0) = 0;
        for (index_t k=0; k<P.nPatches; ++k)
        {
            SparseMatrix M = assembleMass(P, k);
            Matrix cf = localCoefficients(P, k, ref.u[k]);
            lref(0,0) += squaredL2(M, cf);
        }
        // reference compliance needs the local right-hand sides
        for (index_t k=0; k<P.nPatches; ++k)
        {
            Raw raw;
            assembleRaw(P, k, raw);
            cref(0,0) += raw.f.col(0).dot(ref.u[k].col(0));
        }
        const real_t l2ref = std::sqrt(lref(0,0));
        chkL2   = std::abs(l2norm - l2ref) / l2ref;
        chkComp = std::abs(compliance - cref(0,0)) / std::abs(cref(0,0));
        if (!(chkMult <= o.gateMult)) { status = 1; failures += " check-multiplier"; }
        if (!(chkL2   <= o.gateL2))   { status = 1; failures += " check-l2"; }
        if (!(chkComp <= o.gateComp)) { status = 1; failures += " check-compliance"; }
    }

#ifdef IETI_HAVE_ELASTICITY
    if (o.checkDirect)
    {
        gsConstantFunction<> body(0.0, dim);
        std::vector<real_t> tr = P.traction;
        gsConstantFunction<> g(gsVector<>::Map(tr.data(), dim), dim);
        gsBoundaryConditions<> bcInfo;
        for (size_t s=0; s<P.clampSides.size(); ++s)
            for (index_t d=0; d<dim; ++d)
                bcInfo.addCondition(P.clampSides[s].patch, P.clampSides[s],
                                    condition_type::dirichlet, 0, d);
        for (size_t s=0; s<P.tractionSides.size(); ++s)
            bcInfo.addCondition(P.tractionSides[s].patch, P.tractionSides[s],
                                condition_type::neumann, &g);
        gsElasticityAssembler<real_t> assembler(P.mp, P.mb, bcInfo, body);
        assembler.options().setReal("YoungsModulus", o.young);
        assembler.options().setReal("PoissonsRatio", o.poisson);
        assembler.assemble();
        gsSparseSolver<>::SimplicialLDLT solver(assembler.matrix());
        gsVector<> solVector = solver.solve(assembler.rhs());
        gsMultiPatch<> displacement;
        assembler.constructSolution(solVector, assembler.allFixedDofs(), displacement);
        const real_t cdir = assembler.rhs().dot(solVector);

        Matrix e(2,1);
        e.setZero();
        for (size_t i=0; i<ieti.patches().size(); ++i)
        {
            const index_t k = ieti.patches()[i].k;
            Matrix diff = localCoefficients(P, k, u[i]) - displacement.patch(k).coefs();
            e(0,0) += squaredL2(mass[i], diff);
            e(1,0) += squaredL2(mass[i], displacement.patch(k).coefs());
        }
        par.sum(e);
        chkDirL2   = std::sqrt(e(0,0)/e(1,0));
        chkDirComp = std::abs(compliance - cdir) / std::abs(cdir);
        if (!(chkDirL2   <= o.gateDirectL2))   { status = 1; failures += " direct-l2"; }
        if (!(chkDirComp <= o.gateDirectComp)) { status = 1; failures += " direct-compliance"; }
    }
#else
    if (o.checkDirect)
    { status = 1; failures += " direct-check-unavailable"; }
#endif

    tm.total = wallTime() - tStart;

    // Timers: maximum over the ranks
    const double tAssembly = par.maxOf(tm.assembly), tFactor = par.maxOf(tm.factor),
                 tCoarse   = par.maxOf(tm.coarse),   tRhs    = par.maxOf(tm.rhs),
                 tSolveMax = par.maxOf(tm.solve),    tTotal  = par.maxOf(tm.total),
                 tMapper   = par.maxOf(tm.mapper),   tComm   = par.maxOf(commSolve);
    const double nCalls   = (double)par.maxOf((index_t)par.commCalls);
    const real_t condMax  = cond;
    int threads = 1;
#ifdef _OPENMP
    threads = omp_get_max_threads();
#endif

    if (par.rank == 0)
    {
        std::ostringstream os;
        os.precision(10);
        os << "RESULT"
           << " tag=" << (o.tag.empty() ? "-" : o.tag)
           << " ranks=" << par.size
           << " threads=" << threads
           << " dim=" << dim
           << " patches=" << P.nPatches
           << " grid=" << P.grid[0] << "x" << P.grid[1] << "x" << P.grid[2]
           << " degree=" << o.degree
           << " refine=" << o.refine
           << " unknowns=" << P.nUnknowns
           << " multipliers=" << P.nLambda
           << " primal=" << P.nPrimal
           << " primal_type=" << o.primal
           << " partition=" << o.partition
           << " t_mapper=" << tMapper
           << " t_assembly=" << tAssembly
           << " t_factor=" << tFactor
           << " t_coarse=" << tCoarse
           << " t_rhs=" << tRhs
           << " iterations=" << iterations
           << " t_solve=" << tSolveMax
           << " t_iter=" << tSolveMax/std::max<index_t>(iterations,1)
           << " t_total=" << tTotal
           << " t_comm=" << tComm
           << " comm_calls=" << nCalls
           << " cond=" << condMax
           << " relres=" << relRes
           << " compliance=" << compliance
           << " l2norm=" << l2norm
           << " chk_mult=" << chkMult
           << " chk_l2=" << chkL2
           << " chk_comp=" << chkComp
           << " chk_direct_l2=" << chkDirL2
           << " chk_direct_comp=" << chkDirComp
           << " status=" << (status ? "FAIL" : "PASS");
        if (status) os << " failures=" << failures;
        gsInfo << os.str() << "\n";
    }

    return status;
}
