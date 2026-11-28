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

    Parallelization. The patches are distributed over the MPI ranks. Every
    rank holds the patches of the whole domain with their topology (one
    trilinear patch per subdomain for the generated box) and, as compact
    integer data, the neighbours of every patch. Bases, dof mappers and the
    gsIetiMapper are built only for the own patches and two layers of
    neighbours around them, which contain all patches that share a dof with an
    own patch or with one of its neighbours. The global numbering of the
    multipliers and of the primal dofs is that of gsIetiMapper on all
    patches: every rank counts the multipliers and primal dofs whose first
    occurrence lies on its own patches, and one sum of these counts over the
    ranks gives the first global index on each patch. Each rank assembles and
    factorizes only its own patches; its jump matrices have one row per
    multiplier of its patches. The vector of Lagrange multipliers is
    replicated on all ranks. Every application of the Schur complement or of
    the preconditioner evaluates the contributions of the local patches and
    sums them over the ranks with MPI_Allreduce. The coarse matrix is
    assembled on rank 0 from the dense blocks of all ranks, received one rank
    after the other, and factorized there; rank 0 broadcasts the coarse
    solutions (--coarse-on=all assembles and factorizes it on every rank).
    Because the conjugate gradient iteration only acts on replicated vectors,
    gsConjugateGradient is used unchanged.

    With --multipliers=distributed a rank holds only the multipliers that
    touch its patches. A multiplier couples two patches, so it lives on one
    rank or on two; an operator application adds the contribution of the
    other rank with one MPI_Neighbor_alltoallv over the graph of adjacent
    ranks (MPI_Dist_graph_create_adjacent), and the conjugate gradient
    iteration reduces one scalar per inner product. The adjacent ranks and
    the shared multipliers come from a search over the own patches and the
    patches near them (linkcell: halo cells of the patch grid and a pair
    search over the Greville points of the coupled dofs; mapper: the inverse
    of the dof mapper of the local patches), followed by one exchange of
    candidate lists between adjacent ranks.

    The checks (--check, --check-direct, --check-topology) build the problem
    on all patches on every rank in addition and compare with it; this
    includes the global numbering, which must equal that of gsIetiMapper on
    all patches.

    The program checks its own result and exits with a non-zero code if a
    check fails (option --check compares with the serial IETI-DP implementation
    of the library classes, option --check-direct with a direct solve of the
    assembled global system by gsElasticityAssembler).

    The result is printed as one line of key=value fields. The fields mem_*
    give the resident set size (MiB, maximum over the ranks) at the end of
    each setup phase, rss_peak_max and rss_peak_mean the peak resident set
    size of the whole run (MiB, maximum and mean over the ranks).

    This file is part of the G+Smo library.

    This Source Code Form is subject to the terms of the Mozilla Public
    License, v. 2.0. If a copy of the MPL was not distributed with this
    file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#include <gismo.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <iterator>
#include <limits>
#include <map>
#include <numeric>
#include <set>
#include <sstream>

#ifdef _OPENMP
#include <omp.h>
#endif

#if defined(__unix__) || defined(__APPLE__)
#include <sys/resource.h>
#include <unistd.h>
#define IETI_HAVE_RUSAGE 1
#endif

#ifdef IETI_WITH_LINKCELL
#include <linkcell.h>
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

/// Peak resident set size of this process (MiB), -1 if unknown
double peakRss()
{
#ifdef IETI_HAVE_RUSAGE
    struct rusage ru;
    if (getrusage(RUSAGE_SELF, &ru) != 0) return -1;
#  ifdef __APPLE__
    return ru.ru_maxrss / 1048576.0;   // bytes
#  else
    return ru.ru_maxrss / 1024.0;      // kilobytes
#  endif
#else
    return -1;
#endif
}

/// Current resident set size of this process (MiB): /proc/self/statm on
/// Linux, the peak elsewhere
double currentRss()
{
#if defined(__linux__)
    std::FILE* f = std::fopen("/proc/self/statm", "r");
    if (f)
    {
        long size = 0, resident = 0;
        const int n = std::fscanf(f, "%ld %ld", &size, &resident);
        std::fclose(f);
        if (n == 2) return resident * (double)sysconf(_SC_PAGESIZE) / 1048576.0;
    }
#endif
    return peakRss();
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
    explicit Par(MPI_Comm c) : commTime(0), commCalls(0), commBytes(0), comm(c)
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
    /// Sum over the ranks, result on rank 0 only
    void sumToRoot(Matrix& m)
    {
        if (size==1) return;
        const double t = MPI_Wtime();
        if (rank==0)
            MPI_Reduce(MPI_IN_PLACE, m.data(), (int)m.size(), MPI_DOUBLE, MPI_SUM, 0, comm);
        else
            MPI_Reduce(m.data(), NULL, (int)m.size(), MPI_DOUBLE, MPI_SUM, 0, comm);
        commTime  += MPI_Wtime() - t;
        commCalls += 1;
        commBytes += 8.0*m.size();
    }
    /// Broadcast from rank 0; m must have its final size on all ranks
    void bcast(Matrix& m)
    {
        if (size==1) return;
        const double t = MPI_Wtime();
        MPI_Bcast(m.data(), (int)m.size(), MPI_DOUBLE, 0, comm);
        commTime  += MPI_Wtime() - t;
        commCalls += 1;
        commBytes += 8.0*m.size();
    }
    void barrier() const { if (size>1) MPI_Barrier(comm); }
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
    /// Concatenation of the vectors of all ranks, in rank order, on rank 0
    /// (empty elsewhere)
    template<class T> std::vector<T> gatherRoot(const std::vector<T>& in, MPI_Datatype dt) const
    {
        if (size==1) return in;
        std::vector<int> cnt(rank==0 ? size : 1), dsp(rank==0 ? size : 1);
        int n = (int)in.size();
        MPI_Gather(&n, 1, MPI_INT, cnt.data(), 1, MPI_INT, 0, comm);
        int tot = 0;
        if (rank==0) for (int i=0; i<size; ++i) { dsp[i] = tot; tot += cnt[i]; }
        std::vector<T> out(tot);
        MPI_Gatherv(in.data(), n, dt, out.data(), cnt.data(), dsp.data(), dt, 0, comm);
        return out;
    }
    /// Sum over the ranks, on every rank
    void sumAll(std::vector<long long>& v) const
    {
        if (size>1 && !v.empty())
            MPI_Allreduce(MPI_IN_PLACE, v.data(), (int)v.size(), MPI_LONG_LONG, MPI_SUM, comm);
    }
    static MPI_Datatype dtype(const int*)    { return MPI_INT;    }
    static MPI_Datatype dtype(const double*) { return MPI_DOUBLE; }
    /// Ranks adjacent to this one: the given list and every rank that lists
    /// this rank, ascending. One gather of all lists (ranks times degree ints).
    std::vector<int> symmetricNeighbours(const std::vector<int>& nbr) const
    {
        std::vector<int> edges;
        for (size_t i=0; i<nbr.size(); ++i) { edges.push_back(rank); edges.push_back(nbr[i]); }
        const std::vector<int> all = allgatherv(edges, MPI_INT);
        std::vector<int> out;
        for (size_t i=0; i+1<all.size(); i+=2)
        {
            if (all[i] == rank && all[i+1] != rank) out.push_back(all[i+1]);
            if (all[i+1] == rank && all[i] != rank) out.push_back(all[i]);
        }
        std::sort(out.begin(), out.end());
        out.erase(std::unique(out.begin(), out.end()), out.end());
        return out;
    }
    /// Communicator with the adjacent ranks as sources and destinations
    MPI_Comm neighbourGraph(const std::vector<int>& nbr) const
    {
        MPI_Comm g;
        const int deg = (int)nbr.size();
        MPI_Dist_graph_create_adjacent(comm, deg, deg ? nbr.data() : NULL, MPI_UNWEIGHTED,
            deg, deg ? nbr.data() : NULL, MPI_UNWEIGHTED, MPI_INFO_NULL, 0, &g);
        return g;
    }
#else
    Par() : rank(0), size(1), commTime(0), commCalls(0), commBytes(0) {}
    void   sum(Matrix&) {}
    void   sumToRoot(Matrix&) {}
    void   bcast(Matrix&) {}
    void   barrier() const {}
    double maxOf(double v) const { return v; }
    double sumOf(double v) const { return v; }
    index_t minOf(index_t v) const { return v; }
    index_t maxOf(index_t v) const { return v; }
    template<class T> std::vector<T> allgatherv(const std::vector<T>& in, int) const { return in; }
    template<class T> std::vector<T> gatherRoot(const std::vector<T>& in, int) const { return in; }
    void sumAll(std::vector<long long>&) const {}
    static int dtype(const int*)    { return 0; }
    static int dtype(const double*) { return 0; }
#endif
    template<class T> std::vector<T> gather(const std::vector<T>& in) const
    { return allgatherv(in, dtype((const T*)NULL)); }
    template<class T> std::vector<T> gatherToRoot(const std::vector<T>& in) const
    { return gatherRoot(in, dtype((const T*)NULL)); }
};

//---------------------------------------------------------------------------
// Problem definition
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
    real_t  innerTol;          // tolerance of the inner CG of the fastdiag local solver
    index_t innerMaxIt;
    bool    check, checkDirect, checkTopology, fastDiag, quiet;
    std::string cost;          // n or n2 (patch weight of the linkcell partition)
    index_t lcMethod;          // linkcell partition method
    std::string tag;
    bool    setupOnly;         // stop after the coarse problem, print the timing split
    index_t emulateRanks;      // setup-only: build the local patches of rank 0 out of this many
    std::string coarse;        // gather (triplets to all ranks) or reduce (pattern from the primal indices)
    std::string coarseSolver;  // lu (SparseLU, COLAMD) or ldlt (SimplicialLDLT, AMD)
    bool    coarseRoot;        // factorize the coarse matrix on rank 0 only and broadcast its solutions
    std::string coarseOn;      // root or all: where the coarse matrix is factorized
    std::string multipliers;   // replicated (one reduction per application) or distributed (neighbour exchange)
    std::string dualNeighbours; // linkcell (halo and pair search) or mapper (inverse of the local dof mapper)
    real_t  gateMult, gateL2, gateComp, gateDirectL2, gateDirectComp;
};

/// The domain and the discretization that one rank needs.
///
/// Global part (every rank): the patches with their geometry as given and
/// their topology, the boundary sides that are clamped or loaded, the
/// patches that share a vertex with each patch, and the shared components
/// (edges, faces) in the order of gsBoxTopology::allComponents. This is
/// compact: a generated box has one trilinear patch per subdomain.
///
/// Local part: bases, boundary conditions and gsIetiMapper for the local
/// patches only, the own patches and their neighbours up to a small depth.
/// A local patch is complete if all its neighbours are local; the coupled
/// dofs of a complete patch then have the same groups as in the dof mapper
/// of all patches. The global numbering of the multipliers and the primal
/// dofs is that of gsIetiMapper on all patches: per patch counts, summed
/// over the ranks, give the first global index of the entities whose first
/// occurrence (lowest patch, then lowest basis index) lies on that patch.
struct Problem
{
    index_t dim;
    index_t nPatches;                         ///< patches of the whole domain
    index_t grid[3];                          ///< patches per direction, 0 if unknown
    gsMultiPatch<>                   global;  ///< all patches and their topology
    std::vector<int>                 nbrStart, nbr; ///< neighbours of patch k: nbr[nbrStart[k]..nbrStart[k+1])
    /// [d]: ((first patch, component index), position) of each component of
    /// dimension d (1 or 2) shared by several patches, sorted by the key;
    /// the position counts in the order of allComponents of all patches
    std::vector<std::pair<std::pair<int,int>,int> > compPos[3];
    index_t nComp[3];
    real_t  lambda, mu;
    std::vector<real_t> traction;
    std::vector<patchSide> clampSides, tractionSides;   ///< global patch indices
    gsConstantFunction<> zero;

    std::vector<index_t>             gid;      ///< global index of each local patch, ascending
    std::vector<char>                complete; ///< all neighbours of the local patch are local
    gsMultiPatch<>                   mp;       ///< local patches and the interfaces between them
    gsMultiBasis<>                   mb;
    std::vector<gsBoundaryConditions<> > bc;
    std::vector<gsIetiMapper<> >     mapper;   ///< one per displacement component, on the local patches

    /// [r]: global index of each local multiplier (primal dof) of component
    /// r, -1 for those not numbered (first occurrence on an incomplete patch)
    std::vector<std::vector<index_t> > lambdaGlobal, primalGlobal;
    index_t nLambda, nPrimal, nUnknowns;      ///< global sizes

    Problem() : dim(0), nPatches(0), nLambda(0), nPrimal(0), nUnknowns(0)
    { grid[0] = grid[1] = grid[2] = 0; nComp[0] = nComp[1] = nComp[2] = 0; }

    bool isLocal(index_t k) const { return std::binary_search(gid.begin(), gid.end(), k); }
    /// Local index of the global patch k
    index_t loc(index_t k) const
    {
        const std::vector<index_t>::const_iterator it = std::lower_bound(gid.begin(), gid.end(), k);
        GISMO_ENSURE(it != gid.end() && *it == k, "Patch " << k << " is not local.");
        return (index_t)(it - gid.begin());
    }
    /// Position of the shared component of dimension d with first patch k (global) and index c
    index_t compPosition(index_t d, index_t k, index_t c) const
    {
        const std::pair<std::pair<int,int>,int> key(std::make_pair((int)k, (int)c), -1);
        const std::vector<std::pair<std::pair<int,int>,int> >& v = compPos[d];
        const std::vector<std::pair<std::pair<int,int>,int> >::const_iterator it =
            std::lower_bound(v.begin(), v.end(), key);
        GISMO_ENSURE(it != v.end() && it->first == key.first, "Unknown component " << c << " of patch " << k);
        return it->second;
    }
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

/// Wall times of the phases (seconds). The mapper phase is split into the
/// topology of the multipatch, the bases, the dof mappers (gsIetiMapper::init),
/// the primal constraints and the jump matrices; the coarse phase into the
/// wait for the slowest rank, the assembly with its communication and the
/// factorization.
struct Timers
{
    double mapper, topo, basis, dofmap, primals, jumps;
    double assembly, factor, wait, coarse, cassemble, cfactor, rhs, solve, total, dual;
    Timers() : mapper(0), topo(0), basis(0), dofmap(0), primals(0), jumps(0),
               assembly(0), factor(0), wait(0), coarse(0), cassemble(0), cfactor(0),
               rhs(0), solve(0), total(0), dual(0) {}
};

/// Resident set size (MiB) at the end of the phases of Timers: after the
/// topology, the bases, the dof mappers, the primal constraints, the jump
/// matrices, the multiplier layout, the local setup of the own patches and
/// the coarse problem
struct Memory
{
    double topo, basis, dofmap, primals, jumps, dual, local, coarse;
    Memory() : topo(0), basis(0), dofmap(0), primals(0), jumps(0), dual(0), local(0), coarse(0) {}

    /// Fields mem_<phase>=max over the ranks, rss_peak_max and rss_peak_mean
    std::string fields(const Par& par) const
    {
        const double v[] = { topo, basis, dofmap, primals, jumps, dual, local, coarse };
        const char* k[] = { "mem_topo", "mem_basis", "mem_dofmap", "mem_primals", "mem_jumps",
                            "mem_dual", "mem_local", "mem_coarse" };
        std::ostringstream os;
        os.precision(6);
        for (size_t i=0; i<sizeof(v)/sizeof(v[0]); ++i)
            os << " " << k[i] << "=" << par.maxOf(v[i]);
        const double peak = peakRss();
        os << " rss_peak_max=" << par.maxOf(peak) << " rss_peak_mean=" << par.sumOf(peak)/par.size;
        return os.str();
    }
};

/// The global part of the problem: patches, topology, material, the clamped
/// and loaded sides, the neighbours of every patch and the shared components
void buildGlobal(const Options& o, Problem& P, Timers& t, Memory& m)
{
    double t0 = wallTime();
    gsMultiPatch<>& G = P.global;
    P.grid[0] = P.grid[1] = P.grid[2] = 0;
    if (!o.xml.empty())
    {
        gsMultiPatch<>::uPtr in = gsReadFile<>(o.xml);
        GISMO_ENSURE(in, "Cannot read multipatch from " << o.xml);
        G = *in;
        P.dim = G.geoDim();
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
                        G.addPatch( *gsNurbsCreator<>::BSplineCube(1.0, i, j, k) );
                    else
                        G.addPatch( *gsNurbsCreator<>::BSplineSquare(1.0, i, j) );
                }
        G.computeTopology();
    }
    const index_t dim = P.dim;
    P.nPatches = G.nPatches();

    P.lambda = o.young*o.poisson / ((1+o.poisson)*(1-2*o.poisson));
    P.mu     = o.young / (2*(1+o.poisson));
    P.traction = parseReals(o.traction);
    P.traction.resize(dim, 0.0);
    P.zero = gsConstantFunction<>(0.0, dim);

    // Clamped and loaded sides
    real_t xmin = 1e300, xmax = -1e300;
    for (index_t k=0; k<P.nPatches; ++k)
    {
        xmin = std::min(xmin, G[k].coefs().col(0).minCoeff());
        xmax = std::max(xmax, G[k].coefs().col(0).maxCoeff());
    }
    const real_t eps = 1e-8*(xmax-xmin);
    for (gsMultiPatch<>::const_biterator it = G.bBegin(); it < G.bEnd(); ++it)
    {
        const real_t x = sideMidpoint(G, *it)(0,0);
        if (x < xmin + eps)      P.clampSides.push_back(*it);
        else if (x > xmax - eps) P.tractionSides.push_back(*it);
    }
    GISMO_ENSURE(!P.clampSides.empty(), "No face to clamp found.");

    // Neighbours (a shared vertex) and the shared edges and faces
    {
        const std::vector< std::vector<patchComponent> > comps = G.allComponents();
        std::vector< std::vector<int> > adj(P.nPatches);
        for (index_t d=0; d<3; ++d) { P.compPos[d].clear(); P.nComp[d] = 0; }
        for (size_t c=0; c<comps.size(); ++c)
        {
            const std::vector<patchComponent>& g = comps[c];
            const index_t d = g[0].dim();
            if (d == 0)
            {
                for (size_t a=0; a<g.size(); ++a)
                    for (size_t b=0; b<g.size(); ++b)
                        if (g[a].patch() != g[b].patch())
                            adj[g[a].patch()].push_back((int)g[b].patch());
            }
            else if (d < dim && g.size() > 1)
                P.compPos[d].push_back(std::make_pair(std::make_pair((int)g[0].patch(), (int)g[0].index()),
                                                      (int)P.nComp[d]++));
        }
        for (index_t d=0; d<3; ++d) std::sort(P.compPos[d].begin(), P.compPos[d].end());
        P.nbrStart.assign(P.nPatches+1, 0);
        P.nbr.clear();
        for (index_t k=0; k<P.nPatches; ++k)
        {
            std::sort(adj[k].begin(), adj[k].end());
            adj[k].erase(std::unique(adj[k].begin(), adj[k].end()), adj[k].end());
            P.nbr.insert(P.nbr.end(), adj[k].begin(), adj[k].end());
            P.nbrStart[k+1] = (int)P.nbr.size();
            std::vector<int>().swap(adj[k]);
        }
    }
    t.topo += wallTime() - t0;
    m.topo = currentRss();
}

/// The given patches and their neighbours up to the given depth, ascending
std::vector<index_t> ring(const Problem& P, const std::vector<index_t>& seed, int depth)
{
    std::vector<char> in(P.nPatches, 0);
    std::vector<index_t> cur, all;
    for (size_t q=0; q<seed.size(); ++q)
        if (!in[seed[q]]) { in[seed[q]] = 1; cur.push_back(seed[q]); }
    all = cur;
    for (int d=0; d<depth; ++d)
    {
        std::vector<index_t> next;
        for (size_t q=0; q<cur.size(); ++q)
            for (int j=P.nbrStart[cur[q]]; j<P.nbrStart[cur[q]+1]; ++j)
                if (!in[P.nbr[j]]) { in[P.nbr[j]] = 1; next.push_back(P.nbr[j]); }
        all.insert(all.end(), next.begin(), next.end());
        cur.swap(next);
    }
    std::sort(all.begin(), all.end());
    return all;
}

/// Local part: patches (global indices, ascending), interfaces between them,
/// bases, boundary conditions and the IETI mappers. Replaces a previous
/// local part.
void buildLocal(const Options& o, Problem& P, const std::vector<index_t>& patches, Timers& t, Memory& m)
{
    double t0 = wallTime();
    const index_t dim = P.dim;
    P.gid = patches;
    const index_t nl = P.gid.size();
    P.complete.assign(nl, 1);
    for (index_t l=0; l<nl; ++l)
        for (int j=P.nbrStart[P.gid[l]]; j<P.nbrStart[P.gid[l]+1]; ++j)
            if (!P.isLocal(P.nbr[j])) { P.complete[l] = 0; break; }

    P.mp = gsMultiPatch<>();
    for (index_t l=0; l<nl; ++l)
        P.mp.addPatch(P.global.patch(P.gid[l]));
    for (gsMultiPatch<>::const_iiterator it = P.global.iBegin(); it != P.global.iEnd(); ++it)
        if (P.isLocal(it->first().patch) && P.isLocal(it->second().patch))
        {
            boundaryInterface bi = *it;
            bi.first().patch  = P.loc(it->first().patch);
            bi.second().patch = P.loc(it->second().patch);
            P.mp.addInterface(bi);
        }
    P.mp.addAutoBoundaries();

    // Bases: degree o.degree, o.refine uniform refinements
    P.mb = gsMultiBasis<>(P.mp, true);
    for (index_t l=0; l<nl; ++l)
        P.mb[l].setDegreePreservingMultiplicity(o.degree);
    for (index_t i=0; i<o.refine; ++i)
        P.mb.uniformRefine();

    // Boundary conditions on the true boundary sides of the local patches
    P.bc.assign(dim, gsBoundaryConditions<>());
    for (size_t s=0; s<P.clampSides.size(); ++s)
        if (P.isLocal(P.clampSides[s].patch))
            for (index_t r=0; r<dim; ++r)
                P.bc[r].addCondition(patchSide(P.loc(P.clampSides[s].patch), P.clampSides[s].side()),
                                     condition_type::dirichlet, &P.zero);
    for (size_t s=0; s<P.tractionSides.size(); ++s)
        if (P.isLocal(P.tractionSides[s].patch))
            for (index_t r=0; r<dim; ++r)
                P.bc[r].addCondition(patchSide(P.loc(P.tractionSides[s].patch), P.tractionSides[s].side()),
                                     condition_type::neumann, &P.zero);
    for (index_t r=0; r<dim; ++r)
        P.bc[r].setGeoMap(P.mp);
    t.basis += wallTime() - t0;
    m.basis = currentRss();
    t0 = wallTime();

    // IETI mappers, one per displacement component
    P.mapper.assign(dim, gsIetiMapper<>());
    for (index_t r=0; r<dim; ++r)
    {
        gsExprAssembler<> assembler;
        gsExprAssembler<>::space u = assembler.getSpace(P.mb);
        u.setup(P.bc[r], dirichlet::interpolation, 0);
        P.mapper[r].init(P.mb, u.mapper(), u.fixedPart());
    }
    t.dofmap += wallTime() - t0;
}

/// Coupled dofs of the local dof mapper of component r: the members (local
/// patch, basis index) of every coupled index, ascending (the order of
/// gsIetiMapper::computeJumpMatrices), and how many of them are patch corners
struct Groups
{
    std::vector<index_t> start;
    std::vector< std::pair<index_t,index_t> > member;
    std::vector<index_t> corners;
    index_t size() const { return (index_t)start.size()-1; }
    index_t members(index_t c) const { return start[c+1]-start[c]; }
    index_t first(index_t c) const { return member[start[c]].first; }
};

Groups coupledGroups(const Problem& P, index_t r)
{
    const gsDofMapper& dm = P.mapper[r].dofMapperGlobal();
    const index_t nc = dm.coupledSize(), nl = P.gid.size(), dim = P.dim;
    Groups G;
    G.start.assign(nc+1, 0);
    for (index_t k=0; k<nl; ++k)
        for (index_t i=0; i<(index_t)dm.patchSize(k); ++i)
            if (dm.is_coupled(i,k)) ++G.start[dm.cindex(i,k)+1];
    for (index_t c=0; c<nc; ++c) G.start[c+1] += G.start[c];
    std::vector<index_t> fill(G.start.begin(), G.start.end()-1);
    G.member.resize(G.start[nc]);
    for (index_t k=0; k<nl; ++k)
        for (index_t i=0; i<(index_t)dm.patchSize(k); ++i)
            if (dm.is_coupled(i,k)) G.member[fill[dm.cindex(i,k)]++] = std::make_pair(k, i);
    G.corners.assign(nc, 0);
    for (index_t k=0; k<nl; ++k)
        for (boxCorner it = boxCorner::getFirst(dim); it!=boxCorner::getEnd(dim); ++it)
        {
            const index_t idx = P.mb[k].functionAtCorner(it);
            if (dm.is_coupled(idx,k)) ++G.corners[dm.cindex(idx,k)];
        }
    return G;
}

/// Whether a shared component carries an average for component r: one of
/// its patches has a free dof on it (gsIetiMapper::interfaceAveragesAsPrimals)
bool componentUsed(const Problem& P, index_t r, const std::vector<patchComponent>& g)
{
    for (size_t i=0; i<g.size(); ++i)
    {
        const index_t k = g[i].patch();
        const gsDofMapper& dl = P.mapper[r].dofMapperLocal(k);
        gsMatrix<index_t> indices;
        P.mb[k].componentBasis_withIndices(g[i], indices, false);
        for (index_t j=0; j<indices.size(); ++j)
            if (dl.is_free_index(dl.index(indices(j,0),0)))
                return true;
    }
    return false;
}

/// Dimensions of the averages that are primal dofs
std::vector<index_t> averageDims(const Problem& P, const Options& o)
{
    std::vector<index_t> d;
    if (o.primal != "v") d.push_back(1);
    if (o.primal == "vef" && P.dim==3) d.push_back(2);
    return d;
}

/// Contributions of patches to the global numbering, summed over the ranks:
/// per component r and patch k the multipliers, vertex primal dofs and free
/// dofs whose first occurrence lies on k; per component and shared edge or
/// face whether it carries an average
struct Counts
{
    index_t dim, N, c1, c2;
    std::vector<long long> data;
    explicit Counts(const Problem& P) : dim(P.dim), N(P.nPatches), c1(P.nComp[1]), c2(P.nComp[2]),
        data(P.dim*(3*P.nPatches + P.nComp[1] + P.nComp[2]), 0) {}
    long long& patch(index_t r, index_t k, int what) { return data[(r*N + k)*3 + what]; }
    long long  patch(index_t r, index_t k, int what) const { return data[(r*N + k)*3 + what]; }
    long long& used(index_t r, index_t d, index_t pos) { return data[dim*3*N + r*(c1+c2) + (d==1 ? 0 : c1) + pos]; }
    long long  used(index_t r, index_t d, index_t pos) const { return data[dim*3*N + r*(c1+c2) + (d==1 ? 0 : c1) + pos]; }
};

/// Adds the contributions of the given local patches, which must be complete
void countPatches(const Problem& P, const Options& o, const std::vector<index_t>& localPatches, Counts& C)
{
    const index_t nl = P.gid.size();
    std::vector<char> contrib(nl, 0);
    for (size_t q=0; q<localPatches.size(); ++q)
    {
        GISMO_ENSURE(P.complete[localPatches[q]], "Patch " << P.gid[localPatches[q]]
                     << " contributes to the numbering but has neighbours that are not local.");
        contrib[localPatches[q]] = 1;
    }
    const std::vector<index_t> dims = averageDims(P, o);
    std::vector< std::vector<patchComponent> > comps;
    if (!dims.empty()) comps = P.mp.allComponents();
    for (index_t r=0; r<P.dim; ++r)
    {
        const Groups G = coupledGroups(P, r);
        for (index_t c=0; c<G.size(); ++c)
        {
            if (!contrib[G.first(c)]) continue;
            const index_t k = P.gid[G.first(c)];
            const long long n = G.members(c);
            C.patch(r,k,2) += 1;
            if (G.corners[c] == 0) C.patch(r,k,0) += n*(n-1)/2;
            if (G.corners[c] >= 2) C.patch(r,k,1) += 1;
        }
        const gsDofMapper& dm = P.mapper[r].dofMapperGlobal();
        for (index_t l=0; l<nl; ++l)
            if (contrib[l])
                for (index_t i=0; i<(index_t)dm.patchSize(l); ++i)
                    if (dm.is_free(i,l) && !dm.is_coupled(i,l)) C.patch(r,P.gid[l],2) += 1;
        for (size_t q=0; q<dims.size(); ++q)
            for (size_t c=0; c<comps.size(); ++c)
            {
                const std::vector<patchComponent>& g = comps[c];
                if (g[0].dim() != dims[q] || g.size() < 2 || !contrib[g[0].patch()]) continue;
                if (componentUsed(P, r, g))
                    C.used(r, dims[q], P.compPosition(dims[q], P.gid[g[0].patch()], g[0].index())) = 1;
            }
    }
}

/// Global indices of the local multipliers and primal dofs, from the summed
/// counts. The local indices are those of gsIetiMapper on the local patches.
void numberLocal(Problem& P, const Options& o, const Counts& C)
{
    const index_t N = P.nPatches, nl = P.gid.size();
    const std::vector<index_t> dims = averageDims(P, o);
    std::vector< std::vector<patchComponent> > comps;
    if (!dims.empty()) comps = P.mp.allComponents();
    P.lambdaGlobal.assign(P.dim, std::vector<index_t>());
    P.primalGlobal.assign(P.dim, std::vector<index_t>());
    long long lamOff = 0, priOff = 0, unknowns = 0;
    for (index_t r=0; r<P.dim; ++r)
    {
        std::vector<long long> MP(N+1, 0), VP(N+1, 0);
        for (index_t k=0; k<N; ++k)
        {
            MP[k+1] = MP[k] + C.patch(r,k,0);
            VP[k+1] = VP[k] + C.patch(r,k,1);
            unknowns += C.patch(r,k,2);
        }
        // Averages: global position among the used shared components of each dimension
        std::vector< std::vector<long long> > avgId(3);
        long long avgBase[3] = {0, 0, 0}, nAvg = 0;
        for (size_t q=0; q<dims.size(); ++q)
        {
            const index_t d = dims[q];
            avgBase[d] = VP[N] + nAvg;
            avgId[d].assign(P.nComp[d], -1);
            for (index_t pos=0; pos<P.nComp[d]; ++pos)
                if (C.used(r,d,pos)) avgId[d][pos] = nAvg++;
        }

        const Groups G = coupledGroups(P, r);
        std::vector<long long> within(nl, 0), withinV(nl, 0);
        std::vector<index_t>& lg = P.lambdaGlobal[r];
        std::vector<index_t>& pg = P.primalGlobal[r];
        for (index_t c=0; c<G.size(); ++c)
        {
            const index_t l0 = G.first(c), k0 = P.gid[l0];
            const bool known = P.complete[l0];
            const long long n = G.members(c);
            const long long mult = G.corners[c] == 0 ? n*(n-1)/2 : 0;
            const long long base = lamOff + MP[k0] + within[l0];
            for (long long j=0; j<mult; ++j) lg.push_back(known ? (index_t)(base + j) : -1);
            within[l0] += mult;
            if (G.corners[c] >= 2)
            {
                pg.push_back(known ? (index_t)(priOff + VP[k0] + withinV[l0]) : -1);
                withinV[l0] += 1;
            }
        }
        for (size_t q=0; q<dims.size(); ++q)
            for (size_t c=0; c<comps.size(); ++c)
            {
                const std::vector<patchComponent>& g = comps[c];
                const index_t d = dims[q];
                if (g[0].dim() != d || g.size() < 2 || !componentUsed(P, r, g)) continue;
                if (!P.complete[g[0].patch()]) { pg.push_back(-1); continue; }
                const long long id = avgId[d][P.compPosition(d, P.gid[g[0].patch()], g[0].index())];
                GISMO_ENSURE(id >= 0, "Average on a component of patch " << P.gid[g[0].patch()]
                             << " is not counted by its owner.");
                pg.push_back((index_t)(priOff + avgBase[d] + id));
            }
        lamOff += MP[N];
        priOff += VP[N] + nAvg;
    }
    GISMO_ENSURE(lamOff < std::numeric_limits<index_t>::max() && unknowns < std::numeric_limits<index_t>::max(),
                 "Global sizes exceed index_t.");
    P.nLambda = (index_t)lamOff;
    P.nPrimal = (index_t)priOff;
    P.nUnknowns = (index_t)unknowns;
}

/// Primal constraints of the given local patches (vertex values and the
/// averages of the shared edges and faces)
void buildPrimals(Problem& P, const Options& o, const std::vector<index_t>& localPatches)
{
    for (index_t r=0; r<P.dim; ++r)
    {
        P.mapper[r].cornersAsPrimals();
        if (o.primal != "v")
            P.mapper[r].interfaceAveragesAsPrimals(P.mp, 1, localPatches);
        if (o.primal == "vef" && P.dim==3)
            P.mapper[r].interfaceAveragesAsPrimals(P.mp, 2, localPatches);
        GISMO_ENSURE(P.mapper[r].nPrimalDofs() == (index_t)P.primalGlobal[r].size(),
                     "Primal dofs of the local mapper and of the numbering differ.");
    }
}

/// Jump matrices of the given local patches
void buildJumps(Problem& P, const std::vector<index_t>& localPatches)
{
    for (index_t r=0; r<P.dim; ++r)
    {
        P.mapper[r].computeJumpMatrices(true, true, localPatches);
        GISMO_ENSURE(P.mapper[r].nLagrangeMultipliers() == (index_t)P.lambdaGlobal[r].size(),
                     "Multipliers of the local mapper and of the numbering differ.");
    }
}

/// Local indices of the given global patches
std::vector<index_t> localIndices(const Problem& P, const std::vector<index_t>& patches)
{
    std::vector<index_t> out(patches.size());
    for (size_t q=0; q<patches.size(); ++q) out[q] = P.loc(patches[q]);
    return out;
}

/// Whether the numbering of a problem built on all patches is that of
/// gsIetiMapper (identity of local and global indices)
bool numberingIsIdentity(const Problem& P)
{
    index_t lo = 0, po = 0;
    for (index_t r=0; r<P.dim; ++r)
    {
        for (size_t j=0; j<P.lambdaGlobal[r].size(); ++j)
            if (P.lambdaGlobal[r][j] != lo + (index_t)j) return false;
        for (size_t j=0; j<P.primalGlobal[r].size(); ++j)
            if (P.primalGlobal[r][j] != po + (index_t)j) return false;
        lo += P.lambdaGlobal[r].size();
        po += P.primalGlobal[r].size();
    }
    return lo == P.nLambda && po == P.nPrimal;
}

/// The problem on all patches with the global numbering, for the checks
/// (serial reference, direct solve, topology); its numbering must be that
/// of gsIetiMapper on all patches
void buildReference(const Options& o, Problem& R, int& status, std::string& failures)
{
    Timers t;
    Memory m;
    buildGlobal(o, R, t, m);
    std::vector<index_t> all(R.nPatches);
    for (index_t k=0; k<R.nPatches; ++k) all[k] = k;
    buildLocal(o, R, all, t, m);
    Counts C(R);
    countPatches(R, o, all, C);
    numberLocal(R, o, C);
    buildPrimals(R, o, std::vector<index_t>());
    buildJumps(R, std::vector<index_t>());
    if (!numberingIsIdentity(R)) { status = 1; failures += " numbering"; }
}

/// Free dofs of global patch k (all components) with the dofs of its
/// clamped sides eliminated; the weight of the linkcell partition
index_t patchFreeDofs(const Problem& P, const Options& o, index_t k)
{
    gsBasis<>::uPtr b = P.global.patch(k).basis().clone();
    b->setDegreePreservingMultiplicity(o.degree);
    for (index_t i=0; i<o.refine; ++i) b->uniformRefine();
    std::vector<index_t> el;
    for (size_t s=0; s<P.clampSides.size(); ++s)
        if (P.clampSides[s].patch == k)
        {
            const gsMatrix<index_t> bd = b->boundary(P.clampSides[s].side());
            el.insert(el.end(), bd.data(), bd.data()+bd.size());
        }
    std::sort(el.begin(), el.end());
    el.erase(std::unique(el.begin(), el.end()), el.end());
    return P.dim * (b->size() - (index_t)el.size());
}

//---------------------------------------------------------------------------
// Patch-local data
//---------------------------------------------------------------------------

/// Jump matrix of global patch k (local mapper), one row per multiplier of
/// the patch; rows receives the global indices of these rows, ascending
JumpMatrix patchJump(const Problem& P, index_t k, std::vector<index_t>& rows)
{
    const index_t l = P.loc(k);
    gsSparseEntries<real_t> se;
    rows.clear();
    index_t offset = 0;
    for (index_t r=0; r<P.dim; ++r)
    {
        const JumpMatrix& J = P.mapper[r].jumpMatrix(l);
        for (index_t i=0; i<J.outerSize(); ++i)
            for (JumpMatrix::InnerIterator it(J, i); it; ++it)
            {
                const index_t g = P.lambdaGlobal[r][it.row()];
                GISMO_ENSURE(g >= 0, "Multiplier of patch " << k << " has no global index.");
                rows.push_back(g);
                se.add(g, offset + it.col(), it.value());
            }
        offset += J.cols();
    }
    std::sort(rows.begin(), rows.end());
    rows.erase(std::unique(rows.begin(), rows.end()), rows.end());
    for (index_t i=0; i<(index_t)se.size(); ++i)
        se[i] = gsSparseEntries<real_t>::Triplet(
            (index_t)(std::lower_bound(rows.begin(), rows.end(), se[i].row()) - rows.begin()),
            se[i].col(), se[i].value());
    JumpMatrix B(rows.size(), offset);
    B.setFrom(se);
    return B;
}

/// The jump matrix with the global row indices
JumpMatrix expandRows(const JumpMatrix& B, const std::vector<index_t>& rows, index_t n)
{
    gsSparseEntries<real_t> se;
    se.reserve(B.nonZeros());
    for (index_t i=0; i<B.outerSize(); ++i)
        for (JumpMatrix::InnerIterator it(B, i); it; ++it)
            se.add(rows[it.row()], it.col(), it.value());
    JumpMatrix R(n, B.cols());
    R.setFrom(se);
    return R;
}

struct Raw
{
    SparseMatrix              A;
    Matrix                    f;
    JumpMatrix                B;        ///< one row per multiplier of the patch
    std::vector<index_t>      rows;     ///< global index of each row of B
    std::vector<index_t>      skeleton;
    std::vector<SparseVector> constraints;
    std::vector<index_t>      primalIdx;
};

/// Stiffness matrix, right-hand side, jump matrix, skeleton and primal
/// constraints of global patch k
void assembleRaw(const Problem& P, index_t k, Raw& out)
{
    const index_t dim = P.dim;
    const index_t l = P.loc(k);

    gsMultiPatch<>  mpl = P.mp[l];
    gsMultiBasis<>  mbl(P.mb[l]);
    std::vector<gsBoundaryConditions<> > bcl(dim);
    for (index_t r=0; r<dim; ++r)
    {
        P.bc[r].getConditionsForPatch(l, bcl[r]);
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
        P.mapper[r].initFeSpace(v[r], l);
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

    out.B = patchJump(P, k, out.rows);

    out.skeleton.clear();
    out.constraints.clear();
    out.primalIdx.clear();
    index_t offset = 0;
    const index_t n = out.A.rows();
    for (index_t r=0; r<dim; ++r)
    {
        const std::vector<index_t> sd = P.mapper[r].skeletonDofs(l);
        for (size_t j=0; j<sd.size(); ++j) out.skeleton.push_back(sd[j]+offset);
        const std::vector<SparseVector>& cond = P.mapper[r].primalConstraints(l);
        for (size_t j=0; j<cond.size(); ++j)
        {
            SparseVector tmp(n);
            for (SparseVector::InnerIterator it(cond[j]); it; ++it)
                tmp[offset + it.row()] = it.value();
            out.constraints.push_back(give(tmp));
        }
        const std::vector<index_t>& ind = P.mapper[r].primalDofIndices(l);
        for (size_t j=0; j<ind.size(); ++j)
        {
            const index_t g = P.primalGlobal[r][ind[j]];
            GISMO_ENSURE(g >= 0, "Primal dof of patch " << k << " has no global index.");
            out.primalIdx.push_back(g);
        }
        offset += P.mapper[r].jumpMatrix(l).cols();
    }
    GISMO_ASSERT(offset == n, "Local dimension mismatch on patch " << k);
}

/// Mass matrix of the scalar space on a patch, including all dofs
SparseMatrix assembleMass(const Problem& P, index_t k)
{
    gsMultiPatch<> mpl = P.mp[P.loc(k)];
    gsMultiBasis<> mbl(P.mb[P.loc(k)]);
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
    const index_t l = P.loc(k);
    for (index_t r=0; r<P.dim; ++r)
    {
        const index_t sz = P.mapper[r].jumpMatrix(l).cols();
        Matrix cr = P.mapper[r].incorporateFixedPart(l, u.middleRows(offset, sz));
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

#ifdef IETI_WITH_LINKCELL
/// Patch centroids, in physical space (z = 0 in 2D)
std::vector<double> patchCentroids(const Problem& P)
{
    std::vector<double> xyz(3*P.nPatches, 0.0);
    for (index_t k=0; k<P.nPatches; ++k)
    {
        gsMatrix<> supp = P.global[k].support();
        gsMatrix<> pt(P.global.parDim(),1), x;
        for (index_t i=0; i<P.global.parDim(); ++i) pt(i,0) = 0.5*(supp(i,0)+supp(i,1));
        P.global[k].eval_into(pt, x);
        for (index_t i=0; i<x.rows(); ++i) xyz[3*k+i] = x(i,0);
    }
    return xyz;
}

/// Box that contains all points with a margin; the margin exceeds every
/// cutoff used, so that periodic images never produce pairs.
lc_cell lcBox(const std::vector<double>& xyz, double margin)
{
    double lo[3] = {1e300,1e300,1e300}, hi[3] = {-1e300,-1e300,-1e300};
    for (size_t i=0; i<xyz.size()/3; ++i)
        for (int d=0; d<3; ++d)
        {
            lo[d] = std::min(lo[d], xyz[3*i+d]);
            hi[d] = std::max(hi[d], xyz[3*i+d]);
        }
    lc_cell c;
    c.ax = hi[0]-lo[0]+4*margin; c.ay = 0; c.az = 0;
    c.bx = 0; c.by = hi[1]-lo[1]+4*margin; c.bz = 0;
    c.cx = 0; c.cy = 0; c.cz = hi[2]-lo[2]+4*margin;
    c.ox = lo[0]-2*margin; c.oy = lo[1]-2*margin; c.oz = lo[2]-2*margin;
    return c;
}

/// Largest patch extent
double patchExtent(const Problem& P)
{
    double e = 0;
    for (index_t k=0; k<P.nPatches; ++k)
        for (index_t d=0; d<P.dim; ++d)
            e = std::max(e, (double)(P.global[k].coefs().col(d).maxCoeff() - P.global[k].coefs().col(d).minCoeff()));
    return e;
}

/// Rank of every patch from linkcell: the patch centroids are binned with
/// cells of one patch extent, every cell carries the summed cost of its
/// patches and the cells are cut into compact parts.
std::vector<int> ownerLinkcell(const Problem& P, const Options& o, int parts)
{
    const std::vector<double> xyz = patchCentroids(P);
    const double ext = patchExtent(P);
    const lc_cell box = lcBox(xyz, 2*ext);
    const double cutoff = 1.5*ext;
    const size_t n = P.nPatches;
    size_t ncell = 0;
    int nxyz[3];
    double width[3];
    GISMO_ENSURE(0 == lc_mesh_cut(xyz.data(), n, &box, cutoff, NULL, ext, parts,
        (uint8_t)o.lcMethod, 0.0, 0, nxyz, width, NULL, NULL, 0, 0, &ncell),
        "lc_mesh_cut query failed: " << lc_last_error());
    std::vector<int> domain(ncell), pointCell(n);
    GISMO_ENSURE(0 == lc_mesh_cut(xyz.data(), n, &box, cutoff, NULL, ext, parts,
        (uint8_t)o.lcMethod, 0.0, 0, nxyz, width, domain.data(), pointCell.data(), ncell, n, &ncell),
        "lc_mesh_cut failed: " << lc_last_error());
    std::vector<uint64_t> weight(ncell, 0);
    for (index_t k=0; k<P.nPatches; ++k)
    {
        const index_t nk = patchFreeDofs(P, o, k);
        weight[pointCell[k]] += (o.cost == "n2") ? (uint64_t)nk*(uint64_t)nk : (uint64_t)nk;
    }
    GISMO_ENSURE(0 == lc_partition_weights(nxyz, weight.data(), ncell, parts,
        (uint8_t)o.lcMethod, 0.0, domain.data()), "lc_partition_weights failed: " << lc_last_error());
    std::vector<int> owner(n);
    for (size_t k=0; k<n; ++k) owner[k] = domain[pointCell[k]];
    return owner;
}

/// Pairs of patches that own coincident interface dofs, found with
/// lc_pairs_within_csr on the physical images of the Greville points of all
/// coupled dofs. The cutoff is a small fraction of the element size.
std::set<std::pair<int,int> > patchPairsLinkcell(const Problem& P, const Options& o, size_t& nDofPairs)
{
    std::vector<double> pts;
    std::vector<int> pointPatch;
    const gsDofMapper& dm = P.mapper[0].dofMapperGlobal();
    double h = 1e300;
    for (index_t k=0; k<P.nPatches; ++k)
    {
        gsMatrix<> anc = P.mb[k].anchors();
        gsMatrix<> x;
        P.mp[k].eval_into(anc, x);
        for (index_t i=0; i<anc.cols(); ++i)
            if (dm.is_coupled(i,k))
            {
                for (index_t d=0; d<3; ++d) pts.push_back(d<x.rows() ? x(d,i) : 0.0);
                pointPatch.push_back((int)k);
            }
        double ext = 1e300;
        for (index_t d=0; d<P.dim; ++d)
            ext = std::min(ext, (double)(P.mp[k].coefs().col(d).maxCoeff() - P.mp[k].coefs().col(d).minCoeff()));
        h = std::min(h, ext / std::pow((double)P.mb[k].numElements(), 1.0/P.dim));
    }
    const double eps = 1e-3*h;
    const lc_cell box = lcBox(pts, 2*h);
    const size_t n = pointPatch.size();
    size_t needOff = 0, needCols = 0;
    int pbc[3] = {0,0,0};
    GISMO_ENSURE(0 == lc_pairs_within_csr(pts.data(), n, &box, pbc, eps, NULL, 2*h, 1,
        NULL, 0, NULL, NULL, 0, &needOff, &needCols), "lc_pairs_within_csr query failed: " << lc_last_error());
    std::vector<int> off(needOff), cols(needCols);
    GISMO_ENSURE(0 == lc_pairs_within_csr(pts.data(), n, &box, pbc, eps, NULL, 2*h, 1,
        off.data(), needOff, cols.data(), NULL, needCols, &needOff, &needCols),
        "lc_pairs_within_csr failed: " << lc_last_error());
    std::set<std::pair<int,int> > result;
    nDofPairs = needCols;
    for (size_t i=0; i<n; ++i)
        for (int c=off[i]; c<off[i+1]; ++c)
        {
            int a = pointPatch[i], b = pointPatch[cols[c]];
            if (a != b) result.insert(std::make_pair(std::min(a,b), std::max(a,b)));
        }
    GISMO_UNUSED(o);
    return result;
}
#endif

/// Patch pairs that share a coupled dof, from the global dof mapper
std::set<std::pair<int,int> > patchPairsMapper(const Problem& P)
{
    const gsDofMapper& dm = P.mapper[0].dofMapperGlobal();
    std::vector<std::vector<int> > groups(dm.coupledSize());
    for (index_t k=0; k<P.nPatches; ++k)
        for (index_t i=0; i<(index_t)dm.patchSize(k); ++i)
            if (dm.is_coupled(i,k)) groups[dm.cindex(i,k)].push_back((int)k);
    std::set<std::pair<int,int> > result;
    for (size_t g=0; g<groups.size(); ++g)
        for (size_t a=0; a<groups[g].size(); ++a)
            for (size_t b=a+1; b<groups[g].size(); ++b)
                result.insert(std::make_pair(std::min(groups[g][a],groups[g][b]),
                                             std::max(groups[g][a],groups[g][b])));
    return result;
}

/// Patch pairs that share a face (or edge in 2D): the interfaces of
/// gsMultiPatch::computeTopology
std::set<std::pair<int,int> > patchPairsTopology(const Problem& P)
{
    std::set<std::pair<int,int> > result;
    for (gsMultiPatch<>::const_iiterator it = P.mp.iBegin(); it != P.mp.iEnd(); ++it)
    {
        const int a = it->first().patch, b = it->second().patch;
        result.insert(std::make_pair(std::min(a,b), std::max(a,b)));
    }
    return result;
}

/// Rank of every patch
std::vector<int> ownerOf(const Problem& P, const Options& o, int size)
{
    const index_t N = P.nPatches;
    std::vector<int> owner(N, 0);
    if (o.partition == "linkcell")
    {
#ifdef IETI_WITH_LINKCELL
        if (size > 1) return ownerLinkcell(P, o, size);
        return owner;
#else
        GISMO_ERROR("--partition=linkcell needs a build with GISMO_WITH_LINKCELL.");
#endif
    }
    std::vector<index_t> order(N);
    std::iota(order.begin(), order.end(), 0);
    if (o.partition == "morton" && P.grid[0] > 0)
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
    for (int r=0; r<size; ++r)
    {
        const index_t lo = (index_t)(((long long)r*N)/size);
        const index_t hi = (index_t)(((long long)(r+1)*N)/size);
        for (index_t q=lo; q<hi; ++q) owner[order[q]] = r;
    }
    return owner;
}

struct PartStats { index_t cutDofs, nbrMax; double nbrAvg; index_t minPatches, maxPatches; };

/// Patches per rank (minimum and maximum)
PartStats partitionStats(const std::vector<int>& owner, int size)
{
    PartStats st;
    st.cutDofs = -1; st.nbrMax = -1; st.nbrAvg = -1;
    std::vector<index_t> cnt(size, 0);
    for (size_t k=0; k<owner.size(); ++k) cnt[owner[k]]++;
    st.minPatches = *std::min_element(cnt.begin(), cnt.end());
    st.maxPatches = *std::max_element(cnt.begin(), cnt.end());
    return st;
}

/// Interface size between ranks: coupled dof groups (first component) that
/// touch more than one rank, each counted by the rank of its first patch,
/// and the number of ranks adjacent to each rank, from the local patches
void cutStats(const Problem& P, const std::vector<int>& owner, Par& par, PartStats& st)
{
    const Groups G = coupledGroups(P, 0);
    long long cut = 0;
    std::set<int> nb;
    for (index_t c=0; c<G.size(); ++c)
    {
        std::set<int> rk;
        for (index_t j=G.start[c]; j<G.start[c+1]; ++j) rk.insert(owner[P.gid[G.member[j].first]]);
        if (!rk.count(par.rank)) continue;
        for (std::set<int>::const_iterator a=rk.begin(); a!=rk.end(); ++a)
            if (*a != par.rank) nb.insert(*a);
        if (owner[P.gid[G.first(c)]] == par.rank) cut += rk.size()-1;
    }
    st.cutDofs = (index_t)par.sumOf((double)cut) * P.dim;
    st.nbrMax  = par.maxOf((index_t)nb.size());
    st.nbrAvg  = par.sumOf((double)nb.size()) / par.size;
}

std::vector<index_t> distribute(const std::vector<int>& owner, int rank)
{
    std::vector<index_t> mine;
    for (size_t k=0; k<owner.size(); ++k)
        if (owner[k] == rank) mine.push_back((index_t)k);
    return mine;
}

//---------------------------------------------------------------------------
// Layout of the multiplier vector
//---------------------------------------------------------------------------

/// Layout of the multiplier vector on one rank.
///
/// Replicated: every rank holds all nLambda entries and every operator
/// application ends with one reduction of the whole vector.
///
/// Distributed: a rank holds the multipliers that touch its patches, in
/// ascending global order. A multiplier couples two patches, so it lives on
/// one rank or on two; the lower rank owns it. All vectors are kept
/// consistent (both holders carry the full value). An operator application
/// produces the contribution of the own patches and adds the contribution of
/// the other holder with one neighbourhood exchange; inner products sum the
/// owned entries and reduce one scalar.
struct Dual
{
    bool    distributed;
    index_t nGlobal, nLocal, nOwned, nShared, nNbr;
    std::vector<index_t> glob;     ///< global index of each multiplier of the own patches, ascending: the rows of the local jump matrices
    Matrix  weight;                ///< 1 for the owned multipliers, 0 for the others
    std::vector<int> nbr;          ///< adjacent ranks, ascending
    std::vector<int> cnt, dsp;     ///< shared multipliers per adjacent rank
    std::vector<int> idx;          ///< local indices of the shared multipliers, by adjacent rank, ascending global index
    mutable std::vector<double> sbuf, rbuf;
#ifdef GISMO_WITH_MPI
    MPI_Comm graph;
#endif

    Dual() : distributed(false), nGlobal(0), nLocal(0), nOwned(0), nShared(0), nNbr(0)
#ifdef GISMO_WITH_MPI
    , graph(MPI_COMM_NULL)
#endif
    {}

    index_t local(index_t g) const
    {
        const std::vector<index_t>::const_iterator it = std::lower_bound(glob.begin(), glob.end(), g);
        GISMO_ENSURE(it != glob.end() && *it == g, "Multiplier " << g << " is not held by this rank.");
        return (index_t)(it - glob.begin());
    }

    /// Number of rows of the local jump matrices
    index_t nRows() const { return (index_t)glob.size(); }

    /// A patch jump matrix whose row i is the global multiplier rows[i], with
    /// the rows of the local jump matrices (the entries of glob)
    JumpMatrix rows(const std::vector<index_t>& rows, const JumpMatrix& B) const
    {
        gsSparseEntries<real_t> se;
        se.reserve(B.nonZeros());
        for (index_t i=0; i<B.outerSize(); ++i)
            for (JumpMatrix::InnerIterator it(B, i); it; ++it)
                se.add(local(rows[it.row()]), it.col(), it.value());
        JumpMatrix R(nRows(), B.cols());
        R.setFrom(se);
        return R;
    }

    /// A vector of the layout restricted to the rows of the local jump
    /// matrices (tmp holds the result if it is not x itself)
    const Matrix& restrict(const Matrix& x, Matrix& tmp) const
    {
        if (distributed) return x;
        tmp.resize(nRows(), x.cols());
        for (index_t i=0; i<nRows(); ++i) tmp.row(i) = x.row(glob[i]);
        return tmp;
    }

    /// The consistent vector y of the layout from the contributions yl of the
    /// own patches (rows of the local jump matrices); yl is consumed
    void accumulate(Par& par, Matrix& yl, Matrix& y) const
    {
        if (!distributed)
        {
            y.setZero(nGlobal, yl.cols());
            for (index_t i=0; i<nRows(); ++i) y.row(glob[i]) = yl.row(i);
            par.sum(y);
            return;
        }
        y.swap(yl);
        exchange(par, y);
    }

    /// Adds the contributions of the adjacent ranks (distributed layout)
    void exchange(Par& par, Matrix& y) const
    {
        GISMO_UNUSED(par); GISMO_UNUSED(y);
#ifdef GISMO_WITH_MPI
        if (nbr.empty()) return;
        const double t = MPI_Wtime();
        for (size_t i=0; i<idx.size(); ++i) sbuf[i] = y(idx[i],0);
        MPI_Neighbor_alltoallv(sbuf.data(), cnt.data(), dsp.data(), MPI_DOUBLE,
                               rbuf.data(), cnt.data(), dsp.data(), MPI_DOUBLE, graph);
        for (size_t i=0; i<idx.size(); ++i) y(idx[i],0) += rbuf[i];
        par.commTime  += MPI_Wtime() - t;
        par.commCalls += 1;
        par.commBytes += 8.0*idx.size();
#endif
    }

    /// Inner product of two consistent vectors
    real_t dot(Par& par, const Matrix& a, const Matrix& b) const
    {
        if (!distributed) return a.col(0).dot(b.col(0));
        real_t s = 0;
        for (index_t i=0; i<nLocal; ++i) s += weight(i,0)*a(i,0)*b(i,0);
#ifdef GISMO_WITH_MPI
        if (par.size > 1)
        {
            const double t = MPI_Wtime();
            real_t r = s;
            MPI_Allreduce(&s, &r, 1, MPI_DOUBLE, MPI_SUM, par.comm);
            s = r;
            par.commTime  += MPI_Wtime() - t;
            par.commCalls += 1;
            par.commBytes += 8.0;
        }
#endif
        return s;
    }

    real_t norm(Par& par, const Matrix& a) const { return math::sqrt(dot(par, a, a)); }

    /// The vector in the global numbering, on every rank (for the checks)
    Matrix global(Par& par, const Matrix& x) const
    {
        if (!distributed) return x;
        Matrix g;
        g.setZero(nGlobal, 1);
        for (index_t i=0; i<nLocal; ++i)
            if (weight(i,0) != 0) g(glob[i],0) = x(i,0);
        par.sum(g);
        return g;
    }
};

/// A coupled dof of an own patch that coincides with a dof of a patch of
/// another rank: (own patch, basis index, other patch)
struct Partner
{
    index_t k, i, other;
    bool operator<(const Partner& o) const
    { return k != o.k ? k < o.k : (i != o.i ? i < o.i : other < o.other); }
    bool operator==(const Partner& o) const { return k == o.k && i == o.i && other == o.other; }
};

/// Partners from the inverse of the dof mapper of the local patches, which
/// holds the complete groups of the coupled dofs of the own patches
std::vector<Partner> partnersMapper(const Problem& P, const std::vector<int>& owner, int rank)
{
    const gsDofMapper& dm = P.mapper[0].dofMapperGlobal();
    std::vector<std::vector<index_t> > groups(dm.coupledSize());
    const index_t nl = P.gid.size();
    for (index_t l=0; l<nl; ++l)
        for (index_t i=0; i<(index_t)dm.patchSize(l); ++i)
            if (dm.is_coupled(i,l)) groups[dm.cindex(i,l)].push_back(P.gid[l]);
    std::vector<Partner> out;
    for (index_t l=0; l<nl; ++l)
    {
        const index_t k = P.gid[l];
        if (owner[k] != rank) continue;
        GISMO_ENSURE(P.complete[l], "Own patch " << k << " has neighbours that are not local.");
        for (index_t i=0; i<(index_t)dm.patchSize(l); ++i)
        {
            if (!dm.is_coupled(i,l)) continue;
            const std::vector<index_t>& g = groups[dm.cindex(i,l)];
            for (size_t a=0; a<g.size(); ++a)
                if (owner[g[a]] != rank)
                { Partner p; p.k = k; p.i = i; p.other = g[a]; out.push_back(p); }
        }
    }
    std::sort(out.begin(), out.end());
    out.erase(std::unique(out.begin(), out.end()), out.end());
    return out;
}

/// Patches of other ranks that touch a patch of this rank, from the inverse
/// of the dof mapper of the local patches
std::vector<index_t> haloPatchesMapper(const Problem& P, const std::vector<int>& owner, int rank)
{
    const std::vector<Partner> pr = partnersMapper(P, owner, rank);
    std::set<index_t> h;
    for (size_t q=0; q<pr.size(); ++q) h.insert(pr[q].other);
    return std::vector<index_t>(h.begin(), h.end());
}

#ifdef IETI_WITH_LINKCELL
/// Candidate patches of other ranks near the patches of this rank: every
/// patch whose centroid lies within sqrt(3) times the largest patch extent of
/// the centroid of an own patch (lc_pairs_within_csr on the centroids). Two
/// patches that share a vertex are never farther apart than that, since each
/// centroid lies inside its patch's bounding box.
std::vector<index_t> haloPatchesLinkcell(const Problem& P, const std::vector<int>& owner, int rank, int parts)
{
    std::vector<index_t> out;
    if (parts < 2) return out;
    const std::vector<double> xyz = patchCentroids(P);
    const double ext = patchExtent(P);
    const double cutoff = std::sqrt(3.0) * ext * (1.0 + 1e-9);
    const lc_cell box = lcBox(xyz, 2*cutoff);
    const size_t n = P.nPatches;
    size_t needOff = 0, needCols = 0;
    int pbc[3] = {0,0,0};
    GISMO_ENSURE(0 == lc_pairs_within_csr(xyz.data(), n, &box, pbc, cutoff, NULL, cutoff, 0,
        NULL, 0, NULL, NULL, 0, &needOff, &needCols), "lc_pairs_within_csr query failed: " << lc_last_error());
    std::vector<int> off(needOff), cols(std::max<size_t>(needCols, 1));
    GISMO_ENSURE(0 == lc_pairs_within_csr(xyz.data(), n, &box, pbc, cutoff, NULL, cutoff, 0,
        off.data(), needOff, cols.data(), NULL, needCols, &needOff, &needCols),
        "lc_pairs_within_csr failed: " << lc_last_error());
    std::set<index_t> h;
    for (size_t k=0; k<n; ++k)
    {
        if (owner[k] != rank) continue;
        for (int c=off[k]; c<off[k+1]; ++c)
            if (owner[cols[c]] != rank) h.insert((index_t)cols[c]);
    }
    out.assign(h.begin(), h.end());
    return out;
}

/// Partners from lc_pairs_within_csr on the physical images of the Greville
/// points of the coupled dofs of the own patches and of the halo patches.
/// The cutoff is a small fraction of the element size.
std::vector<Partner> partnersLinkcell(const Problem& P, const std::vector<int>& owner, int rank,
                                      const std::vector<index_t>& halo)
{
    std::vector<Partner> out;
    if (halo.empty()) return out;
    std::vector<index_t> patches = halo;
    for (size_t l=0; l<P.gid.size(); ++l)
        if (owner[P.gid[l]] == rank) patches.push_back(P.gid[l]);
    std::vector<double> pts;
    std::vector<index_t> pointPatch, pointDof;
    const gsDofMapper& dm = P.mapper[0].dofMapperGlobal();
    double h = 1e300;
    for (size_t q=0; q<patches.size(); ++q)
    {
        const index_t k = patches[q], l = P.loc(k);
        GISMO_ENSURE(P.complete[l], "Patch " << k << " of the pair search has neighbours that are not local.");
        gsMatrix<> anc = P.mb[l].anchors();
        gsMatrix<> x;
        P.mp[l].eval_into(anc, x);
        for (index_t i=0; i<anc.cols(); ++i)
            if (dm.is_coupled(i,l))
            {
                for (index_t d=0; d<3; ++d) pts.push_back(d<x.rows() ? x(d,i) : 0.0);
                pointPatch.push_back(k);
                pointDof.push_back(i);
            }
        double ext = 1e300;
        for (index_t d=0; d<P.dim; ++d)
            ext = std::min(ext, (double)(P.mp[l].coefs().col(d).maxCoeff() - P.mp[l].coefs().col(d).minCoeff()));
        h = std::min(h, ext / std::pow((double)P.mb[l].numElements(), 1.0/P.dim));
    }
    const double eps = 1e-3*h;
    const lc_cell box = lcBox(pts, 2*h);
    const size_t n = pointPatch.size();
    size_t needOff = 0, needCols = 0;
    int pbc[3] = {0,0,0};
    GISMO_ENSURE(0 == lc_pairs_within_csr(pts.data(), n, &box, pbc, eps, NULL, 2*h, 0,
        NULL, 0, NULL, NULL, 0, &needOff, &needCols), "lc_pairs_within_csr query failed: " << lc_last_error());
    std::vector<int> off(needOff), cols(std::max<size_t>(needCols, 1));
    GISMO_ENSURE(0 == lc_pairs_within_csr(pts.data(), n, &box, pbc, eps, NULL, 2*h, 0,
        off.data(), needOff, cols.data(), NULL, needCols, &needOff, &needCols),
        "lc_pairs_within_csr failed: " << lc_last_error());
    for (size_t a=0; a<n; ++a)
    {
        if (owner[pointPatch[a]] != rank) continue;
        for (int c=off[a]; c<off[a+1]; ++c)
        {
            const index_t ko = pointPatch[cols[c]];
            if (owner[ko] == rank) continue;
            Partner p; p.k = pointPatch[a]; p.i = pointDof[a]; p.other = ko;
            out.push_back(p);
        }
    }
    std::sort(out.begin(), out.end());
    out.erase(std::unique(out.begin(), out.end()), out.end());
    return out;
}
#endif

/// Patches of other ranks that can share multipliers with this rank, from
/// the linkcell search over the patch centroids; empty for
/// --dual-neighbours=mapper, which reads them from the local dof mapper
std::vector<index_t> haloPatchesSearch(const Problem& P, const Options& o, const std::vector<int>& owner,
                                       int rank, int parts)
{
#ifdef IETI_WITH_LINKCELL
    if (o.dualNeighbours == "linkcell") return haloPatchesLinkcell(P, owner, rank, parts);
#endif
    GISMO_ENSURE(o.dualNeighbours == "mapper", "--dual-neighbours=" << o.dualNeighbours
                 << " is not available (mapper, or linkcell with GISMO_WITH_LINKCELL).");
    GISMO_UNUSED(P); GISMO_UNUSED(owner); GISMO_UNUSED(rank); GISMO_UNUSED(parts);
    return std::vector<index_t>();
}

/// Global rows of the multipliers of a patch: all rows, and the rows per
/// column of each displacement component
struct PatchRows
{
    std::vector<index_t> all;
    std::vector<std::vector<std::vector<index_t> > > byCol;  // [component][column] -> rows
};

PatchRows patchRows(const Problem& P, index_t k, bool withColumns)
{
    PatchRows pr;
    pr.byCol.resize(P.dim);
    const index_t l = P.loc(k);
    for (index_t r=0; r<P.dim; ++r)
    {
        const JumpMatrix& J = P.mapper[r].jumpMatrix(l);
        if (withColumns) pr.byCol[r].resize(J.cols());
        for (index_t i=0; i<J.outerSize(); ++i)
            for (JumpMatrix::InnerIterator it(J, i); it; ++it)
            {
                const index_t g = P.lambdaGlobal[r][it.row()];
                GISMO_ENSURE(g >= 0, "Multiplier of patch " << k << " has no global index.");
                pr.all.push_back(g);
                if (withColumns) pr.byCol[r][it.col()].push_back(g);
            }
    }
    std::sort(pr.all.begin(), pr.all.end());
    pr.all.erase(std::unique(pr.all.begin(), pr.all.end()), pr.all.end());
    return pr;
}

/// Builds the layout of the multiplier vector of this rank.
///
/// Distributed layout: the partners give, for every adjacent rank, the
/// multipliers in the columns of the dofs that coincide with a dof of that
/// rank (a superset of the shared ones). The adjacent ranks exchange these
/// lists once over the neighbourhood graph; the multipliers shared by two
/// ranks are the intersection of the two lists, in ascending global order
/// on both sides.
///
/// With emulate > 0 (one process, rank 0 of emulate ranks) the lists of the
/// other side are the rows of the jump matrices of the halo patches, which
/// the caller has built.
Dual buildDual(const Problem& P, const Problem& Pchk, const Options& o, Par& par, const std::vector<int>& owner,
               const std::vector<index_t>& mine, const std::vector<index_t>& halo, int& status,
               std::string& failures)
{
    Dual D;
    D.nGlobal = P.nLambda;
    D.distributed = (o.multipliers == "distributed");
    if (!D.distributed)
    {
        for (size_t m=0; m<mine.size(); ++m)
        {
            const PatchRows pr = patchRows(P, mine[m], false);
            D.glob.insert(D.glob.end(), pr.all.begin(), pr.all.end());
        }
        std::sort(D.glob.begin(), D.glob.end());
        D.glob.erase(std::unique(D.glob.begin(), D.glob.end()), D.glob.end());
        D.nLocal = D.nOwned = P.nLambda;
        return D;
    }
    const int rank = par.rank;
    const bool emulate = o.emulateRanks > 0;

    std::vector<Partner> partners;
#ifdef IETI_WITH_LINKCELL
    if (o.dualNeighbours == "linkcell")
    {
        partners = partnersLinkcell(P, owner, rank, halo);
        if (o.check || o.checkTopology)
        {
            // The pair search must find the couplings of the dof mapper, on
            // every rank: the counts are summed, so that rank 0 reports a
            // mismatch of any rank, and each rank with a mismatch prints it.
            const std::vector<Partner> ref = partnersMapper(Pchk, owner, rank);
            std::vector<Partner> missing, extra;
            std::set_difference(ref.begin(), ref.end(), partners.begin(), partners.end(),
                                std::back_inserter(missing));
            std::set_difference(partners.begin(), partners.end(), ref.begin(), ref.end(),
                                std::back_inserter(extra));
            const std::vector<index_t> refHalo = haloPatchesMapper(Pchk, owner, rank);
            std::vector<index_t> haloMissing;
            std::set_difference(refHalo.begin(), refHalo.end(), halo.begin(), halo.end(),
                                std::back_inserter(haloMissing));
            if (!missing.empty() || !extra.empty() || !haloMissing.empty())
            {
                std::cout << "# dual-partners rank=" << rank << " missing=" << missing.size()
                          << " extra=" << extra.size() << " halo=" << halo.size()
                          << " halo_missing=" << haloMissing.size();
                if (!haloMissing.empty())
                    std::cout << " first_halo_missing=" << haloMissing[0]
                              << " owner=" << owner[haloMissing[0]];
                if (!missing.empty())
                    std::cout << " first_missing=" << missing[0].k << ":" << missing[0].i
                              << ":" << missing[0].other;
                std::cout << std::endl;
            }
            if (par.sumOf((double)(missing.size() + extra.size())) > 0)
            { status = 1; failures += " dual-partners"; }
            if (par.sumOf((double)haloMissing.size()) > 0)
            { status = 1; failures += " dual-halo"; }
        }
    }
    else
#endif
        partners = partnersMapper(P, owner, rank);

    // Local multipliers and the candidates per adjacent rank
    std::map<int, std::vector<long long> > cand;
    size_t q = 0;
    for (size_t m=0; m<mine.size(); ++m)
    {
        const index_t k = mine[m];
        const PatchRows pr = patchRows(P, k, true);
        D.glob.insert(D.glob.end(), pr.all.begin(), pr.all.end());
        while (q < partners.size() && partners[q].k < k) ++q;
        for (; q < partners.size() && partners[q].k == k; ++q)
        {
            std::vector<long long>& c = cand[owner[partners[q].other]];
            for (index_t r=0; r<P.dim; ++r)
            {
                const gsDofMapper& dl = P.mapper[r].dofMapperLocal(P.loc(k));
                if (!dl.is_free(partners[q].i, 0)) continue;
                const std::vector<index_t>& rows = pr.byCol[r][dl.index(partners[q].i, 0)];
                c.insert(c.end(), rows.begin(), rows.end());
            }
        }
    }
    std::sort(D.glob.begin(), D.glob.end());
    D.glob.erase(std::unique(D.glob.begin(), D.glob.end()), D.glob.end());
    D.nLocal = (index_t)D.glob.size();
    for (std::map<int, std::vector<long long> >::iterator it = cand.begin(); it != cand.end(); ++it)
    {
        std::sort(it->second.begin(), it->second.end());
        it->second.erase(std::unique(it->second.begin(), it->second.end()), it->second.end());
    }

    std::vector<std::vector<long long> > other;   // candidates of the adjacent ranks, by adjacent rank
    if (emulate)
    {
        std::map<int, std::vector<long long> > rowsOf;
        for (size_t hq=0; hq<halo.size(); ++hq)
        {
            const PatchRows pr = patchRows(P, halo[hq], false);
            std::vector<long long>& v = rowsOf[owner[halo[hq]]];
            v.insert(v.end(), pr.all.begin(), pr.all.end());
        }
        for (std::map<int, std::vector<long long> >::iterator it = cand.begin(); it != cand.end(); ++it)
        {
            std::vector<long long>& v = rowsOf[it->first];
            std::sort(v.begin(), v.end());
            v.erase(std::unique(v.begin(), v.end()), v.end());
            D.nbr.push_back(it->first);
            other.push_back(v);
        }
    }
    else
    {
#ifdef GISMO_WITH_MPI
        std::vector<int> first;
        for (std::map<int, std::vector<long long> >::iterator it = cand.begin(); it != cand.end(); ++it)
            first.push_back(it->first);
        D.nbr = par.symmetricNeighbours(first);
        D.graph = par.neighbourGraph(D.nbr);
        const int deg = (int)D.nbr.size();
        std::vector<int> scnt(std::max(deg,1), 0), sdsp(std::max(deg,1), 0),
                         rcnt(std::max(deg,1), 0), rdsp(std::max(deg,1), 0);
        std::vector<long long> sendIds;
        for (int j=0; j<deg; ++j)
        {
            const std::vector<long long>& c = cand[D.nbr[j]];
            sdsp[j] = (int)sendIds.size();
            scnt[j] = (int)c.size();
            sendIds.insert(sendIds.end(), c.begin(), c.end());
        }
        MPI_Neighbor_alltoall(scnt.data(), 1, MPI_INT, rcnt.data(), 1, MPI_INT, D.graph);
        int tot = 0;
        for (int j=0; j<deg; ++j) { rdsp[j] = tot; tot += rcnt[j]; }
        std::vector<long long> recvIds(std::max(tot,1));
        sendIds.resize(std::max<size_t>(sendIds.size(), 1));
        MPI_Neighbor_alltoallv(sendIds.data(), scnt.data(), sdsp.data(), MPI_LONG_LONG,
                               recvIds.data(), rcnt.data(), rdsp.data(), MPI_LONG_LONG, D.graph);
        for (int j=0; j<deg; ++j)
            other.push_back(std::vector<long long>(recvIds.begin()+rdsp[j], recvIds.begin()+rdsp[j]+rcnt[j]));
#endif
    }

    D.weight.setOnes(D.nLocal, 1);
    D.cnt.assign(std::max<size_t>(D.nbr.size(), 1), 0);
    D.dsp.assign(std::max<size_t>(D.nbr.size(), 1), 0);
    for (size_t j=0; j<D.nbr.size(); ++j)
    {
        const std::vector<long long>& c = cand[D.nbr[j]];
        std::vector<long long> shared;
        std::set_intersection(c.begin(), c.end(), other[j].begin(), other[j].end(),
                              std::back_inserter(shared));
        D.dsp[j] = (int)D.idx.size();
        D.cnt[j] = (int)shared.size();
        if (!shared.empty()) D.nNbr += 1;
        for (size_t a=0; a<shared.size(); ++a)
        {
            const index_t l = D.local((index_t)shared[a]);
            D.idx.push_back((int)l);
            if (D.nbr[j] < rank) D.weight(l,0) = 0;
        }
    }
    D.nShared = (index_t)D.idx.size();
    D.nOwned  = (index_t)D.weight.sum();
    D.sbuf.assign(std::max<size_t>(D.idx.size(), 1), 0.0);
    D.rbuf.assign(std::max<size_t>(D.idx.size(), 1), 0.0);

    // A multiplier held by two ranks appears once in the exchange lists of
    // each; every multiplier has exactly one owner.
    {
        std::vector<int> sorted(D.idx);
        std::sort(sorted.begin(), sorted.end());
        if (std::adjacent_find(sorted.begin(), sorted.end()) != sorted.end())
        { status = 1; failures += " dual-shared-twice"; }
    }
    if (!emulate)
    {
        const double owned = par.sumOf((double)D.nOwned);
        if (owned != (double)P.nLambda)
        { status = 1; failures += " dual-ownership"; }
    }
    return D;
}

/// Seconds per CG iteration spent in the multiplier-vector arithmetic of the
/// given patches: two applications of B^T and B per patch, the three vector
/// updates and the three inner products. The local solves are left out.
double dualKernelTime(const std::vector<JumpMatrix>& B, index_t n, index_t reps)
{
    Matrix x = Matrix::Ones(n,1), p = Matrix::Ones(n,1), r = Matrix::Ones(n,1), z = Matrix::Ones(n,1), y;
    real_t sink = 0;
    const double t0 = wallTime();
    for (index_t it=0; it<reps; ++it)
    {
        for (int pass=0; pass<2; ++pass)
        {
            y.setZero(n,1);
            for (size_t k=0; k<B.size(); ++k)
            {
                Matrix w = B[k].transpose() * p;
                y += B[k] * w;
            }
        }
        sink += p.col(0).dot(y.col(0));
        x += 1e-3 * p;
        r -= 1e-3 * y;
        sink += r.col(0).squaredNorm();
        sink += r.col(0).dot(z.col(0));
        p = z + 0.5 * p;
    }
    const double t = (wallTime() - t0) / reps;
    if (sink != sink) gsWarn << "dualKernelTime: not a number\n";
    return t;
}

//---------------------------------------------------------------------------
// Distributed IETI-DP
//---------------------------------------------------------------------------


/// Counters of the inner iterations
struct InnerStats
{
    double dirCalls, dirIters, conCalls, conIters;
    InnerStats() : dirCalls(0), dirIters(0), conCalls(0), conIters(0) {}
};

/// Sides (1..2*dim) of global patch k that carry a clamp and/or an interface
std::vector<int> patchSides(const Problem& P, index_t k, bool clamp, bool ifc)
{
    std::set<int> sides;
    if (clamp)
        for (size_t i=0; i<P.clampSides.size(); ++i)
            if (P.clampSides[i].patch == k) sides.insert((int)P.clampSides[i].side());
    if (ifc)
        for (gsMultiPatch<>::const_iiterator it = P.global.iBegin(); it != P.global.iEnd(); ++it)
        {
            if (it->first().patch == k)  sides.insert((int)it->first().side());
            if (it->second().patch == k) sides.insert((int)it->second().side());
        }
    return std::vector<int>(sides.begin(), sides.end());
}

/// Homogeneous Dirichlet conditions on the given sides of a single patch,
/// one set per displacement component
std::vector<gsBoundaryConditions<> > sidesAsBc(const Problem& P, const std::vector<int>& sides)
{
    std::vector<gsBoundaryConditions<> > bc(P.dim);
    for (index_t r=0; r<P.dim; ++r)
        for (size_t i=0; i<sides.size(); ++i)
            bc[r].addCondition(patchSide(0, boxSide(sides[i])), condition_type::dirichlet,
                               const_cast<gsConstantFunction<>*>(&P.zero));
    return bc;
}

/// Conjugate gradient iteration for a symmetric positive definite sparse
/// matrix; the preconditioner is an operator. Used as inexact solver for the
/// Dirichlet problems of the scaled Dirichlet preconditioner.
class InnerCg : public Op
{
public:
    InnerCg(const SparseMatrix& A, OpPtr M, real_t tol, index_t maxit,
            memory::shared_ptr<InnerStats> st)
    : m_A(A), m_M(give(M)), m_tol(tol), m_maxit(maxit), m_st(st) {}

    void apply(const Matrix& b, Matrix& x) const
    {
        const index_t n = m_A.rows();
        x.setZero(n, 1);
        Matrix r = b, z, Ap;
        m_M->apply(r, z);
        Matrix p = z;
        real_t rz = r.col(0).dot(z.col(0));
        const real_t rz0 = rz;
        index_t it = 0;
        if (rz0 > 0)
            for (; it<m_maxit; ++it)
            {
                Ap = m_A * p;
                const real_t a = rz / p.col(0).dot(Ap.col(0));
                x += a * p;
                r -= a * Ap;
                m_M->apply(r, z);
                const real_t rzn = r.col(0).dot(z.col(0));
                if (rzn <= m_tol*m_tol*rz0) { ++it; break; }
                p = z + (rzn/rz) * p;
                rz = rzn;
            }
        m_st->dirCalls += 1;
        m_st->dirIters += it;
    }
    index_t rows() const { return m_A.rows(); }
    index_t cols() const { return m_A.rows(); }
private:
    SparseMatrix m_A;
    OpPtr m_M;
    real_t m_tol;
    index_t m_maxit;
    memory::shared_ptr<InnerStats> m_st;
};

/// Inexact solver for the local saddle point system
///     A x + C^T mu = w,  C x = z
/// (A singular on the rigid body modes, C the primal constraints) by the
/// preconditioned conjugate gradient method projected onto the null space of C
/// (Gould, Hribar and Nocedal, SIAM J. Sci. Comput. 23, 2001), with the
/// preconditioner M^{-1} that is an exact inverse of a Kronecker sum. The
/// constraint C x = z is satisfied to rounding by a particular solution
/// M^{-1} C^T (C M^{-1} C^T)^{-1} z; only the residual in the null space of C
/// is reduced iteratively. The input has the size n + nc (w, then z), the
/// output the size n + nc (x, then zero).
class ConstrainedCg : public Op
{
    typedef gsEigen::FullPivLU<gsEigen::Matrix<real_t,gsEigen::Dynamic,gsEigen::Dynamic> > DenseLDLT;
public:
    ConstrainedCg(const SparseMatrix& A, const std::vector<SparseVector>& constraints,
                  OpPtr M, real_t tol, index_t maxit, memory::shared_ptr<InnerStats> st)
    : m_A(A), m_M(give(M)), m_tol(tol), m_maxit(maxit), m_st(st)
    {
        const index_t n = A.rows(), nc = constraints.size();
        m_C.resize(nc, n);
        gsSparseEntries<real_t> se;
        for (index_t i=0; i<nc; ++i)
            for (SparseVector::InnerIterator it(constraints[i]); it; ++it)
                se.add(i, it.row(), it.value());
        m_C.setFrom(se);
        Matrix Ct;
        Ct.setZero(n, nc);
        for (index_t i=0; i<nc; ++i)
            for (SparseVector::InnerIterator it(constraints[i]); it; ++it)
                Ct(it.row(), i) = it.value();
        m_MCt.resize(n, nc);
        for (index_t i=0; i<nc; ++i)
        {
            Matrix col = Ct.col(i), res;
            m_M->apply(col, res);
            m_MCt.col(i) = res.col(0);
        }
        Matrix S = m_C * m_MCt;
        m_S.compute(S);
    }

    void apply(const Matrix& in, Matrix& out) const
    {
        const index_t n = m_A.rows(), nc = m_C.rows();
        Matrix w = in.topRows(n), z = in.bottomRows(nc);
        Matrix x;
        x.setZero(n, 1);
        if (z.norm() > 0)
            x = m_MCt * m_S.solve(z);
        static int dbg = 0;
        if (dbg < 3 && z.norm() > 0)
        {
            ++dbg;
            gsInfo << "diagnostic particular solution: |C x - z| " << (Matrix(m_C * x) - z).norm()
                   << ", |z| " << z.norm() << ", |x| " << x.norm() << ", rank of S " << m_S.rank()
                   << " of " << nc << "\n";
        }
        Matrix r = w - m_A * x;
        Matrix v;
        project(r, v);
        Matrix p = v, Ap;
        real_t rz = r.col(0).dot(v.col(0));
        const real_t rz0 = rz;
        index_t it = 0;
        if (rz0 > 0)
            for (; it<m_maxit; ++it)
            {
                Ap = m_A * p;
                const real_t a = rz / p.col(0).dot(Ap.col(0));
                x += a * p;
                r -= a * Ap;
                project(r, v);
                const real_t rzn = r.col(0).dot(v.col(0));
                if (rzn <= m_tol*m_tol*rz0) { ++it; break; }
                p = v + (rzn/rz) * p;
                rz = rzn;
            }
        static int dbg2 = 0;
        if (dbg2 < 2 && z.norm() > 0)
        {
            ++dbg2;
            Matrix Cd;
            Cd.setZero(nc, n);
            for (index_t k2=0; k2<m_C.outerSize(); ++k2)
                for (SparseMatrix::InnerIterator it2(m_C, k2); it2; ++it2)
                    Cd(it2.row(), it2.col()) = it2.value();
            gsInfo << "diagnostic constrained CG end: iterations " << it << ", |C x - z| " << (Cd*x - z).norm()
                   << ", |x| " << x.norm() << ", rz0 " << rz0 << ", rz " << rz << ", |C| " << Cd.norm() << "\n";
        }
        m_st->conCalls += 1;
        m_st->conIters += it;
        out.setZero(n+nc, 1);
        out.topRows(n) = x;
    }
    index_t rows() const { return m_A.rows() + m_C.rows(); }
    index_t cols() const { return rows(); }
private:
    /// v = M^{-1} r - M^{-1} C^T (C M^{-1} C^T)^{-1} C M^{-1} r
    void project(const Matrix& r, Matrix& v) const
    {
        m_M->apply(r, v);
        // A second pass removes the loss of feasibility by rounding that
        // accumulates over the iterations with a single projection.
        for (int pass=0; pass<2; ++pass)
        {
            Matrix t = m_S.solve(Matrix(m_C * v));
            v -= m_MCt * t;
        }
    }
    SparseMatrix m_A;
    SparseMatrix m_C;
    OpPtr m_M;
    Matrix m_MCt;
    DenseLDLT m_S;
    real_t m_tol;
    index_t m_maxit;
    memory::shared_ptr<InnerStats> m_st;
};

/// Block diagonal fast diagonalization operator for the whole patch (all
/// unknowns, natural conditions on the interfaces, Dirichlet on the clamped
/// sides) that is the exact inverse of
///   sum_k (mu + (lambda+mu) delta_ck) vol/e_k^2 (d_k u_c, d_k v_c) + alpha (u_c, v_c)
/// on the parameter domain, for a box patch with edge lengths e_k and volume
/// vol. alpha > 0 makes the blocks of floating patches invertible.
OpPtr makeNeumannFd(const Problem& P, index_t k)
{
    const index_t dim = P.dim;
    gsVector<> e(dim);
    real_t vol = 1, emax = 0;
    for (index_t d=0; d<dim; ++d)
    {
        e[d] = P.global[k].coefs().col(d).maxCoeff() - P.global[k].coefs().col(d).minCoeff();
        vol *= e[d];
        emax = std::max(emax, e[d]);
    }
    const real_t alpha = 1e-2 * P.mu * vol / (emax*emax);
    const std::vector<gsBoundaryConditions<> > bc = sidesAsBc(P, patchSides(P, k, true, false));
    gsBlockOp<>::Ptr blk = gsBlockOp<>::make(dim, dim);
    for (index_t c=0; c<dim; ++c)
    {
        gsVector<> beta(dim);
        for (index_t d=0; d<dim; ++d)
            beta[d] = (P.mu + (c==d ? P.lambda+P.mu : 0.0)) * vol / (e[d]*e[d]);
        blk->addOperator(c, c, gsPatchPreconditionersCreator<>::fastDiagonalizationOp(
            P.mb[P.loc(k)], bc[c], gsAssembler<>::defaultOptions(), alpha, beta));
    }
    return blk;
}

/// Dirichlet fast diagonalization operator on the interior dofs of the patch
/// (those not on a clamped or interface side)
OpPtr makeDirichletFd(const Problem& P, index_t k)
{
    return gsPatchPreconditionersCreator<>::elasticityFastDiagonalizationOp(
        P.mb[P.loc(k)], sidesAsBc(P, patchSides(P, k, true, true)), gsAssembler<>::defaultOptions(),
        P.lambda, P.mu, &P.mp[P.loc(k)]);
}

/// Direct solver of a sparse matrix as a linear operator
template <class Solver>
class SparseDirectOp : public Op
{
public:
    explicit SparseDirectOp(const SparseMatrix& A) : m_n(A.rows())
    {
        m_solver.compute(A);
        GISMO_ENSURE(m_solver.info() == gsEigen::Success, "Sparse factorization failed.");
    }
    void apply(const Matrix& x, Matrix& y) const { y = m_solver.solve(x); }
    index_t rows() const { return m_n; }
    index_t cols() const { return m_n; }
private:
    Solver  m_solver;
    index_t m_n;
};

/// The sparse LU factorization of gsSparseSolver (SparseLU, COLAMD) that
/// releases its copy of the input matrix after the factorization: the
/// solves use the factors and the permutations only. The copy is as large as
/// the matrix.
class LeanSparseLU : public gsSparseSolver<real_t>::LU
{
public:
    template<class M> void compute(const M& A)
    {
        gsSparseSolver<real_t>::LU::compute(A);
        const index_t n = this->m_mat.rows();
        this->m_mat.resize(n, n);
        this->m_mat.data().squeeze();
    }
};

OpPtr makeCoarseSolver(const SparseMatrix& A, const std::string& kind)
{
    if (kind == "ldlt")
        return memory::make_shared(new SparseDirectOp< gsEigen::SimplicialLDLT<
            gsEigen::SparseMatrix<real_t>, gsEigen::Lower, gsEigen::AMDOrdering<int> > >(A));
    GISMO_ENSURE(kind == "lu", "Unknown coarse solver " << kind << " (lu or ldlt).");
    return memory::make_shared(new SparseDirectOp<LeanSparseLU>(A));
}

/// Compressed rows of the coarse matrix: the union of the dense blocks
/// pdof x pdof of all patches. lists holds, for each patch, the number of its
/// primal dofs followed by their global indices.
struct CoarsePattern
{
    std::vector<int> start, col;

    CoarsePattern(index_t n, const std::vector<int>& lists)
    {
        // The patches (position of their list) that hold each primal dof
        std::vector<int> cnt(n+1, 0);
        for (size_t q=0; q<lists.size(); q += 1+lists[q])
            for (int a=0; a<lists[q]; ++a)
                ++cnt[lists[q+1+a]+1];
        for (index_t i=0; i<n; ++i) cnt[i+1] += cnt[i];
        std::vector<int> fill(cnt.begin(), cnt.end()-1), holder(cnt[n]);
        for (size_t q=0; q<lists.size(); q += 1+lists[q])
            for (int a=0; a<lists[q]; ++a)
                holder[fill[lists[q+1+a]]++] = (int)q;
        // Row i: the union of the lists of its patches
        start.assign(n+1, 0);
        std::vector<int> row;
        for (index_t i=0; i<n; ++i)
        {
            row.clear();
            for (int h=cnt[i]; h<cnt[i+1]; ++h)
                row.insert(row.end(), lists.begin()+holder[h]+1, lists.begin()+holder[h]+1+lists[holder[h]]);
            std::sort(row.begin(), row.end());
            row.erase(std::unique(row.begin(), row.end()), row.end());
            col.insert(col.end(), row.begin(), row.end());
            start[i+1] = (int)col.size();
        }
    }
    index_t nnz() const { return col.size(); }
    /// Position of entry (i,j), which must be in the pattern
    index_t pos(index_t i, index_t j) const
    {
        return std::lower_bound(col.begin()+start[i], col.begin()+start[i+1], (int)j) - col.begin();
    }
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
    IetiMpi(const Problem& P, Par& par, const Options& o, std::vector<index_t> mine, const Dual& dual)
    : m_P(P), m_par(par), m_opt(o), m_mine(give(mine)), m_dual(dual), m_stats(new InnerStats), m_schurOp(*this), m_precOp(*this) {}

    void setup(Timers& t, Memory& m);

    /// Distributed Schur complement of the IETI-DP system
    class Schur : public Op
    {
    public:
        explicit Schur(const IetiMpi& s) : m_s(s) {}
        void apply(const Matrix& x, Matrix& y) const { m_s.applySchur(x, y); }
        index_t rows() const { return m_s.m_dual.nLocal; }
        index_t cols() const { return m_s.m_dual.nLocal; }
    private:
        const IetiMpi& m_s;
    };

    /// Distributed scaled Dirichlet preconditioner
    class Prec : public Op
    {
    public:
        explicit Prec(const IetiMpi& s) : m_s(s) {}
        void apply(const Matrix& x, Matrix& y) const { m_s.applyPrec(x, y); }
        index_t rows() const { return m_s.m_dual.nLocal; }
        index_t cols() const { return m_s.m_dual.nLocal; }
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

    const InnerStats& stats() const { return *m_stats; }

private:
    void applySchur(const Matrix& x, Matrix& y) const;
    void applyPrec(const Matrix& x, Matrix& y) const;
    /// Solution of the coarse problem. With --coarse-on=root the right-hand
    /// side is needed on rank 0 only and the solution is broadcast.
    Matrix coarseSolve(const Matrix& g) const
    {
        Matrix out;
        if (m_P.nPrimal == 0) return out;
        if (!m_opt.coarseRoot)
        {
            m_coarse->apply(g, out);
            return out;
        }
        if (m_par.rank == 0)
            m_coarse->apply(g, out);
        else
            out.resize(m_P.nPrimal, 1);
        m_par.bcast(out);
        return out;
    }
    /// Sum of the primal contributions over the ranks, where coarseSolve needs it
    void sumPrimal(Matrix& g) const
    {
        if (m_opt.coarseRoot) m_par.sumToRoot(g);
        else                  m_par.sum(g);
    }
    void setupCoarse(const std::vector<int>& ti, const std::vector<int>& tj,
                     const std::vector<double>& tv, Timers& t);
    void setupCoarseRoot(const std::vector<int>& lists, const std::vector<double>& values,
                         SparseMatrix& Ac) const;

    const Problem&      m_P;
    Par&                m_par;
    Options             m_opt;
    std::vector<index_t> m_mine;
    const Dual&         m_dual;
    std::vector<OwnedPatch> m_patches;
    OpPtr               m_coarse;
    Matrix              m_fc;
    Matrix              m_rhs;
    gsScaledDirichletPrec<> m_precLocal;
    OpPtr               m_precLocalOp;
    memory::shared_ptr<InnerStats> m_stats;
    Schur               m_schurOp;
    Prec                m_precOp;
    friend class Schur;
    friend class Prec;
};

void IetiMpi::setup(Timers& t, Memory& mem)
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
        raw.B = m_dual.rows(raw.rows, raw.B);
        OwnedPatch p;
        p.k = k;
        p.n = raw.A.rows();
        p.f = raw.f;

        const bool inexactD = (m_opt.localSolver == "fastdiag" || m_opt.localSolver == "fastdiag-dirichlet");
        const bool inexact  = (m_opt.localSolver == "fastdiag");
        const index_t nc = raw.constraints.size();

        // Scaled Dirichlet preconditioner: Schur complement on the skeleton
        {
            gsScaledDirichletPrec<>::Blocks blocks =
                gsScaledDirichletPrec<>::matrixBlocks(raw.A, raw.skeleton);
            OpPtr dirichletSolver;
            if (inexactD)
            {
                OpPtr M = makeDirichletFd(m_P, k);
                GISMO_ENSURE(M->rows() == blocks.A11.rows(), "Fast diagonalization of patch " << k
                    << " has " << M->rows() << " unknowns, the interior block " << blocks.A11.rows()
                    << ": the patch is not a box with interfaces on the sides of its skeleton.");
                dirichletSolver = memory::make_shared(new InnerCg(blocks.A11, M, m_opt.innerTol,
                    m_opt.innerMaxIt, m_stats));
                if (m_opt.check && q == 0 && m_par.rank == 0)
                {
                    Matrix b = Matrix::Random(blocks.A11.rows(), 1), xi, xe;
                    dirichletSolver->apply(b, xi);
                    makeSparseCholeskySolver(blocks.A11)->apply(b, xe);
                    gsInfo << "diagnostic Dirichlet solve: relative residual "
                           << (blocks.A11*xi - b).norm()/b.norm() << ", difference to the exact solve "
                           << (xi-xe).norm()/xe.norm() << "\n";
                }
            }
            else
                dirichletSolver = makeSparseCholeskySolver(blocks.A11);
            m_precLocal.addSubdomain(
                gsScaledDirichletPrec<>::restrictJumpMatrix(raw.B, raw.skeleton).moveToPtr(),
                gsScaledDirichletPrec<>::schurComplement(blocks, dirichletSolver)
            );
        }

        const index_t m = raw.primalIdx.size();
        p.pdof = raw.primalIdx;
        p.Psi.setZero(p.n, m);
        if (inexact)
        {
            // Local saddle point system solved by projected CG; the primal
            // basis is computed with the same solver.
            gsSparseEntries<real_t> seE;
            for (index_t i=0; i<p.n; ++i) seE.add(i, i, 1.0);
            p.E.resize(p.n+nc, p.n);
            p.E.setFrom(seE);
            p.solver = memory::make_shared(new ConstrainedCg(raw.A, raw.constraints,
                makeNeumannFd(m_P, k), m_opt.innerTol, m_opt.innerMaxIt, m_stats));
            for (index_t j=0; j<m; ++j)
            {
                Matrix in;
                in.setZero(p.n+nc, 1);
                in(p.n+j, 0) = 1.0;
                Matrix out;
                p.solver->apply(in, out);
                p.Psi.col(j) = out.topRows(p.n).col(0);
            }
            if (m_opt.check && q == 0 && m_par.rank == 0)
            {
                Matrix Cd;
                Cd.setZero(nc, p.n);
                for (index_t i=0; i<nc; ++i)
                    for (SparseVector::InnerIterator it(raw.constraints[i]); it; ++it)
                        Cd(i, it.row()) = it.value();
                {
                    OpPtr Mn = makeNeumannFd(m_P, k);
                    Matrix x1 = Matrix::Random(Mn->cols(), 1), y1 = Matrix::Random(Mn->cols(), 1), Mx, My;
                    Mn->apply(x1, Mx);
                    Mn->apply(y1, My);
                    gsInfo << "diagnostic Neumann fast diagonalization: size " << Mn->rows() << " (local " << p.n
                           << "), y'Mx " << y1.col(0).dot(Mx.col(0)) << " x'My " << x1.col(0).dot(My.col(0))
                           << ", x'Mx " << x1.col(0).dot(Mx.col(0)) << "\n";
                }
                Matrix R = Cd * p.Psi;
                for (index_t i=0; i<nc; ++i) R(i,i) -= 1.0;
                SparseMatrix modA, em, embB;
                Matrix rhsB;
                gsPrimalSystem<>::incorporateConstraints(raw.constraints, false, raw.A, modA, em, embB, rhsB);
                OpPtr lu = makeSparseLUSolver(modA);
                Matrix w;
                w.setZero(p.n+nc, 1);
                w.topRows(p.n) = Matrix::Random(p.n, 1);
                Matrix se, si;
                lu->apply(w, se);
                p.solver->apply(w, si);
                gsInfo << "diagnostic constrained solve: |C Psi - I| " << R.norm()
                       << ", difference to the exact solve " << (se.topRows(p.n)-si.topRows(p.n)).norm()/se.topRows(p.n).norm()
                       << " (residual-free norm of the exact solution " << se.topRows(p.n).norm() << ")\n";
            }
        }
        else
        {
            SparseMatrix modA, embBasis;
            Matrix rhsBasis;
            gsPrimalSystem<>::incorporateConstraints(raw.constraints, m_opt.eliminateVertices,
                raw.A, modA, p.E, embBasis, rhsBasis);
            if (m_opt.localSolver == "ldlt")
                p.solver = makeSparseCholeskySolver(modA);
            else
                p.solver = memory::make_shared(new SparseDirectOp<LeanSparseLU>(modA));
            SparseMatrix Psi = gsPrimalSystem<>::primalBasis(
                p.solver, embBasis, rhsBasis, raw.primalIdx, nPrimal);
            for (index_t j=0; j<m; ++j)
                for (SparseMatrix::InnerIterator it(Psi, p.pdof[j]); it; ++it)
                    p.Psi(it.row(), j) = it.value();
        }

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

    mem.local = currentRss();

    // The wait for the slowest rank is measured separately from the coarse work.
    double t0 = wallTime();
    m_par.barrier();
    t.wait += wallTime() - t0;

    // Coarse problem
    t0 = wallTime();
    if (nPrimal > 0)
    {
        setupCoarse(ti, tj, tv, t);
        m_par.sum(fc);
    }
    m_fc = fc;
    t.coarse += wallTime() - t0;
    mem.coarse = currentRss();

    // Right-hand side of the Schur complement system
    t0 = wallTime();
    m_precLocal.setupMultiplicityScaling();
    m_precLocalOp = m_precLocal.preconditioner();

    Matrix d;
    d.setZero(m_dual.nRows(), 1);
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
    m_dual.accumulate(m_par, d, m_rhs);
    t.rhs += wallTime() - t0;
}

/// Assembles and factorizes the coarse matrix from the local triplets
/// (ti, tj, tv) of all ranks. --coarse=gather gathers the triplets on all
/// ranks. --coarse=reduce with --coarse-on=all gathers the primal indices of
/// all patches, builds the sparsity pattern on every rank and sums the values
/// with one reduction; with --coarse-on=root only rank 0 receives the primal
/// indices, builds the pattern and adds the dense blocks of one rank after
/// the other, so that no other rank holds data of the size of the coarse
/// matrix.
void IetiMpi::setupCoarse(const std::vector<int>& ti, const std::vector<int>& tj,
                          const std::vector<double>& tv, Timers& t)
{
    const index_t nPrimal = m_P.nPrimal;
    double t0 = wallTime();
    SparseMatrix Ac(nPrimal, nPrimal);
    const bool haveMatrix = !m_opt.coarseRoot || m_par.rank == 0;
    if (m_opt.coarse == "gather")
    {
        std::vector<int>    gi = m_par.gather(ti);
        std::vector<int>    gj = m_par.gather(tj);
        std::vector<double> gv = m_par.gather(tv);
        gsSparseEntries<real_t> se;
        se.reserve(gv.size());
        for (size_t i=0; i<gv.size(); ++i)
            se.add(gi[i], gj[i], gv[i]);
        Ac.setFrom(se);
    }
    else
    {
        GISMO_ENSURE(m_opt.coarse == "reduce", "Unknown coarse assembly " << m_opt.coarse
                     << " (gather or reduce).");
        std::vector<int> lists;
        for (size_t q=0; q<m_patches.size(); ++q)
        {
            lists.push_back((int)m_patches[q].pdof.size());
            for (size_t a=0; a<m_patches[q].pdof.size(); ++a)
                lists.push_back((int)m_patches[q].pdof[a]);
        }
        if (m_opt.coarseRoot)
        {
            setupCoarseRoot(lists, tv, Ac);
            Ac.makeCompressed();
            t.cassemble += wallTime() - t0;
            t0 = wallTime();
            if (haveMatrix)
                m_coarse = makeCoarseSolver(Ac, m_opt.coarseSolver);
            t.cfactor += wallTime() - t0;
            return;
        }
        const CoarsePattern pat(nPrimal, m_par.gather(lists));
        Matrix val;
        val.setZero(pat.nnz(), 1);
        for (size_t i=0; i<tv.size(); ++i)
            val(pat.pos(ti[i], tj[i]), 0) += tv[i];
        m_par.sum(val);
        if (haveMatrix)
        {
            gsSparseEntries<real_t> se;
            se.reserve(pat.nnz());
            for (index_t r=0; r<nPrimal; ++r)
                for (int p=pat.start[r]; p<pat.start[r+1]; ++p)
                    se.add(r, pat.col[p], val(p,0));
            Ac.setFrom(se);
        }
    }
    Ac.makeCompressed();
    t.cassemble += wallTime() - t0;

    t0 = wallTime();
    if (haveMatrix)
        m_coarse = makeCoarseSolver(Ac, m_opt.coarseSolver);
    t.cfactor += wallTime() - t0;
}

/// Coarse matrix on rank 0 from the primal indices (lists) and the dense
/// blocks (values, row by row per patch) of all ranks; the blocks of rank q
/// are received after those of rank q-1
void IetiMpi::setupCoarseRoot(const std::vector<int>& lists, const std::vector<double>& values,
                              SparseMatrix& Ac) const
{
    const index_t nPrimal = m_P.nPrimal;
    const std::vector<int> all = m_par.gatherToRoot(lists);
    std::vector<int> len(1, (int)lists.size());
    len = m_par.gatherToRoot(len);
    if (m_par.rank != 0)
    {
#ifdef GISMO_WITH_MPI
        MPI_Send(values.data(), (int)values.size(), MPI_DOUBLE, 0, 77, m_par.comm);
#endif
        return;
    }
    const CoarsePattern pat(nPrimal, all);
    std::vector<double> val(pat.nnz(), 0.0), buf;
    size_t p = 0;
    for (int q=0; q<m_par.size; ++q)
    {
        const std::vector<double>* v = &values;
#ifdef GISMO_WITH_MPI
        if (q > 0)
        {
            MPI_Status st;
            int n = 0;
            MPI_Probe(q, 77, m_par.comm, &st);
            MPI_Get_count(&st, MPI_DOUBLE, &n);
            buf.resize(n);
            MPI_Recv(buf.data(), n, MPI_DOUBLE, q, 77, m_par.comm, MPI_STATUS_IGNORE);
            v = &buf;
        }
#endif
        size_t t = 0;
        const size_t end = p + len[q];
        for (; p < end; p += 1+all[p])
        {
            const int m = all[p];
            for (int a=0; a<m; ++a)
                for (int b=0; b<m; ++b)
                    val[pat.pos(all[p+1+a], all[p+1+b])] += (*v)[t++];
        }
        GISMO_ENSURE(t == v->size(), "Coarse blocks of rank " << q << " do not match its primal indices.");
    }
    gsSparseEntries<real_t> se;
    se.reserve(pat.nnz());
    for (index_t r=0; r<nPrimal; ++r)
        for (int c=pat.start[r]; c<pat.start[r+1]; ++c)
            se.add(r, pat.col[c], val[c]);
    Ac.resize(nPrimal, nPrimal);
    Ac.setFrom(se);
}

void IetiMpi::applySchur(const Matrix& xin, Matrix& y) const
{
    const index_t nPrimal = m_P.nPrimal;
    Matrix tmp, yl;
    const Matrix& x = m_dual.restrict(xin, tmp);
    yl.setZero(m_dual.nRows(), 1);
    Matrix g;
    g.setZero(nPrimal, 1);

    for (size_t i=0; i<m_patches.size(); ++i)
    {
        const OwnedPatch& p = m_patches[i];
        Matrix w = p.B.transpose() * x;
        Matrix s;
        p.solver->apply(p.E * w, s);
        yl += p.B * (p.E.transpose() * s);
        if (nPrimal > 0)
        {
            Matrix gl = p.Psi.transpose() * w;
            for (size_t a=0; a<p.pdof.size(); ++a) g(p.pdof[a],0) += gl(a,0);
        }
    }
    if (nPrimal > 0)
    {
        sumPrimal(g);
        Matrix yc = coarseSolve(g);
        for (size_t i=0; i<m_patches.size(); ++i)
        {
            const OwnedPatch& p = m_patches[i];
            Matrix yc_l(p.pdof.size(), 1);
            for (size_t a=0; a<p.pdof.size(); ++a) yc_l(a,0) = yc(p.pdof[a],0);
            yl += p.B * (p.Psi * yc_l);
        }
    }
    m_dual.accumulate(m_par, yl, y);
}

void IetiMpi::applyPrec(const Matrix& xin, Matrix& y) const
{
    Matrix tmp, yl;
    m_precLocalOp->apply(m_dual.restrict(xin, tmp), yl);
    m_dual.accumulate(m_par, yl, y);
}

std::vector<Matrix> IetiMpi::reconstruct(const Matrix& lambdaIn) const
{
    Matrix tmp;
    const Matrix& lambda = m_dual.restrict(lambdaIn, tmp);
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
        sumPrimal(g);
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

/// Result of cgDistributed
struct CgInfo
{
    index_t iterations;
    real_t  error, cond;
};

/// Preconditioned conjugate gradients with the inner products of the
/// distributed multiplier layout. The recurrences, the stopping test
/// (residual norm over right-hand side norm) and the Lanczos estimate of the
/// condition number are those of gsConjugateGradient.
CgInfo cgDistributed(const Op& A, const Op& M, const Dual& D, Par& par, const Matrix& rhs,
                     Matrix& x, real_t tol, index_t maxIt)
{
    CgInfo info;
    info.iterations = 0;
    info.error = 0;
    info.cond = -1;
    std::vector<real_t> delta(1, 0), gamma;
    const real_t rhsNorm = D.norm(par, rhs);
    if (rhsNorm == 0)
    {
        x.setZero(rhs.rows(), 1);
        return info;
    }
    Matrix tmp, update, res;
    A.apply(x, tmp);
    res = rhs - tmp;
    info.error = D.norm(par, res) / rhsNorm;
    if (info.error < tol) return info;
    M.apply(res, update);
    real_t absNew = D.dot(par, res, update);
    while (info.iterations < maxIt)
    {
        ++info.iterations;
        A.apply(update, tmp);
        const real_t alpha = absNew / D.dot(par, update, tmp);
        delta.back() += 1./alpha;
        x   += alpha * update;
        res -= alpha * tmp;
        info.error = D.norm(par, res) / rhsNorm;
        if (info.error < tol) break;
        M.apply(res, tmp);
        const real_t absOld = absNew;
        absNew = D.dot(par, res, tmp);
        const real_t beta = absNew / absOld;
        update = tmp + beta * update;
        gamma.push_back(-math::sqrt(beta)/alpha);
        delta.push_back(beta/alpha);
    }
    if (info.error < tol)
    {
        gsLanczosMatrix<real_t> L(gamma, delta);
        info.cond = L.maxEigenvalue()/L.minEigenvalue();
    }
    return info;
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
        raw.B = expandRows(raw.B, raw.rows, P.nLambda);
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
    o.tol = 1e-12; o.maxIt = 500; o.innerTol = 1e-10; o.innerMaxIt = 1000;
    o.check = o.checkDirect = o.checkTopology = o.fastDiag = o.quiet = false;
    o.cost = "n2"; o.lcMethod = 1;
    o.setupOnly = false; o.emulateRanks = 0;
    o.coarse = "reduce"; o.coarseSolver = "lu"; o.coarseOn = "root";
    o.multipliers = "replicated";
#ifdef IETI_WITH_LINKCELL
    o.dualNeighbours = "linkcell";
#else
    o.dualNeighbours = "mapper";
#endif
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
    cmd.addString("",  "local-solver", "Local solver: lu, ldlt (ldlt only with --eliminate-vertices and primal v) fastdiag-dirichlet (inner CG preconditioned by fast diagonalization for the Dirichlet solves of the preconditioner, exact constrained solves) or fastdiag (inexact Dirichlet and constrained solves; box patches only)", o.localSolver);
    cmd.addReal  ("",  "inner-tol", "Relative tolerance of the inner CG of --local-solver=fastdiag", o.innerTol);
    cmd.addInt   ("",  "inner-maxit", "Maximum number of inner CG iterations", o.innerMaxIt);
    cmd.addString("",  "partition", "Patch distribution: lex, morton or linkcell (needs GISMO_WITH_LINKCELL)", o.partition);
    cmd.addString("",  "cost",      "Patch weight of the linkcell partition: n (unknowns) or n2 (squared)", o.cost);
    cmd.addInt   ("",  "lc-method", "linkcell partition method (1 Hilbert, 3 Hilbert + Fiduccia-Mattheyses)", o.lcMethod);
    cmd.addSwitch("",  "check-topology", "Compare the interfaces found by linkcell with gsMultiPatch::computeTopology", o.checkTopology);
    cmd.addReal  ("t", "tol",       "Relative residual tolerance of the CG solver", o.tol);
    cmd.addInt   ("",  "maxit",     "Maximum number of CG iterations", o.maxIt);
    cmd.addSwitch("",  "check",     "Compare with the serial IETI-DP of the library classes", o.check);
    cmd.addSwitch("",  "check-direct", "Compare with a direct solve of the global system (gsElasticityAssembler)", o.checkDirect);
    cmd.addSwitch("q", "quiet",     "Print only the result line", o.quiet);
    cmd.addString("",  "tag",       "Label copied to the result line", o.tag);
    cmd.addSwitch("",  "setup-only", "Stop after the coarse problem and print the timing split of the setup", o.setupOnly);
    cmd.addInt   ("",  "emulate-ranks", "With --setup-only on one rank: build the local patches of rank 0 of this many ranks (the counts of the other ranks for the global numbering are built one rank at a time) and stop after the mapper phase", o.emulateRanks);
    cmd.addString("",  "coarse",    "Assembly of the coarse matrix: gather (triplets on all ranks) or reduce (pattern from the primal indices; with --coarse-on=root the dense blocks are added on rank 0, with --coarse-on=all the values are summed with one reduction)", o.coarse);
    cmd.addString("",  "coarse-solver", "Coarse solver: lu (SparseLU, COLAMD ordering) or ldlt (SimplicialLDLT, AMD ordering)", o.coarseSolver);
    cmd.addString("",  "coarse-on", "Where the coarse matrix is factorized: root (rank 0, the coarse solutions are broadcast) or all (every rank)", o.coarseOn);
    cmd.addString("",  "multipliers", "Multiplier vector: replicated (every rank holds all multipliers, one reduction per operator application) or distributed (a rank holds the multipliers of its patches, one neighbourhood exchange per operator application)", o.multipliers);
    cmd.addString("",  "dual-neighbours", "Neighbour search of --multipliers=distributed: linkcell (halo cells and pair search over the own and the halo patches; needs GISMO_WITH_LINKCELL) or mapper (inverse of the dof mapper of the local patches)", o.dualNeighbours);
    try { cmd.getValues(argc, argv); } catch (int rv) { return rv; }
    GISMO_ENSURE(o.multipliers == "replicated" || o.multipliers == "distributed",
                 "--multipliers is replicated or distributed.");
    GISMO_ENSURE(o.coarseOn == "root" || o.coarseOn == "all", "--coarse-on is root or all.");
    o.coarseRoot = (o.coarseOn == "root");
    if (o.traction.empty()) o.traction = (o.dim==3) ? "0,0,-1" : "0,-1";

    const double tStart = wallTime();
    Timers tm;
    Memory mem;
    int status = 0;
    std::string failures;

    // Global part of the problem (every rank): patches, topology, neighbours
    Problem P;
    double t0 = wallTime();
    buildGlobal(o, P, tm, mem);
    GISMO_ENSURE(o.emulateRanks == 0 || (o.setupOnly && par.size == 1),
                 "--emulate-ranks needs --setup-only and one MPI rank.");

    GISMO_ENSURE(par.size <= P.nPatches, "More ranks than patches.");
    if (o.localSolver == "fastdiag" || o.localSolver == "fastdiag-dirichlet")
    {
        GISMO_ENSURE(!o.eliminateVertices, "--local-solver=fastdiag does not eliminate vertices.");
        // The inexact operator differs from the exact one by the inner tolerance.
        o.gateMult = o.gateL2 = o.gateComp = o.gateDirectL2 = o.gateDirectComp = 1e-6;
    }
    if (o.localSolver == "ldlt")
        GISMO_ENSURE(o.eliminateVertices && o.primal == "v",
                     "--local-solver=ldlt requires --eliminate-vertices and --primal=v.");

    double tPart0 = wallTime();
    const int parts = o.emulateRanks > 0 ? (int)o.emulateRanks : par.size;
    const bool emulate = o.emulateRanks > 0;
    const std::vector<int> owner = ownerOf(P, o, parts);
    const double tPartition = wallTime() - tPart0;
    if (o.checkTopology)
    {
        // Every rank computes the partition itself; all must hold the same one
        double sig = 0;
        for (size_t k=0; k<owner.size(); ++k) sig += (double)(k+1) * (double)(owner[k]+1);
        if (par.maxOf(sig) != -par.maxOf(-sig))
        {
            status = 1; failures += " owner-differs";
            std::cout << "# owner-differs rank=" << par.rank << " signature=" << sig << std::endl;
        }
    }
    PartStats pst = partitionStats(owner, parts);
    GISMO_ENSURE(pst.minPatches > 0, "Partition leaves a rank without patches.");
    std::vector<index_t> mine = distribute(owner, par.rank);

    // Patches of other ranks near the own ones from the linkcell search
    // (distributed multipliers)
    const bool distributed = (o.multipliers == "distributed");
    double t1 = wallTime();
    std::vector<index_t> haloSearch;
    if (distributed)
        haloSearch = haloPatchesSearch(P, o, owner, par.rank, parts);
    tm.dual += wallTime() - t1;

    // Contributions of the other ranks to the global numbering; the
    // emulation of one rank builds them one rank at a time
    Counts counts(P);
    if (emulate)
    {
        Timers tq;
        Memory mq;
        for (int q=1; q<parts; ++q)
        {
            const std::vector<index_t> own = distribute(owner, q);
            buildLocal(o, P, ring(P, own, 1), tq, mq);
            countPatches(P, o, localIndices(P, own), counts);
        }
    }

    // Local part: the own patches and two layers of neighbours, so that the
    // coupled dofs of the own patches and of their neighbours have complete
    // groups (one more layer for the emulation, which reads the multipliers
    // of the halo patches)
    {
        std::vector<index_t> local = ring(P, mine, emulate ? 3 : 2);
        if (!haloSearch.empty())
        {
            const std::vector<index_t> h = ring(P, haloSearch, emulate ? 2 : 1);
            local.insert(local.end(), h.begin(), h.end());
            std::sort(local.begin(), local.end());
            local.erase(std::unique(local.begin(), local.end()), local.end());
        }
        buildLocal(o, P, local, tm, mem);
    }
    t1 = wallTime();
    const std::vector<index_t> mineLocal = localIndices(P, mine);
    countPatches(P, o, mineLocal, counts);
    par.sumAll(counts.data);
    numberLocal(P, o, counts);
    tm.dofmap += wallTime() - t1;
    mem.dofmap = currentRss();

    t1 = wallTime();
    buildPrimals(P, o, mineLocal);
    tm.primals += wallTime() - t1;
    mem.primals = currentRss();

    t1 = wallTime();
    std::vector<index_t> halo = haloSearch;
    if (distributed && o.dualNeighbours == "mapper")
        halo = haloPatchesMapper(P, owner, par.rank);
    tm.dual += wallTime() - t1;
    t1 = wallTime();
    {
        std::vector<index_t> jumpSet = mine;
        if (emulate) jumpSet.insert(jumpSet.end(), halo.begin(), halo.end());
        std::sort(jumpSet.begin(), jumpSet.end());
        buildJumps(P, localIndices(P, jumpSet));
    }
    tm.jumps += wallTime() - t1;
    mem.jumps = currentRss();
    tm.mapper = wallTime() - t0;

    // The problem on all patches, on every rank, for the checks
    Problem R;
    const bool needReference = o.check || o.checkDirect || o.checkTopology;
    if (needReference)
    {
        buildReference(o, R, status, failures);
        if (R.nLambda != P.nLambda || R.nPrimal != P.nPrimal || R.nUnknowns != P.nUnknowns)
        { status = 1; failures += " numbering-sizes"; }
    }

    t1 = wallTime();
    const Dual dual = buildDual(P, needReference ? R : P, o, par, owner, mine, halo, status, failures);
    tm.dual += wallTime() - t1;
    mem.dual = currentRss();
    if (par.maxOf((index_t)status) != 0 && status == 0)
    { status = 1; failures += " dual-layout-on-another-rank"; }
    if (!emulate) cutStats(P, owner, par, pst);


    // Timing split of the setup (maximum over the ranks)
    struct SetupReport
    {
        static void print(const Par& par, const Options& o, const Problem& P, const Timers& t,
                          const Memory& m, double tPartition, index_t owned, const char* stage)
        {
            const double v[] = { t.mapper, t.topo, t.basis, t.dofmap, t.primals, t.jumps,
                                 t.assembly, t.factor, t.wait, t.coarse, t.cassemble, t.cfactor };
            const char* k[] = { "t_mapper", "t_topo", "t_basis", "t_dofmap", "t_primals", "t_jumps",
                                "t_assembly", "t_factor", "t_wait", "t_coarse", "t_cassemble", "t_cfactor" };
            std::ostringstream os;
            os.precision(6);
            os << "SETUP stage=" << stage << " tag=" << (o.tag.empty() ? "-" : o.tag)
               << " ranks=" << par.size << " emulate=" << o.emulateRanks
               << " patches=" << P.nPatches << " owned=" << owned
               << " grid=" << P.grid[0] << "x" << P.grid[1] << "x" << P.grid[2]
               << " degree=" << o.degree << " refine=" << o.refine
               << " unknowns=" << P.nUnknowns << " multipliers=" << P.nLambda << " primal=" << P.nPrimal
               << " primal_type=" << o.primal << " partition=" << o.partition
               << " coarse=" << o.coarse << " coarse_solver=" << o.coarseSolver
               << " coarse_root=" << (o.coarseRoot ? 1 : 0)
               << " t_partition=" << par.maxOf(tPartition);
            for (size_t i=0; i<sizeof(v)/sizeof(v[0]); ++i)
                os << " " << k[i] << "=" << par.maxOf(v[i]);
            os << m.fields(par);
            if (par.rank == 0) gsInfo << os.str() << "\n";
        }
    };
    if (o.setupOnly && o.emulateRanks > 0)
    {
        SetupReport::print(par, o, P, tm, mem, tPartition, mine.size(), "mapper");
        if (dual.distributed)
        {
            // Multiplier-vector arithmetic of this rank in both layouts
            std::vector<JumpMatrix> Brep, Bloc;
            for (size_t q=0; q<mine.size(); ++q)
            {
                std::vector<index_t> rows;
                const JumpMatrix B = patchJump(P, mine[q], rows);
                Brep.push_back(expandRows(B, rows, P.nLambda));
                Bloc.push_back(dual.rows(rows, B));
            }
            const index_t reps = 20;
            const double tRep = dualKernelTime(Brep, P.nLambda, reps);
            const double tLoc = dualKernelTime(Bloc, dual.nLocal, reps);
            std::ostringstream os;
            os.precision(6);
            os << "DUAL tag=" << (o.tag.empty() ? "-" : o.tag)
               << " emulate=" << o.emulateRanks << " patches=" << P.nPatches << " owned=" << mine.size()
               << " halo_patches=" << halo.size()
               << " grid=" << P.grid[0] << "x" << P.grid[1] << "x" << P.grid[2]
               << " degree=" << o.degree << " refine=" << o.refine
               << " multipliers=" << P.nLambda << " primal=" << P.nPrimal
               << " dual_neighbours=" << o.dualNeighbours
               << " dual_local=" << dual.nLocal << " dual_owned=" << dual.nOwned
               << " dual_shared=" << dual.nShared << " dual_nbr=" << dual.nNbr
               << " t_dual=" << tm.dual
               << " t_vec_replicated=" << tRep << " t_vec_distributed=" << tLoc
               << " bytes_iter_replicated=" << 16.0*P.nLambda + 8.0*P.nPrimal
               << " bytes_iter_distributed=" << 16.0*dual.nShared + 8.0*P.nPrimal + 24.0
               << " status=" << (status ? "FAIL" : "PASS");
            if (status) os << " failures=" << failures;
            gsInfo << os.str() << "\n";
        }
        return status;
    }

    IetiMpi ieti(P, par, o, mine, dual);
    ieti.setup(tm, mem);
    if (o.setupOnly)
    {
        SetupReport::print(par, o, P, tm, mem, tPartition, mine.size(), "coarse");
        return 0;
    }

    // Solve
    Matrix lambda;
    lambda.setZero(dual.nLocal, 1);
    index_t iterations = 0;
    bool converged = false;
    real_t cond = -1;
    par.commTime = 0; par.commCalls = 0; par.commBytes = 0;
    t0 = wallTime();
    if (dual.distributed)
    {
        const CgInfo ci = cgDistributed(ieti.schur(), ieti.prec(), dual, par, ieti.rhs(), lambda, o.tol, o.maxIt);
        iterations = ci.iterations;
        converged  = ci.error < o.tol;
        cond       = ci.cond;
    }
    else
    {
        gsConjugateGradient<> PCG(memory::make_shared_not_owned(static_cast<Op*>(&ieti.schur())),
                                  memory::make_shared_not_owned(static_cast<Op*>(&ieti.prec())));
        PCG.setCalcEigenvalues(true);
        PCG.setTolerance(o.tol);
        PCG.setMaxIterations(o.maxIt);
        Matrix hist;
        PCG.solveDetailed(ieti.rhs(), lambda, hist);
        iterations = hist.rows()-1;
        converged  = hist(iterations,0) < o.tol;
        cond       = PCG.getConditionNumber();
    }
    const double tSolve = wallTime() - t0;
    tm.solve = tSolve;
    const double commSolve = par.commTime;
    const double commBytesSolve = par.commBytes;
    const index_t commCallsSolve = par.commCalls;

    // True relative residual of the multiplier system
    Matrix Fl;
    ieti.schur().apply(lambda, Fl);
    const real_t relRes = dual.norm(par, ieti.rhs() - Fl) / dual.norm(par, ieti.rhs());

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
        libraryReference(R, o, ref);
        Matrix Fref;
        ref.schur->apply(dual.global(par, lambda), Fref);
        chkMult = (Fref - ref.rhs).norm() / ref.rhs.norm();

        Matrix part(2,1);
        part.setZero();
        Matrix cref(1,1);
        cref(0,0) = 0;
        Matrix lref(1,1);
        lref(0,0) = 0;
        for (index_t k=0; k<R.nPatches; ++k)
        {
            SparseMatrix M = assembleMass(R, k);
            Matrix cf = localCoefficients(R, k, ref.u[k]);
            lref(0,0) += squaredL2(M, cf);
        }
        // reference compliance needs the local right-hand sides
        for (index_t k=0; k<R.nPatches; ++k)
        {
            Raw raw;
            assembleRaw(R, k, raw);
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
        gsConstantFunction<> body(gsVector<>::Zero(dim), dim);
        std::vector<real_t> tr = P.traction;
        gsConstantFunction<> g(gsVector<>::Map(tr.data(), dim), dim);
        gsBoundaryConditions<> bcInfo;
        for (size_t s=0; s<R.clampSides.size(); ++s)
            for (index_t d=0; d<dim; ++d)
                bcInfo.addCondition(R.clampSides[s].patch, R.clampSides[s],
                                    condition_type::dirichlet, 0, d);
        for (size_t s=0; s<R.tractionSides.size(); ++s)
            bcInfo.addCondition(R.tractionSides[s].patch, R.tractionSides[s],
                                condition_type::neumann, &g);
        gsElasticityAssembler<real_t> assembler(R.mp, R.mb, bcInfo, body);
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

    // Topology cross-check
    index_t lcPairs = -1;
    std::string topo = "NA";
    if (o.checkTopology)
    {
        const std::set<std::pair<int,int> > viaMapper = patchPairsMapper(R);
        const std::set<std::pair<int,int> > viaTopo   = patchPairsTopology(R);
        bool ok = std::includes(viaMapper.begin(), viaMapper.end(), viaTopo.begin(), viaTopo.end());
#ifdef IETI_WITH_LINKCELL
        size_t nDofPairs = 0;
        const std::set<std::pair<int,int> > viaLc = patchPairsLinkcell(R, o, nDofPairs);
        lcPairs = (index_t)viaLc.size();
        ok = ok && (viaLc == viaMapper);
        if (par.rank == 0)
            gsInfo << "topology: patch pairs linkcell=" << viaLc.size() << " dof-mapper=" << viaMapper.size()
                   << " computeTopology(faces)=" << viaTopo.size() << " coincident-dof-pairs=" << nDofPairs << "\n";
        topo = ok ? "PASS" : "FAIL";
#else
        topo = ok ? "PASS-noLinkcell" : "FAIL";
#endif
        if (!ok) { status = 1; failures += " topology"; }
    }

    tm.total = wallTime() - tStart;

    // Timers: maximum over the ranks
    const double tAssembly = par.maxOf(tm.assembly), tFactor = par.maxOf(tm.factor),
                 tCoarse   = par.maxOf(tm.coarse),   tRhs    = par.maxOf(tm.rhs),
                 tSolveMax = par.maxOf(tm.solve),    tTotal  = par.maxOf(tm.total),
                 tMapper   = par.maxOf(tm.mapper),   tComm   = par.maxOf(commSolve);
    Timers tMax;
    tMax.topo = par.maxOf(tm.topo); tMax.dofmap = par.maxOf(tm.dofmap);
    tMax.primals = par.maxOf(tm.primals); tMax.jumps = par.maxOf(tm.jumps);
    tMax.wait = par.maxOf(tm.wait); tMax.cassemble = par.maxOf(tm.cassemble);
    tMax.cfactor = par.maxOf(tm.cfactor);
    const double nCalls   = (double)par.maxOf(commCallsSolve);
    const double commBytes = par.maxOf(commBytesSolve);
    const index_t dualLocal = par.maxOf(dual.nLocal), dualShared = par.maxOf(dual.nShared),
                  dualNbr = par.maxOf(dual.nNbr);
    const double tDual = par.maxOf(tm.dual);
    const real_t condMax  = cond;
    const double dirCalls = par.sumOf(ieti.stats().dirCalls), dirIters = par.sumOf(ieti.stats().dirIters),
                 conCalls = par.sumOf(ieti.stats().conCalls), conIters = par.sumOf(ieti.stats().conIters);
    const std::string memFields = mem.fields(par);
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
           << " t_partition=" << tPartition
           << " cut_dofs=" << pst.cutDofs
           << " nbr_max=" << pst.nbrMax
           << " nbr_avg=" << pst.nbrAvg
           << " patches_min=" << pst.minPatches
           << " patches_max=" << pst.maxPatches
           << " lc_pairs=" << lcPairs
           << " topology=" << topo
           << " t_mapper=" << tMapper
           << " t_topo=" << tMax.topo
           << " t_dofmap=" << tMax.dofmap
           << " t_primals=" << tMax.primals
           << " t_jumps=" << tMax.jumps
           << " t_assembly=" << tAssembly
           << " t_factor=" << tFactor
           << " t_wait=" << tMax.wait
           << " t_coarse=" << tCoarse
           << " t_cassemble=" << tMax.cassemble
           << " t_cfactor=" << tMax.cfactor
           << " coarse=" << o.coarse
           << " coarse_solver=" << o.coarseSolver
           << " coarse_root=" << (o.coarseRoot ? 1 : 0)
           << " t_rhs=" << tRhs
           << " iterations=" << iterations
           << " t_solve=" << tSolveMax
           << " t_iter=" << tSolveMax/std::max<index_t>(iterations,1)
           << " t_total=" << tTotal
           << " t_comm=" << tComm
           << " comm_calls=" << nCalls
           << " comm_bytes=" << commBytes
           << " multiplier_layout=" << o.multipliers
           << " dual_neighbours=" << (dual.distributed ? o.dualNeighbours : std::string("-"))
           << " dual_local=" << dualLocal
           << " dual_shared=" << dualShared
           << " dual_nbr=" << dualNbr
           << " t_dual=" << tDual
           << " local_solver=" << o.localSolver
           << " inner_dir_calls=" << dirCalls
           << " inner_dir_iters=" << dirIters
           << " inner_con_calls=" << conCalls
           << " inner_con_iters=" << conIters
           << " cond=" << condMax
           << " relres=" << relRes
           << " compliance=" << compliance
           << " l2norm=" << l2norm
           << " chk_mult=" << chkMult
           << " chk_l2=" << chkL2
           << " chk_comp=" << chkComp
           << " chk_direct_l2=" << chkDirL2
           << " chk_direct_comp=" << chkDirComp
           << memFields
           << " status=" << (status ? "FAIL" : "PASS");
        if (status) os << " failures=" << failures;
        gsInfo << os.str() << "\n";
    }

    return status;
}
