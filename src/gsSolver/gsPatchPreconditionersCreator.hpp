/** @file gsPatchPreconditionersCreator.hpp

    @brief Provides preconditioners that live on the parameter domain.

    This file is part of the G+Smo library.

    This Source Code Form is subject to the terms of the Mozilla Public
    License, v. 2.0. If a copy of the MPL was not distributed with this
    file, You can obtain one at http://mozilla.org/MPL/2.0/.

    Author(s): S. Takacs, C. Hofreither
*/
#pragma once

#include <gsSolver/gsSumOp.h>
#include <gsSolver/gsProductOp.h>
#include <gsSolver/gsKroneckerOp.h>
#include <gsSolver/gsBlockOp.h>
#include <gsSolver/gsMatrixOp.h>
#include <gsAssembler/gsExprAssembler.h>
#include <gsAssembler/gsQuadrature.h>
#include <gsNurbs/gsTensorBSplineBasis.h>

#include <algorithm>

namespace gismo
{

namespace {

template<typename T>
gsBoundaryConditions<T> boundaryConditionsForDirection( const gsBoundaryConditions<T>& bc, index_t direction )
{
    gsBoundaryConditions<T> result;

    for ( index_t i = 1; i <= 2; ++i)
    {
        patchSide global(0,i+2*direction), local(0,i);
        const boundary_condition<T>* cond = bc.getConditionFromSide(global);
        if (cond)
            result.addCondition(local,cond->type(),cond->function());
    }
    return result;
}

template<typename T>
gsBoundaryConditions<T> boundaryConditionsForComponent( const gsBoundaryConditions<T>& bc, short_t component )
{
    gsBoundaryConditions<T> result;

    for ( typename gsBoundaryConditions<T>::const_bciterator it = bc.beginAll(); it != bc.endAll(); ++it )
        for ( typename gsBoundaryConditions<T>::const_iterator cond = it->second.begin(); cond != it->second.end(); ++cond )
            if ( cond->unknown() == 0 && cond->patch() == 0
                 && ( cond->unkComponent() == -1 || cond->unkComponent() == component )
                 && !result.getConditionFromSide(cond->ps) )
                result.addCondition(cond->ps,cond->type(),cond->function());
    return result;
}

template<typename T>
void eliminateDirichlet1D(const gsBoundaryConditions<T>& bc,
                          const gsOptionList& opt, gsSparseMatrix<T> & result)
{
    dirichlet::strategy ds = (dirichlet::strategy)opt.askInt("DirichletStrategy",dirichlet::elimination);
    if (ds == dirichlet::elimination)
    {
        patchSide west(0,boundary::west), east(0,boundary::east);
        index_t i = 0;
        if (bc.getConditionFromSide( west ) && bc.getConditionFromSide( west )->type() == condition_type::dirichlet ) i += 1;
        if (bc.getConditionFromSide( east ) && bc.getConditionFromSide( east )->type() == condition_type::dirichlet ) i += 2;
        if (i%2 + i/2 >= result.rows() || i%2 + i/2 >= result.cols())
            result.resize(0,0);
        else switch ( i )
             {
             case 0: break;
             case 1: result = result.block( 1, 1, result.rows()-1, result.cols()-1 ); break;
             case 2: result = result.block( 0, 0, result.rows()-1, result.cols()-1 ); break;
             case 3: result = result.block( 1, 1, result.rows()-2, result.cols()-2 ); break;
             }
    }
    else
        GISMO_ERROR("Unknown Dirichlet strategy.");
}

template<typename T>
gsSparseMatrix<T> assembleMass(const gsBasis<T>& basis)
{
    gsExprAssembler<T> mass(1,1);
    mass.setIntegrationDomain(basis.domain());
    typename gsExprAssembler<T>::space u = mass.getSpace(basis);
    mass.initMatrix();
    mass.assemble( u * u.tr() );
    gsSparseMatrix<T> result;
    mass.matrix_into(result);
    return result;
}

template<typename T>
gsSparseMatrix<T> assembleStiffness(const gsBasis<T>& basis)
{
    gsExprAssembler<T> stiff(1,1);
    stiff.setIntegrationDomain(basis.domain());
    typename gsExprAssembler<T>::space u = stiff.getSpace(basis);
    stiff.initMatrix();
    stiff.assemble( gismo::expr::grad(u) * gismo::expr::grad(u).tr() );
    gsSparseMatrix<T> result;
    stiff.matrix_into(result);
    return result;
}


// Quadrature nodes and weights of all elements of a univariate basis, with
// the values, the derivatives and the indices of the basis functions that
// are active at the nodes (one column per node)
template<typename T>
struct univariateQuadrature
{
    gsMatrix<T>       nodes;
    gsVector<T>       weights;
    gsMatrix<T>       values;
    gsMatrix<T>       derivs;
    gsMatrix<index_t> actives;
    T                 volume; // sum of the weights
};

template<typename T>
void setupUnivariateQuadrature(const gsBasis<T>& basis, const gsOptionList& opt,
                               univariateQuadrature<T>& result)
{
    typename gsQuadRule<T>::uPtr rule = gsQuadrature::getPtr(basis, opt);
    memory::shared_ptr< gsDomain<T> > domain = basis.domain();
    typename gsBasis<T>::domainIter domIt    = domain->beginAll();
    typename gsBasis<T>::domainIter domItEnd = domain->endAll();

    std::vector<T> nodes, weights;
    gsMatrix<T> elNodes;
    gsVector<T> elWeights;
    for (; domIt < domItEnd; ++domIt )
    {
        rule->mapTo( domIt.lowerCorner(), domIt.upperCorner(), elNodes, elWeights );
        for ( index_t q=0; q<elWeights.rows(); ++q )
        {
            nodes.push_back( elNodes(0,q) );
            weights.push_back( elWeights[q] );
        }
    }

    const index_t n = (index_t)nodes.size();
    result.nodes.resize(1,n);
    result.weights.resize(n);
    for ( index_t q=0; q<n; ++q )
    {
        result.nodes(0,q)  = nodes[q];
        result.weights[q]  = weights[q];
    }
    result.volume = result.weights.sum();
    basis.eval_into  (result.nodes, result.values );
    basis.deriv_into (result.nodes, result.derivs );
    basis.active_into(result.nodes, result.actives);
}

// Assembles the univariate matrix with the entries
// sum_q weights[q] coef[q] f_i(nodes[q]) f_j(nodes[q]), where the f_i are
// the derivatives of the basis functions if stiffness is true and their
// values otherwise
template<typename T>
gsSparseMatrix<T> assembleWeighted(const univariateQuadrature<T>& qu,
                                   const gsVector<T>& coef, bool stiffness, index_t sz)
{
    const gsMatrix<T>& f = stiffness ? qu.derivs : qu.values;
    const index_t K = f.rows();
    gsSparseEntries<T> entries;
    entries.reserve( K*K*f.cols() );
    for ( index_t q=0; q<f.cols(); ++q )
    {
        const T w = qu.weights[q] * coef[q];
        for ( index_t i=0; i<K; ++i )
            for ( index_t j=0; j<K; ++j )
                entries.add( qu.actives(i,q), qu.actives(j,q), w * f(i,q) * f(j,q) );
    }
    gsSparseMatrix<T> result(sz,sz);
    result.setFrom(entries);
    result.makeCompressed();
    return result;
}

// Averages of the diagonal of |det J| J^{-1} J^{-T} over the anchors of the
// basis of the geometry
template<typename T>
gsVector<T> averageMetricDiagonal(const gsGeometry<T>& geo)
{
    const index_t d = geo.domainDim();
    gsMatrix<T> pts, jacs;
    geo.basis().anchors_into(pts);
    geo.jacobian_into(pts, jacs);
    gsVector<T> result = gsVector<T>::Zero(d);
    for ( index_t i=0; i<pts.cols(); ++i )
    {
        const gsMatrix<T> J = jacs.block(0, i*d, d, d);
        const T detJ = math::abs(J.determinant());
        const gsMatrix<T> Jinv = J.inverse();
        for ( index_t k=0; k<d; ++k )
            result[k] += detJ * Jinv.row(k).squaredNorm();
    }
    result /= (T)(pts.cols());
    return result;
}

// Sums of |det J| (J^{-1})_{kc}^2 over the tensor grid of the quadrature
// nodes, where J is the Jacobian of the geometry. For the direction k, the
// component c and the direction j, result[(k*d+c)*d+j] has one entry per
// node of the direction j: the sum over the nodes of all other directions,
// weighted with their quadrature weights. The grid is traversed line by
// line, so the memory is linear in the univariate sizes.
//
// Returns false if the determinant vanishes or changes its sign.
template<typename T>
bool assembleMetricSums(const gsGeometry<T>& geo,
                        const std::vector< univariateQuadrature<T> >& qu,
                        std::vector< gsVector<T> >& result)
{
    const index_t d = (index_t)qu.size();
    result.resize(d*d*d);
    for ( index_t i=0; i<d*d; ++i )
        for ( index_t j=0; j<d; ++j )
            result[i*d+j].setZero( qu[j].weights.rows() );

    const index_t n0 = qu[0].weights.rows();
    gsMatrix<T> pts(d,n0), jacs, J, Jinv, line(d,d);
    pts.row(0) = qu[0].nodes;
    gsVector<index_t> idx = gsVector<index_t>::Zero(d);
    T sign = 0;

    // Loop over the lines in the direction 0
    while (true)
    {
        T w = 1;
        for ( index_t l=1; l<d; ++l )
        {
            pts.row(l).setConstant( qu[l].nodes(0,idx[l]) );
            w *= qu[l].weights[idx[l]];
        }
        geo.jacobian_into(pts, jacs);

        line.setZero();
        for ( index_t i=0; i<n0; ++i )
        {
            J = jacs.block(0, i*d, d, d);
            const T detJ = J.determinant();
            if (sign == 0)
                sign = detJ > 0 ? 1 : -1;
            if (!(sign * detJ > 0))
                return false;
            Jinv = J.inverse();
            for ( index_t k=0; k<d; ++k )
                for ( index_t c=0; c<d; ++c )
                {
                    const T val = sign * detJ * Jinv(k,c) * Jinv(k,c);
                    result[(k*d+c)*d][i] += w * val;
                    line(k,c) += qu[0].weights[i] * val;
                }
        }

        for ( index_t j=1; j<d; ++j )
        {
            T wj = 1;
            for ( index_t l=1; l<d; ++l )
                if (l != j)
                    wj *= qu[l].weights[idx[l]];
            for ( index_t k=0; k<d; ++k )
                for ( index_t c=0; c<d; ++c )
                    result[(k*d+c)*d+j][idx[j]] += wj * line(k,c);
        }

        // Next line
        index_t l = 1;
        for ( ; l<d; ++l )
        {
            if (++idx[l] < qu[l].weights.rows())
                break;
            idx[l] = 0;
        }
        if (l == d)
            break;
    }
    return true;
}

// Univariate matrices of the separable approximation
//   c_k(xi) ~ a_k(xi_k) prod_{j!=k} m_j(xi_j)
// of the coefficients c_k = sum_c factors(k,c) |det J| (J^{-1})_{kc}^2,
// given the sums from assembleMetricSums: stiffness matrices weighted with
// a_k and mass matrices weighted with m_k.
//
// a_k is the mean of c_k over the hyperplanes orthogonal to the direction k.
// m_j is the geometric mean over k!=j of the means of c_k over the
// hyperplanes orthogonal to the direction j, each divided by the mean of c_k
// over the whole parameter domain. All means are weighted with the
// quadrature weights. The approximation is exact if each c_k is of the
// given form; for a constant Jacobian, a_k = c_k and m_j = 1.
template<typename T>
void assembleTensorSeparable(
    const gsBasis<T>& basis,
    const gsBoundaryConditions<T>& bc,
    const gsOptionList& opt,
    const std::vector< univariateQuadrature<T> >& qu,
    const std::vector< gsVector<T> >& sums,
    const gsMatrix<T>& factors,
    std::vector< gsSparseMatrix<T> >& local_stiff,
    std::vector< gsSparseMatrix<T> >& local_mass
    )
{
    const index_t d = basis.dim();

    // mean[k*d+j]: means of c_k over the hyperplanes orthogonal to j
    std::vector< gsVector<T> > mean(d*d);
    gsVector<T> total(d);
    for ( index_t k=0; k<d; ++k )
    {
        for ( index_t j=0; j<d; ++j )
        {
            T vol = 1;
            for ( index_t l=0; l<d; ++l )
                if (l != j)
                    vol *= qu[l].volume;
            mean[k*d+j].setZero( qu[j].weights.rows() );
            for ( index_t c=0; c<d; ++c )
                mean[k*d+j] += (factors(k,c) / vol) * sums[(k*d+c)*d+j];
        }
        total[k] = qu[k].weights.dot(mean[k*d+k]) / qu[k].volume;
    }

    local_stiff.resize(d);
    local_mass.resize(d);
    for ( index_t j=0; j<d; ++j )
    {
        const gsVector<T>& a = mean[j*d+j];
        gsVector<T> m = gsVector<T>::Ones( qu[j].weights.rows() );
        for ( index_t k=0; k<d; ++k )
            if (k != j)
                m.array() *= mean[k*d+j].array() / total[k];
        if (d > 2)
            for ( index_t q=0; q<m.rows(); ++q )
                m[q] = math::pow( m[q], (T)(1)/(T)(d-1) );

        GISMO_ASSERT ( (a.array() > 0).all() && (m.array() > 0).all(),
            "gsPatchPreconditionersCreator: the separable weights have to be positive." );

        // The i-th univariate matrix belongs to the direction d-1-i
        const index_t sz = basis.component(j).size();
        local_stiff[d-1-j] = assembleWeighted(qu[j], a, true , sz);
        local_mass [d-1-j] = assembleWeighted(qu[j], m, false, sz);
        const gsBoundaryConditions<T> bcj = boundaryConditionsForDirection(bc,j);
        eliminateDirichlet1D(bcj, opt, local_stiff[d-1-j]);
        eliminateDirichlet1D(bcj, opt, local_mass [d-1-j]);
    }
}

template<typename T>
std::vector< gsSparseMatrix<T> > assembleTensorMass(
    const gsBasis<T>& basis,
    const gsBoundaryConditions<T>& bc,
    const gsOptionList& opt
    )
{
    const index_t d = basis.dim();
    std::vector< gsSparseMatrix<T> > result(d);
    for ( index_t i=0; i!=d; ++i )
    {
        result[i] = assembleMass(basis.component(d-1-i));
        eliminateDirichlet1D(boundaryConditionsForDirection(bc,d-1-i), opt, result[i]);
    }
    return result;
}

template<typename T>
std::vector< gsSparseMatrix<T> > assembleTensorStiffness(
    const gsBasis<T>& basis,
    const gsBoundaryConditions<T>& bc,
    const gsOptionList& opt
    )
{
    const index_t d = basis.dim();
    std::vector< gsSparseMatrix<T> > result(d);
    for ( index_t i=0; i!=d; ++i )
    {
        result[i] = assembleStiffness(basis.component(d-1-i));
        eliminateDirichlet1D(boundaryConditionsForDirection(bc,d-1-i), opt, result[i]);
    }
    return result;
}

// Banded square matrix with the half bandwidth p; the entry (i,i+o-p) is
// stored at v[i*(2p+1)+o]
template<typename T>
struct lowRankBanded
{
    index_t n, p;
    std::vector<T> v;

    lowRankBanded() : n(0), p(0) {}
    lowRankBanded(index_t n_, index_t p_) : n(n_), p(p_), v(n_*(2*p_+1), (T)(0)) {}

    T& at(index_t i, index_t o)       { return v[i*(2*p+1)+o]; }
    T  at(index_t i, index_t o) const { return v[i*(2*p+1)+o]; }

    lowRankBanded transposed() const
    {
        lowRankBanded result(n,p);
        for ( index_t i=0; i<n; ++i )
            for ( index_t o=0; o<=2*p; ++o )
            {
                const index_t j = i+o-p;
                if (j >= 0 && j < n)
                    result.at(j,2*p-o) = at(i,o);
            }
        return result;
    }

    // y += this applied along the direction m of a tensor with the sizes sz,
    // where the first direction runs fastest
    void applyAlong(index_t m, const std::vector<index_t>& sz, const T* x, T* y) const
    {
        index_t stride = 1, outer = 1;
        for ( index_t j=0; j<m; ++j )
            stride *= sz[j];
        for ( index_t j=m+1; j<(index_t)sz.size(); ++j )
            outer *= sz[j];
        for ( index_t l=0; l<outer; ++l )
            for ( index_t i=0; i<n; ++i )
            {
                T* yi = y + (l*n+i)*stride;
                for ( index_t o=0; o<=2*p; ++o )
                {
                    const index_t j = i+o-p;
                    const T b = at(i,o);
                    if (j < 0 || j >= n || b == (T)(0))
                        continue;
                    const T* xj = x + (l*n+j)*stride;
                    for ( index_t t=0; t<stride; ++t )
                        yi[t] += b * xj[t];
                }
            }
    }
};

// Banded univariate matrix with the entries
//   sum_q sqw[q] coef[q*stride] f_i(nodes[q]) g_j(nodes[q]),
// where sqw are the square roots of the quadrature weights, f_i are the test
// and g_j the trial functions, each a basis function or its derivative
template<typename T>
lowRankBanded<T> assembleLowRankFactor(const univariateQuadrature<T>& qu,
                                       const gsVector<T>& sqw,
                                       const T* coef, index_t stride,
                                       bool derivTest, bool derivTrial, index_t sz)
{
    const gsMatrix<T>& f = derivTest  ? qu.derivs : qu.values;
    const gsMatrix<T>& g = derivTrial ? qu.derivs : qu.values;
    const index_t K = f.rows(), p = K-1;
    lowRankBanded<T> result(sz,p);
    for ( index_t q=0; q<f.cols(); ++q )
    {
        const T w = sqw[q] * coef[q*stride];
        if (w == (T)(0))
            continue;
        for ( index_t a=0; a<K; ++a )
        {
            const index_t i = qu.actives(a,q);
            const T wa = w * f(a,q);
            for ( index_t b=0; b<K; ++b )
                result.at(i,qu.actives(b,q)-i+p) += wa * g(b,q);
        }
    }
    return result;
}

// Tensor train SVD of the tensor x (first index fastest, overwritten) with
// the sizes sz and the absolute tolerance tol. The core m is stored in
// cores[m] with the entry (a,q,b) at a + ranks[m]*(q + sz[m]*b), where
// ranks[0] = ranks[d] = 1. Returns false if the tensor is truncated to zero.
template<typename T>
bool tensorTrainSvd(std::vector<T>& x, const std::vector<index_t>& sz, T tol,
                    std::vector< std::vector<T> >& cores, std::vector<index_t>& ranks)
{
    typedef gsEigen::Matrix<T,Dynamic,Dynamic> Matrix;
    const index_t d = (index_t)sz.size();
    cores.resize(d);
    ranks.assign(d+1,1);
    const T tol2 = tol * tol / (T)(d > 1 ? d-1 : 1);
    index_t total = (index_t)x.size();
    for ( index_t m=0; m+1<d; ++m )
    {
        const index_t rows = ranks[m] * sz[m], cols = total / sz[m];
        gsEigen::Map<Matrix> unfolding(x.data(), rows, cols);
        // For a wide unfolding X, the SVD of X^T = Q R is computed from the
        // SVD of the small factor R^T = U S W^T, so X = U S (Q W)^T
        Matrix U, V;
        gsEigen::Matrix<T,Dynamic,1> s;
        if (rows < cols)
        {
            gsEigen::HouseholderQR<Matrix> qr(unfolding.transpose());
            const Matrix Rt = qr.matrixQR().topRows(rows).template triangularView<gsEigen::Upper>().transpose();
            gsEigen::JacobiSVD<Matrix> svd(Rt, gsEigen::ComputeFullU | gsEigen::ComputeFullV);
            U = svd.matrixU();
            s = svd.singularValues();
            V = Matrix::Zero(cols, rows);
            V.topRows(rows) = svd.matrixV();
            V.applyOnTheLeft(qr.householderQ());
        }
        else
        {
            gsEigen::BDCSVD<Matrix> svd(unfolding, gsEigen::ComputeThinU | gsEigen::ComputeThinV);
            U = svd.matrixU();
            s = svd.singularValues();
            V = svd.matrixV();
        }
        index_t r = s.rows();
        T tail = 0;
        while ( r > 0 && tail + s[r-1]*s[r-1] <= tol2 )
        {
            tail += s[r-1]*s[r-1];
            --r;
        }
        if (r == 0)
            return false;
        cores[m].assign(U.data(), U.data() + rows*r);
        const Matrix rest = s.head(r).asDiagonal() * V.leftCols(r).transpose();
        x.assign(rest.data(), rest.data() + rest.size());
        total = cols;
        ranks[m+1] = r;
    }
    cores[d-1] = x;
    return true;
}

// One coefficient function as a sum of Kronecker products in the block of
// the test component e and the trial component c. factors[m][a*ranks[m+1]+b]
// is the univariate matrix of the direction m.
template<typename T>
struct lowRankTerm
{
    index_t e, c;
    std::vector<index_t> ranks;
    std::vector< std::vector< lowRankBanded<T> > > factors;

    lowRankTerm transposed() const
    {
        lowRankTerm result(*this);
        std::swap(result.e, result.c);
        for ( size_t m=0; m<factors.size(); ++m )
            for ( size_t i=0; i<factors[m].size(); ++i )
                result.factors[m][i] = factors[m][i].transposed();
        return result;
    }
};

// Low rank representation of the elasticity stiffness matrix of a patch
template<typename T>
struct lowRankElasticity
{
    index_t d, N;
    std::vector<index_t> sz, deg;                  // univariate sizes and degrees
    std::vector< lowRankTerm<T> > terms;
    std::vector< std::vector<index_t> > dofs;      // dofs[c][I]: unknown or -1

    index_t tensorSize() const
    {
        index_t result = 1;
        for ( index_t m=0; m<d; ++m )
            result *= sz[m];
        return result;
    }

    // y[e] += sum of the terms of the blocks (e,c) applied to x[c]
    void apply(const std::vector< std::vector<T> >& x, std::vector< std::vector<T> >& y) const
    {
        const index_t n = tensorSize();
        std::vector< std::vector<T> > cur, next;
        for ( size_t i=0; i<terms.size(); ++i )
        {
            const lowRankTerm<T>& t = terms[i];
            next.assign(t.ranks[d-1], std::vector<T>(n,(T)(0)));
            for ( index_t a=0; a<t.ranks[d-1]; ++a )
                t.factors[d-1][a].applyAlong(d-1, sz, x[t.c].data(), next[a].data());
            for ( index_t m=d-2; m>=0; --m )
            {
                cur.swap(next);
                next.assign(t.ranks[m], std::vector<T>(n,(T)(0)));
                for ( index_t a=0; a<t.ranks[m]; ++a )
                    for ( index_t b=0; b<t.ranks[m+1]; ++b )
                        t.factors[m][a*t.ranks[m+1]+b].applyAlong(m, sz, cur[b].data(), next[a].data());
            }
            std::vector<T>& ye = y[t.e];
            for ( index_t j=0; j<n; ++j )
                ye[j] += next[0][j];
        }
    }

    // Adds the term to the band tensor out, which has the entry of the row
    // I = i_0 + n_0 (i_1 + ...) and the offsets o_0, ..., o_{d-1} at
    // I*W + o_{d-1} + w_{d-1} (o_{d-2} + w_{d-2} (...))
    void addToBand(const lowRankTerm<T>& t, std::vector<T>& out) const
    {
        std::vector< std::vector<T> > Q(t.ranks[d-1]), Qn;
        index_t Nr = sz[d-1], Wr = 2*t.factors[d-1][0].p+1;
        for ( index_t a=0; a<t.ranks[d-1]; ++a )
            Q[a] = t.factors[d-1][a].v;
        for ( index_t m=d-2; m>=0; --m )
        {
            const index_t nm = sz[m], wm = 2*t.factors[m][0].p+1, Wn = wm*Wr;
            Qn.assign(m == 0 ? 0 : t.ranks[m], std::vector<T>());
            for ( index_t a=0; a<t.ranks[m]; ++a )
            {
                std::vector<T>& dst = (m == 0) ? out : Qn[a];
                if (m != 0)
                    dst.assign(nm*Nr*Wn, (T)(0));
                for ( index_t b=0; b<t.ranks[m+1]; ++b )
                {
                    const lowRankBanded<T>& B = t.factors[m][a*t.ranks[m+1]+b];
                    const std::vector<T>& q = Q[b];
                    for ( index_t Ir=0; Ir<Nr; ++Ir )
                        for ( index_t i=0; i<nm; ++i )
                        {
                            T* row = dst.data() + (i+nm*Ir)*Wn;
                            const T* qr = q.data() + Ir*Wr;
                            for ( index_t o=0; o<wm; ++o )
                            {
                                const T bv = B.at(i,o);
                                if (bv == (T)(0))
                                    continue;
                                T* ro = row + o*Wr;
                                for ( index_t D=0; D<Wr; ++D )
                                    ro[D] += bv * qr[D];
                            }
                        }
                }
            }
            Q.swap(Qn);
            Nr *= nm;
            Wr = Wn;
        }
    }

    gsSparseMatrix<T> matrix() const
    {
        const index_t n = tensorSize();
        std::vector<index_t> w(d);
        index_t W = 1;
        for ( index_t m=0; m<d; ++m )
        {
            w[m] = 2*deg[m]+1;
            W *= w[m];
        }
        std::vector< std::vector<T> > band(d*d);
        for ( size_t i=0; i<terms.size(); ++i )
        {
            std::vector<T>& bd = band[terms[i].e*d+terms[i].c];
            if (bd.empty())
                bd.assign(n*W, (T)(0));
            addToBand(terms[i], bd);
        }

        std::vector< std::pair<index_t,index_t> > owner(N);
        for ( index_t c=0; c<d; ++c )
            for ( index_t I=0; I<n; ++I )
                if (dofs[c][I] >= 0)
                    owner[dofs[c][I]] = std::make_pair(c,I);

        // The matrix is symmetric, so the column (c,J) is the row (c,J)
        gsSparseMatrix<T> result(N,N);
        result.reserve(N*d*W);
        std::vector< std::pair<index_t,T> > col;
        std::vector<index_t> idx(d);
        for ( index_t g=0; g<N; ++g )
        {
            const index_t c = owner[g].first, J = owner[g].second;
            index_t rem = J;
            for ( index_t m=0; m<d; ++m )
            {
                idx[m] = rem % sz[m];
                rem /= sz[m];
            }
            col.clear();
            for ( index_t e=0; e<d; ++e )
            {
                const std::vector<T>& bd = band[c*d+e];
                if (bd.empty())
                    continue;
                for ( index_t D=0; D<W; ++D )
                {
                    index_t Dr = D, I = 0;
                    bool inside = true;
                    for ( index_t m=d-1; m>=0 && inside; --m )
                    {
                        const index_t j = idx[m] + Dr % w[m] - (w[m]-1)/2;
                        Dr /= w[m];
                        inside = (j >= 0 && j < sz[m]);
                        I = I * sz[m] + j;
                    }
                    // Entries of basis functions without a common element
                    // stay exactly zero and are not stored
                    if (inside && dofs[e][I] >= 0 && bd[J*W+D] != (T)(0))
                        col.push_back(std::make_pair(dofs[e][I], bd[J*W+D]));
                }
            }
            std::sort(col.begin(), col.end());
            result.startVec(g);
            for ( size_t k=0; k<col.size(); ++k )
                result.insertBack(col[k].first, g) = col[k].second;
        }
        result.finalize();
        return result;
    }
};

template<typename T>
class lowRankElasticityOp : public gsLinearOperator<T>
{
public:
    explicit lowRankElasticityOp(const lowRankElasticity<T>& lr) : m_lr(lr) {}

    void apply(const gsMatrix<T>& input, gsMatrix<T>& x) const
    {
        const index_t d = m_lr.d, n = m_lr.tensorSize();
        x.setZero(m_lr.N, input.cols());
        std::vector< std::vector<T> > in(d, std::vector<T>(n)), out(d, std::vector<T>(n));
        for ( index_t k=0; k<input.cols(); ++k )
        {
            for ( index_t c=0; c<d; ++c )
                for ( index_t I=0; I<n; ++I )
                {
                    const index_t g = m_lr.dofs[c][I];
                    in[c][I] = g < 0 ? (T)(0) : input(g,k);
                    out[c][I] = 0;
                }
            m_lr.apply(in, out);
            for ( index_t c=0; c<d; ++c )
                for ( index_t I=0; I<n; ++I )
                {
                    const index_t g = m_lr.dofs[c][I];
                    if (g >= 0)
                        x(g,k) = out[c][I];
                }
        }
    }

    index_t rows() const { return m_lr.N; }
    index_t cols() const { return m_lr.N; }

private:
    lowRankElasticity<T> m_lr;
};

// Sets up the low rank representation of the elasticity stiffness matrix
template<typename T>
void setupLowRankElasticity(const gsBasis<T>& basis,
                            const gsGeometry<T>& geo,
                            const std::vector< gsBoundaryConditions<T> >& bc,
                            const gsOptionList& opt,
                            T lambda, T mu, T tol,
                            lowRankElasticity<T>& result)
{
    const index_t d = basis.dim();
    GISMO_ASSERT ( geo.domainDim() == d && geo.targetDim() == d,
        "gsPatchPreconditionersCreator: geometry and basis do not fit." );
    GISMO_ASSERT ( (index_t)bc.size() == d,
        "gsPatchPreconditionersCreator: needs boundary conditions for each component." );
    GISMO_ENSURE ( (dirichlet::strategy)opt.askInt("DirichletStrategy",dirichlet::elimination) == dirichlet::elimination,
        "gsPatchPreconditionersCreator: only the elimination of Dirichlet conditions is supported." );

    GISMO_ASSERT ( d >= 2, "gsPatchPreconditionersCreator: needs a basis of dimension 2 or more." );

    result.d = d;
    result.sz.resize(d);
    result.deg.resize(d);
    std::vector< univariateQuadrature<T> > qu(d);
    std::vector< gsVector<T> > sqw(d);
    std::vector<index_t> nq(d);
    index_t ntot = 1;
    for ( index_t m=0; m<d; ++m )
    {
        setupUnivariateQuadrature(basis.component(m), opt, qu[m]);
        sqw[m] = qu[m].weights.array().sqrt();
        result.sz[m] = basis.component(m).size();
        result.deg[m] = qu[m].actives.rows()-1;
        nq[m] = qu[m].weights.rows();
        ntot *= nq[m];
    }

    // Unknowns: per component, the sides with Dirichlet conditions are eliminated
    const index_t n = result.tensorSize();
    result.dofs.assign(d, std::vector<index_t>(n,-1));
    index_t offset = 0;
    std::vector<index_t> lo(d), hi(d), idx(d);
    for ( index_t c=0; c<d; ++c )
    {
        for ( index_t m=0; m<d; ++m )
        {
            const gsBoundaryConditions<T> bcm = boundaryConditionsForDirection(bc[c],m);
            const boundary_condition<T>* west = bcm.getConditionFromSide( patchSide(0,boundary::west) );
            const boundary_condition<T>* east = bcm.getConditionFromSide( patchSide(0,boundary::east) );
            lo[m] = (west && west->type() == condition_type::dirichlet) ? 1 : 0;
            hi[m] = result.sz[m] - ((east && east->type() == condition_type::dirichlet) ? 1 : 0);
        }
        for ( index_t I=0; I<n; ++I )
        {
            index_t rem = I;
            bool free = true;
            for ( index_t m=0; m<d; ++m )
            {
                const index_t i = rem % result.sz[m];
                rem /= result.sz[m];
                free = free && i >= lo[m] && i < hi[m];
            }
            if (free)
                result.dofs[c][I] = offset++;
        }
    }
    result.N = offset;

    // |det J| and J^{-1} on the tensor grid of the quadrature nodes
    std::vector<T> detJ(ntot), Jinv(ntot*d*d);
    {
        const index_t chunk = 16384;
        gsMatrix<T> pts, jacs, J, Ji;
        T sign = 0;
        for ( index_t s=0; s<ntot; s+=chunk )
        {
            const index_t len = math::min(chunk, ntot-s);
            pts.resize(d,len);
            for ( index_t i=0; i<len; ++i )
            {
                index_t rem = s+i;
                for ( index_t m=0; m<d; ++m )
                {
                    pts(m,i) = qu[m].nodes(0, rem % nq[m]);
                    rem /= nq[m];
                }
            }
            geo.jacobian_into(pts, jacs);
            for ( index_t i=0; i<len; ++i )
            {
                J = jacs.block(0, i*d, d, d);
                const T det = J.determinant();
                if (sign == 0)
                    sign = det > 0 ? 1 : -1;
                GISMO_ENSURE ( sign * det > 0,
                    "gsPatchPreconditionersCreator: the geometry map is not regular at a quadrature node." );
                detJ[s+i] = sign * det;
                Ji = J.inverse();
                for ( index_t a=0; a<d; ++a )
                    for ( index_t b=0; b<d; ++b )
                        Jinv[(s+i)*d*d + a + d*b] = Ji(a,b);
            }
        }
    }

    // The coefficient functions A^{ce}_{kl} with e>c, or e==c and k<=l; the
    // others are their transposes
    struct coefIndex { index_t c, e, k, l; };
    std::vector<coefIndex> fs;
    for ( index_t c=0; c<d; ++c )
        for ( index_t e=c; e<d; ++e )
            for ( index_t k=0; k<d; ++k )
                for ( index_t l=(e == c ? k : 0); l<d; ++l )
                {
                    const coefIndex ci = {c, e, k, l};
                    fs.push_back(ci);
                }

    std::vector< std::vector<T> > values(fs.size(), std::vector<T>(ntot));
    std::vector<T> norms(fs.size());
    T maxNorm = 0;
    for ( size_t f=0; f<fs.size(); ++f )
    {
        const coefIndex& ci = fs[f];
        std::vector<T>& v = values[f];
        for ( index_t q=0; q<ntot; ++q )
        {
            const T* Ji = &Jinv[q*d*d];
            T metric = 0;
            if (ci.c == ci.e)
                for ( index_t j=0; j<d; ++j )
                    metric += Ji[ci.k+d*j] * Ji[ci.l+d*j];
            const T a = lambda * Ji[ci.k+d*ci.c] * Ji[ci.l+d*ci.e]
                      + mu * metric
                      + mu * Ji[ci.k+d*ci.e] * Ji[ci.l+d*ci.c];
            index_t rem = q;
            T w = 1;
            for ( index_t m=0; m<d; ++m )
            {
                w *= sqw[m][rem % nq[m]];
                rem /= nq[m];
            }
            v[q] = detJ[q] * a * w;
        }
        T s = 0;
        for ( index_t q=0; q<ntot; ++q )
            s += v[q] * v[q];
        norms[f] = math::sqrt(s);
        maxNorm = math::max(maxNorm, norms[f]);
    }
    std::vector<T>().swap(Jinv);

    const T abstol = tol * maxNorm;
    result.terms.clear();
    std::vector< std::vector<T> > cores;
    std::vector<index_t> ranks;
    for ( size_t f=0; f<fs.size(); ++f )
    {
        if (norms[f] <= abstol)
            continue;
        const bool nonzero = tensorTrainSvd(values[f], nq, abstol, cores, ranks);
        std::vector<T>().swap(values[f]);
        if (!nonzero)
            continue;
        const coefIndex& ci = fs[f];
        lowRankTerm<T> t;
        t.e = ci.e;
        t.c = ci.c;
        t.ranks = ranks;
        t.factors.resize(d);
        for ( index_t m=0; m<d; ++m )
            for ( index_t a=0; a<ranks[m]; ++a )
                for ( index_t b=0; b<ranks[m+1]; ++b )
                    t.factors[m].push_back( assembleLowRankFactor(qu[m], sqw[m],
                        cores[m].data() + a + ranks[m]*nq[m]*b, ranks[m],
                        m == ci.l, m == ci.k, result.sz[m]) );
        if (ci.e != ci.c || ci.k != ci.l)
            result.terms.push_back(t.transposed());
        result.terms.push_back(t);
    }
}

} // anonymous namespace

template<typename T>
gsSparseMatrix<T> gsPatchPreconditionersCreator<T>::massMatrix(
    const gsBasis<T>& basis,
    const gsBoundaryConditions<T>& bc,
    const gsOptionList& opt
    )
{
    std::vector< gsSparseMatrix<T> > local_mass = assembleTensorMass(basis, bc, opt);
    const index_t d = local_mass.size();
    gsSparseMatrix<T> result = local_mass[d-1];
    for (index_t i=d-2; i>-1; --i)
        result = local_mass[i].kron(result);
    return result;
}

template<typename T>
typename gsPatchPreconditionersCreator<T>::OpUPtr gsPatchPreconditionersCreator<T>::massMatrixOp(
    const gsBasis<T>& basis,
    const gsBoundaryConditions<T>& bc,
    const gsOptionList& opt
    )
{
    const index_t d = basis.dim();

    std::vector< gsSparseMatrix<T> > local_mass = assembleTensorMass(basis, bc, opt);

    std::vector<OpPtr> local_mass_op(d);
    for (index_t i=0; i<d; ++i)
        local_mass_op[i] = makeMatrixOp(local_mass[i].moveToPtr());

    return gsKroneckerOp<T>::make(local_mass_op);
}

template<typename T>
typename gsPatchPreconditionersCreator<T>::OpUPtr gsPatchPreconditionersCreator<T>::massMatrixInvOp(
    const gsBasis<T>& basis,
    const gsBoundaryConditions<T>& bc,
    const gsOptionList& opt
    )
{
    const index_t d = basis.dim();

    std::vector< gsSparseMatrix<T> > local_mass = assembleTensorMass(basis, bc, opt);

    std::vector<OpPtr> local_mass_op(d);
    for (index_t i=0; i<d; ++i)
        local_mass_op[i] = makeSparseCholeskySolver(local_mass[i]);

    return gsKroneckerOp<T>::make(local_mass_op);
}

template<typename T>
gsSparseMatrix<T> gsPatchPreconditionersCreator<T>::stiffnessMatrix(
    const gsBasis<T>& basis,
    const gsBoundaryConditions<T>& bc,
    const gsOptionList& opt,
    T alpha,
    T beta
    )
{
    const index_t d = basis.dim();

    std::vector< gsSparseMatrix<T> > local_stiff = assembleTensorStiffness(basis, bc, opt);
    std::vector< gsSparseMatrix<T> > local_mass  = assembleTensorMass(basis, bc, opt);

    if ( beta!=1 )
    {
        for (index_t i=0; i!=d; ++i)
            local_stiff[i] *= beta;
    }

    gsSparseMatrix<T> K = give(local_stiff[d-1]);
    gsSparseMatrix<T> M = give(local_mass [d-1]);

    for (index_t i=d-2; i>-1; --i)
    {
        K  = local_mass[i].kron(K);
        K += local_stiff[i].kron(M);
        if ( i != 0 || alpha != 0 )
            M = local_mass[i].kron(M);
    }
    if (alpha==1)
        K += M;
    else if (alpha!=0)
        K += alpha * M;
    return K;
}

template<typename T>
typename gsPatchPreconditionersCreator<T>::OpUPtr gsPatchPreconditionersCreator<T>::stiffnessMatrixOp(
    const gsBasis<T>& basis,
    const gsBoundaryConditions<T>& bc,
    const gsOptionList& opt,
    T alpha,
    T beta
    )
{
    const index_t d = basis.dim();

    std::vector< gsSparseMatrix<T> > local_stiff = assembleTensorStiffness(basis, bc, opt);
    std::vector< gsSparseMatrix<T> > local_mass  = assembleTensorMass(basis, bc, opt);

    std::vector<OpUPtr> local_stiff_op(d);
    std::vector<OpPtr > local_mass_op (d);
    for (index_t i=0; i<d; ++i)
    {
        if (beta!=1)
            local_stiff[i] *= beta;
        local_stiff_op[i] = makeMatrixOp(local_stiff[i].moveToPtr());
        local_mass_op [i] = makeMatrixOp(local_mass [i].moveToPtr());
    }
    OpUPtr K = give(local_stiff_op[0]);
    OpPtr  M = give(local_mass_op [0]);

    for (index_t i=1; i<d; ++i)
    {
        K = gsSumOp<T>::make(
            gsKroneckerOp<T>::make( give(K),      local_mass_op [i]  ),
            gsKroneckerOp<T>::make( M,       give(local_stiff_op[i]) )
            );
        if ( i < d-1 || alpha != 0 )
            M = gsKroneckerOp<T>::make(M, local_mass_op[i]);
    }
    if (alpha==1)
        K = gsSumOp<T>::make( give(K), M );
    else if (alpha!=0)
        K = gsSumOp<T>::make( give(K), gsScaledOp<T>::make( M, alpha ) );

    return K;
}

template<typename T>
typename gsPatchPreconditionersCreator<T>::OpUPtr gsPatchPreconditionersCreator<T>::fastDiagonalizationOp(
    const gsBasis<T>& basis,
    const gsBoundaryConditions<T>& bc,
    const gsOptionList& opt,
    T alpha,
    T beta,
    T gamma
    )
{
    GISMO_ASSERT ( beta != 0, "gsPatchPreconditionersCreator<T>::fastDiagonalizationOp() does not work for beta==0." );

    return fastDiagonalization(basis, bc, opt, alpha, gsVector<T>::Constant(basis.dim(),beta), gamma);
}

template<typename T>
typename gsPatchPreconditionersCreator<T>::OpUPtr gsPatchPreconditionersCreator<T>::fastDiagonalizationOp(
    const gsBasis<T>& basis,
    const gsBoundaryConditions<T>& bc,
    const gsOptionList& opt,
    T alpha,
    const gsVector<T>& beta
    )
{
    GISMO_ASSERT ( beta.rows() == basis.dim(),
        "gsPatchPreconditionersCreator<T>::fastDiagonalizationOp() needs one beta per direction." );
    GISMO_ASSERT ( (beta.array() > 0).all(),
        "gsPatchPreconditionersCreator<T>::fastDiagonalizationOp() needs positive betas." );

    return fastDiagonalization(basis, bc, opt, alpha, beta, 0);
}

template<typename T>
typename gsPatchPreconditionersCreator<T>::OpUPtr gsPatchPreconditionersCreator<T>::fastDiagonalizationOp(
    const gsBasis<T>& basis,
    const gsGeometry<T>& geo,
    const gsBoundaryConditions<T>& bc,
    const gsOptionList& opt
    )
{
    const index_t d = basis.dim();
    GISMO_ASSERT ( geo.domainDim() == d && geo.targetDim() == d,
        "gsPatchPreconditionersCreator<T>::fastDiagonalizationOp(): geometry and basis do not fit." );

    std::vector< univariateQuadrature<T> > qu(d);
    for ( index_t k=0; k<d; ++k )
        setupUnivariateQuadrature(basis.component(k), opt, qu[k]);

    std::vector< gsVector<T> > sums;
    if ( !assembleMetricSums(geo, qu, sums) )
    {
        // The geometry map is not regular: one average per direction
        return fastDiagonalization(basis, bc, opt, 0, averageMetricDiagonal(geo), 0);
    }

    std::vector< gsSparseMatrix<T> > local_stiff, local_mass;
    const gsMatrix<T> factors = gsMatrix<T>::Ones(d,d);
    assembleTensorSeparable(basis, bc, opt, qu, sums, factors, local_stiff, local_mass);
    return fastDiagonalization(local_stiff, local_mass, 0, 0);
}

template<typename T>
typename gsPatchPreconditionersCreator<T>::OpUPtr gsPatchPreconditionersCreator<T>::elasticityFastDiagonalizationOp(
    const gsBasis<T>& basis,
    const gsBoundaryConditions<T>& bc,
    const gsOptionList& opt,
    T lambda,
    T mu,
    const gsGeometry<T>* geo
    )
{
    std::vector< gsBoundaryConditions<T> > bcs;
    for ( index_t c=0; c<basis.dim(); ++c )
        bcs.push_back( boundaryConditionsForComponent(bc,(short_t)c) );
    return elasticityFastDiagonalizationOp(basis, bcs, opt, lambda, mu, geo);
}

template<typename T>
typename gsPatchPreconditionersCreator<T>::OpUPtr gsPatchPreconditionersCreator<T>::elasticityFastDiagonalizationOp(
    const gsBasis<T>& basis,
    const std::vector< gsBoundaryConditions<T> >& bc,
    const gsOptionList& opt,
    T lambda,
    T mu,
    const gsGeometry<T>* geo
    )
{
    GISMO_ASSERT ( (index_t)bc.size() == basis.dim(),
        "gsPatchPreconditionersCreator<T>::elasticityFastDiagonalizationOp() needs boundary conditions for each component." );
    GISMO_ASSERT ( mu > 0 && lambda + mu > 0,
        "gsPatchPreconditionersCreator<T>::elasticityFastDiagonalizationOp() needs mu>0 and lambda+mu>0." );

    const index_t d = basis.dim();
    typename gsBlockOp<T>::uPtr result = gsBlockOp<T>::make(d,d);

    gsVector<T> scaling = gsVector<T>::Ones(d);
    if (geo)
    {
        GISMO_ASSERT ( geo->domainDim() == d && geo->targetDim() == d,
            "gsPatchPreconditionersCreator<T>::elasticityFastDiagonalizationOp(): geometry and basis do not fit." );

        std::vector< univariateQuadrature<T> > qu(d);
        for ( index_t k=0; k<d; ++k )
            setupUnivariateQuadrature(basis.component(k), opt, qu[k]);

        std::vector< gsVector<T> > sums;
        if ( assembleMetricSums(*geo, qu, sums) )
        {
            // The coefficient of the direction k in the block of the component c is
            // mu (|det J| J^{-1} J^{-T})_{kk} + (lambda+mu) |det J| (J^{-1})_{kc}^2
            std::vector< gsSparseMatrix<T> > local_stiff, local_mass;
            for ( index_t c=0; c<d; ++c )
            {
                gsMatrix<T> factors = gsMatrix<T>::Constant(d,d,mu);
                factors.col(c).array() += lambda+mu;
                assembleTensorSeparable(basis, bc[c], opt, qu, sums, factors, local_stiff, local_mass);
                result->addOperator(c, c, fastDiagonalization(local_stiff, local_mass, 0, 0) );
            }
            return result;
        }

        // The geometry map is not regular: one average per direction
        scaling = averageMetricDiagonal(*geo);
    }

    for ( index_t c=0; c<d; ++c )
    {
        gsVector<T> beta = mu * scaling;
        beta[c] += (lambda+mu) * scaling[c];
        result->addOperator(c, c,
            fastDiagonalization(basis, bc[c], opt, 0, beta, 0) );
    }
    return result;
}

template<typename T>
gsSparseMatrix<T> gsPatchPreconditionersCreator<T>::elasticityLowRankMatrix(
    const gsBasis<T>& basis,
    const gsGeometry<T>& geo,
    const std::vector< gsBoundaryConditions<T> >& bc,
    const gsOptionList& opt,
    T lambda,
    T mu,
    T tol
    )
{
    lowRankElasticity<T> lr;
    setupLowRankElasticity(basis, geo, bc, opt, lambda, mu, tol, lr);
    return lr.matrix();
}

template<typename T>
gsSparseMatrix<T> gsPatchPreconditionersCreator<T>::elasticityLowRankMatrix(
    const gsBasis<T>& basis,
    const gsGeometry<T>& geo,
    const gsBoundaryConditions<T>& bc,
    const gsOptionList& opt,
    T lambda,
    T mu,
    T tol
    )
{
    std::vector< gsBoundaryConditions<T> > bcs;
    for ( index_t c=0; c<basis.dim(); ++c )
        bcs.push_back( boundaryConditionsForComponent(bc,(short_t)c) );
    return elasticityLowRankMatrix(basis, geo, bcs, opt, lambda, mu, tol);
}

template<typename T>
typename gsPatchPreconditionersCreator<T>::OpUPtr gsPatchPreconditionersCreator<T>::elasticityLowRankOp(
    const gsBasis<T>& basis,
    const gsGeometry<T>& geo,
    const std::vector< gsBoundaryConditions<T> >& bc,
    const gsOptionList& opt,
    T lambda,
    T mu,
    T tol
    )
{
    lowRankElasticity<T> lr;
    setupLowRankElasticity(basis, geo, bc, opt, lambda, mu, tol, lr);
    return OpUPtr( new lowRankElasticityOp<T>(lr) );
}

template<typename T>
typename gsPatchPreconditionersCreator<T>::OpUPtr gsPatchPreconditionersCreator<T>::elasticityLowRankOp(
    const gsBasis<T>& basis,
    const gsGeometry<T>& geo,
    const gsBoundaryConditions<T>& bc,
    const gsOptionList& opt,
    T lambda,
    T mu,
    T tol
    )
{
    std::vector< gsBoundaryConditions<T> > bcs;
    for ( index_t c=0; c<basis.dim(); ++c )
        bcs.push_back( boundaryConditionsForComponent(bc,(short_t)c) );
    return elasticityLowRankOp(basis, geo, bcs, opt, lambda, mu, tol);
}

template<typename T>
typename gsPatchPreconditionersCreator<T>::OpUPtr gsPatchPreconditionersCreator<T>::fastDiagonalization(
    const gsBasis<T>& basis,
    const gsBoundaryConditions<T>& bc,
    const gsOptionList& opt,
    T alpha,
    const gsVector<T>& beta,
    T gamma
    )
{
    const index_t d = basis.dim();

    // Assemble univariate
    std::vector< gsSparseMatrix<T> > local_stiff = assembleTensorStiffness(basis, bc, opt);
    std::vector< gsSparseMatrix<T> > local_mass  = assembleTensorMass(basis, bc, opt);

    // The i-th univariate matrix belongs to the direction d-1-i
    for (index_t i=0; i!=d; ++i)
        local_stiff[i] *= beta[d-1-i];

    return fastDiagonalization(local_stiff, local_mass, alpha, gamma);
}

template<typename T>
typename gsPatchPreconditionersCreator<T>::OpUPtr gsPatchPreconditionersCreator<T>::fastDiagonalization(
    const std::vector< gsSparseMatrix<T> >& local_stiff,
    const std::vector< gsSparseMatrix<T> >& local_mass,
    T alpha,
    T gamma
    )
{
    const index_t d = (index_t)local_stiff.size();

    // Determine overall size
    index_t sz = 1;
    for ( index_t i=0; i<d; ++i )
        sz *= local_stiff[i].rows();

    // Initialize the diagonal with 1
    gsMatrix<T> diag;
    diag.setConstant(sz,1,alpha); // This is the pure-mass part!

    index_t glob = sz; // Indexing value for setting up the Kronecker product

    typedef typename gsMatrix<T>::GenSelfAdjEigenSolver EVSolver;
    typedef typename EVSolver::EigenvectorsType evMatrix;
    typedef typename EVSolver::RealVectorType evVector;
    EVSolver ges;

    std::vector<OpPtr> Qop(d);
    std::vector<OpPtr> QTop(d);
    gsMatrix<T> ev;

    T avg_term = gamma;

    // Now, setup the Q's and update the D's
    for ( index_t i=0; i<d; ++i )
    {
        // Solve generalized eigenvalue problem
        ges.compute(local_stiff[i], local_mass[i], gsEigen::ComputeEigenvectors);
        // Q^T M Q = I, or M = Q^{-T} Q^{-1}
        // Q^T K Q = D, or K = Q^{-T} D Q^{-1}

        if (gamma != (T)(0))
        {
            gsMatrix<T> etrans = ges.eigenvectors().transpose()*local_mass[i]*gsMatrix<T>::Ones(local_mass[i].rows(),1);
            GISMO_ASSERT((etrans.block(1, 0, etrans.rows()-1, 1).array() < (T)(1)/100000000).all(),
                "gsPatchPreconditionerCreator::fastDiagonalizationOp: gamma!=0 only allowed for pure Neumann.");
            avg_term *= etrans(0,0) * etrans(0,0);
        }

        // From the eigenvalues, we setup the matrix D already in an Kroneckerized way.
        const evVector & D = ges.eigenvalues();

        const index_t loc = D.rows();
        glob /= loc;
        const index_t glob2 = sz / loc / glob;

        for ( index_t l=0; l<loc; ++l )
            for ( index_t m=0; m<glob; ++m )
                for ( index_t n=0; n<glob2; ++n )
                    diag( m + l*glob + n*loc*glob, 0 ) += D(l,0);

        // Finally, we store the eigenvectors
        ev.swap(const_cast<evMatrix&>(ges.eigenvectors()));

        // These are the operators representing the eigenvectors
        typename gsMatrixOp< gsMatrix<T> >::Ptr matrOp = makeMatrixOp( ev.moveToPtr() );
        Qop [i] = matrOp;
        // Here we are safe as long as we do not want to apply QTop after Qop got destroyed.
        QTop[i] = makeMatrixOp( matrOp->matrix().transpose() );
    }

    GISMO_ASSERT( glob == 1,
        "gsPatchPreconditionerCreator::fastDiagonalizationOp: Internal error." );

    diag(0,0) += avg_term;

    for ( index_t l=0; l<sz; ++l )
        diag( l, 0 ) = (T)(1)/diag( l, 0 );

    memory::unique_ptr< gsEigen::DiagonalMatrix<T,Dynamic> > diag_mat( new gsEigen::DiagonalMatrix<T,Dynamic>( give(diag) ) );

    return gsProductOp<T>::make(
        gsKroneckerOp<T>::make(QTop),
        makeMatrixOp(give(diag_mat)),
        gsKroneckerOp<T>::make(Qop)
        );
}

namespace {

// Get the tilde basis
template<typename T>
void tildeSpaceBasis_oneside(const gsTensorBSplineBasis<1,T>& basis, bool isLeftHandSide, gsMatrix<T>& tildeBasis, gsMatrix<T>& complBasis, bool bc = 0, const bool odd = true)
{
    // bc == false: Neumann (or any other not-eliminating bc), bc == true: Dirichlet

    const index_t b = (index_t)bc;
    const index_t p = basis.degree();
    const T h = basis.knots().maxIntervalLength();

    if (p-b<=0)
    {
        tildeBasis.resize(0,0); complBasis.resize(0,0);
        return;
    }

    const T u = (isLeftHandSide ? basis.knots().first() : basis.knots().last());
    gsMatrix<T> U(1,1);
    U(0,0) = u;

    std::vector< gsMatrix<T> > allDerivs;
    basis.evalAllDers_into(U, p-1, allDerivs);

    // Collect all derivatives in matrix (rows: derivatives, columns: basis functions)
    // normalize with h^(deriv)
    // use only odd derivatives if odd = true and only even derivatives if odd = false
    // skip last (on left) or first (on right) basis function since it's always in S-tilde
    gsMatrix<T> derivs;
    // If we have Dirichlet bc, we reduce the number of rows and columns by 1. We basically
    // eliminate the first row and the first column (first basis function, 0th derivative)
    // for the right-side, we have to remove the last basis function.
    derivs.setZero(p-b, p-b);

    const index_t offset = (isLeftHandSide ? b : 1);

    index_t i_start;
    if (odd) i_start = 1;
    else i_start = 2*b;

    for (index_t i = i_start; i < p; i += 2)
        for (index_t j = 0; j < p-b; ++j)
            derivs(i-b, j) = math::pow(h, i) * allDerivs[i](j+offset);

    typename gsMatrix<T>::JacobiSVD svd = derivs.jacobiSvd(gsEigen::ComputeFullV);

    index_t n_tilde;
    if (odd) n_tilde = (p + 1) / 2 - b;
    else n_tilde = p / 2 - b;

    tildeBasis = svd.matrixV().rightCols(n_tilde);
    complBasis = svd.matrixV().leftCols(p - b - n_tilde);
}

// Compute a basis for S-tilde and one for its orthogonal complement
template<typename T>
void tildeSpaceBasis(const gsBasis<T>& basis, gsSparseMatrix<T>& B_tilde, gsSparseMatrix<T>& B_compl, const gsBoundaryConditions<T>& bc, const bool odd = true)
{
    GISMO_ASSERT( nullptr != (dynamic_cast<const gsTensorBSplineBasis<1,T>*>(&basis)),
                    "gsPatchPreconditionersCreator<T>::getTildeSpaceBasisTransformation and "
                    "gsPatchPreconditionersCreator<T>::subspaceCorrectedMassSmootherOp only work with tensor-B-spline bases." );
    const gsTensorBSplineBasis<1,T>& bbasis = static_cast<const gsTensorBSplineBasis<1,T>&>(basis);

    patchSide west(0,boundary::west), east(0,boundary::east);
    bool bwest = ( bc.getConditionFromSide( west ) && bc.getConditionFromSide( west )->type() == condition_type::dirichlet );
    bool beast = ( bc.getConditionFromSide( east ) && bc.getConditionFromSide( east )->type() == condition_type::dirichlet );

    gsMatrix<T> b_L, b_compl_L;
    gsMatrix<T> b_R, b_compl_R;

    // Contruct space with vanishing odd derivatives
    tildeSpaceBasis_oneside(bbasis, true,  b_L, b_compl_L, bwest, odd);
    tildeSpaceBasis_oneside(bbasis, false, b_R, b_compl_R, beast, odd);

    const index_t n = bbasis.size() - (index_t)bwest - (index_t)beast;
    const index_t n_L = b_L.cols();
    const index_t m_L = b_L.rows();
    const index_t n_R = b_R.cols();
    const index_t m_R = b_R.rows();
    const index_t n_c_L = b_compl_L.cols();
    const index_t m_c_L = b_compl_L.rows();
    const index_t n_c_R = b_compl_R.cols();
    const index_t m_c_R = b_compl_R.rows();
    const index_t n_I = n - n_L - n_R - n_c_L - n_c_R;

    //GISMO_ENSURE ( n_I >= 0, "tildeSpaceBasis: Too few knots for that spline degree." );
    if ( n_I <= 0 )
    {
        static bool warned = false;
        if (!warned)
        {
            gsWarn << "tildeSpaceBasis was called with too few knots for that spline degree.\n"
                   << "So, we assume that S_tilde is empty.\n";
            warned = true;
        }

        B_tilde.resize(n,0);

        gsSparseEntries<T> E_compl;
        E_compl.reserve(n);
        for (index_t i = 0; i < n; ++i)
            E_compl.add(i,i,1.0);
        B_compl.resize(n,n);
        B_compl.setFrom(E_compl);
        return;
    }

    gsSparseEntries<T> E_tilde, E_compl;

    // put b_L into upper left block of S-tilde basis
    for (index_t j = 0; j < n_L; ++j)
    {
        for (index_t i = 0; i < m_L; ++i)
            E_tilde.add(i, j, b_L(i, j));
    }
    // fill identity matrix into interior part of S-tilde basis
    for (index_t j = 0; j < n_I; ++j)
    {
        E_tilde.add(m_L + j, n_L + j, 1.0);
    }
    // put b_R into lower right block of S-tilde basis
    for (index_t j = 0; j < n_R; ++j)
    {
        for (index_t i = 0; i < m_R; ++i)
            E_tilde.add(m_L + n_I + i, n_L + n_I + j, b_R(i, j));
    }
    B_tilde.resize(n, n_L + n_I + n_R);
    B_tilde.setFrom(E_tilde);

    // put b_compl_L into upper left block of complement basis
    for (index_t j = 0; j < n_c_L; ++j)
    {
        for (index_t i = 0; i < m_c_L; ++i)
            E_compl.add(i, j, b_compl_L(i, j));
    }
    // put b_compl_R into lower right block of complement basis
    for (index_t j = 0; j < n_c_R; ++j)
    {
        for (index_t i = 0; i < m_c_R; ++i)
            E_compl.add(m_c_L + n_I + i, n_c_L + j, b_compl_R(i, j));
    }
    B_compl.resize(n, n_c_L + n_c_R);
    B_compl.setFrom(E_compl);

}

template<typename T>
void constructTildeSpaceBasis(
    const gsBasis<T>& basis,
    const gsBoundaryConditions<T>& bc,
    const gsOptionList& opt,
    std::vector< gsSparseMatrix<T> >& B_tilde,
    std::vector< gsSparseMatrix<T> >& B_l2compl,
    const bool odd = true
    )
{
    const index_t d = basis.dim();
    B_tilde.resize(d);
    B_l2compl.resize(d);
    dirichlet::strategy ds = (dirichlet::strategy)opt.askInt("DirichletStrategy",dirichlet::elimination);
    if (ds == dirichlet::elimination)
    {
        for ( index_t i=0; i<d; ++i )
            tildeSpaceBasis(basis.component(d-1-i), B_tilde[i], B_l2compl[i], boundaryConditionsForDirection(bc,d-1-i), odd);
    }
    else
        GISMO_ERROR("Unknown Dirichlet strategy.");

}

// Constructs a matrix for swapping a tensor product
// from A (x) B (x) C (x) D (x) E  to   A (x) D (x) C (x) B (x) E,
// where only the dimensions of those matrices have to be given.
// Note that also those A, B, etc. could be tensor products (here as dimension just provide the products)
// Note that also those A, B, etc. could vanish. Then just provide a 1 as dimension, i.e., a scalar.
// So, literally every thinkable swap is possible.
template<typename T>
gsSparseMatrix<T> kroneckerSwap( index_t e, index_t d, index_t c, index_t b, index_t a )
{
    const index_t sz = a*b*c*d*e;
    gsSparseMatrix<T> result(sz,sz);
    gsSparseEntries<T> entries;
    entries.reserve(sz);
    for ( index_t i=0; i<a; ++i )
        for ( index_t j=0; j<b; ++j )
            for ( index_t k=0; k<c; ++k )
                for ( index_t l=0; l<d; ++l )
                    for ( index_t m=0; m<e; ++m )
                        entries.add( i+a*(j+b*(k+c*(l+d*m))), i+a*(l+d*(k+c*(j+b*m))), 1. );

    result.setFrom(entries);
    result.makeCompressed();

    return result;
}

} // anonymous namespace

template<typename T>
std::pair< std::vector< gsSparseMatrix<T> >, std::vector< gsSparseMatrix<T> > > gsPatchPreconditionersCreator<T>::getTildeSpaceBasisTransformation(
    const gsBasis<T>& basis,
    const gsBoundaryConditions<T>& bc,
    const gsOptionList& opt
    )
{
    std::vector< gsSparseMatrix<T> > B_tilde, B_l2compl;
    constructTildeSpaceBasis( basis, bc, opt, B_tilde, B_l2compl, !opt.askSwitch( "UseVanishingEvenDerivatives", false ) );
    return std::pair< std::vector< gsSparseMatrix<T> >, std::vector< gsSparseMatrix<T> > >( B_tilde, B_l2compl );
}

template<typename T>
typename gsPatchPreconditionersCreator<T>::OpUPtr gsPatchPreconditionersCreator<T>::subspaceCorrectedMassSmootherOp(
    const gsBasis<T>& basis,
    const gsBoundaryConditions<T>& bc,
    const gsOptionList& opt,
    T sigma,
    T alpha,
    T beta
    )
{
    GISMO_ASSERT ( beta != 0, "gsPatchPreconditionersCreator<T>::subspaceCorrectedMassSmootherOp() does not work for beta==0." );

    // Get some properties
    const index_t d = basis.dim();
    const T h = basis.getMinCellLength();

    // Assemble univariate
    std::vector< gsSparseMatrix<T> > local_stiff = assembleTensorStiffness(basis, bc, opt);
    std::vector< gsSparseMatrix<T> > local_mass  = assembleTensorMass(basis, bc, opt);

    if (beta != 1)
    {
        for (index_t i=0; i!=d; ++i)
            local_stiff[i] *= beta;
    }

    // Setup of basis
    std::vector< gsSparseMatrix<T> > B_tilde(d), B_l2compl(d);
    constructTildeSpaceBasis(basis, bc, opt, B_tilde, B_l2compl);

    std::vector< gsSparseMatrix<T> > M_compl(d), K_compl(d), B_compl(d);
    std::vector< typename gsLinearOperator<T>::Ptr > M_tilde_inv(d);
    for ( index_t i=0; i<d; ++i )
    {
        // Transform the complement
        typename gsLinearOperator<T>::Ptr M_inv = makeSparseCholeskySolver(local_mass[i]);
        gsMatrix<T> B_compl_dense;
        M_inv->apply( B_l2compl[i], B_compl_dense );
        B_compl[i] = B_compl_dense.sparseView();

        // Setup of matrices and corresponding solvers
        gsSparseMatrix<T> M_tilde = B_tilde[i].transpose() * local_mass[i] * B_tilde[i];
        M_tilde_inv[i] = makeSparseCholeskySolver( M_tilde );
        M_compl[i] = B_compl[i].transpose() * local_mass[i] * B_compl[i];
        K_compl[i] = B_compl[i].transpose() * local_stiff[i] * B_compl[i];
    }

    // Setup of final operator
    typename gsSumOp<T>::uPtr result = gsSumOp<T>::make();

    for ( index_t type = 0; type < (1<<d); ++ type )
    {
        std::vector< typename gsLinearOperator<T>::Ptr > correction(0);
        gsSparseMatrix<T> transfer;

        std::vector< gsSparseMatrix<T>* > transfers(d);

        index_t numberInteriors = 0;

        // Setup of transfer
        for ( index_t j = 0; j<d; ++ j )
        {
            if ( type & ( 1 << j ) )
                transfers[j] = &(B_compl[d-1-j]);
            else
            {
                transfers[j] = &(B_tilde[d-1-j]);
                ++numberInteriors;
            }

            if ( j == 0 )
                transfer = *(transfers[j]);
            else
                transfer = transfers[j]->kron(transfer);

        }

        // If the subspace is not present, ignore it.
        if ( transfer.cols() == 0 )
            continue;

        // Setup of swap, where the boundary part is shifted to the begin

        index_t left = 1, current = transfers[d-1]->cols(), right = 1;
        for ( index_t j = 0; j < d-1; ++j )
            left *= transfers[j]->cols();

        for ( index_t j = d-1; j >= 0; --j )
        {
            if ( type & ( 1 << j ) )
            {
                transfer = transfer * kroneckerSwap<T>( right, current, left, 1, 1 );
                if ( j > 0 )
                {
                    left *= current;
                    current = transfers[j-1]->cols();
                    left /= current;
                }
            }
            else
            {
                if ( j > 0 )
                {
                    right *= current;
                    current = transfers[j-1]->cols();
                    left /= current;
                }
            }

        }

        // Setup of interior correction
        for ( index_t j = d-1; j>=0; --j )
        {
            if ( ! ( type & ( 1 << j ) ) )
                correction.push_back( M_tilde_inv[d-1-j] );
        }

        // If we are in the interior, we have to do the scaling here as there is no boundary correction.
        if ( numberInteriors == d )
            correction[0] = gsScaledOp<T>::make( correction[0], (T)(1)/( alpha + beta*(T)(numberInteriors)/(sigma*h*h) ) );

        // Setup of bondary correction
        if ( numberInteriors < d )
        {
            gsSparseMatrix<T> bc_matrix;

            {
                gsSparseMatrix<T> s(0,0);
                for ( index_t k = d-1; k>=0; --k )
                {
                    if ( type & ( 1 << k ) )
                    {
                        if ( s.rows() == 0 )
                            s = M_compl[d-1-k];
                        else
                            s = M_compl[d-1-k].kron(s);
                    }
                }
                bc_matrix = ( alpha + beta*(T)(numberInteriors)/(sigma*h*h) ) * s;
            }

            for ( index_t j = d-1; j>=0; --j )
            {
                if ( type & ( 1 << j ) )
                {
                    gsSparseMatrix<T> s(0,0);
                    for ( index_t k = d-1; k>=0; --k )
                    {
                        if ( type & ( 1 << k ) )
                        {
                            gsSparseMatrix<T>* chosenMatrix;
                            if ( j == k )
                                chosenMatrix = &(K_compl[d-1-k]);
                            else
                                chosenMatrix = &(M_compl[d-1-k]);

                            if ( s.rows() == 0 )
                                s = *chosenMatrix;
                            else
                                s = s.kron(*chosenMatrix);
                        }
                    }
                    bc_matrix += s;
                }
            }

            correction.push_back(makeSparseCholeskySolver(bc_matrix));
        }

        typename gsMatrixOp< gsSparseMatrix<T> >::Ptr transOp = makeMatrixOp(transfer.moveToPtr());
        // Setup of whole operator
        // The correction is the Kronecker-product of the operators in the vector correction.
        result->addOperator(
            gsProductOp<T>::make(
                makeMatrixOp( transOp->matrix().transpose() ),
                gsKroneckerOp<T>::make( correction ),
                transOp
                )
            );
    }

    return give(result);

}

} // namespace gismo
