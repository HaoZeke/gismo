/** @file elasticity_low_rank_example.cpp

  @brief Low-rank assembly of linear elasticity on one tensor B-spline
  patch, compared with gsElasticityAssembler

  Low-rank assembly of linear elasticity on one tensor B-spline patch,
  measured against gsElasticityAssembler.
  
  The stiffness matrix of displacement component c (trial) against e (test) is
    K_ec = sum_{k,l} int A^{ce}_{kl}(xi) d_k B_J d_l B_I dxi,
    A^{ce}_{kl} = |det J| (lambda Ji_kc Ji_le + mu delta_ce (Ji Ji^T)_kl + mu Ji_ke Ji_lc),
  with Ji = J^{-1}. Each A^{ce}_{kl}, times the square roots of the quadrature
  weights, is sampled on the tensor grid of the quadrature nodes and compressed
  by a TT-SVD with absolute tolerance eps * max_f ||A_f||. A TT term with ranks
  r_1..r_{d-1} is the sum of prod r_m Kronecker products of banded univariate
  matrices, which give (a) a sparse matrix and (b) a matrix-free operator.

  Needs the optional module gsElasticity (-DGISMO_OPTIONAL=gsElasticity);
  without it the example prints a note and exits.

  This file is part of the G+Smo library.

  This Source Code Form is subject to the terms of the Mozilla Public
  License, v. 2.0. If a copy of the MPL was not distributed with this
  file, You can obtain one at http://mozilla.org/MPL/2.0/.

  Author(s): R. Goswami
*/

#include <gismo.h>

#ifdef gsElasticity_ENABLED
#if defined(__unix__) || defined(__APPLE__)
#include <sys/resource.h>
#endif
#include <gsElasticity/src/gsElasticityAssembler.h>
#include <gsElasticity/src/gsBaseUtils.h>
#include <gsElasticity/src/gsGeoUtils.h>
#include <algorithm>
#include <cstdio>
using namespace gismo;
typedef real_t T;
typedef gsEigen::Matrix<T, gsEigen::Dynamic, gsEigen::Dynamic> Mat;

// Banded n x n matrix of half bandwidth p: entry (i, i+o-p) at v[i*(2p+1)+o].
struct Banded {
  index_t n = 0, p = 0;
  std::vector<T> v;
  void init(index_t n_, index_t p_) { n = n_; p = p_; v.assign(n * (2 * p + 1), 0); }
  T &at(index_t i, index_t o) { return v[i * (2 * p + 1) + o]; }
  T at(index_t i, index_t o) const { return v[i * (2 * p + 1) + o]; }
  Banded transposed() const {
    Banded b; b.init(n, p);
    for (index_t i = 0; i < n; ++i)
      for (index_t o = 0; o <= 2 * p; ++o) {
        const index_t j = i + o - p;
        if (j >= 0 && j < n) b.at(j, 2 * p - o) = at(i, o);
      }
    return b;
  }
};

struct UniQuad {
  gsMatrix<T> nodes, values, derivs;
  gsVector<T> weights, sqw;
  gsMatrix<index_t> actives;
  index_t n = 0, p = 0;
};

void setupUniQuad(const gsBasis<T> &b, const gsOptionList &opt, UniQuad &qu) {
  typename gsQuadRule<T>::uPtr rule = gsQuadrature::getPtr(b, opt);
  std::vector<T> nd, wt;
  gsMatrix<T> en; gsVector<T> ew;
  typename gsBasis<T>::domainIter it = b.domain()->beginAll(), end = b.domain()->endAll();
  for (; it < end; ++it) {
    rule->mapTo(it.lowerCorner(), it.upperCorner(), en, ew);
    for (index_t q = 0; q < ew.rows(); ++q) { nd.push_back(en(0, q)); wt.push_back(ew[q]); }
  }
  const index_t m = (index_t)nd.size();
  qu.nodes.resize(1, m); qu.weights.resize(m);
  for (index_t q = 0; q < m; ++q) { qu.nodes(0, q) = nd[q]; qu.weights[q] = wt[q]; }
  qu.sqw = qu.weights.array().sqrt();
  b.eval_into(qu.nodes, qu.values);
  b.deriv_into(qu.nodes, qu.derivs);
  b.active_into(qu.nodes, qu.actives);
  qu.n = b.size(); qu.p = b.maxDegree();
}

// sum_q sqw[q] g[q*gs] f^{dtest}_i(q) f^{dtrial}_j(q), row i = test function
Banded uniMatrix(const UniQuad &qu, const T *g, index_t gs, bool dtest, bool dtrial) {
  Banded B; B.init(qu.n, qu.p);
  const gsMatrix<T> &fi = dtest ? qu.derivs : qu.values, &fj = dtrial ? qu.derivs : qu.values;
  const index_t K = fi.rows();
  for (index_t q = 0; q < fi.cols(); ++q) {
    const T w = qu.sqw[q] * g[q * gs];
    if (w == 0) continue;
    for (index_t a = 0; a < K; ++a) {
      const index_t i = qu.actives(a, q);
      const T wa = w * fi(a, q);
      for (index_t b = 0; b < K; ++b) B.at(i, qu.actives(b, q) - i + qu.p) += wa * fj(b, q);
    }
  }
  return B;
}

// TT core m: g[a + rl*(q + nq*b)]
struct Core { index_t rl, nq, rr; std::vector<T> g; };

// TT-SVD of F (size prod nq, first index fastest; overwritten), absolute tolerance tau.
// Returns false if the tensor is truncated to zero.
static bool svdQR = true;
bool ttsvd(std::vector<T> &F, const std::vector<index_t> &nq, T tau, std::vector<Core> &cores) {
  const index_t d = (index_t)nq.size();
  cores.assign(d, Core());
  std::vector<T> &buf = F;
  index_t rl = 1, total = (index_t)F.size();
  const T tau2 = tau * tau / (T)(d - 1);
  for (index_t m = 0; m + 1 < d; ++m) {
    const index_t rows = rl * nq[m], cols = total / nq[m];
    gsEigen::Map<Mat> C(buf.data(), rows, cols);
    Mat U, V;
    gsEigen::Matrix<T, gsEigen::Dynamic, 1> s;
    const bool wide = svdQR && rows < cols;
    gsEigen::HouseholderQR<Mat> qr;
    if (wide) {
      // X^T = Q R, R^T = U S W^T, so X = U S (Q W)^T
      qr.compute(C.transpose());
      const Mat Rt = qr.matrixQR().topRows(rows).template triangularView<gsEigen::Upper>().transpose();
      gsEigen::JacobiSVD<Mat> svd(Rt, gsEigen::ComputeFullU | gsEigen::ComputeFullV);
      U = svd.matrixU(); s = svd.singularValues(); V = svd.matrixV();
    } else {
      gsEigen::BDCSVD<Mat> svd(C, gsEigen::ComputeThinU | gsEigen::ComputeThinV);
      U = svd.matrixU(); s = svd.singularValues(); V = svd.matrixV();
    }
    index_t r = s.rows();
    T tail = 0;
    while (r > 0 && tail + s[r - 1] * s[r - 1] <= tau2) { tail += s[r - 1] * s[r - 1]; --r; }
    if (r == 0) return false;
    if (wide) {  // only the kept columns of Q W
      Mat QW = Mat::Zero(cols, r);
      QW.topRows(rows) = V.leftCols(r);
      QW.applyOnTheLeft(qr.householderQ());
      V.swap(QW);
    }
    Core &c = cores[m];
    c.rl = rl; c.nq = nq[m]; c.rr = r;
    c.g.assign(U.data(), U.data() + rows * r);
    Mat next = s.head(r).asDiagonal() * V.leftCols(r).transpose();
    buf.assign(next.data(), next.data() + next.size());
    total = cols; rl = r;
  }
  Core &c = cores[d - 1];
  c.rl = rl; c.nq = nq[d - 1]; c.rr = 1; c.g = buf;
  return true;
}

// One coefficient function as a sum of Kronecker products: block (e,c),
// B[m][a*r[m+1]+b] is the univariate matrix of direction m.
struct Term {
  index_t e, c;
  std::vector<index_t> r;
  std::vector<std::vector<Banded>> B;
  Term transposed() const {
    Term t; t.e = c; t.c = e; t.r = r; t.B = B;
    for (auto &v : t.B) for (auto &b : v) b = b.transposed();
    return t;
  }
  size_t bytes() const { size_t s = 0; for (auto &v : B) for (auto &b : v) s += b.v.size() * sizeof(T); return s; }
};

// out += B applied along direction m of a tensor with sizes n (first fastest)
void applyDir(const Banded &B, index_t m, const std::vector<index_t> &n, const T *x, T *y) {
  index_t s = 1, outer = 1;
  for (index_t j = 0; j < m; ++j) s *= n[j];
  for (index_t j = m + 1; j < (index_t)n.size(); ++j) outer *= n[j];
  const index_t nm = n[m], p = B.p;
  for (index_t o = 0; o < outer; ++o)
    for (index_t i = 0; i < nm; ++i) {
      T *yo = y + (o * nm + i) * s;
      for (index_t q = 0; q <= 2 * p; ++q) {
        const index_t j = i + q - p;
        if (j < 0 || j >= nm) continue;
        const T b = B.at(i, q);
        if (b == 0) continue;
        const T *xo = x + (o * nm + j) * s;
        for (index_t t = 0; t < s; ++t) yo[t] += b * xo[t];
      }
    }
}

struct LowRankElasticity {
  index_t d = 0, N = 0;
  std::vector<index_t> n;               // univariate sizes
  std::vector<Term> terms;              // including the mirrored ones
  std::vector<std::vector<index_t>> map; // map[c][I]: free index or -1
  size_t bytes() const { size_t s = 0; for (auto &t : terms) s += t.bytes(); return s; }
  index_t ntens() const { index_t s = 1; for (auto v : n) s *= v; return s; }

  // y += sum over terms of block (e,c) applied to xc (full tensors)
  void applyFull(const std::vector<std::vector<T>> &x, std::vector<std::vector<T>> &y) const {
    const index_t nt = ntens();
    std::vector<std::vector<T>> cur, nxt;
    for (const Term &t : terms) {
      // direction d-1 first
      nxt.assign(t.r[d - 1], std::vector<T>(nt, 0));
      for (index_t a = 0; a < t.r[d - 1]; ++a) applyDir(t.B[d - 1][a], d - 1, n, x[t.c].data(), nxt[a].data());
      for (index_t m = d - 2; m >= 0; --m) {
        cur.swap(nxt);
        nxt.assign(t.r[m], std::vector<T>(nt, 0));
        for (index_t a = 0; a < t.r[m]; ++a)
          for (index_t b = 0; b < t.r[m + 1]; ++b)
            applyDir(t.B[m][a * t.r[m + 1] + b], m, n, cur[b].data(), nxt[a].data());
      }
      std::vector<T> &ye = y[t.e];
      for (index_t i = 0; i < nt; ++i) ye[i] += nxt[0][i];
    }
  }

  // Band tensor of the term over directions 0..d-1, added to out
  // (row I = i_0 + n_0 (i_1 + ...), band index D = o_{d-1} + w_{d-1} (o_{d-2} + ...)),
  // out has ntens()*W entries.
  void addBand(const Term &t, std::vector<T> &out) const {
    std::vector<std::vector<T>> Q(t.r[d - 1]), Qn;
    index_t Nr = n[d - 1], Wr = 2 * t.B[d - 1][0].p + 1;
    for (index_t a = 0; a < t.r[d - 1]; ++a) Q[a] = t.B[d - 1][a].v;
    for (index_t m = d - 2; m >= 0; --m) {
      const index_t nm = n[m], wm = 2 * t.B[m][0].p + 1, Wn = wm * Wr;
      Qn.assign(m == 0 ? 0 : t.r[m], std::vector<T>());
      for (index_t a = 0; a < t.r[m]; ++a) {
        std::vector<T> &dst = (m == 0) ? out : Qn[a];
        if (m != 0) dst.assign(nm * Nr * Wn, 0);
        for (index_t b = 0; b < t.r[m + 1]; ++b) {
          const Banded &Bm = t.B[m][a * t.r[m + 1] + b];
          const std::vector<T> &q = Q[b];
          for (index_t Ir = 0; Ir < Nr; ++Ir)
            for (index_t i = 0; i < nm; ++i) {
              T *row = dst.data() + (i + nm * Ir) * Wn;
              const T *qr = q.data() + Ir * Wr;
              for (index_t o = 0; o < wm; ++o) {
                const T bv = Bm.at(i, o);
                if (bv == 0) continue;
                T *ro = row + o * Wr;
                for (index_t D = 0; D < Wr; ++D) ro[D] += bv * qr[D];
              }
            }
        }
      }
      Q.swap(Qn); Nr *= nm; Wr = Wn;
    }
  }

  // Sparse matrix on the free unknowns
  gsSparseMatrix<T> sparse() const {
    const index_t nt = ntens();
    std::vector<index_t> w(d);
    index_t W = 1;
    for (index_t m = 0; m < d; ++m) { w[m] = 2 * terms[0].B[m][0].p + 1; W *= w[m]; }
    std::vector<std::vector<T>> band(d * d);
    for (const Term &t : terms) {
      std::vector<T> &bd = band[t.e * d + t.c];
      if (bd.empty()) bd.assign(nt * W, 0);
      addBand(t, bd);
    }
    // inverse map
    std::vector<std::pair<index_t, index_t>> inv(N);
    for (index_t c = 0; c < d; ++c)
      for (index_t I = 0; I < nt; ++I) if (map[c][I] >= 0) inv[map[c][I]] = std::make_pair(c, I);
    // The matrix is symmetric: column (c,J) is filled from row (c,J).
    gsSparseMatrix<T> K(N, N);
    K.reserve((size_t)N * d * W);
    std::vector<std::pair<index_t, T>> col;
    std::vector<index_t> idx(d);
    for (index_t g = 0; g < N; ++g) {
      const index_t c = inv[g].first, J = inv[g].second;
      index_t rem = J;
      for (index_t m = 0; m < d; ++m) { idx[m] = rem % n[m]; rem /= n[m]; }
      col.clear();
      for (index_t e = 0; e < d; ++e) {
        const std::vector<T> &bd = band[c * d + e];
        if (bd.empty()) continue;
        for (index_t D = 0; D < W; ++D) {
          // D = o_{d-1} + w_{d-1} (o_{d-2} + ...), I = i_0 + n_0 (i_1 + ...)
          index_t Dr = D, I = 0; bool ok = true;
          for (index_t m = d - 1; m >= 0; --m) {
            const index_t o = Dr % w[m]; Dr /= w[m];
            const index_t j = idx[m] + o - (w[m] - 1) / 2;
            if (j < 0 || j >= n[m]) { ok = false; break; }
            I = I * n[m] + j;
          }
          if (!ok) continue;
          const index_t row = map[e][I];
          // entries of basis functions without a common element stay exactly zero
          if (row < 0 || bd[J * W + D] == 0) continue;
          col.push_back(std::make_pair(row, bd[J * W + D]));
        }
      }
      std::sort(col.begin(), col.end());
      K.startVec(g);
      for (auto &rv : col) K.insertBack(rv.first, g) = rv.second;
    }
    K.finalize();
    return K;
  }
};

class LowRankOp : public gsLinearOperator<T> {
public:
  explicit LowRankOp(const LowRankElasticity &lr) : m(lr) {}
  void apply(const gsMatrix<T> &in, gsMatrix<T> &out) const {
    const index_t nt = m.ntens();
    out.setZero(m.N, in.cols());
    std::vector<std::vector<T>> x(m.d, std::vector<T>(nt)), y(m.d, std::vector<T>(nt));
    for (index_t k = 0; k < in.cols(); ++k) {
      for (index_t c = 0; c < m.d; ++c)
        for (index_t I = 0; I < nt; ++I) { const index_t g = m.map[c][I]; x[c][I] = g < 0 ? 0 : in(g, k); y[c][I] = 0; }
      m.applyFull(x, y);
      for (index_t c = 0; c < m.d; ++c)
        for (index_t I = 0; I < nt; ++I) { const index_t g = m.map[c][I]; if (g >= 0) out(g, k) = y[c][I]; }
    }
  }
  index_t rows() const { return m.N; }
  index_t cols() const { return m.N; }
private:
  const LowRankElasticity &m;
};

struct Timing { double jac = 0, coef = 0, dec = 0, uni = 0; index_t maxr = 0, nkron = 0, nterm = 0; };

// Coefficients |det J| and J^{-1} on the grid of the quadrature nodes
bool evalGeometry(const gsGeometry<T> &geo, const std::vector<UniQuad> &qu,
                  std::vector<T> &det, std::vector<T> &jinv) {
  const index_t d = (index_t)qu.size();
  index_t nq = 1;
  for (auto &q : qu) nq *= q.nodes.cols();
  det.resize(nq); jinv.resize(nq * d * d);
  const index_t chunk = 16384;
  gsMatrix<T> pts, jac;
  T sign = 0;
  for (index_t s = 0; s < nq; s += chunk) {
    const index_t m = std::min(chunk, nq - s);
    pts.resize(d, m);
    for (index_t i = 0; i < m; ++i) {
      index_t rem = s + i;
      for (index_t k = 0; k < d; ++k) { const index_t len = qu[k].nodes.cols(); pts(k, i) = qu[k].nodes(0, rem % len); rem /= len; }
    }
    geo.jacobian_into(pts, jac);
    for (index_t i = 0; i < m; ++i) {
      const Mat J = jac.block(0, i * d, d, d);
      const T dj = J.determinant();
      if (sign == 0) sign = dj > 0 ? 1 : -1;
      if (!(sign * dj > 0)) return false;
      det[s + i] = sign * dj;
      const Mat Ji = J.inverse();
      for (index_t a = 0; a < d * d; ++a) jinv[(s + i) * d * d + a] = Ji(a % d, a / d);
    }
  }
  return true;
}

// Low-rank representation for tolerance eps. det/jinv from evalGeometry.
void buildLowRank(const std::vector<UniQuad> &qu, const std::vector<T> &det, const std::vector<T> &jinv,
                  T lambda, T mu, T eps, LowRankElasticity &lr, Timing &tm) {
  const index_t d = (index_t)qu.size();
  std::vector<index_t> nq(d);
  index_t ntot = 1;
  for (index_t m = 0; m < d; ++m) { nq[m] = qu[m].nodes.cols(); ntot *= nq[m]; }
  // unique coefficient functions: (c,e,k,l) with e>c, or e==c and k<=l
  struct F { index_t c, e, k, l; bool mirror; };
  std::vector<F> fs;
  for (index_t c = 0; c < d; ++c)
    for (index_t e = c; e < d; ++e)
      for (index_t k = 0; k < d; ++k)
        for (index_t l = (e == c ? k : 0); l < d; ++l)
          fs.push_back(F{c, e, k, l, e != c || k != l});
  gsStopwatch sw;
  std::vector<std::vector<T>> vals(fs.size(), std::vector<T>(ntot));
  std::vector<T> nrm(fs.size());
  T fmax = 0;
  for (size_t f = 0; f < fs.size(); ++f) {
    const F &ff = fs[f];
    std::vector<T> &v = vals[f];
    for (index_t q = 0; q < ntot; ++q) {
      const T *Ji = &jinv[q * d * d];  // Ji[a + d*b] = (J^{-1})_{ab}
      T g = 0;
      if (ff.c == ff.e) for (index_t mm = 0; mm < d; ++mm) g += Ji[ff.k + d * mm] * Ji[ff.l + d * mm];
      T a = lambda * Ji[ff.k + d * ff.c] * Ji[ff.l + d * ff.e] + mu * g + mu * Ji[ff.k + d * ff.e] * Ji[ff.l + d * ff.c];
      index_t rem = q; T w = 1;
      for (index_t m = 0; m < d; ++m) { w *= qu[m].sqw[rem % nq[m]]; rem /= nq[m]; }
      v[q] = det[q] * a * w;
    }
    T s = 0; for (T x : v) s += x * x;
    nrm[f] = std::sqrt(s); fmax = std::max(fmax, nrm[f]);
  }
  tm.coef += sw.stop();
  lr.terms.clear();
  tm.maxr = 0; tm.nkron = 0; tm.nterm = 0;
  const T tau = eps * fmax;
  for (size_t f = 0; f < fs.size(); ++f) {
    if (nrm[f] <= tau) continue;
    sw.restart();
    std::vector<Core> cores;
    const bool ok = ttsvd(vals[f], nq, tau, cores);
    std::vector<T>().swap(vals[f]);
    tm.dec += sw.stop();
    if (!ok) continue;
    sw.restart();
    Term t; t.e = fs[f].e; t.c = fs[f].c;
    t.r.assign(d + 1, 1);
    for (index_t m = 1; m < d; ++m) { t.r[m] = cores[m - 1].rr; tm.maxr = std::max(tm.maxr, t.r[m]); }
    t.B.resize(d);
    for (index_t m = 0; m < d; ++m) {
      const Core &co = cores[m];
      for (index_t a = 0; a < co.rl; ++a)
        for (index_t b = 0; b < co.rr; ++b)
          t.B[m].push_back(uniMatrix(qu[m], co.g.data() + a + co.rl * co.nq * b, co.rl, m == fs[f].l, m == fs[f].k));
    }
    index_t nk = 1; for (index_t m = 1; m < d; ++m) nk *= t.r[m];
    tm.nkron += fs[f].mirror ? 2 * nk : nk;
    tm.nterm += fs[f].mirror ? 2 : 1;
    if (fs[f].mirror) { lr.terms.push_back(t.transposed()); }
    lr.terms.push_back(std::move(t));
    tm.uni += sw.stop();
  }
}

#if defined(__unix__) || defined(__APPLE__)
static double rssMB() { struct rusage u; getrusage(RUSAGE_SELF, &u); return u.ru_maxrss / 1024.0; }
#else
static double rssMB() { return 0; } // peak memory is printed on POSIX systems only
#endif

int main(int argc, char **argv) {
  int p = 2, r = 3, g = 0, sparseMax = 200000;
  T nu = 0.3;
  std::string epsList = "1e-2,1e-4,1e-6,1e-8,1e-10,1e-12";
  gsCmdLine cmd("low-rank elasticity assembly");
  cmd.addInt("p", "degree", "spline degree", p);
  cmd.addInt("r", "refine", "uniform refinements", r);
  cmd.addInt("g", "geo", "0 unit cube, 1 quarter annulus lifted to 3D, 2 deformed cube", g);
  cmd.addReal("n", "nu", "Poisson ratio", nu);
  cmd.addString("e", "eps", "comma separated tolerances", epsList);
  bool quadErr = false, bdc = false;
  cmd.addSwitch("B", "bdcsvd", "SVD of each unfolding by BDCSVD instead of QR plus SVD of the triangular factor", bdc);
  cmd.addSwitch("Q", "quad", "also assemble with one more Gauss node per direction and print the difference", quadErr);
  try { cmd.getValues(argc, argv); } catch (int rv) { return rv; }
  svdQR = !bdc;
  const T E = 210e9, lam = E * nu / ((1 + nu) * (1 - 2 * nu)), mu = E / (2 * (1 + nu));
  const char *gname = g == 0 ? "cube" : g == 1 ? "annulus3d" : "deformed";

  gsMultiPatch<T> geo;
  if (g == 1) {
    gsTensorBSpline<2, T>::uPtr q = gsNurbsCreator<T>::BSplineFatQuarterAnnulus(1, 2);
    geo.addPatch(*gsNurbsCreator<T>::lift3D(*q, 1));
  } else if (g == 2) {
    // degree 2, 4 elements per direction; interior control points moved by a smooth
    // field that is not a product of univariate functions
    gsTensorBSpline<3, T>::uPtr cube = gsNurbsCreator<T>::BSplineCube(1);
    cube->degreeElevate(1);
    cube->uniformRefine(); cube->uniformRefine();
    gsMatrix<T> &C = cube->coefs();
    const index_t nc = cube->basis().component(0).size();
    const T a = 0.1, pi = 3.14159265358979323846;
    for (index_t i = 0; i < C.rows(); ++i) {
      const index_t i0 = i % nc, i1 = (i / nc) % nc, i2 = i / (nc * nc);
      if (i0 == 0 || i1 == 0 || i2 == 0 || i0 == nc - 1 || i1 == nc - 1 || i2 == nc - 1) continue;
      const T x = C(i, 0), y = C(i, 1), z = C(i, 2);
      C(i, 0) += a * std::sin(2 * pi * y) * std::sin(pi * z) * std::cos(pi * x / 2);
      C(i, 1) += a * std::sin(2 * pi * z) * std::sin(pi * x) * std::cos(pi * y / 2);
      C(i, 2) += a * std::sin(2 * pi * x) * std::sin(pi * y) * std::cos(pi * z / 2);
    }
    geo.addPatch(*cube);
  } else
    geo.addPatch(*gsNurbsCreator<T>::BSplineCube(1));
  geo.computeTopology();
  gsMultiBasis<T> basis(geo, true);
  basis.setDegree(p);
  for (int i = 0; i < r; ++i) basis.uniformRefine();

  gsFunctionExpr<T> rhs("0", "0", "0", 3), load("0", "0", "-1e5", 3);
  gsBoundaryConditions<T> bc;
  for (int c = 0; c < 3; ++c) bc.addCondition(0, boundary::west, condition_type::dirichlet, nullptr, c);
  bc.addCondition(0, boundary::south, condition_type::dirichlet, nullptr, 1);
  bc.addCondition(0, boundary::east, condition_type::neumann, &load);
  bc.setGeoMap(geo);
  gsElasticityAssembler<T> asmb(geo, basis, bc, rhs);
  asmb.options().setReal("YoungsModulus", E);
  asmb.options().setReal("PoissonsRatio", nu);
  asmb.options().setInt("MaterialLaw", material_law::hooke);
  asmb.options().setInt("DirichletStrategy", dirichlet::elimination);
  gsStopwatch sw;
  asmb.assemble();
  const double tasm = sw.stop();
  const gsSparseMatrix<T> &K = asmb.matrix();
  const gsMatrix<T> &b = asmb.rhs();
  const double rssRef = rssMB();
  const double kMB = (K.nonZeros() * (sizeof(T) + sizeof(int)) + (K.cols() + 1) * sizeof(int)) / 1048576.0;
  std::printf("LR geo=%s p=%d r=%d dofs=%ld nnz=%ld asm_gsel=%.3f K_MB=%.1f rss_MB=%.0f\n", gname, p, r,
              long(K.rows()), long(K.nonZeros()), tasm, kMB, rssRef);
  std::fflush(stdout);

  if (quadErr) {
    // Quadrature error of the default rule: the same assembler with p+2 nodes
    gsElasticityAssembler<T> a2(geo, basis, bc, rhs);
    a2.options() = asmb.options();
    a2.options().setInt("quB", asmb.options().getInt("quB") + 1);
    a2.assemble();
    std::printf("LR geo=%s p=%d r=%d quad_relerr=%.2e\n", gname, p, r,
                double((a2.matrix() - K).norm() / K.norm()));
    std::fflush(stdout);
  }

  // Preconditioner and reference solution
  gsConstantFunction<T> zero(0, 3);
  std::vector<gsBoundaryConditions<T>> bcs(3);
  for (int c = 0; c < 3; ++c) bcs[c].addCondition(0, boundary::west, condition_type::dirichlet, &zero);
  bcs[1].addCondition(0, boundary::south, condition_type::dirichlet, &zero);
  gsOptionList popt = gsAssembler<T>::defaultOptions();
  sw.restart();
  gsLinearOperator<T>::Ptr P = gsPatchPreconditionersCreator<T>::elasticityFastDiagonalizationOp(
      basis[0], bcs, popt, lam, mu, &geo.patch(0));
  const double tprec = sw.stop();
  gsMatrix<T> xref = gsMatrix<T>::Zero(K.rows(), 1);
  {
    gsConjugateGradient<T> cg(K, P);
    cg.setTolerance(1e-12); cg.setMaxIterations(5000);
    sw.restart();
    cg.solve(b, xref);
    std::printf("LR geo=%s p=%d r=%d ref_cg it=%d solve=%.3f prec_setup=%.3f\n", gname, p, r,
                int(cg.iterations()), sw.stop(), tprec);
  }
  // SpMV time of the assembled matrix
  {
    gsMatrix<T> y;
    sw.restart();
    for (int i = 0; i < 10; ++i) y.noalias() = K * xref;
    std::printf("LR geo=%s p=%d r=%d spmv=%.5f\n", gname, p, r, sw.stop() / 10);
  }

  // Low-rank data independent of the tolerance
  const gsBasis<T> &B0 = basis[0];
  const index_t d = 3;
  LowRankElasticity lr;
  lr.d = d; lr.N = K.rows(); lr.n.resize(d);
  sw.restart();
  std::vector<UniQuad> qu(d);
  for (index_t m = 0; m < d; ++m) { setupUniQuad(B0.component(m), asmb.options(), qu[m]); lr.n[m] = qu[m].n; }
  const double tquad = sw.stop();
  const gsSparseSystem<T> &sys = asmb.system();
  lr.map.assign(d, std::vector<index_t>(lr.ntens(), -1));
  index_t off = 0;
  for (index_t c = 0; c < d; ++c) {
    const gsDofMapper &dm = sys.colMapper(c);
    for (index_t I = 0; I < lr.ntens(); ++I) {
      const index_t gi = dm.index(I, 0);
      if (dm.is_free_index(gi)) lr.map[c][I] = off + gi;
    }
    off += dm.freeSize();
  }
  GISMO_ENSURE(off == K.rows(), "free dof count mismatch");
  sw.restart();
  std::vector<T> det, jinv;
  const bool regular = evalGeometry(geo.patch(0), qu, det, jinv);
  const double tjac = sw.stop() + tquad;
  GISMO_ENSURE(regular, "geometry map not regular at a quadrature node");
  index_t nqt = 1; for (auto &q : qu) nqt *= q.nodes.cols();
  std::printf("LR geo=%s p=%d r=%d nq=%ld jac=%.3f geo_MB=%.1f\n", gname, p, r, long(nqt), tjac,
              (det.size() + jinv.size()) * sizeof(T) / 1048576.0);
  std::fflush(stdout);

  const T Knorm = K.norm();
  std::stringstream ss(epsList);
  std::string tok;
  while (std::getline(ss, tok, ',')) {
    const T eps = std::stod(tok);
    Timing tm;
    buildLowRank(qu, det, jinv, lam, mu, eps, lr, tm);
    const double tsetup = tjac + tm.coef + tm.dec + tm.uni;
    // (b) matrix-free
    LowRankOp::Ptr op(new LowRankOp(lr));
    gsMatrix<T> xm = gsMatrix<T>::Zero(K.rows(), 1), y;
    sw.restart();
    op->apply(xref, y);
    const double tapply = sw.stop();
    const T opErr = (y - K * xref).norm() / (K * xref).norm();
    // CG with the matrix-free operator only while an application is cheap
    int itm = -1;
    double tcgm = 0;
    if (tapply < 0.5) {
      gsConjugateGradient<T> cgm(op, P);
      cgm.setTolerance(1e-12); cgm.setMaxIterations(5000);
      sw.restart();
      cgm.solve(b, xm);
      tcgm = sw.stop();
      itm = int(cgm.iterations());
    }
    std::printf("LR geo=%s p=%d r=%d eps=%.0e maxrank=%ld terms=%ld kron=%ld setup=%.3f (jac=%.3f coef=%.3f dec=%.3f uni=%.3f) "
                "factors_MB=%.2f apply=%.4f op_relerr=%.2e mf_it=%d mf_solve=%.3f mf_solerr=%.2e\n",
                gname, p, r, double(eps), long(tm.maxr), long(tm.nterm), long(tm.nkron), tsetup, tjac, tm.coef, tm.dec, tm.uni,
                lr.bytes() / 1048576.0, tapply, double(opErr), itm, tcgm,
                itm < 0 ? -1.0 : double((xm - xref).norm() / xref.norm()));
    std::fflush(stdout);
    if (K.rows() > sparseMax) continue;
    // (a) sparse matrix
    sw.restart();
    gsSparseMatrix<T> Klr = lr.sparse();
    const double tsp = sw.stop();
    const T merr = (Klr - K).norm() / Knorm;
    gsConjugateGradient<T> cgs(Klr, P);
    cgs.setTolerance(1e-12); cgs.setMaxIterations(5000);
    gsMatrix<T> xs = gsMatrix<T>::Zero(K.rows(), 1);
    cgs.solve(b, xs);
    std::printf("LR geo=%s p=%d r=%d eps=%.0e sparse=%.3f total_a=%.3f mat_relerr=%.2e nnz_lr=%ld sp_it=%d sp_solerr=%.2e rss_MB=%.0f\n",
                gname, p, r, double(eps), tsp, tsetup + tsp, double(merr), long(Klr.nonZeros()),
                int(cgs.iterations()), double((xs - xref).norm() / xref.norm()), rssMB());
    std::fflush(stdout);
  }
  return 0;
}

#else // gsElasticity_ENABLED

int main()
{
    gismo::gsInfo << "This example needs the optional module gsElasticity (-DGISMO_OPTIONAL=gsElasticity).\n";
    return 0;
}

#endif // gsElasticity_ENABLED
