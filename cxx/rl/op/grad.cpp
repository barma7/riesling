#include "grad.hpp"

#include "../log/log.hpp"
#include "../sys/threads.hpp"
#include "top-impl.hpp"

namespace rl::TOps {

namespace {

/*
 * These difference functions have circular boundary conditions and are scaled to have max
 * eigenvalue = 1
 */

template <bool fwd, typename T1, typename T2, int N>
inline void ForwardDiff(T1 const &a, T2 &&b, Sz<N> const shape, Index const dim, float const s = 1.f)
{
  Sz<N> sz = shape, sz1 = shape, f0, fp1, fm1;
  fp1[dim] = 1;
  fm1[dim] = sz[dim] - 1;
  sz[dim] -= 1;
  sz1[dim] = 1;
  if constexpr (fwd) {
    Log::Debug("Grad", "Forward forward differences dim {}", dim, dim);
    b.slice(f0, sz).device(Threads::TensorDevice()) += (a.slice(fp1, sz) - a.slice(f0, sz)) * b.slice(f0, sz).constant(s / 2.f);
    b.slice(fm1, sz1).device(Threads::TensorDevice()) +=
      (a.slice(f0, sz1) - a.slice(fm1, sz1)) * b.slice(fm1, sz1).constant(s / 2.f);
  } else {
    Log::Debug("Grad", "Adjoint forward differences dim {}", dim, dim);
    b.slice(fp1, sz).device(Threads::TensorDevice()) +=
      (a.slice(f0, sz) - a.slice(fp1, sz)) * b.slice(f0, sz).constant(s / 2.f);
    b.slice(f0, sz1).device(Threads::TensorDevice()) +=
      (a.slice(fm1, sz1) - a.slice(f0, sz1)) * b.slice(f0, sz1).constant(s / 2.f);
  }
}

template <bool fwd, typename T1, typename T2, int N>
inline void BackwardDiff(T1 const &a, T2 &&b, Sz<N> const shape, Index const dim, float const s = 1.f)
{
  Sz<N> sz = shape, sz1 = shape, f0, fp1, fm1;
  fp1[dim] = 1;
  fm1[dim] = sz[dim] - 1;
  sz[dim] -= 1;
  sz1[dim] = 1;
  if constexpr (fwd) {
    Log::Debug("Grad", "Forward backward differences dim {}", dim);
    b.slice(fp1, sz).device(Threads::TensorDevice()) +=
      (a.slice(fp1, sz) - a.slice(f0, sz)) * b.slice(f0, sz).constant(s / 2.f);
    b.slice(f0, sz1).device(Threads::TensorDevice()) +=
      (a.slice(f0, sz1) - a.slice(fm1, sz1)) * b.slice(fm1, sz1).constant(s / 2.f);
  } else {
    Log::Debug("Grad", "Adjoint backward differences dim {}", dim);
    b.slice(f0, sz).device(Threads::TensorDevice()) += (a.slice(f0, sz) - a.slice(fp1, sz)) * b.slice(f0, sz).constant(s);
    b.slice(fm1, sz1).device(Threads::TensorDevice()) +=
      (a.slice(fm1, sz1) - a.slice(f0, sz1)) * b.slice(f0, sz1).constant(s / 2.f);
  }
}
/*
 * Fused 3D gradients over dims 0, 1, 2 for a batch of V volumes, circular boundaries, used for the 5D / 6D shapes of the
 * TV and TGV regularizers. The slice expressions above make one pass over the data per difference (3 for Grad, 9 for
 * GradVec) and ran 10-20x slower than memory bandwidth allows. These compute the same differences with the same
 * coefficients in one pass. The GradVec adjoint keeps the weights of BackwardDiff<false> (interior s, last point s/2),
 * so results do not change. Note that these weights make it about 2x the true adjoint of BackwardDiff<true>.
 */
struct Vol3
{
  Index nx, ny, nz, V;
  auto  N() const -> Index { return nx * ny * nz; }
  auto  NV() const -> Index { return nx * ny * nz * V; }
};

template <bool add> inline void Put(Cx &y, Cx const v)
{
  if constexpr (add) {
    y += v;
  } else {
    y = v;
  }
}

// Runs f(row pointer offset, j, k, j-1, j+1, k-1, k+1 row offsets) for every row of every volume, in parallel
template <typename F> void ForRows(Vol3 const g, F const &f)
{
  Threads::ChunkFor(
    [&](Index const lo, Index const hi) {
      for (Index r = lo; r < hi; r++) {
        Index const j = r % g.ny, k = (r / g.ny) % g.nz, v = r / (g.ny * g.nz);
        Index const jm = j == 0 ? g.ny - 1 : j - 1, jp = j + 1 == g.ny ? 0 : j + 1;
        Index const km = k == 0 ? g.nz - 1 : k - 1, kp = k + 1 == g.nz ? 0 : k + 1;
        auto const  row = [&](Index const jj, Index const kk) { return v * g.N() + (kk * g.ny + jj) * g.nx; };
        f(row(j, k), j, k, row(jm, k), row(jp, k), row(j, km), row(j, kp));
      }
    },
    g.ny * g.nz * g.V);
}

// y_d (+)= c * (x[i + e_d] - x[i]), d = 0, 1, 2
template <bool add> void Grad3Forward(Cx const *x, Cx *y, Vol3 const g, float const c)
{
  Index const NV = g.NV();
  ForRows(g, [&](Index const o, Index, Index, Index, Index const ojp, Index, Index const okp) {
    Cx const *x0 = x + o, *xj = x + ojp, *xk = x + okp;
    Cx       *y0 = y + o, *y1 = y + NV + o, *y2 = y + 2 * NV + o;
    for (Index i = 0; i < g.nx; i++) {
      Index const ip = i + 1 == g.nx ? 0 : i + 1;
      Put<add>(y0[i], (x0[ip] - x0[i]) * c);
      Put<add>(y1[i], (xj[i] - x0[i]) * c);
      Put<add>(y2[i], (xk[i] - x0[i]) * c);
    }
  });
}

// x (+)= c * sum_d (y_d[i - e_d] - y_d[i])
template <bool add> void Grad3Adjoint(Cx const *y, Cx *x, Vol3 const g, float const c)
{
  Index const NV = g.NV();
  ForRows(g, [&](Index const o, Index, Index, Index const ojm, Index, Index const okm, Index) {
    Cx const *y0 = y + o, *y1 = y + NV + o, *y2 = y + 2 * NV + o, *y1m = y + NV + ojm, *y2m = y + 2 * NV + okm;
    Cx       *x0 = x + o;
    for (Index i = 0; i < g.nx; i++) {
      Index const im = i == 0 ? g.nx - 1 : i - 1;
      Put<add>(x0[i], (y0[im] - y0[i]) * c + (y1m[i] - y1[i]) * c + (y2m[i] - y2[i]) * c);
    }
  });
}

// Symmetrized gradient of a 3-component field: with B_d(u) = u[i] - u[i - e_d],
// y0..2 (+)= cd * B_d(x_d), y3 (+)= co * (B_0(x1) + B_1(x0)), y4 ... (x2, x0), y5 ... (x2, x1)
template <bool add> void GradVec3Forward(Cx const *x, Cx *y, Vol3 const g, float const cd, float const co)
{
  Index const NV = g.NV();
  ForRows(g, [&](Index const o, Index, Index, Index const ojm, Index, Index const okm, Index) {
    Cx const *x0 = x + o, *x1 = x + NV + o, *x2 = x + 2 * NV + o;
    Cx const *x0j = x + ojm, *x1j = x + NV + ojm, *x2j = x + 2 * NV + ojm;
    Cx const *x0k = x + okm, *x1k = x + NV + okm, *x2k = x + 2 * NV + okm;
    Cx       *y0 = y + o, *y1 = y + NV + o, *y2 = y + 2 * NV + o, *y3 = y + 3 * NV + o, *y4 = y + 4 * NV + o,
       *y5 = y + 5 * NV + o;
    for (Index i = 0; i < g.nx; i++) {
      Index const im = i == 0 ? g.nx - 1 : i - 1;
      Put<add>(y0[i], (x0[i] - x0[im]) * cd);
      Put<add>(y1[i], (x1[i] - x1j[i]) * cd);
      Put<add>(y2[i], (x2[i] - x2k[i]) * cd);
      Put<add>(y3[i], (x1[i] - x1[im]) * co + (x0[i] - x0j[i]) * co);
      Put<add>(y4[i], (x2[i] - x2[im]) * co + (x0[i] - x0k[i]) * co);
      Put<add>(y5[i], (x2[i] - x2j[i]) * co + (x1[i] - x1k[i]) * co);
    }
  });
}

// Adjoint as in BackwardDiff<false>: with A_d(u; w) = (u[i] - u[i + e_d]) * (w, or w / 2 at the last point along d),
// x0 (+)= A_0(y0; sd) + A_1(y3; so) + A_2(y4; so), x1 (+)= A_0(y3; so) + A_1(y1; sd) + A_2(y5; so),
// x2 (+)= A_0(y4; so) + A_1(y5; so) + A_2(y2; sd)
template <bool add> void GradVec3Adjoint(Cx const *y, Cx *x, Vol3 const g, float const sd, float const so)
{
  Index const NV = g.NV();
  ForRows(g, [&](Index const o, Index const j, Index const k, Index, Index const ojp, Index, Index const okp) {
    Cx const *y0 = y + o, *y1 = y + NV + o, *y2 = y + 2 * NV + o, *y3 = y + 3 * NV + o, *y4 = y + 4 * NV + o,
             *y5 = y + 5 * NV + o;
    Cx const *y1j = y + NV + ojp, *y3j = y + 3 * NV + ojp, *y5j = y + 5 * NV + ojp;
    Cx const *y2k = y + 2 * NV + okp, *y4k = y + 4 * NV + okp, *y5k = y + 5 * NV + okp;
    Cx       *x0 = x + o, *x1 = x + NV + o, *x2 = x + 2 * NV + o;
    bool const  lastj = j + 1 == g.ny, lastk = k + 1 == g.nz;
    float const sdj = lastj ? sd / 2.f : sd, soj = lastj ? so / 2.f : so;
    float const sdk = lastk ? sd / 2.f : sd, sok = lastk ? so / 2.f : so;
    for (Index i = 0; i < g.nx; i++) {
      bool const  lasti = i + 1 == g.nx;
      Index const ip = lasti ? 0 : i + 1;
      float const sdi = lasti ? sd / 2.f : sd, soi = lasti ? so / 2.f : so;
      Put<add>(x0[i], (y0[i] - y0[ip]) * sdi + (y3[i] - y3j[i]) * soj + (y4[i] - y4k[i]) * sok);
      Put<add>(x1[i], (y3[i] - y3[ip]) * soi + (y1[i] - y1j[i]) * sdj + (y5[i] - y5k[i]) * sok);
      Put<add>(x2[i], (y4[i] - y4[ip]) * soi + (y5[i] - y5j[i]) * soj + (y2[i] - y2k[i]) * sdk);
    }
  });
}

template <int NG> auto IsXYZ(Sz<NG> const &d) -> bool { return NG == 3 && d[0] == 0 && d[1] == 1 && d[2] == 2; }
} // namespace

template <int ND, int NG> Grad<ND, NG>::Grad(InDims const ish, Sz<NG> const d)
  : Parent("Grad", ish, AddBack(ish, (Index)d.size()))
  , dims_{d}
{
}

template <int ND, int NG> auto Grad<ND, NG>::Make(InDims const ish, Sz<NG> const d) -> std::shared_ptr<Grad>
{
  return std::make_shared<Grad>(ish, d);
}

template <int ND, int NG> void Grad<ND, NG>::forward(InCMap x, OutMap y, float const s) const
{
  if constexpr (ND == 5 && NG == 3) {
    if (IsXYZ<NG>(dims_)) {
      auto const  time = this->startForward(x, y, false);
      float const scale = s / std::sqrt(NG);
      Grad3Forward<false>(x.data(), y.data(), Vol3{x.dimension(0), x.dimension(1), x.dimension(2), x.dimension(3) * x.dimension(4)}, scale / 2.f);
      this->finishForward(y, time, false);
      return;
    }
  }
  y.setZero();
  iforward(x, y, s);
}

template <int ND, int NG> void Grad<ND, NG>::adjoint(OutCMap y, InMap x, float const s) const
{
  if constexpr (ND == 5 && NG == 3) {
    if (IsXYZ<NG>(dims_)) {
      auto const  time = this->startAdjoint(y, x, false);
      float const scale = s / std::sqrt(NG);
      Grad3Adjoint<false>(y.data(), x.data(), Vol3{x.dimension(0), x.dimension(1), x.dimension(2), x.dimension(3) * x.dimension(4)}, scale / 2.f);
      this->finishAdjoint(x, time, false);
      return;
    }
  }
  x.setZero();
  iadjoint(y, x, s);
}

template <int ND, int NG> void Grad<ND, NG>::iforward(InCMap x, OutMap y, float const s) const
{
  auto const  time = this->startForward(x, y, true);
  float const scale = s / std::sqrt(NG);
  if constexpr (ND == 5 && NG == 3) {
    if (IsXYZ<NG>(dims_)) {
      Grad3Forward<true>(x.data(), y.data(), Vol3{x.dimension(0), x.dimension(1), x.dimension(2), x.dimension(3) * x.dimension(4)}, scale / 2.f);
      this->finishForward(y, time, true);
      return;
    }
  }
  for (Index ii = 0; ii < NG; ii++) {
    ForwardDiff<true>(x, y.template chip<ND>(ii), x.dimensions(), dims_[ii], scale);
  }
  this->finishForward(y, time, true);
}

template <int ND, int NG> void Grad<ND, NG>::iadjoint(OutCMap y, InMap x, float const s) const
{
  auto const  time = this->startAdjoint(y, x, true);
  float const scale = s / std::sqrt(NG);
  if constexpr (ND == 5 && NG == 3) {
    if (IsXYZ<NG>(dims_)) {
      Grad3Adjoint<true>(y.data(), x.data(), Vol3{x.dimension(0), x.dimension(1), x.dimension(2), x.dimension(3) * x.dimension(4)}, scale / 2.f);
      this->finishAdjoint(x, time, true);
      return;
    }
  }
  for (Index ii = 0; ii < NG; ii++) {
    ForwardDiff<false>(y.template chip<ND>(ii), x, x.dimensions(), dims_[ii], scale);
  }
  this->finishAdjoint(x, time, true);
}

template struct Grad<1, 1>;
template struct Grad<2, 2>;
template struct Grad<3, 3>;

template struct Grad<5, 1>;
template struct Grad<5, 3>;

template <int ND, int NG> Div<ND, NG>::Div(OutDims const sh, Sz<NG> const d)
  : Parent("Grad", AddBack(sh, (Index)d.size()), sh)
  , dims_{d}
{
}

template <int ND, int NG> auto Div<ND, NG>::Make(OutDims const sh, Sz<NG> const d) -> std::shared_ptr<Div>
{
  return std::make_shared<Div>(sh, d);
}

template <int ND, int NG> void Div<ND, NG>::forward(InCMap x, OutMap y, float const s) const
{
  y.setZero();
  iforward(x, y, s);
}

template <int ND, int NG> void Div<ND, NG>::adjoint(OutCMap y, InMap x, float const s) const
{
  x.setZero();
  iadjoint(y, x, s);
}

template <int ND, int NG> void Div<ND, NG>::iforward(InCMap x, OutMap y, float const s) const
{
  auto const time = this->startForward(x, y, true);

  float const scale = s / std::sqrt(2 * NG);
  for (Index ii = 0; ii < NG; ii++) {
    BackwardDiff<true>(x.template chip<ND>(ii), y, y.dimensions(), dims_[ii], scale);
  }
  this->finishForward(y, time, true);
}

template <int ND, int NG> void Div<ND, NG>::iadjoint(OutCMap y, InMap x, float const s) const
{
  auto const  time = this->startAdjoint(y, x, true);
  float const scale = s / std::sqrt(2 * NG);
  for (Index ii = 0; ii < NG; ii++) {
    BackwardDiff<false>(y, x.template chip<ND>(ii), y.dimensions(), dims_[ii], scale);
  }
  this->finishAdjoint(x, time, true);
}

template struct Div<1, 1>;
template struct Div<2, 2>;
template struct Div<3, 3>;
template struct Div<5, 3>;

template <int ND, int NG> GradVec<ND, NG>::GradVec(InDims const ish, Sz<NG> const dims)
  : Parent("GradV", ish, AddBack(FirstN<ND - 1>(ish), (Index)((dims.size() * (dims.size() + 1)) / 2)))
  , dims_{dims}
{
  if ((Index)dims.size() != ishape[ND - 1]) {
    throw(Log::Failure("gradv", "Symmetrized gradient only, dims were {} and {}", dims.size(), ishape[ND - 1]));
  }
}

template <int ND, int NG> auto GradVec<ND, NG>::Make(InDims const ish, Sz<NG> const d) -> std::shared_ptr<GradVec>
{
  return std::make_shared<GradVec>(ish, d);
}

template <int ND, int NG> void GradVec<ND, NG>::forward(InCMap x, OutMap y, float const s) const
{
  if constexpr (ND == 6 && NG == 3) {
    if (IsXYZ<NG>(dims_)) {
      auto const  time = this->startForward(x, y, false);
      float const scale = s / std::sqrt(2 * NG * (NG + 1) / 2);
      float const scale2 = scale / std::sqrt(2);
      GradVec3Forward<false>(x.data(), y.data(), Vol3{x.dimension(0), x.dimension(1), x.dimension(2), x.dimension(3) * x.dimension(4)}, scale / 2.f, scale2 / 2.f);
      this->finishForward(y, time, false);
      return;
    }
  }
  y.setZero();
  iforward(x, y, s);
}

template <int ND, int NG> void GradVec<ND, NG>::iforward(InCMap x, OutMap y, float const s) const
{
  auto const time = this->startForward(x, y, true);
  auto const sz = FirstN<ND - 1>(x.dimensions());
  /*
   * Grad applied to a vector produces a tensor. Here it is flattened back into a vector
   */
  float const scale = s / std::sqrt(2 * NG * (NG + 1) / 2);
  if constexpr (ND == 6 && NG == 3) {
    if (IsXYZ<NG>(dims_)) {
      float const scale2 = scale / std::sqrt(2);
      GradVec3Forward<true>(x.data(), y.data(), Vol3{x.dimension(0), x.dimension(1), x.dimension(2), x.dimension(3) * x.dimension(4)}, scale / 2.f, scale2 / 2.f);
      this->finishForward(y, time, false);
      return;
    }
  }
  Index yind = NG;
  for (Index ii = 0; ii < NG; ii++) {
    BackwardDiff<true>(x.template chip<ND - 1>(ii), y.template chip<ND - 1>(ii), sz, dims_[ii], scale);
    for (Index ij = ii + 1; ij < NG; ij++) {
      BackwardDiff<true>(x.template chip<ND - 1>(ij), y.template chip<ND - 1>(yind), sz, dims_[ii], scale / std::sqrt(2));
      BackwardDiff<true>(x.template chip<ND - 1>(ii), y.template chip<ND - 1>(yind), sz, dims_[ij], scale / std::sqrt(2));
      yind++;
    }
  }
  this->finishForward(y, time, false);
}

template <int ND, int NG> void GradVec<ND, NG>::adjoint(OutCMap y, InMap x, float const s) const
{
  if constexpr (ND == 6 && NG == 3) {
    if (IsXYZ<NG>(dims_)) {
      auto const  time = this->startAdjoint(y, x, false);
      float const scale = s / std::sqrt(2 * NG * (NG + 1) / 2);
      float const scale2 = scale / std::sqrt(2);
      GradVec3Adjoint<false>(y.data(), x.data(), Vol3{x.dimension(0), x.dimension(1), x.dimension(2), x.dimension(3) * x.dimension(4)}, scale, scale2);
      this->finishAdjoint(x, time, false);
      return;
    }
  }
  x.setZero();
  iadjoint(y, x, s);
}

template <int ND, int NG> void GradVec<ND, NG>::iadjoint(OutCMap y, InMap x, float const s) const
{
  auto const time = this->startAdjoint(y, x, true);
  auto const sz = FirstN<ND - 1>(x.dimensions());
  /*
   *  This is the tensor form of Div (see wikipedia page) but with the tensor flattened into a vector
   */
  float const scale = s / std::sqrt(2 * NG * (NG + 1) / 2);
  if constexpr (ND == 6 && NG == 3) {
    if (IsXYZ<NG>(dims_)) {
      float const scale2 = scale / std::sqrt(2);
      GradVec3Adjoint<true>(y.data(), x.data(), Vol3{x.dimension(0), x.dimension(1), x.dimension(2), x.dimension(3) * x.dimension(4)}, scale, scale2);
      this->finishAdjoint(x, time, false);
      return;
    }
  }
  Index yind = NG;
  for (Index ii = 0; ii < NG; ii++) {
    BackwardDiff<false>(y.template chip<ND - 1>(ii), x.template chip<ND - 1>(ii), sz, dims_[ii], scale);
    for (Index ij = ii + 1; ij < NG; ij++) {
      BackwardDiff<false>(y.template chip<ND - 1>(yind), x.template chip<ND - 1>(ii), sz, dims_[ij], scale / std::sqrt(2));
      BackwardDiff<false>(y.template chip<ND - 1>(yind), x.template chip<ND - 1>(ij), sz, dims_[ii], scale / std::sqrt(2));
      yind++;
    }
  }
  this->finishAdjoint(x, time, false);
}

template struct GradVec<2, 1>;
template struct GradVec<3, 2>;
template struct GradVec<4, 3>;
template struct GradVec<6, 3>;

} // namespace rl::TOps
