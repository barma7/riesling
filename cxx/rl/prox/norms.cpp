#include "norms.hpp"

#include "../algo/common.hpp"
#include "../log/log.hpp"
#include "../sys/threads.hpp"
#include "../tensors.hpp"

namespace rl::Proxs {

namespace {
/*
 * Multiplies every block of x (the elements that share all non-normed coordinates) by factor(|block|), when the leading
 * dims are not normed, the usual case (e.g. norms over the basis and gradient components of each voxel). The previous
 * code built each block from a shuffle/reshape/chip expression, i.e. generic index arithmetic for every element, which
 * took ~3 s for a 3-component TGV field of a 180x180x326 K=5 image. Here pass 1 adds |x|^2 into one value per block,
 * streaming over x in memory order; pass 2 scales each element. Threads own ranges of the leading (contiguous) dims
 * across all other dims, so no two threads touch the same block. Returns false if dim 0 is normed (not handled).
 */
template <int O, int D, typename F> auto BlockScale(Cx *x, Sz<O> const &shape, Sz<D> const &normDims, F const &factor) -> bool
{
  auto const normed = [&](Index const d) { return std::find(normDims.cbegin(), normDims.cend(), d) != normDims.cend(); };
  Index      K = 0;
  while (K < O && !normed(K)) {
    K++;
  }
  if (K == 0) { return false; }
  Index L = 1, nOuter = 1, nBlockOuter = 1;
  for (Index d = 0; d < K; d++) {
    L *= shape[d];
  }
  std::array<Index, O> bstride{};
  for (Index d = K; d < O; d++) {
    nOuter *= shape[d];
    if (!normed(d)) {
      bstride[d] = nBlockOuter;
      nBlockOuter *= shape[d];
    }
  }
  // Element offset of outer combination c is c * L; its blocks start at blockOuter[c] * L
  std::vector<Index> blockOuter(nOuter);
  for (Index c = 0; c < nOuter; c++) {
    Index rem = c, b = 0;
    for (Index d = K; d < O; d++) {
      b += (rem % shape[d]) * bstride[d];
      rem /= shape[d];
    }
    blockOuter[c] = b;
  }
  std::vector<float> f(nBlockOuter * L, 0.f);
  constexpr Index    chunk = 4096;
  Threads::ChunkFor(
    [&](Index const lo, Index const hi) {
      for (Index v0 = lo; v0 < hi; v0 += chunk) {
        Index const v1 = std::min(v0 + chunk, hi);
        for (Index c = 0; c < nOuter; c++) {
          float const *xc = reinterpret_cast<float const *>(x + c * L);
          float       *fc = f.data() + blockOuter[c] * L;
          for (Index v = v0; v < v1; v++) {
            fc[v] += xc[2 * v] * xc[2 * v] + xc[2 * v + 1] * xc[2 * v + 1];
          }
        }
        for (Index bo = 0; bo < nBlockOuter; bo++) {
          float *fb = f.data() + bo * L;
          for (Index v = v0; v < v1; v++) {
            fb[v] = factor(std::sqrt(fb[v]));
          }
        }
        for (Index c = 0; c < nOuter; c++) {
          float       *xc = reinterpret_cast<float *>(x + c * L);
          float const *fc = f.data() + blockOuter[c] * L;
          for (Index v = v0; v < v1; v++) {
            xc[2 * v] *= fc[v];
            xc[2 * v + 1] *= fc[v];
          }
        }
      }
    },
    L);
  return true;
}
} // namespace

auto L1::Make(float const λ, Index const sz) -> Prox::Ptr { return std::make_shared<L1>(λ, sz); }
auto L1::Make(float const λ, CMap b, Ops::Op::Ptr P) -> Prox::Ptr { return std::make_shared<L1>(λ, b, P); }

L1::L1(float const λ_, Index const sz_)
  : Prox(sz_)
  , λ{λ_}
  , b{nullptr, 0}
{
  Log::Print("L1Prox", "λ {}", λ);
}

L1::L1(float const λ_, CMap b_, Ops::Op::Ptr P_)
  : Prox(b_.size())
  , λ{λ_}
  , b{b_}
  , P{P_}
{
  if (P && P->rows() != b.size()) {
    throw(Log::Failure("L1Prox", "Preconditioner size {} did not match data size {}", P->rows(), b.size()));
  }

  Log::Print("L1Prox", "λ {} with bias", λ);
}

void L1::apply(float const α, Map x) const
{
  float const t = α * λ;
  float const nx = ParallelNorm(x);
  Threads::ChunkFor(
    [t, &x](Index lo, Index hi) {
      for (Index ii = lo; ii < hi; ii++) {
        float const ax = std::abs(x[ii]);
        x[ii] = ax > t ? (1.f - t / ax) * x[ii] : 0.f;
      }
    },
    x.size());
  Log::Debug("L1Prox", "α {:4.3E} λ {:4.3E} t {:4.3E} |x| {:4.3E} |z| {:4.3E}", α, λ, t, nx, ParallelNorm(x));
}

void L1::conj(float const α, Map x) const
{
  float const nx = ParallelNorm(x);
  if (b.size()) {
    if (P) {
      P->iforward(b, x, -(α * λ));
    } else {
      x.device(Threads::CoreDevice()) = x - (α * λ) * b;
    }
  }
  x.device(Threads::CoreDevice()) = λ * x.array() / x.array().abs().max(λ).cast<Cx>();
  Log::Debug("L1Prox", "Conjugate λ {:4.3E} |x| {:4.3E} |z| {:4.3E}", λ, nx, ParallelNorm(x));
}

auto L1I::Make(float const λ, Index const sz) -> Prox::Ptr { return std::make_shared<L1I>(λ, sz); }

L1I::L1I(float const λ_, Index const sz_)
  : Prox(sz_)
  , λ{λ_}
{
  Log::Print("L1Prox", "λ {}", λ);
}

void L1I::apply(float const α, Map x) const
{
  float const t = α * λ;
  float const nx = ParallelNorm(x);
  Threads::ChunkFor(
    [t, &x](Index lo, Index hi) {
      for (Index ii = lo; ii < hi; ii++) {
        float const xr = x[ii].real();
        float const xi = x[ii].imag();
        float const ax = std::abs(xi);
        x[ii] = Cx(xr, ax > t ? (1.f - t / ax) * xi : 0.f);
      }
    },
    x.size());
  Log::Debug("L1IProx", "α {} λ {} t {} |x| {} |z| {}", α, λ, t, nx, ParallelNorm(x));
}

void L1I::conj(float const α, Map x) const
{
  float const nx = ParallelNorm(x); // Cursed users might do this in place and overwrite x
  Threads::ChunkFor(
    [λ = this->λ, &x](Index lo, Index hi) {
      for (Index ii = lo; ii < hi; ii++) {
        float const xi = x[ii].imag();
        x[ii] = Cx(0.f, λ * xi / std::max(std::abs(xi), λ));
      }
    },
    x.size());
  Log::Debug("L1IProx", "Conjugate λ {} |x| {} |z| {}", λ, nx, ParallelNorm(x));
}

template <int O, int D> auto L2<O, D>::Make(float const λ, Sz<O> const &s, Sz<D> const &d) -> Prox::Ptr
{
  return std::make_shared<L2>(λ, s, d);
}

template <int O, int D> L2<O, D>::L2(float const λ_, Sz<O> const &s, Sz<D> const &d)
  : Prox(Product(s))
  , λ{λ_}
  , shape{s}
  , normDims(d)
{
  Log::Print("L2Prox", "λ {} shape {} norm dims {}", λ_, shape, normDims);
  Sz<O> all;
  std::iota(all.begin(), all.end(), 0);
  std::set_difference(all.cbegin(), all.cend(), normDims.cbegin(), normDims.cend(), otherDims.begin());
}

template <int O, int D> void L2<O, D>::apply(float const α, Map x) const
{
  Eigen::TensorMap<CxN<O>> xm(x.data(), shape);
  float const              nx = ParallelNorm(x);
  Index const              nElems = std::transform_reduce(normDims.cbegin(), normDims.cend(), 1L, std::multiplies{},
                                                          [sh = this->shape](Index const id) { return sh[id]; });
  Index const              nBlocks = std::transform_reduce(otherDims.cbegin(), otherDims.cend(), 1L, std::multiplies{},
                                                           [sh = this->shape](Index const id) { return sh[id]; });
  Sz<O> const              shuff = Concatenate(normDims, otherDims);
  Sz<D + 1>                rsh;
  std::transform(normDims.cbegin(), normDims.cend(), rsh.begin(), [sh = this->shape](Index const id) { return sh[id]; });
  rsh[D] = nBlocks;
  Sz<1> const rel{nElems};
  float const t = α * λ; // * std::sqrt(nElems);
  if (BlockScale<O, D>(x.data(), shape, normDims, [t](float const norm) { return norm > t ? 1.f - t / norm : 0.f; })) {
    Log::Debug("L2Prox", "α {} λ {} t {} |x| {} |z| {}", α, λ, t, nx, ParallelNorm(x));
    return;
  }
  Threads::ChunkFor(
    [&](Index lo, Index hi) {
      for (Index ib = lo; ib < hi; ib++) {
        auto       xblk = xm.shuffle(shuff).reshape(rsh).template chip<D>(ib);
        auto const norm = Norm<false>(xblk);
        if (norm > t) {
          xblk *= xblk.constant(1.f - t / norm);
        } else {
          xblk.setZero();
        }
      }
    },
    nBlocks);
  Log::Debug("L2Prox", "α {} λ {} t {} |x| {} |z| {}", α, λ, t, nx, ParallelNorm(x));
}

template <int O, int D> void L2<O, D>::conj(float const, Map x) const
{
  Eigen::TensorMap<CxN<O>> xm(x.data(), shape);
  float const              nx = ParallelNorm(x);

  Index const nElems = std::transform_reduce(normDims.cbegin(), normDims.cend(), 1L, std::multiplies{},
                                             [sh = this->shape](Index const id) { return sh[id]; });
  Index const nBlocks = std::transform_reduce(otherDims.cbegin(), otherDims.cend(), 1L, std::multiplies{},
                                              [sh = this->shape](Index const id) { return sh[id]; });
  Sz<O> const shuff = Concatenate(normDims, otherDims);
  Sz<D + 1>   rsh;
  std::transform(normDims.cbegin(), normDims.cend(), rsh.begin(), [sh = this->shape](Index const id) { return sh[id]; });
  rsh[D] = nBlocks;
  Sz<1> const rel{nElems};

  if (BlockScale<O, D>(x.data(), shape, normDims, [l = λ](float const norm) { return norm > l ? l / norm : 1.f; })) {
    Log::Debug("L2Prox", "Conjugate λ {} |x| {} |z| {}", λ, nx, ParallelNorm(x));
    return;
  }
  Threads::ChunkFor(
    [&](Index lo, Index hi) {
      for (Index ib = lo; ib < hi; ib++) {
        auto       xblk = xm.shuffle(shuff).reshape(rsh).template chip<D>(ib);
        auto const norm = Norm<false>(xblk);
        if (norm > λ) { xblk *= xblk.constant(λ / norm); }
      }
    },
    nBlocks);
  Log::Debug("L2Prox", "Conjugate λ {} |x| {} |z| {}", λ, nx, ParallelNorm(x));
}

template struct L2<1, 1>;
template struct L2<5, 1>;
template struct L2<5, 2>;
template struct L2<6, 1>;
template struct L2<6, 2>;
template struct L2<6, 3>;

auto SumOfSquares::Make(CMap b_, Ops::Op::Ptr P_, bool const w) -> Prox::Ptr
{
  return std::make_shared<SumOfSquares>(b_, P_, w);
}

SumOfSquares::SumOfSquares(CMap b_, Ops::Op::Ptr P_, bool const w)
  : Prox(b_.size())
  , b{b_}
  , P{P_}
  , weighted{w}
{
  if (weighted && !P) { throw(Log::Failure("SoS", "Weighted least squares needs the weights (preconditioner)")); }
  if (P && P->rows() != b.size()) {
    throw(Log::Failure("SoS", "Preconditioner size {} did not match data size {}", P->rows(), b.size()));
  }
  Log::Print("SoS", "|b| {}", ParallelNorm(b));
}

void SumOfSquares::apply(float const α, Map x) const
{
  float const nx = ParallelNorm(x);
  /* Worked this out by following §2.2 of Parykh and Boyd */
  x.device(Threads::CoreDevice()) = (x - b) / (1.f + α) + b;
  Log::Debug("SoS", "α {} |x| {} |z| {}", α, nx, ParallelNorm(x));
}

void SumOfSquares::conj(float const α, Map x) const
{
  float const nx = ParallelNorm(x);
  if (P && weighted) { // f*(y) = ½|y|²_P⁻¹ + <y, b> with the dual step αP: y = (x - αPb) / (1 + α)
    P->iforward(b, x, -α);
    x.device(Threads::CoreDevice()) = x / Cx(1.f + α);
  } else if (P) {
    P->iforward(b, x, -α);
    P->inverse(CMap(x.data(), x.size()), x, α, 1.f);
  } else {
    x.device(Threads::CoreDevice()) = (x - α * b) / (1.f + α);
  }
  Log::Debug("SoS", "α {} |x| {} |z| {}", α, nx, ParallelNorm(x));
}

} // namespace rl::Proxs
