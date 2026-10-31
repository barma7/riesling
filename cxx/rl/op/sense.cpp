#include "sense.hpp"

#include "../log/log.hpp"
#include "../sense/sense.hpp"
#include "../sys/threads.hpp"
#include "../tensors.hpp"
#include "top-impl.hpp"

namespace rl::TOps {

namespace {
/*
 * x(v, b) (+)= s * sum_c y(v, b, c) * conj(maps(v, b or 0, c)), v = flattened voxel index.
 * Replaces an Eigen reduction over the outermost (channel) dimension, which ran ~10x slower than
 * memory bandwidth allows. Works on blocks of voxels so that the block of maps stays in cache
 * while all basis images use it. Complex products are written out in real arithmetic, so they
 * vectorise without fast-math.
 */
template <bool accumulate>
void SENSEAdjoint(Cx const *y, Cx const *maps, Cx *x, Index const N, Index const nB, Index const nMB, Index const nC,
                  float const s)
{
  constexpr Index block = 2048;
  Threads::ChunkFor(
    [&](Index const lo, Index const hi) {
      std::vector<float> acc(2 * block);
      for (Index v0 = lo; v0 < hi; v0 += block) {
        Index const n = std::min(block, hi - v0);
        for (Index ib = 0; ib < nB; ib++) {
          std::fill_n(acc.data(), 2 * n, 0.f);
          Index const mb = (nMB == 1) ? 0 : ib;
          for (Index ic = 0; ic < nC; ic++) {
            float const *__restrict yp = reinterpret_cast<float const *>(y + v0 + N * (ib + nB * ic));
            float const *__restrict mp = reinterpret_cast<float const *>(maps + v0 + N * (mb + nMB * ic));
            float *__restrict a = acc.data();
            for (Index iv = 0; iv < n; iv++) {
              float const yr = yp[2 * iv], yi = yp[2 * iv + 1], mr = mp[2 * iv], mi = mp[2 * iv + 1];
              a[2 * iv] += yr * mr + yi * mi;
              a[2 * iv + 1] += yi * mr - yr * mi;
            }
          }
          float *__restrict xp = reinterpret_cast<float *>(x + v0 + N * ib);
          float const *__restrict a = acc.data();
          for (Index iv = 0; iv < 2 * n; iv++) {
            if constexpr (accumulate) {
              xp[iv] += s * a[iv];
            } else {
              xp[iv] = s * a[iv];
            }
          }
        }
      }
    },
    N);
}
} // namespace

auto SENSEOp::Make(Cx5 const &maps, Index const nB) -> Ptr { return std::make_shared<SENSEOp>(maps, nB); }

SENSEOp::SENSEOp(Cx5 const &maps, Index const nB)
  : Parent("SENSEOp")
  , maps_{maps}
{
  for (int ii = 0; ii < 3; ii++) {
    ishape[ii] = maps_.dimension(ii);
    oshape[ii] = maps_.dimension(ii);
    resX.set(ii, maps_.dimension(ii));
  }
  ishape[3] = nB;
  oshape[3] = nB;
  resX.set(3, nB);
  oshape[4] = maps_.dimension(4);
  brdX.set(4, maps_.dimension(4));

  if (maps_.dimension(3) == 1) {
    brdMaps.set(3, nB);
  } else if (maps_.dimension(3) == nB) {
    brdMaps.set(3, 1);
  } else {
    throw Log::Failure("SENSEOp", "Maps had basis size {} expected {}", maps_.dimension(3), nB);
  }
}

void SENSEOp::forward(InCMap x, OutMap y, float const s) const
{
  auto const time = this->startForward(x, y, false);
  y.device(Threads::TensorDevice()) = x.reshape(resX).broadcast(brdX) * maps_.broadcast(brdMaps) * y.constant(s);
  this->finishForward(y, time, false);
}

void SENSEOp::iforward(InCMap x, OutMap y, float const s) const
{
  auto const time = this->startForward(x, y, true);
  y.device(Threads::TensorDevice()) += x.reshape(resX).broadcast(brdX) * maps_.broadcast(brdMaps) * y.constant(s);
  this->finishForward(y, time, true);
}

void SENSEOp::adjoint(OutCMap y, InMap x, float const s) const
{
  auto const time = this->startAdjoint(y, x, false);
  SENSEAdjoint<false>(y.data(), maps_.data(), x.data(), Product(FirstN<3>(ishape)), ishape[3], maps_.dimension(3),
                      oshape[4], s);
  this->finishAdjoint(x, time, false);
}

void SENSEOp::iadjoint(OutCMap y, InMap x, float const s) const
{
  auto const time = this->startAdjoint(y, x, true);
  SENSEAdjoint<true>(y.data(), maps_.data(), x.data(), Product(FirstN<3>(ishape)), ishape[3], maps_.dimension(3),
                     oshape[4], s);
  this->finishAdjoint(x, time, true);
}

auto SENSEOp::nChannels() const -> Index { return oshape[4]; }
auto SENSEOp::mapDimensions() const -> Sz3 { return FirstN<3>(ishape); }
auto SENSEOp::maps() const -> Cx5 { return maps_; }

auto MakeSENSE(Cx5 const &maps, Index const nB) -> SENSEOp::Ptr { return std::make_shared<SENSEOp>(maps, nB); }

} // namespace rl::TOps
