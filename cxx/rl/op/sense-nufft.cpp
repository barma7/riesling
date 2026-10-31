#include "sense-nufft.hpp"

#include "../fft.hpp"
#include "../log/log.hpp"
#include "../sys/threads.hpp"
#include "../tensors.hpp"
#include "apodize.hpp"
#include "top-impl.hpp"

#include <cmath>

namespace rl::TOps {

template <typename KF>
SENSENUFFT<KF>::SENSENUFFT(GridOpts<3> const &opts, TrajectoryN<3> const &traj, Cx5 const &maps, Basis::CPtr basis)
  : Parent("SENSENUFFT")
  , gridder{opts, traj, maps.dimension(4), basis}
  , maps_{maps}
  , workspace{gridder.ishape}
{
  Sz3 const   mat = FirstN<3>(maps_.dimensions());
  Index const nB = gridder.ishape[3];
  if (mat != traj.matrixForFOV(opts.fov)) {
    throw Log::Failure("SENSENUFFT", "SENSE maps matrix {} does not match the FOV matrix {}", mat, traj.matrixForFOV(opts.fov));
  }
  if (maps_.dimension(3) != 1 && maps_.dimension(3) != nB) {
    throw Log::Failure("SENSENUFFT", "Maps had basis size {} expected 1 or {}", maps_.dimension(3), nB);
  }
  ishape = AddBack(mat, nB);
  oshape = gridder.oshape;
  // The correction NUFFT's Apodize multiplies by (its adjoint of a grid of ones; real, as it is stored with zero imaginary part)
  Apodize<3, 2, KF> apo(AddBack(mat, 1, 1), AddBack(FirstN<3>(gridder.ishape), 1, 1), opts.osamp);
  Cx5               ones(apo.oshape);
  ones.setConstant(Cx(1.f));
  Cx5 const w = apo.adjoint(ones);
  apo_ = w.real().reshape(mat);
  for (int ii = 0; ii < 3; ii++) {
    padLeft_[ii] = (gridder.ishape[ii] - mat[ii] + 1) / 2;
  }
  Log::Print("SENSENUFFT", "ishape {} oshape {} grid {}", ishape, oshape, gridder.ishape);
}

/*
 * workspace(pad(v), b, c) = ((x(v, b) * maps(v, b or 0, c)) * apo(v)) * s, zero outside the image. The same products in the
 * same order as SENSEOp (Eigen's complex product: one fused multiply-add per component) followed by Apodize.
 */
template <typename KF> void SENSENUFFT<KF>::imageToGrid(InCMap x, float const s) const
{
  auto const  time = Log::Now();
  Index const nx = ishape[0], ny = ishape[1], nz = ishape[2], nB = ishape[3];
  Index const gx = workspace.dimension(0), gy = workspace.dimension(1), gz = workspace.dimension(2);
  Index const nC = workspace.dimension(4), nMB = maps_.dimension(3);
  Index const N = nx * ny * nz, G = gx * gy * gz;
  Index const px = padLeft_[0], py = padLeft_[1], pz = padLeft_[2];
  Cx const   *xd = x.data();
  Cx const   *md = maps_.data();
  float const *ad = apo_.data();
  Cx         *wd = workspace.data();
  Threads::ChunkFor(
    [&](Index const lo, Index const hi) {
      for (Index row = lo; row < hi; row++) { // Grid row (all of dimension 0) at (jg, kg)
        Index const j = row % gy - py, k = row / gy - pz;
        bool const  inside = j >= 0 && j < ny && k >= 0 && k < nz;
        Index const v0 = nx * (j + ny * k);
        for (Index ic = 0; ic < nC; ic++) {
          for (Index ib = 0; ib < nB; ib++) {
            Cx *w = wd + gx * row + G * (ib + nB * ic);
            if (!inside) {
              std::fill_n(w, gx, Cx(0.f));
              continue;
            }
            std::fill_n(w, px, Cx(0.f));
            std::fill_n(w + px + nx, gx - px - nx, Cx(0.f));
            Index const        mb = (nMB == 1) ? 0 : ib;
            float const *__restrict xp = reinterpret_cast<float const *>(xd + v0 + N * ib);
            float const *__restrict mp = reinterpret_cast<float const *>(md + v0 + N * (mb + nMB * ic));
            float const *__restrict ap = ad + v0;
            float *__restrict wp = reinterpret_cast<float *>(w + px);
            for (Index iv = 0; iv < nx; iv++) {
              float const xr = xp[2 * iv], xi = xp[2 * iv + 1], mr = mp[2 * iv], mi = mp[2 * iv + 1];
              float const tr = std::fma(xr, mr, -(xi * mi));
              float const ti = std::fma(xr, mi, xi * mr);
              wp[2 * iv] = (tr * ap[iv]) * s;
              wp[2 * iv + 1] = (ti * ap[iv]) * s;
            }
          }
        }
      }
    },
    gy * gz);
  Log::Debug("SNUFFT", "SENSE+apodize forward took {}", Log::ToNow(time));
}

/*
 * x(v, b) (+)= s * sum_c (workspace(pad(v), b, c) * apo(v)) * conj(maps(v, b or 0, c)), as Apodize's adjoint followed by
 * SENSEOp's. Works on image rows so that the row of maps stays in cache for all basis images.
 */
template <typename KF> template <bool accumulate> void SENSENUFFT<KF>::gridToImage(InMap x, float const s) const
{
  auto const  time = Log::Now();
  Index const nx = ishape[0], ny = ishape[1], nz = ishape[2], nB = ishape[3];
  Index const gx = workspace.dimension(0), gy = workspace.dimension(1), gz = workspace.dimension(2);
  Index const nC = workspace.dimension(4), nMB = maps_.dimension(3);
  Index const N = nx * ny * nz, G = gx * gy * gz;
  Index const px = padLeft_[0], py = padLeft_[1], pz = padLeft_[2];
  Cx         *xd = x.data();
  Cx const   *md = maps_.data();
  float const *ad = apo_.data();
  Cx const   *wd = workspace.data();
  Threads::ChunkFor(
    [&](Index const lo, Index const hi) {
      std::vector<float> acc(2 * nx);
      for (Index row = lo; row < hi; row++) { // Image row at (j, k)
        Index const j = row % ny, k = row / ny;
        Index const v0 = nx * row;
        Index const g0 = px + gx * ((j + py) + gy * (k + pz));
        float const *__restrict ap = ad + v0;
        for (Index ib = 0; ib < nB; ib++) {
          std::fill(acc.begin(), acc.end(), 0.f);
          Index const mb = (nMB == 1) ? 0 : ib;
          for (Index ic = 0; ic < nC; ic++) {
            float const *__restrict wp = reinterpret_cast<float const *>(wd + g0 + G * (ib + nB * ic));
            float const *__restrict mp = reinterpret_cast<float const *>(md + v0 + N * (mb + nMB * ic));
            float *__restrict a = acc.data();
            for (Index iv = 0; iv < nx; iv++) {
              float const yr = wp[2 * iv] * ap[iv], yi = wp[2 * iv + 1] * ap[iv];
              float const mr = mp[2 * iv], mi = mp[2 * iv + 1];
              a[2 * iv] += yr * mr + yi * mi;
              a[2 * iv + 1] += yi * mr - yr * mi;
            }
          }
          float *__restrict xp = reinterpret_cast<float *>(xd + v0 + N * ib);
          float const *__restrict a = acc.data();
          for (Index iv = 0; iv < 2 * nx; iv++) {
            if constexpr (accumulate) {
              xp[iv] += s * a[iv];
            } else {
              xp[iv] = s * a[iv];
            }
          }
        }
      }
    },
    ny * nz);
  Log::Debug("SNUFFT", "SENSE+apodize adjoint took {}", Log::ToNow(time));
}

template <typename KF> void SENSENUFFT<KF>::forward(InCMap x, OutMap y, float const s) const
{
  auto const time = this->startForward(x, y, false);
  imageToGrid(x, s);
  CxNMap<5> wsm(workspace.data(), workspace.dimensions());
  FFT::Forward(wsm, Sz3{0, 1, 2});
  gridder.forward(CxNCMap<5>(workspace.data(), workspace.dimensions()), y);
  this->finishForward(y, time, false);
}

template <typename KF> void SENSENUFFT<KF>::iforward(InCMap x, OutMap y, float const s) const
{
  auto const time = this->startForward(x, y, true);
  imageToGrid(x, s);
  CxNMap<5> wsm(workspace.data(), workspace.dimensions());
  FFT::Forward(wsm, Sz3{0, 1, 2});
  gridder.iforward(CxNCMap<5>(workspace.data(), workspace.dimensions()), y);
  this->finishForward(y, time, true);
}

template <typename KF> void SENSENUFFT<KF>::adjoint(OutCMap y, InMap x, float const s) const
{
  auto const time = this->startAdjoint(y, x, false);
  CxNMap<5>  wsm(workspace.data(), workspace.dimensions());
  gridder.adjoint(y, wsm);
  FFT::Adjoint(wsm, Sz3{0, 1, 2});
  gridToImage<false>(x, s);
  this->finishAdjoint(x, time, false);
}

template <typename KF> void SENSENUFFT<KF>::iadjoint(OutCMap y, InMap x, float const s) const
{
  auto const time = this->startAdjoint(y, x, true);
  CxNMap<5>  wsm(workspace.data(), workspace.dimensions());
  gridder.adjoint(y, wsm);
  FFT::Adjoint(wsm, Sz3{0, 1, 2});
  gridToImage<true>(x, s);
  this->finishAdjoint(x, time, true);
}

template struct SENSENUFFT<ExpSemi<4>>;
template struct SENSENUFFT<ExpSemi<6>>;
template struct SENSENUFFT<ExpSemi<8>>;
template struct SENSENUFFT<TopHat<1>>;

auto MakeSENSENUFFT(GridOpts<3> const &opts, TrajectoryN<3> const &traj, Cx5 const &maps, Basis::CPtr basis)
  -> TOp<4, 3>::Ptr
{
  if (opts.tophat) { return std::make_shared<SENSENUFFT<TopHat<1>>>(opts, traj, maps, basis); }
  switch (opts.kW) {
  case 4: return std::make_shared<SENSENUFFT<ExpSemi<4>>>(opts, traj, maps, basis);
  case 6: return std::make_shared<SENSENUFFT<ExpSemi<6>>>(opts, traj, maps, basis);
  case 8: return std::make_shared<SENSENUFFT<ExpSemi<8>>>(opts, traj, maps, basis);
  default: throw(Log::Failure("SENSENUFFT", "Kernel width {} not supported", opts.kW));
  }
}

} // namespace rl::TOps
