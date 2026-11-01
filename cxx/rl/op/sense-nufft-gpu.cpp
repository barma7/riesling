#include "sense-nufft-gpu.hpp"

#include "../kernel/kernel.hpp"
#include "../log/log.hpp"
#include "../sys/threads.hpp"
#include "../tensors.hpp"
#include "apodize.hpp"
#include "top-impl.hpp"

#ifdef RL_CUDA
#include "../gpu/snufft.hpp"
#endif

#include <stdexcept>
#include <type_traits>

namespace rl::TOps {

auto GPUAvailable() -> bool
{
#ifdef RL_CUDA
  return true;
#else
  return false;
#endif
}

auto HostPinner() -> std::function<std::shared_ptr<void>(void *, size_t)>
{
#ifdef RL_CUDA
  return [](void *p, size_t const bytes) -> std::shared_ptr<void> {
    auto const t = Log::Now();
    if (!gpu::PinHost(p, bytes)) {
      Log::Print("GPU", "Could not page-lock {:.2f} GB of host memory, copies will be slower", bytes / 1.e9);
      return nullptr;
    }
    Log::Print("GPU", "Page-locked {:.2f} GB of host memory in {}", bytes / 1.e9, Log::ToNow(t));
    return std::shared_ptr<void>(p, [](void *q) { gpu::UnpinHost(q); });
  };
#else
  return nullptr;
#endif
}

auto KSpaceWeights(TOp<5, 5>::Ptr M) -> Re3
{
  if (!M) { return Re3(); }
  Cx5 ones(M->ishape);
  ones.setConstant(Cx(1.f));
  Cx5 const   w = M->forward(ones);
  Index const nC = w.dimension(0), nS = w.dimension(1), nT = w.dimension(2), nO = w.dimension(3) * w.dimension(4);
  Cx const   *wd = w.data();
  Index const nK = nC * nS * nT;
  bool        same = true;
  for (Index ik = 0; ik < nK; ik++) {
    if (wd[ik].imag() != 0.f) { throw Log::Failure("GPU", "k-space weights are not real"); }
    if (wd[ik] != wd[ik - ik % nC]) { same = false; }
  }
  for (Index io = 1; io < nO; io++) { // The device applies the same weights to every slab and time point
    for (Index ik = 0; ik < nK; ik++) {
      if (wd[ik + nK * io] != wd[ik]) { throw Log::Failure("GPU", "k-space weights differ between time points"); }
    }
  }
  Re3 r(same ? 1 : nC, nS, nT);
  for (Index it = 0; it < nT; it++) {
    for (Index is = 0; is < nS; is++) {
      for (Index ic = 0; ic < r.dimension(0); ic++) {
        r(ic, is, it) = wd[ic + nC * (is + nS * it)].real();
      }
    }
  }
  return r;
}

#ifdef RL_CUDA
namespace {
// Apodization correction (as SENSENUFFT: Apodize's adjoint of a grid of ones), kernel β and scale
template <typename KF> auto KernelTerms(float const osamp, Sz3 const mat, Sz3 const grid, float &β, float &scale) -> Re3
{
  Apodize<3, 2, KF> apo(AddBack(mat, 1, 1), AddBack(grid, 1, 1), osamp);
  Cx5               ones(apo.oshape);
  ones.setConstant(Cx(1.f));
  Cx5 const     w = apo.adjoint(ones);
  Kernel<3, KF> k(osamp);
  scale = k.scale;
  if constexpr (!std::is_same_v<KF, TopHat<1>>) { β = k.f.β; }
  return w.real().reshape(mat);
}

struct SENSENUFFTGPUNormal final : TOp<4, 4>
{
  TOP_INHERIT(4, 4)
  TOP_DECLARE(SENSENUFFTGPUNormal)
  SENSENUFFTGPUNormal(std::shared_ptr<gpu::SNUFFT> i, Sz4 const shape)
    : Parent("SNUFFTGPU-N", shape, shape)
    , impl_{i}
  {
  }
  void iforward(InCMap x, OutMap y, float const s = 1.f) const;
  void iadjoint(OutCMap y, InMap x, float const s = 1.f) const;

private:
  std::shared_ptr<gpu::SNUFFT> impl_;
};

void SENSENUFFTGPUNormal::forward(InCMap x, OutMap y, float const s) const
{
  auto const time = this->startForward(x, y, false);
  impl_->normal(x.data(), y.data(), s);
  this->finishForward(y, time, false);
}

void SENSENUFFTGPUNormal::adjoint(OutCMap y, InMap x, float const s) const // Hermitian
{
  auto const time = this->startAdjoint(y, x, false);
  impl_->normal(y.data(), x.data(), s);
  this->finishAdjoint(x, time, false);
}

void SENSENUFFTGPUNormal::iforward(InCMap x, OutMap y, float const s) const
{
  auto const time = this->startForward(x, y, true);
  InTensor   t(ishape);
  impl_->normal(x.data(), t.data(), s);
  y.device(Threads::TensorDevice()) += t;
  this->finishForward(y, time, true);
}

void SENSENUFFTGPUNormal::iadjoint(OutCMap y, InMap x, float const s) const
{
  auto const time = this->startAdjoint(y, x, true);
  InTensor   t(ishape);
  impl_->normal(y.data(), t.data(), s);
  x.device(Threads::TensorDevice()) += t;
  this->finishAdjoint(x, time, true);
}
} // namespace
#endif

SENSENUFFTGPU::SENSENUFFTGPU(
  GridOpts<3> const &opts, TrajectoryN<3> const &traj, Cx5 const &maps, Basis::CPtr basis, Re3 const &weights)
  : Parent("SENSENUFFTGPU")
{
#ifndef RL_CUDA
  throw Log::Failure("GPU", "riesling was built without CUDA (configure with -DRL_CUDA=ON)");
#else
  Sz3 const   mat = FirstN<3>(maps.dimensions());
  Index const nB = basis ? basis->nB() : 1, nMB = maps.dimension(3), nC = maps.dimension(4);
  Index const nS = traj.nSamples(), nT = traj.nTraces();
  if (mat != traj.matrixForFOV(opts.fov)) {
    throw Log::Failure("GPU", "SENSE maps matrix {} does not match the FOV matrix {}", mat, traj.matrixForFOV(opts.fov));
  }
  if (nMB != 1 && nMB != nB) { throw Log::Failure("GPU", "Maps had basis size {} expected 1 or {}", nMB, nB); }
  if (weights.size() && (weights.dimension(1) != nS || weights.dimension(2) != nT ||
                         (weights.dimension(0) != 1 && weights.dimension(0) != nC))) {
    throw Log::Failure("GPU", "Weights {} do not match k-space {}x{}x{}", weights.dimensions(), nC, nS, nT);
  }
  Sz3 const grid = MulToEven(traj.matrixForFOV(opts.fov), opts.osamp);
  grid_ = grid;
  ishape = AddBack(mat, nB);
  oshape = Sz3{nC, nS, nT};

  // Grid coordinates, the same expression as TrajectoryN::toCoordLists
  using Arrayf = Eigen::Array<float, 3, 1>;
  Arrayf omat, tmat;
  for (Index ii = 0; ii < 3; ii++) {
    tmat[ii] = traj.matrix()[ii];
    omat[ii] = grid[ii];
  }
  Arrayf const osamp = omat / tmat;
  Arrayf const k0 = omat / 2;
  Re3          coords(3, nS, nT);
  for (int32_t it = 0; it < nT; it++) {
    for (int32_t is = 0; is < nS; is++) {
      Arrayf const p = traj.point(is, it) * 1.f;
      Arrayf const k = p * osamp + k0;
      for (Index ii = 0; ii < 3; ii++) {
        coords(ii, is, it) = k[ii];
      }
    }
  }

  gpu::SNUFFTSetup set;
  Re3              apo;
  if (opts.tophat) {
    apo = KernelTerms<TopHat<1>>(opts.osamp, mat, grid, set.beta, set.kScale);
  } else {
    switch (opts.kW) {
    case 4: apo = KernelTerms<ExpSemi<4>>(opts.osamp, mat, grid, set.beta, set.kScale); break;
    case 6: apo = KernelTerms<ExpSemi<6>>(opts.osamp, mat, grid, set.beta, set.kScale); break;
    case 8: apo = KernelTerms<ExpSemi<8>>(opts.osamp, mat, grid, set.beta, set.kScale); break;
    default: throw Log::Failure("GPU", "Kernel width {} not supported", opts.kW);
    }
  }
  set.nx = mat[0], set.ny = mat[1], set.nz = mat[2];
  set.gx = grid[0], set.gy = grid[1], set.gz = grid[2];
  set.nB = nB, set.nC = nC, set.nMB = nMB;
  set.nSamp = nS, set.nTrace = nT;
  set.kW = opts.kW;
  set.tophat = opts.tophat;
  set.coords = coords.data();
  set.maps = maps.data();
  set.apo = apo.data();
  if (basis) {
    set.basis = basis->B.data();
    set.nBS = basis->nSample();
    set.nBT = basis->nTrace();
    set.basisScale = basis->scale;
  }
  if (weights.size()) {
    set.weights = weights.data();
    set.weightsPerChannel = weights.dimension(0) > 1;
  }
  try {
    impl_ = std::make_shared<gpu::SNUFFT>(set);
  } catch (std::exception const &e) {
    throw Log::Failure("GPU", "{}", e.what());
  }
  Log::Print("GPU",
             "SENSE+NUFFT on the GPU: image {} grid {} basis {} channels {}, {} of {} samples inside the grid, {:.2f} GB on "
             "the device",
             mat, grid, nB, nC, impl_->nValid(), nS * nT, impl_->deviceBytes() / 1.e9);
#endif
}

SENSENUFFTGPU::~SENSENUFFTGPU() { logTimes(); }

auto SENSENUFFTGPU::device() const -> std::shared_ptr<gpu::SNUFFT> { return impl_; }

void SENSENUFFTGPU::logTimes() const
{
#ifdef RL_CUDA
  if (!impl_) { return; }
  auto const t = impl_->times();
  if (t.calls == 0) { return; }
  Log::Print("GPU",
             "{} calls, device time (s): copy in {:.2f} expand {:.2f} FFT {:.2f} interp {:.2f} spread {:.2f} iFFT {:.2f} "
             "collect {:.2f} copy out {:.2f}",
             t.calls, t.copyIn / 1e3, t.expand / 1e3, t.fft / 1e3, t.interp / 1e3, t.spread / 1e3, t.ifft / 1e3,
             t.collect / 1e3, t.copyOut / 1e3);
#endif
}

auto SENSENUFFTGPU::gridAdjoint(Cx3 const &y) const -> Cx5
{
#ifdef RL_CUDA
  if (y.dimensions() != oshape) { throw Log::Failure("GPU", "y dims {} expected {}", y.dimensions(), oshape); }
  Cx5 grid(AddBack(grid_, ishape[3], oshape[0]));
  impl_->gridAdjoint(y.data(), grid.data());
  return grid;
#else
  return Cx5();
#endif
}

auto SENSENUFFTGPU::normal() const -> TOp<4, 4>::Ptr
{
#ifdef RL_CUDA
  return std::make_shared<SENSENUFFTGPUNormal>(impl_, ishape);
#else
  return nullptr;
#endif
}

void SENSENUFFTGPU::forward(InCMap x, OutMap y, float const s) const
{
  auto const time = this->startForward(x, y, false);
#ifdef RL_CUDA
  impl_->forward(x.data(), y.data(), s);
#endif
  this->finishForward(y, time, false);
}

void SENSENUFFTGPU::iforward(InCMap x, OutMap y, float const s) const
{
  auto const time = this->startForward(x, y, true);
#ifdef RL_CUDA
  OutTensor t(oshape);
  impl_->forward(x.data(), t.data(), s);
  y.device(Threads::TensorDevice()) += t;
#endif
  this->finishForward(y, time, true);
}

void SENSENUFFTGPU::adjoint(OutCMap y, InMap x, float const s) const
{
  auto const time = this->startAdjoint(y, x, false);
#ifdef RL_CUDA
  impl_->adjoint(y.data(), x.data(), s);
#endif
  this->finishAdjoint(x, time, false);
}

void SENSENUFFTGPU::iadjoint(OutCMap y, InMap x, float const s) const
{
  auto const time = this->startAdjoint(y, x, true);
#ifdef RL_CUDA
  InTensor t(ishape);
  impl_->adjoint(y.data(), t.data(), s);
  x.device(Threads::TensorDevice()) += t;
#endif
  this->finishAdjoint(x, time, true);
}

auto MakeSENSENUFFTGPU(
  GridOpts<3> const &opts, TrajectoryN<3> const &traj, Cx5 const &maps, Basis::CPtr basis, Re3 const &weights)
  -> std::shared_ptr<SENSENUFFTGPU>
{
  return std::make_shared<SENSENUFFTGPU>(opts, traj, maps, basis, weights);
}

} // namespace rl::TOps
