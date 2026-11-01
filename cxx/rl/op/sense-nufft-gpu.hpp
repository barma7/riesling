#pragma once

#include "../basis/basis.hpp"
#include "../trajectory.hpp"
#include "grid-opts.hpp"
#include "top.hpp"

#include <functional>

namespace rl::gpu {
class SNUFFT;
}

namespace rl::TOps {

/*
 * SENSENUFFT computed on a CUDA GPU (rl/gpu/snufft.cu): the same operator as TOps::SENSENUFFT (3D, one slab), to rounding.
 * Host vectors are copied to the device and back on every call; the maps, trajectory and weights stay on the device.
 *
 * normal() gives Aᴴ W A, W the k-space weights passed here (the preconditioner M of the recon), as one device pass per
 * call: the ADMM CG x-update needs only that, and then the k-space data of the iterations never leave the device.
 */
struct SENSENUFFTGPU final : TOp<4, 3>
{
  TOP_INHERIT(4, 3)
  TOP_DECLARE(SENSENUFFTGPU)
  // weights: (1 or nC, nSamp, nTrace), or empty for none (normal() is then Aᴴ A)
  SENSENUFFTGPU(GridOpts<3> const &opts, TrajectoryN<3> const &traj, Cx5 const &maps, Basis::CPtr basis, Re3 const &weights);
  ~SENSENUFFTGPU();
  void iforward(InCMap x, OutMap y, float const s = 1.f) const;
  void iadjoint(OutCMap y, InMap x, float const s = 1.f) const;

  auto normal() const -> TOp<4, 4>::Ptr; // Aᴴ W A on the image space
  auto gridAdjoint(Cx3 const &y) const -> Cx5; // Gridding adjoint only (x, y, z on the grid, b, c), for accuracy checks
  void logTimes() const;                 // Device time per stage so far (debug log)
  auto device() const -> std::shared_ptr<gpu::SNUFFT>; // The device operator (device-pointer interface), for GPU solvers

private:
  std::shared_ptr<gpu::SNUFFT> impl_;
  Sz3                          grid_;
};

auto MakeSENSENUFFTGPU(GridOpts<3> const &opts,
                       TrajectoryN<3> const &traj,
                       Cx5 const           &maps,
                       Basis::CPtr          basis,
                       Re3 const           &weights) -> std::shared_ptr<SENSENUFFTGPU>;

// Weights of a k-space weighting operator (a diagonal TensorScale, e.g. the preconditioner): M applied to ones. Returned
// as (1, nSamp, nTrace) when they are the same for all channels, else (nC, nSamp, nTrace); empty for M == nullptr.
auto KSpaceWeights(TOp<5, 5>::Ptr M) -> Re3;

auto GPUAvailable() -> bool; // Built with CUDA

// For ADMM::pinHost: page-locks host memory until the returned token is released; nullptr without CUDA
auto HostPinner() -> std::function<std::shared_ptr<void>(void *, size_t)>;

} // namespace rl::TOps
