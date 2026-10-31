#pragma once

#include "grid.hpp"

namespace rl::TOps {

/*
 * NUFFT(SENSE(x)) in 3D: the same operator as Compose(SENSEOp, NUFFT), without the channel images in between. The SENSE
 * product, the apodization correction and the padding to the grid are one pass that writes the gridding workspace; the
 * adjoint crops, apodizes and sums over channels in one pass. Saves a pass and a temporary of the size of the grid.
 */
template <typename KF> struct SENSENUFFT final : TOp<4, 3>
{
  TOP_INHERIT(4, 3)
  TOP_DECLARE(SENSENUFFT)
  SENSENUFFT(GridOpts<3> const &opts, TrajectoryN<3> const &traj, Cx5 const &maps, Basis::CPtr basis);
  void iforward(InCMap x, OutMap y, float const s = 1.f) const;
  void iadjoint(OutCMap y, InMap x, float const s = 1.f) const;

private:
  Grid<3, KF> gridder;
  Cx5         maps_;
  Re3         apo_; // Apodization correction on the image grid, as TOps::Apodize applies it
  Sz3         padLeft_;
  Cx5 mutable workspace;

  void                            imageToGrid(InCMap x, float const s) const;
  template <bool accumulate> void gridToImage(InMap x, float const s) const;
};

auto MakeSENSENUFFT(GridOpts<3> const &opts, TrajectoryN<3> const &traj, Cx5 const &maps, Basis::CPtr basis)
  -> TOp<4, 3>::Ptr;

} // namespace rl::TOps
