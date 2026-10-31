#include "rl/basis/basis.hpp"
#include "rl/log/log.hpp"
#include "rl/op/compose.hpp"
#include "rl/op/nufft.hpp"
#include "rl/op/sense-nufft.hpp"
#include "rl/op/sense.hpp"
#include "rl/sys/threads.hpp"
#include "rl/tensors.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

using namespace rl;
using namespace Catch;

namespace {
template <typename T> auto RelDiff(T const &a, T const &b) -> float
{
  T const d = a - b;
  return Norm<false>(d) / Norm<false>(b);
}
} // namespace

TEST_CASE("SENSENUFFT", "[op]")
{
  Threads::SetGlobalThreadCount(GENERATE(1, 4));
  float const osamp = GENERATE(1.f, 1.3f, 2.f);
  Index const nMB = GENERATE(1, 3);  // One set of maps for all basis images, or one per basis image
  bool const  smallFOV = GENERATE(false, true);
  Index const nC = 3, nS = 16, nT = 8, nB = 3;
  Sz3 const   matrix{12, 10, 8};
  Re3         points(3, nS, nT);
  points.setRandom();
  for (Index id = 0; id < 3; id++) {
    points.chip<0>(id) = points.chip<0>(id) * (0.45f * matrix[id]); // Some points near the edge
  }
  TrajectoryN<3> const traj(points, matrix);
  Basis                basis(nB, 1, nT);
  basis.B.setRandom();
  GridOpts<3> opts{.osamp = osamp};
  if (smallFOV) { opts.fov = GridOpts<3>::Arrayf(10.f, 8.f, 6.f); } // Image matrix smaller than the grid: padding
  Sz3 const imgMatrix = traj.matrixForFOV(opts.fov);

  Cx5 maps(AddBack(imgMatrix, nMB, nC));
  maps.setRandom();
  auto const ref = TOps::MakeCompose(TOps::MakeSENSE(maps, nB), TOps::MakeNUFFT<3>(opts, traj, nC, &basis));
  auto const fused = TOps::MakeSENSENUFFT(opts, traj, maps, &basis);
  REQUIRE(fused->ishape == ref->ishape);
  REQUIRE(fused->oshape == ref->oshape);

  Cx4 x(ref->ishape);
  Cx3 y(ref->oshape);
  x.setRandom();
  y.setRandom();
  float const s = 0.7f;
  float const tol = 1.e-5f; // Multi-threaded gridding adjoint sums in a varying order

  Cx3 const yr = ref->forward(x, s);
  Cx3 const yf = fused->forward(x, s);
  CHECK(RelDiff(yf, yr) == Approx(0).margin(tol));
  Cx4 const xr = ref->adjoint(y, s);
  Cx4 const xf = fused->adjoint(y, s);
  CHECK(RelDiff(xf, xr) == Approx(0).margin(tol));

  // In-place versions add to the output
  Cx3 yri = y, yfi = y;
  ref->iforward(CxNCMap<4>(x.data(), x.dimensions()), CxNMap<3>(yri.data(), yri.dimensions()), s);
  fused->iforward(CxNCMap<4>(x.data(), x.dimensions()), CxNMap<3>(yfi.data(), yfi.dimensions()), s);
  CHECK(RelDiff(yfi, yri) == Approx(0).margin(tol));
  Cx4 xri = x, xfi = x;
  ref->iadjoint(CxNCMap<3>(y.data(), y.dimensions()), CxNMap<4>(xri.data(), xri.dimensions()), s);
  fused->iadjoint(CxNCMap<3>(y.data(), y.dimensions()), CxNMap<4>(xfi.data(), xfi.dimensions()), s);
  CHECK(RelDiff(xfi, xri) == Approx(0).margin(tol));

  // Dot test, relative to the norms (random complex vectors give dot products far smaller than the norms)
  auto const xx = Dot<false>(x, xf);
  auto const yy = Dot<false>(yf, y);
  CHECK(std::abs(yy - xx) / (Norm<false>(yf) * Norm<false>(y)) == Approx(0).margin(1.e-5));

  // Repeated calls reuse the workspace: the padding must be zero again after an adjoint
  Cx3 const yf2 = fused->forward(x, s);
  CHECK(RelDiff(yf2, yr) == Approx(0).margin(tol));
}
