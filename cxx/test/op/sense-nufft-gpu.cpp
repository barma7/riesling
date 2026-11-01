#include "rl/basis/basis.hpp"
#include "rl/log/log.hpp"
#include "rl/op/sense-nufft-gpu.hpp"
#include "rl/op/sense-nufft.hpp"
#include "rl/op/tensorscale.hpp"
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

TEST_CASE("SENSENUFFTGPU", "[op][gpu]")
{
  float const osamp = GENERATE(1.f, 1.3f, 2.f);
  Index const kW = GENERATE(0, 4, 6); // 0: top-hat
  Index const nMB = GENERATE(1, 3);
  bool const  smallFOV = GENERATE(false, true);
  bool const  useBasis = GENERATE(true, false);
  Index const wMode = GENERATE(0, 1, 2); // Weights: none, shared by the channels, per channel
  Index const nC = 3, nS = 16, nT = 8, nB = useBasis ? 3 : 1;
  if (nMB > nB) { return; }
  if (kW == 6 && smallFOV) { return; } // The 7-tap kernel does not fit the small image (Apodize)
  INFO("osamp " << osamp << " kW " << kW << " nMB " << nMB << " smallFOV " << smallFOV << " basis " << useBasis << " weights "
                << wMode);
  Sz3 const matrix{12, 10, 8};
  Re3       points(3, nS, nT);
  points.setRandom();
  for (Index id = 0; id < 3; id++) {
    points.chip<0>(id) = points.chip<0>(id) * (0.45f * matrix[id]);
  }
  points(0, 3, 2) = std::numeric_limits<float>::quiet_NaN(); // Ignored by both
  TrajectoryN<3> const traj(points, matrix);
  Basis                basis(nB, 1, nT);
  basis.B.setRandom();
  basis.B(1, 0, 3) = Cx(0.f); // A zero entry: skipped
  GridOpts<3> opts{.osamp = osamp, .tophat = kW == 0, .kW = kW == 0 ? 4 : kW};
  if (smallFOV) { opts.fov = GridOpts<3>::Arrayf(10.f, 8.f, 6.f); }
  Sz3 const imgMatrix = traj.matrixForFOV(opts.fov);

  Cx5 maps(AddBack(imgMatrix, nMB, nC));
  maps.setRandom();
  Re3 W;
  if (wMode > 0) {
    W.resize(wMode == 1 ? 1 : nC, nS, nT);
    W.setRandom();
    W = W.abs() + 0.1f;
  }
  auto const cpu = TOps::MakeSENSENUFFT(opts, traj, maps, useBasis ? &basis : nullptr);
  auto const gpu = TOps::MakeSENSENUFFTGPU(opts, traj, maps, useBasis ? &basis : nullptr, W);
  REQUIRE(gpu->ishape == cpu->ishape);
  REQUIRE(gpu->oshape == cpu->oshape);

  Cx4 x(cpu->ishape);
  Cx3 y(cpu->oshape);
  x.setRandom();
  y.setRandom();
  float const s = 0.7f;
  /* Rounding differences (cuFFT vs DUCC, summation order) are amplified by the conditioning of the operator: without
   * oversampling the apodization correction is large at the edge of the FOV, more so for wider kernels. The CPU's own fused
   * and composed SENSE+NUFFT adjoints (different summation order) differ by 3e-5 (width 6) and 5e-5 (width 8) at osamp 1.
   */
  float const tol = osamp > 1.f ? 2.e-5f : (kW == 6 ? 3.e-4f : 5.e-5f);

  Cx3 const yc = cpu->forward(x, s);
  Cx3 const yg = gpu->forward(x, s);
  CHECK(RelDiff(yg, yc) == Approx(0).margin(tol));
  Cx4 const xc = cpu->adjoint(y, s);
  Cx4 const xg = gpu->adjoint(y, s);
  CHECK(RelDiff(xg, xc) == Approx(0).margin(tol));

  Cx3 yci = y, ygi = y;
  cpu->iforward(CxNCMap<4>(x.data(), x.dimensions()), CxNMap<3>(yci.data(), yci.dimensions()), s);
  gpu->iforward(CxNCMap<4>(x.data(), x.dimensions()), CxNMap<3>(ygi.data(), ygi.dimensions()), s);
  CHECK(RelDiff(ygi, yci) == Approx(0).margin(tol));
  Cx4 xci = x, xgi = x;
  cpu->iadjoint(CxNCMap<3>(y.data(), y.dimensions()), CxNMap<4>(xci.data(), xci.dimensions()), s);
  gpu->iadjoint(CxNCMap<3>(y.data(), y.dimensions()), CxNMap<4>(xgi.data(), xgi.dimensions()), s);
  CHECK(RelDiff(xgi, xci) == Approx(0).margin(tol));

  // Normal operator: Aᴴ W A on the device against the CPU operator's forward, the weights, its adjoint
  Cx3 yw = cpu->forward(x);
  if (wMode == 1) {
    yw = yw * W.cast<Cx>().broadcast(Sz3{nC, 1, 1});
  } else if (wMode == 2) {
    yw = yw * W.cast<Cx>();
  }
  Cx4 const nc = cpu->adjoint(yw, s);
  auto      N = gpu->normal();
  Cx4 const ng = N->forward(x, s);
  CHECK(RelDiff(ng, nc) == Approx(0).margin(tol));
  Cx4 const na = N->adjoint(x, s);
  CHECK(RelDiff(na, nc) == Approx(0).margin(tol));

  // Repeated calls: workspace state must not leak between calls
  Cx3 const yg2 = gpu->forward(x, s);
  CHECK(RelDiff(yg2, yc) == Approx(0).margin(tol));
  Cx4 const ng2 = N->forward(x, s);
  CHECK(RelDiff(ng2, nc) == Approx(0).margin(tol));
}

TEST_CASE("KSpaceWeights", "[op][gpu]")
{
  REQUIRE(TOps::KSpaceWeights(nullptr).size() == 0);
  Sz5 const shape{3, 4, 5, 1, 2};
  Re2       w2(4, 5);
  w2.setRandom();
  auto const shared = std::make_shared<TOps::TensorScale<5, 1, 2>>(shape, w2.cast<Cx>());
  Re3 const  r2 = TOps::KSpaceWeights(shared);
  REQUIRE(r2.dimensions() == Sz3{1, 4, 5});
  CHECK(Norm<false>(Re2(r2.chip<0>(0) - w2)) == 0.f);
  Re3 w3(3, 4, 5);
  w3.setRandom();
  auto const per = std::make_shared<TOps::TensorScale<5, 0, 2>>(shape, w3.cast<Cx>());
  Re3 const  r3 = TOps::KSpaceWeights(per);
  REQUIRE(r3.dimensions() == Sz3{3, 4, 5});
  CHECK(Norm<false>(Re3(r3 - w3)) == 0.f);
}
