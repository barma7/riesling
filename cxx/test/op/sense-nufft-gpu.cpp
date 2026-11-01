#include "rl/basis/basis.hpp"
#include "rl/gpu/snufft.hpp"
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

/*
 * Device-pointer interface (for solvers that keep their vectors on the device) against the host interface: forward,
 * adjoint (write and accumulate), normal operator, and the fused PDHG data-term dual steps against the same steps
 * computed on the host from the operator's forward and the weights.
 */
TEST_CASE("SNUFFT-device", "[op][gpu]")
{
  float const osamp = GENERATE(1.3f, 2.f);
  bool const  useBasis = GENERATE(true, false);
  Index const wMode = GENERATE(0, 1, 2);
  Index const nC = 3, nS = 16, nT = 8, nB = useBasis ? 3 : 1;
  INFO("osamp " << osamp << " basis " << useBasis << " weights " << wMode);
  Sz3 const matrix{12, 10, 8};
  Re3       points(3, nS, nT);
  points.setRandom();
  for (Index id = 0; id < 3; id++) {
    points.chip<0>(id) = points.chip<0>(id) * (0.45f * matrix[id]);
  }
  points(0, 3, 2) = std::numeric_limits<float>::quiet_NaN(); // Not in device order
  TrajectoryN<3> const traj(points, matrix);
  Basis                basis(nB, 1, nT);
  basis.B.setRandom();
  GridOpts<3> opts{.osamp = osamp, .kW = 4};
  Cx5         maps(AddBack(traj.matrixForFOV(opts.fov), 1, nC));
  maps.setRandom();
  Re3 W;
  if (wMode > 0) {
    W.resize(wMode == 1 ? 1 : nC, nS, nT);
    W.setRandom();
    W = W.abs() + 0.1f;
  }
  auto const  op = TOps::MakeSENSENUFFTGPU(opts, traj, maps, useBasis ? &basis : nullptr, W);
  auto const  dev = op->device();
  Index const nK = dev->nK();
  REQUIRE(nK == nS * nT - 1);

  Cx4 x(op->ishape), x0(op->ishape);
  Cx3 y(op->oshape), b(op->oshape);
  x.setRandom();
  x0.setRandom();
  y.setRandom();
  b.setRandom();
  y.chip<2>(2).chip<1>(3).setZero(); // The invalid sample is 0 in anything that comes back from device order
  b.chip<2>(2).chip<1>(3).setZero();
  float const tol = 1.e-6f, s = 0.7f;
  size_t const     xb = x.size() * sizeof(Cx), kb = nC * nK * sizeof(Cx);
  gpu::DeviceBuffer dx(xb), dx2(xb), dk(kb), db(kb);
  auto const        X = dx.as<gpu::Cxf>(), X2 = dx2.as<gpu::Cxf>();
  auto const        K = dk.as<gpu::Cxf>(), B = db.as<gpu::Cxf>();

  // Device order round trip
  dev->toDeviceOrder(reinterpret_cast<gpu::Cxf const *>(y.data()), K);
  Cx3 yr(op->oshape);
  dev->fromDeviceOrder(K, reinterpret_cast<gpu::Cxf *>(yr.data()));
  CHECK(RelDiff(yr, y) == 0.f);

  // Forward
  dx.upload(x.data(), xb);
  dev->forwardD(X, K, s);
  Cx3 yd(op->oshape);
  dev->fromDeviceOrder(K, reinterpret_cast<gpu::Cxf *>(yd.data()));
  CHECK(RelDiff(yd, Cx3(op->forward(x, s))) == Approx(0).margin(tol));

  // Adjoint, written and accumulated
  dev->toDeviceOrder(reinterpret_cast<gpu::Cxf const *>(y.data()), K);
  dev->adjointD(K, X2, s, false);
  Cx4 xd(op->ishape);
  dx2.download(xd.data(), xb);
  Cx4 const xa = op->adjoint(y, s);
  CHECK(RelDiff(xd, xa) == Approx(0).margin(tol));
  dx2.upload(x0.data(), xb);
  dev->adjointD(K, X2, s, true);
  dx2.download(xd.data(), xb);
  CHECK(RelDiff(xd, Cx4(x0 + xa)) == Approx(0).margin(tol));

  // Normal operator
  dev->normalD(X, X2, s);
  dx2.download(xd.data(), xb);
  CHECK(RelDiff(xd, Cx4(op->normal()->forward(x, s))) == Approx(0).margin(tol));

  // PDHG data-term dual steps: u = y + σ W (A x - b), then u / (1 + σW), u / (1 + σ), u / max(|u|, 1)
  float const σ = 0.6f;
  Cx3 const   Ax = op->forward(x);
  Re3         Wf(op->oshape);
  Wf.setConstant(1.f);
  if (wMode == 1) {
    Wf = W.broadcast(Sz3{nC, 1, 1});
  } else if (wMode == 2) {
    Wf = W;
  }
  for (auto const kind : {gpu::DataDual::LS, gpu::DataDual::WLS, gpu::DataDual::LAD}) {
    Cx3 const u = y + (Ax - b) * (Wf * σ).cast<Cx>();
    Cx3       e(op->oshape);
    if (kind == gpu::DataDual::LS) {
      e = u / (Wf * σ + 1.f).cast<Cx>();
    } else if (kind == gpu::DataDual::WLS) {
      e = u / Cx(1.f + σ);
    } else {
      e = u / u.abs().cwiseMax(1.f).cast<Cx>();
    }
    e.chip<2>(2).chip<1>(3).setZero();
    dev->toDeviceOrder(reinterpret_cast<gpu::Cxf const *>(y.data()), K);
    dev->toDeviceOrder(reinterpret_cast<gpu::Cxf const *>(b.data()), B);
    dev->dataDualD(X, K, B, σ, kind);
    Cx3 r(op->oshape);
    dev->fromDeviceOrder(K, reinterpret_cast<gpu::Cxf *>(r.data()));
    INFO("kind " << int(kind));
    CHECK(RelDiff(r, e) == Approx(0).margin(tol));
  }
}
