#include "rl/precon.hpp"
#include "rl/io/hd5.hpp"
#include <filesystem>
#include "rl/basis/basis.hpp"
#include "rl/log/log.hpp"

#include <limits>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

using namespace rl;
using namespace Catch;

TEST_CASE("Preconditioner", "[precon]")
{
  Index const M = GENERATE(16);
  Sz3 const   matrix{M, M, M};
  Re3         points(3, 4, 1);
  points.setZero();
  points(0, 0, 0) = -0.25f * M;
  points(0, 3, 0) = 0.25f * M;
  Trajectory const traj(points, matrix);
  auto const       sc = KSpaceSingle(GridOpts<3>(), traj);
  INFO("Weights\n" << sc);
  CHECK(sc(0, 0) == Approx(1.f).margin(1.e-3f));
  CHECK(sc(1, 0) == Approx(0.5f).margin(1.e-3f));
  CHECK(sc(2, 0) == Approx(0.5f).margin(1.e-3f));
  CHECK(sc(3, 0) == Approx(1.f).margin(1.e-3f));
}

namespace {
using Z = std::complex<double>;

// Independent tiny encoding matrix: normalized DFT rows, expanded over basis
// coefficients, with optional coil- and coefficient-dependent sensitivities.
auto Reference(Trajectory const &traj, Basis::CPtr basis, Cx5 const &maps, float lambda = 0.f) -> Re3
{
  auto const mat = traj.matrix();
  Index const N = Product(mat), nb = basis ? basis->nB() : 1;
  Index const nc = maps.dimension(4), ns = traj.nSamples(), nt = traj.nTraces();
  Index const nr = nc * ns * nt;
  std::vector<std::vector<Z>> rows(nr, std::vector<Z>(N * nb));
  std::vector<double> norm(nr, 0.);
  for (Index t = 0; t < nt; ++t) {
    for (Index s = 0; s < ns; ++s) {
      for (Index c = 0; c < nc; ++c) {
        Index const i = c + nc * (s + ns * t);
        for (Index b = 0; b < nb; ++b) {
          Z const bv = basis ? Z(basis->B(b, s % basis->nSample(), t % basis->nTrace())) * double(basis->scale) : Z(1.);
          for (Index z = 0; z < mat[2]; ++z) {
            for (Index y = 0; y < mat[1]; ++y) {
              for (Index x = 0; x < mat[0]; ++x) {
                Index const v = x + mat[0] * (y + mat[1] * z);
                double const phase = -2. * M_PI * (traj.points()(0, s, t) * double(x - mat[0] / 2) / mat[0] +
                  traj.points()(1, s, t) * double(y - mat[1] / 2) / mat[1] +
                  traj.points()(2, s, t) * double(z - mat[2] / 2) / mat[2]);
                Z const m = maps.data()[v + N * ((maps.dimension(3) == 1 ? 0 : b) + maps.dimension(3) * c)];
                rows[i][v + N * b] = bv * m * std::polar(1. / std::sqrt(double(N)), phase);
                norm[i] += std::norm(rows[i][v + N * b]);
              }
            }
          }
        }
      }
    }
  }
  Re3 result(nc, ns, nt);
  for (Index i = 0; i < nr; ++i) {
    double den = 0.;
    for (Index j = 0; j < nr; ++j) {
      Z dot = 0.;
      for (Index v = 0; v < N * nb; ++v) { dot += rows[i][v] * std::conj(rows[j][v]); }
      den += std::norm(dot);
    }
    result.data()[i] = norm[i] == 0. ? 1.f : float((1. + lambda) / (den / norm[i] + lambda));
  }
  return result;
}

void CheckWeights(Re3 const &actual, Re3 const &expected, double tol = 0.003)
{
  REQUIRE(actual.dimensions() == expected.dimensions());
  for (Index i = 0; i < actual.size(); ++i) {
    INFO("Row " << i << " expected " << expected.data()[i]);
    CHECK(std::isfinite(actual.data()[i]));
    CHECK(actual.data()[i] > 0.f);
    CHECK(actual.data()[i] == Approx(expected.data()[i]).epsilon(tol));
  }
}
}

TEST_CASE("Ong weights match a direct complex encoding matrix", "[precon]")
{
  Index const M = 16, ns = 2, nt = 3;
  Re3 points(3, ns, nt);
  points.setZero();
  bool const colocated = GENERATE(true, false);
  if (!colocated) {
    for (Index t = 0; t < nt; ++t) {
      for (Index s = 0; s < ns; ++s) {
        points(0, s, t) = 0.31f * float(s + 2 * t);
        points(1, s, t) = -0.24f * float(t + s);
        points(2, s, t) = 0.17f * float(t - s);
      }
    }
  }
  Trajectory traj(points, Sz3{M, M, M});
  Basis basis(2, ns, nt);
  bool const complex = GENERATE(false, true);
  // Negative overlaps, unequal norms, a tiny row, and an exactly zero row.
  for (Index t = 0; t < nt; ++t) {
    for (Index s = 0; s < ns; ++s) {
      basis.B(0, s, t) = Cx(t == 1 ? -1.1f : 0.8f + 0.2f * s, complex ? 0.3f * t : 0.f);
      basis.B(1, s, t) = Cx(0.4f - 0.6f * t, complex ? 0.5f * s - 0.2f : 0.f);
    }
  }
  basis.B(0, 1, 1) = Cx(1.e-10f, 0.f);
  basis.B(1, 1, 1) = Cx(0.f, 0.f);
  basis.B(0, 1, 2) = basis.B(1, 1, 2) = 0.f;
  basis.scale = 1.7f;
  Cx5 ones(M, M, M, 1, 1);
  ones.setConstant(1.f);
  float const lambda = GENERATE(0.f, 0.3f);
  Re3 single(1, ns, nt);
  single.chip<0>(0) = KSpaceSingle(GridOpts<3>(), traj, lambda, &basis);
  CheckWeights(single, Reference(traj, &basis, ones, lambda));
  CheckWeights(KSpaceMulti(ones, GridOpts<3>(), traj, lambda, &basis), single);

  Index const nmb = GENERATE(1, 2);
  Cx5 maps(M, M, M, nmb, 2);
  for (Index c = 0; c < 2; ++c) {
    for (Index b = 0; b < nmb; ++b) {
      for (Index z = 0; z < M; ++z) {
        for (Index y = 0; y < M; ++y) {
          for (Index x = 0; x < M; ++x) {
            maps(x, y, z, b, c) = std::polar(0.4f + 0.2f * c + 0.1f * b + 0.008f * x,
                                            0.07f * (c + 1) * y + 0.09f * b * z);
          }
        }
      }
    }
  }
  CheckWeights(KSpaceMulti(maps, GridOpts<3>(), traj, lambda, &basis), Reference(traj, &basis, maps, lambda));
  if (nmb == 1) {
    CheckWeights(KSpaceMulti(maps, GridOpts<3>(), traj, lambda), Reference(traj, nullptr, maps, lambda));
  }
}

TEST_CASE("Generated weights preserve identity frames and ignore the basis on request", "[precon]")
{
  Re3 points(3, 1, 3);
  points.setZero();
  Trajectory traj(points, Sz3{16, 16, 16});
  Basis identity(3, 1, 3);
  identity.B.setZero();
  for (Index t = 0; t < 3; ++t) { identity.B(t, 0, t) = 1.f; }
  auto w = KSpaceSingle(GridOpts<3>(), traj, 0.f, &identity);
  for (Index t = 0; t < 3; ++t) { CHECK(w(0, t) == Approx(1.f).epsilon(0.002)); }
  Basis constant;
  Re2 const noBasis = KSpaceSingle(GridOpts<3>(), traj);
  Re2 const withConstant = KSpaceSingle(GridOpts<3>(), traj, 0.f, &constant);
  CHECK(Norm<true>(Re2(noBasis - withConstant)) == Approx(0.f).margin(1.e-6f));
  PreconOpts opts;
  opts.basis = false;
  auto const op = MakeKSpacePrecon(opts, GridOpts<3>(), traj, &identity, 1, Sz0{});
  Cx3 input(1, 1, 3);
  input.setConstant(1.f);
  auto const output = op->forward(input);
  for (Index t = 0; t < 3; ++t) { CHECK(output(0, 0, t).real() == Approx(noBasis(0, t)).epsilon(0.002)); }
  CHECK_THROWS(KSpaceSingle(GridOpts<3>(), traj, -1.f));
  CHECK_THROWS(KSpaceSingle(GridOpts<3>(), traj, std::numeric_limits<float>::infinity()));
  constant.B(0, 0, 0) = Cx(std::numeric_limits<float>::quiet_NaN(), 0.f);
  CHECK_THROWS(KSpaceSingle(GridOpts<3>(), traj, 0.f, &constant));
}

TEST_CASE("Discarded trajectory rows receive neutral weights", "[precon]")
{
  Re3 points(3, 2, 1);
  points.setZero();
  points(0, 1, 0) = 100.f;
  Trajectory traj(points, Sz3{16, 16, 16});
  auto const w = KSpaceSingle(GridOpts<3>(), traj);
  CHECK(w(0, 0) == Approx(1.f).epsilon(0.002));
  CHECK(w(1, 0) == 1.f);
}

// Grid indices round to nearest, so a kept sample close enough to the +k wall lands one past the
// last index and the gridder ignores it. Trajectory::init() only removes samples beyond the wall.
TEST_CASE("Samples that round off the grid receive neutral weights", "[precon]")
{
  GridOpts<3> gridOpts;
  gridOpts.osamp = GENERATE(1.f, 1.3f);
  float const wall = GENERATE(8.f, 7.9f);
  Re3         points(3, 3, 1);
  points.setZero();
  points(2, 1, 0) = wall;
  points(2, 2, 0) = -wall; // Rounds to index 0, so this one stays on the grid
  Trajectory const traj(points, Sz3{16, 16, 16});

  auto const single = KSpaceSingle(gridOpts, traj);
  CHECK(single(1, 0) == 1.f);
  CHECK(single(2, 0) > 0.f);

  Cx5 maps(16, 16, 16, 1, 2);
  maps.setConstant(Cx(0.5f, 0.f));
  auto const multi = KSpaceMulti(maps, gridOpts, traj, 0.f, nullptr);
  for (Index c = 0; c < 2; ++c) {
    CHECK(multi(c, 1, 0) == 1.f);
    CHECK(multi(c, 2, 0) > 0.f);
  }
}

TEST_CASE("Two dimensional weights and unitary basis invariance", "[precon]")
{
  Index const M = 16;
  Re3 points(2, 3, 1);
  points.setZero();
  points(0, 1, 0) = 0.37f;
  points(1, 2, 0) = -0.61f;
  TrajectoryN<2> traj(points, Sz2{M, M});
  Basis basis(2, 3, 1);
  basis.B(0, 0, 0) = Cx(1.f, 0.2f);
  basis.B(1, 0, 0) = Cx(-0.3f, 0.7f);
  basis.B(0, 1, 0) = Cx(-1.f, -0.4f);
  basis.B(1, 1, 0) = Cx(0.1f, 0.6f);
  basis.B(0, 2, 0) = Cx(0.2f, 0.5f);
  basis.B(1, 2, 0) = Cx(0.8f, -0.2f);
  Re2 const weights = KSpaceSingle(GridOpts<2>(), traj, 0.f, &basis);
  for (Index i = 0; i < 3; ++i) {
    double denom = 0., norm = 0.;
    for (Index b = 0; b < 2; ++b) { norm += std::norm(Z(basis.entry(b, i, 0))); }
    for (Index j = 0; j < 3; ++j) {
      Z temporal = 0., spatial = 0.;
      for (Index b = 0; b < 2; ++b) { temporal += Z(basis.entry(b, i, 0)) * std::conj(Z(basis.entry(b, j, 0))); }
      for (Index x = 0; x < M; ++x) {
        for (Index y = 0; y < M; ++y) {
          double const phase = 2. * M_PI * ((points(0, i, 0) - points(0, j, 0)) * (x - M / 2) +
            (points(1, i, 0) - points(1, j, 0)) * (y - M / 2)) / M;
          spatial += std::polar(1. / double(M * M), phase);
        }
      }
      denom += std::norm(temporal * spatial);
    }
    CHECK(weights(i, 0) == Approx(norm / denom).epsilon(0.003));
  }
  Basis rotated = basis;
  for (Index s = 0; s < 3; ++s) {
    Cx const a = basis.B(0, s, 0), b = basis.B(1, s, 0);
    rotated.B(0, s, 0) = (a + Cx(0.f, 1.f) * b) / std::sqrt(2.f);
    rotated.B(1, s, 0) = (Cx(0.f, 1.f) * a + b) / std::sqrt(2.f);
  }
  Re2 const changed = KSpaceSingle(GridOpts<2>(), traj, 0.f, &rotated);
  for (Index i = 0; i < 3; ++i) { CHECK(changed(i, 0) == Approx(weights(i, 0)).epsilon(0.0001)); }
}

TEST_CASE("Supplied single and multichannel weights retain their values", "[precon]")
{
  Re3 points(3, 3, 1);
  points.setZero();
  Trajectory traj(points, Sz3{16, 16, 16});
  Basis basis(2, 1, 1);
  for (bool multi : {false, true}) {
    std::string const name = multi ? "test-precon-multi.h5" : "test-precon-single.h5";
    {
      HD5::Writer writer(name);
      if (multi) {
        Re3 w(2, 3, 1);
        for (Index i = 0; i < w.size(); ++i) { w.data()[i] = 0.2f + i; }
        writer.writeTensor(HD5::Keys::Weights, w.dimensions(), w.data(), {"channel", "sample", "trace"});
      } else {
        Re2 w(3, 1);
        for (Index i = 0; i < w.size(); ++i) { w.data()[i] = 0.2f + i; }
        writer.writeTensor(HD5::Keys::Weights, w.dimensions(), w.data(), {"sample", "trace"});
      }
    }
    PreconOpts opts{.type = name, .λ = 7.f, .basis = false};
    auto const op = MakeKSpacePrecon(opts, GridOpts<3>(), traj, &basis, 2, Sz0{});
    Cx3 input(2, 3, 1);
    input.setConstant(1.f);
    auto const result = op->forward(input);
    for (Index s = 0; s < 3; ++s) {
      for (Index c = 0; c < 2; ++c) {
        CHECK(result(c, s, 0).real() == Approx(0.2f + (multi ? c + 2 * s : s)).epsilon(1.e-6));
      }
    }
    std::filesystem::remove(name);
  }
}

TEST_CASE("Zero sensitivity channels and inconsistent coefficient maps", "[precon]")
{
  Re3 points(3, 2, 1);
  points.setZero();
  Trajectory traj(points, Sz3{16, 16, 16});
  Cx5 maps(16, 16, 16, 1, 2);
  maps.setZero();
  for (Index v = 0; v < 16 * 16 * 16; ++v) { maps.data()[v] = 1.f; }
  auto const w = KSpaceMulti(maps, GridOpts<3>(), traj, 0.f);
  for (Index s = 0; s < 2; ++s) {
    CHECK(w(0, s, 0) == Approx(0.5f).epsilon(0.002));
    CHECK(w(1, s, 0) == 1.f);
  }
  Cx5 mismatch(16, 16, 16, 2, 1);
  mismatch.setConstant(1.f);
  CHECK_THROWS(KSpaceMulti(mismatch, GridOpts<3>(), traj, 0.f));
}
