#include "rl/op/grid.hpp"
#include "rl/kernel/tophat.hpp"
#include "rl/log/log.hpp"
#include "rl/tensors.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

using namespace rl;
using namespace Catch;

TEST_CASE("Grid1", "[op]")
{
  // Log::SetDisplayLevel(Log::Display::High);
  Threads::SetGlobalThreadCount(1);
  Index const M = GENERATE(16, 32);
  auto const  matrix = Sz1{M};
  Re3         points(1, 3, 1);
  points.setZero();
  points(0, 0, 0) = -0.4f * M;
  points(0, 2, 0) = 0.4f * M;
  TrajectoryN<1> const traj(points, matrix);

  float const osamp = GENERATE(1.3f, 2.f);
  auto        grid = TOps::Grid<1, ExpSemi<6>>::Make(GridOpts<1>{.osamp = osamp}, traj, 1);
  Cx3         noncart(grid->oshape);
  Cx3         cart(grid->ishape);
  noncart.setConstant(1.f);
  cart = grid->adjoint(noncart);
  INFO("M " << M << " OS " << osamp);
  INFO("noncart\n" << noncart);
  INFO("cart\n" << cart);
  auto const cs = std::sqrt(cart.size());
  CHECK((Norm<false>(cart) - Norm<false>(noncart)) / cs == Approx(0.f).margin(3e-4f));
  noncart = grid->forward(cart);
  INFO("noncart\n" << noncart);
  CHECK((Norm<false>(cart) - Norm<false>(noncart)) / cs == Approx(0.f).margin(3e-4f));
}

TEST_CASE("Grid2", "[op]")
{
  // Log::SetDisplayLevel(Log::Display::High);
  Threads::SetGlobalThreadCount(1);
  Index const M = GENERATE(16, 32);
  auto const  matrix = Sz2{M, M};
  Re3         points(2, 3, 1);
  points.setZero();
  points(0, 0, 0) = -0.4f * M;
  points(1, 0, 0) = -0.4f * M;
  points(0, 2, 0) = 0.4f * M;
  points(1, 2, 0) = 0.4f * M;
  TrajectoryN<2> const traj(points, matrix);

  float const osamp = GENERATE(1.3f, 2.f);
  auto        grid = TOps::Grid<2, ExpSemi<6>>::Make(GridOpts<2>{.osamp = osamp}, traj, 1);
  Cx3         noncart(grid->oshape);
  Cx4         cart(grid->ishape);
  noncart.setConstant(1.f);
  cart = grid->adjoint(noncart);
  INFO("M " << M << " OS " << osamp);
  INFO("noncart\n" << noncart);
  INFO("cart\n" << cart);
  auto const cs = std::sqrt(cart.size());
  CHECK((Norm<false>(cart) - Norm<false>(noncart)) / cs == Approx(0.f).margin(2e-4f));
  noncart = grid->forward(cart);
  INFO("noncart\n" << noncart);
  CHECK((Norm<false>(cart) - Norm<false>(noncart)) / cs == Approx(0.f).margin(2e-4f));
}

TEST_CASE("Grid3", "[op]")
{
  // Log::SetDisplayLevel(Log::Display::High);
  Threads::SetGlobalThreadCount(1);
  Index const M = GENERATE(16, 32);
  auto const  matrix = Sz3{M, M, M};
  Re3         points(3, 3, 1);
  points.setZero();
  points(0, 0, 0) = -0.4f * M;
  points(1, 0, 0) = -0.4f * M;
  points(2, 0, 0) = -0.4f * M;
  points(0, 2, 0) = 0.4f * M;
  points(1, 2, 0) = 0.4f * M;
  points(2, 2, 0) = 0.4f * M;
  TrajectoryN<3> const traj(points, matrix);

  float const osamp = GENERATE(1.3f, 2.f);
  auto        grid = TOps::Grid<3>::Make(GridOpts<3>{.osamp = osamp}, traj, 1);
  Cx3         noncart(grid->oshape);
  Cx5         cart(grid->ishape);
  noncart.setConstant(1.f);
  cart = grid->adjoint(noncart);
  INFO("M " << M << " OS " << osamp);
  // INFO("noncart\n" << noncart);
  // INFO("cart\n" << cart);
  auto const cs = std::sqrt(cart.size());
  CHECK((Norm<false>(cart) - Norm<false>(noncart)) / cs == Approx(0.f).margin(2e-4f));
  noncart = grid->forward(cart);
  // INFO("noncart\n" << noncart);
  CHECK((Norm<false>(cart) - Norm<false>(noncart)) / cs == Approx(0.f).margin(2e-4f));
}

TEST_CASE("Grid3-Basis-Channels", "[op]")
{
  Threads::SetGlobalThreadCount(GENERATE(1, 4));
  Index const M = 24, nC = 3, nS = 16, nT = 8, nB = 3;
  Re3         points(3, nS, nT);
  points.setRandom();
  points = points * (0.45f * M); // Some points near the edge, so the wrapped copies are used too
  TrajectoryN<3> const traj(points, Sz3{M, M, M});

  Basis basis(nB, 1, nT);
  basis.B.setRandom();
  basis.B(1, 0, 0) = 0.f; // Traces 0 and 1 have zero basis entries, which are skipped
  basis.B(0, 0, 1) = 0.f;
  basis.B(2, 0, 1) = 0.f;

  auto grid = TOps::Grid<3>::Make(GridOpts<3>{.osamp = 1.3f}, traj, nC, &basis);
  Cx5  x(grid->ishape), yx(grid->ishape);
  Cx3  y(grid->oshape), xy(grid->oshape);
  x.setRandom();
  y.setRandom();
  xy = grid->forward(x);
  yx = grid->adjoint(y);
  auto const xx = Dot<false>(x, yx);
  auto const yy = Dot<false>(xy, y);
  CHECK(std::abs((yy - xx) / (yy + xx)) == Approx(0).margin(1.e-5));

  // Same as gridding each channel and basis vector on its own, without a basis
  auto      grid1 = TOps::Grid<3>::Make(GridOpts<3>{.osamp = 1.3f}, traj, 1);
  Sz5 const sz1 = AddBack(FirstN<3>(grid->ishape), 1, 1);
  Cx3       ref(grid->oshape);
  ref.setZero();
  for (Index ic = 0; ic < nC; ic++) {
    for (Index ib = 0; ib < nB; ib++) {
      Cx5 const x1 = x.slice(Sz5{0, 0, 0, ib, ic}, sz1);
      Cx3 const y1 = grid1->forward(x1);
      for (Index it = 0; it < nT; it++) {
        for (Index is = 0; is < nS; is++) {
          ref(ic, is, it) += y1(0, is, it) * basis.entry(ib, is, it);
        }
      }
    }
  }
  Cx3 const diff = xy - ref;
  CHECK(Norm<false>(diff) / Norm<false>(ref) == Approx(0).margin(1.e-5));
}

TEST_CASE("GridB", "[op]")
{
  Threads::SetGlobalThreadCount(1);
  Index const M = 6;
  auto const  matrix = Sz1{M};
  Re3         points(1, 6, 1);
  points.setZero();
  points(0, 0, 0) = -3.f;
  points(0, 1, 0) = -2.f;
  points(0, 2, 0) = -1.f;
  points(0, 3, 0) = 0;
  points(0, 4, 0) = 1.f;
  points(0, 5, 0) = 2.f;
  TrajectoryN<1> const traj(points, matrix);

  Basis basis(2, 6, 1);
  basis.B.setZero();
  basis.B(0, 0, 0) = 1.f;
  basis.B(0, 1, 0) = 1.f;
  basis.B(0, 2, 0) = 1.f;
  basis.B(1, 3, 0) = 1.f;
  basis.B(1, 4, 0) = 1.f;
  basis.B(1, 5, 0) = 1.f;
  using GType = TOps::Grid<1, rl::TopHat<1>>;
  auto grid = GType::Make(GridOpts<1>{.osamp = 1.f}, traj, 1, &basis);
  Cx3  noncart(grid->oshape);
  noncart.setConstant(1.f);
  Cx3 cart = grid->adjoint(noncart);
  INFO("GRID\n" << cart);
  CHECK(cart(0, 0, 0).real() == Approx(1.f).margin(1e-2f));
  CHECK(cart(1, 0, 0).real() == Approx(1.f).margin(1e-2f));
  CHECK(cart(2, 0, 0).real() == Approx(1.f).margin(1e-2f));
  CHECK(cart(3, 0, 0).real() == Approx(0.f).margin(1e-2f));
  CHECK(cart(4, 0, 0).real() == Approx(0.f).margin(1e-2f));
  CHECK(cart(5, 0, 0).real() == Approx(0.f).margin(1e-2f));
  CHECK(cart(0, 1, 0).real() == Approx(0.f).margin(1e-2f));
  CHECK(cart(1, 1, 0).real() == Approx(0.f).margin(1e-2f));
  CHECK(cart(2, 1, 0).real() == Approx(0.f).margin(1e-2f));
  CHECK(cart(3, 1, 0).real() == Approx(1.f).margin(1e-2f));
  CHECK(cart(4, 1, 0).real() == Approx(1.f).margin(1e-2f));
  CHECK(cart(5, 1, 0).real() == Approx(1.f).margin(1e-2f));
  Cx3 nc2 = grid->forward(cart);
  INFO("NC\n" << nc2);
  CHECK(Norm<false>(nc2 - noncart) == Approx(0.f).margin(1e-2f));
}
