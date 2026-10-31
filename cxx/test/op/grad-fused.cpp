#include "rl/op/grad.hpp"
#include "rl/sys/threads.hpp"
#include "rl/tensors.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

using namespace rl;
using namespace Catch;

/*
 * Grad<5, 3> and GradVec<6, 3> over dims 0, 1, 2 use fused kernels. They must give the same results as the generic
 * code (Grad<3, 3>, GradVec<4, 3>) applied to each volume of the batch dims 3 and 4. Non-cubic sizes, so swapped
 * axes would show.
 */
namespace {
Index const nx = 12, ny = 10, nz = 8, nb = 2, nt = 3;

template <typename T> auto RelDiff(T const &a, T const &b) -> float
{
  T const d = a - b;
  return Norm<false>(d) / Norm<false>(b);
}

} // namespace

TEST_CASE("Grad-fused", "[op]")
{
  Threads::SetGlobalThreadCount(GENERATE(1, 4));
  auto g5 = TOps::Grad<5, 3>::Make(Sz5{nx, ny, nz, nb, nt}, Sz3{0, 1, 2});
  auto g3 = TOps::Grad<3, 3>::Make(Sz3{nx, ny, nz}, Sz3{0, 1, 2});
  Cx5  x(g5->ishape), x0(g5->ishape), xref(g5->ishape);
  Cx6  y(g5->oshape), y0(g5->oshape), yref(g5->oshape);
  x.setRandom();
  y.setRandom();
  x0.setRandom();
  y0.setRandom();

  // forward / adjoint
  for (Index it = 0; it < nt; it++) {
    for (Index ib = 0; ib < nb; ib++) {
      Cx3 const xv = x.chip<4>(it).chip<3>(ib);
      Cx4 const yv = y.chip<4>(it).chip<3>(ib);
      yref.chip<4>(it).chip<3>(ib) = g3->forward(xv);
      xref.chip<4>(it).chip<3>(ib) = g3->adjoint(yv);
    }
  }
  CHECK(RelDiff(Cx6(g5->forward(x)), yref) == Approx(0).margin(1.e-6));
  CHECK(RelDiff(Cx5(g5->adjoint(y)), xref) == Approx(0).margin(1.e-6));

  // In-place versions add to the output
  Cx6 yi = y0;
  g5->iforward(TOps::Grad<5, 3>::InCMap(x.data(), x.dimensions()), TOps::Grad<5, 3>::OutMap(yi.data(), yi.dimensions()));
  CHECK(RelDiff(yi, Cx6(y0 + yref)) == Approx(0).margin(1.e-6));
  Cx5 xi = x0;
  g5->iadjoint(TOps::Grad<5, 3>::OutCMap(y.data(), y.dimensions()), TOps::Grad<5, 3>::InMap(xi.data(), xi.dimensions()));
  CHECK(RelDiff(xi, Cx5(x0 + xref)) == Approx(0).margin(1.e-6));
}

TEST_CASE("GradVec-fused", "[op]")
{
  Threads::SetGlobalThreadCount(GENERATE(1, 4));
  auto g6 = TOps::GradVec<6, 3>::Make(Sz6{nx, ny, nz, nb, nt, 3}, Sz3{0, 1, 2});
  auto g4 = TOps::GradVec<4, 3>::Make(Sz4{nx, ny, nz, 3}, Sz3{0, 1, 2});
  Cx6  x(g6->ishape), x0(g6->ishape), xref(g6->ishape);
  Cx6  y(g6->oshape), y0(g6->oshape), yref(g6->oshape);
  x.setRandom();
  y.setRandom();
  x0.setRandom();
  y0.setRandom();

  for (Index it = 0; it < nt; it++) {
    for (Index ib = 0; ib < nb; ib++) {
      Cx4 const xv = x.chip<4>(it).chip<3>(ib);
      Cx4 const yv = y.chip<4>(it).chip<3>(ib);
      yref.chip<4>(it).chip<3>(ib) = g4->forward(xv);
      xref.chip<4>(it).chip<3>(ib) = g4->adjoint(yv);
    }
  }
  CHECK(RelDiff(Cx6(g6->forward(x)), yref) == Approx(0).margin(1.e-6));
  CHECK(RelDiff(Cx6(g6->adjoint(y)), xref) == Approx(0).margin(1.e-6));

  Cx6 yi = y0;
  g6->iforward(TOps::GradVec<6, 3>::InCMap(x.data(), x.dimensions()),
               TOps::GradVec<6, 3>::OutMap(yi.data(), yi.dimensions()));
  CHECK(RelDiff(yi, Cx6(y0 + yref)) == Approx(0).margin(1.e-6));
  Cx6 xi = x0;
  g6->iadjoint(TOps::GradVec<6, 3>::OutCMap(y.data(), y.dimensions()),
               TOps::GradVec<6, 3>::InMap(xi.data(), xi.dimensions()));
  CHECK(RelDiff(xi, Cx6(x0 + xref)) == Approx(0).margin(1.e-6));
}
