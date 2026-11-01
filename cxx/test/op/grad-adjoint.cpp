#include "rl/algo/eig.hpp"
#include "rl/op/grad.hpp"
#include "rl/op/laplacian.hpp"
#include "rl/op/ops.hpp"
#include "rl/sys/threads.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

using namespace rl;
using namespace Catch;

/*
 * Adjoint (dot) tests and norms of the regularizer transforms. ADMM's fixed point, PDHG and the PDHG step sizes all
 * assume that adjoint() is the true adjoint of forward(), and PDHG's step sizes need the norms. Non-cubic sizes, so
 * swapped axes would show.
 */
namespace {
Index const nx = 12, ny = 10, nz = 8, nb = 2, nt = 3;

// |<F x, y> - <x, F' y>| relative to |F x| |y|, for random complex x and y
auto DotErr(Ops::Op::Ptr const &F) -> float
{
  Ops::Op::Vector const x = Ops::Op::Vector::Random(F->cols()), y = Ops::Op::Vector::Random(F->rows());
  Ops::Op::Vector const Fx = F->forward(x), Fy = F->adjoint(y);
  return std::abs(Fx.dot(y) - x.dot(Fy)) / (Fx.norm() * y.norm());
}

auto Norm2(Ops::Op::Ptr const &F) -> float { return PowerMethodForward(F, nullptr, 200).val; }
} // namespace

TEST_CASE("Grad-adjoint", "[op]")
{
  Threads::SetGlobalThreadCount(GENERATE(1, 4));
  Sz5 const sh5{nx, ny, nz, nb, nt};
  auto      grad3 = TOps::Grad<3, 3>::Make(Sz3{nx, ny, nz}, Sz3{0, 1, 2});
  auto      grad5 = TOps::Grad<5, 3>::Make(sh5, Sz3{0, 1, 2}); // Fused
  auto      gradt = TOps::Grad<5, 1>::Make(sh5, Sz1{4});       // TVT
  auto      div3 = TOps::Div<3, 3>::Make(Sz3{nx, ny, nz}, Sz3{0, 1, 2});
  auto      div5 = TOps::Div<5, 3>::Make(sh5, Sz3{0, 1, 2});
  auto      gv4 = TOps::GradVec<4, 3>::Make(Sz4{nx, ny, nz, 3}, Sz3{0, 1, 2});
  auto      gv6 = TOps::GradVec<6, 3>::Make(grad5->oshape, Sz3{0, 1, 2}); // Fused
  auto      gv3 = TOps::GradVec<3, 2>::Make(Sz3{nx, ny, 2}, Sz2{0, 1});
  auto      lap = TOps::Laplacian<3>::Make(Sz3{nx, ny, nz});
  auto      iso = TOps::IsoΔ3D<5>::Make(sh5);

  // The TGV terms as riesling/regularizers.cpp builds them: F1 [x; v] = ∇x - v, F2 [x; v] = ε(v)
  Index const n = grad5->cols(), nv = grad5->rows();
  auto        ext_x = std::make_shared<Ops::Extract>(n + nv, 0, n);
  auto        ext_v = std::make_shared<Ops::Extract>(n + nv, n, nv);
  auto        tgv1 = Ops::Sub(Ops::Mul(grad5, ext_x), ext_v);
  auto        tgv2 = Ops::Mul(gv6, ext_v);

  SECTION("Dot")
  {
    float const tol = 1.e-5f;
    CHECK(DotErr(grad3) == Approx(0).margin(tol));
    CHECK(DotErr(grad5) == Approx(0).margin(tol));
    CHECK(DotErr(gradt) == Approx(0).margin(tol));
    CHECK(DotErr(div3) == Approx(0).margin(tol));
    CHECK(DotErr(div5) == Approx(0).margin(tol));
    CHECK(DotErr(gv4) == Approx(0).margin(tol));
    CHECK(DotErr(gv6) == Approx(0).margin(tol));
    CHECK(DotErr(gv3) == Approx(0).margin(tol));
    CHECK(DotErr(lap) == Approx(0).margin(tol));
    CHECK(DotErr(iso) == Approx(0).margin(tol));
    CHECK(DotErr(tgv1) == Approx(0).margin(tol));
    CHECK(DotErr(tgv2) == Approx(0).margin(tol));
  }

  SECTION("Norms")
  {
    // λmax(F'F): the gradients are scaled to 1 (reached for even sizes); ∇x - v adds the identity on v
    float const g = Norm2(grad5), t = Norm2(gradt), v = Norm2(gv6), f1 = Norm2(tgv1), f2 = Norm2(tgv2);
    INFO("Grad " << g << " GradT " << t << " GradVec " << v << " TGV1 " << f1 << " TGV2 " << f2);
    CHECK(g == Approx(1.f).margin(1.e-3f));
    CHECK(t <= 1.f + 1.e-3f);
    CHECK(v <= 1.f + 1.e-3f);
    CHECK(f1 == Approx(2.f).margin(2.e-3f));
    CHECK(f2 == Approx(v).margin(1.e-3f));
  }
}
