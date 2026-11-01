#include "rl/gpu/pdhg.hpp"
#include "rl/op/grad.hpp"
#include "rl/op/ops.hpp"
#include "rl/prox/llr.hpp"
#include "rl/prox/norms.hpp"
#include "rl/tensors.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

using namespace rl;
using namespace Catch;

/*
 * The GPU PDHG's regularizer steps against the CPU's: the dual step with the conjugate prox, y = prox*(y + σ T x̂), and
 * the primal step X += a Tᴴ y, for the transforms and proxes riesling/regularizers.cpp builds. Non-cubic sizes.
 */
namespace {
Index const nx = 12, ny = 10, nz = 8, nb = 2, nt = 3;
using Vector = Ops::Op::Vector;

auto RelDiff(Vector const &a, Vector const &b) -> float { return (a - b).norm() / b.norm(); }

struct Dev
{
  gpu::DeviceBuffer d;
  explicit Dev(Vector const &v)
    : d(v.size() * sizeof(Cx))
  {
    d.upload(v.data(), v.size() * sizeof(Cx));
  }
  auto p() const -> gpu::Cxf * { return d.as<gpu::Cxf>(); }
  auto get(Index const n) const -> Vector
  {
    Vector v(n);
    d.download(v.data(), n * sizeof(Cx));
    return v;
  }
};

auto Setup(bool const tgv) -> gpu::PDHGSetup
{
  gpu::PDHGSetup s;
  s.nx = nx, s.ny = ny, s.nz = nz, s.nB = nb, s.nT = nt;
  s.tgv = tgv;
  return s;
}
} // namespace

TEST_CASE("PDHG-GPU-terms", "[gpu]")
{
  Sz5 const   sh5{nx, ny, nz, nb, nt};
  Index const n = Product(sh5);
  auto        grad = TOps::Grad<5, 3>::Make(sh5, Sz3{0, 1, 2});
  auto        gradt = TOps::Grad<5, 1>::Make(sh5, Sz1{4});
  auto        gv = TOps::GradVec<6, 3>::Make(grad->oshape, Sz3{0, 1, 2});
  auto        ext_x = std::make_shared<Ops::Extract>(4 * n, 0, n);
  auto        ext_v = std::make_shared<Ops::Extract>(4 * n, n, 3 * n);
  auto        tgv1 = Ops::Sub(Ops::Mul(grad, ext_x), ext_v);
  auto        tgv2 = Ops::Mul(gv, ext_v);

  auto const  T = GENERATE(gpu::Transform::Identity, gpu::Transform::Grad, gpu::Transform::GradT, gpu::Transform::TGV1,
                          gpu::Transform::TGV2);
  bool const  tgv = T == gpu::Transform::TGV1 || T == gpu::Transform::TGV2;
  Index const nX = tgv ? 4 * n : n;
  Ops::Op::Ptr F = T == gpu::Transform::Grad    ? Ops::Op::Ptr(grad)
                   : T == gpu::Transform::GradT ? Ops::Op::Ptr(gradt)
                   : T == gpu::Transform::TGV1  ? tgv1
                   : T == gpu::Transform::TGV2  ? tgv2
                                                : Ops::Op::Ptr(nullptr);
  Index const  nY = F ? F->rows() : n;
  Index const  G = nY / n;
  Sz6 const    sh6{nx, ny, nz, nb, nt, G};
  INFO("transform " << int(T));

  Vector xh = Vector::Random(nX), y = Vector::Random(nY), X0 = Vector::Random(nX);
  float const σ = 0.37f, λ = 0.6f, a = -0.45f;
  Vector      u = y;
  if (F) {
    F->iforward(xh, u, σ);
  } else {
    u += σ * xh;
  }

  auto s = Setup(tgv);
  // Proxes: L1, L1I, L2 over b, g, b + g, t, b + t + g
  struct Case
  {
    gpu::ProxKind  P;
    unsigned       mask;
    Proxs::Prox::Ptr cpu;
  };
  std::vector<Case> cases{{gpu::ProxKind::L1, 0, Proxs::L1::Make(λ, nY)}, {gpu::ProxKind::L1I, 0, Proxs::L1I::Make(λ, nY)}};
  cases.push_back({gpu::ProxKind::L2, 1u << 3, Proxs::L2<6, 1>::Make(λ, sh6, Sz1{3})});
  cases.push_back({gpu::ProxKind::L2, 1u << 4, Proxs::L2<6, 1>::Make(λ, sh6, Sz1{4})});
  cases.push_back({gpu::ProxKind::L2, 1u << 3 | 1u << 4, Proxs::L2<6, 2>::Make(λ, sh6, Sz2{3, 4})});
  if (G > 1) {
    cases.push_back({gpu::ProxKind::L2, 1u << 5, Proxs::L2<6, 1>::Make(λ, sh6, Sz1{5})});
    cases.push_back({gpu::ProxKind::L2, 1u << 3 | 1u << 5, Proxs::L2<6, 2>::Make(λ, sh6, Sz2{3, 5})});
    cases.push_back({gpu::ProxKind::L2, 1u << 3 | 1u << 4 | 1u << 5, Proxs::L2<6, 3>::Make(λ, sh6, Sz3{3, 4, 5})});
  }
  for (auto const &c : cases) {
    INFO("prox " << int(c.P) << " mask " << c.mask);
    Vector e = u;
    c.cpu->conj(σ, e);
    gpu::Term const t{.T = T, .P = c.P, .lambda = λ, .normMask = c.mask};
    Dev             dx(xh), dy(y);
    gpu::TermDual(t, s, dx.p(), dy.p(), σ);
    CHECK(RelDiff(dy.get(nY), e) == Approx(0).margin(1.e-6));
  }

  // Primal step: X += a Tᴴ y
  Vector eX = X0;
  if (F) {
    F->iadjoint(y, eX, a);
  } else {
    eX += a * y;
  }
  gpu::Term const t{.T = T};
  Dev             dX(X0), dy(y);
  gpu::TermAdjoint(t, s, dy.p(), dX.p(), a, a);
  CHECK(RelDiff(dX.get(nX), eX) == Approx(0).margin(1.e-6));
}

TEST_CASE("PDHG-GPU-LLR", "[gpu]")
{
  Sz5 const   sh5{nx, ny, nz, nb, nt};
  Index const n = Product(sh5);
  Index const patch = GENERATE(3, 5), window = GENERATE(1, 3);
  if (window > patch) { return; }
  INFO("patch " << patch << " window " << window);
  float const σ = 0.37f;
  Vector      xh = Vector::Random(n), y = Vector::Random(n);
  // λ (scaled by the CPU prox as there) in the range of the singular values, so some are clipped and some not
  Proxs::LLR<5> cpu(0.08f, patch, window, false, sh5);
  Vector        e = y + σ * xh;
  cpu.conj(σ, e);
  gpu::Term const t{.T = gpu::Transform::Identity, .P = gpu::ProxKind::LLR, .lambda = cpu.λ, .llrPatch = int(patch),
                    .llrWindow = int(window)};
  Dev       dx(xh), dy(y), dz(Vector::Zero(n));
  int const shift[3] = {0, 0, 0};
  gpu::LLRDual(t, Setup(false), dx.p(), dy.p(), dz.p(), σ, shift);
  Vector const r = dz.get(n);
  INFO("scaled λ " << cpu.λ << " |u| " << (y + σ * xh).norm() << " |e| " << e.norm());
  CHECK((e - (y + σ * xh)).norm() > 1.e-2f * e.norm()); // The projection did clip
  CHECK(RelDiff(r, e) == Approx(0).margin(2.e-6));
}
