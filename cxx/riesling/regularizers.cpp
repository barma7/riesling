#include "regularizers.hpp"

#include "rl/algo/eig.hpp"
#include "rl/log/log.hpp"
#include "rl/op/fft.hpp"
#include "rl/op/grad.hpp"
#include "rl/op/laplacian.hpp"
#include "rl/prox/l1-wavelets.hpp"
#include "rl/prox/llr.hpp"
#include "rl/prox/norms.hpp"
#include "rl/prox/stack.hpp"

namespace rl {

RegOpts::RegOpts(args::Subparser &parser)
  : iso(parser, "ISO", "Isotropic/joint dims (b/g/bg)", {"iso"})

  , l1(parser, "L1", "Simple L1 regularization", {"l1"})
  , l1i(parser, "L1", "L1 regularization of the imaginary part", {"l1i"})
  , tv(parser, "TV", "Total Variation", {"tv"})
  , tv2(parser, "TV", "TV + Laplacian", {"tv2"})
  , tgv(parser, "TGV", "Total Generalized Variation", {"tgv"})
  , tvt(parser, "TVT", "Total Variation along time dimension", {"tvt"})

  , llr(parser, "L", "LLR regularization", {"llr"})
  , llrPatch(parser, "S", "Patch size for LLR (default 5)", {"llr-patch"}, 5)
  , llrWin(parser, "S", "Window size for LLR (default 3)", {"llr-win"}, 3)
  , llrShift(parser, "S", "Enable random LLR shifting", {"llr-shift"})

  , wavelets(parser, "L", "L1 Wavelet denoising", {"wavelets"})
  , waveDims(parser, "W", "Wavelet transform dimensions (0,1,2)", {"wavelet-dims"}, std::vector<Index>{0, 1, 2})
  , waveWidth(parser, "W", "Wavelet width (4/6/8)", {"wavelet-width"}, 6)
{
}

auto Regularizers(RegOpts &opts, TOps::TOp<5, 5>::Ptr const &recon) -> Regularizers_t
{
  Ops::Op::Ptr             A = recon;
  auto const               shape = recon->ishape;
  Index const              nx = Product(shape);
  std::vector<Regularizer> regs;
  Ops::Op::Ptr             ext; // With TGV the unknowns are [x; v]: reads x out of them

  if (opts.tgv) {
    auto grad_x = TOps::Grad<5, 3>::Make(shape, Sz3{0, 1, 2});
    auto ext_x = std::make_shared<Ops::Extract>(A->cols() + grad_x->rows(), 0, A->cols());
    auto ext_v = std::make_shared<Ops::Extract>(A->cols() + grad_x->rows(), A->cols(), grad_x->rows());
    auto op1 = Ops::Sub(Ops::Mul(grad_x, ext_x), ext_v);

    auto grad_v = TOps::GradVec<6, 3>::Make(grad_x->oshape, Sz3{0, 1, 2});
    auto op2 = Ops::Mul(grad_v, ext_v);

    Proxs::Prox::Ptr prox_x, prox_v;
    if (opts.iso) {
      if (opts.iso.Get() == "b") {
        prox_x = Proxs::L2<6, 1>::Make(opts.tgv.Get(), grad_x->oshape, Sz1{3});
        prox_v = Proxs::L2<6, 1>::Make(opts.tgv.Get(), grad_v->oshape, Sz1{3});
      } else if (opts.iso.Get() == "g") {
        prox_x = Proxs::L2<6, 1>::Make(opts.tgv.Get(), grad_x->oshape, Sz1{5});
        prox_v = Proxs::L2<6, 1>::Make(opts.tgv.Get(), grad_v->oshape, Sz1{5});
      } else if (opts.iso.Get() == "bg") {
        prox_x = Proxs::L2<6, 2>::Make(opts.tgv.Get(), grad_x->oshape, Sz2{3, 5});
        prox_v = Proxs::L2<6, 2>::Make(opts.tgv.Get(), grad_v->oshape, Sz2{3, 5});
      } else {
        throw Log::Failure("Regs", "Isotropic dims must be b, g, or bg");
      }
    } else {
      prox_x = Proxs::L1::Make(opts.tgv.Get(), op1->rows());
      prox_v = Proxs::L1::Make(opts.tgv.Get(), op2->rows());
    }
    // λmax(T'T): 1 + λmax(∇'∇) = 2 for ∇x - v, 0.25 for the symmetrized gradient (test/op/grad-adjoint.cpp)
    regs.push_back({op1, prox_x, grad_x->oshape, 2.f});
    regs.push_back({op2, prox_v, grad_v->oshape, 0.25f});
    A = std::make_shared<Ops::Multiply>(A, ext_x);
    ext = ext_x;
  }

  // The other terms act on x; with TGV their transforms read x out of [x; v] (TGV can be combined with them)
  auto const onX = [&ext](Ops::Op::Ptr const &T) -> Ops::Op::Ptr { return ext ? (T ? Ops::Mul(T, ext) : ext) : T; };

  if (opts.tv) {
    auto             grad = TOps::Grad<5, 3>::Make(shape, Sz3{0, 1, 2});
    Proxs::Prox::Ptr prox;
    if (opts.iso) {
      if (opts.iso.Get() == "b") {
        prox = Proxs::L2<6, 1>::Make(opts.tv.Get(), grad->oshape, Sz1{3});
      } else if (opts.iso.Get() == "t") {
        prox = Proxs::L2<6, 1>::Make(opts.tv.Get(), grad->oshape, Sz1{4});
      } else if (opts.iso.Get() == "g") {
        prox = Proxs::L2<6, 1>::Make(opts.tv.Get(), grad->oshape, Sz1{5});
      } else if (opts.iso.Get() == "bt") {
        prox = Proxs::L2<6, 2>::Make(opts.tv.Get(), grad->oshape, Sz2{3, 4});
      } else if (opts.iso.Get() == "bg") {
        prox = Proxs::L2<6, 2>::Make(opts.tv.Get(), grad->oshape, Sz2{3, 5});
      } else if (opts.iso.Get() == "gt") {
        prox = Proxs::L2<6, 2>::Make(opts.tv.Get(), grad->oshape, Sz2{4, 5});
      } else if (opts.iso.Get() == "bgt") {
        prox = Proxs::L2<6, 3>::Make(opts.tv.Get(), grad->oshape, Sz3{3, 4, 5});
      } else {
        throw Log::Failure("Regs", "Valid dims are bgt");
      }
    } else {
      prox = Proxs::L1::Make(opts.tv.Get(), grad->rows());
    }
    regs.push_back({onX(grad), prox, grad->oshape});
  }

  if (opts.tv2) {
    auto grad = TOps::Grad<5, 3>::Make(shape, Sz3{0, 1, 2});
    auto lap = TOps::IsoΔ3D<5>::Make(shape);
    auto both = Ops::VStack::Make({grad, lap});

    Proxs::Prox::Ptr pg, pl;
    float const      σ = 0.77; // Bock et al 2008
    if (opts.iso) {
      if (opts.iso.Get() == "b") {
        pg = Proxs::L2<6, 1>::Make(opts.tv2.Get() * σ, grad->oshape, Sz1{3});
        pl = Proxs::L2<5, 1>::Make(opts.tv2.Get() * (1 - σ), lap->oshape, Sz1{3});
      } else if (opts.iso.Get() == "t") {
        pg = Proxs::L2<6, 1>::Make(opts.tv2.Get() * σ, grad->oshape, Sz1{4});
        pl = Proxs::L2<5, 1>::Make(opts.tv2.Get() * (1 - σ), lap->oshape, Sz1{4});
      } else if (opts.iso.Get() == "bt") {
        pg = Proxs::L2<6, 2>::Make(opts.tv2.Get() * σ, grad->oshape, Sz2{3, 4});
        pl = Proxs::L2<5, 2>::Make(opts.tv2.Get() * (1 - σ), lap->oshape, Sz2{3, 4});
      } else {
        throw Log::Failure("Regs", "Isotropic dims must bebt");
      }
    } else {
      pg = Proxs::L1::Make(opts.tv2.Get() * σ, grad->rows());
      pl = Proxs::L1::Make(opts.tv2.Get() * (1 - σ), lap->rows());
    }
    auto prox = Proxs::Stack::Make({pg, pl});
    regs.push_back({onX(both), prox, grad->oshape, 2.01f}); // λmax(∇'∇ + Δ'Δ) <= 1 + 1.01 (test/op/grad-adjoint.cpp)
  }

  if (opts.tvt) {
    auto grad = TOps::Grad<5, 1>::Make(shape, Sz1{4});
    auto prox = Proxs::L1::Make(opts.tvt.Get(), grad->rows());
    regs.push_back({onX(grad), prox, grad->oshape});
  }

  if (opts.wavelets) {
    regs.push_back({onX(nullptr),
                    std::make_shared<Proxs::L1Wavelets>(opts.wavelets.Get(), shape, opts.waveWidth.Get(), opts.waveDims.Get()),
                    shape});
  }

  if (opts.llr) {
    regs.push_back(
      {onX(nullptr), std::make_shared<Proxs::LLR<5>>(opts.llr.Get(), opts.llrPatch.Get(), opts.llrWin.Get(), opts.llrShift, shape),
       shape});
  }

  if (opts.l1) {
    Proxs::Prox::Ptr prox;
    if (opts.iso) { // The images are (i, j, k, b, t): b is dimension 3, t dimension 4
      if (opts.iso.Get() == "b") {
        prox = std::make_shared<Proxs::L2<5, 1>>(opts.l1.Get(), shape, Sz1{3});
      } else if (opts.iso.Get() == "t") {
        prox = std::make_shared<Proxs::L2<5, 1>>(opts.l1.Get(), shape, Sz1{4});
      } else if (opts.iso.Get() == "bt") {
        prox = std::make_shared<Proxs::L2<5, 2>>(opts.l1.Get(), shape, Sz2{3, 4});
      } else {
        throw Log::Failure("Regs", "Valid dims are bt");
      }
    } else {
      prox = Proxs::L1::Make(opts.l1.Get(), nx);
    }
    regs.push_back({onX(nullptr), prox, shape});
  }

  if (opts.l1i) {
    Proxs::Prox::Ptr prox = Proxs::L1I::Make(opts.l1i.Get(), nx);
    regs.push_back({onX(nullptr), prox, shape});
  }

  if (regs.size() == 0) { throw Log::Failure("Reg", "Must specify at least one regularizer"); }

  return {regs, A, ext};
}

namespace {
// Normed dimensions of the 6D shape (i, j, k, b, t, g) from --iso, as the L2 proxes Regularizers() builds
auto IsoMask(std::string const &iso, std::string const &allowed) -> unsigned
{
  unsigned mask = 0;
  for (auto const c : iso) {
    if (allowed.find(c) == std::string::npos) { throw Log::Failure("Regs", "Isotropic dims must be {} here", allowed); }
    mask |= c == 'b' ? 1u << 3 : (c == 't' ? 1u << 4 : 1u << 5);
  }
  return mask;
}
} // namespace

auto GPURegularizers(RegOpts &opts, Sz5 const shape) -> GPURegularizers_t
{
  GPURegularizers_t r;
  using gpu::ProxKind;
  using gpu::Transform;
  if (opts.tv2) { throw Log::Failure("Regs", "--tv2 is not implemented in the GPU PDHG"); }
  if (opts.wavelets) { throw Log::Failure("Regs", "Wavelets are not implemented in the GPU PDHG"); }
  if (opts.tgv) {
    r.tgv = true;
    unsigned const mask = opts.iso ? IsoMask(opts.iso.Get(), "bg") : 0u;
    if (opts.iso && opts.iso.Get() != "b" && opts.iso.Get() != "g" && opts.iso.Get() != "bg") {
      throw Log::Failure("Regs", "Isotropic dims must be b, g, or bg");
    }
    ProxKind const P = opts.iso ? ProxKind::L2 : ProxKind::L1;
    // Block norms: ∇ (1) and -I (1) for ∇x - v, the symmetrized gradient (√0.25) for ε(v)
    r.terms.push_back(
      {.T = Transform::TGV1, .P = P, .lambda = opts.tgv.Get(), .normMask = mask, .norm2 = 2.f, .normX = 1.f, .normV = 1.f});
    r.terms.push_back(
      {.T = Transform::TGV2, .P = P, .lambda = opts.tgv.Get(), .normMask = mask, .norm2 = 0.25f, .normX = 0.f, .normV = 0.5f});
  }
  if (opts.tv) {
    unsigned const mask = opts.iso ? IsoMask(opts.iso.Get(), "bgt") : 0u;
    r.terms.push_back({.T = Transform::Grad, .P = opts.iso ? ProxKind::L2 : ProxKind::L1, .lambda = opts.tv.Get(),
                       .normMask = mask, .norm2 = 1.f});
  }
  if (opts.tvt) { r.terms.push_back({.T = Transform::GradT, .P = ProxKind::L1, .lambda = opts.tvt.Get(), .norm2 = 1.f}); }
  if (opts.llr) {
    // The λ scaling of the CPU prox (Proxs::LLR)
    Proxs::LLR<5> const cpu(opts.llr.Get(), opts.llrPatch.Get(), opts.llrWin.Get(), opts.llrShift, shape);
    r.terms.push_back({.T = Transform::Identity,
                       .P = ProxKind::LLR,
                       .lambda = cpu.λ,
                       .llrPatch = int(opts.llrPatch.Get()),
                       .llrWindow = int(opts.llrWin.Get()),
                       .llrShift = opts.llrShift,
                       .norm2 = 1.f});
  }
  if (opts.l1) {
    if (opts.iso && opts.iso.Get() != "b" && opts.iso.Get() != "t" && opts.iso.Get() != "bt") {
      throw Log::Failure("Regs", "Valid dims are bt");
    }
    unsigned const mask = opts.iso ? IsoMask(opts.iso.Get(), "bt") : 0u;
    r.terms.push_back({.T = Transform::Identity, .P = opts.iso ? ProxKind::L2 : ProxKind::L1, .lambda = opts.l1.Get(),
                       .normMask = mask, .norm2 = 1.f});
  }
  if (opts.l1i) { r.terms.push_back({.T = Transform::Identity, .P = ProxKind::L1I, .lambda = opts.l1i.Get(), .norm2 = 1.f}); }
  if (r.terms.empty()) { throw Log::Failure("Reg", "Must specify at least one regularizer"); }
  return r;
}

} // namespace rl
