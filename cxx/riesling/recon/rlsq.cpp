#include "inputs.hpp"
#include "regularizers.hpp"

#include "rl/algo/admm.hpp"
#include "rl/io/hd5.hpp"
#include "rl/log/debug.hpp"
#include "rl/log/log.hpp"
#include "rl/op/pad.hpp"
#include "rl/op/recon.hpp"
#include "rl/precon.hpp"
#include "rl/scaling.hpp"
#include "rl/sense/sense.hpp"

using namespace rl;

namespace {
// Whether the chosen regularizers couple time points (dim 4 of the images), so that they cannot be solved one by one
auto TimeCoupled(RegOpts &r) -> bool
{
  if (r.iso && r.iso.Get().find('t') != std::string::npos) { return true; }
  if (r.tvt || r.llr) { return true; }
  if (r.wavelets) {
    auto const d = r.waveDims.Get();
    if (std::find(d.cbegin(), d.cend(), 4) != d.cend()) { return true; }
  }
  return false;
}
} // namespace

template <int ND> void run_recon_rlsq(args::Subparser &parser)
{
  CoreArgs<ND>                 coreArgs(parser);
  GridArgs<ND>                 gridArgs(parser);
  PreconArgs                   preArgs(parser);
  ReconArgs                    reconArgs(parser);
  SENSEArgs<ND>                senseArgs(parser);
  ADMMArgs                     admmArgs(parser);
  PDHGArgs                     pdhgArgs(parser);
  RegOpts                      regOpts(parser);
  f0Args                       f0Args(parser);
  args::ValueFlag<std::string> scaling(parser, "S", "Data scaling (otsu/bart/number)", {"scale"}, "otsu");
  args::ValueFlag<Index>       debugIters(parser, "I", "Write debug images ever N outer iterations (16)", {"debug-iters"}, 16);
  args::Flag                   debugZ(parser, "Z", "Write regularizer debug images", {"debug-z"});
  args::Flag                   pdhg(parser, "P", "Use PDHG instead of ADMM", {"pdhg", 'p'});
  args::ValueFlag<Index>       restart(parser, "R", "Restart PDHG", {"pdhg-restart", 'r'}, 8);
  args::Flag                   adapt(parser, "A", "Adaptive PDHG", {"adaptive"});
  ArrayFlag<float, ND>         cropFov(parser, "FOV", "Crop FoV in mm (x,y,z)", {"crop-fov"});
  args::Flag splitT(parser, "S", "ADMM: solve each time point on its own (less memory; regularizers must not couple time)",
                    {"split-time"});

  ParseCommand(parser, coreArgs.iname, coreArgs.oname);
  auto const      cmd = parser.GetCommand().Name();
  HD5::Reader     reader(coreArgs.iname.Get());
  Info const      info = reader.readStruct<Info>(HD5::Keys::Info);
  TrajectoryN<ND> traj(reader, info.voxel_size.head<ND>(), coreArgs.matrix.Get());
  auto            noncart = reader.readTensor<Cx5>(coreArgs.dset.Get());
  traj.checkDims(FirstN<3>(noncart.dimensions()));

  auto const basis = LoadBasis(coreArgs.basisFile.Get());

  /* --split-time: the time points (e.g. echoes) are independent problems when the regularizers do not couple them, so
   * solve them one after the other. Only one time point's ADMM state is in memory at a time. The objective is the same:
   * the same encoding operator per time point, the same k-space weights, and the data scale computed from all time
   * points together, as without splitting. Each time point gets its own ADMM penalty and stopping test.
   */
  if (splitT) {
    if (pdhg) { throw Log::Failure(cmd, "--split-time is only implemented for ADMM"); }
    if (f0Args.Nτ) { throw Log::Failure(cmd, "--split-time does not support off-resonance correction"); }
    if (TimeCoupled(regOpts)) {
      throw Log::Failure(cmd, "--split-time needs regularizers that do not couple time (no t in --iso, no --tvt or --llr, "
                              "no wavelets over dim 4)");
    }
    Index const nT = noncart.dimension(4);
    Sz5 const   tshape = AddBack(FirstN<4>(noncart.dimensions()), 1);
    float       scale;
    if (scaling.Get() == "otsu" || scaling.Get() == "bart") {
      auto const Rall = Recon(reconArgs.Get(), preArgs.Get(), gridArgs.Get(), senseArgs.Get(), traj, basis.get(), noncart);
      scale = ScaleData(scaling.Get(), Rall.A, Rall.M, CollapseToVector(noncart));
    } else {
      scale = ScaleData(scaling.Get(), nullptr, nullptr, CollapseToVector(noncart));
    }
    // One operator for a single time point. SENSE self-calibration uses the --sense-tp time point, as without splitting.
    auto sopts = senseArgs.Get();
    if (sopts.tp >= nT) { throw Log::Failure(cmd, "SENSE time point {} but data has {}", sopts.tp, nT); }
    Cx5 const ncSENSE = noncart.slice(Sz5{0, 0, 0, 0, sopts.tp}, tshape);
    sopts.tp = 0;
    auto const   R = Recon(reconArgs.Get(), preArgs.Get(), gridArgs.Get(), sopts, traj, basis.get(), ncSENSE);
    auto const   shape = R.A->ishape;
    TOps::Pad<5> oc(Concatenate(traj.matrixForFOV(cropFov.Get()), LastN<5 - ND>(shape)), shape);
    Cx5          out(AddBack(FirstN<4>(oc.ishape), nT));
    for (Index it = 0; it < nT; it++) {
      Log::Print(cmd, "Time point {} of {}", it + 1, nT);
      Cx5 nc = noncart.slice(Sz5{0, 0, 0, 0, it}, tshape);
      if (scale != 1.f) { nc.device(Threads::TensorDevice()) = nc * Cx(scale); }
      auto [reg, A, ext_x] = Regularizers(regOpts, R.A);
      ADMM::DebugX debug_x = [shape, it, di = debugIters.Get()](Index const ii, ADMM::Vector const &xx) {
        if (Log::IsDebugging() && (ii % di == 0)) {
          Log::Tensor(fmt::format("admm-x-t{}-{:02d}", it, ii), shape, xx.data(), HD5::Dims::Images);
        }
      };
      ADMM opt{A, R.M, reg, admmArgs.Get(), debug_x, nullptr};
      auto x = opt.run(CollapseToConstVector(nc));
      if (ext_x) { x = ext_x->forward(x); }
      if (scale != 1.f) { x.device(Threads::CoreDevice()) = x / Cx(scale); }
      auto const xm = AsConstTensorMap(x, R.A->ishape);
      out.chip<4>(it) = oc.adjoint(xm).template chip<4>(0);
      if (coreArgs.residual) { // Residual of the unscaled data, stored in place of it
        Cx5 r = noncart.slice(Sz5{0, 0, 0, 0, it}, tshape);
        r -= R.A->forward(xm);
        noncart.template chip<4>(it) = r.chip<4>(0);
      }
    }
    HD5::Writer writer(coreArgs.oname.Get());
    writer.writeStruct(HD5::Keys::Info, info);
    writer.writeTensor(HD5::Keys::Data, out.dimensions(), out.data(), HD5::Dims::Images);
    if (coreArgs.residual) {
      writer.writeTensor(HD5::Keys::Residual, noncart.dimensions(), noncart.data(), HD5::Dims::Noncartesian);
      traj.write(writer);
    }
    if (Log::Saved().size()) { writer.writeStrings("log", Log::Saved()); }
    Log::Print(cmd, "Finished");
    return;
  }

  auto const R = f0Args.Nτ ? Recon(reconArgs.Get(), preArgs.Get(), gridArgs.Get(), senseArgs.Get(), traj, f0Args.Get(), noncart,
                                   reader.readTensor<Re3>("f0map"))
                           : Recon(reconArgs.Get(), preArgs.Get(), gridArgs.Get(), senseArgs.Get(), traj, basis.get(), noncart);
  auto const shape = R.A->ishape;
  float const scale = ScaleData(scaling.Get(), R.A, R.M, CollapseToVector(noncart));
  if (scale != 1.f) { noncart.device(Threads::TensorDevice()) = noncart * Cx(scale); }
  auto [reg, A, ext_x] = Regularizers(regOpts, R.A);

  Ops::Op::Vector x;
  if (pdhg) {
    PDHG::Debug debug = [shape, ext_x, di = debugIters.Get()](Index const ii, PDHG::Vector const &dx, PDHG::Vector const &x̅) {
      if (Log::IsDebugging() && (ii % di == 0)) {
        if (ext_x) {
          auto xit = ext_x->forward(dx);
          Log::Tensor(fmt::format("pdhg-x-{:02d}", ii), shape, xit.data(), HD5::Dims::Images);
          xit = ext_x->forward(x̅);
          Log::Tensor(fmt::format("pdhg-xbar-{:02d}", ii), shape, xit.data(), HD5::Dims::Images);
        } else {
          Log::Tensor(fmt::format("pdhg-x-{:02d}", ii), shape, dx.data(), HD5::Dims::Images);
          Log::Tensor(fmt::format("pdhg-xbar-{:02d}", ii), shape, x̅.data(), HD5::Dims::Images);
        }
      }
    };
    x = restart ? PDHG::Restarted(restart.Get(), CollapseToConstVector(noncart), A, R.M, reg, pdhgArgs.Get(), debug)
                : PDHG::Run(CollapseToConstVector(noncart), A, R.M, reg, pdhgArgs.Get(), debug);
  } else {
    ADMM::DebugX debug_x = [shape, di = debugIters.Get()](Index const ii, ADMM::Vector const &xx) {
      if (Log::IsDebugging() && (ii % di == 0)) {
        Log::Tensor(fmt::format("admm-x-{:02d}", ii), shape, xx.data(), HD5::Dims::Images);
      }
    };
    ADMM::DebugZ debug_z = [&, di = debugIters.Get()](Index const ii, Index const ir, ADMM::CMap const Fx, ADMM::CMap const z,
                                                      ADMM::CMap const u) {
      if (Log::IsDebugging() && debugZ && (ii % di == 0)) {
        if (std::holds_alternative<Sz5>(reg[ir].shape)) {
          auto const Fshape = std::get<Sz5>(reg[ir].shape);
          Log::Tensor(fmt::format("admm-Fx-{:02d}-{:02d}", ir, ii), Fshape, Fx.data(), HD5::Dims::Images);
          Log::Tensor(fmt::format("admm-z-{:02d}-{:02d}", ir, ii), Fshape, z.data(), HD5::Dims::Images);
          Log::Tensor(fmt::format("admm-u-{:02d}-{:02d}", ir, ii), Fshape, u.data(), HD5::Dims::Images);
        }
        if (std::holds_alternative<Sz6>(reg[ir].shape)) {
          auto const Fshape = std::get<Sz6>(reg[ir].shape);
          Log::Tensor(fmt::format("admm-Fx-{:02d}-{:02d}", ir, ii), Fshape, Fx.data(), {"b", "i", "j", "k", "t", "g"});
          Log::Tensor(fmt::format("admm-z-{:02d}-{:02d}", ir, ii), Fshape, z.data(), {"b", "i", "j", "k", "t", "g"});
          Log::Tensor(fmt::format("admm-u-{:02d}-{:02d}", ir, ii), Fshape, u.data(), {"b", "i", "j", "k", "t", "g"});
        }
      }
    };
    ADMM opt{A, R.M, reg, admmArgs.Get(), debug_x, debug_z};
    x = opt.run(CollapseToConstVector(noncart));
  }
  if (ext_x) { x = ext_x->forward(x); }
  if (scale != 1.f) { x.device(Threads::CoreDevice()) = x / Cx(scale); }
  auto const xm = AsConstTensorMap(x, R.A->ishape);

  TOps::Pad<5> oc(Concatenate(traj.matrixForFOV(cropFov.Get()), LastN<5 - ND>(shape)), R.A->ishape);
  auto         out = oc.adjoint(xm);
  HD5::Writer  writer(coreArgs.oname.Get());
  writer.writeStruct(HD5::Keys::Info, info);
  writer.writeTensor(HD5::Keys::Data, out.dimensions(), out.data(), HD5::Dims::Images);
  if (coreArgs.residual) {
    Log::Print(cmd, "Calculating K-space residual");
    fmt::print("|xm| {} |noncart| {}\n", Norm<true>(xm), Norm<true>(noncart));
    if (scale != 1.f) { noncart.device(Threads::TensorDevice()) = noncart / Cx(scale); }
    noncart -= R.A->forward(xm);
    fmt::print("|xm| {} |noncart| {}\n", Norm<true>(xm), Norm<true>(noncart));
    writer.writeTensor(HD5::Keys::Residual, noncart.dimensions(), noncart.data(), HD5::Dims::Noncartesian);
    traj.write(writer);
  }
  if (Log::Saved().size()) { writer.writeStrings("log", Log::Saved()); }
  Log::Print(cmd, "Finished");
}

void main_recon_rlsq(args::Subparser &parser) { run_recon_rlsq<3>(parser); }
void main_recon_rlsq2(args::Subparser &parser) { run_recon_rlsq<2>(parser); }
