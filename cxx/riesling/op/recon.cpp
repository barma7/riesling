#include "inputs.hpp"

#include "rl/io/hd5.hpp"
#include "rl/log/log.hpp"
#include "rl/op/recon.hpp"
#include "rl/op/sense-nufft-gpu.hpp"
#include "rl/sys/threads.hpp"
#include "rl/types.hpp"

using namespace rl;

/*
 * Applies the normal operator Aᴴ M A of a recon (SENSE, basis, k-space weights M as recon-lsq/rlsq build them) to
 * x0 = Aᴴ M b, b the input data, a number of times, and writes x0 and the last result. For timing the operator (e.g. with
 * and without --gpu) and comparing implementations on real data.
 */
void main_op_recon(args::Subparser &parser)
{
  CoreArgs<3>            coreArgs(parser);
  GridArgs<3>            gridArgs(parser);
  PreconArgs             preArgs(parser);
  ReconArgs              reconArgs(parser);
  SENSEArgs<3>           senseArgs(parser);
  args::ValueFlag<Index> repeat(parser, "N", "Number of applications of the normal operator (3)", {"repeat"}, 3);
  args::ValueFlag<std::string> gridAdj(parser, "F", "Write the GPU gridding adjoint of the data (no FFT) to F and stop",
                                       {"grid-adj"});

  ParseCommand(parser, coreArgs.iname, coreArgs.oname);
  auto const     cmd = parser.GetCommand().Name();
  HD5::Reader    reader(coreArgs.iname.Get());
  Info const     info = reader.readStruct<Info>(HD5::Keys::Info);
  TrajectoryN<3> traj(reader, info.voxel_size.head<3>(), coreArgs.matrix.Get());
  auto const     noncart = reader.readTensor<Cx5>(coreArgs.dset.Get());
  traj.checkDims(FirstN<3>(noncart.dimensions()));
  auto const basis = LoadBasis(coreArgs.basisFile.Get());
  if (gridAdj) { // Gridding adjoint of the first slab and time point on the GPU, as op-grid --adj writes it
    Index const nC = noncart.dimension(0);
    Cx5         ones(AddBack(traj.matrixForFOV(gridArgs.Get().fov), 1, nC));
    ones.setConstant(Cx(1.f)); // The maps are not used by the gridding
    auto const G = TOps::MakeSENSENUFFTGPU(gridArgs.Get(), traj, ones, basis.get(), Re3());
    Cx3 const  y = noncart.chip<4>(0).chip<3>(0);
    Cx5 const  g = G->gridAdjoint(y);
    HD5::Writer writer(gridAdj.Get());
    writer.writeStruct(HD5::Keys::Info, info);
    writer.writeTensor(HD5::Keys::Data, AddBack(g.dimensions(), 1), g.data(), HD5::Dims::Channels);
    Log::Print(cmd, "Finished");
    return;
  }
  auto const R = Recon(reconArgs.Get(), preArgs.Get(), gridArgs.Get(), senseArgs.Get(), traj, basis.get(), noncart);

  auto t = Log::Now();
  Cx5  Mb = noncart;
  if (R.M) { R.M->forward(noncart, Mb); }
  Cx5 const x0 = R.A->adjoint(Mb);
  Log::Print(cmd, "x0 = Aᴴ M b took {}", Log::ToNow(t));

  Cx5 y(x0.dimensions()), yA(R.A->oshape), yM(R.A->oshape);
  for (Index ii = 0; ii < repeat.Get(); ii++) {
    t = Log::Now();
    if (R.N) {
      R.N->forward(x0, y);
    } else {
      R.A->forward(x0, yA);
      if (R.M) {
        R.M->forward(yA, yM);
        R.A->adjoint(yM, y);
      } else {
        R.A->adjoint(yA, y);
      }
    }
    Log::Print(cmd, "Aᴴ M A {} took {}", ii, Log::ToNow(t));
  }

  HD5::Writer writer(coreArgs.oname.Get());
  writer.writeStruct(HD5::Keys::Info, info);
  writer.writeTensor(HD5::Keys::Data, y.dimensions(), y.data(), HD5::Dims::Images);
  writer.writeTensor("x0", x0.dimensions(), x0.data(), HD5::Dims::Images);
  Log::Print(cmd, "Finished");
}
