#pragma once

#include "args.hpp"

#include "rl/algo/regularizer.hpp"
#include "rl/gpu/pdhg.hpp"

#include <variant>

namespace rl {

struct RegOpts
{
  RegOpts(args::Subparser &parser);

  args::ValueFlag<std::string> iso;

  args::ValueFlag<float> l1;
  args::ValueFlag<float> l1i;
  args::ValueFlag<float> tv;
  args::ValueFlag<float> tv2;
  args::ValueFlag<float> tgv;
  args::ValueFlag<float> tvt;

  args::ValueFlag<float> llr;
  args::ValueFlag<Index> llrPatch;
  args::ValueFlag<Index> llrWin;
  args::Flag             llrShift;

  args::ValueFlag<float>                                   wavelets;
  args::ValueFlag<std::vector<Index>, VectorReader<Index>> waveDims;
  args::ValueFlag<Index>                                   waveWidth;
};

struct Regularizers_t
{
  std::vector<Regularizer> regs;
  Ops::Op::Ptr             A;
  Ops::Op::Ptr             ext_x;
};

auto Regularizers(RegOpts &regOpts, TOps::TOp<5, 5>::Ptr const &A) -> Regularizers_t;

/* The same regularizers for the GPU PDHG (rl/gpu/pdhg.hpp): the same transforms, proxes, λ and norm bounds as
 * Regularizers(), for all of its options (TGV, TV, TV + Laplacian, TVT, wavelets, LLR, L1, L1I; also combined). --tv2 is
 * built as two terms (gradient and Laplacian), the same problem as the CPU's stacked term. tgv: the unknowns are [x; v].
 */
struct GPURegularizers_t
{
  std::vector<gpu::Term> terms;
  bool                   tgv = false;
};
auto GPURegularizers(RegOpts &regOpts, Sz5 const shape) -> GPURegularizers_t;

} // namespace rl
