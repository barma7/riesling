#pragma once

namespace rl {
struct ReconOpts
{
  bool decant, lowmem;
  bool gpu = false; // SENSE + NUFFT on a CUDA GPU (3D, one slab)
};

struct f0Opts
{
  float τ0, τacq;
  Index Nτ;
};

} // namespace rl
