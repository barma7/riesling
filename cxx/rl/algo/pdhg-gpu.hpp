#pragma once

#include "../gpu/pdhg.hpp"
#include "../op/sense-nufft-gpu.hpp"
#include "pdhg.hpp"

namespace rl::PDHG {

/*
 * PDHG on the GPU (rl/gpu/pdhg.hpp): data (nC, nSamp, nTrace, 1, nTime) on the host, the operator of one time point
 * (Recon::G). s: the sizes and the regularizer terms (GPURegularizers in riesling/regularizers.hpp); opts as for the CPU
 * PDHG (adaptive steps and restarts are not implemented). Logs as the CPU version; debug(it, [x; v]) every debugEvery
 * iterations. Returns x (nx, ny, nz, nB, nTime).
 */
auto RunGPU(Cx5 const                                 &data,
            TOps::SENSENUFFTGPU const                 &op,
            gpu::PDHGSetup                             s,
            Opts const                                &opts,
            std::function<void(Index, Vector const &)> debug,
            Index                                      debugEvery) -> Vector;

} // namespace rl::PDHG
