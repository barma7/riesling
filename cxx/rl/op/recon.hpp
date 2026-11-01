#pragma once

#include "../io/reader.hpp"
#include "../op/compose.hpp"
#include "../precon.hpp"
#include "../sense/sense.hpp"
#include "recon-opts.hpp"

/*
 *  Full recon operators
 */

namespace rl::TOps {
struct SENSENUFFTGPU;
}

namespace rl {
template <int ND> struct Recon
{
  Recon(ReconOpts const       &rOpts,
        PreconOpts const      &pOpts,
        GridOpts<ND> const    &gridOpts,
        SENSE::Opts<ND> const &senseOpts,
        TrajectoryN<ND> const &traj,
        Basis::CPtr            basis,
        Cx5 const             &data);

  Recon(ReconOpts const       &rOpts,
        PreconOpts const      &pOpts,
        GridOpts<ND> const    &gridOpts,
        SENSE::Opts<ND> const &senseOpts,
        TrajectoryN<ND> const &traj,
        f0Opts const          &f0Opts,
        Cx5 const             &data,
        Re3 const             &f0map);
  TOps::TOp<5, 5>::Ptr A, M;
  TOps::TOp<5, 5>::Ptr N; // Aᴴ M A as one operator (--gpu), else nullptr
  std::shared_ptr<TOps::SENSENUFFTGPU> G; // The GPU operator of one time point (--gpu), for GPU solvers
};
} // namespace rl