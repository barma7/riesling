#pragma once

#include "prox.hpp"

namespace rl::Proxs {

/*
 * Locally Low-Rank Regularizer
 *
 * The patch size is the volume over which the low-rank (SVD) calculation is performed.
 * Window size is the volume within this that is copied to the output. Set to 1 to get true sliding-window.
 */
template<int D>
struct LLR final : Prox
{
  PROX_INHERIT
  float λ;
  Index patchSize, windowSize;
  Sz<D>   shape;
  bool  shift;
  LLR(float const, Index const, Index const, bool const, Sz<D> const);

  void apply(float const α, Map x) const;
  void conj(float const α, Map x) const;
  auto dualBound2() const -> double; // Spectral norm λ per window block: λ² min(window rows, images) per window
};

} // namespace rl::Proxs
