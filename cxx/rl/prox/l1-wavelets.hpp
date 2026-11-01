#pragma once

#include "../op/pad.hpp"
#include "../op/wavelets.hpp"
#include "norms.hpp"

namespace rl::Proxs {

struct L1Wavelets final : Prox
{
  PROX_INHERIT

  L1Wavelets(float const λ, Sz5 const shape, Index const width, std::vector<Index> const dims);
  void apply(float const α, Map x) const;
  void conj(float const α, Map x) const;
  auto dualBound2() const -> double { return l1.dualBound2(); } // Orthonormal wavelets: as L1 on the coefficients

private:
  Ops::Op::Ptr waves;
  L1           l1;
};

} // namespace rl::Proxs
