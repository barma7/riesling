#pragma once

#include "../op/ops.hpp"
#include "regularizer.hpp"

namespace rl {

struct ADMM
{
  using Op = Ops::Op;
  using Vector = typename Op::Vector;
  using Map = typename Op::Map;
  using CMap = typename Op::CMap;
  using DebugX = std::function<void(Index const, Vector const &)>;
  using DebugZ = std::function<void(Index const, Index const, CMap, CMap, CMap)>; // Maps: no copies of work vectors

  struct Opts
  {
    Index iters0 = 4;    // Number of inner iterations on first outer iteration
    Index iters1 = 1;    // Number of inner iterations on subsequent outer iterations
    float aTol = 1.e-6f; // LSMR tolerance parameters
    float bTol = 1.e-6f;
    float cTol = 1.e-6f;

    Index outerLimit; // Number of outer iterations
    float ε;          // Combined apply/dual tolerance parameter

    float ρ;       // Penalty parameter
    bool  balance; // Apply residual balancing scheme

    float μ; // Residual balancing parameters
    float τmax;

    float ɑ = 0.f; // Over-relaxation parameter, set 1 < ɑ < 2

    bool cg = true; // x-update with CG on the normal equations; false: LSMR on the stacked system
  };

  Op::Ptr                  A;    // Forward model
  Op::Ptr                  Minv; // Pre-conditioner
  std::vector<Regularizer> regs;
  Opts                     opts;
  DebugX                   debug_x = nullptr;
  DebugZ                   debug_z = nullptr;
  Op::Ptr                  AhWA = nullptr; // Optional Aᴴ W A in one operator (a GPU): CG uses it instead of A, Minv, Aᴴ
  // Optional: page-lock host memory for the run (the CG vectors that AhWA reads and writes, for faster copies to a GPU).
  // Returns a token that unlocks it when released.
  std::function<std::shared_ptr<void>(void *, size_t)> pinHost = nullptr;

  auto run(Vector const &b) const -> Vector;
  auto run(CMap b) const -> Vector;
};

} // namespace rl
