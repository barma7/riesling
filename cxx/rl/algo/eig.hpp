#pragma once

#include "../log/log.hpp"
#include "../op/ops.hpp"
#include <optional>

namespace rl {

struct PowerReturn
{
  float            val;
  Eigen::VectorXcf vec;
};

auto PowerMethodForward(std::shared_ptr<Ops::Op> op, std::shared_ptr<Ops::Op> M, Index const iterLimit) -> PowerReturn;
auto PowerMethodAdjoint(std::shared_ptr<Ops::Op> op, std::shared_ptr<Ops::Op> M, Index const iterLimit) -> PowerReturn;

/* Largest eigenvalue of A' P A (P: a real, non-negative diagonal or nullptr) by Lanczos iteration: much closer after a
 * few tens of iterations than the power method when the top of the spectrum is dense (gridding operators, gradients).
 * Still an estimate from below. */
auto LanczosMax(std::shared_ptr<Ops::Op> A, std::shared_ptr<Ops::Op> P, Index const iterLimit) -> float;

} // namespace rl
