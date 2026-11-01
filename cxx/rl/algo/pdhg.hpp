#pragma once

#include "../op/ops.hpp"
#include "../prox/prox.hpp"
#include "../prox/stack.hpp"
#include "regularizer.hpp"

namespace rl::PDHG {
using Op = Ops::Op;
using Prox = Proxs::Prox;
using Vector = typename Op::Vector;
using CMap = typename Op::CMap;
using Debug = std::function<void(Index const, Vector const &, Vector const &)>;
struct Opts
{
  bool  adaptive, lad;
  bool  weighted = false; // Data term ½|Ax - b|²_P (ADMM's objective) instead of ½|Ax - b|² with P only as dual step
  Index imax;
  float tol, λE; // λE: λmax(A'PA), <= 0 to estimate it
  float ratio = 1.f; // τ / σ, the primal over the dual step (στ = 1/L either way)
  bool  termSteps = false; // Per-term steps (Pock & Chambolle 2011, block form); GPU only
  bool  autoRatio = false; // Start from the data, then adapt to (|x|/|y|)² at 50, 100, ... (pdhg.cpp, DataRatio)
};

auto Run(Vector const &b, Op::Ptr E, Op::Ptr P, std::vector<Regularizer> const &regs, Opts opts, Debug d = nullptr) -> Vector;
auto Run(CMap b, Op::Ptr E, Op::Ptr P, std::vector<Regularizer> const &regs, Opts opts, Debug d = nullptr) -> Vector;

auto Restarted(Index const r, Vector const &b, Op::Ptr E, Op::Ptr P, std::vector<Regularizer> const &regs, Opts opts, Debug d = nullptr)
  -> Vector;
auto Restarted(Index const r, CMap b, Op::Ptr E, Op::Ptr P, std::vector<Regularizer> const &regs, Opts opts, Debug d = nullptr) -> Vector;
} // namespace rl::PDHG
