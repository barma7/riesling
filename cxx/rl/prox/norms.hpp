#pragma once

#include "prox.hpp"
#include "rl/op/op.hpp"

namespace rl::Proxs {

struct L1 final : Prox
{
  PROX_INHERIT
  float       λ;
  static auto Make(float const λ, Index const sz) -> Prox::Ptr;
  static auto Make(float const λ, CMap b, Ops::Op::Ptr P) -> Prox::Ptr;
  L1(float const λ, Index const sz);
  L1(float const λ, CMap b, Ops::Op::Ptr P);
  void apply(float const α, Map x) const;
  void conj(float const α, Map x) const;
  auto dualBound2() const -> double { return double(λ) * λ * sz; }

private:
  CMap const   b;
  Ops::Op::Ptr P;
};

struct L1I final : Prox
{
  PROX_INHERIT
  float       λ;
  static auto Make(float const λ, Index const sz) -> Prox::Ptr;
  L1I(float const λ, Index const sz);
  void apply(float const α, Map x) const;
  void conj(float const α, Map x) const;
  auto dualBound2() const -> double { return double(λ) * λ * sz; }

private:
  Ops::Op::Ptr P;
};

template <int O, int D> struct L2 final : Prox
{
  PROX_INHERIT
  float       λ;
  Index       blockSize;
  static auto Make(float const λ, Sz<O> const &shape, Sz<D> const &dims) -> Prox::Ptr;
  L2(float const λ, Sz<O> const &shape, Sz<D> const &dims);
  void apply(float const α, Map x) const;
  void conj(float const α, Map x) const;
  auto dualBound2() const -> double; // λ² x the number of blocks

private:
  Sz<O>     shape;
  Sz<D>     normDims;
  Sz<O - D> otherDims;
};

/* ½|x - b|², or with weighted ½|x - b|²_P (P a positive diagonal). In PDHG the preconditioner P is also the dual step of
 * the data term (Ong et al. 2019): without weighted it only speeds up convergence, with weighted it is also the data
 * weighting of the objective (as in RIESLING's ADMM / LSMR). Only conj() knows about the weighting.
 */
struct SumOfSquares final : Prox
{
  PROX_INHERIT
  static auto Make(CMap b, Ops::Op::Ptr precon = nullptr, bool const weighted = false) -> Prox::Ptr;
  SumOfSquares(CMap b, Ops::Op::Ptr precon, bool const weighted = false);
  void apply(float const α, Map x) const;
  void conj(float const α, Map x) const;

private:
  CMap const   b;
  Ops::Op::Ptr P;
  bool         weighted;
};

} // namespace rl::Proxs
