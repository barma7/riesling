#pragma once

#include "../op/ops.hpp"
#include "../types.hpp"

namespace rl::Proxs {

struct Prox
{
  using Vector = Eigen::Vector<Cx, Eigen::Dynamic>;
  using Map = typename Vector::AlignedMapType;
  using CMap = typename Eigen::Map<Vector const, Eigen::AlignedMax>;
  using Ptr = std::shared_ptr<Prox>;

  Prox(Index const sz);

  void         apply(float const α, Vector &x) const;
  virtual void apply(float const α, Map x) const = 0;

  void         conj(float const α, Vector &x) const;
  virtual void conj(float const α, Map x) const = 0;

  /* Upper bound of |y|² for y in the set the conjugate prox projects onto (λ per element, block or patch for the norms):
   * the size of the dual solution, for PDHG's automatic step ratio. 0 if not known. */
  virtual auto dualBound2() const -> double { return 0.; }

  virtual ~Prox() {};

  Index sz;
};

#define PROX_INHERIT                                                                                                           \
  using Vector = typename Prox::Vector;                                                                                        \
  using Map = typename Prox::Map;                                                                                              \
  using CMap = typename Prox::CMap;                                                                                            \
  using Ptr = Prox::Ptr;                                                                                                       \
  using Prox::apply;                                                                                                           \
  using Prox::conj;

struct Conjugate final : Prox
{
  PROX_INHERIT
  static auto Make(Prox::Ptr p) -> Prox::Ptr;
  Conjugate(Prox::Ptr p);

  void apply(float const α, Map x) const;
  void conj(float const α, Map x) const;

private:
  Prox::Ptr p;
};

/* The Proximal Operator for f(x) = 0, which comes up in PDHG */
struct Null final : Prox
{
  PROX_INHERIT
  static auto Make(Index const sz) -> Prox::Ptr;
  Null(Index const sz);

  void apply(float const α, Map x) const;
  void conj(float const α, Map x) const;
};

} // namespace rl::Proxs
