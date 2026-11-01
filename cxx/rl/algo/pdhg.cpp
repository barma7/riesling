#include "pdhg.hpp"

#include "../log/log.hpp"
#include "../prox/norms.hpp"
#include "../prox/stack.hpp"
#include "../tensors.hpp"
#include "common.hpp"
#include "eig.hpp"
#include "iter.hpp"

namespace rl::PDHG {

struct PDResType
{
  float primal;
  float dual;
};
auto PDResiduals(Vector const               &x,
                 Vector const               &xold,
                 std::vector<Vector> const  &ys,
                 std::vector<Vector> const  &yolds,
                 std::vector<Op::Ptr> const &As,
                 std::vector<Op::Ptr> const &Ps,
                 float const                 τ,
                 float const                 σ) -> PDResType
{
  Vector xd(x.size()), p(x.size());
  xd.device(Threads::CoreDevice()) = xold - x;
  float pres = 0.f, dres = 0.f;
  for (Index ii = 0; ii < ys.size(); ii++) {
    p.device(Threads::CoreDevice()) = xd / τ;
    Vector yd(ys[ii].size()), d(ys[ii].size());
    yd.device(Threads::CoreDevice()) = yolds[ii] - ys[ii];
    if (As[ii] && Ps[ii]) { Ps[ii]->forward(yd, yd); }
    d.device(Threads::CoreDevice()) = yd / σ;
    if (As[ii] && Ps[ii]) {
      As[ii]->iadjoint(yd, p, -1.f);
      As[ii]->forward(xd, yd);
      Ps[ii]->iforward(yd, d, -1.f);
    } else if (As[ii]) {
      As[ii]->iadjoint(yd, p, -1.f);
      As[ii]->iforward(xd, d, -1.f);
    } else {
      p.device(Threads::CoreDevice()) -= yd;
      d.device(Threads::CoreDevice()) -= xd;
    }
    pres += std::pow(ParallelNorm(p), 2);
    dres += std::pow(ParallelNorm(d), 2);
  }
  pres = std::sqrt(pres);
  dres = std::sqrt(dres);
  return {pres, dres};
}

/* L >= λmax of K'ΣK for the stacked operator K = [A; T_1; ...] with the data rows weighted by the preconditioner P, so
 * that the step sizes σ = τ = 1/√L satisfy στ‖K‖² <= 1. λmax(A'PA) is estimated (Lanczos, 2% margin) unless given;
 * each regularizer carries an upper bound of λmax(T'T). Until 2026-11 L was λE (default 1) + the number of
 * regularizers, too small when λmax(A'PA) > 1 (~2.5 for a 3D radial subspace recon with RIESLING's weights) and for TGV
 * (∇x - v has λmax 2).
 */
auto StepL(Op::Ptr E, Op::Ptr P, std::vector<Regularizer> const &regs, float const λE) -> float
{
  float const λ = λE > 0.f ? λE : 1.02f * LanczosMax(E, P, 24);
  float       L = λ;
  for (auto const &r : regs) {
    L += r.norm2;
  }
  Log::Print("PDHG", "λmax(A'PA) {:4.3E}{} L {:4.3E}", λ, λE > 0.f ? " (given)" : " (estimated)", L);
  return L;
}

/* Automatic step ratio (Opts::autoRatio). Chambolle & Pock (2011) bound the error after k iterations by
 * (|x* - x0|²/τ + |y* - y0|²_Σ⁻¹) / k; with στ = 1/L fixed that is smallest for τ/σ = (|x*| / |y*|)² from x0 = y0 = 0, the
 * dual norm in the metric of the dual steps (|y|²/P for the preconditioned data term).
 *
 * DataRatio: the start, from the data, since the iterates estimate the norms badly while the ratio is far off (a small
 * ratio keeps x small, which pushes the estimate further down). |x*| from one exact line-search step of the data term
 * from 0, x̃ = α AᴴPb, α = |AᴴPb|² / <AᴴPb, AᴴPA AᴴPb>; the data dual from its residual |Ax̃ - b|²_P = |b|²_P - α|AᴴPb|²;
 * the regularizers' duals from their bounds (Prox::dualBound2).
 *
 * AutoRatio: afterwards, as the primal weight of PDLP (Applegate et al. 2021, "Practical large-scale linear programming
 * using primal-dual hybrid gradient"), the ratio moves halfway (in log space, at most a factor 2) towards the estimate
 * from the iterates, at iterations 50, 100, 200, ... only, so the steps settle.
 */
auto DataRatio(CMap b, Op::Ptr E, Op::Ptr P, std::vector<Regularizer> const &regs) -> float
{
  Vector const Pb = P ? P->forward(Vector(b)) : Vector(b);
  Vector const x1 = E->adjoint(Pb);
  Vector       Ax = E->forward(x1);
  if (P) { Ax = P->forward(Ax); }
  Vector const g = E->adjoint(Ax);
  double const n1 = std::pow(double(ParallelNorm(x1)), 2), q = std::real(x1.dot(g)), bP = std::real(b.dot(Pb));
  double const α = n1 / q, x2 = α * α * n1, yd2 = std::max(bP - α * n1, 0.);
  double       yr2 = 0.;
  for (auto const &r : regs) {
    yr2 += r.P->dualBound2();
  }
  double const ratio = x2 / (yd2 + yr2);
  Log::Print("PDHG", "Step ratio from the data: |x̃|² {:4.3E}, |y_data|² {:4.3E}, |y_terms|² <= {:4.3E}: τ/σ {:4.3E}", x2, yd2,
             yr2, ratio);
  return float(ratio);
}

void AutoRatio(Vector const &x, std::vector<Vector> const &ys, std::vector<Op::Ptr> const &Ps, float const L, float &ratio,
               float &σ, float &τ)
{
  double y2 = 0.;
  for (size_t ix = 0; ix < ys.size(); ix++) {
    if (Ps[ix]) {
      Vector t(ys[ix].size());
      Ps[ix]->inverse(ys[ix], t);
      y2 += std::real(ys[ix].dot(t));
    } else {
      y2 += std::pow(double(ParallelNorm(ys[ix])), 2);
    }
  }
  double const x2 = std::pow(double(ParallelNorm(x)), 2);
  if (!(y2 > 0.) || !(x2 > 0.)) { return; }
  double const target = x2 / y2;
  float const  r =
    std::clamp(float(std::exp(0.5 * std::log(target) + 0.5 * std::log(double(ratio)))), ratio / 2.f, 2.f * ratio);
  Log::Print("PDHG", "Step ratio τ/σ {:4.3E} -> {:4.3E} (target (|x|/|y|)² {:4.3E})", ratio, r, target);
  ratio = r;
  σ = 1.f / std::sqrt(L * ratio);
  τ = std::sqrt(ratio / L);
}

auto Run(Vector const &b, Op::Ptr A, Op::Ptr P, std::vector<Regularizer> const &regs, Opts opts, Debug d) -> Vector
{
  return Run(CMap{b.data(), b.rows()}, A, P, regs, opts, d);
}

auto Run(CMap b, Op::Ptr E, Op::Ptr P, std::vector<Regularizer> const &regs, Opts opts, Debug debug) -> Vector
{
  if (b.rows() != E->rows()) { throw(Log::Failure("PDHG", "b had {} rows, expected {}", b.rows(), E->rows())); }
  if (regs.size() < 1) { throw(Log::Failure("PDHG", "Requires at least one regularizer")); }
  Index const                   nx = regs.size() + 1;
  std::vector<Op::Ptr>          As(nx), Ps(nx); /* Transforms, preconditioners */
  std::vector<Proxs::Prox::Ptr> proxs(nx);      /* Proximal operators */

  /* Data consistency term */
  As[0] = E;
  Ps[0] = P;
  proxs[0] = opts.lad ? Proxs::L1::Make(1.f, b, P) : Proxs::SumOfSquares::Make(b, P, opts.weighted);
  /* Regularizers */
  for (Index ir = 0; ir < regs.size(); ir++) {
    As[ir + 1] = regs[ir].T;
    Ps[ir + 1] = nullptr; /* Regularizers are not preconditioned. Yet. */
    proxs[ir + 1] = regs[ir].P;
  }

  float const L = StepL(E, P, regs, opts.λE);
  float       σ = 1.f / std::sqrt(L * opts.ratio); // στ = 1/L, τ/σ = ratio
  float       τ = std::sqrt(opts.ratio / L);
  float       α = 0.5f;
  float const s = 1.f;
  float const η = 0.95f;
  float const Δ = 1.5f;
  Log::Print("PDHG", "{}{}σ {:4.3E} τ {:4.3E} Convergence Tolerance {}", opts.adaptive ? "Adaptive " : "",
             opts.lad ? "LAD " : (opts.weighted ? "weighted LS " : ""), σ, τ, opts.tol);
  Vector x(E->cols()), x̅(E->cols()), xold(E->cols());
  x.setZero();
  x̅.setZero();
  xold.setZero();

  std::vector<Vector> ys(nx), yolds(nx);
  for (Index ir = 0; ir < nx; ir++) {
    if (As[ir]) {
      ys[ir] = Vector(As[ir]->rows());
      yolds[ir] = Vector(As[ir]->rows());
    } else {
      ys[ir] = Vector(E->cols());
      yolds[ir] = Vector(E->cols());
    }
    ys[ir].setZero();
    yolds[ir].setZero();
  }

  if (opts.autoRatio && opts.adaptive) { throw Log::Failure("PDHG", "--pdhg-auto-ratio and --adaptive exclude each other"); }
  if (opts.termSteps) { throw Log::Failure("PDHG", "--pdhg-term-steps is only implemented for the GPU PDHG (--gpu)"); }
  float ratio = opts.ratio;
  Index nextRatio = 50;
  if (opts.autoRatio) {
    ratio = DataRatio(b, E, P, regs);
    σ = 1.f / std::sqrt(L * ratio);
    τ = std::sqrt(ratio / L);
  }
  Iterating::Starting();
  for (Index ii = 0; ii < opts.imax; ii++) {
    /* PDHG steps are
     * yn = proxF*σ(y + σAx̅)
     * xn = proxGτ(x - τA'yn)
     * x̅ = xn + (xn - x)
     *
     *      But G(x) = 0 so proxGτ(x) = x
     */
    for (Index ix = 0; ix < nx; ix++) {
      yolds[ix] = ys[ix];
      if (As[ix] && Ps[ix]) {
        auto ytemp = As[ix]->forward(x̅);
        Ps[ix]->iforward(ytemp, ys[ix], σ);
      } else if (As[ix]) {
        As[ix]->iforward(x̅, ys[ix], σ);
      } else {
        ys[ix].device(Threads::CoreDevice()) += σ * x̅;
      }
      proxs[ix]->conj(σ, ys[ix]);
    }

    xold.device(Threads::CoreDevice()) = x;
    for (Index ix = 0; ix < nx; ix++) {
      if (As[ix]) {
        As[ix]->iadjoint(ys[ix], x, -τ);
      } else {
        x.device(Threads::CoreDevice()) -= τ * ys[ix];
      }
    }
    x̅.device(Threads::CoreDevice()) = 2.f * x - xold;

    if (debug) { debug(ii, x, x̅); }
    float const normx = ParallelNorm(x);
    if (opts.autoRatio && ii + 1 == nextRatio) {
      nextRatio *= 2;
      AutoRatio(x, ys, Ps, L, ratio, σ, τ);
    }

    if (opts.adaptive) {
      auto const res = PDResiduals(x, xold, ys, yolds, As, Ps, τ, σ);

      if (res.primal > s * res.dual * Δ) { /* Large primal */
        τ = τ / (1.f - α);
        σ = σ * (1.f - α);
        α = α * η;
      } else if (res.primal < s * res.dual / Δ) { /* Large dual */
        τ = τ * (1.f - α);
        σ = σ / (1.f - α);
        α = α * η;
      }

      Log::Print("PDHG", "{:02d}: |x| {:4.3E} |P| {:4.3E} |D| {:4.3E} σ {:4.3E} τ {:4.3E} α {:4.3E}", ii, normx, res.primal,
                 res.dual, σ, τ, α);
      if (res.primal / normx < opts.tol) {
        Log::Print("PDHG", "Primal tolerance reached");
        break;
      }
    } else {
      xold.device(Threads::CoreDevice()) -= x;
      float const normdx = ParallelNorm(xold);
      float const rel = normdx / normx;
      Log::Print("PDHG", "{:02d}: |x| {:4.3E} |Δx|/|x| {:4.3E}", ii, normx, rel);
      if (rel < opts.tol) {
        Log::Print("PDHG", "Δ tolerance reached");
        break;
      }
    }

    if (Iterating::ShouldStop("PDHG")) { break; }
  }
  Iterating::Finished();
  return x;
}

auto Restarted(Index const r, Vector const &b, Op::Ptr A, Op::Ptr P, std::vector<Regularizer> const &regs, Opts opts, Debug d)
  -> Vector
{
  return Restarted(r, CMap{b.data(), b.rows()}, A, P, regs, opts, d);
}

auto Restarted(Index const nInner, CMap b, Op::Ptr E, Op::Ptr P, std::vector<Regularizer> const &regs, Opts opts, Debug debug)
  -> Vector
{
  if (b.rows() != E->rows()) { throw(Log::Failure("PDHG", "b had {} rows, expected {}", b.rows(), E->rows())); }
  if (regs.size() < 1) { throw(Log::Failure("PDHG", "Requires at least one regularizer")); }
  if (opts.autoRatio) { throw Log::Failure("PDHG", "--pdhg-auto-ratio is not implemented with restarts"); }
  auto                          dev = Threads::CoreDevice();
  Index const                   nx = regs.size() + 1;
  std::vector<Op::Ptr>          As(nx), Ps(nx); /* Transforms, preconditioners */
  std::vector<Proxs::Prox::Ptr> proxs(nx);      /* Proximal operators */

  /* Data consistency term */
  As[0] = E;
  Ps[0] = P;
  proxs[0] = opts.lad ? Proxs::L1::Make(1.f, b, P) : Proxs::SumOfSquares::Make(b, P, opts.weighted);
  /* Regularizers */
  for (Index ir = 0; ir < regs.size(); ir++) {
    As[ir + 1] = regs[ir].T;
    Ps[ir + 1] = nullptr; /* Regularizers are not preconditioned. Yet. */
    proxs[ir + 1] = regs[ir].P;
  }

  float const L = StepL(E, P, regs, opts.λE);
  float       σ = 1.f / std::sqrt(L * opts.ratio); // στ = 1/L, τ/σ = ratio
  float       τ = std::sqrt(opts.ratio / L);
  float       α = 0.5f;
  float const s = 1.f;
  float const η = 0.95f;
  float const Δ = 1.5f;
  Log::Print("PDHG", "{}{}σ {:4.3E} τ {:4.3E} Convergence Tolerance {}", opts.adaptive ? "Adaptive " : "",
             opts.lad ? "LAD " : (opts.weighted ? "weighted LS " : ""), σ, τ, opts.tol);
  Vector x(E->cols()), x̂(E->cols()) /* Extrapolated/Momentum */, x̅(E->cols()) /* Average of iterations */, xold(E->cols());
  x̅.setZero();

  std::vector<Vector> ys(nx), y̅s(nx), yolds(nx);
  for (Index ir = 0; ir < nx; ir++) {
    if (As[ir]) {
      ys[ir] = Vector(As[ir]->rows());
      y̅s[ir] = Vector(As[ir]->rows());
      if (opts.adaptive) { yolds[ir] = Vector(As[ir]->rows()); }
    } else {
      ys[ir] = Vector(E->cols());
      y̅s[ir] = Vector(E->cols());
      if (opts.adaptive) { yolds[ir] = Vector(E->cols()); }
    }
    y̅s[ir].setZero();
  }

  Index const nOuter = opts.imax / nInner;
  Iterating::Starting();
  for (Index io = 0; io < nOuter; io++) {
    x.device(dev) = x̅;
    x̂.device(dev) = x̅;
    for (Index ir = 0; ir < nx; ir++) {
      ys[ir] = y̅s[ir];
    }
    for (Index ii = 0; ii < nInner; ii++) {
      /* PDHG steps are
       * yn = proxF*σ(y + σAx̂)
       * xn = proxGτ(x - τA'yn)
       * x̂ = xn + (xn - x)
       *
       *      But G(x) = 0 so proxGτ(x) = x
       */
      Index const itot = io * nInner + ii;
      for (Index ix = 0; ix < nx; ix++) {
        if (opts.adaptive) { yolds[ix].device(dev) = ys[ix]; }
        if (As[ix] && Ps[ix]) {
          auto ytemp = As[ix]->forward(x̂);
          Ps[ix]->iforward(ytemp, ys[ix], σ);
        } else if (As[ix]) {
          As[ix]->iforward(x̂, ys[ix], σ);
        } else {
          ys[ix].device(dev) += σ * x̂;
        }
        proxs[ix]->conj(σ, ys[ix]);
        y̅s[ix].device(dev) += (ys[ix] - y̅s[ix]) / (ii + 2); /* Running average formula */
      }

      xold.device(dev) = x;
      for (Index ix = 0; ix < nx; ix++) {
        if (As[ix]) {
          As[ix]->iadjoint(ys[ix], x, -τ);
        } else {
          x.device(dev) -= τ * ys[ix];
        }
      }
      x̂.device(dev) = 2.f * x - xold;
      x̅.device(dev) += (x - x̅) / (ii + 2); /* Running average formula */

      if (debug) { debug(itot, x, x̅); }
      float const normx = ParallelNorm(x);
      float const normx̅ = ParallelNorm(x̅);
      if (opts.adaptive) {
        auto const res = PDResiduals(x, xold, ys, yolds, As, Ps, τ, σ);

        if (res.primal > s * res.dual * Δ) { /* Large primal */
          τ = τ / (1.f - α);
          σ = σ * (1.f - α);
          α = α * η;
        } else if (res.primal < s * res.dual / Δ) { /* Large dual */
          τ = τ * (1.f - α);
          σ = σ / (1.f - α);
          α = α * η;
        }

        Log::Print("PDHG", "{:02d}: |x̅| {:4.3E} |x| {:4.3E} |P| {:4.3E} |D| {:4.3E} σ {:4.3E} τ {:4.3E} α {:4.3E}", itot, normx̅,
                   normx, res.primal, res.dual, σ, τ, α);
        if (res.primal / normx < opts.tol) {
          Log::Print("PDHG", "Primal tolerance reached");
          break;
        }
      } else {
        xold.device(dev) -= x;
        float const normdx = ParallelNorm(xold);
        float const rel = normdx / normx;
        Log::Print("PDHG", "{:02d}: |x̅| {:4.3E} |x| {:4.3E} |Δx|/|x| {:4.3E}", itot, normx̅, normx, rel);
        if (rel < opts.tol) {
          Log::Print("PDHG", "Δ tolerance reached");
          break;
        }
      }

      if (Iterating::ShouldStop("PDHG")) { break; }
    }
    if (Iterating::ShouldStop("PDHG")) { break; }
  }
  Iterating::Finished();
  return x̅;
}

} // namespace rl::PDHG
