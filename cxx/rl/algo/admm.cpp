#include "admm.hpp"

#include "../log/log.hpp"
#include "../op/top.hpp"
#include "../tensors.hpp"
#include "common.hpp"
#include "iter.hpp"
#include "lsmr.hpp"

#include <future>

namespace rl {

auto ADMM::run(Vector const &b) const -> Vector { return run(CMap{b.data(), b.rows()}); }

auto ADMM::run(CMap b) const -> Vector
{
  /* See https://web.stanford.edu/~boyd/papers/admm/lasso/lasso_lsqr.html
   * For the least squares part we are solving:
  A'x = b'
  [     A            [     b
    √ρ F_1             √ρ (z_1 - u_1)
    √ρ F_2     x =     √ρ (z_2 - u_2)
      ...              ...
    √ρ F_n ]           √ρ (z_n - u_n) ]

    Then for the regularizers, in turn we do:

    z_i = prox_λ/ρ(F_i * x + u_{i-1})
    u_i = F_i * x + u_{i-1} - z_i
    */
  if (b.rows() != A->rows()) { throw Log::Failure("ADMM", "b was size {} expected {}", b.rows(), A->rows()); }
  auto &dev = Threads::CoreDevice(); // Not const: Eigen ignores a const device and evaluates on one thread

  Index const                                  R = regs.size();
  std::vector<Vector>                          z(R), u(R);
  std::vector<std::shared_ptr<Ops::DiagScale>> ρdiags(R);
  std::vector<std::shared_ptr<Ops::Op>>        scaled_ops(R);

  float ρ = opts.ρ;
  for (Index ir = 0; ir < R; ir++) {
    Index const sz = regs[ir].T ? regs[ir].T->rows() : A->cols();
    z[ir].resize(sz);
    z[ir].setZero();
    u[ir].resize(sz);
    u[ir].setZero();
    ρdiags[ir] = std::make_shared<Ops::DiagScale>(sz, std::sqrt(ρ));
    scaled_ops[ir] = regs[ir].T ? std::static_pointer_cast<Ops::Op>(std::make_shared<Ops::Multiply>(ρdiags[ir], regs[ir].T))
                                : std::static_pointer_cast<Ops::Op>(ρdiags[ir]);
  }

  std::shared_ptr<Op> reg = Ops::VStack::Make(scaled_ops);
  std::shared_ptr<Op> Aʹ = Ops::VStack::Make(A, scaled_ops);
  std::shared_ptr<Op> Minvʹ;
  if (Minv == nullptr) {
    Minvʹ = nullptr;
  } else {
    std::shared_ptr<Op> I = std::make_shared<Ops::Identity>(reg->rows());
    Minvʹ = std::make_shared<Ops::DStack>(Minv, I);
  }
  LSMR lsmr{Aʹ, Minvʹ, nullptr, LSMR::Opts{opts.iters0, opts.aTol, opts.bTol, opts.cTol}};

  Vector x(A->cols());
  x.setZero();

  /* LSMR solves the x-update on the stacked system above, with right-hand side bʹ. CG instead solves its normal equations
   *   (Aᴴ W A + ρ Σ F_iᴴ F_i) x = Aᴴ W b + ρ Σ F_iᴴ (z_i - u_i)
   * where W is the k-space weighting (Minv), the same minimiser. It only needs vectors of the size of x, and keeps
   * q = Aᴴ W A x up to date from the products it computes anyway, so the warm start costs no extra pass of A.
   */
  Vector bʹ, AhWb, q;
  if (AhWA && (AhWA->rows() != A->cols() || AhWA->cols() != A->cols())) {
    throw Log::Failure("ADMM", "AᴴWA is {}x{}, expected {}x{}", AhWA->rows(), AhWA->cols(), A->cols(), A->cols());
  }
  if (opts.cg) {
    Log::Print("ADMM", "x-update: CG on the normal equations{}", AhWA ? ", AᴴWA in one operator" : "");
    Vector Wb(A->rows());
    if (Minv) {
      Minv->forward(b, Map(Wb.data(), Wb.size()));
    } else {
      Wb.device(dev) = b;
    }
    AhWb = A->adjoint(Wb);
    q.resize(A->cols());
    q.setZero(); // x = 0
  } else {
    bʹ.resize(Aʹ->rows());
    bʹ.setZero();
    bʹ.head(A->rows()).device(dev) = b;
  }

  /* Work vectors in one allocation, made once: the CG x-update and the z/u-update use the same memory one after the other.
   * Fresh allocations every outer iteration cost page faults (the kernel zeroes each page on its first write), and
   * separate ones for the two phases would add up in memory.
   */
  Index const cols = A->cols();
  Index       maxT = 0, maxZ = 0; // Largest F_i x, largest z_i
  for (auto const &rg : regs) {
    if (rg.T) { maxT = std::max(maxT, rg.T->rows()); }
    maxZ = std::max(maxZ, rg.T ? rg.T->rows() : cols);
  }
  auto const  pad = [](Index const n) { return (n + 15) / 16 * 16; }; // Keeps each vector aligned for Map
  Index const nyA = AhWA ? 0 : A->rows(), nyW = (!AhWA && Minv) ? A->rows() : 0; // k-space work vectors
  Index const cgSize = opts.cg ? 4 * pad(cols) + pad(nyA) + pad(nyW) + pad(maxT) : 0;
  Index const zuSize = 2 * pad(maxZ) + pad(cols);
  Vector      work;
  auto const  carve = [&work, &pad](std::initializer_list<Index> const sizes) { // Consecutive vectors in work
    std::vector<Map> v;
    Index            offset = 0;
    for (Index const n : sizes) {
      v.emplace_back(work.data() + offset, n);
      offset += pad(n);
    }
    return v;
  };
  if (opts.cg) { work.resize(std::max(cgSize, zuSize)); }
  std::shared_ptr<void> pinned; // p and Ap (work vectors 1 and 2 of the CG); released before work
  if (opts.cg && AhWA && pinHost) { pinned = pinHost(work.data() + pad(cols), 2 * pad(cols) * sizeof(Cx)); }

  auto cgUpdate = [&](Index const imax) {
    auto v = carve({cols, cols, cols, cols, nyA, nyW, maxT});
    Map &r = v[0], &p = v[1], &Ap = v[2], &Gp = v[3], &yA = v[4], &yW = v[5], &t = v[6];
    // r = Aᴴ W b - q + ρ Σ F_iᴴ (z_i - u_i - F_i x)
    r.device(dev) = AhWb - q;
    for (Index ir = 0; ir < R; ir++) {
      if (regs[ir].T) {
        Map tm(t.data(), regs[ir].T->rows());
        regs[ir].T->forward(CMap(x.data(), cols), tm);
        tm.device(dev) = (z[ir] - u[ir] - tm) * ρ;
        regs[ir].T->iadjoint(CMap(t.data(), tm.size()), Map(r.data(), cols));
      } else {
        r.device(dev) += (z[ir] - u[ir] - x) * ρ;
      }
    }
    p.device(dev) = r;
    float rr = CheckedDot(r, r);
    Log::Print("CG", "IT |r|       α         |x|");
    Log::Print("CG", "00 {:4.3E}", std::sqrt(rr));
    for (Index ii = 0; ii < imax; ii++) {
      /* Ap = Aᴴ W A p, Gp = ρ Σ F_iᴴ F_i p. A single AᴴWA (on a GPU) runs on its own thread while the regularizer terms
       * run here on the CPU threads: both only read p.
       */
      std::future<void> ApDone;
      if (AhWA) {
        ApDone = std::async(std::launch::async, [&] { AhWA->forward(CMap(p.data(), cols), Ap); });
      } else {
        A->forward(CMap(p.data(), cols), yA);
        if (Minv) {
          Minv->forward(CMap(yA.data(), yA.size()), yW);
          A->adjoint(CMap(yW.data(), yW.size()), Ap);
        } else {
          A->adjoint(CMap(yA.data(), yA.size()), Ap);
        }
      }
      for (Index ir = 0; ir < R; ir++) {
        if (regs[ir].T) {
          Map tm(t.data(), regs[ir].T->rows());
          regs[ir].T->forward(CMap(p.data(), cols), tm);
          tm.device(dev) = tm * ρ;
          if (ir == 0) {
            regs[ir].T->adjoint(CMap(t.data(), tm.size()), Map(Gp.data(), cols));
          } else {
            regs[ir].T->iadjoint(CMap(t.data(), tm.size()), Map(Gp.data(), cols));
          }
        } else if (ir == 0) {
          Gp.device(dev) = p * ρ;
        } else {
          Gp.device(dev) += p * ρ;
        }
      }
      if (ApDone.valid()) { ApDone.get(); }
      // pᴴ N p is real for the Hermitian N; its imaginary part is rounding only, so take the real part
      float const pNp = std::real(ParallelDot(p, Ap)) + std::real(ParallelDot(p, Gp));
      if (!(pNp > 0.f)) {
        Log::Print("CG", "pᴴNp {} is not positive, stopping", pNp);
        break;
      }
      float const α = rr / pNp;
      x.device(dev) = x + p * α;
      q.device(dev) = q + Ap * α;
      r.device(dev) = r - (Ap + Gp) * α;
      float const rrNew = CheckedDot(r, r);
      p.device(dev) = r + p * (rrNew / rr);
      rr = rrNew;
      Log::Print("CG", "{:02d} {:4.3E} {:4.3E} {:4.3E}", ii + 1, std::sqrt(rr), α, ParallelNorm(x));
      if (rr == 0.f) { break; }
    }
  };

  Log::Print("ADMM", "Abs ε {}", opts.ε);
  Iterating::Starting();
  for (Index io = 0; io < opts.outerLimit; io++) {
    if (opts.cg) {
      cgUpdate(io == 0 ? opts.iters0 : opts.iters1);
    } else {
      Index start = A->rows();
      for (Index ir = 0; ir < R; ir++) {
        Index rr = regs[ir].T ? regs[ir].T->rows() : A->cols();
        bʹ.segment(start, rr).device(dev) = std::sqrt(ρ) * (z[ir] - u[ir]);
        start += rr;
        ρdiags[ir]->scale = std::sqrt(ρ);
      }
      x = lsmr.run(bʹ, x);
      lsmr.opts.imax = opts.iters1;
    }
    if (debug_x) { debug_x(io, x); }

    if (!opts.cg) { work.resize(zuSize); } // LSMR has its own vectors: only hold these while updating z and u
    float normFx = 0.f, normz = 0.f, normu = 0.f, pRes = 0.f, dRes = 0.f;
    for (Index ir = 0; ir < R; ir++) {
      float       nz, nu, nP, nD;
      Index const sz = z[ir].size();
      /* w = u + F x (relaxed), then the new z in place (prox); Fx = F x, then z_new - z_old; Fh = Fᴴ(...) for the
       * residuals. The old z stays in z[ir] until the end, so no copy of it is needed. Same arithmetic as updating u
       * in place: (u + Fx) is rounded once either way.
       */
      auto zv = carve({sz, sz, cols});
      Map &w = zv[0], &Fx = zv[1], &Fh = zv[2];
      // Note that in the Boyd primer relaxation is defined as ɑ * Ax - (1.f - ɑ) * (Bz - c) but this comes from the
      // constraint that Ax + Bz = c Our constraint is Fx = z, which defines A = F and B = -I, and hence the minus sign
      // becomes a plus in this line below, which matches the code examples on Boyd's website
      if (regs[ir].T) {
        regs[ir].T->forward(CMap(x.data(), cols), Fx);
        if (opts.ɑ > 0.f) {
          w.device(dev) = u[ir] + (opts.ɑ * Fx + (1.f - opts.ɑ) * z[ir]);
          regs[ir].P->apply(1.f / ρ, w);
          u[ir].device(dev) = (u[ir] + (opts.ɑ * Fx + (1.f - opts.ɑ) * z[ir])) - w;
        } else {
          w.device(dev) = u[ir] + Fx;
          regs[ir].P->apply(1.f / ρ, w);
          u[ir].device(dev) = (u[ir] + Fx) - w;
        }
        if (debug_z) { debug_z(io, ir, CMap(Fx.data(), sz), CMap(w.data(), sz), CMap(u[ir].data(), sz)); }
        float const nFx = ParallelNorm(Fx);
        nz = ParallelNorm(w);
        nP = ParallelNorm(Fx - w);
        Fx.device(dev) = w - z[ir];
        z[ir].device(dev) = w;
        regs[ir].T->adjoint(CMap(u[ir].data(), sz), Fh);
        nu = ParallelNorm(Fh);
        regs[ir].T->adjoint(CMap(Fx.data(), sz), Fh);
        nD = ParallelNorm(Fh);
        Log::Print("ADMM", "Reg {:02d} |Fx| {:3.2E} |z| {:3.2E} |F'u| {:3.2E}", ir, nFx, nz, nu);
      } else {
        if (opts.ɑ > 0.f) {
          w.device(dev) = u[ir] + (opts.ɑ * x + (1.f - opts.ɑ) * z[ir]);
          regs[ir].P->apply(1.f / ρ, w);
          u[ir].device(dev) = (u[ir] + (opts.ɑ * x + (1.f - opts.ɑ) * z[ir])) - w;
        } else {
          w.device(dev) = u[ir] + x;
          regs[ir].P->apply(1.f / ρ, w);
          u[ir].device(dev) = (u[ir] + x) - w;
        }
        if (debug_z) { debug_z(io, ir, CMap(x.data(), sz), CMap(w.data(), sz), CMap(u[ir].data(), sz)); }
        nz = ParallelNorm(w);
        nu = ParallelNorm(u[ir]);
        nP = ParallelNorm(x - w);
        nD = ParallelNorm(w - z[ir]);
        z[ir].device(dev) = w;
        Log::Print("ADMM", "Reg {:02d} |z| {:3.2E} |F'u| {:3.2E}", ir, nz, nu);
      }
      normz += nz * nz;
      normu += nu * nu;
      pRes += nP * nP;
      dRes += nD * nD;
    }
    if (!opts.cg) { work.resize(0); }
    float const normx = ParallelNorm(x);
    normz = std::sqrt(normz);
    normu = std::sqrt(normu);
    pRes = std::sqrt(pRes) / std::max(normFx, normz);
    dRes = std::sqrt(dRes) / normu;

    Log::Print("ADMM", "{:02d} |x| {:3.2E} |z| {:3.2E} |F'u| {:3.2E} ρ {:3.2E} |Pr| {:3.2E} |Du| {:3.2E}", io, normx, normz,
               normu, ρ, pRes, dRes);

    if ((pRes < opts.ε) && (dRes < opts.ε)) {
      Log::Print("ADMM", "Primal and dual tolerances achieved, stopping");
      break;
    }
    if (opts.balance && io > 0) {
      // ADMM Penalty Parameter Selection by Residual Balancing, Wohlberg 2017
      float const ratio = std::sqrt(pRes / dRes);
      float const τ = (ratio < 1.f) ? std::max(1.f / opts.τmax, 1.f / ratio) : std::min(opts.τmax, ratio);
      if (pRes > opts.μ * dRes) {
        ρ *= τ;
        for (Index ir = 0; ir < R; ir++) {
          u[ir].device(dev) = u[ir] / τ;
        }
      } else if (dRes > opts.μ * pRes) {
        ρ /= τ;
        for (Index ir = 0; ir < R; ir++) {
          u[ir].device(dev) = u[ir] * τ;
        }
      }
    }
    if (Iterating::ShouldStop("ADMM")) { break; }
  }
  Iterating::Finished();
  return x;
}

} // namespace rl
