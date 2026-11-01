#pragma once

#include "snufft.hpp"

#include <functional>
#include <string>
#include <vector>

/*
 * PDHG with all of its vectors on a CUDA device (rl/gpu/pdhg.cu). Plain C++ interface, like snufft.hpp.
 *
 * min_x D(Ax - b) + Σ_i g_i(T_i x): the data term D through the operator's fused dual step (SNUFFT::dataDualD), the
 * regularizers g_i(T_i x) as the CPU's (riesling/regularizers.cpp): the same transforms (rl/op/grad.cpp, scaled the
 * same way, circular boundaries) and the same conjugate proxes (rl/prox). With TGV the unknowns are [x; v].
 *
 * Layout: x is (nx, ny, nz, nB, nT), the first dimension fastest; v and the transforms' outputs add a last dimension of
 * G components (6D: nx, ny, nz, nB, nT, G), as the CPU tensors.
 */
namespace rl::gpu {

enum class Transform : int
{
  Identity, // T x = x (G = 1)
  Grad,     // ∇x over the three spatial dimensions (TOps::Grad<5, 3>, G = 3)
  GradT,    // Difference along the time dimension (TOps::Grad<5, 1> on dimension 4, G = 1)
  TGV1,     // ∇x - v (G = 3)
  TGV2      // Symmetrized gradient of v (TOps::GradVec<6, 3>, G = 6)
};

enum class ProxKind : int
{
  L1,  // λ|·|₁ (Proxs::L1)
  L1I, // λ|Im ·|₁ (Proxs::L1I)
  L2,  // λ Σ_blocks |·|₂ over the dimensions in normMask (Proxs::L2)
  LLR  // λ Σ_patches |·|_* over (nB, nT) (Proxs::LLR, λ already scaled as there)
};

struct Term
{
  Transform T = Transform::Identity;
  ProxKind  P = ProxKind::L1;
  float     lambda = 0.f;
  unsigned  normMask = 0; // L2: bit d set = dimension d of the 6D shape is normed (only 3 = b, 4 = t, 5 = g)
  int       llrPatch = 5, llrWindow = 3;
  bool      llrShift = false; // Random window shift per application, as the CPU option
  float     norm2 = 1.f;      // Upper bound of λmax(T'T), for the step sizes
};

struct PDHGSetup
{
  int64_t           nx = 0, ny = 0, nz = 0, nB = 1, nT = 1;
  bool              tgv = false; // Unknowns [x; v], v with 3 components
  std::vector<Term> terms;
  DataDual          data = DataDual::LS;
  int64_t           maxIts = 100;
  float             tol = 1.e-4f;  // Stop when |Δx| / |x| < tol
  float             lambdaE = 0.f; // λmax(Aᴴ W A); <= 0: estimate it (Lanczos on the device, 2% margin)
  int               lanczosIts = 24;
};

struct PDHGStatus
{
  int64_t it;
  float   normx, normdx; // |[x; v]| and |Δ[x; v]| after the iteration
};

// Size of the transform's output for an image of n = nx ny nz nB nT elements
auto TermRows(Transform const T, int64_t const n) -> int64_t;
// Device memory the solver allocates (bytes), beyond the operator's
auto PDHGDeviceBytes(PDHGSetup const &s, int64_t nC, int64_t nK) -> size_t;

/*
 * Runs PDHG. A: the operator of one time point (nx, ny, nz, nB images, nC channels); b: the data of each time point in
 * the operator's host layout (nC, nSamp, nTrace). x: the result (nx, ny, nz, nB, nT, host). After every iteration
 * status() is called (logging, stopping: return false to stop) and, every debugEvery iterations, debug() with the host
 * copy of [x; v] (nullptr: none). log() gets the step sizes and the estimate of λE.
 */
void RunPDHG(SNUFFT const                                        &A,
             PDHGSetup const                                     &s,
             std::vector<Cxf const *> const                      &b,
             Cxf                                                 *x,
             std::function<bool(PDHGStatus const &)> const        &status,
             std::function<void(int64_t, Cxf const *)> const     &debug,
             int64_t                                              debugEvery,
             std::function<void(std::string const &)> const      &log);

/*
 * Building blocks, exposed for the tests (device pointers). xh: [x; v] (v only with TGV), y: the term's dual.
 * TermDual: y = conj prox_σ(y + σ T xh) for L1, L1I and L2 terms; LLRDual: the same for LLR terms (identity transform),
 * written to z (the patches overlap neighbouring windows, so not in place). TermAdjoint: X += a Tᴴ y.
 */
void TermDual(Term const &t, PDHGSetup const &s, Cxf const *xh, Cxf *y, float sigma);
void LLRDual(Term const &t, PDHGSetup const &s, Cxf const *xh, Cxf const *y, Cxf *z, float sigma, int const shift[3]);
void TermAdjoint(Term const &t, PDHGSetup const &s, Cxf const *y, Cxf *X, float a);

} // namespace rl::gpu
