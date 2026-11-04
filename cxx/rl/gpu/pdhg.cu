#include "pdhg.hpp"

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace rl::gpu {

namespace {

void Check(cudaError_t const e, char const *what)
{
  if (e != cudaSuccess) { throw std::runtime_error(std::string("GPU PDHG ") + what + ": " + cudaGetErrorString(e)); }
}

constexpr int Threads = 256;
auto          Blocks(long long const n) -> unsigned { return unsigned((n + Threads - 1) / Threads); }

// Image geometry: element (i, j, k, b, t, c) at i + nx (j + ny k) + N (b + nB (t + nT c))
struct Vol
{
  int       nx, ny, nz, nB, nT;
  long long N, nI; // Voxels, elements of one image (N nB nT)

  __host__ __device__ auto off(int const i, int const j, int const k, int const bt, int const c) const -> long long
  {
    return i + (long long)nx * (j + (long long)ny * k) + N * (bt + (long long)nB * nT * c);
  }
};

auto MakeVol(PDHGSetup const &s) -> Vol
{
  Vol g;
  g.nx = int(s.nx), g.ny = int(s.ny), g.nz = int(s.nz), g.nB = int(s.nB), g.nT = int(s.nT);
  g.N = s.nx * s.ny * s.nz;
  g.nI = g.N * s.nB * s.nT;
  return g;
}

// Coefficients of the transforms, computed as in rl/op/grad.cpp (same float rounding)
struct Coef
{
  float g;      // Grad<5, 3>: (1 / sqrt(3)) / 2
  float t;      // Grad<5, 1>: 1 / 2
  float vd, vo; // GradVec<6, 3>: diagonal (1 / sqrt(12)) / 2, off-diagonal (1 / sqrt(12)) / sqrt(2) / 2
  float lap;    // IsoΔ3D: 1 / (48 sqrt(28.3)), times -200 (centre), 20 (faces), 6 (edges), 1 (corners)
};

auto MakeCoef() -> Coef
{
  float const gs = 1.f / std::sqrt(3);
  float const vs = 1.f / std::sqrt(2 * 3 * (3 + 1) / 2);
  float const vs2 = vs / std::sqrt(2);
  float const ls = 1.f / (48.f * std::sqrt(28.3));
  return Coef{gs / 2.f, 1.f / 2.f, vs / 2.f, vs2 / 2.f, ls};
}

__device__ inline auto Add(float2 const a, float2 const b) -> float2 { return make_float2(a.x + b.x, a.y + b.y); }
__device__ inline auto Sub(float2 const a, float2 const b) -> float2 { return make_float2(a.x - b.x, a.y - b.y); }
__device__ inline auto Scale(float2 const a, float const s) -> float2 { return make_float2(a.x * s, a.y * s); }
__device__ inline auto Nxt(int const i, int const n) -> int { return i + 1 == n ? 0 : i + 1; }

// Adds v of every active thread to *acc: one atomic per group of coalesced threads
__device__ inline void AtomicSum(double *acc, double const v)
{
  namespace cg = cooperative_groups;
  auto const   grp = cg::coalesced_threads();
  double const s = cg::reduce(grp, v, cg::plus<double>());
  if (grp.thread_rank() == 0) { atomicAdd(acc, s); }
}

// |(y_old - y_new) / σ + e|², the dual residual of one element (e = (T (x̂ - x)) of the element)
__device__ inline auto Res2(float2 const yo, float2 const yn, float const sigma, float2 const e) -> double
{
  float const rx = (yo.x - yn.x) / sigma + e.x, ry = (yo.y - yn.y) / sigma + e.y;
  return double(rx) * rx + double(ry) * ry;
}
__device__ inline auto Prv(int const i, int const n) -> int { return i == 0 ? n - 1 : i - 1; }

auto Components(Transform const T) -> int
{
  switch (T) {
  case Transform::Identity: return 1;
  case Transform::Grad: return 3;
  case Transform::GradT: return 1;
  case Transform::TGV1: return 3;
  case Transform::TGV2: return 6;
  case Transform::Lap: return 1;
  }
  return 1;
}

// Weight of a neighbour of the isotropic Laplacian (IsoΔ3D): -200 at the centre, 20 faces, 6 edges, 1 corners
__device__ inline auto LapW(int const di, int const dj, int const dk) -> float
{
  int const n = (di != 0) + (dj != 0) + (dk != 0);
  return n == 0 ? -200.f : (n == 1 ? 20.f : (n == 2 ? 6.f : 1.f));
}

/*
 * Element (i, j, k, b, t, comp) of T xh, xh = [x; v]. Circular boundaries. Grad: forward differences (x[i + e_d] - x[i]),
 * GradVec: backward differences (u[i] - u[i - e_d]) as rl/op/grad.cpp.
 */
template <Transform T>
__device__ inline auto Elem(Vol const &g, Coef const &c, float2 const *__restrict__ xh, int const i, int const j, int const k,
                            int const b, int const t, int const comp) -> float2
{
  int const bt = b + g.nB * t;
  if constexpr (T == Transform::Identity) {
    return xh[g.off(i, j, k, bt, 0)];
  } else if constexpr (T == Transform::Grad || T == Transform::TGV1) {
    float2 const x0 = xh[g.off(i, j, k, bt, 0)];
    float2 const xn = comp == 0 ? xh[g.off(Nxt(i, g.nx), j, k, bt, 0)]
                                : (comp == 1 ? xh[g.off(i, Nxt(j, g.ny), k, bt, 0)] : xh[g.off(i, j, Nxt(k, g.nz), bt, 0)]);
    float2 e = Scale(Sub(xn, x0), c.g);
    if constexpr (T == Transform::TGV1) { e = Sub(e, xh[g.nI + g.off(i, j, k, bt, comp)]); }
    return e;
  } else if constexpr (T == Transform::GradT) {
    float2 const x0 = xh[g.off(i, j, k, bt, 0)];
    float2 const xn = xh[g.off(i, j, k, b + g.nB * Nxt(t, g.nT), 0)];
    return Scale(Sub(xn, x0), c.t);
  } else if constexpr (T == Transform::Lap) { // Interior points only, as IsoΔ3D
    if (i < 1 || i > g.nx - 2 || j < 1 || j > g.ny - 2 || k < 1 || k > g.nz - 2) { return make_float2(0.f, 0.f); }
    float2 acc = make_float2(0.f, 0.f);
    for (int dk = -1; dk <= 1; dk++) {
      for (int dj = -1; dj <= 1; dj++) {
        for (int di = -1; di <= 1; di++) {
          float2 const u = xh[g.off(i + di, j + dj, k + dk, bt, 0)];
          float const  w = LapW(di, dj, dk);
          acc = make_float2(fmaf(w, u.x, acc.x), fmaf(w, u.y, acc.y));
        }
      }
    }
    return Scale(acc, c.lap);
  } else { // TGV2: v at xh + nI, components 0, 1, 2
    float2 const *v = xh + g.nI;
    auto const    B = [&](int const vc, int const d) {
      float2 const u0 = v[g.off(i, j, k, bt, vc)];
      float2 const um = d == 0 ? v[g.off(Prv(i, g.nx), j, k, bt, vc)]
                               : (d == 1 ? v[g.off(i, Prv(j, g.ny), k, bt, vc)] : v[g.off(i, j, Prv(k, g.nz), bt, vc)]);
      return Sub(u0, um);
    };
    switch (comp) {
    case 0: return Scale(B(0, 0), c.vd);
    case 1: return Scale(B(1, 1), c.vd);
    case 2: return Scale(B(2, 2), c.vd);
    case 3: return Add(Scale(B(1, 0), c.vo), Scale(B(0, 1), c.vo));
    case 4: return Add(Scale(B(2, 0), c.vo), Scale(B(0, 2), c.vo));
    default: return Add(Scale(B(2, 1), c.vo), Scale(B(1, 2), c.vo));
    }
  }
}

/*
 * y = conj prox(y + σ T xh), one thread per norm block: the elements that share the voxel and the non-normed coordinates
 * among (b, t, g); the normed ones (mask bits 3, 4, 5) are looped over. L1 and L1I: blocks of one element.
 */
template <Transform T, ProxKind P>
__global__ void DualK(Vol const g, Coef const c, float2 const *__restrict__ xh, float2 *__restrict__ y, float const sigma,
                      float const lambda, int const G, unsigned const mask, long long const nOuter, int const nInner,
                      float2 const *__restrict__ x, double *__restrict__ res)
{
  long long const o = blockIdx.x * (long long)blockDim.x + threadIdx.x;
  if (o >= nOuter) { return; }
  long long const vx = o % g.N;
  long long       r = o / g.N;
  int const       S[3] = {g.nB, g.nT, G};
  int             base[3] = {0, 0, 0};
  for (int d = 0; d < 3; d++) {
    if (!((mask >> (3 + d)) & 1u)) {
      base[d] = int(r % S[d]);
      r /= S[d];
    }
  }
  int const  i = int(vx % g.nx), j = int((vx / g.nx) % g.ny), k = int(vx / ((long long)g.nx * g.ny));
  auto const coords = [&](int q, int (&cc)[3]) {
    for (int d = 0; d < 3; d++) {
      if ((mask >> (3 + d)) & 1u) {
        cc[d] = q % S[d];
        q /= S[d];
      } else {
        cc[d] = base[d];
      }
    }
  };
  auto const u = [&](int const (&cc)[3], long long &at) {
    at = g.off(i, j, k, cc[0] + g.nB * cc[1], cc[2]);
    float2 const e = Elem<T>(g, c, xh, i, j, k, cc[0], cc[1], cc[2]);
    float2 const yo = y[at];
    return make_float2(fmaf(sigma, e.x, yo.x), fmaf(sigma, e.y, yo.y));
  };
  double     r2 = 0.;
  auto const write = [&](int const (&cc)[3], long long const at, float2 const yn) { // y = yn, and the residual
    if (res) {
      float2 const yo = y[at];
      float2 const e = Sub(Elem<T>(g, c, xh, i, j, k, cc[0], cc[1], cc[2]), Elem<T>(g, c, x, i, j, k, cc[0], cc[1], cc[2]));
      r2 += Res2(yo, yn, sigma, e);
    }
    y[at] = yn;
  };
  if constexpr (P == ProxKind::L2) {
    float n2 = 0.f;
    for (int q = 0; q < nInner; q++) {
      int       cc[3];
      long long at;
      coords(q, cc);
      float2 const uu = u(cc, at);
      n2 += uu.x * uu.x + uu.y * uu.y;
    }
    float const n = sqrtf(n2);
    float const f = n > lambda ? lambda / n : 1.f;
    for (int q = 0; q < nInner; q++) {
      int       cc[3];
      long long at;
      coords(q, cc);
      float2 const yn = Scale(u(cc, at), f);
      write(cc, at, yn);
    }
  } else {
    int       cc[3];
    long long at;
    coords(0, cc);
    float2 const uu = u(cc, at);
    if constexpr (P == ProxKind::L1) {
      write(cc, at, Scale(uu, lambda / fmaxf(hypotf(uu.x, uu.y), lambda)));
    } else { // L1I
      write(cc, at, make_float2(0.f, lambda * uu.y / fmaxf(fabsf(uu.y), lambda)));
    }
  }
  if (res) { AtomicSum(res, r2); }
}

// X += Tᴴ y, scaled by a on x and by av on v. One thread per element of x (Identity, Grad, GradT, TGV1), or per voxel and
// (b, t) for TGV2's v
template <Transform T>
__global__ void AdjK(Vol const g, Coef const c, float2 const *__restrict__ y, float2 *__restrict__ X, float const a,
                     float const av)
{
  long long const e = blockIdx.x * (long long)blockDim.x + threadIdx.x;
  if (e >= g.nI) { return; }
  if constexpr (T == Transform::Identity) {
    X[e] = make_float2(fmaf(a, y[e].x, X[e].x), fmaf(a, y[e].y, X[e].y));
    return;
  }
  long long const vx = e % g.N;
  int const       bt = int(e / g.N), b = bt % g.nB, t = bt / g.nB;
  int const       i = int(vx % g.nx), j = int((vx / g.nx) % g.ny), k = int(vx / ((long long)g.nx * g.ny));
  if constexpr (T == Transform::Grad || T == Transform::TGV1) {
    // ∇ᴴ y: Σ_d (y_d[i - e_d] - y_d[i])
    float2 s = Sub(y[g.off(Prv(i, g.nx), j, k, bt, 0)], y[g.off(i, j, k, bt, 0)]);
    s = Add(s, Sub(y[g.off(i, Prv(j, g.ny), k, bt, 1)], y[g.off(i, j, k, bt, 1)]));
    s = Add(s, Sub(y[g.off(i, j, Prv(k, g.nz), bt, 2)], y[g.off(i, j, k, bt, 2)]));
    float const ag = a * c.g;
    X[e] = make_float2(fmaf(ag, s.x, X[e].x), fmaf(ag, s.y, X[e].y));
    if constexpr (T == Transform::TGV1) { // v -= a y
      for (int d = 0; d < 3; d++) {
        long long const  at = g.off(i, j, k, bt, d);
        float2 const     yy = y[at];
        float2          &xv = X[g.nI + at];
        xv = make_float2(fmaf(-av, yy.x, xv.x), fmaf(-av, yy.y, xv.y));
      }
    }
  } else if constexpr (T == Transform::Lap) {
    // Transpose of the interior-only stencil: the contributions of the interior neighbours (also next to the boundary)
    float2 acc = make_float2(0.f, 0.f);
    for (int dk = -1; dk <= 1; dk++) {
      for (int dj = -1; dj <= 1; dj++) {
        for (int di = -1; di <= 1; di++) {
          int const ni = i - di, nj = j - dj, nk = k - dk;
          if (ni < 1 || ni > g.nx - 2 || nj < 1 || nj > g.ny - 2 || nk < 1 || nk > g.nz - 2) { continue; }
          float2 const u = y[g.off(ni, nj, nk, bt, 0)];
          float const  w = LapW(di, dj, dk);
          acc = make_float2(fmaf(w, u.x, acc.x), fmaf(w, u.y, acc.y));
        }
      }
    }
    float const al = a * c.lap;
    X[e] = make_float2(fmaf(al, acc.x, X[e].x), fmaf(al, acc.y, X[e].y));
  } else if constexpr (T == Transform::GradT) {
    float2 const s = Sub(y[g.off(i, j, k, b + g.nB * Prv(t, g.nT), 0)], y[g.off(i, j, k, bt, 0)]);
    float const  at = a * c.t;
    X[e] = make_float2(fmaf(at, s.x, X[e].x), fmaf(at, s.y, X[e].y));
  } else { // TGV2: v += a GradVecᴴ y, with A_d(u) = u[i] - u[i + e_d]
    auto const A = [&](int const yc, int const d) {
      float2 const u0 = y[g.off(i, j, k, bt, yc)];
      float2 const un = d == 0 ? y[g.off(Nxt(i, g.nx), j, k, bt, yc)]
                               : (d == 1 ? y[g.off(i, Nxt(j, g.ny), k, bt, yc)] : y[g.off(i, j, Nxt(k, g.nz), bt, yc)]);
      return Sub(u0, un);
    };
    float2 const v0 = Add(Scale(A(0, 0), c.vd), Add(Scale(A(3, 1), c.vo), Scale(A(4, 2), c.vo)));
    float2 const v1 = Add(Scale(A(3, 0), c.vo), Add(Scale(A(1, 1), c.vd), Scale(A(5, 2), c.vo)));
    float2 const v2 = Add(Scale(A(4, 0), c.vo), Add(Scale(A(5, 1), c.vo), Scale(A(2, 2), c.vd)));
    float2 *V = X + g.nI;
    for (int d = 0; d < 3; d++) {
      float2 const    w = d == 0 ? v0 : (d == 1 ? v1 : v2);
      long long const at = g.off(i, j, k, bt, d);
      V[at] = make_float2(fmaf(av, w.x, V[at].x), fmaf(av, w.y, V[at].y));
    }
  }
}

/*
 * LLR, conjugate prox of λ Σ |patch|_* for the dual u = y + σ xh: per window (CPU rl/patches.cpp: windows of W³ voxels,
 * shifted, each in a patch of P³ voxels clamped to the volume), the patch matrix (P³ rows, nB nT columns) is decomposed
 * by one-sided (Hestenes) Jacobi, its singular values clipped at λ, and the window's rows written to z. One warp per
 * window, the patch and V in shared memory. One-sided Jacobi works on the patch itself, as accurate as the CPU's float SVD
 * (the Gram matrix would square the condition number).
 */
struct LLRGeo
{
  int P, W, inset, shift[3], nW[3], N, M; // N = nB nT columns, M = P³ rows
};

__device__ inline auto WarpSum(float v) -> float
{
  for (int o = 16; o > 0; o >>= 1) {
    v += __shfl_xor_sync(0xffffffffu, v, o);
  }
  return v;
}

__device__ inline auto WarpSumD(double v) -> double
{
  for (int o = 16; o > 0; o >>= 1) {
    v += __shfl_xor_sync(0xffffffffu, v, o);
  }
  return v;
}

__global__ void LLRK(Vol const g, LLRGeo const L, float2 const *__restrict__ xh, float2 const *__restrict__ y,
                     float2 *__restrict__ z, float const sigma, float const lambda, int const warps,
                     float2 const *__restrict__ x, double *__restrict__ res)
{
  extern __shared__ float2 sh[];
  int const       lane = threadIdx.x & 31, wib = threadIdx.x >> 5;
  long long const w = (long long)blockIdx.x * warps + wib;
  long long const nWin = (long long)L.nW[0] * L.nW[1] * L.nW[2];
  if (w >= nWin) { return; }
  float2 *A = sh + (long long)wib * (L.M * L.N + L.N * L.N + L.N); // Columns of the patch matrix, then V, then f
  float2 *V = A + L.M * L.N;
  float2 *F = V + L.N * L.N;
  int const wi[3] = {int(w % L.nW[0]), int((w / L.nW[0]) % L.nW[1]), int(w / ((long long)L.nW[0] * L.nW[1]))};
  int const dims[3] = {g.nx, g.ny, g.nz};
  int       stW[3], szW[3], stP[3], stW2[3];
  for (int d = 0; d < 3; d++) {
    int const st = (wi[d] - 1) * L.W + L.shift[d];
    stW[d] = max(st, 0);
    szW[d] = L.W + min(min(st, 0), dims[d] - stW[d] - L.W);
    if (szW[d] < 1) { return; } // Whole warp
    stP[d] = min(max(st - L.inset, 0), dims[d] - L.P);
    stW2[d] = stW[d] - stP[d];
  }
  // Load u = y + σ xh
  for (int q = lane; q < L.M * L.N; q += 32) {
    int const       r = q % L.M, col = q / L.M;
    int const       ri = r % L.P, rj = (r / L.P) % L.P, rk = r / (L.P * L.P);
    long long const at = g.off(stP[0] + ri, stP[1] + rj, stP[2] + rk, col, 0);
    float2 const    yo = y[at], xo = xh[at];
    A[q] = make_float2(fmaf(sigma, xo.x, yo.x), fmaf(sigma, xo.y, yo.y));
  }
  for (int q = lane; q < L.N * L.N; q += 32) {
    V[q] = make_float2((q % L.N) == (q / L.N) ? 1.f : 0.f, 0.f);
  }
  __syncwarp();
  // Sweeps over all column pairs until they are orthogonal to float precision
  for (int sweep = 0; sweep < 30; sweep++) {
    bool rotated = false;
    for (int ci = 0; ci < L.N - 1; ci++) {
      for (int cj = ci + 1; cj < L.N; cj++) {
        float2 *ai = A + (long long)ci * L.M, *aj = A + (long long)cj * L.M;
        float   al = 0.f, be = 0.f, gr = 0.f, gi = 0.f;
        for (int r = lane; r < L.M; r += 32) {
          float2 const p = ai[r], q = aj[r];
          al += p.x * p.x + p.y * p.y;
          be += q.x * q.x + q.y * q.y;
          gr += p.x * q.x + p.y * q.y; // conj(p) q
          gi += p.x * q.y - p.y * q.x;
        }
        al = WarpSum(al);
        be = WarpSum(be);
        gr = WarpSum(gr);
        gi = WarpSum(gi);
        float const gabs = hypotf(gr, gi);
        if (!(gabs > 1.e-7f * sqrtf(al * be)) || gabs == 0.f) { continue; } // Same decision in every lane
        rotated = true;
        float const  zeta = (be - al) / (2.f * gabs);
        float const  tt = copysignf(1.f, zeta) / (fabsf(zeta) + sqrtf(1.f + zeta * zeta));
        float const  cs = rsqrtf(1.f + tt * tt), sn = cs * tt;
        float2 const ph = make_float2(gr / gabs, -gi / gabs); // e^{-iφ}, γ = |γ| e^{iφ}
        for (int r = lane; r < L.M; r += 32) {
          float2 const p = ai[r], q = aj[r];
          float2 const bq = make_float2(ph.x * q.x - ph.y * q.y, ph.x * q.y + ph.y * q.x);
          ai[r] = make_float2(cs * p.x - sn * bq.x, cs * p.y - sn * bq.y);
          aj[r] = make_float2(sn * p.x + cs * bq.x, sn * p.y + cs * bq.y);
        }
        float2 *vi = V + (long long)ci * L.N, *vj = V + (long long)cj * L.N;
        for (int r = lane; r < L.N; r += 32) {
          float2 const p = vi[r], q = vj[r];
          float2 const bq = make_float2(ph.x * q.x - ph.y * q.y, ph.x * q.y + ph.y * q.x);
          vi[r] = make_float2(cs * p.x - sn * bq.x, cs * p.y - sn * bq.y);
          vj[r] = make_float2(sn * p.x + cs * bq.x, sn * p.y + cs * bq.y);
        }
        __syncwarp();
      }
    }
    if (!rotated) { break; }
  }
  // Singular values = column norms; clip at λ: the column scales f_k = min(1, λ / s_k), in shared memory (the output loop
  // below is divergent, so no shuffles there)
  for (int col = 0; col < L.N; col++) {
    float n2 = 0.f;
    for (int r = lane; r < L.M; r += 32) {
      float2 const p = A[(long long)col * L.M + r];
      n2 += p.x * p.x + p.y * p.y;
    }
    n2 = WarpSum(n2);
    float const s = sqrtf(n2);
    if (lane == 0) { F[col] = make_float2(s > lambda ? lambda / s : 1.f, 0.f); }
  }
  __syncwarp();
  // Window rows: z(row, j) = Σ_k A(row, k) f_k conj(V(j, k))
  int const nWv = szW[0] * szW[1] * szW[2];
  double    r2 = 0.;
  for (int q = lane; q < nWv * L.N; q += 32) {
    int const wv = q % nWv, jc = q / nWv;
    int const wx = wv % szW[0], wy = (wv / szW[0]) % szW[1], wz = wv / (szW[0] * szW[1]);
    int const r = (stW2[0] + wx) + L.P * ((stW2[1] + wy) + L.P * (stW2[2] + wz));
    float2    acc = make_float2(0.f, 0.f);
    for (int kc = 0; kc < L.N; kc++) {
      float const  f = F[kc].x;
      float2 const a = A[(long long)kc * L.M + r];
      float2 const v = V[(long long)kc * L.N + jc];
      // a f conj(v)
      acc.x += f * (a.x * v.x + a.y * v.y);
      acc.y += f * (a.y * v.x - a.x * v.y);
    }
    long long const at = g.off(stW[0] + wx, stW[1] + wy, stW[2] + wz, jc, 0);
    z[at] = acc;
    if (res) { r2 += Res2(y[at], acc, sigma, Sub(xh[at], x[at])); }
  }
  if (res) {
    r2 = WarpSumD(r2);
    if (lane == 0) { atomicAdd(res, r2); }
  }
}

/*
 * Wavelets as TOps::Wavelets: per transformed dimension, every line of L samples is transformed at sizes L, L/2, ... down
 * to the smallest size the CPU uses (halving while the half is even and > 4), each level on the first sz samples
 * (periodic Daubechies filters Cc / Cr, offset -N/2); the adjoint runs the levels the other way. One block per line, the
 * line in shared memory. The adjoint is the gather form of the CPU's scatter: sample p receives, for each tap k, the
 * pair ii with 2 ii - N/2 + k = p (mod sz).
 */
struct WaveCoef
{
  float c[8], r[8];
  int   n;
};

auto MakeWaveCoef(int const N) -> WaveCoef
{
  WaveCoef w{};
  w.n = N;
  float const c4[4] = {0.6830127f, 1.1830127f, 0.3169873f, -0.1830127f};
  float const c6[6] = {0.47046721f, 1.14111692f, 0.650365f, -0.19093442f, -0.12083221f, 0.0498175f};
  float const c8[8] = {0.32580343f, 1.01094572f, 0.89220014f, -0.03957503f, -0.26450717f, 0.0436163f, 0.0465036f, -0.01498699f};
  float const *src = N == 4 ? c4 : (N == 6 ? c6 : (N == 8 ? c8 : nullptr));
  if (!src) { throw std::runtime_error("GPU wavelets: width must be 4, 6 or 8"); }
  for (int i = 0; i < N; i++) {
    w.c[i] = src[i] / static_cast<float>(M_SQRT2);
  }
  float sign = 1.f;
  for (int i = 0; i < N; i++) {
    w.r[i] = sign * w.c[N - 1 - i];
    sign = -sign;
  }
  return w;
}

__device__ inline auto WrapI(int const i, int const n) -> int
{
  int const t = i + n;
  return t - n * (t / n);
}

__global__ void WaveK(float2 *__restrict__ x, long long const nLines, int const L, long long const stride,
                      long long const inner, long long const outerStride, int const minSz, bool const adjoint,
                      WaveCoef const w)
{
  extern __shared__ float2 wsh[];
  float2         *a = wsh, *b = wsh + L;
  long long const line = blockIdx.x;
  if (line >= nLines) { return; }
  // Line start: the dimensions before the transformed one (inner) and after it (outer)
  long long const base = (line % inner) + (line / inner) * outerStride;
  for (int p = threadIdx.x; p < L; p += blockDim.x) {
    a[p] = x[base + p * stride];
  }
  __syncthreads();
  int const off = -w.n / 2;
  auto      level = [&](int const sz) {
    if (sz < 4 || sz % 2) { return; } // As TOps::Wavelets::wav1
    int const h = sz / 2;
    if (!adjoint) {
      for (int ii = threadIdx.x; ii < h; ii += blockDim.x) {
        float2 lo = make_float2(0.f, 0.f), hi = make_float2(0.f, 0.f);
        for (int k = 0; k < w.n; k++) {
          float2 const v = a[WrapI(2 * ii + off + k, sz)];
          lo = make_float2(fmaf(w.c[k], v.x, lo.x), fmaf(w.c[k], v.y, lo.y));
          hi = make_float2(fmaf(w.r[k], v.x, hi.x), fmaf(w.r[k], v.y, hi.y));
        }
        b[ii] = lo;
        b[ii + h] = hi;
      }
    } else {
      for (int p = threadIdx.x; p < sz; p += blockDim.x) {
        float2 acc = make_float2(0.f, 0.f);
        for (int k = 0; k < w.n; k++) {
          int const q = WrapI(p - off - k, sz);
          if (q % 2) { continue; }
          int const    ii = q / 2;
          float2 const lo = a[ii], hi = a[ii + h];
          acc = make_float2(acc.x + w.c[k] * lo.x + w.r[k] * hi.x, acc.y + w.c[k] * lo.y + w.r[k] * hi.y);
        }
        b[p] = acc;
      }
    }
    __syncthreads();
    for (int p = threadIdx.x; p < sz; p += blockDim.x) {
      a[p] = b[p];
    }
    __syncthreads();
  };
  if (!adjoint) {
    for (int sz = L; sz >= minSz; sz /= 2) {
      level(sz);
    }
  } else {
    for (int sz = minSz; sz <= L; sz *= 2) {
      level(sz);
    }
  }
  for (int p = threadIdx.x; p < L; p += blockDim.x) {
    x[base + p * stride] = a[p];
  }
}

// y = y + s x (elementwise)
__global__ void AxpyK(float2 *__restrict__ y, float2 const *__restrict__ x, float const s, long long const n)
{
  long long const e = blockIdx.x * (long long)blockDim.x + threadIdx.x;
  if (e >= n) { return; }
  y[e] = make_float2(fmaf(s, x[e].x, y[e].x), fmaf(s, x[e].y, y[e].y));
}

// L1 conjugate prox: y = λ y / max(|y|, λ)
__global__ void ClipK(float2 *__restrict__ y, float const lambda, long long const n)
{
  long long const e = blockIdx.x * (long long)blockDim.x + threadIdx.x;
  if (e >= n) { return; }
  y[e] = Scale(y[e], lambda / fmaxf(hypotf(y[e].x, y[e].y), lambda));
}

// Dual residual of an identity-transform term: Σ |(yold - y) / σ + (xh - x)|²
__global__ void IdResK(float2 const *__restrict__ yold, float2 const *__restrict__ y, float2 const *__restrict__ xh,
                       float2 const *__restrict__ x, float const sigma, long long const n, double *__restrict__ res)
{
  long long const e = blockIdx.x * (long long)blockDim.x + threadIdx.x;
  if (e >= n) { return; }
  AtomicSum(res, Res2(yold[e], y[e], sigma, Sub(xh[e], x[e])));
}

auto MakeLLR(Term const &t, Vol const &g, int const shift[3]) -> LLRGeo
{
  LLRGeo L;
  L.P = t.llrPatch;
  L.W = t.llrWindow;
  L.inset = (L.P - L.W) / 2;
  int const dims[3] = {g.nx, g.ny, g.nz};
  for (int d = 0; d < 3; d++) {
    L.shift[d] = shift[d];
    L.nW[d] = dims[d] / L.W + 2;
    if (dims[d] < L.P) { throw std::runtime_error("GPU LLR: patch larger than the image"); }
  }
  L.N = g.nB * g.nT;
  L.M = L.P * L.P * L.P;
  if (L.N > 32) { throw std::runtime_error("GPU LLR: at most 32 images (basis x time) per patch"); }
  return L;
}

// Reductions in double: partial sums per block, added on the host
constexpr int RBlocks = 1024;

__global__ void ExtrapK(float2 *__restrict__ X, float2 *__restrict__ Xh, long long const n, long long const nx,
                        double *__restrict__ part)
{
  // Xh holds the previous X: Xh = 2 X - Xh; sums |X|², |X - Xprev|² and |X - Xprev|² of the first nx elements (x of [x; v])
  __shared__ double s0[Threads], s1[Threads], s2[Threads];
  double            a0 = 0., a1 = 0., a2 = 0.;
  for (long long e = blockIdx.x * (long long)blockDim.x + threadIdx.x; e < n; e += (long long)gridDim.x * blockDim.x) {
    float2 const x = X[e], p = Xh[e];
    float2 const d = make_float2(x.x - p.x, x.y - p.y);
    double const d2 = double(d.x) * d.x + double(d.y) * d.y;
    a0 += double(x.x) * x.x + double(x.y) * x.y;
    a1 += d2;
    if (e < nx) { a2 += d2; }
    Xh[e] = make_float2(2.f * x.x - p.x, 2.f * x.y - p.y);
  }
  s0[threadIdx.x] = a0;
  s1[threadIdx.x] = a1;
  s2[threadIdx.x] = a2;
  __syncthreads();
  for (int o = blockDim.x / 2; o > 0; o >>= 1) {
    if (threadIdx.x < o) {
      s0[threadIdx.x] += s0[threadIdx.x + o];
      s1[threadIdx.x] += s1[threadIdx.x + o];
      s2[threadIdx.x] += s2[threadIdx.x + o];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    part[3 * blockIdx.x] = s0[0];
    part[3 * blockIdx.x + 1] = s1[0];
    part[3 * blockIdx.x + 2] = s2[0];
  }
}

// Re(aᴴ b) (b == a: |a|²)
__global__ void DotK(float2 const *__restrict__ a, float2 const *__restrict__ b, long long const n, double *__restrict__ part)
{
  __shared__ double s0[Threads];
  double            acc = 0.;
  for (long long e = blockIdx.x * (long long)blockDim.x + threadIdx.x; e < n; e += (long long)gridDim.x * blockDim.x) {
    acc += double(a[e].x) * b[e].x + double(a[e].y) * b[e].y;
  }
  s0[threadIdx.x] = acc;
  __syncthreads();
  for (int o = blockDim.x / 2; o > 0; o >>= 1) {
    if (threadIdx.x < o) { s0[threadIdx.x] += s0[threadIdx.x + o]; }
    __syncthreads();
  }
  if (threadIdx.x == 0) { part[blockIdx.x] = s0[0]; }
}

// w = w - a v - b u (Lanczos)
__global__ void Lanczos3K(float2 *__restrict__ w, float2 const *__restrict__ v, float2 const *__restrict__ u, float const a,
                          float const b, long long const n)
{
  long long const e = blockIdx.x * (long long)blockDim.x + threadIdx.x;
  if (e >= n) { return; }
  w[e] = make_float2(w[e].x - a * v[e].x - b * u[e].x, w[e].y - a * v[e].y - b * u[e].y);
}

__global__ void ScaleK(float2 *__restrict__ dst, float2 const *__restrict__ src, float const s, long long const n)
{
  long long const e = blockIdx.x * (long long)blockDim.x + threadIdx.x;
  if (e >= n) { return; }
  dst[e] = make_float2(src[e].x * s, src[e].y * s);
}

struct Reducer
{
  DeviceBuffer        part{3 * RBlocks * sizeof(double)};
  std::vector<double> h = std::vector<double>(3 * RBlocks);

  auto dot(float2 const *a, float2 const *b, long long const n) -> double
  {
    DotK<<<RBlocks, Threads>>>(a, b, n, part.as<double>());
    Check(cudaGetLastError(), "dot");
    part.download(h.data(), RBlocks * sizeof(double));
    double s = 0.;
    for (int i = 0; i < RBlocks; i++) {
      s += h[i];
    }
    return s;
  }

  // Extrapolation, returns |X|², |ΔX|² and |ΔX|² of the first nx elements
  auto extrapolate(float2 *X, float2 *Xh, long long const n, long long const nx) -> std::tuple<double, double, double>
  {
    ExtrapK<<<RBlocks, Threads>>>(X, Xh, n, nx, part.as<double>());
    Check(cudaGetLastError(), "extrapolate");
    part.download(h.data(), 3 * RBlocks * sizeof(double));
    double a = 0., b = 0., c = 0.;
    for (int i = 0; i < RBlocks; i++) {
      a += h[3 * i];
      b += h[3 * i + 1];
      c += h[3 * i + 2];
    }
    return {a, b, c};
  }
};

// Largest eigenvalue of a symmetric tridiagonal matrix (diagonal a, off-diagonal b) by bisection with Sturm counts
auto TridiagMax(std::vector<double> const &a, std::vector<double> const &b) -> double
{
  int const n = int(a.size());
  double    lo = a[0], hi = a[0];
  for (int i = 0; i < n; i++) {
    double const r = (i > 0 ? std::abs(b[i - 1]) : 0.) + (i + 1 < n ? std::abs(b[i]) : 0.);
    lo = std::min(lo, a[i] - r);
    hi = std::max(hi, a[i] + r);
  }
  auto const countBelow = [&](double const x) { // Eigenvalues < x
    int    c = 0;
    double d = 1.;
    for (int i = 0; i < n; i++) {
      d = (a[i] - x) - (i > 0 ? b[i - 1] * b[i - 1] / d : 0.);
      if (d == 0.) { d = -1.e-300; }
      if (d < 0.) { c++; }
    }
    return c;
  };
  for (int it = 0; it < 200 && hi - lo > 1.e-12 * std::max(1., std::abs(hi)); it++) {
    double const m = 0.5 * (lo + hi);
    if (countBelow(m) == n) {
      hi = m;
    } else {
      lo = m;
    }
  }
  return hi;
}

// λmax(Aᴴ W A) of one time point by Lanczos on the device
auto LanczosMaxD(SNUFFT const &A, Vol const &g, int const its, std::function<void(std::string const &)> const &log) -> float
{
  long long const n = g.N * g.nB;
  DeviceBuffer    bv(n * sizeof(float2)), bu(n * sizeof(float2)), bw(n * sizeof(float2));
  {
    std::mt19937                          gen(42);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    std::vector<float2>                   h(n);
    for (auto &e : h) {
      e = make_float2(dist(gen), dist(gen));
    }
    bv.upload(h.data(), n * sizeof(float2));
  }
  bu.zero();
  Reducer     red;
  float const nv = float(std::sqrt(red.dot(bv.as<float2>(), bv.as<float2>(), n)));
  ScaleK<<<Blocks(n), Threads>>>(bv.as<float2>(), bv.as<float2>(), 1.f / nv, n);
  std::vector<double> al, be;
  double              b = 0., lam = 0.;
  for (int it = 0; it < its; it++) {
    float2 *v = bv.as<float2>(), *u = bu.as<float2>(), *w = bw.as<float2>();
    A.normalD(reinterpret_cast<Cxf const *>(v), reinterpret_cast<Cxf *>(w), 1.f);
    double const a = red.dot(v, w, n);
    Lanczos3K<<<Blocks(n), Threads>>>(w, v, u, float(a), float(b), n);
    al.push_back(a);
    b = std::sqrt(red.dot(w, w, n));
    lam = TridiagMax(al, be);
    if (b <= 1.e-6 * lam) { break; } // Invariant subspace
    be.push_back(b);
    std::swap(bu, bv); // u = previous v
    ScaleK<<<Blocks(n), Threads>>>(bv.as<float2>(), w, float(1. / b), n);
  }
  log("Lanczos λmax(AᴴWA) " + std::to_string(lam) + " after " + std::to_string(al.size()) + " iterations");
  return float(lam);
}

// Device time per phase of the iteration (CUDA events), for the log
struct PhaseTimer
{
  static constexpr int N = 7;
  cudaEvent_t          ev[N];
  double               ms[N - 1] = {};
  PhaseTimer()
  {
    for (auto &e : ev) {
      Check(cudaEventCreate(&e), "event");
    }
  }
  ~PhaseTimer()
  {
    for (auto &e : ev) {
      cudaEventDestroy(e);
    }
  }
  void mark(int const i) { Check(cudaEventRecord(ev[i]), "event record"); }
  void add() // After the last mark of an iteration
  {
    Check(cudaEventSynchronize(ev[N - 1]), "event sync");
    for (int i = 0; i < N - 1; i++) {
      float t = 0.f;
      Check(cudaEventElapsedTime(&t, ev[i], ev[i + 1]), "event elapsed");
      ms[i] += t;
    }
  }
};

/*
 * Step sizes: the base steps σ0 of each dual block (data, then the terms) and τ0 of the primal blocks x and v; the ratio R
 * scales them as σ = σ0 / √R, τ = τ0 √R, which keeps the convergence condition |Σ^½ K T^½| <= 1 and sets τ/σ.
 * Scalar (Chambolle & Pock 2011): σ0 = τ0 = 1/√L for every block, L >= λmax(KᴴΣ̃K).
 * Per term (Pock & Chambolle 2011, "Diagonal preconditioning for first order primal-dual algorithms in convex
 * optimization", their α = 1 rule applied to blocks): with n_ij an upper bound of the norm of K's block (dual block i,
 * primal block j), σ0_i = 1 / Σ_j n_ij and τ0_j = 1 / Σ_i n_ij. The matrix of the n_ij bounds |K|, and by the Schur test
 * these steps make |diag(√σ0) n diag(√τ0)| <= 1, so |Σ0^½ K T0^½| <= 1 without any L. Blocks: the data term
 * (n = √λmax(AᴴWA) on x; its dual step already contains W), each term (normX on x, normV on v).
 */
struct Steps
{
  float              sD0, tx0, tv0;
  std::vector<float> s0;
  float              sD, tx, tv;
  std::vector<float> s;

  void scale(float const R)
  {
    float const r = std::sqrt(R);
    sD = sD0 / r;
    tx = tx0 * r;
    tv = tv0 * r;
    s.resize(s0.size());
    for (size_t i = 0; i < s0.size(); i++) {
      s[i] = s0[i] / r;
    }
  }
};

auto MakeSteps(PDHGSetup const &s, float const lamE, float const L) -> Steps
{
  Steps st;
  if (s.termSteps) {
    float const nD = std::sqrt(lamE);
    float       cx = nD, cv = 0.f;
    for (auto const &t : s.terms) {
      st.s0.push_back(1.f / (t.normX + t.normV));
      cx += t.normX;
      cv += t.normV;
    }
    st.sD0 = 1.f / nD;
    st.tx0 = 1.f / cx;
    st.tv0 = cv > 0.f ? 1.f / cv : st.tx0;
  } else {
    float const b = 1.f / std::sqrt(L);
    st.sD0 = st.tx0 = st.tv0 = b;
    st.s0.assign(s.terms.size(), b);
  }
  return st;
}

auto GB(size_t const b) -> std::string
{
  char s[32];
  std::snprintf(s, sizeof(s), "%.2f GB", double(b) / 1.e9);
  return s;
}

// Upper bound of |y|² for the dual of a term: the conjugate prox projects onto a set of radius λ per block (L1, L1I: per
// element; L2: per norm block; LLR: spectral norm λ per window, so |.|_F² <= λ² min(rows, columns) of its window block)
auto DualBound2(Term const &t, PDHGSetup const &s) -> double
{
  int const       G = Components(t.T);
  long long const N = s.nx * s.ny * s.nz;
  double const    l2 = double(t.lambda) * t.lambda;
  if (t.P == ProxKind::L2) {
    int const S[3] = {int(s.nB), int(s.nT), G};
    double    blocks = double(N);
    for (int d = 0; d < 3; d++) {
      if (!((t.normMask >> (3 + d)) & 1u)) { blocks *= S[d]; }
    }
    return l2 * blocks;
  } else if (t.P == ProxKind::LLR) {
    double const W3 = double(t.llrWindow) * t.llrWindow * t.llrWindow;
    return l2 * std::min(W3, double(s.nB * s.nT)) * (double(N) / W3);
  }
  return l2 * double(N * s.nB * s.nT * G);
}

/*
 * Scales of the residuals (--pdhg-check), fixed by the data so that one tolerance fits different problems (as PDLP's
 * relative residuals): |AᴴWb| for the primal residual |Kᴴy| (the size of the data term's pull on x at x = 0) and |b|_W for
 * the dual residual (in the metric of the data term). Uses X and the data dual as scratch and leaves them zero.
 */
auto ResidualScales(SNUFFT const &A, PDHGSetup const &s, Vol const &g, DeviceBuffer &bX, DeviceBuffer &bYd,
                    DeviceBuffer const &bB, long long const nCK, Reducer &red) -> std::pair<double, double>
{
  long long const nX = g.N * g.nB;
  double          n1 = 0., bW = 0.;
  for (int64_t t = 0; t < s.nT; t++) {
    auto const X = bX.as<Cxf>() + t * nX;
    auto const Y = bYd.as<Cxf>() + t * nCK;
    auto const B = bB.as<Cxf>() + t * nCK;
    Check(cudaMemcpy(Y, B, nCK * sizeof(float2), cudaMemcpyDeviceToDevice), "copy");
    A.weightD(Y);
    A.adjointD(Y, X, 1.f, false);
    n1 += red.dot(reinterpret_cast<float2 const *>(X), reinterpret_cast<float2 const *>(X), nX);
    bW += A.weightedNorm2D(B);
  }
  bX.zero();
  bYd.zero();
  return {std::sqrt(n1), std::sqrt(bW)};
}

/*
 * Starting value of the automatic step ratio, from the data, before any iteration: the iterates estimate |x*| and |y*|
 * badly while the ratio is far off (a small ratio keeps x small, which pushes the estimate further down). |x*| from one
 * exact line-search step of the weighted least squares from 0, x̃ = α AᴴWb, α = |AᴴWb|² / <AᴴWb, AᴴWA AᴴWb>; the data
 * dual from its residual, |W(Ax̃ - b)|²_W⁻¹ = |b|²_W - α |AᴴWb|²; the terms' duals from their bounds (DualBound2).
 * Uses X, X̂ and the data dual as scratch and leaves them zero.
 */
auto DataRatio(SNUFFT const &A, PDHGSetup const &s, Vol const &g, Steps const &st, DeviceBuffer &bX, DeviceBuffer &bXh,
               DeviceBuffer &bYd, DeviceBuffer const &bB, long long const nCK, Reducer &red,
               std::function<void(std::string const &)> const &log) -> double
{
  long long const nX = g.N * g.nB;
  double          n1 = 0., q = 0., bW = 0.;
  for (int64_t t = 0; t < s.nT; t++) {
    auto const X = bX.as<Cxf>() + t * nX, Xh = bXh.as<Cxf>() + t * nX;
    auto const Y = bYd.as<Cxf>() + t * nCK;
    auto const B = bB.as<Cxf>() + t * nCK;
    Check(cudaMemcpy(Y, B, nCK * sizeof(float2), cudaMemcpyDeviceToDevice), "copy");
    A.weightD(Y);
    A.adjointD(Y, X, 1.f, false); // AᴴWb
    A.normalD(X, Xh, 1.f);        // AᴴWA AᴴWb
    auto const x = reinterpret_cast<float2 const *>(X);
    n1 += red.dot(x, x, nX);
    q += red.dot(x, reinterpret_cast<float2 const *>(Xh), nX);
    bW += A.weightedNorm2D(B);
  }
  double const alpha = n1 / q, x2 = alpha * alpha * n1, yd2 = std::max(bW - alpha * n1, 0.);
  double       yr2 = 0.;
  for (size_t ir = 0; ir < s.terms.size(); ir++) {
    yr2 += DualBound2(s.terms[ir], s) / st.s0[ir];
  }
  // The norms in the metrics of the base steps: |x|²/τ0, |y|²/σ0 (with only scalar steps, the same factor everywhere)
  double const r = (x2 / st.tx0) / (yd2 / st.sD0 + yr2);
  log("Step ratio from the data: |x̃|² " + std::to_string(x2) + ", |y_data|² " + std::to_string(yd2) +
      ", Σ |y_term|²/σ0 <= " + std::to_string(yr2) + ": R " + std::to_string(r));
  bX.zero();
  bXh.zero();
  bYd.zero();
  return r;
}

} // namespace

auto TermRows(Transform const T, int64_t const n) -> int64_t { return Components(T) * n; }

auto PDHGDeviceBytes(PDHGSetup const &s, int64_t const nC, int64_t const nK) -> size_t
{
  int64_t const nI = s.nx * s.ny * s.nz * s.nB * s.nT;
  int64_t const nP = s.tgv ? 4 * nI : nI;
  int64_t       n = 2 * nP + 2 * s.nT * nC * nK; // X, X̂; data dual and data
  bool          llr = false;
  for (auto const &t : s.terms) {
    n += TermRows(t.T, nI);
    llr = llr || t.P == ProxKind::LLR;
  }
  bool wave = false;
  for (auto const &t : s.terms) {
    wave = wave || t.P == ProxKind::Wavelets;
  }
  if (llr || (wave && s.checkEvery > 0)) { n += nI; } // Second dual buffer (LLR; wavelets with residuals)
  return size_t(n) * sizeof(float2) + 3 * RBlocks * sizeof(double);
}

void TermDual(Term const &t, PDHGSetup const &s, Cxf const *xh, Cxf *y, float const sigma, Cxf const *x, double *res)
{
  Vol const  g = MakeVol(s);
  Coef const c = MakeCoef();
  int const  G = Components(t.T);
  unsigned   mask = t.P == ProxKind::L2 ? t.normMask : 0u;
  int        S[3] = {int(s.nB), int(s.nT), G}, nInner = 1;
  long long  nOuter = g.N;
  for (int d = 0; d < 3; d++) {
    if ((mask >> (3 + d)) & 1u) {
      nInner *= S[d];
    } else {
      nOuter *= S[d];
    }
  }
  auto const X = reinterpret_cast<float2 const *>(xh);
  auto const Y = reinterpret_cast<float2 *>(y);
#define RL_DUAL(TT, PP)                                                                                                        \
  DualK<TT, PP><<<Blocks(nOuter), Threads>>>(g, c, X, Y, sigma, t.lambda, G, mask, nOuter, nInner,                         \
                                             reinterpret_cast<float2 const *>(x), res)
#define RL_DUAL_P(TT)                                                                                                          \
  switch (t.P) {                                                                                                               \
  case ProxKind::L1: RL_DUAL(TT, ProxKind::L1); break;                                                                       \
  case ProxKind::L1I: RL_DUAL(TT, ProxKind::L1I); break;                                                                     \
  case ProxKind::L2: RL_DUAL(TT, ProxKind::L2); break;                                                                       \
  default: throw std::runtime_error("GPU PDHG: LLR goes through LLRDual");                                                   \
  }
  switch (t.T) {
  case Transform::Identity: RL_DUAL_P(Transform::Identity); break;
  case Transform::Grad: RL_DUAL_P(Transform::Grad); break;
  case Transform::GradT: RL_DUAL_P(Transform::GradT); break;
  case Transform::TGV1: RL_DUAL_P(Transform::TGV1); break;
  case Transform::TGV2: RL_DUAL_P(Transform::TGV2); break;
  case Transform::Lap: RL_DUAL_P(Transform::Lap); break;
  }
#undef RL_DUAL_P
#undef RL_DUAL
  Check(cudaGetLastError(), "term dual");
}

void LLRDual(Term const &t, PDHGSetup const &s, Cxf const *xh, Cxf const *y, Cxf *z, float const sigma, int const shift[3],
             Cxf const *x, double *res)
{
  if (t.T != Transform::Identity) { throw std::runtime_error("GPU PDHG: LLR acts on the image (identity transform)"); }
  Vol const    g = MakeVol(s);
  LLRGeo const L = MakeLLR(t, g, shift);
  size_t const perWarp = size_t(L.M * L.N + L.N * L.N + L.N) * sizeof(float2);
  int const    warps = int(std::min<size_t>(4, (48 * 1024) / perWarp));
  if (warps < 1) { throw std::runtime_error("GPU LLR: patch x images too large for shared memory"); }
  long long const nWin = (long long)L.nW[0] * L.nW[1] * L.nW[2];
  unsigned const  nb = unsigned((nWin + warps - 1) / warps);
  LLRK<<<nb, 32 * warps, warps * perWarp>>>(g, L, reinterpret_cast<float2 const *>(xh), reinterpret_cast<float2 const *>(y),
                                            reinterpret_cast<float2 *>(z), sigma, t.lambda, warps,
                                            reinterpret_cast<float2 const *>(x), res);
  Check(cudaGetLastError(), "LLR");
}

void WaveletTransform(Term const &t, PDHGSetup const &s, Cxf *y, bool const adjoint)
{
  long long const sh[5] = {s.nx, s.ny, s.nz, s.nB, s.nT};
  WaveCoef const  w = MakeWaveCoef(t.waveWidth);
  long long const n = sh[0] * sh[1] * sh[2] * sh[3] * sh[4];
  for (int d = 0; d < 5; d++) {
    if (!((t.waveDims >> d) & 1u)) { continue; }
    int const L = int(sh[d]);
    if (L % 2) { throw std::runtime_error("GPU wavelets: transformed dimensions must be even (as TOps::Wavelets)"); }
    int minSz = L;
    while ((minSz / 2) % 2 == 0 && minSz > 4) {
      minSz /= 2;
    }
    long long inner = 1;
    for (int e = 0; e < d; e++) {
      inner *= sh[e];
    }
    long long const nLines = n / L;
    int const       threads = std::min(256, ((L + 31) / 32) * 32);
    WaveK<<<unsigned(nLines), threads, 2 * L * sizeof(float2)>>>(reinterpret_cast<float2 *>(y), nLines, L, inner, inner,
                                                                 inner * L, minSz, adjoint, w);
    Check(cudaGetLastError(), "wavelets");
  }
}

void WaveletDual(Term const &t, PDHGSetup const &s, Cxf const *xh, Cxf *y, float const sigma, Cxf const *x,
                 Cxf const *yold, double *res)
{
  if (t.T != Transform::Identity) { throw std::runtime_error("GPU PDHG: wavelets act on the image (identity transform)"); }
  long long const n = s.nx * s.ny * s.nz * s.nB * s.nT;
  auto const      Y = reinterpret_cast<float2 *>(y);
  AxpyK<<<Blocks(n), Threads>>>(Y, reinterpret_cast<float2 const *>(xh), sigma, n);
  WaveletTransform(t, s, y, false);
  ClipK<<<Blocks(n), Threads>>>(Y, t.lambda, n);
  WaveletTransform(t, s, y, true);
  if (res) {
    IdResK<<<Blocks(n), Threads>>>(reinterpret_cast<float2 const *>(yold), Y, reinterpret_cast<float2 const *>(xh),
                                   reinterpret_cast<float2 const *>(x), sigma, n, res);
  }
  Check(cudaGetLastError(), "wavelet dual");
}

void TermAdjoint(Term const &t, PDHGSetup const &s, Cxf const *y, Cxf *X, float const a, float const av)
{
  Vol const  g = MakeVol(s);
  Coef const c = MakeCoef();
  auto const Y = reinterpret_cast<float2 const *>(y);
  auto const XX = reinterpret_cast<float2 *>(X);
  switch (t.T) {
  case Transform::Identity: AdjK<Transform::Identity><<<Blocks(g.nI), Threads>>>(g, c, Y, XX, a, av); break;
  case Transform::Grad: AdjK<Transform::Grad><<<Blocks(g.nI), Threads>>>(g, c, Y, XX, a, av); break;
  case Transform::GradT: AdjK<Transform::GradT><<<Blocks(g.nI), Threads>>>(g, c, Y, XX, a, av); break;
  case Transform::TGV1: AdjK<Transform::TGV1><<<Blocks(g.nI), Threads>>>(g, c, Y, XX, a, av); break;
  case Transform::TGV2: AdjK<Transform::TGV2><<<Blocks(g.nI), Threads>>>(g, c, Y, XX, a, av); break;
  case Transform::Lap: AdjK<Transform::Lap><<<Blocks(g.nI), Threads>>>(g, c, Y, XX, a, av); break;
  }
  Check(cudaGetLastError(), "term adjoint");
}

void RunPDHG(SNUFFT const                                    &A,
             PDHGSetup const                                 &s,
             std::vector<Cxf const *> const                  &b,
             Cxf                                             *x,
             std::function<bool(PDHGStatus const &)> const    &status,
             std::function<void(int64_t, Cxf const *)> const &debug,
             int64_t const                                    debugEvery,
             std::function<void(std::string const &)> const  &log)
{
  Vol const       g = MakeVol(s);
  long long const nP = s.tgv ? 4 * g.nI : g.nI; // [x; v]
  long long const nX = g.N * g.nB;              // One time point of x
  long long const nCK = A.nC() * A.nK();        // One time point of k-space, device order
  if ((long long)b.size() != s.nT) { throw std::runtime_error("GPU PDHG: one data set per time point"); }
  bool llr = false, wave = false;
  for (auto const &t : s.terms) {
    if (!s.tgv && (t.T == Transform::TGV1 || t.T == Transform::TGV2)) {
      throw std::runtime_error("GPU PDHG: TGV terms need the [x; v] unknowns");
    }
    if (t.P == ProxKind::LLR && t.T != Transform::Identity) { throw std::runtime_error("GPU PDHG: LLR on the image only"); }
    llr = llr || t.P == ProxKind::LLR;
    wave = wave || t.P == ProxKind::Wavelets;
    if (t.P == ProxKind::Wavelets && t.T != Transform::Identity) {
      throw std::runtime_error("GPU PDHG: wavelets on the image only");
    }
  }

  // Step sizes σ = τ = 1/√L, L >= λmax(Kᴴ diag(W, I, ...) K) for K = [A; T_1; ...]
  float const lamE = s.lambdaE > 0.f ? s.lambdaE : 1.02f * LanczosMaxD(A, g, s.lanczosIts, log);
  float       L = lamE;
  for (auto const &t : s.terms) {
    L += t.norm2;
  }
  float ratio = s.ratio;
  Steps st = MakeSteps(s, lamE, L);
  st.scale(ratio);
  log("λmax(AᴴWA) " + std::to_string(lamE) + (s.lambdaE > 0.f ? " (given)" : " (estimated, x 1.02)") + ", L " +
      std::to_string(L) + (s.termSteps ? ", per-term steps" : "") + ", R " + std::to_string(ratio) + ": σ_data " +
      std::to_string(st.sD) + ", τ_x " + std::to_string(st.tx) + (s.tgv ? ", τ_v " + std::to_string(st.tv) : std::string()));
  for (size_t ir = 0; ir < s.terms.size(); ir++) {
    log("  term " + std::to_string(ir) + ": σ " + std::to_string(st.s[ir]));
  }

  size_t const need = PDHGDeviceBytes(s, A.nC(), A.nK()), avail = DeviceFreeBytes();
  log("Device memory for the iterations " + GB(need) + " (free " + GB(avail) + ")");
  if (need + (size_t(256) << 20) > avail) {
    throw std::runtime_error("GPU PDHG needs " + GB(need) + " of device memory for its vectors, " + GB(avail) +
                             " are free. Solving the time points one at a time (--split-time), a smaller basis or fewer "
                             "regularizer terms need less.");
  }
  DeviceBuffer bX(nP * sizeof(float2)), bXh(nP * sizeof(float2));
  DeviceBuffer bYd(s.nT * nCK * sizeof(float2)), bB(s.nT * nCK * sizeof(float2));
  // Second dual buffer: LLR writes its result there (the patches overlap); wavelets keep the old dual there for the residual
  DeviceBuffer bZ((llr || (wave && s.checkEvery > 0)) ? g.nI * sizeof(float2) : 0);
  std::vector<DeviceBuffer> bY;
  for (auto const &t : s.terms) {
    bY.emplace_back(TermRows(t.T, g.nI) * sizeof(float2));
    bY.back().zero();
  }
  bX.zero();
  bXh.zero();
  bYd.zero();
  for (int64_t t = 0; t < s.nT; t++) {
    A.toDeviceOrder(b[t], bB.as<Cxf>() + t * nCK);
  }

  Reducer             red;
  PhaseTimer          tm;
  int64_t             nIts = 0, nextRatio = 50;
  if (s.autoRatio) {
    ratio = float(DataRatio(A, s, g, st, bX, bXh, bYd, bB, nCK, red, log));
    st.scale(ratio);
  }
  /* Residuals (checkEvery > 0), for stopping. Primal: the optimality condition for x is Kᴴy = 0 (no prox on x), and
   * Kᴴy_k = T⁻¹(x_{k-1} - x_k) from the primal step itself: free. Dual: (y_{k-1} - y_k) Σ⁻¹ + K(x̂_{k-1} - x_{k-1}) is how
   * far y_k is from the dual optimality condition at x_{k-1} (from the prox's own optimality condition), computed in the
   * dual steps of check iterations (one more forward operator for the data term). Goldstein et al. 2015 ("Adaptive
   * primal-dual splitting methods for statistical learning and image processing") define the same pair.
   */
  double       scaleP = 1., scaleD = 1.;
  DeviceBuffer bRes((1 + s.terms.size()) * sizeof(double)); // [0] data term, [1 + i] term i
  if (s.checkEvery > 0) {
    std::tie(scaleP, scaleD) = ResidualScales(A, s, g, bX, bYd, bB, nCK, red);
    log("Residuals every " + std::to_string(s.checkEvery) + " iterations, relative to |AᴴWb| " + std::to_string(scaleP) +
        " and |b|_W " + std::to_string(scaleD) + ", tolerance " + std::to_string(s.tol));
  }
  std::mt19937        gen(std::random_device{}());
  std::vector<float2> hx;
  for (int64_t it = 0; it < s.maxIts; it++) {
    auto const X = bX.as<Cxf>(), Xh = bXh.as<Cxf>();
    bool const chk = s.checkEvery > 0 && it > 0 && it % s.checkEvery == 0;
    double    *res = chk ? bRes.as<double>() : nullptr;
    if (chk) { bRes.zero(); }
    // Dual steps at x̂: y = prox*_σ(y + σ K x̂)
    tm.mark(0);
    for (int64_t t = 0; t < s.nT; t++) {
      if (chk) {
        A.dataDualD(Xh + t * nX, bYd.as<Cxf>() + t * nCK, bB.as<Cxf>() + t * nCK, st.sD, s.data, X + t * nX, res);
      } else {
        A.dataDualD(Xh + t * nX, bYd.as<Cxf>() + t * nCK, bB.as<Cxf>() + t * nCK, st.sD, s.data);
      }
    }
    tm.mark(1);
    for (size_t ir = 0; ir < s.terms.size(); ir++) {
      auto const &t = s.terms[ir];
      if (t.P == ProxKind::LLR) {
        int shift[3] = {0, 0, 0};
        if (t.llrShift) {
          std::uniform_int_distribution<> d(0, t.llrWindow - 1);
          for (auto &e : shift) {
            e = d(gen);
          }
        }
        LLRDual(t, s, Xh, bY[ir].as<Cxf>(), bZ.as<Cxf>(), st.s[ir], shift, X, chk ? res + 1 + ir : nullptr);
        std::swap(bY[ir], bZ);
      } else if (t.P == ProxKind::Wavelets) {
        if (chk) { Check(cudaMemcpyAsync(bZ.as<Cxf>(), bY[ir].as<Cxf>(), g.nI * sizeof(float2), cudaMemcpyDeviceToDevice), "copy"); }
        WaveletDual(t, s, Xh, bY[ir].as<Cxf>(), st.s[ir], X, chk ? bZ.as<Cxf>() : nullptr, chk ? res + 1 + ir : nullptr);
      } else {
        TermDual(t, s, Xh, bY[ir].as<Cxf>(), st.s[ir], X, chk ? res + 1 + ir : nullptr);
      }
    }
    // Primal step x = x - τ Kᴴ y (G = 0), then x̂ = 2 x - x_old; x̂ holds x_old meanwhile
    tm.mark(2);
    Check(cudaMemcpyAsync(Xh, X, nP * sizeof(float2), cudaMemcpyDeviceToDevice), "copy");
    tm.mark(3);
    for (int64_t t = 0; t < s.nT; t++) {
      A.adjointD(bYd.as<Cxf>() + t * nCK, X + t * nX, -st.tx, true);
    }
    tm.mark(4);
    for (size_t ir = 0; ir < s.terms.size(); ir++) {
      TermAdjoint(s.terms[ir], s, bY[ir].as<Cxf>(), X, -st.tx, -st.tv);
    }
    tm.mark(5);
    auto const [nx2, ndx2, ndxx2] =
      red.extrapolate(reinterpret_cast<float2 *>(X), reinterpret_cast<float2 *>(Xh), nP, g.nI);
    tm.mark(6);
    tm.add();
    nIts++;
    PDHGStatus ps{it, float(std::sqrt(nx2)), float(std::sqrt(ndx2))};
    if (chk) {
      std::vector<double> r(1 + s.terms.size());
      bRes.download(r.data(), r.size() * sizeof(double));
      // |T⁻¹ Δx|: the step of the iteration that just ended (block-wise with per-term steps)
      double const p2 = ndxx2 / (double(st.tx) * st.tx) + std::max(ndx2 - ndxx2, 0.) / (double(st.tv) * st.tv);
      ps.primal = float(std::sqrt(p2) / scaleP);
      double d2 = 0.;
      for (double const e : r) {
        d2 += e;
      }
      ps.dual = float(std::sqrt(d2) / scaleD);
      ps.dualData = float(std::sqrt(r[0]) / scaleD);
      for (size_t ir = 0; ir < s.terms.size(); ir++) {
        ps.dualTerms.push_back(float(std::sqrt(r[1 + ir]) / scaleD));
      }
      ps.normxx = float(std::sqrt(red.dot(reinterpret_cast<float2 const *>(X), reinterpret_cast<float2 const *>(X), g.nI)));
      ps.normdxx = float(std::sqrt(ndxx2));
    }
    bool const go = status(ps);
    /* Automatic step ratio. The PDHG error after k iterations is bounded by (|x* - x0|² / τ + |y* - y0|²_Σ⁻¹) / k
     * (Chambolle & Pock 2011); with στ fixed that is smallest for τ / σ = (|x* - x0| / |y* - y0|)², the dual norm in the
     * metric of the dual steps (|y|² / W for the data term). It starts from the data (DataRatio); from x0 = y0 = 0 the
     * iterates then estimate the norms. As the "primal weight" of PDLP (Applegate et al. 2021) the ratio moves halfway
     * there in log space, here at iterations 50, 100, 200, ... and by at most a factor 2: few changes, so the steps
     * settle.
     */
    if (s.autoRatio && it + 1 == nextRatio) {
      nextRatio *= 2;
      double yd2 = 0., yr2 = 0.;
      for (int64_t t = 0; t < s.nT; t++) {
        yd2 += A.dualNorm2D(bYd.as<Cxf>() + t * nCK);
      }
      for (size_t ir = 0; ir < s.terms.size(); ir++) {
        auto const y = bY[ir].as<float2>();
        yr2 += red.dot(y, y, TermRows(s.terms[ir].T, g.nI)) / st.s0[ir];
      }
      // In the metrics of the base steps: |x|²/τ0_x + |v|²/τ0_v and |y|²/σ0
      auto const   Xp = reinterpret_cast<float2 const *>(X);
      double const xx2 = red.dot(Xp, Xp, g.nI), xv2 = s.tgv ? std::max(nx2 - xx2, 0.) : 0.;
      double const x2m = xx2 / st.tx0 + xv2 / st.tv0;
      double const y2 = yd2 / st.sD0 + yr2;
      if (y2 > 0. && x2m > 0.) {
        double const target = x2m / y2;
        float const  r = std::clamp(float(std::exp(0.5 * std::log(target) + 0.5 * std::log(double(ratio)))), ratio / 2.f,
                                    2.f * ratio);
        log("Step ratio τ/σ " + std::to_string(ratio) + " -> " + std::to_string(r) + " (target (|x|/|y|)² " +
            std::to_string(target) + ": |x|² " + std::to_string(x2m) + ", |y_data|² " + std::to_string(yd2 / st.sD0) +
            ", |y_terms|² " + std::to_string(yr2) + ", base metrics)");
        ratio = r;
        st.scale(ratio);
      }
    }
    if (debug && debugEvery > 0 && it % debugEvery == 0) {
      hx.resize(nP);
      bX.download(hx.data(), nP * sizeof(float2));
      debug(it, reinterpret_cast<Cxf const *>(hx.data()));
    }
    if (!go) { break; }
    if (s.checkEvery > 0) {
      if (chk && ps.primal < s.tol && ps.dual < s.tol) {
        log("Residual tolerance reached");
        break;
      }
    } else if (ps.normdx < s.tol * ps.normx) {
      log("Δ tolerance reached");
      break;
    }
  }
  if (nIts) {
    char m[256];
    std::snprintf(m, sizeof(m),
                  "Device time per iteration (ms): data dual %.0f, regularizer duals %.0f, copy %.0f, data adjoint %.0f, "
                  "regularizer adjoints %.0f, extrapolation %.0f",
                  tm.ms[0] / nIts, tm.ms[1] / nIts, tm.ms[2] / nIts, tm.ms[3] / nIts, tm.ms[4] / nIts, tm.ms[5] / nIts);
    log(m);
  }
  bX.download(x, g.nI * sizeof(float2));
}

} // namespace rl::gpu
