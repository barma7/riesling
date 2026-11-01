#include "snufft.hpp"

#include <cuda_runtime.h>
#include <cufft.h>

#include <algorithm>
#include <cfenv>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace rl::gpu {

namespace {

void Check(cudaError_t const e, char const *what)
{
  if (e != cudaSuccess) { throw std::runtime_error(std::string("CUDA ") + what + ": " + cudaGetErrorString(e)); }
}

void Check(cufftResult const r, char const *what)
{
  if (r != CUFFT_SUCCESS) { throw std::runtime_error(std::string("cuFFT ") + what + ": error " + std::to_string(int(r))); }
}

template <typename T> struct DBuf
{
  T     *p = nullptr;
  size_t n = 0;
  DBuf() = default;
  DBuf(DBuf const &) = delete;
  DBuf &operator=(DBuf const &) = delete;
  ~DBuf() { release(); }
  void alloc(size_t const nn)
  {
    release();
    if (nn) { Check(cudaMalloc(&p, nn * sizeof(T)), "cudaMalloc"); }
    n = nn;
  }
  void upload(T const *h, size_t const nn)
  {
    alloc(nn);
    if (nn) { Check(cudaMemcpy(p, h, nn * sizeof(T), cudaMemcpyHostToDevice), "upload"); }
  }
  void release()
  {
    if (p) { cudaFree(p); }
    p = nullptr;
    n = 0;
  }
  auto bytes() const -> size_t { return n * sizeof(T); }
};

struct Geo
{
  int       nx, ny, nz, gx, gy, gz, px, py, pz;
  long long N, G; // Image voxels, grid points
  int       nB, nC, nMB;
  float     sigma; // Sign of the folded FFT shifts, (-1)^((gx + gy + gz) / 2)
  float     fs;    // Unitary FFT scale 1/sqrt(G)
};

struct Kern
{
  float beta, hw, L, scale; // f(z) = exp(beta sqrt(1 - z²) - 1), z = ((i + L) - offset) / hw; dimension 0 times scale
};

// Valid samples in device order (sorted by tile, then grid cell)
struct Samples
{
  short4 const *ki;   // Nearest grid point
  float4 const *off;  // Offset from it (x, y, z), w = k-space weight when the weights are shared by the channels
  int const    *idx;  // sample + nSamp * trace
  int const    *brow; // Offset of the basis entries, -1 without basis
  long long     n;
};

// Fused PDHG data-term dual step at the end of the interpolation (kind 0: plain output), see SNUFFT::dataDualD
struct Dual
{
  float2       *y = nullptr; // Dual variable, device order
  float2 const *b = nullptr; // Data, device order
  float         sigma = 0.f;
  int           kind = 0;
};

__device__ inline auto CMul(float2 const a, float2 const b) -> float2
{
  return make_float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

__device__ inline auto Wrap(int const i, int const n) -> int { return i < 0 ? i + n : (i >= n ? i - n : i); }

// 1D kernel weights with the checkerboard sign of the folded FFT shift: (-1)^i on tap i
template <int FW> __device__ inline void Weights1(float const o, Kern const k, float const scale, float (&w)[FW])
{
  if constexpr (FW == 1) {
    w[0] = scale;
  } else {
#pragma unroll
    for (int i = 0; i < FW; i++) {
      float const z = ((i + k.L) - o) / k.hw;
      float const z2 = z * z;
      float const f = z2 > 1.f ? 0.f : expf(k.beta * sqrtf(1.f - z2) - 1.f);
      w[i] = (i & 1) ? -(f * scale) : (f * scale);
    }
  }
}

// ws(g, b) = sigma (-1)^(gx+gy+gz) fs * ((x(v, b) * maps(v, b or 0, c)) * apo(v)) * s inside the image, 0 in the padding
__global__ void Expand(Geo const g,
                       float2 const *__restrict__ x,
                       float2 const *__restrict__ maps,
                       float const *__restrict__ apo,
                       float2 *__restrict__ ws,
                       int const   c,
                       float const s)
{
  long long const gi = blockIdx.x * (long long)blockDim.x + threadIdx.x;
  if (gi >= g.G) { return; }
  int const       ix = gi % g.gx;
  long long const r = gi / g.gx;
  int const       iy = r % g.gy, iz = r / g.gy;
  int const       i = ix - g.px, j = iy - g.py, k = iz - g.pz;
  if (i < 0 || i >= g.nx || j < 0 || j >= g.ny || k < 0 || k >= g.nz) {
    for (int b = 0; b < g.nB; b++) {
      ws[gi + g.G * b] = make_float2(0.f, 0.f);
    }
    return;
  }
  float const     sg = ((ix + iy + iz) & 1) ? -(g.sigma * g.fs) : (g.sigma * g.fs);
  long long const v = i + g.nx * (j + (long long)g.ny * k);
  float const     a = apo[v];
  for (int b = 0; b < g.nB; b++) {
    int const    mb = (g.nMB == 1) ? 0 : b;
    float2 const xv = x[v + g.N * b];
    float2 const m = maps[v + g.N * (mb + (long long)g.nMB * c)];
    float const  tr = fmaf(xv.x, m.x, -(xv.y * m.y));
    float const  ti = fmaf(xv.x, m.y, xv.y * m.x);
    ws[gi + g.G * b] = make_float2(((tr * a) * s) * sg, ((ti * a) * s) * sg);
  }
}

/*
 * x(v, b) (+)= sigma (-1)^(gx+gy+gz) fs * (ws(g, b) * apo(v)) * conj(maps(v, b or 0, c)), summed over the channels: the first
 * channel writes, the last multiplies by s. With accumulate, every channel adds s times its term (x += s Aᴴ y).
 */
__global__ void Collect(Geo const g,
                        float2 const *__restrict__ ws,
                        float2 const *__restrict__ maps,
                        float const *__restrict__ apo,
                        float2 *__restrict__ x,
                        int const   c,
                        float const s,
                        bool const  first,
                        bool const  last,
                        bool const  accumulate = false)
{
  long long const v = blockIdx.x * (long long)blockDim.x + threadIdx.x;
  if (v >= g.N) { return; }
  int const       i = v % g.nx;
  long long const r = v / g.nx;
  int const       j = r % g.ny, k = r / g.ny;
  int const       ix = i + g.px, iy = j + g.py, iz = k + g.pz;
  long long const gi = ix + g.gx * (iy + (long long)g.gy * iz);
  float const     sg = ((ix + iy + iz) & 1) ? -(g.sigma * g.fs) : (g.sigma * g.fs);
  float const     a = apo[v] * sg;
  for (int b = 0; b < g.nB; b++) {
    int const    mb = (g.nMB == 1) ? 0 : b;
    float2 const w = ws[gi + g.G * b];
    float2 const m = maps[v + g.N * (mb + (long long)g.nMB * c)];
    float const  yr = w.x * a, yi = w.y * a;
    float2       acc = make_float2(yr * m.x + yi * m.y, yi * m.x - yr * m.y);
    if (accumulate) {
      float2 const o = x[v + g.N * b];
      x[v + g.N * b] = make_float2(fmaf(s, acc.x, o.x), fmaf(s, acc.y, o.y));
      continue;
    }
    if (!first) {
      float2 const o = x[v + g.N * b];
      acc.x += o.x;
      acc.y += o.y;
    }
    if (last) {
      acc.x *= s;
      acc.y *= s;
    }
    x[v + g.N * b] = acc;
  }
}

/*
 * Interpolation (gridding forward) for channel c: y = Σ_b phi_b Σ_taps k * ws(tap, b), phi_b the scaled basis entries of the
 * sample. One thread per sample; for each tap the NBT basis images are read together. Output either to the full k-space,
 * out[c + nC * idx] (forward), or in device order times the k-space weight (normal operator).
 */
template <int FW, int NBT> __global__ void __launch_bounds__(256) Interp(Geo const g,
                                                                     Kern const k,
                                                                     Samples const smp,
                                                                     float2 const *__restrict__ ws,
                                                                     float2 const *__restrict__ basis,
                                                                     float const bscale,
                                                                     float const *__restrict__ wch, // Per-channel weights
                                                                     bool const  weighted,
                                                                     float2 *__restrict__ out,
                                                                     bool const  toFull,
                                                                     int const   c,
                                                                     Dual const  d)
{
  long long const is = blockIdx.x * (long long)blockDim.x + threadIdx.x;
  if (is >= smp.n) { return; }
  short4 const ki = smp.ki[is];
  float4 const of = smp.off[is];
  float        w0[FW], w1[FW], w2[FW];
  Weights1<FW>(of.x, k, k.scale, w0);
  Weights1<FW>(of.y, k, 1.f, w1);
  Weights1<FW>(of.z, k, 1.f, w2);
  // Parity of the first tap; the signs of the others are in the weights
  float const base = ((ki.x + ki.y + ki.z - 3 * (FW / 2)) & 1) ? -1.f : 1.f;
  int         tx[FW];
#pragma unroll
  for (int i = 0; i < FW; i++) {
    tx[i] = Wrap(ki.x + i - FW / 2, g.gx);
  }
  int const brow = smp.brow[is];
  float2    y = make_float2(0.f, 0.f);
  for (int b0 = 0; b0 < g.nB; b0 += NBT) {
    int const nb = min(NBT, g.nB - b0);
    float2    a[NBT];
#pragma unroll
    for (int b = 0; b < NBT; b++) {
      a[b] = make_float2(0.f, 0.f);
    }
    for (int i2 = 0; i2 < FW; i2++) {
      int const tz = Wrap(ki.z + i2 - FW / 2, g.gz);
      for (int i1 = 0; i1 < FW; i1++) {
        int const     ty = Wrap(ki.y + i1 - FW / 2, g.gy);
        float const   w12 = w1[i1] * w2[i2];
        float2 const *row = ws + g.G * b0 + g.gx * (ty + (long long)g.gy * tz);
#pragma unroll
        for (int i0 = 0; i0 < FW; i0++) {
          float const w = w0[i0] * w12;
#pragma unroll
          for (int b = 0; b < NBT; b++) {
            if (b < nb) {
              float2 const v = __ldg(row + g.G * b + tx[i0]);
              a[b].x = fmaf(w, v.x, a[b].x);
              a[b].y = fmaf(w, v.y, a[b].y);
            }
          }
        }
      }
    }
#pragma unroll
    for (int b = 0; b < NBT; b++) {
      if (b < nb) {
        float2 phi = make_float2(1.f, 0.f);
        if (brow >= 0) {
          float2 const e = basis[brow + b0 + b];
          phi = make_float2(e.x * bscale, e.y * bscale);
          if (phi.x == 0.f && phi.y == 0.f) { continue; }
        }
        float2 const p = CMul(a[b], phi);
        y.x += p.x;
        y.y += p.y;
      }
    }
  }
  y.x *= base;
  y.y *= base;
  if (d.kind) {
    long long const o = smp.n * c + is;
    float const     W = wch ? wch[o] : of.w;
    float2 const    yo = d.y[o], bo = d.b[o];
    float const     sw = d.sigma * W;
    float2          u = make_float2(fmaf(sw, y.x - bo.x, yo.x), fmaf(sw, y.y - bo.y, yo.y));
    float const     f = d.kind == 1 ? 1.f / (1.f + sw) : (d.kind == 2 ? 1.f / (1.f + d.sigma) : 1.f / fmaxf(hypotf(u.x, u.y), 1.f));
    d.y[o] = make_float2(u.x * f, u.y * f);
    return;
  }
  if (weighted) {
    float const W = wch ? wch[smp.n * c + is] : of.w;
    y.x *= W;
    y.y *= W;
  }
  if (toFull) {
    out[c + (long long)g.nC * smp.idx[is]] = y;
  } else {
    out[is] = y;
  }
}

template <int FW> void LaunchInterp(unsigned const nb, Geo const &g, Kern const &k, Samples const &smp, float2 const *ws,
                                    float2 const *basis, float const bscale, float const *wch, bool const weighted,
                                    float2 *out, bool const toFull, int const c, Dual const d)
{
  switch (std::min(g.nB, 8)) {
  case 1: Interp<FW, 1><<<nb, 256>>>(g, k, smp, ws, basis, bscale, wch, weighted, out, toFull, c, d); break;
  case 2: Interp<FW, 2><<<nb, 256>>>(g, k, smp, ws, basis, bscale, wch, weighted, out, toFull, c, d); break;
  case 3: Interp<FW, 3><<<nb, 256>>>(g, k, smp, ws, basis, bscale, wch, weighted, out, toFull, c, d); break;
  case 4: Interp<FW, 4><<<nb, 256>>>(g, k, smp, ws, basis, bscale, wch, weighted, out, toFull, c, d); break;
  case 5: Interp<FW, 5><<<nb, 256>>>(g, k, smp, ws, basis, bscale, wch, weighted, out, toFull, c, d); break;
  case 6: Interp<FW, 6><<<nb, 256>>>(g, k, smp, ws, basis, bscale, wch, weighted, out, toFull, c, d); break;
  default: Interp<FW, 8><<<nb, 256>>>(g, k, smp, ws, basis, bscale, wch, weighted, out, toFull, c, d); break;
  }
}

/*
 * Spreading (gridding adjoint) for channel c: ws(tap, b) += k * conj(phi_b) * y, bin by bin. The samples are sorted by
 * tile of TileSz³ grid cells, then by cell, and split into bins of at most MaxBin. One block per bin: the samples' 1D
 * weights and values are staged in shared memory with the first sample of each cell. Each thread then takes points of the
 * tile (plus the kernel's halo) and sums, in registers, the contributions of exactly the samples whose kernel reaches the
 * point: for each of the FW² rows of cells in reach one contiguous run of samples. The point is then added to the grid,
 * once per bin, the only atomics.
 *
 * Why not an atomic add per tap: a centre-out trajectory starts every spoke at the k-space centre, and the atomics there
 * (about 10⁶ per point, in sequence) were both slow and inaccurate (float sums of that length). Here a point sums at most
 * MaxBin terms per bin, then one partial sum per bin; the summation order within a bin is fixed. Shared-memory atomics are
 * emulated with compare-and-swap on this GPU generation (sm_75) and were 4x slower; per-point accumulators over tiles of
 * 4 x 4 x 2 points with warp-level culling of the samples were ~10% slower.
 */
constexpr int TileSz = 8; // At most 8: the cell inside the tile is part of the sort key (< 512)
constexpr int MaxBin = 256;

struct Bin
{
  int x0, y0, z0; // Tile origin (grid cell)
  int start, count;
};

template <int FW, int NBT> __global__ void __launch_bounds__(256) SpreadGather(Geo const g,
                                                                           Kern const k,
                                                                           Samples const smp,
                                                                           Bin const *__restrict__ bins,
                                                                           float2 const *__restrict__ in,
                                                                           bool const fromFull,
                                                                           float2 const *__restrict__ basis,
                                                                           float const bscale,
                                                                           float2 *__restrict__ ws,
                                                                           int const c,
                                                                           int const b0,
                                                                           int const nb)
{
  constexpr int T = TileSz, H = FW / 2, P = T + 2 * H, P3 = P * P * P, T3 = T * T * T;
  __shared__ short  scell[MaxBin]; // Cell of the sample inside the tile
  __shared__ char   scx[MaxBin];   // Its x
  __shared__ float  sw[MaxBin][3 * FW];
  __shared__ float2 sv[MaxBin][NBT];
  __shared__ short  cstart[T3 + 1]; // First sample of each cell

  Bin const bin = bins[blockIdx.x];
  int const n = bin.count;
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    long long const is = bin.start + i;
    short4 const    ki = smp.ki[is];
    float4 const    of = smp.off[is];
    float2 const    y0 = fromFull ? in[c + (long long)g.nC * smp.idx[is]] : in[is];
    float const     base = ((ki.x + ki.y + ki.z - 3 * (FW / 2)) & 1) ? -1.f : 1.f;
    float2 const    y = make_float2(y0.x * base, y0.y * base);
    scell[i] = short(((ki.z - bin.z0) * T + (ki.y - bin.y0)) * T + (ki.x - bin.x0));
    scx[i] = char(ki.x - bin.x0);
    float w[FW];
    Weights1<FW>(of.x, k, k.scale, w);
#pragma unroll
    for (int d = 0; d < FW; d++) {
      sw[i][d] = w[d];
    }
    Weights1<FW>(of.y, k, 1.f, w);
#pragma unroll
    for (int d = 0; d < FW; d++) {
      sw[i][FW + d] = w[d];
    }
    Weights1<FW>(of.z, k, 1.f, w);
#pragma unroll
    for (int d = 0; d < FW; d++) {
      sw[i][2 * FW + d] = w[d];
    }
    int const brow = smp.brow[is];
#pragma unroll
    for (int b = 0; b < NBT; b++) {
      float2 v = y;
      if (brow >= 0 && b < nb) {
        float2 const e = basis[brow + b0 + b];
        float2 const phi = make_float2(e.x * bscale, -(e.y * bscale)); // conj
        v = (phi.x == 0.f && phi.y == 0.f) ? make_float2(0.f, 0.f) : CMul(phi, y);
      }
      sv[i][b] = v;
    }
  }
  __syncthreads();
  for (int cc = threadIdx.x; cc <= T3; cc += blockDim.x) { // Lower bound of cell cc in the sorted cells
    int lo = 0, hi = n;
    while (lo < hi) {
      int const mid = (lo + hi) / 2;
      if (scell[mid] < cc) {
        lo = mid + 1;
      } else {
        hi = mid;
      }
    }
    cstart[cc] = short(lo);
  }
  __syncthreads();

  for (int pi = threadIdx.x; pi < P3; pi += blockDim.x) {
    int const px = pi % P, py = (pi / P) % P, pz = pi / (P * P);
    int const rxlo = max(0, px - FW + 1), rxhi = min(T - 1, px);
    if (rxlo > rxhi) { continue; }
    float2 acc[NBT], acc2[NBT];
#pragma unroll
    for (int b = 0; b < NBT; b++) {
      acc[b] = make_float2(0.f, 0.f);
      acc2[b] = make_float2(0.f, 0.f);
    }
    bool any = false;
    for (int dz = 0; dz < FW; dz++) {
      int const rz = pz - dz;
      if (rz < 0 || rz >= T) { continue; }
      for (int dy = 0; dy < FW; dy++) {
        int const ry = py - dy;
        if (ry < 0 || ry >= T) { continue; }
        int const row = (rz * T + ry) * T;
        int const j1 = cstart[row + rxhi + 1];
        int       j = cstart[row + rxlo];
        any = any || j < j1;
        // Two samples at a time into two sums (independent chains of loads and FMAs)
        for (; j + 1 < j1; j += 2) {
          float const w0 = sw[j][px - scx[j]] * (sw[j][FW + dy] * sw[j][2 * FW + dz]);
          float const w1 = sw[j + 1][px - scx[j + 1]] * (sw[j + 1][FW + dy] * sw[j + 1][2 * FW + dz]);
#pragma unroll
          for (int b = 0; b < NBT; b++) {
            float2 const v0 = sv[j][b], v1 = sv[j + 1][b];
            acc[b].x = fmaf(w0, v0.x, acc[b].x);
            acc[b].y = fmaf(w0, v0.y, acc[b].y);
            acc2[b].x = fmaf(w1, v1.x, acc2[b].x);
            acc2[b].y = fmaf(w1, v1.y, acc2[b].y);
          }
        }
        if (j < j1) {
          float const w0 = sw[j][px - scx[j]] * (sw[j][FW + dy] * sw[j][2 * FW + dz]);
#pragma unroll
          for (int b = 0; b < NBT; b++) {
            float2 const v0 = sv[j][b];
            acc[b].x = fmaf(w0, v0.x, acc[b].x);
            acc[b].y = fmaf(w0, v0.y, acc[b].y);
          }
        }
      }
    }
    if (!any) { continue; }
#pragma unroll
    for (int b = 0; b < NBT; b++) {
      acc[b].x += acc2[b].x;
      acc[b].y += acc2[b].y;
    }
    int const       ix = Wrap(bin.x0 - H + px, g.gx), iy = Wrap(bin.y0 - H + py, g.gy), iz = Wrap(bin.z0 - H + pz, g.gz);
    long long const gi = ix + g.gx * (iy + (long long)g.gy * iz);
#pragma unroll
    for (int b = 0; b < NBT; b++) {
      if (b < nb && (acc[b].x != 0.f || acc[b].y != 0.f)) {
        float *q = reinterpret_cast<float *>(ws + gi + g.G * (b0 + b));
        atomicAdd(q, acc[b].x);
        atomicAdd(q + 1, acc[b].y);
      }
    }
  }
}

template <int FW> void LaunchSpread(int const nB, unsigned const nBins, Geo const &g, Kern const &k, Samples const &smp,
                                    Bin const *bins, float2 const *in, bool const fromFull, float2 const *basis,
                                    float const bscale, float2 *ws, int const c)
{
  for (int b0 = 0; b0 < nB; b0 += 8) { // At most 8 basis images per pass (shared memory and registers)
    int const nb = std::min(8, nB - b0);
    switch (nb) {
    case 1: SpreadGather<FW, 1><<<nBins, 256>>>(g, k, smp, bins, in, fromFull, basis, bscale, ws, c, b0, nb); break;
    case 2: SpreadGather<FW, 2><<<nBins, 256>>>(g, k, smp, bins, in, fromFull, basis, bscale, ws, c, b0, nb); break;
    case 3: SpreadGather<FW, 3><<<nBins, 256>>>(g, k, smp, bins, in, fromFull, basis, bscale, ws, c, b0, nb); break;
    case 4: SpreadGather<FW, 4><<<nBins, 256>>>(g, k, smp, bins, in, fromFull, basis, bscale, ws, c, b0, nb); break;
    case 5: SpreadGather<FW, 5><<<nBins, 256>>>(g, k, smp, bins, in, fromFull, basis, bscale, ws, c, b0, nb); break;
    case 6: SpreadGather<FW, 6><<<nBins, 256>>>(g, k, smp, bins, in, fromFull, basis, bscale, ws, c, b0, nb); break;
    default: SpreadGather<FW, 8><<<nBins, 256>>>(g, k, smp, bins, in, fromFull, basis, bscale, ws, c, b0, nb); break;
    }
    Check(cudaGetLastError(), "spread");
  }
}

// Undo the folded FFT-shift signs: ws(g) *= (-1)^(gx + gy + gz)
__global__ void Unsign(Geo const g, float2 *__restrict__ ws)
{
  long long const gi = blockIdx.x * (long long)blockDim.x + threadIdx.x;
  if (gi >= g.G) { return; }
  int const       ix = gi % g.gx;
  long long const r = gi / g.gx;
  int const       iy = r % g.gy, iz = r / g.gy;
  if ((ix + iy + iz) & 1) {
    for (int b = 0; b < g.nB; b++) {
      ws[gi + g.G * b].x = -ws[gi + g.G * b].x;
      ws[gi + g.G * b].y = -ws[gi + g.G * b].y;
    }
  }
}

constexpr int Threads = 256;
auto          Blocks(long long const n) -> unsigned { return unsigned((n + Threads - 1) / Threads); }

struct Event
{
  cudaEvent_t e;
  Event() { Check(cudaEventCreate(&e), "event"); }
  ~Event() { cudaEventDestroy(e); }
};

} // namespace

// Device order (channel-major, the operator's sample order) from / to the full k-space (channel fastest)
__global__ void ToDeviceOrder(float2 const *__restrict__ k, int const *__restrict__ idx, long long const n, int const nC,
                              float2 *__restrict__ dk)
{
  long long const i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
  if (i >= n * nC) { return; }
  long long const is = i % n;
  int const       c = int(i / n);
  dk[i] = k[c + (long long)nC * idx[is]];
}

__global__ void FromDeviceOrder(float2 const *__restrict__ dk, int const *__restrict__ idx, long long const n, int const nC,
                                float2 *__restrict__ k)
{
  long long const i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
  if (i >= n * nC) { return; }
  long long const is = i % n;
  int const       c = int(i / n);
  k[c + (long long)nC * idx[is]] = dk[i];
}

// Partial sums (per block, double) of |y|² / W over a k-space vector in device order
__global__ void InvWeightedNorm2(float2 const *__restrict__ y, float4 const *__restrict__ off, float const *__restrict__ wch,
                                 long long const n, int const nC, double *__restrict__ part, bool const inverse)
{
  __shared__ double sh[256];
  double            a = 0.;
  for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < n * nC; i += (long long)gridDim.x * blockDim.x) {
    float const  W = wch ? wch[i] : off[i % n].w;
    float2 const v = y[i];
    double const m = double(v.x) * v.x + double(v.y) * v.y;
    a += inverse ? m / W : m * W;
  }
  sh[threadIdx.x] = a;
  __syncthreads();
  for (int o = blockDim.x / 2; o > 0; o >>= 1) {
    if (threadIdx.x < o) { sh[threadIdx.x] += sh[threadIdx.x + o]; }
    __syncthreads();
  }
  if (threadIdx.x == 0) { part[blockIdx.x] = sh[0]; }
}

__global__ void WeightK(float2 *__restrict__ y, float4 const *__restrict__ off, float const *__restrict__ wch, long long const n,
                        int const nC)
{
  long long const i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
  if (i >= n * nC) { return; }
  float const W = wch ? wch[i] : off[i % n].w;
  y[i] = make_float2(y[i].x * W, y[i].y * W);
}

struct SNUFFT::Impl
{
  SNUFFTSetup set;
  Geo         geo;
  Kern        kern;
  int         FW;

  DBuf<short4> ki;
  DBuf<float4> off;
  DBuf<int>    idx, brow;
  DBuf<float>  wch;
  DBuf<Bin>    bins;
  DBuf<float2> maps, basis, ws, yk, xin, xout, kfull;
  DBuf<float>  apo;
  long long    nValid = 0;
  float        bscale = 1.f;
  bool         weighted = false;
  cufftHandle  plan;

  mutable SNUFFTTimes times;
  mutable Event       ev[8];

  Impl(SNUFFTSetup const &s);
  ~Impl() { cufftDestroy(plan); }

  auto samples() const -> Samples { return Samples{ki.p, off.p, idx.p, brow.p, nValid}; }
  void interp(int const c, bool const weightedOut, float2 *out, bool const toFull, Dual const d = {}) const;
  void hostBuffers(); // The staging vectors of the host-pointer interface, allocated when first used
  void spread(int const c, float2 const *in, bool const fromFull) const;
  void fft(bool const fwd) const;
  void record(int const i) const { Check(cudaEventRecord(ev[i].e), "event record"); }
  auto elapsed(int const a, int const b) const -> double
  {
    float ms = 0.f;
    Check(cudaEventSynchronize(ev[b].e), "event sync");
    Check(cudaEventElapsedTime(&ms, ev[a].e, ev[b].e), "event elapsed");
    return ms;
  }
};

SNUFFT::Impl::Impl(SNUFFTSetup const &s)
  : set{s}
{
  Check(cudaSetDevice(s.device), "cudaSetDevice");
  // Wait for the device without spinning a CPU core: the CPU computes other terms while the device works. Only possible
  // before the context exists; if another operator created it, keep its setting.
  if (auto const e = cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync); e != cudaSuccess && e != cudaErrorSetOnActiveProcess) {
    Check(e, "cudaSetDeviceFlags");
  }
  cudaGetLastError(); // Clear cudaErrorSetOnActiveProcess
  for (auto const d : {s.gx, s.gy, s.gz}) {
    if (d % 2) { throw std::runtime_error("GPU SENSE-NUFFT: grid sizes must be even"); }
  }
  geo.nx = s.nx, geo.ny = s.ny, geo.nz = s.nz;
  geo.gx = s.gx, geo.gy = s.gy, geo.gz = s.gz;
  geo.px = (s.gx - s.nx + 1) / 2, geo.py = (s.gy - s.ny + 1) / 2, geo.pz = (s.gz - s.nz + 1) / 2;
  geo.N = s.nx * s.ny * s.nz;
  geo.G = s.gx * s.gy * s.gz;
  geo.nB = s.nB, geo.nC = s.nC, geo.nMB = s.nMB;
  geo.sigma = (((s.gx + s.gy + s.gz) / 2) % 2) ? -1.f : 1.f;
  geo.fs = 1.f / std::sqrt(float(geo.G));
  if (s.tophat) {
    FW = 1;
    kern = Kern{0.f, 0.5f, 0.f, s.kScale};
  } else {
    FW = ((s.kW + 1) / 2) * 2 + 1;
    if (FW != 5 && FW != 7 && FW != 9) { throw std::runtime_error("GPU SENSE-NUFFT: kernel width not supported"); }
    kern = Kern{s.beta, s.kW / 2.f, 0.5f - FW / 2.f, s.kScale};
  }

  // Valid samples, their nearest grid point and offset, exactly as TrajectoryN::toCoordLists
  std::fesetround(FE_TONEAREST);
  long long const      nST = s.nSamp * s.nTrace;
  std::vector<int>     valid;
  std::vector<short4>  hki(nST);
  std::vector<float4>  hoff(nST);
  std::vector<int64_t> key(nST);
  valid.reserve(nST);
  long long const gsz[3] = {s.gx, s.gy, s.gz};
  for (long long ist = 0; ist < nST; ist++) {
    float const *p = s.coords + 3 * ist;
    bool         ok = true;
    float        kr[3], ko[3];
    for (int d = 0; d < 3; d++) {
      if (p[d] != p[d]) { ok = false; }
      kr[d] = std::nearbyint(p[d]);
      ko[d] = p[d] - kr[d];
      if (kr[d] < 0.f || kr[d] >= float(gsz[d])) { ok = false; }
    }
    if (!ok) { continue; }
    int const ix = int(kr[0]), iy = int(kr[1]), iz = int(kr[2]);
    hki[ist] = make_short4(short(ix), short(iy), short(iz), 0);
    hoff[ist] = make_float4(ko[0], ko[1], ko[2], 1.f);
    // Sort key: tile, then the cell inside it
    int64_t const T = TileSz;
    int64_t const tile = (ix / T) + ((s.gx + T - 1) / T) * ((iy / T) + ((s.gy + T - 1) / T) * int64_t(iz / T));
    key[ist] = tile * 512 + ((iz % T) * T + (iy % T)) * T + (ix % T);
    valid.push_back(int(ist));
  }
  nValid = valid.size();
  std::stable_sort(valid.begin(), valid.end(), [&](int const a, int const b) { return key[a] < key[b]; });

  std::vector<short4> ski(nValid);
  std::vector<float4> soff(nValid);
  std::vector<int>    sidx(nValid), sbrow(nValid);
  for (long long iv = 0; iv < nValid; iv++) {
    int const ist = valid[iv];
    ski[iv] = hki[ist];
    soff[iv] = hoff[ist];
    sidx[iv] = ist;
    if (s.basis) {
      long long const is = ist % s.nSamp, it = ist / s.nSamp;
      sbrow[iv] = int(s.nB * ((is % s.nBS) + s.nBS * (it % s.nBT)));
    } else {
      sbrow[iv] = -1;
    }
  }
  weighted = s.weights != nullptr;
  if (weighted) {
    if (s.weightsPerChannel) {
      std::vector<float> w(nValid * s.nC);
      for (long long ic = 0; ic < s.nC; ic++) {
        for (long long iv = 0; iv < nValid; iv++) {
          w[ic * nValid + iv] = s.weights[ic + s.nC * sidx[iv]];
        }
      }
      wch.upload(w.data(), w.size());
    } else {
      for (long long iv = 0; iv < nValid; iv++) {
        soff[iv].w = s.weights[sidx[iv]];
      }
    }
  }
  { // Bins: runs of samples in the same tile, at most MaxBin each
    std::vector<Bin> hb;
    int64_t const    ntx = (s.gx + TileSz - 1) / TileSz, nty = (s.gy + TileSz - 1) / TileSz;
    for (long long iv = 0; iv < nValid;) {
      int64_t const tile = key[valid[iv]] / 512;
      long long     end = iv;
      while (end < nValid && key[valid[end]] / 512 == tile && end - iv < MaxBin) {
        end++;
      }
      hb.push_back(Bin{int(TileSz * (tile % ntx)), int(TileSz * ((tile / ntx) % nty)), int(TileSz * (tile / (ntx * nty))),
                       int(iv), int(end - iv)});
      iv = end;
    }
    bins.upload(hb.data(), hb.size());
  }
  ki.upload(ski.data(), nValid);
  off.upload(soff.data(), nValid);
  idx.upload(sidx.data(), nValid);
  brow.upload(sbrow.data(), nValid);
  { // Free the host copies before the large device allocations
    std::vector<short4>().swap(hki);
    std::vector<float4>().swap(hoff);
    std::vector<int64_t>().swap(key);
  }

  maps.upload(reinterpret_cast<float2 const *>(s.maps), geo.N * s.nMB * s.nC);
  apo.upload(s.apo, geo.N);
  if (s.basis) {
    basis.upload(reinterpret_cast<float2 const *>(s.basis), s.nB * s.nBS * s.nBT);
    bscale = s.basisScale;
  }
  ws.alloc(geo.G * s.nB);
  yk.alloc(nValid);

  int n[3] = {int(s.gz), int(s.gy), int(s.gx)};
  Check(cufftPlanMany(&plan, 3, n, nullptr, 1, int(geo.G), nullptr, 1, int(geo.G), CUFFT_C2C, int(s.nB)), "plan");
}

void SNUFFT::Impl::fft(bool const fwd) const
{
  Check(cufftExecC2C(plan, reinterpret_cast<cufftComplex *>(ws.p), reinterpret_cast<cufftComplex *>(ws.p),
                     fwd ? CUFFT_FORWARD : CUFFT_INVERSE),
        "exec");
}

void SNUFFT::Impl::hostBuffers()
{
  if (xin.n == 0) {
    xin.alloc(geo.N * set.nB);
    xout.alloc(geo.N * set.nB);
  }
}

void SNUFFT::Impl::interp(int const c, bool const weightedOut, float2 *out, bool const toFull, Dual const d) const
{
  auto const smp = samples();
  auto const nb = Blocks(nValid);
  switch (FW) {
  case 1: LaunchInterp<1>(nb, geo, kern, smp, ws.p, basis.p, bscale, wch.p, weightedOut, out, toFull, c, d); break;
  case 5: LaunchInterp<5>(nb, geo, kern, smp, ws.p, basis.p, bscale, wch.p, weightedOut, out, toFull, c, d); break;
  case 7: LaunchInterp<7>(nb, geo, kern, smp, ws.p, basis.p, bscale, wch.p, weightedOut, out, toFull, c, d); break;
  case 9: LaunchInterp<9>(nb, geo, kern, smp, ws.p, basis.p, bscale, wch.p, weightedOut, out, toFull, c, d); break;
  }
  Check(cudaGetLastError(), "interp");
}

void SNUFFT::Impl::spread(int const c, float2 const *in, bool const fromFull) const
{
  Check(cudaMemsetAsync(ws.p, 0, ws.bytes()), "memset");
  auto const     smp = samples();
  unsigned const nBins = bins.n;
  switch (FW) {
  case 1: LaunchSpread<1>(geo.nB, nBins, geo, kern, smp, bins.p, in, fromFull, basis.p, bscale, ws.p, c); break;
  case 5: LaunchSpread<5>(geo.nB, nBins, geo, kern, smp, bins.p, in, fromFull, basis.p, bscale, ws.p, c); break;
  case 7: LaunchSpread<7>(geo.nB, nBins, geo, kern, smp, bins.p, in, fromFull, basis.p, bscale, ws.p, c); break;
  case 9: LaunchSpread<9>(geo.nB, nBins, geo, kern, smp, bins.p, in, fromFull, basis.p, bscale, ws.p, c); break;
  }
}

DeviceBuffer::DeviceBuffer(size_t const bytes)
  : n_{bytes}
{
  if (n_) { Check(cudaMalloc(&p_, n_), "cudaMalloc"); }
}

DeviceBuffer::~DeviceBuffer()
{
  if (p_) { cudaFree(p_); }
}

DeviceBuffer::DeviceBuffer(DeviceBuffer &&o) noexcept
  : p_{o.p_}
  , n_{o.n_}
{
  o.p_ = nullptr;
  o.n_ = 0;
}

DeviceBuffer &DeviceBuffer::operator=(DeviceBuffer &&o) noexcept
{
  if (this != &o) {
    if (p_) { cudaFree(p_); }
    p_ = o.p_;
    n_ = o.n_;
    o.p_ = nullptr;
    o.n_ = 0;
  }
  return *this;
}

void DeviceBuffer::upload(void const *h, size_t const bytes)
{
  if (bytes > n_) { throw std::runtime_error("DeviceBuffer: upload larger than the buffer"); }
  Check(cudaMemcpy(p_, h, bytes, cudaMemcpyHostToDevice), "upload");
}

void DeviceBuffer::download(void *h, size_t const bytes) const
{
  if (bytes > n_) { throw std::runtime_error("DeviceBuffer: download larger than the buffer"); }
  Check(cudaMemcpy(h, p_, bytes, cudaMemcpyDeviceToHost), "download");
}

void DeviceBuffer::zero()
{
  if (n_) { Check(cudaMemset(p_, 0, n_), "memset"); }
}

auto DeviceFreeBytes() -> size_t
{
  size_t f = 0, t = 0;
  Check(cudaMemGetInfo(&f, &t), "cudaMemGetInfo");
  return f;
}

auto PinHost(void *p, size_t const bytes) -> bool
{
  if (cudaHostRegister(p, bytes, cudaHostRegisterDefault) != cudaSuccess) {
    cudaGetLastError();
    return false;
  }
  return true;
}

void UnpinHost(void *p)
{
  cudaHostUnregister(p);
  cudaGetLastError();
}

SNUFFT::SNUFFT(SNUFFTSetup const &s)
  : impl_{std::make_unique<Impl>(s)}
{
}

SNUFFT::~SNUFFT() = default;

auto SNUFFT::nValid() const -> int64_t { return impl_->nValid; }

auto SNUFFT::deviceBytes() const -> size_t
{
  auto const &m = *impl_;
  return m.ki.bytes() + m.off.bytes() + m.idx.bytes() + m.brow.bytes() + m.wch.bytes() + m.bins.bytes() + m.maps.bytes() +
         m.basis.bytes() +
         m.ws.bytes() + m.yk.bytes() + m.xin.bytes() + m.xout.bytes() + m.kfull.bytes() + m.apo.bytes();
}

auto SNUFFT::times() const -> SNUFFTTimes { return impl_->times; }
void SNUFFT::resetTimes() const { impl_->times = SNUFFTTimes{}; }

void SNUFFT::forward(Cxf const *x, Cxf *y, float const s) const
{
  auto &m = *impl_;
  m.hostBuffers();
  auto &t = m.times;
  if (m.kfull.n == 0) { m.kfull.alloc(m.set.nC * m.set.nSamp * m.set.nTrace); }
  m.record(0);
  Check(cudaMemcpy(m.xin.p, x, m.xin.bytes(), cudaMemcpyHostToDevice), "copy x");
  Check(cudaMemset(m.kfull.p, 0, m.kfull.bytes()), "memset k"); // Invalid samples stay zero
  m.record(1);
  t.copyIn += m.elapsed(0, 1);
  for (int c = 0; c < m.geo.nC; c++) {
    Expand<<<Blocks(m.geo.G), Threads>>>(m.geo, m.xin.p, m.maps.p, m.apo.p, m.ws.p, c, s);
    Check(cudaGetLastError(), "expand");
    m.record(2);
    m.fft(true);
    m.record(3);
    m.interp(c, false, m.kfull.p, true);
    m.record(4);
    Check(cudaEventSynchronize(m.ev[4].e), "sync");
    t.expand += m.elapsed(1, 2);
    t.fft += m.elapsed(2, 3);
    t.interp += m.elapsed(3, 4);
    m.record(1);
  }
  Check(cudaMemcpy(y, m.kfull.p, m.kfull.bytes(), cudaMemcpyDeviceToHost), "copy y");
  m.record(5);
  Check(cudaEventSynchronize(m.ev[5].e), "sync");
  t.copyOut += m.elapsed(1, 5);
  t.calls++;
}

void SNUFFT::adjoint(Cxf const *y, Cxf *x, float const s) const
{
  auto &m = *impl_;
  m.hostBuffers();
  auto &t = m.times;
  if (m.kfull.n == 0) { m.kfull.alloc(m.set.nC * m.set.nSamp * m.set.nTrace); }
  m.record(0);
  Check(cudaMemcpy(m.kfull.p, y, m.kfull.bytes(), cudaMemcpyHostToDevice), "copy y");
  m.record(1);
  t.copyIn += m.elapsed(0, 1);
  for (int c = 0; c < m.geo.nC; c++) {
    m.spread(c, m.kfull.p, true);
    m.record(2);
    m.fft(false);
    m.record(3);
    Collect<<<Blocks(m.geo.N), Threads>>>(m.geo, m.ws.p, m.maps.p, m.apo.p, m.xout.p, c, s, c == 0, c == m.geo.nC - 1);
    Check(cudaGetLastError(), "collect");
    m.record(4);
    Check(cudaEventSynchronize(m.ev[4].e), "sync");
    t.spread += m.elapsed(1, 2);
    t.ifft += m.elapsed(2, 3);
    t.collect += m.elapsed(3, 4);
    m.record(1);
  }
  Check(cudaMemcpy(x, m.xout.p, m.xout.bytes(), cudaMemcpyDeviceToHost), "copy x");
  m.record(5);
  Check(cudaEventSynchronize(m.ev[5].e), "sync");
  t.copyOut += m.elapsed(1, 5);
  t.calls++;
}

void SNUFFT::gridAdjoint(Cxf const *y, Cxf *grid) const
{
  auto &m = *impl_;
  if (m.kfull.n == 0) { m.kfull.alloc(m.set.nC * m.set.nSamp * m.set.nTrace); }
  Check(cudaMemcpy(m.kfull.p, y, m.kfull.bytes(), cudaMemcpyHostToDevice), "copy y");
  for (int c = 0; c < m.geo.nC; c++) {
    m.spread(c, m.kfull.p, true);
    Unsign<<<Blocks(m.geo.G), Threads>>>(m.geo, m.ws.p);
    Check(cudaGetLastError(), "unsign");
    Check(cudaMemcpy(grid + m.ws.n * c, m.ws.p, m.ws.bytes(), cudaMemcpyDeviceToHost), "copy grid");
  }
}

void SNUFFT::normal(Cxf const *x, Cxf *y, float const s) const
{
  auto &m = *impl_;
  m.hostBuffers();
  auto &t = m.times;
  m.record(0);
  Check(cudaMemcpy(m.xin.p, x, m.xin.bytes(), cudaMemcpyHostToDevice), "copy x");
  m.record(1);
  t.copyIn += m.elapsed(0, 1);
  for (int c = 0; c < m.geo.nC; c++) {
    Expand<<<Blocks(m.geo.G), Threads>>>(m.geo, m.xin.p, m.maps.p, m.apo.p, m.ws.p, c, 1.f);
    Check(cudaGetLastError(), "expand");
    m.record(2);
    m.fft(true);
    m.record(3);
    m.interp(c, m.weighted, m.yk.p, false);
    m.record(4);
    m.spread(c, m.yk.p, false);
    m.record(5);
    m.fft(false);
    m.record(6);
    Collect<<<Blocks(m.geo.N), Threads>>>(m.geo, m.ws.p, m.maps.p, m.apo.p, m.xout.p, c, s, c == 0, c == m.geo.nC - 1);
    Check(cudaGetLastError(), "collect");
    m.record(7);
    Check(cudaEventSynchronize(m.ev[7].e), "sync");
    t.expand += m.elapsed(1, 2);
    t.fft += m.elapsed(2, 3);
    t.interp += m.elapsed(3, 4);
    t.spread += m.elapsed(4, 5);
    t.ifft += m.elapsed(5, 6);
    t.collect += m.elapsed(6, 7);
    m.record(1);
  }
  Check(cudaMemcpy(y, m.xout.p, m.xout.bytes(), cudaMemcpyDeviceToHost), "copy x");
  m.record(2);
  Check(cudaEventSynchronize(m.ev[2].e), "sync");
  t.copyOut += m.elapsed(1, 2);
  t.calls++;
}

auto SNUFFT::nK() const -> int64_t { return impl_->nValid; }
auto SNUFFT::nC() const -> int64_t { return impl_->set.nC; }

void SNUFFT::sync() const { Check(cudaDeviceSynchronize(), "sync"); }

namespace {
// Σ W^±1 |y|² over a k-space vector in device order
auto WeightedNorm2(float2 const *y, float4 const *off, float const *wch, long long const n, int const nC, bool const inverse)
  -> double
{
  int const    nb = 1024;
  DBuf<double> part;
  part.alloc(nb);
  InvWeightedNorm2<<<nb, 256>>>(y, off, wch, n, nC, part.p, inverse);
  Check(cudaGetLastError(), "weighted norm");
  std::vector<double> h(nb);
  Check(cudaMemcpy(h.data(), part.p, nb * sizeof(double), cudaMemcpyDeviceToHost), "copy norm");
  double s = 0.;
  for (auto const v : h) {
    s += v;
  }
  return s;
}
} // namespace

auto SNUFFT::dualNorm2D(Cxf const *y) const -> double
{
  auto const &m = *impl_;
  return WeightedNorm2(reinterpret_cast<float2 const *>(y), m.off.p, m.wch.n ? m.wch.p : nullptr, m.nValid, int(m.set.nC), true);
}

auto SNUFFT::weightedNorm2D(Cxf const *y) const -> double
{
  auto const &m = *impl_;
  return WeightedNorm2(reinterpret_cast<float2 const *>(y), m.off.p, m.wch.n ? m.wch.p : nullptr, m.nValid, int(m.set.nC),
                       false);
}

void SNUFFT::weightD(Cxf *y) const
{
  auto const     &m = *impl_;
  long long const n = m.nValid * m.set.nC;
  WeightK<<<Blocks(n), Threads>>>(reinterpret_cast<float2 *>(y), m.off.p, m.wch.n ? m.wch.p : nullptr, m.nValid, int(m.set.nC));
  Check(cudaGetLastError(), "weight");
}

void SNUFFT::toDeviceOrder(Cxf const *k, Cxf *dk) const
{
  auto const     &m = *impl_;
  long long const nFull = m.set.nC * m.set.nSamp * m.set.nTrace, n = m.nValid * m.set.nC;
  DBuf<float2>    full;
  full.upload(reinterpret_cast<float2 const *>(k), nFull);
  ToDeviceOrder<<<Blocks(n), Threads>>>(full.p, m.idx.p, m.nValid, int(m.set.nC), reinterpret_cast<float2 *>(dk));
  Check(cudaGetLastError(), "to device order");
  Check(cudaDeviceSynchronize(), "sync");
}

void SNUFFT::fromDeviceOrder(Cxf const *dk, Cxf *k) const
{
  auto const     &m = *impl_;
  long long const nFull = m.set.nC * m.set.nSamp * m.set.nTrace, n = m.nValid * m.set.nC;
  DBuf<float2>    full;
  full.alloc(nFull);
  Check(cudaMemset(full.p, 0, full.bytes()), "memset");
  FromDeviceOrder<<<Blocks(n), Threads>>>(reinterpret_cast<float2 const *>(dk), m.idx.p, m.nValid, int(m.set.nC), full.p);
  Check(cudaGetLastError(), "from device order");
  Check(cudaMemcpy(k, full.p, full.bytes(), cudaMemcpyDeviceToHost), "copy k");
}

void SNUFFT::forwardD(Cxf const *x, Cxf *y, float const s) const
{
  auto const &m = *impl_;
  auto const  xd = reinterpret_cast<float2 const *>(x);
  auto const  yd = reinterpret_cast<float2 *>(y);
  for (int c = 0; c < m.geo.nC; c++) {
    Expand<<<Blocks(m.geo.G), Threads>>>(m.geo, xd, m.maps.p, m.apo.p, m.ws.p, c, s);
    Check(cudaGetLastError(), "expand");
    m.fft(true);
    m.interp(c, false, yd + m.nValid * c, false);
  }
}

void SNUFFT::adjointD(Cxf const *y, Cxf *x, float const s, bool const add) const
{
  auto const &m = *impl_;
  auto const  yd = reinterpret_cast<float2 const *>(y);
  auto const  xd = reinterpret_cast<float2 *>(x);
  for (int c = 0; c < m.geo.nC; c++) {
    m.spread(c, yd + m.nValid * c, false);
    m.fft(false);
    Collect<<<Blocks(m.geo.N), Threads>>>(m.geo, m.ws.p, m.maps.p, m.apo.p, xd, c, s, c == 0, c == m.geo.nC - 1, add);
    Check(cudaGetLastError(), "collect");
  }
}

void SNUFFT::normalD(Cxf const *x, Cxf *y, float const s) const
{
  auto const &m = *impl_;
  auto const  xd = reinterpret_cast<float2 const *>(x);
  auto const  yd = reinterpret_cast<float2 *>(y);
  for (int c = 0; c < m.geo.nC; c++) {
    Expand<<<Blocks(m.geo.G), Threads>>>(m.geo, xd, m.maps.p, m.apo.p, m.ws.p, c, 1.f);
    Check(cudaGetLastError(), "expand");
    m.fft(true);
    m.interp(c, m.weighted, m.yk.p, false);
    m.spread(c, m.yk.p, false);
    m.fft(false);
    Collect<<<Blocks(m.geo.N), Threads>>>(m.geo, m.ws.p, m.maps.p, m.apo.p, yd, c, s, c == 0, c == m.geo.nC - 1);
    Check(cudaGetLastError(), "collect");
  }
}

void SNUFFT::dataDualD(Cxf const *xhat, Cxf *y, Cxf const *b, float const sigma, DataDual const kind) const
{
  auto const &m = *impl_;
  auto const  xd = reinterpret_cast<float2 const *>(xhat);
  Dual        d;
  d.y = reinterpret_cast<float2 *>(y);
  d.b = reinterpret_cast<float2 const *>(b);
  d.sigma = sigma;
  d.kind = int(kind);
  for (int c = 0; c < m.geo.nC; c++) {
    Expand<<<Blocks(m.geo.G), Threads>>>(m.geo, xd, m.maps.p, m.apo.p, m.ws.p, c, 1.f);
    Check(cudaGetLastError(), "expand");
    m.fft(true);
    m.interp(c, false, nullptr, false, d);
  }
}

} // namespace rl::gpu
