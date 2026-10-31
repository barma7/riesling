#pragma once

#include "../kernel/kernel.hpp"
#include "../types.hpp"

#include <mutex>
#include <vector>

/*
 * 3D gridding with a transposed subgrid. The grid keeps riesling's [i, j, k, b, c] layout,
 * but each thread's subgrid stores all basis vectors and channels of one grid point next to
 * each other ([point][b][c], c fastest). Every kernel tap then reads or updates nB * nC
 * contiguous values, which vectorises, instead of nB * nC separate taps in strided planes.
 *
 * The basis is applied once per sample (forward: after the kernel sum, adjoint: before it).
 * Basis vectors that are zero for a sample are skipped, as in GFunc, so sparse bases
 * (e.g. time frames, one non-zero entry per trace) do not pay for the zero entries.
 */
namespace rl::GridT {

template <int SGFW> inline auto Point(Index const ix, Index const iy, Index const iz) -> Index
{
  return ix + SGFW * (iy + SGFW * iz);
}

// Grid -> transposed subgrid
template <int SGFW, bool Wrapped> inline void Load(Eigen::Array<int16_t, 3, 1> const corner, Cx5CMap const &x, Cx *sg)
{
  Index const nB = x.dimension(3), nC = x.dimension(4), nBC = nB * nC;
  for (Index ib = 0; ib < nB; ib++) {
    for (Index ic = 0; ic < nC; ic++) {
      Index const bc = ic + nC * ib;
      for (Index iz = 0; iz < SGFW; iz++) {
        Index const iiz = Wrapped ? Wrap<Index>(iz + corner[2], x.dimension(2)) : iz + corner[2];
        for (Index iy = 0; iy < SGFW; iy++) {
          Index const iiy = Wrapped ? Wrap<Index>(iy + corner[1], x.dimension(1)) : iy + corner[1];
          for (Index ix = 0; ix < SGFW; ix++) {
            Index const iix = Wrapped ? Wrap<Index>(ix + corner[0], x.dimension(0)) : ix + corner[0];
            sg[Point<SGFW>(ix, iy, iz) * nBC + bc] = x(iix, iiy, iiz, ib, ic);
          }
        }
      }
    }
  }
}

// Transposed subgrid -> grid (accumulate). Locks one grid plane at a time for each b and c, as
// SubgridToGrid does: holding a plane for all b and c at once made 16 threads slower than 8.
template <int SGFW, bool Wrapped>
inline void Store(std::vector<std::mutex> &m, Eigen::Array<int16_t, 3, 1> const corner, Cx const *sg, Cx5Map &x)
{
  Index const nB = x.dimension(3), nC = x.dimension(4), nBC = nB * nC;
  for (Index ib = 0; ib < nB; ib++) {
    for (Index ic = 0; ic < nC; ic++) {
      Index const bc = ic + nC * ib;
      for (Index iz = 0; iz < SGFW; iz++) {
        Index const      iiz = Wrapped ? Wrap<Index>(iz + corner[2], x.dimension(2)) : iz + corner[2];
        std::scoped_lock lock(m[iiz]);
        for (Index iy = 0; iy < SGFW; iy++) {
          Index const iiy = Wrapped ? Wrap<Index>(iy + corner[1], x.dimension(1)) : iy + corner[1];
          for (Index ix = 0; ix < SGFW; ix++) {
            Index const iix = Wrapped ? Wrap<Index>(ix + corner[0], x.dimension(0)) : ix + corner[0];
            x(iix, iiy, iiz, ib, ic) += sg[Point<SGFW>(ix, iy, iz) * nBC + bc];
          }
        }
      }
    }
  }
}

// acc[j] = sum over kernel taps of k * sg[tap][off + j], j < len
template <int FW, int SGFW>
inline void Gather(Eigen::Array<int16_t, 3, 1> const c,
                   FixedTensor<float, 3, FW> const  &k,
                   Cx const                         *sg,
                   Index const                       nBC,
                   Index const                       off,
                   Index const                       len,
                   Cx                               *acc)
{
  float *__restrict a = reinterpret_cast<float *>(acc);
  Index const n2 = 2 * len;
  for (Index j = 0; j < n2; j++) {
    a[j] = 0.f;
  }
  for (Index iz = 0; iz < FW; iz++) {
    Index const iiz = iz + c[2] - FW / 2;
    for (Index iy = 0; iy < FW; iy++) {
      Index const iiy = iy + c[1] - FW / 2;
      Cx const   *row = sg + Point<SGFW>(c[0] - FW / 2, iiy, iiz) * nBC + off;
      for (Index ix = 0; ix < FW; ix++) {
        float const                w = k(ix, iy, iz);
        float const *__restrict p = reinterpret_cast<float const *>(row + ix * nBC);
        for (Index j = 0; j < n2; j++) {
          a[j] += w * p[j];
        }
      }
    }
  }
}

// sg[tap][off + j] += k * v[j], j < len
template <int FW, int SGFW>
inline void Scatter(Eigen::Array<int16_t, 3, 1> const c,
                    FixedTensor<float, 3, FW> const  &k,
                    Cx const                         *v,
                    Index const                       nBC,
                    Index const                       off,
                    Index const                       len,
                    Cx                               *sg)
{
  float const *__restrict vv = reinterpret_cast<float const *>(v);
  Index const n2 = 2 * len;
  for (Index iz = 0; iz < FW; iz++) {
    Index const iiz = iz + c[2] - FW / 2;
    for (Index iy = 0; iy < FW; iy++) {
      Index const iiy = iy + c[1] - FW / 2;
      Cx         *row = sg + Point<SGFW>(c[0] - FW / 2, iiy, iiz) * nBC + off;
      for (Index ix = 0; ix < FW; ix++) {
        float const          w = k(ix, iy, iz);
        float *__restrict p = reinterpret_cast<float *>(row + ix * nBC);
        for (Index j = 0; j < n2; j++) {
          p[j] += w * vv[j];
        }
      }
    }
  }
}

} // namespace rl::GridT
