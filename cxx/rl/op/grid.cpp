#include "grid.hpp"

#include "../log/log.hpp"
#include "../sys/threads.hpp"
#include "grid-func.hpp"
#include "grid-subgrid.hpp"
#include "grid-transposed.hpp"
#include "top-impl.hpp"

namespace rl {

namespace TOps {

namespace {
/*
 * Each coordinate list is processed by one thread. Centre-out (and especially ramp-sampled UTE)
 * trajectories put a large fraction of all samples into the few subgrids at the k-space centre
 * (11.7% in one subgrid for a 3D UTE dataset), so one thread did far more than its share. Split
 * long lists into chunks with the same corner; the threads then share the centre between them.
 */
constexpr size_t MaxCoordsPerList = 32768;

template <typename CL> void SplitLists(std::vector<CL> &lists, size_t const maxCoords)
{
  std::vector<CL> split;
  split.reserve(lists.size());
  for (auto &list : lists) {
    if (list.coords.size() <= maxCoords) {
      split.push_back(std::move(list));
      continue;
    }
    size_t const nChunks = (list.coords.size() + maxCoords - 1) / maxCoords;
    size_t const chunk = (list.coords.size() + nChunks - 1) / nChunks;
    for (size_t start = 0; start < list.coords.size(); start += chunk) {
      CL part;
      part.corner = list.corner;
      part.coords.assign(list.coords.begin() + start, list.coords.begin() + std::min(start + chunk, list.coords.size()));
      split.push_back(std::move(part));
    }
  }
  // Largest first, so the threads pick up the big chunks before the many small lists
  std::stable_sort(split.begin(), split.end(), [](CL const &a, CL const &b) { return a.coords.size() > b.coords.size(); });
  lists = std::move(split);
}
} // namespace

template <int ND, typename KF, int SG>
auto Grid<ND, KF, SG>::Make(GridOpts<ND> const &opts, TrajectoryN<ND> const &traj, Index const nC, Basis::CPtr b)
  -> std::shared_ptr<Grid<ND, KF>>
{
  return std::make_shared<Grid<ND, KF>>(opts, traj, nC, b);
}

template <int ND, typename KF, int SG>
Grid<ND, KF, SG>::Grid(GridOpts<ND> const &opts, TrajectoryN<ND> const &traj, Index const nC, Basis::CPtr b)
  : Parent(fmt::format("Grid{}D", ND))
  , kernel(opts.osamp)
  , basis{b}
{
  static_assert(ND < 4);
  auto const osMatrix = MulToEven(traj.matrixForFOV(opts.fov), opts.osamp);
  gridLists = traj.toCoordLists(osMatrix, kernel.FullWidth, SGSZ, false);
  auto const nLists = gridLists.size();
  SplitLists(gridLists, MaxCoordsPerList);
  Log::Debug("Grid", "Split {} coordinate lists into {} (at most {} coords each)", nLists, gridLists.size(), MaxCoordsPerList);
  ishape = AddBack(osMatrix, basis ? basis->nB() : 1, nC);
  oshape = Sz3{nC, traj.nSamples(), traj.nTraces()};
  mutexes = std::vector<std::mutex>(osMatrix[ND - 1]);
  Log::Debug("Grid", "ishape {} oshape {}", this->ishape, this->oshape);
}

template <int ND, typename KF, int SG> void
Grid<ND, KF, SG>::forwardTask(std::atomic<size_t> &next, float const s, CxNCMap<ND + 2> const x, Cx3Map y) const
{
  if constexpr (ND == 3) { // Transposed subgrid, see grid-transposed.hpp
    constexpr int   FW = KF::FullWidth;
    Index const     nB = basis ? basis->nB() : 1, nC = y.dimension(0), nBC = nB * nC;
    std::vector<Cx> sg(SGFW * SGFW * SGFW * nBC), acc(nBC), b(nB);
    for (size_t is = next++; is < gridLists.size(); is = next++) {
      auto const &list = gridLists[is];
      auto const  corner = SubgridCorner<ND, SGSZ, FW>(list.corner);
      if (InBounds<ND, SGFW>(corner, FirstN<ND>(x.dimensions()))) {
        GridT::Load<SGFW, false>(corner, x, sg.data());
      } else {
        GridT::Load<SGFW, true>(corner, x, sg.data());
      }
      for (auto const &m : list.coords) {
        typename KType::Tensor const k = kernel(m.offset) * s;
        if (basis) {
          Cx const *const bp = basis->entries(m.sample, m.trace);
          bool            dense = true;
          for (Index ib = 0; ib < nB; ib++) {
            b[ib] = bp[ib] * Cx(basis->scale);
            dense = dense && (b[ib] != Cx(0));
          }
          if (dense) {
            GridT::Gather<FW, SGFW>(m.cart, k, sg.data(), nBC, 0, nBC, acc.data());
            for (Index ic = 0; ic < nC; ic++) {
              Cx sum = 0.f;
              for (Index ib = 0; ib < nB; ib++) {
                sum += acc[ic + nC * ib] * b[ib];
              }
              y(ic, m.sample, m.trace) += sum;
            }
          } else {
            for (Index ib = 0; ib < nB; ib++) {
              if (b[ib] == Cx(0)) { continue; }
              GridT::Gather<FW, SGFW>(m.cart, k, sg.data(), nBC, nC * ib, nC, acc.data());
              for (Index ic = 0; ic < nC; ic++) {
                y(ic, m.sample, m.trace) += acc[ic] * b[ib];
              }
            }
          }
        } else {
          GridT::Gather<FW, SGFW>(m.cart, k, sg.data(), nBC, 0, nC, acc.data());
          for (Index ic = 0; ic < nC; ic++) {
            y(ic, m.sample, m.trace) += acc[ic];
          }
        }
      }
    }
    return;
  }
  CxN<ND + 2> sx(AddBack(Constant<ND>(SGFW), basis ? basis->nB() : 1, y.dimension(0)));
  for (size_t is = next++; is < gridLists.size(); is = next++) {
    auto const &list = gridLists[is];
    auto const  corner = SubgridCorner<ND, SGSZ, KF::FullWidth>(list.corner);
    if (InBounds<ND, SGFW>(corner, FirstN<ND>(x.dimensions()))) {
      GridToSubgrid<ND, SGFW>::FastCopy(corner, x, sx);
    } else {
      GridToSubgrid<ND, SGFW>::SlowCopy(corner, x, sx);
    }
    for (auto const &m : list.coords) {
      typename KType::Tensor const k = kernel(m.offset) * s;
      if (basis) {
        GFunc<ND, KF::FullWidth>::Gather(basis, m.cart, m.sample, m.trace, k, sx, y);
      } else {
        GFunc<ND, KF::FullWidth>::Gather(m.cart, m.sample, m.trace, k, sx, y);
      }
    }
  }
}

template <int ND, typename KF, int SG> void Grid<ND, KF, SG>::forward(InCMap x, OutMap y, float const s) const
{
  auto const time = this->startForward(x, y, false);
  y.device(Threads::TensorDevice()) = y.constant(0.f);
  std::atomic<size_t> next = 0; // Lists are handed out dynamically, largest first
  Threads::StridedFor(gridLists.size(), [&](Index const, Index const) { forwardTask(next, s, x, y); });
  this->finishForward(y, time, false);
}

template <int ND, typename KF, int SG> void Grid<ND, KF, SG>::iforward(InCMap x, OutMap y, float const s) const
{
  auto const          time = this->startForward(x, y, true);
  std::atomic<size_t> next = 0;
  Threads::StridedFor(gridLists.size(), [&](Index const, Index const) { forwardTask(next, s, x, y); });
  this->finishForward(y, time, true);
}

template <int ND, typename KF, int SG>
void Grid<ND, KF, SG>::adjointTask(std::atomic<size_t> &next, float const s, Cx3CMap y, CxNMap<ND + 2> x) const

{
  if constexpr (ND == 3) { // Transposed subgrid, see grid-transposed.hpp
    constexpr int   FW = KF::FullWidth;
    Index const     nB = basis ? basis->nB() : 1, nC = y.dimension(0), nBC = nB * nC;
    std::vector<Cx> sg(SGFW * SGFW * SGFW * nBC), v(nBC), b(nB);
    for (size_t is = next++; is < gridLists.size(); is = next++) {
      auto const &list = gridLists[is];
      std::fill(sg.begin(), sg.end(), Cx(0.f));
      for (auto const &m : list.coords) {
        typename KType::Tensor const k = kernel(m.offset) * s;
        if (basis) {
          Cx const *const bp = basis->entries(m.sample, m.trace);
          bool            dense = true;
          for (Index ib = 0; ib < nB; ib++) {
            b[ib] = std::conj(bp[ib] * Cx(basis->scale));
            dense = dense && (b[ib] != Cx(0));
          }
          if (dense) {
            for (Index ib = 0; ib < nB; ib++) {
              for (Index ic = 0; ic < nC; ic++) {
                v[ic + nC * ib] = b[ib] * y(ic, m.sample, m.trace);
              }
            }
            GridT::Scatter<FW, SGFW>(m.cart, k, v.data(), nBC, 0, nBC, sg.data());
          } else {
            for (Index ib = 0; ib < nB; ib++) {
              if (b[ib] == Cx(0)) { continue; }
              for (Index ic = 0; ic < nC; ic++) {
                v[ic] = b[ib] * y(ic, m.sample, m.trace);
              }
              GridT::Scatter<FW, SGFW>(m.cart, k, v.data(), nBC, nC * ib, nC, sg.data());
            }
          }
        } else {
          for (Index ic = 0; ic < nC; ic++) {
            v[ic] = y(ic, m.sample, m.trace);
          }
          GridT::Scatter<FW, SGFW>(m.cart, k, v.data(), nBC, 0, nC, sg.data());
        }
      }
      auto const corner = SubgridCorner<ND, SGSZ, FW>(list.corner);
      if (InBounds<ND, SGFW>(corner, FirstN<ND>(x.dimensions()))) {
        GridT::Store<SGFW, false>(mutexes, corner, sg.data(), x);
      } else {
        GridT::Store<SGFW, true>(mutexes, corner, sg.data(), x);
      }
    }
    return;
  }
  CxN<ND + 2> sx(AddBack(Constant<ND>(SGFW), basis ? basis->nB() : 1, y.dimension(0)));
  for (size_t is = next++; is < gridLists.size(); is = next++) {
    auto const &list = gridLists[is];
    sx.setZero();
    for (auto const &m : list.coords) {
      typename KType::Tensor const k = kernel(m.offset) * s;
      if (basis) {
        GFunc<ND, KF::FullWidth>::Scatter(basis, m.cart, m.sample, m.trace, k, y, sx);
      } else {
        GFunc<ND, KF::FullWidth>::Scatter(m.cart, m.sample, m.trace, k, y, sx);
      }
    }
    auto const corner = SubgridCorner<ND, SGSZ, KF::FullWidth>(list.corner);
    if (InBounds<ND, SGFW>(corner, FirstN<ND>(x.dimensions()))) {
      SubgridToGrid<ND, SGFW>::FastCopy(mutexes, corner, sx, x);
    } else {
      SubgridToGrid<ND, SGFW>::SlowCopy(mutexes, corner, sx, x);
    }
  }
}

template <int ND, typename KF, int SG> void Grid<ND, KF, SG>::adjoint(OutCMap y, InMap x, float const s) const
{
  auto const time = this->startAdjoint(y, x, false);
  x.device(Threads::TensorDevice()) = x.constant(0.f);
  std::atomic<size_t> next = 0;
  Threads::StridedFor(gridLists.size(), [&](Index const, Index const) { adjointTask(next, s, y, x); });
  this->finishAdjoint(x, time, false);
}

template <int ND, typename KF, int SG> void Grid<ND, KF, SG>::iadjoint(OutCMap y, InMap x, float const s) const
{
  auto const          time = this->startAdjoint(y, x, true);
  std::atomic<size_t> next = 0;
  Threads::StridedFor(gridLists.size(), [&](Index const, Index const) { adjointTask(next, s, y, x); });
  this->finishAdjoint(x, time, true);
}

template struct Grid<1, rl::ExpSemi<4>>;
template struct Grid<2, rl::ExpSemi<4>>;
template struct Grid<3, rl::ExpSemi<4>>;

template struct Grid<1, rl::ExpSemi<6>>;
template struct Grid<2, rl::ExpSemi<6>>;
template struct Grid<3, rl::ExpSemi<6>>;

template struct Grid<1, rl::ExpSemi<8>>;
template struct Grid<2, rl::ExpSemi<8>>;
template struct Grid<3, rl::ExpSemi<8>>;

template struct Grid<1, rl::TopHat<1>>;
template struct Grid<2, rl::TopHat<1>>;
template struct Grid<3, rl::TopHat<1>>;

} // namespace TOps
} // namespace rl