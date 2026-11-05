#include "precon.hpp"

#include "fft.hpp"
#include "io/reader.hpp"
#include "log/log.hpp"
#include "op/ndft.hpp"
#include "op/nufft.hpp"
#include "op/tensorscale.hpp"
#include "op/top-id.hpp"
#include "sys/threads.hpp"

#include "log/debug.hpp"

#include <vector>

namespace rl {

namespace {
using DCx = std::complex<double>;

auto BasisValue(Basis::CPtr basis, Index b, Index s, Index t) -> DCx
{
  if (!basis) { return 1.; }
  return DCx(basis->B(b, s % basis->nSample(), t % basis->nTrace())) * double(basis->scale);
}

void CheckInputs(float λ, Basis::CPtr basis)
{
  if (!std::isfinite(λ) || λ < 0.f) { throw Log::Failure("Precon", "Regularization must be finite and nonnegative"); }
  if (basis) {
    if (!std::isfinite(basis->scale) || basis->nB() < 1 || basis->nSample() < 1 || basis->nTrace() < 1) {
      throw Log::Failure("Precon", "Invalid basis dimensions or scale");
    }
    for (Index i = 0; i < basis->B.size(); ++i) {
      if (!std::isfinite(basis->B.data()[i].real()) || !std::isfinite(basis->B.data()[i].imag())) {
        throw Log::Failure("Precon", "Basis entries must be finite");
      }
    }
  }
}

// q_bd(i) conj(q_bd(j)) sums to |B(i)^H B(j)|^2. Stream scalar NUFFTs,
// rather than allocating K^2 image grids. Do not construct a Basis here: its
// normalization belongs to the encoding and must not be applied a second time.
auto ProductFeature(Basis::CPtr basis, Index b, Index d, Cx3 &q) -> bool
{
  if (basis) {
    bool overlap = false;
    for (Index t = 0; t < basis->nTrace(); ++t) {
      for (Index s = 0; s < basis->nSample(); ++s) {
        overlap = overlap || (basis->B(b, s, t) != Cx(0.f) && basis->B(d, s, t) != Cx(0.f));
      }
    }
    if (!overlap || basis->scale == 0.f) { return false; }
  }
  bool nonzero = false;
  for (Index t = 0; t < q.dimension(2); ++t) {
    for (Index s = 0; s < q.dimension(1); ++s) {
      auto const v = BasisValue(basis, b, s, t) * std::conj(BasisValue(basis, d, s, t));
      q(0, s, t) = Cx(v);
      nonzero = nonzero || v != DCx(0.);
    }
  }
  return nonzero;
}

// Normalize the target feature before multiplying, using double precision for
// the ratio. This also handles tiny (but nonzero) basis rows without a 0/0.
void Accumulate(Re3 &denom, Index channel, Cx3 const &value, Basis::CPtr basis,
                Index b, Index d, std::vector<double> const &mapNorm, float scale, float multiplicity)
{
  Index const nB = basis ? basis->nB() : 1;
  for (Index t = 0; t < value.dimension(2); ++t) {
    for (Index s = 0; s < value.dimension(1); ++s) {
      double norm = 0.;
      for (Index k = 0; k < nB; ++k) { norm += std::norm(BasisValue(basis, k, s, t)) * mapNorm[k]; }
      if (norm == 0.) { continue; }
      auto const q = BasisValue(basis, b, s, t) * std::conj(BasisValue(basis, d, s, t)) / norm;
      denom(channel, s, t) += float((q * DCx(value(0, s, t))).real() * scale * multiplicity);
    }
  }
}

// Samples the gridder ignored have no row in this calculation, whatever the trajectory says: besides
// invalid points, a sample within half a grid cell of the +k wall rounds to one index past the end.
auto Invert(Re3 weights, Basis::CPtr basis, std::vector<std::vector<double>> const &norms, BN<2> const &covered, float λ) -> Re3
{
  for (Index t = 0; t < weights.dimension(2); ++t) {
    for (Index s = 0; s < weights.dimension(1); ++s) {
      for (Index c = 0; c < weights.dimension(0); ++c) {
        double norm = 0.;
        for (Index b = 0; b < Index(norms[c].size()); ++b) { norm += std::norm(BasisValue(basis, b, s, t)) * norms[c][b]; }
        if (!covered(s, t)) { norm = 0.; }
        float const den = weights(c, s, t);
        if (norm > 0. && (!std::isfinite(den) || den <= 0.f)) {
          throw Log::Failure("Precon", "Nonpositive or nonfinite row denominator {} at channel {} sample {} trace {}", den, c, s, t);
        }
        weights(c, s, t) = norm == 0. ? 1.f : (1.f + λ) / (den + λ);
        if (!std::isfinite(weights(c, s, t)) || weights(c, s, t) <= 0.f) {
          throw Log::Failure("Precon", "Generated weight was not finite and positive");
        }
      }
    }
  }
  Log::Print("Precon", "Finished min {} max {}", Minimum(weights), Maximum(weights));
  return weights;
}
} // namespace

/* Ong, Uecker & Lustig (2020), section VI:
 * p_i = ||a_i||^2 / sum_j |a_i^H a_j|^2.
 * The spatial autocorrelation accounts for the nominal image support. */
template <int ND> auto KSpaceSingle(GridOpts<ND> const &gridOpts, TrajectoryN<ND> const &traj, float const λ, Basis::CPtr basis)
  -> Re2
{
  CheckInputs(λ, basis);
  Log::Print("Precon", "Ong squared-overlap weights, basis rank {}, regularization {}", basis ? basis->nB() : 0, λ);
  TrajectoryN<ND> newTraj(traj.points() * 2.f, MulToEven(traj.matrix(), 2), traj.voxelSize() / 2.f);
  TOps::NUFFT<ND> nufft(gridOpts, newTraj, 1, nullptr);
  CxN<ND + 2> psf(nufft.ishape), xcor(nufft.ishape), temp(nufft.ishape);
  CxN<ND + 2> ones(AddBack(traj.matrix(), 1, 1));
  ones.setConstant(1.f);
  TOps::Pad<ND + 2> pad(ones.dimensions(), nufft.ishape);
  pad.forward(ones, xcor);
  FFT::Forward(xcor, FirstN<ND>(Sz3{0, 1, 2}));
  xcor.device(Threads::TensorDevice()) = xcor * xcor.conjugate();
  FFT::Adjoint(xcor, FirstN<ND>(Sz3{0, 1, 2}));
  float const scale = std::pow(Product(FirstN<ND>(psf.dimensions())), 1.5f) /
                      Product(traj.matrix()) / Product(traj.matrix());
  Cx3 q(nufft.oshape), input(nufft.oshape), value(nufft.oshape);
  Re3 weights(nufft.oshape);
  weights.setZero();
  Index const nB = basis ? basis->nB() : 1;
  std::vector<double> const norms(nB, 1.);
  for (Index b = 0; b < nB; ++b) {
    for (Index d = b; d < nB; ++d) {
      if (!ProductFeature(basis, b, d, q)) { continue; }
      input.device(Threads::TensorDevice()) = q.conjugate();
      nufft.adjoint(input, psf);
      temp.device(Threads::TensorDevice()) = psf * xcor;
      nufft.forward(temp, value);
      // The spatial kernel is real: reversed complex products are conjugates.
      Accumulate(weights, 0, value, basis, b, d, norms, scale, b == d ? 1.f : 2.f);
    }
  }
  return Invert(std::move(weights), basis, {norms}, nufft.covered(), λ).template chip<0>(0);
}

template auto KSpaceSingle(GridOpts<2> const &, TrajectoryN<2> const &, float const, Basis::CPtr) -> Re2;
template auto KSpaceSingle(GridOpts<3> const &, TrajectoryN<3> const &, float const, Basis::CPtr) -> Re2;

auto KSpaceMulti(Cx5 const &smaps, GridOpts<3> const &gridOpts, Trajectory const &traj, float const λ, Basis::CPtr basis) -> Re3
{
  CheckInputs(λ, basis);
  Log::Print("Precon", "Ong squared-overlap weights, basis rank {}, regularization {}", basis ? basis->nB() : 0, λ);
  Index const nB = basis ? basis->nB() : 1, nMB = smaps.dimension(3), nC = smaps.dimension(4);
  if (nMB != 1 && nMB != nB) { throw Log::Failure("Precon", "SENSE maps had basis dimension {}, expected 1 or {}", nMB, nB); }
  for (Index i = 0; i < smaps.size(); ++i) {
    if (!std::isfinite(smaps.data()[i].real()) || !std::isfinite(smaps.data()[i].imag())) {
      throw Log::Failure("Precon", "SENSE maps must be finite");
    }
  }
  Trajectory newTraj(traj.points() * 2.f, MulToEven(traj.matrix(), 2), traj.voxelSize() / 2.f);
  TOps::NUFFT<3> nufft(gridOpts, newTraj, 1, nullptr);
  Sz5 const mapShape = AddBack(FirstN<3>(smaps.dimensions()), 1, 1);
  TOps::Pad<5> pad(mapShape, nufft.ishape);
  Cx5 map(mapShape), fb(nufft.ishape), fd(nufft.ishape), xcor(nufft.ishape), psf(nufft.ishape), temp(nufft.ishape);
  Cx3 q(nufft.oshape), input(nufft.oshape), value(nufft.oshape);
  Re3 weights(nC, traj.nSamples(), traj.nTraces());
  weights.setZero();
  Index const N = Product(FirstN<3>(smaps.dimensions()));
  std::vector<std::vector<double>> norms(nC, std::vector<double>(nB, 0.));
  for (Index c = 0; c < nC; ++c) {
    for (Index b = 0; b < nB; ++b) {
      for (Index v = 0; v < N; ++v) { norms[c][b] += std::norm(DCx(smaps.data()[v + N * ((nMB == 1 ? 0 : b) + nMB * c)])); }
    }
  }
  float const scale = std::pow(Product(FirstN<3>(nufft.ishape)), 1.5f) / Product(traj.matrix());
  for (Index b = 0; b < nB; ++b) {
    for (Index d = b; d < nB; ++d) {
      if (!ProductFeature(basis, b, d, q)) { continue; }
      input.device(Threads::TensorDevice()) = q.conjugate();
      nufft.adjoint(input, psf);
      for (Index ci = 0; ci < nC; ++ci) {
        xcor.setZero();
        for (Index cj = 0; cj < nC; ++cj) {
          auto product = [&](Index k, Cx5 &out) {
            Index const mb = nMB == 1 ? 0 : k;
            map.device(Threads::TensorDevice()) = smaps.slice(Sz5{0, 0, 0, mb, ci}, mapShape) *
                                                  smaps.slice(Sz5{0, 0, 0, mb, cj}, mapShape).conjugate();
            pad.forward(map, out);
            FFT::Forward(out, Sz3{0, 1, 2});
          };
          product(b, fb);
          if (b == d || nMB == 1) {
            xcor.device(Threads::TensorDevice()) += fb * fb.conjugate();
          } else {
            product(d, fd);
            xcor.device(Threads::TensorDevice()) += fb * fd.conjugate();
          }
        }
        FFT::Adjoint(xcor, Sz3{0, 1, 2});
        temp.device(Threads::TensorDevice()) = psf * xcor;
        nufft.forward(temp, value);
        Accumulate(weights, ci, value, basis, b, d, norms[ci], scale, b == d ? 1.f : 2.f);
      }
    }
  }
  return Invert(std::move(weights), basis, norms, nufft.covered(), λ);
}

template <int NB> auto LoadKSpacePrecon(std::string const &fname, Index const nSamp, Index const nTrace, Sz<3 + NB> const shape)
  -> TOps::TOp<3 + NB>::Ptr
{
  Log::Print("Precon", "Loading supplied weights unchanged from {}", fname);
  HD5::Reader reader(fname);
  Index const o = reader.order(HD5::Keys::Weights);
  if (o == 2) {
    Re2 const w = reader.readTensor<Re2>(HD5::Keys::Weights);
    if (w.dimension(0) != nSamp || w.dimension(1) != nTrace) {
      throw Log::Failure("Precon", "Preconditioner dimensions on disk {}x{} did not match trajectory {}x{}", w.dimension(0),
                         w.dimension(1), nSamp, nTrace);
    }
    return std::make_shared<TOps::TensorScale<3 + NB, 1, NB>>(shape, w.cast<Cx>());
  } else if (o == 3) {
    Re3 const w = reader.readTensor<Re3>(HD5::Keys::Weights);
    if (w.dimension(0) != shape[0]) {
      throw Log::Failure("Precon", "Preconditioner on disk had {} channels, expected {}", w.dimension(0), shape[0]);
    }
    if (w.dimension(1) != nSamp || w.dimension(2) != nTrace) {
      throw Log::Failure("Precon", "Preconditioner dimensions on disk {}x{} did not match trajectory {}x{}", w.dimension(1),
                         w.dimension(2), nSamp, nTrace);
    }
    return std::make_shared<TOps::TensorScale<3 + NB, 0, NB>>(shape, w.cast<Cx>());
  } else {
    throw Log::Failure("Precon", "On-disk weights had order {}, expected 2 or 3", o);
  }
}

template <int ND, int NB> auto MakeKSpacePrecon(PreconOpts const      &opts,
                                                GridOpts<ND> const    &gridOpts,
                                                TrajectoryN<ND> const &traj,
                                                Basis::CPtr            basis,
                                                Index const            nC,
                                                Sz<NB> const           bshape) -> TOps::TOp<3 + NB>::Ptr
{
  if (!opts.basis && basis) { Log::Print("Precon", "Ignoring temporal basis for generated weights"); }
  auto const shape = Concatenate(Sz3{nC, traj.nSamples(), traj.nTraces()}, bshape);
  if (opts.type == "" || opts.type == "none") {
    Log::Print("Precon", "Using no preconditioning");
    return nullptr;
  } else if (opts.type == "single") {
    Re2 const w = KSpaceSingle(gridOpts, traj, opts.λ, opts.basis ? basis : nullptr);
    return std::make_shared<TOps::TensorScale<3 + NB, 1, NB>>(shape, w.cast<Cx>());
  } else if (opts.type == "multi") {
    throw Log::Failure("Precon", "Multichannel preconditioner requested without SENSE maps");
  } else {
    return LoadKSpacePrecon<NB>(opts.type, traj.nSamples(), traj.nTraces(), shape);
  }
}

template auto
MakeKSpacePrecon(PreconOpts const &, GridOpts<2> const &, TrajectoryN<2> const &, Basis::CPtr, Index const, Sz0 const)
  -> TOps::TOp<3>::Ptr;
template auto
MakeKSpacePrecon(PreconOpts const &, GridOpts<2> const &, TrajectoryN<2> const &, Basis::CPtr, Index const, Sz1 const)
  -> TOps::TOp<4>::Ptr;
template auto
MakeKSpacePrecon(PreconOpts const &, GridOpts<2> const &, TrajectoryN<2> const &, Basis::CPtr, Index const, Sz2 const)
  -> TOps::TOp<5>::Ptr;
template auto
MakeKSpacePrecon(PreconOpts const &, GridOpts<3> const &, TrajectoryN<3> const &, Basis::CPtr, Index const, Sz0 const)
  -> TOps::TOp<3>::Ptr;
template auto
MakeKSpacePrecon(PreconOpts const &, GridOpts<3> const &, TrajectoryN<3> const &, Basis::CPtr, Index const, Sz1 const)
  -> TOps::TOp<4>::Ptr;
template auto
MakeKSpacePrecon(PreconOpts const &, GridOpts<3> const &, TrajectoryN<3> const &, Basis::CPtr, Index const, Sz2 const)
  -> TOps::TOp<5>::Ptr;

template <int ND, int NB> auto MakeKSpacePrecon(PreconOpts const      &opts,
                                                GridOpts<ND> const    &gridOpts,
                                                TrajectoryN<ND> const &traj,
                                                Basis::CPtr            basis,
                                                Cx5 const             &smaps,
                                                Sz<NB> const           bshape) -> TOps::TOp<3 + NB, 3 + NB>::Ptr
{
  Index const nC = smaps.dimension(4);
  if (!opts.basis && basis) { Log::Print("Precon", "Ignoring temporal basis for generated weights"); }
  auto const  shape = Concatenate(Sz3{nC, traj.nSamples(), traj.nTraces()}, bshape);
  if (opts.type == "" || opts.type == "none") {
    Log::Print("Precon", "Using no preconditioning");
    return nullptr;
  } else if (opts.type == "single") {
    Re2 const w = KSpaceSingle(gridOpts, traj, opts.λ, opts.basis ? basis : nullptr);
    return std::make_shared<TOps::TensorScale<3 + NB, 1, NB>>(shape, w.cast<Cx>());
  } else if (opts.type == "multi") {
    Re3 const w = KSpaceMulti(smaps, gridOpts, traj, opts.λ, opts.basis ? basis : nullptr);
    return std::make_shared<TOps::TensorScale<3 + NB, 0, NB>>(shape, w.cast<Cx>());
  } else {
    return LoadKSpacePrecon<NB>(opts.type, traj.nSamples(), traj.nTraces(), shape);
  }
}

template auto
MakeKSpacePrecon(PreconOpts const &, GridOpts<3> const &, TrajectoryN<3> const &, Basis::CPtr, Cx5 const &, Sz<0> const)
  -> TOps::TOp<3, 3>::Ptr;
template auto
MakeKSpacePrecon(PreconOpts const &, GridOpts<3> const &, TrajectoryN<3> const &, Basis::CPtr, Cx5 const &, Sz<1> const)
  -> TOps::TOp<4, 4>::Ptr;
template auto
MakeKSpacePrecon(PreconOpts const &, GridOpts<3> const &, TrajectoryN<3> const &, Basis::CPtr, Cx5 const &, Sz<2> const)
  -> TOps::TOp<5, 5>::Ptr;

} // namespace rl
