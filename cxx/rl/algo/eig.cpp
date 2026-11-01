#include "eig.hpp"

#include "common.hpp"
#include "iter.hpp"

#include <Eigen/Eigenvalues>

namespace rl {

auto PowerMethodForward(std::shared_ptr<Ops::Op> A, std::shared_ptr<Ops::Op> P, Index const iterLimit) -> PowerReturn
{
  Log::Print("Power", "A'PA");
  Eigen::VectorXcf vec = Eigen::VectorXcf::Random(A->cols());
  float            val = ParallelNorm(vec);
  vec /= val;
  Iterating::Starting();
  for (auto ii = 0; ii < iterLimit; ii++) {
    if (P) {
      vec = A->adjoint(P->adjoint(A->forward(vec)));
    } else {
      vec = A->adjoint(A->forward(vec));
    }
    val = ParallelNorm(vec);
    vec.device(Threads::CoreDevice()) = vec / val;
    Log::Print("Power", "{} Eigenvalue {}", ii, val);
    if (Iterating::ShouldStop("Power")) { break; }
  }
  Iterating::Finished();
  return {val, vec};
}

auto PowerMethodAdjoint(std::shared_ptr<Ops::Op> A, std::shared_ptr<Ops::Op> P, Index const iterLimit) -> PowerReturn
{
  Log::Print("Power", "PAA'");
  Eigen::VectorXcf vec = Eigen::VectorXcf::Random(A->rows());
  float            val = ParallelNorm(vec);
  vec /= val;
  Iterating::Starting();
  for (auto ii = 0; ii < iterLimit; ii++) {
    if (P) {
      vec = P->adjoint(A->forward(A->adjoint(vec)));
    } else {
      vec = A->forward(A->adjoint(vec));
    }
    val = ParallelNorm(vec);
    vec.device(Threads::CoreDevice()) = vec / val;
    Log::Print("Power", "{} Eigenvalue {}", ii, val);
    if (Iterating::ShouldStop("Power")) { break; }
  }
  Iterating::Finished();
  return {val, vec};
}

auto LanczosMax(std::shared_ptr<Ops::Op> A, std::shared_ptr<Ops::Op> P, Index const iterLimit) -> float
{
  Log::Print("Lanczos", "λmax of A'PA, {} iterations", iterLimit);
  auto            &dev = Threads::CoreDevice();
  Eigen::VectorXcf v = Eigen::VectorXcf::Random(A->cols()), vprev = Eigen::VectorXcf::Zero(A->cols()), w(A->cols());
  v /= ParallelNorm(v);
  std::vector<float> α, β;
  float              b = 0.f, λ = 0.f;
  for (Index ii = 0; ii < iterLimit; ii++) {
    if (P) {
      w = A->adjoint(P->adjoint(A->forward(v)));
    } else {
      w = A->adjoint(A->forward(v));
    }
    float const a = std::real(v.dot(w));
    w.device(dev) = w - a * v - b * vprev;
    α.push_back(a);
    b = ParallelNorm(w);
    // Largest eigenvalue of the tridiagonal matrix so far
    Index const     n = α.size();
    Eigen::MatrixXf T = Eigen::MatrixXf::Zero(n, n);
    for (Index ij = 0; ij < n; ij++) {
      T(ij, ij) = α[ij];
      if (ij + 1 < n) { T(ij, ij + 1) = T(ij + 1, ij) = β[ij]; }
    }
    λ = Eigen::SelfAdjointEigenSolver<Eigen::MatrixXf>(T, Eigen::EigenvaluesOnly).eigenvalues().maxCoeff();
    Log::Print("Lanczos", "{} λmax {}", ii, λ);
    if (b <= 1.e-6f * λ) { break; } // Invariant subspace
    β.push_back(b);
    vprev.device(dev) = v;
    v.device(dev) = w / b;
  }
  return λ;
}

} // namespace rl
