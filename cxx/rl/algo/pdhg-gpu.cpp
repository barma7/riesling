#include "pdhg-gpu.hpp"

#include "../log/log.hpp"
#include "iter.hpp"

namespace rl::PDHG {

auto RunGPU(Cx5 const                                 &data,
            TOps::SENSENUFFTGPU const                 &op,
            gpu::PDHGSetup                             s,
            Opts const                                &opts,
            std::function<void(Index, Vector const &)> debug,
            Index const                                debugEvery) -> Vector
{
#ifdef RL_CUDA
  if (opts.adaptive) { throw Log::Failure("PDHG", "Adaptive steps are not implemented on the GPU"); }
  if (data.dimension(3) != 1) { throw Log::Failure("PDHG", "GPU PDHG: one slab only"); }
  s.data = opts.lad ? gpu::DataDual::LAD : (opts.weighted ? gpu::DataDual::WLS : gpu::DataDual::LS);
  s.maxIts = opts.imax;
  s.tol = opts.tol;
  s.lambdaE = opts.λE;
  s.ratio = opts.ratio;
  s.autoRatio = opts.autoRatio;
  s.termSteps = opts.termSteps;
  s.checkEvery = opts.checkEvery;
  s.nT = data.dimension(4);
  Index const                   nTP = data.dimension(0) * data.dimension(1) * data.dimension(2);
  std::vector<gpu::Cxf const *> b(s.nT);
  for (Index it = 0; it < s.nT; it++) {
    b[it] = reinterpret_cast<gpu::Cxf const *>(data.data() + it * nTP);
  }
  Log::Print("PDHG", "GPU {}{} terms, {} time point{}", opts.lad ? "LAD, " : (opts.weighted ? "weighted LS, " : ""),
             s.terms.size(), s.nT, s.nT > 1 ? "s" : "");
  Index const nI = s.nx * s.ny * s.nz * s.nB * s.nT;
  Vector      x(nI), xv;
  auto const  status = [](gpu::PDHGStatus const &st) {
    if (st.primal >= 0.f) {
      Log::Print("PDHG", "{:02d}: |x| {:4.3E} |Δx|/|x| {:4.3E} |P| {:4.3E} |D| {:4.3E}", st.it, st.normx, st.normdx / st.normx,
                 st.primal, st.dual);
    } else {
      Log::Print("PDHG", "{:02d}: |x| {:4.3E} |Δx|/|x| {:4.3E}", st.it, st.normx, st.normdx / st.normx);
    }
    return !Iterating::ShouldStop("PDHG");
  };
  std::function<void(int64_t, gpu::Cxf const *)> dbg = nullptr;
  if (debug) {
    dbg = [&](int64_t const it, gpu::Cxf const *xvh) {
      xv = Eigen::Map<Vector const>(reinterpret_cast<Cx const *>(xvh), s.tgv ? 4 * nI : nI);
      debug(it, xv);
    };
  }
  Iterating::Starting();
  gpu::RunPDHG(*op.device(), s, b, reinterpret_cast<gpu::Cxf *>(x.data()), status, dbg, debugEvery,
               [](std::string const &m) { Log::Print("PDHG", "{}", m); });
  Iterating::Finished();
  return x;
#else
  throw Log::Failure("PDHG", "riesling was built without CUDA (RL_CUDA)");
#endif
}

} // namespace rl::PDHG
