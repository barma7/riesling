#pragma once

#include <complex>
#include <cstdint>
#include <memory>

/*
 * SENSE + NUFFT (3D) on a CUDA device. Plain C++ interface (no Eigen, no CUDA types) so that it can be used from riesling's
 * Eigen code and compiled by nvcc on its own; see rl/op/sense-nufft-gpu.hpp for the operator built on it.
 *
 * Same operator as TOps::SENSENUFFT: the gridding coordinates, kernel (exponential of a semicircle, or top-hat), kernel
 * scale, apodization, padding and unitary centred FFT are those of the CPU code, so results agree to rounding. The FFT
 * shifts are folded into the passes before and after the FFT as checkerboard signs.
 */
namespace rl::gpu {

using Cxf = std::complex<float>;

struct SNUFFTSetup
{
  int64_t nx = 0, ny = 0, nz = 0; // Image matrix
  int64_t gx = 0, gy = 0, gz = 0; // Gridding matrix (even sizes)
  int64_t nB = 1, nC = 1, nMB = 1; // Basis images, channels, sets of maps (1 or nB)
  int64_t nSamp = 0, nTrace = 0;   // k-space is (nC, nSamp, nTrace), channel fastest

  int   kW = 4;           // Kernel width W: FullWidth = (W + 1) / 2 * 2 + 1 taps per dimension
  bool  tophat = false;   // Top-hat (nearest neighbour) kernel, 1 tap
  float beta = 0.f;          // Exponential of a semicircle parameter
  float kScale = 1.f;     // Kernel normalisation (rl::Kernel::scale)

  float const *coords = nullptr; // 3 x nSamp x nTrace grid coordinates (point * osamp + grid / 2), NaN = invalid sample
  Cxf const   *maps = nullptr;   // nx, ny, nz, nMB, nC
  float const *apo = nullptr;    // nx, ny, nz: apodization correction on the image grid

  Cxf const *basis = nullptr; // nB x nBS x nBT (riesling Basis::B), or nullptr for none (nB = 1)
  int64_t    nBS = 1, nBT = 1;
  float      basisScale = 1.f;

  float const *weights = nullptr; // k-space weights for the normal operator: (nSamp, nTrace) or (nC, nSamp, nTrace)
  bool         weightsPerChannel = false;

  int device = 0;
};

// Accumulated device time per stage (ms), for profiling
struct SNUFFTTimes
{
  double copyIn = 0, expand = 0, fft = 0, interp = 0, spread = 0, ifft = 0, collect = 0, copyOut = 0;
  long   calls = 0;
};

// Data term of a primal-dual (PDHG) iteration, see SNUFFT::dataDualD
enum class DataDual : int
{
  LS = 1,  // ½‖Ax - b‖², the k-space weights W only as the dual step (RIESLING's PDHG; Ong et al. 2019)
  WLS = 2, // ½‖Ax - b‖²_W, the objective of RIESLING's ADMM / LSMR
  LAD = 3  // ‖Ax - b‖₁ (least absolute deviations), W as the dual step
};

// Device memory held by host code (e.g. a solver's vectors), freed with the object
class DeviceBuffer
{
public:
  DeviceBuffer() = default;
  explicit DeviceBuffer(size_t bytes);
  ~DeviceBuffer();
  DeviceBuffer(DeviceBuffer &&o) noexcept;
  DeviceBuffer &operator=(DeviceBuffer &&o) noexcept;
  DeviceBuffer(DeviceBuffer const &) = delete;
  DeviceBuffer &operator=(DeviceBuffer const &) = delete;

  template <typename T> auto as() const -> T * { return static_cast<T *>(p_); }
  auto bytes() const -> size_t { return n_; }
  void upload(void const *h, size_t bytes);
  void download(void *h, size_t bytes) const;
  void zero();

private:
  void  *p_ = nullptr;
  size_t n_ = 0;
};

auto DeviceFreeBytes() -> size_t; // Free device memory now

// Page-lock host memory (faster copies to and from the device); false if not possible
auto PinHost(void *p, size_t bytes) -> bool;
void UnpinHost(void *p);

class SNUFFT
{
public:
  explicit SNUFFT(SNUFFTSetup const &s);
  ~SNUFFT();
  SNUFFT(SNUFFT const &) = delete;
  SNUFFT &operator=(SNUFFT const &) = delete;

  // Host pointers. x: nx, ny, nz, nB; y: nC, nSamp, nTrace
  void forward(Cxf const *x, Cxf *y, float s) const;       // y = s A x
  void adjoint(Cxf const *y, Cxf *x, float s) const;       // x = s Aᴴ y
  void normal(Cxf const *x, Cxf *y, float s) const;        // y = s Aᴴ W A x, W the k-space weights (identity if none)
  void gridAdjoint(Cxf const *y, Cxf *grid) const;         // Spreading only: grid (gx, gy, gz, nB, nC), for accuracy checks
  auto nValid() const -> int64_t;                          // Samples inside the grid (the others are ignored)
  auto deviceBytes() const -> size_t;                      // Device memory held
  auto times() const -> SNUFFTTimes;
  void resetTimes() const;

  /* Device pointers, for solvers that keep their vectors on the device. Images as above; k-space in "device order":
   * channel-major, nC x nK(), the valid samples in the operator's internal order. Asynchronous (default stream).
   */
  auto nK() const -> int64_t;                                 // Samples per channel in device order (= nValid())
  auto nC() const -> int64_t;                                 // Channels
  void toDeviceOrder(Cxf const *k, Cxf *dk) const;            // Host k (nC, nSamp, nTrace) -> device dk
  void fromDeviceOrder(Cxf const *dk, Cxf *k) const;          // Device dk -> host k (the invalid samples 0)
  void forwardD(Cxf const *x, Cxf *y, float s) const;         // y = s A x
  void adjointD(Cxf const *y, Cxf *x, float s, bool add) const; // x = s Aᴴ y, or x += s Aᴴ y with add
  void normalD(Cxf const *x, Cxf *y, float s) const;          // y = s Aᴴ W A x
  /* PDHG dual step of the data term, fused with the forward operator: u = y + σ W (A x̂ - b), then y = u / (1 + σW) (LS),
   * u / (1 + σ) (WLS) or u / max(|u|, 1) (LAD). W: the k-space weights of the setup (1 without).
   */
  void dataDualD(Cxf const *xhat, Cxf *y, Cxf const *b, float sigma, DataDual kind) const;
  void sync() const; // Wait for the device
  // Σ |y|² / W over a k-space vector in device order: the norm of a data-term dual in the metric of its dual step σW
  auto dualNorm2D(Cxf const *y) const -> double;
  auto weightedNorm2D(Cxf const *y) const -> double; // Σ W |y|² (|b|²_W)
  void weightD(Cxf *y) const;                        // y = W y in device order

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace rl::gpu
