Design notes: speed, memory and solvers
========================================

This page explains the main changes made to reduce the run time and memory of regularized 3D non-Cartesian reconstructions (``recon-rlsq``). For each one it gives what was wrong or slow in the earlier code, what changed, and the literature behind the choice. It is meant for developers, and for users who want to know what the new options do before relying on them. The options themselves are documented in :doc:`recon`.

The changes fall in three groups:

- **Same result, less time or memory.** Faster kernels, fewer passes over memory, an optional GPU operator. These keep the objective and agree with the previous code to floating-point rounding; they are tested against it.
- **Fixes.** Places where the code did not compute what it stated: the adjoints of some regularizer transforms, the PDHG step sizes, the dimensions of one regularizer. These change results, on purpose.
- **Solver options.** Ways to reach the same solution sooner: the ADMM x-update solver, PDHG step-size selection and stopping, a GPU-resident PDHG. Most are opt-in. Two defaults changed: the ADMM x-update solver and the residual-balancing tolerance ``--mu``.

.. contents::
   :local:
   :depth: 1

Principles
----------

*Keep the objective.* A faster reconstruction that converges to a different image is a different method. The GPU operator therefore reproduces the CPU operator (the same kernel, coordinates, apodization and FFT scaling) instead of calling an existing GPU NUFFT library. The GPU regularizers use the CPU's transforms and coefficients. Solver changes are judged by how fast they approach the solution of the same problem, not by how close they stay to the old iterates.

*Most of the time is memory traffic.* The operations of an iteration (SENSE products, apodization, finite differences, proximal operators, vector updates) do a few floating-point operations per byte, so memory bandwidth sets their speed (the roofline model [Williams2009]_). Several of them ran an order of magnitude below that bound because of generic tensor expressions, extra passes or temporaries. The fixes fuse passes, keep data in cache, and allocate large vectors once.

*Derive solver parameters from the problem.* Step sizes and ratios that were assumed (operator norms of 1, a primal/dual step ratio of 1) are now computed or estimated from the operator and the data. They then carry over to other trajectories, bases, regularizers and data scales.

*Test exactness.* Every new kernel is checked against the code it replaces or against an independent reference: dot tests for adjoints, and CPU against GPU for every operator and regularizer step (`Testing`_).

CPU operators
-------------

**Build flags.** Release builds passed no ``-march`` flag, so Eigen and the DUCC FFT compiled for SSE2 only, without FMA or AVX. The ``RL_ARCH`` CMake option (``bootstrap.sh -a ARCH``) passes ``-march`` to the C++ sources, for example ``native`` (this machine only), ``x86-64-v3`` (AVX2 and FMA) or ``x86-64-v4`` (AVX-512). The default, empty, keeps the portable build. On an AVX-512 workstation an iteration ran about a third faster.

**Gridding: load balance.** Each subgrid's list of samples was processed by one thread, with lists assigned round-robin. Centre-out trajectories (radial, UTE, ZTE) put a large share of all samples in the subgrids at the centre of k-space; in one 3D radial dataset a single subgrid held over a tenth of them. One thread then did most of the work, and going beyond about eight threads gave no speed-up. Long lists are now split into chunks of at most 32768 samples, which are handed to threads dynamically, largest first. FINUFFT's spreader is built on the same idea of bounded subproblems [Barnett2019]_. A side effect is better accuracy. The adjoint now sums the k-space centre, where every spoke contributes, in chunks rather than in one long sequence. This lowered the rounding error at the central grid point by more than two orders of magnitude, as expected from blocked summation [Higham2002]_.

**Gridding: vectorization.** For every sample, each of the 5×5×5 kernel taps was applied as a separate short loop per basis vector and channel, over strided planes of the subgrid. Little of this vectorized. The per-thread subgrid is now stored transposed, with all basis vectors and channels of a grid point next to each other. Each tap is then one contiguous loop over all basis vectors and channels, and the basis is applied once per sample. Zero basis entries are still skipped, so sparse bases (e.g. time frames) keep their cost.

**Fused SENSE + NUFFT.** The 3D multi-channel encoding operator was a composition of SENSE and the NUFFT. Every application allocated the channel images as a temporary (image × basis × channels, as large as the gridding workspace), filled them, then copied them into the oversampled grid while applying the apodization with a generic broadcast-and-pad expression. ``TOps::SENSENUFFT`` (``rl/op/sense-nufft.cpp``) instead writes image × maps × apodization straight into the oversampled grid. Its adjoint crops, apodizes and sums over channels in one pass per row. The products are taken in the same order as before, so results agree to rounding, and the missing temporary lowers peak memory. ``--lowmem`` and ``--decant`` keep their own operators.

**Kernels.** Three operations ran 10–20× below memory bandwidth and are now one-pass loops over blocks of voxels with the same arithmetic. Other shapes keep the generic code.

- The SENSE adjoint, a reduction over channels.
- The 3D finite differences of TV, TV2 and TGV, which made one pass per difference: 3 for the gradient, 9 for TGV's symmetrized gradient.
- The block L2 proximal operator of the isotropic norms, which did generic index arithmetic for every element.

**Threads, allocations, norms.**

- ADMM held a ``const`` copy of the thread-pool device. Eigen uses its threaded evaluator only with a non-const device, so every vector operation in ADMM silently ran on one thread.
- ADMM allocated its work vectors (several image- and k-space-sized ones) in every outer iteration. The kernel serves allocations that large with fresh pages, zeroed on first touch, so that cost recurred in every iteration. The vectors are now allocated once and shared by the x-update and the z/u-update, so peak memory does not grow.
- Norms used Eigen's ``stableNorm``, several times slower than one pass over memory. ``ParallelNorm`` sums squares in single precision over blocks and adds the block sums in double [Higham2002]_. It falls back to ``stableNorm`` if a block sum overflows.

ADMM
----

ADMM [Boyd2011]_ alternates three steps:

- an x-update, a least-squares problem in the data term and the penalties :math:`\rho \|F_i x - z_i + u_i\|^2`;
- the proximal operators of the regularizers (the z-update);
- the dual updates (u).

**x-update: CG on the normal equations.** The x-update was solved with LSMR [Fong2011]_ on the stacked system :math:`[A; \sqrt{\rho} F_1; \dots]`. LSMR keeps vectors in the range of that system, so each of its work vectors also holds every regularizer output: for TGV that is nine image-sized fields per vector. Its warm start costs one more application of :math:`A` per outer iteration. In single precision it also stalled well above the residual that CG reaches. The default is now CG on the normal equations of the same subproblem:

.. math::

   \Big(A^H W A + \rho \sum_i F_i^H F_i\Big)\, x = A^H W b + \rho \sum_i F_i^H (z_i - u_i)

Here :math:`W` are the k-space weights, so the minimiser is unchanged.

- CG needs only image-sized vectors.
- :math:`A^H W b` is computed once.
- :math:`A^H W A x` is updated from the products CG computes anyway, so the warm start is free.

The usual objection to CG on the normal equations is that it squares the condition number, which limits the accuracy it can reach in floating point; that is the reason LSQR and LSMR exist [Paige1982]_. It matters little here. The k-space preconditioner [Ong2020]_ keeps the system well conditioned, and the x-update runs only a few warm-started iterations, far from that limit. ``--admm-inner=lsmr`` selects the previous solver.

The x-update is inexact by design: a fixed number of warm-started CG iterations, set by ``--max-its0`` and ``--max-its1``. ADMM converges with inexact x-updates whose errors are summable [Eckstein1992]_; in practice a fixed number of warm-started iterations works. That number matters: in our tests, 12 inner iterations per outer iteration reached a given accuracy several times sooner than 3. More than 12 was not tested; the previous default, 64, makes each outer iteration several times more expensive than 12. The default is now 12 (``--max-its0``, ``--max-its1``).

**Residual balancing tolerance.** Residual balancing [He2000]_ [Wohlberg2017]_ rescales the penalty ρ whenever the primal and dual residuals differ by more than a factor μ. With μ = 1.2 and an inexact x-update it fired in almost every outer iteration. Each time ρ and the scaled duals u were rescaled, the residuals oscillated and progress towards the solution slowed. μ = 10 is the value Boyd et al. suggest ([Boyd2011]_, §3.4.1) and is now the default. Balancing still corrects a poor starting ρ but no longer chatters. ``--mu=1.2`` restores the old behaviour. Only the path to the solution changes, not the objective.

**Time points one at a time.** ``--split-time`` handles time points (e.g. echoes) that no regularizer couples: none of ``--tvt``, ``--llr``, ``t`` in ``--iso`` or wavelets over time. Such problems are a sum of independent problems, and ADMM used to hold the state of all of them at once. With the option, each time point is solved on its own with a one-time-point encoding operator, so the solver state of only one is in memory. The data scale is still computed from all time points. The option is available for ADMM and PDHG, and refused with off-resonance correction or regularizers that couple time.

Basis-aware k-space weights
--------------------------

The k-space weight generator follows the diagonal row formula in section VI of [Ong2020]_,

.. math::

   p_i = \frac{\|a_i\|^2}{\sum_j |a_i^H a_j|^2}.

Including a temporal basis in the old NUFFT calculation introduced its signed overlap
:math:`G_{ij}` into the sum, followed by an absolute value. Negative overlaps could
cancel positive contributions and produce extremely large reciprocal weights. The
formula requires :math:`|G_{ij}|^2` and division of the denominator by :math:`G_{ii}`.
Taking an absolute value or clipping the resulting weights does not correct that formula.
Single-channel generation retains the historical NUFFT/autocorrelation approximation
over the nominal trajectory matrix. Increasing ``--fov`` does not change that nominal
support, so these are not exact row-formula weights for every enlarged-FoV reconstruction
operator. This fix corrects temporal overlaps without changing that spatial approximation.

The implementation expands the squared overlap into products
:math:`q_{bd}(i)=B_b(i)\overline{B_d(i)}`. Each product uses the same scalar NUFFT
workspace; off-diagonal pairs contribute twice their real part, including for complex
bases. Exactly zero products are skipped. This avoids allocating a grid for every
product, at the cost of more NUFFT passes for dense bases. The encoding basis and
its normalization are unchanged. Multichannel generation also includes coil row norms
and cross-correlations between coefficient-dependent sensitivity maps.

**Objective compatibility.** In ADMM/LSMR and ``--pdhg-weighted``, these weights are
:math:`W` in the data objective, so correcting them can change the minimizer and the
regularization balance. In unweighted PDHG they are solver step sizes. This fix does
not introduce an independent solver preconditioner for a fixed weighted objective.
Use ``--precon=file`` to retain an existing objective's weights, or
``--precon-no-basis`` to generate weights without temporal overlaps. The latter still
uses the basis for reconstruction; it is not an exact restoration of every historical
version's clipping and regularization behavior.

Tests compare tiny explicit complex Fourier encoding matrices against both generators,
including negative overlaps, identity frames, row scaling, and coefficient-dependent
coil maps. Correct weights alone do not establish faster convergence or quantitative
map accuracy; those require comparisons with the intended objective held fixed.

Fixes to the regularizers
-------------------------

**Exact adjoints.** The discrete divergence must be the negative adjoint of the discrete gradient [Chambolle2004]_. The backward differences used by the divergence and by TGV's symmetrized gradient weighted interior points by *s* instead of the forward operator's *s*/2, so both adjoints were about twice the true ones (a dot test gives 1.92 instead of 1). Two consequences:

- ADMM's x-update used the wrong transpose, so its fixed point weighted TGV's second-order term at about twice the nominal λ.
- PDHG assumes the true adjoint in its iteration and its step sizes. With a mismatched adjoint, it is not guaranteed to converge to the solution of the stated problem [Lorenz2023]_.

The two Laplacians of ``--tv2`` apply their stencil at interior points only, but used the forward operator as the adjoint, which is not the transpose next to the boundary. All of these are now exact (``rl/op/grad.cpp``, ``rl/op/laplacian.cpp``) and covered by dot tests. **This changes the TGV objective, and TV2's at the boundary, so λ values tuned with earlier versions may need re-tuning.**

**TGV with other regularizers.** The regularizer setup returned as soon as it had built TGV's terms, so any other regularizer given with ``--tgv`` was silently ignored. With TGV the unknowns are :math:`[x; v]` [Bredies2010]_ [Knoll2011]_; the other terms now act on the :math:`x` part.

**Dimensions of the L1 norms.** With ``--l1`` and ``--iso``: images are ordered (i, j, k, basis, time). ``--iso=b`` normed dimension 4 (time) instead of 3 (basis), and ``t`` and ``bt`` named a dimension 5 that images do not have. Now ``b`` = 3 and ``t`` = 4, as the other regularizers use them. Results of ``--l1 --iso`` from earlier versions were not what the option describes.

**Smaller fixes.**

- ``op-grid --fwd`` built the wrong operator for Cartesian input.
- Log durations printed milliseconds without zero padding, so 1.041 s appeared as "1.41 seconds".
- The DUCC dependency is now fetched from its author's GitHub repository, since the original archive server stopped serving anonymous downloads.

GPU
---

The GPU code is an optional build (``-DRL_CUDA=ON``, see the README). It is used with ``--gpu`` by the recon commands, for 3D single-slab recons without ``--lowmem`` or ``--decant``.

**SENSE + NUFFT on the device.** There was no usable GPU path before; the experimental ``gewurz`` tool uses an explicit DFT, far too slow for 3D data. ``rl/gpu/snufft.cu`` implements the fused operator and reproduces the CPU's to rounding:

- the same gridding coordinates and rounding to the grid;
- the same exponential-of-semicircle kernel [Barnett2021]_ and normalization;
- the CPU's apodization weights;
- the same padding;
- a unitary centred FFT (cuFFT), with the FFT shifts folded into the neighbouring passes.

Spreading (the adjoint of the interpolation) does not use one atomic addition per kernel tap: for centre-out trajectories that is slow, and inaccurate at the k-space centre, where around 10⁶ contributions land on each grid point. Instead the samples are sorted by tile of grid cells and split into bins of bounded size, and each thread gathers the contributions to its grid points in registers. It follows the same idea as cuFINUFFT's load-balanced, blocked spreading [Shih2021]_. The result is at least as accurate as the CPU's against a double-precision sum. The normal operator :math:`A^H W A` runs in one device pass, so ADMM's iterations move only image-sized vectors over PCIe, from page-locked host memory.

**ADMM with the GPU operator.** For full-size 3D recons with a basis, ADMM's state (the unknowns, z and u for every regularizer, the CG vectors) does not fit in typical GPU memory, so it stays in host memory. Each CG iteration computes :math:`A^H W A p` on the device while the CPU computes :math:`\rho \sum_i F_i^H F_i p`; both only read :math:`p`. With the operator on the GPU, the CPU work on the ADMM and regularizer vectors becomes the limit.

**GPU-resident PDHG** (``--pdhg --gpu``). PDHG [Chambolle2011]_ has no inner solve. Its state is the unknowns, their extrapolation and one dual variable per term (k-space-sized for the data term). For problems that fit in device memory, everything can stay on the device, and no vectors cross PCIe, which is an order of magnitude slower than device memory. Design (``rl/gpu/pdhg.cu``):

- The data term's dual step is fused into the last pass of the forward operator, so it needs no k-space temporary. The primal step accumulates :math:`-\tau A^H y` inside the adjoint.
- Each regularizer term is one kernel. It evaluates the transform on the fly (gradient, time gradient, TGV's two terms, the 27-point Laplacian) and applies the conjugate proximal operator: L1, L1 of the imaginary part, or L2 over any combination of the basis, time and gradient dimensions.
- LLR uses one warp per window, with a one-sided Jacobi SVD [Hestenes1958]_ [Demmel1992]_ of the small patch matrix in shared memory. Jacobi is accurate for small matrices and parallelizes across the column pairs.
- Wavelets use one thread block per line, with the line in shared memory: the periodic Daubechies transform of ``TOps::Wavelets``.
- The memory needed is checked before anything is allocated, and the error message says what reduces it. Roughly, the iteration needs two copies of the unknowns, one dual per term, and two k-space vectors (data and dual). For TGV that is 17 image-sized vectors plus the two k-space vectors.
- The device time per phase of the iteration is logged at the end, so no profiler is needed to see where the time goes. Typically the data term takes most of it.

Every regularizer of ``recon-rlsq`` (TGV, TV, TV2, TVT, L1 with or without ``--iso``, L1 of the imaginary part, LLR, wavelets, alone or combined) runs on the device. Ten iterations agree with the CPU PDHG to about 2·10⁻⁶. Least squares, weighted least squares and least absolute deviations (``--lad``) are supported.

PDHG step sizes and stopping
----------------------------

RIESLING's PDHG solves :math:`\min_x \sum_i g_i(K_i x)`, with :math:`K_0 = A` and :math:`g_0` the data term, preconditioned in k-space [Ong2020]_. Each iteration is:

.. math::

   y_i \leftarrow \mathrm{prox}_{\sigma_i g_i^*}\big(y_i + \sigma_i K_i \hat x\big), \qquad
   x \leftarrow x - \tau \sum_i K_i^H y_i, \qquad
   \hat x \leftarrow 2x - x_\text{old}

It converges if :math:`\|\Sigma^{1/2} K T^{1/2}\| \le 1` [Chambolle2011]_ [Pock2011]_, where :math:`K` stacks the :math:`K_i` and :math:`\Sigma`, :math:`T` are the dual and primal steps. With scalar steps this condition reads :math:`\sigma \tau L \le 1` with :math:`L \ge \|K\|^2`.

**Operator norm.** :math:`L` was :math:`\lambda_E` plus the number of regularizers, with :math:`\lambda_E = 1` by default. That assumes every regularizer transform has norm 1 and that the preconditioned encoding operator has :math:`\lambda_\max = 1`. Neither holds in general:

- With a temporal basis, :math:`\lambda_\max(A^H W A)` can be well above 1.
- TGV's first term :math:`\nabla x - v` has :math:`\lambda_\max = 2`.

Steps that are too large void the convergence guarantee. Now :math:`L = \lambda_E + \sum_i \nu_i`. Each regularizer carries an upper bound :math:`\nu_i` of :math:`\lambda_\max(T_i^H T_i)`: 1 for the identity and the scaled gradients, 2 and 0.25 for TGV's two terms, 2.01 for TV2. :math:`\lambda_E` is estimated with about 20 Lanczos iterations and a 2% margin. For an operator whose spectrum is dense near the top, Lanczos converges much faster than the power method [Kuczynski1992]_. Precomputing the eigenvalue with ``riesling eig`` is no longer needed; ``--lambda-E`` still overrides the estimate.

**Weighted data term.** ADMM minimises :math:`\tfrac12\|Ax-b\|_W^2 + R(x)`. PDHG minimised :math:`\tfrac12\|Ax-b\|^2 + R(x)` and used :math:`W` only as the dual step size of the data term. The two solvers therefore solved different problems, and λ values did not carry over between them. ``--pdhg-weighted`` makes PDHG solve ADMM's problem. With the dual step :math:`\sigma W`, the conjugate proximal step of :math:`\tfrac12\|z-b\|_W^2` is :math:`y \leftarrow (y + \sigma W(A\hat x - b)) / (1 + \sigma)`.

**Step ratio.** The condition above fixes only the product :math:`\sigma\tau`; the ratio :math:`\tau/\sigma` is free, and with constant steps PDHG is sensitive to it. Chambolle and Pock bound the error after :math:`k` iterations by :math:`(\|x^*-x_0\|^2/\tau + \|y^*-y_0\|^2/\sigma)/k` [Chambolle2011]_. With :math:`\sigma\tau = 1/L` held fixed and a start from zero, this bound is smallest for :math:`\tau/\sigma = (\|x^*\|/\|y^*\|)^2`. That depends on the data scale, λ and the terms, so the ratio of 1 used before can be far off: in our tests the best ratio was between 10 and 100. ``--pdhg-ratio=R`` sets it by hand. ``--pdhg-auto-ratio`` chooses it in two stages:

#. *Start, from the data.* :math:`\|x^*\|` is estimated from one exact line-search step of the data term along :math:`A^H W b`, and the data dual from that step's residual. The regularizer duals are bounded by the radius of the set their conjugate proximal operator projects onto, e.g. :math:`\lambda^2 n` for an L1 term on :math:`n` elements.
#. *Updates, from the iterates.* At iterations 50, 100, 200, … the ratio moves halfway, in log space, towards :math:`(\|x_k\|/\|y_k\|)^2`, by at most a factor of 2. This is the "primal weight" update of PDLP [Applegate2021]_.

The iterates are not used from the start because they estimate the norms badly while the ratio is far off. A ratio that is too small keeps :math:`x` small, which pushes the estimate further down; started from 1, the estimate collapsed.

**Per-term step sizes** (``--pdhg-term-steps``, GPU only). The blocks of :math:`K` have very different norms. In TGV, for example, the symmetrized gradient of :math:`v` has :math:`\lambda_\max(T^H T) = 0.25` against 2 for :math:`\nabla x - v`, and the data block has norm :math:`\sqrt{\lambda_E}`. One σ and one τ sized for the largest block are too small for the others. Pock and Chambolle's diagonal preconditioning [Pock2011]_ chooses steps from the entries of :math:`K`. Applied to blocks with their α = 1 rule, it takes an upper bound :math:`n_{ij}` of the norm of each block (dual term *i*, primal block *j*) and sets :math:`\sigma_i = 1/\sum_j n_{ij}` and :math:`\tau_j = 1/\sum_i n_{ij}`. By the Schur test these steps satisfy the convergence condition without a norm of the whole stacked operator. The global ratio still applies on top (:math:`\sigma_i/\sqrt R`, :math:`\tau_j\sqrt R`), because it reflects the size of the solution, not of :math:`K`. Per-term steps with the ratio left at 1 were slow; use them together with ``--pdhg-auto-ratio``.

**Stopping test** (``--pdhg-check=N``, GPU only). PDHG stopped when :math:`\|\Delta x\|/\|x\|` fell below a tolerance. How close that is to the solution depends on the convergence rate: with poor steps the iterates move slowly while still far away. The test also ignores the duals. ADMM stops on its primal and dual residuals, and PDHG has the same pair [Goldstein2015]_:

- *Primal residual.* There is no proximal operator on :math:`x`, so optimality is :math:`K^H y = 0`. The primal step itself gives :math:`K^H y_k = T^{-1}(x_{k-1} - x_k)`, so this residual is free.
- *Dual residual.* :math:`\Sigma^{-1}(y_{k-1} - y_k) + K(\hat x - x)` follows from each proximal step's own optimality condition. It is computed inside the dual-step kernels, so it needs no extra memory, only one extra forward operator for the data term.

Both are taken relative to :math:`\|A^H W b\|` and :math:`\|b\|_W`, which the data fix, so one tolerance fits different problems (as PDLP's relative residuals). The residuals are computed every *N* iterations, and the run stops when both are below ``--pdhg-tol``. With *N* = 10 they cost 2–3% of the run time. In our tests the default tolerance of 10⁻⁴ stopped about 10⁻³ (relative) from the converged solution, comparable with ADMM's ``--eps=1e-3``. ``--pdhg-check=0``, the default, keeps the plain test at no cost; ``--pdhg-tol=0`` runs all ``--max-its`` iterations.

Choosing a solver
-----------------

- **ADMM** (the default) is robust and has few parameters. Its state lives in host memory, so it handles any problem size; ``--gpu`` moves the encoding operator to the GPU. Tune the number of inner CG iterations (``--max-its1``) and use ``--split-time`` when the time points are independent.
- **PDHG on the GPU** suits problems whose state fits in device memory. In our tests on 3D radial recons with TGV it reached a given accuracy 1.3–3× sooner than ADMM with the GPU operator, and used a fraction of ADMM's host memory. Suggested options: ``--pdhg --gpu --pdhg-weighted --pdhg-auto-ratio --pdhg-term-steps``. Add ``--pdhg-check=10`` to stop on accuracy, or, for repeated recons of the same kind, use a tuned ``--pdhg-ratio`` and a fixed ``--max-its``.
- **PDHG on the CPU** has the corrected step sizes, ``--pdhg-weighted``, ``--pdhg-ratio`` and ``--pdhg-auto-ratio``. It also keeps ``--adaptive`` steps and ``--pdhg-restart``, which the GPU PDHG does not have.

Testing
-------

The tests are in ``cxx/test`` (``riesling-tests``; the GPU tests are tagged ``[gpu]`` and built with ``RL_CUDA``):

- ``op/grad-adjoint.cpp``: dot tests of every regularizer transform (gradient, divergence, symmetrized gradient, time gradient, both Laplacians, the TGV terms as the recon builds them), their norm bounds, and the Lanczos estimate.
- ``op/grad-fused.cpp``, the L2 prox and gridding tests: the fused kernels against the generic code.
- ``op/sense-nufft.cpp``: the fused SENSE + NUFFT against the composition of the two operators.
- ``op/sense-nufft-gpu.cpp``: the GPU operator against the CPU's (forward, adjoint, normal operator, the device interface and the data-term dual steps).
- ``op/pdhg-gpu.cpp``: every GPU transform × proximal operator dual step and every adjoint against the CPU's, plus LLR and wavelets.

Known limitations
-----------------

- ``--pdhg-term-steps`` and ``--pdhg-check`` are GPU only; ``--adaptive`` and ``--pdhg-restart`` are CPU only, and ``--pdhg-auto-ratio`` does not combine with them.
- The GPU PDHG needs the whole problem in device memory. Large bases with TGV may not fit; ``--split-time`` or ADMM with ``--gpu`` are the alternatives.
- The CG x-update runs a fixed number of iterations (``--max-its0``, ``--max-its1``); the LSMR tolerances do not apply to it.
- One GPU, 3D single-slab recons only.

References
----------

.. [Applegate2021] D. Applegate et al., "Practical large-scale linear programming using primal-dual hybrid gradient", NeurIPS 2021. `arXiv:2106.04756 <https://arxiv.org/abs/2106.04756>`_
.. [Barnett2019] A. H. Barnett, J. Magland, L. af Klinteberg, "A parallel nonuniform fast Fourier transform library based on an 'exponential of semicircle' kernel", SIAM J. Sci. Comput. 41(5), C479–C504, 2019. `doi:10.1137/18M120885X <https://doi.org/10.1137/18M120885X>`_
.. [Barnett2021] A. H. Barnett, "Aliasing error of the exp(β√(1−z²)) kernel in the nonuniform fast Fourier transform", Appl. Comput. Harmon. Anal. 51, 1–16, 2021. `doi:10.1016/j.acha.2020.10.002 <https://doi.org/10.1016/j.acha.2020.10.002>`_
.. [Boyd2011] S. Boyd, N. Parikh, E. Chu, B. Peleato, J. Eckstein, "Distributed optimization and statistical learning via the alternating direction method of multipliers", Found. Trends Mach. Learn. 3(1), 1–122, 2011. `doi:10.1561/2200000016 <https://doi.org/10.1561/2200000016>`_
.. [Bredies2010] K. Bredies, K. Kunisch, T. Pock, "Total generalized variation", SIAM J. Imaging Sci. 3(3), 492–526, 2010. `doi:10.1137/090769521 <https://doi.org/10.1137/090769521>`_
.. [Chambolle2004] A. Chambolle, "An algorithm for total variation minimization and applications", J. Math. Imaging Vis. 20, 89–97, 2004. `doi:10.1023/B:JMIV.0000011325.36760.1e <https://doi.org/10.1023/B:JMIV.0000011325.36760.1e>`_
.. [Chambolle2011] A. Chambolle, T. Pock, "A first-order primal-dual algorithm for convex problems with applications to imaging", J. Math. Imaging Vis. 40, 120–145, 2011. `doi:10.1007/s10851-010-0251-1 <https://doi.org/10.1007/s10851-010-0251-1>`_
.. [Demmel1992] J. Demmel, K. Veselić, "Jacobi's method is more accurate than QR", SIAM J. Matrix Anal. Appl. 13(4), 1204–1245, 1992. `doi:10.1137/0613074 <https://doi.org/10.1137/0613074>`_
.. [Eckstein1992] J. Eckstein, D. P. Bertsekas, "On the Douglas–Rachford splitting method and the proximal point algorithm for maximal monotone operators", Math. Program. 55, 293–318, 1992. `doi:10.1007/BF01581204 <https://doi.org/10.1007/BF01581204>`_
.. [Fong2011] D. C.-L. Fong, M. Saunders, "LSMR: An iterative algorithm for sparse least-squares problems", SIAM J. Sci. Comput. 33(5), 2950–2971, 2011. `doi:10.1137/10079687X <https://doi.org/10.1137/10079687X>`_
.. [Goldstein2015] T. Goldstein, M. Li, X. Yuan, "Adaptive primal-dual splitting methods for statistical learning and image processing", NeurIPS 2015, 2089–2097. `proceedings <https://papers.nips.cc/paper/2015/hash/cd758e8f59dfdf06a852adad277986ca-Abstract.html>`_
.. [He2000] B. S. He, H. Yang, S. L. Wang, "Alternating direction method with self-adaptive penalty parameters for monotone variational inequalities", J. Optim. Theory Appl. 106, 337–356, 2000. `doi:10.1023/A:1004603514434 <https://doi.org/10.1023/A:1004603514434>`_
.. [Hestenes1958] M. R. Hestenes, "Inversion of matrices by biorthogonalization and related results", J. SIAM 6(1), 51–90, 1958. `doi:10.1137/0106005 <https://doi.org/10.1137/0106005>`_
.. [Higham2002] N. J. Higham, *Accuracy and Stability of Numerical Algorithms*, 2nd ed., SIAM, 2002, ch. 4 (summation). `doi:10.1137/1.9780898718027 <https://doi.org/10.1137/1.9780898718027>`_
.. [Knoll2011] F. Knoll, K. Bredies, T. Pock, R. Stollberger, "Second order total generalized variation (TGV) for MRI", Magn. Reson. Med. 65(2), 480–491, 2011. `doi:10.1002/mrm.22595 <https://doi.org/10.1002/mrm.22595>`_
.. [Kuczynski1992] J. Kuczyński, H. Woźniakowski, "Estimating the largest eigenvalue by the power and Lanczos algorithms with a random start", SIAM J. Matrix Anal. Appl. 13(4), 1094–1122, 1992. `doi:10.1137/0613066 <https://doi.org/10.1137/0613066>`_
.. [Lorenz2023] D. A. Lorenz, F. Schneppe, "Chambolle–Pock's primal-dual method with mismatched adjoint", Appl. Math. Optim. 87, 22, 2023. `doi:10.1007/s00245-022-09933-5 <https://doi.org/10.1007/s00245-022-09933-5>`_
.. [Ong2020] F. Ong, M. Uecker, M. Lustig, "Accelerating non-Cartesian MRI reconstruction convergence using k-space preconditioning", IEEE Trans. Med. Imaging 39(5), 1646–1654, 2020. `doi:10.1109/TMI.2019.2954121 <https://doi.org/10.1109/TMI.2019.2954121>`_
.. [Paige1982] C. C. Paige, M. A. Saunders, "LSQR: An algorithm for sparse linear equations and sparse least squares", ACM Trans. Math. Softw. 8(1), 43–71, 1982. `doi:10.1145/355984.355989 <https://doi.org/10.1145/355984.355989>`_
.. [Pock2011] T. Pock, A. Chambolle, "Diagonal preconditioning for first order primal-dual algorithms in convex optimization", ICCV 2011, 1762–1769. `doi:10.1109/ICCV.2011.6126441 <https://doi.org/10.1109/ICCV.2011.6126441>`_
.. [Shih2021] Y. Shih, G. Wright, J. Andén, J. Blaschke, A. H. Barnett, "cuFINUFFT: a load-balanced GPU library for general-purpose nonuniform FFTs", IEEE IPDPS Workshops (PDSEC) 2021. `arXiv:2102.08463 <https://arxiv.org/abs/2102.08463>`_
.. [Williams2009] S. Williams, A. Waterman, D. Patterson, "Roofline: an insightful visual performance model for multicore architectures", Commun. ACM 52(4), 65–76, 2009. `doi:10.1145/1498765.1498785 <https://doi.org/10.1145/1498765.1498785>`_
.. [Wohlberg2017] B. Wohlberg, "ADMM penalty parameter selection by residual balancing", 2017. `arXiv:1704.06209 <https://arxiv.org/abs/1704.06209>`_
