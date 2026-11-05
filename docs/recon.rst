Reconstruction
==============

The main reconstruction tool you should use is `recon-rlsq`_. This solves the regularized least-squares reconstruction problem using the Alternating-Directions Method-of-Multipliers algorithm. It supports several common regularizers including L1-wavelets, Total Variation and Total Generalized Variation. To run an unregularized reconstruction use `recon-lsq`_. ``recon-lsq`` and ``recon-rlsq`` will self-calibrate sensitivity maps from the input data if no maps are supplied. These can be calculated explicitly with the `sense-calib`_ command, see there for details of the calibration.

* `recon-lsq`_
* `recon-rlsq`_
* `recon-rss`_
* `sense-calib`_
* `denoise`_

*Common Options*

* ``--basis=basis.h5``

    RIESLING supports sub-space reconstruction using the specified basis vectors.

* ``--tophat``

    Use a top-hat kernel/nearest-neighbour gridding, see `C. Oesterle, M. Markl, R. Strecker, F. M. Kraemer, and J. Hennig, ‘Spiral reconstruction by regridding to a large rectilinear matrix: A practical solution for routine systems’, Journal of Magnetic Resonance Imaging, vol. 10, no. 1, pp. 84–92, Jul. 1999 <http://doi.wiley.com/10.1002/%28SICI%291522-2586%28199907%2910%3A1%3C84%3A%3AAID-JMRI12%3E3.0.CO%3B2-D>`_. The default kernel is a width 4 exponential of a semi-circle, see `A. H. Barnett, ‘Aliasing error of the exp ⁡ ( β 1 − z 2 ) kernel in the nonuniform fast Fourier transform’, Applied and Computational Harmonic Analysis, vol. 51, pp. 1–16, Mar. 2021 <https://linkinghub.elsevier.com/retrieve/pii/S1063520320300725>`_.

* ``--osamp=S``

    Grid oversampling factor, default 1.3. See `P. J. Beatty, D. G. Nishimura, and J. M. Pauly, ‘Rapid gridding reconstruction with a minimal oversampling ratio’, IEEE Transactions on Medical Imaging, vol. 24, no. 6, pp. 799–808, Jun. 2005 <http://ieeexplore.ieee.org/document/1435541/>`_.

* ``--fov=F,F,F``, ``--crop-fov=F,F,F``

    Set the fields of view to use during iterations and the final cropping respectively. Matrix sizes will be calculated using the header voxel-size information. In situations where there is significant signal outside the nominal FOV, but the data was acquired oversampled, then this can be used to prevent aliasing artefacts and improve image quality. `C. A. Baron, N. Dwork, J. M. Pauly, and D. G. Nishimura, ‘Rapid compressed sensing reconstruction of 3D non-Cartesian MRI’, Magnetic Resonance in Medicine, vol. 79, no. 5, pp. 2685–2692, May 2018 <http://doi.wiley.com/10.1002/mrm.26928>`_. The default for both is the matrix size multipled by the voxel size in the header.

* ``--matrix=M,M,M``

    Override the matrix size in the header. The voxel sizes will be rescaled to keep the FOV the same.

* ``--lowmem``

    3D non-cartesian reconstructions can consume large amounts of memory. By default RIESLING will reconstruct all channels simultaneously, requiring that both the oversampled grid and the sensitivity maps for each are held in RAM. Enabling this option swaps to a scheme where only one grid and sensitivity map are kept in RAM. This requires repeating the NUFFT calculations for each channel, trading memory size for reconstruction speed.

* ``--gpu``

    Run the SENSE + NUFFT operator on a CUDA GPU. Requires a build with ``-DRL_CUDA=ON`` (see the README). Only for 3D single-slab reconstructions, and not with ``--lowmem`` or ``--decant``. The GPU operator is the same operator as the CPU's to floating-point rounding (same kernel, apodization and scaling), so results do not change. ``recon-rlsq`` then uses it in the ADMM x-update, or, with ``--pdhg``, runs the whole PDHG iteration on the GPU. See :doc:`design`.

* ``--precon=none/single/multi/file``

    Choose a diagonal k-space preconditioner. The default is Frank Ong's preconditioner. See `F. Ong, M. Uecker, and M. Lustig, ‘Accelerating Non-Cartesian MRI Reconstruction Convergence Using k-Space Preconditioning’, IEEE Trans. Med. Imaging, vol. 39, no. 5, pp. 1646–1654, May 2020 <https://ieeexplore.ieee.org/document/8906069/>`_.

    Generated weights use Ong's row formula ``p_i = ||a_i||^2 / sum_j |a_i^H a_j|^2`` with a NUFFT/autocorrelation approximation. The single-channel spatial kernel retains the nominal trajectory-matrix support even when ``--fov`` enlarges the reconstruction support. Temporal overlaps are squared in magnitude, including complex bases, and the row norm is included. Multichannel weights include coil-map row norms and cross-correlations between coefficient-dependent maps. This corrects previous signed-overlap cancellation and multichannel scaling; regenerating weights can therefore change results. Existing weights files are loaded unchanged.

* ``--precon-no-basis``

    Generate weights without the temporal basis while retaining the basis in the reconstruction operator. This also applies to ``riesling precon``. It has no effect when ``--precon`` names an existing weights file. Single-channel weights then depend only on the sampling geometry. For multichannel weights, coefficient-dependent maps require a matching basis; use shared maps for basis-free generation.

    In LSMR, ADMM and ``--pdhg-weighted``, these weights define the data term, so changing them changes the objective and can require re-tuning the regularization strength. Unweighted PDHG uses them as solver step sizes. A separate choice of objective weights and solver preconditioner is not provided by this option. To reproduce an existing weighted reconstruction exactly, continue supplying its saved weights file.

    Generation streams at most ``K(K+1)/2`` temporal products through one scalar NUFFT workspace, including complex bases. Exactly zero products are skipped (identity frames need only K passes). Dense bases require more NUFFT passes than the former K-vector calculation, but no K-squared set of image grids. The encoding basis normalization is unchanged.

    ``--precon-l`` must be finite and nonnegative; the returned weight is ``(1+l)/(p_i^-1+l)``. Exactly zero encoding rows, including discarded trajectory samples and samples that round off the grid at the +k edge, receive neutral weight 1. Nonfinite or nonpositive denominators for nonzero encoding rows cause an error instead of being hidden by an absolute value.


recon-lsq
---------

Reconstructs the image as the least-squares solution to an inverse problem. The system matrix includes sensitivity maps and the NUFFT. The LSMR algorithm is used as this does not require forming the normal equations, keeping the condition number low. In combination with pre-conditioning in k-space this ensures convergence in a handful of iterations.

*Usage*

.. code-block:: bash

    riesling recon-lsq input.h5 output.h5

*Important Options*

* ``--max-its=N``, ``--atol=A``, ``--btol=B``, ``--ctol=C``

    Termination conditions. Reasoable image quality can be obtained in less than five iterations. The a and b tolerances are relative to how accurate the solution has become, c is a tolerance on the condition number of the system.

* ``--lambda=L``

    Apply basic Tikohonov/L2 regularization to the reconstruction.

* ``--tacq=T``, ``--Nt=N``

    Off-resonance correction parameters. To use these you must also supply an off-resonance map in the input. This should be a dataset labelled ``f0map`` and must be the same matrix size as your image at the reconstruction FOV. These two parameters are the total acquisition time and the number of time segments you want for the reconstruction. The units of ``tacq`` must match ``f0map``, i.e. if your off-resonance map is in Hz then your acquisition time is in seconds.

recon-rlsq
----------

By default, uses the Alternating Directions Method-of-Multipliers (ADMM) to add regularizers to the least-squares reconstruction problem. This is similar to the BART ``pics`` command. See `S. Boyd, ‘Distributed Optimization and Statistical Learning via the Alternating Direction Method of Multipliers’ doi: 10.1561/2200000016 <http://www.nowpublishers.com/article/Details/MAL-016>`_. The Primal Dual Hybrid Gradient (PDHG) algorithm is also implemented. Its step sizes are derived from the maximum eigenvalue of the encoding operator, which is estimated at the start of the run. With ``--gpu`` the whole PDHG iteration runs on the GPU, which is usually the fastest option when the problem fits in GPU memory. The design of both solvers and the reasoning behind their options are described in :doc:`design`.

*Usage*

.. code-block:: bash

    riesling recon-rlsq input.h5 output.h5 --tgv=1e-3

*Regularizers*

See `denoise`_. The same regularizers are available for ``recon-rlsq``.

*Common Options*

* ``--scale=bart/otsu/S``

    The optimal regularization strength λ depends both on the particular regularizer and the typical intensity values in the unregularized image. To make values of λ roughly comparable, it is usual to scale the data such that the intensity values are approximately 1 during the optimization (and then unscale the final image). By default ``riesling`` will perform a NUFFT and then use Otsu's method to find the median foreground intensity as the scaling factor (specify ``otsu`` to make this explicit). The BART automatic scaling can be chosen with ``bart``. Alternately a fixed numeric *multiplicative* scaling factor can be specified, which will skip the initial NUFFT. If you already know the approximate scaling of your data (from a test recon), this option will be the fastest.

*ADMM Options*

    The Alternating-Directions Method-of-Multipliers is a very robust algorithm for solving non-smooth regularized least-squares. However, it requires solving an inverse problem on every iteration, which itself must be solved using an iterative scheme. This means it can be very slow. However, there is only parameter for the algorithm (rho) and the adaptive scheme implemented in riesling means that you should not have to adjust the default parameter. ADMM is hence currently the default choice as it is essentially guaranteed to converge to a sensible answer, given enough iterations.

* ``--max-its0=N``, ``--max-its1=N``, ``--atol=A``, ``--btol=B``, ``--ctol=C``

    These control the inner optimization for ADMM (the x update step). As this step is warm-started, it is possible to control the maximum number of iterations for the zeroth and subsequent steps independently (default 12 each; earlier versions used 64). The tolerances only apply to LSMR: the CG x-update always runs exactly this many iterations. Very few inner iterations make each outer iteration cheaper but need many more outer iterations; in tests 12 reached a given accuracy several times sooner than 3.

* ``--admm-inner=cg/lsmr``

    The solver for the x update step. ``cg`` (the default) uses Conjugate Gradients on the normal equations of the x update, including the k-space preconditioner weights, so it solves the same problem as ``lsmr``. It only needs vectors the size of the image, which uses considerably less memory than LSMR on the stacked system (where every vector also holds all regularizer outputs), and its warm start needs no extra pass of the encoding operator. ``lsmr`` is the previous default.

* ``--split-time``

    Solve each time point (e.g. echo) as its own problem, one after the other. When the regularizers do not couple time points (no ``t`` in ``--iso``, no ``--tvt`` or ``--llr``, no wavelets over the time dimension) the problem is a sum of independent problems, so the solution is the same, but only one time point's solver state is in memory at a time. The data scale is computed from all time points together, as without this option. Each time point gets its own ADMM penalty and stopping test. Works with ADMM and PDHG (CPU and GPU); not with off-resonance correction.

* ``--max-outer-its=N``

    The maximum number of ADMM iterations (default 64).

* ``--eps=E``

    Primal and dual convergence tolerance for ADMM. Default value is 1e-3.

* ``--rho=P``

    Coupling factor for ADMM. The default value of 1 is robust, and will be adjusted inside the algorithm according to `ADMM Penalty Parameter Selection by Residual Balancing <http://arxiv.org/abs/1704.06209>`_.

* ``--mu=M``, ``--tau=T``

    Residual balancing rescales ρ when the primal and dual residuals differ by more than a factor ``mu`` (default 10, the value suggested by Boyd et al.); ``tau`` limits the rescaling factor (default 10). Earlier versions used ``mu`` = 1.2, which with an inexact x-update rescales ρ on almost every iteration and slows convergence; ``--mu=1.2`` restores that behaviour.

*PDHG Options*

    The preconditioned Primal-Dual Hybrid Gradient does not require an inner solve, so its iterations are much cheaper than ADMM's. Its convergence depends on the step sizes; these are now derived from the operator norms and, optionally, chosen automatically. For problems that fit in GPU memory, ``--pdhg --gpu --pdhg-weighted --pdhg-auto-ratio --pdhg-term-steps`` is the suggested starting point, with ``--pdhg-check=10`` to stop on accuracy. See :doc:`design` for the reasoning behind each option.

* ``--pdhg``

    Switches to the PDHG algorithm instead of ADMM. With ``--gpu`` the whole iteration (operator, regularizers, vector updates) runs on the GPU, for every regularizer of ``recon-rlsq``. The run stops with a message if the problem does not fit in GPU memory; ``--split-time``, a smaller basis or fewer regularizers reduce it.

* ``--pdhg-weighted``

    Weight the data term by the k-space preconditioner weights, i.e. minimise the same objective as ADMM. Without it PDHG minimises the unweighted least-squares problem and uses the weights only as its dual step size (Ong et al.), so its solution and suitable λ differ from ADMM's. Not with ``--lad``.

* ``--lad``

    Least absolute deviations data term (L1 norm of the residual) instead of least squares.

* ``--lambda-E=l``

    The maximum eigenvalue of the preconditioned encoding operator, used for the step sizes. By default it is estimated at the start of the run with a few Lanczos iterations, so ``riesling eig`` is no longer needed; give a value to skip the estimate. The step sizes also include an upper bound of each regularizer transform's norm (e.g. 2 for the first TGV term), so that the convergence condition holds.

* ``--pdhg-ratio=R``

    The ratio τ/σ of the primal to the dual step size (default 1). Only their product is fixed by the convergence condition, and the best ratio depends on the data scale, λ and the regularizers; values of 10–100 are common. Useful when the same kind of reconstruction is repeated and the ratio has been tuned.

* ``--pdhg-auto-ratio``

    Choose τ/σ automatically: a first estimate from the data, then a few damped updates from the iterates (at iterations 50, 100, 200, …). Overrides ``--pdhg-ratio``. Not with ``--adaptive`` or ``--pdhg-restart``.

* ``--pdhg-term-steps``

    Use a separate dual step size per regularizer term and a separate primal step for the image and TGV's auxiliary field (Pock & Chambolle's diagonal preconditioning, applied to blocks), instead of one σ and one τ. Combine with ``--pdhg-auto-ratio``. GPU only.

* ``--max-its=N``, ``--pdhg-tol=t``, ``--pdhg-check=N``

    Termination conditions for PDHG: the maximum number of iterations (default 128) and the tolerance (default 1e-4). By default PDHG stops when the relative change of the image falls below the tolerance. With ``--pdhg-check=N`` (GPU only) it instead computes the relative primal and dual residuals every N iterations and stops when both are below the tolerance, which tracks the distance to the solution more reliably; with N = 10 this costs a few percent of the run time. ``--pdhg-tol=0`` always runs ``--max-its`` iterations. In adaptive mode the tolerance applies to the primal residual.

* ``--adaptive``

    Enables adaptive step-sizes for PDHG (Goldstein et al.), which can lead to faster convergence. However, calculating the step-sizes requires additional applications of the encoding operator, such that each iteration will likely be slower and more memory is required. Use with caution. CPU only.

* ``--pdhg-restart=R``

    Restarted averaged PDHG: runs blocks of R iterations, restarts each block from the average of the previous block's iterates, and returns the final average. The tolerance only ends the current block early. CPU only, not with ``--pdhg-auto-ratio``.

recon-rss
---------

Perform a basic reconstruction using root-sum-of-squares channel combination. Very fast but worst image quality. Does not calculate or use sensitivity maps. Useful for testing.

*Usage*

.. code-block:: bash

    riesling recon-rss input.h5 output.h5

sense-calib
-----------

Sensitivity maps are an integral part of any reconstruction from a multi-channel coil. Calculating high quality sensitivity maps is a difficult and open research question for two reasons. First, the multi-channel reconstruction problem is ill-posed as there is no unique solution (if the sensitivities are multiplied and the image divided by an arbitrary field the same data will result), and second because sensitivities exist in the background of the image where we cannot acquire signal.

RIESLING estimates sensitivities assuming that a fully-sampled calibration region with consistent contrast has been acquired in the data. This is true for the majority of non-cartesian sequences, see `E. N. Yeh et al., ‘Inherently self-calibrating non-cartesian parallel imaging’, Magnetic Resonance in Medicine, vol. 54, no. 1, pp. 1–8, Jul. 2005, <http://doi.wiley.com/10.1002/mrm.20517>`_, and this step is hence incorporated into the reconstruction commands. However, there are many situations where it is beneficial to calculate the sensitivities up-front, potentially from alternate data. There is hence an explicit ``sense-calib`` command for this. All the relevant options to this command are also exposed for the reconstruction commands.

Note that RIESLING calculates and stores the sensitivity kernels in k-space, not the maps themselves. If you want to see the maps, a separate ``sense-maps`` command is provided to convert between them.

The FOV and oversampling used in the calibration must match your reconstruction.

The algorithm used by RIESLING consists of these steps:

1. Reconstruct low-resolution images for each channel from a fully-sampled calibration region (inverse NUFFT).
2. Obtain a reference image either from fully-sampled single channel data or by taking the root-sum-squares across the multi-channel images.
3. Solve the inverse problem :math:`c = RFPs` where :math:`s` are the sensitivity kernels in k-space, :math:`c` are the channel images, :math:`P` is a padding operator, :math:`F` is an FT, and :math:`R` is an operator that multiplies each channel by the reference image.

To ensure the maps are smooth and have support in the background region, the forward model is modified to incorporate regularization with a Sobolev Norm term :math:`λW = (1 + |k|^2)^{l/2}` (where :math:`k` is k-space co-ordinate, i.e. penalises high frequency terms) and a mask :math:`M` over the object:

.. math::
    c' = A s\\
    c' = \begin{bmatrix}
        c\\
        0
    \end{bmatrix}\\
    A = \begin{bmatrix}
        M R F P\\
        λW
    \end{bmatrix}

This problem is badly conditioned, and even with a preconditioner can take approximately 100 iterations to converge. However due to the small matrix sizes this should only take a few seconds. See `H. C. M. Holme, S. Rosenzweig, F. Ong, R. N. Wilke, M. Lustig, and M. Uecker, ‘ENLIVE: An Efficient Nonlinear Method for Calibrationless and Robust Parallel Imaging’, Scientific Reports, vol. 9, no. 1, Dec. 2019, <http://www.nature.com/articles/s41598-019-39888-7>`_ for the regularizer.

*Usage*

.. code-block:: bash

    riesling sense-calib input.h5 kernels.h5

*Important Options*

* ``--ref=reference.h5``

    Use the supplied data to reconstruct the reference image (i.e. from a body coil acquisition) instead of using the root-sum-squares of the channels.

* ``--sense-lambda=λ``

    The amount of regularization to apply to the sensitivities. Over regularization will result in the per-voxel sensitivities reducing.

* ``--sense-l=L``

    The L parameter to the Sobolev Norm weights. Higher numbers increase the regularization strength in a highly non-linear fashion.

* ``--sense-res=R``

    The resolution of the initial reconstructions for the sensitivity maps. Because sensitivities are generally agreed to be smooth, only a low resolution reconstruction is required and the default is 6mm isotropic. However, the resulting images must have a sufficiently large matrix size to extract the kernels from.

* ``--sense-width=K``

    The width of the sensitivity kernels in k-space on the nominal grid. The value specified here will be mulitipled by the oversampling factor to produce the final kernel size. Hence, if you override the default oversampling in the main reconstruction you must also do so here.

* ``--sense-tp=T``

    If the input data contains multiple timepoints, use this one to calculate the sensitivities (default is first volume).

denoise
----------

Denoise an image using a proximal operator possibly combined with the Primal Dual Hybrid Gradient algorithm. See `F. Ong, Accelerating Non-Cartesian MRI Reconstruction Convergence Using k-Space Preconditioning’ doi: 10.1109/TMI.2019.2954121 <https://ieeexplore.ieee.org/document/8906069/>`_

*Usage*

.. code-block:: bash

    riesling denoise input.h5 output.h5 --tgv=1e-3

*Important Options*

* ``--max-its=N``, ``--res-tol=r``, ``--delta-tol=d``, ``--lambda-G=l``

    These control the PDHG algorithm for regularizers that cannot be solved using only a proximal operator (i.e. those that have a non-invertible transform). The residual tolerance is calculated relative to the initial data, and the delta-tolerance will terminate the algorithm when the x update becomes small enough. ``--lambda-G`` is the maximum eigenvalue for the regularizer transform and is used to determine the step-size. The default value of 16 is sufficiently large to cover all the regularizers implemented in ``riesling``.

* ``--scale=bart/otsu/S``

    The optimal regularization strength λ depends both on the particular regularizer and the typical intensity values in the unregularized image. To make values of λ roughly comparable, it is usual to scale the data such that the intensity values are approximately 1 during the optimization (and then unscale the final image). By default ``riesling`` will perform a NUFFT and then use Otsu's method to find the median foreground intensity as the scaling factor (specify ``otsu`` to make this explicit). The BART automatic scaling can be chosen with ``bart``. Alternately a fixed numeric *multiplicative* scaling factor can be specified, which will skip the initial NUFFT. If you already know the approximate scaling of your data (from a test recon), this option will be the fastest.

*Regularization Options*

Multiple regularizers can be specified simultaneously, each with a different regularization strength λ and options. At least one regularizer must be specified, there is no default option at present.

* ``--l1=λ``

    Basic L1 regularization in the image domain, i.e. λ|x|.

* ``--tv=λ``

    Classic `Total Variation <https://linkinghub.elsevier.com/retrieve/pii/016727899290242F>`_ regularization, i.e. λ|∇x|

* ``--tv2=λ``

    Second-order TV, i.e. the gradient and the isotropic Laplacian. Has comparable quality to TGV but much lower memory consumption and much faster convergence. See `A Combined First and Second Order Variational Approach for Image Reconstruction <http://link.springer.com/10.1007/s10851-013-0445-4>`_.

* ``--tgv=λ``, ``--tgvl2=λ``

    `Total Generalized Variation <http://doi.wiley.com/10.1002/mrm.22595>`_ and `TGV on the L2 voxelwise norm <http://ieeexplore.ieee.org/document/7466848/>`_. The latter is useful for multichannel images. Note that due to the way the TGV problem is formulated, it consumes significantly more memory and is slower than TV for the same data. TGV can be combined with the other regularizers, which then act on the image. The adjoints of TGV's operators are now exact (earlier versions weighted the second-order term about twice its nominal λ), so λ values tuned with earlier versions may need re-tuning.

* ``--iso=b|g|bg|bt|gt|bgt``

    Isotropic or joint denoising on the specified dimensions (basis, spatial gradients, time) for TV, TV2, TGV or L1. Not all regularizers support all combinations. For ``--l1`` only ``b``, ``t`` and ``bt`` apply (earlier versions normed the wrong dimensions). See `F. Knoll, M. Holler, T. Koesters, R. Otazo, K. Bredies, and D. K. Sodickson, ‘Joint MR-PET Reconstruction Using a Multi-Channel Image Regularizer’, IEEE Trans. Med. Imaging, vol. 36, no. 1, pp. 1–16, Jan. 2017, doi: 10.1109/TMI.2016.2564989. <http://ieeexplore.ieee.org/document/7466848/>`_.

* ``--llr=λ``, ``--llr-patch=N``, ``--llr-win=N``, ``--llr-shift``

    `Locally Low-Rank <https://onlinelibrary.wiley.com/doi/abs/10.1002/mrm.26102>`_ regularization. The patch size determines the region to calculate the SVD over, the window size determines the region that is copied to the output image. Set the window size to 1 to calculate an SVD for each output voxel. Set the window size equal to the patch size to use the entire patch. The ``--llr-shift`` option employs the random patch shifting strategy, this may not converge.

* ``--wavelets=λ``, ``--wavelet-width=W``, ``--wavelet-dims=0,1,1,1``

    L1-wavelets of width W (default 6). The number of levels is the maximum possible. Which of the basis,X,Y,Z dimensions to be transformed can be specified with the ``--wavelet-dims`` option.
