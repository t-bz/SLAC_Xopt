import traceback
from copy import deepcopy
from typing import Callable, Dict
import time
import numpy as np

import pandas as pd
import torch
from botorch import fit_gpytorch_mll

from botorch.models.gp_regression import SingleTaskGP
from botorch.models.transforms import Normalize, Standardize
from gpytorch import ExactMarginalLogLikelihood
from gpytorch.kernels import MaternKernel, PolynomialKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.priors import GammaPrior
from pandas import DataFrame
from xopt import Evaluator, VOCS, Xopt
from xopt.generators import UpperConfidenceBoundGenerator
from xopt.generators.bayesian.models.standard import StandardModelConstructor
from xopt.numerical_optimizer import GridOptimizer
from emitopt.utils import get_quad_strength_conversion_factor

from scripts.custom_turbo import QuadScanTurbo
from scripts.utils.visualization import visualize_step

def perform_sampling(
    vocs,
    turbo_length,
    beamsize_evaluator,
    dump_file,
    generator_kwargs,
    initial_points,
    n_iterations,
    quad_strength_key,
    initial_data=None,
    visualize=False
):
    # run points to determine emittance
    # ===================================
    
    # set up Xopt object
    # use beta to control the relative spacing between points and the observed minimum
    turbo_controller = QuadScanTurbo(vocs, length=turbo_length)
    model_constructor = StandardModelConstructor(use_low_noise_prior=False)
    generator = UpperConfidenceBoundGenerator(
        vocs=vocs,
        beta=100.0,
        numerical_optimizer=GridOptimizer(n_grid_points=100),
        model_constructor=model_constructor,
        turbo_controller=turbo_controller,
        **generator_kwargs,
    )

    beamsize_evaluator = Evaluator(function=beamsize_evaluator)
    X = Xopt(generator=generator, evaluator=beamsize_evaluator, vocs=vocs)
    X.options.dump_file = dump_file

    # add old data if specified
    if initial_data is not None:
        X.add_data(initial_data)

    # evaluate initial points if specified
    if initial_points is not None:
        X.evaluate_data(initial_points)

    if len(X.data) == 0:
        raise RuntimeError(
            "no data added to model during initialization, "\
            "must specify either initial_data or initial_points"\
            "to perform sampling"
        )

    if visualize > 1:
        visualize_step(X.generator, f"{X.vocs.objective_names[0]}, step:{1}")
    X.step()
       
    # perform exploration
    for i in range(n_iterations - 1):
        if visualize > 1:
            visualize_step(X.generator, f"{X.vocs.objective_names[0]}, step:{i + 2}")
        X.step()
        

    # get minimum point
    turbo_controller = X.generator.turbo_controller
    min_pt = (
        turbo_controller.center_x, 
        turbo_controller.best_value
    )

    # return data
    return X.data, min_pt, X


def characterize_emittance(
    xvocs: VOCS,
    yvocs: VOCS,
    beamsize_evaluator: Callable,
    beamline_config,
    quad_strength_key: str,
    rms_x_key: str,
    rms_y_key: str,
    initial_points: DataFrame = None,
    initial_data: DataFrame = None,
    n_iterations: int = 5,
    turbo_length: float = 1.0,
    generator_kwargs: Dict = None,
    visualize: int = 0,
    dump_file: str = None,
):
    """
    Script to evaluate beam emittance using an automated quadrupole scan.

    In order to work properly the callable passed as `beamsize_evaluator` should have
    the following properties:
        - It should accept a dictionary containing the quadrupole strength parameter
        identified by `quad_strength_key` and return a dictionary containing the keys
        `rms_x_key` and 'rms_y_key'.
        - Quadrupole strengths should be in units of [m^{-2}] (or should be scaled by
        `quadrupole_strength_scale_factor` to [m^{-2}]) with positive values denote
        focusing in the horizontal plane
        - RMS values returned should be in units of [mm]

    Note: rms values are specified in [mm] to avoid numerical fitting issues of
    hyperparameters in the GP models.

    Parameters
    ----------
    vocs: VOCS
        Xopt style VOCS object to describe the Bayesian exploration problem for
        quadrupole scan

    beamsize_evaluator : Callable
        Xopt style callable function that is evaluated during the optimization run.

    beamline_config : BeamlineConfig
        Beamline configuration object containing info about the beamline used for the
        quad scan.

    quad_strength_key : str
        Dictionary key used to specify the geometric quadrupole strength.

    initial_data : DataFrame
        Initial points to evaluate in order to initialize characterization.

    rms_x_key : str
        Dictionary key used to specify the measurement of horizontal beam size.

    rms_y_key : str
        Dictionary key used to specify the measurement of vertical beam size.

    n_iterations : int, optional
        Number of exploration steps to run. Default: 5

    turbo_length : float, optional
        Scale factor for turbo lengthscale. In units of GP lengthscales. Default: 1.0

    generator_kwargs : dict, optional
        Dictionary passed to generator to customize Bayesian Exploration.

    quad_scan_analysis_kwargs : dict, optional
        Dictionary used to customize quadrupole scan analysis / emittance calculation.

    dump_file : str, optional
        Filename to specify dump file for Xopt object.

    Returns
    -------
    result : dict
        Results dictionary object containing the following keys. Note emittance units
        are [mm-mrad].
        - `x_emittance` : median value of the horizontal emittance
        - `x_emittance_05` : 5% quantile value of the horizontal emittance
        - `x_emittance_95` : 95% quantile value of the horizontal emittance
        - `x_emittance_var : variance of horizontal emittance
        - `y_emittance` : median value of the vertical emittance
        - `y_emittance_05` : 5% quantile value of the vertical emittance
        - `y_emittance_95` : 95% quantile value of the vertical emittance
        - `y_emittance_var : variance of vertical emittance

    X : Xopt
        Xopt object containing the evaluator, generator, vocs and data objects.

    """

    # check for proper vocs object
    # assert quad_strength_key in vocs.variable_names
    # assert (rms_x_key in vocs.observables) and (rms_y_key in vocs.observables)

    # set up kwarg objects
    generator_kwargs = generator_kwargs or {}

    # perform sampling for X
    print("sampling points for x emittance")
    start = time.perf_counter()
    gen_data_x, min_pt_x, X = perform_sampling(
        xvocs,
        turbo_length,
        beamsize_evaluator,
        dump_file,
        generator_kwargs,
        initial_points,
        n_iterations,
        quad_strength_key,
        initial_data=initial_data,
        visualize=visualize,
    )
    print(f"Runtime: {time.perf_counter() - start}")

    # perform sampling for Y
    print("sampling points for y emittance")
    start = time.perf_counter()
    gen_data_y, min_pt_y, X = perform_sampling(
        yvocs,
        turbo_length,
        beamsize_evaluator,
        dump_file,
        generator_kwargs,
        None,
        n_iterations,
        quad_strength_key,
        initial_data=gen_data_x,
        visualize=visualize,
    )
    print(f"Runtime: {time.perf_counter() - start}")

    return analyze_data(
        gen_data_y, 
        beamline_config, 
        quad_strength_key, 
        rms_x_key, 
        rms_y_key,
        [min_pt_x, min_pt_y],
        visualize
    ), X


def analyze_data(
    analysis_data, 
    beamline_config, 
    quad_strength_key, 
    rms_x_key, 
    rms_y_key, 
    minimum_pts, 
    visualize
):
    # get subset of data for analysis, drop Nan measurements
    analysis_data = deepcopy(analysis_data)[
        [quad_strength_key, rms_x_key, rms_y_key]
    ].dropna()
    

    key = [rms_x_key, rms_y_key]
    rmat = [beamline_config.transport_matrix_x, beamline_config.transport_matrix_y]
    name = ["x", "y"]
    beta0 = [beamline_config.design_beta_x, beamline_config.design_beta_y]
    alpha0 = [beamline_config.design_alpha_x, beamline_config.design_alpha_y]
    
    result = {}

    for i in range(2):
        # make a copy of the analysis data
        data = deepcopy(analysis_data)

        # window data via a fixed width around the minimum point
        min_loc =  minimum_pts[i][0][quad_strength_key]
        width = 2.5
        data = data[
           pd.DataFrame(
               (
                   data[quad_strength_key] < min_loc + width / 2,
                   data[quad_strength_key] > min_loc - width / 2,
               )
           ).all()
        ]

        # window data via a fixed multiple of the minimum value
        min_multiplier = 2
        max_val = minimum_pts[i][1] * min_multiplier
        data = data[data[key[i]] < max_val]

        # get data from xopt object and scale to [m^{-2}]
        k = (
            data[quad_strength_key].to_numpy(dtype=np.double)
            * beamline_config.pv_to_focusing_strength
        )
        
        # flip sign of focusing strengths for y
        if name[i] == "y":
            k = -k
        
        rms = data[key[i]].to_numpy(dtype=np.double)

        # get transport matrix from quad to screen
        rmat_quad_to_screen = torch.tensor(rmat[i]).double()

        # calculate emittances (note negative sign in y-calculation)
        print(f"creating emittance fit {name[i]}")
        start = time.perf_counter()
        stats = get_valid_emit_bmag_samples_from_quad_scan(
            k,
            rms,
            beamline_config.scan_quad_length,
            rmat_quad_to_screen,
            beta0=beta0[i],
            alpha0=alpha0[i],
            visualize=visualize > 0,
        )
        print(f"Runtime: {time.perf_counter() - start}")

        # return emittance results in [mm-mrad]
        gamma = beamline_config.beam_energy / 0.511e-3
        result = result | {
            f"{name[i]}_emittance": float(gamma * torch.quantile(stats[0], 0.5)),
            f"{name[i]}_emittance_05": float(
                gamma * torch.quantile(stats[0], 0.05)
            ),
            f"{name[i]}_emittance_95": float(
                gamma * torch.quantile(stats[0], 0.95)
            ),
            f"{name[i]}_emittance_var": float(torch.var(gamma * stats[0])),
        }
        if stats[1] is not None:
            result[f"bmag_{name[i]}_median"] = float(torch.quantile(stats[1], 0.5))

    return result


from emitopt.utils import build_quad_rmat, plot_valid_thick_quad_fits, propagate_sig


def compute_emit_bmag_thick_quad(
    k, y_batch, q_len, rmat_quad_to_screen, beta0=1.0, alpha0=0.0
):
    """
    A function that computes the emittance(s) corresponding to a set of quadrupole measurement scans
    using a thick quad model.

    Parameters:
        k: 1d torch tensor of shape (n_steps_quad_scan,)
            representing the measurement quad geometric focusing strengths in [m^-2]
            used in the emittance scan

        y_batch: 2d torch tensor of shape (n_scans x n_steps_quad_scan),
                where each row represents the mean-square beamsize outputs in [m^2] of an emittance scan
                with inputs given by k

        q_len: float defining the (longitudinal) quadrupole length or "thickness" in [m]

        rmat_quad_to_screen: the (fixed) 2x2 R matrix describing the transport from the end of the
                measurement quad to the observation screen.

        beta0: the design beta twiss parameter at the screen

        alpha0: the design alpha twiss parameter at the screen

    Returns:
        emit: shape (n_scans x 1) containing the geometric emittance fit results for each scan
        bmag_min: (n_scans x 1) containing the bmag corresponding to the optimal point for each scan
        sig: shape (n_scans x 3 x 1) containing column vectors of [sig11, sig12, sig22]
        is_valid: 1d tensor identifying physical validity of the emittance fit results

    SOURCE PAPER: http://www-library.desy.de/preparch/desy/thesis/desy-thesis-05-014.pdf
    """

    # construct the A matrix from eq. (3.2) & (3.3) of source paper
    quad_rmats = build_quad_rmat(k, q_len)  # result shape (len(k) x 2 x 2)
    total_rmats = (
        rmat_quad_to_screen.reshape(1, 2, 2) @ quad_rmats
    )  # result shape (len(k) x 2 x 2)

    amat = torch.tensor([])  # prepare the A matrix
    for rmat in total_rmats:
        r11, r12 = rmat[0, 0], rmat[0, 1]
        amat = torch.cat(
            (amat, torch.tensor([[r11**2, 2.0 * r11 * r12, r12**2]])), dim=0
        )
    # amat result shape (len(k) x 3)

    # get sigma matrix elements just before measurement quad from pseudo-inverse
    sig = amat.pinverse().unsqueeze(0) @ y_batch.unsqueeze(
        -1
    )  # shapes (1 x 3 x len(k)) @ (n_scans x len(k) x 1)
    # result shape (n_scans x 3 x 1) containing column vectors of [sig11, sig12, sig22]

    # compute emit
    emit = torch.sqrt(sig[:, 0, 0] * sig[:, 2, 0] - sig[:, 1, 0] ** 2).reshape(
        -1, 1
    )  # result shape (n_scans x 1)

    # check sigma matrix and emit for physical validity
    is_valid = torch.logical_and(sig[:, 0, 0] > 0, sig[:, 2, 0] > 0)  # result 1d tensor
    is_valid = torch.logical_and(
        is_valid, ~torch.isnan(emit.flatten())
    )  # result 1d tensor

    # propagate beam parameters to screen
    twiss_at_screen = propagate_sig(sig, emit, total_rmats)[1]
    # result shape (n_scans x len(k) x 3 x 1)

    if alpha0 is not None and beta0 is not None:
        # get design gamma0 from design beta0, alpha0
        gamma0 = (1 + alpha0**2) / beta0

        # compute bmag
        bmag = 0.5 * (
            twiss_at_screen[:, :, 0, 0] * gamma0
            - 2 * twiss_at_screen[:, :, 1, 0] * alpha0
            + twiss_at_screen[:, :, 2, 0] * beta0
        )
        # result shape (n_scans, n_steps_quad_scan)

        # select minimum bmag from quad scan
        bmag_min, bmag_min_id = torch.min(
            bmag, dim=1, keepdim=True
        )  # result shape (n_scans, 1)
    else:
        bmag_min = None

    return emit, bmag_min, sig, is_valid


def get_valid_emit_bmag_samples_from_quad_scan(
    k,
    y,
    q_len,
    rmat_quad_to_screen,
    beta0=1.0,
    alpha0=0.0,
    n_samples=10000,
    n_steps_quad_scan=10,
    covar_module=None,
    visualize=False,
    tkwargs=None,
):
    """
    A function that produces a distribution of possible (physically valid) emittance values corresponding
    to a single quadrupole measurement scan. Data is first modeled by a SingleTaskGP, virtual measurement
    scan samples are then drawn from the model posterior, the samples are modeled by thick-quad transport
    to obtain fits to the beam parameters, and physically invalid results are discarded.

    Parameters:

        k: 1d numpy array of shape (n_steps_quad_scan,)
        representing the measurement quad geometric focusing strengths in [m^-2]
        used in the emittance scan

        y: 1d numpy array of shape (n_steps_quad_scan, )
            representing the root-mean-square beam size measurements in [m] of an emittance scan
            with inputs given by k

        q_len: float defining the (longitudinal) quadrupole length or "thickness" in [m]

        rmat_quad_to_screen: the (fixed) 2x2 R matrix describing the transport from the end of the
                measurement quad to the observation screen.

        beta0: the design beta twiss parameter at the screen

        alpha0: the design alpha twiss parameter at the screen

        n_samples: the number of virtual measurement scan samples to evaluate for our "Bayesian" estimate

        n_steps_quad_scan: the number of steps in our virtual measurement scans

        covar_module: the covariance module to be used in fitting of the SingleTaskGP
                    (modeling the function y**2 vs. k)
                    If None, uses ScaleKernel(MaternKernel()).

        visualize: boolean. Set to True to plot the parabolic fitting results.

        tkwargs: dict containing the tensor device and dtype

    Returns:
        emits_valid: a tensor of physically valid emittance results from sampled measurement scans.

        bmag_valid: (n_valid_scans x 1) containing the bmag corresponding to the optimal point
                        from each physically valid fit.

        sig_valid: tensor, shape (n_valid_scans x 3 x 1), containing the computed
                        sig11, sig12, sig22 corresponding to each physically valid
                        fit.

        sample_validity_rate: a float between 0 and 1 that describes the rate at which the samples
                        were physically valid/retained.
    """
    if tkwargs is None:
        tkwargs = {"dtype": torch.double, "device": "cpu"}

    k = torch.tensor(k, **tkwargs)
    y = torch.tensor(y, **tkwargs)

    k_virtual, bss = fit_gp_quad_scan(
        k=k,
        y=y,
        n_samples=n_samples,
        n_steps_quad_scan=n_steps_quad_scan,
        covar_module=covar_module,
        tkwargs=tkwargs,
    )

    (emit, bmag, sig, is_valid) = compute_emit_bmag_thick_quad(
        k=k_virtual,
        y_batch=bss,
        q_len=q_len,
        rmat_quad_to_screen=rmat_quad_to_screen,
        beta0=beta0,
        alpha0=alpha0,
    )

    sample_validity_rate = (torch.sum(is_valid) / is_valid.shape[0]).reshape(1)

    # filter on physical validity
    cut_ids = torch.tensor(range(emit.shape[0]))[is_valid]
    emit_valid = torch.index_select(emit, dim=0, index=cut_ids)
    if bmag is not None:
        bmag_valid = torch.index_select(bmag, dim=0, index=cut_ids)
    else:
        bmag_valid = None

    sig_valid = torch.index_select(sig, dim=0, index=cut_ids)

    if visualize:
        plot_valid_thick_quad_fits(
            k,
            y,
            q_len,
            rmat_quad_to_screen,
            emit=emit_valid,
            bmag=bmag_valid,
            sig=sig_valid,
        )
    return emit_valid, bmag_valid, sig_valid, sample_validity_rate


def plot_valid_thick_quad_fits(
    k, y, q_len, rmat_quad_to_screen, emit, bmag, sig, ci=0.95, tkwargs=None
):
    """
    A function to plot the physically valid fit results
    produced by get_valid_emit_bmag_samples_from_quad_scan().

    Parameters:

        k: 1d numpy array of shape (n_steps_quad_scan,)
        representing the measurement quad geometric focusing strengths in [m^-2]
        used in the emittance scan

        y: 1d numpy array of shape (n_steps_quad_scan, )
            representing the root-mean-square beam size measurements in [m] of an emittance scan
            with inputs given by k

        sig: tensor, shape (n_scans x 3 x 1), containing the computed sig11, sig12, sig22
                corresponding to each measurement scan

        emit: shape (n_scans x 1) containing the geometric emittance fit results for each scan

        q_len: float defining the (longitudinal) quadrupole length or "thickness" in [m]

        rmat_quad_to_screen: the (fixed) 2x2 R matrix describing the transport from the end of the
                measurement quad to the observation screen.

        ci: "Confidence interval" for plotting upper/lower quantiles.

        tkwargs: dict containing the tensor device and dtype
    """
    from matplotlib import pyplot as plt

    if tkwargs is None:
        tkwargs = {"dtype": torch.double, "device": "cpu"}

    k_fit = torch.linspace(k.min(), k.max(), 100, **tkwargs)
    quad_rmats = build_quad_rmat(k_fit, q_len)  # result shape (len(k_fit) x 2 x 2)
    total_rmats = (
        rmat_quad_to_screen.reshape(1, 2, 2) @ quad_rmats
    )  # result shape (len(k_fit) x 2 x 2)
    sig_final = propagate_sig(sig, emit, total_rmats)[
        0
    ]  # result shape len(sig) x len(k_fit) x 3 x 1
    bss_fit = sig_final[:, :, 0, 0]

    upper_quant = torch.quantile(bss_fit.sqrt(), q=0.5 + ci / 2.0, dim=0)
    lower_quant = torch.quantile(bss_fit.sqrt(), q=0.5 - ci / 2.0, dim=0)

    fig, axs = plt.subplots(3)
    fig.set_size_inches(5, 9)

    ax = axs[0]
    fit = ax.fill_between(
        k_fit.detach().numpy(),
        lower_quant * 1.0e6,
        upper_quant * 1.0e6,
        alpha=0.3,
        label='"Bayesian" Thick-Quad Model',
        zorder=1,
    )

    obs = ax.scatter(
        k, y * 1.0e6, marker="x", s=120, c="orange", label="Measurements", zorder=2
    )
    ax.set_title("Beam Size at Screen")
    ax.set_xlabel(r"Measurement Quad Geometric Focusing Strength ($[k]=m^{-2}$)")
    ax.set_ylabel(r"r.m.s. Beam Size ($[\sigma]=\mu m$)")
    ax.legend(handles=[obs, fit])

    ax = axs[1]
    ax.hist(emit.flatten(), density=True)
    ax.set_title("Geometric Emittance Distribution")
    ax.set_xlabel(r"Geometric Emittance ($[\epsilon]=m*rad$)")
    ax.set_ylabel("Probability Density")

    if bmag is not None:
        ax = axs[2]
        ax.hist(bmag.flatten(), range=(1, 5), bins=20, density=True)
        ax.set_title(r"$\beta_{mag}$ Distribution")
        ax.set_xlabel(r"$\beta_{mag}$ at Screen")
        ax.set_ylabel("Probability Density")

    plt.tight_layout()


def fit_gp_quad_scan(
    k,
    y,
    n_samples=10000,
    n_steps_quad_scan=10,
    covar_module=None,
    tkwargs=None,
):
    """
    A function that fits a GP model to an emittance beam size measurement quad scan
    and returns a set of "virtual scans" (functions sampled from the GP model posterior).
    The GP is fit to the BEAM SIZE SQUARED, and the virtual quad scans are NOT CHECKED
    for physical validity.

    Parameters:

        k: 1d numpy array of shape (n_steps_quad_scan,)
        representing the measurement quad geometric focusing strengths in [m^-2]
        used in the emittance scan

        y: 1d numpy array of shape (n_steps_quad_scan, )
            representing the root-mean-square beam size measurements in [m] of an emittance scan
            with inputs given by k

        covar_module: the covariance module to be used in fitting of the SingleTaskGP
                    (modeling the function y**2 vs. k)
                    If None, uses ScaleKernel(MaternKernel()).

        tkwargs: dict containing the tensor device and dtype

        n_samples: the number of virtual measurement scan samples to evaluate for our "Bayesian" estimate

        n_steps_quad_scan: the number of steps in our virtual measurement scans


    Returns:
        k_virtual: a 1d tensor representing the inputs for the virtual measurement scans.
                    All virtual scans are evaluated at the same set of input locations.

        bss: a tensor of shape (n_samples x n_steps_quad_scan) where each row repesents
        the beam size squared results of a virtual quad scan evaluated at the points k_virtual.
    """

    if tkwargs is None:
        tkwargs = {"dtype": torch.double, "device": "cpu"}

    k = torch.tensor(k, **tkwargs)
    y = torch.tensor(y, **tkwargs)

    if covar_module is None:
        covar_module = ScaleKernel(
            PolynomialKernel(2), outputscale_prior=GammaPrior(2.0, 0.15)
        )

    model = SingleTaskGP(
        k.reshape(-1, 1),
        y.pow(2).reshape(-1, 1),
        covar_module=covar_module,
        input_transform=Normalize(1),
        outcome_transform=Standardize(1),
        likelihood=GaussianLikelihood(
            noise_prior=GammaPrior(1.0, 1.0),
        ),
    )
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)

    k_virtual = torch.linspace(k.min(), k.max(), n_steps_quad_scan, **tkwargs)

    p = model.posterior(k_virtual.reshape(-1, 1))
    bss = p.sample(torch.Size([n_samples])).reshape(-1, n_steps_quad_scan)

    return k_virtual, bss
