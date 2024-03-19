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
from xopt.generators.bayesian import UpperConfidenceBoundGenerator
from xopt.generators.bayesian.models.standard import StandardModelConstructor
from xopt.numerical_optimizer import GridOptimizer
from emitopt.analysis import compute_emit_bayesian

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
    n_interpolate_points,
    quad_strength_key,
    initial_data=None,
    visualize=False,
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
        gp_constructor=model_constructor,
        numerical_optimizer=GridOptimizer(n_grid_points=100),
        turbo_controller=turbo_controller,
        n_interpolate_points=n_interpolate_points,
        n_monte_carlo_samples=32,
        **generator_kwargs,
    )
    beamsize_evaluator = Evaluator(function=beamsize_evaluator)
    X = Xopt(generator=generator, evaluator=beamsize_evaluator, vocs=vocs)
    X.dump_file = dump_file

    # add old data if specified
    if initial_data is not None:
        X.add_data(initial_data)

    # evaluate initial points if specified
    if initial_points is not None:
        X.evaluate_data(initial_points)

    if len(X.data) == 0:
        raise RuntimeError(
            "no data added to model during initialization, "
            "must specify either initial_data or initial_points"
            "to perform sampling"
        )

    if visualize > 1:
        visualize_step(X.generator, f"{X.vocs.objective_names[0]}, step:{1}")
    X.step()

    # perform exploration
    for i in range(n_iterations - 1):
        print(i)
        if visualize > 1:
            visualize_step(X.generator, f"{X.vocs.objective_names[0]}, step:{i + 2}")
        X.step()
    print("done")

    # get minimum point
    turbo_controller = X.generator.turbo_controller
    min_pt = (turbo_controller.center_x, turbo_controller.best_value)

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
    n_interpolate_points: int = 5,
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
        n_interpolate_points,
        quad_strength_key,
        initial_data=initial_data,
        visualize=visualize,
    )
    print(f"Runtime: {time.perf_counter() - start}")

    # perform sampling for Y
    print("sampling points for y emittance - note: 1/2 n_iterations")
    start = time.perf_counter()
    gen_data_y, min_pt_y, X = perform_sampling(
        yvocs,
        turbo_length,
        beamsize_evaluator,
        dump_file,
        generator_kwargs,
        None,
        int(n_iterations / 2),
        n_interpolate_points,
        quad_strength_key,
        initial_data=gen_data_x,
        visualize=visualize,
    )
    print(f"Runtime: {time.perf_counter() - start}")

    return (
        analyze_data(
            gen_data_y,
            beamline_config,
            quad_strength_key,
            rms_x_key,
            rms_y_key,
            [min_pt_x, min_pt_y],
            visualize,
        ),
        X,
    )


def analyze_data(
    analysis_data,
    beamline_config,
    quad_strength_key,
    rms_x_key,
    rms_y_key,
    minimum_pts,
    visualize,
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
        # min_loc =  minimum_pts[i][0][quad_strength_key]
        # width = 2.5
        # data = data[
        #   pd.DataFrame(
        #       (
        #           data[quad_strength_key] < min_loc + width / 2,
        #           data[quad_strength_key] > min_loc - width / 2,
        #       )
        #   ).all()
        # ]

        # window data via a fixed multiple of the minimum value
        # min_multiplier = 2
        # max_val = minimum_pts[i][1] * min_multiplier
        # data = data[data[key[i]] < max_val]

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
        stats = compute_emit_bayesian(
            k,
            rms,
            beamline_config.scan_quad_length,
            rmat_quad_to_screen,
            beta0=beta0[i],
            alpha0=alpha0[i],
            visualize=True,
        )
        print(f"Runtime: {time.perf_counter() - start}")

        # return emittance results in [mm-mrad]
        gamma = beamline_config.beam_energy / 0.511e-3
        result = result | {
            f"{name[i]}_emittance": float(gamma * torch.quantile(stats[0], 0.5)),
            f"{name[i]}_emittance_05": float(gamma * torch.quantile(stats[0], 0.05)),
            f"{name[i]}_emittance_95": float(gamma * torch.quantile(stats[0], 0.95)),
            f"{name[i]}_emittance_var": float(torch.var(gamma * stats[0])),
        }
        if stats[1] is not None:
            result[f"bmag_{name[i]}_median"] = float(torch.quantile(stats[1], 0.5))

    return result
