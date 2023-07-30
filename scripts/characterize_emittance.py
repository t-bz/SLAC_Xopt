from typing import Callable, Dict
import numpy as np
from copy import deepcopy

import torch
from emitopt.utils import get_valid_emit_samples_from_quad_scan
from xopt import VOCS, Xopt, Evaluator
from xopt.generators import BayesianExplorationGenerator
from xopt.numerical_optimizer import GridOptimizer

import traceback

def characterize_emittance(
        vocs: VOCS,
        beamsize_evaluator: Callable,
        quad_length: float,
        drift_length: float,
        beam_energy: float,
        quad_strength_key: str,
        rms_x_key: str,
        rms_y_key: str,
        quad_strength_scale_factor: float = 1.0,
        n_iterations: int = 10,
        n_initial: int = 5,
        generator_kwargs: Dict = None,
        quad_scan_analysis_kwargs: Dict = None,
        dump_file: str = None
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

    quad_length : float
        Effective magnetic length of the quadrupole in [m].

    drift_length : float
        Drift length from quadrupole to beam size measurement location in [m].

    beam_energy : float
        Beam energy in GeV.

    quad_strength_key : str
        Dictionary key used to specify the geometric quadrupole strength.

    quad_strength_scale_factor : float, optional
        Scale factor that scales quadrupole strength parameters given by
        `quad_strength_key` to [m^{-2}]

    rms_x_key : str
        Dictionary key used to specify the measurement of horizontal beam size.

    rms_y_key : str
        Dictionary key used to specify the measurement of vertical beam size.

    n_iterations : int, optional
        Number of exploration steps to run. Default: 5

    n_initial : int, optional
        Number of initial random samples to take before performing exploration
        steps. Default: 5

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
        - `x_emittance_median` : median value of the horizontal emittance
        - `x_emittance_05` : 5% quantile value of the horizontal emittance
        - `x_emittance_95` : 95% quantile value of the horizontal emittance
        - `y_emittance_median` : median value of the vertical emittance
        - `y_emittance_05` : 5% quantile value of the vertical emittance
        - `y_emittance_95` : 95% quantile value of the vertical emittance
    X : Xopt
        Xopt object containing the evaluator, generator, vocs and data objects.

    """

    # check for proper vocs object
    assert quad_strength_key in vocs.variable_names
    assert (rms_x_key in vocs.observables) and (rms_y_key in vocs.observables)

    # set up kwarg objects
    generator_kwargs = generator_kwargs or {}
    quad_scan_analysis_kwargs = quad_scan_analysis_kwargs or {}

    # set up Xopt object
    generator = BayesianExplorationGenerator(
        vocs=vocs,
        numerical_optimizer=GridOptimizer(n_grid_points=100),
        **generator_kwargs
    )
    beamsize_evaluator = Evaluator(function=beamsize_evaluator)
    X = Xopt(generator=generator, evaluator=beamsize_evaluator, vocs=vocs)
    X.options.dump_file = dump_file

    # evaluate random samples
    X.random_evaluate(n_initial)

    # check to make sure the correct data is returned before performing exploration
    assert rms_y_key in X.data.columns
    assert rms_x_key in X.data.columns

    # perform exploration
    for i in range(n_iterations):
        X.step()

    try:
        analysis_data = deepcopy(X.data)[[quad_strength_key, rms_x_key, rms_y_key]].dropna()
        
        # get data from xopt object and scale to [m^{-2}]
        k = analysis_data[quad_strength_key].to_numpy(dtype=np.double) * quad_strength_scale_factor
        rms_x = analysis_data[rms_x_key].to_numpy(dtype=np.double)
        rms_y = analysis_data[rms_y_key].to_numpy(dtype=np.double)
    
        # calculate emittances (note negative sign in y-calculation)
        x_emit_stats = get_valid_emit_samples_from_quad_scan(
            k,
            rms_x,
            quad_length,
            drift_length,
            **quad_scan_analysis_kwargs
        )
        y_emit_stats = get_valid_emit_samples_from_quad_scan(
            -k,
            rms_y,
            quad_length,
            drift_length,
            **quad_scan_analysis_kwargs
        )
    
        # return emittance results in [mm-mrad]
        gamma = beam_energy / 0.511e-3
        result = {
            "x_emittance_median": float(gamma*torch.quantile(x_emit_stats[0], 0.5)),
            "x_emittance_05": float(gamma*torch.quantile(x_emit_stats[0], 0.05)),
            "x_emittance_95": float(gamma*torch.quantile(x_emit_stats[0], 0.95)),
            "y_emittance_median": float(gamma*torch.quantile(y_emit_stats[0], 0.5)),
            "y_emittance_05": float(gamma*torch.quantile(y_emit_stats[0], 0.05)),
            "y_emittance_95": float(gamma*torch.quantile(y_emit_stats[0], 0.95)),
        }
        
    except Exception:
        print(traceback.format_exc())
        result = {}

    finally:
        return result, X
        
