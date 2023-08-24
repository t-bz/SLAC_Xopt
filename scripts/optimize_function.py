from typing import Callable, Dict

import numpy as np

import pandas as pd
from pandas import DataFrame
from xopt import Evaluator, VOCS, Xopt
from xopt.generators import ExpectedImprovementGenerator


def optimize_function(
    vocs: VOCS,
    evaluator_function: Callable,
    n_iterations: int = 5,
    n_initial: int = 5,
    initial_points: DataFrame = None,
    results_dir: str = None,
    generator_kwargs: Dict = None,
) -> Xopt:
    """
    Function to minimize a given function using Xopt's ExpectedImprovementGenerator.

    Details:
    - Initializes BO with a set number of random evaluations given by `n_initial`
    - Raises errors if they occur during calls of `evaluator_function`
    - Runs the generator for `n_iteration` steps
    - Identifies and re-evaluates the best observed point

    Parameters
    ----------
    vocs: VOCS
        Xopt style VOCS object to describe the optimization problem

    evaluator_function : Callable
        Xopt style callable function that is evaluated during the optimization run.

    n_iterations : int, optional
        Number of optimization steps to run. Default: 5

    n_initial : int, optional
        Number of initial random samples to take before performing optimization
        steps. Default: 5

    generator_kwargs : dict, optional
        Dictionary passed to generator to customize Expected Improvement BO.

    Returns
    -------
    X : Xopt
        Xopt object containing the evaluator, generator, vocs and data objects.

    """

    # set up Xopt object
    generator_kwargs = generator_kwargs or {}
    beamsize_evaluator = Evaluator(function=evaluator_function)
    generator = ExpectedImprovementGenerator(vocs=vocs, **generator_kwargs)
    generator.numerical_optimizer.max_iter = 100
    # generator.turbo_controller = "optimize"

    X = Xopt(vocs=vocs, generator=generator, evaluator=beamsize_evaluator)
    X.options.strict = True

    if results_dir is not None:
        X.options.dump_file = f"{results_dir}/optimize_record.yml"

    if initial_points is not None:
        X.evaluate_data(initial_points)
    else:
        # evaluate random initial points
        X.random_evaluate(n_initial)

    # run optimization
    for i in range(n_iterations):
        print(f"step {i}")
        X.step()

    # get best config and re-evaluate it
    best_config = X.data[X.vocs.variable_names + X.vocs.constant_names].iloc[
        np.argmin(X.data[X.vocs.objective_names].to_numpy())
    ]
    X.evaluate_data(pd.DataFrame(best_config.to_dict(), index=[1]))

    return X
