from typing import List, Callable, Dict

import numpy as np
from emitopt.utils import get_quad_strength_conversion_factor
from pydantic import BaseModel, Field, PositiveFloat
from xopt import VOCS

from scripts.characterize_emittance import characterize_emittance
from scripts.evaluate_function.screen_image import measure_background, \
    measure_beamsize
from scripts.optimize_function import optimize_function


class ScreenEmittanceConfig(BaseModel):
    scan_variable: str
    scan_variable_range: List = Field(min_length=2, max_length=2)
    screen_name: str
    quad_length: PositiveFloat
    drift_length: PositiveFloat
    beam_energy: PositiveFloat
    tuning_variables: Dict[str, List] = None
    roi: list = Field(None, min_length=4, max_length=4)
    threshold: float = 0.0
    measure_background_flag: bool = False
    background_file: str = None
    visualize: bool = False


# define function to measure beam size
def eval_beamsize(input_dict):
    results = measure_beamsize(input_dict)
    results["S_x_mm"] = results["Sx"] * 1e3
    results["S_y_mm"] = results["Sy"] * 1e3

    # add total beam size
    results["total_size"] = np.sqrt(results["Sx"] ** 2 + results["Sy"] ** 2)
    return results


def screen_emittance_measurement_from_config(config: ScreenEmittanceConfig):
    return automatic_screen_emittance_measurement(
        **config.dict()
    )


def automatic_screen_emittance_measurement(
        scan_variable: str,
        scan_variable_range: list,
        screen_name: str,
        quad_length: float,
        drift_length: float,
        beam_energy: float,
        tuning_variables: dict = None,
        roi: list = None,
        threshold: float = 0.0,
        backgroud_file: str = None,
        measure_background_flag: bool = False,
        visualize: bool = False
):
    """
        Function that orchestrates an autonomous quadrupole scan in order to measure the
        beam emittance using a diagnostic screen.

        A region of interest (ROI) is specified as
        :                +------------------+
        :                |                  |
        :              height               |
        :                |                  |
        :               (xy)---- width -----+

        Parameters
        ----------

        scan_variable : str
            Control parameter name for scanning quadrupole focusing strength.

        scan_variable_range : List
            Range of values for scan variable

        screen_name : str
            Diagnostic screen name.

        quad_length : float
            Effective length of quadrupole in [m].

        drift_length : float
            Drift length from quadrupole to beam size measurement location in [m].

        beam_energy : float
            Beam energy in [GeV].

        tuning_variables : dict, optional
            Dict of Xopt-style tuning variable ranges

        roi : list, optional
            list containing roi bounding box elements [x, y, width, height]

        threshold : float, optional
            Optional minimum pixel intensity.

        backgroud_file : str, optional
            Optional path to file containing a background image used to subtract from measurements.

        measure_background_flag : bool, default: False
            Flag to measure background before performing emittance characterization.

        visualize : bool, default: False
            Flag to visualize quadrupole scan modeling and emittance calculation.


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
            Xopt object containing the evaluator, generator, vocs and data objects for
            the quadrupole scan.

        """

    # measure background if requested
    # TODO: add in command to turn beam off
    if measure_background_flag and backgroud_file is None:
        measure_background(screen_name)
        backgroud_file = f"{screen_name}_background.npy".replace(":", "_")
    elif not measure_background_flag and backgroud_file is not None:
        pass
    elif measure_background_flag and backgroud_file is not None:
        RuntimeError("cannot specify a background file and specify measure background "
                     "equals True")

    measurement_options = {
        "screen": screen_name,
        "background": backgroud_file,
        "threshold": threshold,
        "roi": np.array(roi),
        "bb_half_width": 3.0,  # half width of the bounding box in terms of std
        "visualize": True
    }

    image_constraints = {
        "bb_penalty": ["LESS_THAN", 0.0],
        "log10_total_intensity": ["GREATER_THAN", 4]
    }

    return automatic_emittance_measurement(
        eval_beamsize,
        scan_variable_range=scan_variable_range,
        scan_variable=scan_variable,
        quad_length=quad_length,
        drift_length=drift_length,
        beam_energy=beam_energy,
        constants=measurement_options,
        constraints=image_constraints,
        tuning_variables=tuning_variables,
        visualize=visualize
    )


def automatic_emittance_measurement(
        beamsize_callable: Callable,
        scan_variable_range: List,
        scan_variable: str,
        quad_length: float,
        drift_length: float,
        beam_energy: float,
        constants: dict = None,
        constraints: dict = None,
        tuning_variables: dict = None,
        pv_scale_factor: float = 1.0,
        visualize: bool = False,
        optimization_kwargs: Dict = None
):
    """
    Function that orchestrates an autonomous quadrupole scan in order to measure the
    beam emittance.

    Parameters
    ----------
    beamsize_callable : Callable
        Function that retrieves the rms size of the beam in x and y. The function
        should return a dict with the following keys:
        - `S_x_mm`: rms beam size in x in [mm].
        - `S_y_mm`: rms beam size in y in [mm].
        - `total_beam_size`: geometric mean of total beam size in [mm].
        - Any keys specified in the `constraints` dictionary passed to this function.

    scan_variable : str
        Control parameter name for scanning quadrupole focusing strength.

    scan_variable_range : List
        Range of values for scan variable

    quad_length : float
        Effective length of quadrupole in [m].

    drift_length : float
        Drift length from quadrupole to beam size measurement location in [m].

    beam_energy : float
        Beam energy in [GeV].

    constants : dict, optional
        Dict containing constant values to be passed to the `beamsize_callable`
        function.

    constraints : dict, optional
        Dict containing Xopt-like constraint specifications of constraints that
        should be satisfied during the routine.

    tuning_variables : dict, optional
        Dict of Xopt-style tuning variable ranges

    pv_scale_factor : float, default: 1.0
        Scale factor that maps PV values to integrated gradient in [kG].

    visualize : bool, default: False
        Flag to visualize quadrupole scan modeling and emittance calculation.

    optimization_kwargs : dict, optional
        Keyword arguments passed to tuning variable optimizer.


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
        Xopt object containing the evaluator, generator, vocs and data objects for
        the quadrupole scan.

    """

    # if tuning variables are specified then run optimization to minimize the beam
    # size before performing the quad scan
    if tuning_variables is not None:
        opt_vocs = VOCS(
            variables=tuning_variables,
            constants={scan_variable: 0.0} | constants,
            constraints=constraints,
            objectives={"total_size": "MINIMIZE"}
        )

        opt_x = optimize_function(
            opt_vocs, beamsize_callable, **optimization_kwargs
        )

        tuning_constants = opt_x.data.iloc[-1][tuning_variables].to_dict()

    else:
        tuning_constants = {}

    # perform quadrupole scan
    int_grad_to_geo_focusing_strength = get_quad_strength_conversion_factor(
        beam_energy, quad_length
    )
    quad_str_scale_factor = pv_scale_factor * int_grad_to_geo_focusing_strength

    emit_vocs = VOCS(
        variables={scan_variable: scan_variable_range},
        observables=["S_x_mm", "S_y_mm"],
        constraints=constraints,
        constants=tuning_constants | constants
    )

    emit_results, emit_Xopt = characterize_emittance(
        emit_vocs,
        beamsize_callable,
        quad_length,
        drift_length,
        quad_strength_key=scan_variable,
        quad_strength_scale_factor=quad_str_scale_factor,
        rms_x_key="S_x_mm",
        rms_y_key="S_y_mm",
        quad_scan_analysis_kwargs={"visualize": visualize}
    )

    return emit_results, emit_Xopt
