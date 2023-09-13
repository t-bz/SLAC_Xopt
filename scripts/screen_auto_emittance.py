import json
import os
from abc import ABC, abstractmethod
from time import sleep, time
from typing import Callable, Dict, List
from copy import deepcopy

import numpy as np
import pandas as pd
import yaml
from emitopt.utils import get_quad_strength_conversion_factor
from epics import caget, caget_many, caput
from pydantic import BaseModel, PositiveFloat, PositiveInt
from xopt import VOCS

from scripts.characterize_emittance import characterize_emittance
from scripts.image import ImageDiagnostic
from scripts.automatic_emittance import BaseEmittanceMeasurement, BeamlineConfig

import pandas as pd
def explode_all_columns(data: pd.DataFrame):
    """explode all data columns in dataframes that are lists or np.arrays"""
    list_types = []
    for name, val in data.iloc[0].items():
        if isinstance(val, list) or isinstance(val, np.ndarray):
            list_types += [name]

    if len(list_types):
        try:
            return data.explode(list_types)
        except ValueError:
            raise ValueError("evaluator outputs that are lists must match in size")
    else:
        return data

class ScreenEmittanceMeasurement(BaseEmittanceMeasurement):
    image_diagnostic: ImageDiagnostic
    minimum_log_intensity: PositiveFloat = 4.0
    n_shots: PositiveInt = 3

    def eval_beamsize(self, inputs):
        # set PVs
        for k, v in inputs.items():
            print(f"CAPUT {k} {v}")
            caput(k, v)

        sleep(self.wait_time)

        # get beam sizes from image diagnostic
        results = self.image_diagnostic.measure_beamsize(self.n_shots, **inputs)
        results["S_x_mm"] = np.array(results["Sx"]) * 1e-3
        results["S_y_mm"] = np.array(results["Sy"]) * 1e-3

        # get other PV's NOTE: Measurements not synchronous with beamsize measurements!
        results = results | dict(
            zip(self.secondary_observables, caget_many(self.secondary_observables))
        )

        # add total beam size
        results["total_size"] = np.sqrt(
            np.array(results["Sx"]) ** 2 + np.array(results["Sy"]) ** 2
        )
        return results

    def fast_scan(self, n_points=5):
        """ 
        perform a fast, rough scan of the parameter space 
        
        """
        old_pv_value = caget(self.beamline_config.scan_quad_pv)
        
        scan_points = np.linspace(
            *self.beamline_config.scan_quad_range,
            n_points
        )

        print(f"CAPUT {self.beamline_config.scan_quad_pv} {scan_points[0]}")
        caput(self.beamline_config.scan_quad_pv, scan_points[0])
        sleep(3.0)

        results = []
        for point in scan_points:
            print(f"CAPUT {self.beamline_config.scan_quad_pv} {point}")
            caput(self.beamline_config.scan_quad_pv, point)

            sleep(1.0)

            result = self.image_diagnostic.measure_beamsize(3)
            result["S_x_mm"] = np.array(result["Sx"]) * 1e-3
            result["S_y_mm"] = np.array(result["Sy"]) * 1e-3
            result[self.beamline_config.scan_quad_pv] = point
            results += [result]
            

        # reset old pv
        caput(self.beamline_config.scan_quad_pv, old_pv_value)

        return explode_all_columns(pd.DataFrame(results))
            

    @property
    def base_vocs(self):
        IMAGE_CONSTRAINTS = {
            "bb_penalty": ["LESS_THAN", 0.0],
            "log10_total_intensity": ["GREATER_THAN", self.image_diagnostic.min_log_intensity],
        }

        # create measurement vocs
        base_vocs = VOCS(
            variables={
                self.beamline_config.scan_quad_pv: self.beamline_config.scan_quad_range
            },
            constraints=IMAGE_CONSTRAINTS,
            observables=["S_x_mm", "S_y_mm"],
            constants=self.constants,
        )

        return base_vocs

    @property
    def x_measurement_vocs(self):
        vocs = deepcopy(self.base_vocs)
        vocs.objectives = {"S_x_mm": "MINIMIZE"}

        return vocs

    @property
    def y_measurement_vocs(self):
        vocs = deepcopy(self.base_vocs)
        vocs.objectives = {"S_y_mm": "MINIMIZE"}

        return vocs

    def get_initial_data(self):
        return self.fast_scan()        

    # def get_initial_points(self):
    #     # grab current point
    #     current_val = {
    #         self.beamline_config.scan_quad_pv: caget(self.beamline_config.scan_quad_pv)
    #     }
    #     init_point = pd.DataFrame(current_val, index=[0])
    #     # append two random points
    #     init_point = pd.concat(
    #         (init_point, pd.DataFrame(self.x_measurement_vocs.random_inputs(2))),
    #         ignore_index=True,
    #     )

    #     return init_point