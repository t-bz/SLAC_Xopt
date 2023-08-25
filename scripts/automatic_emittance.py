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


class BeamlineConfig(BaseModel):
    # parameters of the measurement quadrupole
    scan_quad_pv: str
    scan_quad_range: List[float]
    scan_quad_length: PositiveFloat
    pv_to_integrated_gradient: float = 1.0

    # parameters of the transport from the quad to the screen
    transport_matrix_x: List[List]
    transport_matrix_y: List[List]

    # beam parameters
    beam_energy: PositiveFloat # in GeV

    # design twiss for matching calculation
    design_beta_x: float = None
    design_beta_y: float = None
    design_alpha_x: float = None
    design_alpha_y: float = None

    @property
    def pv_to_focusing_strength(self):
        """
        returns a scale factor that translates pv values to geometric focusing
        strength
        """
        int_grad_to_geo_focusing_strength = get_quad_strength_conversion_factor(
            self.beam_energy, self.scan_quad_length
        )
        return self.pv_to_integrated_gradient * int_grad_to_geo_focusing_strength

    @property
    def design_twiss(self):
        return [
            self.design_beta_x,
            self.design_alpha_x,
            self.design_beta_y,
            self.design_alpha_y,
        ]


class BaseEmittanceMeasurement(BaseModel, ABC):
    beamline_config: BeamlineConfig
    wait_time: PositiveFloat = 2.0
    n_iterations: PositiveInt = 5
    turbo_length: PositiveFloat = 1.0
    run_dir: str = os.getcwd()
    secondary_observables: list = []
    constants: dict = {}
    visualize: bool = False
    _dump_file: str = None

    class Config:
        extra = "forbid"
        underscore_attrs_are_private = True

    @abstractmethod
    def eval_beamsize(self, inputs):
        pass

    @property
    def dump_file(self):
        return self._dump_file

    @property
    @abstractmethod
    def x_measurement_vocs(self):
        pass

    @property
    @abstractmethod
    def y_measurement_vocs(self):
        pass

    @abstractmethod
    def get_initial_points(self):
        pass

    def dump_yaml(self, fname=None):
        """dump data to file"""
        fname = fname or f"{self.run_dir}emittance_config.yaml"
        output = json.loads(self.json())
        with open(fname, "w") as f:
            yaml.dump(output, f)

    def yaml(self):
        return yaml.dump(self.dict(), default_flow_style=None, sort_keys=False)

    def run(self):
        # set up location to store data
        if not os.path.exists(self.run_dir):
            os.mkdir(self.run_dir)

        self._dump_file = os.path.join(
            self.run_dir, f"emittance_characterize_{int(time())}.yml"
        )

        # run scan
        emit_results, emit_Xopt = characterize_emittance(
            self.x_measurement_vocs,
            self.y_measurement_vocs,
            self.eval_beamsize,
            self.beamline_config,
            quad_strength_key=self.beamline_config.scan_quad_pv,
            rms_x_key="S_x_mm",
            rms_y_key="S_y_mm",
            initial_points=self.get_initial_points(),
            n_iterations=self.n_iterations,
            turbo_length=self.turbo_length,
            quad_scan_analysis_kwargs={"visualize": self.visualize},
            dump_file=self.dump_file,
        )
        

        return emit_results, emit_Xopt


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

    @property
    def base_vocs(self):
        IMAGE_CONSTRAINTS = {
            "bb_penalty": ["LESS_THAN", 0.0],
            "log10_total_intensity": ["GREATER_THAN", self.minimum_log_intensity],
        }

        # create measurement vocs
        base_vocs = VOCS(
            variables={
                self.beamline_config.scan_quad_pv: self.beamline_config.scan_quad_range
            },
            constraints=IMAGE_CONSTRAINTS,
            observables=["S_x_mm","S_y_mm"],
            constants=self.constants,
        )

        return base_vocs
        

    @property
    def x_measurement_vocs(self):
        vocs = deepcopy(self.base_vocs)  
        vocs.objectives={"S_x_mm":"MINIMIZE"}

        return vocs

    @property
    def y_measurement_vocs(self):
        vocs = deepcopy(self.base_vocs)   
        vocs.objectives={"S_y_mm":"MINIMIZE"}

        return vocs


    def get_initial_points(self):
        # grab current point
        current_val = {
            self.beamline_config.scan_quad_pv: caget(self.beamline_config.scan_quad_pv)
        }
        init_point = pd.DataFrame(current_val, index=[0])
        # append two random points
        init_point = pd.concat(
            (
                init_point, pd.DataFrame(self.x_measurement_vocs.random_inputs(2))
            ), ignore_index=True)
        
        return init_point
