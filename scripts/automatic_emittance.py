import json
from time import sleep
from typing import List, Callable, Dict

import numpy as np
import pandas as pd
import yaml
from emitopt.utils import get_quad_strength_conversion_factor
from epics import caput, caget_many, caget
from pydantic import BaseModel, PositiveFloat, PositiveInt
from xopt import VOCS

from scripts.characterize_emittance import characterize_emittance
from scripts.optimize_function import optimize_function
from scripts.image import ImageDiagnostic
import os


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
    beam_energy: PositiveFloat

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


class ScreenEmittanceMeasurement(BaseModel):
    image_diagnostic: ImageDiagnostic
    beamline_config: BeamlineConfig
    minimum_log_intensity: PositiveFloat = 4.0
    wait_time: PositiveFloat = 2.0
    n_shots: PositiveInt = 3
    n_init: PositiveInt = 3
    n_iterations: PositiveInt = 5
    run_dir: str = None
    secondary_observables: list = []
    constants: dict = {}
    visualize: bool = False

    def eval_beamsize(self, inputs):
        # set PVs
        for k, v in inputs.items():
            print(f'CAPUT {k} {v}')
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
        results["total_size"] = np.sqrt(np.array(results["Sx"]) ** 2 + np.array(results["Sy"]) ** 2)
        return results

    @property
    def measurement_vocs(self):
        # standard image constraints
        IMAGE_CONSTRAINTS = {
            "bb_penalty": ["LESS_THAN", 0.0],
            "log10_total_intensity": ["GREATER_THAN", self.minimum_log_intensity]
        }

        # create measurement vocs
        emit_vocs = VOCS(
            variables={
                self.beamline_config.scan_quad_pv: self.beamline_config.scan_quad_range
            },
            observables=["S_x_mm", "S_y_mm"],
            constraints=IMAGE_CONSTRAINTS,
            constants=self.constants
        )

        return emit_vocs

    def run(self):
        # set up location to store data
        if not os.path.exists(self.run_dir):
            os.mkdir(self.run_dir)

        # grab current point
        current_val = {
            self.beamline_config.scan_quad_pv: caget(self.beamline_config.scan_quad_pv)
        }
        init_point = pd.DataFrame(current_val, index=[0])

        # run scan
        emit_results, emit_Xopt = characterize_emittance(
            self.measurement_vocs,
            self.eval_beamsize,
            self.beamline_config,
            quad_strength_key=self.beamline_config.scan_quad_pv,
            rms_x_key="S_x_mm",
            rms_y_key="S_y_mm",
            initial_data=init_point,
            n_iterations=self.n_iterations,
            generator_kwargs={"turbo_controller":"optimize"},
            quad_scan_analysis_kwargs={"visualize": self.visualize},
            dump_file=f"{self.run_dir}/xopt_run.yml"
        )

        return emit_results, emit_Xopt

    def dump_yaml(self, fname=None):
        """dump data to file"""
        fname = fname or f"{self.run_dir}emittance_config.yaml"
        output = json.loads(self.json())
        with open(fname, "w") as f:
            yaml.dump(output, f)

    def yaml(self):
        return yaml.dump(self.dict(), default_flow_style=None, sort_keys=False)
