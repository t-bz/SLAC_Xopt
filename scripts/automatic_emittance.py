import json
import os
from abc import ABC, abstractmethod
from time import sleep
import time
from typing import Callable, Dict, List
from copy import deepcopy
import traceback

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
    beam_energy: PositiveFloat  # in GeV

    # design twiss for matching calculation
    design_beta_x: float = None
    design_beta_y: float = None
    design_alpha_x: float = None
    design_alpha_y: float = None

    @property
    def pv_to_focusing_strength(self):
        """
        returns a scale factor that translates pv values to geometric focusing
        strength in m^{-2}
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
    visualize: int = 0
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

    def get_initial_points(self):
        return None

    def get_initial_data(self):
        return None

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
            self.run_dir, f"emittance_characterize_{int(time.time())}.yml"
        )

        # generate initial points
        print("getting initial points to measure")
        initial_points = self.get_initial_points()
        #print(initial_points)

        # generate initial data
        print("getting initial data")
        start = time.perf_counter()
        initial_data = self.get_initial_data()
        print(f"initial data gathering took: {time.perf_counter() - start} s")

        # get old setting
        old_pv_value = caget(self.beamline_config.scan_quad_pv)

        
        # run scan
        try:
            emit_results, emit_Xopt = characterize_emittance(
                self.x_measurement_vocs,
                self.y_measurement_vocs,
                self.eval_beamsize,
                self.beamline_config,
                quad_strength_key=self.beamline_config.scan_quad_pv,
                rms_x_key="S_x_mm",
                rms_y_key="S_y_mm",
                initial_points=initial_points,
                initial_data=initial_data,
                n_iterations=self.n_iterations,
                turbo_length=self.turbo_length,
                visualize=self.visualize,
                dump_file=self.dump_file,
            )
    
            # add self info to dump file
            info = yaml.safe_load(open(self.dump_file))
            info = info | {"emittance_measurement": self.dict()}
    
            with open(self.dump_file, "w") as f:
                yaml.dump(info, f)

        except Exception:
            print(traceback.format_exc())
        finally:
            caput(self.beamline_config.scan_quad_pv,old_pv_value)
        
        return emit_results, emit_Xopt
        

