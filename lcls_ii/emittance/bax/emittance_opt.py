import os.path

# add to python path
import sys
import time
from copy import deepcopy

from typing import Dict, List, Optional

import numpy as np

import pandas as pd
import torch
import yaml
from epics import caget_many
from gpytorch.kernels import MaternKernel, PolynomialKernel, ScaleKernel

from pydantic import (
    BaseModel,
    conlist,
    DirectoryPath,
    FilePath,
    PositiveFloat,
    PositiveInt,
)

sys.path.append("../../")
sys.path.append("../../../")

from lcls_ii.common import get_pv_objects, measure_pvs, set_magnet_strengths
from lcls_ii.emittance.bax.test_algorithm import MinimizeEmitBmag
from scripts.image import ImageDiagnostic
from xopt import Evaluator, VOCS, Xopt
from xopt.generators.bayesian import BayesianExplorationGenerator
from xopt.generators.bayesian.bax_generator import BaxGenerator
from xopt.generators.bayesian.models.standard import StandardModelConstructor
from xopt.numerical_optimizer import GridOptimizer
from xopt.utils import get_local_region


class BAXEmittanceOpt(BaseModel):
    # filesystem settings
    data_dir: DirectoryPath
    pv_list: FilePath
    image_diagnostic: ImageDiagnostic

    # problem definition
    tuning_variables: Dict[str, conlist(float, min_length=2, max_length=2)]
    measurement_variable: Dict[str, conlist(float, min_length=2, max_length=2)]

    # beam physics properties
    quad_length: PositiveFloat
    rmat_x: List
    rmat_y: List
    beam_energy_gev: PositiveFloat
    pv_to_geometric_focusing_strength_scale: PositiveFloat
    x_design_twiss: Optional[List]
    y_design_twiss: Optional[List]

    # algorithm properties
    n_interpolate_points: Optional[PositiveInt] = 3
    n_grid_points: Optional[PositiveInt] = 5
    n_bax_samples: Optional[PositiveInt] = 20
    n_virtual_fitting_points: Optional[PositiveInt] = 11
    jitter: Optional[PositiveFloat] = 0.1

    n_random_samples: Optional[PositiveInt] = 10
    n_exploration_steps: Optional[PositiveInt] = 5
    n_optimization_steps: Optional[PositiveInt] = 10
    local_region_fraction: Optional[PositiveFloat] = 0.25

    X_bayes_exp: Optional[Xopt] = None
    X_bax: Optional[Xopt] = None

    class Config:
        extra = "forbid"
        arbitrary_types_allowed = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # load in pv objects
        self._pv_objects = get_pv_objects(self.pv_list)

    def get_vocs(self):
        vocs = VOCS(
            variables=self.tuning_variables | self.measurement_variable,
            constraints={"bb_penalty": ["LESS_THAN", 0.0]},
            observables=["xrms_sq", "yrms_sq"],
        )

        return vocs

    def get_tune_meas_dims(self, vocs: VOCS):
        # ID which index is the measurement dim
        variable_names = vocs.variable_names
        measurement_dim = variable_names.index(
            list(self.measurement_variable.keys())[0]
        )
        tuning_dims = list(range(len(variable_names)))
        tuning_dims.remove(measurement_dim)

        return tuning_dims, measurement_dim

    def get_model_construtor(self):
        vocs = self.get_vocs()

        tuning_dims, measurement_dim = self.get_tune_meas_dims(vocs)

        covar_module = MaternKernel(
            ard_num_dims=len(tuning_dims),
            active_dims=tuning_dims,
            lengthscale_prior=None,
        ) * PolynomialKernel(power=2, active_dims=[measurement_dim])
        scaled_covar_module = ScaleKernel(covar_module)

        # prepare options for Xopt generator
        covar_module_dict = {
            "xrms_sq": scaled_covar_module,
            "yrms_sq": deepcopy(scaled_covar_module),
        }
        # covar_module_dict = {}
        model_constructor = StandardModelConstructor(
            covar_modules=covar_module_dict, use_low_noise_prior=True
        )

        return model_constructor

    def measure_beamsize(self, inputs):
        pv_objects = self._pv_objects
        set_magnet_strengths(inputs, pv_objects, validate=True)
        time.sleep(0.5)
        # measure all pvs - except for names in inputs
        results = measure_pvs(
            [name for name in pv_objects.keys() if name not in inputs], pv_objects
        )

        # do some calculations
        results["time"] = time.time()

        # add beam size measurement to results dict
        beamsize_results = self.image_diagnostic.measure_beamsize(1)
        results["Sx_mm"] = np.array(beamsize_results["Sx"]) * 1e-3
        results["Sy_mm"] = np.array(beamsize_results["Sy"]) * 1e-3

        # add beam size squared (mm^2)
        results["xrms_sq"] = results["Sx_mm"] ** 2
        results["yrms_sq"] = results["Sy_mm"] ** 2
        results = beamsize_results | results
        return results

    def run_bayesian_exploration(self, dump_filename: str = "bayes_exp.yml"):
        # create Xopt components
        vocs = self.get_vocs()
        model_constructor = self.get_model_construtor()

        generator = BayesianExplorationGenerator(
            vocs=vocs,
            gp_constructor=model_constructor,
            numerical_optimizer=GridOptimizer(n_grid_points=self.n_grid_points),
            n_interpolate_points=self.n_interpolate_points,
        )

        evaluator = Evaluator(function=self.measure_beamsize)

        self.X_bayes_exp = Xopt(
            vocs=vocs,
            generator=generator,
            evaluator=evaluator,
            strict=True,
            dump_file=os.path.join(self.data_dir, dump_filename),
        )

        # random initial evaluation
        current_value = dict(
            zip(
                self.X_bayes_exp.vocs.variable_names,
                caget_many(self.X_bayes_exp.vocs.variable_names),
            )
        )
        random_sample_region = get_local_region(
            current_value, self.X_bayes_exp.vocs, fraction=self.local_region_fraction
        )
        self.X_bayes_exp.random_evaluate(
            self.n_random_evaluate, custom_bounds=random_sample_region
        )

        # running Bayesian exploration
        start = time.time()
        for i in range(self.n_steps):
            self.X_bayes_exp.step()
        print(time.time() - start)

        return self.X_bayes_exp

    def get_algorithm(self):
        tuning_dim, measurement_dim = self.get_tune_meas_dims(self.get_vocs())

        # define BAX generator
        algo_kwargs = {
            "x_key": "xrms_sq",
            "y_key": "yrms_sq",
            "scale_factor": self.pv_to_geometric_focusing_strength_scale,
            "q_len": self.quad_length,
            "rmat_x": torch.tensor(self.rmat_x),
            "rmat_y": torch.tensor(self.rmat_y),
            "n_samples": self.n_bax_samples,
            "meas_dim": measurement_dim,
            "n_steps_measurement_param": self.n_virtual_fitting_points,
            "thick_quad": False,
            "init": "smallest",
            "scipy_options": {"maxiter": 25},
            "jitter": self.jitter,
            "twiss0_x": torch.tensor(self.x_design_twiss),
            "twiss0_y": torch.tensor(self.y_design_twiss),
        }
        algo = MinimizeEmitBmag(**algo_kwargs)

        return algo

    def run_emittance_optimization(
        self, load_data_filename: str = None, dump_filename: str = "emittance_opt.yml"
    ):
        # define Xopt object
        # create Xopt components
        vocs = self.get_vocs()
        model_constructor = self.get_model_construtor()

        evaluator = Evaluator(function=self.measure_beamsize)

        generator = BaxGenerator(
            vocs=vocs,
            gp_constructor=model_constructor,
            algorithm=self.get_algorithm(),
            n_interpolate_points=self.n_interpolate_points,
        )

        self.X_bax = Xopt(
            evaluator=evaluator,
            generator=generator,
            vocs=vocs,
            dump_file=os.path.join(self.data_dir, dump_filename),
        )

        # load previous data (possibly from file)
        if self.X_bayes_exp is not None:
            self.X_bax.add_data(self.X_bayes_exp.data)
        else:
            if load_data_filename is not None:
                self.X_bax.add_data(pd.DataFrame(yaml.safe_load(load_data_filename)))
            else:
                raise RuntimeError(
                    "need to run bayesian exploration first or provide "
                    "a data yaml file to initialize BAX"
                )

        # run BAX
        start = time.time()
        for i in range(self.n_steps):
            print(i)
            self.X_bax.step()
        print(time.time() - start)

        return self.X_bax
