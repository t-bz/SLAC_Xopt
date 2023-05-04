import os
import time
import torch
import pandas as pd
from typing import Optional, Callable, Type
from xopt import Xopt
from xopt.vocs import VOCS
from xopt.evaluator import Evaluator
from xopt.generators import UpperConfidenceBoundGenerator, \
    ExpectedImprovementGenerator
from xopt.generators.bayesian.options import ModelOptions
from xopt.generators.bayesian.expected_improvement import BayesianOptions
from xopt.generators.bayesian.upper_confidence_bound import UCBOptions
from gpytorch.means.mean import Mean
from gpytorch.means import ConstantMean

import custom_mean
import dynamic_custom_mean
from dynamic_custom_mean import DynamicCustomMean, Flatten, OccasionalModel, \
    OccasionalConstant
from utils import calc_mae, calc_corr, print_runtime


class BOAgent:
    def __init__(
            self,
            prior_mean: Mean,
            vocs: VOCS,
            bo_config: Optional[dict] = None,
    ):
        """Wrapper for BO runs using Xopt with custom mean definitions.

        Args:
            prior_mean: Prior mean module.
            vocs: VOCS definition for Xopt.
            bo_config: Collection of BO parameters.
        """
        self.mean = prior_mean
        self.vocs = vocs
        if bo_config is None:
            bo_config = {}
        self.bo_config = bo_config
        self.acq_name = bo_config.get("acq_name", "EI")
        self.n_init = bo_config.get("n_init", 3)
        self.n_step = bo_config.get("n_step", 50)
        self.n_run = bo_config.get("n_run", 1)
        self.n_test = bo_config.get("n_test", 10000)
        self.get_optimum = bo_config.get("get_optimum", True)
        self.n_opt = int(self.get_optimum)
        self.objective_name = bo_config.get(
            "objective_name", "negative_sigma_xy")
        if not self.objective_name == "negative_sigma_xy":
            raise ValueError(
                f"objective_name {self.objective_name} is not supported"
            )
        self.path = bo_config.get("path", './BO/')
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        # storage of data, parameters and metrics
        self._data_init, self._data_test = None, None
        self.data = {}
        self.mean_params = {}
        self.mean_variables = {}
        metrics = ["mae_prior", "corr_prior", "mae_posterior",
                   "corr_posterior", "mae_samples", "corr_samples"]
        self.metrics = {}
        for metric in metrics:
            self.metrics[metric] = torch.full(
                (self.n_run, self.n_step), torch.nan)

    def _get_data_set(self, evaluate: Callable, n_samples: int = 3,
                      file: Optional[str] = None) -> pd.DataFrame:
        if os.path.isfile(file):
            data_set = pd.read_csv(file)
        else:
            inputs = [self.vocs.random_inputs() for _ in
                      range(n_samples)]  # seed=0
            outputs = [evaluate(input_dict)[self.objective_name] for input_dict
                       in inputs]
            data_set = pd.DataFrame(inputs)
            data_set[self.objective_name] = outputs
            data_set.to_csv(file)
        return data_set

    def get_init_data(self, evaluate: Callable):
        file = self.path + "data_init.csv"
        self._data_init = self._get_data_set(evaluate, self.n_init, file)

    def get_test_data(self, evaluate: Callable):
        file = self.path + "data_test.csv"
        self._data_test = self._get_data_set(evaluate, self.n_test, file)

    def _calculate_metrics(self, mean, model, run_data) -> dict:
        x_test_variables = torch.tensor(
            self._data_test[self.vocs.variable_names].values)
        y_test = torch.tensor(
            self._data_test[self.vocs.objective_names].values).squeeze()
        y_test_prior = mean(x_test_variables).squeeze()
        y_test_posterior = model.posterior(
            x_test_variables.unsqueeze(1)).mean.detach().squeeze()
        x_samples_variables = torch.tensor(
            run_data[self.vocs.variable_names].values)
        y_samples = torch.tensor(
            run_data[self.vocs.objective_names].values).squeeze()
        y_samples_prior = mean(x_samples_variables).squeeze()
        metrics = {}
        for metric in self.metrics.keys():
            if metric.endswith("prior"):
                if metric.startswith("mae"):
                    v = calc_mae(y_test, y_test_prior)
                elif metric.startswith("corr"):
                    v = calc_corr(y_test, y_test_prior)
                else:
                    raise ValueError(f"Unknown prior metric {metric}")
            elif metric.endswith("posterior"):
                if metric.startswith("mae"):
                    v = calc_mae(y_test, y_test_posterior)
                elif metric.startswith("corr"):
                    v = calc_corr(y_test, y_test_posterior)
                else:
                    raise ValueError(f"Unknown posterior metric {metric}")
            elif metric.endswith("samples"):
                if metric.startswith("mae"):
                    v = calc_mae(y_samples, y_samples_prior)
                elif metric.startswith("corr"):
                    v = calc_corr(y_samples, y_samples_prior)
                else:
                    raise ValueError(f"Unknown samples metric {metric}")
            else:
                raise ValueError(f"Unknown metric {metric}")
            metrics[metric] = v
        return metrics

    def run(self, evaluate: Callable):
        # create initial and test data sets
        if self._data_init is None:
            self.get_init_data(evaluate)
        if self._data_test is None:
            self.get_test_data(evaluate)
        # prepare data storage
        self.data["x_names"] = \
            self.vocs.variable_names + self.vocs.constant_names
        self.data["y_names"] = self.vocs.objective_names
        for key in ["x", "y"]:
            size = (self.n_run, self.n_init + self.n_step + self.n_opt,
                    len(self.data[f"{key}_names"]))
            self.data[key] = torch.full(size=size, fill_value=torch.nan)
        mean_variables = _lookup_mean_variables(self.mean.__class__)
        # run BO
        t0 = time.time()
        for i_run in range(self.n_run):
            # define prior mean
            mean_class = self.mean.__class__
            if issubclass(mean_class, DynamicCustomMean):
                mean = mean_class(self.mean.model, step=0, **self.mean.config)
            else:
                mean = self.mean
            # Xopt definitions
            model_options = ModelOptions(
                name="trainable_mean_standard",
                mean_modules={self.objective_name: mean},
            )
            if self.acq_name == "EI":
                generator_options = BayesianOptions(model=model_options)
                generator = ExpectedImprovementGenerator(
                    self.vocs, options=generator_options)
            else:
                generator_options = UCBOptions(model=model_options)
                generator = UpperConfidenceBoundGenerator(
                    self.vocs, options=generator_options)
            evaluator = Evaluator(function=evaluate)
            X = Xopt(generator=generator, evaluator=evaluator,
                     vocs=self.vocs, data=self._data_init.copy(deep=True))
            # optimization loop
            for i_step in range(self.n_step):
                # define prior mean
                if issubclass(mean_class, DynamicCustomMean):
                    mean = mean_class(self.mean.model, step=i_step,
                                      **self.mean.config)
                else:
                    mean = self.mean
                X.generator.options.model.mean_modules[
                    self.objective_name] = mean
                # optimization step
                X.step()
                # store parameters
                for name, param in mean.named_parameters():
                    if param.requires_grad:
                        if name.startswith("raw_"):
                            name = name[4:]
                        if name not in self.mean_params:
                            self.mean_params[name] = torch.full(
                                (self.n_run, self.n_step), torch.nan)
                        self.mean_params[name][i_run, i_step] = getattr(
                            mean, name).detach()
                # store variables
                for name in mean_variables:
                    if name not in self.mean_variables:
                        self.mean_variables[name] = torch.full(
                            (self.n_run, self.n_step), torch.nan)
                    self.mean_variables[name][i_run, i_step] = getattr(
                        mean, name)
                # store metrics
                with torch.no_grad():
                    metrics = self._calculate_metrics(mean, X.generator.model,
                                                      X.data)
                for metric in self.metrics.keys():
                    self.metrics[metric][i_run, i_step] = metrics[metric]
            # get optimum
            run_data = X.data.copy(deep=True)
            if self.get_optimum:
                x_opt = X.generator.get_optimum()
                y_opt = pd.DataFrame(evaluate(x_opt.to_dict("index")[0]),
                                     index=[0])
                data_opt = pd.concat([x_opt, y_opt], axis=1)
                run_data = pd.concat([run_data, data_opt], axis=0)
            # store data
            self.data["x"][i_run] = torch.tensor(
                run_data[self.vocs.variable_names +
                         self.vocs.constant_names].values
            )
            self.data["y"][i_run] = torch.tensor(
                run_data[self.vocs.objective_names].values)
        # print runtime
        print_runtime(t0, time.time())

    def save(self, file: str, save_history: bool = True):
        mean_config = {}
        if hasattr(self.mean, "config"):
            mean_config.update(self.mean.config)
        data = {
            "mean_name": self.mean.__class__.__name__,
            "mean_config": mean_config,
            "vocs": self.vocs,
            "bo_config": self.bo_config,
            "path": self.path
        }
        if save_history:
            history = {
                "data": self.data,
                "metrics": self.metrics,
                "mean_params": self.mean_params,
                "mean_variables": self.mean_variables
            }
            data.update(history)
        torch.save(data, file)


def load_bo_agent(
        file: str,
        model: Optional[torch.nn.Module] = None,
        step: Optional[int] = None,
        load_history: bool = True) -> BOAgent:
    data = torch.load(file)
    # build prior mean
    mean_name, mean_config = data["mean_name"], data["mean_config"]
    mean_class = _get_mean_class(mean_name)
    if model is not None:
        mean_config["model"] = model
    if step is not None:
        mean_config["step"] = step
    prior_mean = mean_class(**mean_config)
    # build BO agent
    agent = BOAgent(prior_mean, data["vocs"], data["bo_config"])
    if load_history:
        agent.data = data["data"]
        agent.mean_params = data["mean_params"]
        agent.mean_variables = data["mean_variables"]
        agent.metrics = data["metrics"]
    return agent


def _get_mean_class(name: str):
    if name == ConstantMean.__name__:
        return ConstantMean
    elif name in custom_mean.__dict__.keys():
        return custom_mean.__dict__[name]
    elif name in dynamic_custom_mean.__dict__.keys():
        return dynamic_custom_mean.__dict__[name]
    else:
        raise ValueError(f"Unknown mean class: {name}")


def _lookup_mean_variables(mean_class: Type[Mean]) -> list:
    variable_names = []
    if issubclass(mean_class, DynamicCustomMean):
        variable_names.append("step")
        if mean_class.__name__ == Flatten.__name__:
            variable_names.append("w")
        elif mean_class.__name__ in [OccasionalConstant.__name__,
                                     OccasionalModel.__name__]:
            variable_names.append("use_constant")
    return variable_names
