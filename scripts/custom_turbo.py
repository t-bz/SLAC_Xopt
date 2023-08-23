import numpy as np
import pandas as pd
import torch
from botorch.models import ModelListGP
from pandas import DataFrame
from xopt.generators.bayesian.turbo import OptimizeTurboController


class QuadScanTurbo(OptimizeTurboController):
    def get_trust_region(self, model: ModelListGP):
        if not isinstance(model, ModelListGP):
            raise RuntimeError("getting trust region requires a ModelListGP")

        if self.center_x is None:
            raise RuntimeError("need to set best point first, call `update_state`")

        # get bounds width
        bounds = torch.tensor(self.vocs.bounds, **self.tkwargs)
        bound_widths = bounds[1] - bounds[0]

        # Scale the TR to be proportional to the lengthscales of the objective model
        x_center = torch.tensor(
            [self.center_x[ele] for ele in self.vocs.variable_names], **self.tkwargs
        )
        lengthscales = model.models[0].covar_module.base_kernel.lengthscale.detach()

        # calculate the ratios of lengthscales for each axis
        weights = lengthscales / torch.prod(lengthscales) ** (1 / self.dim)

        # calculate the tr bounding box
        tr_lb = torch.clamp(
            x_center - weights * self.length * lengthscales * bound_widths / 2.0,
            *bounds
        )
        tr_ub = torch.clamp(
            x_center + weights * self.length * lengthscales * bound_widths / 2.0,
            *bounds
        )
        return torch.cat((tr_lb, tr_ub), dim=0)

    def update_state(self, data: DataFrame, previous_batch_size: int = 1) -> None:
        """
        Update turbo state class using min of data points that are feasible.
        Otherwise raise an error

        NOTE: this is the opposite of botorch which assumes maximization, xopt assumes
        minimization

        Parameters
        ----------
        data : DataFrame
            Entire data set containing previous measurements. Requires at least one
            valid point.

        previous_batch_size : int, default = 1
            Number of candidates in previous batch evaluation

        Returns
        -------
            None

        """
        # get locations of valid data samples
        feas_data = self.vocs.feasibility_data(data)

        if len(data[feas_data["feasible"]]) == 0:
            raise RuntimeError(
                "turbo requires at least one valid point in training " "dataset"
            )
        else:
            self._set_best_point(data[feas_data["feasible"]])

    def _set_best_point(self, data):
        # aggregate data to get the mean value
        mean_pivot_table = pd.pivot_table(
            data,
            values=self.vocs.objective_names[0],
            columns=self.vocs.variable_names,
            aggfunc=np.mean
        )

        # only use points that have enough data
        min_n_points = 2
        list_pivot_table = pd.pivot_table(
            data,
            values=self.vocs.objective_names[0],
            columns=self.vocs.variable_names,
            aggfunc=list
        )
        valid_points = []
        for name, val in list_pivot_table.to_dict().items():
            if np.count_nonzero(
                    ~np.isnan(
                        np.array(val[self.vocs.objective_names[0]]))
            ) >= min_n_points:
                valid_points += [name]

        mean_pivot_table = mean_pivot_table[valid_points].T

        # get location and value of best (mean) point so far
        best_idx = mean_pivot_table.to_numpy().argmin()
        self.center_x = {mean_pivot_table.index.name: mean_pivot_table.index[best_idx]}
        self.best_value = mean_pivot_table.to_numpy().min()
