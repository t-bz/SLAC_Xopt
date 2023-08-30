from copy import deepcopy
from typing import Dict

import torch
import numpy as np
import matplotlib.pyplot as plt
from epics import caget_many

from xopt import Xopt
from lume_model.torch.model import PyTorchModel
from lume_model.variables import InputVariable

from xopt import VOCS


def update_variables(
    variable_ranges: Dict,
    input_variables: Dict,
    inputs_small: torch.Tensor,
    from_machine_state: bool = False,
) -> Dict[str, list]:
    updated_variables = {}
    if not from_machine_state:
        # use variable_ranges and update invalid defaults with median from data samples
        for k, v in variable_ranges.items():
            default_on_file = input_variables[k].default
            if not v[0] < default_on_file < v[1]:
                # QUAD:IN20:361:BCTRL: -2.79151
                # QUAD:IN20:371:BCTRL: 2.2584
                # QUAD:IN20:425:BCTRL: 0.131524
                print("Redefined default value for:", k)
                idx = list(input_variables.keys()).index(k)
                sample_median = torch.median(inputs_small[:, idx]).item()
                if not v[0] < sample_median < v[1]:
                    raise ValueError("Sample median not in range!")
                updated_variables[k] = [*v, sample_median]
            else:
                updated_variables[k] = [*v, default_on_file]
    else:
        # get machine values to set ranges and defaults
        machine_values = caget_many(list(variable_ranges.keys()))
        relative_range = 0.05
        for i, k in enumerate(variable_ranges.keys()):
            min_value = machine_values[i] - relative_range * machine_values[i]
            max_value = machine_values[i] + relative_range * machine_values[i]
            updated_variables[k] = [min_value, max_value, machine_values[i]]
    return updated_variables


def create_vocs(
    measurement_options: Dict,
    image_constraints: Dict[str, list],
    updated_variables: Dict[str, list],
    case: int = 1,
) -> VOCS:
    case_1_variables = [
        "SOLN:IN20:121:BCTRL",
        "QUAD:IN20:121:BCTRL",
        "QUAD:IN20:122:BCTRL",
    ]
    if case == 1:
        constants = {k: v for k, v in measurement_options.items()}
        for k, v in updated_variables.items():
            if k not in case_1_variables:
                constants[k] = v[-1]
        vocs = VOCS(
            variables={k: updated_variables[k][:2] for k in case_1_variables},
            constants=constants,
            objectives={"total_size": "MINIMIZE"},
            constraints=image_constraints,
        )
    else:
        vocs = VOCS(
            variables={k: v[:2] for k, v in updated_variables.items()},
            constants=measurement_options,
            objectives={"total_size": "MINIMIZE"},
            constraints=image_constraints,
        )
    return vocs


def update_input_variables_to_transformer(
    lume_model, transformer_loc: int
) -> list[InputVariable]:
    """Returns input variables updated to the transformer at the given location.

    Updated are the value ranges and default of the input variables. This allows, e.g., to add a
    calibration transformer and to update the input variable specification accordingly.

    Args:
        lume_model: The LUME-model for which the input variables shall be updated.
        transformer_loc: The location of the input transformer to adjust for.

    Returns:
        The updated input variables.
    """
    x_old = {
        "min": torch.tensor(
            [var.value_range[0] for var in lume_model.input_variables.values()],
            dtype=torch.double,
        ),
        "max": torch.tensor(
            [var.value_range[1] for var in lume_model.input_variables.values()],
            dtype=torch.double,
        ),
        "default": torch.tensor(
            [var.default for var in lume_model.input_variables.values()],
            dtype=torch.double,
        ),
    }
    x_new = {}
    for key in x_old.keys():
        x = x_old[key]
        # compute previous limits at transformer location
        for i in range(transformer_loc):
            x = lume_model.input_transformers[i].transform(x)
        # untransform of transformer to adjust for
        x = lume_model.input_transformers[transformer_loc].untransform(x)
        # backtrack through transformers
        for transformer in lume_model.input_transformers[:transformer_loc][::-1]:
            x = transformer.untransform(x)
        x_new[key] = x
    updated_variables = deepcopy(lume_model.input_variables)
    for i, var in enumerate(updated_variables.values()):
        var.value_range = [x_new["min"][i].item(), x_new["max"][i].item()]
        var.default = x_new["default"][i].item()
    return updated_variables


def plot_model_in_2d(
    X: Xopt,
    output_name: str = None,
    variable_names: tuple[str, str] = None,
    n_grid: int = 50,
    figsize: tuple[float, float] = (10, 8),
) -> tuple[plt.Figure, np.ndarray]:
    """Displays GP model predictions for selected output in 2D.

    The GP model is displayed with respect to the two named input variables. If None are given,
    the list of variables in X.vocs is used. Feasible samples are indicated with orange "+"-marks,
    infeasible samples with red "x"-marks. Feasibility is calculated with respect to all constraints
    unless the selected output is a constraint itself, in which case only that one is considered.

    Args:
        X: Xopt object with a Bayesian generator containing a trained GP model.
        output_name: Selects the GP model to display.
        variable_names: The two variables for which the model is displayed.
          Defaults to X.vocs.variable_names.
        n_grid: Number of grid points per dimension used to display the model predictions.
        figsize: Size of the matplotlib figure.

    Returns:
        The matplotlib figure and axes objects.
    """

    # define output and variable names
    if output_name is None:
        output_name = X.vocs.output_names[0]
    if variable_names is None:
        variable_names = X.vocs.variable_names
    if not len(variable_names) == 2:
        raise ValueError(f"Number of variables should be 2, not {len(variable_names)}.")

    # generate input mesh
    x_last = X.data[X.vocs.variable_names].iloc[-1]
    x_lim = torch.tensor([X.vocs.variables[k] for k in variable_names])
    x_i = [torch.linspace(*x_lim[i], n_grid) for i in range(x_lim.shape[0])]
    x_mesh = torch.meshgrid(*x_i, indexing="ij")
    x_v = torch.hstack([ele.reshape(-1, 1) for ele in x_mesh]).double()
    x = torch.stack(
        [x_v[:, variable_names.index(k)] if k in variable_names else x_last[k] * torch.ones(x_v.shape[0])
         for k in X.vocs.variable_names],
        dim=-1,
    )

    # compute model predictions
    gp = X.generator.model.models[X.generator.vocs.output_names.index(output_name)]
    with torch.no_grad():
        _x = gp.input_transform.transform(x)
        _x = gp.mean_module(_x)
        prior_mean = gp.outcome_transform.untransform(_x)[0]
        posterior_mean = gp.posterior(x).mean
        posterior_std = torch.sqrt(torch.diagonal(gp.posterior(x).mvn.covariance_matrix))

    # determine feasible samples
    if "feasible_" + output_name in X.vocs.feasibility_data(X.data).columns:
        feasible = X.vocs.feasibility_data(X.data)["feasible_" + output_name]
    else:
        feasible = X.vocs.feasibility_data(X.data)["feasible"]
    feasible_samples = X.data[variable_names][feasible]
    infeasible_samples = X.data[variable_names][~feasible]

    # plot data
    nrows, ncols = 2, 2
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10, 8))
    z = [posterior_mean, prior_mean, posterior_std]
    labels = ["Posterior Mean", "Prior Mean", "Posterior SD"]
    for i in range(nrows * ncols):
        ax = axs[i // ncols, i % nrows]
        if i >= len(z):
            ax.axis("off")
        else:
            pcm = ax.pcolormesh(*x_mesh, z[i].detach().squeeze().reshape(n_grid, n_grid))
            if not feasible_samples.empty:
                ax.plot(*feasible_samples.to_numpy().T, "+C1")
            if not infeasible_samples.empty:
                ax.plot(*infeasible_samples.to_numpy().T, "xC3")
            ax.locator_params(axis="both", nbins=5)
            ax.set_title(labels[i])
            ax.set_xlabel(variable_names[0])
            if i % nrows == 0:
                ax.set_ylabel(variable_names[1])
            cbar = fig.colorbar(pcm, ax=ax)
            cbar.set_label(output_name)
    fig.tight_layout()

    return fig, axs