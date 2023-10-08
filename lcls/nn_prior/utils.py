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


def display_model(
    X: Xopt,
    output_name: str = None,
    variable_names: tuple[str, str] = None,
    idx: int = -1,
    reference_point: dict = None,
    constrained_acqf: bool = True,
    n_grid: int = 50,
    figsize: tuple[float, float] = None,
    show_samples: bool = True,
    fading_samples: bool = True,
) -> tuple[plt.Figure, np.ndarray]:
    """Displays GP model predictions for selected output.

    The GP model is displayed with respect to the named input variables. If None are given,
    the list of variables in X.vocs is used. Feasible samples are indicated with orange "+"-marks,
    infeasible samples with red "o"-marks. Feasibility is calculated with respect to all constraints
    unless the selected output is a constraint itself, in which case only that one is considered.

    Args:
        X: Xopt object with a Bayesian generator containing a trained GP model.
        output_name: Selects the GP model to display.
        variable_names: The variables for which the model is displayed (maximum of 2).
          Defaults to X.vocs.variable_names.
        idx: Index of the last sample to use.
        reference_point: Reference point determining the value of variables not in variable_names.
          Defaults to last used sample.
        constrained_acqf: Determines whether the constrained or base acquisition function is shown.
        n_grid: Number of grid points per dimension used to display the model predictions.
        figsize: Size of the matplotlib figure. Defaults to (6, 4) for 1D and (10, 8) for 2D.
        show_samples: Determines whether samples are shown.
        fading_samples: Determines whether older samples are shown as more transparent.

    Returns:
        The matplotlib figure and axes objects.
    """

    # define output and variable names
    if output_name is None:
        output_name = X.vocs.output_names[0]
    if variable_names is None:
        variable_names = X.vocs.variable_names
    dim = len(variable_names)
    if not dim in [1, 2]:
        raise ValueError(f"Number of variables should be 1 or 2, not {dim}.")

    # generate input mesh
    if reference_point is None:
        reference_point = X.data[X.vocs.variable_names].iloc[idx].to_dict()
    x_lim = torch.tensor([X.vocs.variables[k] for k in variable_names])
    x_i = [torch.linspace(*x_lim[i], n_grid) for i in range(x_lim.shape[0])]
    x_mesh = torch.meshgrid(*x_i, indexing="ij")
    x_v = torch.hstack([ele.reshape(-1, 1) for ele in x_mesh]).double()
    x = torch.stack(
        [x_v[:, variable_names.index(k)] if k in variable_names else reference_point[k] * torch.ones(x_v.shape[0])
         for k in X.vocs.variable_names],
        dim=-1,
    )

    # compute model predictions
    gp = X.generator.model.models[X.generator.vocs.output_names.index(output_name)]
    with torch.no_grad():
        _x = gp.input_transform.transform(x)
        _x = gp.mean_module(_x)
        prior_mean = gp.outcome_transform.untransform(_x)[0]
        posterior = gp.posterior(x)
        posterior_mean = posterior.mean
        posterior_sd = torch.sqrt(posterior.mvn.variance)
        if constrained_acqf:
            acqf_values = X.generator.get_acquisition(X.generator.model)(x.unsqueeze(1))
        else:
            acqf_values = X.generator.get_acquisition(X.generator.model).base_acqusition(x.unsqueeze(1))

    # determine feasible samples
    max_idx = idx + 1
    if max_idx == 0:
        max_idx = None
    if "feasible_" + output_name in X.vocs.feasibility_data(X.data).columns:
        feasible = X.vocs.feasibility_data(X.data).iloc[:max_idx]["feasible_" + output_name]
    else:
        feasible = X.vocs.feasibility_data(X.data).iloc[:max_idx]["feasible"]
    feasible_samples = X.data.iloc[:max_idx][variable_names][feasible]
    feasible_index = X.data.iloc[:max_idx].index.values.astype(int)[feasible]
    infeasible_samples = X.data.iloc[:max_idx][variable_names][~feasible]
    infeasible_index = X.data.iloc[:max_idx].index.values.astype(int)[~feasible]
    idx_min = np.min(X.data.iloc[:max_idx].index.values.astype(int))
    idx_max = np.max(X.data.iloc[:max_idx].index.values.astype(int))
    alpha_min = 0.1

    # plot configuration
    if dim == 1:
        sharex = True
        nrows, ncols = 2, 1
        if figsize is None:
            figsize = (6, 4)
    else:
        sharex = False
        nrows, ncols = 2, 2
        if figsize is None:
            figsize = (10, 8)
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, sharex=True)

    # plot data
    z = [posterior_mean, prior_mean, posterior_sd, acqf_values]
    labels = ["Posterior Mean", "Prior Mean", "Posterior SD"]
    if dim == 1:
        labels[2] = "Posterior CL ($\pm 2\,\sigma$)"
    if constrained_acqf:
        labels.append("Constrained Acquisition Function")
    else:
        labels.append("Base Acquisition Function")

    for i in range(nrows * ncols):
        ax = axs.flatten()[i]
        
        # base plot
        if dim == 1:
            x_axis = x[:, X.vocs.variable_names.index(variable_names[0])].squeeze().numpy()
            if i == 0:
                ax.plot(x_axis, z[1].detach().squeeze().numpy(), "C2--", label=labels[1])
                ax.plot(x_axis, z[0].detach().squeeze().numpy(), "C0", label=labels[0])
                ax.fill_between(x_axis, z[0].detach().squeeze().numpy() - 2 * z[2].detach().squeeze().numpy(),
                                z[0].detach().squeeze().numpy() + 2 * z[2].detach().squeeze().numpy(),
                                color="C0", alpha=0.25, label=labels[2])
            else:
                ax.plot(x_axis, z[-1].detach().squeeze().numpy(), label=labels[-1])
        else:
            pcm = ax.pcolormesh(x_mesh[0].numpy(), x_mesh[1].numpy(), z[i].detach().squeeze().reshape(n_grid, n_grid).numpy())
        
        # plot samples
        if show_samples:
            x_0_feasible, x_1_feasible = None, None
            x_0_infeasible, x_1_infeasible = None, None
            if dim == 1 and i == 0:
                if not feasible_samples.empty:
                    x_0_feasible = feasible_samples.to_numpy()
                    x_1_feasible = X.data.iloc[:max_idx][output_name][feasible].to_numpy()
                if not infeasible_samples.empty:
                    x_0_infeasible = infeasible_samples.to_numpy()
                    x_1_infeasible = X.data.iloc[:max_idx][output_name][~feasible].to_numpy()
            elif dim == 2:
                if not feasible_samples.empty:
                    x_0_feasible, x_1_feasible = feasible_samples.to_numpy().T
                if not infeasible_samples.empty:
                    x_0_infeasible, x_1_infeasible = infeasible_samples.to_numpy().T
            if x_0_feasible is not None and x_1_feasible is not None:
                if fading_samples and idx_min < idx_max:
                    for j in range(len(feasible_index)):
                        alpha = alpha_min + (1 - alpha_min) * ((feasible_index[j] - idx_min) / (idx_max - idx_min))
                        ax.scatter(x_0_feasible[j], x_1_feasible[j], marker="+", c="C1", alpha=alpha)
                else:
                    ax.scatter(x_0_feasible, x_1_feasible, marker="+", c="C1")
            if x_0_infeasible is not None and x_1_infeasible is not None:
                if fading_samples and idx_min < idx_max:
                    for j in range(len(infeasible_index)):
                        alpha = alpha_min + (1 - alpha_min) * ((infeasible_index[j] - idx_min) / (idx_max - idx_min))
                        ax.scatter(x_0_infeasible[j], x_1_infeasible[j], marker="o", c="C3", alpha=alpha)
                else:
                    ax.scatter(x_0_infeasible, x_1_infeasible, marker="o", c="C3")

        # plot labels
        if dim == 1:
            if i == 0:
                ax.set_ylabel(output_name)
            else:
                ax.set_xlabel(variable_names[0])
                ax.set_ylabel(r"$\alpha\,$[{}]".format(X.vocs.output_names[0]))
            ax.legend()
        else:
            ax.locator_params(axis="both", nbins=5)
            ax.set_title(labels[i])
            ax.set_xlabel(variable_names[0])
            if i % nrows == 0:
                ax.set_ylabel(variable_names[1])
            cbar = fig.colorbar(pcm, ax=ax)
            if i == 2:
                cbar_label = r"$\sigma\,$[{}]".format(output_name)
            elif i == 3:
                cbar_label = r"$\alpha\,$[{}]".format(X.vocs.output_names[0])
            else:
                cbar_label = output_name
            cbar.set_label(cbar_label)
    
    fig.tight_layout()
    return fig, axs