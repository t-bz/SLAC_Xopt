from copy import deepcopy
from typing import Dict

import torch
from epics import caget_many

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
