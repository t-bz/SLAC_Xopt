from typing import Dict

import torch
from epics import caget_many

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
    case_1_variables = ["SOLN:IN20:121:BCTRL", "QUAD:IN20:121:BCTRL", "QUAD:IN20:122:BCTRL"]
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
