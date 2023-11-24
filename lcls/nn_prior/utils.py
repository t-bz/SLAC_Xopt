import datetime
import os
from copy import deepcopy
from typing import Any, Union

import numpy as np
import pandas as pd
import torch
import yaml
from epics import caget_many
from torch import Tensor

from lume_model.variables import InputVariable
from lume_model.models import TorchModel, TorchModule
from xopt.generators.bayesian.bayesian_generator import BayesianGenerator


class ObjectiveModel(torch.nn.Module):
    """Defines how to compute the objective based on model predictions"""
    def __init__(
        self,
        model: TorchModule,
        objective_scale: float = 1e-3,  # in mm
        objective_offset: float = None,
        include_roundness: bool = True,
    ):
        """Initializes ObjectiveModel.

        Args:
            model: LCLS injector surrogate model.
            objective_scale: Scaling factor for the objective. Defaults to 1e-3 which yields a transverse
              beamsize given in mm.
            objective_offset: Adds a fixed offset to the objective.
            include_roundness: Whether to include a term penalizing asymmetric beam shapes.
        """
        super(ObjectiveModel, self).__init__()
        self.model = model
        self.objective_scale = objective_scale
        self.objective_offset = objective_offset
        if objective_offset is None:
            self.objective_offset = 0.0
        self.include_roundness = include_roundness

    def function(self, sigma_x: Tensor, sigma_y: Tensor) -> Tensor:
        """Computes the objective from the given beamsizes in x and y.

        Args:
            sigma_x: Beamsize in x (in micrometer).
            sigma_y: Beamsize in y (in micrometer).

        Returns:
            The objective.
        """
        # using this calculation due to occasional negative values
        sigma_xy = torch.sqrt(sigma_x ** 2 + sigma_y ** 2)
        if self.include_roundness:
            result = sigma_xy + torch.abs(sigma_x - sigma_y)
        else:
            result = sigma_xy
        return self.objective_scale * result + self.objective_offset

    def forward(self, x) -> Tensor:
        idx_sigma_x = self.model.output_order.index("OTRS:IN20:571:XRMS")
        idx_sigma_y = self.model.output_order.index("OTRS:IN20:571:YRMS")
        sigma_x = self.model(x)[..., idx_sigma_x]
        sigma_y = self.model(x)[..., idx_sigma_y]
        return self.function(sigma_x, sigma_y)


def load_model(
        input_variables: list[str],
        model_path: Union[str, os.PathLike] = "lcls_cu_injector_nn_model/",
        calibration_path: Union[str, os.PathLike] = "calibration/",
        reg: str = None,
):
    """Loads the specified version of the LCLS injector surrogate model from file.

    Args:
        input_variables: Input variables for the model.
        model_path: Directory of the models files.
        calibration_path: Directory of the calibration files.
        reg: Amount of regularization used in training the calibration layers. Select from [None, "low", "mid",
          "high"] to load the corresponding transformers.

    Returns:
        The LCLS injector surrogate model.
    """
    # load TorchModel
    lume_model = TorchModel(model_path + "model/pv_model.yml")
    # replace keys in input variables
    for var in lume_model.input_variables:
        var.name = var.name.replace("BACT", "BCTRL")
    # load calibration transformers
    if reg is not None:
        input_nn_to_cal = torch.load(calibration_path + f"input_nn_to_cal_{reg}_reg.pt")
        output_nn_to_cal = torch.load(calibration_path + f"output_nn_to_cal_{reg}_reg.pt")
        lume_model.input_transformers = lume_model.input_transformers + [input_nn_to_cal]
        lume_model.output_transformers = [output_nn_to_cal] + lume_model.output_transformers
    # wrap in TorchModule
    lume_module = TorchModule(
        model=lume_model,
        input_order=input_variables,
        output_order=lume_model.output_names[0:2],
    )
    return ObjectiveModel(lume_module)


def get_performance_stats(data: Tensor, confidence_level: float = 0.9) -> tuple[Tensor, Tensor, Tensor]:
    """Computes the median and confidence level across several BO runs.

    Args:
        data: Performance data across several BO runs.
        confidence_level: Confidence level used to calculate the lower and upper bound.

    Returns:
        Median, lower bound and upper bound corresponding to the confidence level.
    """
    if not 0.0 <= confidence_level <= 1.0:
        raise ValueError("Confidence level must be between 0 and 1.")
    q = 0.9 + (1 - confidence_level) / 2
    m = torch.nanmedian(data, dim=0).values
    lb = torch.nanquantile(data, q=q, dim=0)
    ub = torch.nanquantile(data, q=1 - q, dim=0)
    return m, lb, ub


def update_variables(
    variable_ranges: dict,
    input_variables: dict,
    inputs_small: Tensor,
    from_machine_state: bool = False,
) -> dict[str, list]:
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


def get_model_predictions(input_dict, generator: BayesianGenerator = None) -> dict[str, Any]:
    """Computes the prior and posterior GP model predictions.

    Args:
        input_dict: Inputs for which to compute the GP model predictions.
        generator: Bayesian generator containing the GP model.

    Returns:
        GP model predictions for the prior mean, posterior mean and posterior standard deviation.
    """
    output_dict = {}
    if generator is not None:
        for output_name in generator.vocs.output_names:
            if generator.model is not None:
                gp = generator.model.models[generator.vocs.output_names.index(output_name)]
                x = torch.tensor(
                    [input_dict[k] for k in generator.vocs.variable_names], dtype=torch.double
                ).unsqueeze(0)
                with torch.no_grad():
                    _x = gp.input_transform.transform(x)
                    _x = gp.mean_module(_x)
                    prior_mean = gp.outcome_transform.untransform(_x)[0].item()
                    posterior = gp.posterior(x)
                    posterior_mean = posterior.mean.item()
                    posterior_sd = torch.sqrt(posterior.mvn.variance).item()
            else:
                prior_mean, posterior_mean, posterior_sd = [np.nan] * 3
            output_dict[output_name + "_prior_mean"] = prior_mean
            output_dict[output_name + "_posterior_mean"] = posterior_mean
            output_dict[output_name + "_posterior_sd"] = posterior_sd
    return output_dict


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


class FixedEvalModel(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super(FixedEvalModel, self).__init__()
        self.model = model

    def forward(self, x) -> Tensor:
        self.model.eval()
        return self.model(x)


def load_xopt_data(file: Union[str, os.PathLike]) -> pd.DataFrame:
    """Loads data from a Xopt YAML-file.

    Args:
        file: Path to the YAML-file.

    Returns:
        Loaded data with sorted index.
    """
    with open(file) as f:
        df = pd.DataFrame(yaml.safe_load(f)["data"])
    df.index = map(int, df.index)
    df = df.sort_index(axis=0)
    return df


def get_running_optimum(data: pd.DataFrame, objective_name: str, maximize: bool) -> np.ndarray:
    """Returns the running optimum for the given data and objective.

    Parameters
    ----------
    data: pd.DataFrame
        Data for which the running optimum shall be calculated.
    objective_name: str
        Name of the data column containing the objective values.
    maximize: bool
        If True, consider the problem a maximization problem (minimization otherwise).

    Returns
    -------
    np.ndarray
        Running optimum for the given data.

    """
    get_opt = np.max if maximize else np.min
    return np.array([get_opt(data[objective_name].iloc[:i + 1]) for i in range(len(data))])


def calculate_mae(a: Tensor, b: Tensor) -> Tensor:
    """Calculates the mean absolute error (MAE) for the given tensors.

    Args:
        a: First tensor.
        b: Second tensor.

    Returns:
        The mean absolute error (MAE).
    """
    return torch.nn.functional.l1_loss(a, b)


def calculate_correlation(a: Tensor, b: Tensor) -> Tensor:
    """Calculates the correlation for the given tensors.

    Args:
        a: First tensor.
        b: Second tensor.

    Returns:
        The correlation.
    """
    corr = torch.corrcoef(torch.stack([a.squeeze(), b.squeeze()]))
    return corr[0, 1]


def isotime():
    """Returns the time formatted according to ISO."""
    t = datetime.datetime.utcnow()
    return t.replace(tzinfo=datetime.timezone.utc).astimezone().replace(microsecond=0).isoformat()


def numpy_save(run_dir: Union[str, os.PathLike], spec: str = "x"):
    path = os.path.join(run_dir, "Data", "")
    if not os.path.exists(path):
        os.makedirs(path)
    pvname_list = [
        "SOLN:IN20:121:BACT",
        "QUAD:IN20:121:BACT",
        "QUAD:IN20:122:BACT",
        "QUAD:IN20:361:BACT",
        "QUAD:IN20:371:BACT",
        "QUAD:IN20:425:BACT",
        "QUAD:IN20:441:BACT",
        "QUAD:IN20:511:BACT",
        "QUAD:IN20:525:BACT",
        "SOLN:IN20:121:BCTRL",
        "QUAD:IN20:121:BCTRL",
        "QUAD:IN20:122:BCTRL",
        "QUAD:IN20:361:BCTRL",
        "QUAD:IN20:371:BCTRL",
        "QUAD:IN20:425:BCTRL",
        "QUAD:IN20:441:BCTRL",
        "QUAD:IN20:511:BCTRL",
        "QUAD:IN20:525:BCTRL",
        "CAMR:IN20:186:IMAGE",
        "CAMR:IN20:186:N_OF_ROW",
        "CAMR:IN20:186:N_OF_COL",
        "CAMR:IN20:186:Y",
        "CAMR:IN20:186:X",
        "CAMR:IN20:186:YRMS",
        "CAMR:IN20:186:XRMS",
        "CAMR:IN20:186:RESOLUTION",
        "IRIS:LR20:130:CONFG_SEL",
        "ACCL:IN20:300:L0A_PDES",
        "ACCL:IN20:300:L0A_ADES",
        "ACCL:IN20:400:L0B_PDES",
        "ACCL:IN20:400:L0B_ADES",
        "ACCL:IN20:300:L0A_S_PV",  # PHASE AVG
        "ACCL:IN20:400:L0B_S_PV",  # PHASE AVG
        "ACCL:IN20:300:L0A_S_AV",  # AMP AVG
        "ACCL:IN20:400:L0B_S_AV",  # AMP AVG
        "GUN:IN20:1:GN1_ADES",
        "GUN:IN20:1:GN1_S_AV",
        "GUN:IN20:1:GN1_S_PV",
        "LASR:IN20:196:PWR1H",
        "LASR:IN20:475:PWR1H",
        "SIOC:SYS0:ML01:CALCOUT008",
        "REFS:IN20:751:EDES",
        "CAMR:IN20:186:ZERNIKE_COEFF",
        "FBCK:BCI0:1:CHRG_S",
        "PMTR:LR20:121:PWR",
        "BPMS:IN20:731:X",  # energy BPM
        "TCAV:IN20:490:TC0_C_1_TCTL",  # if 1, TCAV is on
        "KLYS:LI20:51:BEAMCODE1_TCTL",  # if 1, TCAV is on
        "LASR:LR20:1:UV_LASER_MODE",
        "LASR:LR20:1:IR_LASER_MODE",
        "IOC:BSY0:MP01:LSHUTCTL",
        "WPLT:LR20:220:LHWP_ANGLE",
        "OTRS:IN20:621:PNEUMATIC",
    ]
    img_list_yags2 = [
        "YAGS:IN20:995:IMAGE",
        "YAGS:IN20:995:XRMS",
        "YAGS:IN20:995:YRMS",
        "YAGS:IN20:995:ROI_XNP",
        "YAGS:IN20:995:ROI_YNP",
        "YAGS:IN20:995:X",
        "YAGS:IN20:995:Y",
        "YAGS:IN20:995:RESOLUTION",
        "YAGS:IN20:995:BLEN",
    ]
    img_list_otr2 = [
        "OTRS:IN20:571:IMAGE",
        "OTRS:IN20:571:XRMS",
        "OTRS:IN20:571:YRMS",
        "OTRS:IN20:571:ROI_XNP",
        "OTRS:IN20:571:ROI_YNP",
        "OTRS:IN20:571:X",
        "OTRS:IN20:571:Y",
        "OTRS:IN20:571:RESOLUTION",
    ]
    img_list_otr3 = [
        "OTRS:IN20:621:IMAGE",
        "OTRS:IN20:621:XRMS",
        "OTRS:IN20:621:YRMS",
        "OTRS:IN20:621:ROI_XNP",
        "OTRS:IN20:621:ROI_YNP",
        "OTRS:IN20:621:X",
        "OTRS:IN20:621:Y",
        "OTRS:IN20:621:RESOLUTION",
    ]
    img_list_yag02 = [
        "YAGS:IN20:241:IMAGE",
        "YAGS:IN20:241:XRMS",
        "YAGS:IN20:241:YRMS",
        "YAGS:IN20:241:ROI_XNP",
        "YAGS:IN20:241:ROI_YNP",
        "YAGS:IN20:241:X",
        "YAGS:IN20:241:Y",
        "YAGS:IN20:241:RESOLUTION",
    ]
    img_list_yag03 = [
        "YAGS:IN20:351:IMAGE",
        "YAGS:IN20:351:XRMS",
        "YAGS:IN20:351:YRMS",
        "YAGS:IN20:351:ROI_XNP",
        "YAGS:IN20:351:ROI_YNP",
        "YAGS:IN20:351:X",
        "YAGS:IN20:351:Y",
        "YAGS:IN20:351:RESOLUTION",
    ]
    if spec == "z":
        img_list = img_list_yags2
    elif spec == "x":
        img_list = img_list_otr3
    else:
        raise ValueError(f"Spec {spec} isn't recognized.")
    ts = isotime()
    # read pvs
    values = caget_many(pvname_list)
    imgs = caget_many(img_list)
    # save to file
    f_values = os.path.join(path, f"values_{ts}.npz")
    np.savez(f_values, **dict(zip(pvname_list, values)))
    f_imgs = os.path.join(path, f"imgs_{ts}.npz")
    np.savez(f_imgs, **dict(zip(img_list, imgs)))
