import datetime
import os
from typing import Any, Union

import numpy as np
import pandas as pd
import torch
import yaml
from epics import caget_many
from torch import Tensor

from lume_model.models import TorchModel, TorchModule
from xopt.generators.bayesian.bayesian_generator import BayesianGenerator
from xopt.vocs import VOCS


class ObjectiveModel(torch.nn.Module):
    """Defines how to compute the objective based on model predictions"""
    def __init__(
        self,
        model: TorchModule,
        objective_scale: float = None,
        objective_offset: float = None,
        include_roundness: bool = True,
        use_sim_model: bool = False,
    ):
        """Initializes ObjectiveModel.

        Args:
            model: LCLS injector surrogate model.
            objective_scale: Scaling factor for the objective. Defaults to 1e-3 (or 1e3 for the simulation
              model) which yields a transverse beamsize given in mm.
            objective_offset: Adds a fixed offset to the objective.
            include_roundness: Whether to include a term penalizing asymmetric beam shapes.
            use_sim_model: If True, the default objective scaling is set to 1e3 (1e-3 otherwise) and the
              simulation variable names are used to extract sigma_x and sigma_y from the model output.
        """
        super(ObjectiveModel, self).__init__()
        self.model = model
        self.objective_scale = objective_scale
        if self.objective_scale is None:
            self.objective_scale = 1e3 if use_sim_model else 1e-3
        self.objective_offset = objective_offset
        if self.objective_offset is None:
            self.objective_offset = 0.0
        self.include_roundness = include_roundness
        self.use_sim_model = use_sim_model

    def function(self, sigma_x: Tensor, sigma_y: Tensor) -> Tensor:
        """Computes the objective from the given beamsizes in x and y.

        Args:
            sigma_x: Beamsize in x.
            sigma_y: Beamsize in y.

        Returns:
            The objective value.
        """
        # using this calculation due to occasional negative values
        sigma_xy = torch.sqrt(sigma_x ** 2 + sigma_y ** 2)
        v = sigma_xy + torch.abs(sigma_x - sigma_y) if self.include_roundness else sigma_xy
        return self.objective_scale * v + self.objective_offset

    def forward(self, x) -> Tensor:
        sigma_x_label = "sigma_x" if self.use_sim_model else "OTRS:IN20:571:XRMS"
        sigma_y_label = "sigma_y" if self.use_sim_model else "OTRS:IN20:571:YRMS"
        idx_sigma_x = self.model.output_order.index(sigma_x_label)
        idx_sigma_y = self.model.output_order.index(sigma_y_label)
        sigma_x = self.model(x)[..., idx_sigma_x]
        sigma_y = self.model(x)[..., idx_sigma_y]
        return self.function(sigma_x, sigma_y)


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


def load_model(
        input_variables: list[str] = None,
        model_path: Union[str, os.PathLike] = "lcls_cu_injector_nn_model/",
        calibration_path: Union[str, os.PathLike] = "calibration/",
        reg: str = None,
        use_sim_model: bool = False,
):
    """Loads the specified version of the LCLS injector surrogate model from file.

    Args:
        input_variables: Input variables for the model.
        model_path: Directory of the models files.
        calibration_path: Directory of the calibration files.
        reg: Amount of regularization used in training the calibration layers. Select from [None, "low", "mid",
          "high"] to load the corresponding transformers.
        use_sim_model: If True, sim_model is loaded (pv_model otherwise).

    Returns:
        The LCLS injector surrogate model.
    """
    if use_sim_model:
        # load TorchModel
        lume_model = TorchModel(model_path + "model/sim_model.yml")
    else:
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


def generate_input_mesh(vocs: VOCS, n: int = 10) -> pd.DataFrame:
    """Generates an input mesh with n^n_variables points.

    Args:
        vocs: VOCS object defining the variables.
        n: Number of base points.

    Returns:
        Input mesh data.
    """
    x_lim = torch.tensor([vocs.variables[k] for k in vocs.variable_names], dtype=torch.double)
    x_i = [torch.linspace(*x_lim[i], n, dtype=torch.double) for i in range(x_lim.shape[0])]
    x_mesh = torch.meshgrid(*x_i, indexing="ij")
    x = torch.hstack([ele.reshape(-1, 1) for ele in x_mesh])
    return pd.DataFrame({k: x[:, vocs.variable_names.index(k)] for k in vocs.variable_names})


def isotime():
    """Returns the time formatted according to ISO."""
    t = datetime.datetime.utcnow()
    return t.replace(tzinfo=datetime.timezone.utc).astimezone().replace(microsecond=0).isoformat()


def numpy_save(run_dir: Union[str, os.PathLike], device_name: str = "OTR3"):
    """Reads and stores a list of relevant PVs to a numpy file.

    Args:
        run_dir: Directory to which the numpy file is saved.
        device_name: Measurement device used, should be in ["YAGS2", "YAG02", "YAG03", "OTR2", "OTR3", "WIRE561"].
          Defaults to "OTR3".
    """
    path = os.path.join(run_dir, "Data", "")
    if not os.path.exists(path):
        os.makedirs(path)
    pvs = [
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
    device_to_pvs = {
        "YAGS2": [
            "YAGS:IN20:995:IMAGE",
            "YAGS:IN20:995:XRMS",
            "YAGS:IN20:995:YRMS",
            "YAGS:IN20:995:ROI_XNP",
            "YAGS:IN20:995:ROI_YNP",
            "YAGS:IN20:995:X",
            "YAGS:IN20:995:Y",
            "YAGS:IN20:995:RESOLUTION",
            "YAGS:IN20:995:BLEN",
        ],
        "YAG02": [
            "YAGS:IN20:241:IMAGE",
            "YAGS:IN20:241:XRMS",
            "YAGS:IN20:241:YRMS",
            "YAGS:IN20:241:ROI_XNP",
            "YAGS:IN20:241:ROI_YNP",
            "YAGS:IN20:241:X",
            "YAGS:IN20:241:Y",
            "YAGS:IN20:241:RESOLUTION",
        ],
        "YAG03": [
            "YAGS:IN20:351:IMAGE",
            "YAGS:IN20:351:XRMS",
            "YAGS:IN20:351:YRMS",
            "YAGS:IN20:351:ROI_XNP",
            "YAGS:IN20:351:ROI_YNP",
            "YAGS:IN20:351:X",
            "YAGS:IN20:351:Y",
            "YAGS:IN20:351:RESOLUTION",
        ],
        "OTR2": [
            "OTRS:IN20:571:IMAGE",
            "OTRS:IN20:571:XRMS",
            "OTRS:IN20:571:YRMS",
            "OTRS:IN20:571:ROI_XNP",
            "OTRS:IN20:571:ROI_YNP",
            "OTRS:IN20:571:X",
            "OTRS:IN20:571:Y",
            "OTRS:IN20:571:RESOLUTION",
        ],
        "OTR3": [
            "OTRS:IN20:621:IMAGE",
            "OTRS:IN20:621:XRMS",
            "OTRS:IN20:621:YRMS",
            "OTRS:IN20:621:ROI_XNP",
            "OTRS:IN20:621:ROI_YNP",
            "OTRS:IN20:621:X",
            "OTRS:IN20:621:Y",
            "OTRS:IN20:621:RESOLUTION",
        ],
        "WIRE561": [
            "WIRE:IN20:561:XRMS",
            "WIRE:IN20:561:YRMS",
        ]
    }
    name = device_name.upper()
    if name not in device_to_pvs.keys():
        raise ValueError(f"Device name {name} isn't recognized.")
    ts = isotime()
    # read pvs
    img_pvs = device_to_pvs[name]
    values = caget_many(pvs)
    img_values = caget_many(img_pvs)
    # save to file
    f_values = os.path.join(path, f"values_{ts}.npz")
    np.savez(f_values, **dict(zip(pvs, values)))
    f_img = os.path.join(path, f"img_values_{ts}.npz")
    np.savez(f_img, **dict(zip(img_pvs, img_values)))
