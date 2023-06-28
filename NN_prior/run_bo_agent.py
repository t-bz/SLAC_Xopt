import os
import sys

import torch
from gpytorch.means import ConstantMean

from lume_model.torch import LUMEModule

from bo_agent import BOAgent
from utils import NegativeTransverseBeamSize
from utils import load_surrogate, load_corr_model, create_vocs
from custom_mean import CustomMean, TrainableFlatten
from dynamic_custom_mean import Flatten, OccasionalConstant, OccasionalModel
from metric_informed_custom_mean import CorrelationThreshold, CorrelatedFlatten


# define prior mean
mean_class = CustomMean
mean_kwargs = {}

# select correlated model
n_epoch = int(sys.argv[1])

# check for GPUs
use_cuda, device = False, "cpu"
if torch.cuda.is_available():
    use_cuda, device = True, "cuda"

# output directory
path = "./BO/"
output_dir = path + "{}/".format(mean_class.__name__)
for i, (k, v) in enumerate(mean_kwargs.items()):
    if k not in ["step", "metrics"]:
        output_dir += f"{k}="
        if v is None:
            output_dir += "None"
        elif isinstance(v, tuple):
            v_str = []
            for v_i in v:
                if isinstance(v_i, float):
                    v_str.append("{:.2f}".format(v_i))
                else:
                    v_str.append("{}".format(v_i))
            output_dir += "{}-{}".format(*v_str)
        else:
            output_dir += "{:.2f}".format(v)
        if not i == len(mean_kwargs) - 1:
            output_dir += "_"
if not output_dir.endswith("/"):
    output_dir += "/"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# load surrogate model and define objective
surrogate = load_surrogate(
    "configs/lcls_variables.yml",
    "configs/normalization.json",
    "torch_model.pt",
    device=device,
)
objective_name = "negative_sigma_xy"
vocs = create_vocs(surrogate, objective_name)
surrogate_module = LUMEModule(surrogate, vocs.variable_names, ["sigma_x", "sigma_y"])
Objective = NegativeTransverseBeamSize
ground_truth = Objective(surrogate_module)


# Xopt evaluator function
def evaluate(input_dict):
    model_result = surrogate.evaluate(input_dict)
    obj_kwargs = {k: model_result[k] for k in surrogate_module.output_order}
    obj_value = NegativeTransverseBeamSize.function(
        **obj_kwargs).detach().item()
    return {objective_name: obj_value}


# load correlated model
if issubclass(mean_class, CustomMean):
    corr_model = load_corr_model(
        "configs/lcls_variables.yml",
        "corr_models/x_transformer.pt",
        "corr_models/y_transformer.pt",
        "corr_models/{:d}ep.pt".format(n_epoch),
        device=device,
    )
    corr_module = LUMEModule(corr_model, vocs.variable_names,
                             ["sigma_x", "sigma_y"])
    mean_kwargs["model"] = Objective(corr_module)

# create BOAgent
prior_mean = mean_class(**mean_kwargs).to(device)
bo_config = {"n_run": 2, "path": path, "use_cuda": use_cuda, "n_step": 3}
bo_agent = BOAgent(prior_mean, vocs, bo_config)
bo_agent.run(evaluate)

# save BOAgent configuration and data history
if issubclass(mean_class, CustomMean):
    file_name = "{:d}ep.pt".format(n_epoch)
else:
    file_name = "constant.pt"
bo_agent.save(output_dir + file_name)
