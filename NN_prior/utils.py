import json
import torch
from abc import abstractmethod
from typing import Callable, Optional, Union
from lume_model.utils import variables_from_yaml
from lume_model.torch import LUMEModule, PyTorchModel
from botorch.models.transforms import Normalize
from botorch.models.transforms.input import AffineInputTransform

from xopt.vocs import VOCS


def load_surrogate(variable_file: str, normalizations_file: str,
                   model_file: str, **kwargs) -> PyTorchModel:
    """Loads the LCLS injector NN surrogate model from file.

    Args:
        variable_file: Path to a yaml-file defining the in- and output
          variables of the surrogate model.
        normalizations_file: Path to a json-file defining the normalization
          transformers for the surrogate model.
        model_file: Path to the torch-file defining the NN model.

    Keyword Args:
        device (Union[torch.device, str]): Device on which the model will be
          evaluated. Defaults to "cpu".

    Returns:
        The LCLS injector NN surrogate model as a PyTorchModel.
    """
    device = kwargs.get("device", "cpu")
    with open(variable_file) as f:
        input_variables, output_variables = variables_from_yaml(f)

    with open(normalizations_file, 'r') as f:
        norm_data = json.load(f)

    transformers = []
    for ele in ["x", "y"]:
        scale = torch.tensor(norm_data[f"{ele}_scale"], dtype=torch.double)
        min_val = torch.tensor(norm_data[f"{ele}_min"], dtype=torch.double)
        transform = AffineInputTransform(
            len(norm_data[f"{ele}_min"]),
            1 / scale,
            -min_val / scale,
        )
        transformers.append(transform)

    surrogate = PyTorchModel(
        model_file,
        input_variables, output_variables,
        input_transformers=[transformers[0]],
        output_transformers=[transformers[1]],
        device=device,
    )
    return surrogate


def load_corr_model(variable_file: str, x_transformer_file: str,
                    y_transformer_file: str, model_file: str,
                    **kwargs) -> PyTorchModel:
    """Loads a model correlated to the LCLS injector NN surrogate from file.

    Args:
        variable_file: Path to a yaml-file defining the in- and output
          variables of the correlated model (same a surrogate).
        x_transformer_file: Path to a torch-file defining the input
          transformer for the correlated model.
        y_transformer_file: Path to a torch-file defining the output
          transformer for the correlated model.
        model_file: Path to the torch-file defining the NN model.

    Keyword Args:
        device (Union[torch.device, str]): Device on which the model will be
          evaluated. Defaults to "cpu".

    Returns:
        The correlated model as a PyTorchModel.
    """
    device = kwargs.get("device", "cpu")
    with open(variable_file) as f:
        input_variables, output_variables = variables_from_yaml(f)

    x_transformer = Normalize(len(input_variables))
    x_transformer.eval()
    x_transformer.load_state_dict(torch.load(x_transformer_file))
    y_transformer = Normalize(len(output_variables))
    y_transformer.eval()
    y_transformer.load_state_dict(torch.load(y_transformer_file))

    corr_model = PyTorchModel(
        model_file,
        input_variables, output_variables,
        input_transformers=[x_transformer],
        output_transformers=[y_transformer],
        device=device,
    )
    return corr_model


class Objective(torch.nn.Module):
    def __init__(
            self,
            model: Union[torch.nn.Module, LUMEModule],
            **kwargs
    ):
        """Calculates an objective based on the output of the given model.

        Args:
            model: The model.

        Keyword Args:
            unit (str): Denotes the unit of the objective.
              Defaults to "arb. unit".

        Attributes:
            name (str): Name of the objective.
        """
        super(Objective, self).__init__()
        self.model = model
        self.name = "Negative Transverse Beam Size"
        self.unit = kwargs.get("unit", "arb. unit")

    @staticmethod
    @abstractmethod
    def function(*args, **kwargs) -> Union[torch.Tensor, float]:
        pass


class NegativeTransverseBeamSize(Objective):
    def __init__(
            self,
            model: Union[torch.nn.Module, LUMEModule],
            **kwargs
    ):
        """
        Calculates the transverse beam size based on the output of the
        given model.

        Args:
            model: The model.

        Keyword Args:
            unit (str): Inherited from Objective. Defaults to "mm".

        Attributes:
            name (str): Inherited from Objective.
        """
        super(NegativeTransverseBeamSize, self).__init__(model)
        self.name = "Negative Transverse Beam Size"
        self.unit = kwargs.get("unit", "mm")

    @staticmethod
    def function(sigma_x: Union[torch.Tensor, float],
                 sigma_y: Union[torch.Tensor, float]) \
            -> Union[torch.Tensor, float]:
        # using this calculation for the transverse beam size due to
        # occasional negative values
        return -torch.sqrt(sigma_x ** 2 + sigma_y ** 2) * 1e3

    def forward(self, x):
        if isinstance(self.model, LUMEModule):
            idx_sigma_x = self.model.output_order.index("sigma_x")
            idx_sigma_y = self.model.output_order.index("sigma_y")
            sigma_x = self.model(x)[..., idx_sigma_x]
            sigma_y = self.model(x)[..., idx_sigma_y]
        else:
            sigma_x, sigma_y = self.model(x)[..., 0], self.model(x)[..., 1]
        return self.function(sigma_x, sigma_y)


class LinearMismatch(torch.nn.Module):
    def __init__(
            self,
            function: Callable[[torch.Tensor], torch.Tensor],
            **kwargs
    ):
        """Adds linear mismatches to the given function.

        Args:
            function: A function.

        Keyword Args:
            x_dim (int): The input dimension. Defaults to 1.
            x_shift (torch.Tensor): A tensor of shape (x_dim).
              Defaults to zeros.
            x_scale (torch.Tensor): A tensor of shape (x_dim). Defaults to ones.
            y_shift (torch.Tensor): A tensor of shape (1). Defaults to zero.
            y_scale (torch.Tensor): A tensor of shape (1). Defaults to one.
        """
        super(LinearMismatch, self).__init__()
        self.function = function
        assert callable(self.function), \
            "Expected function to be callable"

        # parameters for mismatch in x
        x_dim = kwargs.get("x_dim", 1)
        x_shape = torch.Size([x_dim])
        self.x_shift = kwargs.get("x_shift", torch.zeros(x_shape))
        assert self.x_shift.shape == x_shape, \
            f"Expected tensor with {x_shape}, but got: {self.x_shift.shape}"
        self.x_scale = kwargs.get("x_scale", torch.ones(x_shape))
        assert self.x_scale.shape == x_shape, \
            f"Expected tensor with {x_shape}, but got: {self.x_scale.shape}"

        # parameters for mismatch in y
        y_shape = torch.Size([1])
        self.y_shift = kwargs.get("y_shift", torch.zeros(y_shape))
        assert self.y_shift.shape == y_shape, \
            f"Expected tensor with {y_shape}, but got: {self.y_shift.shape}"
        self.y_scale = kwargs.get("y_scale", torch.ones(y_shape))
        assert self.y_scale.shape == y_shape, \
            f"Expected tensor with {y_shape}, but got: {self.y_scale.shape}"

    def forward(self, x):
        """Expects tensor of shape (n_batch, n_samples, n_dim)"""
        mismatched_x = self.x_scale * x + self.x_shift
        y = self.y_scale * self.function(mismatched_x) + self.y_shift
        return y


def running_max(y: torch.Tensor) -> torch.Tensor:
    """Returns the running maximum for the given data.

    The respective maxima are formed over the last dimension.

    Args:
        y: Data for which the running maximum/maxima shall be calculated.

    Returns:
        The running maximum for the given data.
    """
    y_opt = torch.stack(
        [torch.max(y[..., :i + 1], dim=-1)[0] for i in range(y.shape[-1])])
    p = 1 + torch.arange(len(y_opt.shape))
    p[-1] = 0
    return torch.permute(y_opt, p.tolist())


def calc_corr(a: torch.Tensor, b: torch.Tensor,
              cutoff_value: Optional[float] = None) -> torch.Tensor:
    """Calculates the correlation between the given tensors.

    Args:
        a: First tensor.
        b: Second tensor.
        cutoff_value: If not None, data samples are only used if the value in
          the first tensor is above the given value.
    """
    if cutoff_value is not None:
        a_c = a[torch.where(a > cutoff_value)[0]]
        b_c = b[torch.where(a > cutoff_value)[0]]
    else:
        a_c, b_c = a, b
    corr = torch.corrcoef(torch.stack([a_c.squeeze(), b_c.squeeze()]))
    return corr[0, 1]


def calc_mae(a: torch.Tensor, b: torch.Tensor,
             cutoff_value: Optional[float] = None) -> torch.Tensor:
    """Calculates the mean absolute error between the given tensors.

    Args:
        a: First tensor.
        b: Second tensor.
        cutoff_value: If not None, data samples are only used if the value in
          the first tensor is above the given value.
    """
    if cutoff_value is not None:
        a_c = a[torch.where(a > cutoff_value)[0]]
        b_c = b[torch.where(a > cutoff_value)[0]]
    else:
        a_c, b_c = a, b
    return torch.nn.functional.l1_loss(a_c, b_c)


def create_vocs(surrogate: PyTorchModel,
                objective_name: str = "negative_sigma_xy") -> VOCS:
    """Creates VOCS object for xopt.

    Args:
        surrogate: The surrogate model.
        objective_name: The name of the objective (only "negative_sigma_xy"
          is supported for now).

    Returns:
        The VOCS object.
    """
    if not objective_name == "negative_sigma_xy":
        raise ValueError(f"objective_name {objective_name} is not supported")
    xopt_variables = {}
    constant_variables = {}
    for input_name, variable in surrogate.input_variables.items():
        if variable.value_range[0] == variable.value_range[1]:
            constant_variables[input_name] = variable.value_range[0]
        else:
            xopt_variables[input_name] = variable.value_range
    vocs = VOCS(
        variables=xopt_variables,
        objectives={objective_name: "MAXIMIZE"},
        constants=constant_variables
    )
    return vocs


def print_runtime(t0: float, t: float):
    """Prints runtime in human-readable format.

    Args:
        t0: Starting time in seconds.
        t: Stopping time in seconds.
    """
    t_hour = (t - t0) / 3600
    if t_hour <= 1.0:
        t_min = 60 * t_hour
        if t_min <= 1.0:
            t_print = "{:.2f} sec".format(60 * t_min)
        else:
            t_print = "{:.2f} min".format(t_min)
    else:
        t_print = "{:.2f} hours".format(t_hour)
    print("Runtime: {}".format(t_print))
