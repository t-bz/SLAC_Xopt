import os
import json
import torch
import numpy as np

from typing import Callable, Optional
from botorch.models.transforms import Normalize
from transformers import create_sim_to_nn_transformers
from transformed_model import KeyedTransformedModel


class Surrogate:
    def __init__(self, **kwargs):
        """Loads the surrogate model and complementary information.

        Keyword Args:
            path (str): The file path to the root directory. Defaults to
              the directory this file is in.

        Attributes:
            model_info (dict): Holds information about the surrogate model.
            pv_info (dict): Holds information about the transformations of
              in- and outputs of the surrogate model to pv values.
            transform_info (dict): Holds information about the normalization
              of the model in- and outputs.
            untransformed_model (torch.nn.Module): The NN model without
              additional transformations.
            input_transformer (torch.nn.Module): The input transformer
              (normalization) for the surrogate model.
            outcome_transformer (torch.nn.Module): The outcome transformer
              (normalization) for the surrogate model.
            x_dim (int): The input dimension.
            y_dim (int): The output dimension.
            raw_x_lim (torch.Tensor): Defines the input domain.
            fixed_feature_x_lim (torch.Tensor): Defines the input domain
              excluding dimensions with fixed value inputs.
            fixed_feature_columns (list): Index list of the fixed value
              input dimensions.
            fixed_feature_values (list): Value list of the fixed value
              input dimensions.
            x_lim (torch.Tensor): Input domain with fixed value dimensions
              adjusted to the range of +-1 percent.
            model (torch.nn.Module): The transformed model.
        """
        self.path = kwargs.get("path")
        if self.path is None:
            self.path = os.path.dirname(__file__)
        # load data
        f_model_info = os.path.join(self.path, "configs/model_info.json")
        self.model_info = json.load(open(f_model_info))
        f_pv_info = os.path.join(self.path, "configs/pv_info.json")
        self.pv_info = json.load(open(f_pv_info))
        f_transform_info = os.path.join(self.path, "configs/normalization.json")
        self.transform_info = json.load(open(f_transform_info))
        f_model = os.path.join(self.path, "torch_model.pt")
        self.untransformed_model = torch.load(f_model).double()
        transformers = create_sim_to_nn_transformers(f_transform_info)
        self.input_transformer, self.outcome_transformer = transformers
        # dimensions and input domain
        self.x_dim = len(self.model_info["model_in_list"])
        self.y_dim = len(self.model_info["model_out_list"])
        train_input_mins = self.model_info["train_input_mins"]
        train_input_maxs = self.model_info["train_input_maxs"]
        self.raw_x_lim = torch.tensor([train_input_mins,
                                       train_input_maxs]).double().T
        # handle fixed value dimensions
        x_lim = self.raw_x_lim.clone()
        self.fixed_feature_x_lim = x_lim[x_lim[:, 0] < x_lim[:, 1]]
        self.fixed_feature_columns, self.fixed_feature_values = [], []
        for i in range(x_lim.shape[0]):
            if not x_lim[i, 0] < x_lim[i, 1]:
                old_value = x_lim[i, 0].clone()
                delta = 1e-2 * x_lim[i, 0]
                x_lim[i] = torch.tensor([old_value - delta, old_value + delta])
                self.fixed_feature_columns.append(i)
                self.fixed_feature_values.append(old_value)
        self.x_lim = x_lim
        # transformed model
        self.model = KeyedTransformedModel(
            self.untransformed_model,
            self.input_transformer,
            self.outcome_transformer,
            self.model_info["model_in_list"],
            self.model_info["model_out_list"]
        )

    @property
    def raw_x_data(self) -> torch.Tensor:
        """Example x data used during training."""
        f = os.path.join(self.path, "data/x_raw_small.npy")
        raw_x_data = np.load(f, allow_pickle=True)
        return torch.from_numpy(raw_x_data)

    @property
    def raw_y_data(self) -> torch.Tensor:
        """Example y data used during training."""
        f = os.path.join(self.path, "data/y_raw_small.npy")
        raw_y_data = np.load(f, allow_pickle=True)
        return torch.from_numpy(raw_y_data.astype(np.float64))

    def sample_x(self, n: int, fixed_feature: bool = True,
                 seed: Optional[int] = None) -> torch.Tensor:
        """Draws uniformly distributed samples from the input domain.

        Args:
            n: The number of samples to generate.
            fixed_feature: Whether drawn samples have constant values for the
              dimensions with fixed inputs values. If False, the data is
              sampled in the range +-1 percent.
            seed: The initial seed for the random number generator.
        """
        gen = torch.Generator()
        if isinstance(seed, int):
            gen.manual_seed(seed)
        if fixed_feature:
            x_lim = self.raw_x_lim
        else:
            x_lim = self.x_lim
        r = torch.rand(n, x_lim.shape[0], generator=gen)
        return (x_lim[:, 1] - x_lim[:, 0]) * r + x_lim[:, 0]


class NegativeTransverseBeamSize(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, **kwargs):
        """
        Calculates the transverse beam size based on the output of the
        given model.

        Args:
            model: The model.

        Keyword Args:
            unit (str): Denotes the unit of the output. Defaults to (mm).
            unit_factor (float): Denotes the required factor to obtain
              the given unit.
        """
        super(NegativeTransverseBeamSize, self).__init__()
        self.model = model
        self.name = "Negative Transverse Beam Size"
        self.unit = kwargs.get("unit", "(mm)")
        self.unit_factor = kwargs.get("unit_factor", 1e3)

    def forward(self, x):
        sigma_x, sigma_y = self.model(x)[..., 0], self.model(x)[..., 1]
        return -torch.sqrt(sigma_x ** 2 + sigma_y ** 2)


class MismatchedGroundTruth(torch.nn.Module):
    def __init__(self, ground_truth: Callable, **kwargs):
        """
        Prediction of the ground truth function with optional linear
        mismatches in x and y.

        Args:
            ground_truth: Ground truth function.

        Keyword Args:
            x_dim (int): The input dimension. Defaults to 1.
            x_shift (torch.Tensor): A tensor of shape (x_dim).
              Defaults to zeros.
            x_scale (torch.Tensor): A tensor of shape (x_dim). Defaults to ones.
            y_shift (torch.Tensor): A tensor of shape (1). Defaults to zero.
            y_scale (torch.Tensor): A tensor of shape (1). Defaults to one.
        """
        super(MismatchedGroundTruth, self).__init__()
        self.ground_truth = ground_truth
        assert callable(self.ground_truth), \
            f"Expected ground_truth to be callable"

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
        y = self.y_scale * self.ground_truth(mismatched_x) + self.y_shift
        return y


def running_max(y: torch.Tensor) -> torch.Tensor:
    """
    Returns the running maximum for the given data. The respective maxima are
    formed over the last dimension.

    Args:
        y: Data for which the running maximum/maxima shall be calculated.
    """
    y_opt = torch.stack(
        [torch.max(y[..., :i + 1], dim=-1)[0] for i in range(y.shape[-1])])
    p = 1 + torch.arange(len(y_opt.shape))
    p[-1] = 0
    return torch.permute(y_opt, p.tolist())


def load_correlated_model(n_epoch: int, surrogate: Surrogate,
                          **kwargs) -> (KeyedTransformedModel, float):
    """Loads a model and its correlation to the surrogate model.

    Args:
        n_epoch: The number of epochs the correlated models has been trained
          for.
        surrogate: The surrogate model.

    Keyword Args:
        path (str): The file path to the root directory. Defaults to the
          subdirectory "corr_models/" of  the directory this file is in.

    Returns:
        The model and its correlation as a tuple.
    """
    default_path = os.path.join(os.path.dirname(__file__), "corr_models/")
    path = kwargs.get("path", default_path)
    # load in-/ouput transformers (normalization)
    x_transformer = Normalize(surrogate.x_dim)
    x_transformer.eval()
    x_transformer.load_state_dict(torch.load(path + "x_transformer.pt"))
    y_transformer = Normalize(surrogate.y_dim)
    y_transformer.eval()
    y_transformer.load_state_dict(torch.load(path + "y_transformer.pt"))
    # load model
    untransformed_model = torch.load(path + "{:d}ep.pt".format(n_epoch))
    model = KeyedTransformedModel(
        untransformed_model,
        x_transformer,
        y_transformer,
        surrogate.model_info["model_in_list"],
        surrogate.model_info["model_out_list"]
    )
    # load correlations
    correlations = torch.load(path + "correlations.pt")
    return model, correlations[n_epoch - 1]


def calc_correlation(y_gt: torch.Tensor, y_corr: torch.Tensor,
                     cutoff_value: Optional[float] = None) -> torch.Tensor:
    """Calculates the correlation between the given tensors.

    Args:
        y_gt: The ground truth values.
        y_corr: The correlated values.
        cutoff_value: If not None, data samples are only used if y_gt is above
          the given value.
    """
    if cutoff_value is not None:
        y_0 = y_gt[torch.where(y_gt > cutoff_value)[0]]
        y_1 = y_corr[torch.where(y_gt > cutoff_value)[0]]
    else:
        y_0, y_1 = y_gt, y_corr
    corr = torch.corrcoef(torch.stack([y_1.squeeze(), y_0.squeeze()]))
    return corr[0, 1]


def calc_mean_absolute_error(y_gt: torch.Tensor, y_corr: torch.Tensor,
                             cutoff_value: Optional[
                                 float] = None) -> torch.Tensor:
    """Calculates the mean absolute error between the given tensors.

    Args:
        y_gt: The ground truth values.
        y_corr: The correlated values.
        cutoff_value: If not None, data samples are only used if y_gt is above
          the given value.
    """
    if cutoff_value is not None:
        y_0 = y_gt[torch.where(y_gt > cutoff_value)[0]]
        y_1 = y_corr[torch.where(y_gt > cutoff_value)[0]]
    else:
        y_0, y_1 = y_gt, y_corr
    return torch.nn.functional.l1_loss(y_1, y_0)
