import torch
from typing import Optional, Tuple
from gpytorch.means import ConstantMean

from custom_mean import CustomMean


class DynamicCustomMean(CustomMean):
    def __init__(
            self,
            model: torch.nn.Module,
            step: int,
            **kwargs
    ):
        """Dynamic custom prior mean adjusting with the step number.

        Args:
            model: Representation of the model.
            step: Step number in a sampling sequence.
        """
        super().__init__(model, **kwargs)
        self.model = model
        self.step = step

    def forward(self, x):
        return self.model(x)


class Flatten(DynamicCustomMean, ConstantMean):
    def __init__(
            self,
            model: torch.nn.Module,
            step: int,
            **kwargs
    ):
        """Prior mean composed of a weighted sum with a constant prior.

        The output is a step-dependent, weighted sum of the prior mean
        derived from the given model and a constant prior:
        y = (1 - w) * model(x) + w * constant_mean.
        The weighting parameter w is linearly increased from its minimal
        to its maximal value over a fixed sequence of steps.

        Args:
            model: Inherited from DynamicCustomMean.
            step: Inherited from DynamicCustomMean.

        Keyword Args:
            w_lim (Tuple[float, float]): Minimum and maximum value of
              weighting parameter w. Defaults to (0.0, 1.0).
            step_range (Tuple[int, int]): Step range over which weighting
              parameter w is changed from minimum to maximum value.
              Defaults to (0, 10).
        """
        super().__init__(model, step)
        self.w_lim = kwargs.get("w_lim", (0.0, 1.0))
        self.step_range = kwargs.get("step_range", (0, 10))

    @property
    def w(self):
        step_delta = self.step_range[1] - self.step_range[0]
        m = (self.w_lim[1] - self.w_lim[0]) / step_delta
        if self.step < self.step_range[0]:
            w = self.w_lim[0]
        else:
            w = self.w_lim[0] + m * (self.step - self.step_range[0])
        return torch.clip(torch.tensor(w), min=self.w_lim[0], max=self.w_lim[1])

    def forward(self, x):
        w = self.w
        return (1 - w) * self.model(x) + w * self.constant


class OccasionalConstant(DynamicCustomMean, ConstantMean):
    def __init__(
            self,
            model: torch.nn.Module,
            step: int,
            **kwargs
    ):
        """Prior mean which occasionally reverts to a constant prior.

        Reverts to a constant prior at every n-th step, that is, if
        (step + 1) % n == 0. If defined, there is also a probability of
        reverting to a constant prior at every step.

        Args:
            model: Inherited from DynamicCustomMean.
            step: Inherited from DynamicCustomMean.

        Keyword Args:
            n (Optional[int]): If not None, a constant prior is used at
              every n-th step. Defaults to 2.
            prob (Optional[float]): If not None, determines the probability of
              reverting to a constant prior at every step. Defaults to None.

        Attributes:
            use_constant (bool): Whether a constant prior is used.
        """
        super().__init__(model, step)
        self.n = kwargs.get("n", 2)
        self.prob = kwargs.get("prob")
        self.use_constant = False
        r = torch.rand(1)
        if self.n is not None:
            if not (self.step + 1) % self.n == 0:
                self.use_constant = True
        elif self.prob is not None and not self.use_constant:
            if r < self.prob:
                self.use_constant = True

    def _forward_constant(self, x):
        constant = self.constant.unsqueeze(-1)  # *batch_shape x 1
        return constant.expand(
            torch.broadcast_shapes(constant.shape, x.shape[:-1]))

    def forward(self, x):
        if self.use_constant:
            return self._forward_constant(x)
        else:
            return self.model(x)


class OccasionalModel(OccasionalConstant):
    def __init__(
            self,
            model: torch.nn.Module,
            step: int,
            **kwargs
    ):
        """Prior mean which occasionally reverts to a model-based prior.

        Reverts to a model-based prior at every n-th step, that is, if
        (step + 1) % n == 0. If defined, there is also a probability of
        reverting to a model-based prior at every step.

        Args:
            model: Inherited from DynamicCustomMean.
            step: Inherited from DynamicCustomMean.

        Keyword Args:
            n (Optional[int]): If not None, a model-based prior is used at
              every n-th step. Defaults to 2.
            prob (Optional[float]): If not None, determines the probability of
              reverting to a model-based prior at every step. Defaults to None.

        Attributes:
            use_constant (bool): Whether a constant prior is used.
        """
        super().__init__(model, step, **kwargs)
        self.use_constant = not self.use_constant
