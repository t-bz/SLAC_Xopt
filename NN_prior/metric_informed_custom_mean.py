import torch

from custom_mean import CustomMean
from gpytorch.means import ConstantMean


class MetricInformedCustomMean(CustomMean):
    def __init__(self, model: torch.nn.Module, metrics: dict, **kwargs):
        """Adaptive custom prior mean adjusting based on given metrics.

        Args:
            model: Representation of the model.
            metrics: A dictionary of metrics assessing the model quality.
        """
        super().__init__(model, **kwargs)
        self.model = model
        self.metrics = metrics

    def forward(self, x):
        return self.model(x)


class CorrelationThreshold(MetricInformedCustomMean, ConstantMean):
    def __init__(self, model: torch.nn.Module, metrics: dict, **kwargs):
        """Prior mean reverting to a constant if correlation is below threshold.

        Reverts to a constant prior mean unless the model correlation found in the metrics is above a
        given threshold.

        Args:
            model: Inherited from MetricInformedCustomMean.
            metrics: Inherited from MetricInformedCustomMean.

        Keyword Args:
            threshold (float): Correlation threshold above which the model is used as a prior. Defaults to 0.8.

        Attributes:
            correlation (float): Correlation value extracted from metrics.
            use_constant (bool): Whether a constant prior is used.
        """
        super().__init__(model, metrics, **kwargs)
        self.threshold = kwargs.get("threshold", 0.8)
        self.correlation = metrics.get("correlation", 0.0)
        self.use_constant = self.correlation < self.threshold

    def _forward_constant(self, x):
        constant = self.constant.unsqueeze(-1)  # *batch_shape x 1
        return constant.expand(torch.broadcast_shapes(constant.shape, x.shape[:-1]))

    def forward(self, x):
        if self.use_constant:
            return self._forward_constant(x)
        else:
            return self.model(x)


class CorrelatedFlatten(MetricInformedCustomMean, ConstantMean):
    def __init__(self, model: torch.nn.Module, metrics: dict, **kwargs):
        """Prior mean composed of a weighted sum with a constant prior.

        The output is a weighted sum of the prior mean derived from the given model and a constant prior:
        y = w * model(x) + (1 - w) * constant_mean, with the weighting parameter w being determined by the given
        correlation and offset: w = correlation - offset.

        Args:
            model: Inherited from MetricInformedCustomMean.
            metrics: Inherited from MetricInformedCustomMean.

        Keyword Args:
            w_lim (Tuple[float, float]): Minimum and maximum value of weighting parameter w.
              Defaults to (0.0, 1.0).
            w_offset (float): Offset for w parameter. Defaults to 0.0.

        Attributes:
            correlation (float): Correlation value extracted from metrics.
            w (float): Weighting parameter.
        """
        super().__init__(model, metrics, **kwargs)
        self.w_lim = kwargs.get("w_lim", (0.0, 1.0))
        self.w_offset = kwargs.get("w_offset", 0.0)
        self.correlation = metrics.get("correlation", 1.0)

    @property
    def w(self):
        w = self.correlation - self.w_offset
        if not isinstance(w, torch.Tensor):
            w = torch.tensor(w)
        return torch.clip(w, min=self.w_lim[0], max=self.w_lim[1])

    def forward(self, x):
        w = self.w
        return w * self.model(x) + (1 - w) * self.constant
