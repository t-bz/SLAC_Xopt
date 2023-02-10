import torch
from gpytorch.means.mean import Mean
from gpytorch.priors import NormalPrior
from transformed_model import TransformedModel


class CustomMean(TransformedModel, Mean):
    def __init__(
            self,
            model: torch.nn.Module,
            gp_input_transform: torch.nn.Module,
            gp_outcome_transform: torch.nn.Module,
    ):
        """Custom prior mean for a GP based on an arbitrary model.

        Args:
            model: Representation of the model.
            gp_input_transform: Module used to transform inputs in the GP.
            gp_outcome_transform: Module used to transform outcomes in the GP.
        """
        super(CustomMean, self).__init__(model, gp_input_transform,
                                         gp_outcome_transform)

    def evaluate_model(self, x):
        """Placeholder method which can be used to modify model calls."""
        return self.model(x)

    def forward(self, x):
        # set transformers to eval mode
        self.input_transformer.eval()
        self.outcome_transformer.eval()
        # transform inputs to mean space
        x_model = self.input_transformer.untransform(x)
        # evaluate model
        y_model = self.evaluate_model(x_model)
        # transform outputs
        y = self.outcome_transformer(y_model)[0]
        self.input_transformer.eval()
        self.outcome_transformer.eval()
        return y


class LinearInputNodes(CustomMean):
    def __init__(
            self,
            model: torch.nn.Module,
            gp_input_transform: torch.nn.Module,
            gp_outcome_transform: torch.nn.Module,
            **kwargs,
    ):
        """Prior mean with learnable linear input transformation.

        Inputs are passed through decoupled linear transformation nodes
        with learnable shift and scaling parameters:
        y = model(x_scale * (x + x_shift)).

        Args:
            model: Inherited from CustomMean.
            gp_input_transform: Inherited from CustomMean.
            gp_outcome_transform: Inherited from CustomMean.

        Keyword Args:
            x_dim (int): The input dimension. Defaults to 1.
            x_shift_prior (gpytorch.priors.Prior): Prior over x_shift.
              Defaults to a Normal distribution.
            x_scale_prior (gpytorch.priors.Prior): Prior over x_scale.
              Defaults to a Normal distribution centered at 1.

        Attributes:
            x_shift (torch.nn.Parameter): Parameter tensor of size x_dim.
            x_scale (torch.nn.Parameter): Parameter tensor of size x_dim.
        """
        super().__init__(model, gp_input_transform, gp_outcome_transform)
        self.x_dim = kwargs.get("x_dim", 1)
        self.x_shift = torch.nn.Parameter(torch.randn(self.x_dim))
        x_shift_prior = kwargs.get(
            "x_shift_prior",
            NormalPrior(loc=torch.zeros((1, self.x_dim)),
                        scale=torch.ones((1, self.x_dim)))
        )
        self.register_prior("x_shift_prior", x_shift_prior, "x_shift")
        self.x_scale = torch.nn.Parameter(torch.randn(self.x_dim))
        x_scale_prior = kwargs.get(
            "x_scale_prior",
            NormalPrior(loc=torch.ones((1, self.x_dim)),
                        scale=torch.ones((1, self.x_dim)))
        )
        self.register_prior("x_shift_prior", x_scale_prior, "x_shift")

    def input_nodes(self, x):
        return self.x_scale * x + self.x_shift

    def evaluate_model(self, x):
        return self.model(self.input_nodes(x))
