import torch
from gpytorch.means.mean import Mean
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
