from gpytorch.means.mean import Mean
from transformed_model import TransformedModel


class CustomMean(TransformedModel, Mean):
    def __init__(
            self,
            model,
            gp_input_transform,
            gp_outcome_transform,
    ):
        """
        Custom prior mean for a GP based on an arbitrary model

        :param model: torch.nn.Module representation of the model
        :param gp_input_transform: module used to transform inputs in the GP
        :param gp_outcome_transform: module used to transform outcomes in the GP
        """

        super(CustomMean, self).__init__(model, gp_input_transform, gp_outcome_transform)

    def forward(self, x):
        # set transformers to eval mode
        self.input_transformer.eval()
        self.outcome_transformer.eval()

        # transform inputs to mean space
        x_model = self.input_transformer.untransform(x)

        # evaluate model
        y_model = self.model(x_model)

        # transform outputs
        y = self.outcome_transformer(y_model)[0]

        self.input_transformer.eval()
        self.outcome_transformer.eval()

        return y
