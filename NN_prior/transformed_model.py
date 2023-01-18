import torch
from abc import ABC, abstractmethod


class Transformer(ABC, torch.nn.Module):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def untransform(self, x):
        pass


class IdentityTransformer(Transformer):
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(x):
        return x

    @staticmethod
    def untransform(x):
        return x


class InvertedTransformer(Transformer):
    def __init__(self, base_transformer):
        super().__init__()
        self.base_transformer = base_transformer

    def forward(self, x):
        return self.base_transformer.untransform(x)

    def untransform(self, x):
        return self.base_transformer(x)


class TransformedModel(torch.nn.Module):
    """
    Fixed model that requires an input and outcome transform to evaluate

    Transformer objects must have a forward method that transforms tensors into input
    coordinates for model and an untransform method that does the reverse.
    """

    def __init__(self, model, input_transformer, outcome_transformer):
        super().__init__()
        self.model = model
        self.model.eval()
        self.model.requires_grad_(False)
        self.input_transformer = input_transformer
        self.outcome_transformer = outcome_transformer

    def forward(self, x):
        # set transformers to eval mode
        self.input_transformer.eval()
        self.outcome_transformer.eval()

        # transform inputs to model space
        x_model = self.input_transformer(x)

        # evaluate model
        y_model = self.model(x_model)

        # transform outputs
        y = self.outcome_transformer(y_model)

        self.input_transformer.eval()
        self.outcome_transformer.eval()

        return y
