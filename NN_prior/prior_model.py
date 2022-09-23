import torch
from botorch import fit_gpytorch_model
from botorch.models import ModelListGP, SingleTaskGP
from botorch.models.transforms import Normalize, Standardize
from gpytorch import ExactMarginalLogLikelihood
from gpytorch.means.mean import Mean


class CustomMean(Mean):
    def __init__(
        self,
        model,
        model_input_names,
        model_output_names,
        input_names,
        output_name,
        gp_input_transform,
        gp_outcome_transform,
    ):
        """
        Custom prior mean for a GP based on an arbitrary model

        :param model: torch.nn.Module representation of the model
        :param model_input_names: list of feature names for model input
        :param model_output_names: list of feature names for model output
        :param input_names: list of feature names for input
        :param output_name: feature name for output
        :param gp_input_transform: module used to transform inputs in the GP
        :param gp_outcome_transform: module used to transform outcomes in the GP
        """

        super(CustomMean, self).__init__()
        self.input_names = input_names
        self.model_input_names = model_input_names
        self.NN_model = model
        self.NN_model.eval()
        self.NN_model.requires_grad_(False)

        self.gp_input_transform = gp_input_transform
        self.gp_outcome_transform = gp_outcome_transform

        # get ordering of column names to reshape the input x
        self.input_indicies = []
        for ele in input_names:
            self.input_indicies.append(model_input_names.index(ele))

        # get model output index
        self.output_index = model_output_names.index(output_name)

    def forward(self, x):
        """
        takes in input_transform(x) from GP, returns outcome_transform(y)
        """
        self.gp_outcome_transform.eval()
        self.gp_input_transform.eval()

        # untransform inputs
        x = self.gp_input_transform.untransform(x)

        # need to reorder columns
        x = x[self.input_indicies]

        # evaluate in batches
        m = self.NN_model(x)  # real x |-> real y
        m = self.gp_outcome_transform(m)  # real y -> standardized y

        self.gp_outcome_transform.train()
        self.gp_input_transform.train()

        return m


def create_nn_prior_model(data, vocs):
    tkwargs = {"dtype": torch.double, "device": "cpu"}
    input_data, objective_data, constraint_data = vocs.extract_data(data)
    train_X = torch.tensor(input_data.to_numpy(), **tkwargs)

    assert len(vocs.n_objectives) == 1
    objective_name = vocs.objective_names[0]

    input_transform = Normalize(
        vocs.n_variables, bounds=torch.tensor(vocs.bounds, **tkwargs)
    )
    outcome_transform = Standardize(1)

    # construct prior mean
    model = None
    model_input_names = []
    model_output_names = []
    input_names = []
    output_name = objective_name
    NN_y_transform = None

    prior_mean = CustomMean(
        model,
        model_input_names,
        model_output_names,
        input_names,
        output_name,
        NN_y_transform,
        outcome_transform,
    )

    objective_models = []

    train_Y = torch.tensor(
        objective_data[objective_name].to_numpy(), **tkwargs
    ).unsqueeze(-1)

    objective_models.append(
        SingleTaskGP(
            train_X,
            train_Y,
            input_transform=input_transform,
            outcome_transform=outcome_transform,
            mean_module=prior_mean
        )
    )
    mll = ExactMarginalLogLikelihood(
        objective_models[-1].likelihood, objective_models[-1]
    )
    fit_gpytorch_model(mll)

    return ModelListGP(*objective_models)
