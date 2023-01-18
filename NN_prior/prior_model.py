


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
