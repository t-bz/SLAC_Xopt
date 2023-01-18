import json

import torch
from botorch.models.transforms import Normalize

from transformed_model import TransformedModel
from transformers import create_sim_to_nn_transformers

"""
File to create surrogate models with their respective transformers
"""


def create_simulation_surrogate():
    """create surrogate model that uses simulation values as inputs and outputs"""
    model_fname = "torch_model.pt"
    transformers_fname = "configs/normalization.json"

    model = torch.load(model_fname).double()

    transformers = create_sim_to_nn_transformers(transformers_fname)

    simulation_model = TransformedModel(
        model,
        *transformers
    )

    return simulation_model


def create_experimental_surrogate():
    """ create surrogate model that uses experimental PV's as inputs and outputs"""

    # create simulation surrogate
    simulation_model = create_simulation_surrogate()

    # create transformer to transform values from



