import json

import torch
from pyro.distributions.transforms import Normalize


def create_sim_to_nn_transformers(transformers_fname):
    """ create input/output transformers to translate sim params to nn params and back
    """
    data = json.load(open(transformers_fname))

    transformers = []
    for ele in ["x", "y"]:
        transform = Normalize(len(data[f"{ele}_min"]))
        transform.ranges = 1 / torch.tensor(
            data[f"{ele}_scale"], dtype=torch.double
        ).unsqueeze(0)
        transform.mins = (
                -torch.tensor(data[f"{ele}_min"], dtype=torch.double).unsqueeze(0)
                * transform.ranges
        )

        transform.eval()
        transformers.append(transform)

    return transformers

def reorder_columns
def create_pv_to_sim_transformers():
    """ create transformers to transform from experimental pvs to sim parameters
        NOTE: transformer also needs to re-order
    """
