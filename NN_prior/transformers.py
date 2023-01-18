import json

import torch
from botorch.models.transforms.input import AffineInputTransform


def create_sim_to_nn_transformers(transformers_fname):
    """
    create input/output transformers to translate sim params to nn params and back
    """
    data = json.load(open(transformers_fname))

    transformers = []
    for ele in ["x", "y"]:
        scale = torch.tensor(data[f"{ele}_scale"], dtype=torch.double)
        min_val = torch.tensor(data[f"{ele}_min"], dtype=torch.double)
        transform = AffineInputTransform(
            len(data[f"{ele}_min"]),
            1 / scale,
            -min_val / scale,
        )

        transformers.append(transform)

    return transformers

def create_pv_to_sim_transformers():
    """ create transformers to transform from experimental pvs to sim parameters
        NOTE: transformer also needs to re-order columns
    """
    pass
