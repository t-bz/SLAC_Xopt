import datetime
import logging

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from torch import Tensor

from scripts.utils.batch_minimization import gen_candidates_scipy

logger = logging.getLogger(__name__)


def gaussian_linear_background(x, amp, mu, sigma, offset=0):
    """Gaussian plus linear background fn"""
    return amp * np.exp(-((x - mu) ** 2) / 2 / sigma ** 2) + offset


class GaussianLeastSquares:
    def __init__(self, train_x: Tensor, train_y: Tensor):
        self.train_x = train_x
        self.train_y = train_y

    def forward(self, X):
        amp = X[..., 0].unsqueeze(-1)
        mu = X[..., 1].unsqueeze(-1)
        sigma = X[..., 2].unsqueeze(-1)
        offset = X[..., 3].unsqueeze(-1)
        train_x = self.train_x.repeat(*X.shape[:-1], 1)
        train_y = self.train_y.repeat(*X.shape[:-1], 1)
        pred = amp * torch.exp(-((train_x - mu) ** 2) / 2 / sigma ** 2) + offset
        loss = -torch.sum((pred - train_y) ** 2, dim=-1).sqrt() / len(train_y)

        return loss


def fit_gaussian_linear_background(y, inital_guess=None, show_plots=True,
                                   n_restarts=50):
    """
    Takes a function y and inputs and fits and Gaussian with
    linear bg to it. Returns the best fit estimates of the parameters
    amp, mu, sigma and their associated 1sig error
    """

    x = np.arange(y.shape[0])
    inital_guess = inital_guess or {}

    # specify initial guesses if not provided in initial_guess
    smoothed_y = np.clip(gaussian_filter(y, 5), 0, np.inf)

    offset = inital_guess.pop("offset", np.mean(y[-10:]))
    amplitude = inital_guess.pop("amplitude", smoothed_y.max() - offset)
    # slope = inital_guess.pop("slope", 0)

    # use weighted mean and rms to guess
    center = inital_guess.pop("mu", np.average(x, weights=smoothed_y))
    sigma = inital_guess.pop(
        "sigma", 200
    )

    para0 = torch.tensor([amplitude, center, sigma, offset])

    # generate points +/- 50 percent
    rand_para0 = torch.rand((n_restarts, 4)) - 0.5
    rand_para0[..., 0] = (rand_para0[..., 0] + 1.0) * amplitude
    rand_para0[..., 1] = (rand_para0[..., 1] + 1.0) * center
    rand_para0[..., 2] = (rand_para0[..., 2] + 1.0) * sigma
    rand_para0[..., 3] = rand_para0[..., 3]*200 + offset

    para0 = torch.vstack((para0, rand_para0))

    bounds = torch.tensor((
        (0, 0, 1.0, -1000.0),
        (3000.0, y.shape[0], y.shape[0]*3, 1000.0)
    ))

    # clip on bounds
    para0 = torch.clip(para0, bounds[0], bounds[1])

    # create LSQ model
    model = GaussianLeastSquares(torch.tensor(x), torch.tensor(y))

    candidates, values = gen_candidates_scipy(
        para0,
        model.forward,
        lower_bounds=bounds[0],
        upper_bounds=bounds[1],
    )

    # get best fit from restarts
    candidate = candidates[torch.argmax(values)].detach().numpy()

    plot_fit(x, y, candidate, show_plots=show_plots)

    return candidate


def plot_fit(x, y, para_x, savepath="", show_plots=False, save_plots=False):
    """
    Plot  beamsize fit in x or y direction
    """
    timestamp = (datetime.datetime.now()).strftime("%Y-%m-%d_%H-%M-%S-%f")
    fig = plt.figure(figsize=(7, 5))
    plt.plot(x, y, "b-", label="data")
    plt.plot(
        x,
        gaussian_linear_background(x, *para_x),
        "r-",
        label=f"fit: amp={para_x[0]:.1f}, centroid={para_x[1]:.1f}, sigma="
              f"{para_x[2]:.1f}, offset={para_x[3]:.1f}",
    )
    plt.xlabel("Pixel")
    plt.ylabel("Counts")
    plt.legend(loc="lower center", bbox_to_anchor=(0.5, -0.3))
    plt.tight_layout()

    if save_plots:
        plt.savefig(savepath + f"beamsize_fit_{timestamp}.png", dpi=100)
    if show_plots:
        plt.show()
    plt.close()
