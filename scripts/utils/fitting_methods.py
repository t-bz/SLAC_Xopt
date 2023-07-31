import numpy as np
import datetime
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import torch
import logging

from torch import Tensor

logger = logging.getLogger(__name__)


def gaussian_linear_background(x, amp, mu, sigma, slope=0, offset=0):
    """Gaussian plus linear background fn"""
    return amp * np.exp(-((x - mu) ** 2) / 2 / sigma ** 2) + offset


class GaussianLeastSquares:
    def __int__(self, train_x: Tensor, train_y: Tensor):
        self.train_x = train_x
        self.train_y = train_y

    def forward(self, X):
        amp = X[..., 0]
        mu = X[..., 1]
        sigma = X[..., 2]
        offset = X[..., 3]
        pred = amp * torch.exp(-((self.train_x - mu) ** 2) / 2 / sigma ** 2) + offset
        return torch.sum((pred - self.train_y)**2)


def fit_gaussian_linear_background(y, inital_guess=None, show_plots=True,
                                   cut_area=None):
    """
    Takes a function y and inputs and fits and Gaussian with
    linear bg to it. Returns the best fit estimates of the parameters
    amp, mu, sigma and their associated 1sig error
    """

    x = np.arange(y.shape[0])
    inital_guess = inital_guess or {}

    # specify initial guesses if not provided in initial_guess
    smoothed_y = np.clip(gaussian_filter(y, 5), 0, np.inf)

    plt.figure()
    plt.plot(y)
    plt.plot(smoothed_y)

    offset = inital_guess.pop("offset", np.mean(smoothed_y[-10:]))
    amplitude = inital_guess.pop("amplitude", smoothed_y.max() - offset)
    # slope = inital_guess.pop("slope", 0)

    # use weighted mean and rms to guess
    center = inital_guess.pop("mu", np.average(x, weights=smoothed_y))
    sigma = inital_guess.pop(
        "sigma", 10.0
    )

    para0 = np.array([amplitude, center, sigma, offset])
    print(para0)
    print(smoothed_y.sum())

    # create LSQ model
    model = GaussianLeastSquares(torch.tensor(x), torch.tensor(y))

    try:
        para, para_error = curve_fit(
            gaussian_linear_background, x, y, p0=para0,
            bounds=(
                np.array((0, 0, 0, -np.inf)),
                np.array((np.inf, y.shape[0], np.inf, np.inf))
            )
        )
    except RuntimeError:
        logger.info("Fitting failed, taking initial guess.")
        para = para0
        para_error = np.array([0] * len(para0))

    para[2] = abs(para[2])  # Gaussian width is postivie definite
    # contraints on the output fit parameters
    if para[2] >= len(x):
        para[2] = len(x)

    if abs(para[1]) <= 0:
        para[1] = 0

    if abs(para[1]) >= len(x):
        para[1] = len(x)

    plot_fit(x, y, para, show_plots=show_plots)

    # taking relevant parameters
    para_vals = para[0:3]
    if np.any(np.diag(para_error) < 0) or np.any(np.diag(para_error) == 0):
        # hardcoded 5% error on init guess
        para_err_vals = list(np.array(para_vals) * 5 / 100)
    else:
        para_err_vals = np.sqrt(np.diag(para_error))[0:3]

    return para_vals, para_err_vals


def find_rms_cut_area(y, para0=None, show_plots=False, cut_area=0.05):
    """
    Takes a distribution (ndarray) and the desired cut area (5% is default).
    Returns the amp (max of array), mean of distribution, and rms (std) of dist
    """

    x = np.arange(y.shape[0])
    y = np.array([0 if ele < 0 else ele for ele in y])

    cumsum = np.cumsum(y)
    idLow = int(np.argwhere(cumsum < cut_area / 2 * cumsum[-1])[-1])
    idHigh = int(np.argwhere(cumsum > (1 - cut_area / 2) * cumsum[-1])[0])

    y[0:idLow] = y[idHigh:] = 0

    xx = x[y != 0]
    xp = y[y != 0]

    mean = np.sum(xx * xp) / np.sum(xp)
    mean2 = np.sum(xx * xx * xp) / np.sum(xp)
    var = mean2 - mean ** 2
    std = np.sqrt(var)

    # TODO: better estimate of peak amplitude in case of noise
    amp = max(y)

    para = np.array([amp, mean, std])

    # TODO: implement errors
    para_errors = np.array([0] * len(para))

    if show_plots:
        plot_fit(x, y, para)

    return para, para_errors


def plot_fit(x, y, para_x, savepath="", show_plots=True, save_plots=False):
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
        label=f"fit: amp={para_x[0]:.1f}, centroid={para_x[1]:.1f}, sigma={para_x[2]:.1f}",
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
