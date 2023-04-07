import time
import torch
import matplotlib.pyplot as plt
import utilities as util

from typing import Callable, Optional
from botorch.models import SingleTaskGP
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition import ExpectedImprovement, UpperConfidenceBound, \
    FixedFeatureAcquisitionFunction
from botorch.optim import optimize_acqf


class BayesOpt:
    def __init__(self, surrogate: util.Surrogate,
                 ground_truth: Optional[torch.nn.Module] = None,
                 create_gp: Optional[Callable] = None,
                 **kwargs):
        """Defines a basic configuration of Bayesian Optimization (BO).

        Defines the objective and some functionality to perform Bayesian
        Optimization with different choices for the prior mean.

        Args:
            surrogate: The surrogate model used to construct the ground truth.
            ground_truth: The ground truth function based on the output of the
              surrogate model. Defaults to the Transverse Beam Size.
            create_gp: The function used to create a GP at each iteration of BO.

        Keyword Args:
            n_init (int): The number of samples BO is initialized with.
            n_step (int): The number of BO steps to perform in a single run.

        Attributes:
            x (torch.Tensor): Holds the inputs of the collected data set.
            y (torch.Tensor): Holds the outputs of the collected data set.
        """
        self.surrogate = surrogate
        if ground_truth is None:
            self.ground_truth = util.NegativeTransverseBeamSize(surrogate.model)
        else:
            self.ground_truth = ground_truth
        if create_gp is None:
            self.create_gp = self.create_default_gp
        else:
            self.create_gp = create_gp
        self.n_init = kwargs.get("n_init", 3)
        self.n_step = kwargs.get("n_step", 50)
        self.x, self.y = None, None

    def create_default_gp(self, x: torch.Tensor, y: torch.Tensor,
                          mean_module: Optional[torch.nn.Module] = None,
                          **kwargs) -> SingleTaskGP:
        """Creates a GP based on the given data set and prior mean.

        Args:
            x: The inputs of data set.
            y: The outputs of data set.
            mean_module: The prior mean module.

        Keyword Args:
            noise_value (float): The value of the trainable noise parameter. If
              not None, noise is fixed to this value (not trained). Defaults
              to 1e-4.
        """
        noise_value = kwargs.get("noise_value", 1e-4)
        if mean_module is None:
            in_transformer = Normalize(self.surrogate.x_dim,
                                       bounds=self.surrogate.x_lim.T)
            out_transformer = Standardize(1)
        else:
            in_transformer = mean_module.input_transformer
            out_transformer = mean_module.outcome_transformer
        gp = SingleTaskGP(x, y, mean_module=mean_module,
                          input_transform=in_transformer,
                          outcome_transform=out_transformer)
        if noise_value is not None:
            gp.likelihood.noise = torch.tensor(noise_value)
            gp.likelihood.noise.requires_grad = False
        return gp

    def initialize(self, n_init: Optional[int] = None):
        """Creates the initial data set for BO.

        Args:
            n_init: The number of samples in the initial data set. If None,
              defaults to the value of the eponymous BayesOpt attribute.
        """
        if n_init is None:
            n_init = self.n_init
        x = self.surrogate.sample_x(n_init, seed=0)
        y = self.ground_truth(x).reshape(n_init, 1)
        self.x, self.y = x, y

    def step(self, custom_mean: Optional[torch.nn.Module] = None,
             acq_name: str = "EI", fixed_feature=True, **kwargs):
        """Performs a single step of BO.

        Args:
            custom_mean: The prior mean module.
            acq_name: The name of the acquisition function to be used. Can be
              either "EI" or "UCB".
            fixed_feature: Whether to use a fixed feature acquisition function.

        Keyword Args:
            verbose (int): Determines how much information to print.
              Defaults to 1.
            acq_q (int): The number of candidates provided by the acquisition
              function. Defaults to 1.
            acq_num_restarts (int): The number of starting points for
              multi-start acquisition function optimization. Defaults to 5.
            acq_raw_samples (int): The number of samples for initialization of
              the acquisition function. Defaults to 20.
        """
        verbose = kwargs.get("verbose", 1)
        acq_q = kwargs.get("acq_q", 1)
        acq_num_restarts = kwargs.get("acq_num_restarts", 5)
        acq_raw_samples = kwargs.get("acq_raw_samples", 20)
        if self.x is None or self.y is None:
            raise ValueError("Found no initial data.")
        t0 = time.time()
        # create GP model and do maximum likelihood fits
        gp = self.create_gp(self.x, self.y, custom_mean)
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_mll(mll)
        # create and optimize acquisition function using the GP
        if acq_name == "EI":
            acq = ExpectedImprovement(gp, self.y.max())
        elif acq_name == "UCB":
            acq = UpperConfidenceBound(gp, beta=2.0)
        else:
            info = "Unknown acquisition function: {}".format(acq_name)
            raise ValueError(info)
        if fixed_feature:
            x_dim = self.surrogate.x_dim
            ff_columns = self.surrogate.fixed_feature_columns
            ff_values = self.surrogate.fixed_feature_values
            ff_acq = FixedFeatureAcquisitionFunction(acq, x_dim, ff_columns,
                                                     torch.as_tensor(ff_values))
            ff_candidate, _ = optimize_acqf(
                acq_function=ff_acq,
                bounds=self.surrogate.fixed_feature_x_lim.T,
                q=acq_q,
                num_restarts=acq_num_restarts,
                raw_samples=acq_raw_samples,
            )
            candidate = torch.zeros((1, x_dim)).double()
            candidate[:, ff_columns] = torch.as_tensor(ff_values).double()
            not_ff_columns = [i for i in range(x_dim) if i not in ff_columns]
            candidate[:, not_ff_columns] = ff_candidate
        else:
            candidate, _ = optimize_acqf(
                acq_function=acq,
                bounds=self.surrogate.x_lim.T,
                q=acq_q,
                num_restarts=acq_num_restarts,
                raw_samples=acq_raw_samples,
            )
        # add candidate and observation to data set
        self.x = torch.cat([self.x, candidate], dim=0)
        y_candidate = self.ground_truth(candidate).unsqueeze(dim=-1)
        self.y = torch.cat([self.y, y_candidate], dim=0)
        if verbose > 0:
            print("Runtime: {:.2f} seconds".format(time.time() - t0))

    def run(self, custom_mean: Optional[torch.nn.Module] = None,
            n_step: Optional[int] = None, n_init: Optional[int] = None,
            acq_name: str = "EI", fixed_feature=True, **kwargs):
        """Performs a single run of BO.

        Args:
            custom_mean: The prior mean module.
            n_step: The number of BO steps. If None, defaults to the value of
              the eponymous BayesOpt attribute.
            n_init: The number of samples BO is initialized with. If None,
              defaults to the value of the eponymous BayesOpt attribute.
            acq_name: The name of the acquisition function to be used. Can be
              either "EI" or "UCB".
            fixed_feature: Whether to use a fixed feature acquisition function.

        Keyword Args:
            verbose (int): Determines how much information to print.
              Defaults to 1.
            All other keyword arguments are passed to BayesOpt.step().
        """
        verbose = kwargs.pop("verbose", 1)
        t0 = time.time()
        self.initialize(n_init)
        if n_step is None:
            n_step = self.n_step
        for i in range(n_step):
            # print header and initial objective values
            if verbose > 0 and i == 0:
                print("{:12s} {:>12s}".format("STEP", "OBJECTIVE"))
                for j in range(self.y.shape[0]):
                    info = "{:<12s} {:12.8f}".format("init", self.y[j, 0])
                    if self.y[j, 0] >= torch.max(self.y):
                        info = "\033[0;32m" + info + '\x1b[0m'
                    print(info)
            self.step(custom_mean, acq_name, fixed_feature, verbose=0, **kwargs)
            # print BO samples
            if verbose > 0:
                info = "{:<12d} {:12.8f}".format(i + 1, self.y[-1, 0])
                if self.y[-1, 0] >= torch.max(self.y):
                    info = "\033[0;32m" + info + '\x1b[0m'
                print(info)
        if verbose > 0:
            print("Runtime: {:.2f} seconds".format(time.time() - t0))

    def plot_running_max(self, **kwargs):
        """Plots the running maximum for the current data set.

        Keyword Args:
            figsize (tuple): The size of the figure in inches.
              Defaults to (6, 4).

        Returns:
            The matplotlib figure and axis.
        """
        figsize = kwargs.get("figsize", (6, 4))
        running_max = util.running_max(self.y.squeeze())

        fig, ax = plt.subplots(1, 1, figsize=figsize)
        x = torch.arange(self.n_step + 1)
        y = self.ground_truth.unit_factor * running_max[self.n_init - 1:]
        ax.plot(x, y, "C0-")
        ax.set_xlabel("Step")
        ax.set_ylabel("{} {}".format(self.ground_truth.name,
                                     self.ground_truth.unit))
        fig.tight_layout()
        return fig, ax

    def plot_sample_distribution(self, **kwargs):
        """Plots the distribution of samples for the current data set.

        Compares the distribution of BO samples to the surrogate example data.

        Keyword Args:
            figsize (tuple): The size of the figure in inches.
              Defaults to (6, 4).

        Returns:
            The matplotlib figure and axis.
        """
        figsize = kwargs.get("figsize", (6, 4))
        raw_obj = self.ground_truth(self.surrogate.raw_x_data)
        unit_factor = self.ground_truth.unit_factor
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)
        entries, bins, patches = ax.hist(raw_obj * unit_factor,
                                         bins=50, density=False,
                                         label="raw samples")
        entries, _, _ = ax.hist(self.y[:, 0] * unit_factor, bins=bins,
                                density=False, label="BO samples")
        ax.set_xlabel("{} {}".format(self.ground_truth.name,
                                     self.ground_truth.unit))
        ax.set_ylabel("Counts")
        ax.legend()
        fig.tight_layout()
        return fig, ax
