from typing import Callable, Dict
import numpy as np
from copy import deepcopy

import torch
from xopt import VOCS, Xopt, Evaluator
from xopt.generators import BayesianExplorationGenerator
from xopt.numerical_optimizer import GridOptimizer

import traceback

def characterize_emittance(
        vocs: VOCS,
        beamsize_evaluator: Callable,
        quad_length: float,
        drift_length: float,
        beam_energy: float,
        quad_strength_key: str,
        rms_x_key: str,
        rms_y_key: str,
        quad_strength_scale_factor: float = 1.0,
        n_iterations: int = 10,
        n_initial: int = 5,
        generator_kwargs: Dict = None,
        quad_scan_analysis_kwargs: Dict = None,
        dump_file: str = None
):
    """
    Script to evaluate beam emittance using an automated quadrupole scan.

    In order to work properly the callable passed as `beamsize_evaluator` should have
    the following properties:
        - It should accept a dictionary containing the quadrupole strength parameter
        identified by `quad_strength_key` and return a dictionary containing the keys
        `rms_x_key` and 'rms_y_key'.
        - Quadrupole strengths should be in units of [m^{-2}] (or should be scaled by
        `quadrupole_strength_scale_factor` to [m^{-2}]) with positive values denote
        focusing in the horizontal plane
        - RMS values returned should be in units of [mm]

    Note: rms values are specified in [mm] to avoid numerical fitting issues of
    hyperparameters in the GP models.

    Parameters
    ----------
    vocs: VOCS
        Xopt style VOCS object to describe the Bayesian exploration problem for
        quadrupole scan

    beamsize_evaluator : Callable
        Xopt style callable function that is evaluated during the optimization run.

    quad_length : float
        Effective magnetic length of the quadrupole in [m].

    drift_length : float
        Drift length from quadrupole to beam size measurement location in [m].

    beam_energy : float
        Beam energy in GeV.

    quad_strength_key : str
        Dictionary key used to specify the geometric quadrupole strength.

    quad_strength_scale_factor : float, optional
        Scale factor that scales quadrupole strength parameters given by
        `quad_strength_key` to [m^{-2}]

    rms_x_key : str
        Dictionary key used to specify the measurement of horizontal beam size.

    rms_y_key : str
        Dictionary key used to specify the measurement of vertical beam size.

    n_iterations : int, optional
        Number of exploration steps to run. Default: 5

    n_initial : int, optional
        Number of initial random samples to take before performing exploration
        steps. Default: 5

    generator_kwargs : dict, optional
        Dictionary passed to generator to customize Bayesian Exploration.

    quad_scan_analysis_kwargs : dict, optional
        Dictionary used to customize quadrupole scan analysis / emittance calculation.

    dump_file : str, optional
        Filename to specify dump file for Xopt object.

    Returns
    -------
    result : dict
        Results dictionary object containing the following keys. Note emittance units
        are [mm-mrad].
        - `x_emittance_median` : median value of the horizontal emittance
        - `x_emittance_05` : 5% quantile value of the horizontal emittance
        - `x_emittance_95` : 95% quantile value of the horizontal emittance
        - `y_emittance_median` : median value of the vertical emittance
        - `y_emittance_05` : 5% quantile value of the vertical emittance
        - `y_emittance_95` : 95% quantile value of the vertical emittance
    X : Xopt
        Xopt object containing the evaluator, generator, vocs and data objects.

    """

    # check for proper vocs object
    assert quad_strength_key in vocs.variable_names
    assert (rms_x_key in vocs.observables) and (rms_y_key in vocs.observables)

    # set up kwarg objects
    generator_kwargs = generator_kwargs or {}
    quad_scan_analysis_kwargs = quad_scan_analysis_kwargs or {}

    # set up Xopt object
    generator = BayesianExplorationGenerator(
        vocs=vocs,
        numerical_optimizer=GridOptimizer(n_grid_points=100),
        **generator_kwargs
    )
    beamsize_evaluator = Evaluator(function=beamsize_evaluator)
    X = Xopt(generator=generator, evaluator=beamsize_evaluator, vocs=vocs)
    X.options.dump_file = dump_file

    # evaluate random samples
    X.random_evaluate(1)

    # check to make sure the correct data is returned before performing exploration
    assert rms_y_key in X.data.columns
    assert rms_x_key in X.data.columns

    # perform exploration
    for i in range(n_iterations):
        X.step()

    try:
        analysis_data = deepcopy(X.data)[[quad_strength_key, rms_x_key, rms_y_key]].dropna()
        
        # get data from xopt object and scale to [m^{-2}]
        k = analysis_data[quad_strength_key].to_numpy(dtype=np.double) * quad_strength_scale_factor
        rms_x = analysis_data[rms_x_key].to_numpy(dtype=np.double)
        rms_y = analysis_data[rms_y_key].to_numpy(dtype=np.double)

        rmat_quad_to_screen_x = torch.tensor(
            [[-2.53258966,  3.7431645 ],
            [-1.22424655,  1.4145822 ]]
        ).double()
        rmat_quad_to_screen_y = torch.tensor(
            [[ 4.76367421,  7.53186817],
            [-0.5456758 , -0.65284864]]
        ).double()

        beta0_x = 5.01
        alpha0_x = 0.049
        beta0_y = 5.01
        alpha0_y = 0.049
    
        # calculate emittances (note negative sign in y-calculation)
        x_emit_stats = get_valid_emit_bmag_samples_from_quad_scan(
            k,
            rms_x,
            quad_length,
            rmat_quad_to_screen_x,
            beta0=beta0_x,
            alpha0=alpha0_x,
            **quad_scan_analysis_kwargs
        )
        y_emit_stats = get_valid_emit_bmag_samples_from_quad_scan(
            -k,
            rms_y,
            quad_length,
            rmat_quad_to_screen_y,
            beta0=beta0_y,
            alpha0=alpha0_y,
            **quad_scan_analysis_kwargs
        )
    
        # return emittance results in [mm-mrad]
        gamma = beam_energy / 0.511e-3
        result = {
            "x_emittance_median": float(gamma*torch.quantile(x_emit_stats[0], 0.5)),
            "x_emittance_05": float(gamma*torch.quantile(x_emit_stats[0], 0.05)),
            "x_emittance_95": float(gamma*torch.quantile(x_emit_stats[0], 0.95)),
            "y_emittance_median": float(gamma*torch.quantile(y_emit_stats[0], 0.5)),
            "y_emittance_05": float(gamma*torch.quantile(y_emit_stats[0], 0.05)),
            "y_emittance_95": float(gamma*torch.quantile(y_emit_stats[0], 0.95)),
            "bmag_x_median": float(torch.quantile(x_emit_stats[1], 0.5)),
            "bmag_y_median": float(torch.quantile(y_emit_stats[1], 0.5)),

        }
        
    except Exception:
        print(traceback.format_exc())
        result = {}

    finally:
        return result, X
        
from emitopt.utils import (propagate_sig, 
                            build_quad_rmat, fit_gp_quad_scan, 
                            plot_valid_thick_quad_fits
                          )

def compute_emit_bmag_thick_quad(k, y_batch, q_len, rmat_quad_to_screen, beta0=1., alpha0=0.):
    """
    A function that computes the emittance(s) corresponding to a set of quadrupole measurement scans
    using a thick quad model.

    Parameters:
        k: 1d torch tensor of shape (n_steps_quad_scan,)
            representing the measurement quad geometric focusing strengths in [m^-2]
            used in the emittance scan

        y_batch: 2d torch tensor of shape (n_scans x n_steps_quad_scan),
                where each row represents the mean-square beamsize outputs in [m^2] of an emittance scan
                with inputs given by k

        q_len: float defining the (longitudinal) quadrupole length or "thickness" in [m]
         
        rmat_quad_to_screen: the (fixed) 2x2 R matrix describing the transport from the end of the 
                measurement quad to the observation screen.

        beta0: the design beta twiss parameter at the screen
        
        alpha0: the design alpha twiss parameter at the screen
        
    Returns:
        emit: shape (n_scans x 1) containing the geometric emittance fit results for each scan
        bmag_min: (n_scans x 1) containing the bmag corresponding to the optimal point for each scan
        sig: shape (n_scans x 3 x 1) containing column vectors of [sig11, sig12, sig22]
        is_valid: 1d tensor identifying physical validity of the emittance fit results
        
    SOURCE PAPER: http://www-library.desy.de/preparch/desy/thesis/desy-thesis-05-014.pdf
    """
    
    # construct the A matrix from eq. (3.2) & (3.3) of source paper
    quad_rmats = build_quad_rmat(k, q_len) # result shape (len(k) x 2 x 2)
    total_rmats = rmat_quad_to_screen.reshape(1,2,2) @ quad_rmats # result shape (len(k) x 2 x 2)
    
    amat = torch.tensor([]) # prepare the A matrix
    for rmat in total_rmats:
        r11, r12 = rmat[0,0], rmat[0,1]
        amat = torch.cat((amat, torch.tensor([[r11**2, 2.*r11*r12, r12**2]])), dim=0)
    # amat result shape (len(k) x 3)
    
    # get sigma matrix elements just before measurement quad from pseudo-inverse
    sig = amat.pinverse().unsqueeze(0) @ y_batch.unsqueeze(-1) # shapes (1 x 3 x len(k)) @ (n_scans x len(k) x 1)
    # result shape (n_scans x 3 x 1) containing column vectors of [sig11, sig12, sig22]
    
    # compute emit
    emit = torch.sqrt(sig[:,0,0]*sig[:,2,0] - sig[:,1,0]**2).reshape(-1,1) # result shape (n_scans x 1)

    # check sigma matrix and emit for physical validity
    is_valid = torch.logical_and(sig[:,0,0] > 0, sig[:,2,0] > 0) # result 1d tensor
    is_valid = torch.logical_and(is_valid, ~torch.isnan(emit.flatten())) # result 1d tensor
    
    # propagate beam parameters to screen
    twiss_at_screen = propagate_sig(sig, emit, total_rmats)[1]
    # result shape (n_scans x len(k) x 3 x 1)
    
    # get design gamma0 from design beta0, alpha0
    gamma0 = (1 + alpha0**2) / beta0
    
    # compute bmag
    bmag = 0.5 * (twiss_at_screen[:,:,0,0] * gamma0
                - 2 * twiss_at_screen[:,:,1,0] * alpha0
                + twiss_at_screen[:,:,2,0] * beta0
               )
    # result shape (n_scans, n_steps_quad_scan)
    
    # select minimum bmag from quad scan
    bmag_min, bmag_min_id = torch.min(bmag, dim=1, keepdim=True) # result shape (n_scans, 1) 
    
    return emit, bmag_min, sig, is_valid


def get_valid_emit_bmag_samples_from_quad_scan(
    k,
    y,
    q_len,
    rmat_quad_to_screen,
    beta0=1.,
    alpha0=0.,
    n_samples=10000,
    n_steps_quad_scan=10,
    covar_module=None,
    visualize=False,
    tkwargs=None,
):
    """
    A function that produces a distribution of possible (physically valid) emittance values corresponding
    to a single quadrupole measurement scan. Data is first modeled by a SingleTaskGP, virtual measurement
    scan samples are then drawn from the model posterior, the samples are modeled by thick-quad transport
    to obtain fits to the beam parameters, and physically invalid results are discarded.

    Parameters:

        k: 1d numpy array of shape (n_steps_quad_scan,)
        representing the measurement quad geometric focusing strengths in [m^-2]
        used in the emittance scan

        y: 1d numpy array of shape (n_steps_quad_scan, )
            representing the root-mean-square beam size measurements in [m] of an emittance scan
            with inputs given by k

        q_len: float defining the (longitudinal) quadrupole length or "thickness" in [m]

        rmat_quad_to_screen: the (fixed) 2x2 R matrix describing the transport from the end of the 
                measurement quad to the observation screen.

        beta0: the design beta twiss parameter at the screen
        
        alpha0: the design alpha twiss parameter at the screen
        
        n_samples: the number of virtual measurement scan samples to evaluate for our "Bayesian" estimate

        n_steps_quad_scan: the number of steps in our virtual measurement scans

        covar_module: the covariance module to be used in fitting of the SingleTaskGP 
                    (modeling the function y**2 vs. k)
                    If None, uses ScaleKernel(MaternKernel()).

        visualize: boolean. Set to True to plot the parabolic fitting results.

        tkwargs: dict containing the tensor device and dtype

    Returns:
        emits_valid: a tensor of physically valid emittance results from sampled measurement scans.

        bmag_valid: (n_valid_scans x 1) containing the bmag corresponding to the optimal point 
                        from each physically valid fit.

        sig_valid: tensor, shape (n_valid_scans x 3 x 1), containing the computed 
                        sig11, sig12, sig22 corresponding to each physically valid
                        fit.

        sample_validity_rate: a float between 0 and 1 that describes the rate at which the samples
                        were physically valid/retained.
    """
    if tkwargs is None:
        tkwargs = {"dtype": torch.double, "device": "cpu"}

    k = torch.tensor(k, **tkwargs)
    y = torch.tensor(y, **tkwargs)

    k_virtual, bss = fit_gp_quad_scan(
        k=k,
        y=y,
        n_samples=n_samples,
        n_steps_quad_scan=n_steps_quad_scan,
        covar_module=covar_module,
        tkwargs=tkwargs
    )
    
    (emit, bmag, sig, is_valid) = compute_emit_bmag_thick_quad(k=k_virtual, 
                                                              y_batch=bss, 
                                                              q_len=q_len, 
                                                              rmat_quad_to_screen=rmat_quad_to_screen, 
                                                              beta0=beta0, 
                                                              alpha0=alpha0)

    sample_validity_rate = (torch.sum(is_valid) / is_valid.shape[0]).reshape(1)

    # filter on physical validity
    cut_ids = torch.tensor(range(emit.shape[0]))[is_valid]
    emit_valid = torch.index_select(emit, dim=0, index=cut_ids)
    bmag_valid = torch.index_select(bmag, dim=0, index=cut_ids)
    sig_valid = torch.index_select(sig, dim=0, index=cut_ids)

    if visualize:
        plot_valid_thick_quad_fits(k, 
                                   y, 
                                   q_len, 
                                   rmat_quad_to_screen,
                                   emit=emit_valid, 
                                   bmag=bmag_valid,
                                   sig=sig_valid, 
                                  )
    return emit_valid, bmag_valid, sig_valid, sample_validity_rate


def plot_valid_thick_quad_fits(k, y, q_len, rmat_quad_to_screen, emit, bmag, sig, ci=0.95, tkwargs=None):
    """
    A function to plot the physically valid fit results
    produced by get_valid_emit_bmag_samples_from_quad_scan().

    Parameters:

        k: 1d numpy array of shape (n_steps_quad_scan,)
        representing the measurement quad geometric focusing strengths in [m^-2]
        used in the emittance scan

        y: 1d numpy array of shape (n_steps_quad_scan, )
            representing the root-mean-square beam size measurements in [m] of an emittance scan
            with inputs given by k

        sig: tensor, shape (n_scans x 3 x 1), containing the computed sig11, sig12, sig22
                corresponding to each measurement scan
                
        emit: shape (n_scans x 1) containing the geometric emittance fit results for each scan

        q_len: float defining the (longitudinal) quadrupole length or "thickness" in [m]

        rmat_quad_to_screen: the (fixed) 2x2 R matrix describing the transport from the end of the 
                measurement quad to the observation screen.
                
        ci: "Confidence interval" for plotting upper/lower quantiles.

        tkwargs: dict containing the tensor device and dtype
    """
    from matplotlib import pyplot as plt

    if tkwargs is None:
        tkwargs = {"dtype": torch.double, "device": "cpu"}

    k_fit = torch.linspace(k.min(), k.max(), 10, **tkwargs)
    quad_rmats = build_quad_rmat(k_fit, q_len) # result shape (len(k_fit) x 2 x 2)
    total_rmats = rmat_quad_to_screen.reshape(1,2,2) @ quad_rmats # result shape (len(k_fit) x 2 x 2)
    sig_final = propagate_sig(sig, emit, total_rmats)[0] # result shape len(sig) x len(k_fit) x 3 x 1
    bss_fit = sig_final[:,:,0,0]

    upper_quant = torch.quantile(bss_fit.sqrt(), q=0.5 + ci / 2.0, dim=0)
    lower_quant = torch.quantile(bss_fit.sqrt(), q=0.5 - ci / 2.0, dim=0)
    
    fig, axs = plt.subplots(3)
    fig.set_size_inches(5,9)
    
    ax=axs[0]
    fit = ax.fill_between(
        k_fit.detach().numpy(),
        lower_quant*1.e6,
        upper_quant*1.e6,
        alpha=0.3,
        label='"Bayesian" Thick-Quad Model',
        zorder=1,
    )
    
    obs = ax.scatter(
        k, y*1.e6, marker="x", s=120, c="orange", label="Measurements", zorder=2
    )
    ax.set_title("Beam Size at Screen")
    ax.set_xlabel(r"Measurement Quad Geometric Focusing Strength ($[k]=m^{-2}$)")
    ax.set_ylabel(r"r.m.s. Beam Size ($[\sigma]=\mu m$)")
    ax.legend(handles=[obs, fit])
    
    ax=axs[1]
    ax.hist(emit.flatten(), density=True)
    ax.set_title('Geometric Emittance Distribution')
    ax.set_xlabel(r'Geometric Emittance ($[\epsilon]=m*rad$)')
    ax.set_ylabel('Probability Density')
    
    ax=axs[2]
    ax.hist(bmag.flatten(), range=(1,5), bins=20, density=True)
    ax.set_title(r'$\beta_{mag}$ Distribution')
    ax.set_xlabel(r'$\beta_{mag}$ at Screen')
    ax.set_ylabel('Probability Density')
    
    plt.tight_layout()
    plt.show()
