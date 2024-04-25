import torch
from plot_utils_fix import plot_valid_thick_quad_fits
from emitopt.modeling import get_virtual_meas_scans
from test_beam_dynamics import reconstruct_beam_matrix, compute_bmag


def compute_emit_bayesian(
    k,
    beamsize,
    q_len,
    rmat,
    beta0=None,
    alpha0=None,
    n_samples=10000,
    n_steps_quad_scan=10,
    covar_module=None,
    noise_prior=None,
    visualize=False,
    tkwargs=None,
):
    """
    Produces a distribution of possible (physically valid) emittance values corresponding
    to a single quadrupole measurement scan. Data is first modeled by a SingleTaskGP, virtual measurement
    scan samples are then drawn from the model posterior, the samples are modeled by thick-quad transport
    to obtain fits to the beam parameters, and physically invalid results are discarded.

    Parameters:

        k: 1d numpy array of shape (n_steps_quad_scan,)
        representing the measurement quad geometric focusing strengths in [m^-2]
        used in the emittance scan

        beamsize: 1d numpy array of shape (n_steps_quad_scan, )
            representing the root-mean-square beam size measurements in [m] of an emittance scan
            with inputs given by k

        q_len: float defining the (longitudinal) quadrupole length or "thickness" in [m]
                    
        rmat: tensor containing the (fixed) 2x2 R matrix describing the transport from the end of the 
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
    tkwargs = twkargs if tkwargs else {"dtype": torch.double, "device": "cpu"}

    k = torch.tensor(k, **tkwargs)
    beamsize = torch.tensor(beamsize, **tkwargs)

    k_virtual, bss_virtual = get_virtual_meas_scans(
        k=k,
        y=beamsize,
        n_samples=n_samples,
        n_steps_quad_scan=n_steps_quad_scan,
        covar_module=covar_module,
        noise_prior=noise_prior,
        tkwargs=tkwargs
    )
    
    (emit, bmag, sig, is_valid) = compute_emit_bmag(k=k_virtual, 
                                                      beamsize_squared=bss_virtual, 
                                                      q_len=q_len, 
                                                      rmat=rmat, 
                                                      beta0=beta0, 
                                                      alpha0=alpha0)

    sample_validity_rate = (torch.sum(is_valid) / is_valid.shape[0]).reshape(1)

    # filter on physical validity
    cut_ids = torch.tensor(range(emit.shape[0]))[is_valid]
    emit = torch.index_select(emit, dim=0, index=cut_ids)
    sig = torch.index_select(sig, dim=0, index=cut_ids)

    if bmag is not None:
        bmag = torch.index_select(bmag, dim=0, index=cut_ids)

    if visualize:
        plot_valid_thick_quad_fits(k, 
                                   beamsize, 
                                   q_len,
                                   rmat,
                                   emit,
                                   bmag,
                                   sig
                                  )
    return emit, bmag, sig, sample_validity_rate


def compute_emit_bmag(k, beamsize_squared, q_len, rmat, beta0=None, alpha0=None, thick=True):
    """
    Computes the emittance(s) corresponding to a set of quadrupole measurement scans
    using a thick OR thin quad model.

    Parameters:
        k: torch tensor of shape (n_steps_quad_scan,) or (batchshape x n_steps_quad_scan)
            representing the measurement quad geometric focusing strengths in [m^-2]
            used in the emittance scan(s)

        beamsize_squared: torch tensor of shape (batchshape x n_steps_quad_scan),
                representing the mean-square beamsize outputs in [m^2] of the emittance scan(s)
                with inputs given by k

        q_len: float defining the (longitudinal) quadrupole length or "thickness" in [m]
        
        rmat: tensor shape (2x2) or (batchshape x 2 x 2)
                containing the 2x2 R matrices describing the transport from the end of the 
                measurement quad to the observation screen.

        beta0: float or tensor shape (batchshape x 1) designating the design beta twiss parameter at the screen
        
        alpha0: float or tensor shape (batchshape x 1) designating the design alpha twiss parameter at the screen

    Returns:
        emit: tensor shape (batchshape) containing the geometric emittance fit results for each scan
        bmag_min: tensor shape (batchshape) containing the bmag corresponding to the optimal point for each scan
        sig: shape tensor shape (batchshape x 3 x 1) containing column vectors of [sig11, sig12, sig22]
        is_valid: tensor shape (batchshape) identifying physical validity of the emittance fit results
        
    SOURCE PAPER: http://www-library.desy.de/preparch/desy/thesis/desy-thesis-05-014.pdf
    """
    # get initial sigma 
    sig, total_rmats = reconstruct_beam_matrix(k, beamsize_squared, q_len, rmat, thick=thick)
    
    emit = torch.sqrt(sig[...,0,0]*sig[...,2,0] - sig[...,1,0]**2) # result shape (batchshape)
    
    # check sigma matrix and emit for physical validity
    is_valid = torch.logical_and(sig[...,0,0] > 0, sig[...,2,0] > 0) # result batchshape
    is_valid = torch.logical_and(is_valid, ~torch.isnan(emit)) # result batchshape
    
    if (beta0 is not None) and (alpha0 is not None):
        bmag = compute_bmag(sig, emit, total_rmats, beta0, alpha0) # result batchshape
    elif beta0 is not None or alpha0 is not None:
        print("WARNING: beta0 and alpha0 must both be specified to compute bmag. Skipping bmag calc.")
        bmag = None
    else:
        bmag = None

    return emit, bmag, sig, is_valid


def normalize_emittance(emit, energy):
    gamma = energy / (0.511e-3)  # beam energy (GeV) divided by electron rest energy (GeV)
    beta = 1.0 - 1.0 / (2 * gamma**2)
    emit_n = gamma * beta * emit
    return emit_n
