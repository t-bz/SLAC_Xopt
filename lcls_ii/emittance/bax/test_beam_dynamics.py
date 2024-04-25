import torch


def compute_bmag(sig, emit, total_rmats, beta0, alpha0):
    """
    parameters:
        sig: tensor shape batchshape x 3 x 1 giving the initial beam matrix before the measurement quad
        
        emit: tensor shape batchshape giving the emittance for each initial beam matrix
        
        total_rmats: tensor shape batchshape x nsteps x 2 x 2 giving the rmats that describe transport
                    through the meas quad and to the screen for each step in the measurement scan(s)
        
        beta0: float or tensor shape (batchshape x 1) designating the design beta (twiss) parameter
                at the screen
        
        alpha0: float or tensor shape (batchshape x 1) designating the design alpha (twiss) parameter
                at the screen
    returns:
        bmag_min: tensor shape batchshape containing the minimum (best) bmag from each measurement scan
    """
    twiss_at_screen = propagate_beam_quad_scan(sig, emit, total_rmats)[1]
    # result shape (batchshape x nsteps x 3 x 1)

    # get design gamma0 from design beta0, alpha0
    gamma0 = (1 + alpha0**2) / beta0

    # compute bmag
    bmag = 0.5 * (twiss_at_screen[...,0,0] * gamma0
                - 2 * twiss_at_screen[...,1,0] * alpha0
                + twiss_at_screen[...,2,0] * beta0
               )
    # result shape (batchshape x nsteps)

    # select minimum bmag from quad scan
    # bmag_min, bmag_min_id = torch.min(bmag, dim=-1) # result shape (batchshape)

    return bmag


def reconstruct_beam_matrix(k, beamsize_squared, q_len, rmat, thick=True):
    """
    Reconstructs the beam matrices corresponding to a set of quadrupole measurement scans
    using a thick quad model and the pseudoinverse method.

    Parameters:
        k: torch tensor of shape (n_steps_quad_scan,) or (batchshape x n_steps_quad_scan),
            representing the measurement quad geometric focusing strengths in [m^-2]
            used in a batch of emittance scans

        beamsize_squared: torch tensor of shape (batchshape x n_steps_quad_scan),
                where each row represents the mean-square beamsize outputs in [m^2] of an emittance scan
                with inputs given by k

        q_len: float defining the (longitudinal) quadrupole length or "thickness" in [m]
        
        rmat: tensor shape (2x2) or (batchshape x 2 x 2)
                containing the 2x2 R matrices describing the transport from the end of the 
                measurement quad to the observation screen.
                
    Outputs:
        
    """
    
    # construct the A matrix from eq. (3.2) & (3.3) of source paper
    quad_rmats = build_quad_rmat(k, q_len, thick=thick) # result shape (batchshape x nsteps x 2 x 2)
    total_rmats = rmat.unsqueeze(-3).double() @ quad_rmats.double() 
    # result shape (batchshape x nsteps x 2 x 2)
    
    # prepare the A matrix
    r11, r12 = total_rmats[...,0,0], total_rmats[...,0,1]
    amat = torch.stack((r11**2, 2.*r11*r12, r12**2), dim=-1)
    # amat result (batchshape x nsteps x 3)

    # get sigma matrix elements just before measurement quad from pseudo-inverse
    sig = amat.pinverse() @ beamsize_squared.unsqueeze(-1).double()
    # shapes (batchshape x 3 x nsteps) @ (batchshape x nsteps x 1)
    # result shape (batchshape x 3 x 1) containing column vectors of [sig11, sig12, sig22]
    
    return sig, total_rmats


def propagate_beam_quad_scan(sig_init, emit, rmat):
    """
    parameters:
        sig_init: shape batchshape x 3 x 1
        emit: shape batchshape
        rmat: shape batchshape x nsteps x 2 x 2
    returns:
        sig_final: shape batchshape x nsteps x 3 x 1
        twiss_final: shape batchshape x nsteps x 3 x 1
    """
    temp = torch.tensor([[[1., 0., 0.],
                           [0., -1., 0.],
                           [0., 0., 1.]]], device=sig_init.device).double()
    twiss_init = (temp @ sig_init) @ (1/emit.reshape(*emit.shape,1,1)) # result shape (batchshape x 3 x 1)
    
    twiss_transport = twiss_transport_mat_from_rmat(rmat) # result shape (batchshape x 3 x 3)

    twiss_final = twiss_transport @ twiss_init.unsqueeze(-3)
    # result shape (batchshape x nsteps x 3 x 1)

    sig_final = (temp @ twiss_final) @ emit.reshape(*emit.shape,1,1,1) 
    # result shape (batchshape x nsteps x 3 x 1)
    
    return sig_final, twiss_final


def twiss_transport_mat_from_rmat(rmat):
    c, s, cp, sp = rmat[...,0,0], rmat[...,0,1], rmat[...,1,0], rmat[...,1,1]
    result = torch.stack((
        torch.stack((c**2, -2*c*s, s**2), dim=-1), 
        torch.stack((-c*cp, c*sp + cp*s, -s*sp), dim=-1),
        torch.stack((cp**2, -2*cp*sp, sp**2), dim=-1)), 
        dim=-2
    )
    return result


def build_quad_rmat(k, q_len, thick=True):
    if thick:
        sqrt_k = k.abs().sqrt()

        c = (torch.cos(sqrt_k*q_len)*(k > 0) 
            + torch.cosh(sqrt_k*q_len)*(k < 0) 
            + torch.ones_like(k)*(k == 0)
            )
        s = (1./sqrt_k * torch.sin(sqrt_k*q_len)*(k > 0) 
             + 1./sqrt_k * torch.sinh(sqrt_k*q_len)*(k < 0) 
             + q_len*torch.ones_like(k)*(k == 0)
            )
        cp = (-sqrt_k * torch.sin(sqrt_k*q_len)*(k > 0) 
              + sqrt_k * torch.sinh(sqrt_k*q_len)*(k < 0)
              + torch.zeros_like(k)*(k == 0)
             )
        sp = (torch.cos(sqrt_k*q_len)*(k > 0) 
              + torch.cosh(sqrt_k*q_len)*(k < 0)
              + torch.ones_like(k)*(k == 0)
             )
                       
    else:
        c, s, cp, sp = (torch.ones_like(k), torch.zeros_like(k), -k*q_len, torch.ones_like(k))
        
    result = torch.stack((
        torch.stack((c, s), dim=-1), 
        torch.stack((cp, sp), dim=-1),), 
        dim=-2
    )
     
    return result
