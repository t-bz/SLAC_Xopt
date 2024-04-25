import copy
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Tuple, Union

import torch
from pydantic import Field

from scipy.optimize import minimize
from torch import Tensor
from xopt.generators.bayesian.bax.algorithms import Algorithm

from emitopt.sampling import (
    draw_linear_product_kernel_post_paths,
    draw_product_kernel_post_paths,
)
from botorch.models.model import Model, ModelList
from botorch.sampling.pathwise.posterior_samplers import draw_matheron_paths
from gpytorch.kernels import ProductKernel, MaternKernel

from test_analysis import compute_emit_bmag


def unif_random_sample_domain(n_samples, domain):
    ndim = len(domain)

    # uniform sample, rescaled, and shifted to cover the domain
    x_samples = torch.rand(n_samples, ndim) * torch.tensor(
        [bounds[1] - bounds[0] for bounds in domain]
    ) + torch.tensor([bounds[0] for bounds in domain])

    return x_samples


class ScipyMinimizeEmittanceXY(Algorithm, ABC):
    name = "ScipyMinimizeEmittance"
    x_key: str = Field(None,
        description="key designating the beamsize squared output in x from evaluate function")
    y_key: str = Field(None,
        description="key designating the beamsize squared output in y from evaluate function")
    scale_factor: float = Field(1.0,
        description="factor by which to multiply the quad inputs to get focusing strengths")
    q_len: float = Field(
        description="the longitudinal thickness of the measurement quadrupole"
    )
    rmat_x: Tensor = Field(None,
        description="tensor shape 2x2 containing downstream rmat for x dimension"
    )
    rmat_y: Tensor = Field(None,
        description="tensor shape 2x2 containing downstream rmat for y dimension"
    )
    twiss0_x: Tensor = Field(None,
        description="1d tensor length 2 containing design x-twiss: [beta0_x, alpha0_x] (for bmag)"
    )
    twiss0_y: Tensor = Field(None,
        description="1d tensor length 2 containing design y-twiss: [beta0_y, alpha0_y] (for bmag)"
    )
    meas_dim: int = Field(
        description="index identifying the measurement quad dimension in the model"
    )
    n_steps_measurement_param: int = Field(
        11, description="number of steps to use in the virtual measurement scans"
    )
    scipy_options: dict = Field(
        None, description="options to pass to scipy minimize")
    thick_quad: bool = Field(True,
        description="Whether to use thick-quad (or thin, if False) transport for emittance calc")
    init: str = Field('random',
        description="Either 'random' or 'smallest'. Determines initialization of sample minimization.")
    jitter: float = Field(0.,
        description="Float between 0 and 1 specifying randomness in sample minimization initialization")
    transform_target: bool = Field(True,
        description="Whether to square the emittance in sample minimization to ensure continuity")

    @property
    def observable_names_ordered(self) -> list:  
        # get observable model names in the order they appear in the model (ModelList)
        return [key for key in [self.x_key, self.y_key] if key]

    def get_execution_paths(self, model: ModelList, bounds: Tensor, tkwargs=None, verbose=False):
        if not (self.x_key or self.y_key):
            raise ValueError("must provide a key for x, y, or both.")
        if (self.x_key and self.rmat_x is None) or (self.y_key and self.rmat_y is None):
            raise ValueError("must provide rmat for each transverse dimension (x/y) being modeled.")
    
        tkwargs = tkwargs if tkwargs else {"dtype": torch.double, "device": "cpu"}

        cpu_tkwargs = {"dtype": torch.double, "device": "cpu"}

        temp_id = self.meas_dim + 1
        tuning_domain = torch.cat((bounds.T[: self.meas_dim], bounds.T[temp_id:]))

        cpu_models = [copy.deepcopy(m).cpu() for m in model.models]

        sample_funcs_list = []
        for cpu_model in cpu_models:
            if type(cpu_model.covar_module.base_kernel) == ProductKernel:
                sample_funcs = draw_product_kernel_post_paths(cpu_model, n_samples=self.n_samples)
            if type(cpu_model.covar_module.base_kernel) == MaternKernel:
                sample_funcs = draw_matheron_paths(cpu_model, sample_shape=torch.Size([self.n_samples]))
            sample_funcs_list += [sample_funcs]


        if self.init=='random':
            xs_tuning_init = unif_random_sample_domain(
                self.n_samples, tuning_domain
            ).double()
            x_tuning_init = xs_tuning_init.flatten()
        if self.init=='smallest':
            if len(self.observable_names_ordered) == 1:
                bss = model.models[0].outcome_transform.untransform(model.models[0].train_targets)[0]
            if len(self.observable_names_ordered) == 2:
                bss_x, bss_y = [m.outcome_transform.untransform(m.train_targets)[0]
                                for m in model.models]
                bss = torch.sqrt(bss_x * bss_y)

            x_smallest_observed_beamsize = model.models[0]._original_train_inputs[torch.argmin(bss)].reshape(1,-1)

            tuning_dims = list(range(bounds.shape[1]))
            tuning_dims.remove(self.meas_dim)
            tuning_dims = torch.tensor(tuning_dims)
            x_tuning_best = torch.index_select(x_smallest_observed_beamsize, dim=1, index=tuning_dims)
            # x_tuning_init = x_tuning_best.repeat(self.n_samples,1).flatten()
            x_tuning_random = unif_random_sample_domain(self.n_samples, tuning_domain).double()
            x_tuning_init = ((1.-self.jitter)*x_tuning_best.repeat(self.n_samples,1) + 
                             self.jitter*x_tuning_random).flatten()
        if self.init=='best':
            tuning_dims = list(range(bounds.shape[1]))
            tuning_dims.remove(self.meas_dim)
            tuning_dims = torch.tensor(tuning_dims)
            x_data = model.models[0].input_transform.untransform(model.models[0].train_inputs[0]) #shape n_tuning x ndim
            x_tuning = torch.index_select(x_data, dim=1, index=tuning_dims) #shape n_tuning x n_tuning_dims
            # x_tuning = torch.cat((x_data[:,:self.meas_dim], x_data[:,temp_id:]), dim=1) #shape n_tuning x n_tuning_dims
            emit_init = self.evaluate_posterior_emittance_samples(model, x_tuning, bounds, n_samples=10)[0] #shape n_samples x n_tuning
            emit_init = torch.nan_to_num(emit_init, float('inf'))
            best_samples = torch.min(emit_init, dim=0)[0] #shape n_tuning
            best_tuning_id = torch.argmin(best_samples)
            x_tuning_random = unif_random_sample_domain(self.n_samples, tuning_domain).double()
            x_tuning_init = ((1.-self.jitter)*x_tuning[best_tuning_id].reshape(1,-1).repeat(self.n_samples,1) +
                             self.jitter*x_tuning_random).flatten()
        # minimize
        def target_func_for_scipy(x_tuning_flat):
            return (
                self.sum_samplewise_emittance_target(
                    sample_funcs_list,
                    torch.tensor(x_tuning_flat, **cpu_tkwargs),
                    bounds,
                    cpu_tkwargs
                )
                .detach()
                .numpy()
            )

        def target_func_for_torch(x_tuning_flat):
            return self.sum_samplewise_emittance_target(
                    sample_funcs_list,
                    x_tuning_flat,
                    bounds,
                    cpu_tkwargs
                )

        def target_jac_for_scipy(x):
            return (
                torch.autograd.functional.jacobian(
                    target_func_for_torch, torch.tensor(x, **cpu_tkwargs)
                )
                .detach()
                .numpy()
            )

        
        # get bounds for sample emittance minimization (tuning domain)
        bounds_for_scipy = tuning_domain.repeat(self.n_samples, 1).detach().cpu().numpy()
        
        # perform sample emittance minimization
        res = minimize(
            target_func_for_scipy,
            x_tuning_init.detach().cpu().numpy(),
            jac=target_jac_for_scipy,
            bounds=bounds_for_scipy,
            options=self.scipy_options,
        )

        if verbose:
            print(
                "ScipyMinimizeEmittance evaluated",
                self.n_samples,
                "(pathwise) posterior samples",
                res.nfev,
                "times in get_sample_optimal_tuning_configs().",
            )

            print(
                "ScipyMinimizeEmittance evaluated",
                self.n_samples,
                "(pathwise) posterior sample jacobians",
                res.njev,
                "times in get_sample_optimal_tuning_configs().",
            )

            print(
                "ScipyMinimizeEmittance took",
                res.nit,
                "steps in get_sample_optimal_tuning_configs().",
            )

        x_tuning_best_flat = torch.tensor(res.x, **cpu_tkwargs)
        x_tuning_best = x_tuning_best_flat.reshape(
            self.n_samples, 1, -1
        )  # each row represents its respective sample's optimal tuning config

        emit_best, is_valid = self.evaluate_posterior_emittance_samples(sample_funcs_list, 
                                                                 x_tuning_best, 
                                                                 bounds, 
                                                                 tkwargs=cpu_tkwargs,
                                                                 transform_target=False)[:2]
        
        xs_exe = self.get_meas_scan_inputs(x_tuning_best, bounds, cpu_tkwargs)

        # evaluate posterior samples at input locations
        ys_exe_list = [sample_funcs(xs_exe).reshape(
            self.n_samples, self.n_steps_measurement_param, 1
        ) for sample_funcs in sample_funcs_list]
        ys_exe = torch.cat(ys_exe_list, dim=-1)

        if sum(is_valid) < 3:
            if verbose:
                print("Scipy failed to find at least 3 physically valid solutions.")
            # no cut
            cut_ids = torch.tensor(range(self.n_samples), device="cpu")
        else:
            # only keep the physically valid solutions
            cut_ids = torch.tensor(range(self.n_samples), device="cpu")[is_valid.flatten()]

        xs_exe = torch.index_select(xs_exe, dim=0, index=cut_ids)
        ys_exe = torch.index_select(ys_exe, dim=0, index=cut_ids)
        x_tuning_best_retained = torch.index_select(x_tuning_best, dim=0, index=cut_ids)
        emit_best_retained = torch.index_select(emit_best, dim=0, index=cut_ids)

        results_dict = {
            "xs_exe": xs_exe.to(**tkwargs),
            "ys_exe": ys_exe.to(**tkwargs),
            "x_tuning_best_retained": x_tuning_best_retained.to(**tkwargs),
            "emit_best_retained": emit_best_retained.to(**tkwargs),
            "x_tuning_best": x_tuning_best.to(**tkwargs),
            "emit_best": emit_best.to(**tkwargs),
            "is_valid": is_valid.to(device=tkwargs['device']),
            "sample_funcs_list": sample_funcs_list
        }

        return xs_exe.to(**tkwargs), ys_exe.to(**tkwargs), results_dict

    def get_meas_scan_inputs(self, x_tuning: Tensor, bounds: Tensor, tkwargs: dict=None):
        """
        A function that generates the inputs for virtual emittance measurement scans at the tuning
        configurations specified by x_tuning.

        Parameters:
            x_tuning: a tensor of shape n_points x n_tuning_dims, where each row specifies a tuning
                        configuration where we want to do an emittance scan.
                        >>batchshape x n_tuning_configs x n_tuning_dims (ex: batchshape = n_samples x n_tuning_configs)
        Returns:
            xs: tensor, shape (n_tuning_configs*n_steps_meas_scan) x d,
                where n_tuning_configs = x_tuning.shape[0],
                n_steps_meas_scan = len(x_meas),
                and d = x_tuning.shape[1] -- the number of tuning parameters
                >>batchshape x n_tuning_configs*n_steps x ndim
        """
        # each row of x_tuning defines a location in the tuning parameter space
        # along which to perform a quad scan and evaluate emit

        # expand the x tensor to represent quad measurement scans
        # at the locations in tuning parameter space specified by X
        tkwargs = tkwargs if tkwargs else {"dtype": torch.double, "device": "cpu"}

        x_meas = torch.linspace(
            *bounds.T[self.meas_dim], self.n_steps_measurement_param, **tkwargs
        )
        
        # prepare column of measurement scans coordinates
        x_meas_expanded = x_meas.reshape(-1,1).repeat(*x_tuning.shape[:-1],1)
        
        # repeat tuning configs as necessary and concat with column from the line above
        # to make xs shape: (n_tuning_configs*n_steps_quad_scan) x d ,
        # where d is the full dimension of the model/posterior space (tuning & meas)
        x_tuning_expanded = torch.repeat_interleave(x_tuning, 
                                                    self.n_steps_measurement_param, 
                                                    dim=-2)


        x = torch.cat(
            (x_tuning_expanded[..., :self.meas_dim], x_meas_expanded, x_tuning_expanded[..., self.meas_dim:]), 
            dim=-1
        )

        return x
            
    def sum_samplewise_emittance_target(self, sample_funcs_list, x_tuning_flat, bounds, tkwargs):
        assert len(x_tuning_flat.shape) == 1 and len(x_tuning_flat) == self.n_samples * (bounds.shape[1]-1)
        
        x_tuning = x_tuning_flat.double().reshape(self.n_samples, 1, -1)

        sample_emittance = self.evaluate_posterior_emittance_samples(sample_funcs_list, 
                                                                 x_tuning, 
                                                                 bounds, 
                                                                 tkwargs,
                                                                 transform_target=self.transform_target)[0]

        sample_targets_sum = torch.sum(sample_emittance)

        return sample_targets_sum

    def evaluate_posterior_emittance_samples(self, model, x_tuning, bounds, tkwargs:dict=None, n_samples=10000, transform_target=False):
        # x_tuning must be shape n_tuning_configs x n_tuning_dims
        tkwargs = tkwargs if tkwargs else {"dtype": torch.double, "device": "cpu"}
        x = self.get_meas_scan_inputs(x_tuning, bounds, tkwargs) # result shape n_tuning_configs*n_steps x ndim
        
        if isinstance(model, ModelList):
            assert len(x_tuning.shape)==2
            p = model.posterior(x) 
            bss = p.sample(torch.Size([n_samples])) # result shape n_samples x n_tuning_configs*n_steps x num_outputs (1 or 2)

            x = x.reshape(x_tuning.shape[0], self.n_steps_measurement_param, -1) # result n_tuning_configs x n_steps x ndim
            x = x.repeat(n_samples,1,1,1) 
            # result shape n_samples x n_tuning_configs x n_steps x ndim
            bss = bss.reshape(n_samples, x_tuning.shape[0], self.n_steps_measurement_param, -1)
            # result shape n_samples x n_tuning_configs x n_steps x num_outputs (1 or 2)
        else:
            assert x_tuning.shape[0]==self.n_samples
            beamsize_squared_list = [sample_funcs(x).reshape(*x_tuning.shape[:-1], self.n_steps_measurement_param)
                                     for sample_funcs in model]
            # each tensor in beamsize_squared (list) will be shape n_samples x n_tuning_configs x n_steps

            x = x.reshape(*x_tuning.shape[:-1], self.n_steps_measurement_param, -1)
            # n_samples x n_tuning_configs x n_steps x ndim
            bss = torch.stack(beamsize_squared_list, dim=-1) 
            # result shape n_samples x n_tuning_configs x n_steps x num_outputs (1 or 2)
            
        if self.x_key and not self.y_key:
            k = x[..., self.meas_dim] * self.scale_factor # n_samples x n_tuning x n_steps
            beamsize_squared = bss[...,0] # result shape n_samples x n_tuning x n_steps
            rmat = self.rmat_x.to(**tkwargs).repeat(*bss.shape[:2],1,1) # n_samples x n_tuning x 2 x 2
        elif self.y_key and not self.x_key:
            k = x[..., self.meas_dim] * (-1. * self.scale_factor) # n_samples x n_tuning x n_steps
            beamsize_squared = bss[...,0] # result shape n_samples x n_tuning x n_steps
            rmat = self.rmat_y.to(**tkwargs).repeat(*bss.shape[:2],1,1) # n_samples x n_tuning x 2 x 2
        else:
            k_x = (x[..., self.meas_dim] * self.scale_factor) # n_samples x n_tuning x n_steps
            k_y = k_x * -1. # n_samples x n_tuning x n_steps
            k = torch.cat((k_x, k_y)) # shape (2*n_samples x n_tuning x n_steps)
            
            beamsize_squared = torch.cat((bss[...,0], bss[...,1])) 
            # shape (2*n_samples x n_tuning x n_steps)

            rmat_x = self.rmat_x.to(**tkwargs).repeat(*bss.shape[:2],1,1)
            rmat_y = self.rmat_y.to(**tkwargs).repeat(*bss.shape[:2],1,1)
            rmat = torch.cat((rmat_x, rmat_y)) # shape (2*n_samples x n_tuning x 2 x 2)

        emit, bmag, sig, is_valid = compute_emit_bmag(k, 
                                          beamsize_squared, 
                                          self.q_len, 
                                          rmat,  
                                          thick=self.thick_quad)
        # result shapes: (n_samples x n_tuning), (n_samples x n_tuning), (n_samples x n_tuning x 3 x 1), (n_samples x n_tuning) 
        # or (2*n_samples x n_tuning), (2*n_samples x n_tuning), (2*n_samples x n_tuning x 3 x 1), (2*n_samples x n_tuning) 

        if transform_target:
            emit_squared = sig[...,0,0]*sig[...,2,0] - sig[...,1,0]**2 
            # result shape (n_samples x n_tuning) or (2*n_samples x n_tuning)
            if self.x_key and self.y_key:
                res = (emit_squared[:bss.shape[0]].pow(2) * 
                       emit_squared[bss.shape[0]:].pow(2)).sqrt()
                is_valid = torch.logical_and(is_valid[:bss.shape[0]], is_valid[bss.shape[0]:])
            else:
                res = emit_squared.pow(2)
        else:
            if self.x_key and self.y_key:
                res = (emit[:bss.shape[0]] * emit[bss.shape[0]:]).sqrt()
                is_valid = torch.logical_and(is_valid[:bss.shape[0]], is_valid[bss.shape[0]:])
            else:
                res = emit
            #final shapes: n_samples x n_tuning_configs
        
        validity_rate = torch.sum(is_valid, dim=0)/is_valid.shape[0]
        #shape n_tuning_configs
        
        return res, is_valid, validity_rate

class GridMinimizeEmitBmag(ScipyMinimizeEmittanceXY):
    name = "GridMinimizeEmitBmag"
    x_key: str = Field(None,
        description="key designating the beamsize squared output in x from evaluate function")
    y_key: str = Field(None,
        description="key designating the beamsize squared output in y from evaluate function")
    scale_factor: float = Field(1.0,
        description="factor by which to multiply the quad inputs to get focusing strengths")
    q_len: float = Field(
        description="the longitudinal thickness of the measurement quadrupole"
    )
    rmat_x: Tensor = Field(None,
        description="tensor shape 2x2 containing downstream rmat for x dimension"
    )
    rmat_y: Tensor = Field(None,
        description="tensor shape 2x2 containing downstream rmat for y dimension"
    )
    twiss0_x: Tensor = Field(None,
        description="1d tensor length 2 containing design x-twiss: [beta0_x, alpha0_x] (for bmag)"
    )
    twiss0_y: Tensor = Field(None,
        description="1d tensor length 2 containing design y-twiss: [beta0_y, alpha0_y] (for bmag)"
    )
    meas_dim: int = Field(
        description="index identifying the measurement quad dimension in the model"
    )
    n_steps_measurement_param: int = Field(
        11, description="number of steps to use in the virtual measurement scans"
    )
    thick_quad: bool = Field(True,
        description="Whether to use thick-quad (or thin, if False) transport for emittance calc")
    n_grid_points: int = Field(10,
        description="Number of points in each grid dimension. Only used if method='Grid'.")

    @property
    def observable_names_ordered(self) -> list:  
        # get observable model names in the order they appear in the model (ModelList)
        return [key for key in [self.x_key, self.y_key] if key]

    def get_execution_paths(self, model: ModelList, bounds: Tensor, tkwargs=None, verbose=False):
        if not (self.x_key or self.y_key):
            raise ValueError("must provide a key for x, y, or both.")
        if (self.x_key and self.rmat_x is None) or (self.y_key and self.rmat_y is None):
            raise ValueError("must provide rmat for each transverse dimension (x/y) being modeled.")
    
        tkwargs = tkwargs if tkwargs else {"dtype": torch.double, "device": "cpu"}

        temp_id = self.meas_dim + 1
        tuning_domain = torch.cat((bounds.T[: self.meas_dim], bounds.T[temp_id:]))

        tuning_bounds = tuning_domain.T
        assert isinstance(tuning_bounds, Tensor)
        # create mesh
        if len(tuning_bounds) != 2:
            raise ValueError("tuning_bounds must have the shape [2, ndim]")

        dim = len(tuning_bounds[0])
        # add in a machine eps
        eps = 1e-5
        linspace_list = [
            torch.linspace(
                tuning_bounds.T[i][0] + eps, tuning_bounds.T[i][1] - eps, self.n_grid_points, **tkwargs
            )
            for i in range(dim)
        ]

        xx = torch.meshgrid(*linspace_list, indexing="ij")
        mesh_pts = torch.stack(xx).flatten(start_dim=1).T

        # evaluate the function on grid points
        objective, emit, bmag, is_valid, validity_rate, bss = self.evaluate_objective(model, mesh_pts, bounds,
                                                            tkwargs=tkwargs, n_samples=self.n_samples)
        
        if self.x_key and self.y_key:
            emit = (emit[...,0] * emit[...,1]).sqrt()
            bmag = (bmag[...,0] * bmag[...,1]).sqrt()
            bmag_min, bmag_min_id = torch.min(bmag, dim=-1)
        else:
            emit = emit.squeeze(-1)
            bmag = bmag.squeeze(-1)
            bmag_min, bmag_min_id = torch.min(bmag, dim=-1)

        objective = torch.nan_to_num(objective, float('inf'))
        best_id = torch.argmin(objective, dim=1)
        best_x = torch.index_select(mesh_pts, dim=0, index=best_id).reshape(self.n_samples, 1, -1)
        xs_exe = self.get_meas_scan_inputs(best_x, bounds, tkwargs)

        ys_exe = torch.tensor([], **tkwargs)
        emit_best = torch.tensor([], **tkwargs)
        
        # is there a way to avoid this for loop? probably
        for sample_id in range(self.n_samples):
            ys_exe = torch.cat((ys_exe, torch.index_select(bss[sample_id], dim=0, index=best_id[sample_id])), dim=0)
            emit_best = torch.cat((emit_best, torch.index_select(emit[sample_id], dim=0, index=best_id[sample_id])), dim=0)

        emit_best = emit_best.reshape(self.n_samples, 1)

        results_dict = {
            "xs_exe": xs_exe,
            "ys_exe": ys_exe,
            "x_tuning_best": best_x,
            "emit_best": emit_best,
        }
    


        return xs_exe, ys_exe, results_dict

    def get_meas_scan_inputs(self, x_tuning: Tensor, bounds: Tensor, tkwargs: dict=None):
        """
        A function that generates the inputs for virtual emittance measurement scans at the tuning
        configurations specified by x_tuning.

        Parameters:
            x_tuning: a tensor of shape n_points x n_tuning_dims, where each row specifies a tuning
                        configuration where we want to do an emittance scan.
                        >>batchshape x n_tuning_configs x n_tuning_dims (ex: batchshape = n_samples x n_tuning_configs)
        Returns:
            xs: tensor, shape (n_tuning_configs*n_steps_meas_scan) x d,
                where n_tuning_configs = x_tuning.shape[0],
                n_steps_meas_scan = len(x_meas),
                and d = x_tuning.shape[1] -- the number of tuning parameters
                >>batchshape x n_tuning_configs*n_steps x ndim
        """
        # each row of x_tuning defines a location in the tuning parameter space
        # along which to perform a quad scan and evaluate emit

        # expand the x tensor to represent quad measurement scans
        # at the locations in tuning parameter space specified by X
        tkwargs = tkwargs if tkwargs else {"dtype": torch.double, "device": "cpu"}

        x_meas = torch.linspace(
            *bounds.T[self.meas_dim], self.n_steps_measurement_param, **tkwargs
        )
        
        # prepare column of measurement scans coordinates
        x_meas_expanded = x_meas.reshape(-1,1).repeat(*x_tuning.shape[:-1],1)
        
        # repeat tuning configs as necessary and concat with column from the line above
        # to make xs shape: (n_tuning_configs*n_steps_quad_scan) x d ,
        # where d is the full dimension of the model/posterior space (tuning & meas)
        x_tuning_expanded = torch.repeat_interleave(x_tuning, 
                                                    self.n_steps_measurement_param, 
                                                    dim=-2)


        x = torch.cat(
            (x_tuning_expanded[..., :self.meas_dim], x_meas_expanded, x_tuning_expanded[..., self.meas_dim:]), 
            dim=-1
        )

        return x
            

    def evaluate_objective(self, model, x_tuning, bounds, tkwargs:dict=None, n_samples=10000, use_bmag=True):
        emit, bmag, is_valid, validity_rate, bss = self.evaluate_posterior_emittance_samples(model, 
                                                                                             x_tuning, 
                                                                                             bounds, 
                                                                                             tkwargs, 
                                                                                             n_samples)
        if self.x_key and self.y_key:
            res = (emit[...,0] * emit[...,1]).sqrt()
            if use_bmag:
                bmag_mean = (bmag[...,0] * bmag[...,1]).sqrt()
                bmag_min, bmag_min_id = torch.min(bmag_mean, dim=-1)
                res = bmag_min * res
        else:
            res = emit
            if use_bmag:
                bmag_min, bmag_min_id = torch.min(bmag, dim=-2) # NEED TO CHECK THIS DIM
                res = (bmag_min * res).squeeze(-1)
        return res, emit, bmag, is_valid, validity_rate, bss

    def evaluate_posterior_emittance_samples(self, model, x_tuning, bounds, tkwargs:dict=None, n_samples=10000):
        # x_tuning must be shape n_tuning_configs x n_tuning_dims
        tkwargs = tkwargs if tkwargs else {"dtype": torch.double, "device": "cpu"}
        x = self.get_meas_scan_inputs(x_tuning, bounds, tkwargs) # result shape n_tuning_configs*n_steps x ndim
        
        if isinstance(model, ModelList):
            assert len(x_tuning.shape)==2
            p = model.posterior(x) 
            bss = p.sample(torch.Size([n_samples])) # result shape n_samples x n_tuning_configs*n_steps x num_outputs (1 or 2)

            x = x.reshape(x_tuning.shape[0], self.n_steps_measurement_param, -1) # result n_tuning_configs x n_steps x ndim
            x = x.repeat(n_samples,1,1,1) 
            # result shape n_samples x n_tuning_configs x n_steps x ndim
            bss = bss.reshape(n_samples, x_tuning.shape[0], self.n_steps_measurement_param, -1)
            # result shape n_samples x n_tuning_configs x n_steps x num_outputs (1 or 2)
        else:
            # assert x_tuning.shape[0]==self.n_samples
            assert x_tuning.shape[0]==1
            beamsize_squared_list = [sample_funcs(x).reshape(*x_tuning.shape[:-1], self.n_steps_measurement_param)
                                     for sample_funcs in model]
            # each tensor in beamsize_squared (list) will be shape n_samples x n_tuning_configs x n_steps

            x = x.reshape(*x_tuning.shape[:-1], self.n_steps_measurement_param, -1)
            # n_samples x n_tuning_configs x n_steps x ndim
            bss = torch.stack(beamsize_squared_list, dim=-1) 
            # result shape n_samples x n_tuning_configs x n_steps x num_outputs (1 or 2)
            
        if self.x_key and not self.y_key:
            k = x[..., self.meas_dim] * self.scale_factor # n_samples x n_tuning x n_steps
            beamsize_squared = bss[...,0] # result shape n_samples x n_tuning x n_steps
            rmat = self.rmat_x.to(**tkwargs).repeat(*bss.shape[:2],1,1) # n_samples x n_tuning x 2 x 2
            beta0 = self.twiss0_x[0].repeat(*bss.shape[:2], 1)
            alpha0 = self.twiss0_x[1].repeat(*bss.shape[:2], 1)
        elif self.y_key and not self.x_key:
            k = x[..., self.meas_dim] * (-1. * self.scale_factor) # n_samples x n_tuning x n_steps
            beamsize_squared = bss[...,0] # result shape n_samples x n_tuning x n_steps
            rmat = self.rmat_y.to(**tkwargs).repeat(*bss.shape[:2],1,1) # n_samples x n_tuning x 2 x 2
            beta0 = self.twiss0_y[0].repeat(*bss.shape[:2], 1)
            alpha0 = self.twiss0_y[1].repeat(*bss.shape[:2], 1)
        else:
            k_x = (x[..., self.meas_dim] * self.scale_factor) # n_samples x n_tuning x n_steps
            k_y = k_x * -1. # n_samples x n_tuning x n_steps
            k = torch.cat((k_x, k_y)) # shape (2*n_samples x n_tuning x n_steps)
            
            beamsize_squared = torch.cat((bss[...,0], bss[...,1])) 
            # shape (2*n_samples x n_tuning x n_steps)

            rmat_x = self.rmat_x.to(**tkwargs).repeat(*bss.shape[:2],1,1)
            rmat_y = self.rmat_y.to(**tkwargs).repeat(*bss.shape[:2],1,1)
            rmat = torch.cat((rmat_x, rmat_y)) # shape (2*n_samples x n_tuning x 2 x 2)
            
            beta0_x = self.twiss0_x[0].repeat(*bss.shape[:2], 1)
            beta0_y = self.twiss0_y[0].repeat(*bss.shape[:2], 1)
            beta0 = torch.cat((beta0_x, beta0_y))

            alpha0_x = self.twiss0_x[1].repeat(*bss.shape[:2], 1)
            alpha0_y = self.twiss0_y[1].repeat(*bss.shape[:2], 1)
            alpha0 = torch.cat((alpha0_x, alpha0_y))

        emit, bmag, sig, is_valid = compute_emit_bmag(k, 
                                          beamsize_squared, 
                                          self.q_len, 
                                          rmat, 
                                          beta0,
                                          alpha0,
                                          thick=self.thick_quad)
        # result shapes: (n_samples x n_tuning), (n_samples x n_tuning x nsteps), (n_samples x n_tuning x 3 x 1), (n_samples x n_tuning) 
        # or (2*n_samples x n_tuning), (2*n_samples x n_tuning x nsteps), (2*n_samples x n_tuning x 3 x 1), (2*n_samples x n_tuning) 

        if self.x_key and self.y_key:
            emit = torch.cat((emit[:bss.shape[0]].unsqueeze(-1), emit[bss.shape[0]:].unsqueeze(-1)), dim=-1) # n_samples x n_tuning x 1 or 2
            bmag = torch.cat((bmag[:bss.shape[0]].unsqueeze(-1), bmag[bss.shape[0]:].unsqueeze(-1)), dim=-1) # n_samples x n_tuning x n_steps x 1 or 2
            is_valid = torch.logical_and(is_valid[:bss.shape[0]], is_valid[bss.shape[0]:])
        else:
            emit = emit.unsqueeze(-1)
            bmag = bmag.unsqueeze(-1)
        #final shapes: n_samples x n_tuning_configs (?? NEED TO CHECK THIS, don't think it's correct)
        
        validity_rate = torch.sum(is_valid, dim=0)/is_valid.shape[0]
        #shape n_tuning_configs
        
        return emit, bmag, is_valid, validity_rate, bss

class ScipyBeamAlignment(Algorithm, ABC):
    name = "ScipyBeamAlignment"
    meas_dims: Union[int, list[int]] = Field(
        description="list of indeces identifying the measurement quad dimensions in the model"
    )
    x_key: str = Field(None,
        description="oberservable name for x centroid position"
    )
    y_key: str = Field(None,
        description="oberservable name for y centroid position"
    )

    @property
    def observable_names_ordered(self) -> list:  
        # get observable model names in the order they appear in the model (ModelList)
        return [key for key in [self.x_key, self.y_key] if key]
    
    def get_execution_paths(
        self, model: ModelList, bounds: Tensor, verbose=False
    ) -> Tuple[Tensor, Tensor, Dict]:
        """get execution paths that minimize the objective function"""

        meas_scans = torch.index_select(
            bounds.T, dim=0, index=torch.tensor(self.meas_dims)
        )
        ndim = bounds.shape[1]
        tuning_dims = [i for i in range(ndim) if i not in self.meas_dims]
        tuning_domain = torch.index_select(
            bounds.T, dim=0, index=torch.tensor(tuning_dims)
        )

        device = torch.tensor(1).device
        torch.set_default_tensor_type("torch.DoubleTensor")

        cpu_models = [copy.deepcopy(m).cpu() for m in model.models]
        sample_funcs_list = [
            draw_linear_product_kernel_post_paths(cpu_model, n_samples=self.n_samples)
                for cpu_model in cpu_models
            ]

        xs_tuning_init = unif_random_sample_domain(
            self.n_samples, tuning_domain
        ).double()

        x_tuning_init = xs_tuning_init.flatten()

        # minimize
        def target_func_for_scipy(x_tuning_flat):
            return (
                self.sum_samplewise_misalignment_flat_x(
                    sample_funcs_list,
                    torch.tensor(x_tuning_flat),
                    self.meas_dims,
                    meas_scans.cpu(),
                )
                .detach()
                .cpu()
                .numpy()
            )

        def target_func_for_torch(x_tuning_flat):
            return self.sum_samplewise_misalignment_flat_x(
                sample_funcs_list, x_tuning_flat, self.meas_dims, meas_scans.cpu()
            )

        def target_jac(x):
            return (
                torch.autograd.functional.jacobian(
                    target_func_for_torch, torch.tensor(x)
                )
                .detach()
                .cpu()
                .numpy()
            )

        res = minimize(
            target_func_for_scipy,
            x_tuning_init.detach().cpu().numpy(),
            jac=target_jac,
            bounds=tuning_domain.repeat(self.n_samples, 1).detach().cpu().numpy(),
            options={"eps": 1e-03},
        )
        if verbose:
            print(
                "ScipyBeamAlignment evaluated",
                self.n_samples,
                "(pathwise) posterior samples",
                res.nfev,
                "times in get_sample_optimal_tuning_configs().",
            )

            print(
                "ScipyBeamAlignment evaluated",
                self.n_samples,
                "(pathwise) posterior sample jacobians",
                res.njev,
                "times in get_sample_optimal_tuning_configs().",
            )

            print(
                "ScipyBeamAlignment took",
                res.nit,
                "steps in get_sample_optimal_tuning_configs().",
            )

        x_tuning_best_flat = torch.tensor(res.x)

        x_tuning_best = x_tuning_best_flat.reshape(
            self.n_samples, 1, -1
        )  # each row represents its respective sample's optimal tuning config


        if device.type == "cuda":
            torch.set_default_tensor_type("torch.cuda.DoubleTensor")

        xs = self.get_meas_scan_inputs(x_tuning_best, meas_scans, self.meas_dims)
        xs_exe = xs
        
        # evaluate posterior samples at input locations
        ys_exe_list = [sample_func(xs_exe).reshape(
            self.n_samples, 1+len(self.meas_dims), 1
        ) for sample_func in sample_funcs_list]
        ys_exe = torch.cat(ys_exe_list, dim=-1)
                            
        results_dict = {
            "xs_exe": xs_exe,
            "ys_exe": ys_exe,
            "x_tuning_best": x_tuning_best,
            "sample_funcs_list": sample_funcs_list,
        }

        return xs_exe, ys_exe, results_dict
        
    def sample_funcs_misalignment(
        self,
        sample_funcs_list,
        x_tuning,  # n x d tensor
        meas_dims,  # list of integers
        meas_scans,  # tensor of measurement device(s) scan inputs, shape: len(meas_dims) x 2
    ):
        """
        A function that computes the beam misalignment(s) through a set of measurement quadrupoles
        from a set of pathwise samples taken from a SingleTaskGP model of the beam centroid position with
        respect to some tuning devices and some measurement quadrupoles.

        arguments:
            sample_funcs_list: a list of pathwise posterior samples for x, y, or both 
                        from a SingleTaskGP model of the beam centroid positions (assumes Linear ProductKernel)
            x_tuning: a tensor of shape (n_samples x n_tuning_dims) where the nth row defines a point in
                        tuning-parameter space at which to evaluate the misalignment of the nth
                        posterior pathwise sample given by post_paths
            meas_dims: the dimension indeces of our model that describe the quadrupole measurement devices
            meas_scans: a tensor of measurement scan inputs, shape len(meas_dims) x 2, where the nth row
                        contains two input scan values for the nth measurement quadrupole

         returns:
             misalignment: the sum of the squared slopes of the beam centroid model output with respect to the
                             measurement quads
             xs: the virtual scan inputs
             ys: the virtual scan outputs (beam centroid positions)

        NOTE: meas scans only needs to have 2 values for each device because it is expected that post_paths
                are produced from a SingleTaskGP with Linear ProductKernel (i.e. post_paths should have
                linear output for each dimension).
        """
        xs = self.get_meas_scan_inputs(x_tuning, meas_scans, meas_dims)

        sample_misalignments_sum_list = [] # list to store the sum of the samplewise misalignments in x, y or both
        sample_ys_list = [] # list to store the centroid positions for x, y or both
        for sample_func in sample_funcs_list:
            ys = sample_func(xs)
            ys = ys.reshape(self.n_samples, -1)

            rise = ys[:, 1:] - ys[:, 0].reshape(-1, 1)
            run = (meas_scans[:, 1] - meas_scans[:, 0]).T.repeat(ys.shape[0], 1)
            slope = rise / run

            misalignment = slope.pow(2).sum(dim=1)
            sample_misalignments_sum_list += [misalignment]
            sample_ys_list += [ys]
        
        total_misalignment = sum(sample_misalignments_sum_list)
        return total_misalignment, xs, sample_ys_list

    def get_meas_scan_inputs(self, x_tuning, meas_scans, meas_dims):
        # meas_scans = torch.index_select(
        #     bounds.T, dim=0, index=torch.tensor(self.meas_dims)
        # )    
        n_steps_meas_scan = 1 + len(meas_dims)
        n_tuning_configs = x_tuning.shape[0]

        # construct measurement scan inputs
        xs = torch.repeat_interleave(x_tuning, n_steps_meas_scan, dim=-2)

        for i in range(len(meas_dims)):
            meas_dim = meas_dims[i]
            meas_scan = meas_scans[i]
            full_scan_column = meas_scan[0].repeat(n_steps_meas_scan, 1)
            full_scan_column[i + 1, 0] = meas_scan[1]
            full_scan_column_repeated = full_scan_column.repeat(*x_tuning.shape[:-1], 1)

            xs = torch.cat(
                (xs[..., :meas_dim], full_scan_column_repeated, xs[..., meas_dim:]), dim=-1
            )

        return xs

    def sum_samplewise_misalignment_flat_x(
        self, sample_funcs_list, x_tuning_flat, meas_dims, meas_scans
    ):
        """
        A wrapper function that computes the sum of the samplewise misalignments for more convenient
        minimization with scipy.

        arguments:
            Same as post_path_misalignment() EXCEPT:

            x_tuning_flat: a FLATTENED tensor formerly of shape (n_samples x ndim) where the nth
                            row defines a point in tuning-parameter space at which to evaluate the
                            misalignment of the nth posterior pathwise sample given by post_paths

            NOTE: x_tuning_flat must be 1d (flattened) so the output of this function can be minimized
                    with scipy minimization routines (that expect a 1d vector of inputs)
            NOTE: samplewise is set to True to avoid unncessary computation during simultaneous minimization
                    of the pathwise misalignments.
        """

        x_tuning = x_tuning_flat.double().reshape(self.n_samples, 1, -1)

        return torch.sum(
            self.sample_funcs_misalignment(
                sample_funcs_list, x_tuning, meas_dims, meas_scans
            )[0]
        )
