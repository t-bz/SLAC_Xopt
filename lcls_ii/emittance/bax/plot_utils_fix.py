import torch
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.patches as mpatches
from emitopt.beam_dynamics import build_quad_rmat, propagate_beam_quad_scan

def plot_virtual_emittance_vs_tuning(optimizer, x_origin, ci=0.95, tkwargs:dict=None, n_samples=10000):
    """
    Plots the emittance cross-sections corresponding to the GP posterior beam size model. 
    This function uses n_samples to produce a confidence interval.
    It DOES NOT use the pathwise sample functions, but rather draws new samples using BoTorch's 
    built-in posterior sampling.
    """
    tkwargs = tkwargs if tkwargs else {"dtype": torch.double, "device": "cpu"}
    
    #extract GP models
    model = optimizer.generator.train_model()
    bax_model_ids = [optimizer.generator.vocs.output_names.index(name)
                            for name in optimizer.generator.algorithm.observable_names_ordered]
    bax_model = model.subset_output(bax_model_ids)
    meas_dim = optimizer.generator.algorithm.meas_dim
    
    bounds = optimizer.generator._get_optimization_bounds()
    tuning_domain = torch.cat((bounds.T[: meas_dim], bounds.T[meas_dim + 1:]))
    
    tuning_param_names = optimizer.vocs.variable_names
    del tuning_param_names[meas_dim]
    
    algorithm = optimizer.generator.algorithm
    
    n_tuning_dims = x_origin.shape[1]
    
    fig, axs = plt.subplots(2, n_tuning_dims, sharex='col', sharey='row')
    fig.set_size_inches(3*n_tuning_dims, 6)
        
    for i in range(n_tuning_dims):
        # do a scan of the posterior emittance (via valid sampling)
        x_scan = torch.linspace(*tuning_domain[i], 100, **tkwargs)
        x_tuning = x_origin.repeat(100, 1)
        x_tuning[:,i] = x_scan
        emit, is_valid, validity_rate = algorithm.evaluate_posterior_emittance_samples(bax_model, 
                                                                                   x_tuning, 
                                                                                   bounds,
                                                                                   tkwargs,
                                                                                   n_samples,
                                                                                   transform_target=False)
        quants = torch.tensor([])
        
        for j in range(len(x_scan)):
            cut_ids = torch.tensor(range(len(emit[:,j])), device=tkwargs['device'])[is_valid[:,j]]
            emit_valid = torch.index_select(emit[:,j], dim=0, index=cut_ids)
            q = torch.tensor([(1.-ci)/2., 0.5, (1.+ci)/2.], **tkwargs)
            if len(cut_ids)>=10:
                quant = torch.quantile(emit_valid, q=q, dim=0).reshape(1,-1)
            else:
                quant = torch.tensor([[float('nan'), float('nan'), float('nan')]], **tkwargs)
            quants = torch.cat((quants, quant))

        if n_tuning_dims==1:
            ax = axs[0]
        else:
            ax = axs[0,i]
        ax.fill_between(x_scan, quants[:,0], quants[:,2], alpha=0.3)
        ax.plot(x_scan, quants[:,1])
        ax.axvline(x_origin[0,i], c='r')
        
        ax.set_xlabel(tuning_param_names[i])
        if i==0:
            ax.legend()
            ax.set_ylabel('Emittance')
            
        if n_tuning_dims==1:
            ax = axs[1]
        else:
            ax = axs[1,i]
        ax.plot(x_scan, validity_rate, c='m')
        ax.axvline(x_origin[0,i], c='r')
        ax.set_ylim(0,1)

        ax.set_xlabel(tuning_param_names[i])
        if i==0:
            ax.legend()
            ax.set_ylabel('Sample Validity Rate')
            
    return fig, axs