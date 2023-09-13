from copy import deepcopy

import matplotlib.pyplot as plt
import torch


def visualize_step(generator, title=""):
    test_x = torch.linspace(*generator.vocs.bounds.flatten(), 500).double()

    vocs = generator.vocs
    
    # get the Gaussian process model from the generator
    model = generator.train_model()

    # get acquisition function from generator
    acq = generator.get_acquisition(model)

    # calculate model posterior and acquisition function at each test point
    # NOTE: need to add a dimension to the input tensor for evaluating the
    # posterior and another for the acquisition function, see
    # https://botorch.org/docs/batching for details
    # NOTE: we use the `torch.no_grad()` environment to speed up computation by
    # skipping calculations for backpropagation
    with torch.no_grad():
        posterior = model.posterior(test_x.unsqueeze(1))
        acq_val = acq(test_x.reshape(-1, 1, 1))

    # get mean function and confidence regions
    mean = posterior.mean
    l,u = posterior.mvn.confidence_region()

    # plot model and acquisition function
    n_outputs = vocs.n_outputs
    fig, ax = plt.subplots(n_outputs+1, 1, sharex="all")
    fig.set_size_inches(6, 10)

    # plot model posterior
    for i in range(n_outputs):
        ax[i].plot(test_x, mean[:,i],f"C{i}", label=vocs.output_names[i])
        ax[i].fill_between(test_x, l[:,i], u[:,i], 
                           alpha=0.25, fc=f"C{i}" )
    
        # add data to model plot
        ax[i].plot(
            generator.data[vocs.variable_names],
            generator.data[vocs.output_names[i]],
            f"C{i}o", label="Training data"
        )
        ax[i].set_ylabel(vocs.output_names[i])

        if vocs.output_names[i] in vocs.constraint_names:
            ax[i].axhline(vocs.constraints[vocs.output_names[i]][-1], ls="--")

    # plot acquisition function
    ax[-1].plot(test_x, acq_val.flatten())

    # add trust region info
    turbo_controller = deepcopy(generator.turbo_controller)
    if turbo_controller is not None:
        tr = turbo_controller.update_state(generator.data)
        tr = turbo_controller.get_trust_region(model).flatten()
        for ele in tr:
            for a in ax:
                a.axvline(ele)

    # add legend
    #ax[0].legend()

    ax[-1].set_ylabel(r"$\alpha(x)$")
    ax[-1].set_xlabel("x")
    ax[0].set_title(title)

    #plt.show()

    return fig, ax