"""Random Parameter Reinitialization (RPR) functions."""

import torch
from torch import nn
import numpy as np
from numpy import random


def reinitialize_parameters(
        model: nn.Module,
        reinit_prop: float,
        selection_method: str = "random",
        device: str = None,
        ):
    """
    Reinitialize a proportion of the model parameters randomly.
    """

    model_params = list(model.named_parameters()) # Get trainable parameters

    with torch.no_grad():

        for param_name in model_params:

            # get parameter
            param = model
            for name in param_name[0].split("."):
                param = getattr(param, name)
            dims = list(param.shape)
            num_params = param.numel()

            vmean = torch.mean(param)
            vstd = torch.std(param)

            if selection_method == "random":
                # Randomly select a proportion of the parameters to reinitialize
                num_reinit = int(num_params * reinit_prop)

                # Sample indices of flattened tensor (later converted to coordinates)
                # Note: indices need to be tuples to access location in tensor
                indices = random.choice(np.prod(dims), num_reinit, replace=False)

                # Reinitialize selected parameters inplace
                random_values = torch.normal(vmean, vstd, size=(num_reinit,)).to(device)
                # Note: these random values are not super random, and some training can be picked up by the mean and std
                for index,rval in zip(indices, random_values):
                    index = linear_index_to_coord(index, dims)
                    param[index] = rval

            elif selection_method == "neuron":
                # determine no. of neurons to reinitialize
                # identify parameters that are corresponding weight-bias pairs
                # select neurons to reinitialize
                # reinitialize row of weights and coresponding bias
                print("Neuron selection method not implemented yet.")

            else:
                print(f"Selection method {selection_method} not implemented yet.")
                raise NotImplementedError(f"Selection method {selection_method} not implemented yet.")


def linear_index_to_coord(
        index: int,
        dims: list,
        ):
    """
    Convert a linear index (from flattened dims) to a coordinate in the tensor.
    """
    coord = []
    for dim in reversed(dims):
        coord.append(index % dim)
        index //= dim
    return tuple(reversed(coord))