# FISTA with proximal step replaced by learned denoiser
import torch
# Import algorithms, operators and functions from CIL optimisation module
from cil.optimisation.algorithms import GD, FISTA, PDHG
from cil.optimisation.operators import BlockOperator, GradientOperator,\
                                       GradientOperator
from cil.optimisation.functions import IndicatorBox, MixedL21Norm, L2NormSquared, \
                                       BlockFunction, L1Norm, LeastSquares, \
                                       OperatorCompositionFunction, TotalVariation, Function

# Import CIL Processors for preprocessing
from cil.processors import CentreOfRotationCorrector, Slicer, TransmissionAbsorptionConverter

# Import CIL display function
from cil.utilities.display import show2D

# Import from CIL ASTRA plugin
from cil.plugins.astra import ProjectionOperator

# Import FBP from CIL recon class
from cil.recon import FBP

#Import Total Variation from the regularisation toolkit plugin
from cil.plugins.ccpi_regularisation.functions import FGP_TV

# All external imports
import matplotlib.pyplot as plt
import math
from time import time
import gc
import numpy as np


class DenoiserProximal(Function):
    """
    DenoiserProximal is a custom CIL function that, when evaluated (__call__), returns 0. 
    It implements a proximal operator via a torch-based denoiser. When the
    proximal() method is called, the input CIL data container is converted into a PyTorch
    tensor, processed with the denoiser  using the specified noise level (tau), and then
    wrapped back into a CIL data container.

    Parameters:
        denoiser: The torch-based denoiser which accepts an input tensor and a noise level.
        device: The torch device (e.g., 'cuda' or 'cpu') on which the denoiser runs.
    """

    def __init__(self, denoiser, device='cuda'):
        self.device = torch.device(device)
        self.denoiser = denoiser 
        super(DenoiserProximal, self).__init__()

    def __call__(self, x):
        # This function merely returns 0 as its evaluation.
        return 0 

    def cil_to_torch(self, x):
        """
        Convert a CIL data container to a PyTorch tensor.

        This method extracts the 'array' attribute from the input CIL data container,
        moves the data to the designated device, and adjusts the tensor's shape by squeezing
        out the first dimension and adding a channel dimension. This reshaped tensor is then
        ready to be passed to the denoiser denoiser.

        Parameters:
            x: A CIL data container with an 'array' attribute containing the data.

        Returns:
            torch.Tensor: A PyTorch tensor formatted for the denoiser denoiser.
        """
        return (torch.tensor(x.array(), device=self.device)
                .squeeze(0).unsqueeze(1))
    
    def torch_to_cil(self, x_tens, out):
        """
        Convert a PyTorch tensor to a CIL data container.

        After the denoiser processes the input, this method converts the resulting PyTorch tensor
        back into the format expected by a CIL data container. It performs a reverse of the shaping operations
        applied in cil_to_torch (i.e., removing the channel dimension and adding back the batch dimension)
        and updates the 'array' attribute of the output container.

        Parameters:
            x_tens (torch.Tensor): The processed tensor from the denoiser.
            out: A pre-allocated CIL data container to store the final output data.
        """
        out.array[:] = (x_tens.squeeze(1).unsqueeze(0)
                        .detach().cpu().numpy())
            
    def proximal(self, x, tau, out=None):
        """
        Apply the proximal operator via a torch-based denoiser to a CIL data container.

        This method implements the proximal step by first converting the input CIL data container
        to a PyTorch tensor using cil_to_torch. The tensor is then passed to the denoiser along
        with the provided noise level 'tau'. The output tensor is converted back into a CIL data container using
        torch_to_cil. If no output container is provided (i.e., out is None), a new container is allocated
        based on the geometry of x.

        Parameters:
            x: The input CIL data container to be processed.
            tau (float): A scalar noise level parameter passed to the denoiser.
            out: (Optional) A pre-allocated CIL data container for returning the result. If not provided,
                 a new container is allocated.

        Returns:
            A CIL data container containing the denoiser-processed data.
        """
        if out is None: 
            out = x.geometry.allocate(None)

        with torch.no_grad():
            x_torch = self.cil_to_torch(x)
            x_torch = self.denoiser(x_torch, tau)
            self.torch_to_cil(x_torch, out)
        return out 
        