# Functions and classes used by different loss functions
import numpy as np
import torch
from torch import Tensor

def sigmoid(tensor: Tensor, temp: float) -> Tensor:
    """ temperature controlled sigmoid
    takes as input a torch tensor (tensor) and passes it through a sigmoid, controlled by temperature: temp
    """
    exponent = -tensor / temp
    # clamp the input tensor for stability
    exponent = torch.clamp(exponent, min=-50, max=50)
    y = 1.0 / (1.0 + torch.exp(exponent))
    return y


def compute_aff(x: Tensor, similarity: str = 'cosine') -> Tensor:
    """computes the affinity matrix between an input vector and itself"""
    if similarity == 'cosine':
        x = torch.mm(x, x.t())
    elif similarity == 'euclidean':
        x = x.unsqueeze(0)
        x = torch.cdist(x, x, p=2)
        x = x.squeeze(0)
        # The greater the distance the smaller affinity
        x = -x
    else:
        raise NotImplementedError(f"Incorrect similarity measure: {similarity}")
    return x