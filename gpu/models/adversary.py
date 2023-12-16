#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    Create the adversary, which is just a tensor of the size of the data
"""

# Import relevant modules
import torch
import typing
import torch.nn as nn


#  Create Architecture
class Adversary(nn.Module):
    def __init__(self, dims: int = 28, scale: float = 10.0) -> None:
        super(Adversary, self).__init__()

        # Create a parameter to tune
        weights = torch.zeros((dims, dims))
        weights[0][0] = 1.0
        self.adversary_parameters = nn.Parameter(weights)
        self.scale = scale

    def forward(
        self,
        data: torch.Tensor,
        target: torch.Tensor,
        model: torch.nn.Module,
        loss: typing.Optional,
        index_slice: typing.Optional = None,
    ) -> torch.Tensor:
        """Performs a forward pass of the model,
        computes the loss and returns the loss for gradient computation later"""

        # Compute the model loss, with the parameters changed
        out = model(data + self.adversary_parameters)
        l = loss(
            out, target
        )  # the loss here will result in gradient computations with respect to both the parameters AND the adversary
        l -= (self.scale / 2) * torch.norm(self.adversary_parameters, p="fro") ** 2

        return l
