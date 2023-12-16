#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    Create the PL Game
"""

# Import relevant modules
import torch
import typing
import torch.nn as nn


# Create Architecture
class PLGame(nn.Module):
    def __init__(self, prob_dim: int, scale: float = 20.0) -> None:
        super(PLGame, self).__init__()

        # Create a parameter to tune
        self.params = nn.Parameter(torch.randn(prob_dim))
        self.scale = scale

    def forward(
        self,
        dataP: torch.Tensor,
        dataQ: torch.Tensor,
        dataR: torch.Tensor,
        model: torch.nn.Module,
        loss: typing.Optional = None,
        index_slice: typing.Optional = None,
    ) -> torch.Tensor:
        """Performs a forward pass of the model,
        computes the loss and returns the loss for gradient computation later"""

        # Compute the model loss, with the parameters changed
        out = model(dataP)  # <x, Px>
        cross = (model.params * torch.matmul(dataR, self.params)).sum()  # <x, Ry>
        y_out = (
            -0.5
            * (
                self.params
                * torch.matmul(
                    dataQ + self.scale * torch.eye(len(dataQ)).to(self.params.device),
                    self.params,
                )
            ).sum()
        )  # Scale to make problem strongly concave

        return out + y_out + cross
