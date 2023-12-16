#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    PL Model (https://openreview.net/pdf?id=JSha3zfdmSo)
"""
import torch
import torch.nn as nn


class PL(nn.Module):
    def __init__(self, prob_dim: int) -> None:
        super(PL, self).__init__()

        # Save values
        self.prob_dim = prob_dim

        # Declare layers
        self.params = nn.Parameter(torch.randn(self.prob_dim))

    def forward(self, x):
        """Forward pass of the model"""

        x = torch.matmul(x, self.params)
        x = self.params * x  # inner product

        return 0.5 * x.sum()
