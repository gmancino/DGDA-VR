#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    Implementation of the GT-DA algorithm from:
        https://ieeexplore.ieee.org/document/9054056
"""

# Import packages
import math
import time
import numpy as np

# Import custom functions
from models.base import *


class GTDA(BaseRLR):
    def __init__(
        self,
        feat_mat: np.array,  # [N, num_samples, prob_dim]
        labels: np.array,  # [N, num_samples]
        curr_weights: np.array,  # [N, prob_dim]
        curr_Y: np.array,  # [N, 1, num_samples]
        mixing_matrix: np.array,  # [N, N]
        param_dict: dict,
    ) -> None:
        """
        Save relevant class information
        """

        # Init super
        super().__init__(
            feat_mat, labels, curr_weights, curr_Y, mixing_matrix, param_dict, "GTDA"
        )

        # Collect appropriate parameters
        self.lrX, self.lrY, self.inner_rounds = (
            param_dict["lrX"],
            param_dict["lrY"],
            param_dict["inner_rounds"],
        )

        # Compute full gradients
        self.Dx, self.Dy = self.loss_grad(self.feat_mat, self.labels, self.X, self.Y)

        # Save all the previous terms
        self.Vx = self.Dx.copy()
        self.prev_Vx = self.Vx.copy()

    def one_step(self, iteration: int) -> None:
        """Compute one algorithm iteration"""

        # Get the new parameters - project after the Y update
        if self.use_proj:
            self.X = self.projection_x(
                np.matmul(self.mixing_matrix, self.X.copy()) - self.lrX * self.Dx.copy()
            )
        else:
            self.X = (
                np.matmul(self.mixing_matrix, self.X.copy()) - self.lrX * self.Dx.copy()
            )

        for _ in range(self.inner_rounds):
            # Gradient calculation
            _, self.Dy = self.loss_grad(self.feat_mat, self.labels, self.X, self.Y)

            # Projection step
            pre_proj = self.Y.copy() + self.lrY * self.Dy.copy()
            if self.use_proj:
                self.Y = self.projection_y(pre_proj)
            else:
                self.Y = pre_proj.copy()

        # Get a new X gradient
        self.Vx, _ = self.loss_grad(self.feat_mat, self.labels, self.X, self.Y)

        # Gradient tracking
        self.Dx = (
            np.matmul(self.mixing_matrix, self.Dx.copy())[np.newaxis, :]
            + self.Vx.copy()
            - self.prev_Vx.copy()
            if self.num_nodes == 1
            else np.matmul(self.mixing_matrix, self.Dx.copy())
            + self.Vx.copy()
            - self.prev_Vx.copy()
        )

        # Save previous information
        self.prev_Vx = (
            self.Vx.copy()
        )  # Deterministic gradient, so no need to update further
