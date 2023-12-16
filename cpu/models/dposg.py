#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    DPOSG: https://arxiv.org/pdf/1910.12999.pdf
"""

# Import packages
import math
import time
import numpy as np

# Import custom functions
from models.base import *


class DPOSG(BaseRLR):
    def __init__(
        self,
        feat_mat: np.array,  # [N, num_samples, prob_dim]
        labels: np.array,  # [N, num_samples]
        curr_weights: np.array,  # [N, prob_dim]
        curr_Y: np.array,  # [N, num_samples]
        mixing_matrix: np.array,  # [N, N]
        param_dict: dict,
    ) -> None:
        """
        Save relevant class information
        """

        # Init super
        super().__init__(
            feat_mat, labels, curr_weights, curr_Y, mixing_matrix, param_dict, "DPOSG"
        )

        # Collect appropriate parameters
        self.lrX, self.lrY, self.mini_batch, self.frequency = (
            param_dict["lrX"],
            param_dict["lrY"],
            param_dict["mini_batch"],
            param_dict["frequency"],
        )

        # Save Z and Y
        self.Zx = curr_weights.copy()
        self.Zy = curr_Y.copy()

        # Initial full gradient
        self.Dx, self.Dy = self.loss_grad(self.feat_mat, self.labels, self.X, self.Y)

    def one_step(self, iteration: int) -> None:
        """Compute one algorithm iteration"""

        # Get the new parameters - project after the Y update
        if self.use_proj:
            self.Zx = self.projection_x(
                np.matmul(self.mixing_matrix, self.X.copy()) - self.lrX * self.Dx.copy()
            )
            self.Zy = self.projection_y(
                np.expand_dims(
                    np.matmul(self.mixing_matrix, np.squeeze(self.Y, axis=1)), axis=1
                )
                + self.lrY * self.Dy.copy()
            )
        else:
            self.Zx = (
                np.matmul(self.mixing_matrix, self.X.copy()) - self.lrX * self.Dx.copy()
            )
            self.Zy = (
                np.expand_dims(
                    np.matmul(self.mixing_matrix, np.squeeze(self.Y, axis=1)), axis=1
                )
                + self.lrY * self.Dy.copy()
            )

        # Compute stochastic batched iterates
        random_integers = np.random.randint(
            low=0, high=self.num_samples, size=(self.num_nodes, self.mini_batch)
        )  # [N, mini_batch]

        # Generate the data sets
        mini_feats = np.stack(
            [x[random_integers[ind], :] for ind, x in enumerate(self.feat_mat)]
        )  # [N, mini_batch, prob_dim]
        mini_labels = np.stack(
            [x[random_integers[ind]] for ind, x in enumerate(self.labels)]
        )

        # Compute the gradient differences
        self.Dx, self.Dy = self.loss_grad(mini_feats, mini_labels, self.Zx, self.Zy)

        # Get the new parameters - project after the Y update
        if self.use_proj:
            self.X = self.projection_x(
                np.matmul(self.mixing_matrix, self.X.copy()) - self.lrX * self.Dx.copy()
            )
            self.Y = self.projection_y(
                np.expand_dims(
                    np.matmul(self.mixing_matrix, np.squeeze(self.Y, axis=1)), axis=1
                )
                + self.lrY * self.Dy.copy()
            )
        else:
            self.X = (
                np.matmul(self.mixing_matrix, self.X.copy()) - self.lrX * self.Dx.copy()
            )
            self.Y = (
                np.expand_dims(
                    np.matmul(self.mixing_matrix, np.squeeze(self.Y, axis=1)), axis=1
                )
                + self.lrY * self.Dy.copy()
            )
