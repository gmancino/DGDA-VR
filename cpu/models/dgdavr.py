#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    Proposed method
"""

# Import packages
import math
import time
import numpy as np

# Import custom functions
from models.base import *


class DGDAVR(BaseRLR):
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
        Save relevant class information for the DGDA-VR method
        """

        # Init super
        super().__init__(
            feat_mat, labels, curr_weights, curr_Y, mixing_matrix, param_dict, "DGDA-VR"
        )

        # Collect appropriate parameters
        self.lrX, self.lrY, self.mini_batch, self.frequency, self.mega_batch = (
            param_dict["lrX"],
            param_dict["lrY"],
            param_dict["mini_batch"],
            param_dict["frequency"],
            param_dict["mega_batch"],
        )

        # Compute full gradients
        self.Dx, self.Dy = self.loss_grad(self.feat_mat, self.labels, self.X, self.Y)

        # Save all the previous terms
        self.Vx = self.Dx.copy()
        self.prev_Vx = self.Vx.copy()
        self.Vy = self.Dy.copy()
        self.prev_Vy = self.Vy.copy()
        self.prev_X = self.X.copy()
        self.prev_Y = self.Y.copy()

    def one_step(self, iteration: int) -> None:
        """Compute one algorithm iteration"""

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

        # Compute the gradient using the iteration count
        if iteration % self.frequency == 0:
            # Compute full gradients
            self.Vx, self.Vy = self.loss_grad(
                self.feat_mat, self.labels, self.X, self.Y
            )

        else:
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
            currX, currY = self.loss_grad(mini_feats, mini_labels, self.X, self.Y)
            prevX, prevY = self.loss_grad(
                mini_feats, mini_labels, self.prev_X, self.prev_Y
            )

            # Update
            self.Vx = currX.copy() - prevX.copy() + self.prev_Vx.copy()
            self.Vy = currY.copy() - prevY.copy() + self.prev_Vy.copy()

        # Perform communications
        self.Dx = (
            np.matmul(
                self.mixing_matrix,
                self.Dx.copy() + self.Vx.copy() - self.prev_Vx.copy(),
            )[np.newaxis, :]
            if self.num_nodes == 1
            else np.matmul(
                self.mixing_matrix,
                self.Dx.copy() + self.Vx.copy() - self.prev_Vx.copy(),
            )
        )
        self.Dy = (
            np.matmul(
                self.mixing_matrix,
                self.Dy.copy() + self.Vy.copy() - self.prev_Vy.copy(),
            )[np.newaxis, :]
            if self.num_nodes == 1
            else np.expand_dims(
                np.matmul(
                    self.mixing_matrix,
                    np.squeeze(
                        self.Dy.copy() + self.Vy.copy() - self.prev_Vy.copy(), axis=1
                    ),
                ),
                axis=1,
            )
        )

        # Save previous information
        self.prev_Vx = self.Vx.copy()
        self.prev_Vy = self.Vy.copy()
        self.prev_X = self.X.copy()
        self.prev_Y = self.Y.copy()
