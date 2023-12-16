#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    DM-HSGD: https://proceedings.neurips.cc/paper/2021/file/d994e3728ba5e28defb88a3289cd7ee8-Paper.pdf
"""

# Import packages
import math
import time
import numpy as np

# Import custom functions
from models.base import *


class DMHSGD(BaseRLR):
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
        Save relevant class information for the Decentralized Spider update
        """

        # Init super
        super().__init__(
            feat_mat, labels, curr_weights, curr_Y, mixing_matrix, param_dict, "DM-HSGD"
        )

        # Collect appropriate parameters
        self.lrX, self.lrY, self.mini_batch, self.frequency = (
            param_dict["lrX"],
            param_dict["lrY"],
            param_dict["mini_batch"],
            param_dict["frequency"],
        )
        self.betax, self.betay = param_dict["betax"], param_dict["betay"]

        # Initial full gradient
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

        # Compute the gradient using the iteration count
        if iteration == 1:
            pass

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
            self.Vx = currX.copy() + (1 - self.betax) * (
                self.prev_Vx.copy() - prevX.copy()
            )
            self.Vy = currY.copy() + (1 - self.betay) * (
                self.prev_Vy.copy() - prevY.copy()
            )

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

        # Save previous information
        self.prev_Vx = self.Vx.copy()
        self.prev_Vy = self.Vy.copy()
        self.prev_X = self.X.copy()
        self.prev_Y = self.Y.copy()
