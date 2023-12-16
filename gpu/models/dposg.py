#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    DPOSG: https://proceedings.neurips.cc/paper/2020/file/7e0a0209b929d097bd3e8ef30567a5c1-Paper.pdf
"""

from __future__ import print_function
import argparse
import os
import time
import torch
import numpy
import typing
from mpi4py import MPI

# Custom classes
from models.base import BaseDL


# Declare main class
class DPOSG(BaseDL):
    def __init__(
        self,
        params: typing.Dict,
        mixing_matrix: numpy.array,
        training_data,
        training_labels,
        testing_data,
        testing_labels,
        comm_world,
        comm_size,
        current_rank,
    ):
        super().__init__(
            params,
            mixing_matrix,
            training_data,
            training_labels,
            testing_data,
            testing_labels,
            comm_world,
            comm_size,
            current_rank,
            "dposg",
            f"minibatch{params['mini_batch']}",
        )

        # Initialize the extra variables
        self.Zx = [self.X[ind].detach().clone() for ind in range(len(self.X))]
        self.Zy = [self.Y[ind].detach().clone() for ind in range(len(self.Y))]

        # Check problem instance
        if self.problem == "plgame":
            self.Dx, self.Dy = self.get_stoch_grads_pl(self.X, self.Y)
        else:
            self.Dx, self.Dy = self.get_stoch_grad_robust(self.X, self.Y)

    def one_step(self, iteration: int) -> tuple:
        # ----- PERFORM COMMUNICATION ----- #
        self.X, comm_time1x = self.communicate_with_neighbors(self.X)
        self.Y, comm_time1y = self.communicate_with_neighbors(self.Y)
        # ---------------------------------- #

        # TIME THIS EPOCH
        time_i = time.time()

        # UPDATE WEIGHTS
        self.Zx = [self.X[k] - self.lrX * self.Dx[k] for k in range(len(self.X))]
        self.Zy = [self.Y[k] + self.lrY * self.Dy[k] for k in range(len(self.Y))]

        # Perform the second update using the temporary terms
        if self.problem == "plgame":
            self.Dx, self.Dy = self.get_stoch_grads_pl(self.Zx, self.Zy)
        else:
            self.Dx, self.Dy = self.get_stoch_grad_robust(self.Zx, self.Zy)

        # Another pass
        self.X = [self.X[k] - self.lrX * self.Dx[k] for k in range(len(self.X))]
        self.Y = [self.Y[k] + self.lrY * self.Dy[k] for k in range(len(self.Y))]

        # Update epoch counter
        self.epochs += (self.size * self.mini_batch) / self.num_train

        # END time
        time_i_end = time.time()

        # SAVE TIMES
        comp_time = round(time_i_end - time_i, 4)
        comm_time = comm_time1x + comm_time1y

        return comp_time, comm_time
