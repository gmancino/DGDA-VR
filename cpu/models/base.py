#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    Base class
"""

# Import packages
import math
import time
import numpy as np


# Create the base class for which others use as implementation
class BaseRLR:
    def __init__(
        self,
        feat_mat: np.array,  # [N, num_samples, prob_dim]
        labels: np.array,  # [N, num_samples]
        curr_weights: np.array,  # [N, prob_dim]
        curr_Y: np.array,  # [N, 1, prob_dim]
        mixing_matrix: np.array,  # [N, N]
        param_dict: dict,
        algorithm_name: str = "My Algorithm",
    ) -> None:
        """
        Save relevant class information for the Decentralized Spider update
        """

        # Collect appropriate parameters
        self.l1 = param_dict["l1"]
        self.y_penalty = param_dict["y_penalty"]
        self.use_proj = param_dict["use_proj"]

        # Save the weights, etc.
        self.feat_mat = feat_mat.copy()
        self.labels = labels.copy()
        self.X = curr_weights.copy()
        self.Y = curr_Y.copy()
        self.mixing_matrix = mixing_matrix.copy()

        # Save values for this dataset
        self.num_nodes = self.mixing_matrix.shape[0]
        self.num_samples = self.feat_mat.shape[1]
        self.prob_dim = self.feat_mat.shape[-1]

        # Save the name and when to report iterations
        self.algorithm_name = algorithm_name
        self.report = param_dict["report"]

        # Reshape the feature matrices and labels to the original shapes for calculation of gradients
        self.feat_mat_full = np.reshape(
            self.feat_mat.copy(),
            ((np.prod(self.feat_mat.shape[:2]), self.feat_mat.shape[-1])),
        )  # [N * num_samples, prob_dim]
        self.labels_full = self.labels.flatten().copy()  # [N * num_samples, 1]
        self.total_num_samples = len(self.labels_full)

        # Save all the important information
        self.ConsensusX = []
        self.ConsensusY = []
        self.Gradient = []  # concatenated gradient -> norm
        self.Objective = []
        self.TotalOpt = []
        self.TestAcc = []  # At average

    def one_step(self, iteration: int) -> None:
        """Compute one algorithm iteration"""

        raise NotImplementedError

    # Define the projection function for y
    def projection_y(self, y: np.array) -> np.array:
        """
        Project onto {v: ||v||_2 <= 1}

        [N, 1, prob_dim] -> [N, 1, prob_dim]
        """

        # Compute the scale
        denominator = np.maximum(np.linalg.norm(y, axis=-1), 1.0)

        # Return the max-value computed row_wise
        return y / np.expand_dims(denominator, axis=-1)

    # Define the projection function for x
    def projection_x(self, x: np.array) -> np.array:
        """
        Soft-thresholding for x

        [N, prob_dim] -> [N, prob_dim]
        """
        # Return the max-value computed row_wise
        return np.sign(x) * np.maximum(np.abs(x) - self.l1, 0.0)

    # Define the logistic loss
    def loss(self, feats: np.array, b: np.array, x: np.array, y: np.array) -> np.array:
        """
        a = feature matrix          [N, num_samples, prob_dim]
        b = labels                  [N, num_samples]
        x = model weights           [N, prob_dim]
        y = adversary perturbations [N, 1, prob_dim]
        """

        return (
            np.log(1 + np.square(b - np.einsum("BNi,Bi -> BN", feats + y, x)) / 2).sum()
            - (self.y_penalty / 2)
            * (np.linalg.norm(np.squeeze(y, axis=1), axis=-1, p=2) ** 2).sum()
        )

    # Define corresponding gradient
    def loss_grad(
        self, feats: np.array, b: np.array, x: np.array, y: np.array
    ) -> np.array:
        """
        a = feature matrix          [N, num_samples, prob_dim]
        b = labels                  [N, num_samples]
        x = model weights           [N, prob_dim]
        y = adversary perturbations [N, 1, prob_dim]
        """

        # Compute the common term
        common_term = b - np.einsum("BNi,Bi -> BN", feats + y, x)  # [N, num_samples]

        # Get the coefficient
        coef = 4 * common_term / (np.square(common_term) + 2)  # [N, num_samples]

        # Compute the gradients
        scale = b.shape[1]
        gradX = -np.expand_dims(coef, axis=-1) * (feats + y)
        gradY = -np.expand_dims(coef.sum(1), axis=1) * x

        # Scale and reshape
        gradX = (1 / scale) * gradX.sum(1)  # [N, prob_dim]
        gradY = (1 / scale) * np.expand_dims(gradY, axis=1)

        return gradX, gradY - self.y_penalty * y

    def compute_errors(self, x_test: np.array, y_test: np.array) -> None:
        """Compute the relevant error terms"""

        # Compute the first errors
        avgX = np.mean(self.X, axis=0)  # [prob_dim, ]
        avgY = np.mean(np.squeeze(self.Y, axis=1), axis=0)  # [prob_dim, ]
        self.ConsensusX.append(
            (1 / self.num_nodes) * np.linalg.norm(self.X - avgX, ord="fro") ** 2
        )
        self.ConsensusY.append(
            (1 / self.num_nodes)
            * np.linalg.norm(np.squeeze(self.Y - avgY, axis=1), ord="fro") ** 2
        )

        # Gradient computation
        base = (
            self.labels_full
            - np.matmul(
                self.feat_mat_full + avgY[np.newaxis, :], avgX[:, np.newaxis]
            ).flatten()
        )  # [N * num_samples, ]
        coef = 4 * base / (np.square(base) + 2)

        gradX = -np.expand_dims(coef, axis=-1) * (
            self.feat_mat_full + avgY[np.newaxis, :]
        )
        gradY = -coef.sum() * avgX

        # Scale and reshape
        gradX = (1 / self.total_num_samples) * gradX.sum(0)
        gradY = (1 / self.total_num_samples) * gradY - self.y_penalty * avgY

        # Check if projection is used
        if self.use_proj:
            # Compute the inner terms
            y_inner = avgY.copy() + gradY.copy()
            x_inner = avgX.copy() - gradX.copy()

            # Compute the proximal mappings
            measureY = y_inner / np.maximum(np.linalg.norm(y_inner), 1.0) - avgY
            measureX = avgX - np.sign(x_inner) * np.maximum(
                np.abs(x_inner) - self.l1, 0.0
            )

        else:
            measureX = gradX.copy()
            measureY = gradY.copy()

        # Concatenate and compute norm
        grad = np.concatenate((measureX, measureY))

        # Compute the objective value
        x_obj = self.l1 * np.abs(avgX).sum() if self.use_proj else 0.0
        objective = (1 / self.total_num_samples) * np.log(
            np.square(
                self.labels_full
                - np.matmul(
                    self.feat_mat_full + avgY[np.newaxis, :], avgX[:, np.newaxis]
                ).flatten()
            )
            / 2
            + 1
        ).sum()  # - (self.y_penalty / 2) * np.linalg.norm(avgY) ** 2 + x_obj

        # Save the objective
        self.Objective.append(objective)

        # Compute the gradient about X, using the projected solutions from above
        self.Gradient.append(np.linalg.norm(grad, ord=2) ** 2)

        # Save total
        self.TotalOpt.append(
            self.Gradient[-1] + self.ConsensusX[-1] + self.ConsensusY[-1]
        )

        # Save accuracy
        self.TestAcc.append(self.test(x_test, y_test))

    def test(self, x_test: np.array, y_test: np.array) -> float:
        """Test the average model parameters on new data"""

        # First get the average X and Y
        avgX = np.mean(self.X, axis=0)  # [prob_dim, ]
        avgY = np.mean(np.squeeze(self.Y, axis=1), axis=0)

        # Get the predictions
        pred = np.array(np.sign(np.matmul(x_test + avgY, avgX))).flatten()

        # Save accuracy
        acc = sum(pred == y_test) / len(y_test) * 100.0

        return acc

    def solve(self, outer_iterations: int, x_test: np.array, y_test: np.array) -> None:
        """Use the algorithm to perform all the computations"""

        # Keep track of running time for this algorithm
        running_time = 0

        # Compute the first errors
        self.compute_errors(x_test, y_test)

        # Print some information about this method
        print(f"{'=' * 25} {self.algorithm_name} TRAINING {'=' * 25}")
        print(f"[COMMS]: {outer_iterations}")
        print(
            f"[GRAPH]: N = {self.num_nodes} | rho = {round(np.linalg.eigvalsh(self.mixing_matrix)[-2], 4) if self.num_nodes > 1 else 0}"
        )

        # Perform a for loop
        for i in range(1, outer_iterations + 1):
            # Time the iteration
            t0 = time.time()
            self.one_step(i)
            t1 = time.time()
            running_time += t1 - t0

            # Save errors
            if i % self.report == 0:
                self.compute_errors(x_test, y_test)

        print(
            f"[TIME]: {round(running_time, 2)} sec ({round(running_time / i, 4)} sec / iteration)"
        )
        print(f"{'=' * 25} {self.algorithm_name} FINISHED {'=' * 25}\n")
