#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    Base Class for deep learning with MPI
"""

from __future__ import print_function
import argparse
import os
import time
import torch
import numpy
import typing
from mpi4py import MPI
from torchvision import datasets, transforms

# Custom classes
from models.gtda import GTDA
from models.dposg import DPOSG
from models.dgdavr import DGDAVR
from models.dmhsgd import DMHSGD

# MPI set-up
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()  # get the current processor

# Main script to run
if __name__ == "__main__":
    # ------------------------------------------------
    # Gather arguments
    parser = argparse.ArgumentParser(description="Testing on minimax problem.")

    parser.add_argument(
        "--iterations",
        type=int,
        default=50000,
        help="Total number of communication rounds.",
    )
    parser.add_argument(
        "--comm_pattern",
        type=str,
        default="ring",
        choices=["ring", "random", "complete"],
        help="Communication pattern.",
    )
    parser.add_argument(
        "--problem",
        type=str,
        default="adversary",
        choices=["adversary", "plgame"],
        help="Problem setting.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="fc",
        choices=["fc"],
        help="Model architecture to choose from.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="mnist",
        choices=["mnist", "pl"],
        help="Dataset to choose from.",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="dgdavr",
        choices=["dposg", "dgdavr", "dmhsgd", "gtda"],
        help="Method to use.",
    )
    parser.add_argument(
        "--lrX", type=float, default=1e-1, help="Local learning rate for X."
    )
    parser.add_argument(
        "--lrY", type=float, default=1e-1, help="Local learning rate for Y."
    )
    parser.add_argument(
        "--betaX", type=float, default=1e-2, help="DM-HSGD gradient lr (x)."
    )
    parser.add_argument(
        "--betaY", type=float, default=1e-2, help="DM-HSGD gradient lr (y)."
    )
    parser.add_argument(
        "--trial", type=int, default=1, help="Which starting variables to use."
    )
    parser.add_argument("--mini_batch", type=int, default=1, help="Mini-batch size.")
    parser.add_argument(
        "--comm_rounds", type=int, default=1, help="Communication rounds."
    )
    parser.add_argument("--mega_batch", type=int, default=100, help="Mega-batch size.")
    parser.add_argument(
        "--frequency", type=int, default=100, help="When to compute the mega batch."
    )
    parser.add_argument("--eta", type=float, default=1.0, help="Penalty term.")
    parser.add_argument(
        "--report", type=int, default=25, help="How often to report criteria."
    )

    args = parser.parse_args()

    # ------------------------------------------------
    # Load Data and Transform
    if args.problem == "plgame":
        # Load the data
        data = torch.load("data/PL.pt")
        training_data = (data["P"].float(), data["Q"].float(), data["R"].float())
        training_labels = None
        testing_data = (data["P"].float(), data["Q"].float(), data["R"].float())
        testing_labels = None

    else:
        # Transform
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )

        # Subset data to local agent
        num_samples = 60000 // size
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST("data", train=True, download=False, transform=transform),
            batch_size=args.mini_batch,
            sampler=torch.utils.data.SubsetRandomSampler(
                [
                    i
                    for i in range(
                        int(rank * num_samples), int((rank + 1) * num_samples)
                    )
                ]
            ),
        )

        # Extract and scale the training data
        training_data = train_loader.dataset.transform(
            train_loader.dataset.data[
                int(rank * num_samples) : int((rank + 1) * num_samples)
            ].numpy()
        ).transpose(0, 1)
        training_labels = train_loader.dataset.targets[
            int(rank * num_samples) : int((rank + 1) * num_samples)
        ]

        # Get a full gradient
        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST("data", train=False, download=False, transform=transform),
            batch_size=10000 // size,
            sampler=torch.utils.data.SubsetRandomSampler(
                [
                    i
                    for i in range(
                        int(rank * (10000 // size)), int((rank + 1) * (10000 // size))
                    )
                ]
            ),
        )

        # Extract and scale the training data
        testing_data = test_loader.dataset.transform(
            test_loader.dataset.data[
                int(rank * (10000 // size)) : int((rank + 1) * (10000 // size))
            ].numpy()
        ).transpose(0, 1)
        testing_labels = test_loader.dataset.targets[
            int(rank * (10000 // size)) : int((rank + 1) * (10000 // size))
        ]

    # ------------------------------------------------
    # Set up communication matrix
    mixing_matrix = torch.tensor(
        numpy.load(f"mixing_matrices/{args.comm_pattern}_{size}.dat", allow_pickle=True)
    )

    # ------------------------------------------------
    # Print training info
    if rank == 0:
        print(f"{'=' * 25} STARTING TRAINING {'=' * 25}")
        print(
            f"[GRAPH INFO] {size} agents | rho = {round(torch.sort(torch.eig(mixing_matrix)[0][:, 0])[0][-2].item(), 4)}"
        )
        print(
            f"[TRAINING INFO] mini-batch = {args.mini_batch} | learning rates = ({args.lrX} / {args.lrY})\n"
        )

    # COMMUNICATION BARRIER
    comm.Barrier()

    # ------------------------------------------------
    # Train the model
    algo_params = {
        "problem": args.problem,
        "model": args.model,
        "lrX": args.lrX,
        "lrY": args.lrY,
        "mini_batch": args.mini_batch,
        "frequency": args.frequency,
        "mega_batch": args.mega_batch,
        "report": args.report,
        "dataset_name": args.dataset,
        "trial": args.trial,
        "eta": args.eta,
        "betaX": args.betaX,
        "betaY": args.betaY,
        "comm_rounds": args.comm_rounds,
    }
    if args.method == "dposg":
        solver = DPOSG(
            algo_params,
            mixing_matrix,
            training_data,
            training_labels,
            testing_data,
            testing_labels,
            comm,
            size,
            rank,
        )
    elif args.method == "gtda":
        solver = GTDA(
            algo_params,
            mixing_matrix,
            training_data,
            training_labels,
            testing_data,
            testing_labels,
            comm,
            size,
            rank,
        )
    elif args.method == "dgdavr":
        solver = DGDAVR(
            algo_params,
            mixing_matrix,
            training_data,
            training_labels,
            testing_data,
            testing_labels,
            comm,
            size,
            rank,
        )
    elif args.method == "dmhsgd":
        solver = DMHSGD(
            algo_params,
            mixing_matrix,
            training_data,
            training_labels,
            testing_data,
            testing_labels,
            comm,
            size,
            rank,
        )
    else:
        if rank == 0:
            print(
                f"[ERROR] method {args.method} is not a valid choice. Defaulting to DGDA-VR"
            )
        solver = DGDAVR(
            algo_params,
            mixing_matrix,
            training_data,
            training_labels,
            testing_data,
            testing_labels,
            comm,
            size,
            rank,
        )
    algo_time = solver.solve(args.iterations)
