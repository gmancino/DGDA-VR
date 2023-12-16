#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    Script for running full tests for the RLR problem
"""

# Import packages
import os
import argparse
import numpy as np
from matplotlib import rc
import sklearn.datasets as sk

# Import custom functions
from models.gtda import GTDA
from models.dposg import DPOSG
from models.dmhsgd import DMHSGD
from models.dgdavr import DGDAVR

# Set colors and styles
# Change size
SMALL_SIZE = 18
MEDIUM_SIZE = 22
BIGGER_SIZE = 30

# Set sizes of axis
rc("axes", titlesize=SMALL_SIZE)  # fontsize of the axes title
rc("axes", labelsize=SMALL_SIZE)  # fontsize of the x and y labels
rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
rc("legend", fontsize=SMALL_SIZE)
fd = {"size": MEDIUM_SIZE, "family": "serif", "serif": ["Computer Modern"]}
rc("font", **fd)
rc("text", usetex=True)

# Set colors
cs = {
    "orange": "#E69F00",
    "sky_blue": "#56B4E9",
    "green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "red": "#D55E00",
    "pink": "#CC79A7",
    "black": "#000000",
}


# Run main
if __name__ == "__main__":
    # ------------------------------------------------
    # Parse user arguments
    parser = argparse.ArgumentParser(
        description="Test the various methods on the RLR problem."
    )

    # DATA
    parser.add_argument(
        "--dataset",
        type=str,
        default="a9a",
        choices=["a9a", "ijcnn1"],
        help="Dataset to use.",
    )
    parser.add_argument(
        "--mm", type=str, default="ring", choices=["ring"], help="Mixing matrix to use."
    )
    parser.add_argument("--size", type=int, default=20, help="Mixing matrix to use.")

    # METHOD
    parser.add_argument(
        "--method",
        type=str,
        default="ours",
        choices=["ours", "dmhsgd", "dposg", "gtda"],
        help="Method to use.",
    )

    # TRIALS
    parser.add_argument(
        "--trials", type=int, default=5, help="Number of initial points to try."
    )

    # TRAINING PARAMETERS
    parser.add_argument(
        "--iterations", type=int, default=5000, help="Number of updates."
    )
    parser.add_argument("--batch", type=int, default=32, help="Mini-batch size.")
    parser.add_argument("--mega_batch", type=int, default=1000, help="Mega-batch size.")
    parser.add_argument(
        "--frequency",
        type=int,
        default=32,
        help="How often to compute full gradient/mega-batch gradient.",
    )
    parser.add_argument(
        "--report", type=int, default=50, help="How often to report errors."
    )
    parser.add_argument("--betax", type=float, default=0.01, help="DM-HSGD term (X).")
    parser.add_argument("--betay", type=float, default=0.01, help="DM-HSGD term (Y).")
    parser.add_argument(
        "--inner_rounds", type=int, default=1, help="How many times to update Y."
    )

    args = parser.parse_args()

    # ------------------------------------------------
    # Load data and transform
    [X_train, y_train] = sk.load_svmlight_file(
        f"data/{args.dataset}/{args.dataset}.txt"
    )
    X_train = X_train.todense()
    [X_test, y_test] = sk.load_svmlight_file(
        f"data/{args.dataset}/{args.dataset}_test.txt", n_features=X_train.shape[1]
    )
    X_test = X_test.todense()

    # Load the mixing matrix
    mm = np.array(
        np.load(f"mixing_matrices/mm_{args.mm}_{args.size}.dat", allow_pickle=True)
    )
    N = mm.shape[0]

    # Transform the data into block structure
    # Set number to include (may result in dropping some samples)
    n_include = len(X_train) // N

    # Set data of the shape [N, number_samples, dim_of_problem] for faster computations later on
    split_X_train = np.array_split(
        np.expand_dims(np.array(X_train[: n_include * N, :]), axis=0), N, axis=1
    )  # split along the rows into N roughly equally sized partitions
    X_train_block = np.squeeze(np.stack(split_X_train, axis=0), 1)

    # Split y terms
    split_y_train = np.array_split(
        np.expand_dims(y_train[: n_include * N], axis=0), N, axis=1
    )  # split along the rows into N roughly equally sized partitions
    y_train_block = np.squeeze(np.stack(split_y_train, axis=0), 1)

    # ------------------------------------------------
    # Allocate space for each trial to be saved
    save_consensusX = np.zeros((args.trials, args.iterations // args.report + 1))
    save_consensusY = np.zeros((args.trials, args.iterations // args.report + 1))
    save_total = np.zeros((args.trials, args.iterations // args.report + 1))
    save_gradients = np.zeros((args.trials, args.iterations // args.report + 1))
    save_accuracy = np.zeros((args.trials, args.iterations // args.report + 1))

    # ------------------------------------------------
    # Set up learning rate dictionary
    lrDict = {
        "ours": {
            "a9a": {"lrX": 0.1, "lrY": 0.1},
            "ijcnn1": {"lrX": 0.01, "lrY": 0.1},
        },
        "dmhsgd": {
            "a9a": {"lrX": 0.01, "lrY": 0.1},
            "ijcnn1": {"lrX": 0.01, "lrY": 0.1},
        },
        "dposg": {
            "a9a": {"lrX": 0.01, "lrY": 0.1},
            "ijcnn1": {"lrX": 0.01, "lrY": 0.01},
        },
        "gtda": {
            "a9a": {"lrX": 0.1, "lrY": 0.1},
            "ijcnn1": {"lrX": 0.1, "lrY": 0.1},
        },
    }

    # If we want to stay in line with the theory:
    lrX = lrDict[f"{args.method}"][f"{args.dataset}"]["lrX"]
    lrY = lrDict[f"{args.method}"][f"{args.dataset}"]["lrY"]

    # ------------------------------------------------
    # Run the trials and save
    for t in range(1, args.trials + 1):
        # Load the starting variable
        try:
            init_X = np.array(
                np.load(f"data/{args.dataset}/initial_X{t}.dat", allow_pickle=True)
            )
        except:
            init_X = np.random.normal(0, 1e-1, (N, X_train.shape[1]))
            init_X.dump(f"data/{args.dataset}/initial_X{t}.dat")

        # init_y = np.ones((N, n_include)) / n_include
        init_y = np.ones(init_X.shape)
        init_y /= init_y.shape[1]
        init_y = np.expand_dims(init_y, axis=1)

        # Save parameters and declare model
        param_dict = {
            "use_proj": False,
            "lrX": lrX,
            "lrY": lrY,
            "mini_batch": args.batch,
            "frequency": args.frequency,
            "l1": 0.0,
            "y_penalty": 1.0,
            "report": args.report,
            "betax": args.betax,
            "betay": args.betay,
            "inner_rounds": args.inner_rounds,
            "mega_batch": args.mega_batch,
        }

        # Declare model
        if args.method == "ours":
            model = DGDAVR(X_train_block, y_train_block, init_X, init_y, mm, param_dict)
        elif args.method == "dmhsgd":
            model = DMHSGD(X_train_block, y_train_block, init_X, init_y, mm, param_dict)
        elif args.method == "dposg":
            model = DPOSG(X_train_block, y_train_block, init_X, init_y, mm, param_dict)
        elif args.method == "gtda":
            model = GTDA(X_train_block, y_train_block, init_X, init_y, mm, param_dict)
        else:
            model = DGDAVR(X_train_block, y_train_block, init_X, init_y, mm, param_dict)

        # Run the model
        model.solve(args.iterations, X_test, y_test)

        # Save values
        save_consensusX[t - 1, :] = np.array(model.ConsensusX)
        save_consensusY[t - 1, :] = np.array(model.ConsensusY)
        save_gradients[t - 1, :] = np.array(model.Gradient)
        save_total[t - 1, :] = np.array(model.TotalOpt)
        save_accuracy[t - 1, :] = np.array(model.TestAcc)

    # ------------------------------------------------
    # Compute mean and standard deviations and save
    mean_consX = np.mean(save_consensusX, axis=0)
    std_consX = np.std(save_consensusX, axis=0)
    mean_consY = np.mean(save_consensusY, axis=0)
    std_consY = np.std(save_consensusY, axis=0)
    mean_grad = np.mean(save_gradients, axis=0)
    std_grad = np.std(save_gradients, axis=0)
    mean_total = np.mean(save_total, axis=0)
    std_total = np.std(save_total, axis=0)
    mean_acc = np.mean(save_accuracy, axis=0)
    std_acc = np.std(save_accuracy, axis=0)

    # Save this information
    try:
        os.mkdir(os.path.join(os.getcwd(), f"results/"))
    except:
        pass
    try:
        os.mkdir(os.path.join(os.getcwd(), f"results/{args.dataset}"))
    except:
        pass
    try:
        os.mkdir(
            os.path.join(os.getcwd(), f"results/{args.dataset}/{args.mm}_{args.size}")
        )
    except:
        pass
    try:
        os.mkdir(
            os.path.join(
                os.getcwd(),
                f"results/{args.dataset}/{args.mm}_{args.size}/{args.method}",
            )
        )
    except:
        pass

    # Save path
    path = os.path.join(
        os.getcwd(), f"results/{args.dataset}/{args.mm}_{args.size}/{args.method}"
    )

    # Save results
    np.savetxt(f"{path}/mean_consensusX.txt", mean_consX, fmt="%.16f")
    np.savetxt(f"{path}/std_consensusX.txt", std_consX, fmt="%.16f")
    np.savetxt(f"{path}/mean_consensusY.txt", mean_consY, fmt="%.16f")
    np.savetxt(f"{path}/std_consensusY.txt", std_consY, fmt="%.16f")
    np.savetxt(f"{path}/mean_gradient.txt", mean_grad, fmt="%.16f")
    np.savetxt(f"{path}/std_gradient.txt", std_grad, fmt="%.16f")
    np.savetxt(f"{path}/mean_total.txt", mean_total, fmt="%.16f")
    np.savetxt(f"{path}/std_total.txt", std_total, fmt="%.16f")
    np.savetxt(f"{path}/mean_accuracy.txt", mean_acc, fmt="%.16f")
    np.savetxt(f"{path}/std_accuracy.txt", std_acc, fmt="%.16f")
