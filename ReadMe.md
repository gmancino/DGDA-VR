# DGDA-VR: Decentralized Gradient Decent Ascent - Variance Reduction method

This document provides instructions on how to reproduce the experimental results from the paper: "[Jointly Improving the Sample and Communication Complexities in Decentralized Minimax Optimization](https://arxiv.org/abs/2307.09421)". Covered in this document are:

- package requirements utilized in the experiments
- instructions on how to reproduce the results from the paper (i.e. hyperparameter settings, etc.)
- a description of the main components in this repository, their use, and how to modify them for new use cases

## Experiment set-up

Experiments are divided into two categories: those that require a GPU (located in the `gpu` folder) and those that do not (located in the `cpu`) folder

The CPU experiments were ran on a 2019 MacBook Pro with a 1.7GHz Quad-Core Intel processor and 16 GB of RAM. The computing environment used is Python 3.8 and the following packages are required:
```
numpy==1.19.1
scikit-learn==0.23.1
```

The GPU experiments were ran on clusters of 8 NVIDIA Tesla V100's (each with 32 GiB HBM) connected by dual 100 Gb EDR Infiniband. The operating system utilized is [CentOS](https://www.centos.org) 7 and all experiments are ran within a conda version 4.9.2 environment. All code is written in Python version 3.7.9, using `PyTorch` version 1.6.0 with CUDA version 10.2; for instructions on how to install PyTorch with CUDA see [here](https://pytorch.org/get-started/previous-versions/). The GCC version of the system utilized is 4.8.5. To perform neighbor communication, `mpi4py` was utilized; see [here](https://mpi4py.readthedocs.io/en/stable/install.html) for instructions on how to install this package. A complete list of packages necessary for completing these experiments is located in the [requirements.txt](requirements.txt) file. Comprehensive installation instructions can be found in [Install.md](Install.md)

## Reproducing the experiments

The `cpu/run_trials.py` script reproduces the robust non-convex linear regression experiments from the paper. The learning rates are set in lines 120-137 and follow the values we used in our experiments. When the code is running properly, you will see the following output (using our method as an example):
```
========================= DGDA-VR TRAINING =========================
[COMMS]: 5000
[GRAPH]: N = 20 | rho = 0.9674
[TIME]: 9.13 sec (0.0018 sec / iteration)
========================= DGDA-VR FINISHED =========================
```

The `gpu/main.py` script reproduces the PL game and robust neural network training experiments from the paper. The learning rates are set using `argparse`


## Structure of the code and adding new methods

The code is organized as follows:
```
cpu/
    data/
    mixing_matrices/
    models/
gpu/
    data/
    mixing_matrices/
    models/
```

#### models
The `models` folder contains the base class which each method inherits from and helper functions utilized by all methods. **Each folder contains a class in the `base.py` script; to implement a new method, simply let the method inherit the base class and modify the `one_step` method accordingly**

For the GPU based experiments, we have the helper function which aids in managing the PyTorch tensor updates:

1. `replace_weights.py`: a custom PyTorch optimizer that simply _replaces_ model parameters from a LIST of PyTorch tensors; since all methods update many variables, this is a straightforward way to minimize conflicts when updating model parameters. The `.step()` method requires two parameters:
    - `weights`: a LIST of PyTorch tensors of length `k` where `k` represents the number of model parameters
    - `device`: the torch device (i.e. GPU) the model is saved on

#### data

Contains the relevant data. To add new datasets, just make sure the `run_trials.py` and `main.py` scripts load it appropriately (lines 58-59 + 87-90 and 43-44 + 63-103, respectively)

#### mixing_matrices
The `mixing_matrices` folder contains `Numpy` arrays of size `N x N` where each `(i,j)` entry corresponds to agent `i`'s weighting of agent `j`'s information

To run experiments with different weight matrices/communication patterns or sizes, save the corresponding mixing matrix in this directory


## Citation

Please cite our paper if you use this code in your work:

```
@inproceedings{zhang2024jointly,
   title={Jointly Improving the Sample and Communication Complexities in Decentralized Stochastic Minimax Optimization},
   author={Zhang, Xuan and Mancino-Ball, Gabriel and Aybat, Necdet Serhat and Xu, Yangyang},
   journal={Proceedings of the AAAI Conference on Artificial Intelligence},
   year={2024}
}
```
