#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    Make the PL data from: https://github.com/TrueNobility303/SPIDER-GDA/blob/main/code/GDA/pl_data_generator.m
"""

import numpy as np
import torch
from scipy.linalg import orth

d = 10
r = 5
l = 1
mu = 1e-9
n = 8 * 1000

UA = orth(np.random.randn(d, d))
UB = orth(np.random.randn(d, d))
UC = np.random.randn(d, d)

D = np.diag(np.concatenate((np.random.uniform(mu, l, r), np.zeros(d - r))))

SigmaA = np.matmul(UA, np.matmul(D, UA.transpose()))
SigmaB = np.matmul(UB, np.matmul(D, UB.transpose()))
SigmaC = 0.1 * np.matmul(UC, UC.transpose())

A = np.random.multivariate_normal(np.zeros(d), SigmaA, n)
B = np.random.multivariate_normal(np.zeros(d), SigmaB, n)
C = np.random.multivariate_normal(np.zeros(d), SigmaC, n)

# Save the data
data = {"P": torch.tensor(A), "Q": torch.tensor(B), "R": torch.tensor(C)}
# torch.save(data, 'data/PL.pt')
