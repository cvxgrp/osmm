import torch
import numpy as np
import cvxpy as cp
from osmm import OSMM

x_var = cp.Variable(2)
v = np.array([-2, 2])


def my_g_cvxpy():
    g = cp.norm(x_var - v)
    return x_var, g, [], []


def my_f_torch(lam_torch, x_torch):  # lam * (exp(x1) - exp(x2)) / (x1 - x2) convex but cannot be expressed by cvxpy
    if torch.abs(x_torch[0] - x_torch[1]) > 1e-10:
        return lam_torch * (torch.exp(x_torch[0]) - torch.exp(x_torch[1])) / (x_torch[0] - x_torch[1])
    else:
        return lam_torch * torch.exp(x_torch[0])


osmm_prob = OSMM(my_f_torch, my_g_cvxpy)
osmm_prob.solve(np.ones(1), np.array([1, 2]))
print("solution = ", x_var.value)