import torch
import numpy as np
import cvxpy as cp
from osmm import OSMM
from osmm import AlgMode
x_var = cp.Variable((2, 2))
v = np.eye(2)


def my_g_cvxpy():
    g = cp.norm(x_var - v, 'fro')
    return x_var, g, []


def my_f_torch(lam_torch, x_torch):  # lam * (exp(x1) - exp(x2)) / (x1 - x2) convex but cannot be expressed by cvxpy
    if torch.abs(x_torch[0, 0] - x_torch[0, 1]) > 1e-10:
        return lam_torch * (torch.exp(x_torch[0, 0]) - torch.exp(x_torch[0, 1])) / (x_torch[0, 0] - x_torch[0, 1])
    else:
        return lam_torch * torch.exp(x_torch[0, 0])


osmm_prob = OSMM(my_f_torch, my_g_cvxpy)
result = osmm_prob.solve(1, np.eye(2), alg_mode=AlgMode.Bundle)
print("result = ", result)
print("solution = ", x_var.value)
print(osmm_prob.method_results["iters_taken"])


import time as time


m = 1000
d = 60
X = np.random.randn(d, m)
S = np.cov(X)
mask = np.ones((d, d)) - np.eye(d)
lam = 1.0

Theta_var = cp.Variable((d, d), PSD=True)


def my_g_cvxpy():
    g = cp.trace(S @ Theta_var) + lam * cp.sum(cp.abs(cp.multiply(Theta_var, mask)))
    return Theta_var, g, []


def my_f_torch(w_torch, theta_torch):
    return -torch.log(torch.det(theta_torch))


t0 = time.time()
init_val = 2 * np.eye(d)
osmm_prob = OSMM(my_f_torch, my_g_cvxpy)
osmm_prob.solve(None, init_val, solver='MOSEK')
print("solve time", time.time() - t0)

t1 = time.time()
Theta_var.value = 2 * np.eye(d)
cvx_prob = cp.Problem(cp.Minimize(cp.trace(S @ Theta_var) - cp.log_det(Theta_var)
                                  + lam * cp.sum(cp.abs(cp.multiply(Theta_var, mask)))))
cvx_prob.solve()
print("solve time", time.time() - t1)




