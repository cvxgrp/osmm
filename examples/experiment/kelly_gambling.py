import torch
import numpy as np
import cvxpy as cp
from mosek.fusion import *
import mosek as mosek
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import time

# import sys; sys.path.insert(0, '..')
# from osmm.osmm import OSMM

CPU = torch.device('cpu')
if torch.cuda.is_available():
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
    device = torch.device('cuda')
else:
    torch.set_default_tensor_type(torch.FloatTensor)
    device = CPU
print("device =", device)
np.random.seed(0)#10
np.seterr(all='raise')

N = 1000
n = 1000
pi = np.random.uniform(0, 1, N)
pi = pi / np.sum(pi)
target_mean = np.random.uniform(0.9, 1.1, n)


def generate_random_data():
    R = np.exp(np.random.randn(n, N))
    tmp_mean = np.sum(R * pi, axis=1)
    R = (R.T / tmp_mean * target_mean).T
    W = np.zeros((n + 1, N))
    W[0:n, :] = R
    W[n, :] = pi
    return W


def get_initial_val():
    ini = np.ones(n) / (n - 1)
    return ini


def my_g_cvxpy():
    x_var = cp.Variable(n, nonneg=True)
    g = 0
    constr = [cp.sum(x_var) == 1]
    return x_var, g, constr


def my_f_torch(b_torch=None, W_torch=None):
    r_torch = W_torch[0:n, :]
    pi_torch = W_torch[n, :]
    if b_torch.shape == torch.Size([n]):
        tmp = torch.matmul(r_torch.T, b_torch)
    else:
        tmp = torch.sum(r_torch * b_torch, axis=0)
    objf = -torch.sum(torch.log(tmp) * pi_torch)
    return objf


############################################################
def get_baseline_soln_cvxpy(W, ep=1e-6, compare_MOSEK=False, compare_SCS=False, compare_ECOS=False):
    R = W[0:n, :]
    b_baseline_var = cp.Variable(n, nonneg=True)
    obj_baseline = -cp.log(R.T @ b_baseline_var) @ pi
    constr_baseline = [cp.sum(b_baseline_var) == 1]
    prob_baseline = cp.Problem(cp.Minimize(obj_baseline), constr_baseline)
    print("Start to solve baseline problem by MOSEK")

    prob_baseline_vals = []
    solve_times = []

    if compare_MOSEK:
        print("Start MOSEK")
        t0 = time.time()
        prob_baseline.solve(solver="MOSEK", verbose=True, mosek_params={
            mosek.dparam.intpnt_co_tol_rel_gap: ep,
            mosek.dparam.intpnt_co_tol_mu_red: ep,
            mosek.dparam.intpnt_qo_tol_mu_red: ep,
            mosek.dparam.intpnt_qo_tol_rel_gap: ep,
            mosek.dparam.intpnt_tol_mu_red: ep,
            mosek.dparam.intpnt_tol_rel_gap: ep
        })
        print("  MOSEK + CVXPY time cost ", time.time() - t0)
        solve_times.append(time.time() - t0)
        print("  MOSEK solver time cost", prob_baseline.solver_stats.solve_time)
        print("  Setup time cost", prob_baseline.solver_stats.setup_time)
        print("  Objective value", prob_baseline.value)
        print("  Solver status  " + prob_baseline.status + ".\n")
        prob_baseline_vals.append(prob_baseline.value)

    if compare_SCS:
        for var in prob_baseline.variables():
            var.value = None
        print("Start to solve baseline problem by SCS")
        t0 = time.time()
        prob_baseline.solve(solver="SCS", eps=ep, verbose=True, max_iters=10000)
        print("  SCS + CVXPY time cost ", time.time() - t0)
        solve_times.append(time.time() - t0)
        print("  Objective value", prob_baseline.value)
        print("  Solver status  " + prob_baseline.status + ".\n")
        prob_baseline_vals.append(prob_baseline.value)

    if compare_ECOS:
        for var in prob_baseline.variables():
            var.value = None
        print("Start to solve baseline problem by ECOS")
        t0 = time.time()
        prob_baseline.solve(solver="ECOS", verbose=True, abstol=ep)
        print("  ECOS + CVXPY time cost ", time.time() - t0)
        solve_times.append(time.time() - t0)
        print("  ECOS solver time cost", prob_baseline.solver_stats.solve_time)
        print("  Setup time cost", prob_baseline.solver_stats.setup_time)
        print("  Objective value", prob_baseline.value)
        print("  Solver status  " + prob_baseline.status + ".\n")
        prob_baseline_vals.append(prob_baseline.value)

    return prob_baseline_vals, solve_times


def my_plot_one_result(W, x_best, is_save_fig=False, figname="kelly.pdf"):
    linewidth = 2
    fontsize = 14

    fig = plt.figure(tight_layout=False, figsize=(12, 6))
    gs = gridspec.GridSpec(1, 2)

    b = fig.add_subplot(gs[0, 0])
    b.stem([i for i in range(n)], x_best, markerfmt=' ')
    b.set_ylabel("Investment", fontsize=fontsize)
    b.set_xlabel("Bet Index", fontsize=fontsize)
    b.set_yscale("log")

    payoff_outcomes = np.log(W[0:n, :].T.dot(x_best))
    payoff_mean = np.sum(payoff_outcomes * W[n, :])
    c = fig.add_subplot(gs[0, 1])
    c.hist(payoff_outcomes, bins=40)
    c.set_yscale("log")
    c.axvline(payoff_mean, color='k', linestyle='dashed', linewidth=linewidth)
    c.text(payoff_mean * 1.5, int(N * 0.4), 'Mean: {:.2f}'.format(payoff_mean), fontsize=fontsize)
    print("mean", payoff_mean, np.exp(payoff_mean))
    c.set_ylabel("Number of Samples", fontsize=fontsize)
    c.set_xlabel("Log Return", fontsize=fontsize)
    #####################################################################
    if is_save_fig:
        fig.savefig(figname)