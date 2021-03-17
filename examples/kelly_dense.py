import autograd.numpy as np
import cvxpy as cp

import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import time

from plot_convergence_1trial_1alg import get_best_x

N = 1000000
n_stocks = 100#500
n_w = n_stocks * 2 + 1
n = n_w

mean_return1_stock = 0.3  # 1.0
mu1_stock = np.concatenate([mean_return1_stock + np.random.randn(n_stocks) * mean_return1_stock / 3.])
prob1_stock = 1
prob2_stock = 0

NFACT_stock = 5
factors_1_stock = np.matrix(np.random.uniform(size=(n_stocks, NFACT_stock)))
sigma_idyo_stock = 0.15  # .3
sigma_fact_stock = 0.15  # .30
Sigma1_stock = np.diag([sigma_idyo_stock ** 2] * n_stocks) + factors_1_stock * np.diag(
        [sigma_fact_stock ** 2] * NFACT_stock) * factors_1_stock.T


def generate_stock_option_return():
    stock_option_return = np.ones((N, n_w))

    stock_price = generate_stock_price()  # np.exp(np.random.randn(n_stocks, N))

    exercise_price = np.mean(stock_price, axis=1)
    premium = 0.045 * exercise_price
    share_price_minus_exercise_price = stock_price - exercise_price.reshape((n_stocks, 1))

    stock_option_return[:, 0:n_stocks] = stock_price.T
    stock_option_return[:, n_stocks:n_w - 1] = np.maximum(0, share_price_minus_exercise_price.T) / premium
    return stock_option_return.T


def generate_stock_price():
    stock_prices = np.vstack([
            np.exp(np.random.multivariate_normal(mu1_stock, Sigma1_stock, size=int(N))),
            # np.exp(np.random.multivariate_normal(mu2_stock, Sigma2_stock, size=int(N * prob2_stock)))
        ])
    # np.random.shuffle(stock_prices)
    return stock_prices.T  # n_stock by N


def get_initial_val_kelly():
    ini = np.ones(n) / (n-1)
    return ini


def get_baseline_soln_kelly(R, compare_with_all=False):
    b_baseline_var = cp.Variable(n, nonneg=True)
    obj_baseline = -cp.sum(cp.log(R.T @ b_baseline_var)) / N
    b_baseline_var.value = np.ones(n) / n
    print("ini", obj_baseline.value)
    constr_baseline = [cp.sum(b_baseline_var) == 1]
    prob_baseline = cp.Problem(cp.Minimize(obj_baseline), constr_baseline)
    print("Start to solve baseline problem by MOSEK")
    t0 = time.time()
    prob_baseline.solve(solver="MOSEK")
    print("  MOSEK + CVXPY total time cost ", time.time() - t0)
    print("  MOSEK solver time cost", prob_baseline.solver_stats.solve_time)
    print("  Setup time cost", prob_baseline.solver_stats.setup_time)
    print("  Objective value", prob_baseline.value)
    print("  Solver status  " + prob_baseline.status + ".\n")
    prob_baseline_val = prob_baseline.value
    mosek_solve_time = prob_baseline.solver_stats.solve_time

    if compare_with_all:
        b_baseline_var.value = None
        print("Start to solve baseline problem by SCS")
        t0 = time.time()
        prob_baseline.solve(solver="SCS", verbose=True)  # get the solve time info from verbose=True
        print("  SCS + CVXPY total time cost ", time.time() - t0)
        print("  Objective value", prob_baseline.value)
        print("  Solver status  " + prob_baseline.status + ".\n")

        b_baseline_var.value = None
        print("Start to solve baseline problem by ECOS")
        t0 = time.time()
        prob_baseline.solve(solver="ECOS", verbose=False)
        print("  ECOS + CVXPY time cost ", time.time() - t0)
        print("  ECOS solver time cost", prob_baseline.solver_stats.solve_time)
        print("  Setup time cost", prob_baseline.solver_stats.setup_time)
        print("  Objective value", prob_baseline.value)
        print("  Solver status  " + prob_baseline.status + ".\n")

    return b_baseline_var.value, prob_baseline_val, mosek_solve_time


def get_cvxpy_description_kelly():
    b_var = cp.Variable(n, nonneg=True)
    g = 0 * cp.sum(b_var)
    constr = [cp.sum(b_var) == 1, b_var >= 0]
    return b_var, g, constr


def my_objf_torch_kelly(r_torch=None, b_torch=None, take_mean=True):
    if b_torch.shape == torch.Size([n]):
        tmp = torch.matmul(r_torch.T, b_torch)
    else:
        tmp = torch.sum(r_torch * b_torch, axis=0)
    if take_mean:
        objf = -torch.mean(torch.log(tmp))
    else:
        objf = -torch.log(tmp)
    return objf


#########################################################################
def my_plot_kelly_one_result(W, xs, objfs, is_save_fig=False, figname="kelly.pdf"):
    linewidth = 2
    fontsize = 14

    num_trials, _, num_iters = objfs.shape
    _, _, _, saved_num_iters = xs.shape
    if num_iters == saved_num_iters:
        x_best, _ = get_best_x(objfs, xs, trial_idx=0)
    else:
        x_best, _ = get_best_x(objfs[:, :, num_iters - saved_num_iters:num_iters], xs, trial_idx=0)

    fig = plt.figure(tight_layout=False, figsize=(12, 6))
    gs = gridspec.GridSpec(1, 2)

    b = fig.add_subplot(gs[0, 0])
    b.stem([i for i in range(n)], x_best, markerfmt=' ')
    b.set_ylabel("Investment", fontsize=fontsize)
    b.set_xlabel("Bet Index", fontsize=fontsize)
    b.set_yscale("log")
    b.grid()

    payoff_outcomes = np.log(W.T.dot(x_best))
    c = fig.add_subplot(gs[0, 1])
    c.hist(payoff_outcomes, bins=40)
    c.set_yscale("log")
    c.axvline(payoff_outcomes.mean(), color='k', linestyle='dashed', linewidth=linewidth)
    c.text(payoff_outcomes.mean() * 1.5, int(N * 0.4), 'Mean: {:.2f}'.format(payoff_outcomes.mean()), fontsize=fontsize)
    print("mean", payoff_outcomes.mean())
    c.set_ylabel("Number of Samples", fontsize=fontsize)
    c.set_xlabel("Log Return", fontsize=fontsize)
    c.grid()
    #####################################################################
    if is_save_fig:
        fig.savefig(figname)