import torch
import numpy as np
import cvxpy as cp
from mosek.fusion import *
import mosek as mosek
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import time
from scipy.stats import norm

CPU = torch.device('cpu')
if torch.cuda.is_available():
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
    device = torch.device('cuda')
else:
    torch.set_default_tensor_type(torch.FloatTensor)
    device = CPU
print("device =", device)
np.random.seed(0)
np.seterr(all='raise')

eta = 0.8
N = 1000000
m = 100
n = m * 3 + 1
n_w = m * 3

sigma = 1.0 / np.sqrt(2)
NFACT = 5
factors_stock = np.random.randn(m, NFACT)
Sigma = (factors_stock.dot(factors_stock.T) / NFACT + np.eye(m)) * (sigma ** 2)
mu = 0.03 * np.sqrt(np.diag(Sigma)) - np.diag(Sigma) / 2

call_strike_prices = np.exp(norm.ppf(0.8) * np.sqrt(np.diag(Sigma)) + mu)
put_strike_prices = np.exp(norm.ppf(0.2) * np.sqrt(np.diag(Sigma)) + mu)

r = np.zeros(m)
d1_call = np.log(1.0 / call_strike_prices) / np.sqrt(np.diag(Sigma)) + 0.5 * np.sqrt(np.diag(Sigma)) + r / np.sqrt(
    np.diag(Sigma))
d2_call = np.log(1.0 / call_strike_prices) / np.sqrt(np.diag(Sigma)) - 0.5 * np.sqrt(np.diag(Sigma)) + r / np.sqrt(
    np.diag(Sigma))
call_option_prices = norm.cdf(d1_call) - call_strike_prices * norm.cdf(d2_call) * np.exp(-r)

d1_put = np.log(1.0 / put_strike_prices) / np.sqrt(np.diag(Sigma)) + 0.5 * np.sqrt(np.diag(Sigma)) + r / np.sqrt(
    np.diag(Sigma))
d2_put = np.log(1.0 / put_strike_prices) / np.sqrt(np.diag(Sigma)) - 0.5 * np.sqrt(np.diag(Sigma)) + r / np.sqrt(
    np.diag(Sigma))
put_option_prices = put_strike_prices * norm.cdf(-d2_put) * np.exp(-r) - norm.cdf(-d1_put)


def generate_random_data():
    total_return = np.zeros((N, n_w))
    stock_price = np.exp(np.random.multivariate_normal(mu, Sigma, size=int(N))) #N by m
    total_return[:, 0:m] = stock_price
    stock_price_minus_call_strike_price = (stock_price.T - call_strike_prices.reshape((m, 1))).T
    total_return[:, m:2 * m] = np.maximum(0, stock_price_minus_call_strike_price) / call_option_prices
    put_strike_price_minus_stock_price = (put_strike_prices.reshape((m, 1)) - stock_price.T).T
    total_return[:, 2 * m:3 * m] = np.maximum(0, put_strike_price_minus_stock_price) / put_option_prices
    return total_return.T


def get_initial_val():
    ini = np.ones(n) / (n - 1)
    #     ini = np.zeros(n)
    return ini


def my_g_cvxpy():
    b_var = cp.Variable(n)
    g = 0
    constr = [cp.sum(b_var[1:n]) == 1, b_var[1:n] >= -0.1,
              cp.sum(cp.abs(b_var[1:n])) <= 1.6]#, b_var[m + 1:n] == 0]
    return b_var, g, constr


def my_f_torch(b_torch=None, W_torch=None):
    if b_torch.shape == torch.Size([n]):
        tmp = torch.matmul(W_torch.T, b_torch[1:n])
    else:
        tmp = torch.sum(W_torch * b_torch[1:n, :], axis=0)
    objf = 1.0 / (1.0 - eta) * torch.mean(torch.relu(-tmp + b_torch[0])) - b_torch[0]
    return objf


def my_plot_one_result(W, x_best, is_save_fig=False, figname="cvar.pdf"):
    linewidth = 2
    fontsize = 10

    fig = plt.figure(tight_layout=True, figsize=(5, 5))
    gs = gridspec.GridSpec(2, 1)

    a = fig.add_subplot(gs[0, 0])
    a.stem([i for i in range(1, n)], x_best[1:n], markerfmt=' ', label="Long", linefmt="blue")
    a.stem([i for i in range(1, n)], -x_best[1:n], markerfmt=' ', label="Short", linefmt="green")
    a.set_ylabel("Investment ($\eta$={:.2f})".format(eta), fontsize=fontsize)
    a.set_xlabel("Asset", fontsize=fontsize)
    a.set_yscale("log")
    a.set_ylim([1e-4, 1e-1])
    a.grid()
    plt.legend(fontsize=fontsize)

    ###################################################################
    payoff_outcomes = W.T.dot(x_best[1:n])
    c = fig.add_subplot(gs[1, 0])
    counts, bin_edges = np.histogram(payoff_outcomes, bins=200, density=True)
    cdf = np.cumsum(counts)
    c.plot(bin_edges[1:], cdf / cdf[-1])
    c.set_ylabel(r"Empirical CDF ($\eta$={:.2f})".format(eta), fontsize=fontsize)
    c.set_xlabel("Total return", fontsize=fontsize)
    #     c.set_xscale("log")
    #     c.set_yscale("log")
    c.axvline(payoff_outcomes.mean(), color='k', linestyle='dashed', linewidth=linewidth)
    c.text(payoff_outcomes.mean() * 1.01, 0.5, 'Mean: {:.3f}'.format(payoff_outcomes.mean()), fontsize=fontsize)
    print("mean", payoff_outcomes.mean())
    c.grid()

    ####################################################################
    if is_save_fig:
        fig.savefig(figname)


b_baseline_var = cp.Variable(n)  # (beta, b)


def get_baseline_soln_cvxpy(R, ep=1e-6, compare_MOSEK=False, compare_SCS=False, compare_ECOS=False):
    cvar_baseline = b_baseline_var[0] + 1.0 / (1.0 - eta) * cp.sum(
        cp.pos(-R.T @ b_baseline_var[1:n] - b_baseline_var[0])) / N
    obj_baseline = cvar_baseline
    constr_baseline = [b_baseline_var[1:n] >= -0.1, cp.sum(b_baseline_var[1:n]) == 1,
                       cp.sum(cp.abs(b_baseline_var[1:n])) <= 1.6]
    prob_baseline = cp.Problem(cp.Minimize(obj_baseline), constr_baseline)

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