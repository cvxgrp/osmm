import torch
import numpy as np
import cvxpy as cp
from mosek.fusion import *
import mosek as mosek
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import time

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

eta = 0.1
N = 10000
n_stocks = 1249
n_w = n_stocks * 2 + 1
n = n_w + 1
mean_return1_stock = 0.3
mu1_stock = np.concatenate([mean_return1_stock + np.random.randn(n_stocks) * mean_return1_stock / 3.])
NFACT_stock = 5
factors_1_stock = np.matrix(np.random.uniform(size=(n_stocks, NFACT_stock)))
sigma_idyo_stock = 0.15  # .3
sigma_fact_stock = 0.15  # .30
Sigma1_stock = np.diag([sigma_idyo_stock ** 2] * n_stocks) \
               + factors_1_stock * np.diag([sigma_fact_stock ** 2] * NFACT_stock) * factors_1_stock.T


def generate_stock_price(N_M):
    stock_prices = np.exp(np.random.multivariate_normal(mu1_stock, Sigma1_stock, size=int(N_M)))
    return stock_prices.T  # n_stock by N


stock_price_for_strike_price = generate_stock_price(N * 10)
strike_price = np.mean(stock_price_for_strike_price, axis=1)
premium = 0.045 * strike_price


def generate_random_data():
    stock_option_return = np.ones((N, n_w))
    stock_price = generate_stock_price(N)
    stock_price_minus_strike_price = stock_price - strike_price.reshape((n_stocks, 1))
    stock_option_return[:, 0:n_stocks] = stock_price.T
    stock_option_return[:, n_stocks:n_w - 1] = np.maximum(0, stock_price_minus_strike_price.T) / premium
    return stock_option_return.T


def get_initial_val():
    ini = np.ones(n) / (n - 1)
    return ini


def my_g_cvxpy():
    b_var = cp.Variable(n)
    g = 0
    constr = [cp.sum(b_var[1:n]) == 1, b_var[1:n] >= 0, b_var[0] >= 1e-8]
    return b_var, g, constr


def my_f_torch(r_torch=None, b_torch=None, take_mean=True):
    if b_torch.shape == torch.Size([n]):
        tmp = torch.matmul(r_torch.T, b_torch[1:n])
    else:
        tmp = torch.sum(r_torch * b_torch[1:n, :], axis=0)
    if take_mean:
        objf = 1.0 / (1.0 - eta) * torch.mean(torch.relu(-tmp + b_torch[0])) - b_torch[0]
    else:
        objf = 1.0 / (1.0 - eta) * torch.relu(-tmp + b_torch[0]) - b_torch[0]
    return objf


#########################################################################
### baseline and plot
def get_baseline_soln_cvxpy(R, ep=1e-6, compare_MOSEK=False, compare_SCS=False, compare_ECOS=False):
    b_baseline_var = cp.Variable(n)  # (beta, b)
    cvar_baseline = b_baseline_var[0] + 1.0 / (1.0 - eta) * cp.sum(
        cp.pos(-R.T @ b_baseline_var[1:n] - b_baseline_var[0])) / N
    obj_baseline = cvar_baseline
    constr_baseline = [b_baseline_var[1:n] >= 0, cp.sum(b_baseline_var[1:n]) == 1, b_baseline_var[0] <= -1e-8]
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


def get_baseline_soln_mosek(R):
    M = Model()
    b_baseline_var = M.variable(n_w, Domain.greaterThan(0.))
    a_baseline_var = M.variable(1, Domain.greaterThan(0.))
    z = M.variable(N, Domain.greaterThan(0.))
    objf = Expr.sub(Expr.mul(Expr.sum(z), 1.0 / N / (1 - eta)), a_baseline_var)
    M.objective(ObjectiveSense.Minimize, objf)
    M.constraint(Expr.sum(b_baseline_var), Domain.equalsTo(1))
    M.constraint(Expr.sub(Expr.add(Expr.mul(Matrix.dense(-R.T), b_baseline_var), Expr.repeat(a_baseline_var, N, 0)), z),
                 Domain.lessThan(0.))
    M.solve()
    return np.concatenate([a_baseline_var.level(), b_baseline_var.level()]), np.sum(z.level()) / N / (
                1 - eta) - a_baseline_var.level()


def my_plot_one_result(W, x_best, is_save_fig=False, figname="cvar.pdf"):
    linewidth = 2
    fontsize = 14

    fig = plt.figure(tight_layout=True, figsize=(12, 6))
    gs = gridspec.GridSpec(1, 2)

    a = fig.add_subplot(gs[0, 0])
    a.stem([i for i in range(1, n)], x_best[1:n], markerfmt=' ')
    a.set_ylabel("Investment ($\eta$={:.2f})".format(eta), fontsize=fontsize)
    a.set_xlabel("Asset", fontsize=fontsize)
    a.set_yscale("log")
    a.grid()

    ###################################################################
    payoff_outcomes = W.T.dot(x_best[1:n])
    c = fig.add_subplot(gs[0, 1])
    counts, bin_edges = np.histogram(payoff_outcomes, bins=200, density=True)
    cdf = np.cumsum(counts)
    c.plot(bin_edges[1:], cdf / cdf[-1])
    c.set_ylabel(r"Empirical CDF ($\eta$={:.2f})".format(eta), fontsize=fontsize)
    c.set_xlabel("Payoff", fontsize=fontsize)
    c.set_xscale("log")
    c.set_yscale("log")
    c.axvline(payoff_outcomes.mean(), color='k', linestyle='dashed', linewidth=linewidth)
    c.text(payoff_outcomes.mean() * 1.1, 0.5, 'Mean: {:.2f}'.format(payoff_outcomes.mean()), fontsize=fontsize)
    print("mean", payoff_outcomes.mean())
    c.grid()

    ####################################################################
    if is_save_fig:
        fig.savefig(figname)