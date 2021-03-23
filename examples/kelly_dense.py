import torch
import numpy as np
import cvxpy as cp
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
np.random.seed(1)
np.seterr(all='raise')

N = 1000000
n_stocks = 500
n_w = n_stocks * 2 + 1
n = n_w

mean_return_stock = 0.3
mu_stock = np.concatenate([mean_return_stock + np.random.randn(n_stocks) * mean_return_stock / 3.])

NFACT_stock = 5
U_stock = np.matrix(np.random.uniform(size=(n_stocks, NFACT_stock)))
sigma_stock = 0.15
Sigma_stock = np.diag([sigma_stock ** 2] * n_stocks) + U_stock * np.diag([sigma_stock ** 2] * NFACT_stock) * U_stock.T


def generate_stock_price(N_M, take_mean=False):
    stock_prices = np.vstack([
            np.exp(np.random.multivariate_normal(mu_stock, Sigma_stock, size=int(N_M)))
    ])
    if take_mean:
        return np.mean(stock_prices.T, axis=1)
    return stock_prices.T  # n_stock by N_M


strike_price = generate_stock_price(N * 10, take_mean=True)


def generate_random_data():
    stock_option_return = np.ones((N, n_w))
    stock_price = generate_stock_price(N)
    premium = 0.045 * strike_price
    stock_price_minus_strike_price = stock_price - strike_price.reshape((n_stocks, 1))
    stock_option_return[:, 0:n_stocks] = stock_price.T
    stock_option_return[:, n_stocks:n_w - 1] = np.maximum(0, stock_price_minus_strike_price.T) / premium
    return stock_option_return.T


def get_initial_val():
    ini = np.ones(n) / (n-1)
    return ini


def get_cvxpy_description():
    b_var = cp.Variable(n, nonneg=True)
    g = 0
    constr = [cp.sum(b_var) == 1]
    return b_var, g, constr


def my_objf_torch(r_torch=None, b_torch=None, take_mean=True):
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
def get_baseline_soln_cvxpy(R, compare_with_all=False):
    b_baseline_var = cp.Variable(n, nonneg=True)
    obj_baseline = -cp.sum(cp.log(R.T @ b_baseline_var)) / N
    constr_baseline = [cp.sum(b_baseline_var) == 1]
    prob_baseline = cp.Problem(cp.Minimize(obj_baseline), constr_baseline)
    b_baseline_var.value = np.ones(n) / n
    print("Initial obj value", obj_baseline.value)
    print("Start to solve baseline problem by MOSEK")
    t0 = time.time()
    ep = 1e-6
    prob_baseline.solve(solver="MOSEK", verbose=True, mosek_params={
        # mosek.dparam.intpnt_co_tol_pfeas: ep,
        # mosek.dparam.intpnt_co_tol_dfeas: ep,
        mosek.dparam.intpnt_co_tol_rel_gap: ep,
        # mosek.dparam.intpnt_co_tol_infeas: ep,
        mosek.dparam.intpnt_co_tol_mu_red: ep,
        # mosek.dparam.intpnt_qo_tol_dfeas: ep,
        # mosek.dparam.intpnt_qo_tol_infeas: ep,
        mosek.dparam.intpnt_qo_tol_mu_red: ep,
        # mosek.dparam.intpnt_qo_tol_pfeas: ep,
        mosek.dparam.intpnt_qo_tol_rel_gap: ep,
        # mosek.dparam.intpnt_tol_dfeas: ep,
        # mosek.dparam.intpnt_tol_infeas: ep,
        mosek.dparam.intpnt_tol_mu_red: ep,
        # mosek.dparam.intpnt_tol_pfeas: ep,
        mosek.dparam.intpnt_tol_rel_gap: ep
    })
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
        prob_baseline.solve(solver="SCS", eps=ep, verbose=True)  # get the solve time info from verbose=True
        print("  SCS + CVXPY total time cost ", time.time() - t0)
        print("  Objective value", prob_baseline.value)
        print("  Solver status  " + prob_baseline.status + ".\n")

        b_baseline_var.value = None
        print("Start to solve baseline problem by ECOS")
        t0 = time.time()
        prob_baseline.solve(solver="ECOS", verbose=True, abstol=ep)#, reltol=ep, feastol=ep)
        print("  ECOS + CVXPY time cost ", time.time() - t0)
        print("  ECOS solver time cost", prob_baseline.solver_stats.solve_time)
        print("  Setup time cost", prob_baseline.solver_stats.setup_time)
        print("  Objective value", prob_baseline.value)
        print("  Solver status  " + prob_baseline.status + ".\n")

    return b_baseline_var.value, prob_baseline_val, mosek_solve_time


def my_plot_one_result(W, x_best, is_save_fig=False, figname="kelly.pdf"):
    linewidth = 2
    fontsize = 16

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