import torch
import autograd.numpy as np
import cvxpy as cp
from mosek.fusion import *
import mosek as mosek
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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

n_product = 500
n = n_product + 1
N = 1000
n_w = 2 * n_product
prod_linear = np.random.uniform(low=0.2, high=0.9, size=(n_product))
prod_linear_2 = 0.5 * prod_linear
prod_change_pnts = np.random.uniform(low=0.01, high=0.03, size=(n_product))
prod_amount_bounds = 5 * prod_change_pnts
prod_linear_torch = torch.tensor(prod_linear, dtype=torch.float)
prod_linear_2_torch = torch.tensor(prod_linear_2, dtype=torch.float)
prod_change_pnts_torch = torch.tensor(prod_change_pnts, dtype=torch.float)


cost_bound = 1.0
eta = 0.9

mean_return1 = .2
sigma_idyo = .05
sigma_fact = .3
NFACT = 5

factors_1 = np.random.randn(n_w, NFACT)
Sigma1 = 0.1 * factors_1.dot(factors_1.T)
mu1 = np.random.uniform(low=-mean_return1, high=0, size=(n_w))


def generate_random_data():
    # draw batch returns
    def draw_batch_returns(N_DRAWS):
        result = np.vstack([
            np.exp(np.random.multivariate_normal(mu1, Sigma1, size=int(N_DRAWS))),
        ])
        np.random.shuffle(result)
        return result

    learning_sample = draw_batch_returns(N)
    return learning_sample.T


init_val = np.ones(n) * 0.001
g_var = cp.Variable(n, nonneg=True)
g_obj = 0
# production_cost = cp.square(cp.norm(B.T @ g_var[0:n_product])) + prod_linear.T @ g_var[0:n_product]
production_cost = prod_linear.T @ g_var[0:n_product] + prod_linear_2.T @ cp.pos(g_var[0:n_product] - prod_change_pnts)
g_constr = [production_cost <= cost_bound, g_var[0:n - 1] <= prod_amount_bounds, g_var[n - 1] >= 1e-10]


def my_f_torch(q_torch=None, W_torch=None):
    _, batch_size = W_torch.shape
    d_torch = W_torch[0:n_w // 2, :]
    p_torch = W_torch[n_w // 2:n_w, :]
    phi = torch.matmul(prod_linear_torch.T, q_torch[0:n_product]) + torch.matmul(prod_linear_2_torch.T, torch.relu(q_torch[0:n_product] - prod_change_pnts_torch))
#     phi = torch.square(torch.norm(torch.matmul(B_torch.T, q_torch[0:n_product]))) \
#     + torch.matmul(torch.tensor(prod_linear, dtype=torch.float).T, q_torch[0:n_product])
    profits = torch.sum(p_torch * torch.min(d_torch, q_torch[0:n_product, None]), axis=0) - phi
    objf = q_torch[n - 1] * torch.logsumexp(-profits / q_torch[n - 1] - np.log(batch_size) - np.log(1 - eta), 0)
    return objf


#########################################################################
### baseline and plot
def get_baseline_soln_cvxpy(W, ep=1e-6, compare_MOSEK=False, compare_SCS=False, compare_ECOS=False):
    print("in baseline solver")
    D = W[0:n_w // 2, :]
    P = W[n_w // 2:n_w, :]
    q_baseline_var = cp.Variable(n_product, nonneg=True)
    a_baseline_var = cp.Variable(1, pos=True)

    u = cp.Variable(1)
    z = cp.Variable(N, nonneg=True)
    profits = cp.Variable(N)

    production_cost = prod_linear.T @ q_baseline_var + prod_linear_2.T @ cp.pos(q_baseline_var - prod_change_pnts)
    sales = cp.vstack([cp.minimum(D[i, :], q_baseline_var[i]) for i in range(n_product)])  # n_w by N
    revenues = cp.sum(cp.multiply(P, sales), axis=0)  # N

    constr1 = cp.sum(z) / N / (1 - eta) <= a_baseline_var
    constr2 = cp.constraints.exponential.ExpCone(-profits - u, cp.hstack([a_baseline_var for i in range(N)]), z)
    constr3 = profits - revenues + production_cost <= 0

    prob_baseline = cp.Problem(cp.Minimize(u), [production_cost <= cost_bound, q_baseline_var <= prod_amount_bounds,
                                                constr1, constr2, constr3])

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


# def get_baseline_soln_mosek(W):
#     D = W[0:n_product, :]
#     P = W[n_product:n_w, :]
#     M = Model()
#     q_baseline_var = M.variable(n_product, Domain.greaterThan(0.))
#     a_baseline_var = M.variable(1, Domain.greaterThan(0.))
#     u = M.variable(1)
#     z = M.variable(N, Domain.greaterThan(0.))
#     sales = M.variable([n_product, N], Domain.greaterThan(0.))
#     cost = M.variable(1, Domain.greaterThan(0.))
#     M.objective(ObjectiveSense.Minimize, u)
#
#     M.constraint(Expr.sub(sales, Matrix.dense(D)), Domain.lessThan(0.))
#     M.constraint(Expr.sub(sales, Expr.hstack([q_baseline_var for i in range(N)])), Domain.lessThan(0.))
#
#     M.constraint(Expr.sub(Expr.mul(Expr.sum(z), 1.0 / N / (1 - eta)), a_baseline_var), Domain.lessThan(0.))
#
#     neg_profit = Expr.sub(Expr.repeat(cost, N, 0), Expr.sum(Expr.mulElm(Matrix.dense(P), sales), 0))
#
#     M.constraint(Expr.hstack(z, Expr.repeat(a_baseline_var, N, 0), Expr.sub(neg_profit, Expr.repeat(u, N, 0))),
#                  Domain.inPExpCone())
#     M.constraint(cost, Domain.lessThan(cost_bound))
#     M.constraint(Expr.vstack(0.5, cost, Expr.mul(Matrix.dense(B.T), q_baseline_var)), Domain.inRotatedQCone())
#     M.solve()
#     return u.level()


def my_plot_one_result(W, x_best, is_save_fig=False, figname="newsvendor.pdf"):
    linewidth = 2
    fontsize = 14

    fig = plt.figure(tight_layout=True, figsize=(12, 6))
    gs = gridspec.GridSpec(1, 2)

    a = fig.add_subplot(gs[0, 0])  # gs[0, :]
    a.stem([i for i in range(1, n - 1)], x_best[0:n - 2], markerfmt=" ", label="Solution")
    print("a", x_best[-1])
    a.set_xlabel("Product", fontsize=fontsize)
    a.set_ylabel("Quantity of Product", fontsize=fontsize)
    a.set_yscale("log")
#     a.set_ylim([1e-2, 1e2])

    labels = []
    labels.append((mpatches.Patch(color="blue"), "Solution"))
    labels.append((mpatches.Patch(color="orange"), "Demand"))
    a.plot([i for i in range(1, n_product + 1)], np.mean(W[0:n_w // 2, :], axis=1))
    a.fill_between([i for i in range(1, n_product + 1)], np.min(W[0:n_product, :], axis=1),
                   np.max(W[0:n_product, :], axis=1), alpha=0.3, color='orange')
    plt.legend(*zip(*labels), loc=2)

    ###################################################################
    b = fig.add_subplot(gs[0, 1])
    D = W[0:n_w // 2, :]
    P = W[n_w // 2:n_w, :]
    cost = x_best[0:n_product].T.dot(prod_linear) + np.maximum(x_best[0:n_product] - prod_change_pnts, 0).T.dot(prod_linear_2)
#     cost = x_best[0:n_product].dot(A.dot(x_best[0:n_product]))
    profits = np.sum(P.T * (np.minimum(D.T, x_best[0:n_product])), axis=1) - cost
    weights = np.ones_like(profits) / len(profits)
    b.hist(profits, weights=weights, bins=50)
    b.axvline(profits.mean(), color='k', linestyle='dashed', linewidth=linewidth)
    b.text(profits.mean(), 0.1, 'Mean: {:.2f}'.format(profits.mean()), fontsize=fontsize)
    # b.set_title(r"Histogram of Profit ($\eta$={:.2f})".format(eta), fontsize=fontsize)
    b.set_xlabel("Profit", fontsize=fontsize)
    b.set_ylabel(r"Empirical Density ($\eta$={:.2f})".format(eta), fontsize=fontsize)
    b.set_yscale("log")
    print("cost = ", cost)

    #####################################################################
    if is_save_fig:
        fig.savefig(figname)