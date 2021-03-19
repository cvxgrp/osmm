import torch
import autograd.numpy as np
import cvxpy as cp
from mosek.fusion import *
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

n_product = 1000
n = n_product + 1
N = 10000
n_w = 2 * n_product
B = np.random.uniform(low=0, high=0.02, size=(n_product, n_product))
A = B.dot(B.T)
cost_bound = 1.0
eta = 0.5

mean_return1 = .05  # 1.25
mean_return2 = -.1  # 0.85
sigma_idyo = .05
sigma_fact = .30  # .2
NFACT = 5  # 5

factors_1 = np.matrix(np.random.uniform(size=(n_w, NFACT)))
Sigma1 = np.diag([sigma_idyo ** 2] * n_w) + factors_1 * np.diag([sigma_fact ** 2] * NFACT) * factors_1.T

factors_2 = np.matrix(np.random.uniform(size=(n_w, NFACT)))
Sigma2 = np.diag([sigma_idyo ** 2] * n_w) + factors_2 * np.diag([sigma_fact ** 2] * NFACT) * factors_2.T
Sigma2 *= 2

mu1 = mean_return1 + np.random.randn(n_w) * mean_return1 / 3.
mu2 = mean_return2 + np.random.randn(n_w) * mean_return2 / 3.

prob1 = .5
prob2 = .5


def generate_random_data():
    # draw batch returns
    def draw_batch_returns(N_DRAWS):
        result = np.vstack([
            np.exp(np.random.multivariate_normal(mu1, Sigma1, size=int(N_DRAWS * prob1))),
            np.exp(np.random.multivariate_normal(mu2, Sigma2, size=int(N_DRAWS * prob2)))
        ])
        np.random.shuffle(result)
        return result

    learning_sample = draw_batch_returns(N)
    return learning_sample.T


def get_initial_val():
    return np.ones(n)


def get_cvxpy_description():
    q_var = cp.Variable(n)
    g = 0
    constr = [cp.square(cp.norm(B.T @ q_var[0:n_product])) <= cost_bound, q_var[0:n-1] >= 0, q_var[n-1] >= 1e-10]
    return q_var, g, constr


def my_objf_torch(w_torch=None, q_torch=None, take_mean=True):
    if take_mean == False:
        print("take_mean must be true")
        return None
    _, batch_size = w_torch.shape
    d_torch = w_torch[0:n_w // 2, :]
    p_torch = w_torch[n_w // 2:n_w, :]
    B_torch = torch.tensor(B, dtype=torch.float)
    phi = torch.square(torch.norm(torch.matmul(B_torch.T, q_torch[0:n_product])))
    profits = torch.sum(p_torch * torch.min(d_torch, q_torch[0:n_product, None]), axis=0) - phi
    objf = q_torch[n - 1] * torch.logsumexp(-profits / q_torch[n - 1] - np.log(batch_size) - np.log(1 - eta), 0)
    return objf


#########################################################################
### baseline and plot
def get_baseline_soln_cvxpy(W, compare_with_all=False):
    print("in baseline solver")
    D = W[0:n_w // 2, :]
    P = W[n_w // 2:n_w, :]
    q_baseline_var = cp.Variable(n_product, nonneg=True)
    a_baseline_var = cp.Variable(1, pos=True)

    u = cp.Variable(1)
    z = cp.Variable(N, nonneg=True)
    profits = cp.Variable(N)

    cost = cp.square(cp.norm(B.T @ q_baseline_var))
    sales = cp.vstack([cp.minimum(D[i, :], q_baseline_var[i]) for i in range(n_product)])  # n_w by N
    revenues = cp.sum(cp.multiply(P, sales), axis=0)  # N

    constr1 = cp.sum(z) / N / (1 - eta) <= a_baseline_var
    constr2 = cp.constraints.exponential.ExpCone(-profits - u, cp.hstack([a_baseline_var for i in range(N)]), z)
    constr3 = profits - revenues + cost[0] <= 0

    baseline_prob = cp.Problem(cp.Minimize(u), [cost <= cost_bound, constr1, constr2, constr3])

    print("Start to solve baseline problem by MOSEK")
    t0 = time.time()
    baseline_prob.solve(solver="MOSEK", verbose=False)
    print("  MOSEK + CVXPY time cost ", time.time() - t0)
    print("  MOSEK solver time cost", baseline_prob.solver_stats.solve_time)
    print("  Setup time cost", baseline_prob.solver_stats.setup_time)
    print("  Objective value", baseline_prob.value)
    print("  Solver status  " + baseline_prob.status + ".\n")
    prob_baseline_val = baseline_prob.value
    mosek_solve_time = baseline_prob.solver_stats.solve_time

    if compare_with_all:
        q_baseline_var.value = None
        print("Start to solve baseline problem by SCS")
        t0 = time.time()
        baseline_prob.solve(solver="SCS", verbose=True)
        print("  SCS + CVXPY time cost ", time.time() - t0)
        print("  Objective value", baseline_prob.value)
        print("  Solver status  " + baseline_prob.status + ".\n")

        q_baseline_var.value = None
        print("Start to solve baseline problem by ECOS")
        t0 = time.time()
        baseline_prob.solve(solver="ECOS", verbose=False)
        print("  ECOS + CVXPY time cost ", time.time() - t0)
        print("  ECOS solver time cost", baseline_prob.solver_stats.solve_time)
        print("  Setup time cost", baseline_prob.solver_stats.setup_time)
        print("  Objective value", baseline_prob.value)
        print("  Solver status  " + baseline_prob.status + ".\n")
    return q_baseline_var.value, prob_baseline_val, mosek_solve_time


def get_baseline_soln_mosek(W):
    D = W[0:n_product, :]
    P = W[n_product:n_w, :]
    M = Model()
    q_baseline_var = M.variable(n_product, Domain.greaterThan(0.))
    a_baseline_var = M.variable(1, Domain.greaterThan(0.))
    u = M.variable(1)
    z = M.variable(N, Domain.greaterThan(0.))
    sales = M.variable([n_product, N], Domain.greaterThan(0.))
    cost = M.variable(1, Domain.greaterThan(0.))
    M.objective(ObjectiveSense.Minimize, u)

    M.constraint(Expr.sub(sales, Matrix.dense(D)), Domain.lessThan(0.))
    M.constraint(Expr.sub(sales, Expr.hstack([q_baseline_var for i in range(N)])), Domain.lessThan(0.))

    M.constraint(Expr.sub(Expr.mul(Expr.sum(z), 1.0 / N / (1 - eta)), a_baseline_var), Domain.lessThan(0.))

    neg_profit = Expr.sub(Expr.repeat(cost, N, 0), Expr.sum(Expr.mulElm(Matrix.dense(P), sales), 0))

    M.constraint(Expr.hstack(z, Expr.repeat(a_baseline_var, N, 0), Expr.sub(neg_profit, Expr.repeat(u, N, 0))),
                 Domain.inPExpCone())
    M.constraint(cost, Domain.lessThan(cost_bound))
    M.constraint(Expr.vstack(0.5, cost, Expr.mul(Matrix.dense(B.T), q_baseline_var)), Domain.inRotatedQCone())
    M.solve()
    return np.concatenate([q_baseline_var.level(), a_baseline_var.level()]), u.level()


def my_plot_one_result(W, x_best, is_save_fig=False, figname="newsvendor.pdf"):
    linewidth = 2
    fontsize = 14

    fig = plt.figure(tight_layout=True, figsize=(12, 6))
    gs = gridspec.GridSpec(1, 2)

    a = fig.add_subplot(gs[0, 0])  # gs[0, :]
    a.stem([i for i in range(1, n - 1)], x_best[0:n - 2], markerfmt=" ", label="Solution")
    print("t and a", x_best[n - 2:n])
    a.set_xlabel("Product", fontsize=fontsize)
    a.set_ylabel("Quantity of Product", fontsize=fontsize)
    a.set_yscale("log")
    a.set_ylim([1e-2, 1e2])

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
    cost = x_best[0:n_product].dot(A.dot(x_best[0:n_product]))
    profits = np.sum(P.T * (np.minimum(D.T, x_best[0:n_product])), axis=1) - cost
    weights = np.ones_like(profits) / len(profits)
    b.hist(profits, weights=weights, bins=50)
    b.axvline(profits.mean(), color='k', linestyle='dashed', linewidth=linewidth)
    b.text(profits.mean() * 1.6, 0.1, 'Mean: {:.2f}'.format(profits.mean()), fontsize=fontsize)
    # b.set_title(r"Histogram of Profit ($\eta$={:.2f})".format(eta), fontsize=fontsize)
    b.set_xlabel("Profit", fontsize=fontsize)
    b.set_ylabel(r"Empirical Density ($\eta$={:.2f})".format(eta), fontsize=fontsize)
    b.set_yscale("log")
    print("cost = ", cp.quad_form(x_best[0:n_product], A).value)

    #####################################################################
    if is_save_fig:
        fig.savefig(figname)
