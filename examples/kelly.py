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
np.random.seed(10)
np.seterr(all='raise')

N = 10000
num_horses = 10
n = 1 + num_horses + num_horses * (num_horses - 1) // 2 + num_horses * (num_horses - 1) * (num_horses - 2) // 6
n_w = n
mu = [1, 1, 1.5, 2.0, 3.0, 3.0, 13.5, 3.8, 4.0, 4.0]
sigma = [0.5, 0.8, 0.8, 0.5, 0.9, 1.0, 1.0, 1.1, 0.9, 1.2]


def generate_unit_payoff(N_payoff, num_horses, mu, sigma):
    n = 1 + num_horses + num_horses * (num_horses - 1) // 2 + num_horses * (num_horses - 1) * (num_horses - 2) // 6
    place_count = np.array([num_horses - 1 - i for i in range(num_horses - 1)])
    show_count_0 = np.array([(num_horses - 1 - i) * (num_horses - 2 - i) / 2 for i in range(num_horses - 2)])
    show_count_1 = np.array([num_horses - 2 - i for i in range(num_horses - 2)])  # 8,7,...,1
    finish_time = np.zeros((num_horses, N_payoff))
    for i in range(num_horses):
        finish_time[i, :] = np.random.randn(N_payoff) * sigma[i] + mu[i]
    # compute payoff
    payoff_count = np.zeros(n - 1)
    for i in range(N_payoff):
        order = np.argsort(finish_time[:, i])
        payoff_count[order[0]] += 1
        place_idx = np.sum(place_count[0: min(order[0], order[1])]) + max(order[0], order[1]) \
                    - min(order[0], order[1]) - 1
        payoff_count[10 + place_idx] += 1
        first_three_ordered_by_idx = np.sort(order[0:3])
        show_idx = int(np.sum(show_count_0[0: first_three_ordered_by_idx[0]]) \
                       + np.sum(show_count_1[first_three_ordered_by_idx[0]: first_three_ordered_by_idx[1] - 1]) \
                       + first_three_ordered_by_idx[2] - first_three_ordered_by_idx[1] - 1)
        payoff_count[10 + 45 + show_idx] += 1
    fac = np.exp(np.random.randn(n - 1) * 1.25 - 1.5)
    # fac = np.array([1.0 for i in range(0)] + list(fac))
    payoff = N_payoff / np.maximum(1, payoff_count) * fac
    return payoff


payoff = generate_unit_payoff(N * 10, num_horses, mu, sigma)


def generate_random_data():
    place_count = np.array([num_horses - 1 - i for i in range(num_horses - 1)])
    show_count_0 = np.array([(num_horses - 1 - i) * (num_horses - 2 - i) / 2 for i in range(num_horses - 2)])
    show_count_1 = np.array([num_horses - 2 - i for i in range(num_horses - 2)])  # 8,7,...,1
    # generate outcomes
    finish_time = np.zeros((num_horses, N))
    for i in range(num_horses):
        finish_time[i, :] = np.random.randn(N) * sigma[i] + mu[i]
    place_num = int(num_horses * (num_horses - 1) / 2)
    show_num = int(num_horses * (num_horses - 1) * (num_horses - 2) / 6)
    win_bets_return = np.zeros((num_horses, N))
    place_bets_return = np.zeros((place_num, N))
    show_bets_retrun = np.zeros((show_num, N))
    # finish_time -= np.multiply(finish_time, (np.random.uniform(0, 1, size=(num_horses, N)) <= 0.1) * 0.2)
    for i in range(N):
        order = np.argsort(finish_time[:, i])
        win_bets_return[order[0], i] = payoff[order[0]]
        place_idx = np.sum(place_count[0: min(order[0], order[1])]) \
                    + max(order[0], order[1]) - min(order[0], order[1]) - 1
        place_bets_return[place_idx, i] = payoff[place_idx + 10]
        first_three_ordered_by_idx = np.sort(order[0:3])
        show_idx = int(np.sum(show_count_0[0: first_three_ordered_by_idx[0]]) \
                       + np.sum(show_count_1[first_three_ordered_by_idx[0]: first_three_ordered_by_idx[1] - 1]) \
                       + first_three_ordered_by_idx[2] - first_three_ordered_by_idx[1] - 1)
        show_bets_retrun[show_idx, i] = payoff[show_idx + 55]
    return np.concatenate([win_bets_return, place_bets_return, show_bets_retrun, np.ones((1, N))])


def get_initial_val():
    ini = np.ones(n) / n
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
### baseline and plot
def get_baseline_soln_cvxpy(R, compare_with_all=False):
    b_baseline_var = cp.Variable(n, nonneg=True)
    obj_baseline = -cp.sum(cp.log(R.T @ b_baseline_var)) / N
    constr_baseline = [cp.sum(b_baseline_var) == 1]
    prob_baseline = cp.Problem(cp.Minimize(obj_baseline), constr_baseline)
    b_baseline_var.value = np.ones(n) / n
    print("ini", obj_baseline.value)
    print("Start to solve baseline problem by MOSEK")
    ep = 1e-6
    t0 = time.time()
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


def get_baseline_soln_kelly_mosek(R):
    M = Model()
    b_baseline_var = M.variable(n, Domain.greaterThan(0.))
    z = M.variable(N)
    objf = Expr.mul(Expr.sum(z), 1.0 / N)
    M.objective(ObjectiveSense.Minimize, objf)
    M.constraint(Expr.sum(b_baseline_var), Domain.equalsTo(1))
    M.constraint(Expr.hstack(Expr.mul(Matrix.dense(R.T), b_baseline_var), Expr.constTerm(N, 1.0), Expr.neg(z)), Domain.inPExpCone())
    M.solve()
    return b_baseline_var.level(), np.sum(z.level()) / N


def my_plot_one_result(W, x_best, is_save_fig=False, figname="kelly.pdf"):
    linewidth = 2
    fontsize = 14

    outcomes_count = np.zeros(n - 1)
    for i in range(N):
        idx = W[0:n - 1, i] > 0
        outcomes_count[idx] += 1

    fig = plt.figure(tight_layout=False, figsize=(18, 6))
    gs = gridspec.GridSpec(1, 3)

    a = fig.add_subplot(gs[0, 0])
    a.stem([i for i in range(n - 1)], outcomes_count[0:n - 1] / N, markerfmt=' ')
    a.set_ylabel("Empirical Probability of Winning", fontsize=fontsize)
    a.set_xlabel("Bet Index", fontsize=fontsize)
    # inset axes....
    axins = a.inset_axes([0.5, 0.5, 0.47, 0.47])
    axins.bar([i for i in range(10)], outcomes_count[0:10] / N)
    axins.set_xticklabels('')
    axins.set_xlabel("Candidate Index")
    a.indicate_inset_zoom(axins)
    a.set_ylim([0, 0.6])

    b = fig.add_subplot(gs[0, 1])
    b.stem([i for i in range(n)], x_best, markerfmt=' ')
    b.set_ylabel("Investment", fontsize=fontsize)
    b.set_xlabel("Bet Index", fontsize=fontsize)
    b.set_yscale("log")

    payoff_outcomes = np.log(W.T.dot(x_best))
    c = fig.add_subplot(gs[0, 2])
    c.hist(payoff_outcomes, bins=40)
    c.set_yscale("log")
    c.axvline(payoff_outcomes.mean(), color='k', linestyle='dashed', linewidth=linewidth)
    c.text(payoff_outcomes.mean() * 1.5, int(N * 0.4), 'Mean: {:.2f}'.format(payoff_outcomes.mean()), fontsize=fontsize)
    print("mean", payoff_outcomes.mean())
    c.set_ylabel("Number of Samples", fontsize=fontsize)
    c.set_xlabel("Log Return", fontsize=fontsize)
    #####################################################################
    if is_save_fig:
        fig.savefig(figname)