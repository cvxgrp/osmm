import torch
import autograd.numpy as np
import cvxpy as cp
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import time
from osmm import OSMM

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

t = 0
q = 30
all_edges = []
for i in range(q - 1):
    for j in range(i + 1, q):
        if i == 0 and j == q - 1:
            continue
        all_edges.append([i, j])

m = len(all_edges) // 2
N = 10000
A = np.zeros((q, m))
connected_edges = np.random.choice(range(len(all_edges)), m, replace=False)
p_max = 1
u_max = 0.1
n = q + m + 1
n_w = q * 2
D = np.eye(q)

for i in range(m):
    edge_idx = connected_edges[i]
    A[all_edges[edge_idx][0], i] = 1
    A[all_edges[edge_idx][1], i] = -1
    D[all_edges[edge_idx][0], all_edges[edge_idx][1]] = 1
    D[all_edges[edge_idx][1], all_edges[edge_idx][0]] = 1


def generate_random_data():
    Sigma = np.linalg.inv(D.dot(D.T))
    d = -np.exp(np.random.multivariate_normal(np.zeros(q), np.eye(q) * 0.5, size=int(N)) + 0.1).T
    d[0:q // 2, :] = -np.exp(np.random.multivariate_normal(np.zeros(q // 2), np.eye(q // 2) * 0.5, size=int(N)) - 2).T
    s = np.exp(np.random.multivariate_normal(np.ones(q), Sigma / np.linalg.norm(Sigma), size=int(N))).T
    W = np.zeros((n_w, N))
    W[0:q, :] = s
    W[q:q * 2, :] = d
    return W


W = generate_random_data()
W_validation = generate_random_data()


def get_initial_val():
    p_var = cp.Variable(q)
    u_var = cp.Variable(m)
    constr = [A @ u_var + p_var == 0, cp.norm(u_var, 'inf') <= u_max, cp.norm(p_var, 'inf') <= p_max]
    fea_baseline = cp.Problem(cp.Minimize(cp.norm(u_var - 0) + cp.norm(p_var - 10)), constr)
    fea_baseline.solve(solver="MOSEK")
    ini = np.zeros(n)
    ini[0:q] = p_var.value
    ini[q:q + m] = u_var.value
    ini[q + m] = 1.0
    return ini


def get_cvxpy_description():
    x_var = cp.Variable(q + m + 1)
    constr = [A @ x_var[q:q + m] + x_var[0:q] == 0, cp.norm(x_var[q:q + m], 'inf') <= u_max,
              cp.norm(x_var[0:q], 'inf') <= p_max, x_var[q + m] >= 1e-8]
    g = 0
    return x_var, g, constr


def my_objf_torch(w_torch=None, x_torch=None, take_mean=True):
    s_torch = w_torch[0:q, :]
    d_torch = w_torch[q:q * 2, :]
    if x_torch.shape == torch.Size([n]):
        objf_s = torch.sum(torch.relu(-d_torch.T - s_torch.T - x_torch[0:q]), axis=1) - t
    else:
        objf_s = torch.sum(torch.relu(-d_torch - s_torch - x_torch[0:q, :]), axis=0) - t
    if take_mean:
        result = torch.mean(torch.square(torch.relu(objf_s + x_torch[q + m]))) / x_torch[q + m]
        return result


osmm_prob = OSMM(f_torch=my_objf_torch, g_cvxpy=get_cvxpy_description, get_initial_val=get_initial_val,
                 W=W, W_validate=W_validation)
osmm_prob.solve(solver="MOSEK")

#########################################################################
### baseline and plot
def get_baseline_soln_energy(W, compare_with_all=False):
    s = W[0:q, :]
    d = W[q:q * 2, :]
    p_var = cp.Variable(q)
    u_var = cp.Variable(m)
    a_var = cp.Variable(1)
    constr = [A @ u_var + p_var == 0, cp.norm(u_var, 'inf') <= u_max, cp.norm(p_var, 'inf') <= p_max, a_var >= 1e-8]
    loss = cp.sum(cp.pos(cp.vstack([-d[i, :] - s[i, :] - p_var[i] for i in range(q)])), axis=0)
    print("loss shape", loss.shape)
    objf = cp.quad_over_lin(cp.pos(loss - t + a_var), a_var) / N
    prob_baseline = cp.Problem(cp.Minimize(objf), constr)
    print("Start MOSEK")
    t0 = time.time()
    prob_baseline.solve(solver="MOSEK")
    print("  MOSEK + CVXPY time cost ", time.time() - t0)
    print("  MOSEK solver time cost", prob_baseline.solver_stats.solve_time)
    print("  Setup time cost", prob_baseline.solver_stats.setup_time)
    print("  Objective value", prob_baseline.value)
    print("  Solver status  " + prob_baseline.status + ".\n")
    prob_baseline_val = prob_baseline.value
    mosek_solve_time = prob_baseline.solver_stats.solve_time

    if compare_with_all:
        p_var.value = None
        u_var.value = None
        a_var.value = None
        print("Start to solve baseline problem by SCS")
        t0 = time.time()
        prob_baseline.solve(solver="SCS", verbose=True)
        print("  SCS + CVXPY time cost ", time.time() - t0)
        print("  Objective value", prob_baseline.value)
        print("  Solver status  " + prob_baseline.status + ".\n")

        p_var.value = None
        u_var.value = None
        a_var.value = None
        print("Start to solve baseline problem by ECOS")
        t0 = time.time()
        prob_baseline.solve(solver="ECOS", verbose=False)
        print("  ECOS + CVXPY time cost ", time.time() - t0)
        print("  ECOS solver time cost", prob_baseline.solver_stats.solve_time)
        print("  Setup time cost", prob_baseline.solver_stats.setup_time)
        print("  Objective value", prob_baseline.value)
        print("  Solver status  " + prob_baseline.status + ".\n")
    return np.concatenate([p_var.value, u_var.value, a_var.value]), prob_baseline_val, mosek_solve_time


def my_plot_energy_one_result(W, x_best, is_save_fig=False, figname="energy.pdf"):
    linewidth = 2
    fontsize = 14

    fig = plt.figure(tight_layout=True, figsize=(18, 6))
    gs = gridspec.GridSpec(1, 3)

    a = fig.add_subplot(gs[0, 0:2])
    NN = 1000
    W_data = np.zeros((NN * 2 * 30, 3))
    for i in range(30):
        W_data[i * NN:(i + 1) * NN, 0] = W[i, 0:NN]
        W_data[(i + 30) * NN:(i + 31) * NN, 0] = -W[i + 30, 0:NN]
        W_data[i * NN:(i + 1) * NN, 1] = i  # node index
        W_data[(i + 30) * NN:(i + 31) * NN, 1] = i
        W_data[(i + 30) * NN:(i + 31) * NN, 2] = 1
        W_data[(i) * NN:(i + 1) * NN, 2] = 0

    W_df = pd.DataFrame(data=W_data, columns=["Amount of Energy", "Node Index", "Category"])
    W_df["Category"] = ["Renewable Source"] * NN * 30 + ["Demand"] * NN * 30
    W_df["Node Index"] = W_df["Node Index"].astype(int)

    sns.stripplot(data=W_df, x="Node Index", y="Amount of Energy", hue="Category", linewidth=1,
                  dodge=True, alpha=0.15, marker='.', palette={"Renewable Source": "b", "Demand": ".85"}, ax=a)
    a.set_xlabel("Node Index", fontsize=fontsize)
    a.set_ylabel("Amount of Energy", fontsize=fontsize)
    a.set_yscale("log")
    a.grid()

    b = fig.add_subplot(gs[0, 2])
    counts, bin_edges = np.histogram(np.sum(np.maximum(0, -W[q:2 * q, :].T - W[0:q, :].T - x_best[0:q]), axis=1),
                                     bins=50, density=True)
    cdf = np.cumsum(counts)
    b.plot(bin_edges[0:len(bin_edges)-1], cdf / cdf[-1])
    shorfall_mean = np.mean(np.sum(np.maximum(0, -W[q:2 * q, :].T - W[0:q, :].T - x_best[0:q]), axis=1))
    b.axvline(shorfall_mean, color='k', linestyle='dashed', linewidth=linewidth)
    b.text(shorfall_mean * 2, 0.8, 'Mean: {:.2f}'.format(shorfall_mean), fontsize=fontsize)
    b.set_ylabel("Empirical CDF", fontsize=fontsize)
    b.set_xlabel("Energy Shortfall", fontsize=fontsize)
    b.grid()
    if is_save_fig:
        fig.savefig(figname)