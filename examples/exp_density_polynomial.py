import autograd.numpy as np
import cvxpy as cp
import torch
from scipy.stats import multivariate_normal

import numpy.polynomial as polynomial

from plot_convergence_1trial_1alg import get_best_x
import matplotlib.gridspec as gridspec
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import seaborn as sns
sns.set(style="ticks")
import time


N_0 = 1000000
lam_reg = 0#1e-6

d = 2
m = 2000
Sigma1 = np.diag([0.5 ** 2] * d)
Sigma2 = np.diag([0.5 ** 2] * d)
Sigma3 = np.diag([0.5 ** 2] * d)
mu1 = 1.0 * np.ones(d)
mu2 = -1.0 * np.ones(d)
mu3 = np.array([1.0, -1.0])
prob1 = .4
prob2 = .3
prob3 = .3
y = np.vstack([
    np.random.multivariate_normal(mu1, Sigma1, size=int(m * prob1)),
    np.random.multivariate_normal(mu2, Sigma2, size=int(m * prob2)),
    np.random.multivariate_normal(mu3, Sigma3, size=int(m * prob3))
    ]).T

print("max abs y before normalization", np.max(np.abs(y)))
y = y / np.max(np.abs(y)) #normalized to [-1, 1]^2

y_mean = np.mean(y, axis=1)
y_cov = np.cov(y - y_mean[:, None])
print("y cov", y_cov.shape, y_cov)

degree = 4
n = int((degree + 1) * (degree + 2) / 2 - 1)


def phi(x):
    _, num_points = x.shape
    phi_val = np.zeros((n, num_points))
    count = 0
    for i in range(degree + 1):
        for j in range(degree - i + 1):
            if i > 0 or j > 0:
                # phi_val[count, :] = np.power(x[0, :], i) * np.power(x[1, :], j)
                coeff_Legendre = np.zeros((degree + 1, degree + 1))
                coeff_Legendre[i, j] = 1
                phi_val[count, :] = polynomial.legendre.legval2d(x[0, :], x[1, :], coeff_Legendre)
                count += 1
    return phi_val


def Dphi():
    z = np.random.multivariate_normal(y_mean, y_cov * 1.1, N_0)
    z_inf_norms = np.max(np.abs(z), axis=1)
    z_acc = z[z_inf_norms <= 1, :]
    log_sampling_normalizer_pdf = multivariate_normal.logpdf(z_acc, y_mean, y_cov * 1.1)
    z_acc = z_acc.T
    _, N = z_acc.shape
    log_pi_z = - log_sampling_normalizer_pdf.reshape((1, N)) - np.log(N_0)
    Dphi_mtx = np.zeros((N, n, d)) #d=2
    Dphi_mtx_transpose = np.zeros((N, d, n)) #d=2
    count = 0
    for i in range(degree + 1):
        for j in range(degree - i + 1):
            if i > 0 or j > 0:
                coeff_Legendre = np.zeros((degree + 1, degree + 1))
                coeff_Legendre[i, j] = 1
                partial_x_coeff = polynomial.legendre.legder(coeff_Legendre, m=1, axis=0)
                partial_y_coeff = polynomial.legendre.legder(coeff_Legendre, m=1, axis=1)
                Dphi_mtx[:, count, 0] = polynomial.legendre.legval2d(z_acc[0, :], z_acc[1, :], partial_x_coeff)
                Dphi_mtx[:, count, 1] = polynomial.legendre.legval2d(z_acc[0, :], z_acc[1, :], partial_y_coeff)
                Dphi_mtx_transpose[:, 0, count] = Dphi_mtx[:, count, 0]
                Dphi_mtx_transpose[:, 1, count] = Dphi_mtx[:, count, 1]
                count += 1
    Dphi_DphiT_mtx = np.matmul(Dphi_mtx, Dphi_mtx_transpose)
    result = np.sum(Dphi_DphiT_mtx.T * np.exp(log_pi_z), axis=2)
    return result


phi_y = phi(y)
c = np.sum(phi_y, axis=1)

D_reg = Dphi()
n_w = n + 1


def generate_random_samples_exp_density():
    z = np.random.multivariate_normal(y_mean, y_cov * 1.1, N_0)
    z_inf_norms = np.max(np.abs(z), axis=1)
    z_acc = z[z_inf_norms <= 1, :]
    log_sampling_normalizer_pdf = multivariate_normal.logpdf(z_acc, y_mean, y_cov * 1.1)
    z_acc = z_acc.T
    _, N = z_acc.shape
    print("N_0 = ", N_0, "N=", N)
    log_pi_z = - log_sampling_normalizer_pdf.reshape((1, N)) - np.log(N_0)
    phi_z = phi(z_acc)
    W = np.concatenate([phi_z, log_pi_z])
    return W


def get_baseline_soln_exp_density(W, compare_with_all=False):
    phi_z = W[0:n, :]
    log_pi_z = W[n_w - 1, :]
    theta_var_baseline = cp.Variable(n)
    objf_baseline = c.T @ theta_var_baseline / m + cp.log_sum_exp(-phi_z.T @ theta_var_baseline + log_pi_z) \
                    + lam_reg * cp.quad_form(theta_var_baseline, D_reg)
    prob_baseline = cp.Problem(cp.Minimize(objf_baseline))
    print("Start to solve baseline problem by MOSEK")
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
        theta_var_baseline.value = None
        print("Start to solve baseline problem by SCS")
        t0 = time.time()
        prob_baseline.solve(solver="SCS", verbose=True)
        print("  SCS + CVXPY time cost ", time.time() - t0)
        print("  Objective value", prob_baseline.value)
        print("  Solver status  " + prob_baseline.status + ".\n")

        theta_var_baseline.value = None
        print("Start to solve baseline problem by ECOS")
        t0 = time.time()
        prob_baseline.solve(solver="ECOS", verbose=False)
        print("  ECOS + CVXPY time cost ", time.time() - t0)
        print("  ECOS solver time cost", prob_baseline.solver_stats.solve_time)
        print("  Setup time cost", prob_baseline.solver_stats.setup_time)
        print("  Objective value", prob_baseline.value)
        print("  Solver status  " + prob_baseline.status + ".\n")

    return theta_var_baseline.value, prob_baseline_val, mosek_solve_time


def get_initial_val_exp_density():
    return np.ones(n) / n


def get_cvxpy_description_exp_density():
    theta_var = cp.Variable(n)
    g = 0 * cp.sum(theta_var) + lam_reg * cp.quad_form(theta_var, D_reg)
    constr = []
    return theta_var, g, constr


def my_objf_torch_exp_density(w_torch=None, theta_torch=None, take_mean=True):
    _, batch_size = w_torch.shape
    phi_z_torch = w_torch[0:n, :]
    log_pi_z_torch = w_torch[n_w - 1, :]
    A_theta = torch.logsumexp(torch.matmul(-phi_z_torch.T, theta_torch) + log_pi_z_torch, 0)
    return A_theta + torch.matmul(theta_torch.T,  torch.tensor(c / m, dtype=torch.float))


def my_plot_exp_density_one_result(W, xs, objfs, is_save_fig=False, figname="exp_density.pdf"):
    font = {'family': 'serif',
            'size': 16,
            }

    num_trials, _, num_iters = objfs.shape
    _, _, _, saved_num_iters = xs.shape
    if num_iters == saved_num_iters:
        x_best, objf_best = get_best_x(objfs, xs, trial_idx=0)
    else:
        x_best, objf_best = get_best_x(objfs[:, :, num_iters - saved_num_iters:num_iters], xs, trial_idx=0)

    A_theta_value = objf_best - c.T.dot(x_best) / m

    plt.rcParams.update({'font.size': 16})
    fig = plt.figure(tight_layout=True, figsize=(16, 5))
    gs = gridspec.GridSpec(1, 2)

    a = fig.add_subplot(gs[0, 0])
    delta = 0.025
    axis_range = 1
    x_axis = y_axis = np.arange(-axis_range, axis_range, delta)
    X, Y = np.meshgrid(x_axis, y_axis)
    points_for_plots = np.zeros((2, len(x_axis) ** 2))
    points_for_plots[0, :] = X.reshape((1, len(x_axis) ** 2), order="F")
    points_for_plots[1, :] = Y.reshape((1, len(x_axis) ** 2), order="F")
    phi_for_plots = phi(points_for_plots)
    phi_T_theta_for_plots = x_best.dot(phi_for_plots)
    log_prob = -phi_T_theta_for_plots.reshape(len(x_axis), len(x_axis), order="F")
    im = a.imshow(np.maximum(-6, log_prob - A_theta_value), interpolation=None, origin='lower',
                  extent=[-axis_range, axis_range, -axis_range, axis_range])
    cbar = fig.colorbar(im)
    cbar.set_label('Estimated Log Density', fontdict=font)
    cbar.ax.tick_params(labelsize=16)
    a.scatter(y[0, :], y[1, :], alpha=0.5, s=0.1)
    a.tick_params(labelsize=16)

    # a.scatter([grid[i][0] for i in range(grid_num ** 2)], [grid[i][1] for i in range(grid_num ** 2)])

    # ####################################################################
    c_fig = fig.add_subplot(gs[0, 1], projection='3d')
    c_fig.plot_surface(X, Y, np.exp(np.maximum(-20, log_prob - A_theta_value)))
    c_fig.set_zlabel("Estimated Density", fontdict=font)
    c_fig.tick_params(labelsize=16)
    plt.xticks(np.array([-1, -0.5, 0.0, 0.5, 1.0]), [-1, -0.5, 0.0, 0.5, 1.0])
    plt.yticks(np.array([-1, -0.5, 0.0, 0.5, 1.0]), [-1, -0.5, 0.0, 0.5, 1.0])

    # ###################################################################
    # b = fig.add_subplot(gs[0, 2])
    # b.stem([i for i in range(0, n)], x_best, markerfmt=" ", label="Estimated " + r"$\theta_i$")
    # b.legend()

    #####################################################################
    if is_save_fig:
        fig.savefig(figname)