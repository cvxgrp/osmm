import torch
import autograd.numpy as np
import cvxpy as cp
import mosek as mosek
from scipy.stats import multivariate_normal
import numpy.polynomial as polynomial
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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

N_0 = 10000
lam_reg = 0
importance_sampling = False
d = 2
m = 2000
Sigma1 = np.diag([0.5 ** 2] * d)
Sigma2 = np.diag([0.5 ** 2] * d)
Sigma3 = np.diag([0.5 ** 2] * d)
mu1 = 1.0 * np.ones(d)
mu2 = -1.0 * np.ones(d)
mu3 = np.array([1.0, -1.0])
prob1 = 0.4
prob2 = .3
prob3 = .3
y = np.vstack([
    np.random.multivariate_normal(mu1, Sigma1, size=int(m * prob1)),
    np.random.multivariate_normal(mu2, Sigma2, size=int(m * prob2)),
    np.random.multivariate_normal(mu3, Sigma3, size=int(m * prob3))
]).T

print("normalization constant", np.max(np.abs(y)))
y = y / np.max(np.abs(y))  # normalized to [-1, 1]^2

y_mean = np.mean(y, axis=1)
y_cov = np.cov(y - y_mean[:, None])

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
    Dphi_mtx = np.zeros((N, n, d))  # d=2
    Dphi_mtx_transpose = np.zeros((N, d, n))  # d=2
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


def generate_random_data():
    if importance_sampling:
        z = np.random.multivariate_normal(y_mean, y_cov * 1.1, N_0)
        z_inf_norms = np.max(np.abs(z), axis=1)
        z_acc = z[z_inf_norms <= 1, :]
        log_sampling_normalizer_pdf = multivariate_normal.logpdf(z_acc, y_mean, y_cov * 1.1)
        z_acc = z_acc.T
    else:
        z_acc = np.zeros((2, N_0))
        for j in range(int(np.sqrt(N_0))):
            for i in range(int(np.sqrt(N_0))):
                z_acc[0, i + j * int(np.sqrt(N_0))] = -1 + i * 2 / int(np.sqrt(N_0))
                z_acc[1, i + j * int(np.sqrt(N_0))] = -1 + j * 2 / int(np.sqrt(N_0))
        log_sampling_normalizer_pdf = np.log(np.ones(N_0) / 4)
    _, N = z_acc.shape
    log_pi_z = - log_sampling_normalizer_pdf.reshape((1, N)) - np.log(N_0)
    phi_z = phi(z_acc)
    W = np.concatenate([phi_z, log_pi_z])
    return W


init_val = np.ones(n) / n
g_var = cp.Variable(n)
g_obj = lam_reg * cp.quad_form(g_var, D_reg)
g_constr = []

def my_f_torch(theta_torch=None, W_torch=None):
    _, batch_size = W_torch.shape
    phi_z_torch = W_torch[0:n, :]
    log_pi_z_torch = W_torch[n_w - 1, :]
    A_theta = torch.logsumexp(torch.matmul(-phi_z_torch.T, theta_torch) + log_pi_z_torch, 0)
    return A_theta + torch.matmul(theta_torch.T, torch.tensor(c / m, dtype=torch.float))


#########################################################################
### baseline and plot
def get_baseline_soln_cvxpy(W, ep=1e-6, compare_MOSEK=False, compare_SCS=False, compare_ECOS=False):
    phi_z = W[0:n, :]
    log_pi_z = W[n_w - 1, :]
    theta_var_baseline = cp.Variable(n)
    objf_baseline = c.T @ theta_var_baseline / m + cp.log_sum_exp(-phi_z.T @ theta_var_baseline + log_pi_z) \
                    + lam_reg * cp.quad_form(theta_var_baseline, D_reg)
    prob_baseline = cp.Problem(cp.Minimize(objf_baseline))

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


def my_plot_exp_density_one_result(x_best, objfs, iters_taken, y, is_save_fig=False, figname="exp_density.pdf"):
    font = {'family': 'serif',
            'size': 16,
            }

    best_iter = np.argmin(objfs[0:iters_taken])
    # x_best = Xs[:, best_iter]
    objf_best = objfs[best_iter]
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

    # ####################################################################
    c_fig = fig.add_subplot(gs[0, 1], projection='3d')
    c_fig.plot_surface(X, Y, np.exp(np.maximum(-20, log_prob - A_theta_value)))
    c_fig.set_zlabel("Estimated Density", fontdict=font)
    c_fig.tick_params(labelsize=16)
    plt.xticks(np.array([-1, -0.5, 0.0, 0.5, 1.0]), [-1, -0.5, 0.0, 0.5, 1.0])
    plt.yticks(np.array([-1, -0.5, 0.0, 0.5, 1.0]), [-1, -0.5, 0.0, 0.5, 1.0])

    #####################################################################
    if is_save_fig:
        fig.savefig(figname)