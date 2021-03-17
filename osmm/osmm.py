import cvxpy as cp
import numpy as np
import torch
from functools import partial

import time
import datetime

from osmm_update import OsmmUpdate
from algmode import *


class OSMM:
    def __init__(self, f_torch, g_cvxpy, get_initial_val, W, W_validate=None, f_hess=None, f_allow_batch=False,
                 save_timing=False, store_x_all_iters=False):
        self.f_torch = f_torch
        self.W_torch = torch.tensor(W, dtype=torch.float, requires_grad=False)
        if W_validate is not None:
            self.W_torch_validate = torch.tensor(W_validate, dtype=torch.float, requires_grad=False)
            self.eval_validate = True
        else:
            self.eval_validate = False
        if f_hess is not None:
            self.f_hess = f_hess  # numpy
        else:
            self.f_hess = self.f_hess_value
        self.x_var_cvxpy, self.g_cvxpy, self.constr_cvxpy = g_cvxpy()
        try:
            _ = self.g_cvxpy.value
        except Exception as e:
            self.g_cvxpy = 0 * cp.sum(self.x_var_cvxpy)
        self.get_initial_val = get_initial_val
        self.f_allow_batch = f_allow_batch
        self.is_save = save_timing
        self.store_x_all_iters = store_x_all_iters
        now = datetime.datetime.now()
        mmddyyhhmm = ("%d_%d_%d_%d_%d" % (now.month, now.day, now.year, now.hour, now.minute))
        self.part_of_out_fn = mmddyyhhmm
        self.n = self.x_var_cvxpy.size
        self.method_results = {}

    def _f(self, x, take_mean=True):
        x_torch = torch.tensor(x, dtype=torch.float, requires_grad=True)
        if not take_mean and self.f_allow_batch:
            f_torch = self.f_torch(self.W_torch, x_torch, take_mean=False)
        else:
            f_torch = self.f_torch(self.W_torch, x_torch)
        return f_torch, x_torch

    def f_value(self, x, take_mean=True):
        if take_mean:
            return float(self._f(x)[0])
        else:
            return self._f(x, take_mean=False)[0].cpu().detach().numpy()

    def f_validate_value(self, x):
        x_torch = torch.tensor(x, dtype=torch.float, requires_grad=True)
        f_torch = self.f_torch(self.W_torch_validate, x_torch)
        return float(f_torch)

    def f_grad_value(self, x):
        f_torch, x_torch = self._f(x)
        f_torch.backward()
        return np.array(x_torch.grad.cpu())

    # def f_Jacobian_value(self, x, w, max_minibatch_size=1000):
    #     n_w, H_rank = w.shape
    #     result = np.zeros((self.n, H_rank))
    #     minibatch_size = min(max_minibatch_size, H_rank)
    #     minibatch_num = H_rank // minibatch_size
    #     for j in range(0, minibatch_num):
    #         w_minibatch = np.array(w[:, j * minibatch_size:(j + 1) * minibatch_size]).reshape((n_w, minibatch_size))
    #         x_torch = torch.tensor(x, dtype=torch.float)
    #         x_torch = x_torch.squeeze()
    #         x_torch = x_torch.repeat(minibatch_size, 1)  ### minibatch_size by n
    #         x_torch.requires_grad_(True)
    #         w_torch = torch.tensor(w_minibatch, dtype=torch.float, requires_grad=False) ###
    #         if self.f_allow_batch:
    #             f_torch = self.f_torch(w_torch, x_torch.T, take_mean=False)
    #         else:
    #             f_torch = torch.zeros(minibatch_size)
    #             for i in range(minibatch_size):
    #                 f_torch[i] = self.f_torch(w_torch[:, i].reshape((n_w, 1)), x_torch[i, :])
    #         jac_objf = f_torch.repeat(minibatch_size, 1).T
    #         jac_objf.backward(torch.eye(minibatch_size))
    #         jac_minibatch = np.array(x_torch.grad.data.cpu()).T
    #         result[:, j * minibatch_size:(j + 1) * minibatch_size] = jac_minibatch
    #     return result

    def f_hess_value(self, x):
        x_torch = torch.tensor(x, dtype=torch.float, requires_grad=True)
        my_f_partial = partial(self.f_torch, self.W_torch)
        result_torch = torch.autograd.functional.hessian(my_f_partial, x_torch)
        return np.array(result_torch.cpu())

    def f_hess_tr_Hutchinson(self, x, max_iter=100, tol=1e-3):
        est_tr = 0
        new_est_tr = 0
        it = 0
        while it < max_iter:
            f_torch, x_torch = self._f(x)
            f_torch.backward(create_graph=True)
            grad_x = x_torch.grad
            grad_x.requires_grad_(True)
            # print("grad x", torch.norm(grad_x))
            v = (np.random.rand(self.n) < 0.5) * 2 - 1.0
            v_torch = torch.tensor(v, dtype=torch.float, requires_grad=False)
            gv_objf = torch.matmul(v_torch.T, grad_x)
            gv_objf.backward()  # retain_graph=True)
            Hv = x_torch.grad.detach().cpu().numpy()
            vTHv = float(v.T.dot(Hv))
            new_est_tr = (est_tr * it + float(vTHv)) / (it + 1)
            if np.abs(new_est_tr - est_tr) < tol * np.abs(est_tr):
                break
            est_tr = new_est_tr
            it += 1
            # print(it, "Hv", np.linalg.norm(Hv), "VTHV", vTHv, "est", est_tr)
        print("Hutchinson #iters", it, "rel. incr.", np.abs(new_est_tr - est_tr) / np.abs(est_tr), "est. tr.", est_tr)
        return est_tr

    def solve(self, max_num_rounds=100, alg_mode=AlgMode.LowRankNewSampPlusBundle,
              H_rank=20, M=20, solver="ECOS", tau_min=1e-3, mu_min=1e-4, mu_max=1e5, mu_0=1.0,
              gamma_inc=1.1, gamma_dec=0.8, ini_by_Hutchison=True,
              alpha=0.05, beta=0.5, j_max=10, ep=1e-15,
              minibatch_size=None, H_0_min=1e-5, num_samples_Jacobian=10, num_iter_eval_Lk=10):

        if num_samples_Jacobian < H_rank:
            num_samples_Jacobian = H_rank
        if self.store_x_all_iters:
            self.method_results["X_iters"] = np.zeros((self.n, max_num_rounds))
        else:
            self.method_results["X_iters"] = np.zeros((self.n, M))
        self.method_results["v_norm_iters"] = np.zeros(max_num_rounds)
        self.method_results["x_best"] = np.zeros(self.n)
        self.method_results["objf_iters"] = np.zeros(max_num_rounds)
        self.method_results["objf_validation_iters"] = np.zeros(max_num_rounds)
        self.method_results["lambd_iters"] = np.zeros(max_num_rounds)
        self.method_results["mu_iters"] = np.ones(max_num_rounds)
        self.method_results["t_iters"] = np.zeros(max_num_rounds)
        self.method_results["lower_bound_iters"] = -np.ones(max_num_rounds) * np.inf
        self.method_results["runtime_iters"] = np.zeros(max_num_rounds)
        self.method_results["iters_taken"] = max_num_rounds
        self.method_results["num_f_evas_line_search_iters"] = np.zeros(max_num_rounds)
        self.method_results["opt_res_iters"] = np.zeros(max_num_rounds)
        self.method_results["q_norm_iters"] = np.zeros(max_num_rounds)
        self.method_results["f_grad_norm_iters"] = np.zeros(max_num_rounds)
        self.method_results["time_cost_detail_iters"] = np.zeros((6, max_num_rounds))

        osmm_method = OsmmUpdate(self)

        subprob, x_k, objf_k, objf_validation_k, f_k, f_grad_k, g_k, lam_k, f_grads_iters_value, f_const_iters_value, \
        G_k, diag_H_k \
            = osmm_method.initialization(H_rank, M, alg_mode, tau_min, ini_by_Hutchison, H_0_min,
                                         minibatch_size, num_samples_Jacobian)

        round_idx = 1
        lbfgs_update_round_idx = 1
        lower_bound_k = -np.inf
        mu_k = mu_0
        S_lbfgs = np.zeros((self.n, H_rank))
        Y_lbfgs = np.zeros((self.n, H_rank))

        self.method_results["X_iters"][:, 0] = x_k
        self.method_results["objf_iters"][0] = objf_k
        self.method_results["lambd_iters"][0] = lam_k
        if self.eval_validate:
            self.method_results["objf_validation_iters"][0] = objf_validation_k
        if objf_k < np.inf:
            self.method_results["x_best"] = x_k

        update_func = partial(osmm_method.update_func, alg_mode=alg_mode, H_rank=H_rank, pieces_num=M, solver=solver,
                              num_samples_Jacobian=num_samples_Jacobian, minibatch_size=minibatch_size,
                              mu_min=mu_min, tau_min=tau_min, mu_max=mu_max, ep=ep,
                              gamma_inc=gamma_inc, gamma_dec=gamma_dec,
                              beta=beta, j_max=j_max, alpha=alpha, num_iter_eval_Lk=num_iter_eval_Lk)

        while round_idx < max_num_rounds:
            start_time = time.time()

            x_k_plus_one, objf_k_plus_one, g_k_plus_one, lower_bound_k_plus_one, f_grad_k_plus_one,\
            f_grads_iters_value, f_const_iters_value, G_k_plus_one, diag_H_k_plus_one, lam_k_plus_one, mu_k_plus_one,\
            Y_lbfgs, S_lbfgs, lbfgs_update_round_idx \
                = update_func(subprob, round_idx, objf_k, g_k, lower_bound_k, f_grad_k,
                              f_grads_iters_value, f_const_iters_value, G_k, diag_H_k, lam_k, mu_k,
                              Y_lbfgs, S_lbfgs, lbfgs_update_round_idx)

            end_time = time.time()
            runtime = end_time - start_time
            self.method_results["runtime_iters"][round_idx] = runtime
            round_idx += 1
            objf_k = objf_k_plus_one
            g_k = g_k_plus_one
            lower_bound_k = lower_bound_k_plus_one
            f_grad_k = f_grad_k_plus_one
            G_k = G_k_plus_one
            diag_H_k = diag_H_k_plus_one
            lam_k = lam_k_plus_one
            mu_k = mu_k_plus_one
            if objf_k <= np.min(self.method_results["objf_iters"][1:round_idx]):
                self.method_results["x_best"] = x_k_plus_one

        print("      Time elapsed (secs): %f." % np.sum(self.method_results["runtime_iters"]))
        print("")