import cvxpy as cp
import numpy as np
import torch
from functools import partial
import time

from .osmm_update import OsmmUpdate
from .alg_mode import AlgMode


class OSMM:
    def __init__(self, f_torch, g_cvxpy):
        self.f_torch = f_torch
        self.f_hess = self.f_hess_value
        self.W_torch = None
        self.W_torch_validate = None
        self.x_var_cvxpy, self.g_cvxpy, self.constr_cvxpy, self.additional_vars_list = g_cvxpy()
        try:
            _ = self.g_cvxpy.value
        except Exception as e:
            self.g_cvxpy = 0 * cp.sum(self.x_var_cvxpy)
        self.init_val = None
        self.store_x_all_iters = True
        self.n = self.x_var_cvxpy.size
        self.method_results = {}

    def _f(self, x, take_mean=True):
        x_torch = torch.tensor(x, dtype=torch.float, requires_grad=True)
        f_torch = self.f_torch(self.W_torch, x_torch)
        return f_torch, x_torch

    def f_value(self, x, take_mean=True):
        return float(self._f(x)[0])
        # else:
        #     return self._f(x, take_mean=False)[0].cpu().detach().numpy()

    def f_validate_value(self, x):
        x_torch = torch.tensor(x, dtype=torch.float, requires_grad=True)
        f_torch = self.f_torch(self.W_torch_validate, x_torch)
        return float(f_torch)

    def f_grad_value(self, x):
        f_torch, x_torch = self._f(x)
        f_torch.backward()
        return np.array(x_torch.grad.cpu())

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
            v = (np.random.rand(self.n) < 0.5) * 2 - 1.0
            v_torch = torch.tensor(v, dtype=torch.float, requires_grad=False)
            gv_objf = torch.matmul(v_torch.T, grad_x)
            gv_objf.backward()
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

    def solve(self, W, init_val, W_validate=None, max_iter=200, hessian_rank=20, gradient_memory=20, solver="ECOS",
              alg_mode=AlgMode.LowRankQNBundle,
              ini_by_Hutchison=True, stop_early=True, num_iter_eval_Lk=10, tau_min=1e-3, mu_min=1e-4, mu_max=1e5,
              mu_0=1.0, gamma_inc=1.1, gamma_dec=0.8, alpha=0.05, beta=0.5, j_max=10, ep=1e-15,
              eps_gap_abs=1e-4, eps_gap_rel=1e-4, eps_res_abs=1e-4, eps_res_rel=1e-4):

        self.W_torch = torch.tensor(W, dtype=torch.float, requires_grad=False)
        if W_validate is not None:
            self.W_torch_validate = torch.tensor(W_validate, dtype=torch.float, requires_grad=False)

        if alg_mode == AlgMode.Exact or alg_mode == AlgMode.BFGSBundle:
            hessian_rank = self.n
        if hessian_rank == 0:
            alg_mode = AlgMode.Bundle
            hessian_rank = 20

        if self.store_x_all_iters:
            self.method_results["var_iters"] = np.zeros((self.n, max_iter))
        else:
            self.method_results["var_iters"] = np.zeros((self.n, gradient_memory))
        self.method_results["v_norm_iters"] = np.zeros(max_iter)
        self.method_results["soln"] = np.zeros(self.n)
        self.method_results["objf_iters"] = np.zeros(max_iter)
        self.method_results["objf_validation_iters"] = np.zeros(max_iter)
        self.method_results["lambd_iters"] = np.zeros(max_iter)
        self.method_results["mu_iters"] = np.ones(max_iter)
        self.method_results["t_iters"] = np.zeros(max_iter)
        self.method_results["lower_bound_iters"] = -np.ones(max_iter) * np.inf
        self.method_results["runtime_iters"] = np.zeros(max_iter)
        self.method_results["iters_taken"] = max_iter
        self.method_results["num_f_evas_line_search_iters"] = np.zeros(max_iter)
        self.method_results["opt_res_iters"] = np.zeros(max_iter)
        self.method_results["q_norm_iters"] = np.zeros(max_iter)
        self.method_results["f_grad_norm_iters"] = np.zeros(max_iter)
        self.method_results["time_cost_detail_iters"] = np.zeros((4, max_iter))
        self.method_results["soln_additional_vars"] = [None] * len(self.additional_vars_list)

        osmm_method = OsmmUpdate(self)

        subprob, x_k, objf_k, objf_validation_k, f_k, f_grad_k, g_k, lam_k, f_grads_iters_value, f_const_iters_value, \
        G_k, diag_H_k \
            = osmm_method.initialization(init_val, hessian_rank, gradient_memory, alg_mode, tau_min, ini_by_Hutchison)
        lower_bound_k = -np.inf
        mu_k = mu_0

        self.method_results["var_iters"][:, 0] = x_k
        self.method_results["objf_iters"][0] = objf_k
        self.method_results["lambd_iters"][0] = lam_k
        if self.W_torch_validate is not None:
            self.method_results["objf_validation_iters"][0] = objf_validation_k
        if objf_k < np.inf:
            self.method_results["soln"] = x_k

        update_func = partial(osmm_method.update_func, alg_mode=alg_mode, H_rank=hessian_rank,
                              pieces_num=gradient_memory, solver=solver,
                              mu_min=mu_min, tau_min=tau_min, mu_max=mu_max, ep=ep,
                              gamma_inc=gamma_inc, gamma_dec=gamma_dec,
                              beta=beta, j_max=j_max, alpha=alpha, num_iter_eval_Lk=num_iter_eval_Lk,
                              eps_gap_abs=eps_gap_abs, eps_gap_rel=eps_gap_rel, eps_res_abs=eps_res_abs,
                              eps_res_rel=eps_res_rel)

        round_idx = 1
        stopping_criteria_satisfied = False
        while round_idx < max_iter and (not stop_early or round_idx < 10 or not stopping_criteria_satisfied):
            start_time = time.time()

            stopping_criteria_satisfied, x_k_plus_one, objf_k_plus_one, g_k_plus_one, lower_bound_k_plus_one, \
            f_grad_k_plus_one, f_grads_iters_value, f_const_iters_value, G_k_plus_one, diag_H_k_plus_one, \
            lam_k_plus_one, mu_k_plus_one, additional_vars_value \
                = update_func(subprob, round_idx, objf_k, g_k, lower_bound_k, f_grad_k,
                              f_grads_iters_value, f_const_iters_value, G_k, diag_H_k, lam_k, mu_k)

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
                self.method_results["soln"] = x_k_plus_one
                self.method_results["soln_additional_vars"] = additional_vars_value


        print("      Time elapsed (secs): %f." % np.sum(self.method_results["runtime_iters"]))
        print("")