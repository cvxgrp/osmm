import cvxpy as cp
import numpy as np
import torch
from functools import partial
import time

from .osmm_update import OsmmUpdate
from .alg_mode import AlgMode
from .f_torch import FTorch

class OSMM:
    def __init__(self, f_torch, g_cvxpy):
        # self.f_torch = f_torch
        self.f_torch = FTorch(f_torch)
        self.f_hess = self.f_hess_value
        self.x_var_cvxpy, self.g_objf, self.g_constrs = g_cvxpy()

        try:
            _ = self.g_objf.value
        except Exception as e:
            self.g_objf = cp.Parameter(value=0)

        self.g_additional_var_soln = {}
        for var in self.g_objf.variables():
            if var is not self.x_var_cvxpy and var not in self.g_additional_var_soln:
                self.g_additional_var_soln[var] = None
        for constr in self.g_constrs:
            for var in constr.variables():
                if var is not self.x_var_cvxpy and var not in self.g_additional_var_soln:
                    self.g_additional_var_soln[var] = None

        self.n = self.x_var_cvxpy.size
        if len(self.x_var_cvxpy.shape) <= 1:
            self.n0 = self.n
            self.n1 = 0
        else:
            self.n0, self.n1 = self.x_var_cvxpy.shape
        self.store_var_all_iters = True
        self.method_results = {}
        self.W_torch = None
        self.W_torch_validate = None

    def _f(self, x):
        x_torch = torch.tensor(x, dtype=torch.float, requires_grad=True)
        f_torch = self.f_torch.eval_func(x_torch, self.W_torch)
        return f_torch, x_torch

    def f_value(self, x):
        return float(self._f(x)[0])

    def f_validate_value(self, x):
        x_torch = torch.tensor(x, dtype=torch.float, requires_grad=True)
        f_torch = self.f_torch.eval_func(x_torch, self.W_torch_validate)
        return float(f_torch)

    def f_grad_value(self, x):
        f_torch, x_torch = self._f(x)
        f_torch.backward()
        return np.array(x_torch.grad.cpu())

    def f_hess_value(self, x):
        x_torch = torch.tensor(x, dtype=torch.float, requires_grad=True)
        my_f_partial = partial(self.f_torch.eval_func, W_torch=self.W_torch)
        result_torch = torch.autograd.functional.hessian(my_f_partial, x_torch)
        return np.array(result_torch.cpu())

    def f_hess_tr_Hutchinson(self, x, max_iter=100, tol=1e-3):
        est_tr = 0
        it = 0
        while it < max_iter:
            f_torch, x_torch = self._f(x)
            f_torch.backward(create_graph=True)
            grad_x = x_torch.grad
            grad_x_tmp = np.array(x_torch.grad.detach().cpu().numpy())
            grad_x.requires_grad_(True)
            if self.n1 == 0:
                v = (np.random.rand(self.n) < 0.5) * 2 - 1.0
                v_torch = torch.tensor(v, dtype=torch.float, requires_grad=False)
                gv_objf = torch.matmul(v_torch.T, grad_x)
                gv_objf.backward()
                Hv = x_torch.grad.detach().cpu().numpy() - grad_x_tmp
                vTHv = float(v.T.dot(Hv))
            else:
                v = (np.random.rand(self.n0, self.n1) < 0.5) * 2 - 1.0
                v_torch = torch.tensor(v, dtype=torch.float, requires_grad=False)
                gv_objf = torch.sum(v_torch * grad_x)
                gv_objf.backward()
                Hv = x_torch.grad.detach().cpu().numpy() - grad_x_tmp
                vTHv = float(np.sum(v * Hv))
            new_est_tr = (est_tr * it + float(vTHv)) / (it + 1)
            if np.abs(new_est_tr - est_tr) < tol * np.abs(est_tr):
                break
            est_tr = new_est_tr
            it += 1
        # print("Hutchinson #iters", it, "rel. incr.", np.abs(new_est_tr - est_tr) / np.abs(est_tr), "est. tr.", est_tr)
        return est_tr

    def solve(self, init_val, max_iter=200, hessian_rank=20, gradient_memory=20, solver="ECOS",
              eps_gap_abs=1e-4, eps_gap_rel=1e-3, eps_res_abs=1e-4, eps_res_rel=1e-3, check_gap_frequency=10,
              store_var_all_iters=True, verbose=False, use_termination_criteria=True, use_cvxpy_param=False,
              use_Hutchinson_init=True, tau_min=1e-3, mu_min=1e-4, mu_max=1e5, mu_0=1.0, gamma_inc=1.1, gamma_dec=0.8,
              alpha=0.05, beta=0.5, j_max=10, ep=1e-15):

        assert hessian_rank >= 0
        assert gradient_memory >= 1

        if hessian_rank == 0:
            alg_mode = AlgMode.Bundle
            hessian_rank = 1
        else:
            alg_mode = AlgMode.LowRankQNBundle

        if self.W_torch is not None:
            del self.W_torch
        if self.W_torch_validate is not None:
            del self.W_torch_validate
        if self.f_torch.W is not None:
            self.W_torch = torch.tensor(self.f_torch.W, dtype=torch.float, requires_grad=False)
        if self.f_torch.W_validate is not None:
            self.W_torch_validate = torch.tensor(self.f_torch.W_validate, dtype=torch.float, requires_grad=False)

        self.store_var_all_iters = store_var_all_iters
        if self.store_var_all_iters:
            if self.n1 == 0:
                self.method_results["var_iters"] = np.zeros((self.n, max_iter))
            else:
                self.method_results["var_iters"] = np.zeros((self.n0, self.n1, max_iter))
        else:
            if self.n1 == 0:
                self.method_results["var_iters"] = np.zeros((self.n, 1))
            else:
                self.method_results["var_iters"] = np.zeros((self.n0, self.n1, 1))
        self.method_results["soln"] = None
        self.method_results["objf_iters"] = np.ones(max_iter) * np.inf
        self.method_results["objf_validate_iters"] = np.ones(max_iter) * np.inf
        self.method_results["lower_bound_iters"] = -np.ones(max_iter) * np.inf
        self.method_results["f_grad_norm_iters"] = np.zeros(max_iter)
        self.method_results["rms_res_iters"] = np.zeros(max_iter)
        self.method_results["q_norm_iters"] = np.zeros(max_iter)
        self.method_results["v_norm_iters"] = np.zeros(max_iter)
        self.method_results["lam_iters"] = np.zeros(max_iter)
        self.method_results["mu_iters"] = np.ones(max_iter)
        self.method_results["t_iters"] = np.zeros(max_iter)
        self.method_results["num_f_evals_iters"] = np.zeros(max_iter)
        self.method_results["time_iters"] = np.zeros(max_iter)
        self.method_results["time_detail_iters"] = np.zeros((4, max_iter))
        self.method_results["total_iters"] = 0

        osmm_method = OsmmUpdate(self, hessian_rank, gradient_memory, use_cvxpy_param, solver, tau_min, mu_min, mu_max,
                                 gamma_inc, gamma_dec, alpha, beta, j_max, eps_gap_abs, eps_gap_rel, eps_res_abs,
                                 eps_res_rel, verbose, alg_mode, check_gap_frequency)

        objf_k, objf_validate_k, f_k, f_grad_k, g_k, lam_k, f_grads_memory, f_consts_memory, G_k\
            = osmm_method.initialization(init_val, use_Hutchinson_init)
        lower_bound_k = -np.inf
        mu_k = mu_0
        x_k = init_val
        if self.n1 == 0:
            self.method_results["var_iters"][:, 0] = x_k
        else:
            self.method_results["var_iters"][:, :, 0] = x_k
        self.method_results["objf_iters"][0] = objf_k
        self.method_results["objf_validate_iters"][0] = objf_validate_k
        self.method_results["f_grad_norm_iters"][0] = np.linalg.norm(f_grad_k)
        self.method_results["lam_iters"][0] = lam_k
        self.method_results["mu_iters"][0] = mu_k

        iter_idx = 1
        stopping_criteria_satisfied = False
        while iter_idx < max_iter and (not use_termination_criteria or iter_idx < 10 or not stopping_criteria_satisfied):
            iter_start_time = time.time()

            stopping_criteria_satisfied, x_k_plus_one, objf_k_plus_one, f_k_plus_one, g_k_plus_one, \
            lower_bound_k_plus_one, f_grad_k_plus_one, f_grads_memory, f_consts_memory, G_k_plus_one, \
            lam_k_plus_one, mu_k_plus_one \
                = osmm_method.update_func(iter_idx, objf_k, f_k, g_k, lower_bound_k, f_grad_k,
                                          f_grads_memory, f_consts_memory, G_k, lam_k, mu_k, ep)

            iter_end_time = time.time()
            iter_runtime = iter_end_time - iter_start_time
            self.method_results["time_iters"][iter_idx] = iter_runtime
            iter_idx += 1
            objf_k = objf_k_plus_one
            g_k = g_k_plus_one
            f_k = f_k_plus_one
            lower_bound_k = lower_bound_k_plus_one
            f_grad_k = f_grad_k_plus_one
            G_k = G_k_plus_one
            lam_k = lam_k_plus_one
            mu_k = mu_k_plus_one

        self.x_var_cvxpy.value = self.method_results["soln"]
        for var in self.g_additional_var_soln:
            var.value = self.g_additional_var_soln[var]
        total_iters = self.method_results["total_iters"]
        result_objf = np.min(self.method_results["objf_iters"][0:total_iters + 1])
        if verbose:
            if self.W_torch_validate is not None:
                print("      Terminated. Num iterations = {}, objf = {:.3e}, lower bound = {:.3e}, "
                      "RMS residual = {:.3e}, sampling acc = {:.3e}."
                      .format(total_iters, result_objf, lower_bound_k, self.method_results["rms_res_iters"][total_iters],
                              np.abs(self.method_results["objf_validate_iters"][total_iters] - objf_k)))
            else:
                print("      Terminated. Num iterations = {}, objf = {:.3e}, lower bound = {:.3e}, "
                      "RMS residual = {:.3e}."
                      .format(total_iters, result_objf, lower_bound_k, self.method_results["rms_res_iters"][total_iters]))
            print("      Time elapsed (secs): %f." % np.sum(self.method_results["time_iters"]))
            print("")
        return result_objf