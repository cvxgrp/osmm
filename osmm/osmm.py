import cvxpy as cp
import numpy as np
import torch
import time

from .osmm_update import OsmmUpdate
from .alg_mode import AlgMode
from .f_torch import FTorch
from .g_cvxpy import GCvxpy


class OSMM:
    def __init__(self):
        self.f_torch = FTorch()
        self.g_cvxpy = GCvxpy()
        self.n = 0
        self.n0 = 0
        self.n1 = 0
        self.store_var_all_iters = True
        self.method_results = {}

    def solve(self, init_val, max_iter=200, hessian_rank=20, gradient_memory=20, solver="ECOS",
              eps_gap_abs=1e-4, eps_gap_rel=1e-3, eps_res_abs=1e-4, eps_res_rel=1e-3, check_gap_frequency=10,
              method="LowRankQNBundle", update_curvature_frequency=1, min_iter=3,
              store_var_all_iters=True, verbose=False, use_termination_criteria=True, use_cvxpy_param=False,
              use_Hutchinson_init=True, tau_min=1e-3, mu_min=1e-4, mu_max=1e5, mu_0=1.0, gamma_inc=1.1, gamma_dec=0.8,
              alpha=0.05, beta=0.5, j_max=10, ep=1e-15, trust_param_zero=False,
              all_active_cuts=False):

        assert hessian_rank >= 0
        assert gradient_memory >= 1
        assert self.f_torch.function is not None
        assert self.g_cvxpy.variable is not None
        assert method == "LowRankQNBundle" or method == "LowRankDiagEVD"

        if method == "LowRankQNBundle":
            alg_mode = AlgMode.LowRankQNBundle
        else:
            alg_mode = AlgMode.LowRankDiagEVD
        if hessian_rank == 0:
            alg_mode = AlgMode.Bundle
            hessian_rank = 1

        # set variable dimension
        self.n = self.g_cvxpy.variable.size
        if len(self.g_cvxpy.variable.shape) <= 1:
            self.n0 = self.n
            self.n1 = 0
        else:
            self.n0, self.n1 = self.g_cvxpy.variable.shape

        # prepare g_cvxpy
        try:
            _ = self.g_cvxpy.objective.value
        except Exception as e:
            self.g_cvxpy.objective = cp.Parameter(value=0)
        self.g_cvxpy.additional_var_soln = {}
        for var in self.g_cvxpy.objective.variables():
            if var is not self.g_cvxpy.variable and var not in self.g_cvxpy.additional_var_soln:
                self.g_cvxpy.additional_var_soln[var] = None
        for constr in self.g_cvxpy.constraints:
            for var in constr.variables():
                if var is not self.g_cvxpy.variable and var not in self.g_cvxpy.additional_var_soln:
                    self.g_cvxpy.additional_var_soln[var] = None

        # allocate torch memory for data
        # if self.f_torch.W_torch is not None:
        #     del self.f_torch.W_torch
        # if self.f_torch.W_validate_torch is not None:
        #     del self.f_torch.W_validate_torch
        # if self.f_torch.W is not None:
        #     self.f_torch.W_torch = torch.tensor(self.f_torch.W, dtype=torch.float, requires_grad=False)
        # if self.f_torch.W_validate is not None:
        #     self.f_torch.W_validate_torch = torch.tensor(self.f_torch.W_validate, dtype=torch.float,
        #                                                  requires_grad=False)

        # allocate memory for results
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
        self.method_results["time_detail_iters"] = np.zeros((5, max_iter))
        self.method_results["total_iters"] = 0

        # method
        osmm_method = OsmmUpdate(self, hessian_rank, gradient_memory, use_cvxpy_param, solver, tau_min, mu_min, mu_max,
                                 gamma_inc, gamma_dec, alpha, beta, j_max, eps_gap_abs, eps_gap_rel, eps_res_abs,
                                 eps_res_rel, verbose, alg_mode, check_gap_frequency, update_curvature_frequency,
                                 trust_param_zero)

        # initialization
        objf_k, objf_validate_k, f_k, f_grad_k, g_k, lam_k, f_grads_memory, f_consts_memory, G_k, H_diag_k \
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

        # iterative update
        iter_idx = 1
        stopping_criteria_satisfied = False
        while iter_idx < max_iter and (
                not use_termination_criteria or iter_idx <= min_iter or not stopping_criteria_satisfied):
            iter_start_time = time.time()

            stopping_criteria_satisfied, x_k_plus_one, objf_k_plus_one, f_k_plus_one, g_k_plus_one, \
            lower_bound_k_plus_one, f_grad_k_plus_one, f_grads_memory, f_consts_memory, G_k_plus_one, \
            H_diag_k_plus_one, lam_k_plus_one, mu_k_plus_one \
                = osmm_method.update_func(iter_idx, objf_k, f_k, g_k, lower_bound_k, f_grad_k,
                                          f_grads_memory, f_consts_memory, G_k, H_diag_k, lam_k, mu_k, ep, all_active_cuts)

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
            H_diag_k = H_diag_k_plus_one
            lam_k = lam_k_plus_one
            mu_k = mu_k_plus_one

        # get solution
        self.g_cvxpy.variable.value = self.method_results["soln"]
        for var in self.g_cvxpy.additional_var_soln:
            var.value = self.g_cvxpy.additional_var_soln[var]
        total_iters = self.method_results["total_iters"]
        result_objf = np.min(self.method_results["objf_iters"][0:total_iters + 1])

        if verbose:
            if self.f_torch.W_validate_torch is not None:
                print("      Terminated. Num iterations = {}, objf = {:.3e}, lower bound = {:.3e}, "
                      "RMS residual = {:.3e}, sampling acc = {:.3e}."
                      .format(total_iters, result_objf, lower_bound_k,
                              self.method_results["rms_res_iters"][total_iters],
                              np.abs(self.method_results["objf_validate_iters"][total_iters] - objf_k)))
            else:
                print("      Terminated. Num iterations = {}, objf = {:.3e}, lower bound = {:.3e}, "
                      "RMS residual = {:.3e}."
                      .format(total_iters, result_objf, lower_bound_k,
                              self.method_results["rms_res_iters"][total_iters]))
            print("      Time elapsed (secs): %f." % np.sum(self.method_results["time_iters"]))
            print("")

        return result_objf