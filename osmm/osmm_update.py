import numpy as np
import cvxpy as cp
import time

from .subproblem import Subproblem
from .curvature_updates import CurvatureUpdate
from .hessian_mode import HessianMode
from .bundle_mode import BundleMode


class OsmmUpdate:
    def __init__(self, osmm_problem, hessian_rank, gradient_memory, use_cvxpy_param, solver, tau_min, mu_min, mu_max,
                 gamma_inc, gamma_dec, alpha, beta, j_max, eps_gap_abs, eps_gap_rel, eps_res_abs, eps_res_rel,
                 verbose, hessian_mode, bundle_mode, check_gap_frequency, update_curvature_frequency, trust_param_zero,
                 exact_g_line_search):
        self.n = osmm_problem.n
        self.n0 = osmm_problem.n0
        self.n1 = osmm_problem.n1
        self.f_torch = osmm_problem.f_torch
        self.g_cvxpy = osmm_problem.g_cvxpy
        self.method_results = osmm_problem.method_results
        self.store_var_all_iters = osmm_problem.store_var_all_iters

        self.curvature_update = CurvatureUpdate(osmm_problem.f_torch, hessian_rank, osmm_problem.n, osmm_problem.n0,
                                                osmm_problem.n1)
        if use_cvxpy_param:
            self.subprobs_param = Subproblem(osmm_problem.g_cvxpy.variable, osmm_problem.g_cvxpy.constraints,
                                             osmm_problem.g_cvxpy.objective,
                                             hessian_rank=hessian_rank, gradient_memory=gradient_memory)
        else:
            self.subprobs_param = None
        self.hessian_rank = hessian_rank
        self.gradient_memory = gradient_memory
        self.solver = solver
        self.tau_min = tau_min
        self.mu_min = mu_min
        self.mu_max = mu_max
        self.gamma_inc = gamma_inc
        self.gamma_dec = gamma_dec
        self.alpha = alpha
        self.beta = beta
        self.j_max = j_max
        self.eps_gap_abs = eps_gap_abs
        self.eps_gap_rel = eps_gap_rel
        self.eps_res_abs = eps_res_abs
        self.eps_res_rel = eps_res_rel
        self.verbose = verbose
        self.hessian_mode = hessian_mode
        self.bundle_mode = bundle_mode
        self.check_gap_frequency = check_gap_frequency
        self.update_curvature_frequency = update_curvature_frequency
        self.trust_param_zero = trust_param_zero
        self.exact_g_line_search = exact_g_line_search
        self.num_active_cuts = 1

    def initialization(self, x_0, use_Hutchison_init):
        f_0 = self.f_torch.f_value(x_0)
        f_grad_0 = self.f_torch.f_grad_value(x_0)
        if np.isnan(f_0) or np.isinf(f_0):
            print("Initial value not valid.")

        self.g_cvxpy.variable.value = x_0
        g_0 = self.g_cvxpy.g_value(solver=self.solver)
        objf_0 = f_0 + g_0
        if objf_0 < np.inf:
            self.method_results["soln"] = x_0
            for var in self.g_cvxpy.additional_var_soln:
                self.g_cvxpy.additional_var_soln[var] = var.value

        if self.f_torch.W_validate_torch is not None:
            f_validate_0 = self.f_torch.f_validate_value(x_0)
            objf_validate_0 = f_validate_0 + g_0
        else:
            objf_validate_0 = None

        if use_Hutchison_init and (not self.trust_param_zero):
            est_hess_tr = self.f_torch.f_hess_tr_Hutchinson(x_0, max_iter=100)
            lam_0 = max(self.tau_min, est_hess_tr / self.n)
        elif not self.trust_param_zero:
            lam_0 = self.tau_min
        else:
            lam_0 = 0

        if self.n1 == 0:
            f_grad_0_vec = f_grad_0
            x_0_vec = x_0
        else:
            f_grad_0_vec = f_grad_0.flatten(order='F')
            x_0_vec = x_0.flatten(order='F')

        f_grads_memory = f_grad_0_vec.repeat(self.gradient_memory).reshape((self.n, self.gradient_memory), order='C')
        f_consts_memory = np.ones(self.gradient_memory) * (f_0 - f_grad_0_vec.T.dot(x_0_vec))

        if self.hessian_mode == HessianMode.LowRankDiagEVD:
            G_0, H_diag_0 = self.curvature_update.low_rank_diag_update(x_0)
        else:
            G_0 = np.zeros((self.n, self.hessian_rank))
            H_diag_0 = np.zeros(self.n)
        return objf_0, objf_validate_0, f_0, f_grad_0, g_0, lam_0, f_grads_memory, f_consts_memory, G_0, H_diag_0

    def _stopping_criteria(self, objf_k_plus_one, objf_validation_k_plus_one, L_k_plus_one, t_k,
                           rms_res_k_plus_one, q_norm_k_plus_one, f_grad_norm_k_plus_one):
        if objf_validation_k_plus_one is not None:
            tmp_eps_gap_abs = np.abs(objf_k_plus_one - objf_validation_k_plus_one)
        else:
            tmp_eps_gap_abs = self.eps_gap_abs
        if objf_k_plus_one is not None and objf_k_plus_one < np.inf:
            if objf_k_plus_one - L_k_plus_one <= tmp_eps_gap_abs + self.eps_gap_rel * np.abs(objf_k_plus_one):
                return True
        if t_k == 1 and rms_res_k_plus_one <= self.eps_res_abs \
                + self.eps_res_rel * (q_norm_k_plus_one + f_grad_norm_k_plus_one) / np.sqrt(self.n):
            return True
        return False

    def _line_search(self, x_k_plus_half, xk, v_k_vec, g_k_plus_half, g_k, objf_k, G_k, H_diag_k, lam_k, ep=1e-15):
        begin_evaluate_f_time = time.time()
        f_x_k_plus_half = self.f_torch.f_value(x_k_plus_half)
        end_evaluate_f_time = time.time()

        # tol=1e-5
        # if f_x_k_plus_half < np.inf and np.linalg.norm(v_k) <= tol * np.linalg.norm(xk) and \
        #                                  f_x_k_plus_half + g_k_plus_half - objf_k <= tol * np.abs(objf_k):
        #     print("t = 1 because of too small increment")
        #     return x_k_plus_half, f_x_k_plus_half, 1.0, 0, end_evaluate_f_time - begin_evaluate_f_time

        desc = np.square(max(ep, np.linalg.norm(G_k.T.dot(v_k_vec)))) + \
               np.square(max(ep, np.linalg.norm(v_k_vec * np.sqrt(H_diag_k)))) \
               + lam_k * np.square(max(ep, np.linalg.norm(v_k_vec)))
        f_tmp = f_x_k_plus_half
        phi_line_search = f_tmp + g_k_plus_half
        t = 1.0
        j = 0
        while j < self.j_max and (f_tmp is None or np.isnan(f_tmp) or
                                  phi_line_search > objf_k - 0.5 * self.alpha * t * desc):
            t = t * self.beta
            f_tmp = self.f_torch.f_value(t * x_k_plus_half + (1 - t) * xk)
            phi_line_search = f_tmp + t * g_k_plus_half + (1 - t) * g_k
            j += 1
        return t * x_k_plus_half + (1 - t) * xk, f_tmp, t, j + 1, end_evaluate_f_time - begin_evaluate_f_time

    #### revision
    def _line_search_exact_g(self, x_k_plus_half, xk, v_k_vec, g_k_plus_half, objf_k, G_k, H_diag_k, lam_k, var_val_k,
                             ep=1e-15):
        begin_evaluate_f_time = time.time()
        f_x_k_plus_half = self.f_torch.f_value(x_k_plus_half)
        end_evaluate_f_time = time.time()

        desc = np.square(max(ep, np.linalg.norm(G_k.T.dot(v_k_vec)))) + \
               np.square(max(ep, np.linalg.norm(v_k_vec * np.sqrt(H_diag_k)))) \
               + lam_k * np.square(max(ep, np.linalg.norm(v_k_vec)))
        f_tmp = f_x_k_plus_half
        g_tmp = g_k_plus_half
        h_tmp = f_tmp + g_tmp
        t = 1.0
        j = 0
        var_val_k_plus_one = [var.value for var in self.g_cvxpy.all_var_list]
        while j < self.j_max and (f_tmp is None or np.isnan(f_tmp) or
                                  h_tmp > objf_k - 0.5 * self.alpha * t * desc):
            t = t * self.beta
            x_tmp = t * x_k_plus_half + (1 - t) * xk
            f_tmp = self.f_torch.f_value(x_tmp)
            # evaluate g at x_tmp
            for i in range(len(var_val_k)):
                if var_val_k[i] is not None and var_val_k_plus_one[i] is not None:
                    var = self.g_cvxpy.all_var_list[i]
                    var.value = t * var_val_k[i] + (1 - t) * var_val_k_plus_one[i]
            g_tmp = self.g_cvxpy.g_value()
            h_tmp = f_tmp + g_tmp
            j += 1
        return t * x_k_plus_half + (1 - t) * xk, f_tmp, t, j + 1, end_evaluate_f_time - begin_evaluate_f_time, g_tmp
    ####

    def _update_curvature_after_solve(self, G_k, x_k_plus_one, x_k_plus_one_vec, xk_vec, f_grad_k_plus_one_vec,
                                      f_grad_k_vec):
        if self.hessian_mode == HessianMode.LowRankQN:
            G_k_plus_one = self.curvature_update.low_rank_quasi_Newton_update(G_k, x_k_plus_one_vec, xk_vec,
                                                                              f_grad_k_plus_one_vec, f_grad_k_vec)
            H_diag_k_plus_one = np.zeros(self.n)
        elif self.hessian_mode == HessianMode.LowRankDiagEVD:
            G_k_plus_one, H_diag_k_plus_one = self.curvature_update.low_rank_diag_update(x_k_plus_one)
        else:
            G_k_plus_one = np.zeros((self.n, self.hessian_rank))
            H_diag_k_plus_one = np.zeros(self.n)
        return G_k_plus_one, H_diag_k_plus_one

    def _update_l_k(self, iter_idx, x_k_plus_one_vec, f_k_plus_one, f_grad_k_plus_one_vec, f_grads_memory,
                    f_consts_memory, bundle_dual):
        if self.bundle_mode == BundleMode.AllActive:
            if bundle_dual is not None:
                active_cuts_idxs = np.where(bundle_dual >= 1e-3)[0]
                if len(active_cuts_idxs) < self.gradient_memory:
                    f_grads_memory[:, 0: len(active_cuts_idxs)] = f_grads_memory[:, active_cuts_idxs]
                    f_consts_memory[0: len(active_cuts_idxs)] = f_consts_memory[active_cuts_idxs]
                    f_grads_memory[:, len(active_cuts_idxs)] = f_grad_k_plus_one_vec
                    f_consts_memory[len(active_cuts_idxs): self.gradient_memory] = \
                        f_k_plus_one - f_grad_k_plus_one_vec.dot(x_k_plus_one_vec)
                    self.num_active_cuts = len(active_cuts_idxs) + 1
                else:
                    print("Not enough gradient memory to store all active cutting-planes. "
                          "Please increase the gradient memory size or switch to the latest M cutting-plane mode.")
                    min_dual_idx = np.argmin(bundle_dual)
                    f_grads_memory[:, min_dual_idx] = f_grad_k_plus_one_vec
                    f_consts_memory[min_dual_idx] = f_k_plus_one - f_grad_k_plus_one_vec.dot(x_k_plus_one_vec)
                    self.num_active_cuts = self.gradient_memory
            else:
                if self.num_active_cuts < self.gradient_memory:
                    f_grads_memory[:, self.num_active_cuts] = f_grad_k_plus_one_vec
                    f_consts_memory[self.num_active_cuts] = f_k_plus_one - f_grad_k_plus_one_vec.dot(x_k_plus_one_vec)
                    self.num_active_cuts += 1
                else:
                    f_grads_memory[:, iter_idx % self.gradient_memory] = f_grad_k_plus_one_vec
                    f_consts_memory[iter_idx % self.gradient_memory] = f_k_plus_one - f_grad_k_plus_one_vec.dot(
                        x_k_plus_one_vec)
                    self.num_active_cuts = self.gradient_memory
        else:
            # if bundle_dual is not None:
            #     active_cuts_idxs = np.where(bundle_dual >= 1e-5)[0]
            #     self.num_active_cuts = len(active_cuts_idxs)
            #     if iter_idx % self.gradient_memory in active_cuts_idxs:
            #         print("active cuts being removed")
            f_grads_memory[:, iter_idx % self.gradient_memory] = f_grad_k_plus_one_vec
            f_consts_memory[iter_idx % self.gradient_memory] = f_k_plus_one - f_grad_k_plus_one_vec.dot(x_k_plus_one_vec)
        return f_grads_memory, f_consts_memory

    def _update_trust_params(self, mu_k, tk, tau_k_plus_one):
        if self.trust_param_zero:
            lam_k_plus_one = 0
            mu_k_plus_one = 0
        else:
            if tk >= 0.99:
                mu_k_plus_one = max(mu_k * self.gamma_dec, self.mu_min)
            else:
                mu_k_plus_one = min(mu_k * self.gamma_inc, self.mu_max)
            lam_k_plus_one = mu_k_plus_one * max(self.tau_min, tau_k_plus_one)
        return lam_k_plus_one, mu_k_plus_one

    def _get_subproblems(self, iter_idx, xk, f_grads_memory, f_consts_memory, lam_k, G_k, H_diag_k):
        if self.subprobs_param is not None:
            if self.solver == "OSQP":
                tentative_update_subp = cp.Problem(cp.Minimize(self.subprobs_param.f_hat_k + self.g_cvxpy.objective
                                                               + self.subprobs_param.trust_penalty),
                                                   self.g_cvxpy.constraints + self.subprobs_param.bundle_constr)
                lower_bound_subp = cp.Problem(cp.Minimize(self.subprobs_param.l_k + self.g_cvxpy.objective),
                                              self.g_cvxpy.constraints + self.subprobs_param.bundle_constr)
            else:
                tentative_update_subp = self.subprobs_param.cvxpy_subp
                lower_bound_subp = self.subprobs_param.lower_bound_subp
            self.subprobs_param.f_grads_iters_para.value = f_grads_memory
            self.subprobs_param.f_const_iters_para.value = f_consts_memory
            self.subprobs_param.sqrt_lam_para.value = np.sqrt(lam_k)
            self.subprobs_param.x_prev_times_sqrt_lam_para.value = xk * np.sqrt(lam_k)
            self.subprobs_param.G_para.value = G_k
            self.subprobs_param.G_T_x_prev_para.value = G_k.T.dot(xk.flatten(order='F'))
            self.subprobs_param.sqrt_H_diag_para.value = np.sqrt(H_diag_k)
            self.subprobs_param.sqrt_H_diag_mul_x_prev_para.value = np.sqrt(H_diag_k) * xk.flatten(order='F')
            bundle_constr = self.subprobs_param.bundle_constr
        else:
            l_k = cp.Variable()
            trust_penalty = 0.5 * cp.sum_squares(self.g_cvxpy.variable * np.sqrt(lam_k) - xk * np.sqrt(lam_k))
            if len(self.g_cvxpy.variable.shape) <= 1:
                f_curvature = 0.5 * cp.sum_squares(G_k.T @ self.g_cvxpy.variable - G_k.T.dot(xk)) \
                              + 0.5 * cp.sum_squares(cp.multiply(self.g_cvxpy.variable, np.sqrt(H_diag_k))
                                                     - xk * np.sqrt(H_diag_k))
                if self.bundle_mode == BundleMode.LatestM and iter_idx < self.gradient_memory:
                    bundle_constr = [f_grads_memory[:, 0: iter_idx].T @ self.g_cvxpy.variable \
                                     + f_consts_memory[0: iter_idx] <= l_k]
                elif self.bundle_mode == BundleMode.AllActive and self.num_active_cuts < self.gradient_memory:
                    bundle_constr = [f_grads_memory[:, 0: self.num_active_cuts].T @ self.g_cvxpy.variable \
                                     + f_consts_memory[0: self.num_active_cuts] <= l_k]
                else:
                    bundle_constr = [f_grads_memory.T @ self.g_cvxpy.variable + f_consts_memory <= l_k]
            else:
                f_curvature = 0.5 * cp.sum_squares(G_k.T @ cp.vec(self.g_cvxpy.variable)
                                                   - G_k.T.dot(xk.flatten(order='F'))) \
                              + 0.5 * cp.sum_squares(cp.multiply(cp.vec(self.g_cvxpy.variable), np.sqrt(H_diag_k))
                                                     - xk.flatten(order='F') * np.sqrt(H_diag_k))
                if self.bundle_mode == BundleMode.LatestM and iter_idx < self.gradient_memory:
                    bundle_constr = [f_grads_memory[:, 0: iter_idx].T @ cp.vec(self.g_cvxpy.variable) \
                                     + f_consts_memory[0: iter_idx] <= l_k]
                elif self.bundle_mode == BundleMode.AllActive and self.num_active_cuts < self.gradient_memory:
                    bundle_constr = [f_grads_memory[:, 0: self.num_active_cuts].T @ cp.vec(self.g_cvxpy.variable) \
                                     + f_consts_memory[0: self.num_active_cuts] <= l_k]
                else:
                    bundle_constr = [f_grads_memory.T @ cp.vec(self.g_cvxpy.variable) + f_consts_memory <= l_k]
            tentative_update_subp = cp.Problem(cp.Minimize(l_k + f_curvature + self.g_cvxpy.objective + trust_penalty),
                                               self.g_cvxpy.constraints + bundle_constr)
            lower_bound_subp = cp.Problem(cp.Minimize(l_k + self.g_cvxpy.objective),
                                          self.g_cvxpy.constraints + bundle_constr)
        return tentative_update_subp, lower_bound_subp, bundle_constr

    def update_func(self, iter_idx, objf_k, f_k, g_k, lower_bound_k, f_grad_k, f_grads_memory, f_consts_memory, G_k,
                    H_diag_k, lam_k, mu_k, ep):
        if self.store_var_all_iters:
            if self.n1 == 0:
                xk = np.array(self.method_results["var_iters"][:, iter_idx - 1])
            else:
                xk = np.array(self.method_results["var_iters"][:, :, iter_idx - 1])
        else:
            if self.n1 == 0:
                xk = np.array(self.method_results["var_iters"][:, 0])
            else:
                xk = np.array(self.method_results["var_iters"][:, :, 0])

        # tentative update
        tentative_update_subp, lower_bound_subp, bundle_constr = \
            self._get_subproblems(iter_idx, xk, f_grads_memory, f_consts_memory, lam_k, G_k, H_diag_k)
        begin_solve_time = time.time()
        subp_solver_success = True
        var_val_prev = [var.value for var in self.g_cvxpy.all_var_list]
        try:
            tentative_update_subp.solve(solver=self.solver)
            if (tentative_update_subp.status != "optimal" and tentative_update_subp.status != "inaccurate_optimal") or \
                    self.g_cvxpy.variable.value is None:
                subp_solver_success = False
        except Exception as e:
            subp_solver_success = False
            if self.verbose:
                print("tentative update error:", e)
        bundle_dual = bundle_constr[0].dual_value
        if not subp_solver_success:
            for i in range(len(var_val_prev)):
                var = self.g_cvxpy.all_var_list[i]
                var.value = var_val_prev[i]
        end_solve_time = time.time()
        x_k_plus_half = self.g_cvxpy.variable.value
        v_k = x_k_plus_half - xk
        if self.n1 == 0:
            v_k_vec = v_k
        else:
            v_k_vec = v_k.flatten(order='F')
        g_k_plus_half = self.g_cvxpy.objective.value  # no need to optimize over hidden var again

        # line search and evaluate g at x_k_plus_one
        if subp_solver_success:
            if self.exact_g_line_search:
                x_k_plus_one, f_k_plus_one, tk, num_f_evals, f_eval_time_cost, g_k_plus_one = \
                    self._line_search_exact_g(x_k_plus_half, xk, v_k_vec, g_k_plus_half, objf_k, G_k, H_diag_k, lam_k,
                                              var_val_prev)
            else:
                x_k_plus_one, f_k_plus_one, tk, num_f_evals, f_eval_time_cost \
                    = self._line_search(x_k_plus_half, xk, v_k_vec, g_k_plus_half, g_k, objf_k, G_k, H_diag_k, lam_k)
                if tk < 1.0:
                    for i in range(len(var_val_prev)):
                        if var_val_prev[i] is not None:
                            var = self.g_cvxpy.all_var_list[i]
                            var.value = tk * var.value + (1 - tk) * var_val_prev[i]
                    g_k_plus_one = self.g_cvxpy.g_value()
                else:
                    g_k_plus_one = g_k_plus_half
        else:
            x_k_plus_one = xk
            f_k_plus_one = f_k
            tk = 0
            num_f_evals = 0
            f_eval_time_cost = 0
            g_k_plus_one = g_k
        objf_k_plus_one = f_k_plus_one + g_k_plus_one

        # update best result found
        if objf_k_plus_one < np.min(self.method_results["objf_iters"][0:iter_idx]):
            self.method_results["soln"] = x_k_plus_one
            for var in self.g_cvxpy.additional_var_soln:
                self.g_cvxpy.additional_var_soln[var] = var.value

        # evaluate validate function
        if self.f_torch.W_validate_torch is not None:
            f_validation_k_plus_one = self.f_torch.f_validate_value(x_k_plus_one)
            objf_validation_k_plus_one = f_validation_k_plus_one + g_k_plus_one
        else:
            objf_validation_k_plus_one = None

        # update q and residual
        begin_evaluate_f_grad_time = time.time()
        f_grad_k_plus_one = self.f_torch.f_grad_value(x_k_plus_one)
        end_evaluate_f_grad_time = time.time()
        if self.n1 == 0:
            f_grad_k_plus_one_vec = f_grad_k_plus_one
            x_k_plus_one_vec = x_k_plus_one
        else:
            f_grad_k_plus_one_vec = f_grad_k_plus_one.flatten(order='F')
            x_k_plus_one_vec = x_k_plus_one.flatten(order='F')
        if bundle_dual is None:
            q_k_plus_one_vec = np.inf
            rms_res = np.inf
        else:
            if self.subprobs_param is None and self.bundle_mode == BundleMode.LatestM and iter_idx < self.gradient_memory:
                q_k_plus_one_vec = - G_k.dot(G_k.T.dot(v_k_vec)) - H_diag_k * v_k_vec - lam_k * v_k_vec \
                                   - f_grads_memory[:, 0:iter_idx].dot(bundle_dual)
            elif self.bundle_mode == BundleMode.AllActive and self.num_active_cuts < self.gradient_memory:
                q_k_plus_one_vec = - G_k.dot(G_k.T.dot(v_k_vec)) - H_diag_k * v_k_vec - lam_k * v_k_vec \
                                   - f_grads_memory[:, 0:self.num_active_cuts].dot(bundle_dual)
            else:
                q_k_plus_one_vec = - G_k.dot(G_k.T.dot(v_k_vec)) - H_diag_k * v_k_vec - lam_k * v_k_vec \
                                    - f_grads_memory.dot(bundle_dual)
            rms_res = np.linalg.norm(f_grad_k_plus_one_vec + q_k_plus_one_vec) / np.sqrt(self.n)

        # update G_k and H_diag_k
        begin_curve_time = time.time()
        if iter_idx % self.update_curvature_frequency == 0:
            G_k_plus_one, H_diag_k_plus_one = self._update_curvature_after_solve(G_k, x_k_plus_one, x_k_plus_one_vec,
                                                                                 xk.flatten(order='F'),
                                                                                 f_grad_k_plus_one_vec,
                                                                                 f_grad_k.flatten(order='F'))
        else:
            G_k_plus_one = G_k
            H_diag_k_plus_one = H_diag_k
        end_curve_time = time.time()

        # update trust param
        tr_H_k_plus_one = np.square(max(ep, np.linalg.norm(G_k_plus_one, 'fro')))
        tau_k_plus_one = tr_H_k_plus_one / self.n / min(self.hessian_rank, iter_idx)
        lam_k_plus_one, mu_k_plus_one = self._update_trust_params(mu_k, tk, tau_k_plus_one)

        # update lower bound and print info
        begin_eval_L_k_time = time.time()
        lower_bound_k_plus_one = lower_bound_k
        if iter_idx % self.check_gap_frequency == 0:
            var_val_k_plus_one = [var.value for var in self.g_cvxpy.all_var_list]
            try:
                lower_bound_subp.solve(solver=self.solver)
                lower_bound_k_plus_one = max(lower_bound_k, lower_bound_subp.value)
            except Exception as e:
                if self.verbose:
                    print("lower bound problem error:", e)
                lower_bound_k_plus_one = lower_bound_k
            if self.verbose:
                if self.f_torch.W_validate_torch is not None:
                    print_str = "iter = {}, objf = {:.3e}, lower bound = {:.3e}, RMS residual = {:.3e}, " \
                                "sampling acc = {:.3e}, ||G||_F = {:.3e}"
                    print(print_str.format(iter_idx, objf_k_plus_one, lower_bound_k_plus_one, rms_res,
                                           np.abs(objf_validation_k_plus_one - objf_k_plus_one),
                                           np.linalg.norm(G_k_plus_one, 'fro')))
                else:
                    print_str = "iter = {}, objf = {:.3e}, lower bound = {:.3e}, RMS residual = {:.3e}, " \
                                "||G||_F = {:.3e}"
                    print(print_str.format(iter_idx, objf_k_plus_one, lower_bound_k_plus_one, rms_res,
                                           np.linalg.norm(G_k_plus_one, 'fro')))
            for i in range(len(var_val_k_plus_one)):
                self.g_cvxpy.all_var_list[i].value = var_val_k_plus_one[i]
        end_eval_L_k_time = time.time()

        # update l_k
        f_grads_memory, f_consts_memory = self._update_l_k(iter_idx, x_k_plus_one_vec, f_k_plus_one,
                                                           f_grad_k_plus_one_vec, f_grads_memory, f_consts_memory,
                                                           bundle_dual)

        # check stopping criteria
        stopping_criteria_satisfied = self._stopping_criteria(objf_k_plus_one, objf_validation_k_plus_one,
                                                              lower_bound_k_plus_one, tk, rms_res,
                                                              np.linalg.norm(q_k_plus_one_vec),
                                                              np.linalg.norm(f_grad_k_plus_one))

        # save update
        self.method_results["time_detail_iters"][0, iter_idx] = f_eval_time_cost
        self.method_results["time_detail_iters"][1, iter_idx] = end_evaluate_f_grad_time - begin_evaluate_f_grad_time
        self.method_results["time_detail_iters"][2, iter_idx] = end_solve_time - begin_solve_time
        self.method_results["time_detail_iters"][3, iter_idx] = end_eval_L_k_time - begin_eval_L_k_time
        self.method_results["time_detail_iters"][4, iter_idx] = end_curve_time - begin_curve_time
        if self.store_var_all_iters:
            if self.n1 == 0:
                self.method_results["var_iters"][:, iter_idx] = x_k_plus_one
            else:
                self.method_results["var_iters"][:, :, iter_idx] = x_k_plus_one
        else:
            if self.n1 == 0:
                self.method_results["var_iters"][:, 0] = x_k_plus_one
            else:
                self.method_results["var_iters"][:, :, 0] = x_k_plus_one
        self.method_results["v_norm_iters"][iter_idx] = np.linalg.norm(v_k_vec)
        self.method_results["objf_iters"][iter_idx] = objf_k_plus_one
        self.method_results["objf_validate_iters"][iter_idx] = objf_validation_k_plus_one
        self.method_results["num_f_evals_iters"][iter_idx] = num_f_evals
        self.method_results["q_norm_iters"][iter_idx] = np.linalg.norm(q_k_plus_one_vec)
        self.method_results["f_grad_norm_iters"][iter_idx] = np.linalg.norm(f_grad_k_plus_one)
        self.method_results["rms_res_iters"][iter_idx] = rms_res
        self.method_results["lam_iters"][iter_idx] = lam_k_plus_one
        self.method_results["mu_iters"][iter_idx] = mu_k_plus_one
        self.method_results["t_iters"][iter_idx] = tk
        self.method_results["lower_bound_iters"][iter_idx] = lower_bound_k_plus_one
        self.method_results["total_iters"] = iter_idx

        return stopping_criteria_satisfied, x_k_plus_one, objf_k_plus_one, f_k_plus_one, g_k_plus_one, \
               lower_bound_k_plus_one, f_grad_k_plus_one, f_grads_memory, f_consts_memory, G_k_plus_one, \
               H_diag_k_plus_one, lam_k_plus_one, mu_k_plus_one