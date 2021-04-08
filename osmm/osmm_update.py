import numpy as np
import cvxpy as cp
import time

from .subproblem import Subproblem
from .curvature_updates import CurvatureUpdate
from .alg_mode import AlgMode


class OsmmUpdate:
    def __init__(self, osmm_problem):
        self.osmm_problem = osmm_problem
        self.curvature_update = CurvatureUpdate(osmm_problem)

    def initialization(self, init_val, hessian_rank, gradient_memory, tau_min, init_by_Hutchison):
        x_0 = init_val
        f_0 = self.osmm_problem.f_value(x_0)
        f_grad_0 = self.osmm_problem.f_grad_value(x_0)

        subprobs = Subproblem(self.osmm_problem.x_var_cvxpy, self.osmm_problem.g_constrs, self.osmm_problem.g_objf,
                              hessian_rank=hessian_rank, gradient_memory=gradient_memory)
        subprobs.x_for_g_para.value = x_0
        try:
            subprobs.g_eval_subp.solve()
            if subprobs.g_eval_subp.value is not None and subprobs.g_eval_subp.value < np.inf:
                g_0 = self.osmm_problem.g_objf.value
            else:
                g_0 = np.inf
        except Exception as e:
            g_0 = np.inf

        objf_0 = f_0 + g_0
        if objf_0 < np.inf:
            self.osmm_problem.method_results["soln"] = x_0
            for var in self.osmm_problem.g_additional_var_val:
                self.osmm_problem.g_additional_var_val[var] = var.value

        if self.osmm_problem.W_torch_validate is not None:
            f_validate_0 = self.osmm_problem.f_validate_value(x_0)
            objf_validate_0 = f_validate_0 + g_0
        else:
            objf_validate_0 = None

        if init_by_Hutchison:
            est_hess_tr = self.osmm_problem.f_hess_tr_Hutchinson(x_0, max_iter=100)
            lam_0 = max(tau_min, est_hess_tr / self.osmm_problem.n)
        else:
            lam_0 = tau_min

        if self.osmm_problem.n1 == 0:
            f_grads_memory = f_grad_0.repeat(gradient_memory).reshape((self.osmm_problem.n, gradient_memory), order='C')
            f_consts_memory = np.ones(gradient_memory) * (f_0 - f_grad_0.T.dot(x_0))
        else:
            f_grad_0_vec = f_grad_0.flatten(order='F')
            f_grads_memory = f_grad_0_vec.repeat(gradient_memory).reshape((self.osmm_problem.n, gradient_memory), order='C')
            f_consts_memory = np.ones(gradient_memory) * (f_0 - f_grad_0_vec.T.dot(x_0.flatten(order='F')))

        G_0 = np.zeros((self.osmm_problem.n, hessian_rank))
        return subprobs, x_0, objf_0, objf_validate_0, f_0, f_grad_0, g_0, lam_0, f_grads_memory, f_consts_memory, G_0

    def _stopping_criteria(self, objf_k_plus_one, objf_validation_k_plus_one, L_k_plus_one, t_k,
                           rms_res_k_plus_one, q_norm_k_plus_one, f_grad_norm_k_plus_one,
                           eps_gap_abs, eps_gap_rel, eps_res_abs, eps_res_rel):
        if objf_validation_k_plus_one is not None:
            eps_gap_abs = np.abs(objf_k_plus_one - objf_validation_k_plus_one)
        if objf_k_plus_one is not None and objf_k_plus_one < np.inf:
            if objf_k_plus_one - L_k_plus_one <= eps_gap_abs + eps_gap_rel * np.abs(objf_k_plus_one):
                return True
        if t_k == 1 and rms_res_k_plus_one <= eps_res_abs \
                + eps_res_rel * (q_norm_k_plus_one + f_grad_norm_k_plus_one) / np.sqrt(self.osmm_problem.n):
            return True
        return False

    def _line_search(self, x_k_plus_half, xk, v_k_vec, g_k_plus_half, g_k, objf_k, G_k, lam_k, beta, j_max, alpha,
                     ep=1e-15):
        begin_evaluate_f_time = time.time()
        f_x_k_plus_half = self.osmm_problem.f_value(x_k_plus_half)
        end_evaluate_f_time = time.time()

        # tol=1e-5
        # if f_x_k_plus_half < np.inf and np.linalg.norm(v_k) <= tol * np.linalg.norm(xk) and \
        #                                  f_x_k_plus_half + g_k_plus_half - objf_k <= tol * np.abs(objf_k):
        #     print("t = 1 because of too small increment")
        #     return x_k_plus_half, f_x_k_plus_half, 1.0, 0, end_evaluate_f_time - begin_evaluate_f_time

        desc = np.square(max(ep, np.linalg.norm(G_k.T.dot(v_k_vec)))) + lam_k * np.square(max(ep, np.linalg.norm(v_k_vec)))
        f_tmp = f_x_k_plus_half
        phi_line_search = f_tmp + g_k_plus_half
        t = 1.0
        j = 0
        while j < j_max and phi_line_search > objf_k - 0.5 * alpha * t * desc:
            t = t * beta
            f_tmp = self.osmm_problem.f_value(t * x_k_plus_half + (1 - t) * xk)
            phi_line_search = f_tmp + t * g_k_plus_half + (1 - t) * g_k
            j += 1
        return t * x_k_plus_half + (1 - t) * xk, f_tmp, t, j + 1, end_evaluate_f_time - begin_evaluate_f_time

    def _update_curvature_after_solve(self, alg_mode, G_k, x_k_plus_one_vec, xk_vec, f_grad_k_plus_one_vec,
                                      f_grad_k_vec, hessian_rank):
        if alg_mode == AlgMode.LowRankQNBundle:
            G_k_plus_one = self.curvature_update.low_rank_quasi_Newton_update(G_k, x_k_plus_one_vec, xk_vec,
                                                                              f_grad_k_plus_one_vec, f_grad_k_vec,
                                                                              hessian_rank)
        else:
            G_k_plus_one = np.zeros((self.osmm_problem.n, hessian_rank))
        return G_k_plus_one

    def _update_l_k(self, iter_idx, gradient_memory, x_k_plus_one_vec, f_k_plus_one, f_grad_k_plus_one_vec,
                    f_grads_memory, f_consts_memory):
        if iter_idx < gradient_memory:
            num_iters_remain = gradient_memory - iter_idx
            f_grads_memory[:, iter_idx: gradient_memory] = \
                f_grad_k_plus_one_vec.repeat(num_iters_remain).reshape((self.osmm_problem.n, num_iters_remain), order='C')
            f_consts_memory[iter_idx: gradient_memory] = np.ones(num_iters_remain) * \
                                                         (f_k_plus_one - f_grad_k_plus_one_vec.dot(x_k_plus_one_vec))
        else:
            f_grads_memory[:, iter_idx % gradient_memory] = f_grad_k_plus_one_vec
            f_consts_memory[iter_idx % gradient_memory] = f_k_plus_one - f_grad_k_plus_one_vec.dot(x_k_plus_one_vec)
        return f_grads_memory, f_consts_memory

    def _update_trust_params(self, mu_k, tk, tau_k_plus_one, tau_min, mu_min, mu_max, gamma_inc, gamma_dec):
        if tk >= 0.99:
            mu_k_plus_one = max(mu_k * gamma_dec, mu_min)
        else:
            mu_k_plus_one = min(mu_k * gamma_inc, mu_max)
        lam_k_plus_one = mu_k_plus_one * max(tau_min, tau_k_plus_one)
        return lam_k_plus_one, mu_k_plus_one

    def update_func(self, subprobs, iter_idx, objf_k, g_k, lower_bound_k, f_grad_k,
                    f_grads_memory, f_consts_memory, G_k, lam_k, mu_k, verbose=False,
                    alg_mode=None, hessian_rank=None, gradient_memory=None, solver=None,
                    gamma_inc=None, gamma_dec=None, mu_min=None, tau_min=None, mu_max=None, ep=None,
                    beta=None, j_max=None, alpha=None, check_gap_frequency=None,
                    eps_gap_abs=None, eps_gap_rel=None, eps_res_abs=None, eps_res_rel=None):
        if solver == "OSQP":
            cvxpy_subp = cp.Problem(cp.Minimize(subprobs.f_hat_k + self.osmm_problem.g_objf + subprobs.trust_penalty),
                                    self.osmm_problem.g_constrs + subprobs.bundle_constr)
            lower_bound_subp = cp.Problem(cp.Minimize(subprobs.l_k + self.osmm_problem.g_objf),
                                          self.osmm_problem.g_constrs + subprobs.bundle_constr)
            g_eval_subp = cp.Problem(cp.Minimize(self.osmm_problem.g_objf), self.osmm_problem.g_constrs +
                                     [self.osmm_problem.x_var_cvxpy == subprobs.x_for_g_para])
        else:
            cvxpy_subp = subprobs.cvxpy_subp
            lower_bound_subp = subprobs.lower_bound_subp
            g_eval_subp = subprobs.g_eval_subp

        if self.osmm_problem.store_var_all_iters:
            if self.osmm_problem.n1 == 0:
                xk = np.array(self.osmm_problem.method_results["var_iters"][:, iter_idx - 1])
            else:
                xk = np.array(self.osmm_problem.method_results["var_iters"][:, :, iter_idx - 1])
        else:
            if self.osmm_problem.n1 == 0:
                xk = np.array(self.osmm_problem.method_results["var_iters"][:, 0])
            else:
                xk = np.array(self.osmm_problem.method_results["var_iters"][:, :, 0])
        subprobs.x_prev_para.value = xk
        subprobs.f_grads_iters_para.value = f_grads_memory
        subprobs.f_const_iters_para.value = f_consts_memory
        subprobs.sqrt_lam_para.value = np.sqrt(lam_k)
        subprobs.x_prev_times_sqrt_lam_para.value = xk * np.sqrt(lam_k)
        subprobs.G_para.value = G_k
        subprobs.G_T_x_prev_para.value = G_k.T.dot(xk.flatten(order='F'))

        begin_solve_time = time.time()
        subp_solver_success = True
        try:
            cvxpy_subp.solve(solver=solver, verbose=False)
        except Exception as e:
            subp_solver_success = False
            if verbose:
                print("tentative update error:", e)
        if self.osmm_problem.x_var_cvxpy.value is None:
            self.osmm_problem.x_var_cvxpy.value = xk
        end_solve_time = time.time()

        x_k_plus_half = self.osmm_problem.x_var_cvxpy.value
        v_k = x_k_plus_half - xk
        g_k_plus_half = self.osmm_problem.g_objf.value
        bundle_dual = subprobs.bundle_constr[0].dual_value

        if self.osmm_problem.n1 == 0:
            v_k_vec = v_k
        else:
            v_k_vec = v_k.flatten(order='F')

        if subp_solver_success:
            x_k_plus_one, f_k_plus_one, tk, num_f_evals, f_eval_time_cost \
                = self._line_search(x_k_plus_half, xk, v_k_vec, g_k_plus_half, g_k, objf_k, G_k, lam_k, beta, j_max, alpha)
        else:
            x_k_plus_one = xk
            f_k_plus_one = objf_k - g_k
            tk = 0
            num_f_evals = 0
            f_eval_time_cost = 0

        self.osmm_problem.x_var_cvxpy.value = x_k_plus_one
        if len(self.osmm_problem.g_additional_var_val) > 0:
            ub_g_k_plus_one = self.osmm_problem.g_objf.value
            subprobs.x_for_g_para.value = x_k_plus_one
            tmp_var_val = [var.value for var in subprobs.g_eval_subp.variables()]
            g_eval_success = True
            try:
                g_eval_subp.solve(solver=solver)
                if self.osmm_problem.g_objf is None or self.osmm_problem.g_objf.value == np.inf:
                    g_eval_success = False
            except Exception as e:
                g_eval_success = False
            if g_eval_success:
                g_k_plus_one = self.osmm_problem.g_objf.value
            else:
                g_k_plus_one = ub_g_k_plus_one
                for i in range(len(tmp_var_val)):
                    var = subprobs.g_eval_subp.variables()[i]
                    var.value = tmp_var_val[i]
        else:
            g_k_plus_one = self.osmm_problem.g_objf.value
        objf_k_plus_one = f_k_plus_one + g_k_plus_one

        if objf_k_plus_one < np.min(self.osmm_problem.method_results["objf_iters"][0:iter_idx]):
            self.osmm_problem.method_results["soln"] = x_k_plus_one
            for var in self.osmm_problem.g_additional_var_val:
                self.osmm_problem.g_additional_var_val[var] = var.value

        if self.osmm_problem.W_torch_validate is not None:
            f_validation_k_plus_one = self.osmm_problem.f_validate_value(x_k_plus_one)
            objf_validation_k_plus_one = f_validation_k_plus_one + g_k_plus_one
        else:
            objf_validation_k_plus_one = None

        begin_evaluate_f_grad_time = time.time()
        f_grad_k_plus_one = self.osmm_problem.f_grad_value(x_k_plus_one)
        end_evaluate_f_grad_time = time.time()

        if self.osmm_problem.n1 == 0:
            f_grad_k_plus_one_vec = f_grad_k_plus_one
            x_k_plus_one_vec = x_k_plus_one
        else:
            f_grad_k_plus_one_vec = f_grad_k_plus_one.flatten(order='F')
            x_k_plus_one_vec = x_k_plus_one.flatten(order='F')

        if bundle_dual is None:
            q_k_plus_one_vec = np.inf
            rms_res = np.inf
        else:
            if gradient_memory == 1:
                q_k_plus_one_vec = - G_k.dot(G_k.T.dot(v_k_vec)) - lam_k * v_k_vec - f_grads_memory[:, 0]
            else:
                q_k_plus_one_vec = - G_k.dot(G_k.T.dot(v_k_vec)) - lam_k * v_k_vec - f_grads_memory.dot(bundle_dual)
            rms_res = np.linalg.norm(f_grad_k_plus_one_vec + q_k_plus_one_vec) / np.sqrt(self.osmm_problem.n)

        f_grads_memory, f_consts_memory = self._update_l_k(iter_idx, gradient_memory, x_k_plus_one_vec, f_k_plus_one,
                                                           f_grad_k_plus_one_vec, f_grads_memory, f_consts_memory)

        G_k_plus_one = self._update_curvature_after_solve(alg_mode, G_k, x_k_plus_one_vec, xk.flatten(order='F'),
                                                          f_grad_k_plus_one_vec, f_grad_k.flatten(order='F'), hessian_rank)

        tr_H_k_plus_one = np.square(max(ep, np.linalg.norm(G_k_plus_one, 'fro')))
        tau_k_plus_one = tr_H_k_plus_one / self.osmm_problem.n / hessian_rank

        lam_k_plus_one, mu_k_plus_one = self._update_trust_params(mu_k, tk, tau_k_plus_one,
                                                                  tau_min, mu_min, mu_max, gamma_inc, gamma_dec)

        begin_eval_L_k_time = 0
        end_eval_L_k_time = 0
        lower_bound_k_plus_one = lower_bound_k
        if iter_idx % check_gap_frequency == 0:
            begin_eval_L_k_time = time.time()
            try:
                lower_bound_subp.solve(solver=solver)
                lower_bound_k_plus_one = max(lower_bound_k, lower_bound_subp.value)
            except Exception as e:
                if verbose:
                    print("lower bound problem error:", e)
                lower_bound_k_plus_one = lower_bound_k
            end_eval_L_k_time = time.time()
            if verbose:
                if self.osmm_problem.W_torch_validate is not None:
                    print_str = "iter = {}, objf = {:.3e}, lower bound = {:.3e}, RMS residual = {:.3e}, sampling acc = {:.3e}, ||G||_F = {:.3e}"
                    print(print_str.format(iter_idx, objf_k_plus_one, lower_bound_k_plus_one, rms_res,
                                           np.abs(objf_validation_k_plus_one - objf_k_plus_one),
                                           np.linalg.norm(G_k_plus_one, 'fro')))
                else:
                    print_str = "iter = {}, objf = {:.3e}, lower bound = {:.3e}, RMS residual = {:.3e}, ||G||_F = {:.3e}"
                    print(print_str.format(iter_idx, objf_k_plus_one, lower_bound_k_plus_one, rms_res,
                                           np.linalg.norm(G_k_plus_one, 'fro')))

        stopping_criteria_satisfied = self._stopping_criteria(objf_k_plus_one, objf_validation_k_plus_one,
                                                              lower_bound_k_plus_one, tk, rms_res,
                                                              np.linalg.norm(q_k_plus_one_vec),
                                                              np.linalg.norm(f_grad_k_plus_one),
                                                              eps_gap_abs, eps_gap_rel, eps_res_abs, eps_res_rel)

        self.osmm_problem.method_results["time_detail_iters"][0, iter_idx] = f_eval_time_cost
        self.osmm_problem.method_results["time_detail_iters"][1, iter_idx] = end_evaluate_f_grad_time - begin_evaluate_f_grad_time
        self.osmm_problem.method_results["time_detail_iters"][2, iter_idx] = end_solve_time - begin_solve_time
        self.osmm_problem.method_results["time_detail_iters"][3, iter_idx] = end_eval_L_k_time - begin_eval_L_k_time
        if self.osmm_problem.store_var_all_iters:
            if self.osmm_problem.n1 == 0:
                self.osmm_problem.method_results["var_iters"][:, iter_idx] = x_k_plus_one
            else:
                self.osmm_problem.method_results["var_iters"][:, :, iter_idx] = x_k_plus_one
        else:
            if self.osmm_problem.n1 == 0:
                self.osmm_problem.method_results["var_iters"][:, 0] = x_k_plus_one
            else:
                self.osmm_problem.method_results["var_iters"][:, :, 0] = x_k_plus_one
        self.osmm_problem.method_results["v_norm_iters"][iter_idx] = np.linalg.norm(v_k_vec)
        self.osmm_problem.method_results["objf_iters"][iter_idx] = objf_k_plus_one
        if self.osmm_problem.W_torch_validate is not None:
            self.osmm_problem.method_results["objf_validate_iters"][iter_idx] = objf_validation_k_plus_one
        self.osmm_problem.method_results["num_f_evals_iters"][iter_idx] = num_f_evals
        self.osmm_problem.method_results["q_norm_iters"][iter_idx] = np.linalg.norm(q_k_plus_one_vec)
        self.osmm_problem.method_results["f_grad_norm_iters"][iter_idx] = np.linalg.norm(f_grad_k_plus_one)
        self.osmm_problem.method_results["rms_res_iters"][iter_idx] = rms_res
        self.osmm_problem.method_results["lam_iters"][iter_idx] = lam_k_plus_one
        self.osmm_problem.method_results["mu_iters"][iter_idx] = mu_k_plus_one
        self.osmm_problem.method_results["t_iters"][iter_idx] = tk
        self.osmm_problem.method_results["lower_bound_iters"][iter_idx] = lower_bound_k_plus_one
        self.osmm_problem.method_results["total_iters"] = iter_idx

        return stopping_criteria_satisfied, x_k_plus_one, objf_k_plus_one, g_k_plus_one, lower_bound_k_plus_one, \
               f_grad_k_plus_one, f_grads_memory, f_consts_memory, G_k_plus_one, lam_k_plus_one, mu_k_plus_one
