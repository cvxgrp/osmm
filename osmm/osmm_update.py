import numpy as np
import cvxpy as cp
import time

from .subproblem import Subproblem
from .curvature_updates import CurvatureUpdate
from .alg_mode import AlgMode


class OsmmUpdate:
    def __init__(self, osmm_problem, hessian_rank, gradient_memory, use_cvxpy_param, solver, tau_min, mu_min, mu_max,
                 gamma_inc, gamma_dec, alpha, beta, j_max, eps_gap_abs, eps_gap_rel, eps_res_abs, eps_res_rel,
                 verbose, alg_mode, check_gap_frequency, update_curvature_frequency, trust_param_zero):
        self.osmm_problem = osmm_problem
        self.curvature_update = CurvatureUpdate(osmm_problem, hessian_rank)
        if use_cvxpy_param:
            self.subprobs_param = Subproblem(osmm_problem.x_var_cvxpy, osmm_problem.g_constrs, osmm_problem.g_objf,
                                             hessian_rank=hessian_rank, gradient_memory=gradient_memory)
        else:
            self.subprobs_param = None
        self.hessian_rank = hessian_rank
        self.gradient_memory = gradient_memory
        self.solver = solver
        self.tau_min = tau_min
        self.eps_gap_abs = eps_gap_abs
        self.eps_gap_rel = eps_gap_rel
        self.eps_res_abs = eps_res_abs
        self.eps_res_rel = eps_res_rel
        self.verbose = verbose
        self.alg_mode = alg_mode
        self.check_gap_frequency = check_gap_frequency
        self.mu_min = mu_min
        self.mu_max = mu_max
        self.gamma_inc = gamma_inc
        self.gamma_dec = gamma_dec
        self.alpha = alpha
        self.beta = beta
        self.j_max = j_max
        self.trust_param_zero = trust_param_zero
        self.update_curvature_frequency = update_curvature_frequency

    def initialization(self, x_0, use_Hutchison_init):
        f_0 = self.osmm_problem.f_value(x_0)
        f_grad_0 = self.osmm_problem.f_grad_value(x_0)

        if self.subprobs_param is not None:
            self.subprobs_param.x_for_g_para.value = x_0
            g_eval_subp = self.subprobs_param.g_eval_subp
        else:
            g_eval_subp = cp.Problem(cp.Minimize(self.osmm_problem.g_objf),
                                     self.osmm_problem.g_constrs + [self.osmm_problem.x_var_cvxpy == x_0])
        try:
            g_eval_subp.solve(solver=self.solver)
            if g_eval_subp.status == "optimal" or g_eval_subp.status == "inaccurate_optimal":
                g_0 = self.osmm_problem.g_objf.value
            else:
                g_0 = np.inf
        except Exception as e:
            g_0 = np.inf
        objf_0 = f_0 + g_0
        if objf_0 < np.inf:
            self.osmm_problem.method_results["soln"] = x_0
            for var in self.osmm_problem.g_additional_var_soln:
                self.osmm_problem.g_additional_var_soln[var] = var.value

        if self.osmm_problem.W_torch_validate is not None:
            f_validate_0 = self.osmm_problem.f_validate_value(x_0)
            objf_validate_0 = f_validate_0 + g_0
        else:
            objf_validate_0 = None

        if use_Hutchison_init and (not self.trust_param_zero):
            est_hess_tr = self.osmm_problem.f_hess_tr_Hutchinson(x_0, max_iter=100)
            lam_0 = max(self.tau_min, est_hess_tr / self.osmm_problem.n)
        elif not self.trust_param_zero:
            lam_0 = self.tau_min
        else:
            lam_0 = 0

        if self.osmm_problem.n1 == 0:
            f_grad_0_vec = f_grad_0
            x_0_vec = x_0
        else:
            f_grad_0_vec = f_grad_0.flatten(order='F')
            x_0_vec = x_0.flatten(order='F')
        f_grads_memory = f_grad_0_vec.repeat(self.gradient_memory).reshape((self.osmm_problem.n, self.gradient_memory),
                                                                           order='C')
        f_consts_memory = np.ones(self.gradient_memory) * (f_0 - f_grad_0_vec.T.dot(x_0_vec))
        if self.alg_mode == AlgMode.ExactHessian:
            G_0 = self.curvature_update.exact_hess_update(x_0)
            H_diag_0 = np.zeros(self.osmm_problem.n)
        elif self.alg_mode == AlgMode.LowRankDiagHessian:
            G_0, H_diag_0 = self.curvature_update.low_rank_diag_update(x_0)
        else:
            G_0 = np.zeros((self.osmm_problem.n, self.hessian_rank))
            H_diag_0 = np.zeros(self.osmm_problem.n)
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
                + self.eps_res_rel * (q_norm_k_plus_one + f_grad_norm_k_plus_one) / np.sqrt(self.osmm_problem.n):
            return True
        return False

    def _line_search(self, x_k_plus_half, xk, v_k_vec, g_k_plus_half, g_k, objf_k, G_k, H_diag_k, lam_k, ep=1e-15):
        begin_evaluate_f_time = time.time()
        f_x_k_plus_half = self.osmm_problem.f_value(x_k_plus_half)
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
        while j < self.j_max and (f_tmp is None or np.isnan(f_tmp) or phi_line_search > objf_k - 0.5 * self.alpha * t * desc):
            t = t * self.beta
            f_tmp = self.osmm_problem.f_value(t * x_k_plus_half + (1 - t) * xk)
            phi_line_search = f_tmp + t * g_k_plus_half + (1 - t) * g_k
            j += 1
        return t * x_k_plus_half + (1 - t) * xk, f_tmp, t, j + 1, end_evaluate_f_time - begin_evaluate_f_time

    def _update_curvature_after_solve(self, G_k, x_k_plus_one_vec, xk_vec, f_grad_k_plus_one_vec, f_grad_k_vec):
        if self.alg_mode == AlgMode.LowRankQNBundle:
            G_k_plus_one = self.curvature_update.low_rank_quasi_Newton_update(G_k, x_k_plus_one_vec, xk_vec,
                                                                              f_grad_k_plus_one_vec, f_grad_k_vec)
            H_diag_k_plus_one = np.zeros(self.osmm_problem.n)
        elif self.alg_mode == AlgMode.ExactHessian:
            G_k_plus_one = self.curvature_update.exact_hess_update(x_k_plus_one_vec)  #####TODO: no vectorization
            H_diag_k_plus_one = np.zeros(self.osmm_problem.n)
        elif self.alg_mode == AlgMode.LowRankDiagHessian:
            G_k_plus_one, H_diag_k_plus_one = self.curvature_update.low_rank_diag_update(
                x_k_plus_one_vec)  # TODO: no vectorization
        else:
            G_k_plus_one = np.zeros((self.osmm_problem.n, self.hessian_rank))
            H_diag_k_plus_one = np.zeros(self.osmm_problem.n)
        return G_k_plus_one, H_diag_k_plus_one

    def _update_l_k(self, iter_idx, x_k_plus_one_vec, f_k_plus_one, f_grad_k_plus_one_vec, f_grads_memory,
                    f_consts_memory):
        if iter_idx < self.gradient_memory:
            num_iters_remain = self.gradient_memory - iter_idx
            f_grads_memory[:, iter_idx: self.gradient_memory] = \
                f_grad_k_plus_one_vec.repeat(num_iters_remain).reshape((self.osmm_problem.n, num_iters_remain), order='C')
            f_consts_memory[iter_idx: self.gradient_memory] = np.ones(num_iters_remain) * \
                                                              (f_k_plus_one - f_grad_k_plus_one_vec.dot(x_k_plus_one_vec))
        else:
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

    def _get_subproblems(self, xk, f_grads_memory, f_consts_memory, lam_k, G_k, H_diag_k):
        if self.subprobs_param is not None:  ####TODO: add H_diag
            if self.solver == "OSQP":
                tentative_update_subp = cp.Problem(cp.Minimize(self.subprobs_param.f_hat_k + self.osmm_problem.g_objf
                                                               + self.subprobs_param.trust_penalty),
                                                   self.osmm_problem.g_constrs + self.subprobs_param.bundle_constr)
                lower_bound_subp = cp.Problem(cp.Minimize(self.subprobs_param.l_k + self.osmm_problem.g_objf),
                                              self.osmm_problem.g_constrs + self.subprobs_param.bundle_constr)
                g_eval_subp = cp.Problem(cp.Minimize(self.osmm_problem.g_objf), self.osmm_problem.g_constrs +
                                         [self.osmm_problem.x_var_cvxpy == self.subprobs_param.x_for_g_para])
            else:
                tentative_update_subp = self.subprobs_param.cvxpy_subp
                lower_bound_subp = self.subprobs_param.lower_bound_subp
                g_eval_subp = self.subprobs_param.g_eval_subp
            self.subprobs_param.x_prev_para.value = xk
            self.subprobs_param.f_grads_iters_para.value = f_grads_memory
            self.subprobs_param.f_const_iters_para.value = f_consts_memory
            self.subprobs_param.sqrt_lam_para.value = np.sqrt(lam_k)
            self.subprobs_param.x_prev_times_sqrt_lam_para.value = xk * np.sqrt(lam_k)
            self.subprobs_param.G_para.value = G_k
            self.subprobs_param.G_T_x_prev_para.value = G_k.T.dot(xk.flatten(order='F'))
            bundle_constr = self.subprobs_param.bundle_constr
        else:
            l_k = cp.Variable()
            trust_penalty = 0.5 * cp.sum_squares(self.osmm_problem.x_var_cvxpy * np.sqrt(lam_k) - xk * np.sqrt(lam_k))
            if len(self.osmm_problem.x_var_cvxpy.shape) <= 1:
                f_curvature = 0.5 * cp.sum_squares(G_k.T @ self.osmm_problem.x_var_cvxpy - G_k.T.dot(xk)) \
                              + 0.5 * cp.sum_squares(cp.multiply(self.osmm_problem.x_var_cvxpy, np.sqrt(H_diag_k))
                                                     - xk * np.sqrt(H_diag_k))
                bundle_constr = [f_grads_memory.T @ self.osmm_problem.x_var_cvxpy + f_consts_memory <= l_k]
            else:
                f_curvature = 0.5 * cp.sum_squares(G_k.T @ cp.vec(self.osmm_problem.x_var_cvxpy)
                                                   - G_k.T.dot(xk.flatten(order='F'))) \
                              + 0.5 * cp.sum_squares(cp.multiply(cp.vec(self.osmm_problem.x_var_cvxpy), np.sqrt(H_diag_k))
                                                     - xk.flatten(order='F') * np.sqrt(H_diag_k))
                bundle_constr = [f_grads_memory.T @ cp.vec(self.osmm_problem.x_var_cvxpy) + f_consts_memory <= l_k]
            tentative_update_subp = cp.Problem(
                cp.Minimize(l_k + f_curvature + self.osmm_problem.g_objf + trust_penalty),
                self.osmm_problem.g_constrs + bundle_constr)
            lower_bound_subp = cp.Problem(cp.Minimize(l_k + self.osmm_problem.g_objf),
                                          self.osmm_problem.g_constrs + bundle_constr)
            g_eval_subp = None
        return tentative_update_subp, lower_bound_subp, g_eval_subp, bundle_constr

    def update_func(self, iter_idx, objf_k, f_k, g_k, lower_bound_k, f_grad_k, f_grads_memory, f_consts_memory, G_k, H_diag_k,
                    lam_k, mu_k, ep):
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

        # tentative update
        tentative_update_subp, lower_bound_subp, g_eval_subp, bundle_constr = \
            self._get_subproblems(xk, f_grads_memory, f_consts_memory, lam_k, G_k, H_diag_k)
        begin_solve_time = time.time()
        subp_solver_success = True
        try:
            tentative_update_subp.solve(solver=self.solver)
            if (tentative_update_subp.status != "optimal" and tentative_update_subp.status != "inaccurate_optimal") or \
                    self.osmm_problem.x_var_cvxpy.value is None:
                subp_solver_success = False
        except Exception as e:
            subp_solver_success = False
            if self.verbose:
                print("tentative update error:", e)
        if not subp_solver_success:
            self.osmm_problem.x_var_cvxpy.value = xk
        end_solve_time = time.time()
        x_k_plus_half = self.osmm_problem.x_var_cvxpy.value
        v_k = x_k_plus_half - xk
        if self.osmm_problem.n1 == 0:
            v_k_vec = v_k
        else:
            v_k_vec = v_k.flatten(order='F')
        g_k_plus_half = self.osmm_problem.g_objf.value
        bundle_dual = bundle_constr[0].dual_value

        # line search
        if subp_solver_success:
            x_k_plus_one, f_k_plus_one, tk, num_f_evals, f_eval_time_cost \
                = self._line_search(x_k_plus_half, xk, v_k_vec, g_k_plus_half, g_k, objf_k, G_k, H_diag_k, lam_k)
        else:
            x_k_plus_one = xk
            f_k_plus_one = f_k
            tk = 0
            num_f_evals = 0
            f_eval_time_cost = 0

        # evaluate g at x_k_plus_one
        self.osmm_problem.x_var_cvxpy.value = x_k_plus_one
        if subp_solver_success and len(self.osmm_problem.g_additional_var_soln) > 0:
            ub_g_k_plus_one = self.osmm_problem.g_objf.value
            if self.subprobs_param is not None:
                self.subprobs_param.x_for_g_para.value = x_k_plus_one
            else:
                g_eval_subp = cp.Problem(cp.Minimize(self.osmm_problem.g_objf),
                                         self.osmm_problem.g_constrs + [self.osmm_problem.x_var_cvxpy == x_k_plus_one])
            tmp_var_val = [var.value for var in g_eval_subp.variables()]
            g_eval_success = True
            try:
                g_eval_subp.solve(solver=self.solver)
                if g_eval_subp.status != "optimal" and g_eval_subp.status != "inaccurate_optimal":
                    g_eval_success = False
                if self.osmm_problem.g_objf is None or self.osmm_problem.g_objf.value == np.inf:
                    g_eval_success = False
            except Exception as e:
                g_eval_success = False
            if g_eval_success:
                g_k_plus_one = self.osmm_problem.g_objf.value
            else:
                g_k_plus_one = ub_g_k_plus_one
                for i in range(len(tmp_var_val)):
                    var = g_eval_subp.variables()[i]
                    var.value = tmp_var_val[i]
        elif subp_solver_success:
            g_k_plus_one = self.osmm_problem.g_objf.value
        else:
            g_k_plus_one = g_k
        objf_k_plus_one = f_k_plus_one + g_k_plus_one

        # update best result found
        if objf_k_plus_one < np.min(self.osmm_problem.method_results["objf_iters"][0:iter_idx]):
            self.osmm_problem.method_results["soln"] = x_k_plus_one
            for var in self.osmm_problem.g_additional_var_soln:
                self.osmm_problem.g_additional_var_soln[var] = var.value

        # evaluate validate function
        if self.osmm_problem.W_torch_validate is not None:
            f_validation_k_plus_one = self.osmm_problem.f_validate_value(x_k_plus_one)
            objf_validation_k_plus_one = f_validation_k_plus_one + g_k_plus_one
        else:
            objf_validation_k_plus_one = None

        # update q and residual
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
            if self.gradient_memory == 1:
                q_k_plus_one_vec = - G_k.dot(G_k.T.dot(v_k_vec)) - H_diag_k * v_k_vec - lam_k * v_k_vec \
                                   - f_grads_memory[:, 0]
            else:
                q_k_plus_one_vec = - G_k.dot(G_k.T.dot(v_k_vec)) - H_diag_k * v_k_vec - lam_k * v_k_vec \
                                   - f_grads_memory.dot(bundle_dual)
            rms_res = np.linalg.norm(f_grad_k_plus_one_vec + q_k_plus_one_vec) / np.sqrt(self.osmm_problem.n)

        # update l_k
        f_grads_memory, f_consts_memory = self._update_l_k(iter_idx, x_k_plus_one_vec, f_k_plus_one,
                                                           f_grad_k_plus_one_vec, f_grads_memory, f_consts_memory)

        # update G_k and H_diag_k
        begin_curve_time = time.time()
        if iter_idx % self.update_curvature_frequency == 0:
            G_k_plus_one, H_diag_k_plus_one = self._update_curvature_after_solve(G_k, x_k_plus_one_vec,
                                                                                 xk.flatten(order='F'),
                                                                                 f_grad_k_plus_one_vec,
                                                                                 f_grad_k.flatten(order='F'))
        else:
            G_k_plus_one = G_k
            H_diag_k_plus_one = H_diag_k
        end_curve_time = time.time()

        # update trust param
        tr_H_k_plus_one = np.square(max(ep, np.linalg.norm(G_k_plus_one, 'fro')))
        tau_k_plus_one = tr_H_k_plus_one / self.osmm_problem.n / min(self.hessian_rank, iter_idx)
        lam_k_plus_one, mu_k_plus_one = self._update_trust_params(mu_k, tk, tau_k_plus_one)

        # update lower bound and print info
        begin_eval_L_k_time = 0
        end_eval_L_k_time = 0
        lower_bound_k_plus_one = lower_bound_k
        if iter_idx % self.check_gap_frequency == 0:
            begin_eval_L_k_time = time.time()
            try:
                lower_bound_subp.solve(solver=self.solver)
                lower_bound_k_plus_one = max(lower_bound_k, lower_bound_subp.value)
            except Exception as e:
                if self.verbose:
                    print("lower bound problem error:", e)
                lower_bound_k_plus_one = lower_bound_k
            end_eval_L_k_time = time.time()
            if self.verbose:
                if self.osmm_problem.W_torch_validate is not None:
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

        # check stopping criteria
        stopping_criteria_satisfied = self._stopping_criteria(objf_k_plus_one, objf_validation_k_plus_one,
                                                              lower_bound_k_plus_one, tk, rms_res,
                                                              np.linalg.norm(q_k_plus_one_vec),
                                                              np.linalg.norm(f_grad_k_plus_one))

        # save update
        self.osmm_problem.method_results["time_detail_iters"][0, iter_idx] = f_eval_time_cost
        self.osmm_problem.method_results["time_detail_iters"][1, iter_idx] = end_evaluate_f_grad_time - \
                                                                             begin_evaluate_f_grad_time
        self.osmm_problem.method_results["time_detail_iters"][2, iter_idx] = end_solve_time - begin_solve_time
        self.osmm_problem.method_results["time_detail_iters"][3, iter_idx] = end_eval_L_k_time - begin_eval_L_k_time
        self.osmm_problem.method_results["time_detail_iters"][4, iter_idx] = end_curve_time - begin_curve_time
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

        return stopping_criteria_satisfied, x_k_plus_one, objf_k_plus_one, f_k_plus_one, g_k_plus_one, \
               lower_bound_k_plus_one, f_grad_k_plus_one, f_grads_memory, f_consts_memory, G_k_plus_one, \
               H_diag_k_plus_one, lam_k_plus_one, mu_k_plus_one