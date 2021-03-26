import numpy as np
import cvxpy as cp
import time

from .subproblem import Subproblem
from .curvature_updates import CurvatureUpdate
from .alg_mode import AlgMode


class OsmmUpdate:
    def __init__(self, problem_instance):
        self.problem_instance = problem_instance
        self.curvature_update = CurvatureUpdate(problem_instance)

    def initialization(self, init_val, H_rank, pieces_num, alg_mode, tau_min, ini_by_Hutchison):
        x_0 = init_val
        f_0 = self.problem_instance.f_value(x_0)
        if self.problem_instance.W_torch_validate is not None:
            f_validation_0 = self.problem_instance.f_validate_value(x_0)
        else:
            f_validation_0 = None
        self.problem_instance.x_var_cvxpy.value = x_0
        g_0 = self.problem_instance.g_cvxpy.value

        if len(self.problem_instance.constr_cvxpy) > 0:
            check_g_fea_prob = cp.Problem(cp.Minimize(0), self.problem_instance.constr_cvxpy)
            try:
                check_g_fea_prob.solve()
                if check_g_fea_prob.value is not None and check_g_fea_prob.value < np.inf:
                    is_initial_feasible = True
                else:
                    is_initial_feasible = False
            except Exception as e:
                is_initial_feasible = False
        else:
            is_initial_feasible = True
        # for constr in self.problem_instance.constr_cvxpy:
        #     if not constr.value():
        #         is_initial_feasible = False
        #         break
        if is_initial_feasible:
            objf_0 = f_0 + g_0
            if self.problem_instance.W_torch_validate is not None:
                objf_validation_0 = f_validation_0 + g_0
            else:
                objf_validation_0 = None
        else:
            objf_0 = np.inf
            if self.problem_instance.W_torch_validate is not None:
                objf_validation_0 = np.inf
            else:
                objf_validation_0 = None

        self.problem_instance.x_var_cvxpy.value = None
        subprob = Subproblem(self.problem_instance.x_var_cvxpy, self.problem_instance.constr_cvxpy,
                             self.problem_instance.g_cvxpy, H_rank=H_rank, pieces_num=pieces_num)

        f_grad_0 = self.problem_instance.f_grad_value(x_0)

        if ini_by_Hutchison:
            begin_Hutchinson_time = time.time()
            est_hess_tr = self.problem_instance.f_hess_tr_Hutchinson(x_0, max_iter=100)
            lam_0 = max(tau_min, est_hess_tr / self.problem_instance.n)
            print("lam_0", lam_0, "Hutchinson time cost, ", time.time() - begin_Hutchinson_time)
        else:
            lam_0 = tau_min

        f_grads_iters_value = f_grad_0.repeat(pieces_num).reshape((self.problem_instance.n, pieces_num))
        f_const_iters_value = np.ones(pieces_num) * (f_0 - f_grad_0.dot(x_0))

        diag_H_0 = np.zeros(self.problem_instance.n)
        G_0 = np.zeros((self.problem_instance.n, H_rank))
        if alg_mode == AlgMode.Exact:
            G_0 = self.curvature_update.exact_hess_update(x_0)

        elif alg_mode == AlgMode.Diag:
            diag_H_0 = self.curvature_update.diag_hess_update(x_0)

        elif alg_mode == AlgMode.Trace:
            diag_H_0 = self.curvature_update.trace_hess_update(x_0)

        elif alg_mode == AlgMode.Hutchinson:
            diag_H_0 = self.curvature_update.Hutchison_update(x_0)

        elif alg_mode == AlgMode.LowRankDiagBundle:
            G_0, diag_H_0 = self.curvature_update.low_rank_diag_update(x_0, H_rank)

        elif alg_mode == AlgMode.LowRankNewSampBundle:
            G_0, diag_H_0 = self.curvature_update.low_rank_new_samp_update(x_0, H_rank)

        elif alg_mode == AlgMode.BFGSBundle:
            G_0 = np.eye(self.problem_instance.n) * np.sqrt(max(tau_min, np.linalg.norm(f_grad_0)))

        return subprob, x_0, objf_0, objf_validation_0, f_0, f_grad_0, g_0, lam_0, f_grads_iters_value, \
               f_const_iters_value, G_0, diag_H_0

    def _stopping_criteria(self, objf_k_plus_one, objf_validation_k_plus_one, L_k_plus_one, t_k, opt_res_norm_k_plus_one,
                          q_norm_k_plus_one, f_grad_norm_k_plus_one,
                          eps_gap_abs, eps_gap_rel, eps_res_abs, eps_res_rel):
        if objf_k_plus_one is not None and objf_k_plus_one < np.inf:
            if objf_validation_k_plus_one is not None:
                if objf_k_plus_one - L_k_plus_one <= np.abs(objf_k_plus_one - objf_validation_k_plus_one) \
                + eps_gap_rel * np.abs(objf_k_plus_one):
                    return True
            else:
                if objf_k_plus_one - L_k_plus_one <= eps_gap_abs + eps_gap_rel * np.abs(objf_k_plus_one):
                    return True
        # if objf_validation_k_plus_one is not None and objf_k_plus_one is not None and objf_k_plus_one < np.inf and \
        #         objf_k_plus_one - L_k_plus_one <= np.abs(objf_k_plus_one - objf_validation_k_plus_one) \
        #         + eps_gap_rel * np.abs(objf_k_plus_one):
        #     return True
        if t_k == 1 and \
                opt_res_norm_k_plus_one <= self.problem_instance.n * eps_res_abs \
                + eps_res_rel * (q_norm_k_plus_one + f_grad_norm_k_plus_one):
            return True
        return False

    def update_func(self, subprob, round_idx, objf_k, g_k, lower_bound_k, f_grad_k,
                    f_grads_iters_value, f_const_iters_value, G_k, diag_H_k, lam_k, mu_k,
                    alg_mode=None, H_rank=None, pieces_num=None, solver=None,
                    gamma_inc=None, gamma_dec=None,
                    mu_min=None, tau_min=None, mu_max=None, ep=None,
                    beta=None, j_max=None, alpha=None, num_iter_eval_Lk=None,
                    eps_gap_abs=None, eps_gap_rel=None, eps_res_abs=None, eps_res_rel=None):
        if solver == "OSQP":
            cvxpy_subp = cp.Problem(cp.Minimize(subprob.f_hat_k + self.problem_instance.g_cvxpy + subprob.trust_pen),
                                    self.problem_instance.constr_cvxpy + subprob.bundle_constr)
            lower_bound_subp = cp.Problem(cp.Minimize(subprob.l_k + self.problem_instance.g_cvxpy),
                                          self.problem_instance.constr_cvxpy + subprob.bundle_constr)
        else:
            cvxpy_subp = subprob.cvxpy_subp
            lower_bound_subp = subprob.lower_bound_subp

        if self.problem_instance.store_x_all_iters or round_idx < pieces_num:
            xk = np.array(self.problem_instance.method_results["var_iters"][:, round_idx - 1])
        else:
            xk = np.array(self.problem_instance.method_results["var_iters"][:, pieces_num - 1])
        subprob.x_prev_para.value = xk
        subprob.f_grads_iters_para.value = f_grads_iters_value
        subprob.f_const_iters_para.value = f_const_iters_value
        subprob.lam_para.value = lam_k
        subprob.G_para.value = G_k
        subprob.diag_H_para.value = diag_H_k

        begin_solve_time = time.time()
        subp_solver_success = True
        try:
            cvxpy_subp.solve(solver=solver, verbose=False)
        except Exception as e:
            subp_solver_success = False
            print("iter.", round_idx, ", solver status: ", cvxpy_subp.status,
                  ", x update is None", self.problem_instance.x_var_cvxpy.value is None, ", error message: ", e)
        if self.problem_instance.x_var_cvxpy.value is None:
            self.problem_instance.x_var_cvxpy.value = xk
        end_solve_time = time.time()

        x_k_plus_half = self.problem_instance.x_var_cvxpy.value
        g_k_plus_half = self.problem_instance.g_cvxpy.value
        bundle_dual = subprob.bundle_constr[0].dual_value
        additional_vars_value = [var.value for var in self.problem_instance.additional_vars_list]

        begin_line_search_time = time.time()
        if subp_solver_success:
            x_k_plus_one, f_k_plus_one, tk, num_f_evas_line_search, f_eva_time_cost \
                = self._line_search(x_k_plus_half, xk, g_k_plus_half, g_k, objf_k, G_k, lam_k, beta, j_max, alpha)
        else:
            x_k_plus_one = xk
            f_k_plus_one = objf_k - g_k
            tk = 0
            num_f_evas_line_search = j_max
            f_eva_time_cost = 0
        end_line_search_time = time.time()

        self.problem_instance.x_var_cvxpy.value = x_k_plus_one
        ub_g_k_plus_one = self.problem_instance.g_cvxpy.value
        ub_objf_k_plus_one = f_k_plus_one + ub_g_k_plus_one
        if self.problem_instance.W_torch_validate is not None:
            f_validation_k_plus_one = self.problem_instance.f_validate_value(x_k_plus_one)
            ub_objf_validation_k_plus_one = f_validation_k_plus_one + ub_g_k_plus_one
        else:
            ub_objf_validation_k_plus_one = None

        begin_evaluate_f_grad_time = time.time()
        f_grad_k_plus_one = self.problem_instance.f_grad_value(x_k_plus_one)
        end_evaluate_f_grad_time = time.time()

        v_k = x_k_plus_half - xk
        if bundle_dual is None:
            q_k_plus_one = np.inf
            opt_residual = np.inf
        else:
            if pieces_num == 1:
                q_k_plus_one = - G_k.dot(G_k.T.dot(v_k)) - lam_k * v_k - f_grads_iters_value[:, 0]
            else:
                q_k_plus_one = - G_k.dot(G_k.T.dot(v_k)) - lam_k * v_k - f_grads_iters_value.dot(bundle_dual)
            opt_residual = np.linalg.norm(f_grad_k_plus_one + q_k_plus_one)

        f_grads_iters_value, f_const_iters_value = self._update_l_k(round_idx, pieces_num, x_k_plus_one,
                                                                    f_k_plus_one, f_grad_k_plus_one,
                                                                    f_grads_iters_value, f_const_iters_value)

        begin_update_curvature_time = time.time()
        G_k_plus_one, diag_H_k_plus_one = \
            self._update_curvature_after_solve(alg_mode, G_k, x_k_plus_one, xk, f_grad_k_plus_one, f_grad_k, H_rank)
        end_update_curvature_time = time.time()

        tr_H_k_plus_one = np.square(max(ep, np.linalg.norm(G_k_plus_one, 'fro')))
        tau_k_plus_one = tr_H_k_plus_one / self.problem_instance.n / H_rank

        lam_k_plus_one, mu_k_plus_one = self._update_trust_params(mu_k, tk, tau_k_plus_one,
                                                                  tau_min, mu_min, mu_max, gamma_inc, gamma_dec)

        begin_evaluate_L_k = 0
        end_evaluate_L_k = 0
        lower_bound_k_plus_one = lower_bound_k
        if round_idx % num_iter_eval_Lk == 0:
            begin_evaluate_L_k = time.time()
            try:
                lower_bound_subp.solve(solver=solver)
                lower_bound_k_plus_one = max(lower_bound_k, lower_bound_subp.value)
            except Exception as e:
                print("lower bound problem error:", e)
                lower_bound_k_plus_one = lower_bound_k
            end_evaluate_L_k = time.time()
            print("iter=", round_idx, "objf_k+1=", ub_objf_k_plus_one, "L_k+1=", lower_bound_k_plus_one,
                  "lam_k+1=", lam_k_plus_one, "tk=", tk, "mu_k+1", mu_k_plus_one,
                  "||G_k+1||_F=", np.linalg.norm(G_k_plus_one, 'fro'), "tau_k+1", tau_k_plus_one)

        stopping_criteria_satisfied = self._stopping_criteria(ub_objf_k_plus_one, ub_objf_validation_k_plus_one,
                                                              lower_bound_k_plus_one, tk, opt_residual,
                                                              np.linalg.norm(q_k_plus_one),
                                                              np.linalg.norm(f_grad_k_plus_one),
                                                              eps_gap_abs, eps_gap_rel, eps_res_abs, eps_res_rel)

        self.problem_instance.method_results["time_cost_detail_iters"][0, round_idx] = f_eva_time_cost
        self.problem_instance.method_results["time_cost_detail_iters"][1, round_idx] = end_evaluate_f_grad_time \
                                                                                           - begin_evaluate_f_grad_time
        self.problem_instance.method_results["time_cost_detail_iters"][2, round_idx] = end_solve_time \
                                                                                           - begin_solve_time
        self.problem_instance.method_results["time_cost_detail_iters"][3, round_idx] = end_evaluate_L_k \
                                                                                           - begin_evaluate_L_k
        if self.problem_instance.store_x_all_iters or round_idx < pieces_num:
            self.problem_instance.method_results["var_iters"][:, round_idx] = x_k_plus_one
        else:
            self.problem_instance.method_results["var_iters"][:, 0:pieces_num - 1] = \
                self.problem_instance.method_results["var_iters"][:, 1:pieces_num]
            self.problem_instance.method_results["var_iters"][:, pieces_num - 1] = x_k_plus_one
        self.problem_instance.method_results["v_norm_iters"][round_idx] = np.linalg.norm(v_k)
        self.problem_instance.method_results["objf_iters"][round_idx] = ub_objf_k_plus_one
        if self.problem_instance.W_torch_validate is not None:
            self.problem_instance.method_results["objf_validation_iters"][round_idx] = ub_objf_validation_k_plus_one
        self.problem_instance.method_results["num_f_evas_line_search_iters"][round_idx] = num_f_evas_line_search
        self.problem_instance.method_results["q_norm_iters"][round_idx] = np.linalg.norm(q_k_plus_one)
        self.problem_instance.method_results["f_grad_norm_iters"][round_idx] = np.linalg.norm(f_grad_k_plus_one)
        self.problem_instance.method_results["opt_res_iters"][round_idx] = opt_residual
        self.problem_instance.method_results["lambd_iters"][round_idx] = lam_k_plus_one
        self.problem_instance.method_results["mu_iters"][round_idx] = mu_k_plus_one
        self.problem_instance.method_results["t_iters"][round_idx] = tk
        self.problem_instance.method_results["lower_bound_iters"][round_idx] = lower_bound_k_plus_one
        self.problem_instance.method_results["iters_taken"] = round_idx

        return stopping_criteria_satisfied, x_k_plus_one, ub_objf_k_plus_one, ub_g_k_plus_one, lower_bound_k_plus_one, \
               f_grad_k_plus_one,f_grads_iters_value, f_const_iters_value, G_k_plus_one, diag_H_k_plus_one, \
               lam_k_plus_one, mu_k_plus_one, additional_vars_value

    def _line_search(self, x_k_plus_half, xk, g_k_plus_half, g_k, objf_k, G_k, lam_k, beta, j_max, alpha, ep=1e-15): #tol=1e-5,
        v_k = x_k_plus_half - xk

        begin_evaluate_f_time = time.time()
        f_x_k_plus_half = self.problem_instance.f_value(x_k_plus_half)
        end_evaluate_f_time = time.time()

        # if f_x_k_plus_half < np.inf and np.linalg.norm(v_k) <= tol * np.linalg.norm(xk) and \
        #                                  f_x_k_plus_half + g_k_plus_half - objf_k <= tol * np.abs(objf_k):
        #     print("t = 1 because of too small increment")
        #     return x_k_plus_half, f_x_k_plus_half, 1.0, 0, end_evaluate_f_time - begin_evaluate_f_time

        desc = np.square(max(ep, np.linalg.norm(np.dot(G_k.T, v_k)))) + lam_k * np.square(max(ep, np.linalg.norm(v_k)))
        f_tmp = f_x_k_plus_half
        phi_line_search = f_tmp + g_k_plus_half
        t = 1.0
        j = 0
        while j < j_max and phi_line_search > objf_k - 0.5 * alpha * t * desc:
            t = t * beta
            f_tmp = self.problem_instance.f_value(t * x_k_plus_half + (1 - t) * xk)
            phi_line_search = f_tmp + t * g_k_plus_half + (1 - t) * g_k
            j += 1
        return t * x_k_plus_half + (1 - t) * xk, f_tmp, t, j, end_evaluate_f_time - begin_evaluate_f_time

    def _update_curvature_after_solve(self, alg_mode, G_k, x_k_plus_one, xk, f_grad_k_plus_one, f_grad_k, H_rank):
        G_k_plus_one = np.zeros((self.problem_instance.n, H_rank))
        diag_H_k_plus_one = np.zeros(self.problem_instance.n)

        if alg_mode == AlgMode.Exact:
            G_k_plus_one = self.curvature_update.exact_hess_update(x_k_plus_one)

        elif alg_mode == AlgMode.Diag:
            diag_H_k_plus_one = self.curvature_update.diag_hess_update(x_k_plus_one)

        elif alg_mode == AlgMode.Trace:
            diag_H_k_plus_one = self.curvature_update.trace_hess_update(x_k_plus_one)

        elif alg_mode == AlgMode.Hutchinson:
            diag_H_k_plus_one = self.curvature_update.Hutchison_update(x_k_plus_one)

        elif alg_mode == AlgMode.LowRankDiagBundle:
            G_k_plus_one, diag_H_k_plus_one = self.curvature_update.low_rank_diag_update(x_k_plus_one, H_rank)

        elif alg_mode == AlgMode.LowRankNewSampBundle:
            G_k_plus_one, diag_H_k_plus_one = self.curvature_update.low_rank_new_samp_update(x_k_plus_one, H_rank)

        elif alg_mode == AlgMode.BFGSBundle:
            G_k_plus_one = self.curvature_update.BFGS_update(G_k, x_k_plus_one, xk, f_grad_k_plus_one, f_grad_k)

        elif alg_mode == AlgMode.LowRankQNBundle:
            G_k_plus_one = self.curvature_update.low_rank_quasi_Newton_update(G_k, x_k_plus_one, xk, f_grad_k_plus_one,
                                                                              f_grad_k, H_rank)

        return G_k_plus_one, diag_H_k_plus_one

    def _update_l_k(self, round_idx, pieces_num, x_k_plus_one, f_k_plus_one, f_grad_k_plus_one, f_grads_iters_value,
                    f_const_iters_value):
        if round_idx < pieces_num:
            num_iters_remain = pieces_num - round_idx
            f_grads_iters_value[:, round_idx: pieces_num] = \
                f_grad_k_plus_one.repeat(num_iters_remain).reshape((self.problem_instance.n, num_iters_remain))
            f_const_iters_value[round_idx: pieces_num] = np.ones(num_iters_remain) * \
                                                         (f_k_plus_one - f_grad_k_plus_one.dot(x_k_plus_one))
        else:
            f_grads_iters_value[:, round_idx % pieces_num] = f_grad_k_plus_one
            f_const_iters_value[round_idx % pieces_num] = f_k_plus_one - f_grad_k_plus_one.dot(x_k_plus_one)
        return f_grads_iters_value, f_const_iters_value

    def _update_trust_params(self, mu_k, tk, tau_k_plus_one, tau_min, mu_min, mu_max, gamma_inc, gamma_dec):
        if tk >= 0.99:
            mu_k_plus_one = max(mu_k * gamma_dec, mu_min)
        else:
            mu_k_plus_one = min(mu_k * gamma_inc, mu_max)
        lam_k_plus_one = mu_k_plus_one * max(tau_min, tau_k_plus_one)
        return lam_k_plus_one, mu_k_plus_one
