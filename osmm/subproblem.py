import cvxpy as cp


class Subproblem:
    def __init__(self, x_var_cvxpy, g_constrs, g_objf, hessian_rank=None, gradient_memory=None):
        n = x_var_cvxpy.size
        self.lam_para = cp.Parameter(nonneg=True)
        self.l_k = cp.Variable(1)
        self.diag_H_para = cp.Parameter(n, nonneg=True)
        self.G_para = cp.Parameter((n, hessian_rank))
        self.f_grads_iters_para = cp.Parameter((n, gradient_memory))
        self.f_const_iters_para = cp.Parameter(gradient_memory)
        if len(x_var_cvxpy.shape) <= 1:
            self.x_for_g_para = cp.Parameter(n)
            self.x_prev_para = cp.Parameter(n)
            self.f_curvature = 0.5 * cp.sum(cp.multiply(self.diag_H_para, cp.square(x_var_cvxpy - self.x_prev_para))) \
                               + 0.5 * cp.sum_squares(self.G_para.T @ (x_var_cvxpy - self.x_prev_para))
            self.bundle_constr = [self.f_grads_iters_para.T @ x_var_cvxpy + self.f_const_iters_para <= self.l_k]
        else:
            n1, n2 = x_var_cvxpy.shape
            self.x_for_g_para = cp.Parameter((n1, n2))
            self.x_prev_para = cp.Parameter((n1, n2))
            self.f_curvature = 0.5 * cp.sum(cp.multiply(self.diag_H_para,
                                                        cp.square(cp.vec(x_var_cvxpy - self.x_prev_para)))) \
                               + 0.5 * cp.sum_squares(self.G_para.T @ cp.vec(x_var_cvxpy - self.x_prev_para))
            self.bundle_constr = [self.f_grads_iters_para.T @ cp.vec(x_var_cvxpy) + self.f_const_iters_para <= self.l_k]

        self.f_hat_k = self.l_k + self.f_curvature
        self.trust_penalty = 0.5 * cp.sum_squares(x_var_cvxpy - self.x_prev_para) * self.lam_para

        self.cvxpy_subp = cp.Problem(cp.Minimize(self.f_hat_k + g_objf + self.trust_penalty),
                                     g_constrs + self.bundle_constr)
        self.lower_bound_subp = cp.Problem(cp.Minimize(self.l_k + g_objf), g_constrs + self.bundle_constr)

        self.g_eval_subp = cp.Problem(cp.Minimize(g_objf), g_constrs + [x_var_cvxpy == self.x_for_g_para])