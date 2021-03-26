import cvxpy as cp


class Subproblem:
    def __init__(self, x_var_cvxpy, g_constrs, g_objf, hessian_rank=None, gradient_memory=None):
        n = x_var_cvxpy.size
        self.x_prev_para = cp.Parameter(n)
        self.diag_H_para = cp.Parameter(n, nonneg=True)
        self.G_para = cp.Parameter((n, hessian_rank))
        self.lam_para = cp.Parameter(nonneg=True)
        self.l_k = cp.Variable(1)

        self.f_curvature = 0.5 * cp.sum(cp.multiply(self.diag_H_para, cp.square(x_var_cvxpy - self.x_prev_para))) \
                           + 0.5 * cp.sum_squares(self.G_para.T @ (x_var_cvxpy - self.x_prev_para))

        self.f_grads_iters_para = cp.Parameter((n, gradient_memory))
        self.f_const_iters_para = cp.Parameter(gradient_memory)
        self.bundle_constr = [self.f_grads_iters_para.T @ x_var_cvxpy + self.f_const_iters_para <= self.l_k]

        self.f_hat_k = self.l_k + self.f_curvature
        self.trust_penalty = 0.5 * cp.sum_squares(x_var_cvxpy - self.x_prev_para) * self.lam_para

        self.cvxpy_subp = cp.Problem(cp.Minimize(self.f_hat_k + g_objf + self.trust_penalty),
                                     g_constrs + self.bundle_constr)
        self.lower_bound_subp = cp.Problem(cp.Minimize(self.l_k + g_objf), g_constrs + self.bundle_constr)