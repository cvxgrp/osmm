import cvxpy as cp


class Subproblem:
    def __init__(self, x_var_cvxpy, constr_cvxpy, g_cvxpy, H_rank=None, pieces_num=None):
        n = x_var_cvxpy.size
        self.x_prev_para = cp.Parameter(n)
        self.diag_H_para = cp.Parameter(n, nonneg=True)
        self.G_para = cp.Parameter((n, H_rank))
        self.lam_para = cp.Parameter(nonneg=True)
        self.l_k = cp.Variable(1)

        self.f_curvature = 0.5 * cp.sum(cp.multiply(self.diag_H_para, cp.square(x_var_cvxpy - self.x_prev_para))) \
                           + 0.5 * cp.sum_squares(self.G_para.T @ (x_var_cvxpy - self.x_prev_para))

        self.f_grads_iters_para = cp.Parameter((n, pieces_num))
        self.f_const_iters_para = cp.Parameter(pieces_num)
        self.bundle_constr = [self.f_grads_iters_para.T @ x_var_cvxpy + self.f_const_iters_para <= self.l_k]

        self.f_hat_k = self.l_k + self.f_curvature
        self.trust_penalty = 0.5 * cp.sum_squares(x_var_cvxpy - self.x_prev_para) * self.lam_para

        self.cvxpy_subp = cp.Problem(cp.Minimize(self.f_hat_k + g_cvxpy + self.trust_penalty),
                                     constr_cvxpy + self.bundle_constr)
        self.lower_bound_subp = cp.Problem(cp.Minimize(self.l_k + g_cvxpy), constr_cvxpy + self.bundle_constr)