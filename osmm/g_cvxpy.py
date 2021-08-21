import cvxpy as cp
import numpy as np


class GCvxpy:
    def __init__(self):
        self.variable = None
        self.objective = 0
        self.constraints = []
        self.additional_var_soln = {}
        self.all_var_list = []

    def eval(self, solver="ECOS"):
        """
        optimize over hidden variables to evaluate g(x)
        :param solver: solver used to minimize g(x, z) over z
        :return: If succeed, g(x, z^star). If solver fails, g(x, z), where z is the value before solving the subproblem.
        """
        if len(self.additional_var_soln) == 0:
            g_tmp = self.objective.value
        else:
            x_tmp = self.variable.value
            var_val_prev = [var.value for var in self.all_var_list]
            ub_g_tmp = None
            for var in self.all_var_list:
                if var.value is None:
                    ub_g_tmp = np.inf
                    break
            if ub_g_tmp is None:
                ub_g_tmp = self.objective.value
                for constr in self.constraints:
                    if not constr.value():
                        ub_g_tmp = np.inf

            g_eval_subp = cp.Problem(cp.Minimize(self.objective), self.constraints + [self.variable == x_tmp])
            # To Do: use cvxpy parameters for the subproblem of evaluating g

            g_eval_success = True
            try:
                g_eval_subp.solve(solver=solver)
                if (g_eval_subp.status != "optimal" and g_eval_subp.status != "inaccurate_optimal") or \
                        self.objective is None or self.objective.value == np.inf:
                    g_eval_success = False
            except Exception as e:
                g_eval_success = False

            if g_eval_success:
                g_tmp = self.objective.value
            else:
                g_tmp = ub_g_tmp
                for i in range(len(var_val_prev)):
                    var = self.all_var_list[i]
                    var.value = var_val_prev[i]
        return g_tmp
