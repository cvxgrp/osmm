import cvxpy as cp
import numpy as np


class GCvxpy:
    def __init__(self):
        self.variable = None
        self.objective = 0
        self.constraints = []
        self.additional_var_soln = {}
        self.all_var_list = []

    def g_value(self, solver="ECOS"):
        """
        optimize over hidden variables to evaluate g(x)
        :param solver: solver used to minimize g(x, z) over z
        :return:
        If x is None or out of the domain of g, returns inf.
        If solver succeeds, g(x, z^star).
        If solver fails, g(x, z), where z is the value before the evaluation.
        """
        if len(self.additional_var_soln) == 0:  # no hidden variable
            if self.variable.value is None:
                return np.inf
            for constr in self.constraints:
                if not constr.value():
                    return np.inf
            g_tmp = self.objective.value
        else:
            x_tmp = self.variable.value
            var_val_prev = [var.value for var in self.all_var_list]

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
                ub_g_tmp = None
                for i in range(len(var_val_prev)):
                    var = self.all_var_list[i]
                    var.value = var_val_prev[i]
                    if var.value is None:
                        ub_g_tmp = np.inf
                if ub_g_tmp is None:
                    ub_g_tmp = self.objective.value
                    for constr in self.constraints:
                        if not constr.value():
                            ub_g_tmp = np.inf
                g_tmp = ub_g_tmp

        return g_tmp