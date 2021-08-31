import unittest
import torch
import numpy as np
from osmm import OSMM
import cvxpy as cp


class HiddenVariableTestCase(unittest.TestCase):
    def setUp(self) -> None:
        n = 30
        full_edges = []
        for i in range(n - 1):
            full_edges += [[i, j] for j in range(i + 1, n)]

        m = len(full_edges) // 2
        N = 1000
        W = np.zeros((2 * n, N))

        np.random.seed(0)
        connected_edges = np.random.choice(range(len(full_edges)), m, replace=False)
        W[0:n, :] = np.exp(np.random.multivariate_normal(np.ones(n), np.eye(n), size=N)).T
        W[n:2 * n, :] = -np.exp(np.random.multivariate_normal(np.zeros(n), np.eye(n), size=N)).T

        A = np.zeros((n, m))
        for i in range(m):
            edge_idx = connected_edges[i]
            A[full_edges[edge_idx][0], i] = 1
            A[full_edges[edge_idx][1], i] = -1

        x_max = 1
        u_max = 0.1
        x_var = cp.Variable(n)
        u_var = cp.Variable(m)
        constrs = [A @ u_var + x_var == 0, cp.norm(u_var, 'inf') <= u_max, cp.norm(x_var, 'inf') <= x_max]

        def my_f_torch(x_torch, W_torch):
            s_torch = W_torch[0:n, :]
            d_torch = W_torch[n:n * 2, :]
            return torch.mean(torch.sum(torch.relu(-d_torch.T - s_torch.T - x_torch), axis=1))

        self.osmm_prob = OSMM()
        self.osmm_prob.f_torch.function = my_f_torch
        self.osmm_prob.f_torch.W_torch = torch.tensor(W, dtype=torch.float)
        self.osmm_prob.g_cvxpy.variable = x_var
        self.osmm_prob.g_cvxpy.objective = 0
        self.osmm_prob.g_cvxpy.constraints = constrs

        self.init_val = np.ones(n)

    def test_default_setting(self):
        result = self.osmm_prob.solve(self.init_val)
        self.assertAlmostEqual(result, 14.015026, delta=1e-2)

    def test_bundle_mode_all_active(self):
        result = self.osmm_prob.solve(self.init_val, bundle_mode="AllActive")
        self.assertAlmostEqual(result, 14.015026, delta=1e-2)

    def test_use_cvxpy_param(self):
        result = self.osmm_prob.solve(self.init_val, use_cvxpy_param=True)
        self.assertAlmostEqual(result, 14.015026, delta=1e-2)

    def test_exact_g_line_search(self):
        result = self.osmm_prob.solve(self.init_val, exact_g_line_search=True)
        self.assertAlmostEqual(result, 14.015026, delta=1e-2)

    def test_not_store_var_all_iters(self):
        result = self.osmm_prob.solve(self.init_val, store_var_all_iters=False)
        self.assertAlmostEqual(result, 14.015026, delta=1e-2)


if __name__ == '__main__':
    unittest.main()
