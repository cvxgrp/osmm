import unittest
import torch
import numpy as np
from osmm import OSMM
import cvxpy as cp


class MatrixVariableTestCase(unittest.TestCase):
    def setUp(self) -> None:
        d = 10
        m = 20
        N = 10000
        np.random.seed(0)
        s = np.random.uniform(low=1.0, high=5.0, size=(d))
        W = np.exp(np.random.randn(2 * m, N))

        def my_f_torch(A_torch, W_torch):
            d_torch = W_torch[0:m, :]
            p_torch = W_torch[m:2 * m, :]
            s_torch = torch.tensor(s, dtype=torch.float, requires_grad=False)
            retail_node_amount = torch.matmul(A_torch, s_torch)
            ave_revenue = torch.sum(p_torch * torch.min(d_torch, retail_node_amount[:, None])) / N
            return -ave_revenue

        A_var = cp.Variable((m, d), nonneg=True)

        self.osmm_prob = OSMM()
        self.osmm_prob.f_torch.function = my_f_torch
        self.osmm_prob.f_torch.W_torch = torch.tensor(W, dtype=torch.float)
        self.osmm_prob.g_cvxpy.variable = A_var
        self.osmm_prob.g_cvxpy.objective = cp.sum(cp.abs(A_var))
        self.osmm_prob.g_cvxpy.constraints = [cp.sum(A_var, axis=0) <= np.ones(d)]

        self.init_val = np.ones((m, d))

    def test_default_setting(self):
        result = self.osmm_prob.solve(self.init_val)
        self.assertAlmostEqual(result, -24.5744, delta=1e-2)

    def test_bundle_mode_all_active(self):
        result = self.osmm_prob.solve(self.init_val, bundle_mode="AllActive")
        self.assertAlmostEqual(result, -24.5744, delta=1e-2)

    def test_use_cvxpy_param(self):
        result = self.osmm_prob.solve(self.init_val, use_cvxpy_param=True)
        self.assertAlmostEqual(result, -24.5744, delta=1e-2)

    def test_exact_g_line_search(self):
        result = self.osmm_prob.solve(self.init_val, exact_g_line_search=True)
        self.assertAlmostEqual(result, -24.5744, delta=1e-2)

    def test_not_store_var_all_iters(self):
        result = self.osmm_prob.solve(self.init_val, store_var_all_iters=False)
        self.assertAlmostEqual(result, -24.5744, delta=1e-2)


if __name__ == '__main__':
    unittest.main()
