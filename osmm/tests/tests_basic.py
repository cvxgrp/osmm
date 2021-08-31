import unittest
import torch
import numpy as np
from osmm import OSMM
import cvxpy as cp


class BasicTestCase(unittest.TestCase):
    def setUp(self) -> None:
        n = 100
        N = 10000
        self.osmm_prob = OSMM()

        def my_f_torch(x_torch, W_torch):
            objf = -torch.mean(torch.log(torch.matmul(W_torch.T, x_torch)))
            return objf

        def my_elementwise_mapping_torch(y_scalar_torch):
            return -torch.log(y_scalar_torch) / N

        self.osmm_prob.f_torch.function = my_f_torch
        self.osmm_prob.f_torch.elementwise_mapping = my_elementwise_mapping_torch

        np.random.seed(0)
        self.osmm_prob.f_torch.W_torch = torch.tensor(np.random.uniform(low=0.5, high=1.5, size=(n, N)),
                                                      requires_grad=False, dtype=torch.float)

        my_var = cp.Variable(n, nonneg=True)
        self.osmm_prob.g_cvxpy.variable = my_var
        self.osmm_prob.g_cvxpy.objective = 0
        self.osmm_prob.g_cvxpy.constraints = [cp.sum(my_var) == 1]

        self.init_val = np.ones(n)

    def test_default_setting(self):
        result = self.osmm_prob.solve(self.init_val)
        self.assertAlmostEqual(result, -0.00259977, delta=1e-4)
        # self.assertEqual(True, False)

    def test_bundle_mode_all_active(self):
        result = self.osmm_prob.solve(self.init_val, bundle_mode="AllActive")
        self.assertAlmostEqual(result, -0.00259977, delta=1e-4)

    def test_hessian_mode_EVD(self):
        result = self.osmm_prob.solve(self.init_val, hessian_mode="LowRankDiagEVD")
        self.assertAlmostEqual(result, -0.00259977, delta=1e-4)

    def test_use_cvxpy_param(self):
        result = self.osmm_prob.solve(self.init_val, use_cvxpy_param=True)
        self.assertAlmostEqual(result, -0.00259977, delta=1e-4)

    def test_exact_g_line_search(self):
        result = self.osmm_prob.solve(self.init_val, exact_g_line_search=True)
        self.assertAlmostEqual(result, -0.00259977, delta=1e-4)

    def test_not_store_var_all_iters(self):
        result = self.osmm_prob.solve(self.init_val, store_var_all_iters=False)
        self.assertAlmostEqual(result, -0.00259977, delta=1e-4)


if __name__ == '__main__':
    unittest.main()
