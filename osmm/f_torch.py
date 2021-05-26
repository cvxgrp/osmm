import numpy as np
import torch
from functools import partial

class FTorch:
    def __init__(self):
        # self.W = None
        # self.W_validate = None
        self.function = None
        self.elementwise_mapping = None
        self.W_torch = None
        self.W_validate_torch = None

    def _f(self, x):
        x_torch = torch.tensor(x, dtype=torch.float, requires_grad=True)
        if self.W_torch is not None:
            f_torch = self.function(x_torch, self.W_torch)
        else:
            f_torch = self.function(x_torch)
        return f_torch, x_torch

    def f_value(self, x):
        return float(self._f(x)[0])

    def f_validate_value(self, x):
        x_torch = torch.tensor(x, dtype=torch.float, requires_grad=False)
        if self.W_torch is not None:
            f_torch = self.function(x_torch, self.W_validate_torch)
        else:
            f_torch = self.function(x_torch)
        return float(f_torch)

    def f_grad_value(self, x):
        f_torch, x_torch = self._f(x)
        f_torch.backward()
        return np.array(x_torch.grad.cpu().numpy())

    def f_hess_value(self, x):
        if self.elementwise_mapping is None:
            x_torch = torch.tensor(x, dtype=torch.float, requires_grad=True)
            my_f_partial = partial(self.function, W_torch=self.W_torch)
            result_torch = torch.autograd.functional.hessian(my_f_partial, x_torch)
        else:
            x_torch = torch.tensor(x, dtype=torch.float, requires_grad=False)
            y_torch = torch.matmul(self.W_torch.T, x_torch)
            y_torch.requires_grad_(True)
            f_tmp = torch.sum(self.elementwise_mapping(y_torch))
            f_tmp.backward(create_graph=True)
            grad_y = y_torch.grad
            y_grad_tmp = torch.tensor(np.array(y_torch.grad.detach().cpu().numpy()), dtype=torch.float,
                                      requires_grad=False)
            grad_y.requires_grad_(True)
            grad_y_sum = torch.sum(grad_y)
            grad_y_sum.backward()
            diag_part = y_torch.grad.detach() - y_grad_tmp
            tmp = self.W_torch * diag_part
            result_torch = torch.matmul(tmp, self.W_torch.T)
        return np.array(result_torch.cpu().numpy())

    def f_hess_tr_Hutchinson(self, x, max_iter=100, tol=1e-3):
        est_tr = 0
        it = 0
        n = x.size
        if len(x.shape) <= 1:
            n0 = n
            n1 = 0
        else:
            n0, n1 = x.shape
        while it < max_iter:
            f_torch, x_torch = self._f(x)
            f_torch.backward(create_graph=True)
            grad_x = x_torch.grad
            grad_x_tmp = np.array(x_torch.grad.detach().cpu().numpy())
            grad_x.requires_grad_(True)
            if n1 == 0:
                v = (np.random.rand(n) < 0.5) * 2 - 1.0
                v_torch = torch.tensor(v, dtype=torch.float, requires_grad=False)
                gv_objf = torch.matmul(v_torch.T, grad_x)
                gv_objf.backward()
                Hv = x_torch.grad.detach().cpu().numpy() - grad_x_tmp
                vTHv = float(v.T.dot(Hv))
            else:
                v = (np.random.rand(n0, n1) < 0.5) * 2 - 1.0
                v_torch = torch.tensor(v, dtype=torch.float, requires_grad=False)
                gv_objf = torch.sum(v_torch * grad_x)
                gv_objf.backward()
                Hv = x_torch.grad.detach().cpu().numpy() - grad_x_tmp
                vTHv = float(np.sum(v * Hv))
            new_est_tr = (est_tr * it + float(vTHv)) / (it + 1)
            if np.abs(new_est_tr - est_tr) < tol * np.abs(est_tr):
                break
            est_tr = new_est_tr
            it += 1
        # print("Hutchinson #iters", it, "rel. incr.", np.abs(new_est_tr - est_tr) / np.abs(est_tr), "est. tr.", est_tr)
        return est_tr