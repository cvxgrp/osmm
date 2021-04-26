# osmm
`osmm` is a Python package for oracle-structured minimization method, which solves problems in the following form
```
minimize f(x, W) + g(x),
```
where `x` is the variable, and `W` contains data and parameters that specify `f`. 
The oracle function `f( ,W)` is convex in `x`, defined by PyTorch, and can be automatically differentiated by PyTorch. 
The structured function `g` is convex, defined by CVXPY, and can contain constraints and variables additional to `x`.

Variable `x` can be a scalar, a vector, or a matrix.

`osmm` is suitable for cases where `W` contains a large data matrix, as will be shown in an example later when we introduce the efficiency.

The implementation is based on our paper Minimizing Oracle-Structured Composite Functions [XXX add link].

## Installation
To install `osmm`, please run 
```
python setup.py install
```

`osmm` requires
* [cvxpy](https://github.com/cvxgrp/cvxpy) >= 1.1.0a4
* [PyTorch](https://pytorch.org/) >= 1.6.0
* Python 3.x

## Usage
`osmm` exposes the `OSMM` class 
```python
OSMM(f_torch, g_cvxpy)
```
which creates an object defining the form of the problem, and a member function of the `OSMM` class
```python
solve(init_val, W)
```
which specifies the problem with data matrix `W`, runs the solve method with initial value `init_val`, and returns the optimal objective value.
If there is no data matrix, then the solve method can be called by the following without `W`.
```python
solve(init_val)
```

### Arguments
The construction method of the `OSMM` class has two required arguments `f_torch` and `g_cvxpy`, which define the form of the problem.
* The first one `f_torch` must be a function with one required input, one optional input, and one output.
    * The first input (required) is a PyTorch tensor for `x`. 
    * The second input (optional) is a PyTorch tensor for `W` and must be named `W_torch`. It is only needed when there is a data matrix in the problem.
    * The output is a PyTorch tensor for the scalar function value of `f`.
* The second one `g_cvxpy` must be a function with no input and three outputs. 
    * The first output is a CVXPY variable for `x`. 
    * The second output is a CVXPY expression for the objective function in `g`. 
    * The third output is a list of constraints contained in `g`.

The `solve` method has one required argument, which gives an initial value of `x`, and several optional arguments.
* `init_val` (required) must be a scalar, a numpy array, or a numpy matrix that is in the same shape as `x`, and it must be in the domain of `f`.
* `W` (optional) is a scalar, a numpy array, or a numpy matrix which specifies the problem to be solved. It is only needed when there is a data matrix in the problem.
* More optional arguments will be introduced later.


### Examples
**1. Basic example.** We take the following Kelly gambling problem as one example
```
minimize - \sum_{i=1}^N log(w_i'x) / N
subject to x >= 0, x'1 = 1,
```
where `x` is an `n` dimensional variable, and `w_i` for `i=1, ..., N` are given data.
The objective function is `f`, the indicator function of the constraints is `g`, and the data matrix `W = [w_1, ..., w_N]`.
The code is as follows.
```python
import torch
import numpy as np
import cvxpy as cp
from osmm import OSMM

n = 100
N = 10000

# Define a CVXPY variable for x.
x_var = cp.Variable(n, nonneg=True)

# Define f by torch.
def my_f_torch(x_torch, W_torch):
    objf = -torch.mean(torch.log(torch.matmul(W_torch.T, x_torch)))
    return objf

# Define g by CVXPY.
def my_g_cvxpy():
    g = 0
    constr = [cp.sum(x_var) == 1]
    return x_var, g, constr

# Generate the data matrix.
W = np.random.uniform(low=0.5, high=1.5, size=(n, N))

# Generate an initial value for x.
init_val = np.ones(n) / n

# Define an OSMM object.
osmm_prob = OSMM(my_f_torch, my_g_cvxpy)

# Call the solve method, and the optimal objective value is returned by it.
opt_obj_val = osmm_prob.solve(init_val, W)

# A solution for x is stored in x_var.value.
print("x solution = ", x_var.value)
```

**2. Matrix variable.** `osmm` accepts matrix variables. Take the following allocation problem as an example
```
minimize - \sum_{i=1}^N p_i' min(As, d_i) / N + t||A||_1
subject to A >= 0, \sum_{j=1}^m A_ij <= 1,
```
where `A` is an `m` by `d` variable.
The `d` dimensional vector `s`, the positive scalar `t`, 
and the `m` dimensional vectors `p_i` and `d_i` are given. 
The notation `|| ||_1` is the sum of absolute values of all entires.
The first term in the objective function is `f`,
the regularization term plus the indicator function of the constraints is `g`,
and the data matrix `W = [(d_1, p_1), ..., (d_N, p_N)]`. 

```python
import torch
import autograd.numpy as np
import cvxpy as cp
from osmm import OSMM

d = 10
m = 20
N = 10000
s = np.random.uniform(low=1.0, high=5.0, size=(d))
W = np.exp(np.random.randn(2 * m, N))
t = 1
A_var = cp.Variable((m, d), nonneg=True)
init_val = np.ones((m, d))

def my_g_cvxpy():
    g = t * cp.sum(cp.abs(A_var))
    constr = [cp.sum(A_var, axis=0) <= np.ones(d)]
    return A_var, g, constr

def my_f_torch(A_torch, W_torch):
    d_torch = W_torch[0:m, :]
    p_torch = W_torch[m:2 * m, :]
    s_torch = torch.tensor(s, dtype=torch.float, requires_grad=False)
    retail_node_amount = torch.matmul(A_torch, s_torch)
    ave_revenue = torch.sum(p_torch * torch.min(d_torch, retail_node_amount[:, None])) / N
    return -ave_revenue
    
osmm_prob = OSMM(my_f_torch, my_g_cvxpy)
result = osmm_prob.solve(init_val, W)
```

**3. Additional variables in g.** `osmm` accepts variables additional to `x` in `g`. Take the following simplified power flow problem as an example
```
minimize \sum_{i=1}^N 1' (-d_i - s_i - x)_+ / N
subject to Au + x = 0, ||u||_inf <= u_max, ||x||_inf <= x_max,
```
where `x` is an `n` dimensional variable, and `u` is an `m` dimensional variable.
The graph incidence matrix `A`, the `n` dimensional vectors `d_i` and `s_i`,
and the positive scalars `u_max` and `x_max` are given.
The objective function corresponds to `f`.
Minimization of the indicator function of the constraints over `u` gives `g`,
so `u` is an additional variable in `g`.
The data matrix `W = [(s_1, d_1), ..., (s_N, d_N)]`. 

```python
import torch
import autograd.numpy as np
import cvxpy as cp
from osmm import OSMM

n = 30
full_edges = []
for i in range(n - 1):
    full_edges += [[i, j] for j in range(i + 1, n)]

m = len(full_edges) // 2
connected_edges = np.random.choice(range(len(full_edges)), m, replace=False)
A = np.zeros((n, m))
for i in range(m):
    edge_idx = connected_edges[i]
    A[full_edges[edge_idx][0], i] = 1
    A[full_edges[edge_idx][1], i] = -1

x_max = 1
u_max = 0.1
N = 1000
W = np.zeros((2 * n, N))
W[0:n, :] = np.exp(np.random.multivariate_normal(np.ones(n), np.eye(n), size=N)).T
W[n:2 * n, :] = -np.exp(np.random.multivariate_normal(np.zeros(n), np.eye(n), size=N)).T

x_var = cp.Variable(n)
init_val = np.ones(n)
u_var = cp.Variable(m)

def my_g_cvxpy():
    constr = [A @ u_var + x_var == 0, cp.norm(u_var, 'inf') <= u_max, cp.norm(x_var, 'inf') <= x_max]
    g = 0
    return x_var, g, constr

def my_f_torch(x_torch, W_torch):
    s_torch = W_torch[0:n, :]
    d_torch = W_torch[n:n * 2, :]
    return torch.mean(torch.sum(torch.relu(-d_torch.T - s_torch.T - x_torch), axis=1))

osmm_prob = OSMM(my_f_torch, my_g_cvxpy)
result = osmm_prob.solve(init_val, W)
```

For more examples, see the [`examples`](examples/) directory.

### Efficiency
`osmm` is efficient when `W` contains a large data matrix, and can be more efficient if PyTorch uses a GPU to compute `f` and its gradient.

We take the Kelly gambling problem as an example again. 
The variable `x_var`, the functions `my_f_torch` and `my_g_cvxpy`, and `init_val` have been defined in the above.
We compare the time cost of `osmm` with CVXPY on a CPU, and show that `osmm` is not as efficient as CVXPY when the data matrix is small with `N=100`,
but is more efficient when the data matrix is large with `N=30,000`.

```python
import time as time
np.random.seed(0)

N = 100
W_small = np.random.uniform(low=0.5, high=1.5, size=(n, N))

t1 = time.time()
opt_obj_val = osmm_prob.solve(init_val, W_small)
print("N = 100, osmm time cost = %.2f, opt value = %.4f" % (time.time() - t1, opt_obj_val))
# N = 100, osmm time cost = 0.19, opt value = -0.0557

cvx_prob = cp.Problem(cp.Minimize(-cp.sum(cp.log(W_small.T @ x_var)) / N), [cp.sum(x_var) == 1])
t2 = time.time()
opt_obj_val = cvx_prob.solve(solver="ECOS")
print("N = 100, cvxpy time cost = %.2f, opt value = %.4f" % (time.time() - t2, opt_obj_val))
# N = 100, cvxpy time cost = 0.09, opt value = -0.0557

N = 30000
W_large = np.random.uniform(low=0.5, high=1.5, size=(n, N))

t3 = time.time()
opt_obj_val = osmm_prob.solve(init_val, W_large)
print("N = 30,000, osmm time cost = %.2f, opt value = %.5f" % (time.time() - t3, opt_obj_val))
# N = 30,000, osmm time cost = 1.12, opt value = -0.00074

cvx_prob = cp.Problem(cp.Minimize(-cp.sum(cp.log(W_large.T @ x_var)) / N), [cp.sum(x_var) == 1])
t4 = time.time()
opt_obj_val = cvx_prob.solve(solver="ECOS")
print("N = 30,000, cvxpy time cost = %.2f, opt value = %.5f" % (time.time() - t4, opt_obj_val))
# N = 30,000, cvxpy time cost = 39.02, opt value = -0.00074
```

### Other optional arguments
Other optinal arguments for the `solve` method are as follows.
* `W_validate` is a scalar, a numpy array, or a numpy matrix in the same shape as `W`. If `W` contains a sampling matrix, then `W_validate` can be used to provide another sampling matrix that gives `f(x, W_validate)`, which is then compared with `f(x, W)` to validate the sampling accuracy. Default is `None`.
* `hessian_rank` is the (maximum) rank of the low-rank quasi-Newton matrix used in the method, and with `hessian_rank=0` the method becomes a proximal bundle algorithm. Default is `20`.
*  `gradient_memory` is the memory in the piecewise affine bundle used in the method, and with `gradient_memory=0` the method becomes a proximal quasi-Newton algorithm. Default is `20`.
* `max_iter` is the maximum number of iterations. Default is `200`.
* `check_gap_frequency` is the number of iterations between when we check the gap. Default is 10.
* `solver` must be one of the solvers supported by CVXPY. Default value is `'ECOS'`.
* `verbose` is a boolean giving the choice of printing information during the iterations. Default value is `False`.
* `use_cvxpy_param` is a boolean giving the choice of using CVXPY parameters. Default value is `False`.
* `store_var_all_iters` is a boolean giving the choice of whether the updates of `x` in all iterations are stored. Default value is `True`.
* The following tolerances are used in the stopping criteria.
    * `eps_gap_abs` and `eps_gap_rel` are absolute and relative tolerances on the gap between upper and lower bounds on the optimal objective, respectively. Default values are `1e-4` and `1e-3`, respectively.
    * `eps_res_abs` and `eps_res_rel` are absolute and relative tolerances on a residue for an optimality condition, respectively. Default values are `1e-4` and `1e-3`, respectively.

### Return values
The optimal objective is returned by the `solve` method.
A solution for `x` and the other variables in `g` can be obtained in the `value` attribute of the corresponding CVXPY variables.

More detailed results are stored in the dictonary `method_results`, which is an attribute of an `OSMM` object. The keys are as follows.
* `"objf_iters"` stores the objective value versus iterations.
* `"lower_bound_iters"` stores lower bound on the optimal objective value versus iterations.
* `"total_iters"` stores the actual number of iterations taken.
* `"objf_validate_iters"` stores the validate objective value versus iterations, when `W_validate` is provided.
* More detailed histories during the iterations are as follows.
  * `"var_iters"` stores the update of `x` versus iterations. It can be turned off by setting the argument `store_var_all_iters=False`.
  * `"time_iters"` stores the time cost per iteration versus iterations.
  * `"rms_res_iters"` stores the RMS value of optimality residue versus iterations.
  * `"f_grad_norm_iters"` stores the norm of the gradient of `f` versus iterations.
  * `"q_norm_iters"` stores the norm of `q` versus iterations.
  * `"v_norm_iters"` stores the norm of `v` versus iterations.
  * `"lam_iters"` stores the value of the penalty parameter versus iterations.
  * `"mu_iters"` stores the value of `mu` versus iterations.
  * `"t_iters"` stores the value of `t` versus iterations.
  * `"num_f_evals_iters"` stores the number of `f` evaluations per iteration versus iterations.
  * `"time_detail_iters"` stores the time costs of computing the value of `f` once, the gradient of `f` once, the tentative update, and the lower bound versus iterations.

## Citing
