# osmm
`osmm` is a Python package for oracle-structured minimization method, which solves problems in the following form
```
minimize f(x, W) + g(x),
```
where `x` is the variable, and `W` contains data and parameters that specify `f`. 
The oracle function `f( ,W)` is convex in `x`, defined by PyTorch, and can be automatically differentiated by PyTorch. 
The structured function `g` is convex, defined by cvxpy, and can contain constraints and variables additional to `x`.

Variable `x` can be a scalar, a vector, or a matrix.

`osmm` is suitable for cases where `W` contains a large data matrix.
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
which generates an object defining the form of the problem, and a member function of the `OSMM` class
```python
solve(W, init_val)
```
which specifies the problem, runs the solve method, and returns the optimal objective value.

### Required arguments
For the construction method of the `OSMM` class, the arguments `f_torch` and `g_cvxpy` define the form of the problem.
* `f_torch` must be a function with two inputs and one output. The first and the second inputs are the PyTorch tensors for `W` and `x`, respectively. The output is a PyTorch tensor for the scalar function value of `f`.
* `g_cvxpy` must be a function with no input and three outputs. The first output is a cvxpy variable for `x`, the second one is a cvxpy expression for the objective function in `g`, the third one is a list of constraints contained in `g`.

For the solve method, the argument `W` specifies the problem to be solved, and `init_val` gives an initial value of `x`.
* `W` must be a scalar, a numpy array, or a numpy matrix.
* `init_val` must be a scalar, a numpy array, or a numpy matrix that is in the same shape as `x`. It must be in the domain of `f`.

### Basic examples
We take the following Kelly gambling problem as an example
```
minimize - \sum_{i=1}^N log(w_i'x) / N
subject to x >= 0, x'1 = 1,
```
where `x` is an `n` dimensional variable, and `w_i` for `i=1,...,N` are given data.
The objective function is `f`, the indicator function of the constraints is `g`, and the data matrix `W = [w_1, ..., w_N]`.
The code is as follows.
```python
import torch
import numpy as np
import cvxpy as cp
from osmm import OSMM

n = 100
N = 10000

# Define a cvxpy variable for x.
x_var = cp.Variable(n, nonneg=True)

# Define f by torch.
def my_f_torch(w_torch, x_torch):
    objf = -torch.mean(torch.log(torch.matmul(w_torch.T, x_torch)))
    return objf

# Define g by cvxpy.
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
opt_obj_val = osmm_prob.solve(W, init_val)

# A solution for x is stored in x_var.value.
print("x solution = ", x_var.value)
```

`osmm` accepts matrix variables. Take the following bipartite graph newsvendor problem as an example
```
minimize - \sum_{i=1}^N p_i' min(As, d_i) / N + t ||A||_1
subject to A >= 0, \sum_{j=1}^m A_ij <= 1,
```
where `A` is an `m` by `d` variable, `m` is the number of sale nodes, and `d` is the number of source nodes.
The amounts of product on the source nodes `s`, the samples of prices on sale nodes `p_i`,
the samples of demands on sale nodes `d_i`, and the regularization parameter `t` are given.
The first term in the objective function, i.e., the negative revenue, is `f`,
the second term plus the indicator function of the constraints is `g`,
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
init_val = np.ones((m, d))
A_var = cp.Variable((m, d))

def my_g_cvxpy():
    g = t * cp.sum(cp.abs(A_var))
    constrs = [A_var >= 0, cp.sum(A_var, axis=0) <= np.ones(m)]
    return A_var, g, constrs

def my_f_torch(w_torch, A_torch):
    d_torch = w_torch[0:m, :]
    p_torch = w_torch[m:2 * m, :]
    source_torch = torch.tensor(source, dtype=torch.float, requires_grad=False)
    q_torch = torch.matmul(A_torch, source_torch)
    revenue = torch.sum(p_torch * torch.min(d_torch, q_torch[:, None])) / N
    return -revenue
    
osmm_prob = OSMM(my_f_torch, my_g_cvxpy)
result = osmm_prob.solve(W, init_val)
```

For more examples, see the [`examples`](examples/) directory.

### Optional arguments
There are some optional arguments for the `solve` method.
* `W_validate` is a numpy matrix, a numpy array, or a scalar in the same shape as `W`. If `W` contains a sampling matrix, then `W_validate` can be used to provide another sampling matrix that gives `f(x, W_validate)`, which is then compared with `f(x, W)` to validate the sampling accuracy.
* `hessian_rank` is the (maximum) rank of the low-rank quasi-Newton matrix used in the method, and with `hessian_rank=0` the method becomes a proximal bundle algorithm. The default value is `20`.
*  `gradient_memory` is the memory in the piecewise affine bundle used in the method, and with `gradient_memory=0` the method becomes a proximal quasi-Newton algorithm. The default value is `20`. Please see the paper for details.
* `max_iter` is the maximum number of iterations. The default value is `200`.
* `solver` must be one of the solvers supported by cvxpy.
* `store_var_all_iters` is a boolean giving the choice of whether the updates of `x` in all iterations are stored. The default value is `True`.
* The following tolerances are used in the stopping criteria.
    * `eps_gap_abs` is the absolute tolerance on the gap between upper and lower bounds on the objective value. The default value is `1e-4`.
    * `eps_gap_rel` is the relative tolerance on the gap between upper and lower bounds on the objective value. The default value is `1e-4`.
    * `eps_res_abs` is the absolute tolerance on a residue for an optimality condition. The default value is `1e-4`.
    * `eps_res_rel` is the relative tolerance on a residue for an optimality condition. The default value is `1e-4`.

### Return values
The optimal objective is returned by the `solve` method.
A solution for `x` and the other variables in `g` can be obtained in the `value` attribute of the corresponding cvxpy variables.

More detailed results are stored in the dictonary `method_results`, which is an attribute of an `OSMM` object. The keys are as follows.
* `"objf_iters"` stores the objective value versus iterations.
* `"lower_bound_iters"` stores lower bound on the optimal objective value versus iterations.
* `"iters_taken"` stores the actual number of iterations taken.
* `"objf_validate_iters"` stores the validate objective value versus iterations, when `W_validate` is provided.
* More detailed histories during the iterations are as follows.
  * `"var_iters"` stores the update of `x` versus iterations. It can be turned off by setting the argument `store_var_all_iters=False` in the `solve()` method.
  * `"runtime_iters"` stores the time cost per iteration versus iterations.
  * `"opt_res_iters"` stores the norm of the optimality residue versus iterations.
  * `"f_grad_norm_iters"` stores the norm of the gradient of `f` versus iterations.
  * `"q_norm_iters"` stores the norm of `q` versus iterations.
  * `"v_norm_iters"` stores the norm of `v` versus iterations.
  * `"lambd_iters"` stores the value of the penalty parameter versus iterations.
  * `"mu_iters"` stores the value of `mu` versus iterations.
  * `"t_iters"` stores the value of `t` versus iterations.
  * `"num_f_evas_line_search_iters"` stores the number of `f` evaluations in the line search versus iterations.
  * `"time_cost_detail_iters"` stores the time costs of evaluating `f` and gradient of `f` (once), the tentative update, and the lower bound versus iterations.

## Efficiency
`osmm` is efficient when `W` contains a large data matrix, and can be more efficient if PyTorch uses a GPU to compute `f` and its gradient.

We take the Kelly gambling problem as an example again. 
The variable `x_var`, the functions `my_f_torch` and `my_g_cvxpy`, and `init_val` have been defined in the above.
We compare the time cost of `osmm` with cvxpy on a CPU, and show that `osmm` is not as efficient as cvxpy when the data matrix is small with `N=100`,
but is more efficient when the data matrix is large with `N=10,000`.

```python
import time as time

N = 100
W_small = np.random.uniform(low=0.5, high=1.5, size=(n, N))

t1 = time.time()
opt_obj_val = osmm_prob.solve(W_small, init_val)
print("N = 100, osmm time cost = %.2f" % (time.time() - t1))
# N = 100, osmm time cost = 0.38

cvx_prob = cp.Problem(cp.Minimize(-cp.sum(cp.log(W_small.T @ x_var)) / N), [cp.sum(x_var) == 1])
t2 = time.time()
cvx_prob.solve()
print("N = 100, cvxpy time cost = %.2f" % (time.time() - t2))
# N = 100, cvxpy time cost = 0.04

N = 10000
W_large = np.random.uniform(low=0.5, high=1.5, size=(n, N))

t3 = time.time()
opt_obj_val = osmm_prob.solve(W_large, init_val)
print("N = 10,000, osmm time cost = %.2f" % (time.time() - t3))
# N = 10,000, osmm time cost = 0.67

cvx_prob = cp.Problem(cp.Minimize(-cp.sum(cp.log(W_large.T @ x_var)) / N), [cp.sum(x_var) == 1])
t4 = time.time()
cvx_prob.solve()
print("N = 10,000, cvxpy time cost = %.2f" % (time.time() - t4))
# N = 10,000, cvxpy time cost = 2.37
```
## Citing
