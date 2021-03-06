# osmm
`osmm` is a Python package for oracle-structured minimization method, which solves problems in the following form

<img src="https://github.com/cvxgrp/osmm/blob/main/readme_figs/eqn1.png" width="20%"/>

where *x* is the variable, and *W* contains data and parameters that specify *f*. 
The oracle function *f( ,W)* is convex in *x*, defined by PyTorch, and can be automatically differentiated by PyTorch. 
The structured function *g* is convex, defined by CVXPY, and can contain constraints and variables additional to *x*.

The variable can be a scalar, a vector, or a matrix. It does not have to be named as `x` in the code.

`osmm` is suitable for cases where *W* contains a large data matrix, as will be shown in an example later when we introduce the efficiency.

The implementation is based on our paper [*Minimizing Oracle-Structured Composite Functions*](https://web.stanford.edu/~boyd/papers/oracle_struc_composite.html)

## Installation
`osmm` requires
* [cvxpy](https://github.com/cvxgrp/cvxpy) >= 1.1.0a4
* [PyTorch](https://pytorch.org/) >= 1.6.0
* Python >= 3.7

To install `osmm`, first clone the repo, and then from inside the directory run 
```python3
python setup.py install
```
It may require root access, and if so, please use `sudo`.
CVXPY will be automatically installed by it (if not installed already),
but PyTorch won't and will need to be additionally installed.

## Usage
**Construct object.** Create an object of the `OSMM` class. For example
```python3
from osmm import OSMM
osmm_prob = OSMM()
```

**Define f by PyTorch.** An `OSMM` object has an attribute `f_torch`, which has the following attributes that define *f*.
* `f_torch.function` must be a function with one required input, one optional input, and one output.
    * The first input (required) is a PyTorch tensor for variable *x*. 
    * The second input (optional) is a PyTorch tensor for *W* and must be named `W_torch`. It is only needed when there is a data matrix in the problem.
    * The output is a PyTorch tensor for the scalar function value of *f*.
* `f_torch.W_torch` is a PyTorch tensor for *W*. The data type for `f_torch.W_torch` should be set as float by `dtype=torch.float`.

To explain the above let us see the following example. Suppose that we have had an `OSMM` object `osmm_prob`.
```python3
import torch
import numpy as np
n = 100
N = 10000

def my_f_torch(x_torch, W_torch):
    objf = -torch.mean(torch.log(torch.matmul(W_torch.T, x_torch)))
    return objf

# Set the attributes.
osmm_prob.f_torch.function = my_f_torch
osmm_prob.f_torch.W_torch = torch.tensor(np.random.uniform(low=0.5, high=1.5, size=(n, N)), requires_grad=False, dtype=torch.float)
```

**Define g by CVXPY.** An `OSMM` object has an attribute `g_cvxpy`, which has the following attributes that define *g*.
* `g_cvxpy.variable` is a CVXPY variable for *x*. 
* `g_cvxpy.objective` is a CVXPY expression for the objective function in *g*. 
* `g_cvxpy.constraints` is a list of constraints contained in *g*.
```python3
import cvxpy as cp
my_var = cp.Variable(n, nonneg=True)
my_g_constr = [cp.sum(my_var) == 1]

# Set the attributes 
osmm_prob.g_cvxpy.variable = my_var
osmm_prob.g_cvxpy.objective = 0
osmm_prob.g_cvxpy.constraints = my_g_constr
```

**Solve.** An `OSMM` object has a `solve` method. The `solve` method has one required argument, which gives an initial value of *x*. It must be a scalar, a numpy array, or a numpy matrix that is in the same shape as *x*, and it must be in the domain of *f*. The `solve` method returns the optimal objective value. A solution is stored in the `value` attribution of the corresponding CVXPY variable.
```python3
init_val = np.ones(n)
result = osmm_prob.solve(init_val)
my_soln = my_var.value
```

## Solve methods
**Default method.** The default is a low-rank quasi-Newton bundle method which works for the general problem introduced at the beginning.

**Other methods.** 
The `osmm` package also supports usage of a low-rank plus diagonal approximated Hessian that is based on eigenvalue decomposition of the exact Hessian,
when the objecitve function *f* has the following form

<img src="https://github.com/cvxgrp/osmm/blob/main/readme_figs/eqn5.png" width="18%"/>

where *F_i* is a convex scalar function, and has second-order derivative which is not everywhere zero.
To use this approximation, a PyTorch description of the elementwise mapping *F=(F_1,...,F_N)* from *R^N* to *R^N* is needed.
```python3
def my_elementwise_mapping_torch(y_scalar_torch):
    return -torch.log(y_scalar_torch) / N
```
It is passed in by setting the `f_torch.elementwise_mapping` attribute.
```python3
osmm_prob.f_torch.elementwise_mapping = my_elementwise_mapping_torch
```
Then when calling the solve method, to use the approximation based on eigen-decomposition of the exact Hessian, run
```python3
osmm_prob.solve(init_val, hessian_mode="LowRankDiagEVD")
```
To use exact Hessian, simply set argument `hessian_rank=n` in the above.

## Examples
**1. Basic example.** We have shown the code step by step in the above for a Kelly gambling problem

<img src="https://github.com/cvxgrp/osmm/blob/main/readme_figs/eqn2.png" width="25%"/>

where *x* is an *n* dimensional variable, and *w_i* for *i=1, ..., N* are given data.
The objective function has been treated as *f*, the indicator function of the constraints has been treated as *g*, and the data matrix *W = [w_1, ..., w_N]*.

**2. Matrix variable.** `osmm` accepts matrix variables. Take the following allocation problem as an example

<img src="https://github.com/cvxgrp/osmm/blob/main/readme_figs/eqn3.png" width="35%"/>

where *A* is an *m* by *d* variable.
The *d* dimensional vector *s*, the positive scalar *t*, 
and the *m* dimensional vectors *p_i* and *d_i* are given. 
The l1 norm notation is the sum of absolute values of all entries.
The first term in the objective function is *f*,
the regularization term plus the indicator function of the constraints is *g*,
and the data matrix *W = [(d_1, p_1), ..., (d_N, p_N)]*. 

```python
import torch
import numpy as np
import cvxpy as cp
from osmm import OSMM

d = 10
m = 20
N = 10000
s = np.random.uniform(low=1.0, high=5.0, size=(d))
W = np.exp(np.random.randn(2 * m, N))
t = 1

def my_f_torch(A_torch, W_torch):
    d_torch = W_torch[0:m, :]
    p_torch = W_torch[m:2 * m, :]
    s_torch = torch.tensor(s, dtype=torch.float, requires_grad=False)
    retail_node_amount = torch.matmul(A_torch, s_torch)
    ave_revenue = torch.sum(p_torch * torch.min(d_torch, retail_node_amount[:, None])) / N
    return -ave_revenue

A_var = cp.Variable((m, d), nonneg=True)

osmm_prob = OSMM()
osmm_prob.f_torch.function = my_f_torch
osmm_prob.f_torch.W_torch = torch.tensor(W, dtype=torch.float)
osmm_prob.g_cvxpy.variable = A_var
osmm_prob.g_cvxpy.objective = t * cp.sum(cp.abs(A_var))
osmm_prob.g_cvxpy.constraints = [cp.sum(A_var, axis=0) <= np.ones(d)]

result = osmm_prob.solve(np.ones((m, d)), verbose=True)
```

**3. Additional variables in g.** `osmm` accepts variables additional to *x* in *g*. Take the following simplified power flow problem as an example


<img src="https://github.com/cvxgrp/osmm/blob/main/readme_figs/eqn4.png" width="45%"/>

where *x* is an *n* dimensional variable, and *u* is an *m* dimensional variable.
The graph incidence matrix *A*, the *n* dimensional vectors *d_i* and *s_i*,
and the positive scalars *u_max* and *x_max* are given.
The objective function corresponds to *f*.
Minimization of the indicator function of the constraints over *u* gives *g*,
so *u* is an additional variable in *g*.
The data matrix *W = [(s_1, d_1), ..., (s_N, d_N)]*. 

```python
import torch
import numpy as np
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
u_var = cp.Variable(m)
constrs = [A @ u_var + x_var == 0, cp.norm(u_var, 'inf') <= u_max, cp.norm(x_var, 'inf') <= x_max]

def my_f_torch(x_torch, W_torch):
    s_torch = W_torch[0:n, :]
    d_torch = W_torch[n:n * 2, :]
    return torch.mean(torch.sum(torch.relu(-d_torch.T - s_torch.T - x_torch), axis=1))

osmm_prob = OSMM()
osmm_prob.f_torch.function = my_f_torch
osmm_prob.f_torch.W_torch = torch.tensor(W, dtype=torch.float)
osmm_prob.g_cvxpy.variable = x_var
osmm_prob.g_cvxpy.objective = 0
osmm_prob.g_cvxpy.constraints = constrs

result = osmm_prob.solve(np.ones(n), verbose=True)
```

For more examples, see the notebooks in the [`examples`](examples/) directory.

## Efficiency
`osmm` is efficient when *W* contains a large data matrix, and can be more efficient if PyTorch uses a GPU to compute *f* and its gradient.

Let us continue with the Kelly gambling example above.
We compare the time cost of `osmm` with CVXPY on a CPU, and show that `osmm` is not as efficient as CVXPY when the data matrix is small with *N=100*,
but is more efficient when the data matrix is large with *N=30,000*.

```python
import time as time
np.random.seed(0)

N = 100
W_small = np.random.uniform(low=0.5, high=1.5, size=(n, N))
osmm_prob.f_torch.W_torch = torch.tensor(W_small, dtype=torch.float, requires_grad=False, dtype=torch.float)

t1 = time.time()
opt_obj_val = osmm_prob.solve(init_val)
print("N = 100, osmm time cost = %.2f, opt value = %.4f" % (time.time() - t1, opt_obj_val))
# N = 100, osmm time cost = 0.19, opt value = -0.0557

cvx_prob = cp.Problem(cp.Minimize(-cp.sum(cp.log(W_small.T @ my_var)) / N), [cp.sum(my_var) == 1])
t2 = time.time()
opt_obj_val = cvx_prob.solve(solver="ECOS")
print("N = 100, cvxpy time cost = %.2f, opt value = %.4f" % (time.time() - t2, opt_obj_val))
# N = 100, cvxpy time cost = 0.09, opt value = -0.0557

N = 30000
W_large = np.random.uniform(low=0.5, high=1.5, size=(n, N))
osmm_prob.f_torch.W_torch = torch.tensor(W_large, dtype=torch.float, requires_grad=False)

t3 = time.time()
opt_obj_val = osmm_prob.solve(init_val)
print("N = 30,000, osmm time cost = %.2f, opt value = %.5f" % (time.time() - t3, opt_obj_val))
# N = 30,000, osmm time cost = 1.12, opt value = -0.00074

cvx_prob = cp.Problem(cp.Minimize(-cp.sum(cp.log(W_large.T @ my_var)) / N), [cp.sum(my_var) == 1])
t4 = time.time()
opt_obj_val = cvx_prob.solve(solver="ECOS")
print("N = 30,000, cvxpy time cost = %.2f, opt value = %.5f" % (time.time() - t4, opt_obj_val))
# N = 30,000, cvxpy time cost = 39.02, opt value = -0.00074
```

## Optional arguments and attributes
Another attribute of `OSMM.f_torch` is `W_validate_torch`, which is a torch tensor in the same shape as `W`. If `W` contains a sampling matrix, then `W_validate_torch` can be used to provide another sampling matrix that gives *f(x, W_validate)*, which is then compared with *f(x, W)* to validate the sampling accuracy. Default is `None`.

Optional arguments for the `solve` method that can change the algorithm used are as follows.
* `hessian_rank` is the (maximum) rank of the low-rank quasi-Newton matrix used in the method, and with `hessian_rank=0` the method becomes a proximal bundle algorithm. Default is `20`.
* `gradient_memory` is the memory in the piecewise affine bundle used in the method, and with `gradient_memory=0` the method becomes a proximal quasi-Newton algorithm. Default is `20`.
* `hessian_mode` either takes `"LowRankQNBundle"`, which is default, or `"LowRankDiagEVD"`, which can be applied only if *f* has a specific form as aforementioned.
* `bundle_mode` either takes `"LatestM"`, which is default and uses the latest cutting-planes in the bundle, or `"AllActive"`, which retains all active cutting-planes for the bundle.
* `solver` must be one of the solvers supported by CVXPY. Default value is `'ECOS'`.
* `verbose` is a boolean giving the choice of printing information during the iterations. Default value is `False`.

Optional arguments for the `solve` method which are less frequently adjusted are as follows.
* `max_iter` is the maximum number of iterations. Default is `200`.
* `check_gap_frequency` is the number of iterations between when we check the gap. Default is 10.
* `update_curvature_frequency` is the number of iterations between when the Hessian is updated. Default is 1.
* `use_cvxpy_param` is a boolean giving the choice of using CVXPY parameters. Default value is `False`.
* `store_var_all_iters` is a boolean giving the choice of whether the updates of *x* in all iterations are stored. Default value is `True`.
* `exact_g_line_search` is a boolean indicating if exact g evaluation is used in line-search. Default value is `False`.
* The following tolerances are used in the stopping criteria.
    * `eps_gap_abs` and `eps_gap_rel` are absolute and relative tolerances on the gap between upper and lower bounds on the optimal objective, respectively. Default values are `1e-4` and `1e-3`, respectively.
    * `eps_res_abs` and `eps_res_rel` are absolute and relative tolerances on a residue for an optimality condition, respectively. Default values are `1e-4` and `1e-3`, respectively.

## Results
The optimal objective is returned by the `solve` method.
A solution for *x* and the other variables in *g* can be obtained in the `value` attribute of the corresponding CVXPY variables.

More detailed results are stored in the dictionary `method_results`, which is an attribute of an `OSMM` object. The keys of the dictionary are as follows.
* `"objf_iters"` stores the objective value versus iterations.
* `"lower_bound_iters"` stores lower bound on the optimal objective value versus iterations.
* `"total_iters"` stores the actual number of iterations taken.
* `"objf_validate_iters"` stores the validate objective value versus iterations, when `W_validate` is provided.
* More detailed histories during the iterations are as follows.
  * `"var_iters"` stores the update of *x* versus iterations. It can be turned off by setting the argument `store_var_all_iters=False`.
  * `"time_iters"` stores the time cost per iteration versus iterations.
  * `"rms_res_iters"` stores the RMS value of optimality residue versus iterations.
  * `"f_grad_norm_iters"` stores the norm of the gradient of *f* versus iterations.
  * `"q_norm_iters"` stores the norm of *q* versus iterations.
  * `"v_norm_iters"` stores the norm of *v* versus iterations.
  * `"lam_iters"` stores the value of the penalty parameter versus iterations.
  * `"mu_iters"` stores the value of *mu* versus iterations.
  * `"t_iters"` stores the value of *t* versus iterations.
  * `"num_f_evals_iters"` stores the number of *f* evaluations per iteration versus iterations.
  * `"time_detail_iters"` stores the time costs of computing each of the following once versus iterations, the value of *f*, the gradient of *f*, the tentative update, the lower bound, and the curvature. 

## Citing
To cite our work, please use the following BibTex entry.

```
@article{oracle_struc_composite,
  author  = {Shen, Xinyue and Ali, Alnur and Boyd, Stephen},
  title   = {Minimizing Oracle-Structured Composite Functions},
  journal = {arXiv},
  year    = {2021},
}
```
