# osmm
`osmm` is a Python package for oracle-structured minimization method, which solves problems in the following form
```
minimize f(x, W) + g(x),
```
where `x` is the variable, and `W` is a given data matrix. The oracle function `f` is defined by Pytorch, and the structured function `g` is defined by cvxpy.

The implementation is based on our paper Minimizing Oracle-Structured Composite Functions.

### Installation
To install `osmm`, please run 
```
python setup.py install
```

`osmm` requires
* [cvxpy](https://github.com/cvxgrp/cvxpy) >= 1.1.0a4
* [PyTorch](https://pytorch.org/) >= 1.6.0
* Python 3.x


### Usage
`osmm` exposes the `OSMM` class 
```python
OSMM(f_torch, g_cvxpy, get_initial_val, W, W_validate=None)
```
which generates an object defining the problem, 
and a member function of the `OSMM` class
```python
solve(solver="ECOS", max_num_rounds=100, r=20, M=20)
```
which runs the solve method.

#### Arguments
The arguments `f_torch`, `g_cvxpy`, `get_initial_val`, `W`, and `W_validate` define the problem.
* `f_torch` must be a function with two inputs and one output. The first and the second inputs are the PyTorch tensors for the data matrix `W` and the variable vector `x`, respectively. The output is a PyTorch tensor for the scalar function value of `f`.
* `g_cvxpy` must be a function with no input and three outputs. The first output is a cvxpy variable for `x`, the second one is a cvxpy expression for the objective function in `g`, and the third one is a list of constraints contained in `g`.
* `get_initial_val` must be a function with no input and one output, which is a numpy array for an initial value of `x`.
* `W` must be a numpy matrix with dimension `n` by `N`, where `n` is the dimension of `x`.
* `W_validate` is not a required argument. For problems in which `W` is a sampling matrix, `W_validate` can be provided as another `n` by `N` sampling matrix used in `f(x, W_validate)`, which is then compared with `f(x, W)` to validate the sampling accuracy.

There are some arguments for the `solve` method. They all have default values, and are not required to be provided by the user.
* `solver` must be one of the solvers supported by cvxpy.
* `max_num_rounds` is the maximum number of iterations.
* `r` is the (maximum) rank of the low-rank quasi-Newton matrix used in the method, and with `r=0` the method becomes a proximal bundle algorithm. The default value is `20`.
*  `M` is the memory in the piecewise affine bundle used in the method, and with `M=0` the method becomes a proximal quasi-Newton algorithm. The default value is `20`. Please see the paper for details on `r` and `M`.

#### Return value
Results after solving are stored in the dictonary `method_results` which is an attribute of an `OSMM` object.
* `"x_best"` stores the solution of `x`.
* `"objf_iters"` stores the objective values during the iterations.
* `"lower_bound_iters"` stores lower bounds on the optimal objective value during the iterations.
* `"iters_taken"` stores the actual number of iterations taken.

#### Example
We take the following Kelly gambling problem as an example
```
minimize - \sum_{i=1}^N [log(w_i'x)] / N
subject to x >= 0, x'1 = 1,
```
where `x` is an `n` dimensional variable, and `w_i` for `i=1,...,N` are given data samples.

The user implements the objective function as `f` by PyTorch and the indicator function of the constraints as `g` by cvxpy,
and gives an initial value of `x` which is in the domain of `f`.
```python
n = 100

def my_f_torch(w_torch, x_torch):
    objf = -torch.mean(torch.log(torch.matmul(w_torch.T, x_torch)))
    return objf
    
def my_g_cvxpy():
    x_var = cp.Variable(n, nonneg=True)
    g = 0
    constr = [cp.sum(x_var) == 1]
    return x_var, g, constr
    
def my_initial_val():
    return np.ones(n) / n
```
Next, with a given `n` by `N` data matrix `W`, the user can define an `OSMM` object.
```python
osmm_prob = OSMM(my_f_torch, my_g_cvxpy, my_initial_val, W)
```
The solve method is called by
```python
osmm_prob.solve()
```
and a solution is stored in `osmm_prob.method_results["x_best"]`.

For more examples, see the [`examples`](examples/) directory.


### Citing
