# osmm
`osmm` is a Python package for oracle-structured minimization method.

The implementation is based on our paper XXX.

### Installation
To install `osmm`, please run 
```
python setup.py install
```

`osmm` requires
* [cvxpy](https://github.com/cvxgrp/cvxpy) >= 1.1.0a4
* [PyTorch](https://pytorch.org/) >= 1.6.0


The implementation is for problems in the following form
```
minimize f(x, W) + g(x),
```
where `x` is the variable, and `W` is a given data matrix.
The user defines `f` by PyTorch, `g` by CVXPY, and an initial point of `x` in the domain of `f`. 
Then by calling the `solve` method in the package, a solution is automatically generated.

### Usage

#### Arguments

#### Return value

#### Example
We take the following Kelly gambling problem as an example
```
minimize - E[log(w^T x)]
subject to x >= 0, x^T 1 = 1,
```
where `x` is an `n` dimensional variable, and the expectation is w.r.t. random variable `w`.
Approximating the expectation with a finite sum, the problem that we solve is
```
minimize - \sum_{i=1}^n [log(w_i^T x)] / N
subject to x >= 0, x^T 1 = 1,
```
where `w_i` for `i=1,...,N` are given data samples.
The objective function and the indicator function of the constraints correspond to functions `f` and `g`, respectively.

The user implements the function `f` by PyTorch and the function `g` by CVXPY as follows.
```python
def my_f_torch(w_torch, x_torch):
    objf = -torch.mean(torch.log(torch.matmul(w_torch.T, b_torch)))
    return objf
    
def my_cvxpy_description():
    x_var = cp.Variable(n, nonneg=True)
    g = 0
    constr = [cp.sum(x_var) == 1]
    return x_var, g, constr
```
Then the user gives an initial value of `x` which is in the domain of `f`.
```python
def my_initial_val():
    return np.ones(n) / n
```
Next, with a given `n` by `N` data matrix `W`, the user can define an `OSMM` object.
```python
osmm_prob = OSMM(my_f_torch, my_cvxpy_description, my_initial_val, W)
```
The user can call the solve method by
```python
osmm_prob.solve()
```
and a solution is stored in `osmm_prob.method_results["x_best"]`.

For more examples, see the [`examples`](examples/) directory.


### Citing
