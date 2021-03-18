# osmm
`osmm` is a Python package for oracle-structured minimization method, which solves problems in the following form
```
minimize f(x, W) + g(x),
```
where `x` is the variable, and `W` is a given data matrix. The oracle function `f` is defined by Pytorch, and the structured function `g` is defined by cvxpy.

The implementation is based on our paper XXX.

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
`osmm` exposes the `OSMM` object 
```python
OSMM(f_torch, g_cvxpy, get_initial_val, W, W_validate=None)
```
and its solve method
```python
OSMM.solve(max_num_rounds=100, H_rank=20, M=20, solver="ECOS", ini_by_Hutchison=True, stop_early=True, alg_mode=AlgMode.LowRankQNBundle, num_iter_eval_Lk=10, tau_min=1e-3, mu_min=1e-4, mu_max=1e5, mu_0=1.0, gamma_inc=1.1, gamma_dec=0.8, alpha=0.05, beta=0.5, j_max=10)
```

#### Arguments

#### Return value

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
    
def my_cvxpy_description():
    x_var = cp.Variable(n, nonneg=True)
    g = 0
    constr = [cp.sum(x_var) == 1]
    return x_var, g, constr
    
def my_initial_val():
    return np.ones(n) / n
```
Next, with a given `n` by `N` data matrix `W`, the user can define an `OSMM` object.
```python
osmm_prob = OSMM(my_f_torch, my_cvxpy_description, my_initial_val, W)
```
The solve method is called by
```python
osmm_prob.solve()
```
and a solution is stored in `osmm_prob.method_results["x_best"]`.

For more examples, see the [`examples`](examples/) directory.


### Citing
