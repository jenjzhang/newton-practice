### Newton's Method

import numbers
import numpy as np
from scipy.optimize import approx_fprime
from scipy.linalg import solve
from scipy.optimize._numdiff import approx_derivative


def grad_f(f, x, h=1e-3):
    """Evaluate the gradient."""
    return approx_derivative(f, x, method='2-point')


def hess_f(f, x, h=1e-3):
    """Evaluate the Hessian."""
    grad_func = lambda v: grad_f(f, v, h=h)
    return approx_derivative(grad_func, x, method='2-point')

def update_hessian(f, x):
    """
    Calculate matrix product of inverse Hessian and grad f
    """
    
    g = grad_f(f, x)
    H = hess_f(f, x)

    print("hessian: ", H)
    
    # Solve H * delta = g   instead of computing inverse explicitly
    delta = solve(H, g)
    return x - delta


def optimize(x_0, f, tol=0.00001, h=1e-3, max_iter=1000):
    """Implements univariate Newton's method. Accepts a starting value and function to optimize, and returns a final x value.
    tol, h, and max_iter are passed as default parameters."""


    iter_count = 0
    x_t = x_0
    print(x_t)
    while iter_count < max_iter:

        x_next = x_t - update_hessian(f, x_t)
        if np.linalg.norm(x_next - x_t) < tol:
            break
        x_t = x_next
        print("x_t: ", x_t)
        iter_count += 1

        if iter_count >= max_iter:
            print("Max number of iterations reached! :(")
            break
        # print(x_t)

    return x_next


print(optimize(np.array([0,0]), lambda v: 3*(v[0]-1)**2 + 2*(v[1]+2)**2))
