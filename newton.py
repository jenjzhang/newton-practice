# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
### Newton's Method

def prime(f, x, h=0.001):
    '''Evaluate the first derivative of a function at a given opoint using a finite difference approach.'''
    return (f(x + h)-f(x - h))/(2 * h)

def double_prime(f, x, h=0.001):
    '''Evaluate the second derivative of a function at a given opoint using a finite difference approach.'''
    return (prime(f, x + h, h)-prime(f, x - h, h))/(2 * h)


def optimize(x_0, f, tol=0.00001, h=0.001):
    '''Implements univariate Newton's method. Accepts a starting value and functiont to optimize, and returns a final x value.
    tol and h are passed as default parameters.'''

    x_t = x_0
    # print(x_t)
    while True:
        x_next = x_t - prime(f, x_t, h) / double_prime(f, x_t, h)
        if abs(x_next - x_t) < tol:
            break
        x_t = x_next
        # print(x_t)

    return x_next

# print(optimize(6, lambda x: x**2-1))
