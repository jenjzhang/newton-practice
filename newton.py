### Newton's Method


def prime(f, x, h=1e-7):
    """Evaluate the first derivative of a function at a given opoint using a finite difference approach."""
    return (f(x + h) - f(x - h)) / (2 * h)


def double_prime(f, x, h=1e-7):
    """Evaluate the second derivative of a function at a given opoint using a finite difference approach."""
    return (prime(f, x + h, h) - prime(f, x - h, h)) / (2 * h)


def optimize(x_0, f, tol=0.00001, h=1e-7, max_iter=1000):
    """Implements univariate Newton's method. Accepts a starting value and function to optimize, and returns a final x value.
    tol, h, and max_iter are passed as default parameters."""

    iter_count = 0
    x_t = x_0
    print(x_t)
    while iter_count < max_iter:
        x_next = x_t - prime(f, x_t, h) / double_prime(f, x_t, h)
        if abs(x_next - x_t) < tol:
            break
        x_t = x_next
        iter_count += 1

        if iter_count >= max_iter:
            print("Max number of iterations reached! :(")
            break
        # print(x_t)

    return x_next


# print(optimize(6, lambda x: x**2+2*x-1))
