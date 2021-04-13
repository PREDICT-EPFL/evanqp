import numpy as np
import cvxpy as cp


def cheby_center(A, b):
    """Computes the Chebyshev center of polytope Ax <= b.
    See https://en.wikipedia.org/wiki/Chebyshev_center
    """
    (n, p) = A.shape
    A1 = np.zeros((n, p+1))
    A1[:, 0:p] = A
    A1[:, p] = np.sqrt(np.sum(A**2, axis=1))

    x = cp.Variable(p+1, name='x')
    prob = cp.Problem(cp.Maximize(x[p]), [A1 @ x <= b])
    prob.solve()

    c = x.value[0:p]
    r = x.value[p]
    return c, r


def cprnd(N, A, b):
    """ Draw from the uniform distribution over a convex polytope.
    Adopted from https://ch.mathworks.com/matlabcentral/fileexchange/34208-uniform-distribution-over-a-convex-polytope
    """
    def stdize(z):
        return z / np.linalg.norm(z)

    (m, p) = A.shape

    def gen_dir():
        return stdize(np.random.randn(p))

    x0, _ = cheby_center(A, b)

    run_up = 10 * (p + 1)
    discard = 5 * (p + 1)

    X = np.zeros((N + run_up + discard, p))

    n = 0
    x = x0

    M = np.zeros(p)

    while n < N + run_up + discard:
        # test whether in run_up or not
        if n < run_up:
            # choose a direction
            u = gen_dir()
        else:
            # choose a previous point at random
            v = X[np.random.randint(n), :].T
            # line sampling direction is from v to sample mean
            u = (v - M) / np.linalg.norm(v - M)
        # determine intersections of x + ut with the polytope
        z = A @ u
        c = (b - A @ x) / z
        t_min = np.max(c[z < 0])
        t_max = np.min(c[z > 0])
        # choose a random point on that line segment
        x = x + (t_min + (t_max - t_min) * np.random.rand()) * u

        X[n, :] = x
        n += 1

        # incremental mean and covariance updates
        delta0 = x - M
        M = M + delta0 / n

    return X[discard+run_up:, :]
