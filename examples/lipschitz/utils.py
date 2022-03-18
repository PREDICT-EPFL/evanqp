import numpy as np
import scipy


def dlqr(A, B, Q, R):
    """Solve the discrete time lqr controller.

    returns state-feedback law: u[k] = -K x[k]
    for discrete-time state-space mode: x[k+1] = A x[k] + B u[k]
    by minimizing cost function: sum x[k].T*Q*x[k] + u[k].T*R*u[k]
    and returns infinite horizon solution S
    """
    S = scipy.linalg.solve_discrete_are(A, B, Q, R)
    K = np.linalg.solve(B.T @ S @ B + R, B.T @ S @ A)
    eig_vals, _ = np.linalg.eig(A - B @ K)
    return K, S, eig_vals
