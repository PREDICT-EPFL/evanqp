import cvxpy as cp
import numpy as np
import scipy


class Set:
    pass


class Polytope(Set):

    def __init__(self, A, b):
        self.A = A
        self.b = b
        self.interior_box = None

    def largest_interior_box(self):
        if self.interior_box is not None:
            return self.interior_box

        n = self.A.shape[1]

        lb = cp.Variable(n)
        ub = cp.Variable(n)
        constraints = [lb <= ub,
                       self.A @ lb <= self.b,
                       self.A @ ub <= self.b]
        objective = cp.Maximize(cp.sum(ub - lb))
        prob = cp.Problem(objective, constraints)
        prob.solve()

        if lb.value is None:
            raise Exception('Could not find largest interior box. Is your polytope compact?')

        self.interior_box = Box(lb.value, ub.value)
        return self.interior_box


class Box(Set):

    def __init__(self, lb, ub):
        self.lb = lb
        self.ub = ub

    def as_polytope(self):
        n = self.lb.shape[0]
        A = scipy.linalg.block_diag(np.eye(n), -np.eye(n))
        b = np.concatenate((self.ub, -self.lb))
        return Polytope(A, b)
