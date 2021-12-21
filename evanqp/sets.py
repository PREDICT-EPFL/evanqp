from abc import ABC, abstractmethod
import cvxpy as cp
import numpy as np
import scipy

from evanqp.utils import cheby_center


class Set(ABC):
    @abstractmethod
    def sample(self):
        """Draws a deterministic sample from the set
        """
        pass


class Polytope(Set):

    def __init__(self, A, b):
        self.A = A
        self.b = b
        self._bounding_box = None

    def sample(self):
        c, r = cheby_center(self.A, self.b)
        return c

    def as_polytope(self):
        return self

    def bounding_box(self):
        if self._bounding_box is not None:
            return self._bounding_box

        n = self.A.shape[1]

        lb = -np.inf * np.ones(n)
        ub = np.inf * np.ones(n)

        x = cp.Variable(n)
        constraints = [self.A @ x <= self.b]
        for i in range(n):
            prob = cp.Problem(cp.Minimize(x[i]), constraints)
            prob.solve()
            if prob.status in [cp.INFEASIBLE, cp.UNBOUNDED]:
                raise Exception('Could not find bounding box. Is your polytope compact?')
            lb[i] = prob.value

            prob = cp.Problem(cp.Maximize(x[i]), constraints)
            prob.solve()
            if prob.status in [cp.INFEASIBLE, cp.UNBOUNDED]:
                raise Exception('Could not find bounding box. Is your polytope compact?')
            ub[i] = prob.value

        self._bounding_box = Box(lb, ub)
        return self._bounding_box


class Box(Set):

    def __init__(self, lb, ub):
        self.lb = lb
        self.ub = ub

    def sample(self):
        return (self.lb + self.ub) / 2

    def as_polytope(self):
        n = self.lb.shape[0]
        A = scipy.linalg.block_diag(np.eye(n), -np.eye(n))
        b = np.concatenate((self.ub, -self.lb))
        return Polytope(A, b)
