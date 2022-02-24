import numpy as np
from gurobipy import GRB, LinExpr

from evanqp.sets import Box, Polytope
from evanqp.zonotope import Zonotope
from evanqp.layers import BoundArithmetic, BaseLayer


class InputLayer(BaseLayer):

    def __init__(self, input_set):
        self.input_set = input_set
        if isinstance(self.input_set, Polytope):
            out_size = self.input_set.A.shape[1]
        elif isinstance(self.input_set, Box):
            out_size = self.input_set.lb.shape[0]
        else:
            raise RuntimeError('Set is not supported.')

        super().__init__(out_size, 0)

    def add_constr(self, model, p_layer=None):
        if isinstance(self.input_set, Polytope):
            A, b = self.input_set.A, self.input_set.b
            for i in range(A.shape[0]):
                model.addConstr(LinExpr(A[i, :], self.vars['out']) <= b[i])

    def add_vars_jacobian(self, model, p_layer):
        self.vars['out_jac'] = np.empty((self.out_size, self.out_size), dtype=object)
        for i in range(self.out_size):
            for j in range(self.out_size):
                if i == j:
                    self.vars['out_jac'][i, j] = model.addVar(vtype=GRB.CONTINUOUS, lb=1.0, ub=1.0)
                else:
                    self.vars['out_jac'][i, j] = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, ub=0.0)

    def add_constr_jacobian(self, model, p_layer):
        pass

    def compute_bounds_jacobian(self, p_layer=None, **kwargs):
        self.jacobian_bounds['out'] = {}
        self.jacobian_bounds['out']['lb'] = np.eye(self.out_size)
        self.jacobian_bounds['out']['ub'] = np.eye(self.out_size)

    def compute_bounds(self, method, p_layer=None, **kwargs):
        if isinstance(self.input_set, Polytope):
            self.bounds['out']['lb'] = self.input_set.bounding_box().lb
            self.bounds['out']['ub'] = self.input_set.bounding_box().ub
        elif isinstance(self.input_set, Box):
            self.bounds['out']['lb'] = self.input_set.lb
            self.bounds['out']['ub'] = self.input_set.ub

        if method == BoundArithmetic.ZONO_ARITHMETIC:
            self.zono_bounds['out'] = Zonotope.zonotope_from_box(self.bounds['out']['lb'], self.bounds['out']['ub'])

    def forward(self, x, warm_start=False):
        return self.input_set.sample()
