from gurobipy import LinExpr

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

    def compute_bounds(self, method, p_layer=None):
        if isinstance(self.input_set, Polytope):
            self.bounds['out']['lb'] = self.input_set.bounding_box().lb
            self.bounds['out']['ub'] = self.input_set.bounding_box().ub
        elif isinstance(self.input_set, Box):
            self.bounds['out']['lb'] = self.input_set.lb
            self.bounds['out']['ub'] = self.input_set.ub

        if method == BoundArithmetic.ZONO_ARITHMETIC:
            self.zono_bounds['out'] = Zonotope.zonotope_from_box(self.bounds['out']['lb'], self.bounds['out']['ub'])
