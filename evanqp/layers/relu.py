import numpy as np
from gurobipy import GRB

from evanqp.layers import BaseLayer, Bound


class ReluLayer(BaseLayer):

    def __init__(self, out_size, depth):
        super().__init__(out_size, depth)

    def add_vars(self, model, linear_approx=False):
        super().add_vars(model)

        self.vars['r'] = [model.addVar(vtype=(GRB.CONTINUOUS if linear_approx else GRB.BINARY),
                                       lb=0,
                                       ub=1)
                          for _ in range(self.out_size)]
        for i in range(self.out_size):
            self.vars['r'][i].setAttr(GRB.Attr.BranchPriority, self.depth)

    def add_constr(self, model, p_layer, linear_approx=False):
        for i in range(self.out_size):
            if p_layer.bounds['out']['lb'][i] >= 0:
                model.addConstr(self.vars['out'][i] == p_layer.vars['out'][i])
            elif p_layer.bounds['out']['ub'][i] <= 0:
                model.addConstr(self.vars['out'][i] == 0)
            else:
                model.addConstr(self.vars['out'][i] >= 0)
                model.addConstr(self.vars['out'][i] >= p_layer.vars['out'][i])
                model.addConstr(self.vars['out'][i] <= p_layer.bounds['out']['ub'][i] * self.vars['r'][i])
                model.addConstr(self.vars['out'][i] <= p_layer.vars['out'][i] - p_layer.bounds['out']['lb'][i] * (1 - self.vars['r'][i]))
                if not linear_approx:
                    model.addConstr((self.vars['r'][i] == 1) >> (p_layer.vars['out'][i] >= 0))
                    model.addConstr((self.vars['r'][i] == 0) >> (p_layer.vars['out'][i] <= 0))

    def compute_bounds(self, method, p_layer):
        if method == Bound.INT_ARITHMETIC:
            self._compute_bounds_ia(p_layer)
        elif method == Bound.ZONO_ARITHMETIC:
            self._compute_bounds_ia(p_layer)
            self._compute_bounds_zono(p_layer)
        else:
            raise NotImplementedError(f'Method {method} not implemented.')

    def _compute_bounds_ia(self, p_layer):
        self.bounds['out']['lb'] = np.clip(p_layer.bounds['out']['lb'], 0, None)
        self.bounds['out']['ub'] = np.clip(p_layer.bounds['out']['ub'], 0, None)

    def _compute_bounds_zono(self, p_layer):
        p_lb, p_ub = p_layer.bounds['out']['lb'], p_layer.bounds['out']['ub']
        self.zono_bounds['out'] = p_layer.zono_bounds['out'].relu(bounds=(p_lb, p_ub))
        lb, ub = self.zono_bounds['out'].concretize()
        self.bounds['out']['lb'] = np.maximum(self.bounds['out']['lb'], lb)
        self.bounds['out']['ub'] = np.minimum(self.bounds['out']['ub'], ub)
