import numpy as np
from gurobipy import LinExpr

from evanqp.layers import BaseLayer, Bound


class LinearLayer(BaseLayer):

    def __init__(self, weight, bias, depth):
        super().__init__(weight.shape[0], depth)

        self.weight = weight
        self.bias = bias

    def add_constr(self, model, p_layer):
        for i in range(self.out_size):
            model.addConstr(self.vars['out'][i] == LinExpr(self.weight[i, :], p_layer.vars['out']) + self.bias[i])

    def compute_bounds(self, method, p_layer):
        if method == Bound.INT_ARITHMETIC:
            self._compute_bounds_ia(p_layer)
        elif method == Bound.ZONO_ARITHMETIC:
            self._compute_bounds_ia(p_layer)
            self._compute_bounds_zono(p_layer)
        else:
            raise NotImplementedError(f'Method {method} not implemented.')

    def _compute_bounds_ia(self, p_layer):
        W_p = np.clip(self.weight, 0, None)
        W_m = np.clip(self.weight, None, 0)

        self.bounds['out']['lb'] = W_m @ p_layer.bounds['out']['ub'] + W_p @ p_layer.bounds['out']['lb'] + self.bias
        self.bounds['out']['ub'] = W_p @ p_layer.bounds['out']['ub'] + W_m @ p_layer.bounds['out']['lb'] + self.bias

    def _compute_bounds_zono(self, p_layer):
        self.zono_bounds['out'] = p_layer.zono_bounds['out'].linear(self.weight, self.bias)
        lb, ub = self.zono_bounds['out'].concretize()
        self.bounds['out']['lb'] = np.maximum(self.bounds['out']['lb'], lb)
        self.bounds['out']['ub'] = np.minimum(self.bounds['out']['ub'], ub)
