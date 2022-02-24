import numpy as np
from gurobipy import GRB, LinExpr

from evanqp.layers import BaseLayer, BoundArithmetic


class LinearLayer(BaseLayer):

    def __init__(self, weight, bias, depth):
        super().__init__(weight.shape[0], depth)

        self.weight = weight
        self.bias = bias

    def add_constr(self, model, p_layer):
        for i in range(self.out_size):
            model.addConstr(self.vars['out'][i] == LinExpr(self.weight[i, :], p_layer.vars['out']) + self.bias[i])

    def add_vars_jacobian(self, model, p_layer):
        self.vars['out_jac'] = np.empty((self.weight.shape[0], p_layer.vars['out_jac'].shape[1]), dtype=object)
        for i in range(self.weight.shape[0]):
            for j in range(p_layer.vars['out_jac'].shape[1]):
                self.vars['out_jac'][i, j] = model.addVar(vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY)

    def add_constr_jacobian(self, model, p_layer):
        for i in range(self.weight.shape[0]):
            for j in range(p_layer.vars['out_jac'].shape[1]):
                model.addConstr(self.vars['out_jac'][i, j] == LinExpr(self.weight[i, :], p_layer.vars['out_jac'][:, j]))

    def compute_bounds_jacobian(self, p_layer, **kwargs):
        W_p = np.clip(self.weight, 0, None)
        W_m = np.clip(self.weight, None, 0)

        self.jacobian_bounds['out'] = {}
        self.jacobian_bounds['out']['lb'] = W_m @ p_layer.jacobian_bounds['out']['ub'] + W_p @ p_layer.jacobian_bounds['out']['lb']
        self.jacobian_bounds['out']['ub'] = W_p @ p_layer.jacobian_bounds['out']['ub'] + W_m @ p_layer.jacobian_bounds['out']['lb']

    def compute_bounds(self, method, p_layer, **kwargs):
        if method == BoundArithmetic.INT_ARITHMETIC:
            self._compute_bounds_ia(p_layer)
        elif method == BoundArithmetic.ZONO_ARITHMETIC:
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

    def forward(self, x, warm_start=False):
        return self.weight @ x + self.bias
