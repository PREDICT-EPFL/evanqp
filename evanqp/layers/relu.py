import numpy as np
from gurobipy import GRB, LinExpr

from evanqp.layers import BaseLayer, BoundArithmetic, LinearLayer


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
        if method == BoundArithmetic.INT_ARITHMETIC:
            self._compute_bounds_ia(p_layer)
        elif method == BoundArithmetic.ZONO_ARITHMETIC:
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

    def compute_ideal_cuts(self, model, p_layer, pp_layer):
        if not isinstance(p_layer, LinearLayer) or not isinstance(pp_layer, BaseLayer):
            return []

        # https://link.springer.com/content/pdf/10.1007/s10107-020-01474-5.pdf Proposition 12,13
        _x = np.asarray(model.cbGetNodeRel(pp_layer.vars['out']))
        _y = np.asarray(model.cbGetNodeRel(self.vars['out']))
        _z = np.asarray(model.cbGetNodeRel(self.vars['r']))

        ineqs = []
        for neuron in range(self.out_size):
            # no cuts are added for a fixed neuron
            if p_layer.bounds['out']['lb'][neuron] > 0 or p_layer.bounds['out']['ub'][neuron] < 0:
                continue

            # check for relaxed binary decision variable
            if _z[neuron] < 0 or _z[neuron] > 1:
                continue

            _L = np.zeros(pp_layer.out_size)
            _U = np.zeros(pp_layer.out_size)
            for i in range(pp_layer.out_size):
                if p_layer.weight[neuron, i] >= 0:
                    _L[i] = pp_layer.bounds['out']['lb'][i]
                    _U[i] = pp_layer.bounds['out']['ub'][i]
                else:
                    _L[i] = pp_layer.bounds['out']['ub'][i]
                    _U[i] = pp_layer.bounds['out']['lb'][i]

            I_map = (p_layer.weight[neuron, :] * _x) < (p_layer.weight[neuron, :] * (_L * (1 - _z[neuron]) + _U * _z[neuron]))

            rhs = p_layer.bias[neuron] * _z[neuron]
            for i in range(pp_layer.out_size):
                if I_map[i]:
                    rhs += p_layer.weight[neuron, i] * (_x[i] - _L[i] * (1 - _z[neuron]))
                else:
                    rhs += p_layer.weight[neuron, i] * _U[i] * _z[neuron]

            # only add most violated constraint
            if _y[neuron] > rhs:
                rhs = LinExpr()
                s = p_layer.bias[neuron]
                for i in range(pp_layer.out_size):
                    if I_map[i]:
                        rhs.addTerms(p_layer.weight[neuron, i], pp_layer.vars['out'][i])
                        rhs.addConstant(-p_layer.weight[neuron, i] * _L[i])
                        rhs.addTerms(p_layer.weight[neuron, i] * _L[i], self.vars['r'][neuron])
                    else:
                        s += p_layer.weight[neuron, i] * _U[i]
                rhs.addTerms(s, self.vars['r'][neuron])

                ineqs.append((self.vars['out'][neuron], rhs))

        return ineqs

    def forward(self, x, warm_start=False):
        if warm_start:
            for i in range(self.out_size):
                self.vars['r'][i].Start = 0 if x[i] <= 0 else 1

        return np.clip(x, 0, None)
