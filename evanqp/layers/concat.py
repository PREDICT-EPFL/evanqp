import numpy as np
from scipy.linalg import block_diag

from evanqp.layers import BaseLayer, BoundArithmetic
from evanqp.zonotope import Zonotope


class ConcatLayer(BaseLayer):

    def add_constr(self, model, p_layer):
        i = 0
        for layer in p_layer:
            for j in range(len(layer.vars['out'])):
                model.addConstr(self.vars['out'][i] == layer.vars['out'][j])
                i += 1

    def compute_bounds(self, method, p_layer, **kwargs):
        self.bounds['out']['lb'] = np.concatenate([layer.bounds['out']['lb'] for layer in p_layer])
        self.bounds['out']['ub'] = np.concatenate([layer.bounds['out']['ub'] for layer in p_layer])

        if method == BoundArithmetic.ZONO_ARITHMETIC:
            head = np.concatenate([layer.zono_bounds['out'].head for layer in p_layer])
            errors = block_diag(*[layer.zono_bounds['out'].errors for layer in p_layer])
            self.zono_bounds['out'] = Zonotope(head, errors)

    def forward(self, x, warm_start=False):
        return np.concatenate(x)
