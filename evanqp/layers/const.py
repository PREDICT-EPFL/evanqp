from evanqp.layers import BaseLayer


class ConstLayer(BaseLayer):

    def __init__(self, x, depth):
        self.x = x

        super().__init__(x.shape[0], depth)

    def add_constr(self, model, p_layer=None):
        for i in range(self.out_size):
            model.addConstr(self.vars['out'][i] == self.x[i])

    def compute_bounds(self, method, p_layer=None, **kwargs):
        self.bounds['out']['lb'] = self.x
        self.bounds['out']['ub'] = self.x

    def forward(self, x, warm_start=False):
        return self.x
