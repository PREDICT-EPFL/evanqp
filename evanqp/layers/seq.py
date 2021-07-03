import torch.nn as nn

from evanqp.layers import BaseLayer, LinearLayer, ReluLayer


class SeqLayer(BaseLayer):

    def __init__(self, layers, depth):
        super().__init__(layers[-1].out_size, depth)

        self.layers = layers

    def add_vars(self, model):
        for layer in self.layers:
            layer.add_vars(model)
        self.vars = self.layers[-1].vars

    def add_constr(self, model, p_layer):
        for layer in self.layers:
            layer.add_constr(model, p_layer)
            p_layer = layer

    def compute_bounds(self, method, p_layer):
        for layer in self.layers:
            layer.compute_bounds(method, p_layer)
            p_layer = layer
        self.bounds = self.layers[-1].bounds
        self.zono_bounds = self.layers[-1].zono_bounds

    def compute_ideal_cuts(self, model, p_layer, pp_layer):
        ineqs = []
        for layer in self.layers:
            ineqs += layer.compute_ideal_cuts(model, p_layer, pp_layer)
            pp_layer = p_layer
            p_layer = layer
        return ineqs

    @staticmethod
    def from_pytorch(pytorch_model, start_depth=1):
        if not isinstance(pytorch_model, nn.Sequential):
            pytorch_model = nn.Sequential(pytorch_model)

        layers = []
        for i, layer in enumerate(pytorch_model):
            if isinstance(layer, nn.Linear):
                weight = layer.weight.detach().cpu().double().numpy()
                bias = layer.bias.detach().cpu().double().numpy()
                layers.append(LinearLayer(weight, bias, i + start_depth))
            elif isinstance(layer, nn.ReLU):
                out_size = layers[i - 1].out_size
                layers.append(ReluLayer(out_size, i + start_depth))
            elif callable(getattr(layer, 'milp_layer')):
                layers.append(layer.milp_layer(i + start_depth))
            else:
                raise NotImplementedError(f'Pytorch Layer {layer} not supported.')

        return SeqLayer(layers, start_depth)

    def forward(self, x, warm_start=False):
        for layer in self.layers:
            x = layer.forward(x, warm_start=warm_start)
        return x
