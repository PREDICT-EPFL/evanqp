import torch.nn as nn


class Sequential(nn.Module):

    def __init__(self, *layers):
        super(Sequential, self).__init__()
        self.layers = nn.ModuleList(layers)

    def forward_until(self, i, x):
        for layer in self.layers[:i + 1]:
            x = layer(x)
        return x

    def forward_from(self, i, x):
        for layer in self.layers[i + 1:]:
            x = layer(x)
        return x

    def __len__(self):
        return len(self.layers)

    def __getitem__(self, i):
        return self.layers[i]

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class SeqNet(nn.Module):

    def __init__(self):
        super(SeqNet, self).__init__()

    def forward(self, x):
        x = self.blocks(x)
        return x

    def forward_until(self, i, x):
        """ Forward until layer i (inclusive) """
        x = self.blocks.forward_until(i, x)
        return x

    def forward_from(self, i, x):
        """ Forward from layer i (exclusive) """
        x = self.blocks.forward_from(i, x)
        return x


class FFNN(SeqNet):

    def __init__(self, sizes, input_size=2, output_size=1, bias=True):
        super(FFNN, self).__init__()

        layers = [nn.Linear(input_size, sizes[0], bias=bias), nn.ReLU()]
        for i in range(1, len(sizes)):
            layers += [
                nn.Linear(sizes[i - 1], sizes[i], bias=bias),
                nn.ReLU(),
            ]
        layers += [nn.Linear(sizes[-1], output_size, bias=bias)]
        self.blocks = Sequential(*layers)
