import torch.nn as nn


class FFNN(nn.Sequential):

    def __init__(self, sizes, input_size=2, output_size=1, bias=True):
        layers = [nn.Linear(input_size, sizes[0], bias=bias), nn.ReLU()]
        for i in range(1, len(sizes)):
            layers += [
                nn.Linear(sizes[i - 1], sizes[i], bias=bias),
                nn.ReLU(),
            ]
        layers += [nn.Linear(sizes[-1], output_size, bias=bias)]

        super().__init__(*layers)
