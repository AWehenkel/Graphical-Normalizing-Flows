import torch.nn as nn


class Conditioner(nn.Module):
    def __init__(self):
        super(Conditioner, self).__init__()

    def forward(self, x, context):
        pass
