import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, in_d, hidden, out_d, act_f=nn.ReLU(), device="cpu"):
        super().__init__()
        self.in_d = in_d
        self.hiddens = hidden
        self.out_d = out_d
        self.act_f = act_f
        self.device = device
        layers_dim = [in_d] + hidden + [out_d]
        layers = []
        for dim_in, dim_out in zip(layers_dim[:-1], layers_dim[1]):
            layers += [nn.Linear(dim_in, dim_out), act_f]
        self.net = nn.Sequential(*layers).to(device)

    def forward(self, x):
        return self.net(x)

    def to(self, device):
        self.net.to(device)
        self.device = device
        return self


