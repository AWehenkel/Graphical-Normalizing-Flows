import torch.nn as nn
import torch
import torch.nn.functional as F


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
        for dim_in, dim_out in zip(layers_dim[:-1], layers_dim[1:]):
            layers += [nn.Linear(dim_in, dim_out), act_f]
        layers.pop()
        self.net = nn.Sequential(*layers).to(device)

    def forward(self, x, context=None):
        return self.net(x)

    def to(self, device):
        self.net.to(device)
        self.device = device
        return self


class MNISTCNN(nn.Module):
    def __init__(self, out_d=10):
        super(MNISTCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, 1)
        self.conv2 = nn.Conv2d(16, 16, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(2304, 128)
        self.fc2 = nn.Linear(128, out_d)
        self.out_d = out_d

    def forward(self, x):
        x = self.conv1(x.view(-1, 1, 28, 28))
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        print(x.shape)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return x


