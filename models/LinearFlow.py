import torch
import torch.nn as nn
from .DAGNN import DAGNN
import math


class LinearFlow(nn.Module):
    def __init__(self, in_d, linear_net=None, emb_net=None, device="cpu", l1_weight=1.):
        super().__init__()
        self.dag = DAGNN(in_d, device=device, soft_thresholding=True, h_thresh=0., net=emb_net)
        self.linear_net = linear_net
        self.lambd = .5
        self.c = .5
        self.eta = 2
        self.gamma = .5
        self.d = in_d
        self.prev_trace = self.dag.get_power_trace(self.c / self.d)
        self.tol = 1e-4
        self.pi = torch.tensor(math.pi).to(device)
        self.l1_weight = l1_weight

    def to(self, device):
        self.dag.to(device)
        self.linear_net.to(device)
        self.prev_trace.to(device)
        return self

    def forward(self, x):
        cond = self.dag(x).view(x.shape[0], -1, self.d).permute(0, 2, 1).contiguous()
        trans = self.linear_net.forward(cond.view(x.shape[0]*self.d, -1)).view(x.shape[0], -1, 2)
        mu, sigma = trans[:, :, 0], trans[:, :, 1]
        return x * torch.exp(sigma) + mu

    def compute_ll(self, x):
        cond = self.dag(x).view(x.shape[0], -1, self.d).permute(0, 2, 1).contiguous().view(x.shape[0]*self.d, -1)
        trans = self.linear_net.forward(cond).view(x.shape[0], -1, 2)
        mu, sigma = trans[:, :, 0], trans[:, :, 1]
        z = x * torch.exp(sigma) + mu
        log_prob_gauss = -.5 * (torch.log(self.pi * 2) + z ** 2).sum(1)
        ll = sigma.sum(1) + log_prob_gauss
        return ll, z

    def DAGness(self):
        return self.dag.get_power_trace(self.c / self.d)

    def constrainA(self, zero_threshold):
        self.dag.constrainA(zero_threshold=zero_threshold)

    def getDag(self):
        return self.dag

    def loss(self, x):
        ll, _ = self.compute_ll(x)
        lag_const = self.dag.get_power_trace(self.c/self.d)
        loss = self.lambd*lag_const + self.c/2*lag_const**2 - ll.mean() + self.l1_weight*self.dag.A.abs().mean()
        return loss

    def update_dual_param(self):
        with torch.no_grad():
            lag_const = self.dag.get_power_trace(self.c / self.d)
            if lag_const > self.tol:
                self.lambd = self.lambd + self.c * lag_const
                # Absolute does not make sense (but copied from DAG-GNN)
                if lag_const.abs() > self.gamma*self.prev_trace.abs():
                    self.c *= self.eta
                self.prev_trace = lag_const
        return lag_const

