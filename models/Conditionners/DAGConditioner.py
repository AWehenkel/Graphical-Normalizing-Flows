import torch
import torch.nn as nn
from .Conditioner import Conditioner
from models.MLP import IdentityNN


class DAGConditioner(Conditioner):
    def __init__(self, in_size, soft_thresholding=True, h_thresh=0., embedding_net=None, gumble_T=1., hot_encoding=False):
        super(DAGConditioner, self).__init__()
        self.A = nn.Parameter(torch.ones(in_size, in_size) * 1.5 + torch.randn((in_size, in_size)) * .02)
        self.in_size = in_size
        self.s_thresh = soft_thresholding
        self.h_thresh = h_thresh
        self.stoch_gate = True
        self.noise_gate = False
        self.embedding_net = embedding_net if embedding_net is not None else IdentityNN()
        self.gumble = True
        self.gumble_T = gumble_T
        self.hot_encoding = hot_encoding
        with torch.no_grad():
            self.constrainA(h_thresh)

    def get_dag(self):
        return self

    def post_process(self, zero_threshold):
        self.stoch_gate = False
        self.noise_gate = False
        self.s_thresh = False
        self.h_thresh = 0.
        self.A *= (self.soft_thresholded_A().clone().abs() > zero_threshold).float()
        self.A *= 1. - torch.eye(self.in_size, device=self.A.device)
        self.A /= self.A + (self.A == 0).float()
        self.A.requires_grad = False
        self.A.grad = None

    def stochastic_gate(self, importance):
        if self.gumble:
            # Gumble soft-max gate
            temp = self.gumble_T
            epsilon = 1e-6
            g1 = -torch.log(-torch.log(torch.rand(importance.shape, device=self.A.device)))
            g2 = -torch.log(-torch.log(torch.rand(importance.shape, device=self.A.device)))
            z1 = torch.exp((torch.log(importance + epsilon) + g1)/temp)
            z2 = torch.exp((torch.log(1 - importance + epsilon) + g2)/temp)
            return z1 / (z1 + z2)

        else:
            beta_1, beta_2 = 3., 10.
            sigma = beta_1/(1. + beta_2*torch.sqrt((importance - .5)**2.))
            mu = importance
            z = torch.randn(importance.shape, device=self.A.device) * sigma + mu + .25
            #non_importance = torch.sqrt((importance - 1.)**2)
            #z = z - non_importance/beta_1
            return torch.relu(z.clamp_max(1.))

    def noiser_gate(self, x, importance):
        noise = torch.randn(importance.shape, device=self.A.device) * torch.sqrt((1 - importance)**2)
        return importance*(x + noise)

    def soft_thresholded_A(self):
        return 2*(torch.sigmoid(2*(self.A**2)) -.5)

    def hard_thresholded_A(self):
        if self.s_thresh:
            return self.soft_thresholded_A()*(self.soft_thresholded_A() > self.h_thresh).float()
        return self.A**2 * (self.A**2 > self.h_thresh).float()

    def forward(self, x, context=None):
        if self.h_thresh > 0:
            if self.stoch_gate:
                e = (x.unsqueeze(1).expand(-1, self.in_size, -1) * self.stochastic_gate(self.hard_thresholded_A().unsqueeze(0)
                                                                                        .expand(x.shape[0], -1, -1)))\
                    .view(x.shape[0] * self.in_size, -1)
            elif self.noise_gate:
                e = self.noiser_gate(x.unsqueeze(1).expand(-1, self.in_size, -1),
                                     self.hard_thresholded_A().unsqueeze(0)
                                     .expand(x.shape[0], -1, -1))\
                    .view(x.shape[0] * self.in_size, -1)
            else:
                e = (x.unsqueeze(1).expand(-1, self.in_size, -1) * self.hard_thresholded_A().unsqueeze(0)
                     .expand(x.shape[0], -1, -1)).view(x.shape[0] * self.in_size, -1)
        elif self.s_thresh:
            if self.stoch_gate:
                e = (x.unsqueeze(1).expand(-1, self.in_size, -1) * self.stochastic_gate(self.soft_thresholded_A().unsqueeze(0)
                                                                                        .expand(x.shape[0], -1, -1))).view(x.shape[0] * self.in_size, -1)
            elif self.noise_gate:
                e = self.noiser_gate(x.unsqueeze(1).expand(-1, self.in_size, -1),
                                     self.soft_thresholded_A().unsqueeze(0).expand(x.shape[0], -1, -1))\
                    .view(x.shape[0] * self.in_size, -1)
            else:
                e = (x.unsqueeze(1).expand(-1, self.in_size, -1) * self.soft_thresholded_A().unsqueeze(0)
                     .expand(x.shape[0], -1, -1)).view(x.shape[0] * self.in_size, -1)
        else:
            e = (x.unsqueeze(1).expand(-1, self.in_size, -1) * self.A.unsqueeze(0).expand(x.shape[0], -1, -1))\
                .view(x.shape[0] * self.in_size, -1)

        if self.hot_encoding:
            hot_encoding = torch.eye(self.in_size).unsqueeze(0).expand(x.shape[0], -1, -1).contiguous().view(-1, self.in_size).to(self.device)
            full_e = torch.cat((e, hot_encoding), 1)
            return self.embedding_net(full_e, context).view(x.shape[0], self.in_size, -1)#.permute(0, 2, 1).contiguous().view(x.shape[0], -1)

        return self.embedding_net(e, context).view(x.shape[0], self.in_size, -1)#.permute(0, 2, 1).contiguous().view(x.shape[0], -1)

    def constrainA(self, zero_threshold=.0001):
        self.A *= (self.A.clone().abs() > zero_threshold).float()
        self.A *= 1. - torch.eye(self.in_size, device=self.A.device)
        return

    def get_power_trace(self, alpha, hutchinson=0):
        if hutchinson != 0:
            h_iter = hutchinson
            trace = 0.
            I = torch.eye(self.in_size, device=self.A.device)
            for j in range(h_iter):
                e0 = torch.randn(self.in_size, 1).to(self.A.device)
                e = e0
                for i in range(self.in_size):
                    e = (I + alpha * self.A ** 2) @ e

                trace += (e0 * e).sum()
            return trace / h_iter - self.in_size

        B = (torch.eye(self.in_size, device=self.A.device) + alpha * self.A ** 2)
        M = torch.matrix_power(B, self.in_size)
        return torch.diag(M).sum() - self.in_size
