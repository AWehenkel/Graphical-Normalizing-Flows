import torch
import torch.nn as nn
from UMNN import IntegrandNetwork, UMNNMAF

class IdentityNN(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class DAGNN(nn.Module):
    def __init__(self, d, device="cpu", soft_thresholding=True, h_thresh=0., net=None):
        super().__init__()
        self.A = nn.Parameter(torch.randn(d, d, device=device)*.1 + .5)
        self.d = d
        self.device = device
        self.s_thresh = soft_thresholding
        self.h_thresh = h_thresh
        self.net = net if net is not None else IdentityNN()

    def to(self, device):
        self.A = self.A.to(device)
        self.device = device
        return self

    def soft_thresholded_A(self):
        return 2*(torch.sigmoid(2*(self.A**2)) -.5)

    def hard_thresholded_A(self):
        if self.s_thresh:
            return torch.relu(self.soft_thresholded_A() - self.h_thresh)
        return torch.relu(self.A**2 - self.h_thresh)

    def forward(self, x):
        if self.h_thresh > 0:
            e = (x.unsqueeze(1).expand(-1, self.d, -1) * self.hard_thresholded_A().unsqueeze(0)
                 .expand(x.shape[0], -1, -1)).view(x.shape[0] * self.d, -1)
        elif self.s_thresh:
            e = (x.unsqueeze(1).expand(-1, self.d, -1) * self.soft_thresholded_A().unsqueeze(0)
                 .expand(x.shape[0], -1, -1)).view(x.shape[0]*self.d, -1)
        else:
            e = (x.unsqueeze(1).expand(-1, self.d, -1) * self.A.unsqueeze(0).expand(x.shape[0], -1, -1))\
                .view(x.shape[0]*self.d, -1)
        return self.net(e).view(x.shape[0], self.d, -1).permute(0, 2, 1).contiguous().view(x.shape[0], -1)

    def constrainA(self, zero_threshold=.0001):
        #self.A /= (self.A.sum(1).unsqueeze(1).expand(-1, self.d) + 1e-5)
        self.A *= (self.A.clone().abs() > zero_threshold).float()
        self.A *= 1. - torch.eye(self.d, device=self.device)
        return

    def get_power_trace(self, alpha, Hutchinson=False):
        if Hutchinson:
            h_iter = 1
            trace = 0.
            I = torch.eye(self.d, device=self.device)
            for j in range(h_iter):
                e0 = torch.randn(self.d, 1).to(self.device)
                e = e0
                for i in range(self.d):
                    e = (I + alpha*self.A**2) @ e

                trace += (e0 * e).sum()
            return trace/h_iter - self.d

        B = (torch.eye(self.d, device=self.device) + alpha*self.A**2)
        M = torch.matrix_power(B, self.d)
        return torch.diag(M).sum() - self.d


class DAGEmbedding(nn.Module):
    def __init__(self, in_d, emb_d=-1, emb_net=None, hiddens_integrand=[50, 50, 50, 50], act_func='ELU', device="cpu"):
        super().__init__()
        self.m_embeding = None
        self.device = device
        self.in_d = in_d
        self.emb_d = in_d if emb_net is None else emb_d
        self.dag = DAGNN(in_d, net=emb_net, device=device)
        self.parallel_nets = IntegrandNetwork(in_d, 1 + in_d + self.emb_d, hiddens_integrand, 1, act_func=act_func,
                                              device=device)

    def to(self, device):
        self.dag.to(device)
        self.parallel_nets.to(device)
        return self

    def make_embeding(self, x_made, context=None):
        b_size = x_made.shape[0]
        self.m_embeding = torch.cat((self.dag.forward(x_made), torch.eye(self.in_d, device=self.device).unsqueeze(0)
                                     .expand(b_size, -1, -1).view(b_size, -1)), 1)
        return self.m_embeding

    def forward(self, x_t):
        return self.parallel_nets.forward(x_t, self.m_embeding)


class DAGNF(nn.Module):
    def __init__(self, in_d, hidden_integrand=[50, 50, 50], emb_net=None, emb_d=-1, act_func='ELU',
                 nb_steps=20, solver="CCParallel", device="cpu", l1_weight=1.):
        super().__init__()
        self.dag_embedding = DAGEmbedding(in_d, emb_d, emb_net, hidden_integrand, act_func, device)
        self.UMNN = UMNNMAF(self.dag_embedding, in_d, nb_steps=nb_steps, device=device, solver=solver)
        self.lambd = .5
        self.c = .5
        self.eta = 2
        self.gamma = .5
        self.d = in_d
        self.prev_trace = self.dag_embedding.dag.get_power_trace(self.c / self.d)
        self.tol = 1e-4
        self.l1_weight = l1_weight

    def to(self, device):
        self.dag_embedding.to(device)
        self.UMNN.to(device)
        self.prev_trace.to(device)
        return self

    def forward(self, x):
        return self.UMNN(x)

    def compute_ll(self, x):
        return self.UMNN.compute_ll(x)

    def DAGness(self):
        return self.dag_embedding.dag.get_power_trace(self.c / self.d)

    def loss(self, x):
        ll, _ = self.UMNN.compute_ll(x)
        lag_const = self.dag_embedding.dag.get_power_trace(self.c/self.d)
        loss = self.lambd*lag_const + self.c/2*lag_const**2 - ll.mean() + self.l1_weight*self.dag_embedding.dag.A.abs().mean()
        return loss

    def update_dual_param(self):
        with torch.no_grad():
            lag_const = self.dag_embedding.dag.get_power_trace(self.c / self.d)
            if lag_const > self.tol:
                self.lambd = self.lambd + self.c * lag_const
                # Absolute does not make sense (but copied from DAG-GNN)
                if lag_const.abs() > self.gamma*self.prev_trace.abs():
                    self.c *= self.eta
                self.prev_trace = lag_const
        return lag_const

