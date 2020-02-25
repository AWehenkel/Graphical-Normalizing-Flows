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
        self.A = nn.Parameter(torch.ones(d, d, device=device)*1. + torch.randn((d, d), device=device)*.02)
        self.d = d
        self.device = device
        self.s_thresh = soft_thresholding
        self.h_thresh = h_thresh
        self.stoch_gate = True
        self.noise_gate = False
        self.net = net if net is not None else IdentityNN()
        self.gumble = True
        with torch.no_grad():
            self.constrainA(h_thresh)

    def to(self, device):
        self.A = self.A.to(device)
        self.device = device
        return self

    def post_process(self, zero_threshold):
        self.stoch_gate = False
        self.noise_gate = False
        self.s_thresh = False
        self.h_thresh = 0.
        self.A *= (self.soft_thresholded_A().clone().abs() > zero_threshold).float()
        self.A *= 1. - torch.eye(self.d, device=self.device)
        self.A /= self.A + (self.A == 0).float()
        self.A.requires_grad = False
        self.A.grad = None

    def stochastic_gate(self, importance):
        if self.gumble:
            # Gumble soft-max gate
            temp = .5
            epsilon = 1e-6
            g1 = -torch.log(-torch.log(torch.rand(importance.shape, device=self.device)))
            g2 = -torch.log(-torch.log(torch.rand(importance.shape, device=self.device)))
            z1 = torch.exp((torch.log(importance + epsilon) + g1)/temp)
            z2 = torch.exp((torch.log(1 - importance + epsilon) + g2)/temp)
            return z1 / (z1 + z2)

        else:
            beta_1, beta_2 = 3., 10.
            sigma = beta_1/(1. + beta_2*torch.sqrt((importance - .5)**2.))
            mu = importance
            z = torch.randn(importance.shape, device=self.device) * sigma + mu + .25
            #non_importance = torch.sqrt((importance - 1.)**2)
            #z = z - non_importance/beta_1
            return torch.relu(z.clamp_max(1.))

    def noiser_gate(self, x, importance):
        noise = torch.randn(importance.shape, device=self.device) * torch.sqrt((1 - importance)**2)
        return importance*(x + noise)

    def soft_thresholded_A(self):
        return 2*(torch.sigmoid(2*(self.A**2)) -.5)

    def hard_thresholded_A(self):
        if self.s_thresh:
            return self.soft_thresholded_A()*(self.soft_thresholded_A() > self.h_thresh).float()
        return self.A**2 * (self.A**2 > self.h_thresh).float()

    def forward(self, x):
        if self.h_thresh > 0:
            if self.stoch_gate:
                e = (x.unsqueeze(1).expand(-1, self.d, -1) * self.stochastic_gate(self.hard_thresholded_A().unsqueeze(0)
                 .expand(x.shape[0], -1, -1))).view(x.shape[0] * self.d, -1)
            elif self.noise_gate:
                e = self.noiser_gate(x.unsqueeze(1).expand(-1, self.d, -1),
                                     self.hard_thresholded_A().unsqueeze(0)
                                     .expand(x.shape[0], -1, -1))\
                    .view(x.shape[0] * self.d, -1)
            else:
                e = (x.unsqueeze(1).expand(-1, self.d, -1) * self.hard_thresholded_A().unsqueeze(0)
                     .expand(x.shape[0], -1, -1)).view(x.shape[0] * self.d, -1)
        elif self.s_thresh:
            if self.stoch_gate:
                e = (x.unsqueeze(1).expand(-1, self.d, -1) * self.stochastic_gate(self.soft_thresholded_A().unsqueeze(0)
                     .expand(x.shape[0], -1, -1))).view(x.shape[0] * self.d, -1)
            elif self.noise_gate:
                e = self.noiser_gate(x.unsqueeze(1).expand(-1, self.d, -1),
                                     self.soft_thresholded_A().unsqueeze(0).expand(x.shape[0], -1, -1))\
                    .view(x.shape[0] * self.d, -1)
            else:
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
        self.lambd = .0
        self.c = 1e-3
        self.eta = 10
        self.gamma = .9
        self.d = in_d
        self.prev_trace = self.dag_embedding.dag.get_power_trace(self.c / self.d)
        self.tol = 1e-15
        self.l1_weight = l1_weight
        self.dag_const = 1.

    def to(self, device):
        self.dag_embedding.to(device)
        self.UMNN.to(device)
        self.prev_trace.to(device)
        return self

    def set_steps_nb(self, nb_steps):
        self.UMNN.set_steps_nb(nb_steps)

    def forward(self, x):
        return self.UMNN(x)

    def compute_ll(self, x):
        return self.UMNN.compute_ll(x)

    def DAGness(self):
        alpha = .1 / self.d
        return self.dag_embedding.dag.get_power_trace(alpha)

    def loss(self, x):
        ll, _ = self.UMNN.compute_ll(x)
        alpha = .1/self.d
        lag_const = self.dag_embedding.dag.get_power_trace(alpha)
        loss = self.dag_const*(self.lambd*lag_const + self.c/2*lag_const**2) - ll.mean() + \
               self.l1_weight*self.dag_embedding.dag.A.abs().mean()
        return loss

    def constrainA(self, zero_threshold):
        self.dag_embedding.dag.constrainA(zero_threshold=zero_threshold)

    def getDag(self):
        return self.dag_embedding.dag

    def update_dual_param(self):
        with torch.no_grad():
            alpha = .1/self.d#self.c / self.d
            lag_const = self.dag_embedding.dag.get_power_trace(alpha)
            if self.dag_const > 0. and lag_const > self.tol:
                self.lambd = self.lambd + self.c * lag_const
                # Absolute does not make sense (but copied from DAG-GNN)
                if lag_const.abs() > self.gamma*self.prev_trace.abs():
                    self.c *= self.eta
                self.prev_trace = lag_const
            elif self.dag_const > 0:
                self.dag_embedding.dag.post_process(1e-1)
                self.dag_const = 0.
        return lag_const

