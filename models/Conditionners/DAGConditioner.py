import torch
import torch.nn as nn
from .Conditioner import Conditioner
from models.MLP import IdentityNN


class DAGMLP(nn.Module):
    def __init__(self, in_size, hidden, out_size, cond_in=0):
        super(DAGMLP, self).__init__()
        in_size = in_size
        l1 = [in_size + cond_in] + hidden
        l2 = hidden + [out_size]
        layers = []
        for h1, h2 in zip(l1, l2):
            layers += [nn.Linear(h1, h2), nn.ReLU()]
        layers.pop()
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class DAGConditioner(Conditioner):
    def __init__(self, in_size, hidden, out_size, cond_in=0, soft_thresholding=True, h_thresh=0., gumble_T=1.,
                 hot_encoding=False, l1=0., nb_epoch_update=1, A_prior=None):
        super(DAGConditioner, self).__init__()
        if A_prior is None:
            self.A = nn.Parameter(torch.ones(in_size, in_size) * 1.5 + torch.randn((in_size, in_size)) * .02)
        else:
            self.A = nn.Parameter(A_prior * 1.5 + torch.randn((in_size, in_size)) * .5)
        self.in_size = in_size
        self.s_thresh = soft_thresholding
        self.h_thresh = h_thresh
        self.stoch_gate = True
        self.noise_gate = False
        in_net = in_size*2 if hot_encoding else in_size
        if issubclass(type(hidden), nn.Module):
            self.embedding_net = hidden
        else:
            self.embedding_net = DAGMLP(in_net, hidden, out_size, cond_in)
        self.gumble = True
        self.hutchinson = False
        self.gumble_T = gumble_T
        self.hot_encoding = hot_encoding
        with torch.no_grad():
            self.constrainA(h_thresh)
        # Buffers related to the optimization of the constraints on A
        self.register_buffer("lambd", torch.tensor(.0))
        self.register_buffer("c", torch.tensor(1e-3))
        self.register_buffer("eta", torch.tensor(10.))
        self.register_buffer("gamma", torch.tensor(.9))
        self.register_buffer("lambd", torch.tensor(.0))
        self.register_buffer("l1_weight", torch.tensor(l1))
        self.register_buffer("dag_const", torch.tensor(1.))
        self.d = in_size
        self.tol = 1e-20
        _, S, _ = torch.svd(self.A**2)
        S = S.abs()
        sigma_max = S.max().item()
        #print(sigma_max)
        self.register_buffer("alpha", torch.tensor(1. / sigma_max) ** 2)
        self.register_buffer("prev_trace", self.get_power_trace())
        #print(self.prev_trace)
        self.nb_epoch_update = nb_epoch_update
        self.no_update = 0
        #exit()

    def get_dag(self):
        return self

    def post_process(self, zero_threshold):
        self.stoch_gate = False
        self.noise_gate = False
        self.s_thresh = False
        self.h_thresh = 0.
        self.A.data = (self.soft_thresholded_A().data.clone().abs() > zero_threshold).float()
        self.A *= 1. - torch.eye(self.in_size, device=self.A.device)
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
            hot_encoding = torch.eye(self.in_size, device=self.A.device).unsqueeze(0).expand(x.shape[0], -1, -1)\
                .contiguous().view(-1, self.in_size)
            e = self.embedding_net(e)
            full_e = torch.cat((e, hot_encoding), 1).view(x.shape[0], self.in_size, -1)
            # TODO Add context
            return full_e

        return self.embedding_net(e).view(x.shape[0], self.in_size, -1)#.permute(0, 2, 1).contiguous().view(x.shape[0], -1)

    def constrainA(self, zero_threshold=.0001):
        self.A *= (self.A.clone().abs() > zero_threshold).float()
        self.A *= 1. - torch.eye(self.in_size, device=self.A.device)
        return

    def get_power_trace(self):
        alpha = min(1, self.alpha)
        if self.hutchinson != 0:
            h_iter = self.hutchinson
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

    def update_dual_param(self):
        with torch.no_grad():
            lag_const = self.get_power_trace()
            if self.dag_const > 0. and lag_const > self.tol:
                self.lambd = self.lambd + self.c * lag_const
                # Absolute does not make sense (but copied from DAG-GNN)
                if lag_const.abs() > self.gamma*self.prev_trace.abs():
                    self.c *= self.eta
                print(self.c)
                self.prev_trace = lag_const
            elif self.dag_const > 0.:
                print("DAGness is very low: %f -> Post processing" % torch.log(lag_const), flush=True)
                self.post_process(1e-1)
                _, S, _ = torch.svd(self.A ** 2)
                S = S.abs()
                sigma_max = S.max().item()
                sigma_max = 1 if sigma_max <= 0 else sigma_max
                self.alpha = torch.tensor(1. / sigma_max) ** 2
                dag_const = self.get_power_trace()
                print("DAGness is still very low: %f" % torch.log(dag_const), flush=True)
                if dag_const > 0.:
                    print("Error in post-processing.", flush=True)
                    self.A.requires_grad = True
                    self.A.grad = self.A*0
                    self.stoch_gate = True
                    self.noise_gate = False
                    self.s_thresh = True
                    self.h_thresh = 0.
                    self.A *= 2
                    _, S, _ = torch.svd(self.A ** 2)
                    S = S.abs()
                    sigma_max = S.max().item()
                    self.alpha = torch.tensor(1. / sigma_max) ** 2
                    self.prev_trace = self.get_power_trace()
                else:
                    self.dag_const = torch.tensor(0.)
                    #self.l1_weight = torch.tensor(0.)
                    print("Post processing successful.")
                    print("Number of edges is %d VS number max is %d" %
                          (int(self.A.sum().item()), ((self.d - 1)*self.d)/2), flush=True)

            else:
                print("DAGness is still very low: %f" % torch.log(self.get_power_trace()), flush=True)
        return lag_const

    def loss(self):
        lag_const = self.get_power_trace()
        loss = self.dag_const*(self.lambd*lag_const + self.c/2*lag_const**2) + self.l1_weight*self.A.abs().mean()
        return loss

    def step(self, epoch_number, loss_avg=0.):
        with torch.no_grad():
            if epoch_number % self.nb_epoch_update == 0:
                if self.in_size < 30:
                    print(self.soft_thresholded_A(), flush=True)
                print(self.loss().abs(), loss_avg.abs(), flush=True)
                if self.loss().abs() < loss_avg.abs()/2 or self.no_update > 2:
                    print("Update param", flush=True)
                    self.update_dual_param()
                    self.no_update = 0
                else:
                    print("No Update param", flush=True)
                    self.no_update += 1
