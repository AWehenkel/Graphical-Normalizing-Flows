import torch
import torch.nn as nn
from UMNN import IntegrandNetwork, UMNNMAF
from .LinearFlow import LinearNormalizer

class IdentityNN(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, context=None):
        return x


class DAGNN(nn.Module):
    def __init__(self, d, device="cpu", soft_thresholding=True, h_thresh=0., net=None, gumble_T=1.):
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
        self.gumble_T = gumble_T
        with torch.no_grad():
            self.constrainA(h_thresh)

    def to(self, device):
        self.A = self.A.to(device)
        self.device = device
        return self

    def get_dag(self):
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
            temp = self.gumble_T
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

    def forward(self, x, context=None):
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
        return self.net(e, context).view(x.shape[0], self.d, -1).permute(0, 2, 1).contiguous().view(x.shape[0], -1)

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
    def __init__(self, in_d, emb_d=-1, emb_net=None, hiddens_integrand=[50, 50, 50, 50], act_func='ELU', device="cpu",
                 gumble_T=1.):
        super().__init__()
        self.m_embeding = None
        self.device = device
        self.in_d = in_d
        self.emb_d = in_d if emb_net is None else emb_d
        self.dag = DAGNN(in_d, net=emb_net, device=device, gumble_T=gumble_T)
        self.parallel_nets = IntegrandNetwork(in_d, 1 + in_d + self.emb_d, hiddens_integrand, 1, act_func=act_func,
                                              device=device)

    def get_dag(self):
        return self.dag

    def to(self, device):
        self.dag.to(device)
        self.parallel_nets.to(device)
        return self

    def make_embeding(self, x_made, context=None):
        b_size = x_made.shape[0]
        self.m_embeding = torch.cat((self.dag.forward(x_made), torch.eye(self.in_d, device=self.device).unsqueeze(0)
                                     .expand(b_size, -1, -1).view(b_size, -1)), 1)
        return self.m_embeding

    def forward(self, x_t, context=None):
        return self.parallel_nets.forward(x_t, self.m_embeding)


class ListModule(object):
    def __init__(self, module, prefix, *args):
        """
        The ListModule class is a container for multiple nn.Module.
        :nn.Module module: A module to add in the list
        :string prefix:
        :list of nn.module args: Other modules to add in the list
        """
        self.module = module
        self.prefix = prefix
        self.num_module = 0
        for new_module in args:
            self.append(new_module)

    def append(self, new_module):
        if not isinstance(new_module, nn.Module):
            raise ValueError('Not a Module')
        else:
            self.module.add_module(self.prefix + str(self.num_module), new_module)
            self.num_module += 1

    def to(self, device):
        for module in self:
            module.to(device)

    def __len__(self):
        return self.num_module

    def __getitem__(self, i):
        i = self.num_module + i if i < 0 else i
        if i < 0 or i >= self.num_module:
            raise IndexError('Out of bound')
        return getattr(self.module, self.prefix + str(i))


class DAGNF(nn.Module):
    def __init__(self, emb_nets, nb_flow=1, **kwargs):
        super().__init__()
        self.device = kwargs['device']
        self.nets = ListModule(self, "DAGFlow")
        for i in range(nb_flow):
            model = DAGStep(emb_net=emb_nets[i], **kwargs)
            self.nets.append(model)

    def to(self, device):
        self.nets.to(device)
        return self

    def set_steps_nb(self, nb_steps):
        for net in self.nets:
            net.set_steps_nb(nb_steps)

    def forward(self, x):
        for net in self.nets:
            x = net.forward(x)
        return x

    def compute_ll(self, x):
        jac_tot = 0.
        if len(self.nets) > 1:
            for id_net in range(len(self.nets) -1):
                net = self.nets[id_net]
                x, jac = net.compute_log_jac(x)
                jac_tot += jac.sum(1)
        ll, z = self.nets[-1].compute_ll(x)
        ll += jac_tot
        return ll, z

    def DAGness(self):
        dagness = []
        for net in self.nets:
            dagness.append(net.DAGness())
        return dagness

    def set_h_threshold(self, threshold):
        for net in self.nets:
            net.dag_embedding.get_dag().h_thresh = threshold

    def set_nb_steps(self, nb_steps):
        for net in self.nets:
            net.set_steps_nb(nb_steps)

    def loss(self, x):
        loss_tot = 0.
        if len(self.nets) > 1:
            for id_net in range(len(self.nets) - 1):
                net = self.nets[id_net]
                x, loss = net.loss(x, only_jac=True)
                loss_tot += loss
        return loss_tot + self.nets[-1].loss(x)

    def constrainA(self, zero_threshold):
        for net in self.nets:
            net.constrainA(zero_threshold=zero_threshold)

    def getDag(self, index=0):
        return self.nets[index].dag_embedding.get_dag()

    def update_dual_param(self):
        for net in self.nets:
            net.update_dual_param()


class DAGStep(nn.Module):
    def __init__(self, in_d, hidden_integrand=[50, 50, 50], emb_net=None, emb_d=-1, act_func='ELU', gumble_T=1.,
                 nb_steps=20, solver="CCParallel", device="cpu", l1_weight=1., linear_normalizer=False):
        super().__init__()
        self.linear_normalizer = linear_normalizer
        if linear_normalizer:
            self.dag_embedding = DAGNN(in_d, device=device, soft_thresholding=True, h_thresh=0., net=IdentityNN(),
                                       gumble_T=gumble_T)
            self.normalizer = LinearNormalizer(self.dag_embedding, emb_net, in_d, device=device)
        else:
            self.dag_embedding = DAGEmbedding(in_d, emb_d, emb_net, hidden_integrand, act_func, device, gumble_T=gumble_T)
            self.normalizer = UMNNMAF(self.dag_embedding, in_d, nb_steps=nb_steps, device=device, solver=solver)
        self.lambd = .0
        self.c = 1e-3
        self.eta = 10
        self.gamma = .9
        self.d = in_d
        self.prev_trace = self.dag_embedding.get_dag().get_power_trace(self.c / self.d)
        self.tol = 1e-20
        self.l1_weight = l1_weight
        self.dag_const = 1.

    def to(self, device):
        self.dag_embedding.to(device)
        self.normalizer.to(device)
        self.prev_trace.to(device)
        return self

    def set_steps_nb(self, nb_steps):
        if not self.linear_normalizer:
            self.normalizer.set_steps_nb(nb_steps)

    def forward(self, x):
        return self.normalizer(x)

    def compute_log_jac(self, x, context=None):
        return self.normalizer.compute_log_jac_bis(x, context)

    def compute_ll(self, x):
        return self.normalizer.compute_ll(x)

    def DAGness(self):
        alpha = .1 / self.d
        return self.dag_embedding.get_dag().get_power_trace(alpha)

    def loss(self, x, only_jac=False):
        if only_jac:
            x, ll = self.compute_log_jac(x)
            ll = ll.sum(1)
        else:
            ll, _ = self.normalizer.compute_ll(x)
        alpha = .1/self.d
        lag_const = self.dag_embedding.get_dag().get_power_trace(alpha)
        loss = self.dag_const*(self.lambd*lag_const + self.c/2*lag_const**2) - ll.mean() + \
               self.l1_weight*self.dag_embedding.get_dag().A.abs().mean()
        if only_jac:
            return x, loss
        return loss

    def constrainA(self, zero_threshold):
        self.dag_embedding.get_dag().constrainA(zero_threshold=zero_threshold)

    def getDag(self):
        return self.dag_embedding.get_dag()

    def update_dual_param(self):
        with torch.no_grad():
            alpha = .1/self.d#self.c / self.d
            lag_const = self.dag_embedding.get_dag().get_power_trace(alpha)
            if self.dag_const > 0. and lag_const > self.tol:
                self.lambd = self.lambd + self.c * lag_const
                # Absolute does not make sense (but copied from DAG-GNN)
                if lag_const.abs() > self.gamma*self.prev_trace.abs():
                    self.c *= self.eta
                self.prev_trace = lag_const
            elif self.dag_const > 0:
                self.dag_embedding.get_dag().post_process(1e-1)
                self.dag_const = 0.
        return lag_const

    def set_h_threshold(self, threshold):
        self.dag_embedding.get_dag().h_thresh = threshold

