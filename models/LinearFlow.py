import torch
import torch.nn as nn
import numpy as np
import math


class LinearNormalizer(nn.Module):
    def __init__(self, conditioner, net, input_size, device="cpu"):
        super(LinearNormalizer, self).__init__()
        #self.model = nn.Sequential(nn.Linear(28**2, 100), nn.Linear(100, 100), nn.Linear(100, 1))
        self.conditioner = conditioner.to(device)
        self.device = device
        self.linear_net = net
        self.input_size = input_size
        self.d = input_size
        self.pi = nn.Parameter(torch.tensor(math.pi).to(self.device), requires_grad=False)

    def forward(self, x, context=None):
        cond = self.conditioner(x, context).view(x.shape[0], -1, self.d).permute(0, 2, 1).contiguous()
        trans = self.linear_net.forward(cond.view(x.shape[0] * self.d, -1)).view(x.shape[0], -1, 2)
        mu, sigma = trans[:, :, 0], torch.exp(trans[:, :, 1])
        sigma.clamp_(-5., 2.)
        mu.clamp_(-5., 5.)
        z = x * sigma + mu
        return z

    def compute_log_jac(self, x, context=None):
        cond = self.conditioner(x, context).view(x.shape[0], -1, self.d).permute(0, 2, 1).contiguous()
        trans = self.linear_net.forward(cond.view(x.shape[0] * self.d, -1)).view(x.shape[0], -1, 2)
        mu, sigma = trans[:, :, 0], trans[:, :, 1]
        sigma.clamp_(-5., 2.)
        mu.clamp_(-5., 5.)
        return sigma

    def compute_log_jac_bis(self, x, context=None):
        cond = self.conditioner(x, context).view(x.shape[0], -1, self.d).permute(0, 2, 1).contiguous()
        trans = self.linear_net.forward(cond.view(x.shape[0] * self.d, -1)).view(x.shape[0], -1, 2)
        mu, sigma = trans[:, :, 0], trans[:, :, 1]
        sigma.clamp_(-5., 2.)
        mu.clamp_(-5., 5.)
        z = x * torch.exp(sigma) + mu
        return z, sigma

    def compute_ll(self, x, context=None):
        cond = self.conditioner(x, context).view(x.shape[0], -1, self.d).permute(0, 2, 1).contiguous()
        trans = self.linear_net.forward(cond.view(x.shape[0] * self.d, -1)).view(x.shape[0], -1, 2)
        mu, sigma = trans[:, :, 0], trans[:, :, 1]
        sigma.clamp_(-5., 2.)
        mu.clamp_(-5., 5.)
        z = x * torch.exp(sigma) + mu
        log_prob_gauss = -.5 * (torch.log(self.pi * 2) + z ** 2).sum(1)
        ll = log_prob_gauss + sigma.sum(1)

        return ll, z

    def compute_ll_bis(self, x, context=None):
        cond = self.conditioner(x, context).view(x.shape[0], -1, self.d).permute(0, 2, 1).contiguous()
        trans = self.linear_net.forward(cond.view(x.shape[0] * self.d, -1)).view(x.shape[0], -1, 2)
        mu, sigma = trans[:, :, 0], trans[:, :, 1]
        sigma.clamp_(-5., 2.)
        mu.clamp_(-5., 5.)
        z = x * torch.exp(sigma) + mu
        log_prob_gauss = -.5 * (torch.log(self.pi * 2) + z ** 2).sum(1)
        ll = log_prob_gauss + sigma.sum(1)

        return ll, z

    def compute_bpp(self, x, alpha=1e-6, context=None):
        d = x.shape[1]
        ll, z = self.computeLL(x, context=context)
        bpp = -ll/(d*np.log(2)) - np.log2(1 - 2*alpha) + 8 \
              + 1/d * (torch.log2(torch.sigmoid(x)) + torch.log2(1 - torch.sigmoid(x))).sum(1)
        z.clamp_(-10., 10.)
        return bpp, ll, z

    # Kind of dichotomy with a factor 100.
    def invert(self, z, iter=10, context=None):
        return None


class CubicNormalizer(nn.Module):
    def __init__(self, conditioner, net, input_size, device="cpu"):
        super(LinearNormalizer, self).__init__()
        self.conditioner = conditioner.to(device)
        self.device = device
        self.linear_net = net
        self.input_size = input_size
        self.d = input_size
        self.pi = torch.tensor(math.pi).to(self.device)

    def forward(self, x, context=None):
        cond = self.conditioner(x, context).view(x.shape[0], -1, self.d).permute(0, 2, 1).contiguous()
        trans = self.linear_net.forward(cond.view(x.shape[0] * self.d, -1)).view(x.shape[0], -1, 3)
        mu, sigma, skew = trans[:, :, 0], torch.exp(trans[:, :, 1]), torch.exp(trans[:, :, 2])
        #sigma.clamp_(-3., 3.)
        z = x * sigma + mu + x**3 * skew
        #z.clamp_(-10., 10.)
        return z

    def compute_log_jac(self, x, context=None):
        cond = self.conditioner(x, context).view(x.shape[0], -1, self.d).permute(0, 2, 1).contiguous()
        trans = self.linear_net.forward(cond.view(x.shape[0] * self.d, -1)).view(x.shape[0], -1, 3)
        mu, sigma, skew = trans[:, :, 0], torch.exp(trans[:, :, 1]), torch.exp(trans[:, :, 2])
        #sigma.clamp_(-3., 3.)
        return sigma + skew * x**2

    def compute_log_jac_bis(self, x, context=None):
        cond = self.conditioner(x, context).view(x.shape[0], -1, self.d).permute(0, 2, 1).contiguous()
        trans = self.linear_net.forward(cond.view(x.shape[0] * self.d, -1)).view(x.shape[0], -1, 3)
        mu, sigma, skew = trans[:, :, 0], torch.exp(trans[:, :, 1]), torch.exp(trans[:, :, 2])
        #sigma.clamp_(-3., 3.)
        z = x * sigma + mu + x**3 * skew
        #z.clamp_(-10., 10.)
        return z, sigma + skew * x**2

    def compute_ll(self, x, context=None):
        cond = self.conditioner(x, context).view(x.shape[0], -1, self.d).permute(0, 2, 1).contiguous()
        trans = self.linear_net.forward(cond.view(x.shape[0] * self.d, -1)).view(x.shape[0], -1, 3)
        mu, sigma, skew = trans[:, :, 0], torch.exp(trans[:, :, 1]), torch.exp(trans[:, :, 2])
        #sigma.clamp_(-3., 3.)
        z = x * sigma + mu + x**3 * skew

        #z.clamp_(-10., 10.)
        log_prob_gauss = -.5 * (torch.log(self.pi * 2) + z ** 2).sum(1)
        ll = log_prob_gauss + sigma.sum(1) + (skew * x**2).sum(1)

        return ll, z

    def compute_ll_bis(self, x, context=None):
        cond = self.conditioner(x, context).view(x.shape[0], -1, self.d).permute(0, 2, 1).contiguous()
        trans = self.linear_net.forward(cond.view(x.shape[0] * self.d, -1)).view(x.shape[0], -1, 3)
        mu, sigma, skew = trans[:, :, 0], torch.exp(trans[:, :, 1]), torch.exp(trans[:, :, 2])
        #sigma.clamp_(-3., 3.)
        z = x * sigma + mu + x**3 * skew

        #z.clamp_(-10., 10.)
        log_prob_gauss = -.5 * (torch.log(self.pi * 2) + z ** 2).sum(1)
        ll = log_prob_gauss + sigma.sum(1) + (skew * x**2).sum(1)

        return ll, z

    def compute_bpp(self, x, alpha=1e-6, context=None):
        d = x.shape[1]
        ll, z = self.computeLL(x, context=context)
        bpp = -ll/(d*np.log(2)) - np.log2(1 - 2*alpha) + 8 \
              + 1/d * (torch.log2(torch.sigmoid(x)) + torch.log2(1 - torch.sigmoid(x))).sum(1)
        z.clamp_(-10., 10.)
        return bpp, ll, z

    # Kind of dichotomy with a factor 100.
    def invert(self, z, iter=10, context=None):
        return None
