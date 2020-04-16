import torch
from Normalizer import Normalizer

class LinearNormalizer(Normalizer):
    def __init__(self):
        super(LinearNormalizer, self).__init__()

    def forward(self, x, h, context=None):
        mu, sigma = h[:, :, 0], torch.exp(h[:, :, 1])
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