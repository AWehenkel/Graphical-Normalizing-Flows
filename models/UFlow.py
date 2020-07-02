import torch
import torch.nn as nn
from .Conditionners import Conditioner, DAGConditioner
from .Normalizers import Normalizer
from .NormalizingFlow import NormalizingFlow, FCNormalizingFlow


class UFlow(FCNormalizingFlow):
    def __init__(self, enc_steps, dec_steps, z_log_density, dropping_factors):
        super(UFlow, self).__init__(enc_steps + dec_steps, z_log_density)
        self.enc_dropping_factors = dropping_factors
        self.dec_gathering_factors = dropping_factors[::-1]
        self.enc_steps = enc_steps
        self.dec_steps = dec_steps

    def forward(self, x, context=None):
        b_size = x.shape[0]
        jac_tot = 0.
        z_all = []
        for step, drop_factors in zip(self.enc_steps, self.enc_dropping_factors):
            z, jac = step(x.contiguous().view(b_size, -1), context)
            d_c, d_h, d_w = drop_factors
            C, H, W = step.img_sizes
            c, h, w = int(C/d_c), int(H/d_h), int(W/d_w)
            z_reshaped = z.view(-1, C, H, W).unfold(1, d_c, d_c).unfold(2, d_h, d_h) \
                    .unfold(3, d_w, d_w).contiguous().view(b_size, c, h, w, -1)
            z_all += [z_reshaped[:, :, :, :, 1:]]
            x = z.view(-1, C, H, W).unfold(1, d_c, d_c).unfold(2, d_h, d_h) \
                    .unfold(3, d_w, d_w).contiguous().view(b_size, c, h, w, -1)[:, :, :, :, 0]

            jac_tot += jac
        for step, gath_factors, z in zip(self.dec_steps[1:], self.dec_gathering_factors[1:], z_all[::-1][1:]):
            d_c, d_h, d_w = gath_factors
            C, H, W = step.img_sizes
            c, h, w = int(C / d_c), int(H / d_h), int(W / d_w)
            z = torch.cat((x.unsqueeze(4), z), 4)
            z = z.view(b_size, c, h, w, d_c, d_h, d_w)
            z = z.permute(0, 1, 2, 3, 6, 4, 5).contiguous().view(b_size, c, h, W, d_c, d_h)
            z = z.permute(0, 1, 2, 5, 3, 4).contiguous().view(b_size, c, H, W, d_c)
            x = z.permute(0, 1, 4, 2, 3).contiguous().view(b_size, C, H, W)
            x, jac = step(x.contiguous().view(b_size, -1), context)
            x = x.contiguous().view(b_size, C, H, W)
            jac_tot += jac
        z = x.view(b_size, -1)
        return z, jac_tot

    def invert(self, z, context=None):
        b_size = z.shape[0]
        z_all = []
        for step, drop_factors in zip(self.dec_steps[::-1][1:], self.enc_dropping_factors[:-1]):
            x = step.invert(z.contiguous().view(b_size, -1), context)
            d_c, d_h, d_w = drop_factors
            C, H, W = step.img_sizes
            c, h, w = int(C / d_c), int(H / d_h), int(W / d_w)
            z_reshaped = z.view(-1, C, H, W).unfold(1, d_c, d_c).unfold(2, d_h, d_h) \
                .unfold(3, d_w, d_w).contiguous().view(b_size, c, h, w, -1)
            z_all += [z_reshaped[:, :, :, :, 1:]]
            x = z.view(-1, C, H, W).unfold(1, d_c, d_c).unfold(2, d_h, d_h) \
                    .unfold(3, d_w, d_w).contiguous().view(b_size, c, h, w, -1)[:, :, :, :, 0]
            z = x

        x = self.enc_steps[-1].invert(z, context)
        for step, gath_factors, z in zip(self.enc_steps[::-1][1:], self.dec_gathering_factors[1:], z_all[::-1]):
            d_c, d_h, d_w = gath_factors
            C, H, W = step.img_sizes
            c, h, w = int(C / d_c), int(H / d_h), int(W / d_w)
            z = torch.cat((x.unsqueeze(4), z), 4)
            z = z.view(b_size, c, h, w, d_c, d_h, d_w)
            z = z.permute(0, 1, 2, 3, 6, 4, 5).contiguous().view(b_size, c, h, W, d_c, d_h)
            z = z.permute(0, 1, 2, 5, 3, 4).contiguous().view(b_size, c, H, W, d_c)
            x = z.permute(0, 1, 4, 2, 3).contiguous().view(b_size, C, H, W)
            x = step.invert(x.contiguous().view(b_size, -1), context).contiguous().view(b_size, C, H, W)
        return x.view(b_size, -1)

