import torch
import torch.nn as nn
from .Conditionners import Conditioner, DAGConditioner
from .Normalizers import Normalizer


class NormalizingFlowStep(nn.Module):
    def __init__(self, conditioner: Conditioner, normalizer: Normalizer):
        super(NormalizingFlowStep, self).__init__()
        self.conditioner = conditioner
        self.normalizer = normalizer

    def forward(self, x, context=None):
        h = self.conditioner(x, context)
        z, jac = self.normalizer(x, h, context)
        return z, jac


class FCNormalizingFlow(nn.Module):
    def __init__(self, steps, z_log_density):
        super(FCNormalizingFlow, self).__init__()
        self.steps = nn.ModuleList()
        self.z_log_density = z_log_density
        for step in steps:
            self.steps.append(step)

    def forward(self, x):
        jac_tot = 0.
        inv_idx = torch.arange(x.shape[1] - 1, -1, -1).long()
        for step in self.steps:
            z, jac = step(x)
            x = z[:, inv_idx]
            jac_tot += torch.log(jac).sum(1)

        log_p_x = self.z_log_density(x) + jac_tot
        return z, log_p_x

    def loss(self, z, log_p_x):
        loss = -log_p_x.mean()
        for step in self.steps:
            if type(step.conditioner) is DAGConditioner:
                loss += step.conditioner.loss()
        return loss

    def DAGness(self):
        dagness = []
        for step in self.steps:
            if type(step.conditioner) is DAGConditioner:
                dagness.append(step.conditioner.get_power_trace())
            else:
                dagness.append(0)
        return dagness

    def step(self, epoch_number, loss_avg):
        for step in self.steps:
            if type(step.conditioner) is DAGConditioner:
                step.conditioner.step(epoch_number, loss_avg)


class CNNormalizingFlow(nn.Module):
    def __init__(self, steps, z_log_density, dropping_factors):
        super(CNNormalizingFlow, self).__init__()
        self.steps = nn.ModuleList()
        self.z_log_density = z_log_density
        self.dropping_factors = dropping_factors
        for step in steps:
            self.steps.append(step)

    def forward(self, x):
        b_size = x.shape[0]
        jac_tot = 0.
        z_all = []
        for step, drop_factors in zip(self.steps, self.dropping_factors):
            z, jac = step(x)
            d_c, d_h, d_w = drop_factors
            C, H, W = step.img_sizes
            c, h, w = C/d_c, H/d_h, W/d_w
            z_reshaped = z.view(-1, C, H, W).unfold(1, d_c, d_c).unfold(2, d_h, d_h) \
                    .unfold(3, d_w, d_w).contiguous().view(b_size, c, h, w, -1)
            z_all += [z_reshaped[:, :, :, 1:].contiguous().view(b_size, -1)]
            x = z.view(-1, C, H, W).unfold(1, d_c, d_c).unfold(2, d_h, d_h) \
                    .unfold(3, d_w, d_w).contiguous().view(b_size, c, h, w, -1)[:, :, :, :, 0] \
                .contiguous().view(b_size, -1)
            jac_tot += jac.sum(1)
        z = torch.cat(z_all, 1)
        log_p_x = self.z_log_density(x) + jac_tot
        return z, log_p_x

    def loss(self, z, log_p_x):
        loss = -log_p_x.mean()
        for step in self.steps:
            if type(step.conditioner) is DAGConditioner:
                loss += step.conditioner.loss()
        return loss

    def DAGness(self):
        dagness = []
        for step in self.steps:
            if type(step.conditioner) is DAGConditioner:
                dagness.append(step.conditioner.get_power_trace())
            else:
                dagness.append(0)
        return dagness

    def step(self, epoch_number, loss_avg):
        for step in self.steps:
            if type(step.conditioner) is DAGConditioner:
                step.conditioner.step(epoch_number, loss_avg)

