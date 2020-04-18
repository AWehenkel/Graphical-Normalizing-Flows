import torch
import torch.nn as nn
from .Normalizers import *
from .Conditionners import *
from .NormalizingFlow import NormalizingFlowStep, FCNormalizingFlow
from math import pi


class NormalLogDensity(nn.Module):
    def __init__(self):
        super(NormalLogDensity, self).__init__()
        self.register_buffer("pi", torch.tensor(pi))

    def forward(self, z):
        return -.5 * (torch.log(self.pi * 2) + z ** 2).sum(1)


def buildFCNormalizingFlow(nb_steps, conditioner_type, conditioner_args, normalizer_type, normalizer_args):
    flow_steps = []
    for step in range(nb_steps):
        conditioner = conditioner_type(**conditioner_args)
        normalizer = normalizer_type(**normalizer_args)
        flow_step = NormalizingFlowStep(conditioner, normalizer)
        flow_steps.append(flow_step)
    return FCNormalizingFlow(flow_steps, NormalLogDensity())
