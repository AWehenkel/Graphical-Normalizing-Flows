import torch
import torch.nn as nn
from .Normalizers import *
from .Conditionners import *
from .NormalizingFlow import NormalizingFlowStep, FCNormalizingFlow, CNNormalizingFlow
from math import pi
from .MLP import MNISTCNN, CIFAR10CNN


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


def buildMNISTNormalizingFlow(nb_inner_steps, normalizer_type, normalizer_args, l1=0.):
    if len(nb_inner_steps) == 3:
        img_sizes = [[1, 28, 28], [1, 14, 14], [1, 7, 7]]
        dropping_factors = [[1, 2, 2], [1, 2, 2]]
        fc_l = [[2304, 128], [400, 64], [16, 16]]

        outter_steps = []
        for i, fc in zip(range(len(fc_l)), fc_l):
            in_size = img_sizes[i][0] * img_sizes[i][1] * img_sizes[i][2]
            inner_steps = []
            for step in range(nb_inner_steps[i]):
                emb_s = 2 if normalizer_type is AffineNormalizer else 30
                hidden = MNISTCNN(fc_l=fc, size_img=img_sizes[i], out_d=emb_s)
                cond = DAGConditioner(in_size, hidden, emb_s, l1=l1)
                norm = normalizer_type(**normalizer_args)
                flow_step = NormalizingFlowStep(cond, norm)
                inner_steps.append(flow_step)
            flow = FCNormalizingFlow(inner_steps, None)
            flow.img_sizes = img_sizes[i]
            outter_steps.append(flow)

        return CNNormalizingFlow(outter_steps, NormalLogDensity(), dropping_factors)
    elif len(nb_inner_steps) == 1:
        inner_steps = []
        for step in range(nb_inner_steps[0]):
            emb_s = 2 if normalizer_type is AffineNormalizer else 30
            hidden = MNISTCNN(fc_l=[2304, 128], size_img=[1, 28, 28], out_d=emb_s)
            cond = DAGConditioner(1*28*28, hidden, emb_s, l1=l1)
            norm = normalizer_type(**normalizer_args)
            flow_step = NormalizingFlowStep(cond, norm)
            inner_steps.append(flow_step)
        flow = FCNormalizingFlow(inner_steps, NormalLogDensity())
        return flow
    else:
        return None


def buildCIFAR10NormalizingFlow(nb_inner_steps, normalizer_type, normalizer_args, l1=0.):
    if len(nb_inner_steps) == 4:
        img_sizes = [[3, 32, 32], [1, 32, 32], [1, 16, 16], [1, 8, 8]]
        dropping_factors = [[3, 1, 1], [1, 2, 2], [1, 2, 2]]
        fc_l = [[400, 128, 84], [576, 128, 32], [64, 32, 32], [16, 32, 32]]
        k_sizes = [5, 3, 3, 2]

        outter_steps = []
        for i, fc in zip(range(len(fc_l)), fc_l):
            in_size = img_sizes[i][0] * img_sizes[i][1] * img_sizes[i][2]
            inner_steps = []
            for step in range(nb_inner_steps[i]):
                emb_s = 2 if normalizer_type is AffineNormalizer else 30
                hidden = CIFAR10CNN(out_d=emb_s, fc_l=fc, size_img=img_sizes[i], k_size=k_sizes[i])
                cond = DAGConditioner(in_size, hidden, emb_s, l1=l1)
                norm = normalizer_type(**normalizer_args)
                flow_step = NormalizingFlowStep(cond, norm)
                inner_steps.append(flow_step)
            flow = FCNormalizingFlow(inner_steps, None)
            flow.img_sizes = img_sizes[i]
            outter_steps.append(flow)

        return CNNormalizingFlow(outter_steps, NormalLogDensity(), dropping_factors)
    elif len(nb_inner_steps) == 1:
        inner_steps = []
        for step in range(nb_inner_steps[0]):
            emb_s = 2 if normalizer_type is AffineNormalizer else 30
            hidden = CIFAR10CNN(fc_l=[400, 128, 84], size_img=[3, 32, 32], out_d=emb_s, k_size=3)
            cond = DAGConditioner(3*32*32, hidden, emb_s, l1=l1)
            norm = normalizer_type(**normalizer_args)
            flow_step = NormalizingFlowStep(cond, norm)
            inner_steps.append(flow_step)
        flow = FCNormalizingFlow(inner_steps, NormalLogDensity())
        return flow
    else:
        return None
