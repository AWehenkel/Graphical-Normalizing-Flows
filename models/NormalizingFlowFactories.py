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
    """
    Function that returns a normalizing flow used in UCI experiments

    @param nb_steps: number of normalizing flow steps

    """
    flow_steps = []
    for step in range(nb_steps):
        conditioner = conditioner_type(**conditioner_args)
        normalizer = normalizer_type(**normalizer_args)
        flow_step = NormalizingFlowStep(conditioner, normalizer)
        flow_steps.append(flow_step)
    return FCNormalizingFlow(flow_steps, NormalLogDensity())


def MNIST_A_prior(in_size, kernel):
    A = torch.zeros(in_size**2, in_size**2)
    row_pix = torch.arange(in_size).view(1, -1).expand(in_size, -1).contiguous().view(-1, 1)
    col_pix = torch.arange(in_size).view(-1, 1).expand(-1, in_size).contiguous().view(-1, 1)

    for i in range(-kernel, kernel + 1):
        for j in range(-kernel, kernel + 1):
            mask = ((col_pix + i) < in_size) * ((col_pix + i) >= 0) * ((row_pix + j) < in_size) * ((row_pix + j) >= 0)
            idx = ((row_pix * in_size + col_pix) * in_size**2 + col_pix + i + in_size * (row_pix + j)) * mask
            A.view(-1)[idx] = 1.
    A.view(-1)[torch.arange(0, in_size**4, in_size**2+1)] = 0
    return A


def buildMNISTNormalizingFlow(nb_inner_steps, normalizer_type, normalizer_args, l1=0., nb_epoch_update=10,
                              hot_encoding=False, prior_kernel=None):
    if len(nb_inner_steps) == 3:
        img_sizes = [[1, 28, 28], [1, 14, 14], [1, 7, 7]]
        dropping_factors = [[1, 2, 2], [1, 2, 2], [1, 1, 1]]
        fc_l = [[2304, 128], [400, 64], [16, 16]]

        outter_steps = []
        for i, fc in zip(range(len(fc_l)), fc_l):
            in_size = img_sizes[i][0] * img_sizes[i][1] * img_sizes[i][2]
            inner_steps = []
            for step in range(nb_inner_steps[i]):
                emb_s = 2 if normalizer_type is AffineNormalizer else 30

                hidden = MNISTCNN(fc_l=fc, size_img=img_sizes[i], out_d=emb_s)
                A_prior = MNIST_A_prior(img_sizes[i][1], prior_kernel) if prior_kernel is not None else None
                cond = DAGConditioner(in_size, hidden, emb_s, l1=l1, nb_epoch_update=nb_epoch_update,
                                      hot_encoding=hot_encoding, A_prior=A_prior)
                if normalizer_type is MonotonicNormalizer:
                    emb_s = 30 + in_size if hot_encoding else 30
                    norm = normalizer_type(**normalizer_args, cond_size=emb_s)
                else:
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
            A_prior = MNIST_A_prior(28, prior_kernel) if prior_kernel is not None else None
            cond = DAGConditioner(1*28*28, hidden, emb_s, l1=l1, nb_epoch_update=nb_epoch_update,
                                  hot_encoding=hot_encoding, A_prior=A_prior)
            if normalizer_type is MonotonicNormalizer:
                emb_s = 30 + 28*28 if hot_encoding else 30
                norm = normalizer_type(**normalizer_args, cond_size=emb_s)
            else:
                norm = normalizer_type(**normalizer_args)
            flow_step = NormalizingFlowStep(cond, norm)
            inner_steps.append(flow_step)
        flow = FCNormalizingFlow(inner_steps, NormalLogDensity())
        return flow
    else:
        return None


def buildCIFAR10NormalizingFlow(nb_inner_steps, normalizer_type, normalizer_args, l1=0., nb_epoch_update=5):
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
                cond = DAGConditioner(in_size, hidden, emb_s, l1=l1, nb_epoch_update=nb_epoch_update)
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
            hidden = CIFAR10CNN(fc_l=[400, 128, 84], size_img=[3, 32, 32], out_d=emb_s, k_size=5)
            cond = DAGConditioner(3*32*32, hidden, emb_s, l1=l1, nb_epoch_update=nb_epoch_update)
            norm = normalizer_type(**normalizer_args)
            flow_step = NormalizingFlowStep(cond, norm)
            inner_steps.append(flow_step)
        flow = FCNormalizingFlow(inner_steps, NormalLogDensity())
        return flow
    else:
        return None
