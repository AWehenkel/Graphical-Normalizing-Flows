import lib.toy_data as toy_data
from models import *
import torch
from timeit import default_timer as timer
import lib.utils as utils
import os
import lib.visualize_flow as vf
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import math
import matplotlib
import seaborn as sns
sns.set()
from matplotlib import gridspec
flatui = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]
sns.palplot(sns.color_palette(flatui))

cond_types = {"DAG": DAGConditioner, "Coupling": CouplingConditioner, "Autoregressive": AutoregressiveConditioner}
norm_types = {"Affine": AffineNormalizer, "Monotonic": MonotonicNormalizer}

def train_toy(toy, load=True, nb_step_dual=300, nb_steps=15, folder="", l1=1., nb_epoch=20000, pre_heating_epochs=10,
              nb_flow=3, cond_type = "Coupling", emb_net = [150, 150, 150]):
    logger = utils.get_logger(logpath=os.path.join(folder, toy, 'logs'), filepath=os.path.abspath(__file__))

    logger.info("Creating model...")

    device = "cpu" if not(torch.cuda.is_available()) else "cuda:0"

    nb_samp = 100
    batch_size = 100

    x_test = torch.tensor(toy_data.inf_train_gen(toy, batch_size=1000)).to(device)
    x = torch.tensor(toy_data.inf_train_gen(toy, batch_size=1000)).to(device)

    dim = x.shape[1]

    norm_type = "Affine"
    save_name = norm_type + str(emb_net) + str(nb_flow)
    solver = "CCParallel"
    int_net = [150, 150, 150]

    conditioner_type = cond_types[cond_type]
    conditioner_args = {"in_size": dim, "hidden": emb_net[:-1], "out_size": emb_net[-1]}
    if conditioner_type is DAGConditioner:
        conditioner_args['l1'] = l1
        conditioner_args['gumble_T'] = .5
        conditioner_args['nb_epoch_update'] = nb_step_dual
        conditioner_args["hot_encoding"] = True
    normalizer_type = norm_types[norm_type]
    if normalizer_type is MonotonicNormalizer:
        normalizer_args = {"integrand_net": int_net, "cond_size": emb_net[-1], "nb_steps": nb_steps,
                           "solver": solver}
    else:
        normalizer_args = {}

    model = buildFCNormalizingFlow(nb_flow, conditioner_type, conditioner_args, normalizer_type, normalizer_args)

    opt = torch.optim.Adam(model.parameters(), 1e-4, weight_decay=1e-5)

    if load:
        logger.info("Loading model...")
        model.load_state_dict(torch.load(folder + toy + '/' + save_name + 'model.pt'))
        model.train()
        opt.load_state_dict(torch.load(folder + toy + '/' + save_name + 'ADAM.pt'))
        logger.info("Model loaded.")

    if True:
        for step in model.steps:
            step.conditioner.stoch_gate = True
            step.conditioner.noise_gate = False
            step.conditioner.gumble_T = .5
    torch.autograd.set_detect_anomaly(True)
    for epoch in range(nb_epoch):
        loss_tot = 0
        start = timer()
        for j in range(0, nb_samp, batch_size):
            cur_x = torch.tensor(toy_data.inf_train_gen(toy, batch_size=batch_size)).to(device)
            z, jac = model(cur_x)
            loss = model.loss(z, jac)
            loss_tot += loss.detach()
            if math.isnan(loss.item()):
                ll, z = model.compute_ll(cur_x)
                print(ll)
                print(z)
                print(ll.max(), z.max())
                exit()
            opt.zero_grad()
            loss.backward(retain_graph=True)
            opt.step()
        model.step(epoch, loss_tot)


        end = timer()
        z, jac = model(x_test)
        ll = (model.z_log_density(z) + jac)
        ll_test = -ll.mean()
        dagness = max(model.DAGness())
        logger.info("epoch: {:d} - Train loss: {:4f} - Test loss: {:4f} - <<DAGness>>: {:4f} - Elapsed time per epoch {:4f} (seconds)".
                    format(epoch, loss_tot.item(), ll_test.item(), dagness, end-start))


        if epoch % 100 == 0 and False:
            with torch.no_grad():
                stoch_gate = model.getDag().stoch_gate
                noise_gate = model.getDag().noise_gate
                s_thresh = model.getDag().s_thresh
                model.getDag().stoch_gate = False
                model.getDag().noise_gate = False
                model.getDag().s_thresh = True
                for threshold in [.95, .1, .01, .0001, 1e-8]:
                    model.set_h_threshold(threshold)
                    # Valid loop
                    z, jac = model(x_test)
                    ll = (model.z_log_density(z) + jac)
                    ll_test = -ll.mean().item()
                    dagness = max(model.DAGness()).item()
                    logger.info("epoch: {:d} - Threshold: {:4f} - Valid log-likelihood: {:4f} - <<DAGness>>: {:4f}".
                                format(epoch, threshold, ll_test, dagness))
                model.getDag().stoch_gate = stoch_gate
                model.getDag().noise_gate = noise_gate
                model.getDag().s_thresh = s_thresh
                model.set_h_threshold(0.)


        if epoch % 500 == 0:
            font = {'family': 'normal',
                    'weight': 'normal',
                    'size': 25}

            matplotlib.rc('font', **font)
            if toy in ["2spirals-8gaussians", "4-2spirals-8gaussians", "8-2spirals-8gaussians", "2gaussians",
                       "4gaussians", "2igaussians", "8gaussians"] or True:
                def compute_ll(x):
                    z, jac = model(x)
                    ll = (model.z_log_density(z) + jac)
                    return ll, z
                with torch.no_grad():
                    npts = 100
                    plt.figure(figsize=(12, 12))
                    gs = gridspec.GridSpec(2, 2, width_ratios=[3, 1], height_ratios=[3, 1])
                    ax = plt.subplot(gs[0])
                    qz_1, qz_2 = vf.plt_flow(compute_ll, ax, npts=npts, device=device)
                    plt.subplot(gs[1])
                    plt.plot(qz_1, np.linspace(-4, 4, npts))
                    plt.ylabel('$x_2$', fontsize=25, rotation=-90, labelpad=20)

                    plt.xticks([])
                    plt.subplot(gs[2])
                    plt.plot(np.linspace(-4, 4, npts), qz_2)
                    plt.xlabel('$x_1$', fontsize=25)
                    plt.yticks([])
                    plt.savefig("%s%s/flow_%s_%d.pdf" % (folder, toy, save_name, epoch))
                    torch.save(model.state_dict(), folder + toy + '/' + save_name + 'model.pt')
                    torch.save(opt.state_dict(), folder + toy + '/'+ save_name + 'ADAM.pt')

toy = "8gaussians"

import argparse
datasets = ["2igaussians", "2gaussians", "8gaussians", "swissroll", "moons", "pinwheel", "cos", "2spirals", "checkerboard", "line", "line-noisy",
            "circles", "joint_gaussian", "2spirals-8gaussians", "4-2spirals-8gaussians", "8-2spirals-8gaussians",
            "8-MIX", "7-MIX", "4gaussians"]

parser = argparse.ArgumentParser(description='')
parser.add_argument("-dataset", default=None, choices=datasets, help="Which toy problem ?")
parser.add_argument("-load", default=False, action="store_true", help="Load a model ?")
parser.add_argument("-folder", default="", help="Folder")
parser.add_argument("-nb_steps_dual", default=50, type=int, help="number of step between updating Acyclicity constraint and sparsity constraint")
parser.add_argument("-l1", default=.0, type=float, help="Maximum weight for l1 regularization")
parser.add_argument("-nb_epoch", default=20000, type=int, help="Number of epochs")

args = parser.parse_args()

for d in ["pinwheel"]:
    for net in [[200, 200, 200, 200]]:
        for nb_flow in [5]:
            if not (os.path.isdir(args.folder + d)):
                os.makedirs(args.folder + d)
            train_toy(d, load=False, nb_epoch=50000, nb_flow=nb_flow, cond_type="Coupling", emb_net=net)

if args.dataset is None:
    toys = datasets
else:
    toys = [args.dataset]

for toy in toys:
    if not(os.path.isdir(args.folder + toy)):
        os.makedirs(args.folder + toy)
    train_toy(toy, load=args.load, folder=args.folder, nb_step_dual=args.nb_steps_dual, l1=args.l1,
              nb_epoch=args.nb_epoch)
