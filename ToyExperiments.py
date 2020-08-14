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
              nb_flow=1, cond_type = "DAG", emb_net = [150, 150, 150, 30]):
    logger = utils.get_logger(logpath=os.path.join(folder, toy, 'logs'), filepath=os.path.abspath(__file__))

    logger.info("Creating model...")

    device = "cpu" if not(torch.cuda.is_available()) else "cuda:0"

    nb_samp = 100
    batch_size = 100

    x = torch.tensor(toy_data.inf_train_gen(toy, batch_size=10000)).to(device)
    x_mu = x.mean(0)
    x_std = x.std(0)

    x_test = torch.tensor(toy_data.inf_train_gen(toy, batch_size=1000)).to(device)
    x_test = (x_test - x_mu.unsqueeze(0).expand(x_test.shape[0], -1)) / x_std.unsqueeze(0).expand(x_test.shape[0], -1)

    dim = x.shape[1]
    print(x_mu.shape)

    norm_type = "Monotonic"
    save_name = norm_type + str(emb_net) + str(nb_flow)
    solver = "CCParallel"
    int_net = [100, 100, 100, 100]

    conditioner_type = cond_types[cond_type]
    print(dim)
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

    model = buildFCNormalizingFlow(nb_flow, conditioner_type, conditioner_args, normalizer_type, normalizer_args).to(device)

    opt = torch.optim.Adam(model.parameters(), 1e-3, weight_decay=1e-5)

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
            cur_x = (cur_x - x_mu.unsqueeze(0).expand(batch_size, -1))/x_std.unsqueeze(0).expand(batch_size, -1)
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



        if epoch % 1000 == 0:
                with torch.no_grad():
                    plt.matshow(model.getConditioners()[0].A.detach().cpu().numpy())
                    plt.colorbar()
                    plt.savefig("%s%s/flow_%s_%d.pdf" % (folder, toy, save_name, epoch))
                    torch.save(model.state_dict(), folder + toy + '/' + save_name + 'model.pt')
                    torch.save(opt.state_dict(), folder + toy + '/'+ save_name + 'ADAM.pt')

toy = "8gaussians"

import argparse
datasets = ["2igaussians", "2gaussians", "8gaussians", "swissroll", "moons", "pinwheel", "cos", "2spirals", "checkerboard", "line", "line-noisy",
            "circles", "joint_gaussian", "2spirals-8gaussians", "4-2spirals-8gaussians", "8-2spirals-8gaussians",
            "8-MIX", "7-MIX", "4gaussians", "woodStructural"]

parser = argparse.ArgumentParser(description='')
parser.add_argument("-dataset", default=None, choices=datasets, help="Which toy problem ?")
parser.add_argument("-load", default=False, action="store_true", help="Load a model ?")
parser.add_argument("-folder", default="", help="Folder")
parser.add_argument("-nb_steps_dual", default=50, type=int, help="number of step between updating Acyclicity constraint and sparsity constraint")
parser.add_argument("-l1", default=.0, type=float, help="Maximum weight for l1 regularization")
parser.add_argument("-nb_epoch", default=20000, type=int, help="Number of epochs")

args = parser.parse_args()


if args.dataset is None:
    toys = datasets
else:
    toys = [args.dataset]

for toy in toys:
    if not(os.path.isdir(args.folder + toy)):
        os.makedirs(args.folder + toy)
    train_toy(toy, load=args.load, folder=args.folder, nb_step_dual=args.nb_steps_dual, l1=args.l1,
              nb_epoch=args.nb_epoch)
