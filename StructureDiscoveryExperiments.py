import lib.toy_data as toy_data
from models import *
import torch
from timeit import default_timer as timer
import lib.utils as utils
import os
import matplotlib.pyplot as plt
import networkx as nx
import math
import seaborn as sns
import UCIdatasets
sns.set()
flatui = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]
sns.palplot(sns.color_palette(flatui))

cond_types = {"DAG": DAGConditioner, "Coupling": CouplingConditioner, "Autoregressive": AutoregressiveConditioner}
norm_types = {"Affine": AffineNormalizer, "Monotonic": MonotonicNormalizer}


def batch_iter(X, batch_size, shuffle=False):
    """
    X: feature tensor (shape: num_instances x num_features)
    """
    if shuffle:
        idxs = torch.randperm(X.shape[0])
    else:
        idxs = torch.arange(X.shape[0])
    if X.is_cuda:
        idxs = idxs.cuda()
    for batch_idxs in idxs.split(batch_size):
        yield X[batch_idxs]

def getDataset(ds_name="8-MIX", device="cpu"):
    if ds_name in ["8-MIX", "7-MIX", "woodStructural"]:
        x_train = torch.tensor(toy_data.inf_train_gen(ds_name, batch_size=20000)).to(device)
        x_mu = x_train.mean(0)
        x_std = x_train.std(0)
        x_train = (x_train - x_mu.unsqueeze(0).expand(x_train.shape[0], -1)) / x_std.unsqueeze(0).expand(x_train.shape[0],
                                                                                                      -1)

        x_test = torch.tensor(toy_data.inf_train_gen(ds_name, batch_size=5000)).to(device)
        x_test = (x_test - x_mu.unsqueeze(0).expand(x_test.shape[0], -1)) / x_std.unsqueeze(0).expand(x_test.shape[0],
                                                                                                      -1)

        x_valid = torch.tensor(toy_data.inf_train_gen(ds_name, batch_size=5000)).to(device)
        x_valid = (x_valid - x_mu.unsqueeze(0).expand(x_test.shape[0], -1)) / x_std.unsqueeze(0).expand(x_valid.shape[0],
                                                                                                      -1)

        ground_truth_A = toy_data.getA(ds_name)
    elif ds_name == "proteins":
        data = UCIdatasets.PROTEINS()
        x_train = torch.from_numpy(data.trn.x).to(device)
        x_test = torch.from_numpy(data.val.x).to(device)
        x_valid = torch.from_numpy(data.tst.x).to(device)
        ground_truth_A = UCIdatasets.proteins.get_adj_matrix()
    else:
        return None

    return (x_train, x_test, x_valid), ground_truth_A








def train_toy(toy, load=True, nb_step_dual=300, nb_steps=15, folder="", l1=1., nb_epoch=20000, pre_heating_epochs=10,
              nb_flow=1, cond_type = "DAG", emb_net = [150, 150, 150, 30]):
    logger = utils.get_logger(logpath=os.path.join(folder, toy, 'logs'), filepath=os.path.abspath(__file__))

    logger.info("Creating model...")

    device = "cpu" if not(torch.cuda.is_available()) else "cuda:0"

    batch_size = 100

    (x_train, x_test, x_valid), ground_truth_A = getDataset(toy, device)
    dim = x_train.shape[1]

    norm_type = "Affine"
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
        for j, cur_x in enumerate(batch_iter(x_train, shuffle=True, batch_size=batch_size)):
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
        loss_tot /= j
        model.step(epoch, loss_tot)

        end = timer()
        loss_tot_valid = 0
        with torch.no_grad():
            for j, cur_x in enumerate(batch_iter(x_valid, shuffle=True, batch_size=batch_size)):
                z, jac = model(cur_x)
                loss = model.loss(z, jac)
                loss_tot_valid += loss.detach()
        loss_tot_valid /= j
        dagness = max(model.DAGness())
        logger.info("epoch: {:d} - Train loss: {:4f} - Valid loss: {:4f} - <<DAGness>>: {:4f} - Elapsed time per epoch {:4f} (seconds)".
                    format(epoch, loss_tot.item(), loss_tot_valid.item(), dagness, end-start))



        if epoch % 50 == 0:
                with torch.no_grad():
                    plt.subplot(1, 2, 1)
                    plt.matshow(model.getConditioners()[0].A.detach().cpu().numpy())
                    plt.colorbar()
                    plt.subplot(1, 2, 2)
                    plt.matshow(ground_truth_A.numpy())
                    plt.savefig("%s%s/flow_%s_%d.pdf" % (folder, toy, save_name, epoch))
                    torch.save(model.state_dict(), folder + toy + '/' + save_name + 'model.pt')
                    torch.save(opt.state_dict(), folder + toy + '/'+ save_name + 'ADAM.pt')

toy = "8gaussians"

import argparse
datasets = ["8-MIX", "7-MIX", "woodStructural", "proteins"]

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
