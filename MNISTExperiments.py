from models import DAGNF, MLP, MNISTCNN
import torch
from timeit import default_timer as timer
import lib.utils as utils
import os
import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
from torchvision import datasets, transforms
from lib.transform import AddUniformNoise, ToTensor, HorizontalFlip, Transpose, Resize
import numpy as np
import math
from UMNN import UMNNMAFFlow


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


def load_data(batch_size=100, cuda=-1):
    data = datasets.MNIST('./MNIST', train=True, download=True,
                          transform=transforms.Compose([
                              AddUniformNoise(),
                              ToTensor()
                          ]))

    train_data, valid_data = torch.utils.data.random_split(data, [50000, 10000])

    test_data = datasets.MNIST('./MNIST', train=False, download=True,
                               transform=transforms.Compose([
                                   AddUniformNoise(),
                                   ToTensor()
                               ]))
    kwargs = {'num_workers': 0, 'pin_memory': True} if cuda > -1 else {}

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, **kwargs)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True, **kwargs)
    return train_loader, valid_loader, test_loader


def train(load=True, nb_step_dual=100, nb_steps=20, path="", l1=.1, nb_epoch=10000,
          int_net=[200, 200, 200], emb_net=[200, 200, 200], b_size=100, umnn_maf=False, min_pre_heating_epochs=30,
          all_args=None, file_number=None, train=True, solver="CC", nb_flow=1, linear_net=False, gumble_T=1.,
          weight_decay=1e-5, learning_rate=1e-3, hutchinson=0):
    logger = utils.get_logger(logpath=os.path.join(path, 'logs'), filepath=os.path.abspath(__file__))
    logger.info(str(all_args))

    logger.info("Creating model...")

    device = "cpu" if not(torch.cuda.is_available()) else "cuda:0"

    if load:
        train = False
        file_number = "_" + file_number if file_number is not None else ""

    batch_size = b_size

    logger.info("Loading data...")
    train_loader, valid_loader, test_loader = load_data(batch_size)

    logger.info("Data loaded.")

    dim = 28**2

    emb_nets = []

    # Fixed for test
    nb_flow = 3
    img_sizes = [[28, 28], [14, 14], [7, 7]]
    dropping_factors = [[2, 2], [2, 2]]
    fc_l = [[2304, 128], [400, 64], [16, 16]]

    for i in range(nb_flow):
        if emb_net is not None:
            net = MNISTCNN(out_d=emb_net[-1], fc_l=fc_l[i], size_img=img_sizes[i]).to(device)
        else:
            net = None
        emb_nets.append(net)
    l1_weight = l1
    model = DAGNF(nb_flow=nb_flow, in_d=dim, hidden_integrand=int_net, emb_d=emb_nets[0].out_d, emb_nets=emb_nets,
                  device=device, l1_weight=l1, nb_steps=nb_steps, solver=solver, linear_normalizer=linear_net,
                  gumble_T=gumble_T, hutchinson=hutchinson, dropping_factors=dropping_factors, img_sizes=img_sizes)

    if min_pre_heating_epochs > 0:
        model.dag_const = 0.
    opt = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    # opt = torch.optim.RMSprop(model.parameters(), lr=1e-3)
    if not umnn_maf and train:
        for net in model.nets:
            net.getDag().stoch_gate = True
            net.getDag().noise_gate = False
    if load:
        logger.info("Loading model...")
        model.load_state_dict(torch.load(path + '/model%s.pt' % file_number, map_location={"cuda:0": device}))
        model.train()
        opt.load_state_dict(torch.load(path + '/ADAM%s.pt' % file_number, map_location={"cuda:0": device}))
        if not train and not umnn_maf:
            for net in model.nets:
                net.dag_embedding.get_dag().stoch_gate = False
                net.dag_embedding.get_dag().noise_gate = False
                net.dag_embedding.get_dag().s_thresh = False



    logger.info("Loading data...")
    train_dl, valid_dl, test_dl = load_data()
    dim = 28**2
    logger.info("Data loaded.")

    for epoch in range(nb_epoch):
        ll_tot = 0
        start = timer()

        # Update constraints
        if epoch % 1 == 0 and not umnn_maf:
            with torch.no_grad():
                model.constrainA(zero_threshold=0.)

        if epoch % nb_step_dual == 0 and epoch != 0 and not umnn_maf and epoch > min_pre_heating_epochs:
            model.update_dual_param()

        if not umnn_maf:
            for net in model.nets:
                dagness = net.DAGness()
                if dagness > 1e-10 and dagness < 1. and epoch > min_pre_heating_epochs:
                    #net.l1_weight = .1
                    net.dag_const = 1.
                    logger.info("Dagness constraint set on.")

        i = 0
        # Training loop
        if train:
            for batch_idx, (cur_x, target) in enumerate(train_loader):
                cur_x = cur_x.view(-1, dim).float().to(device)
                model.set_steps_nb(nb_steps + torch.randint(0, 10, [1])[0].item())
                loss = model.loss(cur_x) if not umnn_maf else -model.compute_ll(cur_x)[0].mean()
                if math.isnan(loss.item()):
                    print("ici")
                    print(loss.item())
                    print(model.compute_ll(cur_x))
                    exit()
                ll_tot += loss.item()
                i += 1
                opt.zero_grad()
                loss.backward(retain_graph=True)
                opt.step()

            ll_tot /= i
        else:
            ll_tot = 0.

        # Valid loop
        ll_test = 0.
        i = 0.
        with torch.no_grad():
            model.set_steps_nb(nb_steps + 20)
            for batch_idx, (cur_x, target) in enumerate(valid_loader):
                cur_x = cur_x.view(-1, dim).float().to(device)
                ll, _ = model.compute_ll(cur_x)
                ll_test += ll.mean().item()
                i += 1
        ll_test /= i

        end = timer()
        if umnn_maf:
            logger.info(
                "epoch: {:d} - Train loss: {:4f} - Valid loss: {:4f} - Elapsed time per epoch {:4f} (seconds)".
                    format(epoch, ll_tot, ll_test, end - start))
        else:
            dagness = max(model.DAGness())
            logger.info(
                "epoch: {:d} - Train loss: {:4f} - Valid log-likelihood: {:4f} - <<DAGness>>: {:4f} - Elapsed time per epoch {:4f} (seconds)".
                format(epoch, ll_tot, ll_test, dagness, end - start))

        if epoch % 10 == 0 and not umnn_maf:
            stoch_gate, noise_gate, s_thresh = [], [], []

            for net in model.nets:
                stoch_gate.append(net.getDag().stoch_gate)
                noise_gate.append(net.getDag().noise_gate)
                s_thresh.append(net.getDag().s_thresh)
                net.getDag().stoch_gate = False
                net.getDag().noise_gate = False
                net.getDag().s_thresh = True
            for threshold in [.95, .5, .1, .01, .0001]:
                model.set_h_threshold(threshold)
                # Valid loop
                ll_test = 0.
                i = 0.
                for batch_idx, (cur_x, target) in enumerate(valid_loader):
                    cur_x = cur_x.view(-1, dim).float().to(device)
                    ll, _ = model.compute_ll(cur_x)
                    ll_test += ll.mean().item()
                    i += 1
                ll_test /= i
                dagness = max(model.DAGness())
                logger.info("epoch: {:d} - Threshold: {:4f} - Valid log-likelihood: {:4f} - <<DAGness>>: {:4f}".
                            format(epoch, threshold, ll_test, dagness))
            i = 0
            model.set_h_threshold(0.)
            for net in model.nets:
                net.getDag().stoch_gate = stoch_gate[i]
                net.getDag().noise_gate = noise_gate[i]
                net.getDag().s_thresh = s_thresh[i]
                i += 1

        if epoch % nb_step_dual == 0:
            logger.info("Saving model N°%d" % epoch)
            torch.save(model.state_dict(), path + '/model_%d.pt' % epoch)
            torch.save(opt.state_dict(), path + '/ADAM_%d.pt' % epoch)

        torch.save(model.state_dict(), path + '/model.pt')
        torch.save(opt.state_dict(), path + '/ADAM.pt')

import argparse

parser = argparse.ArgumentParser(description='')
parser.add_argument("-load", default=False, action="store_true", help="Load a model ?")
parser.add_argument("-folder", default="", help="Folder")
parser.add_argument("-nb_steps_dual", default=100, type=int,
                    help="number of step between updating Acyclicity constraint and sparsity constraint")
parser.add_argument("-l1", default=10., type=float, help="Maximum weight for l1 regularization")
parser.add_argument("-nb_epoch", default=10000, type=int, help="Number of epochs")
parser.add_argument("-b_size", default=2, type=int, help="Batch size")
parser.add_argument("-int_net", default=[100, 100, 100, 100], nargs="+", type=int, help="NN hidden layers of UMNN")
parser.add_argument("-emb_net", default=[100, 100, 100, 2], nargs="+", type=int, help="NN layers of embedding")
parser.add_argument("-UMNN_MAF", default=False, action="store_true", help="replace the DAG-NF by a UMNN-MAF")
parser.add_argument("-nb_steps", default=20, type=int, help="Number of integration steps.")
parser.add_argument("-min_pre_heating_epochs", default=30, type=int, help="Number of heating steps.")
parser.add_argument("-f_number", default=None, type=str, help="Number of heating steps.")
parser.add_argument("-solver", default="CC", type=str, help="Which integral solver to use.",
                    choices=["CC", "CCParallel"])
parser.add_argument("-nb_flow", type=int, default=1, help="Number of steps in the flow.")
parser.add_argument("-test", default=False, action="store_true")
parser.add_argument("-linear_net", default=False, action="store_true")
parser.add_argument("-gumble_T", default=1., type=float, help="Temperature of the gumble distribution.")
parser.add_argument("-weight_decay", default=1e-5, type=float, help="Weight decay value")
parser.add_argument("-learning_rate", default=1e-3, type=float, help="Weight decay value")
parser.add_argument("-hutchinson", default=0, type=int, help="Use a hutchinson trace estimator if non null")

args = parser.parse_args()
from datetime import datetime
now = datetime.now()

path =  "MNIST/" + now.strftime("%m_%d_%Y_%H_%M_%S") if args.folder == "" else args.folder
if not (os.path.isdir(path)):
    os.makedirs(path)
train(load=args.load, path=path, nb_step_dual=args.nb_steps_dual, l1=args.l1, nb_epoch=args.nb_epoch,
      int_net=args.int_net, emb_net=args.emb_net, b_size=args.b_size, all_args=args, umnn_maf=args.UMNN_MAF,
      nb_steps=args.nb_steps, min_pre_heating_epochs=args.min_pre_heating_epochs, file_number=args.f_number,
      solver=args.solver, nb_flow=args.nb_flow, train=not args.test, linear_net=args.linear_net,
      gumble_T=args.gumble_T, weight_decay=args.weight_decay, learning_rate=args.learning_rate, hutchinson=args.hutchinson)
