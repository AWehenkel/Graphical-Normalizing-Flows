from models import DAGNF, MLP, MNISTCNN, DAGStep
import torch
from timeit import default_timer as timer
import lib.utils as utils
import os
import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import UCIdatasets
import numpy as np
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


def load_data(name):

    if name == 'bsds300':
        return UCIdatasets.BSDS300()

    elif name == 'power':
        return UCIdatasets.POWER()

    elif name == 'gas':
        return UCIdatasets.GAS()

    elif name == 'hepmass':
        return UCIdatasets.HEPMASS()

    elif name == 'miniboone':
        return UCIdatasets.MINIBOONE()

    elif name == "digits":
        return UCIdatasets.DIGITS()

    else:
        raise ValueError('Unknown dataset')


def train(dataset="POWER", load=True, nb_step_dual=100, nb_steps=20, path="", l1=.1, nb_epoch=10000,
          int_net=[200, 200, 200], emb_net=[200, 200, 200], b_size=100, umnn_maf=False, min_pre_heating_epochs=30,
          all_args=None, file_number=None, train=True, solver="CC", nb_flow=1):
    logger = utils.get_logger(logpath=os.path.join(path, 'logs'), filepath=os.path.abspath(__file__))
    logger.info(str(all_args))

    logger.info("Creating model...")

    device = "cpu" if not(torch.cuda.is_available()) else "cuda:0"

    if load:
        train = False
        file_number = "_" + file_number if file_number is not None else ""

    batch_size = b_size

    logger.info("Loading data...")
    data = load_data(dataset)
    data.trn.x = torch.from_numpy(data.trn.x).to(device)
    data.val.x = torch.from_numpy(data.val.x).to(device)
    data.tst.x = torch.from_numpy(data.tst.x).to(device)
    logger.info("Data loaded.")

    dim = data.trn.x.shape[1]
    if umnn_maf:
        model = UMNNMAFFlow(nb_flow=nb_flow, nb_in=dim, hidden_derivative=int_net, hidden_embedding=emb_net[:-1],
                            embedding_s=emb_net[-1], nb_steps=nb_steps, device=device).to(device)
    else:
        emb_nets = []
        for i in range(nb_flow):
            if emb_net is not None:
                if dataset == "mnist":
                    net = MNISTCNN()
                else:
                    net = MLP(dim, hidden=emb_net[:-1], out_d=emb_net[-1], device=device)
            else:
                net = None
            emb_nets.append(net)
        l1_weight = l1
        model = DAGNF(nb_flow=nb_flow, in_d=dim, hidden_integrand=int_net, emb_d=emb_nets[0].out_d, emb_nets=emb_nets, device=device,
                      l1_weight=l1, nb_steps=nb_steps, solver=solver)
        if nb_flow == 1:
            model = DAGStep(in_d=dim, hidden_integrand=int_net, emb_d=emb_nets[0].out_d, emb_net=emb_nets[0],
                            device=device, l1_weight=l1, nb_steps=nb_steps, solver=solver)
            model.nets = [model]

    model.dag_const = 0.
    opt = torch.optim.Adam(model.parameters(), 1e-3, weight_decay=1e-5)
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
                net.dag_embedding.dag.stoch_gate = False
                net.dag_embedding.dag.noise_gate = False
                net.dag_embedding.dag.s_thresh = False


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
                    net.l1_weight = .0
                    net.dag_const = 1.
                    logger.info("Dagness constraint set on.")

        i = 0
        # Training loop
        if train:
            for cur_x in batch_iter(data.trn.x, shuffle=True, batch_size=batch_size):
                model.set_steps_nb(nb_steps + torch.randint(0, 10, [1])[0].item())
                loss = model.loss(cur_x) if not umnn_maf else -model.compute_ll(cur_x)[0].mean()
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
            for cur_x in batch_iter(data.val.x, shuffle=True, batch_size=batch_size):
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
            dagness = model.DAGness() if nb_flow == 1 else max(model.DAGness())
            logger.info("epoch: {:d} - Train loss: {:4f} - Valid log-likelihood: {:4f} - <<DAGness>>: {:4f} - Elapsed time per epoch {:4f} (seconds)".
                        format(epoch, ll_tot, ll_test, dagness, end-start))

        if epoch % 10 == 0 and not umnn_maf:
            for threshold in [.1, .01, .0001, 1e-8]:
                model.set_h_threshold(threshold)
                # Valid loop
                ll_test = 0.
                i = 0.
                for cur_x in batch_iter(data.val.x, shuffle=True, batch_size=batch_size):
                    ll, _ = model.compute_ll(cur_x)
                    ll_test += ll.mean().item()
                    i += 1
                ll_test /= i
                dagness = model.DAGness() if nb_flow == 1 else max(model.DAGness())
                logger.info("epoch: {:d} - Threshold: {:4f} - Valid log-likelihood: {:4f} - <<DAGness>>: {:4f}".
                            format(epoch, threshold, ll_test, dagness))
            model.set_h_threshold(0.)

        if epoch % nb_step_dual == 0:

            if not umnn_maf:
                # Plot DAG
                font = {'family': 'normal',
                        'weight': 'bold',
                        'size': 12}

                matplotlib.rc('font', **font)
                A_normal = model.nets[0].dag_embedding.dag.soft_thresholded_A().detach().cpu().numpy().T
                logger.info(str(A_normal))
                A_thresholded = A_normal * (A_normal > .001)
                j = 0
                for A, name in zip([A_normal, A_thresholded], ["normal", "thresholded"]):
                    A /= A.sum() / np.log(dim)
                    ax = plt.subplot(2, 2, 1 + j)
                    plt.title(name + " DAG")
                    G = nx.from_numpy_matrix(A, create_using=nx.DiGraph)
                    pos = nx.layout.spring_layout(G)
                    nx.draw_networkx_nodes(G, pos, node_size=200, node_color='blue', alpha=.7)
                    edges, weights = zip(*nx.get_edge_attributes(G, 'weight').items())
                    nx.draw_networkx_edges(G, pos, node_size=200, arrowstyle='->',
                                           arrowsize=3, connectionstyle='arc3,rad=0.2',
                                           edge_cmap=plt.cm.Blues, width=5 * weights)
                    labels = {}
                    for i in range(dim):
                        labels[i] = str(r'$%d$' % i)
                    nx.draw_networkx_labels(G, pos, labels, font_size=12)

                    ax = plt.subplot(2, 2, 2 + j)
                    out = ax.matshow(np.log(A))
                    plt.colorbar(out, ax=ax)
                    j += 2
                    # vf.plt_flow(model.compute_ll, ax)
                plt.savefig("%s/DAG_%d.pdf" % (path, epoch))
                G.clear()
                plt.clf()

            torch.save(model.state_dict(), path + '/model_%d.pt' % epoch)
            torch.save(opt.state_dict(), path + '/ADAM_%d.pt' % epoch)

        torch.save(model.state_dict(), path + '/model.pt')
        torch.save(opt.state_dict(), path + '/ADAM.pt')

import argparse
datasets = ["power", "gas", "bsds300", "miniboone", "hepmass", "digits"]

parser = argparse.ArgumentParser(description='')
parser.add_argument("-dataset", default=None, choices=datasets, help="Which toy problem ?")
parser.add_argument("-load", default=False, action="store_true", help="Load a model ?")
parser.add_argument("-folder", default="", help="Folder")
parser.add_argument("-nb_steps_dual", default=100, type=int, help="number of step between updating Acyclicity constraint and sparsity constraint")
parser.add_argument("-l1", default=.2, type=float, help="Maximum weight for l1 regularization")
parser.add_argument("-nb_epoch", default=10000, type=int, help="Number of epochs")
parser.add_argument("-b_size", default=100, type=int, help="Batch size")
parser.add_argument("-int_net", default=[100, 100, 100, 100], nargs="+", type=int, help="NN hidden layers of UMNN")
parser.add_argument("-emb_net", default=[100, 100, 100, 10], nargs="+", type=int, help="NN layers of embedding")
parser.add_argument("-UMNN_MAF", default=False, action="store_true", help="replace the DAG-NF by a UMNN-MAF")
parser.add_argument("-nb_steps", default=20, type=int, help="Number of integration steps.")
parser.add_argument("-min_pre_heating_epochs", default=30, type=int, help="Number of heating steps.")
parser.add_argument("-f_number", default=None, type=str, help="Number of heating steps.")
parser.add_argument("-solver", default="CC", type=str, help="Which integral solver to use.",
                    choices=["CC", "CCParallel"])
parser.add_argument("-nb_flow", type=int, default=1, help="Number of steps in the flow.")


args = parser.parse_args()
from datetime import datetime
now = datetime.now()

if args.dataset is None:
    toys = datasets
else:
    toys = [args.dataset]

for toy in toys:
    path = toy + "/" + now.strftime("%m_%d_%Y_%H_%M_%S") if args.folder == "" else args.folder
    if not(os.path.isdir(path)):
        os.makedirs(path)
    train(toy, load=args.load, path=path, nb_step_dual=args.nb_steps_dual, l1=args.l1, nb_epoch=args.nb_epoch,
          int_net=args.int_net, emb_net=args.emb_net, b_size=args.b_size, all_args=args, umnn_maf=args.UMNN_MAF,
          nb_steps=args.nb_steps, min_pre_heating_epochs=args.min_pre_heating_epochs, file_number=args.f_number,
          solver=args.solver, nb_flow=args.nb_flow)
