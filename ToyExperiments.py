import lib.toy_data as toy_data
from models import DAGNF, MLP
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


def train_toy(toy, load=True, nb_step_dual=300, nb_steps=15, folder="", l1=1., nb_epoch=50000, pre_heating_epochs=10):
    logger = utils.get_logger(logpath=os.path.join(folder, toy, 'logs'), filepath=os.path.abspath(__file__))

    logger.info("Creating model...")

    device = "cpu" if not(torch.cuda.is_available()) else "cuda:0"

    nb_samp = 100
    batch_size = 100

    x_test = torch.tensor(toy_data.inf_train_gen(toy, batch_size=1000)).to(device)
    x = torch.tensor(toy_data.inf_train_gen(toy, batch_size=1000)).to(device)

    dim = x.shape[1]
    linear_net = True
    nb_flow = 10
    emb_net = [100, 100, 100, 100, 100, 2]
    emb_nets = []
    for i in range(nb_flow):
        if emb_net is not None:
            net = MLP(dim, hidden=emb_net[:-1], out_d=emb_net[-1], device=device)
        else:
            net = None
        emb_nets.append(net)

    model = DAGNF(nb_flow=nb_flow, in_d=dim, hidden_integrand=[50, 50, 50], emb_d=emb_nets[0].out_d, emb_nets=emb_nets, device=device,
                  l1_weight=l1, nb_steps=nb_steps, linear_normalizer=linear_net)
    model.dag_const = 0.

    if True:
        i = 0
        for net in model.nets:
            with torch.no_grad():
                print("coucou")
                A = torch.zeros(2, 2)
                if i % 2 == 0:
                    A[1, 0] = 1.
                    #A[2, 0] = 1.
                else:
                    A[0, 1] = 1.
                    #A[0, 2] = 1.
                i += 1
                net.getDag().stoch_gate = False
                net.getDag().s_thresh = False
                net.getDag().h_thresh = 0.
                net.getDag().post_process(1e-3)
                net.dag_embedding.get_dag().A.data = A.float().to(device)
                net.dag_const = 0.
                net.getDag().gumble = False

    opt = torch.optim.Adam(model.parameters(), 1e-4, weight_decay=1e-5)
    #opt = torch.optim.RMSprop(model.parameters(), lr=1e-3)

    if load:
        logger.info("Loading model...")
        model.load_state_dict(torch.load(toy + '/model.pt'))
        model.train()
        opt.load_state_dict(torch.load(toy + '/ADAM.pt'))
        logger.info("Model loaded.")

    if True:
        for net in model.nets:
            net.getDag().stoch_gate = True
            net.getDag().noise_gate = False
            net.getDag().gumble_T = .5
    for epoch in range(nb_epoch):
        ll_tot = 0
        start = timer()
        for j in range(0, nb_samp, batch_size):
            cur_x = torch.tensor(toy_data.inf_train_gen(toy, batch_size=batch_size)).to(device)
            loss = model.loss(cur_x)
            ll_tot += loss.item()
            if math.isnan(loss.item()):
                ll, z = model.compute_ll(cur_x)
                print(ll)
                print(z)
                print(ll.max(), z.max())
                exit()
            opt.zero_grad()
            loss.backward(retain_graph=True)
            opt.step()
        if epoch % 1 == 0:
            with torch.no_grad():
                model.constrainA(zero_threshold=0.)

        if epoch % nb_step_dual == 0 and epoch > pre_heating_epochs:
            model.update_dual_param()

        end = timer()
        ll_test, _ = model.compute_ll(x_test)
        ll_test = -ll_test.mean()
        dagness = max(model.DAGness())
        if epoch > pre_heating_epochs:
            for net in model.nets:
                net.l1_weight = .0
                net.dag_const = 1.
        logger.info("epoch: {:d} - Train loss: {:4f} - Test loss: {:4f} - <<DAGness>>: {:4f} - Elapsed time per epoch {:4f} (seconds)".
                    format(epoch, ll_tot, ll_test.item(), dagness, end-start))


        if epoch % 100 == 0:
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
                    ll_test, _ = model.compute_ll(x_test)
                    ll_test = -ll_test.mean().item()
                    dagness = max(model.DAGness()).item()
                    logger.info("epoch: {:d} - Threshold: {:4f} - Valid log-likelihood: {:4f} - <<DAGness>>: {:4f}".
                                format(epoch, threshold, ll_test, dagness))
                model.getDag().stoch_gate = stoch_gate
                model.getDag().noise_gate = noise_gate
                model.getDag().s_thresh = s_thresh
                model.set_h_threshold(0.)


        if epoch % 500 == 0:
            if toy in ["2spirals-8gaussians", "4-2spirals-8gaussians", "8-2spirals-8gaussians", "2gaussians", "2igaussians", "8gaussians"]:
                def compute_ll_2spirals(x):
                    if toy in ["2gaussians", "8gaussians", "2igaussians"]:
                        return model.compute_ll(x.to(device))
                    return model.compute_ll(torch.cat((x, torch.zeros(x.shape[0], dim-2).to(device)), 1))
                def compute_ll_8gaussians(x):
                    if toy in ["2gaussians", "8gaussians", "2igaussians"]:
                        return model.compute_ll(x.to(device))
                    return model.compute_ll(torch.cat((torch.zeros(x.shape[0], dim-2).to(device), x), 1))
                with torch.no_grad():
                    npts = 100
                    plt.figure(figsize=(30, 10))
                    ax = plt.subplot(1, 3, 1, aspect="equal")
                    qz_1, qz_2 = vf.plt_flow(compute_ll_2spirals, ax, npts=npts, device=device)
                    plt.subplot(1, 3, 2)
                    plt.plot(np.linspace(-4, 4, npts), qz_1)
                    plt.subplot(1, 3, 3)
                    plt.plot(np.linspace(-4, 4, npts), qz_2)
                    flow_type = "linear" if linear_net else "monotonic"
                    plt.savefig("%s%s/flow_%s_%d_%d.pdf" % (folder, toy, flow_type, nb_flow, epoch))

            # Plot DAG
            font = {'family': 'normal',
                    'weight': 'bold',
                    'size': 12}

            matplotlib.rc('font', **font)
            for net in model.nets:
                A_normal = net.getDag().soft_thresholded_A().detach().cpu().numpy().T
                print(A_normal)
                #logger.info(str(A_normal))
                A_thresholded = A_normal * (A_normal > .001)
                j = 0
                for A, name in zip([A_normal, A_thresholded], ["normal", "thresholded"]):
                    #A /= A.sum() / np.log(dim)

                    ax = plt.subplot(2, 2, 1 + j)
                    plt.title(name + " DAG")
                    G = nx.from_numpy_matrix(A, create_using=nx.DiGraph)
                    pos = nx.layout.spring_layout(G)
                    nx.draw_networkx_nodes(G, pos, node_size=200, node_color='blue', alpha=.7)
                    if nx.get_edge_attributes(G, 'weight').items() is not None:
                        edges, weights = zip(*nx.get_edge_attributes(G, 'weight').items())
                    nx.draw_networkx_edges(G, pos, node_size=200, arrowstyle='->',
                                                   arrowsize=3, connectionstyle='arc3,rad=0.2',
                                                   edge_cmap=plt.cm.Blues, width=5*weights)
                    labels = {}
                    for i in range(dim):
                        labels[i] = str(r'$%d$' % i)
                    nx.draw_networkx_labels(G, pos, labels, font_size=12)

                    ax = plt.subplot(2, 2, 2 + j)
                    out = ax.matshow(np.log(A))
                    plt.colorbar(out, ax=ax)
                    j += 2

                #vf.plt_flow(model.compute_ll, ax)
                #plt.savefig("%s%s/DAG_%d.pdf" % (folder, toy, epoch))
                torch.save(model.state_dict(), folder + toy + '/model.pt')
                torch.save(opt.state_dict(), folder + toy + '/ADAM.pt')
                G.clear()
                plt.clf()


toy = "8gaussians"

import argparse
datasets = ["2igaussians", "2gaussians", "8gaussians", "swissroll", "moons", "pinwheel", "cos", "2spirals", "checkerboard", "line", "line-noisy",
            "circles", "joint_gaussian", "2spirals-8gaussians", "4-2spirals-8gaussians", "8-2spirals-8gaussians",
            "8-MIX", "7-MIX"]

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
