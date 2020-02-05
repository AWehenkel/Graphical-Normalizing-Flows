import lib.toy_data as toy_data
from models import DAGNF
import torch
from timeit import default_timer as timer
import lib.utils as utils
import os
import lib.visualize_flow as vf
import matplotlib.pyplot as plt
from UMNN import UMNNMAFFlow
import networkx as nx


def train_toy(toy, load=True, nb_steps=20, folder=""):
    logger = utils.get_logger(logpath=os.path.join(folder, toy, 'logs'), filepath=os.path.abspath(__file__))

    logger.info("Creating model...")

    device = "cpu" if not(torch.cuda.is_available()) else "cuda:0"

    nb_samp = 100
    batch_size = 100

    x_test = torch.tensor(toy_data.inf_train_gen(toy, batch_size=1000)).to(device)
    x = torch.tensor(toy_data.inf_train_gen(toy, batch_size=1000)).to(device)

    dim = x.shape[1]
    model = DAGNF(in_d=dim, hiddens_integrand=[200, 200, 200, 200], device=device)

    opt = torch.optim.Adam(model.parameters(), 1e-3, weight_decay=1e-5)

    if load:
        logger.info("Loading model...")
        model.load_state_dict(torch.load(toy + '/model.pt'))
        model.train()
        opt.load_state_dict(torch.load(toy + '/ADAM.pt'))
        logger.info("Model loaded.")

    for epoch in range(10000):
        ll_tot = 0
        start = timer()
        for j in range(0, nb_samp, batch_size):
            cur_x = torch.tensor(toy_data.inf_train_gen(toy, batch_size=batch_size)).to(device)
            print(cur_x.shape)
            #ll, _ = model.compute_ll(cur_x)
            #loss = -ll.mean()#
            loss = model.loss(cur_x)
            ll_tot += loss.item()
            opt.zero_grad()
            loss.backward(retain_graph=True)
            opt.step()
        if epoch % 1 == 0:
            with torch.no_grad():
                model.dag_embedding.dag.constrainA()

        if epoch % 50 == 0 and epoch != 0:
            model.update_dual_param()

        end = timer()
        ll_test, _ = model.compute_ll(x_test)
        ll_test = -ll_test.mean()
        logger.info("epoch: {:d} - Train loss: {:4f} - Test loss: {:4f} - Elapsed time per epoch {:4f} (seconds)".
                    format(epoch, ll_tot, ll_test.item(), end-start))
        if epoch % 100 == 0:
            if toy in ["2spirals-8gaussians", "4-2spirals-8gaussians", "8-2spirals-8gaussians"]:
                def compute_ll_2spirals(x):
                    return model.compute_ll(torch.cat((x, torch.zeros(x.shape[0], dim-2).to(device)), 1))
                def compute_ll_8gaussians(x):
                    return model.compute_ll(torch.cat((torch.zeros(x.shape[0], dim-2).to(device), x), 1))
                with torch.no_grad():
                    ax = plt.subplot(1, 3, 1, aspect="equal")
                    vf.plt_flow(compute_ll_2spirals, ax)
                    ax = plt.subplot(1, 3, 2, aspect="equal")
                    vf.plt_flow(compute_ll_8gaussians, ax)

            # Plot DAG
            A = model.dag_embedding.dag.soft_thresholded_A().detach().numpy().T
            ax = plt.subplot(1, 3, 3)
            G = nx.from_numpy_matrix(A, create_using=nx.DiGraph)
            pos = nx.layout.fruchterman_reingold_layout(G)
            nodes = nx.draw_networkx_nodes(G, pos, node_size=200, node_color='blue', alpha=.7)
            edges, weights = zip(*nx.get_edge_attributes(G, 'weight').items())
            edges = nx.draw_networkx_edges(G, pos, node_size=200, arrowstyle='->',
                                           arrowsize=3, connectionstyle='arc3,rad=0.2',
                                           edge_cmap=plt.cm.Blues, width=weights)
            labels = {}
            for i in range(dim):
                labels[i] = str(r'$%d$' % i)
            nx.draw_networkx_labels(G, pos, labels, font_size=12)

            #vf.plt_flow(model.compute_ll, ax)
            plt.savefig("%s%s/flow_%d.pdf" % (folder, toy, epoch))
            torch.save(model.state_dict(), folder + toy + '/model.pt')
            torch.save(opt.state_dict(), folder + toy + '/ADAM.pt')
            G.clear()
            plt.clf()
            print(A)

toy = "4-2spirals-8gaussians"

import argparse
datasets = ["8gaussians", "swissroll", "moons", "pinwheel", "cos", "2spirals", "checkerboard", "line", "line-noisy",
            "circles", "joint_gaussian", "2spirals-8gaussians", "4-2spirals-8gaussians", "8-2spirals-8gaussians"]

parser = argparse.ArgumentParser(description='')
parser.add_argument("-dataset", default=None, choices=datasets, help="Which toy problem ?")
parser.add_argument("-load", default=False, action="store_true", help="Load a model ?")
parser.add_argument("-folder", default="", help="Folder")
args = parser.parse_args()


if args.dataset is None:
    toys = datasets
else:
    toys = [args.dataset]

for toy in toys:
    if not(os.path.isdir(args.folder + toy)):
        os.makedirs(args.folder + toy)
    train_toy(toy, load=args.load, folder=args.folder)
