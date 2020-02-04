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
import sklearn.datasets as datasets
import numpy as np


def train(load=True, nb_steps=20, nb_flow=1, folder=""):
    dir = "California-Housing"
    logger = utils.get_logger(logpath=os.path.join(folder, dir, 'logs'), filepath=os.path.abspath(__file__))

    logger.info("Creating model...")

    device = "cpu" if not(torch.cuda.is_available()) else "cuda:0"

    #model = UMNNMAFFlow(nb_flow=nb_flow, nb_in=4, hidden_derivative=[100, 100, 100], hidden_embedding=[50, 50, 50],
    #                    embedding_s=10, nb_steps=nb_steps, device=device).to(device)
    dim = 8
    model = DAGNF(in_d=dim, hiddens_integrand=[200, 200, 200, 200], device=device)

    opt = torch.optim.Adam(model.parameters(), 1e-3, weight_decay=1e-5)

    if load:
        logger.info("Loading model...")
        model.load_state_dict(torch.load(dir + '/model.pt'))
        model.train()
        opt.load_state_dict(torch.load(dir + '/ADAM.pt'))
        logger.info("Model loaded.")

    nb_samp = 100
    batch_size = 100

    all_data = datasets.fetch_california_housing(data_home=None, download_if_missing=True, return_X_y=False)
    X = np.append([all_data['data'], all_data['target']])

    x_test = torch.tensor(toy_data.inf_train_gen(dir, batch_size=1000)).to(device)
    x = torch.tensor(toy_data.inf_train_gen(dir, batch_size=1000)).to(device)

    for epoch in range(10000):
        ll_tot = 0
        start = timer()
        for j in range(0, nb_samp, batch_size):
            cur_x = torch.tensor(toy_data.inf_train_gen(dir, batch_size=batch_size)).to(device)
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
            def compute_ll_2spirals(x):
                return model.compute_ll(torch.cat((x, torch.zeros(x.shape[0], dim-2)), 1))
            def compute_ll_8gaussians(x):
                return model.compute_ll(torch.cat((torch.zeros(x.shape[0], dim-2), x), 1))
            ax = plt.subplot(1, 3, 1, aspect="equal")
            vf.plt_flow(compute_ll_2spirals, ax)
            ax = plt.subplot(1, 3, 2, aspect="equal")
            vf.plt_flow(compute_ll_8gaussians, ax)

            # Plot DAG
            A = model.dag_embedding.dag.A.detach().numpy().T
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
            plt.savefig("%s/flow_%d.pdf" % (dir, epoch))
            torch.save(model.state_dict(), dir + '/model.pt')
            torch.save(opt.state_dict(), dir + '/ADAM.pt')
            G.clear()
            plt.clf()
            print(model.dag_embedding.dag.A)

dir = "4-2spirals-8gaussians"
if not(os.path.isdir(dir)):
    os.makedirs(dir)
train_toy(dir)