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
from Datasets import CaliforniaHousingDataset


def train(load=False, nb_steps=20, nb_flow=1, folder=""):
    dir = "California-Housing"
    if not (os.path.isdir(dir)):
        os.makedirs(dir)
    logger = utils.get_logger(logpath=os.path.join(folder, dir, 'logs'), filepath=os.path.abspath(__file__))

    logger.info("Creating model...")

    device = "cpu" if not(torch.cuda.is_available()) else "cuda:0"

    #model = UMNNMAFFlow(nb_flow=nb_flow, nb_in=4, hidden_derivative=[100, 100, 100], hidden_embedding=[50, 50, 50],
    #                    embedding_s=10, nb_steps=nb_steps, device=device).to(device)

    batch_size = 100
    n_worker = 1
    shuffle = True

    d_train = CaliforniaHousingDataset("train")
    d_test = CaliforniaHousingDataset("test", normalize=False).normalize(d_train.mu, d_train.std)
    d_valid = CaliforniaHousingDataset("valid", normalize=False).normalize(d_train.mu, d_train.std)

    dl_train = torch.utils.data.DataLoader(dataset=d_train, batch_size=batch_size, num_workers=n_worker, shuffle=shuffle)
    dl_test = torch.utils.data.DataLoader(dataset=d_test, batch_size=batch_size, num_workers=n_worker, shuffle=shuffle)
    dl_valid = torch.utils.data.DataLoader(dataset=d_valid, batch_size=batch_size, num_workers=n_worker, shuffle=shuffle)

    dim = d_train[0].shape[0]
    model = DAGNF(in_d=dim, hidden_integrand=[200, 200, 200, 200], device=device)

    opt = torch.optim.Adam(model.parameters(), 1e-3, weight_decay=1e-5)

    if load:
        logger.info("Loading model...")
        model.load_state_dict(torch.load(dir + '/model.pt'))
        model.train()
        opt.load_state_dict(torch.load(dir + '/ADAM.pt'))
        logger.info("Model loaded.")

    for epoch in range(10000):
        ll_tot = 0
        i = 0
        start = timer()
        for _, X in enumerate(dl_train):
            cur_x = torch.tensor(X, device=device, dtype=torch.float)
            loss = model.loss(cur_x)
            ll_tot += loss.item()
            i += 1
            opt.zero_grad()
            loss.backward(retain_graph=True)
            opt.step()
        if epoch % 1 == 0:
            with torch.no_grad():
                model.dag_embedding.dag.constrainA()

        if epoch % 1 == 0 and epoch != 0:
            model.update_dual_param()

        ll_tot /= i
        end = timer()
        with torch.no_grad():
            ll_test = 0.
            i = 0
            for _, X in enumerate(dl_test):
                cur_x = torch.tensor(X, device=device, dtype=torch.float)
                ll, _ = model.compute_ll(cur_x)
                ll_test += ll.mean().item()
                i += 1
        ll_test /= i
        logger.info("epoch: {:d} - Train loss: {:4f} - Test log-likelihood: {:4f} - Elapsed time per epoch {:4f} (seconds)".
                    format(epoch, ll_tot, ll_test, end-start))
        if epoch % 5 == 0:

            # Plot DAG
            plt.figure()
            A = model.dag_embedding.dag.A.detach().cpu().numpy().T
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


train()