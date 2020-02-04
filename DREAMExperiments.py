import lib.toy_data as toy_data
from models import DAGNF
import torch
from timeit import default_timer as timer
import lib.utils as utils
import os
import lib.visualize_flow as vf
import matplotlib.pyplot as plt
from UMNN import UMNNMAFFlow
from Dream.dream_reader import DreamData
from Dream.aupr import *

def train_toy(toy, load=True, nb_steps=20, nb_flow=1, folder=""):
    logger = utils.get_logger(logpath=os.path.join(folder, toy, 'logs'), filepath=os.path.abspath(__file__))

    logger.info("Creating model...")

    device = "cpu"

    #model = UMNNMAFFlow(nb_flow=nb_flow, nb_in=4, hidden_derivative=[100, 100, 100], hidden_embedding=[50, 50, 50],
    #                    embedding_s=10, nb_steps=nb_steps, device=device).to(device)
    data_size = 100
    model = DAGNF(in_d=data_size).to(device)

    opt = torch.optim.Adam(model.parameters(), 1e-3, weight_decay=1e-5)

    nb_samp = 100
    batch_size = 100

    dataset = DreamData("Dream/Dream4", data_size=data_size)
    x_test = torch.tensor(toy_data.inf_train_gen(toy, batch_size=1000)).to(device)
    x = torch.tensor(toy_data.inf_train_gen(toy, batch_size=1000)).to(device)

    for epoch in range(10000):
        ll_tot = 0
        start = timer()
        for j in range(0, nb_samp, batch_size):
            cur_x = torch.tensor(dataset.get_full_dataset()).float()
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

        if epoch % 5 == 0:
            model.update_dual_param()
        end = timer()

        if epoch % 1 == 0:
            logger.info("epoch: {:d} - Train loss: {:4f} - Elapsed time per epoch {:4f} (seconds)".
                        format(epoch, ll_tot, end - start))
            torch.save(model.state_dict(), toy + '/model.pt')
            torch.save(opt.state_dict(), toy + '/ADAM.pt')
            print(model.dag_embedding.dag.A)
            print((((model.dag_embedding.dag.A > 0).float().numpy() - dataset.adjacence_matrix.transpose())**2).sum())
            print(new_gene_AUPR(model.dag_embedding.dag.A.detach().float().numpy(), dataset.adjacence_matrix.transpose()))
            print(new_gene_AUPR(model.dag_embedding.dag.A.detach().float().numpy(), dataset.adjacence_matrix))

dir = "DREAM"
if not(os.path.isdir(dir)):
    os.makedirs(dir)
train_toy(dir)