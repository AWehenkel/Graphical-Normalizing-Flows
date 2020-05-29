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
import torch.nn as nn
from UMNN import UMNNMAFFlow
from models.NormalizingFlowFactories import buildMNISTNormalizingFlow, buildCIFAR10NormalizingFlow, buildFCNormalizingFlow
from models.Normalizers import AffineNormalizer, MonotonicNormalizer
from models.Conditionners import *
import torchvision.datasets as dset
import torchvision.transforms as tforms
import matplotlib.animation as animation
import matplotlib
import torchvision

def add_noise(x):
    """
    [0, 1] -> [0, 255] -> add noise -> [0, 1]
    """
    noise = x.new().resize_as_(x).uniform_()
    x = x * 255 + noise
    x = x / 256
    return x


def compute_bpp(ll, x, alpha=1e-6):
    d = x.shape[1]
    bpp = -ll / (d * np.log(2)) - np.log2(1 - 2 * alpha) + 8 \
          + 1 / d * (torch.log2(torch.sigmoid(x)) + torch.log2(1. - torch.sigmoid(x))).sum(1)
    return bpp


def load_data(dataset="MNIST", batch_size=100, cuda=-1):
    if dataset == "MNIST":
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

        train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True, **kwargs)
        valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size, shuffle=True, drop_last=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True, drop_last=True, **kwargs)
    elif len(dataset) == 6 and dataset[:5] == 'MNIST':
        data = datasets.MNIST('./MNIST', train=True, download=True,
                              transform=transforms.Compose([
                                  AddUniformNoise(),
                                  ToTensor()
                              ]))
        label = int(dataset[5])
        idx = data.train_labels == label
        data.targets = data.train_labels[idx]
        data.data = data.train_data[idx]

        train_data, valid_data = torch.utils.data.random_split(data, [5000, idx.sum() - 5000])

        test_data = datasets.MNIST('./MNIST', train=False, download=True,
                                   transform=transforms.Compose([
                                       AddUniformNoise(),
                                       ToTensor()
                                   ]))
        idx = test_data.test_labels == label
        test_data.targets = test_data.test_labels[idx]
        test_data.data = test_data.test_data[idx]

        kwargs = {'num_workers': 0, 'pin_memory': True} if cuda > -1 else {}

        train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True,
                                                   **kwargs)
        valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size, shuffle=True, drop_last=True,
                                                   **kwargs)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True, drop_last=True,
                                                  **kwargs)
    elif dataset == "CIFAR10":
        im_dim = 3
        im_size = 32  # if args.imagesize is None else args.imagesize
        trans = lambda im_size: tforms.Compose([tforms.Resize(im_size), tforms.ToTensor(), add_noise])
        train_data = dset.CIFAR10(
            root="./data", train=True, transform=tforms.Compose([
                tforms.Resize(im_size),
                tforms.RandomHorizontalFlip(),
                tforms.ToTensor(),
                add_noise,
            ]), download=True
        )
        test_data = dset.CIFAR10(root="./data", train=False, transform=trans(im_size), download=True)
        kwargs = {'num_workers': 0, 'pin_memory': True} if cuda > -1 else {}

        train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, drop_last=True, shuffle=True, **kwargs)
        # WARNING VALID = TEST
        valid_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, drop_last=True, shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, drop_last=True, shuffle=True, **kwargs)
    return train_loader, valid_loader, test_loader


cond_types = {"DAG": DAGConditioner, "Coupling": CouplingConditioner, "Autoregressive": AutoregressiveConditioner}


def train(dataset="MNIST", load=True, nb_step_dual=100, nb_steps=20, path="", l1=.1, nb_epoch=10000, b_size=100,
          int_net=[50, 50, 50], all_args=None, file_number=None, train=True, solver="CC", weight_decay=1e-5,
          learning_rate=1e-3, batch_per_optim_step=1, n_gpu=1, norm_type='Affine', nb_flow=[1], hot_encoding=True,
          prior_A_kernel=None, conditioner="DAG", emb_net=None):
    logger = utils.get_logger(logpath=os.path.join(path, 'logs'), filepath=os.path.abspath(__file__))
    logger.info(str(all_args))


    if load:
        file_number = "_" + file_number if file_number is not None else ""

    batch_size = b_size
    best_valid_loss = np.inf

    logger.info("Loading data...")
    train_loader, valid_loader, test_loader = load_data(dataset, batch_size)
    if len(dataset) == 6 and dataset[:5] == 'MNIST':
        dataset = "MNIST"
    alpha = 1e-6 if dataset == "MNIST" else .05

    logger.info("Data loaded.")

    master_device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # -----------------------  Model Definition ------------------- #
    logger.info("Creating model...")
    if norm_type == 'Affine':
        normalizer_type = AffineNormalizer
        normalizer_args = {}
    else:
        normalizer_type = MonotonicNormalizer
        normalizer_args = {"integrand_net": int_net, "nb_steps": 15, "solver": solver}

    if conditioner == "DAG":
        conditioner_type = DAGConditioner
        if dataset == "MNIST":
            inner_model = buildMNISTNormalizingFlow(nb_flow, normalizer_type, normalizer_args, l1,
                                                    nb_epoch_update=nb_step_dual, hot_encoding=hot_encoding,
                                                    prior_kernel=prior_A_kernel)
        elif dataset == "CIFAR10":
            inner_model = buildCIFAR10NormalizingFlow(nb_flow, normalizer_type, normalizer_args, l1,
                                                      nb_epoch_update=nb_step_dual, hot_encoding=hot_encoding)
        else:
            logger.info("Wrong dataset name. Training aborted.")
            exit()
    else:
        dim = 28**2 if dataset == "MNIST" else 32*32*3
        conditioner_type = cond_types[conditioner]
        conditioner_args = {"in_size": dim, "hidden": emb_net[:-1], "out_size": emb_net[-1]}
        if norm_type == 'Monotonic':
            normalizer_args["cond_size"] = emb_net[-1]

        inner_model = buildFCNormalizingFlow(nb_flow[0], conditioner_type, conditioner_args, normalizer_type, normalizer_args)
    model = nn.DataParallel(inner_model, device_ids=list(range(n_gpu))).to(master_device)
    logger.info(str(model))
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    logger.info("Number of parameters: %d" % pytorch_total_params)

    opt = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    if load:
        logger.info("Loading model...")
        model.load_state_dict(torch.load(path + '/model%s.pt' % file_number, map_location={"cuda:0": master_device}))
        model.train()
        opt.load_state_dict(torch.load(path + '/ADAM%s.pt' % file_number, map_location={"cuda:0": master_device}))
        if master_device != "cpu":
            for state in opt.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.cuda()
    logger.info("...Model built.")
    logger.info("Training starts:")

    if load:
        for conditioner in model.module.getConditioners():
            conditioner.alpha = conditioner.getAlpha()

    # ----------------------- Main Loop ------------------------- #
    for epoch in range(nb_epoch):
        ll_tot = 0
        start = timer()
        if train:
            model.to(master_device)
            # ----------------------- Training Loop ------------------------- #
            for batch_idx, (cur_x, target) in enumerate(train_loader):
                cur_x = cur_x.view(batch_size, -1).float().to(master_device)
                for normalizer in model.module.getNormalizers():
                    if type(normalizer) is MonotonicNormalizer:
                        normalizer.nb_steps = nb_steps + torch.randint(0, 10, [1])[0].item()
                z, jac = model(cur_x)
                loss = model.module.loss(z, jac)/(batch_per_optim_step * n_gpu)
                if math.isnan(loss.item()):
                    print("Error Nan in loss")
                    print("Dagness:", model.module.DAGness())
                    exit()
                ll_tot += loss.detach()
                if batch_idx % batch_per_optim_step == 0:
                    opt.zero_grad()

                loss.backward(retain_graph=True)
                if (batch_idx + 1) % batch_per_optim_step == 0:
                    opt.step()

            with torch.no_grad():
                print("Dagness:", model.module.DAGness())

            ll_tot /= (batch_idx + 1)
            torch.cuda.empty_cache()
            model.module.step(epoch, ll_tot)

        else:
            ll_tot = 0.

        # ----------------------- Valid Loop ------------------------- #
        ll_test = 0.
        bpp_test = 0.
        model.to(master_device)
        with torch.no_grad():
            for normalizer in model.module.getNormalizers():
                if type(normalizer) is MonotonicNormalizer:
                    normalizer.nb_steps = 150
            for batch_idx, (cur_x, target) in enumerate(valid_loader):
                cur_x = cur_x.view(batch_size, -1).float().to(master_device)
                z, jac = model(cur_x)
                ll = (model.module.z_log_density(z) + jac)
                ll_test += ll.mean().item()
                bpp_test += compute_bpp(ll, cur_x.view(batch_size, -1).float().to(master_device), alpha).mean().item()
            ll_test /= batch_idx + 1
            bpp_test /= batch_idx + 1
            end = timer()

            dagness = max(model.module.DAGness())
            logger.info(
                "epoch: {:d} - Train loss: {:4f} - Valid log-likelihood: {:4f} - Valid BPP {:4f} - <<DAGness>>: {:4f} "
                "- Elapsed time per epoch {:4f} (seconds)".format(epoch, ll_tot, ll_test, bpp_test, dagness, end - start))
            if model.module.isInvertible() and -ll_test < best_valid_loss:
                logger.info("------- New best validation loss --------")
                torch.save(model.state_dict(), path + '/best_model.pt')
                best_valid_loss = -ll_test
                # Valid loop
                ll_test = 0.
                for batch_idx, (cur_x, target) in enumerate(test_loader):
                    z, jac = model(cur_x.view(batch_size, -1).float().to(master_device))
                    ll = (model.module.z_log_density(z) + jac)
                    ll_test += ll.mean().item()
                    bpp_test += compute_bpp(ll, cur_x.view(batch_size, -1).float().to(master_device), alpha).mean().item()

                ll_test /= batch_idx + 1
                bpp_test /= batch_idx + 1
                logger.info("epoch: {:d} - Test log-likelihood: {:4f} - Test BPP {:4f} - <<DAGness>>: {:4f}".
                            format(epoch, ll_test, bpp_test, dagness))
            if epoch % 10 == 0 and conditioner_type is DAGConditioner:
                stoch_gate, noise_gate, s_thresh = [], [], []
                for conditioner in model.module.getConditioners():
                    stoch_gate.append(conditioner.stoch_gate)
                    noise_gate.append(conditioner.noise_gate)
                    s_thresh.append(conditioner.s_thresh)
                    conditioner.stoch_gate = False
                    conditioner.noise_gate = False
                    conditioner.s_thresh = True
                for threshold in [.95, .5, .1, .01, .0001]:
                    for conditioner in model.module.getConditioners():
                        conditioner.h_thresh = threshold
                    # Valid loop
                    ll_test = 0.
                    bpp_test = 0.
                    for batch_idx, (cur_x, target) in enumerate(valid_loader):
                        cur_x = cur_x.view(batch_size, -1).float().to(master_device)
                        z, jac = model(cur_x)
                        ll = (model.module.z_log_density(z) + jac)
                        ll_test += ll.mean().item()
                        bpp_test += compute_bpp(ll, cur_x.view(batch_size, -1).float().to(master_device), alpha).mean().item()
                    ll_test /= batch_idx + 1
                    bpp_test /= batch_idx + 1
                    dagness = max(model.module.DAGness())
                    logger.info("epoch: {:d} - Threshold: {:4f} - Valid log-likelihood: {:4f} - Valid BPP {:4f} - <<DAGness>>: {:4f}".
                        format(epoch, threshold, ll_test, bpp_test, dagness))
                for i, conditioner in enumerate(model.module.getConditioners()):
                    conditioner.h_thresh = 0.
                    conditioner.stoch_gate = stoch_gate[i]
                    conditioner.noise_gate = noise_gate[i]
                    conditioner.s_thresh = s_thresh[i]




                in_s = 784 if dataset == "MNIST" else 3*32*32
                a_tmp = model.module.getConditioners()[0].soft_thresholded_A()[0, :]
                a_tmp = a_tmp.view(28, 28).cpu().numpy() if dataset == "MNIST" else a_tmp.view(3, 32, 32).cpu().numpy()
                fig, ax = plt.subplots()
                mat = ax.matshow(a_tmp)
                plt.colorbar(mat)
                current_cmap = matplotlib.cm.get_cmap()
                current_cmap.set_bad(color='red')
                mat.set_clim(0, 1.)
                def update(i):
                    A = model.module.getConditioners()[0].soft_thresholded_A()[i, :].cpu().numpy()
                    A[i] = np.nan
                    if dataset == "MNIST":
                        A = A.reshape(28, 28)
                    elif dataset == "CIFAR10":
                        A = A.reshape(3, 32, 32)
                    mat.set_data(A)
                    return mat

                # Set up formatting for the movie files
                Writer = animation.writers['ffmpeg']
                writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
                ani = animation.FuncAnimation(fig, update, range(in_s), interval=100, save_count=0)
                ani.save(path + '/A_epoch_%d.mp4' % epoch, writer=writer)

                deg_out = (model.module.getConditioners()[0].soft_thresholded_A() > 0.).sum(0).cpu().numpy()
                deg_in = (model.module.getConditioners()[0].soft_thresholded_A() > 0.).sum(1).cpu().numpy()
                fig, ax = plt.subplots(1, 2, figsize=(12, 6))
                if dataset == "MNIST":
                    shape = (28, 28)
                elif dataset == "CIFAR10":
                    shape = (3, 32, 32)
                res0 = ax[0].matshow(np.log(deg_in).reshape(shape))
                ax[0].set(title="In degrees")
                fig.colorbar(res0, ax=ax[0])
                res1 = ax[1].matshow(np.log(deg_out.reshape(shape)))
                ax[1].set(title="Out degrees")
                fig.colorbar(res1, ax=ax[1])
                plt.savefig(path + '/A_degrees_epoch_%d.png' % epoch)

            if model.module.isInvertible():
                with torch.no_grad():
                    n_images = 16
                    in_s = 28**2
                    for T in [.1, .25, .5, .75, 1.]:
                        z = torch.randn(n_images, in_s).to(device=master_device) * T
                        x = model.module.invert(z)
                        print((z - model(x)[0]).abs().mean())
                        grid_img = torchvision.utils.make_grid(x.view(n_images, 1, 28, 28), nrow=4)
                        torchvision.utils.save_image(grid_img, path + '/images_%d_%f.png' % (epoch, T))

            if epoch % nb_step_dual == 0:
                logger.info("Saving model N°%d" % epoch)
                torch.save(model.state_dict(), path + '/model_%d.pt' % epoch)
                torch.save(opt.state_dict(), path + '/ADAM_%d.pt' % epoch)

            torch.save(model.state_dict(), path + '/model.pt')
            torch.save(opt.state_dict(), path + '/ADAM.pt')
            torch.cuda.empty_cache()

import argparse

parser = argparse.ArgumentParser(description='')
parser.add_argument("-load", default=False, action="store_true", help="Load a model ?")
parser.add_argument("-folder", default="", help="Folder")
parser.add_argument("-nb_steps_dual", default=100, type=int,
                    help="number of step between updating Acyclicity constraint and sparsity constraint")
parser.add_argument("-l1", default=10., type=float, help="Maximum weight for l1 regularization")
parser.add_argument("-nb_epoch", default=10000, type=int, help="Number of epochs")
parser.add_argument("-b_size", default=1, type=int, help="Batch size")
parser.add_argument("-int_net", default=[50, 50, 50], nargs="+", type=int, help="NN hidden layers of UMNN")
parser.add_argument("-nb_steps", default=20, type=int, help="Number of integration steps.")
parser.add_argument("-f_number", default=None, type=str, help="Number of heating steps.")
parser.add_argument("-solver", default="CC", type=str, help="Which integral solver to use.",
                    choices=["CC", "CCParallel"])
parser.add_argument("-nb_flow", default=[1], nargs="+", type=int, help="Number of steps in the flow.")
parser.add_argument("-test", default=False, action="store_true")
parser.add_argument("-weight_decay", default=1e-5, type=float, help="Weight decay value")
parser.add_argument("-learning_rate", default=1e-3, type=float, help="Weight decay value")
parser.add_argument("-batch_per_optim_step", default=1, type=int, help="Number of batch to accumulate")
parser.add_argument("-nb_gpus", default=1, type=int, help="Number of gpus to train on")
parser.add_argument("-dataset", default="MNIST", type=str, choices=["MNIST", "CIFAR10", "MNIST1"])
parser.add_argument("-normalizer", default="Affine", type=str, choices=["Affine", "Monotonic"])
parser.add_argument("-no_hot_encoding", default=False, action="store_true")
parser.add_argument("-prior_A_kernel", default=None, type=int)

parser.add_argument("-conditioner", default='DAG', choices=['DAG', 'Coupling', 'Autoregressive'], type=str)
parser.add_argument("-emb_net", default=[100, 100, 100, 10], nargs="+", type=int, help="NN layers of embedding")

args = parser.parse_args()
from datetime import datetime
now = datetime.now()

path = args.dataset + "/" + now.strftime("%m_%d_%Y_%H_%M_%S") if args.folder == "" else args.folder
if not (os.path.isdir(path)):
    os.makedirs(path)
train(dataset=args.dataset, load=args.load, path=path, nb_step_dual=args.nb_steps_dual, l1=args.l1, nb_epoch=args.nb_epoch,
      int_net=args.int_net, b_size=args.b_size, all_args=args, nb_flow=args.nb_flow,
      nb_steps=args.nb_steps, file_number=args.f_number, norm_type=args.normalizer,
      solver=args.solver, train=not args.test, weight_decay=args.weight_decay, learning_rate=args.learning_rate,
      batch_per_optim_step=args.batch_per_optim_step, n_gpu=args.nb_gpus, hot_encoding=not args.no_hot_encoding,
      prior_A_kernel=args.prior_A_kernel, conditioner=args.conditioner, emb_net=args.emb_net)
