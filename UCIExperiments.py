from timeit import default_timer as timer
import lib.utils as utils
from datetime import datetime
import yaml
import os
import UCIdatasets
import numpy as np
from models.Normalizers import *
from models.Conditionners import *
from models.NormalizingFlowFactories import buildFCNormalizingFlow
from models.NormalizingFlow import *
import math
import re

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
    elif name == "proteins":
        return UCIdatasets.PROTEINS()
    else:
        raise ValueError('Unknown dataset')


cond_types = {"DAG": DAGConditioner, "Coupling": CouplingConditioner, "Autoregressive": AutoregressiveConditioner}
norm_types = {"affine": AffineNormalizer, "monotonic": MonotonicNormalizer}


def train(dataset="POWER", load=True, nb_step_dual=100, nb_steps=20, path="", l1=.1, nb_epoch=10000,
          int_net=[200, 200, 200], emb_net=[200, 200, 200], b_size=100, all_args=None, file_number=None, train=True,
          solver="CC", nb_flow=1, weight_decay=1e-5, learning_rate=1e-3, cond_type='DAG', norm_type='affine'):
    logger = utils.get_logger(logpath=os.path.join(path, 'logs'), filepath=os.path.abspath(__file__))
    logger.info(str(all_args))

    logger.info("Creating model...")

    device = "cpu" if not(torch.cuda.is_available()) else "cuda:0"

    if load:
        #train = False
        file_number = "_" + file_number if file_number is not None else ""

    batch_size = b_size

    logger.info("Loading data...")
    data = load_data(dataset)
    data.trn.x = torch.from_numpy(data.trn.x).to(device)
    data.val.x = torch.from_numpy(data.val.x).to(device)
    data.tst.x = torch.from_numpy(data.tst.x).to(device)
    logger.info("Data loaded.")

    dim = data.trn.x.shape[1]
    conditioner_type = cond_types[cond_type]
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

    model = buildFCNormalizingFlow(nb_flow, conditioner_type, conditioner_args, normalizer_type, normalizer_args)
    best_valid_loss = np.inf

    opt = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    if load:
        logger.info("Loading model...")
        model.load_state_dict(torch.load(path + '/model%s.pt' % file_number, map_location={"cuda:0": device}))
        model.train()
        if os.path.isfile(path + '/ADAM%s.pt'):
            opt.load_state_dict(torch.load(path + '/ADAM%s.pt' % file_number, map_location={"cuda:0": device}))
            if device != "cpu":
                for state in opt.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.cuda()

    #x = data.trn.x[:20]
    #print(x, model(x))
    #exit()

    for epoch in range(nb_epoch):
        ll_tot = 0
        start = timer()

        # Update constraints
        if conditioner_type is DAGConditioner:
            with torch.no_grad():
                for conditioner in model.getConditioners():
                    conditioner.constrainA(zero_threshold=0.)

        # Training loop
        model.to(device)
        if train:
            for i, cur_x in enumerate(batch_iter(data.trn.x, shuffle=True, batch_size=batch_size)):
                if normalizer_type is MonotonicNormalizer:
                    for normalizer in model.getNormalizers():
                        normalizer.nb_steps = nb_steps + torch.randint(0, 10, [1])[0].item()
                z, jac = model(cur_x)
                #print(z.mean(), jac.mean())
                loss = model.loss(z, jac)
                if math.isnan(loss.item()) or math.isinf(loss.abs().item()):
                    torch.save(model.state_dict(), path + '/NANmodel.pt')
                    print("Error NAN in loss")
                    exit()
                ll_tot += loss.detach()
                opt.zero_grad()
                loss.backward(retain_graph=True)
                opt.step()

            ll_tot /= i + 1
            model.step(epoch, ll_tot)
        else:
            ll_tot = 0.


        # Valid loop
        ll_test = 0.
        with torch.no_grad():
            if normalizer_type is MonotonicNormalizer:
                for normalizer in model.getNormalizers():
                    normalizer.nb_steps = nb_steps + 20
            for i, cur_x in enumerate(batch_iter(data.val.x, shuffle=True, batch_size=batch_size)):
                z, jac = model(cur_x)
                ll = (model.z_log_density(z) + jac)
                ll_test += ll.mean().item()
            ll_test /= i + 1

            end = timer()
            dagness = max(model.DAGness())
            logger.info("epoch: {:d} - Train loss: {:4f} - Valid log-likelihood: {:4f} - <<DAGness>>: {:4f} - Elapsed time per epoch {:4f} (seconds)".
                        format(epoch, ll_tot.item(), ll_test, dagness, end-start))

            if dagness < 1e-20 and -ll_test < best_valid_loss:
                logger.info("------- New best validation loss --------")
                torch.save(model.state_dict(), path + '/best_model.pt')
                best_valid_loss = -ll_test
                # Valid loop
                ll_test = 0.
                for i, cur_x in enumerate(batch_iter(data.tst.x, shuffle=True, batch_size=batch_size)):
                    z, jac = model(cur_x)
                    ll = (model.z_log_density(z) + jac)
                    ll_test += ll.mean().item()
                ll_test /= i + 1

                logger.info("epoch: {:d} - Test log-likelihood: {:4f} - <<DAGness>>: {:4f}".format(epoch, ll_test,
                                                                                                   dagness))
            if epoch % 10 == 0 and conditioner_type is DAGConditioner:
                stoch_gate, noise_gate, s_thresh = [], [], []

                for conditioner in model.getConditioners():
                    stoch_gate.append(conditioner.stoch_gate)
                    noise_gate.append(conditioner.noise_gate)
                    s_thresh.append(conditioner.s_thresh)
                    conditioner.stoch_gate = False
                    conditioner.noise_gate = False
                    conditioner.s_thresh = True
                for threshold in [.95, .5, .1, .01, .0001]:
                    for conditioner in model.getConditioners():
                        conditioner.h_thresh = threshold
                    # Valid loop
                    ll_test = 0.
                    for i, cur_x in enumerate(batch_iter(data.val.x, shuffle=True, batch_size=batch_size)):
                        z, jac = model(cur_x)
                        ll = (model.z_log_density(z) + jac)
                        ll_test += ll.mean().item()
                    ll_test /= i
                    dagness = max(model.DAGness())
                    logger.info("epoch: {:d} - Threshold: {:4f} - Valid log-likelihood: {:4f} - <<DAGness>>: {:4f}".
                                format(epoch, threshold, ll_test, dagness))


                for i, conditioner in enumerate(model.getConditioners()):
                    conditioner.h_thresh = threshold
                    conditioner.stoch_gate = stoch_gate[i]
                    conditioner.noise_gate = noise_gate[i]
                    conditioner.s_thresh = s_thresh[i]

            torch.save(model.state_dict(), path + '/model_%d.pt' % epoch)
            torch.save(opt.state_dict(), path + '/ADAM_%d.pt' % epoch)
            if dataset == "proteins" and conditioner_type is DAGConditioner:
                torch.save(model.getConditioners[0].soft_thresholded_A().detach().cpu(), path + '/A_%d.pt' % epoch)

        torch.save(model.state_dict(), path + '/model.pt')
        torch.save(opt.state_dict(), path + '/ADAM.pt')

import argparse
datasets = ["power", "gas", "bsds300", "miniboone", "hepmass", "digits", "proteins"]

parser = argparse.ArgumentParser(description='')
parser.add_argument("-load_config", default=None, type=str)
# General Parameters
parser.add_argument("-dataset", default=None, choices=datasets, help="Which toy problem ?")
parser.add_argument("-load", default=False, action="store_true", help="Load a model ?")
parser.add_argument("-folder", default="", help="Folder")
parser.add_argument("-f_number", default=None, type=str, help="Number of heating steps.")
parser.add_argument("-test", default=False, action="store_true")
parser.add_argument("-nb_flow", type=int, default=1, help="Number of steps in the flow.")

# Optim Parameters
parser.add_argument("-weight_decay", default=1e-5, type=float, help="Weight decay value")
parser.add_argument("-learning_rate", default=1e-3, type=float, help="Weight decay value")
parser.add_argument("-nb_epoch", default=10000, type=int, help="Number of epochs")
parser.add_argument("-b_size", default=100, type=int, help="Batch size")

# Conditioner Parameters
parser.add_argument("-conditioner", default='DAG', choices=['DAG', 'Coupling', 'Autoregressive'], type=str)
parser.add_argument("-emb_net", default=[100, 100, 100, 10], nargs="+", type=int, help="NN layers of embedding")
    # Specific for DAG:
parser.add_argument("-nb_steps_dual", default=100, type=int, help="number of step between updating Acyclicity constraint and sparsity constraint")
parser.add_argument("-l1", default=.2, type=float, help="Maximum weight for l1 regularization")
parser.add_argument("-gumble_T", default=1., type=float, help="Temperature of the gumble distribution.")

# Normalizer Parameters
parser.add_argument("-normalizer", default='affine', choices=['affine', 'monotonic'], type=str)
parser.add_argument("-int_net", default=[100, 100, 100, 100], nargs="+", type=int, help="NN hidden layers of UMNN")
parser.add_argument("-nb_steps", default=20, type=int, help="Number of integration steps.")
parser.add_argument("-solver", default="CC", type=str, help="Which integral solver to use.",
                    choices=["CC", "CCParallel"])

args = parser.parse_args()

now = datetime.now()
loader = yaml.SafeLoader
loader.add_implicit_resolver(
    u'tag:yaml.org,2002:float',
    re.compile(u'''^(?:
     [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
    |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
    |\\.[0-9_]+(?:[eE][-+][0-9]+)?
    |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
    |[-+]?\\.(?:inf|Inf|INF)
    |\\.(?:nan|NaN|NAN))$''', re.X),
    list(u'-+0123456789.'))
if args.load_config is not None:
    with open("UCIExperimentsConfigurations.yml", 'r') as stream:
        try:
            configs = yaml.load(stream, Loader=loader)[args.load_config]
            for key, val in configs.items():
                setattr(args, key, val)
        except yaml.YAMLError as exc:
            print(exc)


dir_name = args.dataset if args.load_config is None else args.load_config
path = "UCIExperiments/" + dir_name + "/" + now.strftime("%m_%d_%Y_%H_%M_%S") if args.folder == "" else args.folder
if not(os.path.isdir(path)):
    os.makedirs(path)
train(args.dataset, load=args.load, path=path, nb_step_dual=args.nb_steps_dual, l1=args.l1, nb_epoch=args.nb_epoch,
      int_net=args.int_net, emb_net=args.emb_net, b_size=args.b_size, all_args=args,
      nb_steps=args.nb_steps, file_number=args.f_number,  solver=args.solver, nb_flow=args.nb_flow,
      train=not args.test, weight_decay=args.weight_decay, learning_rate=args.learning_rate,
      cond_type=args.conditioner,  norm_type=args.normalizer)
