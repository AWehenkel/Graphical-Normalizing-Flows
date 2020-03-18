import numpy as np
import torch

class PROTEINS:

    class Data:

        def __init__(self, data):

            self.x = data.astype(np.float32)
            self.N = self.x.shape[0]

    def __init__(self):

        trn, val, tst = load_data()

        self.trn = self.Data(trn)
        self.val = self.Data(val)
        self.tst = self.Data(tst)

        self.n_dims = self.trn.x.shape[1]


def load_data():
    dir_f = "Datasets/Human_Protein_Network/"
    train = torch.load(dir_f + "X_train.pkt")
    mu, sigma = train.mean(0), train.std(0)
    valid = torch.load(dir_f + "X_train.pkt")
    test = None
    return (train - mu)/sigma, (valid - mu)/sigma, (valid - mu)/sigma

