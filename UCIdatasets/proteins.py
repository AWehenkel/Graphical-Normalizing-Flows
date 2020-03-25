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

        self.A = get_adj_matrix().T


def get_shd(A):
    A_true = get_adj_matrix()
    return np.abs(A - A_true).sum(), np.abs(A - A_true.T).sum()


def get_adj_matrix():
    A = np.zeros((11, 11))
    # PKC Children
    A[8, 1] = 1
    A[8, 9] = 1
    A[8, 10] = 1
    A[8, 7] = 1
    A[8, 0] = 1
    # PKA Children
    A[7, 1] = 1
    A[7, 9] = 1
    A[7, 10] = 1
    A[7, 5] = 1
    A[7, 0] = 1
    A[7, 6] = 1
    # RAF Child
    A[0, 1] = 1
    # MEK Child
    A[1, 5] = 1
    # P44/42 Child
    A[5, 6] = 1
    #PlcGamma Children
    A[2, 3] = 1
    A[2, 4] = 1
    A[2, 8] = 1
    #PIP3 Children
    A[4, 3] = 1
    A[4, 6] = 1
    #PIP2 Child
    A[3, 8] = 1
    return A




def load_data():
    dir_f = "Datasets/Human_Protein_Network/"
    train = torch.load(dir_f + "X_train.pkt")
    mu, sigma = train.mean(0), train.std(0)
    valid = torch.load(dir_f + "X_train.pkt")
    test = None
    return (train - mu)/sigma, (valid - mu)/sigma, (valid - mu)/sigma

