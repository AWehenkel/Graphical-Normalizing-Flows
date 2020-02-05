import torch
from torch.utils.data import Dataset
import sklearn.datasets as datasets
import numpy as np


class CaliforniaHousingDataset(Dataset):
    def __init__(self, kind="train", normalize=True):
        self.kind = kind
        all_data = datasets.fetch_california_housing(data_home=None, download_if_missing=True, return_X_y=False)
        X = np.append(all_data['data'], all_data['target'].reshape(-1, 1), axis=1)
        if kind == "train":
            self.X = X[:15000]
        elif kind == "test":
            self.X = X[15000:18000]
        elif kind == "valid":
            self.X = X[18000:]
        else:
            raise Exception
        if normalize:
            self.mu = self.X.mean(0).reshape(1, -1)
            self.std = self.X.std(0).reshape(1, -1)
            self.normalize(self.mu, self.std)

    def normalize(self, mu, std):
        self.X = (self.X - np.repeat(mu, self.X.shape[0], axis=0)) / np.repeat(std, self.X.shape[0], axis=0)
        return self

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx]

