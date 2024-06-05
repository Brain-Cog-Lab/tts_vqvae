import torch
import torch.nn as nn
import torch.nn.functional as F


class SnnDecoder(nn.Module):

    def __init__(self, method):
        super().__init__()
        self.decoder = getattr(self, method)

    def sum_sigmoid(self, x):
        """
        sum with temporal and sigmoid
        :param x: (ST,...)
        :return:
        """
        return F.sigmoid(torch.sum(x, 0))

    def mean_sigmoid(self, x):
        return F.sigmoid(torch.mean(x, 0))

    def mean(self, x):
        return torch.mean(x, 0)

    def select_last(self, x):
        return x[-1]

    def dvs(self, x):
        """
        x: (t,b,2,h,w) => (b,t,2,h,w)
        """
        return torch.transpose(x, 0, 1)

    def forward(self, x):
        return self.decoder(x)
