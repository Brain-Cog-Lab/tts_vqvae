import torch
import torch.nn as nn


class SnnEncoder(nn.Module):

    def __init__(self, method="direct", time_step=4):
        super().__init__()
        self.time_step = time_step
        if method == "direct":
            self.encoder = self.direct
        elif method == 'dvs':
            self.encoder = self.dvs

    def direct(self, x):
        return torch.stack([x] * self.time_step, dim=0)

    def dvs(self, x):
        """
        x: dvs data batch, shape (b,t,2,h,w)
        """
        return torch.transpose(x, 0, 1)  # (t,b,2,h,w)

    @torch.no_grad()
    def forward(self, x):
        return self.encoder(x)
