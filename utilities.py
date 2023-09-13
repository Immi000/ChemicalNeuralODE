import torch
from torch.utils.data import Dataset
from parameters import MASSES, REAL_VARS, DEVICE


def deriv(x: torch.Tensor) -> torch.Tensor:
    return torch.gradient(x, dim=1)[0].squeeze(0)

def deriv2(x: torch.Tensor) -> torch.Tensor:
    return deriv(deriv(x))

def mass_conservation_loss(x_true: torch.Tensor, x_pred: torch.Tensor) -> torch.Tensor:
    return torch.sum(torch.abs(torch.sum(MASSES * x_true, dim=-1)  - torch.sum(MASSES * x_pred, dim=-1)), dim=-1).mean()

def l2_loss(y1: torch.Tensor, y2: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.abs(y2 - y1)**2)

def identity_loss(x: torch.Tensor, model) -> torch.Tensor:
    return l2_loss(x, model.decoder(model.encoder(x)))

def deriv_loss(x_true: torch.Tensor, x_pred: torch.Tensor) -> torch.Tensor:
    return l2_loss(deriv(x_pred), deriv(x_true))

def deriv2_loss(x_true: torch.Tensor, x_pred: torch.Tensor) -> torch.Tensor:
    return l2_loss(deriv2(x_pred), deriv2(x_true))


class OSUDataloader(Dataset):
    def __init__(self,
                 data,
                 batch_size=64,
                 shuffle=False,
                 drop_last=False,
                 device=DEVICE,):
        self.data = data
        self.x, self.xdot = self.data[:,:,:REAL_VARS].to(device), self.data[:,:,REAL_VARS:].to(device)
        self.xmin = self.x.min()
        self.xmax = self.x.max()
        self.x = 2 * (self.x - self.xmin) / (self.xmax - self.xmin) - 1
        self.xdot = (self.xdot - self.xdot.min()) / (self.xdot.max() - self.xdot.min())
        if shuffle:
            perm = torch.randperm(self.x.shape[0])
            self.x, self.xdot = self.x[perm,:,:], self.xdot[perm,:,:]
        self.length = self.x.shape[0]
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __len__(self):
        return self.length // self.batch_size - int(self.drop_last)

    def __getitem__(self, i):
        return self.x[self.batch_size * i:self.batch_size * (i + 1), 0, :], \
            self.x[self.batch_size * i:self.batch_size * (i + 1), :, :], \
            self.xdot[self.batch_size * i:self.batch_size * (i + 1), :, :]

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def reshuffle(self):
        perm = torch.randperm(self.x.shape[0])
        self.x, self.xdot = self.x[perm,:,:], self.xdot[perm,:,:]