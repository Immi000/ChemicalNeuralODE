import torch
from parameters import MASSES

def deriv(x):
    return torch.gradient(x, dim=1)[0].squeeze(0)

def deriv2(x):
    return deriv(deriv(x))

def mass_conservation_loss(x_true, x_pred):
    return torch.sum(torch.abs(torch.sum(MASSES * x_true, dim=-1)  - torch.sum(MASSES * x_pred, dim=-1)), dim=-1).mean()


def relative_l2_loss(x_true: torch.tensor, x_pred: torch.tensor):
    return torch.abs((x_pred - x_true) / torch.amax(x_true, dim=1)[:,None,:]).mean()


def l2_loss(y1: torch.tensor, y2: torch.tensor):
    return torch.mean(torch.abs(y2 - y1)**2)

def identity_loss(self, x: torch.tensor, model):
    return self.l2_loss(x, model.decoder(model.encoder(x)))