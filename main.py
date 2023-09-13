from models import ModelWrapper, LinearModelWrapper
from parameters import DEVICE
from utilities import OSUDataloader, l2_loss, mass_conservation_loss, identity_loss, deriv_loss, deriv2_loss
from torch.optim import Adam
import torch
from numpy import loadtxt


def load_data_from_files() -> torch.Tensor:
    dataset = torch.empty((500, 100, 58))
    for i in range(500):
        num = (4 - len(str(i))) * '0' + str(i)
        dataset[i,:,:] = torch.tensor(loadtxt('/export/home/isulzer/ChemicalNeuralODE/outputs/chemistry_' + num + '.dat'))
    return dataset


data = load_data_from_files()
dataloader = OSUDataloader(data,
                           batch_size=256,
                           shuffle=True,
                           drop_last=False)

model = ModelWrapper(latent_vars = 5,                # number of latent dimensions
                     ode_hidden = 5,                 # number of hidden layers in ODE
                     ode_width = 256,                # neurons per hidden layers in ODE
                     width_list = [64, 32, 16, 8],   # neurons per hidden layers in encoder/decoder
                     ).to(DEVICE)

optimizer = Adam(params=model.parameters(),
                 lr=1e-3)                            # learning rate

loss_weights = [100., 1., 1., 1., 1.]                # weights for the different losses

for epoch in range(100):
    for i, (x0, x_true, x_dot) in enumerate(dataloader):
        x_pred = model(x0, torch.linspace(0, 1, 100).to(DEVICE))
        loss = loss_weights[0] * l2_loss(x_true, x_pred) + \
               loss_weights[1] * mass_conservation_loss(x_true, x_pred) + \
               loss_weights[2] * identity_loss(x_true, model) + \
               loss_weights[3] * deriv_loss(x_true, x_pred) + \
               loss_weights[4] * deriv2_loss(x_true, x_pred)
        if epoch == 10 and i == 0:
            # rescale losses
            loss_weights[0] = 1 / l2_loss(x_true, x_pred).item() * 100
            loss_weights[1] = 1 / mass_conservation_loss(x_true, x_pred).item()
            loss_weights[2] = 1 / identity_loss(x_true, model).item()
            loss_weights[3] = 1 / deriv_loss(x_true, x_pred).item()
            loss_weights[4] = 1 / deriv2_loss(x_true, x_pred).item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch}, Loss {loss.item()}")