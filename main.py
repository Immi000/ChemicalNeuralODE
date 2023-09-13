from models import ModelWrapper, LinearModelWrapper
from parameters import DEVICE

model = ModelWrapper(latent_vars = 5,       # number of latent dimensions
                    ode_hidden = 5,         # number of hidden layers in ODE
                    ode_width = 256,        # neurons per hidden layers in ODE
                    width_list=coder_width_list,
                    coder_hidden=coder_hidden_layers
                    ).to(DEVICE)