from models import ModelWrapper, LinearModelWrapper
from parameters import DEVICE

model = ModelWrapper(latent_vars = 5,               # number of latent dimensions
                    ode_hidden = 5,                 # number of hidden layers in ODE
                    ode_width = 256,                # neurons per hidden layers in ODE
                    width_list = [64, 32, 16, 8],   # 
                    coder_hidden = 4,               # number of hidden layers in encoder/decoder
                    ).to(DEVICE)