import torch
from torchdiffeq import odeint_adjoint


class NeuralODE(torch.nn.Module):

    def __init__(self,
                 input_shape: int,
                 output_shape: int,
                 activation: torch.nn.modules.activation = torch.nn.ReLU(),
                 n_hidden: int = 8,
                 layer_width: int = 256):
        super().__init__()

        self.reg_factor = torch.nn.Parameter(torch.tensor(1.))

        self.input_shape = input_shape
        self.activation = activation

        self.mlp = torch.nn.Sequential()
        self.mlp.append(torch.nn.Linear(self.input_shape, layer_width))
        self.mlp.append(self.activation)
        for i in range(n_hidden):
            self.mlp.append(torch.nn.Linear(layer_width, layer_width))
            self.mlp.append(self.activation)
        self.mlp.append(torch.nn.Linear(layer_width, output_shape))


    def forward(self, t, x):
        return self.reg_factor * torch.tanh(self.mlp(x) / self.reg_factor)


class Encoder(torch.nn.Module):

    def __init__(self,
                in_features: int = 29,
                latent_features: int = 5,
                n_hidden: int = 4,
                width_list: list = [32, 16, 8],
                activation: torch.nn.modules.activation = torch.nn.Tanh()):
        super().__init__()
        assert n_hidden == len(width_list) + 1, "n_hidden must equal length of width_list"
        self.in_features = in_features
        self.latent_features = latent_features
        self.n_hidden = n_hidden
        self.width_list = width_list
        self.activation = activation

        self.mlp = torch.nn.Sequential()
        self.mlp.append(torch.nn.Linear(self.in_features, self.width_list[0]))
        self.mlp.append(self.activation)
        for i, width in enumerate(self.width_list[1:]):
            self.mlp.append(torch.nn.Linear(self.width_list[i], width))
            self.mlp.append(self.activation)
        self.mlp.append(torch.nn.Linear(self.width_list[-1], self.latent_features))
        self.mlp.append(activation)

    def forward(self, x):
        return self.mlp(x)


class Decoder(torch.nn.Module):
    
        def __init__(self,
                    out_features: int = 29,
                    latent_features: int = 5,
                    n_hidden: int = 4,
                    width_list: list = [32, 16, 8],
                    activation: torch.nn.modules.activation = torch.nn.Tanh()):
            super().__init__()
            assert n_hidden == len(width_list) + 1, "n_hidden must equal length of width_list"
            self.out_features = out_features
            self.latent_features = latent_features
            self.n_hidden = n_hidden
            self.width_list = width_list
            self.activation = activation
            self.width_list.reverse()
    
            self.mlp = torch.nn.Sequential()
            self.mlp.append(torch.nn.Linear(self.latent_features, self.width_list[0]))
            self.mlp.append(self.activation)
            for i, width in enumerate(self.width_list[1:]):
                self.mlp.append(torch.nn.Linear(self.width_list[i], width))
                self.mlp.append(self.activation)
            self.mlp.append(torch.nn.Linear(self.width_list[-1], self.out_features))
            self.mlp.append(activation)

        def forward(self, x):
            return self.mlp(x)


class ModelWrapper(torch.nn.Module):
    def __init__(self,
                 real_vars: int = 29,
                 latent_vars: int = 5,
                 ode_width: int = 256,
                 ode_hidden: int = 8,
                 coder_hidden: int = 4,
                 width_list: list = [32, 16, 8],
                 coder_activation: torch.nn.modules.activation = torch.nn.Tanh()
                 ):
        super().__init__()
        assert coder_hidden == len(width_list) + 1, "coder_hidden must equal length of width_list"
        self.x_vars = real_vars
        self.z_vars = latent_vars
        self.encoder = Encoder(self.x_vars, self.z_vars, n_hidden=coder_hidden, width_list=width_list, activation=coder_activation)
        self.decoder = Decoder(self.x_vars, self.z_vars, n_hidden=coder_hidden, width_list=width_list, activation=coder_activation)
        self.ode = NeuralODE(self.z_vars, self.z_vars, n_hidden=ode_hidden, layer_width=ode_width)
        self.width = ode_width
        self.hidden = ode_hidden

        # cached variables
        self.z_pred = None
        self.x_pred = None
        self.z_dot = None
        self.relative_error = None


    def forward(self, x0: torch.tensor, t_range: torch.tensor) -> torch.tensor:
        self.z_pred = torch.permute(odeint_adjoint(self.ode, self.encoder(x0), t_range, adjoint_rtol=1e-7, adjoint_atol=1e-9, adjoint_method='dopri8'), (1, 0, 2))
        self.z_dot = torch.gradient(self.z_pred, dim=1)[0]
        self.x_pred = self.decoder(self.z_pred)
        return self.x_pred
        

class LinearODE(torch.nn.Module):

    def __init__(self, latent_params: int = 5):
        super().__init__()
        self.slope = torch.nn.Parameter(torch.randn(latent_params))
        self.latent_params = latent_params

    def forward(self, t):
        self.batch_size = t.shape[0]
        return torch.einsum("bi,j->bij", (t, self.slope))


class LinearModelWrapper(ModelWrapper):
    
    def __init__(self, real_vars: int = 29,
                 latent_vars: int = 5,
                 coder_hidden: int = 4,
                 width_list: list = [32, 16, 8],
                 coder_activation: torch.nn.modules.activation = torch.nn.Tanh()
                 ):
        super().__init__(real_vars=real_vars,
                         latent_vars=latent_vars,
                         ode_width=1,
                         ode_hidden=1,
                         coder_hidden=4,
                         width_list=width_list,
                         coder_activation=coder_activation)
        self.ode = LinearODE(latent_vars)

    def forward(self, x0: torch.tensor, t_range: torch.tensor) -> torch.tensor:
        batch_size = x0.shape[0]
        t = t_range.unsqueeze(0).repeat(batch_size, 1)
        self.z_pred = self.ode(t) + self.encoder(x0).unsqueeze(1)
        self.z_dot = self.ode.slope
        self.x_pred = self.decoder(self.z_pred)
        return self.x_pred
    

class PolynomialODE(torch.nn.Module):

    def __init__(self, degree: int = 2, latent_params: int = 5):
        super().__init__()
        self.coef = torch.nn.Linear(degree,latent_params, bias=False, dtype=torch.float32)
        self.degree = degree

    def forward(self, t):
        return self.coef(self.poly(t))
    
    def poly(self, t):
        t = t.unsqueeze(1)
        return torch.hstack([t ** i for i in range(1, self.degree + 1)]).permute(0, 2, 1)
    
    def integrated_poly(self, t):
        t = t.unsqueeze(1)
        return torch.hstack([t ** (i + 1) / (i + 1) for i in range(self.degree)]).permute(0, 2, 1)

class PolynomialModelWrapper(ModelWrapper):

    def __init__(self, real_vars: int = 29,
                 latent_vars: int = 5,
                 degree: int = 2,
                 coder_hidden: int = 4,
                 width_list: list = [32, 16, 8],
                 coder_activation: torch.nn.modules.activation = torch.nn.Tanh()
                 ):
        super().__init__(real_vars, latent_vars, 1, 1, None, coder_hidden, width_list, coder_activation)
        self.ode = PolynomialODE(degree, latent_vars)

    def forward(self, x0: torch.tensor, t_range: torch.tensor) -> torch.tensor:
        batch_size = x0.shape[0]
        t = t_range.unsqueeze(0).repeat(batch_size, 1)
        self.z_pred = self.ode(t) + self.encoder(x0).unsqueeze(1)
        self.x_pred = self.decoder(self.z_pred)
        return self.x_pred