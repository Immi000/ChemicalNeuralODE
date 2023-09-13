from torch import tensor, device
from torch.cuda import is_available

DEVICE = device('cuda:0' if is_available() else 'cpu')
REAL_VARS = 29
LATENT_VARS = 5

SPECIES = ["C", "C+", "CH", "CH+", "CH2",
           "CH2+", "CH3", "CH3+", "CH4", "CH4+",
           "CH5+", "CO", "CO+", "e", "H",
           "H+", "H2", "H2+", "H2O", "H2O+",
           "H3+", "H3O+", "HCO+", "O", "O+",
           "O2", "O2+", "OH", "OH+"]


# mass of each species in atomic mass units
MASSES = tensor([12.011, 12.011, 13.019, 13.019, 14.027,
                14.027, 15.035, 15.035, 16.043, 16.043,
                17.054, 28.010, 28.010, 0.001, 1.008,
                1.008, 2.016, 2.059, 18.015, 18.015,
                3.024, 19.023, 29.018, 15.999, 15.999,
                31.998, 31.998, 17.007, 17.007]).to(DEVICE)

MODELS_FOLDER = "/export/data/isulzer/Neural-ODEs-Bachelorarbeit/chemical_model/models/"
PLOT_FOLDER = "/export/data/isulzer/Neural-ODEs-Bachelorarbeit/chemical_model/plots/"
