import torch.nn as nn
import torch
import numpy as np
import cantera as ct

COLORS = {
    'BLUE': "\033[94m",
    'ORANGE': "\033[93m",
    'ENDC': '\033[0m',
}
class MethanePinn2(nn.Module):
    def __init__(self, n_species):

        super(MethanePinn2, self).__init__()

        self.n_species = n_species
        self.n_out = self.n_species + 2 # T and P

        self.net = nn.Sequential(
            nn.Linear(self.n_out, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, self.n_out)
        )
    
    def forward(self, state):
        return self.net(state)

def loss_fn(model, gas):

    state = np.array([gas.T, gas.P, *gas.X])
    state = torch.tensor(state, dtype=torch.float32)
    pred_state = model(state)

    r = ct.ConstPressureReactor(gas)
    sim = ct.ReactorNet([r])
    sim.advance(1e-4)
    true_state = torch.tensor(np.array([r.thermo.T, r.thermo.P, *r.thermo.X]), dtype=torch.float32)

    loss = torch.mean((pred_state - true_state) ** 2)
    return loss
