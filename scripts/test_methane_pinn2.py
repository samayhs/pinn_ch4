from helpers.methane_pinn2 import MethanePinn2, loss_fn
import cantera as ct
from scripts.helpers.StateNet1 import SolutionNet, ChemNet
import torch
import matplotlib.pyplot as plt



def test_cantera_sim():
    gas = ct.Solution('./FFCM2.yaml')
    gas.TP = 1000, ct.one_atm
    gas.set_equivalence_ratio(1.0, 'CH4', 'O2:2 N2:7.52')


    model = MethanePinn2(gas.n_species)
    loss = loss_fn(model, gas)
    print(f"Initial loss: {loss.item():.4e}")
    optimizer = torch.optim.Adam(model.parameters(), weight_decay = 1e-7, lr=1e-4)

    for i in range(1000):
        optimizer.zero_grad()
        loss = loss_fn(model, gas)
        loss.backward()
        optimizer.step()
        if i % 100 == 0:
            print(f"Epoch {i} loss: {loss.item():.4e}")

def test_pinn():
    gas = ct.Solution("mechanisms/FFCM2.yaml")
    T0 = 1200
    gas.TP = T0, ct.one_atm
    gas.set_equivalence_ratio(phi=1.0, fuel="CH4", oxidizer="O2:1.0, N2:3.76")
    Y0 = torch.tensor(gas.Y, dtype=torch.float32)

    sol_net = SolutionNet(n_species=gas.n_species)
    chem_net = ChemNet(n_species=gas.n_species)
    sol_net.set_ic(T0, Y0)
    t = torch.linspace(0, 0.01, 100).unsqueeze(1)  # Time from 0 to 10 ms
    print(t.shape)

    try:
        state = sol_net(t)
        omega = chem_net(state)
    except Exception as e:
        print(f"Error occurred: {e}")

    print(state.shape)

    fig, ax = plt.subplots()
    ax.plot(t.detach().numpy(), state[:,0].detach().numpy())
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Temperature (K)')
    ax.set_title('Predicted Temperature Evolution')
    plt.show()

if __name__ == "__main__":

    test_pinn()