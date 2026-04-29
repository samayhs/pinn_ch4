import cantera as ct
from scripts.helpers.StateNet1 import *
import torch
import matplotlib.pyplot as plt
import numpy as np
from itertools import product
import os
import winsound

"""
Cantera Basics
"""

def run_cantera_sim(gas, T, P, phi):

    gas.TP = T, P*ct.one_atm
    gas.set_equivalence_ratio(phi, "CH4", "O2:1.0, N2:3.76")

    r = ct.ConstPressureReactor(gas)
    sim = ct.ReactorNet([r])


    states = ct.SolutionArray(gas)
    while sim.time < 1e-3:
        sim.advance(sim.time + 1e-6)
        states.append(r.phase.state)
    
    return states

'''
Obtain CSVs from Cantera Sims
'''

def get_input_data_from_states(states: ct.SolutionArray, save_csv=False,csv_path=f'results/training_data.csv'):
    N = len(states.T)
    T = states.T.reshape(-1, 1)
    P = states.P.reshape(-1, 1)
    Y = states.Y
    dt = np.full((N, 1), 1e-6)
    output = np.concatenate((T, P, Y, dt), axis=1)
    if save_csv:
        
        np.savetxt(csv_path, output, delimiter=',')
    return torch.tensor(output, dtype=torch.float32)

def get_omega_from_states(states: ct.SolutionArray, save_csv=False,csv_path=f'results/omega_data.csv'):

    omega = states.net_production_rates
    if save_csv:
        
        np.savetxt(csv_path, omega, delimiter=',')
    return torch.tensor(omega, dtype=torch.float32)
'''
Model Training Helper Functions
'''
def generate_training_data(gas):
    """
    Sample random initial conditions as state vectors [T, P, Y_1...Y_K, dt].
    Loads from CSV if available, otherwise runs Cantera and saves.
    """

    T_list = [1500, 1600, 1700]
    P_list = [1]
    phi_list = [0.7, 1, 1.3]

    full_state_data = []
    full_omega_data = []
    for T, P, phi in product(T_list, P_list, phi_list):
        state_path = f"results/states_T={T}K, P={P}atm, phi={phi}"
        omega_path = f"results/omega_T={T}K, P={P}atm, phi={phi}"

        if os.path.exists(state_path) and os.path.exists(omega_path):
            print(f"Loading from CSV: T={T}K, P={P}atm, phi={phi}")
            data = torch.tensor(np.loadtxt(state_path, delimiter=','), dtype=torch.float32)
            omega_data = torch.tensor(np.loadtxt(omega_path, delimiter=','), dtype=torch.float32)
        else:
            print(f"Running Cantera: T={T}K, P={P}atm, phi={phi}")
            states = run_cantera_sim(gas, T, P, phi)
            data = get_input_data_from_states(states, save_csv=True, csv_path=state_path)
            omega_data = get_omega_from_states(states, save_csv=True, csv_path=omega_path)

        full_state_data.append(data)
        full_omega_data.append(omega_data)
    return full_state_data, full_omega_data

"""
Validate Model
"""

def validate_model_hk(gas: ct.Solution, T: torch.Tensor):
    model = StateNet(gas)
    hk_model = model.hk(T).detach().numpy()  # (N, K) molar enthalpies in J/kmol

    species_names = gas.species_names
    n_sp = gas.n_species
    hk_cantera = np.zeros((len(T), n_sp))
    saved_TP = gas.T, gas.P
    for i, temp in enumerate(T):
        gas.TP = temp.item(), saved_TP[1]
        hk_cantera[i, :] = gas.standard_enthalpies_RT * 8314.0 * temp.item()  # J/kmol
    gas.TP = saved_TP

    T_np = T.detach().numpy()
    n_plot = min(6, n_sp)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for k, ax in enumerate(axes.flatten()[:n_plot]):
        ax.plot(T_np, hk_cantera[:, k] / 1e6, 'k-', label='Cantera')
        ax.plot(T_np, hk_model[:, k] / 1e6, 'r--', label='Model')
        ax.set_xlabel('T [K]')
        ax.set_ylabel('$h_k$ [MJ/kmol]')
        ax.set_title(species_names[k])
        ax.legend()
    plt.tight_layout()
    plt.savefig('results/validate_hk.png', dpi=150)
    plt.show()


'''
Train and Evaluate Model

'''

def train(model, state_list, omega_list, n_epochs = 100000):

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Concatenate all conditions into single tensors
    states0 = torch.cat(state_list, dim=0)   # (N_total, 2+K+1)
    omega0 = torch.cat(omega_list, dim=0)    # (N_total, K)
    print(f"Training data: {states0.shape[0]} samples, {states0.shape[1]} features")

    
    loss_history = []

    for epoch in range(n_epochs):
        optimizer.zero_grad()

        idx = torch.randint(0, states0.shape[0], (64,))
        batch_states = states0[idx]
        batch_omega = omega0[idx]

        if epoch % 100 == 0:
            verbose = True
        else: 
            verbose = False
        loss = loss_fn(model, batch_states, batch_omega, verbose=verbose)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        loss_history.append(loss.item())

        if epoch % 100 == 0:
            print(f"Epoch {epoch:4d} | Loss: {loss.item():.4e}")

    # Plot loss
    fig, ax = plt.subplots()
    ax.semilogy(loss_history)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('StateNet Training Loss')
    plt.savefig('results/training_loss.png')
    plt.show()

    torch.save(model.state_dict(), 'results/statenet.pt')
    print("Model saved to results/statenet.pt")

def evaluate(model, gas):
    """Run the trained model autoregressively and compare to Cantera."""
    T0, P0, phi = 1600, 1, 1.0
    gas.TP = T0, P0 * ct.one_atm
    gas.set_equivalence_ratio(phi, "CH4", "O2:1.0, N2:3.76")

    dt = 1e-6
    n_steps = 1000

    # Cantera reference
    r = ct.ConstPressureReactor(gas)
    sim = ct.ReactorNet([r])
    ct_T = []
    for _ in range(n_steps):
        sim.advance(sim.time + dt)
        ct_T.append(r.thermo.T)

    # Model prediction (autoregressive)
    gas.TP = T0, P0 * ct.one_atm
    gas.set_equivalence_ratio(phi, "CH4", "O2:1.0, N2:3.76")
    state = torch.tensor([[gas.T, gas.P, *gas.Y, dt]], dtype=torch.float32)

    model.eval()
    pred_T = []
    with torch.no_grad():
        for _ in range(n_steps):
            state = model(state)
            pred_T.append(state[0, 0].item())

    t = np.arange(1, n_steps + 1) * dt * 1e3  # ms

    fig, ax = plt.subplots()
    ax.plot(t, ct_T, label="Cantera")
    ax.plot(t, pred_T, '--', label="StateNet")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Temperature (K)")
    ax.set_title(f"T={T0}K, P={P0}atm, phi={phi}")
    ax.legend()
    plt.savefig("results/eval_temperature.png")
    plt.show()


if __name__ == "__main__":
    gas = ct.Solution("mechanisms/FFCM2.yaml")
    
    a,b = generate_training_data(gas)
    print(a[1].shape)
    print(b[1].shape)
    print(len(a[1]))














    winsound.Beep(800, 300)
    winsound.Beep(1000, 300)
    winsound.Beep(1200, 500)
    


    
