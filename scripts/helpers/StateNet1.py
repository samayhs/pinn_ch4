"""
PINN for Constant-Volume Homogeneous Reactor
=============================================
Governing equations (from Homo_ODE_version4.py):

    dT/dt  = -( sum_k[ u_k * omega_k ] ) / (rho * cv)
    dYk/dt = omega_k * Wk / rho,   k = 1 ... K

State vector:  y = [T, Y_1, ..., Y_K]
Parameters:    rho (fixed), Wk (fixed molecular weights)

Strategy
--------
- A single network `SolutionNet` maps t -> [T, Y_1, ..., Y_K].
- A second network `ChemNet` maps [T, Y_1, ..., Y_K] -> [omega_1, ..., omega_K]
  (net molar production rates, learned jointly).
- Thermodynamic properties cv and u_k are provided via simple polynomial
  fits (NASA-7 style) or by wrapping Cantera calls during data generation.
- The total loss has three terms:
    L_ic   : initial condition residuals
    L_phys : PDE residuals (the ODEs above)
    L_data : supervised fit to Cantera simulation data (optional but recommended)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Callable
import cantera as ct

# ---------------------------------------------------------------------------
# 1.  Utility: hard-coded initial-condition enforcement via output transform
#     y_net(t) = y0 + t * NN(t)   ensures y(0) = y0 exactly.
# ---------------------------------------------------------------------------

class SolutionNet(nn.Module):
    """
    Maps t -> [T, Y_1, ..., Y_K].
    Uses output transform so IC is satisfied exactly:
        output(t) = y0 + t * raw_net(t)
    """
    def __init__(self, n_species: int, hidden: int = 16, depth: int = 5):
        super().__init__()
        self.n_species = n_species
        self.n_out = 1 + n_species   # T + K species

        layers = [nn.Linear(1, hidden), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        layers.append(nn.Linear(hidden, self.n_out))
        self.net = nn.Sequential(*layers)

        # y0 will be set before training via register_buffer
        self.register_buffer("y0", torch.zeros(self.n_out))

    def set_ic(self, T0: float, Y0: torch.Tensor):
        """
        Sets the initial condition for the solution network.

        T0: initial temperature [K]
        Y0: tensor of shape (K,) containing initial mass fractions
        """
        self.y0 = torch.cat([torch.tensor([T0]), Y0]).to(self.y0.device)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        t: (N, 1)
        returns: (N, 1+K)
        """
        raw = self.net(t)
        return self.y0.unsqueeze(0) + t * raw   # IC-satisfying transform
    


class StateNet(nn.Module):
    """
    Maps state0 -> state(dt) for a time step dt.
    state: [T, P, Y_1, ..., Y_K, dt]
    
    """
    def __init__(self, gas, hidden: int = 128, depth: int = 4):
        super().__init__()
        n_species = gas.n_species
        # self.gas = gas

        self.Wk = torch.tensor(gas.molecular_weights, dtype=torch.float32)

        self.T0, self.P0 = gas.T, gas.P
        state_len = 2 + n_species + 1  # T, P, Y_1...Y_K, dt

        coeffs_low  = []
        coeffs_high = []
        T_mids      = []
        for s in gas.species():
            T_mids.append(s.thermo.coeffs[0])
            coeffs_high.append(s.thermo.coeffs[1:8])
            coeffs_low.append(s.thermo.coeffs[8:15])

        self.register_buffer("coeffs_low", torch.tensor(np.array(coeffs_low), dtype=torch.float32))   # (K, 7)
        self.register_buffer("coeffs_high", torch.tensor(np.array(coeffs_high), dtype=torch.float32)) # (K, 7)
        self.register_buffer("T_mids", torch.tensor(T_mids, dtype=torch.float32))                     # (K,)

        self.chemnet = ChemNet(n_species, hidden, depth)

        layers = [nn.Linear(state_len, hidden), nn.LayerNorm(hidden), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.Tanh()]
        layers.append(nn.Linear(hidden, state_len))
        self.net = nn.Sequential(*layers)

    def cp(self, T: torch.Tensor, Y: torch.Tensor):
        """
        Compute mixture cp at N temperatures with N compositions.
        Returns mass-based mixture cp in J/(kg·K).

        T: (N,) or (N,1)
        Y: (N, K)
        """
        R = 8314.0  # J/(kmol·K)
        T = T  # (N,)

        # Select NASA-7 coefficients based on T vs T_mid for each species
        # mask: (N, K) — True where T > T_mid
        mask = T.unsqueeze(1) > self.T_mids.unsqueeze(0)  # (N,K)

        # coeffs shape: (K, 7), we only need a1-a5 for cp/R
        a = torch.where(
            mask.unsqueeze(2),               # (N, K, 1)
            self.coeffs_high.unsqueeze(0),   # (1, K, 7)
            self.coeffs_low.unsqueeze(0)     # (1, K, 7)
        )  # (N, K, 7)

        # cp_k/R = a1 + a2*T + a3*T^2 + a4*T^3 + a5*T^4
        T2 = T * T
        T3 = T2 * T
        T4 = T3 * T
        # T_powers: (N, 5)
        T_powers = torch.stack([torch.ones_like(T), T, T2, T3, T4], dim=1)

        # cp_k/R for each species: (N, K)
        cp_over_R = torch.einsum('nkj,nj->nk', a[:, :, :5], T_powers)

        # Convert to mass-based cp_k [J/(kg·K)]: cp_k = cp_over_R * R / Wk
        cp_k = cp_over_R * R / self.Wk.unsqueeze(0)  # (N, K)

        # Mixture cp = sum(Y_k * cp_k)
        cp_mix = (Y * cp_k).sum(dim=1)  # (N,)
        return cp_mix

    def hk(self, T: torch.Tensor):
        """
        Compute per-species enthalpies h_k in J/kmol for each species.
        Uses NASA-7: h_k/(R*T) = a1 + a2*T/2 + a3*T^2/3 + a4*T^3/4 + a5*T^4/5 + a6/T

        T: (N,) or (N,1)
        Returns: (N, K) molar enthalpies in J/kmol
        """
        R = 8314.0  # J/(kmol·K)
        T = T.squeeze(-1)  # (N,)

        mask = T.unsqueeze(1) > self.T_mids.unsqueeze(0)  # (N, K)

        a = torch.where(
            mask.unsqueeze(2),
            self.coeffs_high.unsqueeze(0),
            self.coeffs_low.unsqueeze(0)
        )  # (N, K, 7)

        # h_k/(R*T) = a1 + a2*T/2 + a3*T^2/3 + a4*T^3/4 + a5*T^4/5 + a6/T
        T2 = T * T
        T3 = T2 * T
        T4 = T3 * T
        T_inv = 1.0 / T

        # (N, 6) — coefficients multiply [1, T/2, T^2/3, T^3/4, T^4/5, 1/T]
        T_powers = torch.stack([
            torch.ones_like(T), T / 2, T2 / 3, T3 / 4, T4 / 5, T_inv
        ], dim=1)

        # h_k/(R*T) for each species: (N, K)
        h_over_RT = torch.einsum('nkj,nj->nk', a[:, :, :6], T_powers)

        # Convert to molar h_k [J/kmol]: h_k = h_over_RT * R * T
        h_k = h_over_RT * R * T.unsqueeze(1)  # (N, K)
        return h_k

    def forward(self, state0: torch.Tensor) -> torch.Tensor:
        """
        state0: (N, 2+K+1)  — concatenation of T, P and mass fractions at time t
        returns: (N, 2+K+1) — predicted state at time t+dt
        """
        return state0 + self.net(state0)


class ChemNet(nn.Module):
    """
    Maps [T, P, Y_1, ..., Y_K] -> [omega_1, ..., omega_K]
    (net molar production rates in kmol/m^3/s).

    Positivity is NOT enforced here because omega_k can be negative
    (consumption). If you want to split into production/consumption,
    you can use two softplus heads.
    """
    def __init__(self, n_species: int, hidden: int = 128, depth: int = 4):
        super().__init__()
        layers = [nn.Linear(2 + n_species + 1, hidden), nn.LayerNorm(hidden), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.Tanh()]
        layers.append(nn.Linear(hidden, n_species))
        self.net = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        state: (N, 2+K+1)  — concatenation of T, P, mass fractions, and dt
        returns: (N, K)
        """
        return self.net(state)


def loss_fn(model: nn.Module, states0: torch.Tensor, ct_omega: torch.tensor, verbose=False):

    """
    Computes the total loss for the PINN, including:

    """
    # Predicted Data
    states = model(states0) # (N, 2+K+1)  — predicted state at time t+dt
    omega_pred = model.chemnet(states)  # (N, K) predicted net molar production rates

    # 0 Data
    T0 = states0[:, 0]  # (N,)
    P0 = states0[:, 1]  # (N,)
    Yk0 = states0[:, 2:-1] # (N, K) mass fractions at time t
    dt = states0[:, -1].unsqueeze(1)  # (N, 1) time step

    # 1 Data
    T1 = states[:, 0]  # (N,)
    P1 = states[:, 1]  # (N,)
    Yk1 = states[:, 2:-1] # (N, K) mass fractions at time t+dt
    
    # LHS data
    dTdt = (T1 - T0)   # (N,) finite difference approximation of dT/dt
    dYkdt = (Yk1 - Yk0)  # (N, K) finite difference approximation of dYk/dt

    # OBTAIN RHO FROM IDEAL GAS LAW
    Yk = states[:, 2:-1] # (N, K) mass fractions
    W_mix_inv = (Yk / model.Wk).sum(dim=1)  # (N,) inverse mean molecular weight
    W_mix = 1.0 / W_mix_inv  # (N,) mean molecular weight
    R = 8314  # J/(kmol*K)
    R_mix = R / W_mix  # (N,) mixture-specific gas constant

    rho = states[:, 1] * W_mix / (R * states[:, 0])  # (N,) density from ideal gas law

    # Yk residual
    rhs_yk = (omega_pred * model.Wk / rho.unsqueeze(1))*dt  # (N, K) RHS of dYk/dt
    loss_Yk = torch.mean((dYkdt - rhs_yk) ** 2)  # MSE loss for species ODEs

    # dTdt residual
    rhs_dT = (-(model.hk(T1) * omega_pred).sum(dim=1) / (rho * model.cp(T1, Yk1)))*dt
    loss_dTdt = torch.mean((dTdt - rhs_dT)**2)

    # dt Residual
    loss_dt = torch.mean(((dt - 1e-6)/1e-15)**2)

    # Pressure consistency loss
    loss_P = torch.mean((P1 - P0) ** 2)  

    # Net Production Rate Loss
    loss_omega = torch.mean((omega_pred - ct_omega) **2)

    total_loss = loss_dTdt + loss_P + loss_Yk  + loss_omega + loss_dt
    if verbose:
        print(f"Loss dT/dt:{loss_dTdt: .3g}, Loss dYk/dt: {loss_Yk: .3g}, Loss P: {loss_P: .3g}, Data Loss {loss_omega: .3g}, dt loss {loss_dt: .3g}")
        print(f"Total Loss: {total_loss: .3g}")
        # print(f"T: {torch.mean(T1)}")
        # print(f"p: {torch.mean(P1)}")
        # print(f"rho: {torch.mean(rho)}")


    return total_loss


