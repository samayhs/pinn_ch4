import torch
from helpers.StateNet2 import StateNet2


def loss_fn2(model: StateNet2, states0: torch.Tensor, states1: torch.Tensor, verbose = False):
    """
    Combined loss for StateNet2 (mole-fraction formulation).

    Aggregates five terms with adaptive normalization so each contributes ~O(1):
        1. ODE residual    — AD derivatives vs mass-action kinetics
        2. Enthalpy        — h_mix conservation across the time step
        3. Atom            — element conservation across the time step
        4. Mole fraction   — sum(X_k) = 1 normalization
        5. Data            — supervised MSE against Cantera target state

    states0: (N, 2+K+1) raw state at time t   [T, P, X_1..X_K, dt]
    states1: (N, 2+K+1) Cantera target state at time t+dt
    """
    # l_ode      = ode_loss(model, states0)
    l_enthalpy = enthalpy_loss(model, states0)
    l_atom     = atom_conservation_loss(model, states0)
    l_mole     = mole_fraction_loss(model, states0)
    l_data     = data_loss(model, states0, states1, verbose=verbose)
    l_dx       = ode_loss_v2(model, states0)
    # print(l_dx)



    total = 1e-8*l_dx + 1e-9*l_enthalpy + 1e5*l_atom + 1e2*l_mole + 1e4*l_data

    raw_losses = {
        # 'ode': l_ode.item(),
        'dx' : l_dx.item(),
        'enthalpy': l_enthalpy.item(),
        'atom': l_atom.item(),
        'mole': l_mole.item(),
        'data': l_data.item(),
        'net' : total.item()
    }

    return total, raw_losses
    
def ode_loss(model: StateNet2, x_raw: torch.Tensor, verbose=False):
    """
    Physics-informed loss for StateNet2.

    Takes RAW state [T, P, X_1..X_K, dt, t_end] as input.  Internally normalizes
    for the network, computes autograd derivatives in normalized space,
    de-normalizes for physics, and compares AD-derived vs ODE-derived rates.

    x_raw: (N, 2+K+1) — raw [T, P, X_1..X_K]
    model: StateNet2 with normalization stats set (ref_data was provided).
    """


   
    R = 8314.0
    N = x_raw.shape[0]
    K = model.nu_prime.shape[1]

    # Separate t_end from the rest
    t_end     = x_raw[:, -1]     # (N,)

    # Normalize; dt' will be last column
    x_raw.requires_grad_(True)
    x_norm = model.normalize(x_raw)
    x_norm.requires_grad_(True)

    y_norm  = model.net(x_norm)

    T_prime = y_norm[:, 0]
    # index 1 = P', index 2: = X'
    X_prime = y_norm[:, 2:]

    # De-normalize
    T   = T_prime * model.sigma_T + model.mu_T
    lnP = y_norm[:, 1] * model.sigma_lnP + model.mu_lnP
    P   = torch.exp(lnP)
    lnX = X_prime * model.sigma_lnX + model.mu_lnX
    X   = torch.exp(lnX).clamp(min=1e-10)

    c    = P / (R * T)
    conc = X * c.unsqueeze(1)


    # Analytical chemistry
    k_f, k_r = model.compute_arrhenius(T, X, P)
    log_conc = torch.log(conc.clamp(min=1e-10))
    fwd_prod = torch.exp(log_conc @ model.nu_prime.T)
    rev_prod = torch.exp(log_conc @ model.nu_double_prime.T)
    rate_per_rxn = k_f * fwd_prod - k_r * rev_prod
    dconc_dt = rate_per_rxn @ (model.nu_double_prime - model.nu_prime)



    # Autograd derivatives w.r.t. dt' (last column of x_norm)
    dTprime_dtprime = torch.autograd.grad(
        T_prime.sum(), x_norm,
        create_graph=True, retain_graph=True
    )[0][:, -1]

    dXprime_dtprime = torch.zeros(N, K, device=x_raw.device)
    for k in range(K):
        dXprime_dtprime[:, k] = torch.autograd.grad(
            X_prime[:, k].sum(), x_norm,
            create_graph=True, retain_graph=True
        )[0][:, -1]

    # Branch 1: ODE side
    prefactor = (t_end.unsqueeze(1) * model.sigma_t_norm) / model.sigma_lnX
    dXprime_dtprime_ODE = prefactor * (1.0 / X) * (1.0 / c.unsqueeze(1)) * dconc_dt

    # Branch 2: AD side with T correction
    correction = (model.sigma_T / model.sigma_lnX) * (1.0 / T.unsqueeze(1)) \
                 * dTprime_dtprime.unsqueeze(1)
    dXprime_dtprime_adj = dXprime_dtprime - correction

    residual = (dXprime_dtprime_ODE.clamp(min=-1e6, max=1e6) - dXprime_dtprime_adj)


    return (residual ** 2).mean()

def ode_loss_v2(model: StateNet2, x_raw: torch.Tensor):
    """
    Alternative ODE loss: compare dX/dt in physical units instead of 
    log-normalized space. Avoids 1/X blowup at the cost of unequal 
    weighting across species magnitudes.
    """
    R = 8314.0
    N = x_raw.shape[0]
    K = model.nu_prime.shape[1]
    
    t_end = x_raw[:, -1]
    
    # Forward pass with grad tracking
    x_norm = model.normalize(x_raw)
    x_norm.requires_grad_(True)
    y_norm = model.net(x_norm)
    
    # De-normalize to physical
    T_prime = y_norm[:, 0]
    P_norm  = y_norm[:, 1]
    X_prime = y_norm[:, 2:]
    
    T = T_prime * model.sigma_T + model.mu_T
    P = torch.exp(P_norm * model.sigma_lnP + model.mu_lnP)
    X = torch.exp(X_prime * model.sigma_lnX + model.mu_lnX).clamp(min=1e-10)
    
    # Chain-rule factor: dt' / dt
    dt_prime_dt = 1.0 / (t_end * model.sigma_t_norm)  # (N,)
    
    # AD side: dT/dt and dX/dt of de-normalized predictions
    dT_dt_AD = torch.autograd.grad(
        T.sum(), x_norm,
        create_graph=True, retain_graph=True
    )[0][:, -1] * dt_prime_dt   # (N,)
    
    dX_dt_AD = torch.zeros(N, K, device=x_raw.device)
    for k in range(K):
        dX_dt_AD[:, k] = torch.autograd.grad(
            X[:, k].sum(), x_norm,
            create_graph=True, retain_graph=True
        )[0][:, -1] * dt_prime_dt
    
    # Chemistry side: analytical dX/dt
    c    = P / (R * T)
    conc = X * c.unsqueeze(1)
    
    k_f, k_r = model.compute_arrhenius(T, X, P)
    log_conc = torch.log(conc.clamp(min=1e-30))
    fwd_prod = torch.exp(log_conc @ model.nu_prime.T)
    rev_prod = torch.exp(log_conc @ model.nu_double_prime.T)
    rate_per_rxn = k_f * fwd_prod - k_r * rev_prod
    dconc_dt = rate_per_rxn @ (model.nu_double_prime - model.nu_prime)  # (N, K)
    
    # Eq. 21: dX_k/dt = (1/c)·d[X_k]/dt + ([X_k]/(c·T))·dT/dt
    # Use autograd's dT/dt for the second term
    dX_dt_chem = (dconc_dt / c.unsqueeze(1) 
                  + (conc / (c * T).unsqueeze(1)) * dT_dt_AD.unsqueeze(1))
    
    # MSE in physical units
    return ((dX_dt_AD - dX_dt_chem) ** 2).mean()

def enthalpy_loss(model: StateNet2, x_input: torch.Tensor):
    """
    Enforces conservation of total enthalpy: h_mix,0 == h_mix,1.
    h_mix = sum(Y_k * h_k / W_k) = sum(X_k * h_k) / W_mix
    """
    T0 = x_input[:, 0]
    X0 = x_input[:, 2:-2]
    hk0 = model.hk(T0)  # (N, K) J/kmol

    output = model(x_input)
    T1 = output[:, 0]
    X1 = output[:, 2:]
    hk1 = model.hk(T1)

    # X -> Y: Y_k = X_k * W_k / W_mix
    W_mix0 = (X0 * model.Wk).sum(dim=1, keepdim=True)
    Y0 = X0 * model.Wk / W_mix0

    W_mix1 = (X1 * model.Wk).sum(dim=1, keepdim=True)
    Y1 = X1 * model.Wk / W_mix1

    # Mass-specific enthalpy: h_mix = sum(Y_k * h_k / W_k)  [J/kg]
    h_mix0 = (Y0 * hk0 / model.Wk).sum(dim=1)
    h_mix1 = (Y1 * hk1 / model.Wk).sum(dim=1)

    return torch.mean((h_mix1 - h_mix0) ** 2)

def atom_conservation_loss(model: StateNet2, x_input: torch.Tensor):
    """
    Enforces conservation of atoms across a time step.

    In a closed reactor the number of atoms of each element per unit mass
    is invariant:

        n_e / m  =  sum_k( a_{e,k} * X_k ) / W_mix  =  const

    where a_{e,k} is the number of atoms of element e in species k,
    X_k are mole fractions, and W_mix = sum_k(X_k * W_k).

    x_input: (N, 2+K+1) raw state [T, P, X_1..X_K, dt]
    model:   StateNet2 with atom_matrix buffer (K, E)
    """
    X0 = x_input[:, 2:-2]                                  # (N, K)
    W_mix0 = (X0 * model.Wk).sum(dim=1, keepdim=True)      # (N, 1)

    output = model(x_input)
    X1 = output[:, 2:]                                      # (N, K)
    W_mix1 = (X1 * model.Wk).sum(dim=1, keepdim=True)      # (N, 1)

    # Atoms per unit mass: (N, E)
    atoms_per_mass0 = (X0 @ model.atom_matrix) / W_mix0
    atoms_per_mass1 = (X1 @ model.atom_matrix) / W_mix1

    return torch.mean((atoms_per_mass1 - atoms_per_mass0) ** 2)

def mole_fraction_loss(model: StateNet2, x_input: torch.Tensor):

    output = model(x_input)
    X = output[:, 2:]

    return torch.mean((X.sum(dim=1) - 1) ** 2)

def data_loss(model: StateNet2, x_input: torch.Tensor, y: torch.Tensor, verbose = False):
    """
    Supervised MSE loss between model prediction and target state.

    x_input: (N, 2+K+1) raw state at time t
    y:       (N, 2+K+1) target state at time t+dt
    """
    output = model.forward_normalized(x_input)
    y_norm = model.renormalize(y)

    if verbose:
        denorm = model.denormalize(output)
        err = denorm - y
        T_err = denorm[:,0]
        Ty = y[:,0]
        print(f"t_err: {T_err.mean()} - {Ty.mean()}")


    return torch.mean((output -y_norm ) ** 2)

