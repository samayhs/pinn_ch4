import cantera as ct
import torch
import matplotlib.pyplot as plt
import numpy as np
from itertools import product
import os
import glob
import winsound
from torch.utils.data import Dataset, DataLoader
from helpers.StateNet2 import *
import pandas as pd
from pathlib import Path
from helpers.lossfn import loss_fn2
"""
Cantera Basics
"""

def run_cantera_sim(gas, T, P, phi, t_end=1e-3, dt=1e-6):

    gas.TP = T, P*ct.one_atm
    gas.set_equivalence_ratio(phi, "CH4", "O2:1.0, N2:3.76")

    r = ct.ConstPressureReactor(gas)
    sim = ct.ReactorNet([r])


    states = ct.SolutionArray(gas, extra=['t'])
    
    while sim.time < t_end:
        sim.advance(sim.time + dt)
        states.append(r.phase.state, t=sim.time)
    IDT = states.t[np.argmax(states.Y[:, gas.species_index("H")])]
    return states, IDT

'''
Obtain CSVs from Cantera Sims
'''

def get_input_data_from_states(states: ct.SolutionArray, IDT: float, save_csv=False,csv_path=f'results/training_data.csv'):
    
    t = states.t.reshape(-1, 1)
    T = states.T.reshape(-1, 1)
    P = states.P.reshape(-1, 1)
    X = states.X
    N = len(P)
    t_end = (np.ones([1,N]) * IDT).reshape(-1,1)

    
    output = np.concatenate(( T, P, X, t, t_end), axis=1)
    if save_csv:
        
        np.savetxt(csv_path, output, delimiter=',')
    return torch.tensor(output, dtype=torch.float32)

'''
Model Training Helper Functions
'''
def generate_training_data(gas, P_list=None, T_list=None, phi_list= None, rerun=False, state_root='results/'):
    """
    Sample random initial conditions as state vectors [T, P, X_1...Y_K, dt].
    Loads from CSV if available, otherwise runs Cantera and saves.
    """
    if T_list == None:
        T_list = [1500, 1600, 1700]
    if P_list == None:
        P_list = [1]
    if phi_list == None:
        phi_list = [0.7, 1, 1.3]

    full_state_data = []


    # For each outlineed condition
    for T, P, phi in product(T_list, P_list, phi_list):
        # This is what the output file should look like

        state_path = os.path.join(state_root,f"states_T={T}K, P={P}atm, phi={phi}.csv")

        # Do not rerun ct sims if data on them already exists or explicitly told to
        if os.path.exists(state_path) and not rerun:
            print(f"\033[35mLoading from CSV: T={T}K, P={P}atm, phi={phi}\033[0m")

            data = torch.tensor(np.loadtxt(state_path, delimiter=','), dtype=torch.float32)

        else:
            # Run the cantera sim with default parameters
            print(f"Running Cantera: T={T}K, P={P}atm, phi={phi}")
            t_end = 1e-3
            states, IDT = run_cantera_sim(gas, T, P, phi, t_end)

            i = 0
            # try to have 1000ish states per run, and t_end be approx IDT*5
            while IDT * 5 > t_end:
                i += 1
                t_end = IDT * 5
                states, IDT = run_cantera_sim(gas, T, P, phi, t_end, dt=min(1e-4, t_end/1000))
                print(f"Run {i}: IDT:{IDT:.3g}, t_end:{t_end:.3g}")
            data = get_input_data_from_states(states,IDT, save_csv=True, csv_path=state_path)

        full_state_data.append(data)
    return full_state_data

def generate_reference_data(full_state_data):
    
    """
    trajectories: list of (n_rows, 2+K+2) tensors, columns [T, P, X_1..X_K, t, t_end]
    Returns dict of normalization statistics.
    """
    K = full_state_data[0].shape[1] -4
    all_rows = torch.cat(full_state_data, dim=0)  # (total_rows, 2+K+2)
    
    T        = all_rows[:, 0]
    P        = all_rows[:, 1]
    X        = all_rows[:, 2:2+K].clamp(min=1e-10)
    t        = all_rows[:, -2]
    t_end    = all_rows[:, -1]
    
    lnP  = torch.log(P)
    lnX  = torch.log(X)
    t_ratio = t / t_end
    

    
    return {

        'mu_T':         T.mean(),
        'sigma_T':      T.std(),

        'mu_lnP':       lnP.mean(),
        'sigma_lnP':    lnP.std(),
        
        'mu_lnX':       lnX.mean(dim=0),        # (K,)
        'sigma_lnX':    lnX.std(dim=0),         # (K,)

        'mu_t_norm':    t_ratio.mean(),
        'sigma_t_norm': t_ratio.std(),
    }

"""
Validate Model Methods
"""
def validate_model_sk(gas: ct.Solution, T: torch.Tensor):
    model = StateNet2(gas)
    sk_model = model.sk(T).detach().numpy()  # (N, K) dimensionless S_k/R

    species_names = gas.species_names
    n_sp = gas.n_species
    sk_cantera = np.zeros((len(T), n_sp))
    saved_TP = gas.T, gas.P
    for i, temp in enumerate(T):
        gas.TP = temp.item(), saved_TP[1]
        sk_cantera[i, :] = gas.standard_entropies_R
    gas.TP = saved_TP  # restore original state

    T_np = T.detach().numpy()
    n_plot = min(6, n_sp)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for k, ax in enumerate(axes.flatten()[:n_plot]):
        ax.plot(T_np, sk_cantera[:, k], 'k-', label='Cantera')
        ax.plot(T_np, sk_model[:, k], 'r--', label='Model')
        ax.set_xlabel('T [K]')
        ax.set_ylabel('$S_k / R$')
        ax.set_title(species_names[k])
        ax.legend()
    plt.tight_layout()
    plt.savefig('results/validate_sk.png', dpi=150)
    plt.show()

def validate_model_hk(gas: ct.Solution, T: torch.Tensor):
    model = StateNet2(gas)
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

def validate_model_arrhenius(gas: ct.Solution, T: torch.Tensor):
    model = StateNet2(gas)
    n_rxn = gas.n_reactions

    X = torch.tensor(gas.X, dtype=torch.float32).unsqueeze(0).expand(len(T), -1)  # (N, K)
    P = torch.full((len(T),), gas.P, dtype=torch.float32)  # (N,)

    kf_model, kr_model = model.compute_arrhenius(T, X, P)
    kf_model = kf_model.detach().numpy()
    kr_model = kr_model.detach().numpy()

    kf_cantera = np.zeros((len(T), n_rxn))
    kr_cantera = np.zeros((len(T), n_rxn))
    saved_TP = gas.T, gas.P
    for i, temp in enumerate(T):
        gas.TP = temp.item(), saved_TP[1]
        kf_cantera[i, :] = gas.forward_rate_constants
        kr_cantera[i, :] = gas.reverse_rate_constants
    gas.TP = saved_TP

    T_np = T.detach().numpy()
    n_plot = min(6, n_rxn)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for r, ax in enumerate(axes.flatten()[:n_plot]):
        ax.semilogy(T_np, kf_cantera[:, r], 'k-', label='Cantera $k_f$')
        ax.semilogy(T_np, kf_model[:, r],   'r--', label='Model $k_f$')
        ax.set_xlabel('T [K]')
        ax.set_ylabel('$k_f$')
        ax.set_title(f'Rxn {r}: {gas.reaction(r).equation[:40]}')
        ax.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig('results/validate_kf.png', dpi=150)
    plt.show()


'''
Create Dataset
'''

class TrajectoryPairDataset(Dataset):
    """
    Loads trajectory CSVs and serves random (state_t, state_tpdt) pairs.
    
    Each CSV row: [T, P, X_1, ..., X_K, t, t_end]
    t_end is constant per trajectory.
    """
    def __init__(self, csv_dir: str, K: int):
        self.K = K
    
        # Load all trajectories into memory as tensors
        self.trajectories = []
        for csv_path in sorted(Path(csv_dir).glob('*.csv')):
            df = pd.read_csv(csv_path)
            # Columns: T, P, X_1...X_K, t, t_end
            data = torch.from_numpy(df.values).float()
            self.trajectories.append(data)

    
    def __len__(self):
        return len(self.trajectories) *1000
    
    def __getitem__(self, idx):
        traj_idx = idx // 1000
        traj = self.trajectories[traj_idx]
        n_rows = traj.shape[0]
        
        # Pick two random row indices, i < j
        i, j = sorted(np.random.choice(n_rows, size=2, replace=False))
        
        row_t    = traj[i]
        row_tpdt = traj[j]

        # calc dt
        t_start    = row_t[-2]
        dt         = row_tpdt[-2] - t_start
        t_end      = row_t[-1]

        
        # Unpack
        state_t = torch.cat([
            row_t[:2 + self.K],   # 1D, shape (2+K,)
            dt.unsqueeze(0),      # 1D, shape (1,)
            t_end.unsqueeze(0),   # 1D, shape (1,)
        ], dim=0)                 # result: 1D, shape (2+K+2,)
        state_tpdt = row_tpdt[:2 + self.K]

        
        return {

                "x": state_t,

                "y": state_tpdt

                }


'''
Plotting
'''

def plot_training_loss(raw_loss_history, save_path='results/training_loss.png'):
    keys = [k for k in raw_loss_history[0] if any(d[k] > 0 for d in raw_loss_history)]
    n = len(keys)
    rows = (n + 1) // 2
    fig, axes = plt.subplots(rows, 2, figsize=(12, 4 * rows))
    axes = axes.flatten()
    for i, key in enumerate(keys):
        vals = [d[key] for d in raw_loss_history]
        axes[i].semilogy(vals)
        axes[i].set_title(key)
        axes[i].set_xlabel('Epoch')
        axes[i].set_ylabel('Loss')
        axes[i].grid(True, alpha=0.3)
    for i in range(n, len(axes)):
        axes[i].set_visible(False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()

'''
Train and Evaluate Model
'''

# ANSI color codes
_GREEN  = "\033[92m"
_RED    = "\033[91m"
_YELLOW = "\033[93m"
_CYAN   = "\033[96m"
_BOLD   = "\033[1m"
_DIM    = "\033[2m"
_RESET  = "\033[0m"

_prev_losses = {}

def print_losses(epoch, avg_raw):
    global _prev_losses
    header = f"{_BOLD}{_CYAN}Epoch {epoch:4d}{_RESET}"
    parts = []
    for k, v in avg_raw.items():
        if k in _prev_losses:
            prev = _prev_losses[k]
            if v < prev:
                arrow = f"{_GREEN}\u2193{_RESET}"
            elif v > prev:
                arrow = f"{_RED}\u2191{_RESET}"
            else:
                arrow = f"{_YELLOW}={_RESET}"
        else:
            arrow = " "
        style = _BOLD if k == 'net' else _DIM
        parts.append(f"{style}{k}{_RESET} {v:.3e} {arrow}")
    print(f"{header} | {' | '.join(parts)}")
    _prev_losses = dict(avg_raw)

def train(gas: ct.Solution, n_epochs=200, lr=1e-4, save_path='results/statenet2.pt'):
    import time
    global _prev_losses
    _prev_losses = {}


    full_data = generate_training_data(gas)
    ref_data  = generate_reference_data(full_data)

    dataset = TrajectoryPairDataset("results", gas.n_species)
    loader  = DataLoader(dataset, batch_size=250, shuffle=True, drop_last=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model     = StateNet2(gas, ref_data).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.print_ref_data()

    model.train()
    raw_loss_history = []
    t_start = time.time()
    verbose=False
    print(f"sigma_lnP value: {model.sigma_lnP.item():.6e}")
    print(f"sigma_T   value: {model.sigma_T.item():.6e}")

    for epoch in range(n_epochs):
        epoch_raw = {k: 0.0 for k in ['dx','enthalpy','atom','mole','data','net']}
        n_batches = 0
        for batch in loader:
            x_batch = batch["x"].to(device)
            y_batch = batch["y"].to(device)

            optimizer.zero_grad()

            loss, raw = loss_fn2(model, x_batch, y_batch, verbose=verbose)
            if n_batches == 0:  # once per epoch is enough
                with torch.no_grad():
                    y_norm = model.forward_normalized(x_batch)
                    print(f"  {_DIM}y_norm | T: mean={y_norm[:,0].mean():+.3f} std={y_norm[:,0].std():.3f} "
                        f"| P: mean={y_norm[:,1].mean():+.3f} std={y_norm[:,1].std():.3f} "
                        f"| X: mean={y_norm[:,2:].mean():+.3f} std={y_norm[:,2:].std():.3f}{_RESET}")

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            for k in epoch_raw:
                epoch_raw[k] += raw[k]
            n_batches += 1

        avg_raw = {k: v / n_batches for k, v in epoch_raw.items()}
        raw_loss_history.append(avg_raw)

        if epoch % 10 == 0:
            # verbose = True
            print_losses(epoch, avg_raw)
            elapsed = time.time() - t_start
            print(f"  {_DIM}time/epoch {elapsed/(epoch+1):.2f}s | ETA {((elapsed/(epoch+1))*(n_epochs - epoch))/60:.1f} min{_RESET}")
        else:
            verbose=False

    elapsed = time.time() - t_start
    print(f"\n{_BOLD}Final loss: {raw_loss_history[-1]['net']:.4e}{_RESET} | Time: {elapsed/60:.1f}min ({elapsed/n_epochs:.2f}s/epoch)")
    plot_training_loss(raw_loss_history, save_path=f'results/training_loss_lr{lr:.0e}.png')

    torch.save(model.state_dict(), save_path)
    print(f"{_GREEN}Model saved to {save_path}{_RESET}")

    return model, raw_loss_history


'''
Evaluate Model
'''

def evaluate(model_path='results/statenet2.pt',
             results_dir='results',
             eval_dir='results/eval',
             n_rollout_steps=200):
    """
    Evaluate trained StateNet2 against Cantera trajectories, using the true
    t_end (5 * IDT) from each trajectory as an oracle input. This isolates
    the model's ability to integrate the chemistry given a known time scale.

    Computes:
      1. Single-step metrics: MSE per variable for predict state(t+dt) from
         state(t) on random pairs.
      2. Rollout metrics: starting from state(t=0), feed model its own output
         for n_rollout_steps. Reports per-variable MSE, peak-T error, and
         time-to-peak-T error vs Cantera.
    """
    import time
    import os
    def prdb(var):
        print(var)
        if input() == 'y':
            raise KeyboardInterrupt

    os.makedirs(eval_dir, exist_ok=True)

    gas = ct.Solution("mechanisms/FFCM2.yaml")
    full_data = generate_training_data(gas)
    ref_data  = generate_reference_data(full_data)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model  = StateNet2(gas, ref_data).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    K = gas.n_species
    plot_species = [s for s in ['OH', 'CH4', 'CO2', 'H2O', 'O2'] if s in gas.species_names]
    sp_idx = {s: gas.species_index(s) for s in plot_species}

    all_rows = []

    for csv_path in sorted(Path(results_dir).glob('states_*.csv')):
        
        cond_name = csv_path.stem.replace('states_', '')
        print(f"\n{_BOLD}{_CYAN}Evaluating {cond_name}{_RESET}")

        df   = pd.read_csv(csv_path, header=None)
        traj = torch.from_numpy(df.values).float().to(device)
        n_rows    = traj.shape[0]
        t_end_val = traj[0, -1].item()   # constant per trajectory (5 * IDT)

        # =====================================================
        # 1. Single-step
        # =====================================================
        n_pairs = min(1000, n_rows * (n_rows - 1) // 2)
        rng = np.random.default_rng(0)
        i_idx = rng.integers(0, n_rows - 1, size=n_pairs)
        j_idx = np.array([rng.integers(i + 1, n_rows) for i in i_idx])

        rows_t    = traj[i_idx]
        rows_tpdt = traj[j_idx]

        dt    = (rows_tpdt[:, -2] - rows_t[:, -2]).unsqueeze(1)
        t_end = rows_t[:, -1:].clone()
        x_input = torch.cat([rows_t[:, :2 + K], dt, t_end], dim=1)
        y_true  = rows_tpdt[:, :2 + K]

        with torch.no_grad():
            y_pred = model(x_input)

        sq_err = (y_pred - y_true) ** 2
        mse_T = sq_err[:, 0].mean().item()
        mse_P = sq_err[:, 1].mean().item()
        mse_X = sq_err[:, 2:].mean().item()

        # Diagnostic: is the model actually responding to its inputs?
        out_T_std = y_pred[:, 0].std().item()
        in_T_std  = rows_t[:, 0].std().item()

        print(f"  {_DIM}single-step{_RESET} | T MSE: {mse_T:.3e} | P MSE: {mse_P:.3e} | X MSE: {mse_X:.3e}")
        print(f"  {_DIM}           {_RESET} | input T std: {in_T_std:.2f} | output T std: {out_T_std:.2f}")

        # =====================================================
        # 2. Rollout (oracle t_end)
        # =====================================================
        n_steps = min(n_rollout_steps, n_rows - 1)
        ref_idx = np.linspace(0, n_rows - 1, n_steps + 1).astype(int)
        ref_traj = traj[ref_idx]                     # (n_steps+1, 2+K+2)

        pred = [ref_traj[0, :2 + K].clone()]
        with torch.no_grad():
            for s in range(n_steps):
                t_cur  = ref_traj[s,     -2].item()
                t_next = ref_traj[s + 1, -2].item()
                dt_s = torch.tensor([[t_next - t_cur]], device=device)
                te_s = torch.tensor([[t_end_val]],     device=device)

                x_in = torch.cat([pred[-1].unsqueeze(0), dt_s, te_s], dim=1)
                y    = model(x_in).squeeze(0)
                pred.append(y)

        pred_traj = torch.stack(pred, dim=0)
        true_traj = ref_traj[:, :2 + K]

        ro_sq = (pred_traj - true_traj) ** 2
        ro_T  = ro_sq[:, 0].mean().item()
        ro_P  = ro_sq[:, 1].mean().item()
        ro_X  = ro_sq[:, 2:].mean().item()

        # Peak-T metrics
        t_arr        = ref_traj[:, -2].cpu().numpy()
        T_pred_np    = pred_traj[:, 0].cpu().numpy()
        T_true_np    = true_traj[:, 0].cpu().numpy()

        T_peak_true     = T_true_np.max()
        T_peak_pred     = T_pred_np.max()
        T_peak_abs_err  = abs(T_peak_pred - T_peak_true)
        T_peak_rel_err  = T_peak_abs_err / T_peak_true

        # Time of peak T (proxy for trajectory phase alignment, NOT IDT)
        t_peak_true = t_arr[np.argmax(T_true_np)]
        t_peak_pred = t_arr[np.argmax(T_pred_np)]
        t_peak_rel_err = abs(t_peak_pred - t_peak_true) / max(t_peak_true, 1e-12)

        print(f"  {_DIM}rollout    {_RESET} | T MSE: {ro_T:.3e} | P MSE: {ro_P:.3e} | X MSE: {ro_X:.3e}")
        print(f"  {_DIM}           {_RESET} | T_peak true: {T_peak_true:.1f}K | pred: {T_peak_pred:.1f}K | rel err: {T_peak_rel_err:.2%}")
        print(f"  {_DIM}           {_RESET} | t_peak true: {t_peak_true:.3e}s | pred: {t_peak_pred:.3e}s | rel err: {t_peak_rel_err:.2%}")

        # =====================================================
        # Plots
        # =====================================================
        n_plots = 1 + len(plot_species)
        fig, axes = plt.subplots(1, n_plots, figsize=(4 * n_plots, 4))
        if n_plots == 1:
            axes = [axes]

        axes[0].plot(t_arr, T_true_np, 'k-',  label='Cantera', linewidth=2)
        axes[0].plot(t_arr, T_pred_np, 'r--', label='StateNet2', linewidth=1.5)
        axes[0].set_xlabel('t [s]')
        axes[0].set_ylabel('T [K]')
        axes[0].set_title('Temperature')
        axes[0].legend(fontsize=8)
        axes[0].grid(True, alpha=0.3)

        for ax, sp in zip(axes[1:], plot_species):
            k = sp_idx[sp]
            ax.semilogy(t_arr, true_traj[:, 2 + k].cpu().numpy(), 'k-',  label='Cantera', linewidth=2)
            ax.semilogy(t_arr, pred_traj[:, 2 + k].cpu().numpy(), 'r--', label='StateNet2', linewidth=1.5)
            ax.set_xlabel('t [s]')
            ax.set_ylabel(f'X_{sp}')
            ax.set_title(sp)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        fig.suptitle(f'Rollout (oracle t_end): {cond_name}', fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(eval_dir, f'rollout_{cond_name}.png'), dpi=150)
        plt.close(fig)

        all_rows.append({
            'condition':           cond_name,
            'single_step_mse_T':   mse_T,
            'single_step_mse_P':   mse_P,
            'single_step_mse_X':   mse_X,
            'rollout_mse_T':       ro_T,
            'rollout_mse_P':       ro_P,
            'rollout_mse_X':       ro_X,
            'T_peak_true':         T_peak_true,
            'T_peak_pred':         T_peak_pred,
            'T_peak_rel_err':      T_peak_rel_err,
            't_peak_true':         t_peak_true,
            't_peak_pred':         t_peak_pred,
            't_peak_rel_err':      t_peak_rel_err,
        })

    summary_df = pd.DataFrame(all_rows)
    summary_path = os.path.join(eval_dir, 'eval_summary.csv')
    summary_df.to_csv(summary_path, index=False)

    print(f"\n{_BOLD}Summary{_RESET}")
    print(summary_df.to_string(index=False))
    print(f"\n{_GREEN}Plots saved to {eval_dir}/rollout_*.png{_RESET}")
    print(f"{_GREEN}Summary saved to {summary_path}{_RESET}")

    return summary_df