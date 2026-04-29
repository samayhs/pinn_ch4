import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Callable
import cantera as ct
from torch.utils.data import DataLoader

class StateNet2(nn.Module):
    """
    Maps state0 -> state(dt) for a time step dt.
    state: [T, P, X_1, ..., X_K, dt]

    Identical architecture to StateNet but uses mole fractions X
    instead of mass fractions Y.
    """
    def __init__(self, gas: ct.Solution, ref_data = None, hidden: int = 128, depth: int = 4):
        super().__init__()
        n_species = gas.n_species

        Wk = torch.tensor(gas.molecular_weights, dtype=torch.float32)
        self.register_buffer("Wk", Wk)

        
        input_len = 2 + n_species + 1   # T, P, X_1...X_K, t, t_end
        output_len = 2 + n_species      # T, P, X_1...X_K

        ## Forward Pass: y = W_d+1 * g(W_d * ... g(W_2 * g(W_1 * x + b1) + b2) ...) + b_{d+1}
        layers = [nn.Linear(input_len, hidden), nn.ELU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.ELU()]
        layers.append(nn.Linear(hidden, output_len))
        self.net = nn.Sequential(*layers)

        # NASA 7 Coeffs
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

        # --- Atom composition matrix (K, E) ---
        # a_{k,e} = number of atoms of element e in species k
        # For Atom Conservation
        n_elements = gas.n_elements
        atom_matrix = np.zeros((n_species, n_elements))
        for k in range(n_species):
            for e in range(n_elements):
                atom_matrix[k, e] = gas.n_atoms(k, e)
        self.register_buffer("atom_matrix", torch.tensor(atom_matrix, dtype=torch.float32))  # (K, E)


        # --- Stoichiometric matrices from gas ---
        nu_r = gas.reactant_stoich_coeffs
        nu_p = gas.product_stoich_coeffs
        if callable(nu_r):  # Cantera 3.x: method returning sparse matrix
            nu_r, nu_p = nu_r(), nu_p()
        if hasattr(nu_r, 'toarray'):
            nu_r, nu_p = nu_r.toarray(), nu_p.toarray()
        # (K, I) -> (I, K) so matrix ops give per-reaction results
        self.register_buffer("nu_prime", torch.tensor(np.array(nu_r).T, dtype=torch.float32))         # (I, K)
        self.register_buffer("nu_double_prime", torch.tensor(np.array(nu_p).T, dtype=torch.float32))  # (I, K)

        # --- Reaction classification and parameters ---
        n_rxn = gas.n_reactions
        n_sp  = gas.n_species

        # Primary Arrhenius (A, b, Ea): elementary base rate / three-body
        # base rate / falloff k_inf.  Unused for PLog (overwritten in forward).
        A_arr  = np.zeros(n_rxn)
        b_arr  = np.zeros(n_rxn)
        Ea_arr = np.zeros(n_rxn)

        # Falloff low-pressure limit (k_0)
        A0_arr  = np.zeros(n_rxn)
        b0_arr  = np.zeros(n_rxn)
        Ea0_arr = np.zeros(n_rxn)

        # Third-body efficiency matrix (I, K) — default 1.0
        eff_matrix = np.ones((n_rxn, n_sp))

        # Troe parameters [A, T3, T1, T2] per reaction (zero if unused)
        troe_coeffs = np.zeros((n_rxn, 4))

        # SRI parameters [a, b, c, d, e] per reaction (defaults: d=1, e=0)
        sri_coeffs = np.zeros((n_rxn, 5))
        sri_coeffs[:, 3] = 1.0  # default d
        is_sri = np.zeros(n_rxn, dtype=bool)

        # Reaction-type masks
        is_three_body = np.zeros(n_rxn, dtype=bool)
        is_falloff    = np.zeros(n_rxn, dtype=bool)
        is_troe       = np.zeros(n_rxn, dtype=bool)
        is_plog       = np.zeros(n_rxn, dtype=bool)

        # PLog data — collected then padded
        plog_pressures_list = []
        plog_A_list  = []
        plog_b_list  = []
        plog_Ea_list = []
        plog_rxn_indices = []
        max_plog_pts = 0

        for i, rxn in enumerate(gas.reactions()):
            rtype = rxn.reaction_type

            # --- helper to fill efficiency row ---
            def _fill_effs(i, rxn):
                tb = rxn.third_body
                if tb is None:
                    return
                eff_matrix[i, :] = tb.default_efficiency
                for sp_name, eff in tb.efficiencies.items():
                    eff_matrix[i, gas.species_index(sp_name)] = eff

            if rtype.startswith('three-body'):
                is_three_body[i] = True
                r = rxn.rate
                A_arr[i]  = r.pre_exponential_factor
                b_arr[i]  = r.temperature_exponent
                Ea_arr[i] = r.activation_energy
                _fill_effs(i, rxn)


            elif rtype.startswith('falloff'):
                is_falloff[i] = True
                r = rxn.rate
                # high-pressure limit  (stored as primary)
                hi = r.high_rate
                A_arr[i]  = hi.pre_exponential_factor
                b_arr[i]  = hi.temperature_exponent
                Ea_arr[i] = hi.activation_energy
                # low-pressure limit
                lo = r.low_rate
                A0_arr[i]  = lo.pre_exponential_factor
                b0_arr[i]  = lo.temperature_exponent
                Ea0_arr[i] = lo.activation_energy
                # Troe or SRI parameters
                if rtype == 'falloff-Troe':
                    is_troe[i] = True
                    fc = r.falloff_coeffs
                    troe_coeffs[i, :len(fc)] = fc
                elif rtype == 'falloff-SRI':
                    is_sri[i] = True
                    fc = r.falloff_coeffs
                    sri_coeffs[i, :len(fc)] = fc
                _fill_effs(i, rxn)


            elif rtype == 'pressure-dependent-Arrhenius':
                is_plog[i] = True
                rates_list = rxn.rate.rates
                n_pts = len(rates_list)
                max_plog_pts = max(max_plog_pts, n_pts)
                plog_rxn_indices.append(i)
                plog_pressures_list.append(np.array([p for p, _ in rates_list]))
                plog_A_list.append(np.array([r.pre_exponential_factor for _, r in rates_list]))
                plog_b_list.append(np.array([r.temperature_exponent   for _, r in rates_list]))
                plog_Ea_list.append(np.array([r.activation_energy     for _, r in rates_list]))

            else:  # elementary
                r = rxn.rate
                A_arr[i]  = r.pre_exponential_factor
                b_arr[i]  = r.temperature_exponent
                Ea_arr[i] = r.activation_energy
        
        log_A = np.log(np.clip(A_arr, 1e-10, None))
        log_A0 = np.log(np.clip(A0_arr, 1e-10, None))


                

        # --- register primary Arrhenius ---
        self.register_buffer("A_arr",  torch.tensor(A_arr,  dtype=torch.float32))
        self.register_buffer("b_arr",  torch.tensor(b_arr,  dtype=torch.float32))
        self.register_buffer("Ea_arr", torch.tensor(Ea_arr, dtype=torch.float32))
        self.register_buffer("log_A", torch.tensor(log_A,  dtype=torch.float32))


        # --- register falloff low-pressure ---
        self.register_buffer("A0_arr",  torch.tensor(A0_arr,  dtype=torch.float32))
        self.register_buffer("b0_arr",  torch.tensor(b0_arr,  dtype=torch.float32))
        self.register_buffer("Ea0_arr", torch.tensor(Ea0_arr, dtype=torch.float32))
        self.register_buffer("log_A0", torch.tensor(log_A0,  dtype=torch.float32))


        # --- third-body / Troe ---
        self.register_buffer("eff_matrix",  torch.tensor(eff_matrix,  dtype=torch.float32))
        self.register_buffer("troe_coeffs", torch.tensor(troe_coeffs, dtype=torch.float32))


        # --- SRI ---
        self.register_buffer("sri_coeffs", torch.tensor(sri_coeffs, dtype=torch.float32))


        # --- type masks ---
        self.register_buffer("is_three_body", torch.tensor(is_three_body))
        self.register_buffer("is_falloff",    torch.tensor(is_falloff))
        self.register_buffer("is_troe",       torch.tensor(is_troe))
        self.register_buffer("is_sri",        torch.tensor(is_sri))
        self.register_buffer("is_plog",       torch.tensor(is_plog))

        # --- PLog padded arrays ---
        n_plog = len(plog_rxn_indices)
        if n_plog > 0:
            plog_P_pad  = np.zeros((n_plog, max_plog_pts))
            plog_A_pad  = np.zeros((n_plog, max_plog_pts))
            plog_b_pad  = np.zeros((n_plog, max_plog_pts))
            plog_Ea_pad = np.zeros((n_plog, max_plog_pts))
            plog_n_pts  = np.zeros(n_plog, dtype=np.int64)


            log_A_p = np.zeros((n_plog, max_plog_pts))
            for j in range(n_plog):
                n = len(plog_pressures_list[j])
                plog_n_pts[j] = n
                plog_P_pad[j, :n]  = plog_pressures_list[j]
                plog_A_pad[j, :n]  = plog_A_list[j]
                plog_b_pad[j, :n]  = plog_b_list[j]
                plog_Ea_pad[j, :n] = plog_Ea_list[j]

                log_A_p[j, :n] = np.log(np.clip(plog_A_list[j], 1e-10, None))
                



            self.register_buffer("plog_P",    torch.tensor(plog_P_pad,  dtype=torch.float32))
            self.register_buffer("plog_lnP",  torch.log(torch.tensor(plog_P_pad, dtype=torch.float32).clamp(min=1e-10)))
            self.register_buffer("plog_A_p",  torch.tensor(plog_A_pad,  dtype=torch.float32))
            self.register_buffer("plog_b_p",  torch.tensor(plog_b_pad,  dtype=torch.float32))
            self.register_buffer("plog_Ea_p", torch.tensor(plog_Ea_pad, dtype=torch.float32))
            self.register_buffer("plog_n_pts", torch.tensor(plog_n_pts, dtype=torch.long))
            self.register_buffer("plog_idx",   torch.tensor(plog_rxn_indices, dtype=torch.long))

            self.register_buffer("log_A_p",  torch.tensor(log_A_p,  dtype=torch.float32))

        else:
            self.register_buffer("plog_P",     torch.zeros(0))
            self.register_buffer("plog_n_pts", torch.zeros(0, dtype=torch.long))
            self.register_buffer("plog_idx",   torch.zeros(0, dtype=torch.long))








        # --- Normalization statistics from ref_data ---
        # ref_data: (M, 2+K+2) raw [T, P, X_1..X_K, t, t_end]

        mu_T            = ref_data["mu_T"]
        sigma_T         = ref_data["sigma_T"]

        mu_lnP          = ref_data["mu_lnP"]
        sigma_lnP       = ref_data["sigma_lnP"]

        mu_lnX          = ref_data['mu_lnX']      # (K,)
        sigma_lnX       = ref_data['sigma_lnX']   # (K,)

        mu_t_norm       = ref_data['mu_t_norm']
        sigma_t_norm    = ref_data['sigma_t_norm']


        self.register_buffer("mu_T",         mu_T)
        self.register_buffer("sigma_T",      sigma_T)
        self.register_buffer("mu_lnP",       mu_lnP)
        self.register_buffer("sigma_lnP",      sigma_lnP.clamp(min=1e-3))
        self.register_buffer("mu_lnX",       mu_lnX)   # (K,)
        self.register_buffer("sigma_lnX",    sigma_lnX.clamp(min=1e-10))    # (K,)
        self.register_buffer("mu_t_norm",    mu_t_norm)    # (K,)
        self.register_buffer("sigma_t_norm", sigma_t_norm)    # (K,)
    
    def print_ref_data(self):

        print(f"mu_t: {self.mu_T}")
        print(f"sigma_T: {self.sigma_T}")

        print(f"mu_lnP: {self.mu_lnP}")
        print(f"sigma_lnP: {self.sigma_lnP}")

        print(f"mu_lnX: {self.mu_lnX}")
        print(f"sigma_lnX: {self.sigma_lnX}")
        
        print(f"mu_t_norm: {self.mu_t_norm}")
        print(f"sigma_t_norm: {self.sigma_t_norm}")

    def debug_forward(self, x_raw: torch.Tensor):
        """Print intermediate tensors after each layer/activation in the network."""
        if x_raw.dim() == 1:
            x_raw = x_raw.unsqueeze(0)
        with torch.no_grad():
            h = self.normalize(x_raw)
            print(f"Normalized input | shape: {h.shape}, min: {h.min():.4g}, max: {h.max():.4g}, nan: {h.isnan().any()}")
            for i, layer in enumerate(self.net):
                h = layer(h)
                print(f"Layer {i} ({layer.__class__.__name__:>7}) | min: {h.min():.4g}, max: {h.max():.4g}, mean: {h.mean():.4g}, std: {h.std():.4g}, nan: {h.isnan().any()}")

    def cp(self, T: torch.Tensor, X: torch.Tensor):
        """
        Compute mixture cp at N temperatures with N compositions (mole fractions).
        Returns mass-based mixture cp in J/(kg·K).

        T: (N,) or (N,1)
        X: (N, K) mole fractions
        """
        R = 8314.0  # J/(kmol·K)

        # Select NASA-7 coefficients based on T vs T_mid for each species
        mask = T.unsqueeze(1) > self.T_mids.unsqueeze(0)  # (N,K)

        a = torch.where(
            mask.unsqueeze(2),               # (N, K, 1)
            self.coeffs_high.unsqueeze(0),   # (1, K, 7)
            self.coeffs_low.unsqueeze(0)     # (1, K, 7)
        )  # (N, K, 7)

        # cp_k/R = a1 + a2*T + a3*T^2 + a4*T^3 + a5*T^4
        T2 = T * T
        T3 = T2 * T
        T4 = T3 * T
        T_powers = torch.stack([torch.ones_like(T), T, T2, T3, T4], dim=1)  # (N, 5)

        # cp_k/R for each species: (N, K)
        cp_over_R = torch.einsum('nkj,nj->nk', a[:, :, :5], T_powers)

         # Molar cp_k [J/(kmol·K)]
        cp_k_molar = cp_over_R * R  # (N, K)

        # Molar mixture cp = sum(X_k * cp_k_molar)
        cp_mix_molar = (X * cp_k_molar).sum(dim=1)  # (N,)

        # Mean molecular weight W_mix = sum(X_k * W_k)
        W_mix = (X * self.Wk.unsqueeze(0)).sum(dim=1)  # (N,)

        # Mass-based mixture cp = molar cp / W_mix
        cp_mix = cp_mix_molar / W_mix  # (N,) in J/(kg·K)
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

        T2 = T * T
        T3 = T2 * T
        T4 = T3 * T
        T_inv = 1.0 / T

        T_powers = torch.stack([
            torch.ones_like(T), T / 2, T2 / 3, T3 / 4, T4 / 5, T_inv
        ], dim=1)

        h_over_RT = torch.einsum('nkj,nj->nk', a[:, :, :6], T_powers)

        h_k = h_over_RT * R * T.unsqueeze(1)  # (N, K)
        return h_k

    def sk(self, T: torch.Tensor):
        """
        Compute per-species dimensionless entropy S_k/R using NASA-7 polynomials.
        S_k/R = a1*ln(T) + a2*T + a3*T^2/2 + a4*T^3/3 + a5*T^4/4 + a7

        T: (N,) or (N,1)
        Returns: (N, K)
        """
        T = T.squeeze(-1)  # (N,)
        mask = T.unsqueeze(1) > self.T_mids.unsqueeze(0)  # (N, K)

        a = torch.where(
            mask.unsqueeze(2),
            self.coeffs_high.unsqueeze(0),
            self.coeffs_low.unsqueeze(0)
        )  # (N, K, 7)

        T2 = T * T
        T3 = T2 * T
        T4 = T3 * T
        lnT = torch.log(T)

        # S/R = a1*ln(T) + a2*T + a3*T^2/2 + a4*T^3/3 + a5*T^4/4 + a7
        T_powers = torch.stack([lnT, T, T2 / 2, T3 / 3, T4 / 4], dim=1)  # (N, 5)
        s_over_R = torch.einsum('nkj,nj->nk', a[:, :, :5], T_powers) + a[:, :, 6]  # (N, K)
        return s_over_R

    def compute_arrhenius(self, T: torch.Tensor, X: torch.Tensor, P: torch.Tensor):
        """
        Compute forward and reverse rate constants for all reactions,
        handling elementary, three-body, falloff (Troe / Lindemann),
        and pressure-dependent Arrhenius (PLog).

        T: (N,) temperature in K
        X: (N, K) mole fractions
        P: (N,) pressure in Pa
        Returns: (k_f, k_r) each of shape (N, I)
        """
        R = 8314.0  # J/(kmol·K)
        P_atm = 101325.0  # Pa
        T_col = T.unsqueeze(1)  # (N, 1)

        # --- species concentrations and third-body [M] ---
        c_total = P / (R * T)                        # (N,)  [kmol/m^3]
        conc = X * c_total.unsqueeze(1)              # (N, K)
        M = conc @ self.eff_matrix.T                 # (N, I)

        # === 1. Base Arrhenius for all reactions ===

        # log_A = torch.log(self.A_arr.clamp(min=1e-10))
        log_A = self.log_A
        b_logT = self.b_arr * torch.log(T_col)
        Ea_RT = self.Ea_arr / (R * T_col)
        log_k_base = (log_A + b_logT - Ea_RT)
        k_f = torch.exp(log_k_base)                 # (N, I)


        # === 2. Three-body: k_f *= [M] ===
        if self.is_three_body.any():
            tb_mask = self.is_three_body.unsqueeze(0)        # (1, I)
            k_f = torch.where(tb_mask, k_f * M, k_f)

        # === 3. Falloff: k_eff = k_inf * Pr/(1+Pr) * F ===
        if self.is_falloff.any():
            # log_A0 = torch.log(self.A0_arr.clamp(min=1e-10))

            log_k0 = (self.log_A0
                      + self.b0_arr * torch.log(T_col)
                      - self.Ea0_arr / (R * T_col))
            
            k0 = torch.exp(log_k0)                          # (N, I)

            k_inf = torch.exp(log_k_base)                   # re-use primary
            Pr = k0 * M / k_inf.clamp(min=1e-10)            # (N, I)

            # default Lindemann: F = 1
            F = torch.ones_like(Pr)

            # Troe blending where applicable
            if self.is_troe.any():
                A_t  = self.troe_coeffs[:, 0]                # (I,)
                T3_t = self.troe_coeffs[:, 1]
                T1_t = self.troe_coeffs[:, 2]
                T2_t = self.troe_coeffs[:, 3]

                # F_cent; omit exp(-T2/T) term when T2 == 0 (3-param Troe)
                F_cent = ((1.0 - A_t) * torch.exp(-T_col / T3_t.clamp(min=1e-10))
                          + A_t * torch.exp(-T_col / T1_t.clamp(min=1e-10)))
                has_T2 = (T2_t.abs() > 0).unsqueeze(0)      # (1, I)
                F_cent = F_cent + torch.where(has_T2,
                                              torch.exp(-T2_t / T_col),
                                              torch.zeros(1, device=T.device))

                log_Fc = torch.log10(F_cent.clamp(min=1e-10))
                c_t = -0.4 - 0.67 * log_Fc
                n_t = 0.75 - 1.27 * log_Fc
                d_t = 0.14

                log_Pr = torch.log10(Pr.clamp(min=1e-10))
                f1 = (log_Pr + c_t) / (n_t - d_t * (log_Pr + c_t))
                log_F = log_Fc / (1.0 + f1 ** 2)
                F_troe = torch.pow(10.0, log_F)

                troe_mask = self.is_troe.unsqueeze(0)        # (1, I)
                F = torch.where(troe_mask, F_troe, F)

            # SRI blending: F = d * [a*exp(-b/T) + exp(-T/c)]^X * T^e
            #   where X = 1/(1 + log10(Pr)^2)
            if self.is_sri.any():
                a_s = self.sri_coeffs[:, 0]
                b_s = self.sri_coeffs[:, 1]
                c_s = self.sri_coeffs[:, 2]
                d_s = self.sri_coeffs[:, 3]
                e_s = self.sri_coeffs[:, 4]

                base = (a_s * torch.exp(-b_s / T_col)
                        + torch.exp(-T_col / c_s.clamp(min=1e-10)))
                log_Pr_sri = torch.log10(Pr.clamp(min=1e-10))
                X_sri = 1.0 / (1.0 + log_Pr_sri ** 2)
                F_sri = d_s * torch.pow(base.clamp(min=1e-10), X_sri) * torch.pow(T_col, e_s)

                sri_mask = self.is_sri.unsqueeze(0)
                F = torch.where(sri_mask, F_sri, F)

            k_falloff = k_inf * (Pr / (1.0 + Pr)) * F
            fo_mask = self.is_falloff.unsqueeze(0)
            k_f = torch.where(fo_mask, k_falloff, k_f)

        # === 4. PLog: log-pressure interpolation ===
        if self.is_plog.any() and self.plog_P.numel() > 0:
            ln_P = torch.log(P)                              # (N,)
            n_plog = self.plog_idx.shape[0]

            for j in range(n_plog):
                n_pts  = self.plog_n_pts[j]
                lnP_pts = self.plog_lnP[j, :n_pts]          # (n_pts,)
                A_p  = self.plog_A_p[j, :n_pts]
                b_p  = self.plog_b_p[j, :n_pts]
                Ea_p = self.plog_Ea_p[j, :n_pts]

                log_A_p = self.log_A_p[j, :n_pts]
                # log_A_p = torch.log(A_p.clamp(min=1e-10))

                # ln(k) at every tabulated pressure point  (N, n_pts)
                log_k_pts = (log_A_p
                             + b_p * torch.log(T_col)
                             - Ea_p / (R * T_col))

                # clamp query pressure into table bounds
                ln_P_c = ln_P.clamp(min=lnP_pts[0], max=lnP_pts[-1])  # (N,)

                # bracket indices
                idx_hi = torch.searchsorted(lnP_pts, ln_P_c).clamp(1, n_pts - 1)
                idx_lo = idx_hi - 1

                lnk_lo  = log_k_pts.gather(1, idx_lo.unsqueeze(1)).squeeze(1)
                lnk_hi  = log_k_pts.gather(1, idx_hi.unsqueeze(1)).squeeze(1)
                lnP_lo  = lnP_pts[idx_lo]
                lnP_hi  = lnP_pts[idx_hi]

                frac = (ln_P_c - lnP_lo) / (lnP_hi - lnP_lo + 1e-10)
                k_plog = torch.exp(lnk_lo + frac * (lnk_hi - lnk_lo))
                





                k_f[:, self.plog_idx[j]] = k_plog

        # === 5. Reverse rate via equilibrium constant ===
        h_over_RT = self.hk(T) / (R * T.unsqueeze(1))       # (N, K)
        s_over_R  = self.sk(T)                               # (N, K)
        g_over_RT = h_over_RT - s_over_R                     # (N, K)

        nu_net = self.nu_double_prime - self.nu_prime         # (I, K)
        delta_g_over_RT = g_over_RT @ nu_net.T                # (N, I)
        K_p = torch.exp(-delta_g_over_RT)

        sum_nu = nu_net.sum(dim=1)                            # (I,)
        K_c = K_p * (P_atm / (R * T_col)) ** (-sum_nu.unsqueeze(0))

        k_r = k_f / K_c.clamp(min=1e-100)

        return k_f, k_r

    def normalize(self, state_raw: torch.Tensor) -> torch.Tensor:
        """
        Raw [T, P, X_1..X_K, dt, t_end] -> normalized [T', P', lnX'_1..lnX'_K, dt_n].
        """
        T_n  = (state_raw[:, 0] - self.mu_T) / self.sigma_T
        P_n  = (torch.log(state_raw[:, 1]) - self.mu_lnP) / self.sigma_lnP

        lnX  = torch.log(state_raw[:, 2:-2].clamp(min=1e-10))
        X_n  = (lnX - self.mu_lnX) / self.sigma_lnX

        dt_ratio  = state_raw[:, -2:-1] / state_raw[:, -1:]
        dt_n = (dt_ratio - self.mu_t_norm )/ self.sigma_t_norm
        return torch.cat([T_n.unsqueeze(1), P_n.unsqueeze(1), X_n, dt_n], dim=1)

    def denormalize(self, state_norm: torch.Tensor) -> torch.Tensor:
        """
        Normalized [T', P', lnX'_1..lnX'_K] -> raw [T, P, X_1..X_K].
        """
        T   = state_norm[:, 0] * self.sigma_T + self.mu_T
        lnP = state_norm[:, 1] * self.sigma_lnP + self.mu_lnP
        P   = torch.exp(lnP)
        lnX = state_norm[:, 2:] * self.sigma_lnX + self.mu_lnX
        X   = torch.exp(lnX).clamp(min=1e-10)
        return torch.cat([T.unsqueeze(1), P.unsqueeze(1), X], dim=1)

    def renormalize(self, state_raw: torch.Tensor) -> torch.Tensor:
        """
        Raw [T, P, X_1..X_K, dt, t_end] -> normalized [T', P', lnX'_1..lnX'_K].
        """
        T_n  = (state_raw[:, 0] - self.mu_T) / self.sigma_T
        P_n  = (torch.log(state_raw[:, 1]) - self.mu_lnP) / self.sigma_lnP

        lnX  = torch.log(state_raw[:, 2:].clamp(min=1e-10))
        X_n  = (lnX - self.mu_lnX) / self.sigma_lnX

        return torch.cat([T_n.unsqueeze(1), P_n.unsqueeze(1), X_n], dim=1)

    def forward_normalized(self, state_raw: torch.Tensor):
        x_norm = self.normalize(state_raw)
        return self.net(x_norm)
    
    
    def forward(self, state0: torch.Tensor) -> torch.Tensor:
        """
        Takes RAW input [T, P, X_1..X_K, t, t_end], returns RAW predicted [T, P, X_1..X_K].
        """
        x_norm = self.normalize(state0)
        y_norm = self.net(x_norm)
        return self.denormalize(y_norm)
