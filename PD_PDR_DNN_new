#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
11D Cloud Microphysics — Unified Statistical Trials: PD (Alg.2), PDR (Alg.5), and DNN
- EXACTLY shared training subsets per (m, trial) across PD, PDR, and DNN
- Same Tasmanian sparse-grid test set and quadrature weights for evaluation
- Identical seeding protocol for reproducibility
- Saves per-output error-bar plots (mean±std) comparing all three methods
- Also saves averaged learning curves with ±1σ bands

External deps: Tasmanian, parmap_framework, module_runcrm, torch, numpy, matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import legendre
import sys
import time
import warnings
from pathlib import Path
warnings.filterwarnings("ignore", message="Glyph.*missing from font")

# --- External packages paths (customize if needed) ---
sys.path.append('/home/ss24ce/.local/lib/python3.10/site-packages')

import Tasmanian
from parmap_framework import parmap
from module_runcrm import runcrm

# Torch for DNN
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim

# =============================================================================
# CONFIGURATION
# =============================================================================

RNG_SEED = 42

# TRIALS CONFIGURATION
NUM_TRIALS = 20                 # number of trials per training size
TRIALS_RNG_SEED = 123           # base seed for (trial, m) reproducibility

# Training sizes
TRAIN_SIZES_TRIALS = list(range(100, 1401, 100))  # 100, 200, ..., 1400

# SG configuration for 11D
LEVEL_11D = 4
GRID_RULE = "clenshaw-curtis"

# Cloud model I/O
INPUT_FILE  = './cloud_column_model/run_one_crm1d.txt'
OUTPUT_FILE = './cloud_column_model/crm1d_output.txt'
NAMELIST    = './cloud_column_model/namelist_3h_t30-180.f90'

# Outputs from CRM (columns 18..23 inclusive -> 6 outputs)
OUTPUT_NAMES = ['PCP', 'ACC', 'LWP', 'IWP', 'OLR', 'OSR']
K = 6

# Parmap cluster settings
PARMASTER   = 'scispark6.jpl.nasa.gov:8786'
PARNWORKERS = 12
PARMODE     = 'par'

# Target total primal-dual steps (for PD and PDR fairness)
TARGET_TOTAL_STEPS = 2000

# PD early stop (optional)
PD_EARLY_STOP     = False
PD_EARLY_STOP_TOL = 1e-10

# DNN hyperparameters
DNN_LAYERS = 8
DNN_NODES  = 50
DNN_LR     = 1e-3
DNN_BATCH  = 32
DNN_EPOCHS = 20000              # epochs per (m, trial)
DNN_USE_AMP = True              # mixed precision if CUDA available

# =============================================================================
# Algorithms (PD and PDR)
# =============================================================================

def algorithm_2_exact(A, B, w, lambda_param, tau, sigma, T, C_init, Lambda_init,
                      early_stop=False, early_stop_tol=1e-10):
    m, N = A.shape
    C_bar = np.zeros(N)
    C = np.array(C_init, dtype=float, copy=True)
    Lambda = np.array(Lambda_init, dtype=float, copy=True)

    thr_base = tau * lambda_param * w  # (N,)

    for n in range(T):
        P = C - tau * (A.T @ Lambda)
        C_next = np.sign(P) * np.maximum(np.abs(P) - thr_base, 0.0)
        Q = Lambda + sigma * (A @ (2.0 * C_next - C)) - sigma * B
        norm_Q = np.linalg.norm(Q)
        Lambda_next = Q / norm_Q if norm_Q > 1.0 else Q
        C_bar = (n / (n + 1.0)) * C_bar + (1.0 / (n + 1.0)) * C_next

        if early_stop and n > 0:
            denom = max(np.linalg.norm(C), 1.0)
            rel_change = np.linalg.norm(C_next - C) / denom
            if rel_change < early_stop_tol:
                C = C_next
                Lambda = Lambda_next
                break

        C = C_next
        Lambda = Lambda_next

    return C_bar


def algorithm_5_exact(A, B, w, lambda_param, T, R, eps_0, r, s,
                      tau=None, sigma=None, max_iter=10**9,
                      pd_early_stop=False, pd_early_stop_tol=1e-10):
    m, N = A.shape
    C_tilde = np.zeros(N)
    eps_l = np.linalg.norm(B)

    if (tau is None) or (sigma is None):
        norm_A = np.linalg.norm(A, ord=2)
        tau_eff = 1.0 / norm_A
        sigma_eff = 1.0 / norm_A
    else:
        tau_eff = float(tau)
        sigma_eff = float(sigma)

    total_iterations = 0
    for l in range(R):
        eps_l = r * (eps_l + eps_0)
        a_l = s * eps_l
        if a_l > 1e-15:
            C_init_scaled = (C_tilde / a_l) if (np.linalg.norm(C_tilde) > 0) else np.zeros(N)
            Lambda_init_scaled = np.zeros(m)
            B_scaled = B / a_l
            C_result = algorithm_2_exact(
                A, B_scaled, w, lambda_param, tau_eff, sigma_eff, T,
                C_init_scaled, Lambda_init_scaled,
                early_stop=pd_early_stop, early_stop_tol=pd_early_stop_tol
            )
            C_tilde = a_l * C_result
            total_iterations += T
        else:
            total_iterations += T

        if total_iterations >= max_iter:
            break
    return C_tilde

# =============================================================================
# Utilities
# =============================================================================

def compute_intrinsic_weights_legendre(Lambda):
    """ u_ν = ∏_{j=1}^d √(2ν_j + 1) """
    N_basis = Lambda.shape[0]
    d = Lambda.shape[1]
    weights = np.ones(N_basis, dtype=float)
    for n in range(N_basis):
        for k in range(d):
            weights[n] *= np.sqrt(2 * Lambda[n, k] + 1)
    return weights


def map_to_canonical(X, pmin, pmax):
    return 2.0 * (X - pmin) / (pmax - pmin) - 1.0


def multiidx_gen(N, rule, w, base=0, multiidx=np.array([]), MULTI_IDX=np.array([])):
    """ Generate hyperbolic cross multi-index set with rule=HCfunc up to level w """
    if len(multiidx) != N:
        i = base
        while rule(np.append(multiidx, i)) <= w:
            MULTI_IDX = multiidx_gen(N, rule, w, base, np.append(multiidx, i), MULTI_IDX)
            i += 1
    else:
        MULTI_IDX = np.vstack([MULTI_IDX, multiidx]) if MULTI_IDX.size else multiidx
    return MULTI_IDX


def compute_table51_hparams(A, m):
    """Return lambda, tau, sigma, r, T_inner, s based on Table 5.1."""
    normA2 = np.linalg.norm(A, ord=2)
    sigma = 1.0 / normA2
    tau = 1.0 / normA2
    r = np.exp(-1.0)
    T_inner = int(np.ceil((2.0 * normA2) / (r * np.sqrt(sigma * tau))))
    T_inner = max(T_inner, 1)
    s = T_inner / (2.0 * normA2)
    lam = (np.sqrt(25.0 * m)) ** -1
    return lam, tau, sigma, r, T_inner, s, normA2

# ---------------- DNN -----------------
class DNN(nn.Module):
    def __init__(self, input_dim, output_dim, layers, nodes_per_layer):
        super().__init__()
        layers_list = [nn.Linear(input_dim, nodes_per_layer), nn.Tanh()]
        for _ in range(layers):
            layers_list.extend([nn.Linear(nodes_per_layer, nodes_per_layer), nn.Tanh()])
        layers_list.append(nn.Linear(nodes_per_layer, output_dim))
        self.layers = nn.Sequential(*layers_list)

    def forward(self, x):
        return self.layers(x)


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)


def train_dnn_on_subset(xi_train, F_train, epochs, batch_size, lr, layers, nodes,
                        device, use_amp=True, verbose_epochs=(10000, 20000, 30000, 40000, 50000)):
    """
    Train a DNN on the provided subset (inputs in canonical domain, outputs in physical units).
    Output scaling is computed from the subset (train-only min-max) and inverted for predictions.
    Returns: trained model, (F_min, F_max) for inverse scaling.
    """
    # Min-max scale outputs based on THIS subset (train-only)
    F_min = F_train.min(axis=0)
    F_max = F_train.max(axis=0)
    # Prevent zero range
    span = np.where((F_max - F_min) == 0, 1.0, (F_max - F_min))
    F_scaled = 2 * (F_train - F_min) / span - 1

    dataset = TensorDataset(torch.from_numpy(xi_train).float(),
                            torch.from_numpy(F_scaled).float())
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = DNN(input_dim=xi_train.shape[1], output_dim=F_train.shape[1],
                layers=layers, nodes_per_layer=nodes).to(device)
    model.apply(init_weights)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ExponentialLR(
        optimizer, gamma=(5e-7 / lr) ** (1 / max(1, epochs))
    )
    criterion = nn.MSELoss()

    scaler = torch.cuda.amp.GradScaler(enabled=use_amp and (device.type == 'cuda'))

    model.train()
    for epoch in range(epochs):
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp and (device.type == 'cuda')):
                pred = model(xb)
                loss = criterion(pred, yb)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        scheduler.step()
        if (epoch + 1) in verbose_epochs:
            print(f"      DNN epoch {epoch+1}/{epochs} — loss={loss.item():.4e}")

    return model, F_min, F_max

# =============================================================================
# Single-trial runner (shared data + three methods)
# =============================================================================

def run_single_trial_all_methods(trial_idx, train_size, pmin, pmax, d, Lambda, N_basis, weights,
                                 Psi_sg, F_sg, w_pos,
                                 device):
    """Run one trial for a given m using the SAME training set for PD, PDR, and DNN."""
    print(f"    Trial {trial_idx+1}/{NUM_TRIALS} (m={train_size})")

    # Set seed for this specific trial
    np.random.seed(TRIALS_RNG_SEED + trial_idx * 1000 + train_size)

    # Generate random training points for this trial
    x_train = np.random.uniform(pmin, pmax, size=(train_size, d))
    xi_train = map_to_canonical(x_train, pmin, pmax)

    # Run CRM once for this training set
    runs_train = [[INPUT_FILE, OUTPUT_FILE, NAMELIST, i+1, x_train[i].tolist()] for i in range(train_size)]
    pmap = parmap.Parmap(master=PARMASTER, mode=PARMODE, numWorkers=PARNWORKERS)
    F_train_full = np.array(pmap(runcrm, runs_train))
    F_train = F_train_full[:, 18:24]  # K columns

    # Build A matrix (Legendre basis) for PD/PDR
    A = np.ones((train_size, N_basis), dtype=float)
    for n in range(N_basis):
        for kdim in range(d):
            Pk = legendre(Lambda[n, kdim])
            norm = np.sqrt((2 * Lambda[n, kdim] + 1))
            A[:, n] *= norm * Pk(xi_train[:, kdim])
    A /= np.sqrt(train_size)

    # Hyperparameters (Table 5.1)
    lam, tau, sigma, r, T_inner, s, normA2 = compute_table51_hparams(A, train_size)
    R = int(np.ceil(TARGET_TOTAL_STEPS / T_inner))
    R = max(R, 1)
    T_pd = R * T_inner

    # --- Solve PD & PDR for all outputs ---
    errs_pd = np.zeros(K)
    errs_pdr = np.zeros(K)
    errs_dnn = np.zeros(K)
    sparsity_pd = np.zeros(K, dtype=int)
    sparsity_pdr = np.zeros(K, dtype=int)

    # Precompute for DNN: inputs in canonical space
    xi_subset = xi_train.astype(np.float32)

    # Timings (optional)
    t0 = time.time()

    for kout in range(K):
        # PD
        b = F_train[:, kout] / np.sqrt(train_size)
        C_res_pd = algorithm_2_exact(
            A, b, weights, lam, tau, sigma, T_pd,
            C_init=np.zeros(N_basis), Lambda_init=np.zeros(train_size),
            early_stop=PD_EARLY_STOP, early_stop_tol=PD_EARLY_STOP_TOL
        )
        y_eval_pd = Psi_sg @ C_res_pd

        # PDR
        C_res_pdr = algorithm_5_exact(
            A, F_train[:, kout] / np.sqrt(train_size), weights, lam,
            T_inner, R, eps_0=1e-6, r=r, s=s,
            tau=tau, sigma=sigma, max_iter=T_pd,
            pd_early_stop=PD_EARLY_STOP, pd_early_stop_tol=PD_EARLY_STOP_TOL
        )
        y_eval_pdr = Psi_sg @ C_res_pdr

        # Sparsity
        sparsity_pd[kout]  = int(np.count_nonzero(np.abs(C_res_pd)  > 1e-12))
        sparsity_pdr[kout] = int(np.count_nonzero(np.abs(C_res_pdr) > 1e-12))

        # --- DNN (train once per K? We train ONE multi-output model for all K) ---
        # We'll defer DNN training to AFTER the loop to train single model (6 outputs).
        pass

    # Train ONE DNN for all outputs using the SAME training subset
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else device
    model, F_min, F_max = train_dnn_on_subset(
        xi_subset, F_train, epochs=DNN_EPOCHS, batch_size=DNN_BATCH, lr=DNN_LR,
        layers=DNN_LAYERS, nodes=DNN_NODES, device=device, use_amp=DNN_USE_AMP
    )

    # Predict on SG
    model.eval()
    with torch.no_grad():
        xi_can_tensor = torch.from_numpy(Psi_sg[:, :0].copy()).float()  # dummy to satisfy linter
    # Build SG input tensor correctly (xi_can from outer scope is not passed; we reconstruct via Psi_sg context in caller)
    # We'll return model and (F_min, F_max) to caller to compute predictions (needs xi_can there).

    t1 = time.time()
    trial_time = t1 - t0

    # Return all artifacts needed to compute errors in the caller (since we need xi_can there)
    return {
        'A': A,
        'lam': lam, 'tau': tau, 'sigma': sigma, 'r': r, 'T_inner': T_inner, 's': s,
        'R': R, 'T_pd': T_pd,
        'F_train': F_train,
        'C_pd': None, 'C_pdr': None,  # not returning per-k coeffs to save memory
        'sparsity_pd': sparsity_pd,
        'sparsity_pdr': sparsity_pdr,
        'model': model,
        'F_min': F_min,
        'F_max': F_max,
        'xi_train': xi_train,
        'trial_time': trial_time
    }

# =============================================================================
# Main
# =============================================================================

def main():
    np.random.seed(RNG_SEED)

    # --- 11D bounds and names (all vary) ---
    pmin = np.array([50.0, 0.10, 50.0, 0.1, 0.05, 0.05, 0.05, 0.05, 0.05, 1.e-4, 2.e-6], dtype=float)
    pmax = np.array([1000.0, 1.0, 1200.0, 0.90, 5.0, 2.5, 2.5, 1.0, 1.0, 2.e-3, 1.e-3], dtype=float)
    xnames = ['as', 'bs', 'ag', 'bg', 'N0r', 'N0s', 'N0g', 'rhos', 'rhog', 'qc0', 'qi0']

    d = 11
    print("11D Parameter Analysis Setup:")
    for j, name in enumerate(xnames):
        print(f"  {name}: [{pmin[j]:.2e}, {pmax[j]:.2e}]")

    # --- Hyperbolic cross multi-index ---
    HCfunc = lambda x: np.prod(x + 1) - 1
    p_poly = 20
    Lambda = multiidx_gen(d, HCfunc, p_poly).astype(int)
    N_basis = Lambda.shape[0]
    print(f"11D basis size (HC level {p_poly}): {N_basis}")

    # Intrinsic weights (Legendre)
    weights = compute_intrinsic_weights_legendre(Lambda)
    print(f"Intrinsic weights range: [{np.min(weights):.3f}, {np.max(weights):.3f}]")

    # --- Tasmanian Sparse Grid for 11D (TEST SET) ---
    grid = Tasmanian.SparseGrid()
    grid.makeGlobalGrid(d, 0, LEVEL_11D, "level", GRID_RULE)
    grid.clearDomainTransform()

    xi_can = grid.getPoints()           # (m_sg, 11)
    w_can = grid.getQuadratureWeights()
    m_sg = xi_can.shape[0]
    print(f"SG grid size (11D, level {LEVEL_11D}): {m_sg}")

    # Physical points
    x_phys_sg = 0.5 * (xi_can + 1.0) * (pmax - pmin) + pmin

    print("Running CRM for SG projection...")
    runs_sg = [[INPUT_FILE, OUTPUT_FILE, NAMELIST, i+1, x_phys_sg[i].tolist()] for i in range(m_sg)]
    pmap = parmap.Parmap(master=PARMASTER, mode=PARMODE, numWorkers=PARNWORKERS)
    F_sg_full = np.array(pmap(runcrm, runs_sg))
    F_sg = F_sg_full[:, 18:24]

    # Build Ψ on SG points
    Psi_sg = np.ones((m_sg, N_basis), dtype=float)
    for n in range(N_basis):
        for kdim in range(d):
            Pk = legendre(Lambda[n, kdim])
            norm = np.sqrt((2 * Lambda[n, kdim] + 1))
            Psi_sg[:, n] *= norm * Pk(xi_can[:, kdim])

    # SG Gram diagnostics (optional)
    w_prob = w_can / (2.0 ** d)
    G_sg = Psi_sg.T @ (w_prob[:, None] * Psi_sg)
    diag_err = np.max(np.abs(np.diag(G_sg) - 1.0))
    offdiag_err = np.max(np.abs(G_sg - np.diag(np.diag(G_sg))))
    print(f"SG Gram matrix - Diagonal error: {diag_err:.2e}, Off-diagonal error: {offdiag_err:.2e}")

    # Physical quadrature weights
    volume_phys = np.prod(pmax - pmin)
    w_phys = w_can * (volume_phys / (2.0 ** d))
    w_pos = np.abs(w_phys)

    # Storage for statistical results
    results = {
        'train_sizes': TRAIN_SIZES_TRIALS,
        'l2_pd': [],      # (n_sizes, n_trials, K)
        'l2_pdr': [],     # (n_sizes, n_trials, K)
        'l2_dnn': [],     # (n_sizes, n_trials, K)
        'sp_pd': [],
        'sp_pdr': [],
        'times': []
    }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    for i, m in enumerate(TRAIN_SIZES_TRIALS):
        print(f"\n--- Training Size: {m} ({i+1}/{len(TRAIN_SIZES_TRIALS)}) ---")
        size_l2_pd = []
        size_l2_pdr = []
        size_l2_dnn = []
        size_sp_pd = []
        size_sp_pdr = []
        size_times = []

        for trial_idx in range(NUM_TRIALS):
            # Generate shared data and train DNN
            trial_art = run_single_trial_all_methods(
                trial_idx, m, pmin, pmax, d, Lambda, N_basis, weights, Psi_sg, F_sg, w_pos, device
            )

            # Reconstruct PD/PDR predictions per K (we recompute here to get y_eval for errors)
            # Build A and b again using artifacts from trial (already have A and F_train)
            A = trial_art['A']
            lam, tau, sigma, r, T_inner, s = trial_art['lam'], trial_art['tau'], trial_art['sigma'], trial_art['r'], trial_art['T_inner'], trial_art['s']
            R = trial_art['R']
            T_pd = trial_art['T_pd']
            F_train = trial_art['F_train']

            # PD/PDR
            l2_pd = np.zeros(K)
            l2_pdr = np.zeros(K)
            sp_pd = np.zeros(K, dtype=int)
            sp_pdr = np.zeros(K, dtype=int)
            for kout in range(K):
                b = F_train[:, kout] / np.sqrt(m)
                C_pd = algorithm_2_exact(
                    A, b, weights, lam, tau, sigma, T_pd,
                    C_init=np.zeros(A.shape[1]), Lambda_init=np.zeros(m),
                    early_stop=PD_EARLY_STOP, early_stop_tol=PD_EARLY_STOP_TOL
                )
                y_pd = Psi_sg @ C_pd

                C_pdr = algorithm_5_exact(
                    A, b, weights, lam, T_inner, R, eps_0=1e-6, r=r, s=s,
                    tau=tau, sigma=sigma, max_iter=T_pd,
                    pd_early_stop=PD_EARLY_STOP, pd_early_stop_tol=PD_EARLY_STOP_TOL
                )
                y_pdr = Psi_sg @ C_pdr

                # Errors
                y_true = F_sg[:, kout]
                diff_pd = y_true - y_pd
                diff_pdr = y_true - y_pdr
                num_pd = np.dot(diff_pd**2, w_pos)
                num_pdr = np.dot(diff_pdr**2, w_pos)
                denom = np.dot(y_true**2, w_pos)
                l2_pd[kout] = np.sqrt(num_pd / denom) if denom > 0 else np.nan
                l2_pdr[kout] = np.sqrt(num_pdr / denom) if denom > 0 else np.nan

                # Sparsity
                sp_pd[kout] = int(np.count_nonzero(np.abs(C_pd)  > 1e-12))
                sp_pdr[kout] = int(np.count_nonzero(np.abs(C_pdr) > 1e-12))

            # DNN predictions on SG
            model = trial_art['model']
            F_min, F_max = trial_art['F_min'], trial_art['F_max']
            # Prevent zero range
            span = np.where((F_max - F_min) == 0, 1.0, (F_max - F_min))

            model.eval()
            with torch.no_grad():
                xi_can_tensor = torch.from_numpy(xi_can).float().to(device)
                F_sg_pred_scaled = model(xi_can_tensor).cpu().numpy()
            F_sg_pred = (F_sg_pred_scaled + 1)/2 * span + F_min

            l2_dnn = np.zeros(K)
            for kout in range(K):
                diff = F_sg[:, kout] - F_sg_pred[:, kout]
                num = np.dot(diff**2, w_pos)
                denom = np.dot(F_sg[:, kout]**2, w_pos)
                l2_dnn[kout] = np.sqrt(num / denom) if denom > 0 else np.nan

            # Collect
            size_l2_pd.append(l2_pd)
            size_l2_pdr.append(l2_pdr)
            size_l2_dnn.append(l2_dnn)
            size_sp_pd.append(sp_pd)
            size_sp_pdr.append(sp_pdr)
            size_times.append(trial_art['trial_time'])

        # Store arrays
        results['l2_pd'].append(np.array(size_l2_pd))
        results['l2_pdr'].append(np.array(size_l2_pdr))
        results['l2_dnn'].append(np.array(size_l2_dnn))
        results['sp_pd'].append(np.array(size_sp_pd))
        results['sp_pdr'].append(np.array(size_sp_pdr))
        results['times'].append(np.array(size_times))

        # Print quick stats
        print("  PD   - mean L2:", np.mean(size_l2_pd, axis=0), "±", np.std(size_l2_pd, axis=0))
        print("  PDR  - mean L2:", np.mean(size_l2_pdr, axis=0), "±", np.std(size_l2_pdr, axis=0))
        print("  DNN  - mean L2:", np.mean(size_l2_dnn, axis=0), "±", np.std(size_l2_dnn, axis=0))

    # =============================================================================
    # Visualization
    # =============================================================================
    outdir = Path("trials_results")
    outdir.mkdir(exist_ok=True)

    # Convert to arrays
    l2_pd_trials  = np.array(results['l2_pd'])   # (n_sizes, n_trials, K)
    l2_pdr_trials = np.array(results['l2_pdr'])
    l2_dnn_trials = np.array(results['l2_dnn'])

    mean_pd  = np.mean(l2_pd_trials, axis=1)     # (n_sizes, K)
    mean_pdr = np.mean(l2_pdr_trials, axis=1)
    mean_dnn = np.mean(l2_dnn_trials, axis=1)
    std_pd   = np.std(l2_pd_trials, axis=1)
    std_pdr  = np.std(l2_pdr_trials, axis=1)
    std_dnn  = np.std(l2_dnn_trials, axis=1)

    sizes = TRAIN_SIZES_TRIALS

    # Per-output error-bar plots
    for kplot in range(K):
        fig, ax = plt.subplots(1, 1, figsize=(10, 7))
        ax.errorbar(sizes, mean_pd[:, kplot],  yerr=std_pd[:, kplot],  fmt='o-', capsize=5, alpha=0.8, label='PD (Table 5.1)')
        ax.errorbar(sizes, mean_pdr[:, kplot], yerr=std_pdr[:, kplot], fmt='s-', capsize=5, alpha=0.8, label='PDR (Table 5.1)')
        ax.errorbar(sizes, mean_dnn[:, kplot], yerr=std_dnn[:, kplot], fmt='^-', capsize=5, alpha=0.8, label='DNN')

        # Reference slope m^{-1/2}
        ref = 0.1 * (np.array(sizes) / sizes[0]) ** (-0.5)
        ax.loglog(sizes, ref, '--', alpha=0.7, label='m^{-1/2} reference')

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Training Sample Size')
        ax.set_ylabel('L2 Relative Error')
        ax.set_title(f'Statistical Learning Curve: {OUTPUT_NAMES[kplot]} (11D)\n{NUM_TRIALS} trials, error bars = ±1 std')
        ax.legend()
        ax.grid(True, which='both', alpha=0.3)
        fig.tight_layout()
        fig.savefig(outdir / f"statistical_l2_error_PD_PDR_DNN_{OUTPUT_NAMES[kplot]}.png", dpi=150)
        plt.close(fig)
        print(f"Saved: {outdir / f'statistical_l2_error_PD_PDR_DNN_{OUTPUT_NAMES[kplot]}.png'}")

    # Averaged curves with bands
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    avg_pd  = np.mean(mean_pd, axis=1)
    avg_pdr = np.mean(mean_pdr, axis=1)
    avg_dnn = np.mean(mean_dnn, axis=1)
    sd_pd   = np.mean(std_pd, axis=1)
    sd_pdr  = np.mean(std_pdr, axis=1)
    sd_dnn  = np.mean(std_dnn, axis=1)

    ax.fill_between(sizes, avg_pd - sd_pd,   avg_pd + sd_pd,   alpha=0.25, label='PD ±1σ')
    ax.fill_between(sizes, avg_pdr - sd_pdr, avg_pdr + sd_pdr, alpha=0.25, label='PDR ±1σ')
    ax.fill_between(sizes, avg_dnn - sd_dnn, avg_dnn + sd_dnn, alpha=0.25, label='DNN ±1σ')

    ax.loglog(sizes, avg_pd,  'o-', linewidth=2, label='PD (Table 5.1)')
    ax.loglog(sizes, avg_pdr, 's-', linewidth=2, label='PDR (Table 5.1)')
    ax.loglog(sizes, avg_dnn, '^-', linewidth=2, label='DNN')

    ref = 0.1 * (np.array(sizes) / sizes[0]) ** (-0.5)
    ax.loglog(sizes, ref, '--', alpha=0.7, label='m^{-1/2} reference')

    ax.set_xlabel('Training Sample Size')
    ax.set_ylabel('Average L2 Relative Error')
    ax.set_title(f'Statistical Learning Curves: PD vs PDR vs DNN (11D)\n{NUM_TRIALS} trials per size, shaded regions = ±1 std')
    ax.legend()
    ax.grid(True, which='both', alpha=0.3)
    plt.tight_layout()
    fig.savefig(outdir / "statistical_average_learning_curves_PD_PDR_DNN.png", dpi=150)
    plt.close(fig)
    print(f"Saved: {outdir / 'statistical_average_learning_curves_PD_PDR_DNN.png'}")


if __name__ == "__main__":
    main()
