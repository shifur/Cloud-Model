#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
11D Cloud Microphysics — PD/PDR (Table 5.1) + DNN with shared train/test sets
- Uses the same training subsets for PD, PDR, and DNN (first m samples)
- Evaluates all models on the SAME Tasmanian sparse-grid test set
- Plots learning curves and scalability (runtime vs training size) for all
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless plotting on HPC
import matplotlib.pyplot as plt
from scipy.special import legendre
import sys
import time
import warnings
warnings.filterwarnings("ignore", message="Glyph.*missing from font")

# Torch for DNN
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim

# --- External packages paths (customize if needed) ---
sys.path.append('/home/ss24ce/.local/lib/python3.10/site-packages')

import Tasmanian
# from parmap_framework import parmap   # <-- removed
from module_runcrm import runcrm

# GPU-batched PD/PDR solver (keep this file alongside this script)
from gpu_batched_sr_lasso import solve_all_outputs_gpu_batch

# --- lightweight fallback executor to replace parmap --------------------------
def exec_runs(runs, workers=1):
    """
    runs: iterable of [INPUT_FILE, OUTPUT_FILE, NAMELIST, run_id, x_phys_list]
    workers: number of processes; 1 = serial
    """
    if workers <= 1:
        return [runcrm(args) for args in runs]
    try:
        from multiprocessing import Pool
        with Pool(processes=workers) as pool:
            return pool.map(runcrm, runs)
    except Exception:
        # fall back to serial if Pool isn't feasible on this node/queue
        return [runcrm(args) for args in runs]

# =============================================================================
# CONFIGURATION
# =============================================================================

RNG_SEED = 42

# Training sizes (use the SAME list for PD/PDR and DNN)
TRAIN_SIZES = list(range(100, 1401, 100))  # 100..2000 step 100 (change to your needs)

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

# (parmap settings kept for compatibility; master/mode are unused now)
PARMASTER   = 'scispark6.jpl.nasa.gov:8786'
PARNWORKERS = 12
PARMODE     = 'par'

# PD/PDR budget
TARGET_TOTAL_STEPS = 2000

# DNN settings (toggle CPU/GPU runs)
DNN_DO_CPU = True
DNN_DO_GPU = torch.cuda.is_available()
DNN_EPOCHS = 20000            # set high at your own risk; 50k will be very slow
DNN_BATCH  = 128
DNN_LR     = 1e-3
DNN_LAYERS = 8
DNN_HIDDEN = 50
DNN_SCHED_FINAL_LR = 5e-7    # exponential decay target over epochs

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
    """Return lambda, tau, sigma, r, T_inner, s, ||A||_2 based on Table 5.1."""
    normA2 = np.linalg.norm(A, ord=2)
    sigma = 1.0 / normA2
    tau = 1.0 / normA2
    r = np.exp(-1.0)
    T_inner = int(np.ceil((2.0 * normA2) / (r * np.sqrt(sigma * tau))))
    T_inner = max(T_inner, 1)
    s = T_inner / (2.0 * normA2)
    lam = (np.sqrt(25.0 * m)) ** -1
    return lam, tau, sigma, r, T_inner, s, normA2

# === Helper solvers for scalability timing (PD only, CPU/GPU variants) =======

def solve_pd_cpu_batched(A, B, w, hp):
    hp_cpu = dict(hp)
    hp_cpu['run_on_gpu'] = False
    return solve_all_outputs_gpu_batch(A, B, w, hp_cpu, method='pd')

def solve_pd_gpu_batched(A, B, w, hp):
    hp_gpu = dict(hp)
    hp_gpu['run_on_gpu'] = True
    return solve_all_outputs_gpu_batch(A, B, w, hp_gpu, method='pd')

def solve_pd_gpu_sequential(A, B, w, hp):
    """Run PD on GPU but one output at a time (for comparison)."""
    hp_gpu = dict(hp)
    hp_gpu['run_on_gpu'] = True
    N = A.shape[1]
    K = B.shape[1]
    X = np.zeros((N, K), dtype=float)
    for k in range(K):
        Xk = solve_all_outputs_gpu_batch(A, B[:, [k]], w, hp_gpu, method='pd')
        X[:, k] = Xk[:, 0]
    return X

# =============================================================================
# DNN model & helpers
# =============================================================================

class DNN(nn.Module):
    def __init__(self, input_dim, output_dim, layers, hidden):
        super().__init__()
        seq = [nn.Linear(input_dim, hidden), nn.Tanh()]
        for _ in range(layers):
            seq += [nn.Linear(hidden, hidden), nn.Tanh()]
        seq += [nn.Linear(hidden, output_dim)]
        self.net = nn.Sequential(*seq)
    def forward(self, x): return self.net(x)

def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)

def train_dnn(X_np, Y_np, epochs, batch, lr, layers, hidden, device, final_lr):
    # deterministic-ish
    torch.manual_seed(RNG_SEED)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(RNG_SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    model = DNN(X_np.shape[1], Y_np.shape[1], layers, hidden).to(device)
    model.apply(weights_init)
    crit = nn.MSELoss()
    opt = optim.Adam(model.parameters(), lr=lr)
    gamma = (final_lr / lr) ** (1.0 / max(epochs, 1))
    sched = optim.lr_scheduler.ExponentialLR(opt, gamma=gamma)

    ds = TensorDataset(torch.from_numpy(X_np).float(), torch.from_numpy(Y_np).float())
    dl = DataLoader(ds, batch_size=batch, shuffle=True, drop_last=False)

    t0 = time.time()
    model.train()
    for _ in range(epochs):
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            pred = model(xb)
            loss = crit(pred, yb)
            loss.backward()
            opt.step()
        sched.step()
    t_elapsed = time.time() - t0
    return model, t_elapsed

def dnn_predict(model, X_np, device):
    model.eval()
    with torch.no_grad():
        x = torch.from_numpy(X_np).float().to(device)
        y = model(x).cpu().numpy()
    return y

# =============================================================================
# Main
# =============================================================================

def main():
    np.random.seed(RNG_SEED)

    # --- 11D bounds and names ---
    pmin = np.array([50.0, 0.10, 50.0, 0.1, 0.05, 0.05, 0.05, 0.05, 0.05, 1.e-4, 2.e-6], dtype=float)
    pmax = np.array([1000.0, 1.0, 1200.0, 0.90, 5.0, 2.5, 2.5, 1.0, 1.0, 2.e-3, 1.e-3], dtype=float)
    xnames = ['as', 'bs', 'ag', 'bg', 'N0r', 'N0s', 'N0g', 'rhos', 'rhog', 'qc0', 'qi0']

    d = 11
    for j, name in enumerate(xnames):
        print(f"  {name}: [{pmin[j]:.2e}, {pmax[j]:.2e}]")

    # --- Hyperbolic cross multi-index for PD/PDR basis ---
    HCfunc = lambda x: np.prod(x + 1) - 1
    p_poly = 20
    Lambda = multiidx_gen(d, HCfunc, p_poly).astype(int)
    N_basis = Lambda.shape[0]
    print(f"11D basis size (HC level {p_poly}): {N_basis}")

    # Intrinsic weights (Legendre)
    weights = compute_intrinsic_weights_legendre(Lambda)
    print(f"Intrinsic weights range: [{np.min(weights):.3f}, {np.max(weights):.3f}]")

    # --- Tasmanian Sparse Grid for test set (shared across all methods) ---
    grid = Tasmanian.SparseGrid()
    grid.makeGlobalGrid(d, 0, LEVEL_11D, "level", GRID_RULE)
    grid.clearDomainTransform()

    xi_can = grid.getPoints()           # (m_sg, 11)
    w_can = grid.getQuadratureWeights()
    m_sg = xi_can.shape[0]
    print(f"SG grid size (11D, level {LEVEL_11D}): {m_sg}")

    # Physical points for CRM test eval
    x_phys_sg = 0.5 * (xi_can + 1.0) * (pmax - pmin) + pmin

    print("Running CRM for SG projection (shared test set)…")
    runs_sg = [[INPUT_FILE, OUTPUT_FILE, NAMELIST, i+1, x_phys_sg[i].tolist()] for i in range(m_sg)]
    # pmap_exec = parmap.Parmap(master=PARMASTER, mode=PARMODE, numWorkers=PARNWORKERS)
    # F_sg_full = np.array(pmap_exec(runcrm, runs_sg))
    F_sg_full = np.array(exec_runs(runs_sg, workers=PARNWORKERS))
    F_sg = F_sg_full[:, 18:24]  # 6 outputs

    # Build Ψ on SG points (for PD/PDR evaluation)
    Psi_sg = np.ones((m_sg, N_basis), dtype=float)
    for n in range(N_basis):
        for kdim in range(d):
            Pk = legendre(Lambda[n, kdim])
            norm = np.sqrt((2 * Lambda[n, kdim] + 1))
            Psi_sg[:, n] *= norm * Pk(xi_can[:, kdim])

    # Weighted integration weights for L2 error
    volume_phys = np.prod(pmax - pmin)
    w_phys = w_can * (volume_phys / (2.0 ** d))
    w_pos = np.abs(w_phys)

    # =============================================================================
    # Shared Training Set: generate once, reuse for PD/PDR/DNN
    # =============================================================================

    m_train_max = max(TRAIN_SIZES)
    x_train_max = np.random.uniform(pmin, pmax, size=(m_train_max, d))
    xi_train_max = map_to_canonical(x_train_max, pmin, pmax)

    print(f"Generating {m_train_max} training samples (shared)…")
    runs_train_max = [[INPUT_FILE, OUTPUT_FILE, NAMELIST, i+1, x_train_max[i].tolist()] for i in range(m_train_max)]
    # F_train_max_full = np.array(pmap_exec(runcrm, runs_train_max))
    F_train_max_full = np.array(exec_runs(runs_train_max, workers=PARNWORKERS))
    F_train_max = F_train_max_full[:, 18:24]

    # DNN scaling (fit once on full training set)
    F_min = F_train_max.min(axis=0)
    F_max = F_train_max.max(axis=0)
    # (DNN trains on scaled outputs; PD/PDR use physical units with 1/sqrt(m) normalization)

    # =============================================================================
    # Storage
    # =============================================================================

    results = {
        'train_sizes': TRAIN_SIZES,
        # PD/PDR
        'l2_pd': [], 'l2_pdr': [],
        't_pd': [], 't_pdr': [],
        't_cpu_pd': [], 't_gpu_pd_batched': [], 't_gpu_pd_seq': [],
        # DNN
        'l2_dnn_cpu': [], 'l2_dnn_gpu': [],
        't_dnn_cpu': [], 't_dnn_gpu': [],
    }

    # =============================================================================
    # Loop over training sizes (same subset for all methods)
    # =============================================================================

    for m in TRAIN_SIZES:
        print("\n" + "="*70)
        print(f"Training with {m} samples (shared subset across PD/PDR/DNN)")
        print("="*70)

        # Shared subset: first m samples (NOT random) to match PD/PDR behavior
        xi_train = xi_train_max[:m]
        F_train  = F_train_max[:m]

        # ---------------- PD/PDR: build A (Legendre basis on canonical xi_train) -------------
        A = np.ones((m, N_basis), dtype=float)
        for n in range(N_basis):
            for kdim in range(d):
                Pk = legendre(Lambda[n, kdim])
                norm = np.sqrt((2 * Lambda[n, kdim] + 1))
                A[:, n] *= norm * Pk(xi_train[:, kdim])

        # Normalize A and B by 1/sqrt(m) (as decided earlier)
        scale = 1.0 / np.sqrt(m)
        A_scaled = A * scale
        B_scaled = F_train * scale

        # Hyperparams from scaled A
        lam, tau, sigma, r, T_inner, s, normA2 = compute_table51_hparams(A_scaled, m)
        R = int(np.ceil(TARGET_TOTAL_STEPS / T_inner))
        R = max(R, 1)
        T_pd = R * T_inner

        print("Hyperparameters (Table 5.1 on scaled A):")
        print(f"  λ={lam:.6f}, ||A||₂={normA2:.6f}, τ=σ={tau:.6f}, r={r:.6f}, T_inner={T_inner}, s={s:.6f}, R={R}, T_pd={T_pd}")

        hp = {'lambda':lam,'tau':tau,'sigma':sigma,'r':r,'T_inner':T_inner,'s':s,'R':R,'T_pd':T_pd,'run_on_gpu':True}

        # ---- PD (batched, GPU) ----
        t0 = time.time()
        X_pd = solve_all_outputs_gpu_batch(A_scaled, B_scaled, weights, hp, method='pd')
        t_pd = time.time() - t0

        # ---- PDR (batched, GPU) ----
        t0 = time.time()
        X_pdr = solve_all_outputs_gpu_batch(A_scaled, B_scaled, weights, hp, method='pdr')
        t_pdr = time.time() - t0

        # ---- Extra PD timings for scalability graph ----
        t0 = time.time(); _ = solve_pd_cpu_batched(A_scaled, B_scaled, weights, hp); t_cpu_pd = time.time() - t0
        t0 = time.time(); _ = solve_pd_gpu_batched(A_scaled, B_scaled, weights, hp); t_gpu_pd_b = time.time() - t0
        t0 = time.time(); _ = solve_pd_gpu_sequential(A_scaled, B_scaled, weights, hp); t_gpu_pd_s = time.time() - t0

        results['t_pd'].append(t_pd)
        results['t_pdr'].append(t_pdr)
        results['t_cpu_pd'].append(t_cpu_pd)
        results['t_gpu_pd_batched'].append(t_gpu_pd_b)
        results['t_gpu_pd_seq'].append(t_gpu_pd_s)

        # ---- PD/PDR predictions on SG for L2 errors ----
        y_pd  = Psi_sg @ X_pd
        y_pdr = Psi_sg @ X_pdr

        l2_pd = np.zeros(K); l2_pdr = np.zeros(K)
        for k in range(K):
            diff_pd  = F_sg[:, k] - y_pd[:, k]
            diff_pdr = F_sg[:, k] - y_pdr[:, k]
            num_pd  = np.dot(diff_pd**2,  w_pos); den = np.dot(F_sg[:, k]**2, w_pos)
            num_pdr = np.dot(diff_pdr**2, w_pos)
            l2_pd[k]  = np.sqrt(num_pd / den)  if den > 0 else np.nan
            l2_pdr[k] = np.sqrt(num_pdr / den) if den > 0 else np.nan
        results['l2_pd'].append(l2_pd)
        results['l2_pdr'].append(l2_pdr)

        # ---------------------------- DNN (same subset & same SG test) -----------------------
        # Prepare DNN inputs (canonical) and scaled outputs to [-1,1] with global stats
        X_dnn = xi_train
        Y_dnn = 2.0 * (F_train - F_min) / (F_max - F_min + 1e-12) - 1.0

        # DNN CPU
        l2_dnn_cpu_vec = np.full(K, np.nan)
        t_dnn_cpu = np.nan
        if DNN_DO_CPU:
            model_cpu, t_dnn_cpu = train_dnn(
                X_dnn, Y_dnn, epochs=DNN_EPOCHS, batch=DNN_BATCH, lr=DNN_LR,
                layers=DNN_LAYERS, hidden=DNN_HIDDEN,
                device=torch.device('cpu'), final_lr=DNN_SCHED_FINAL_LR
            )
            y_sg_scaled = dnn_predict(model_cpu, xi_can, device=torch.device('cpu'))
            y_sg = (y_sg_scaled + 1.0) * 0.5 * (F_max - F_min) + F_min

            l2_dnn_cpu = np.zeros(K)
            for k in range(K):
                diff = F_sg[:, k] - y_sg[:, k]
                num = np.dot(diff**2, w_pos); den = np.dot(F_sg[:, k]**2, w_pos)
                l2_dnn_cpu[k] = np.sqrt(num / den) if den > 0 else np.nan
            l2_dnn_cpu_vec = l2_dnn_cpu

        # DNN GPU
        l2_dnn_gpu_vec = np.full(K, np.nan)
        t_dnn_gpu = np.nan
        if DNN_DO_GPU:
            device = torch.device('cuda')
            model_gpu, t_dnn_gpu = train_dnn(
                X_dnn, Y_dnn, epochs=DNN_EPOCHS, batch=DNN_BATCH, lr=DNN_LR,
                layers=DNN_LAYERS, hidden=DNN_HIDDEN,
                device=device, final_lr=DNN_SCHED_FINAL_LR
            )
            y_sg_scaled = dnn_predict(model_gpu, xi_can, device=device)
            y_sg = (y_sg_scaled + 1.0) * 0.5 * (F_max - F_min) + F_min

            l2_dnn_gpu = np.zeros(K)
            for k in range(K):
                diff = F_sg[:, k] - y_sg[:, k]
                num = np.dot(diff**2, w_pos); den = np.dot(F_sg[:, k]**2, w_pos)
                l2_dnn_gpu[k] = np.sqrt(num / den) if den > 0 else np.nan
            l2_dnn_gpu_vec = l2_dnn_gpu

        results['t_dnn_cpu'].append(t_dnn_cpu)
        results['t_dnn_gpu'].append(t_dnn_gpu)
        results['l2_dnn_cpu'].append(l2_dnn_cpu_vec)
        results['l2_dnn_gpu'].append(l2_dnn_gpu_vec)

        # Summary line
        print(f"Times (s): PD={t_pd:.2f}, PDR={t_pdr:.2f}, DNN_CPU={t_dnn_cpu:.2f}  DNN_GPU={t_dnn_gpu:.2f}")

    # =============================================================================
    # PLOTS (save to files)
    # =============================================================================
    sizes = np.array(results['train_sizes'])
    l2_pd   = np.array(results['l2_pd'])     # (S,K)
    l2_pdr  = np.array(results['l2_pdr'])
    l2_dnC  = np.array(results['l2_dnn_cpu'])
    l2_dnG  = np.array(results['l2_dnn_gpu'])

    # --- Per-output learning curves: PD, PDR, DNN (GPU if present else CPU) ---
    for k in range(K):
        plt.figure(figsize=(8,6))
        plt.loglog(sizes, l2_pd[:,k],  'o-', label='PD')
        plt.loglog(sizes, l2_pdr[:,k], 's-', label='PDR')
        if DNN_DO_GPU:
            plt.loglog(sizes, l2_dnG[:,k], '^-', label='DNN (GPU)')
        if DNN_DO_CPU:
            plt.loglog(sizes, l2_dnC[:,k], 'd-', label='DNN (CPU)')
        plt.xlabel('Training Sample Size')
        plt.ylabel('L2 Relative Error')
        plt.title(f'Learning Curve ({OUTPUT_NAMES[k]}) — shared train/test')
        plt.grid(True, which='both', alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"learning_curve_{OUTPUT_NAMES[k]}.png", dpi=150)
        plt.close()

    # --- Average L2 across outputs ---
    plt.figure(figsize=(9,6))
    plt.loglog(sizes, np.nanmean(l2_pd,  axis=1), 'o-', label='PD')
    plt.loglog(sizes, np.nanmean(l2_pdr, axis=1), 's-', label='PDR')
    if DNN_DO_GPU: plt.loglog(sizes, np.nanmean(l2_dnG, axis=1), '^-', label='DNN (GPU)')
    if DNN_DO_CPU: plt.loglog(sizes, np.nanmean(l2_dnC, axis=1), 'd-', label='DNN (CPU)')
    plt.xlabel('Training Sample Size'); plt.ylabel('Average L2 Relative Error')
    plt.title('Average Learning Curve — shared train/test')
    plt.grid(True, which='both', alpha=0.3); plt.legend(); plt.tight_layout()
    plt.savefig("learning_curve_average.png", dpi=150); plt.close()

    # --- Scalability: runtime vs training size (compare all) ---
    plt.figure(figsize=(10,6))
    plt.plot(sizes, results['t_cpu_pd'],          'o-', label='PD (CPU batched)')
    plt.plot(sizes, results['t_gpu_pd_seq'],      's-', label='PD (GPU seq-K)')
    plt.plot(sizes, results['t_gpu_pd_batched'],  '^-', label='PD (GPU batched)')
    plt.plot(sizes, results['t_pdr'],             'x-', label='PDR (GPU batched)')
    if DNN_DO_CPU: plt.plot(sizes, results['t_dnn_cpu'], 'd-', label='DNN (CPU)')
    if DNN_DO_GPU: plt.plot(sizes, results['t_dnn_gpu'], 'v-', label='DNN (GPU)')
    plt.xlabel('Training Sample Size'); plt.ylabel('Time (seconds)')
    plt.title('Scalability: runtime vs training size (shared train/test)')
    plt.grid(True, which='both', alpha=0.3); plt.legend(); plt.tight_layout()
    plt.savefig("scalability_runtime.png", dpi=150); plt.close()

    # --- Optional: speedup curves (GPU vs CPU) for DNN ---
    if DNN_DO_CPU and DNN_DO_GPU:
        cpu = np.array(results['t_dnn_cpu']); gpu = np.array(results['t_dnn_gpu'])
        valid = (~np.isnan(cpu)) & (~np.isnan(gpu)) & (gpu>0)
        plt.figure(figsize=(9,6))
        plt.plot(sizes[valid], cpu[valid]/gpu[valid], 'o-')
        plt.xlabel('Training Sample Size'); plt.ylabel('Speedup (×)')
        plt.title('DNN Speedup: CPU → GPU')
        plt.grid(True, which='both', alpha=0.3); plt.tight_layout()
        plt.savefig("dnn_speedup.png", dpi=150); plt.close()

    # --- Print summary ---
    print("\nSummary (average L2 at max size):")
    print("  PD:   ", float(np.nanmean(l2_pd[-1])))
    print("  PDR:  ", float(np.nanmean(l2_pdr[-1])))
    if DNN_DO_CPU: print("  DNN CPU:", float(np.nanmean(l2_dnC[-1])))
    if DNN_DO_GPU: print("  DNN GPU:", float(np.nanmean(l2_dnG[-1])))

if __name__ == "__main__":
    main()
