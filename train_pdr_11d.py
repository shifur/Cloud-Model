# train_pdr_11d.py
# CHANGES from previous version:
#   CHANGE 1: Import PDR_TARGET_STEPS and PDR_MAX_INNER_ITERS from config
#             (was causing ImportError — neither was defined in surrogate_config_11d.py)
#   CHANGE 2: Replace hardcoded R = ceil(2000/T_inner) with PDR_TARGET_STEPS from config
#   CHANGE 3: Replace hardcoded max_it with PDR_MAX_INNER_ITERS from config
#   Everything else is identical to the original file.

import os
import numpy as np
from scipy.special import legendre

from surrogate_config_11d import (
    set_global_seed, SEED, XNAMES, PMIN_11, PMAX_11,
    TRAIN_SIZE, PDR_POLY_LEVEL, PDR_EPS0,
    PDR_TARGET_STEPS,    # CHANGE 1: now defined in config (was missing → ImportError)
    PDR_MAX_INNER_ITERS, # CHANGE 1: now defined in config (was missing → ImportError)
)
from crm_eval_11d_six import run_cloud_11d_six

# ---------- helpers ----------
def map_to_canonical_11d(X):
    return 2.0 * (X - PMIN_11) / (PMAX_11 - PMIN_11) - 1.0

def multiidx_gen(d: int, rule, w: int, base=0, current=np.array([]), acc=np.array([])):
    if len(current) != d:
        i = base
        while rule(np.append(current, i)) <= w:
            acc = multiidx_gen(d, rule, w, base, np.append(current, i), acc)
            i += 1
    else:
        acc = np.vstack([acc, current]) if acc.size else current
    return acc

def compute_intrinsic_weights_legendre(Lambda: np.ndarray) -> np.ndarray:
    w = np.ones(Lambda.shape[0], dtype=float)
    for n in range(Lambda.shape[0]):
        for k in range(Lambda.shape[1]):
            deg = int(Lambda[n, k])
            w[n] *= np.sqrt(2.0 * deg + 1.0)
    return w

def build_design_legendre(Xi: np.ndarray, Lambda: np.ndarray) -> np.ndarray:
    m, d = Xi.shape
    N = Lambda.shape[0]
    A = np.ones((m, N), dtype=float)
    for n in range(N):
        for k in range(d):
            deg = int(Lambda[n, k])
            Pk = legendre(deg)
            A[:, n] *= np.sqrt(2.0 * deg + 1.0) * Pk(Xi[:, k])
    A /= np.sqrt(m)
    return A

def compute_table51_hparams(A: np.ndarray, m: int):
    normA2 = np.linalg.norm(A, ord=2)
    sigma = 1.0 / normA2
    tau   = 1.0 / normA2
    r     = np.exp(-1.0)
    T_in  = int(np.ceil((2.0 * normA2) / (r * np.sqrt(sigma * tau))))
    T_in  = max(1, T_in)
    s     = T_in / (2.0 * normA2)
    lam   = (np.sqrt(25.0 * m)) ** -1
    return lam, tau, sigma, r, T_in, s

# ---------- primal-dual (CPU, simple numpy) ----------
def algorithm_2_exact(A, B, w, lam, tau, sigma, T, C_init, L_init,
                      early_stop=False, tol=1e-10):
    m, N = A.shape
    C_bar = np.zeros(N, dtype=float)
    C = C_init.copy(); Lmb = L_init.copy()
    thr = tau * lam * w
    for n in range(T):
        P = C - tau * (A.T @ Lmb)
        C_next = np.sign(P) * np.maximum(np.abs(P) - thr, 0.0)
        Q = Lmb + sigma * (A @ (2.0 * C_next - C)) - sigma * B
        nQ = np.linalg.norm(Q)
        L_next = Q / nQ if nQ > 1.0 else Q
        C_bar = (n / (n + 1.0)) * C_bar + (1.0 / (n + 1.0)) * C_next
        if early_stop and n > 0:
            rel = np.linalg.norm(C_next - C) / max(np.linalg.norm(C), 1.0)
            if rel < tol: C, Lmb = C_next, L_next; break
        C, Lmb = C_next, L_next
    return C_bar

def algorithm_5_exact(A, B, w, lam, T_inner, R, eps_0, r, s,
                      tau, sigma, max_iter=10**9):
    m, N = A.shape
    C_tilde = np.zeros(N, dtype=float)
    eps_l = np.linalg.norm(B)
    total = 0
    for _ in range(R):
        eps_l = r * (eps_l + eps_0)
        a_l = s * eps_l
        if a_l > 1e-15:
            C_init = C_tilde / a_l if np.linalg.norm(C_tilde) > 0 else np.zeros(N, dtype=float)
            L_init = np.zeros(m, dtype=float)
            B_scaled = B / a_l
            C_res = algorithm_2_exact(A, B_scaled, w, lam, tau, sigma, T_inner, C_init, L_init)
            C_tilde = a_l * C_res
        total += T_inner
        if total >= max_iter: break
    return C_tilde

def main():
    set_global_seed(SEED)
    os.makedirs('models_11d', exist_ok=True)

    # 1) sample X and get Y @120
    X_train = np.column_stack([
        np.random.uniform(PMIN_11[j], PMAX_11[j], size=TRAIN_SIZE) for j in range(11)
    ])
    Y_train = run_cloud_11d_six(X_train)  # (m, 6)

    # 2) basis multi-index (HC rule — matches reference unified_trials.py)
    HCfunc = lambda x: np.prod(x + 1) - 1
    Lambda = multiidx_gen(11, HCfunc, PDR_POLY_LEVEL).astype(int)
    print(f"N_basis = {Lambda.shape[0]}  (PDR_POLY_LEVEL={PDR_POLY_LEVEL})")

    # 3) design + weights
    Xi = map_to_canonical_11d(X_train)
    A  = build_design_legendre(Xi, Lambda)            # (m, N_basis)
    w  = compute_intrinsic_weights_legendre(Lambda)   # (N_basis,)

    # 4) hyperparams — Table 5.1
    lam, tau, sigma, r, T_inner, s = compute_table51_hparams(A, TRAIN_SIZE)
    # CHANGE 2: use PDR_TARGET_STEPS from config instead of hardcoded 2000
    R      = max(1, int(np.ceil(PDR_TARGET_STEPS / T_inner)))
    # CHANGE 3: use PDR_MAX_INNER_ITERS from config instead of hardcoded value
    max_it = min(PDR_MAX_INNER_ITERS, R * T_inner)
    print(f"T_inner={T_inner}, R={R}, T_total={R*T_inner}, max_it={max_it}")

    # 5) solve per output
    root_m   = np.sqrt(TRAIN_SIZE)
    C_list   = []
    for k in range(6):
        B  = Y_train[:, k] / root_m
        Ck = algorithm_5_exact(
            A, B, w, lam, T_inner, R, PDR_EPS0, r, s,
            tau, sigma, max_iter=max_it
        )
        C_list.append(Ck)
    C_six = np.stack(C_list, axis=1)  # (N_basis, 6)

    # 6) save — keys match what pdr_emulator_11d.py expects
    np.savez(
        'models_11d/pdr11d_model.npz',
        Lambda   = Lambda,
        C_six    = C_six,
        pmin11   = PMIN_11,
        pmax11   = PMAX_11,
        m_train  = TRAIN_SIZE,
        xnames   = np.array(XNAMES, dtype=object)
    )
    print("Saved: models_11d/pdr11d_model.npz")

if __name__ == "__main__":
    main()
