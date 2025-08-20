#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gpu_batched_sr_lasso.py

GPU-batched SR-LASSO solvers (PD / PDR) with true matrix-wise batching.

Features
--------
- Dual backend (CuPy GPU / NumPy CPU) with safe fallback
- True batching across outputs: shapes (N, K) and (m, K)
- Minimal host<->device transfers
- Stable float() casts on norms for GPU conditionals
- Configurable dtype (float32 default)

IMPORTANT: Normalization
------------------------
This module DOES NOT normalize inputs. If you want 1/sqrt(m) scaling, pass in
A_scaled = A / sqrt(m) and B_scaled = B / sqrt(m) from the calling code.
(Your main script already does this.)

Public API
----------
solve_all_outputs_gpu_batch(A, F_train, weights, hyperparams, method='pd', dtype=np.float32)
  -> returns coefficients X with shape (N, K) on CPU (NumPy)

Hyperparameters (expected keys in `hyperparams`)
------------------------------------------------
lambda, tau, sigma, r, T_inner, s, R, T_pd, run_on_gpu (bool)
"""

from __future__ import annotations
import numpy as np
import time

# -----------------------------------------------------------------------------
# Backend setup
# -----------------------------------------------------------------------------
try:
    import cupy as cp
    GPU_AVAILABLE = True
    print("CuPy available — GPU acceleration enabled")
except Exception:
    cp = None
    GPU_AVAILABLE = False
    print("CuPy not available — using CPU")

DTYPE_DEFAULT = np.float32   # change to np.float64 for validation

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def is_gpu_array(a) -> bool:
    return (GPU_AVAILABLE and isinstance(a, cp.ndarray))

def to_gpu(a, dtype=DTYPE_DEFAULT):
    if GPU_AVAILABLE:
        return cp.asarray(a, dtype=dtype)
    return np.asarray(a, dtype=dtype)

def to_cpu(a):
    return cp.asnumpy(a) if is_gpu_array(a) else a

def backend_for(*arrays):
    """Return xp = cp or np depending on whether any array is on GPU."""
    if GPU_AVAILABLE and any(is_gpu_array(a) for a in arrays if a is not None):
        return cp
    return np

# -----------------------------------------------------------------------------
# Algorithm 2: Primal-Dual (batched over K outputs)
# -----------------------------------------------------------------------------
def algorithm_2_exact_batched(
    A, B, w, lambda_param, tau, sigma, T,
    C_init=None, Lambda_init=None,
    early_stop: bool = False,
    early_stop_tol: float = 1e-8,
    dtype=DTYPE_DEFAULT,
):
    """
    Batched PD for decoupled SR-LASSO across K outputs.

    Shapes
    ------
    A: (m, N)
    B: (m, K)
    w: (N,)
    returns C_bar: (N, K)
    """
    xp = backend_for(A, B, w)
    on_gpu = (xp is cp)

    # Convert inputs to desired backend/dtype
    A = to_gpu(A, dtype) if on_gpu else np.asarray(A, dtype=dtype)
    B = to_gpu(B, dtype) if on_gpu else np.asarray(B, dtype=dtype)
    w = to_gpu(w, dtype) if on_gpu else np.asarray(w, dtype=dtype)

    m, N = A.shape
    K = B.shape[1]

    C = (xp.zeros((N, K), dtype=dtype) if C_init is None
         else (to_gpu(C_init, dtype) if on_gpu else np.asarray(C_init, dtype=dtype)))
    Lambda = (xp.zeros((m, K), dtype=dtype) if Lambda_init is None
              else (to_gpu(Lambda_init, dtype) if on_gpu else np.asarray(Lambda_init, dtype=dtype)))
    C_bar = xp.zeros_like(C)

    thr_base = (tau * float(lambda_param) * w).reshape(-1, 1)  # (N,1) broadcast to K
    AT = A.T

    for n in range(int(T)):
        # x-update (soft threshold)
        P = C - tau * (AT @ Lambda)                # (N,K)
        C_next = xp.sign(P) * xp.maximum(xp.abs(P) - thr_base, 0.0)

        # y-update: column-wise projection onto unit l2 ball
        Q = Lambda + sigma * (A @ (2.0 * C_next - C)) - sigma * B   # (m,K)
        norms = xp.linalg.norm(Q, axis=0)                           # (K,)
        scale = xp.maximum(1.0, norms)                              # (K,)
        Lambda_next = Q / scale

        # running average
        C_bar = (n / (n + 1.0)) * C_bar + (1.0 / (n + 1.0)) * C_next

        if early_stop and n > 0:
            denom = max(float(xp.linalg.norm(C)), 1.0)
            rel_change = float(xp.linalg.norm(C_next - C)) / denom
            if rel_change < early_stop_tol:
                C = C_next
                Lambda = Lambda_next
                break

        C = C_next
        Lambda = Lambda_next

    return to_cpu(C_bar)

# -----------------------------------------------------------------------------
# Algorithm 5: Restarted Primal-Dual (batched)
# -----------------------------------------------------------------------------
def algorithm_5_exact_batched(
    A, B, w, lambda_param, T, R, eps_0, r, s,
    tau=None, sigma=None, max_iter=10**9,
    pd_early_stop: bool = False,
    pd_early_stop_tol: float = 1e-8,
    dtype=DTYPE_DEFAULT,
):
    """
    Batched Restarted PD (PDR). Calls Algorithm 2 in inner loops.

    A: (m,N), B: (m,K)
    returns C_tilde: (N,K)
    """
    # Use CPU for spectral norm to be robust and reproducible
    normA2 = float(np.linalg.norm(to_cpu(A), ord=2)) if (tau is None or sigma is None) else None
    tau_eff = 1.0 / normA2 if tau is None else float(tau)
    sigma_eff = 1.0 / normA2 if sigma is None else float(sigma)

    xp = backend_for(A, B, w)
    on_gpu = (xp is cp)

    N = A.shape[1]
    K = B.shape[1]
    C_tilde = to_gpu(np.zeros((N, K), dtype=dtype), dtype) if on_gpu else np.zeros((N, K), dtype=dtype)

    total_iters = 0
    eps_l = float(np.linalg.norm(to_cpu(B)))  # global scalar norm

    for _ in range(int(R)):
        eps_l = float(r) * (eps_l + float(eps_0))
        a_l = float(s) * eps_l

        if a_l > 1e-15:
            C_init_scaled = C_tilde / a_l if float(xp.linalg.norm(C_tilde)) > 0 else C_tilde
            Lambda_init_scaled = to_gpu(np.zeros((A.shape[0], K), dtype=dtype), dtype) if on_gpu else np.zeros((A.shape[0], K), dtype=dtype)
            B_scaled = B / a_l

            C_result = algorithm_2_exact_batched(
                A, B_scaled, w, lambda_param, tau_eff, sigma_eff, int(T),
                C_init=to_cpu(C_init_scaled), Lambda_init=to_cpu(Lambda_init_scaled),
                early_stop=pd_early_stop, early_stop_tol=pd_early_stop_tol, dtype=dtype,
            )
            C_tilde = to_gpu(C_result, dtype) * a_l if on_gpu else np.asarray(C_result, dtype=dtype) * a_l
            total_iters += int(T)
        else:
            total_iters += int(T)

        if total_iters >= int(max_iter):
            break

    return to_cpu(C_tilde)

# -----------------------------------------------------------------------------
# TRUE GPU BATCH DRIVER (minimal transfers)
# -----------------------------------------------------------------------------
def solve_all_outputs_gpu_batch(
    A, F_train, weights, hyperparams, method: str = 'pd', dtype=DTYPE_DEFAULT
):
    """
    True batch solve for all K outputs (decoupled SR-LASSO solved matrix-wise).

    Parameters
    ----------
    A : (m, N) NumPy or CuPy array (pass pre-normalized if desired)
    F_train : (m, K) NumPy or CuPy array (pass pre-normalized if desired)
    weights : (N,) NumPy or CuPy array
    hyperparams : dict
        Keys: 'lambda','tau','sigma','r','T_inner','s','R','T_pd','run_on_gpu'
    method : 'pd' or 'pdr'
    dtype : numpy dtype (np.float32 default)

    Returns
    -------
    X : (N, K) NumPy array (CPU) with coefficients.
    """
    lam   = float(hyperparams['lambda'])
    tau   = float(hyperparams['tau'])
    sigma = float(hyperparams['sigma'])
    run_on_gpu = bool(hyperparams.get('run_on_gpu', True)) and GPU_AVAILABLE

    # Move data at most once
    if run_on_gpu:
        A_dev = to_gpu(A, dtype)
        B_dev = to_gpu(F_train, dtype)
        w_dev = to_gpu(weights, dtype)
    else:
        A_dev = np.asarray(A, dtype=dtype)
        B_dev = np.asarray(F_train, dtype=dtype)
        w_dev = np.asarray(weights, dtype=dtype)

    if method.lower() == 'pd':
        T_pd = int(hyperparams['T_pd'])
        X = algorithm_2_exact_batched(
            A_dev, B_dev, w_dev, lam, tau, sigma, T_pd,
            C_init=None, Lambda_init=None, early_stop=False, dtype=dtype,
        )
    elif method.lower() == 'pdr':
        T_inner = int(hyperparams['T_inner'])
        R       = int(hyperparams['R'])
        r       = float(hyperparams['r'])
        s       = float(hyperparams['s'])
        T_pd    = int(hyperparams['T_pd'])
        X = algorithm_5_exact_batched(
            A_dev, B_dev, w_dev, lam, T_inner, R, eps_0=1e-6, r=r, s=s,
            tau=tau, sigma=sigma, max_iter=T_pd, pd_early_stop=False, dtype=dtype,
        )
    else:
        raise ValueError(f"Unknown method: {method!r}")

    # Ensure CPU output for downstream code
    return to_cpu(X)

# -----------------------------------------------------------------------------
# GPU utils (optional)
# -----------------------------------------------------------------------------
def gpu_memory_info():
    if not GPU_AVAILABLE:
        print("GPU not available"); return
    mp = cp.get_default_memory_pool()
    used = mp.used_bytes() / 1024**3
    print(f"GPU memory used: {used:.2f} GB")

def clear_gpu_memory():
    if not GPU_AVAILABLE:
        return
    mp = cp.get_default_memory_pool()
    pmp = cp.get_default_pinned_memory_pool()
    mp.free_all_blocks()
    pmp.free_all_blocks()

# -----------------------------------------------------------------------------
# Minimal self-test (mirrors main: scales A and B by 1/sqrt(m))
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    print("Self-test: gpu_batched_sr_lasso.py")
    np.random.seed(0)
    m, N, K = 128, 256, 6

    # Synthetic design and sparse truth
    A = np.random.randn(m, N).astype(DTYPE_DEFAULT)
    X_true = np.zeros((N, K), dtype=DTYPE_DEFAULT)
    for k in range(K):
        nz = np.random.choice(N, size=20, replace=False)
        X_true[nz, k] = np.random.randn(20).astype(DTYPE_DEFAULT)
    B = (A @ X_true + 0.01 * np.random.randn(m, K)).astype(DTYPE_DEFAULT)
    w = np.ones(N, dtype=DTYPE_DEFAULT)

    # --- Apply 1/sqrt(m) normalization (same as your main) ---
    scale = 1.0 / np.sqrt(m)
    A = A * scale
    B = B * scale

    # Table 5.1-like hyperparams computed on *scaled* A
    normA2 = float(np.linalg.norm(A, ord=2))
    tau = sigma = 1.0 / normA2
    lam = (np.sqrt(25.0 * m)) ** -1
    r = np.exp(-1.0)
    T_inner = int(np.ceil((2.0 * normA2) / (r * np.sqrt(tau * sigma))))
    s = T_inner / (2.0 * normA2)
    R = 2
    T_pd = R * T_inner

    hps = {
        "lambda": lam, "tau": tau, "sigma": sigma, "r": r,
        "T_inner": T_inner, "s": s, "R": R, "T_pd": T_pd,
        "run_on_gpu": GPU_AVAILABLE,
    }

    t0 = time.time()
    X_pd = solve_all_outputs_gpu_batch(A, B, w, hps, method='pd', dtype=DTYPE_DEFAULT)
    t1 = time.time()
    X_pdr = solve_all_outputs_gpu_batch(A, B, w, hps, method='pdr', dtype=DTYPE_DEFAULT)
    t2 = time.time()

    print(f"PD  time: {t1 - t0:.3f}s,  ||X||={np.linalg.norm(X_pd):.3f}")
    print(f"PDR time: {t2 - t1:.3f}s,  ||X||={np.linalg.norm(X_pdr):.3f}")
