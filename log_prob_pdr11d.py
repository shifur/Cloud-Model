# log_prob_pdr11d.py
import numpy as np
from pdr_emulator_11d import PDR11D

_emul = None

# Caches to avoid recomputing every call
_cache = {
    "y_mask_key": None,   # tuple(y_mask)
    "sel_idx": None,      # np.where(sel)[0]
    "log_norm": None,     # constant part of diag Gaussian logpdf
    "inv_var": None       # 1/variance for selected outputs
}

def _fail_blob():
    return np.array([-np.inf] + [np.nan]*6, dtype=float)

def _prep_mask_cache(y_mask, y_sig_six):
    """
    Cache selection indices + constants for diagonal Gaussian likelihood.
    """
    y_mask_key = tuple(np.asarray(y_mask, dtype=int).tolist())
    if _cache["y_mask_key"] == y_mask_key:
        return

    sel = (np.asarray(y_mask, dtype=int) == 1)
    sel_idx = np.where(sel)[0]

    sig = np.asarray(y_sig_six, dtype=float)[sel_idx]
    var = sig * sig
    inv_var = 1.0 / var

    # log N(obs|mu,diag(var)) = -0.5*(d*log(2pi) + sum(log(var)) + sum((obs-mu)^2/var))
    d = sel_idx.size
    log_norm = -0.5 * (d * np.log(2.0 * np.pi) + np.sum(np.log(var)))

    _cache["y_mask_key"] = y_mask_key
    _cache["sel_idx"] = sel_idx
    _cache["log_norm"] = float(log_norm)
    _cache["inv_var"] = inv_var

def log_prob_pdr11d(
    xp, xtrue, L1_six, y_sig_six,
    pmin_full, pmax_full, PMask, y_mask, LType=1
):
    """
    xp: (nxp,) here nxp=11 (vary all)
    L1_six: (6,) truth at 120
    y_sig_six: (6,) sigmas
    y_mask: (6,) 0/1
    """
    global _emul

    if _emul is None:
        _emul = PDR11D('models_11d/pdr11d_model.npz')

    # Cache mask selection + diag Gaussian constants once per y_mask
    if LType == 1:
        _prep_mask_cache(y_mask, y_sig_six)

    x = xtrue.copy()
    pidx = np.nonzero(PMask)[0]
    x[pidx] = xp

    # bounds
    if np.any(x < pmin_full) or np.any(x > pmax_full):
        return -np.inf, _fail_blob()

    y_pred = _emul.predict(x[None, :])[0]

    if LType == 1:
        sel_idx = _cache["sel_idx"]
        mu  = np.asarray(L1_six, dtype=float)[sel_idx]
        obs = np.asarray(y_pred, dtype=float)[sel_idx]
        diff = obs - mu

        # Fast diagonal Gaussian log-likelihood
        log_like = _cache["log_norm"] - 0.5 * np.sum((diff * diff) * _cache["inv_var"])
    else:
        # If you ever add non-Gaussian likelihoods, handle here
        log_like = 0.0

    blob = np.concatenate([[log_like], y_pred])
    return float(log_like), blob
