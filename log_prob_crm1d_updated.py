"""
Compute the log-posterior for the 1D CRM model:
log p(x, y) = log p(x) + log p(y|x)

Returns:
  (log_posterior, blob)
  where blob = [log_likelihood, *y_forward] (length 1 + ny) even on failure
"""

import os
import sys
import uuid
import numpy as np
from cloud_column_model import cloud_column_model
from ensemble_da import P_Gaussian   # function

# ---------- Helpers ----------
def _fail_blob(ny: int):
    """Return (-inf, blob) with fixed shape so emcee never crashes."""
    blob = np.concatenate(([-np.inf], np.full(ny, np.nan, dtype=float)))
    return -np.inf, blob

# Per-process working directory cache
_WORK_DIR = None
def _get_worker_dir(base_dir: str) -> str:
    """
    Returns a per-process scratch directory: <base_dir>/tmp/w<PID>_<GUID>
    Created once per process; reused across calls.
    """
    global _WORK_DIR
    if _WORK_DIR is None:
        os.makedirs(os.path.join(base_dir, "tmp"), exist_ok=True)
        _WORK_DIR = os.path.join(base_dir, "tmp", f"w{os.getpid()}_{uuid.uuid4().hex[:8]}")
        os.makedirs(_WORK_DIR, exist_ok=True)
    return _WORK_DIR

# ---------- Main API ----------
def log_prob_crm1d(
    xp, xtrue, P1, P2, L1, L2, PType, LType,
    pmin, pmax, PMask, LMask, expdir='.'
):
    # -------- Map subset xp into full x --------
    pidx = np.nonzero(PMask)          # indices of perturbed params
    x = xtrue.copy()
    x[pidx] = xp

    nx = len(x)
    ny = len(L1)

    # -------- Bounds checks (user/MCMC) --------
    if pmin is not None:
        for n in range(nx):
            if x[n] < pmin[n]:
                return _fail_blob(ny)
    if pmax is not None:
        for n in range(nx):
            if x[n] > pmax[n]:
                return _fail_blob(ny)

    # -------- Extra CRM-internal bounds (ag,bg) to avoid failed runs --------
    # Indices: ag=2, bg=3 in your x vector
    if nx >= 4:
        ag, bg = x[2], x[3]
        if (ag < 20.0) or (ag > 1180.0) or (bg < 0.01666667) or (bg > 0.98333333):
            return _fail_blob(ny)

    # -------- Types / masks defaults --------
    if PType is None:
        PType = 0
    if LType is None:
        LType = 1
    if PMask is None:
        PMask = np.ones(nx)
    if LMask is None:
        LMask = np.ones(ny)

    # -------- Prior --------
    if PType == 0:
        # Uniform prior inside bounds → log(1) = 0
        log_prior = 0.0
    elif PType == 1:
        xmask = x * PMask
        P1mask = P1 * PMask
        log_prior = P_Gaussian(xmask, P1mask, P2)
    else:
        sys.exit("Unknown Prior Type in log_prob_crm1d.py")

    # -------- Forward model (CRM) --------
    # Use a per-process scratch dir + unique filenames to avoid races
    run_dir = _get_worker_dir(expdir)
    uid = uuid.uuid4().hex[:8]
    input_file    = os.path.join(run_dir, f'run_one_crm1d_{uid}.txt')
    output_file   = os.path.join(run_dir, f'crm1d_output_{uid}.txt')
    namelist_file = os.path.join(expdir, 'namelist_3h_t30-180.f90')  # static is OK

    p_vec = np.ndarray.tolist(x)
    try:
        crm1d = cloud_column_model.CRM1DWrap(input_file, output_file, namelist_file, params=p_vec)
        model_output, crm_status = crm1d()

        # Optionally clean up the temp I/O files if the wrapper leaves them
        for f in (input_file, output_file):
            try:
                if os.path.exists(f):
                    os.remove(f)
            except Exception:
                pass

        if not crm_status:
            return _fail_blob(ny)

        y = np.array(model_output, dtype=float)
        if y.shape[0] != ny:
            return _fail_blob(ny)
    except Exception:
        # e.g., output file missing; never crash emcee
        return _fail_blob(ny)

    # -------- Likelihood --------
    if LType == 0:
        log_likelihood = 0.0
    elif LType == 1:
        idx = np.where(LMask > 0.5)[0]
        if idx.size == 0:
            log_likelihood = 0.0
        else:
            y_sel  = y[idx]
            L1_sel = L1[idx]
            if L2.ndim == 1:
                S = np.diag(L2[idx]**2.0)
            else:
                S = L2[np.ix_(idx, idx)]
            log_likelihood = P_Gaussian(y_sel, L1_sel, S)
    else:
        sys.exit("Unknown Likelihood Type in log_prob_crm1d.py")

    # -------- Return --------
    blob = np.concatenate(([log_likelihood], y))   # [log_like, y...]
    return (log_prior + log_likelihood), blob
