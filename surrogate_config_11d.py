# surrogate_config_11d.py
# ============================================================
#  SINGLE SOURCE OF TRUTH for bounds, DNN architecture,
#  true parameter vector, and output settings.
#  Import from here in ALL scripts — never redefine locally.
# ============================================================

import os
import numpy as np
import random

# --------------------------
#   Reproducibility
# --------------------------
SEED = 1234

def set_global_seed(seed: int = SEED):
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass
    np.random.seed(seed)
    random.seed(seed)


# --------------------------
#   Parameter space (11-D)
# --------------------------
XNAMES = ['as', 'bs', 'ag', 'bg', 'N0r', 'N0s', 'N0g', 'rhos', 'rhog', 'qc0', 'qi0']

# True/reference parameter vector — used by MCMC as likelihood centre and p0 anchor.
# Also used by unified_trials.py to confirm both surrogates invert the same problem.
X_TRUE_11D = np.array(
    [200.0, 0.3, 400.0, 0.4, 0.5, 0.5, 0.5, 0.2, 0.4, 1.0e-3, 6.0e-4],
    dtype=float
)

# ---- Canonical bounds (AUTHORITATIVE) ----
# Previously: unified_trials.py used different values for as, ag, bg, rhos, rhog.
# Now both PD/PDR and DNN/MCMC pipelines use EXACTLY these bounds.
PMIN_11 = np.array(
    [50.0, 0.10, 50.0, 0.10, 0.05, 0.05, 0.05, 0.05, 0.05, 1.0e-4, 2.0e-6],
    dtype=float
)
PMAX_11 = np.array(
    [1000.0, 1.0, 1200.0, 0.90, 5.0, 2.5, 2.5, 1.0, 1.0, 2.0e-3, 1.0e-3],
    dtype=float
)

# Handy aliases so callers don't need to rename
pmin = PMIN_11
pmax = PMAX_11

# Handy indices
IDX_AG = XNAMES.index('ag')   # 2
IDX_BG = XNAMES.index('bg')   # 3

# --------------------------
#   Outputs @ t = 120 min
# --------------------------
YNAMES_SIX = ['PCP', 'ACC', 'LWP', 'IWP', 'OLR', 'OSR']
K_OUTPUTS   = 6

# Per-output observational σ (not σ²)
Y_SIG_SIX = np.array([2.0, 5.0, 0.5, 1.0, 5.0, 5.0], dtype=float)

# Output masks
YMASK_FIG7 = np.array([1, 0, 1, 1, 0, 0], dtype=int)   # no radiation: PCP, LWP, IWP
YMASK_FIG9 = np.array([0, 0, 0, 0, 1, 1], dtype=int)   # radiation:    OLR, OSR

# CRM output column indices in the raw output array
CRM_OUTPUT_COL_START = 18   # columns 18..23 → 6 outputs
CRM_OUTPUT_COL_END   = 24


# --------------------------
#   DNN hyperparameters (AUTHORITATIVE)
# --------------------------
# These values are used by train_dnn_11d.py, unified_trials.py,
# and dnn_emulator_11d.py.  Change here; nowhere else.
DNN_LAYERS   = 8      # number of hidden layers  (matches unified_trials.py)
DNN_NODES    = 50     # nodes per hidden layer   (matches unified_trials.py)
DNN_LR       = 1e-3   # initial Adam learning rate
DNN_BATCH    = 32     # batch size               (matches unified_trials.py)
DNN_EPOCHS   = 20000  # training epochs          (matches unified_trials.py)
DNN_LR_FINAL = 5e-7   # ExponentialLR decays from DNN_LR to this over DNN_EPOCHS


# --------------------------
#   Training sizes
# --------------------------
TRAIN_SIZE       = 5000              # matches unified_trials.py training size
TRAIN_SIZES_LIST = list(range(100, 5001, 100))   # [100, 200, ..., 5000]  matches paper Table 4


# --------------------------
#   CS / PDR hyperparameters
# --------------------------
# CHANGE 1: PDR_POLY_LEVEL corrected from 40 to 20 — matches reference code (p_poly=20)
#           and train_pdr_11d.py. Level 40 was wrong; 20 gives N_basis=4291 for d=11.
PDR_POLY_LEVEL        = 20    # hyperbolic-cross level  (N_basis = 4291 for d=11)

# CHANGE 2: PDR_TARGET_STEPS corrected from 5000 to 2000 — matches reference code
#           (TARGET_TOTAL_STEPS = 2000) and the hardcoded value in train_pdr_11d.py.
PDR_TARGET_STEPS      = 2000  # Ttarget; R = ceil(PDR_TARGET_STEPS / T_inner)

# CHANGE 3: PDR_MAX_INNER_ITERS added — was imported by train_pdr_11d.py but
#           never defined here, causing ImportError on every run.
PDR_MAX_INNER_ITERS   = 2000  # cap on R * T_inner — consistent with PDR_TARGET_STEPS

PDR_EPS0              = 1e-6
PD_EARLY_STOP         = False
PD_EARLY_STOP_TOL     = 1e-10

# MCMC settings — shared by both DNN and PDR MCMC pipelines
# CHANGE 4: centralise MCMC knobs here so both scripts stay in sync
MCMC_NWALK = 32       # number of emcee walkers  (matches mcmc_with_dnn_updated_new.py)
MCMC_NBURN = 300    # burn-in steps             (matches mcmc_with_dnn_updated_new.py)
MCMC_NMCMC = 3000   # production steps          (matches mcmc_with_dnn_updated_new.py)

# Tasmanian sparse grid for test set
SG_LEVEL = 4
SG_RULE  = "clenshaw-curtis"

# Number of statistical trials per training size
NUM_TRIALS = 20


# --------------------------
#   CRM file locations
# --------------------------
EXPDIR   = './cloud_column_model/'
RUN_FILE = 'run_one_crm1d.txt'
OUT_FILE = 'crm1d_output.txt'
NAMELIST = 'namelist_3h_t30-180.f90'

INPUT_FILE_FULL  = os.path.join(EXPDIR, RUN_FILE)
OUTPUT_FILE_FULL = os.path.join(EXPDIR, OUT_FILE)
NAMELIST_FULL    = os.path.join(EXPDIR, NAMELIST)


# --------------------------
#   Output directories
# --------------------------
os.makedirs('models_11d',    exist_ok=True)
os.makedirs('output',        exist_ok=True)
os.makedirs('plots',         exist_ok=True)
os.makedirs('trials_results', exist_ok=True)


# --------------------------
#   Convenience helpers
# --------------------------
def map_to_canonical_11d(X: np.ndarray) -> np.ndarray:
    """Affine map from physical [PMIN_11, PMAX_11] to [-1,1]^11."""
    X = np.asarray(X, dtype=float)
    return 2.0 * (X - PMIN_11) / (PMAX_11 - PMIN_11) - 1.0


def map_to_physical_11d(Xi: np.ndarray) -> np.ndarray:
    """Inverse: [-1,1]^11 → physical domain."""
    Xi = np.asarray(Xi, dtype=float)
    return 0.5 * (Xi + 1.0) * (PMAX_11 - PMIN_11) + PMIN_11


def ensure_within_bounds(X: np.ndarray) -> np.ndarray:
    """Clip to [PMIN_11, PMAX_11]."""
    return np.clip(X, PMIN_11, PMAX_11)
