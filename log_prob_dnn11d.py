# log_prob_dnn11d.py
import numpy as np
from ensemble_da import P_Gaussian
from dnn_emulator_11d import DNN11D

_emul = None
def _fail_blob():
    return np.array([-np.inf] + [np.nan]*6, dtype=float)

def log_prob_dnn11d(xp, xtrue, L1_six, y_sig_six,
                    pmin_full, pmax_full, PMask, y_mask, LType=1):
    global _emul
    if _emul is None:
        _emul = DNN11D('models_11d/dnn11d_model.pt')

    x = xtrue.copy()
    pidx = np.nonzero(PMask)[0]
    x[pidx] = xp
    if np.any(x < pmin_full) or np.any(x > pmax_full):
        return -np.inf, _fail_blob()

    y_pred = _emul.predict(x[None, :])[0]

    sel = (np.asarray(y_mask, dtype=int) == 1)
    mu  = L1_six[sel]
    obs = y_pred[sel]
    Sig = np.diag((y_sig_six[sel] ** 2))
    log_like = P_Gaussian(obs, mu, Sig) if LType == 1 else 0.0

    blob = np.concatenate([[log_like], y_pred])
    return log_like, blob
