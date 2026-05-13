# crm_eval_11d_six.py
import os
import numpy as np
from cloud_column_model import cloud_column_model
from surrogate_config_11d import EXPDIR, RUN_FILE, OUT_FILE, NAMELIST

# Output order per time block: [PCP, ACC, LWP, IWP, OLR, OSR]
IDX0_120 = 3 * 6  # start index for t=120 min (block #3, 0-based)

def run_cloud_11d_six(X11: np.ndarray) -> np.ndarray:
    """
    Input : X11 (m,11) parameters
    Return: Y (m,6) = [PCP, ACC, LWP, IWP, OLR, OSR] @ t=120 min
    """
    X11 = np.atleast_2d(np.asarray(X11, dtype=float))
    m = X11.shape[0]
    Y = np.zeros((m, 6), dtype=float)
    for i in range(m):
        crm = cloud_column_model.CRM1DWrap(
            os.path.join(EXPDIR, RUN_FILE),
            os.path.join(EXPDIR, OUT_FILE),
            os.path.join(EXPDIR, NAMELIST),
            params=np.ndarray.tolist(X11[i])
        )
        out, ok = crm()
        if not ok:
            raise RuntimeError("CRM1D run failed for sample index %d" % i)
        y = np.asarray(out, dtype=float)
        Y[i, :] = y[IDX0_120:IDX0_120+6]
    return Y
