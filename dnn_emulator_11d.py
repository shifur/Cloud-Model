# dnn_emulator_11d.py
"""
Loads the trained DNN checkpoint and exposes a predict() method.
Architecture is read directly from the checkpoint (dnn_layers, dnn_nodes)
so this file is always consistent with whatever train_dnn_11d.py saved.
"""

import numpy as np
import torch
from torch import nn

from surrogate_config_11d import PMIN_11, PMAX_11, map_to_canonical_11d


# =============================================================================
#  DNN definition — must match train_dnn_11d.py exactly.
#  Import the class directly from train_dnn_11d to guarantee consistency.
# =============================================================================
try:
    from train_dnn_11d import DNN
except ImportError:
    # Fallback self-contained definition (keep in sync with train_dnn_11d.py)
    class DNN(nn.Module):
        def __init__(self, in_dim=11, out_dim=6, layers=8, nodes=50):  # matches unified_trials.py
            super().__init__()
            seq = [nn.Linear(in_dim, nodes), nn.Tanh()]
            for _ in range(layers):          # range(layers) — not range(layers-1)
                seq += [nn.Linear(nodes, nodes), nn.Tanh()]
            seq += [nn.Linear(nodes, out_dim)]
            self.net = nn.Sequential(*seq)

        def forward(self, x):
            return self.net(x)


# =============================================================================
#  Emulator wrapper
# =============================================================================
class DNN11D:
    """
    Loads a trained checkpoint and wraps predict() in physical units.

    Parameters
    ----------
    ckpt_path : str
        Path to .pt file saved by train_dnn_11d.py.
    device : str or None
        'cuda', 'cpu', or None (auto).
    """
    def __init__(self, ckpt_path: str = 'models_11d/dnn11d_model.pt',
                 device: str | None = None):
        map_loc = 'cpu' if device is None else device
        ckpt = torch.load(ckpt_path, map_location=map_loc)

        # FIX: read architecture from checkpoint (was hardcoded 6/128 before)
        layers = int(ckpt.get('dnn_layers', 8))
        nodes  = int(ckpt.get('dnn_nodes', 50))

        self.model = DNN(11, 6, layers, nodes)
        self.model.load_state_dict(ckpt['state_dict'])
        self.model.eval()

        self.Ymin   = np.asarray(ckpt['Ymin'],   dtype=np.float64)
        self.Ymax   = np.asarray(ckpt['Ymax'],   dtype=np.float64)
        self.pmin11 = np.asarray(ckpt['pmin11'], dtype=np.float64)
        self.pmax11 = np.asarray(ckpt['pmax11'], dtype=np.float64)
        self.m_train = int(ckpt.get('m_train', 1))

        self._span = np.where(
            (self.Ymax - self.Ymin) == 0.0, 1.0, (self.Ymax - self.Ymin)
        )

        # Verify checkpoint bounds match current config
        if not (np.allclose(self.pmin11, PMIN_11) and
                np.allclose(self.pmax11, PMAX_11)):
            import warnings
            warnings.warn(
                "Checkpoint bounds do not match surrogate_config_11d.py PMIN_11/PMAX_11. "
                "Retrain the DNN after fixing bounds.",
                RuntimeWarning, stacklevel=2
            )

        if device is None:
            self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self._device = torch.device(device)
        self.model.to(self._device)

    def predict(self, X11: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        X11 : (m, 11) or (11,) physical-space inputs

        Returns
        -------
        Y : (m, 6) predictions in physical units
        """
        X = np.atleast_2d(np.asarray(X11, dtype=np.float32))

        # Use checkpoint bounds for canonical mapping (consistent with training)
        Xi = (2.0 * (X - self.pmin11) / (self.pmax11 - self.pmin11) - 1.0).astype(np.float32)

        with torch.no_grad():
            t = torch.from_numpy(Xi).to(self._device)
            y_scaled = self.model(t).cpu().numpy()

        Y = (y_scaled + 1.0) * 0.5 * self._span + self.Ymin
        return Y
