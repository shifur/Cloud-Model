# train_dnn_11d.py
"""
Train the DNN surrogate for the 11D cloud microphysics problem.
Architecture and hyperparameters are imported from surrogate_config_11d.py
so that train_dnn_11d, unified_trials, and dnn_emulator_11d all use
exactly the same model definition.
"""

import os
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from surrogate_config_11d import (
    set_global_seed, SEED,
    XNAMES, PMIN_11, PMAX_11, X_TRUE_11D,
    TRAIN_SIZE,
    DNN_LAYERS, DNN_NODES, DNN_LR, DNN_BATCH, DNN_EPOCHS, DNN_LR_FINAL,
    map_to_canonical_11d,
)
from crm_eval_11d_six import run_cloud_11d_six


# =============================================================================
#  DNN definition  — SINGLE definition; imported by dnn_emulator_11d.py too
# =============================================================================
class DNN(nn.Module):
    """
    Fully-connected network with `layers` hidden layers each of `nodes` nodes.
    Architecture is identical to unified_trials.py's DNN class:
      - 1 input  layer  : Linear(11 → nodes) + Tanh
      - `layers` hidden : Linear(nodes → nodes) + Tanh   (loop range(layers))
      - 1 output layer  : Linear(nodes → 6)
    Total Tanh layers = layers + 1.
    """
    def __init__(self, in_dim: int = 11, out_dim: int = 6,
                 layers: int = DNN_LAYERS, nodes: int = DNN_NODES):  # 8 layers, 50 nodes
        super().__init__()
        seq = [nn.Linear(in_dim, nodes), nn.Tanh()]
        for _ in range(layers):                           # FIX: was range(layers-1) in old file
            seq += [nn.Linear(nodes, nodes), nn.Tanh()]
        seq += [nn.Linear(nodes, out_dim)]
        self.net = nn.Sequential(*seq)

    def forward(self, x):
        return self.net(x)


def _init_weights(m):
    """Xavier uniform init for Linear layers — matches unified_trials.py."""
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)


# =============================================================================
#  Training entry point
# =============================================================================
def main():
    set_global_seed(SEED)
    os.makedirs('models_11d', exist_ok=True)

    # ---- 0. CUDA diagnostics (always printed first) ----
    print("==== Torch/CUDA Diagnostics ====")
    print(f"torch version            : {torch.__version__}")
    print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
    print(f"torch.version.cuda       : {getattr(torch.version, 'cuda', None)}")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"  device {i}: {props.name}, {props.total_memory/1e9:.1f} GB")
        torch.backends.cudnn.benchmark = True
        print("cudnn.benchmark          : True")
    else:
        print("CUDA not available — training on CPU")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    DNN_USE_AMP = True
    use_cuda_amp = DNN_USE_AMP and (device.type == 'cuda')
    print(f"Selected device          : {device}")
    print(f"AMP (mixed precision)    : {use_cuda_amp}")
    print("================================")

    # ---- 1. Generate training data ----
    X_train = np.column_stack([
        np.random.uniform(PMIN_11[j], PMAX_11[j], size=TRAIN_SIZE)
        for j in range(11)
    ]).astype(np.float32)
    print(f"Running CRM for {TRAIN_SIZE} training samples...")
    Y_train = run_cloud_11d_six(X_train).astype(np.float32)

    # ---- 2. Normalise inputs and outputs ----
    Xi = map_to_canonical_11d(X_train).astype(np.float32)

    Ymin = Y_train.min(axis=0)
    Ymax = Y_train.max(axis=0)
    span = np.where((Ymax - Ymin) == 0.0, 1.0, (Ymax - Ymin))
    Y_scaled = (2.0 * (Y_train - Ymin) / span - 1.0).astype(np.float32)

    ds = TensorDataset(torch.from_numpy(Xi), torch.from_numpy(Y_scaled))
    dl = DataLoader(ds, batch_size=DNN_BATCH, shuffle=True, drop_last=False)

    # ---- 3. Build model ----
    print(f"Architecture: {DNN_LAYERS} hidden layers × {DNN_NODES} nodes, "
          f"{DNN_EPOCHS} epochs, batch={DNN_BATCH}, lr={DNN_LR}")

    model = DNN(11, 6, DNN_LAYERS, DNN_NODES).to(device)
    model.apply(_init_weights)                            # FIX: Xavier init (was missing)

    opt = torch.optim.Adam(model.parameters(), lr=DNN_LR)

    # FIX: ExponentialLR scheduler — matches unified_trials.py (was missing entirely)
    gamma = (DNN_LR_FINAL / DNN_LR) ** (1.0 / max(1, DNN_EPOCHS))
    scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=gamma)

    loss_fn = nn.MSELoss()
    log_every = max(1, DNN_EPOCHS // 20)

    # AMP scaler — matches unified_trials.py
    scaler = torch.cuda.amp.GradScaler(enabled=use_cuda_amp)

    # ---- 4. Train ----
    model.train()
    for ep in range(1, DNN_EPOCHS + 1):
        epoch_loss = 0.0
        for xb, yb in dl:
            # non_blocking=True — matches unified_trials.py
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)              # matches unified_trials.py
            if use_cuda_amp:
                with torch.cuda.amp.autocast():
                    pred = model(xb)
                    loss = loss_fn(pred, yb)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                pred = model(xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                opt.step()
            epoch_loss += loss.item() * xb.shape[0]
        scheduler.step()

        if ep % log_every == 0:
            print(f"[{ep:6d}/{DNN_EPOCHS}] MSE={epoch_loss/TRAIN_SIZE:.6e}  "
                  f"lr={scheduler.get_last_lr()[0]:.2e}")

    # ---- 5. Save checkpoint (same keys as dnn_emulator_11d.py expects) ----
    ckpt = {
        'state_dict': model.cpu().state_dict(),
        'Ymin':    Ymin,
        'Ymax':    Ymax,
        'pmin11':  PMIN_11.astype(np.float32),
        'pmax11':  PMAX_11.astype(np.float32),
        'xnames':  np.array(XNAMES, dtype=object),
        'm_train': int(TRAIN_SIZE),
        'dnn_layers': DNN_LAYERS,
        'dnn_nodes':  DNN_NODES,
    }
    out_path = 'models_11d/dnn11d_model.pt'
    torch.save(ckpt, out_path)
    print(f"Saved checkpoint: {out_path}")


if __name__ == "__main__":
    main()
