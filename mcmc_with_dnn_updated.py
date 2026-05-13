# mcmc_with_dnn_updated.py
import os, time
import numpy as np
import xarray as xr
import emcee
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

from surrogate_config_11d import (
    set_global_seed, SEED,
    XNAMES, X_TRUE_11D, PMIN_11, PMAX_11,
    Y_SIG_SIX, YMASK_FIG7, YMASK_FIG9,
)
from crm_eval_11d_six import run_cloud_11d_six
import log_prob_dnn11d

# ------------------------------------------------------------------
#  PATHS TO CLOUD MCMC "TRUE" POSTERIOR CHAINS (11D full CRM)
# ------------------------------------------------------------------
TRUE_CHAIN_FILE_NORAD = "/home/ss24ce/last_time-2/plots/output/mcmc_crm1d__EXP_3_full11_ag400.00000_bg0.40000_norad.nc"
TRUE_CHAIN_FILE_RAD   = "/home/ss24ce/last_time-2/plots/output/mcmc_crm1d__EXP_3_full11_ag400.00000_bg0.40000_rad.nc"

# ==============================================================
#   NUMERICAL INTEGRATION (SciPy old/new compatible)
#   - SciPy simps (old): simps(y, x=None, axis=-1)
#   - SciPy simpson (new): simpson(y, *, x=None, dx=..., axis=-1)
#   We expose a single simps(y, x=None, axis=-1) wrapper.
# ==============================================================

try:
    from scipy.integrate import simps as _scipy_simps  # old SciPy

    def simps(y, x=None, axis=-1):
        return _scipy_simps(y, x=x, axis=axis)

except Exception:
    try:
        from scipy.integrate import simpson as _scipy_simpson  # new SciPy

        def simps(y, x=None, axis=-1):
            # x is keyword-only in scipy.integrate.simpson
            return _scipy_simpson(y, x=x, axis=axis)

    except Exception:
        import numpy as _np

        def simps(y, x=None, axis=-1):
            return _np.trapz(y, x=x, axis=axis)

# ==============================================================
#   SAVE / NETCDF SAFETY HELPERS
# ==============================================================

def _ensure_writable_dir(path: str):
    d = os.path.dirname(path) or "."
    os.makedirs(d, exist_ok=True)
    test_path = os.path.join(d, ".__write_test__")
    with open(test_path, "w") as f:
        f.write("ok")
    os.remove(test_path)

def _safe_float(arr, dtype=np.float32, nan_fill=-1e30):
    """Force numeric dtype and remove NaN/Inf."""
    a = np.asarray(arr)
    if a.dtype == object:
        a = np.array(a.tolist(), dtype=np.float64)
    a = a.astype(dtype, copy=False)
    a = np.nan_to_num(a, nan=nan_fill, posinf=1e30, neginf=-1e30)
    return a

def save_dataset_safely(ds: xr.Dataset, out_nc: str, debug_one_by_one: bool = False):
    _ensure_writable_dir(out_nc)

    if os.path.exists(out_nc):
        os.remove(out_nc)

    if debug_one_by_one:
        print("\n[DEBUG] Writing variables one-by-one to locate failure...")
        for v in list(ds.data_vars):
            try:
                tmp = xr.Dataset({v: ds[v]}, coords=ds.coords, attrs=ds.attrs)
                tmp_path = out_nc.replace(".nc", f".__test_{v}.nc")
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
                tmp.to_netcdf(tmp_path, engine="netcdf4", format="NETCDF4")
                print("  OK:", v, "dtype=", tmp[v].dtype, "shape=", tmp[v].shape)
            except Exception as e:
                print("  FAIL:", v, "dtype=", ds[v].dtype, "shape=", ds[v].shape)
                print("       error:", repr(e))
                raise
        print("[DEBUG] All per-variable writes succeeded. Problem may be multi-var close step.\n")

    encoding = {}
    for v in ds.data_vars:
        enc = {"zlib": True, "complevel": 4}
        if ds[v].ndim == 3:
            enc["chunksizes"] = (min(2000, ds[v].shape[0]), ds[v].shape[1], ds[v].shape[2])
        elif ds[v].ndim == 2:
            enc["chunksizes"] = (min(4000, ds[v].shape[0]), ds[v].shape[1])
        encoding[v] = enc

    try:
        ds.to_netcdf(out_nc, engine="netcdf4", format="NETCDF4", encoding=encoding)
        print("Saved:", out_nc)
        return
    except Exception as e1:
        print("[WARN] netcdf4 write failed:", repr(e1))

    try:
        ds.to_netcdf(out_nc, engine="h5netcdf")
        print("Saved with h5netcdf:", out_nc)
        return
    except Exception as e2:
        print("[ERROR] h5netcdf fallback also failed:", repr(e2))
        raise

# ==============================================================
#   KDE HELPERS  (FIXED: always pass x= as keyword)
# ==============================================================

def kde1d(x, grid, bw=0.15):
    kde = gaussian_kde(x, bw_method=bw)
    z = kde(grid)
    z /= simps(z, x=grid)          # <-- keyword
    return z

def kde2d(x, y, xgrid, ygrid, bw=0.15):
    kde = gaussian_kde(np.vstack([x, y]), bw_method=bw)
    Xg, Yg = np.meshgrid(xgrid, ygrid)
    Z = kde(np.vstack([Xg.ravel(), Yg.ravel()])).reshape(Xg.shape)
    # L1-normalize (keyword-only safe)
    Z /= simps(simps(Z, x=ygrid, axis=0), x=xgrid)
    return Xg, Yg, Z

def _grid_limits(samples, lo, hi, qlo=0.5, qhi=99.5, n=200):
    a = max(lo, np.percentile(samples, qlo))
    b = min(hi, np.percentile(samples, qhi))
    if not np.isfinite(a) or not np.isfinite(b) or a >= b:
        a, b = lo, hi
    return np.linspace(a, b, n)

# ==============================================================
#   JS DIVERGENCE HELPERS
# ==============================================================

def kullback_leibler_divergence(P, Q):
    eps = 1e-12
    P2 = P + eps
    Q2 = Q + eps
    return np.sum(P2 * np.log(P2 / Q2))

def jensen_shannon_divergence(P, Q):
    M = 0.5 * (P + Q)
    return 0.5 * kullback_leibler_divergence(P, M) + 0.5 * kullback_leibler_divergence(Q, M)

# ==============================================================
#   ENERGY DISTANCE HELPERS
# ==============================================================

def energy_distance(X, Y, max_n=10000, seed=None):
    rng = np.random.default_rng(seed)
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    n, d = X.shape
    m, d2 = Y.shape
    assert d == d2

    if n > max_n:
        idx = rng.choice(n, size=max_n, replace=False)
        X = X[idx]; n = max_n
    if m > max_n:
        idx = rng.choice(m, size=max_n, replace=False)
        Y = Y[idx]; m = max_n

    diff_xy = X[:, None, :] - Y[None, :, :]
    dist_xy = np.linalg.norm(diff_xy, axis=-1)

    diff_xx = X[:, None, :] - X[None, :, :]
    dist_xx = np.linalg.norm(diff_xx, axis=-1)

    diff_yy = Y[:, None, :] - Y[None, :, :]
    dist_yy = np.linalg.norm(diff_yy, axis=-1)

    term_xy = (2.0 / (n * m)) * np.sum(dist_xy)
    term_xx = (1.0 / (n * n)) * np.sum(dist_xx)
    term_yy = (1.0 / (m * m)) * np.sum(dist_yy)

    D2 = term_xy - term_xx - term_yy
    D2 = max(D2, 0.0)
    return np.sqrt(D2), D2

def compute_pairwise_ed_matrix(X_dnn, X_true, xnames):
    X_dnn = np.asarray(X_dnn, dtype=float)
    X_true = np.asarray(X_true, dtype=float)
    d = len(xnames)
    ED_mat = np.zeros((d, d), dtype=float)
    for i in range(d):
        for j in range(d):
            if i == j:
                ED_mat[i, j] = 0.0
                continue
            Xp = X_dnn[:, [i, j]]
            Xt = X_true[:, [i, j]]
            ED_ij, _ = energy_distance(Xp, Xt, max_n=10000)
            ED_mat[i, j] = ED_ij
    return ED_mat

def compute_pairwise_js_matrix(X_dnn, X_true, xnames, pmin, pmax,
                               n_grid=80, bw_2d=0.2):
    X_dnn = np.asarray(X_dnn, dtype=float)
    X_true = np.asarray(X_true, dtype=float)
    d = len(xnames)
    JS_mat = np.zeros((d, d), dtype=float)

    for i in range(d):
        for j in range(d):
            if i == j:
                JS_mat[i, j] = 0.0
                continue

            Xp = X_dnn[:, [i, j]]
            Xt = X_true[:, [i, j]]

            x_all = np.concatenate([Xp[:, 0], Xt[:, 0]])
            y_all = np.concatenate([Xp[:, 1], Xt[:, 1]])

            x_grid = _grid_limits(x_all, pmin[i], pmax[i], n=n_grid)
            y_grid = _grid_limits(y_all, pmin[j], pmax[j], n=n_grid)

            _, _, P = kde2d(Xp[:, 0], Xp[:, 1], x_grid, y_grid, bw=bw_2d)
            _, _, Q = kde2d(Xt[:, 0], Xt[:, 1], x_grid, y_grid, bw=bw_2d)

            P_flat = P.ravel()
            Q_flat = Q.ravel()
            P_flat /= np.sum(P_flat)
            Q_flat /= np.sum(Q_flat)
            JS_mat[i, j] = jensen_shannon_divergence(P_flat, Q_flat)

    return JS_mat

# ==============================================================
#   HEATMAP PLOTTING FOR ED / JS MATRICES
# ==============================================================

def plot_matrix_heatmap(M, labels, title, out_png, cmap='Blues', fmt="{:.2f}"):
    d = len(labels)
    fig, ax = plt.subplots(figsize=(1.2*d, 1.2*d), facecolor='white')
    vmin = np.nanmin(M)
    vmax = np.nanmax(M)
    im = ax.imshow(M, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax)

    ax.set_xticks(range(d))
    ax.set_xticklabels(labels, rotation=90)
    ax.set_yticks(range(d))
    ax.set_yticklabels(labels)

    for i in range(d):
        for j in range(d):
            ax.text(j, i, fmt.format(M[i, j]),
                    ha='center', va='center', fontsize=6, color='black')

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.title(title)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png) or '.', exist_ok=True)
    plt.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close(fig)

# ==============================================================
#   EXISTING PLOTTING: ag–bg + 11×11 pairwise grid
# ==============================================================

def plot_ag_bg_marginal(Xa_plot, xnames, pmin, pmax, x_true, out_png, title_note=""):
    ag_i, bg_i = xnames.index('ag'), xnames.index('bg')
    ag, bg = Xa_plot[:, ag_i], Xa_plot[:, bg_i]
    ag_g = _grid_limits(ag, pmin[ag_i], pmax[ag_i])
    bg_g = _grid_limits(bg, pmin[bg_i], pmax[bg_i])
    Xg, Yg, Z = kde2d(ag, bg, ag_g, bg_g, bw=0.15)

    fig = plt.figure(figsize=(5.8, 5.2), facecolor='white')
    gs  = fig.add_gridspec(2, 2, width_ratios=[4, 1.2],
                           height_ratios=[1.2, 4],
                           hspace=0.05, wspace=0.05)
    axx = fig.add_subplot(gs[0,0])
    axy = fig.add_subplot(gs[1,1])
    ax  = fig.add_subplot(gs[1,0])

    ax.contourf(Xg, Yg, Z, levels=60, alpha=0.85, cmap='rainbow')
    ax.plot([x_true[ag_i]], [x_true[bg_i]], 'kx', ms=9, mew=2)
    ax.set_xlabel('ag'); ax.set_ylabel('bg')
    ax.grid(alpha=0.25)

    axx.hist(ag, bins=60, density=True, histtype='step', lw=1.0)
    axy.hist(bg, bins=60, density=True, histtype='step', lw=1.0,
             orientation='horizontal')
    axx.tick_params(labelbottom=False)
    axy.tick_params(labelleft=False)

    plt.suptitle(f'ag–bg marginal posterior {title_note}', y=0.98, fontsize=12)
    os.makedirs(os.path.dirname(out_png) or '.', exist_ok=True)
    plt.savefig(out_png, dpi=130, bbox_inches='tight')
    plt.close(fig)

def plot_pairwise_grid(Xa_plot, xnames, pmin, pmax, x_true,
                       subset=None, out_png='plots/fig_pairgrid_dnn11d.png',
                       title="", bw_1d=0.15, bw_2d=0.15, n_grid=160,
                       figsize_per_cell=1.6, dpi=150):
    if subset is None: subset = xnames[:]
    idxs = [xnames.index(n) for n in subset]; d = len(idxs)
    S = Xa_plot[:, idxs]

    g1, k1 = [], []
    for j, jj in enumerate(idxs):
        g = _grid_limits(S[:, j], pmin[jj], pmax[jj], n=n_grid)
        g1.append(g); k1.append(kde1d(S[:, j], g, bw=bw_1d))

    fig, axes = plt.subplots(d, d,
                             figsize=(figsize_per_cell*d, figsize_per_cell*d),
                             facecolor='white', squeeze=False)
    plt.subplots_adjust(hspace=0.05, wspace=0.05)

    for r in range(d):
        for c in range(d):
            ax = axes[r, c]
            if r == c:
                ax.plot(g1[r], k1[r], lw=1.0); ax.set_yticks([])
                if x_true is not None:
                    ax.axvline(x_true[idxs[r]], ls='--', lw=1.0, color='k', alpha=0.7)
                ax.set_xlabel(subset[r] if r==d-1 else "")
                if r < d-1: ax.set_xticklabels([])
                ax.grid(alpha=0.12)
            elif r > c:
                x = S[:, c]; y = S[:, r]
                gx = _grid_limits(x, pmin[idxs[c]], pmax[idxs[c]], n=n_grid)
                gy = _grid_limits(y, pmin[idxs[r]], pmax[idxs[r]], n=n_grid)
                Xg, Yg, Z = kde2d(x, y, gx, gy, bw=bw_2d)
                ax.contourf(Xg, Yg, Z, levels=40, cmap='rainbow', alpha=0.9)
                if x_true is not None:
                    ax.plot([x_true[idxs[c]]], [x_true[idxs[r]]], 'kx', ms=7, mew=1.8)
                if r != d-1: ax.set_xticklabels([])
                else: ax.set_xlabel(subset[c])
                if c != 0: ax.set_yticklabels([])
                else: ax.set_ylabel(subset[r])
                ax.grid(alpha=0.08)
            else:
                ax.set_visible(False)

    if title: plt.suptitle(title, y=0.995, fontsize=12)
    os.makedirs(os.path.dirname(out_png) or '.', exist_ok=True)
    plt.savefig(out_png, dpi=dpi, bbox_inches='tight')
    plt.close(fig)

# ==============================================================
#   === DIAGNOSTIC HELPERS (R-hat + trace plots) ===
# ==============================================================

def gelman_rubin_rhat(chains_3d: np.ndarray):
    nsteps, nwalkers, nparams = chains_3d.shape
    rhat = np.empty(nparams, dtype=float)

    for k in range(nparams):
        x = chains_3d[:, :, k]
        m = float(nwalkers)
        n = float(nsteps)

        chain_means = np.mean(x, axis=0)
        chain_vars  = np.var(x, axis=0, ddof=1)

        B = n * np.var(chain_means, ddof=1)
        W = np.mean(chain_vars)

        Var_hat = ((n - 1.0) / n) * W + (B / n)
        rhat[k] = np.sqrt(Var_hat / W)

    return rhat

def plot_traces(Xa, param_names, out_png):
    nsteps, nwalkers, nparams = Xa.shape
    fig, axes = plt.subplots(nparams, 1, figsize=(10, 2.0*nparams), sharex=True)
    if nparams == 1:
        axes = [axes]

    for k in range(nparams):
        ax = axes[k]
        ax.plot(Xa[:, :, k], alpha=0.3)
        ax.set_ylabel(param_names[k])
        ax.grid(alpha=0.2)

    axes[-1].set_xlabel("MCMC step")
    fig.suptitle("Trace plots (DNN 11D, all walkers)", y=0.99, fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    os.makedirs(os.path.dirname(out_png) or '.', exist_ok=True)
    plt.savefig(out_png, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("Saved trace plots:", os.path.abspath(out_png))

# ==============================================================
#   TRUE POSTERIOR SAMPLING (FROM CLOUD MCMC CHAIN)
# ==============================================================

def sample_true_from_cloud_chain(nc_path, nsamples=5000):
    if not os.path.exists(nc_path):
        raise FileNotFoundError(f"Cloud MCMC file not found: {nc_path}")

    ds = xr.open_dataset(nc_path)
    if "Xa" not in ds:
        raise RuntimeError("Cloud MCMC file missing 'Xa' variable")

    Xa = ds["Xa"].values
    if Xa.ndim != 3:
        raise RuntimeError(f"Expected Xa with 3 dims, got shape {Xa.shape}")

    nsamp, nch, npar = Xa.shape
    pts = Xa.reshape(-1, npar)
    M = pts.shape[0]
    ns = min(nsamples, M)
    rng = np.random.default_rng(SEED + 999)
    idx = rng.choice(M, size=ns, replace=False)
    return pts[idx, :]

def subsample_cloud(X, max_n=5000, seed=None):
    X = np.asarray(X, dtype=float)
    n = X.shape[0]
    if n <= max_n:
        return X
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=max_n, replace=False)
    return X[idx, :]

# ==============================================================
#   MAIN DNN MCMC + METRIC COMPUTATION
# ==============================================================

def run_case(USE_RADIATION: bool):
    set_global_seed(SEED)
    x_true = X_TRUE_11D.copy()
    PMask  = np.ones_like(x_true)

    y_mask   = YMASK_FIG9 if USE_RADIATION else YMASK_FIG7
    run_note = "_rad" if USE_RADIATION else "_norad"

    L1_six = run_cloud_11d_six(x_true[None, :])[0]

    # MCMC knobs (walkers, burn-in steps, production steps)
    nwalk, nburn, nmcmc = 32, 3000, 50000

    x_sig = np.array([20.0, 0.05, 20.0, 0.05,
                      0.05, 0.05, 0.05,
                      0.05, 0.05,
                      1.e-4, 1.e-5])

    p0 = np.tile(x_true, nwalk).reshape((nwalk, 11)) + \
         np.random.normal(0.0, 1.0, (nwalk, 11)) * x_sig
    for j in range(11):
        lo, hi = PMIN_11[j], PMAX_11[j]
        bad = (p0[:, j] < lo) | (p0[:, j] > hi)
        while np.any(bad):
            p0[bad, j] = x_true[j] + np.random.normal(0.0, x_sig[j], bad.sum())
            bad = (p0[:, j] < lo) | (p0[:, j] > hi)

    lp0, _ = log_prob_dnn11d.log_prob_dnn11d(
        p0[0], x_true, L1_six, Y_SIG_SIX,
        PMIN_11, PMAX_11, PMask, y_mask, 1
    )
    if not np.isfinite(lp0):
        raise RuntimeError("Initial DNN log_prob is -inf; train DNN first.")

    dtype = [("Hx", float, (1+6,))]
    sampler = emcee.EnsembleSampler(
        nwalk, 11, log_prob_dnn11d.log_prob_dnn11d,
        blobs_dtype=dtype,
        args=[x_true, L1_six, Y_SIG_SIX,
              PMIN_11, PMAX_11, PMask, y_mask, 1]
    )

    t0 = time.time()
    state = sampler.run_mcmc(p0, nburn, progress=True)
    sampler.reset()
    sampler.run_mcmc(state, nmcmc, progress=True)
    print(f"DNN MCMC {run_note} took {time.time()-t0:.1f}s")

    Xa      = sampler.get_chain()
    log_pxy = sampler.get_log_prob()

    b = sampler.get_blobs()
    try:
        Hx_struct = b["Hx"]
    except Exception:
        Hx_struct = b

    Hx_struct = np.asarray(Hx_struct)
    if Hx_struct.dtype == object:
        Hx_struct = np.array(Hx_struct.tolist(), dtype=np.float64)

    log_pyx = np.asarray(Hx_struct[:, :, 0])
    HXa     = np.asarray(Hx_struct[:, :, 1:])

    acc = float(np.mean(sampler.acceptance_fraction))
    print(f"Mean acceptance (DNN{run_note}): {acc:.3f}")

    try:
        tau_arr = sampler.get_autocorr_time(quiet=True)
        print("Autocorr times (per parameter, DNN):", tau_arr)
    except Exception as e:
        print("Autocorr time computation failed (DNN):", repr(e))
        tau_arr = None

    try:
        rhat = gelman_rubin_rhat(Xa)
        print("Gelman–Rubin R-hat (per parameter, DNN):", rhat)
    except Exception as e:
        print("R-hat computation failed (DNN):", repr(e))
        rhat = None

    trace_png = f'plots/trace_dnn11d{run_note}.png'
    plot_traces(Xa, XNAMES, trace_png)

    Xa_s      = _safe_float(Xa, dtype=np.float32)
    HXa_s     = _safe_float(HXa, dtype=np.float32)
    log_pxy_s = _safe_float(log_pxy, dtype=np.float32)
    log_pyx_s = _safe_float(log_pyx, dtype=np.float32)

    x_true_s  = _safe_float(x_true, dtype=np.float32)
    PMask_s   = _safe_float(PMask.astype(float), dtype=np.float32)
    L1_six_s  = _safe_float(L1_six, dtype=np.float32)
    L2_six_s  = _safe_float((Y_SIG_SIX**2), dtype=np.float32)
    y_mask_s  = _safe_float(y_mask.astype(float), dtype=np.float32)

    os.makedirs('output', exist_ok=True)
    base = f'_EXP_3_dnn11d{run_note}.nc'
    out_nc = os.path.join('output', base)

    ds = xr.Dataset(
        {"x_true":  (["nx"], x_true_s),
         "PMask":   (["nx"], PMask_s),
         "L1_six":  (["ny"], L1_six_s),
         "L2_six":  (["ny"], L2_six_s),
         "y_mask":  (["ny"], y_mask_s),
         "Xa":      (["nsamples","nchains","nx"], Xa_s),
         "HXa":     (["nsamples","nchains","ny"], HXa_s),
         "p_yx":    (["nsamples","nchains"], log_pyx_s),
         "p_xy":    (["nsamples","nchains"], log_pxy_s)},
        coords={"nsamples": np.arange(Xa_s.shape[0]),
                "nchains":  np.arange(Xa_s.shape[1]),
                "nx":       np.arange(11),
                "ny":       np.arange(6)},
        attrs={"xnames": ",".join(XNAMES),
               "ynames": "PCP@120,ACC@120,LWP@120,IWP@120,OLR@120,OSR@120",
               "Acceptance fraction": float(np.mean(sampler.acceptance_fraction))}
    )

    if tau_arr is not None:
        ds["tau"] = (["nx"], _safe_float(tau_arr, dtype=np.float32))
        ds.tau.attrs["long_name"] = "Autocorrelation time per parameter (DNN 11D)"
    if rhat is not None:
        ds["rhat"] = (["nx"], _safe_float(rhat, dtype=np.float32))
        ds.rhat.attrs["long_name"] = "Gelman–Rubin R-hat per parameter (DNN 11D)"

    save_dataset_safely(ds, out_nc, debug_one_by_one=False)

    Xa_plot = Xa_s.reshape(-1, Xa_s.shape[-1])

    plot_ag_bg_marginal(
        Xa_plot, XNAMES, PMIN_11, PMAX_11, x_true,
        out_png=f'plots/fig_agbg_marginal{base.replace(".nc",".png")}',
        title_note=f"(DNN 11D; {'OLR/OSR' if y_mask[-2:].sum() else 'PCP/LWP/IWP'})"
    )
    plot_pairwise_grid(
        Xa_plot, XNAMES, PMIN_11, PMAX_11, x_true,
        out_png=f'plots/fig_pairgrid{base.replace(".nc",".png")}',
        title=f"DNN 11D pairwise marginals {'(with rad)' if y_mask[-2:].sum() else '(no rad)'}"
    )

    try:
        os.makedirs("mcmc__DNN", exist_ok=True)

        X_dnn_sub = subsample_cloud(Xa_plot, max_n=10000, seed=SEED + 301)
        true_chain_file = TRUE_CHAIN_FILE_RAD if USE_RADIATION else TRUE_CHAIN_FILE_NORAD
        X_true_sub = sample_true_from_cloud_chain(true_chain_file, nsamples=5000)

        ED_self_dnn, _  = energy_distance(X_dnn_sub, X_dnn_sub, max_n=5000, seed=SEED + 302)
        ED_self_true, _ = energy_distance(X_true_sub, X_true_sub, max_n=5000, seed=SEED + 303)
        ED_dnn_true, _  = energy_distance(X_dnn_sub, X_true_sub, max_n=5000, seed=SEED + 304)
        ED_true_dnn, _  = energy_distance(X_true_sub, X_dnn_sub, max_n=5000, seed=SEED + 305)

        print("\n=== ENERGY DISTANCE VALIDATION TESTS (DNN vs Cloud MCMC) ===")
        print(f"Self-distance (DNN vs DNN):         ED = {ED_self_dnn:.6f}")
        print(f"Self-distance (Cloud vs Cloud):     ED = {ED_self_true:.6f}")
        print(f"ED(DNN → Cloud) = {ED_dnn_true:.6f}")
        print(f"ED(Cloud → DNN) = {ED_true_dnn:.6f}")
        print(f"Difference      = {abs(ED_dnn_true - ED_true_dnn):.6e}")
        print("=== END VALIDATION TESTS ===\n")

        ed_png = os.path.join("mcmc__DNN", f"fig_EDmatrix_dnn11d{run_note}.png")
        ED_mat = compute_pairwise_ed_matrix(X_dnn_sub, X_true_sub, XNAMES)
        plot_matrix_heatmap(
            ED_mat, XNAMES,
            title=f"Energy distance (pairwise) DNN vs Cloud {run_note}",
            out_png=ed_png,
            fmt="{:.2f}"
        )

        js_png = os.path.join("mcmc__DNN", f"fig_JSmatrix_dnn11d{run_note}.png")
        JS_mat = compute_pairwise_js_matrix(
            X_dnn_sub, X_true_sub, XNAMES, PMIN_11, PMAX_11,
            n_grid=80, bw_2d=0.2
        )
        plot_matrix_heatmap(
            JS_mat, XNAMES,
            title=f"JS divergence (pairwise) DNN vs Cloud {run_note}",
            out_png=js_png,
            fmt="{:.3f}"
        )

        summary_path = os.path.join('mcmc__DNN', f'summary_dnn11d{run_note}.txt')
        with open(summary_path, 'w') as f:
            f.write("=== DNN vs Cloud MCMC Posterior Metrics ===\n")
            f.write(f"Case: {'WITH radiation (OLR/OSR)' if USE_RADIATION else 'NO radiation (PCP/LWP/IWP)'}\n\n")
            f.write("Energy distance validation:\n")
            f.write(f"  Self ED(DNN):   {ED_self_dnn:.6f}\n")
            f.write(f"  Self ED(Cloud): {ED_self_true:.6f}\n")
            f.write(f"  ED(DNN,Cloud):  {ED_dnn_true:.6f}\n")
            f.write(f"  ED(Cloud,DNN):  {ED_true_dnn:.6f}\n")
            f.write(f"  |ΔED|:          {abs(ED_dnn_true - ED_true_dnn):.6e}\n\n")
            f.write("Pairwise ED matrix (rows/cols = XNAMES order):\n")
            np.savetxt(f, ED_mat, fmt="%.4f")
            f.write("\nPairwise JS matrix:\n")
            np.savetxt(f, JS_mat, fmt="%.6f")
        print("Saved summary:", summary_path)

    except Exception as e:
        print("[WARN] Skipping JS/ED metrics (DNN vs Cloud) due to error:", repr(e))

def main():
    print("\n=== CASE A (Fig.7): NO RADIATION ===")
    run_case(False)
    print("\n=== CASE B (Fig.9): WITH RADIATION ===")
    run_case(True)

if __name__ == "__main__":
    main()
