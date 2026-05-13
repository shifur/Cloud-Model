# mcmc_with_pdr11d.py
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
import log_prob_pdr11d

# simps fallback
try:
    from scipy.integrate import simps
except Exception:
    try:
        from scipy.integrate import simpson as simps
    except Exception:
        import numpy as _np
        def simps(y, x=None, axis=-1):
            return _np.trapz(y, x, axis=axis) if x is not None else _np.trapz(y, dx=1.0, axis=axis)

def kde1d(x, grid, bw=0.15):
    kde = gaussian_kde(x, bw_method=bw)
    z = kde(grid); z /= simps(z, grid)
    return z

def kde2d(x, y, xgrid, ygrid, bw=0.15):
    kde = gaussian_kde(np.vstack([x, y]), bw_method=bw)
    Xg, Yg = np.meshgrid(xgrid, ygrid)
    Z = kde(np.vstack([Xg.ravel(), Yg.ravel()])).reshape(Xg.shape)
    Z /= simps(simps(Z, ygrid, axis=0), xgrid)
    return Xg, Yg, Z

def _grid_limits(samples, lo, hi, qlo=0.5, qhi=99.5, n=200):
    a = max(lo, np.percentile(samples, qlo))
    b = min(hi, np.percentile(samples, qhi))
    if not np.isfinite(a) or not np.isfinite(b) or a >= b: a, b = lo, hi
    return np.linspace(a, b, n)

def plot_ag_bg_marginal(Xa_plot, xnames, pmin, pmax, x_true, out_png, title_note=""):
    ag_i, bg_i = xnames.index('ag'), xnames.index('bg')
    ag, bg = Xa_plot[:, ag_i], Xa_plot[:, bg_i]
    ag_g = _grid_limits(ag, pmin[ag_i], pmax[ag_i])
    bg_g = _grid_limits(bg, pmin[bg_i], pmax[bg_i])
    Xg, Yg, Z = kde2d(ag, bg, ag_g, bg_g, bw=0.15)

    fig = plt.figure(figsize=(5.8, 5.2), facecolor='white')
    gs  = fig.add_gridspec(2, 2, width_ratios=[4, 1.2], height_ratios=[1.2, 4], hspace=0.05, wspace=0.05)
    axx = fig.add_subplot(gs[0,0]); axy = fig.add_subplot(gs[1,1]); ax = fig.add_subplot(gs[1,0])
    ax.contourf(Xg, Yg, Z, levels=60, alpha=0.85, cmap='rainbow')   # filled-only
    ax.plot([x_true[ag_i]], [x_true[bg_i]], 'kx', ms=9, mew=2)
    ax.set_xlabel('ag'); ax.set_ylabel('bg'); ax.grid(alpha=0.25)
    axx.hist(ag, bins=60, density=True, histtype='step', lw=1.0)
    axy.hist(bg, bins=60, density=True, histtype='step', lw=1.0, orientation='horizontal')
    axx.tick_params(labelbottom=False); axy.tick_params(labelleft=False)
    plt.suptitle(f'ag–bg marginal posterior {title_note}', y=0.98, fontsize=12)
    os.makedirs(os.path.dirname(out_png) or '.', exist_ok=True)
    plt.savefig(out_png, dpi=130, bbox_inches='tight'); plt.close(fig)

def plot_pairwise_grid(Xa_plot, xnames, pmin, pmax, x_true,
                       subset=None, out_png='plots/fig_pairgrid_pdr11d.png',
                       title="", bw_1d=0.15, bw_2d=0.15, n_grid=160, figsize_per_cell=1.6, dpi=150):
    if subset is None: subset = xnames[:]
    idxs = [xnames.index(n) for n in subset]; d = len(idxs)
    S = Xa_plot[:, idxs]
    g1, k1 = [], []
    for j, jj in enumerate(idxs):
        g = _grid_limits(S[:, j], pmin[jj], pmax[jj], n=n_grid)
        g1.append(g); k1.append(kde1d(S[:, j], g, bw=bw_1d))

    fig, axes = plt.subplots(d, d, figsize=(figsize_per_cell*d, figsize_per_cell*d), facecolor='white', squeeze=False)
    plt.subplots_adjust(hspace=0.05, wspace=0.05)
    for r in range(d):
        for c in range(d):
            ax = axes[r, c]
            if r == c:
                ax.plot(g1[r], k1[r], lw=1.0); ax.set_yticks([])
                if x_true is not None: ax.axvline(x_true[idxs[r]], ls='--', lw=1.0, color='k', alpha=0.7)
                ax.set_xlabel(subset[r] if r==d-1 else "")
                if r < d-1: ax.set_xticklabels([]); ax.grid(alpha=0.12)
            elif r > c:
                x = S[:, c]; y = S[:, r]
                gx = _grid_limits(x, pmin[idxs[c]], pmax[idxs[c]], n=n_grid)
                gy = _grid_limits(y, pmin[idxs[r]], pmax[idxs[r]], n=n_grid)
                Xg, Yg, Z = kde2d(x, y, gx, gy, bw=bw_2d)
                ax.contourf(Xg, Yg, Z, levels=40, cmap='rainbow', alpha=0.9)  # filled-only
                if x_true is not None: ax.plot([x_true[idxs[c]]], [x_true[idxs[r]]], 'kx', ms=7, mew=1.8)
                if r != d-1: ax.set_xticklabels([]); 
                else: ax.set_xlabel(subset[c])
                if c != 0: ax.set_yticklabels([]); 
                else: ax.set_ylabel(subset[r])
                ax.grid(alpha=0.08)
            else:
                ax.set_visible(False)
    if title: plt.suptitle(title, y=0.995, fontsize=12)
    os.makedirs(os.path.dirname(out_png) or '.', exist_ok=True)
    plt.savefig(out_png, dpi=dpi, bbox_inches='tight'); plt.close(fig)

def run_case(USE_RADIATION: bool):
    set_global_seed(SEED)
    x_true = X_TRUE_11D.copy()
    PMask  = np.ones_like(x_true)
    y_mask = YMASK_FIG9 if USE_RADIATION else YMASK_FIG7
    run_note  = "_rad" if USE_RADIATION else "_norad"
    title_note= "(with radiation: OLR, OSR)" if USE_RADIATION else "(no radiation: PCP, LWP, IWP)"

    # truth for all 6 @120
    L1_six = run_cloud_11d_six(x_true[None, :])[0]

    # MCMC settings
    nwalk, nburn, nmcmc = 64, 500, 3000
    x_sig = np.array([20.0, 0.05, 20.0, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 1.e-4, 1.e-5])

    # init walkers inside bounds
    p0 = np.tile(x_true, nwalk).reshape((nwalk, 11)) + np.random.normal(0.0, 1.0, (nwalk, 11)) * x_sig
    for j in range(11):
        lo, hi = PMIN_11[j], PMAX_11[j]
        bad = (p0[:, j] < lo) | (p0[:, j] > hi)
        while np.any(bad):
            p0[bad, j] = x_true[j] + np.random.normal(0.0, x_sig[j], bad.sum())
            bad = (p0[:, j] < lo) | (p0[:, j] > hi)

    # sanity
    lp0, _ = log_prob_pdr11d.log_prob_pdr11d(p0[0], x_true, L1_six, Y_SIG_SIX, PMIN_11, PMAX_11, PMask, y_mask)
    if not np.isfinite(lp0): raise RuntimeError("Initial PDR log_prob is -inf; train PDR first.")

    dtype = [("Hx", float, (1+6,))]
    sampler = emcee.EnsembleSampler(
        nwalk, 11, log_prob_pdr11d.log_prob_pdr11d,
        blobs_dtype=dtype,
        args=[x_true, L1_six, Y_SIG_SIX, PMIN_11, PMAX_11, PMask, y_mask, 1]
    )

    t0 = time.time()
    state = sampler.run_mcmc(p0, nburn, progress=True)
    sampler.reset()
    sampler.run_mcmc(state, nmcmc, progress=True)
    print(f"PDR MCMC {run_note} took {time.time()-t0:.1f}s")

    Xa      = sampler.get_chain()
    log_pxy = sampler.get_log_prob()
    blobs   = sampler.get_blobs()["Hx"]
    log_pyx = blobs[:, :, 0]
    HXa     = blobs[:, :, 1:]

    os.makedirs('output', exist_ok=True)
    base = f'_EXP_3_pdr11d{run_note}.nc'
    ds = xr.Dataset(
        {"x_true":  (["nx"], x_true),
         "PMask":   (["nx"], PMask.astype(float)),
         "L1_six":  (["ny"], L1_six),
         "L2_six":  (["ny"], (Y_SIG_SIX**2)),   # σ² saved
         "y_mask":  (["ny"], y_mask.astype(float)),
         "Xa":      (["nsamples","nchains","nx"], Xa),
         "HXa":     (["nsamples","nchains","ny"], HXa),
         "p_yx":    (["nsamples","nchains"], log_pyx),
         "p_xy":    (["nsamples","nchains"], log_pxy)},
        coords={"nsamples": np.arange(Xa.shape[0]),
                "nchains":  np.arange(Xa.shape[1]),
                "nx":       np.arange(11),
                "ny":       np.arange(6)},
        attrs={"xnames": ",".join(XNAMES),
               "ynames": "PCP@120,ACC@120,LWP@120,IWP@120,OLR@120,OSR@120",
               "Acceptance fraction": float(np.mean(sampler.acceptance_fraction))}
    )
    out_nc = os.path.join('output', base); ds.to_netcdf(out_nc); print("Saved:", out_nc)

    # plots
    Xa_plot = Xa.reshape(-1, Xa.shape[-1])
    plot_ag_bg_marginal(Xa_plot, XNAMES, PMIN_11, PMAX_11, x_true,
                        out_png=f'plots/fig_agbg_marginal{base.replace(".nc",".png")}',
                        title_note=f"(PDR 11D; {'OLR/OSR' if y_mask[-2:].sum() else 'PCP/LWP/IWP'})")
    plot_pairwise_grid(Xa_plot, XNAMES, PMIN_11, PMAX_11, x_true,
                       out_png=f'plots/fig_pairgrid{base.replace(".nc",".png")}',
                       title=f"PDR 11D pairwise marginals {'(with rad)' if y_mask[-2:].sum() else '(no rad)'}")

def main():
    print("\n=== CASE A (Fig.7): NO RADIATION ===")
    run_case(False)
    print("\n=== CASE B (Fig.9): WITH RADIATION ===")
    run_case(True)

if __name__ == "__main__":
    main()
