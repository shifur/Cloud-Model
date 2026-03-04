# run_mcmc_crm1d_full11.py  —  11D joint posterior + ag–bg marginal + 11×11 pairwise grid
import os
import time
import uuid
import numpy as np
import xarray as xr
import emcee
import multiprocessing as mp
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# Robust simps fallback
try:
    from scipy.integrate import simps
except Exception:
    try:
        from scipy.integrate import simpson as simps
    except Exception:
        import numpy as _np
        def simps(y, x=None, axis=-1):
            if x is None: return _np.trapz(y, dx=1.0, axis=axis)
            else:         return _np.trapz(y, x, axis=axis)

# ----------------
#   MODULES
# ----------------
from cloud_column_model import cloud_column_model
import log_prob_crm1d_updated as log_prob_crm1d  # your existing Code 1 file

# ----------------
#   HELPERS
# ----------------
def safe_to_netcdf(ds: xr.Dataset, final_path: str):
    out_dir = os.path.dirname(final_path)
    os.makedirs(out_dir, exist_ok=True)
    tmp_path = os.path.join(out_dir, f".tmp_{uuid.uuid4().hex}.nc")
    ds.to_netcdf(tmp_path, mode="w")
    os.replace(tmp_path, final_path)
    print("Wrote:", os.path.abspath(final_path))

def kde1d(x, grid, bw=0.15):
    kde = gaussian_kde(x, bw_method=bw)
    z = kde(grid)
    z /= simps(z, x=grid)
    return z

def kde2d(x, y, xgrid, ygrid, bw=0.15):
    kde = gaussian_kde(np.vstack([x, y]), bw_method=bw)
    Xg, Yg = np.meshgrid(xgrid, ygrid)
    Z = kde(np.vstack([Xg.ravel(), Yg.ravel()])).reshape(Xg.shape)
    # L1 normalize for comparable contour levels
    Z /= simps(simps(Z, ygrid, axis=0), xgrid)
    return Xg, Yg, Z

def _grid_limits(samples, lo, hi, qlo=0.5, qhi=99.5, n=160):
    a = max(lo, np.percentile(samples, qlo))
    b = min(hi, np.percentile(samples, qhi))
    if (not np.isfinite(a)) or (not np.isfinite(b)) or (a >= b):
        a, b = lo, hi
    return np.linspace(a, b, n)

def plot_ag_bg_marginal(Xa_plot, xnames, pmin, pmax, x_true, out_png, title_note=""):
    # Xa_plot: [N, 11] flattened samples (post-burn)
    ag_idx = xnames.index('ag')
    bg_idx = xnames.index('bg')
    ag = Xa_plot[:, ag_idx]
    bg = Xa_plot[:, bg_idx]

    # Grids for nice contours (keep within bounds & sample spread)
    ag_min = max(pmin[ag_idx], np.percentile(ag, 0.5))
    ag_max = min(pmax[ag_idx], np.percentile(ag, 99.5))
    bg_min = max(pmin[bg_idx], np.percentile(bg, 0.5))
    bg_max = min(pmax[bg_idx], np.percentile(bg, 99.5))
    ag_grid = np.linspace(ag_min, ag_max, 200)
    bg_grid = np.linspace(bg_min, bg_max, 200)

    Xg, Yg, Z = kde2d(ag, bg, ag_grid, bg_grid, bw=0.15)

    # Figure with 2D contours + marginals
    fig = plt.figure(figsize=(5.8, 5.2), facecolor='white')
    gs  = fig.add_gridspec(2, 2, width_ratios=[4, 1.2], height_ratios=[1.2, 4], hspace=0.05, wspace=0.05)
    ax_histx = fig.add_subplot(gs[0,0])
    ax_histy = fig.add_subplot(gs[1,1])
    ax       = fig.add_subplot(gs[1,0])

    ax.contour(Xg, Yg, Z, levels=10, linewidths=1.2, colors='crimson')
    ax.contourf(Xg, Yg, Z, levels=50, alpha=0.6, cmap='rainbow')
    ax.plot([x_true[ag_idx]], [x_true[bg_idx]], 'kx', ms=10, mew=2)
    ax.set_xlabel('ag'); ax.set_ylabel('bg'); ax.grid(alpha=0.25)

    # 1D marginals
    ax_histx.hist(ag, bins=60, density=True, histtype='step', lw=1.2)
    ax_histy.hist(bg, bins=60, density=True, histtype='step', lw=1.2, orientation='horizontal')
    ax_histx.tick_params(labelbottom=False); ax_histy.tick_params(labelleft=False)

    plt.suptitle(f'ag–bg marginal posterior {title_note}', y=0.98, fontsize=12)
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print("Saved:", os.path.abspath(out_png))

# NEW (2): 11×11 pairwise grid (reference-style)
def plot_pairwise_grid(Xa_plot, xnames, pmin, pmax, x_true,
                       subset=None, out_png='fig_pairgrid.png',
                       title="", bw_1d=0.15, bw_2d=0.15, n_grid=160,
                       figsize_per_cell=1.6, dpi=150):
    if subset is None:
        subset = xnames[:]
    idxs = [xnames.index(n) for n in subset]
    d = len(idxs)
    S = Xa_plot[:, idxs]

    g1, k1 = [], []
    for j, jj in enumerate(idxs):
        g = _grid_limits(S[:, j], pmin[jj], pmax[jj], n=n_grid)
        g1.append(g)
        k1.append(kde1d(S[:, j], g, bw=bw_1d))

    fig, axes = plt.subplots(
        d, d,
        figsize=(figsize_per_cell * d, figsize_per_cell * d),
        facecolor='white',
        squeeze=False
    )
    plt.subplots_adjust(hspace=0.05, wspace=0.05)

    for r in range(d):
        for c in range(d):
            ax = axes[r, c]
            if r == c:
                ax.plot(g1[r], k1[r], lw=1.0)
                ax.set_yticks([])
                if x_true is not None:
                    ax.axvline(x_true[idxs[r]], ls='--', lw=1.0, color='k', alpha=0.7)
                ax.set_xlabel(subset[r] if r == d - 1 else "")
                if r < d - 1:
                    ax.set_xticklabels([])
                ax.grid(alpha=0.12)

            elif r > c:
                x = S[:, c]
                y = S[:, r]
                gx = _grid_limits(x, pmin[idxs[c]], pmax[idxs[c]], n=n_grid)
                gy = _grid_limits(y, pmin[idxs[r]], pmax[idxs[r]], n=n_grid)
                Xg, Yg, Z = kde2d(x, y, gx, gy, bw=bw_2d)
                ax.contourf(Xg, Yg, Z, levels=40, cmap='rainbow', alpha=0.9)
                if x_true is not None:
                    ax.plot([x_true[idxs[c]]], [x_true[idxs[r]]], 'kx', ms=7, mew=1.8)

                if r != d - 1:
                    ax.set_xticklabels([])
                else:
                    ax.set_xlabel(subset[c])

                if c != 0:
                    ax.set_yticklabels([])
                else:
                    ax.set_ylabel(subset[r])

                ax.grid(alpha=0.08)

            else:
                ax.set_visible(False)

    if title:
        plt.suptitle(title, y=0.995, fontsize=12)
    os.makedirs(os.path.dirname(out_png) or '.', exist_ok=True)
    plt.savefig(out_png, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print("Saved:", os.path.abspath(out_png))


def run_one_case(USE_RADIATION: bool):
    np.set_printoptions(precision=5, suppress=True)
    np.random.seed(33)

    # --------------------------
    #    EXPERIMENT SETTINGS
    # --------------------------
    expdir   = './cloud_column_model/'
    exper_tag = '_EXP_3'
    do_parallel = True

    # NEW (1): create a fresh run folder and save EVERYTHING there
    run_tag = time.strftime("%Y%m%d_%H%M%S")
    base_run_dir = os.path.abspath(f"FINAL_CRM11D_{run_tag}")
    out_nc_dir   = os.path.join(base_run_dir, "output")
    out_plot_dir = os.path.join(base_run_dir, "plots")
    os.makedirs(out_nc_dir, exist_ok=True)
    os.makedirs(out_plot_dir, exist_ok=True)
    print(f"[INFO] Outputs will be saved under: {base_run_dir}")

    # MCMC knobs (bump a bit for 11D)
    nwalkers = 32
    nburn    = 40000
    nmcmc    = 400000
    PType    = 0   # flat prior (inside bounds)
    LType    = 1   # Gaussian likelihood

    # parameters (ALL 11 varied)
    xnames = ['as','bs','ag','bg','N0r','N0s','N0g','rhos','rhog','qc0','qi0']
    x_true = np.array([200.0, 0.3, 400.0, 0.4, 0.5, 0.5, 0.5, 0.2, 0.4, 1.e-3, 6.e-4])
    PMask  = np.ones(len(xnames))
    xidx   = np.squeeze(np.nonzero(PMask)).tolist()
    nx     = len(x_true)
    nxp    = int(np.count_nonzero(PMask))

    # proposal scales and hard bounds
    x_sig     = np.array([20.0, 0.05, 20.0, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 1.e-4, 1.e-5])
    pmin_MCMC = np.array([0.0, 0.0, 20.0, 0.01666667, 0.05, 0.05, 0.05, 0.01666667, 0.01666667, 1.e-4, 2.e-6])
    pmax_MCMC = np.array([1000.0, 1.0, 1180.0, 0.98333333, 5.0, 2.5, 2.5, 0.98333333, 0.98333333, 2.e-3, 1.e-3])

    # --------------------------
    #   OBS CHOICE (Fig.7 vs Fig.9)
    # --------------------------
    tnames = ['30','60','90','120','150','180']
    t_mask = np.array([0,0,0,1,0,0])
    ynames = ['PCP','ACC','LWP','IWP','OLR','OSR']

    if USE_RADIATION:
        y_mask    = np.array([0,0,0,0,1,1])
        y_sig_tmp = np.array([2.0,5.0,0.5,1.0,5.0,5.0])
        run_note  = "_rad"
        title_note = "(with radiation: OLR, OSR)"
    else:
        y_mask    = np.array([1,0,1,1,0,0])
        y_sig_tmp = np.array([2.0,5.0,0.5,1.0,10.0,20.0])
        run_note  = "_norad"
        title_note = "(no radiation: PCP, LWP, IWP)"

    n_obs       = len(ynames)
    n_obs_times = len(tnames)
    y_sig  = np.tile(y_sig_tmp, n_obs_times)
    LMask  = np.empty(n_obs * n_obs_times)
    for t in range(n_obs_times):
        for o in range(n_obs):
            idx2 = (t * n_obs) + o
            LMask[idx2] = t_mask[t] * y_mask[o]

    # --------------------------
    #   Prior & Nature run
    # --------------------------
    P1 = np.array(x_true)
    P2 = np.diag(x_sig)
    p_vec = np.ndarray.tolist(x_true)

    try:
        crm1d = cloud_column_model.CRM1DWrap(
            os.path.join(expdir, 'run_one_crm1d.txt'),
            os.path.join(expdir, 'crm1d_output.txt'),
            os.path.join(expdir, 'namelist_3h_t30-180.f90'),
            params=p_vec
        )
        model_output, crm_status = crm1d()
        if not crm_status:
            raise RuntimeError("CRM1D nature run returned crm_status=False")
        L1 = np.array(model_output)
        L2 = np.diag(y_sig**2.0)
    except Exception as e:
        print("WARNING: Nature run failed; using synthetic linearized L1. Reason:", repr(e))
        ny = n_obs * n_obs_times
        rng = np.random.default_rng(12345)
        W = rng.standard_normal((ny, len(x_true))) * 0.02
        b = rng.standard_normal(ny) * 0.01
        L1 = (W @ x_true) + b
        L2 = np.diag(y_sig**2.0)

    # --------------------------
    #   Initialize walkers
    # --------------------------
    p0 = np.tile(x_true[xidx], nwalkers).reshape((nwalkers, nxp)) \
        + np.random.normal(0.0, 1.0, (nwalkers, nxp)) * x_sig[xidx]
    for j in range(nxp):
        low, high = pmin_MCMC[xidx][j], pmax_MCMC[xidx][j]
        bad = (p0[:, j] < low) | (p0[:, j] > high)
        while np.any(bad):
            p0[bad, j] = x_true[xidx][j] + np.random.normal(0.0, x_sig[xidx][j], bad.sum())
            bad = (p0[:, j] < low) | (p0[:, j] > high)
    print(f"Init p0 shape {p0.shape}, vary {nxp} params.")

    lp_test, blob_test = log_prob_crm1d.log_prob_crm1d(
        p0[0], x_true, P1, P2, L1, L2, PType, LType,
        pmin_MCMC, pmax_MCMC, PMask, LMask, expdir=expdir
    )
    if not np.isfinite(lp_test) or isinstance(blob_test, (float, int)):
        print("log_prob_crm1d returned -inf or malformed blob at initialization.")
        print("Candidate x:", p0[0]); print("Bounds:", pmin_MCMC[xidx], pmax_MCMC[xidx])
        raise RuntimeError("Initial test call failed; fix expdir/files or resample p0.")

    # --------------------------
    #   MCMC
    # --------------------------
    dtype = [("Hx", float, (1 + n_obs * n_obs_times))]
    sampler = emcee.EnsembleSampler(
        nwalkers, nxp, log_prob_crm1d.log_prob_crm1d,
        blobs_dtype=dtype,
        moves=[emcee.moves.WalkMove()],
        args=[x_true, P1, P2, L1, L2, PType, LType, pmin_MCMC, pmax_MCMC, PMask, LMask, expdir]
    )

    if do_parallel:
        num_workers = min((os.cpu_count() or 4), 16)
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=num_workers) as pool:
            sampler.pool = pool
            t0 = time.time()
            state = sampler.run_mcmc(p0, nburn, progress=True)
            print(f"Burn-in took {time.time()-t0:.1f}s")
            sampler.reset()
            t0 = time.time()
            sampler.run_mcmc(state, nmcmc, progress=True)
            print(f"Production {nmcmc} steps took {time.time()-t0:.1f}s")
    else:
        t0 = time.time()
        state = sampler.run_mcmc(p0, nburn, progress=True)
        print(f"Burn-in took {time.time()-t0:.1f}s")
        sampler.reset()
        t0 = time.time()
        sampler.run_mcmc(state, nmcmc, progress=True)
        print(f"Production {nmcmc} steps took {time.time()-t0:.1f}s")

    acc = float(np.mean(sampler.acceptance_fraction))
    print(f"Mean acceptance: {acc:.3f}")
    try:
        tau = float(np.mean(sampler.get_autocorr_time(quiet=True)))
    except Exception as e:
        print("Autocorr time unreliable:", repr(e)); tau = np.nan

    # --------------------------
    #   Save chain & blobs (NEW FOLDER)
    # --------------------------
    Xa      = sampler.get_chain()
    log_pxy = sampler.get_log_prob()
    blobs   = sampler.get_blobs()["Hx"]
    log_pyx = blobs[:, :, 0]
    HXa     = blobs[:, :, 1:]

    base_name = f"{exper_tag}_full11_ag{float(x_true[xnames.index('ag')]):.5f}_bg{float(x_true[xnames.index('bg')]):.5f}{run_note}"
    ncfile = os.path.join(out_nc_dir, f"mcmc_crm1d_{base_name}.nc")

    ds = xr.Dataset(
        {"x_true":  (["nx"], x_true),
         "P1":      (["nx"], x_true),
         "P2":      (["nx"], x_sig),
         "PMask":   (["nx"], PMask),
         "L1":      (["ny"], L1),
         "L2":      (["ny"], np.tile(y_sig_tmp, len(tnames))),
         "LMask":   (["ny_all"], np.array(LMask, dtype=float)),
         "Xa":      (["nsamples", "nchains", "nxp"], Xa),
         "HXa":     (["nsamples", "nchains", "ny_all"], HXa),
         "p_yx":    (["nsamples", "nchains"], log_pyx),
         "p_xy":    (["nsamples", "nchains"], log_pxy)},
        coords={"nsamples": (["nsamples"], np.arange(Xa.shape[0])),
                "nchains":  (["nchains"],  np.arange(Xa.shape[1])),
                "nx":       (["nx"],       np.arange(nx)),
                "nxp":      (["nxp"],      np.arange(nxp)),
                "ny":       (["ny"],       np.arange(len(L1))),
                "ny_all":   (["ny_all"],   np.arange(HXa.shape[-1]))},
        attrs={"P Type": PType,
               "L Type": LType,
               "xnames": ",".join(xnames),
               "ynames": ",".join(['PCP','ACC','LWP','IWP','OLR','OSR']),
               "tnames": ",".join(tnames),
               "Acceptance fraction": acc,
               "Autocorrelation time": float(tau) if np.isfinite(tau) else -1.0}
    )
    safe_to_netcdf(ds, ncfile)

    # --------------------------
    #   Plots (NEW FOLDER)
    # --------------------------
    Xa_plot = Xa.reshape(-1, Xa.shape[-1])

    out_png_agbg = os.path.join(out_plot_dir, f'fig_agbg_marginal_{base_name}.png')
    plot_ag_bg_marginal(
        Xa_plot, xnames, pmin_MCMC, pmax_MCMC, x_true,
        out_png=out_png_agbg,
        title_note=title_note
    )

    # NEW (2): 11×11 pairwise grid plot
    out_png_pair = os.path.join(out_plot_dir, f'fig_pairgrid_{base_name}.png')
    plot_pairwise_grid(
        Xa_plot, xnames, pmin_MCMC, pmax_MCMC, x_true,
        out_png=out_png_pair,
        title=f"CRM 11D pairwise marginals {title_note}"
    )

def main():
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

    print("\n=== CASE A (Fig.7-style): NO RADIATION ===")
    run_one_case(USE_RADIATION=False)

    print("\n=== CASE B (Fig.9-style): WITH RADIATION (OLR, OSR) ===")
    run_one_case(USE_RADIATION=True)

if __name__ == "__main__":
    mp.freeze_support()
    main()
