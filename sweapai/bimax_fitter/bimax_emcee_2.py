import numpy as np
import matplotlib.pyplot as plt
plt.ion()

import emcee
from scipy.optimize import minimize
from scipy.ndimage import gaussian_filter1d

# from sweapai.src import misc_funcs

# ============================================================
# MODEL
# ============================================================

def maxwellian_2d(params, vpara, vperp):
    """
    Single drifting anisotropic Maxwellian.

    params = [amp_log10, u, vth_para, vth_perp]
    """

    amp_log10, u, vth_para, vth_perp = params

    return 10**amp_log10 * np.exp(
        -((vpara - u) / vth_para)**2
        - (vperp / vth_perp)**2
    )


def two_component_bimax(theta, vpara, vperp):
    """
    Two-component bi-Maxwellian: core + beam.

    theta =
    [
        u_core,
        u_beam,
        log10_vth_core,
        log10_aniso_core,
        log10_vth_beam,
        log10_aniso_beam,
        amp_core_log10,
        beam_core_log10_ratio
    ]
    """

    (
        u_core,
        u_beam,
        log_vth_core,
        log_aniso_core,
        log_vth_beam,
        log_aniso_beam,
        amp_core,
        beam_core_ratio,
    ) = theta

    vth_core_para = 10**log_vth_core
    aniso_core = 10**log_aniso_core
    vth_core_perp = aniso_core * vth_core_para

    vth_beam_para = 10**log_vth_beam
    aniso_beam = 10**log_aniso_beam
    vth_beam_perp = aniso_beam * vth_beam_para

    amp_beam = amp_core + beam_core_ratio

    core = maxwellian_2d(
        [amp_core, u_core, vth_core_para, vth_core_perp],
        vpara,
        vperp,
    )

    beam = maxwellian_2d(
        [amp_beam, u_beam, vth_beam_para, vth_beam_perp],
        vpara,
        vperp,
    )

    return core + beam


# ============================================================
# INITIALIZATION
# ============================================================

def initial_guess_two_peaks(vdf, vpara, vperp, baseline=1e-8):
    """
    Estimate core and beam initial locations from the VDF itself.
    """

    vdf = np.asarray(vdf, dtype=float).ravel()
    vpara = np.asarray(vpara, dtype=float).ravel()
    vperp = np.asarray(vperp, dtype=float).ravel()

    vdf = vdf / np.nanmax(vdf)

    near_axis = np.abs(vperp) < np.nanpercentile(np.abs(vperp), 25)

    vp = vpara[near_axis]
    f = vdf[near_axis]

    order = np.argsort(vp)
    vp = vp[order]
    f = f[order]

    logf = np.log10(f + baseline)
    logf_smooth = gaussian_filter1d(logf, sigma=3)

    core_idx = np.argmax(logf_smooth)
    u_core_init = vp[core_idx]

    sep = np.abs(vp - u_core_init)
    beam_candidates = sep > 100.0

    if np.any(beam_candidates):
        beam_idx_local = np.argmax(logf_smooth[beam_candidates])
        u_beam_init = vp[beam_candidates][beam_idx_local]
    else:
        u_beam_init = u_core_init + 200.0

    p0 = np.array([
        u_core_init,
        u_beam_init,
        np.log10(70.0),
        np.log10(1.0),
        np.log10(90.0),
        np.log10(1.5),
        0.0,
        -1.2,
    ])

    return p0


# ============================================================
# PROBABILITY
# ============================================================

def log_prior(theta, bounds):
    for value, (lo, hi) in zip(theta, bounds):
        if not (lo < value < hi):
            return -np.inf
    return 0.0


def log_likelihood(theta, vdf, vpara, vperp, weights=None, baseline=1e-6):
    model = two_component_bimax(theta, vpara, vperp)

    log_data = np.log10(vdf + baseline)
    log_model = np.log10(model + baseline)

    residual = log_data - log_model

    if weights is None:
        weights = np.ones_like(residual)

    chi2 = np.sum(weights * residual**2)

    return -0.5 * chi2


def log_probability(theta, vdf, vpara, vperp, bounds, weights=None, baseline=1e-6):
    lp = log_prior(theta, bounds)

    if not np.isfinite(lp):
        return -np.inf

    return lp + log_likelihood(
        theta,
        vdf,
        vpara,
        vperp,
        weights=weights,
        baseline=baseline,
    )


# ============================================================
# FITTER
# ============================================================

def fit_bimaxwellian(
    vdf,
    vpara,
    vperp,
    nwalkers=64,
    nsteps=5000,
    burn=2000,
    thin=10,
    baseline=1e-6,
    run_mcmc=True,
):
    """
    Fit flattened VDF(vpara, vperp) with a two-component bi-Maxwellian.
    """

    vdf = np.asarray(vdf, dtype=float).ravel()
    vpara = np.asarray(vpara, dtype=float).ravel()
    vperp = np.asarray(vperp, dtype=float).ravel()

    good = (
        np.isfinite(vdf)
        & np.isfinite(vpara)
        & np.isfinite(vperp)
        & (vdf > 0)
    )

    vdf = vdf[good]
    vpara = vpara[good]
    vperp = vperp[good]

    vdf = vdf / np.nanmax(vdf)

    p0 = initial_guess_two_peaks(vdf, vpara, vperp, baseline=baseline)

    bounds = [
        (np.nanmin(vpara), np.nanmax(vpara)),       # u_core
        (np.nanmin(vpara), np.nanmax(vpara)),       # u_beam
        (np.log10(20.0), np.log10(300.0)),          # core vth_para
        (np.log10(0.1), np.log10(10.0)),            # core anisotropy
        (np.log10(20.0), np.log10(400.0)),          # beam vth_para
        (np.log10(0.1), np.log10(20.0)),            # beam anisotropy
        (-4.0, 0.5),                                # core log amp
        (-4.0, -0.05),                              # beam/core log ratio
    ]

    log_vdf = np.log10(vdf + baseline)

    weights = np.ones_like(vdf)

    # Beam/collar emphasized
    weights[(log_vdf < -0.5) & (log_vdf > -3.5)] = 20.0

    # Core peak deemphasized
    weights[log_vdf > -0.3] = 2.0

    def neg_log_prob(theta):
        lp = log_probability(
            theta,
            vdf,
            vpara,
            vperp,
            bounds,
            weights=weights,
            baseline=baseline,
        )

        if not np.isfinite(lp):
            return 1e100

        return -lp

    opt = minimize(
        neg_log_prob,
        p0,
        method="Nelder-Mead",
        options={
            "maxiter": 8000,
            "xatol": 1e-6,
            "fatol": 1e-6,
        },
    )

    if opt.success:
        p_start = opt.x
    else:
        print("Warning: optimizer did not fully converge. Using best available result.")
        p_start = opt.x

    if not run_mcmc:
        vdf_fit = two_component_bimax(p_start, vpara, vperp)

        return {
            "best": p_start,
            "lower": None,
            "upper": None,
            "samples": None,
            "sampler": None,
            "vdf_fit": vdf_fit,
            "vdf_data": vdf,
            "vpara": vpara,
            "vperp": vperp,
            "bounds": bounds,
            "weights": weights,
            "optimizer_result": opt,
        }

    ndim = len(p_start)

    if nwalkers < 4 * ndim:
        nwalkers = 4 * ndim

    scale = np.array([
        10.0,
        10.0,
        0.03,
        0.05,
        0.03,
        0.05,
        0.03,
        0.05,
    ])

    pos = p_start + scale * np.random.randn(nwalkers, ndim)

    for j, (lo, hi) in enumerate(bounds):
        pos[:, j] = np.clip(pos[:, j], lo + 1e-8, hi - 1e-8)

    sampler = emcee.EnsembleSampler(
        nwalkers,
        ndim,
        log_probability,
        args=(vdf, vpara, vperp, bounds, weights, baseline),
    )

    sampler.run_mcmc(pos, nsteps, progress=True)

    samples = sampler.get_chain(discard=burn, thin=thin, flat=True)

    best = np.quantile(samples, 0.50, axis=0)
    lower = np.quantile(samples, 0.16, axis=0)
    upper = np.quantile(samples, 0.84, axis=0)

    vdf_fit = two_component_bimax(best, vpara, vperp)

    return {
        "best": best,
        "lower": lower,
        "upper": upper,
        "samples": samples,
        "sampler": sampler,
        "vdf_fit": vdf_fit,
        "vdf_data": vdf,
        "vpara": vpara,
        "vperp": vperp,
        "bounds": bounds,
        "weights": weights,
        "optimizer_result": opt,
    }


# ============================================================
# PLOTTING
# ============================================================

def plot_bimax_comparison(
    vdf,
    vpara,
    vperp,
    theta,
    baseline=1e-6,
    levels=np.linspace(-6, 0, 13),
):
    """
    Compare original VDF, fitted bi-Maxwellian, and residual.
    """

    vdf = np.asarray(vdf, dtype=float).ravel()
    vpara = np.asarray(vpara, dtype=float).ravel()
    vperp = np.asarray(vperp, dtype=float).ravel()

    good = (
        np.isfinite(vdf)
        & np.isfinite(vpara)
        & np.isfinite(vperp)
        & (vdf > 0)
    )

    vdf = vdf[good]
    vpara = vpara[good]
    vperp = vperp[good]

    vdf = vdf / np.nanmax(vdf)

    vdf_fit = two_component_bimax(theta, vpara, vperp)

    log_data = np.log10(vdf + baseline)
    log_fit = np.log10(vdf_fit + baseline)

    residual = log_data - log_fit

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(15, 5),
        constrained_layout=True,
    )

    ax = axes[0]
    cf = ax.tricontourf(vperp, vpara, log_data, levels=levels, cmap="jet")
    ax.scatter(vperp, vpara, c="k", s=2)
    ax.set_title("Original VDF")
    ax.set_xlabel(r"$v_\perp$")
    ax.set_ylabel(r"$v_\parallel$")
    fig.colorbar(cf, ax=ax)

    ax = axes[1]
    cf = ax.tricontourf(vperp, vpara, log_fit, levels=levels, cmap="jet")
    ax.scatter(vperp, vpara, c="k", s=2)
    ax.set_title("Bi-Maxwellian Fit")
    ax.set_xlabel(r"$v_\perp$")
    ax.set_ylabel(r"$v_\parallel$")
    fig.colorbar(cf, ax=ax)

    ax = axes[2]
    vmax = np.nanmax(np.abs(residual))

    cf = ax.tricontourf(
        vperp,
        vpara,
        residual,
        levels=np.linspace(-vmax, vmax, 21),
        cmap="seismic",
    )

    ax.scatter(vperp, vpara, c="k", s=2)
    ax.set_title("Residual: log10(data) - log10(fit)")
    ax.set_xlabel(r"$v_\perp$")
    ax.set_ylabel(r"$v_\parallel$")
    fig.colorbar(cf, ax=ax)

    return fig, axes


def print_fit_result(theta):
    labels = [
        "u_core",
        "u_beam",
        "log10_vth_core",
        "log10_aniso_core",
        "log10_vth_beam",
        "log10_aniso_beam",
        "amp_core_log10",
        "beam_core_log10_ratio",
    ]

    for name, value in zip(labels, theta):
        print(f"{name:25s} = {value:.6g}")

    print()
    print("Derived physical parameters:")
    print(f"vth_core_para       = {10**theta[2]:.6g}")
    print(f"aniso_core          = {10**theta[3]:.6g}")
    print(f"vth_core_perp       = {10**theta[2] * 10**theta[3]:.6g}")
    print(f"vth_beam_para       = {10**theta[4]:.6g}")
    print(f"aniso_beam          = {10**theta[5]:.6g}")
    print(f"vth_beam_perp       = {10**theta[4] * 10**theta[5]:.6g}")
    print(f"amp_core            = {10**theta[6]:.6g}")
    print(f"amp_beam            = {10**(theta[6] + theta[7]):.6g}")


if __name__ == "__main__":
    sleprec = misc_funcs.read_pickle('Outputs/supres_dict_hybrid_8_2000_20200126_142800_142807')
    vpara = sleprec[0]['vpara_supres']
    vperp = sleprec[0]['vperp_supres']
    vdf_rec = np.power(10, sleprec[0]['vdf_supres'][1])
    vdf_rec = np.reshape(vdf_rec, (49, 49))
    vdf_rec = vdf_rec.T.flatten()

    result = fit_bimaxwellian(
        vdf_rec,
        vpara,
        vperp,
        nwalkers=64,
        nsteps=5000,
        burn=2000,
        thin=10,
        run_mcmc=True,
    )

    print_fit_result(result["best"])

    plot_bimax_comparison(
        vdf_rec,
        vpara,
        vperp,
        result["best"],
    )