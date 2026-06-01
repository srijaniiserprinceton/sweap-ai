import numpy as np
import matplotlib.pyplot as plt
plt.ion()

from scipy.optimize import differential_evolution, least_squares
from scipy.ndimage import gaussian_filter1d

from sweapai.src import misc_functions as misc_fn


# ============================================================
# MODEL
# ============================================================

def one_bimax(theta, vpara, vperp):
    """
    theta = [log10_A, u_para, log10_vth_para, log10_vth_perp]
    """
    logA, u, log_vth_para, log_vth_perp = theta

    A = 10.0 ** logA
    vth_para = 10.0 ** log_vth_para
    vth_perp = 10.0 ** log_vth_perp

    return A * np.exp(
        -((vpara - u) / vth_para) ** 2
        - (vperp / vth_perp) ** 2
    )


def two_bimax(theta, vpara, vperp):
    return one_bimax(theta[:4], vpara, vperp) + one_bimax(theta[4:], vpara, vperp)


# ============================================================
# UTILS
# ============================================================

def clean_and_scale(vdf, vpara, vperp, floor=1e-30):
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

    vdf_min = np.nanmin(vdf[vdf > 0])
    vdf_scaled = vdf / vdf_min

    return vdf_scaled, vpara, vperp, vdf_min


def robust_log_residual(model, data, weights, floor):
    return np.sqrt(weights) * (
        np.log10(model + floor) - np.log10(data + floor)
    )


def sort_components_by_amplitude(theta):
    if theta[4] > theta[0]:
        return np.r_[theta[4:8], theta[0:4]]
    return theta


# ============================================================
# ONE-COMPONENT FIT FOR BOUNDS / BACKUP
# ============================================================

def fit_one_component_generic(vdf, vpara, vperp, floor):
    logv = np.log10(vdf + floor)

    vmin, vmax = np.nanmin(vpara), np.nanmax(vpara)
    vpara_range = vmax - vmin
    vperp_range = np.nanmax(vperp) - np.nanmin(vperp)

    dyn = np.nanmax(logv) - np.nanmin(logv)

    pmin = max(5.0, 0.01 * vpara_range)
    tmin = max(5.0, 0.03 * vperp_range)

    # pmax = 0.45 * vpara_range
    pmax = 0.90 * vpara_range
    tmax = 0.90 * vperp_range

    bounds = [
        (np.nanmin(logv), np.nanmax(logv) + 1.0),
        (vmin, vmax),
        (np.log10(pmin), np.log10(pmax)),
        (np.log10(tmin), np.log10(tmax)),
    ]

    weights = np.ones_like(vdf)
    # weights[logv > np.nanmin(logv) + 0.25 * dyn] = 4.0
    # weights[logv > np.nanmin(logv) + 0.60 * dyn] = 10.0

    def residual(theta):
        model = one_bimax(theta, vpara, vperp)
        return robust_log_residual(model, vdf, weights, floor)

    def objective(theta):
        r = residual(theta)
        return np.sum(r * r)

    de = differential_evolution(
        objective,
        bounds,
        maxiter=400,
        popsize=15,
        polish=False,
        seed=1,
    )

    lower = np.array([b[0] for b in bounds])
    upper = np.array([b[1] for b in bounds])

    lsq = least_squares(
        residual,
        de.x,
        bounds=(lower, upper),
        loss="soft_l1",
        f_scale=0.2,
        max_nfev=15000,
        xtol=1e-10,
        ftol=1e-10,
        gtol=1e-10,
    )

    return lsq.x, bounds, lsq


# ============================================================
# CORE PEAK + SHOULDER INITIALIZATION
# ============================================================

def initialize_core_and_shoulder_from_vpara_profile(
    vdf,
    vpara,
    vperp,
    bounds_one,
    floor,
    shoulder_side="higher",
):
    """
    Initialize two components from the v_parallel profile.

    Component 1:
        strongest peak in max-over-vperp profile.

    Component 2:
        strongest shoulder/curvature feature away from the peak.

    This works when the beam is a shoulder, not a clean second peak.
    """

    unique_vpara = np.unique(vpara)
    profile = np.zeros_like(unique_vpara, dtype=float)

    for i, vp in enumerate(unique_vpara):
        mask = vpara == vp
        profile[i] = np.nanmax(vdf[mask])

    log_profile = np.log10(profile + floor)

    if len(unique_vpara) > 5:
        smooth = gaussian_filter1d(log_profile, sigma=2)
    else:
        smooth = log_profile.copy()

    vpara_range = np.nanmax(vpara) - np.nanmin(vpara)
    vperp_range = np.nanmax(vperp) - np.nanmin(vperp)

    # --------------------------------------------------------
    # Component 1 = strongest peak
    # --------------------------------------------------------
    p1 = np.argmax(smooth)
    u1 = unique_vpara[p1]

    # --------------------------------------------------------
    # Component 2 = shoulder away from strongest peak
    # --------------------------------------------------------
    grad = np.gradient(smooth, unique_vpara)
    curv = np.gradient(grad, unique_vpara)

    sep = unique_vpara - u1

    if shoulder_side == "higher":
        candidate = sep > 0.12 * vpara_range
    elif shoulder_side == "lower":
        candidate = sep < -0.12 * vpara_range
    else:
        candidate = np.abs(sep) > 0.12 * vpara_range

    # Shoulder score:
    # high profile value + curvature change.
    # This targets plateau/shoulder regions rather than only local maxima.
    score = smooth + 2.0 * curv

    score[~candidate] = -np.inf

    if np.all(~np.isfinite(score)):
        candidate = np.abs(sep) > 0.12 * vpara_range
        score = smooth + 2.0 * curv
        score[~candidate] = -np.inf

    p2 = np.argmax(score)
    u2 = unique_vpara[p2]

    peak_indices = np.array([p1, p2])

    lower = np.array([b[0] for b in bounds_one])
    upper = np.array([b[1] for b in bounds_one])

    theta_list = []

    for u0 in [u1, u2]:
        local = np.abs(vpara - u0) < 0.16 * vpara_range

        if np.sum(local) < 10:
            local = np.abs(vpara - u0) < 0.25 * vpara_range

        amp0 = np.nanmax(vdf[local])

        w = np.maximum(vdf[local], floor)
        vp = vpara[local]
        vt = vperp[local]

        sig_para = np.sqrt(np.sum(w * (vp - u0) ** 2) / np.sum(w))
        sig_perp = np.sqrt(np.sum(w * vt ** 2) / np.sum(w))

        vth_para = np.sqrt(2.0) * sig_para
        vth_perp = np.sqrt(2.0) * sig_perp

        # Initialization caps only
        vth_para = np.clip(vth_para, 0.02 * vpara_range, 0.18 * vpara_range)
        vth_perp = np.clip(vth_perp, 0.05 * vperp_range, 0.75 * vperp_range)

        theta = np.array([
            np.log10(amp0 + floor),
            u0,
            np.log10(vth_para),
            np.log10(vth_perp),
        ])

        theta = np.clip(theta, lower + 1e-8, upper - 1e-8)
        theta_list.append(theta)

    theta0 = np.r_[theta_list[0], theta_list[1]]

    return theta0, unique_vpara, profile, smooth, peak_indices


# ============================================================
# MAIN FITTER
# ============================================================

def fit_two_bimax_generic(
    vdf,
    vpara,
    vperp,
    floor=1e-12,
    shoulder_side="higher",
    plot=True,
):
    """
    Two-component bi-Maxwellian fitter.

    - Scales VDF by minimum positive VDF.
    - Initializes component 1 at the main v_parallel peak.
    - Initializes component 2 at the shoulder/curvature feature.
    - Does not sort inside least_squares.
    - Sorts only after final fit.
    """

    vdf, vpara, vperp, vdf_min = clean_and_scale(
        vdf,
        vpara,
        vperp,
        floor=floor,
    )

    logv = np.log10(vdf + floor)
    dyn = np.nanmax(logv) - np.nanmin(logv)

    one0, bounds_one, one_result = fit_one_component_generic(
        vdf,
        vpara,
        vperp,
        floor,
    )

    theta0, unique_vpara, profile, smooth_profile, peak_indices = (
        initialize_core_and_shoulder_from_vpara_profile(
            vdf,
            vpara,
            vperp,
            bounds_one,
            floor,
            shoulder_side=shoulder_side,
        )
    )

    lower_one = np.array([b[0] for b in bounds_one])
    upper_one = np.array([b[1] for b in bounds_one])

    vpara_range = np.nanmax(vpara) - np.nanmin(vpara)
    vperp_range = np.nanmax(vperp) - np.nanmin(vperp)

    lower1 = lower_one.copy()
    upper1 = upper_one.copy()

    lower2 = lower_one.copy()
    upper2 = upper_one.copy()

    # Component-specific width bounds.
    upper1[2] = np.log10(0.40 * vpara_range)
    upper1[3] = np.log10(0.90 * vperp_range)

    upper2[2] = np.log10(0.22 * vpara_range)
    upper2[3] = np.log10(0.80 * vperp_range)

    upper1[2] = max(upper1[2], lower1[2] + 1e-6)
    upper1[3] = max(upper1[3], lower1[3] + 1e-6)
    upper2[2] = max(upper2[2], lower2[2] + 1e-6)
    upper2[3] = max(upper2[3], lower2[3] + 1e-6)

    lower = np.r_[lower1, lower2]
    upper = np.r_[upper1, upper2]

    theta0 = np.clip(theta0, lower + 1e-8, upper - 1e-8)

    # --------------------------------------------------------
    # Weights
    # --------------------------------------------------------
    weights = np.ones_like(vdf)

    weights[logv > np.nanmin(logv) + 0.20 * dyn] = 3.0
    weights[logv > np.nanmin(logv) + 0.50 * dyn] = 8.0
    weights[logv > np.nanmin(logv) + 0.80 * dyn] = 15.0

    u_init_1 = theta0[1]
    u_init_2 = theta0[5]

    peak_band = 0.10 * vpara_range

    peak_mask = (
        (np.abs(vpara - u_init_1) < peak_band)
        | (np.abs(vpara - u_init_2) < peak_band)
    )

    weights[peak_mask] *= 2.0

    def joint_residual(theta):
        model = two_bimax(theta, vpara, vperp)
        r = robust_log_residual(model, vdf, weights, floor)

        logA1, u1, lvpa1, lvpe1 = theta[:4]
        logA2, u2, lvpa2, lvpe2 = theta[4:]

        vthp1 = 10.0 ** lvpa1
        vtht1 = 10.0 ** lvpe1
        vthp2 = 10.0 ** lvpa2
        vtht2 = 10.0 ** lvpe2

        min_sep = 0.015 * vpara_range

        p_amp = max(0.0, logA2 - logA1)
        p_sep = max(0.0, min_sep - abs(u2 - u1)) / vpara_range

        max_aspect = 3.0
        p_aspect1 = max(0.0, vthp1 / vtht1 - max_aspect)
        p_aspect2 = max(0.0, vthp2 / vtht2 - max_aspect)

        penalty = np.array([
            40.0 * p_amp,
            25.0 * p_sep,
            60.0 * p_aspect1,
            80.0 * p_aspect2,
        ])

        return np.r_[r, penalty]

    joint = least_squares(
        joint_residual,
        theta0,
        bounds=(lower, upper),
        loss="soft_l1",
        f_scale=0.2,
        max_nfev=60000,
        xtol=1e-11,
        ftol=1e-11,
        gtol=1e-11,
    )

    theta_best = sort_components_by_amplitude(joint.x)

    result = {
        "theta": theta_best,
        "component1_theta": theta_best[:4],
        "component2_theta": theta_best[4:],
        "vdf": vdf,
        "vpara": vpara,
        "vperp": vperp,
        "vdf_fit": two_bimax(theta_best, vpara, vperp),
        "component1_fit": one_bimax(theta_best[:4], vpara, vperp),
        "component2_fit": one_bimax(theta_best[4:], vpara, vperp),
        "theta0": theta0,
        "one_component_initial": one0,
        "one_component_result": one_result,
        "joint_result": joint,
        "vdf_min_used_for_scaling": vdf_min,
        "bounds_one_component": bounds_one,
        "lower_bounds_joint": lower,
        "upper_bounds_joint": upper,
        "unique_vpara": unique_vpara,
        "vpara_profile": profile,
        "vpara_profile_smooth": smooth_profile,
        "peak_indices": peak_indices,
    }

    if plot:
        plot_two_bimax_generic(result, floor=floor)
        plot_vpara_profile_initialization(result)

    return result


# ============================================================
# PLOTTING
# ============================================================

def plot_two_bimax_generic(result, floor=1e-12):
    vdf = result["vdf"]
    vpara = result["vpara"]
    vperp = result["vperp"]

    comp1 = result["component1_fit"]
    comp2 = result["component2_fit"]
    total = result["vdf_fit"]

    log_data = np.log10(vdf + floor)
    log_comp1 = np.log10(comp1 + floor)
    log_comp2 = np.log10(comp2 + floor)
    log_total = np.log10(total + floor)
    resid = log_data - log_total

    vmin = np.floor(np.nanmin(log_data))
    vmax = np.ceil(np.nanmax(log_data))
    levels = np.linspace(vmin, vmax, 14)

    fig, axes = plt.subplots(1, 5, figsize=(24, 5), constrained_layout=True)

    panels = [
        ("Original VDF", log_data),
        ("Component 1", log_comp1),
        ("Component 2", log_comp2),
        ("Component 1 + 2", log_total),
    ]

    for ax, (title, z) in zip(axes[:4], panels):
        cf = ax.tricontourf(vperp, vpara, z, levels=levels, cmap="jet")
        ax.scatter(vperp, vpara, s=2, c="k")
        ax.set_title(title)
        ax.set_xlabel(r"$v_\perp$")
        ax.set_ylabel(r"$v_\parallel$")
        fig.colorbar(cf, ax=ax)

    vmax_resid = np.nanpercentile(np.abs(resid), 98)

    if not np.isfinite(vmax_resid) or vmax_resid == 0:
        vmax_resid = 1.0

    cf = axes[4].tricontourf(
        vperp,
        vpara,
        resid,
        levels=np.linspace(-vmax_resid, vmax_resid, 21),
        cmap="seismic",
    )

    axes[4].scatter(vperp, vpara, s=2, c="k")
    axes[4].set_title("Residual")
    axes[4].set_xlabel(r"$v_\perp$")
    axes[4].set_ylabel(r"$v_\parallel$")
    fig.colorbar(cf, ax=axes[4])

    return fig, axes


def plot_vpara_profile_initialization(result):
    unique_vpara = result["unique_vpara"]
    profile = result["vpara_profile"]
    smooth = result["vpara_profile_smooth"]
    peaks = result["peak_indices"]

    plt.figure(figsize=(7, 4))
    plt.plot(unique_vpara, np.log10(profile), "o-", label="profile")
    plt.plot(unique_vpara, smooth, "-", label="smoothed")
    plt.scatter(
        unique_vpara[peaks],
        smooth[peaks],
        s=90,
        marker="x",
        label="initial core/shoulder",
    )
    plt.xlabel(r"$v_\parallel$")
    plt.ylabel(r"$\log_{10}(\max_{v_\perp} f)$")
    plt.title("Core peak + shoulder initialization")
    plt.legend()
    plt.tight_layout()


# ============================================================
# PRINTING
# ============================================================

def print_two_bimax_generic(result):
    theta = result["theta"]

    labels = [
        "logA_1", "u_1", "log_vth_para_1", "log_vth_perp_1",
        "logA_2", "u_2", "log_vth_para_2", "log_vth_perp_2",
    ]

    for lab, val in zip(labels, theta):
        print(f"{lab:22s} = {val:.6g}")

    print("\nLinear parameters:")
    print(f"A_1          = {10**theta[0]:.6g}")
    print(f"u_1          = {theta[1]:.6g}")
    print(f"vth_para_1   = {10**theta[2]:.6g}")
    print(f"vth_perp_1   = {10**theta[3]:.6g}")

    print()
    print(f"A_2          = {10**theta[4]:.6g}")
    print(f"u_2          = {theta[5]:.6g}")
    print(f"vth_para_2   = {10**theta[6]:.6g}")
    print(f"vth_perp_2   = {10**theta[7]:.6g}")

    print()
    print(f"VDF scaling: divided by min positive VDF = {result['vdf_min_used_for_scaling']:.6e}")

    print("\nJoint upper bounds:")
    print("component 1 upper vth_para =", 10**result["upper_bounds_joint"][2])
    print("component 1 upper vth_perp =", 10**result["upper_bounds_joint"][3])
    print("component 2 upper vth_para =", 10**result["upper_bounds_joint"][6])
    print("component 2 upper vth_perp =", 10**result["upper_bounds_joint"][7])

    print("\nInitialization:")
    print("theta0 =", result["theta0"])


# ============================================================
# USAGE
# ============================================================

if __name__ == "__main__":
    sleprec = misc_funcs.read_pickle(
        "Outputs/supres_dict_hybrid_8_2000_20200126_142800_142807"
    )

    vpara = sleprec[0]["vpara_supres"]
    vperp = sleprec[0]["vperp_supres"]

    vdf_rec = np.power(10, sleprec[0]["vdf_supres"][1])
    vdf_rec = np.reshape(vdf_rec, (49, 49))
    vdf_rec = vdf_rec.T.flatten()

    result = fit_two_bimax_generic(
        vdf_rec,
        vpara,
        vperp,
        floor=1e-12,
        shoulder_side="higher",
        plot=True,
    )

    print_two_bimax_generic(result)