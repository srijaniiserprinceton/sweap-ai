
import time
from dataclasses import dataclass
from typing import Optional, Dict, Tuple

import numpy as np

# These come from the earlier translated files you already have.
from fc_fov_sampling import (
    FrustumSamples,
    uniform_sample_frustum,
    calculate_effective_area_cm2,
    cup_response_numflux_spectrum,
)
from vdf import new_vdf


@dataclass
class DeterministicRayGrid:
    """
    Deterministic frustum-adapted quadrature grid.

    Parameters
    ----------
    s : ndarray, shape (Nrays,)
        Normalized radial coordinates in the unit disk, 0 <= s <= 1.

    varphi : ndarray, shape (Nrays,)
        Azimuthal angle in the unit disk, radians.

    phi : ndarray, shape (Nrays,)
        Cup x-deflection angle for each ray, radians.

    theta : ndarray, shape (Nrays,)
        Cup y-deflection angle for each ray, radians.

    ang_w : ndarray, shape (Nrays,)
        Angular quadrature weight on the unit disk. These weights integrate
        functions over du dv on the unit disk, i.e.
            ∫_disk h(u, v) du dv ≈ Σ_j ang_w[j] h(u_j, v_j)

    xi : ndarray, shape (Nz,)
        Reference slab coordinates on [0, 1].

    z_w : ndarray, shape (Nz,)
        Reference 1D quadrature weights on [0, 1].

    effarea : ndarray, shape (4, Nrays)
        Effective area evaluated once per ray.
    """
    s: np.ndarray
    varphi: np.ndarray
    phi: np.ndarray
    theta: np.ndarray
    ang_w: np.ndarray
    xi: np.ndarray
    z_w: np.ndarray
    effarea: np.ndarray


def gauss_legendre_on_interval(a: float, b: float, n: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return Gauss-Legendre nodes and weights on [a, b]."""
    x, w = np.polynomial.legendre.leggauss(n)
    nodes = 0.5 * (b - a) * x + 0.5 * (b + a)
    weights = 0.5 * (b - a) * w
    return nodes, weights


def build_disk_rule(ns: int = 8,
                    nphi: int = 16,
                    radial_rule: str = "gauss") -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build a deterministic quadrature rule on the unit disk using

        u = s cos(varphi),   v = s sin(varphi),   0<=s<=1, 0<=varphi<2π

    with disk area element:
        du dv = s ds dvarphi

    We choose:
    - trapezoidal rule in varphi
    - either Gauss-Legendre or midpoint rule in s

    Returns
    -------
    s, varphi, u, v, w_disk : arrays of shape (ns*nphi,)
        Quadrature nodes on the unit disk and their weights.
    """
    if radial_rule == "gauss":
        # Nodes/weights on [0,1] for the integral ∫_0^1 f(s) s ds.
        # We use standard Gauss-Legendre on [0,1] and explicitly include the s factor.
        s_nodes, s_weights = gauss_legendre_on_interval(0.0, 1.0, ns)
        radial_weights = s_weights * s_nodes
    elif radial_rule == "midpoint":
        edges = np.linspace(0.0, 1.0, ns + 1)
        s_nodes = 0.5 * (edges[:-1] + edges[1:])
        ds = edges[1:] - edges[:-1]
        radial_weights = ds * s_nodes
    else:
        raise ValueError("radial_rule must be 'gauss' or 'midpoint'")

    varphi_nodes = np.linspace(0.0, 2.0 * np.pi, nphi, endpoint=False)
    varphi_weights = np.full(nphi, 2.0 * np.pi / nphi)

    S, PHI = np.meshgrid(s_nodes, varphi_nodes, indexing="ij")
    WR, WA = np.meshgrid(radial_weights, varphi_weights, indexing="ij")

    s = S.ravel()
    varphi = PHI.ravel()
    u = s * np.cos(varphi)
    v = s * np.sin(varphi)
    w_disk = (WR * WA).ravel()

    return s, varphi, u, v, w_disk


def build_deterministic_ray_grid(alpha_rad: float,
                                 ns: int = 8,
                                 nphi: int = 16,
                                 nz: int = 4,
                                 radial_rule: str = "gauss") -> DeterministicRayGrid:
    """
    Build a fixed deterministic node set with the 'sliding volume' feature.

    Geometry:
        x = z * tan(alpha_max) * u
        y = z * tan(alpha_max) * v
        with u^2 + v^2 <= 1

    For each fixed (u, v), the ray direction is fixed across every slab.
    Therefore A(theta, phi) is evaluated once and reused forever.

    The full frustum integral for one slab [z0, z1] becomes

        ∫_{z0}^{z1} ∫_disk g(x,y,z) (z tan(alpha))^2 du dv dz

    which we approximate by a tensor quadrature:
        Σ_m Σ_j  z_w[m] * ang_w[j] * (z_m tan(alpha))^2 * g(...)

    Here:
    - ang_w integrates du dv on the unit disk
    - z_w integrates dz on [0,1], then gets multiplied by Δz per slab
    """
    s, varphi, u, v, ang_w = build_disk_rule(ns=ns, nphi=nphi, radial_rule=radial_rule)

    tan_alpha = np.tan(alpha_rad)
    # Ray angles are fixed by normalized disk coordinates.
    # Since x/z = tan(alpha)*u and y/z = tan(alpha)*v, these do not depend on z.
    phi = np.arctan(tan_alpha * u)
    theta = np.arctan(tan_alpha * v)

    effarea = calculate_effective_area_cm2(phi, theta)

    # Reference slab coordinate ξ in [0,1], shared by every slab.
    xi, z_w = gauss_legendre_on_interval(0.0, 1.0, nz)

    return DeterministicRayGrid(
        s=s,
        varphi=varphi,
        phi=phi,
        theta=theta,
        ang_w=ang_w,
        xi=xi,
        z_w=z_w,
        effarea=effarea,
    )


def deterministic_response_numflux_spectrum(vz_lo: np.ndarray,
                                            vz_hi: np.ndarray,
                                            fv,
                                            grid: DeterministicRayGrid,
                                            alpha_rad: float) -> np.ndarray:
    """
    Deterministic frustum quadrature with fixed rays and sliding z-nodes.

    Parameters
    ----------
    vz_lo, vz_hi : ndarray, shape (nsteps,)
        Lower and upper z-bounds for each SPC slab.

    fv : VDF object
        Must support:
            f = fv.evaluate(vxyz)
        where vxyz has shape (N, 3).

    grid : DeterministicRayGrid
        Precomputed deterministic rays and quadrature weights.

    alpha_rad : float
        Cup half-angle.

    Returns
    -------
    numflux : ndarray, shape (4, nsteps)
        Synthetic SPC number flux spectrum.
    """
    vz_lo = np.asarray(vz_lo, dtype=float)
    vz_hi = np.asarray(vz_hi, dtype=float)

    nsensors = grid.effarea.shape[0]
    nsteps = vz_lo.size
    nrays = grid.phi.size
    nz = grid.xi.size

    tan_alpha = np.tan(alpha_rad)
    cmperkm = 1.0e5

    # Recover unit-disk coordinates from the stored angles. This avoids storing u,v separately.
    # Because tan(phi)=tan(alpha)*u and tan(theta)=tan(alpha)*v:
    u = np.tan(grid.phi) / tan_alpha
    v = np.tan(grid.theta) / tan_alpha

    numflux = np.zeros((nsensors, nsteps), dtype=float)

    for k in range(nsteps):
        z0 = vz_lo[k]
        dz = vz_hi[k] - vz_lo[k]

        # Slide the same normalized z-nodes into the current slab.
        z_nodes = z0 + dz * grid.xi                      # shape (nz,)
        wz = dz * grid.z_w                               # shape (nz,)

        # Build full tensor nodes for this slab.
        # Shapes after broadcasting:
        #   Z : (nz, nrays)
        Z = z_nodes[:, None]                 # (nz, 1)
        U = u[None, :]                       # (1, nrays)
        V = v[None, :]                       # (1, nrays)

        Zfull = np.broadcast_to(Z, (nz, nrays))   # (nz, nrays)

        X = Zfull * tan_alpha * U
        Y = Zfull * tan_alpha * V

        # Flatten for one VDF call
        vxyz = np.column_stack([X.ravel(), Y.ravel(), Zfull.ravel()])

        fvals = fv.evaluate(vxyz).reshape(nz, nrays)

        # Full quadrature weight:
        #   du dv weight on disk      -> grid.ang_w[j]
        #   dz weight in slab         -> wz[m]
        #   frustum Jacobian          -> (z tan(alpha))^2
        # Therefore:
        #   W[m,j] = wz[m] * ang_w[j] * (z_m tan(alpha))^2
        W = (wz[:, None] *
             grid.ang_w[None, :] *
             (Z * tan_alpha) ** 2)

        vz_component = Zfull # here z itself is v_z in cup coordinates

        # Integrand for each sensor:
        #   cmperkm * v_z * A(theta,phi) * f * dV
        common = cmperkm * vz_component * fvals * W

        for s_idx in range(nsensors):
            numflux[s_idx, k] = np.sum(common * grid.effarea[s_idx][None, :])

    return numflux


def reference_mc_spectrum(vz_lo: np.ndarray,
                          vz_hi: np.ndarray,
                          fv,
                          alpha_rad: float,
                          n_mc: int = 1000,
                          seed: int = 12345) -> np.ndarray:
    """
    Reference Monte Carlo spectrum using your translated .pro-style code.
    """
    rng = np.random.default_rng(seed)
    ref_samples = uniform_sample_frustum(vz_lo[0], vz_hi[0], alpha_rad, n=n_mc, rng=rng)
    ref_effarea = calculate_effective_area_cm2(ref_samples.phi, ref_samples.theta)
    return cup_response_numflux_spectrum(vz_lo, vz_hi, fv, ref_effarea, ref_samples)


def build_default_spectrum_grid(nsteps: int = 30) -> Tuple[np.ndarray, np.ndarray]:
    """Same default spectrum bins used earlier: 100-120, 120-140, ..."""
    vz_lo = 100.0 + 20.0 * np.arange(nsteps)
    vz_hi = vz_lo + 20.0
    return vz_lo, vz_hi


def compare_quadrature_vs_mc(params: Optional[Dict[str, float]] = None,
                             alpha_deg: float = 44.0,
                             ns: int = 8,
                             nphi: int = 16,
                             nz: int = 4,
                             radial_rule: str = "gauss",
                             n_mc: int = 1000,
                             n_repeat: int = 3,
                             mc_seed: int = 12345) -> Dict[str, object]:
    """
    Compare deterministic fixed-ray quadrature against the reference Monte Carlo code.

    Returns
    -------
    dict with:
        det_flux, mc_flux, abs_err, rel_err, time_det, time_mc, grid
    """
    if params is None:
        params = {"vx": 100.0, "vy": 50.0, "vz": 250.0, "w": 50.0, "n": 1000.0}

    alpha_rad = np.deg2rad(alpha_deg)
    vz_lo, vz_hi = build_default_spectrum_grid()
    fv = new_vdf("maxwellian_vdf", params)

    # Build deterministic grid once.
    grid = build_deterministic_ray_grid(
        alpha_rad=alpha_rad,
        ns=ns,
        nphi=nphi,
        nz=nz,
        radial_rule=radial_rule,
    )

    # Time deterministic solver
    t0 = time.perf_counter()
    for _ in range(n_repeat):
        det_flux = deterministic_response_numflux_spectrum(vz_lo, vz_hi, fv, grid, alpha_rad)
    time_det = (time.perf_counter() - t0) / n_repeat

    # Time reference Monte Carlo solver
    t0 = time.perf_counter()
    for _ in range(n_repeat):
        mc_flux = reference_mc_spectrum(vz_lo, vz_hi, fv, alpha_rad, n_mc=n_mc, seed=mc_seed)
    time_mc = (time.perf_counter() - t0) / n_repeat

    abs_err = np.abs(det_flux - mc_flux)
    rel_err = abs_err / (np.abs(mc_flux) + 1e-30)

    return {
        "det_flux": det_flux,
        "mc_flux": mc_flux,
        "abs_err": abs_err,
        "rel_err": rel_err,
        "time_det": time_det,
        "time_mc": time_mc,
        "grid": grid,
        "num_det_nodes_per_slab": grid.phi.size * grid.xi.size,
        "num_rays": grid.phi.size,
        "num_z_nodes": grid.xi.size,
    }


def convergence_scan(params: Optional[Dict[str, float]] = None,
                     alpha_deg: float = 44.0,
                     configs: Optional[list] = None,
                     n_mc_ref: int = 10000,
                     mc_seed: int = 12345) -> list:
    """
    Run a small convergence scan for several deterministic quadrature settings.

    Each config should be a dict like:
        {"ns": 6, "nphi": 12, "nz": 4, "radial_rule": "gauss"}
    """
    if params is None:
        params = {"vx": 100.0, "vy": 50.0, "vz": 250.0, "w": 50.0, "n": 1000.0}

    if configs is None:
        configs = [
            {"ns": 4, "nphi": 12, "nz": 3, "radial_rule": "gauss"},
            {"ns": 6, "nphi": 12, "nz": 4, "radial_rule": "gauss"},
            {"ns": 8, "nphi": 16, "nz": 4, "radial_rule": "gauss"},
            {"ns": 10, "nphi": 16, "nz": 5, "radial_rule": "gauss"},
        ]

    alpha_rad = np.deg2rad(alpha_deg)
    vz_lo, vz_hi = build_default_spectrum_grid()
    fv = new_vdf("maxwellian_vdf", params)

    # Reference Monte Carlo once, dense.
    mc_flux = reference_mc_spectrum(vz_lo, vz_hi, fv, alpha_rad, n_mc=n_mc_ref, seed=mc_seed)

    results = []
    for cfg in configs:
        grid = build_deterministic_ray_grid(alpha_rad=alpha_rad, **cfg)

        t0 = time.perf_counter()
        det_flux = deterministic_response_numflux_spectrum(vz_lo, vz_hi, fv, grid, alpha_rad)
        time_det = time.perf_counter() - t0

        abs_err = np.abs(det_flux - mc_flux)
        rel_err = abs_err / (np.abs(mc_flux) + 1e-30)

        results.append({
            **cfg,
            "num_rays": grid.phi.size,
            "num_det_nodes_per_slab": grid.phi.size * grid.xi.size,
            "time_det": time_det,
            "max_rel_err": float(np.max(rel_err)),
            "mean_rel_err": float(np.mean(rel_err)),
            "median_rel_err": float(np.median(rel_err)),
        })

    return results



def angular_constants_from_det(grid):
    # C_s = ∫_disk A_s du dv  ≈ Σ_j w_j A_s(j)
    return np.array([np.sum(grid.ang_w * grid.effarea[s]) for s in range(4)])

def angular_constants_from_spectrum(flux, vz_lo, vz_hi, alpha_rad):
    # From:
    # F_s(k) = 1e5 * tan^2(alpha) * ((z1^4-z0^4)/4) * C_s
    denom = 1e5 * (np.tan(alpha_rad)**2) * 0.25 * (vz_hi**4 - vz_lo**4)
    # divide each sensor spectrum by denom, should be flat in k
    C = flux / denom[None, :]
    return C


if __name__ == "__main__":
    # Example one-shot comparison
    out = compare_quadrature_vs_mc(
        params={"vx": 100.0, "vy": 50.0, "vz": 250.0, "w": 50.0, "n": 1000.0},
        alpha_deg=44.0,
        ns=16,
        nphi=24,
        nz=12,
        radial_rule="gauss",
        n_mc=1000,
        n_repeat=3,
    )

    print("Deterministic time per call:", out["time_det"])
    print("Reference MC time per call:", out["time_mc"])
    print("Deterministic nodes per slab:", out["num_det_nodes_per_slab"])
    print("Max relative error:", np.max(out["rel_err"]))
    print("Mean relative error:", np.mean(out["rel_err"]))

    print("\nConvergence scan:")
    scan = convergence_scan()
    for row in scan:
        print(row)

    alpha_rad = np.deg2rad(44.0)
    vz_lo = 100.0 + 20.0 * np.arange(30)
    vz_hi = vz_lo + 20.0

    # Build deterministic grid once.
    grid = build_deterministic_ray_grid(
        alpha_rad=alpha_rad,
        ns=16,
        nphi=24,
        nz=12,
        radial_rule='gauss',
    )

    # deterministic constants from weights directly
    C_det_direct = angular_constants_from_det(grid)

    # constants inferred from full spectra
    C_det_from_flux = angular_constants_from_spectrum(out['det_flux'], vz_lo, vz_hi, alpha_rad)
    C_mc_from_flux  = angular_constants_from_spectrum(out['mc_flux'],  vz_lo, vz_hi, alpha_rad)

    print("det direct:", C_det_direct)
    print("det from flux mean over bins:", C_det_from_flux.mean(axis=1))
    print("mc  from flux mean over bins:", C_mc_from_flux.mean(axis=1))
    print("mc std over bins:", C_mc_from_flux.std(axis=1))
