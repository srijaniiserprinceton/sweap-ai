from __future__ import annotations

"""
All-in-one Python translation of the three IDL / .pro files:

- fc_fov_sampling.pro
- vdf.pro
- psp_swp_spc_coldspot.pro

This module is designed to be runnable as a starting point for
forward-modeling and convergence tests.

Notes
-----
1. The SPC geometry code below imports the translated helper modules that were
   generated from the uploaded .pro files:
      - vdf_py.py
      - psp_spc_coldspot_py.py
   Those files should be kept in the same directory as this script.

2. The effective-area wrapper preserves the original IDL logic exactly:
      area_cm2 = 100 * psp_swp_spc_coldspot(phi_deg, theta_deg)
   even though the comment in the original code about mm^2 -> cm^2 is a bit
   confusing physically. This script mirrors the .pro implementation.

3. The reference-sample / sliding-volume method is also preserved exactly:
   samples are generated once in the first slab, then mapped into later slabs
   by holding the ray direction fixed and rescaling the weights by the Jacobian.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
from line_profiler import profile

# -----------------------------------------------------------------------------
# Import the translated support modules from the same folder.
# -----------------------------------------------------------------------------

_HERE = Path(__file__).resolve().parent
import sys
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from vdf import VDF, new_vdf  # type: ignore
from psp_spc_coldspot_MT import psp_swp_spc_coldspot  # type: ignore


# =============================================================================
# Data container
# =============================================================================

@dataclass
class FrustumSamples:
    """
    Sample points inside one SPC field-of-view frustum.

    Attributes
    ----------
    phi, theta : ndarray, shape (N,)
        Deflection angles [radians].
        The original IDL defines
            phi   = atan(x / z)
            theta = atan(y / z)
        implemented robustly here using atan2(x, z) and atan2(y, z).

    r : ndarray, shape (N,)
        Speed magnitude sqrt(x^2 + y^2 + z^2).

    dvol : ndarray, shape (N,)
        Volume weight associated with each point.

    zlower, zupper : float
        Lower and upper z-bounds of the slab.

    alpha_rad : float
        Half-angle of the SPC cone [radians].
    """
    phi: np.ndarray
    theta: np.ndarray
    r: np.ndarray
    dvol: np.ndarray
    zlower: float
    zupper: float
    alpha_rad: float


# =============================================================================
# Effective area wrapper
# =============================================================================

def calculate_effective_area_cm2(phi_rad: np.ndarray,
                                 theta_rad: np.ndarray) -> np.ndarray:
    """
    Effective area wrapper matching the original IDL helper.

    Parameters
    ----------
    phi_rad, theta_rad : ndarray, shape (N,)
        Deflection angles in radians.

    Returns
    -------
    effarea : ndarray, shape (4, N)
        Effective area for the four SPC collectors, following the same
        convention as the original IDL code.
    """
    phi_deg = np.rad2deg(phi_rad)
    theta_deg = np.rad2deg(theta_rad)
    return 100.0 * psp_swp_spc_coldspot(phi_deg, theta_deg)


# =============================================================================
# Frustum geometry and sampling
# =============================================================================

def uniform_sample_frustum(zlower: float,
                           zupper: float,
                           alpha_rad: float,
                           n: int = 1000,
                           rng: Optional[np.random.Generator] = None) -> FrustumSamples:
    """
    Draw uniform random points in the 3D frustum

        z in [zlower, zupper]
        sqrt(x^2 + y^2) < z * tan(alpha_rad)

    using rejection sampling from a bounding rectangular prism.

    This is a direct Python translation of `uniform_sample_frustum` from the
    IDL code, with one robustness improvement: if the first batch of proposals
    does not produce enough accepted points, we keep sampling until N points are
    obtained.
    """
    if rng is None:
        rng = np.random.default_rng()

    z0 = float(zlower)
    dz0 = float(zupper - zlower)

    # Radii of the frustum at the lower and upper z-boundaries.
    r0 = z0 * np.tan(alpha_rad)
    r1 = (z0 + dz0) * np.tan(alpha_rad)

    # Exact volume of a conical frustum.
    vol = (dz0 * np.pi / 3.0) * (r0**2 + r1**2 + r0 * r1)

    # Bounding rectangular prism used for rejection sampling.
    recvol = dz0 * (2.0 * r1)**2

    # Same heuristic oversampling factor as the IDL code.
    n0_guess = max(1, int(np.ceil(1.1 * n * recvol / vol)))

    x_keep_all = []
    y_keep_all = []
    z_keep_all = []
    naccepted = 0

    while naccepted < n:
        xrand = 2.0 * r1 * (rng.random(n0_guess) - 0.5)
        yrand = 2.0 * r1 * (rng.random(n0_guess) - 0.5)
        zrand = z0 + dz0 * rng.random(n0_guess)

        inside = np.sqrt(xrand**2 + yrand**2) / zrand < np.tan(alpha_rad)

        x_keep = xrand[inside]
        y_keep = yrand[inside]
        z_keep = zrand[inside]

        x_keep_all.append(x_keep)
        y_keep_all.append(y_keep)
        z_keep_all.append(z_keep)
        naccepted += x_keep.size

    x = np.concatenate(x_keep_all)[:n]
    y = np.concatenate(y_keep_all)[:n]
    z = np.concatenate(z_keep_all)[:n]

    # Ray-like coordinates used by the original code.
    phi = np.arctan2(x, z)
    theta = np.arctan2(y, z)
    r = np.sqrt(x**2 + y**2 + z**2)

    # Each initial sample has equal volume weight because the accepted points
    # are uniform in the frustum.
    dvol = np.full(n, vol / n, dtype=float)

    return FrustumSamples(
        phi=phi,
        theta=theta,
        r=r,
        dvol=dvol,
        zlower=zlower,
        zupper=zupper,
        alpha_rad=alpha_rad,
    )


def rescale_frustum_samples(zlower: float,
                            zupper: float,
                            samples: FrustumSamples,
                            debug: bool = False) -> FrustumSamples:
    """
    Slide one set of frustum samples into a new z-slab while keeping the ray
    direction fixed.

    This is the key speed-saving idea in the original `.pro` code:
    - keep the same angular directions (phi, theta),
    - map each point to the same fractional location in the new slab,
    - update the volume weight by the exact local Jacobian.

    The weight update is:

        dV_new = dV_old * (dz1 / dz0) * (r_new / r_old)^2

    which is exactly what the IDL code uses.
    """
    tan_phi = np.tan(samples.phi)
    tan_theta = np.tan(samples.theta)

    cosalpha = 1.0 / np.sqrt(tan_phi**2 + tan_theta**2 + 1.0)
    tanalpha_max = np.tan(samples.alpha_rad)

    # Recover the original z positions from (r, phi, theta).
    z0 = samples.r * cosalpha

    # Fractional position of each point within the old slab.
    zlo0 = samples.zlower
    dz0 = samples.zupper - samples.zlower
    lam = (z0 - zlo0) / dz0

    # Map to the new slab at the same fractional position.
    dz1 = zupper - zlower
    newz = zlower + dz1 * lam
    newr = newz / cosalpha

    # Jacobian-rescaled volume weights.
    newdvol = samples.dvol * (dz1 / dz0) * (newr / samples.r)**2

    if debug:
        radius0 = zlower * tanalpha_max
        radius1 = zupper * tanalpha_max
        newvol = (dz1 * np.pi / 3.0) * (radius0**2 + radius1**2 + radius0 * radius1)
        err_percent = 100.0 * (np.sum(newdvol) - newvol) / newvol
        print(f"net volume transform error: {err_percent:.8f}%")

    return FrustumSamples(
        phi=samples.phi.copy(),
        theta=samples.theta.copy(),
        r=newr,
        dvol=newdvol,
        zlower=zlower,
        zupper=zupper,
        alpha_rad=samples.alpha_rad,
    )


def rays_to_xyzfc(samples: FrustumSamples) -> np.ndarray:
    """
    Convert the ray representation (phi, theta, r) back into Cartesian
    cup-frame coordinates (x, y, z).

    Returns
    -------
    ndarray, shape (N, 3)
        Cartesian velocities [km/s].
    """
    tan_phi = np.tan(samples.phi)
    tan_theta = np.tan(samples.theta)

    cosalpha = 1.0 / np.sqrt(tan_phi**2 + tan_theta**2 + 1.0)
    z = samples.r * cosalpha
    x = tan_phi * z
    y = tan_theta * z

    return np.column_stack([x, y, z])


# =============================================================================
# Forward model
# =============================================================================

def cup_response_numflux_1step(vz_lo: float,
                               vz_hi: float,
                               fv: VDF,
                               effarea: np.ndarray,
                               ps_samples: FrustumSamples) -> np.ndarray:
    """
    Compute the number-flux response for one SPC modulator step.

    Mathematically, this approximates

        F_s = const * ∫ v_z * A_s(phi, theta) * f(v) d^3v

    over the frustum corresponding to [vz_lo, vz_hi].
    """
    nsensors = effarea.shape[0]
    numflux = np.zeros(nsensors, dtype=float)

    # Slide the reference cloud into this slab.
    slab_samples = rescale_frustum_samples(vz_lo, vz_hi, ps_samples)

    # Convert to Cartesian velocity coordinates.
    vxyz = rays_to_xyzfc(slab_samples)

    # Evaluate the VDF at the new locations.
    f = fv.evaluate(vxyz)

    # Same conversion factor as in the original IDL code.
    cmperkm = 1.0e5
    vz = vxyz[:, 2]

    for s in range(nsensors):
        integrand = vz * effarea[s, :] * f * slab_samples.dvol
        numflux[s] = cmperkm * np.nansum(integrand)

    return numflux


def cup_response_numflux_spectrum(vz_lo: np.ndarray,
                                  vz_hi: np.ndarray,
                                  fv: VDF,
                                  effarea: np.ndarray,
                                  ps_samples: FrustumSamples) -> np.ndarray:
    """
    Compute the full synthetic SPC spectrum for many modulator bins.

    Parameters
    ----------
    vz_lo, vz_hi : ndarray, shape (nsteps,)
        Lower and upper z-boundaries of the bins.

    fv : VDF
        VDF object created with new_vdf(...).

    effarea : ndarray, shape (4, N)
        Effective area for the four collectors, evaluated once on the
        reference rays.

    ps_samples : FrustumSamples
        Reference sample set.

    Returns
    -------
    ndarray, shape (4, nsteps)
        Synthetic number-flux spectrum.
    """
    vz_lo = np.asarray(vz_lo, dtype=float)
    vz_hi = np.asarray(vz_hi, dtype=float)
    nsteps = vz_lo.size
    nsensors = effarea.shape[0]

    numflux = np.zeros((nsensors, nsteps), dtype=float)

    for i in range(nsteps):
        numflux[:, i] = cup_response_numflux_1step(
            vz_lo=vz_lo[i],
            vz_hi=vz_hi[i],
            fv=fv,
            effarea=effarea,
            ps_samples=ps_samples,
        )

    return numflux


# =============================================================================
# Convenience / test routines
# =============================================================================
@profile
def integration_test(debug: bool = False,
                     n: int = 1000,
                     samples: Optional[FrustumSamples] = None,
                     params: Optional[Dict[str, float]] = None,
                     rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """
    Reproduce the behavior of the IDL `integration_test` routine.

    This is the easiest entry point to check that the translated forward model
    is running.
    """
    if rng is None:
        rng = np.random.default_rng()

    if params is None:
        params = {"vx": 100.0, "vy": 50.0, "vz": 250.0, "w": 50.0, "n": 1000.0}

    spc_fov_radians = np.deg2rad(44.0)

    # Define the SPC modulator bins.
    nsteps = 30
    vlo = 100.0 + 20.0 * np.arange(nsteps)
    vhi = vlo + 20.0

    # Create the VDF object using the translated vdf module.
    myvdf = new_vdf("maxwellian_vdf", params)

    # Generate or reuse the reference sample cloud in the first slab.
    if samples is None:
        ps_samples = uniform_sample_frustum(vlo[0], vhi[0], spc_fov_radians, n=n, rng=rng)
    else:
        ps_samples = samples

    # Evaluate the geometric response once on the fixed rays.
    effarea = calculate_effective_area_cm2(ps_samples.phi, ps_samples.theta)

    # Compute the synthetic SPC spectrum.
    numflux = cup_response_numflux_spectrum(vlo, vhi, myvdf, effarea, ps_samples)

    if debug:
        print("integration_test completed")
        print(f"  samples used: {ps_samples.r.size}")
        print(f"  output shape: {numflux.shape}")
        print(f"  min/max flux: {np.nanmin(numflux):.6e}, {np.nanmax(numflux):.6e}")

    return numflux


def timetest() -> tuple[np.ndarray, np.ndarray]:
    """
    Rough timing test matching the purpose of the IDL `timetest`.

    Returns
    -------
    ns : ndarray
        Sample counts used.

    times : ndarray
        Average time per integration_test call [seconds].
    """
    import time

    ns = np.array([10, 33, 100, 333, 1000, 3333, 10000], dtype=int)
    times = np.zeros_like(ns, dtype=float)

    rng = np.random.default_rng(12345)
    spc_fov_radians = np.deg2rad(44.0)

    for j, n in enumerate(ns):
        print(f"Timing n = {n}")
        t0 = time.perf_counter()

        samples = uniform_sample_frustum(100.0, 200.0, spc_fov_radians, n=n, rng=rng)

        for _ in range(10):
            _ = integration_test(n=n, samples=samples, rng=rng)

        times[j] = (time.perf_counter() - t0) / 10.0

    return ns, times


def restest() -> dict:
    """
    Monte-Carlo convergence test analogous to the IDL `restest`.

    Returns
    -------
    dict
        Contains sample sizes, flux realizations, mean, and scatter estimate.
    """
    ns = np.array([10, 33, 100, 333, 1000, 3333, 10000], dtype=int)
    fluxes = np.zeros((4, 30, len(ns), 10), dtype=float)
    params = {"vx": 100.0, "vy": 50.0, "vz": 150.0, "w": 100.0, "n": 1000.0}

    rng = np.random.default_rng(12345)

    for j, n in enumerate(ns):
        print(f"restest n = {n}")
        for i in range(10):
            fluxes[:, :, j, i] = integration_test(n=n, params=params, rng=rng)

    means = np.mean(fluxes, axis=3)
    meansq = np.mean(fluxes**2, axis=3)
    var = np.sqrt(np.maximum(meansq - means**2, 0.0))

    return {
        "ns": ns,
        "fluxes": fluxes,
        "means": means,
        "var": var,
    }


# =============================================================================
# Example command-line run
# =============================================================================

if __name__ == "__main__":
    print("Running one integration_test() example...")
    out = integration_test(debug=True, n=1000)
    print("First few values from sensor 0:", out[0, :5])

    print("\nRunning a quick timing test...")
    ns, times = timetest()
    for n, t in zip(ns, times):
        print(f"n = {n:6d}   avg time = {t:.6f} s")
