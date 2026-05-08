from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Optional, Tuple, Union
from line_profiler import profile

import numpy as np

try:
    from scipy.integrate import dblquad
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False

ArrayLike = Union[float, np.ndarray]
DTOR = np.pi / 180.0


@dataclass(frozen=True)
class SPCGeometryParams:
    rl: float = 10.86
    re: float = 39.9
    rc: float = 23.94
    ze: float = 70.5
    zl: float = 16.9
    epsilon: float = 0.36
    dz_coll: float = 1.981
    al: float = np.degrees(np.arctan2(39.9 - 10.86, 70.5 - 16.9))
    ac: float = np.degrees(np.arctan2(23.94 - 10.86, 16.9))
    amax: float = np.degrees(np.arctan2(23.94 + 39.9, 70.5))



def setparams() -> SPCGeometryParams:
    """Python version of the IDL `setparams` procedure."""
    return SPCGeometryParams()



def _clip_circle_domain(value: float, tol: float = 1e-9) -> float:
    if value > -tol:
        return max(value, 0.0)
    return value



def int_uppersemicircle_dx(*, r: float = 1.0, xc: float = 0.0, yc: float = 0.0,
                           x0: Optional[float] = None, x1: Optional[float] = None) -> float:
    """
    Integrate the upper semicircle y(x) over x in [x0, x1].

    This is a direct translation of the IDL helper.
    """
    r = abs(float(r))
    if x0 is None:
        x0 = xc - r
    if x1 is None:
        x1 = xc + r
    if x1 == x0:
        return 0.0

    lower, upper = (x1, x0) if x1 < x0 else (x0, x1)

    if lower >= (xc + r) or upper <= (xc - r):
        return 0.0
    lower = max(lower, xc - r)
    upper = min(upper, xc + r)

    num_upper = upper - xc
    num_lower = lower - xc

    denom_uparg = -xc**2 + r**2 + 2.0 * xc * upper - upper**2
    denom_loarg = -xc**2 + r**2 + 2.0 * xc * lower - lower**2
    denom_uparg = _clip_circle_domain(denom_uparg)
    denom_loarg = _clip_circle_domain(denom_loarg)

    denom_upper = 0.0 if (r**2 + 2.0 * xc * upper) == (upper**2 + xc**2) else math.sqrt(denom_uparg)
    denom_lower = 0.0 if (r**2 + 2.0 * xc * lower) == (lower**2 + xc**2) else math.sqrt(denom_loarg)

    arg_upper = yc * upper + 0.5 * num_upper * denom_upper + 0.5 * (r**2) * math.atan2(num_upper, denom_upper)
    arg_lower = yc * lower + 0.5 * num_lower * denom_lower + 0.5 * (r**2) * math.atan2(num_lower, denom_lower)

    diff = arg_upper - arg_lower
    if not np.isfinite(diff) or arg_upper == arg_lower:
        return 0.0
    return diff



def int_lowersemicircle_dx(*, r: float = 1.0, xc: float = 0.0, yc: float = 0.0,
                           x0: Optional[float] = None, x1: Optional[float] = None) -> float:
    """
    Integrate the lower semicircle y(x) over x in [x0, x1].

    This mirrors the IDL implementation exactly.
    """
    r = abs(float(r))
    if x0 is None:
        x0 = xc - r
    if x1 is None:
        x1 = xc + r
    if x1 == x0:
        return 0.0

    lower, upper = (x1, x0) if x1 < x0 else (x0, x1)

    if lower >= (xc + r) or upper <= (xc - r):
        return 0.0
    lower = max(lower, xc - r)
    upper = min(upper, xc + r)

    num_upper = xc - upper
    num_lower = xc - lower
    denom_uparg = -xc**2 + r**2 + 2.0 * xc * upper - upper**2
    denom_loarg = -xc**2 + r**2 + 2.0 * xc * lower - lower**2
    denom_uparg = _clip_circle_domain(denom_uparg)
    denom_loarg = _clip_circle_domain(denom_loarg)

    denom_upper = 0.0 if (r**2 + 2.0 * xc * upper) == (upper**2 + xc**2) else math.sqrt(denom_uparg)
    denom_lower = 0.0 if (r**2 + 2.0 * xc * lower) == (lower**2 + xc**2) else math.sqrt(denom_loarg)

    arg_upper = yc * upper + 0.5 * num_upper * denom_upper + 0.5 * (r**2) * math.atan2(num_upper, denom_upper)
    arg_lower = yc * lower + 0.5 * num_lower * denom_lower + 0.5 * (r**2) * math.atan2(num_lower, denom_lower)

    diff = arg_upper - arg_lower
    if not np.isfinite(diff) or arg_upper == arg_lower:
        return 0.0
    return diff



def intersection_of_circles(*, x1: float, y1: float, r1: float,
                            x2: float, y2: float, r2: float) -> np.ndarray:
    """Return the two x-coordinates of circle-circle intersections."""
    d = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    if (d == 0.0) and (abs(r1) == abs(r2)):
        return np.array([x1 - r1, x1 + r1], dtype=float)

    if d > (abs(r2) + abs(r1)):
        return np.array([np.nan, np.nan], dtype=float)
    if d < abs(abs(r2) - abs(r1)):
        return np.array([np.nan, np.nan], dtype=float)

    a = (r1**2 - r2**2 + d**2) / (2.0 * d)
    h = math.sqrt(max(r1**2 - a**2, 0.0))

    x3 = x1 + a * (x2 - x1) / d
    y3 = y1 + a * (y2 - y1) / d

    x4 = x3 + h * (y2 - y1) / d
    x5 = x3 - h * (y2 - y1) / d

    return np.array([min(x4, x5), max(x4, x5)], dtype=float)



def area_of_intersection(*, x1: float, x2: float, y1: float, y2: float,
                         r1: float, r2: float) -> float:
    """Area of overlap of two circles, integrated piecewise in x."""
    intercepts = intersection_of_circles(x1=x1, x2=x2, y1=y1, y2=y2, r1=r1, r2=r2)
    xmax = min(x1 + abs(r1), x2 + abs(r2))
    xmin = max(x1 - abs(r1), x2 - abs(r2))
    xpoints = np.array([xmin, intercepts[0], intercepts[1], xmax], dtype=float)
    xi = xpoints[np.isfinite(xpoints)]

    area = 0.0
    for i in range(len(xi) - 1):
        low1 = int_lowersemicircle_dx(r=r1, xc=x1, yc=y1, x0=xi[i], x1=xi[i + 1])
        low2 = int_lowersemicircle_dx(r=r2, xc=x2, yc=y2, x0=xi[i], x1=xi[i + 1])
        high1 = int_uppersemicircle_dx(r=r1, xc=x1, yc=y1, x0=xi[i], x1=xi[i + 1])
        high2 = int_uppersemicircle_dx(r=r2, xc=x2, yc=y2, x0=xi[i], x1=xi[i + 1])
        thisarea = min(high1, high2) - max(low1, low2)
        area += max(thisarea, 0.0)
    return area



def circle_x_at_yeq(y: float, *, xc: float, yc: float, r: float) -> np.ndarray:
    arg = r**2 - (y - yc) ** 2
    if arg < 0.0:
        return np.array([np.nan, np.nan], dtype=float)
    root = math.sqrt(arg)
    return np.array([xc - root, xc + root], dtype=float)



def circle_xintercepts(*, xc: float, yc: float, r: float) -> np.ndarray:
    arg = r**2 - yc**2
    if arg < 0.0:
        return np.array([np.nan, np.nan], dtype=float)
    root = math.sqrt(arg)
    return np.array([xc - root, xc + root], dtype=float)



def quadrant_area_of_intersection(*, x1: float, x2: float, y1: float, y2: float,
                                  r1: float, r2: float, x3: Optional[float] = None,
                                  y3: Optional[float] = None, r3: Optional[float] = None,
                                  epsilon: Union[float, Tuple[float, float], np.ndarray] = 0.0) -> np.ndarray:
    """
    Compute overlap areas in the four quadrants after excluding a central slit.

    Returns
    -------
    np.ndarray, shape (4,)
        [area_pp, area_mp, area_pm, area_mm]
        where p/m indicates positive/negative x and positive/negative y.
    """
    if np.ndim(epsilon) == 0:
        eps_x = float(epsilon)
        eps_y = float(epsilon)
    else:
        eps = np.asarray(epsilon, dtype=float).ravel()
        if eps.size == 1:
            eps_x = float(eps[0])
            eps_y = float(eps[0])
        elif eps.size >= 2:
            eps_x = float(eps[0])
            eps_y = float(eps[1])
        else:
            eps_x = 0.0
            eps_y = 0.0

    if r3 is None:
        x3, y3, r3 = x1, y1, r1

    intercepts1 = intersection_of_circles(x1=x1, x2=x2, y1=y1, y2=y2, r1=r1, r2=r2)
    intercepts2 = intersection_of_circles(x1=x1, x2=x3, y1=y1, y2=y3, r1=r1, r2=r3)
    intercepts3 = intersection_of_circles(x1=x2, x2=x3, y1=y2, y2=y3, r1=r2, r2=r3)
    intercepts = np.concatenate([intercepts1, intercepts2, intercepts3])

    x_intercepts = np.concatenate([
        circle_x_at_yeq(eps_y / 2.0, xc=x1, yc=y1, r=r1),
        circle_x_at_yeq(eps_y / 2.0, xc=x2, yc=y2, r=r2),
        circle_x_at_yeq(eps_y / 2.0, xc=x3, yc=y3, r=r3),
        circle_x_at_yeq(-eps_y / 2.0, xc=x1, yc=y1, r=r1),
        circle_x_at_yeq(-eps_y / 2.0, xc=x2, yc=y2, r=r2),
        circle_x_at_yeq(-eps_y / 2.0, xc=x3, yc=y3, r=r3),
    ])

    xmax = min(x1 + abs(r1), x2 + abs(r2), x3 + abs(r3))
    xmin = max(x1 - abs(r1), x2 - abs(r2), x3 - abs(r3))
    if xmin >= xmax:
        return np.zeros(4, dtype=float)

    xpoints = np.concatenate([
        np.array([xmin]),
        intercepts,
        np.array([xmax, 0.0]),
        x_intercepts,
        np.array([-eps_x / 2.0, eps_x / 2.0]),
    ])
    xi = np.sort(xpoints[np.isfinite(xpoints) & (xpoints >= xmin) & (xpoints <= xmax)])
    if xi.size < 2:
        return np.zeros(4, dtype=float)

    area_pp = area_mp = area_pm = area_mm = 0.0

    rhs = np.where(xi >= (eps_x / 2.0))[0]
    lhs = np.where(xi <= (-eps_x / 2.0))[0]

    def _accumulate(xseg: np.ndarray) -> Tuple[float, float]:
        upper_area = 0.0
        lower_area = 0.0
        for i in range(len(xseg) - 1):
            xa, xb = xseg[i], xseg[i + 1]
            low1 = int_lowersemicircle_dx(r=r1, xc=x1, yc=y1, x0=xa, x1=xb)
            low2 = int_lowersemicircle_dx(r=r2, xc=x2, yc=y2, x0=xa, x1=xb)
            low3 = int_lowersemicircle_dx(r=r3, xc=x3, yc=y3, x0=xa, x1=xb)
            high1 = int_uppersemicircle_dx(r=r1, xc=x1, yc=y1, x0=xa, x1=xb)
            high2 = int_uppersemicircle_dx(r=r2, xc=x2, yc=y2, x0=xa, x1=xb)
            high3 = int_uppersemicircle_dx(r=r3, xc=x3, yc=y3, x0=xa, x1=xb)
            gap = (eps_y / 2.0) * (xb - xa)

            low = max(low1, low2, low3)
            high = min(high1, high2, high3)

            dAp = max((high - gap), 0.0) - max((low - gap), 0.0)
            dAm = min((high + gap), 0.0) - min((low + gap), 0.0)
            upper_area += max(dAp, 0.0)
            lower_area += max(-dAm, 0.0)
        return upper_area, lower_area

    if rhs.size >= 2:
        upp, low = _accumulate(xi[rhs])
        area_pp += upp
        area_pm += low

    if lhs.size >= 2:
        upp, low = _accumulate(xi[lhs])
        area_mp += upp
        area_mm += low

    return np.array([area_pp, area_mp, area_pm, area_mm], dtype=float)



def transparency_factor(phi: ArrayLike, theta: ArrayLike) -> np.ndarray:
    """
    Grid transparency correction factor.

    Inputs are in degrees, matching the IDL routine.
    """
    phi = np.asarray(phi, dtype=float)
    theta = np.asarray(theta, dtype=float)

    tanalpha = np.sqrt(np.tan(phi * DTOR) ** 2 + np.tan(theta * DTOR) ** 2)
    gridthick = 0.1
    gridcell = 1.0
    tau_0 = 0.888
    trans_area = tau_0 * gridcell
    l = np.sqrt(trans_area)
    ngrids = 8
    d = (gridthick / l) * tanalpha

    clockangles = np.arange(ngrids, dtype=float) / 90.0

    out = np.ones_like(d, dtype=float)
    for ang in clockangles:
        out *= tau_0 * (1.0 - d * np.sin(ang * DTOR)) * (1.0 - d * np.cos(ang * DTOR))
    return out


@profile
def psp_swp_spc_coldspot(phi: ArrayLike = 0.0, theta: ArrayLike = 0.0) -> np.ndarray:
    """
    Cold-plasma SPC effective area model.

    Parameters
    ----------
    phi, theta : float or array-like
        Horizontal and vertical incidence angles in degrees.

    Returns
    -------
    np.ndarray
        Shape (4, N), one effective area per quadrant / collector region.
        If scalar inputs are given, returns shape (4, 1).
    """
    params = setparams()

    phi = np.atleast_1d(np.asarray(phi, dtype=float))
    theta = np.atleast_1d(np.asarray(theta, dtype=float))
    if phi.shape != theta.shape:
        phi, theta = np.broadcast_arrays(phi, theta)

    mask = (np.cos(phi * DTOR) > 0.0) & (np.cos(theta * DTOR) > 0.0)

    r1 = params.re
    x1 = params.ze * np.tan(phi * DTOR)
    y1 = params.ze * np.tan(theta * DTOR)

    r2 = params.rl
    x2 = params.zl * np.tan(phi * DTOR)
    y2 = params.zl * np.tan(theta * DTOR)

    r3 = params.rc
    x3 = 0.0
    y3 = 0.0

    eps_x = np.maximum(params.epsilon - params.dz_coll * np.tan(np.abs(phi) * DTOR), 0.0)
    eps_y = np.maximum(params.epsilon - params.dz_coll * np.tan(np.abs(theta) * DTOR), 0.0)

    areas = np.zeros((4, phi.size), dtype=float)
    transp = transparency_factor(phi, theta)

    for i in range(phi.size):
        quad = quadrant_area_of_intersection(
            x1=float(x1[i]), x2=float(x2[i]), y1=float(y1[i]), y2=float(y2[i]),
            r1=r1, r2=r2, x3=x3, y3=y3, r3=r3,
            epsilon=(float(eps_x[i]), float(eps_y[i])),
        )
        areas[:, i] = transp[i] * quad

    areas *= mask.astype(float)
    return areas


# Alias used elsewhere in the IDL file.
coldspot_areas = psp_swp_spc_coldspot


def psp_swp_spc_coldspot_xyz(x: ArrayLike, y: ArrayLike, z: ArrayLike = 1.0) -> np.ndarray:
    """Evaluate the coldspot model from x, y, z coordinates."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)
    phi = np.degrees(np.arctan2(x, z))
    theta = np.degrees(np.arctan2(y, z))
    return psp_swp_spc_coldspot(phi, theta)



def distrospot_area(f: np.ndarray, phi: ArrayLike, theta: ArrayLike,
                    dphi: float, dtheta: float) -> np.ndarray:
    """
    Effective area for a discrete angular distribution.
    """
    areas = coldspot_areas(phi, theta)
    norma = np.sum(f * dphi * dtheta)
    if norma == 0.0:
        return np.zeros(4, dtype=float)
    weights = f / norma
    return np.array([
        np.sum(areas[0, :] * weights * dphi * dtheta),
        np.sum(areas[1, :] * weights * dphi * dtheta),
        np.sum(areas[2, :] * weights * dphi * dtheta),
        np.sum(areas[3, :] * weights * dphi * dtheta),
    ], dtype=float)



def gaussian_spread_area(phi: float, theta: float, spread: float, num: int) -> np.ndarray:
    """
    Approximate warm-plasma effective area using a discrete Gaussian in tan(angle).

    This is the direct numerical-grid method from the IDL file.
    """
    n = int(2 * np.fix(np.sqrt(float(num)) / 2.0))

    pmin = np.degrees(np.arctan(np.tan(phi * DTOR) - 4.0 * spread))
    pmax = np.degrees(np.arctan(np.tan(phi * DTOR) + 4.0 * spread))
    tmin = np.degrees(np.arctan(np.tan(theta * DTOR) - 4.0 * spread))
    tmax = np.degrees(np.arctan(np.tan(theta * DTOR) + 4.0 * spread))

    p = pmin + (pmax - pmin) * np.arange(n + 1, dtype=float) / n
    t = tmin + (tmax - tmin) * np.arange(n + 1, dtype=float) / n

    pgrid, tgrid = np.meshgrid(p, t, indexing="xy")
    pgrid = pgrid.ravel()
    tgrid = tgrid.ravel()

    dp = 8.0 * spread / n
    dt = 8.0 * spread / n
    alphamax = 42.162

    exfov = (np.tan(pgrid * DTOR) ** 2 + np.tan(tgrid * DTOR) ** 2) < (np.tan(alphamax * DTOR) ** 2)
    exfov2 = (np.abs(pgrid) < alphamax) & (np.abs(tgrid) < alphamax)
    keep = exfov & exfov2
    if np.count_nonzero(keep) <= 1:
        return np.zeros(4, dtype=float)

    pgrid = pgrid[keep]
    tgrid = tgrid[keep]

    f = np.exp(-((np.tan(phi * DTOR) - np.tan(pgrid * DTOR)) ** 2 +
                 (np.tan(theta * DTOR) - np.tan(tgrid * DTOR)) ** 2) / (spread ** 2))
    return distrospot_area(f, pgrid, tgrid, dp, dt)


@profile
def psp_swp_spc_warmspot(phi: float, theta: float, spread: float, num: int = 20) -> np.ndarray:
    """
    Warm-plasma effective area.

    For small spread, this falls back to the cold model.
    Here the robust, directly runnable implementation is the discrete-grid method
    (`gaussian_spread_area`) from the IDL file.

    The original IDL also includes a second 'intelligent' 2D integration pathway
    using INT_2D. Reproducing that exactly would require more of the original IDL
    numerical integration environment. For practical Python use, this wrapper calls
    the discrete Gaussian method.
    """
    if spread < 0.001:
        return psp_swp_spc_coldspot(phi, theta)[:, 0]
    return gaussian_spread_area(phi, theta, spread, num)


if __name__ == "__main__":
    phi = np.array([0.0, 5.0, 10.0])
    theta = np.array([0.0, 0.0, 5.0])
    areas = psp_swp_spc_coldspot(phi, theta)
    print("coldspot shape:", areas.shape)
    print("coldspot:\n", areas)
    print("warmspot example:", psp_swp_spc_warmspot(6.0, 10.0, 0.25, num=400))
