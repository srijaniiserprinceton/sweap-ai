from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Sequence, Union
import numpy as np

ArrayLike = Union[np.ndarray, Sequence[float]]


def _as_n_by_3(vxyz: ArrayLike) -> np.ndarray:
    """
    Normalize velocity input to shape (N, 3).

    Accepted input styles:
    - single vector: [vx, vy, vz]
    - array of vectors: [[vx1, vy1, vz1], [vx2, vy2, vz2], ...]
    - IDL-like 3xN array
    - IDL-like Nx3 array
    """
    arr = np.asarray(vxyz, dtype=float)

    if arr.ndim == 1:
        if arr.size != 3:
            raise ValueError("1D vxyz input must have exactly 3 elements.")
        return arr.reshape(1, 3)

    if arr.ndim != 2:
        raise ValueError("vxyz must be a 1D 3-vector or a 2D array of 3-vectors.")

    if arr.shape[1] == 3:
        return arr
    if arr.shape[0] == 3:
        return arr.T

    raise ValueError("2D vxyz input must have shape (N, 3) or (3, N).")



def empty_vdf(vxyz: ArrayLike, _extra: Any = None) -> np.ndarray:
    """
    Return zero everywhere.

    Direct translation of the IDL helper `empty_vdf`.
    """
    arr = _as_n_by_3(vxyz)
    return np.zeros(arr.shape[0], dtype=float)



def maxwellian_vdf(vxyz: ArrayLike, params: Dict[str, float]) -> np.ndarray:
    """
    Evaluate a drifting isotropic 3D Maxwellian.

    Parameters
    ----------
    vxyz : array-like
        Velocity coordinates. Accepts shape (N, 3), (3, N), or a single 3-vector.

    params : dict
        Must contain keys:
            vx, vy, vz : drift velocity components
            w          : thermal width
            n          : density / amplitude

    Returns
    -------
    np.ndarray
        Maxwellian values with shape (N,).

    Notes
    -----
    This follows the IDL expression exactly:

        g = (n / (sqrt(pi) * w)^3) * exp(-((vx-ux)^2 + (vy-uy)^2 + (vz-uz)^2) / w^2)
    """
    arr = _as_n_by_3(vxyz)

    ux = float(params["vx"])
    uy = float(params["vy"])
    uz = float(params["vz"])
    w = float(params["w"])
    n = float(params["n"])

    dvx = arr[:, 0] - ux
    dvy = arr[:, 1] - uy
    dvz = arr[:, 2] - uz

    g = (n / ((np.sqrt(np.pi) * w) ** 3)) * np.exp(-(dvx**2 + dvy**2 + dvz**2) / (w**2))
    return g


@dataclass
class VDF:
    """
    Lightweight Python version of the IDL `vdf` object.

    It stores:
    - the evaluation function name or callable
    - any user data / parameters needed by that function
    """
    func_name: str = ""
    udata: Optional[Any] = None

    _FUNCTIONS: Dict[str, Callable[[ArrayLike, Any], np.ndarray]] = None

    def __post_init__(self) -> None:
        # Registry of built-in VDF functions.
        self._FUNCTIONS = {
            "empty_vdf": empty_vdf,
            "maxwellian_vdf": maxwellian_vdf,
        }

    def evaluate(self, vxyz: ArrayLike) -> np.ndarray:
        """Evaluate the currently selected VDF."""
        if callable(self.func_name):
            return self.func_name(vxyz, self.udata)

        if self.func_name not in self._FUNCTIONS:
            raise ValueError(f"Unknown VDF function: {self.func_name!r}")

        return self._FUNCTIONS[self.func_name](vxyz, self.udata)

    def set_function(self, func_name: Union[str, Callable[[ArrayLike, Any], np.ndarray]]) -> None:
        """Set the underlying VDF evaluation function."""
        self.func_name = func_name

    def set_udata(self, udata: Any) -> None:
        """Set the support data / parameters used by the evaluation function."""
        self.udata = udata



def new_vdf(func_name: Union[str, Callable[[ArrayLike, Any], np.ndarray]],
            udata: Optional[Any] = None) -> VDF:
    """
    Convenience constructor matching the IDL helper `new_vdf`.
    """
    obj = VDF()
    obj.set_udata(udata)
    obj.set_function(func_name)
    return obj


if __name__ == "__main__":
    # Basic self-test mirroring the IDL example.
    f = new_vdf("empty_vdf")
    print("empty_vdf([0,0,0]) =", f.evaluate([0, 0, 0]))

    g = new_vdf("maxwellian_vdf", {"vx": 100.0, "vy": 0.0, "vz": 0.0, "w": 50.0, "n": 100.0})
    print(
        "maxwellian test =",
        g.evaluate([[80, 0, 0], [90, 0, 0], [100, 0, 0], [110, 0, 0]]),
    )
