"""The detector's sensors as LUCiD's simulator sees them.

Every table here files a detected photon against the sensor the simulator reports (its
``sensor_id``), and fiTQun evaluates the tables at the sensors it is given. Both must be the
simulator's own: a separate geometry file can list the sensors in another order (no row of
sk_geometry.npz is the simulator's sensor of that index) and at positions the simulator moved
(it snaps PMTs onto the wall: half of SK's barrel PMTs by 2.87 cm), and either files light
against the wrong place without any error.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Sensors:
    """Sensor ``i`` is the simulator's ``sensor_id == i``. Lengths in metres."""

    positions_m: np.ndarray   # (n, 3)
    normals: np.ndarray       # (n, 3) unit, pointing into the detector: the propagation's normals
    radius_m: float           # cylinder radius
    halfheight_m: float       # cylinder half-height
    pmt_radius_m: float       # sensor radius


def from_detector_config(detector_config) -> Sensors:
    """The sensors of the detector LUCiD builds from ``detector_config`` (a cylinder)."""
    from lucid.geometry import generate_detector

    det = generate_detector(str(detector_config))
    if not all(hasattr(det, a) for a in ("barr_points", "tcap_points", "bcap_points")):
        raise ValueError(f"{detector_config}: the fiTQun tables need a cylinder detector")
    pos = np.asarray(det.all_points, dtype=np.float64)    # barrel, then top cap, then bottom cap
    on_cap = np.arange(len(pos)) >= len(det.barr_points)
    return Sensors(positions_m=pos, normals=radial_normals(pos, on_cap=on_cap), radius_m=float(det.r),
                   halfheight_m=float(det.H) / 2.0, pmt_radius_m=float(det.S_radius))


def radial_normals(positions: np.ndarray, *, on_cap: np.ndarray) -> np.ndarray:
    """The propagation's inward normals for sensors at ``positions``: the axis for barrel
    sensors, -z / +z for sensors on the top / bottom cap (``on_cap``)."""
    pos = np.asarray(positions, dtype=np.float64)
    rho = np.hypot(pos[:, 0], pos[:, 1])
    safe = np.where(rho > 0, rho, 1.0)
    barrel = np.column_stack([-pos[:, 0] / safe, -pos[:, 1] / safe, np.zeros(len(pos))])
    cap = np.column_stack([np.zeros(len(pos)), np.zeros(len(pos)), -np.sign(pos[:, 2])])
    return np.where(np.asarray(on_cap)[:, None], cap, barrel)
