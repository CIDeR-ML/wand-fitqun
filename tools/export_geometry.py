#!/usr/bin/env python3
"""Write LUCiD's detector geometry in the plain form SadQun and the WCSim
converter read.

SadQun deliberately does not link numpy or HDF5: the only thing WCSim ever did
for fiTQun was fill PMTpos/PMTdir/PMTQE, so a text file is enough and keeps the
container to ROOT alone.

The geometry is the simulator's, read from the LUCiD sensor file the hits come with
(``--sensor``): its ``config/sensor_positions`` are the positions the simulator detected
light at, in the order its ``sensor_idx`` counts. The npz's own order is a different
permutation (for SK_WAND not one of the 11096 rows agrees; rows differ by up to 48 m), and
half of its barrel PMTs sit 2.87 cm inside the wall the simulator snaps them onto. The
normals are the simulator's too: barrel PMTs face the axis and caps face along -z / +z,
where the npz directions follow flat wall panels, 1-4 deg off. The npz supplies only
which PMTs are on a cap, the detector dimensions and the QE.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from lucid_fitqun.fitqun.sensors import radial_normals  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("npz", type=Path, help="LUCiD sk_geometry.npz")
    p.add_argument("-o", "--output", type=Path, required=True)
    p.add_argument("--sensor", type=Path, required=True,
                   help="a LUCiD wc_sensor_*.h5: the simulator's sensor positions, in the "
                        "order its sensor_idx counts")
    p.add_argument("--match-tol-cm", type=float, default=5.0)
    a = p.parse_args()

    g = np.load(a.npz)
    pos_cm = g["positions_mm"] / 10.0          # mm -> cm, fiTQun works in cm
    dirs = np.asarray(g["directions"], dtype=np.float64)

    import h5py
    with h5py.File(a.sensor, "r") as f:
        sensor_cm = np.asarray(f["config/sensor_positions"], dtype=np.float64) * 100.0
    if len(sensor_cm) != len(pos_cm):
        raise SystemExit(f"sensor file has {len(sensor_cm)} sensors, npz has {len(pos_cm)}")
    # Which npz row each sensor is, by nearest position: the simulator snapped some
    # PMTs up to ~3 cm onto the wall, so the match is not exact.
    from scipy.spatial import cKDTree
    dist, idx = cKDTree(pos_cm).query(sensor_cm)
    if dist.max() > a.match_tol_cm:
        raise SystemExit(f"cannot match sensor to npz geometry: max {dist.max():.1f} cm")
    if len(set(idx.tolist())) != len(idx):
        raise SystemExit("sensor-to-npz match is not one-to-one")
    print(f"{a.sensor.name}: the sensor file's positions and sensor_idx order "
          f"(max offset from the npz {dist.max():.2f} cm)")
    on_cap = np.abs(dirs[idx, 2]) >= 0.5
    pos_cm, dirs = sensor_cm, radial_normals(sensor_cm, on_cap=on_cap)
    qe = g["PMTQE"][idx] if "PMTQE" in g.files else np.ones(len(pos_cm))

    lines = [
        f"n_pmt {len(pos_cm)}",
        f"pmt_radius_cm {float(g['sensor_radius']) * 100.0:.6f}",
        f"det_radius_cm {float(g['radius']) * 100.0:.6f}",
        f"det_halfz_cm {float(g['height']) * 100.0 / 2.0:.6f}",
    ]
    for i, (p_, d_, q_) in enumerate(zip(pos_cm, dirs, qe)):
        lines.append(f"pmt {i} {p_[0]:.6f} {p_[1]:.6f} {p_[2]:.6f} "
                     f"{d_[0]:.6f} {d_[1]:.6f} {d_[2]:.6f} {float(q_):.6f}")

    a.output.write_text("\n".join(lines) + "\n")
    print(f"{a.output}: {len(pos_cm)} PMTs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
