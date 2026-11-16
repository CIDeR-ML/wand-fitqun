"""The sensors every table and every fiTQun geometry use are the simulator's own.

The pencil test fires photons straight at chosen PMTs through LUCiD's propagation and reduces
them with the production ShardBuilder: a hit must land on the aimed PMT and be filed in the
100 cm shell at normal incidence. Built from sk_geometry.npz in its own order instead, no row
is the simulator's sensor of that index and the shell stays empty.
"""
import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from lucid_fitqun.fitqun import scattable, sensors

REPO = Path(__file__).resolve().parents[1]
LUCID_CONFIG = Path(importlib.util.find_spec("lucid").origin).parents[1] / "config"
GEOM = LUCID_CONFIG / "SK_WAND_geom_config.json"
PHYSICS = LUCID_CONFIG / "SK_WAND_physics_config.json"
NPZ = LUCID_CONFIG / "sk_geometry.npz"


@pytest.fixture(scope="module")
def sk():
    return sensors.from_detector_config(GEOM)


def test_sensors_are_the_simulators(sk):
    from lucid.geometry import generate_detector

    det = generate_detector(str(GEOM))
    assert np.array_equal(sk.positions_m, np.asarray(det.all_points, dtype=np.float64))
    assert np.allclose(np.linalg.norm(sk.normals, axis=1), 1.0)
    surface = scattable.surface_for(sk.normals[:, 2])
    nb, nt = len(det.barr_points), len(det.tcap_points)
    assert (surface[:nb] == "sidescattable").all()
    assert (surface[nb:nb + nt] == "topscattable").all()
    assert (surface[nb + nt:] == "botscattable").all()
    rho = np.hypot(sk.positions_m[:nb, 0], sk.positions_m[:nb, 1])
    assert np.allclose(rho, sk.radius_m)                               # snapped onto the wall
    inward = np.einsum("ij,ij->i", sk.normals[:nb, :2], sk.positions_m[:nb, :2]) / rho
    assert np.allclose(inward, -1.0)                                   # barrel faces the axis
    assert np.allclose(np.abs(sk.positions_m[nb:, 2]), sk.halfheight_m)


def test_the_npz_is_not_the_simulators_geometry(sk):
    """Why nothing may read the npz in its place: another order, and half the barrel inset."""
    from lucid.geometry import generate_detector

    g = np.load(NPZ)
    file_order = np.linalg.norm(g["positions_mm"] / 1000.0 - sk.positions_m, axis=1)
    assert np.median(file_order) > 10.0                                # metres, not centimetres
    raw = np.asarray(generate_detector(str(GEOM)).raw_positions, dtype=np.float64)
    snap = np.linalg.norm(sk.positions_m - raw, axis=1)
    assert snap.max() > 0.02 and np.median(snap) < 1e-6


def _targets(sk):
    """A barrel PMT the simulator snapped, an unsnapped one, a top-cap and a bottom-cap PMT,
    each far enough from the other surfaces for a full 100 cm shell."""
    from lucid.geometry import generate_detector

    raw = np.asarray(generate_detector(str(GEOM)).raw_positions, dtype=np.float64)
    moved = np.linalg.norm(sk.positions_m - raw, axis=1) > 0.01
    z, rho = sk.positions_m[:, 2], np.hypot(sk.positions_m[:, 0], sk.positions_m[:, 1])
    barrel = np.abs(sk.normals[:, 2]) < 0.5
    central = barrel & (np.abs(z) < 2.0)
    out = {"barrel, snapped": int(np.flatnonzero(central & moved)[0]),
           "barrel, not snapped": int(np.flatnonzero(central & ~moved)[0])}
    for name, sign in (("top cap", 1), ("bottom cap", -1)):
        out[name] = int(np.flatnonzero(~barrel & (np.sign(z) == sign) & (rho < 5.0))[0])
    return out


def test_pencil_beams_land_on_the_aimed_pmt(sk):
    import jax
    from lucid.simulation.shotgun import setup_shotgun_simulator
    from lucid.sources.shotgun_source import shotgun_source, stack_shotgun_sources
    from lucid_fitqun.fitqun.sample_reduce import ShardBuilder

    n = 256
    targets = _targets(sk)
    origins = [np.tile(sk.positions_m[i] + 1.0 * sk.normals[i], (n, 1)) for i in targets.values()]
    dirs = [np.tile(-sk.normals[i], (n, 1)) for i in targets.values()]
    sim = setup_shotgun_simulator(str(GEOM), physics_config=str(PHYSICS), n_photons=n,
                                  hits="photon_hits", K=12, tts_sigma_ns=1.0)
    srcs = [shotgun_source(o.astype(np.float32), d.astype(np.float32), n_photons=n)
            for o, d in zip(origins, dirs)]
    det, sid, _, ind = sim.batch(stack_shotgun_sources(srcs), jax.random.split(jax.random.PRNGKey(5), len(srcs)))
    det, sid, ind = (np.asarray(x) for x in (det, sid, ind))

    for k, (name, i) in enumerate(targets.items()):
        builder = ShardBuilder.for_sensors(sk, shell_radii_cm=[100.0])
        builder.add(detected=det[k], sensor_id=sid[k], indirect=ind[k],
                    emission_pos_m=origins[k], emission_dir=dirs[k])
        direct = det[k].astype(bool) & ~ind[k].astype(bool)
        assert direct.sum() > 20, name
        assert (sid[k][direct] == i).all(), f"{name}: the simulator put a direct hit on another PMT"
        counts = builder.shard.angular_counts[100.0]
        assert counts.sum() == direct.sum(), f"{name}: direct hits filed outside the 100 cm shell"
        assert counts[-1] == direct.sum(), f"{name}: direct hits not at normal incidence"
        surface = scattable.surface_for(sk.normals[i:i + 1, 2])[0]
        assert int(builder.shard.direct[surface].count.sum()) == direct.sum(), f"{name}: direct table"


def test_export_geometry_hands_fitqun_the_simulators_sensors(sk, tmp_path):
    h5py = pytest.importorskip("h5py")
    order = np.random.default_rng(0).permutation(len(sk.positions_m))   # any sensor-file order
    sensor = tmp_path / "wc_sensor_0000.h5"
    with h5py.File(sensor, "w") as f:
        f["config/sensor_positions"] = sk.positions_m[order].astype(np.float32)
    out = tmp_path / "geom.txt"
    subprocess.run([sys.executable, str(REPO / "tools" / "export_geometry.py"), str(NPZ),
                    "--sensor", str(sensor), "-o", str(out)], check=True, capture_output=True)
    rows = np.array([[float(x) for x in line.split()[2:8]]
                     for line in out.read_text().splitlines() if line.startswith("pmt ")])
    assert len(rows) == len(order)
    assert np.allclose(rows[:, :3], sk.positions_m[order].astype(np.float32) * 100.0, atol=1e-4)
    assert np.allclose(rows[:, 3:], sk.normals[order], atol=1e-6)
