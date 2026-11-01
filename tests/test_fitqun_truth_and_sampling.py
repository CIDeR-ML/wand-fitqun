"""fiTQun tuning modules that had no test: truth, the source placement, and the profile reader.

- ``truth``: momentum from kinetic energy (closed form p^2 = T^2 + 2Tm), the primary is
  ``track_idx == 0``, which events are skipped, and the m -> cm text export.
- ``tools/fitqun/fq_residuals.py``: the longitudinal/transverse vertex split it quotes for fiTQun,
  on a hand-built fiTQun tree.
- ``isotropic_sample.translate_uniform`` / ``translate_shell``: uniform in VOLUME (r^2 in a
  cylinder, r^3 in a shell), one vertex per event, the shell radius converted cm -> m.
- ``cprofile.accumulate``: PhotonSim writes mm, the profile is in cm.

Each builds its inputs by hand (h5py, uproot), so nothing here needs PhotonSim or a dataset.
"""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from lucid.production.fitqun import cprofile, isotropic_sample, truth

uproot = pytest.importorskip("uproot")
h5py = pytest.importorskip("h5py")

REPO = Path(__file__).resolve().parents[1]
M_MU, M_E = 105.6583755, 0.51099895      # PDG RPP 2022, MeV


def _p_from_t(t, m):
    """p^2 = (T + m)^2 - m^2 = T^2 + 2 T m."""
    return np.sqrt(t * t + 2.0 * t * m)


# --- truth ---------------------------------------------------------------------------------

def _write_truth(tmp_path, events):
    """labl/step HDF5 in the layout ``truth.read_tracks`` reads; one group per event."""
    labl, step = tmp_path / "labl.h5", tmp_path / "step.h5"
    with h5py.File(labl, "w") as fl, h5py.File(step, "w") as fs:
        for i, ev in enumerate(events):
            key = f"event_{i:03d}"
            inter = fl.create_group(f"{key}/per_interaction")
            inter["primary_pdgs_data"] = np.asarray(ev["pdgs"])
            inter["primary_energies_data"] = np.asarray(ev["kinetic"])
            for ax, v in zip("xyz", ev["vertex"]):
                inter[f"vertex_{ax}"] = np.array([v])
            inter["t0"] = np.array([ev["t0"]])
            fl.create_group(f"{key}/per_event")["contained"] = ev["contained"]
            st = fs.create_group(key)
            seg = ev["segments"]                # (track_idx, time, dir, beta)
            st["track_idx"] = np.array([s[0] for s in seg])
            st["time"] = np.array([s[1] for s in seg])
            for j, ax in enumerate("xyz"):
                st[f"dir_{ax}"] = np.array([s[2][j] for s in seg])
            st["beta_start"] = np.array([s[3] for s in seg])
    return labl, step


def _beta(t, m):
    return _p_from_t(t, m) / (t + m)


def _truth_events():
    t_mu, t_e = 500.0, 800.0
    return [
        # 0: a muon. Track 1 (a delta ray) starts earlier and points elsewhere; the primary is
        #    track 0, and within it the EARLIEST segment (dir given unnormalised).
        dict(pdgs=[13], kinetic=[t_mu], vertex=(1.0, -2.0, 0.5), t0=3.0, contained=True,
             segments=[(0, 2.0, (0.0, 0.6, 0.0), 0.9), (1, 0.5, (1.0, 0.0, 0.0), 0.79),
                       (0, 1.0, (0.0, 0.0, 2.0), _beta(t_mu, M_MU))]),
        # 1: two primaries -- which one a single-ring fit should find is undefined: skipped
        dict(pdgs=[13, 11], kinetic=[t_mu, t_e], vertex=(0, 0, 0), t0=0.0, contained=True,
             segments=[(0, 0.0, (0.0, 0.0, 1.0), _beta(t_mu, M_MU))]),
        # 2: beta says p = 58 MeV/c, energy says 596: the wrong track; skipped
        dict(pdgs=[13], kinetic=[t_mu], vertex=(0, 0, 0), t0=0.0, contained=True,
             segments=[(0, 0.0, (0.0, 0.0, 1.0), 0.48)]),
        # 3: a proton has no mass in PDG_MASSES: skipped
        dict(pdgs=[2212], kinetic=[t_mu], vertex=(0, 0, 0), t0=0.0, contained=True,
             segments=[(0, 0.0, (0.0, 0.0, 1.0), 0.7)]),
        # 4: an electron, not contained
        dict(pdgs=[11], kinetic=[t_e], vertex=(-3.0, 0.0, 4.0), t0=-1.5, contained=False,
             segments=[(0, 0.0, (1.0, 1.0, 0.0), _beta(t_e, M_E))]),
    ]


def test_read_tracks_takes_track_zero_and_momentum_from_kinetic_energy(tmp_path):
    labl, step = _write_truth(tmp_path, _truth_events())
    tracks = truth.read_tracks(labl, step)
    assert [t.event for t in tracks] == [0, 4]

    mu, el = tracks
    assert mu.pdg == 13 and mu.momentum_mev == pytest.approx(_p_from_t(500.0, M_MU), rel=1e-12)
    assert mu.momentum_mev == pytest.approx(596.37, abs=0.01)
    np.testing.assert_allclose(mu.direction, [0.0, 0.0, 1.0])       # earliest track-0 segment
    np.testing.assert_allclose(mu.vertex_m, [1.0, -2.0, 0.5])
    assert (mu.t0_ns, mu.contained) == (3.0, True)
    assert mu.kinetic_mev == pytest.approx(500.0, rel=1e-12)          # and back

    assert el.momentum_mev == pytest.approx(_p_from_t(800.0, M_E), rel=1e-12)
    np.testing.assert_allclose(el.direction, [2 ** -0.5, 2 ** -0.5, 0.0])
    assert el.contained is False


def test_momentum_from_beta_is_beta_gamma_m():
    """The cross-check: p = beta gamma m. At beta = 0.9973, beta gamma = 13.5806, so a muon has
    1434.9 MeV/c (the module docstring's 1434.8 is the same event with beta unrounded)."""
    assert truth._momentum_from_beta(0.9973, 13) == pytest.approx(1434.9, abs=0.05)
    b = _beta(500.0, M_MU)
    assert truth._momentum_from_beta(b, 13) == pytest.approx(_p_from_t(500.0, M_MU), rel=1e-9)


def test_write_text_converts_metres_to_cm(tmp_path):
    tr = truth.TruthTrack(event=7, pdg=13, vertex_m=np.array([1.5, -2.0, 0.25]),
                          direction=np.array([0.0, 0.6, 0.8]), momentum_mev=596.25,
                          t0_ns=3.0, contained=True)
    line = truth.write_text([tr], tmp_path / "t.txt").read_text().split()
    assert line[:3] == ["track", "7", "13"]
    np.testing.assert_allclose([float(x) for x in line[3:]],
                               [150.0, -200.0, 25.0, 3.0, 0.0, 0.6, 0.8, 596.25])


# --- tools/fitqun/fq_residuals.py --------------------------------------------------------

def _load_tool(name):
    path = REPO / "tools" / "fitqun" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_tools_fitqun_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_fq_residuals_splits_the_vertex_error_along_the_track(tmp_path):
    """True muon at (1, -2, 0.5) m going +z. fiTQun's mu- hypothesis (index 2) is put 40 cm
    downstream and 30 cm sideways, its direction 5 degrees off, its momentum 10% high. So
    dL = +40 cm, dT = 30 cm, dtheta = 5 deg, dp/p = 0.10, and dwall is the distance to the SK_WAND
    barrel, 1696.23 - |(100, -200)| cm."""
    events = _truth_events()
    labl, step = _write_truth(tmp_path, events)
    p_true = _p_from_t(500.0, M_MU)

    n_ev = len(events)
    pos = np.zeros((n_ev, 1, 7, 3), np.float32)
    dirs = np.tile(np.float32([0, 0, 1]), (n_ev, 1, 7, 1))
    mom = np.full((n_ev, 1, 7), 100.0, np.float32)
    pc = np.zeros((n_ev, 1, 7), np.int32)
    pos[0, 0, 2] = [100.0 + 30.0, -200.0, 50.0 + 40.0]
    th = np.radians(5.0)
    dirs[0, 0, 2] = [np.sin(th), 0.0, np.cos(th)]
    mom[0, 0, 2] = 1.1 * p_true
    fq = tmp_path / "fq.root"
    with uproot.recreate(fq) as f:
        f["fiTQun"] = {"fq1rpos": pos, "fq1rdir": dirs, "fq1rmom": mom, "fq1rpcflg": pc}

    out = tmp_path / "res.txt"
    mod = _load_tool("fq_residuals")
    assert mod.main(["--fq", str(fq), "--labl", str(labl), "--step", str(step),
                     "--pdg", "13", "-o", str(out)]) == 0
    rows = np.atleast_2d(np.loadtxt(out))
    # truth.read_tracks keeps events 0 and 4 (an electron, read as the mu- hypothesis here)
    assert rows.shape == (2, 7)
    dL, dT, dth, dp, dwall, is_pc, contained = rows[0]
    assert dL == pytest.approx(40.0, abs=1e-3)
    assert dT == pytest.approx(30.0, abs=1e-3)
    assert dth == pytest.approx(5.0, abs=1e-4)
    assert dp == pytest.approx(0.10, abs=1e-5)
    assert dwall == pytest.approx(1696.23 - np.hypot(100.0, 200.0), abs=0.01)   # %.6g output
    assert (is_pc, contained) == (0.0, 1.0)


# --- isotropic_sample --------------------------------------------------------------------

R_M, HZ_M, PMT_M = 16.9, 18.1, 0.254      # SK-like extent and a 20" PMT radius, metres
N = 20_000


def test_translate_uniform_fills_the_pmt_enclosed_cylinder_uniformly_in_volume():
    """scatTableLooper.C's volume (rmax = R - r_pmt, zmax = H - r_pmt), uniform in volume:
    P(rho < rmax/2) = 1/4 (sampling rho linearly gives 1/2), z flat in [-zmax, zmax]. Each
    event gets its own vertex, shared by all of its photons."""
    rng = np.random.default_rng(0)
    event_id = np.repeat(np.arange(N), 2)                    # two photons per event
    origins = np.zeros((2 * N, 3), np.float32)
    v = isotropic_sample.translate_uniform(origins, rng, det_radius_m=R_M, det_halfheight_m=HZ_M,
                                           pmt_radius_m=PMT_M, event_id=event_id)
    np.testing.assert_array_equal(v[0::2], v[1::2])
    v = v[0::2].astype(np.float64)
    rmax, zmax = R_M - PMT_M, HZ_M - PMT_M
    rho = np.hypot(v[:, 0], v[:, 1])
    assert rho.max() <= rmax * (1 + 1e-6) and np.abs(v[:, 2]).max() <= zmax * (1 + 1e-6)
    assert rho.max() > 0.99 * rmax and np.abs(v[:, 2]).max() > 0.99 * zmax
    assert np.mean(rho < rmax / 2) == pytest.approx(0.25, abs=0.015)     # SE 0.003
    assert np.mean(v[:, 2] > 0) == pytest.approx(0.5, abs=0.015)
    assert len(np.unique(v, axis=0)) == N                     # one vertex per event


def test_translate_shell_draws_inside_the_shell_uniformly_in_volume():
    """A 100 +- 50 cm shell (the one fit_cos.C fits) on the inward hemisphere of one barrel PMT:
    distance in [0.5, 1.5) m, cos(eta) in [0, 1]. Uniform in volume means r^3 is uniform, so
    the median r is cbrt((0.5^3 + 1.5^3)/2) = 1.205 m (a linear-in-r draw: 1.0 m). Checked where
    cos(eta) > 0.9, where no vertex is rejected for leaving the source volume; there the
    fraction below 1.205 m is 1/2 (linear in r would give 0.70)."""
    rng = np.random.default_rng(1)
    pmt = np.array([[R_M, 0.0, 0.0]])
    axis = np.array([[-1.0, 0.0, 0.0]])
    v = isotropic_sample.translate_shell(
        np.zeros((N, 3), np.float32), rng, pmt_pos_m=pmt, pmt_axes=axis,
        shell_r_cm=100.0, shell_dr_cm=50.0, det_radius_m=R_M, det_halfheight_m=HZ_M,
        pmt_radius_m=PMT_M, event_id=np.arange(N)).astype(np.float64)
    d = v - pmt[0]
    r = np.linalg.norm(d, axis=1)
    ceta = d @ axis[0] / r
    assert r.min() >= 0.5 - 1e-5 and r.max() < 1.5 + 1e-5
    assert ceta.min() >= -1e-6 and ceta.max() <= 1.0 + 1e-6
    assert np.hypot(v[:, 0], v[:, 1]).max() <= (R_M - PMT_M) * (1 + 1e-6)

    fwd = ceta > 0.9
    assert fwd.sum() > 1500
    r_med = np.cbrt((0.5 ** 3 + 1.5 ** 3) / 2)
    assert np.mean(r[fwd] < r_med) == pytest.approx(0.5, abs=0.04)   # SE ~0.012


# --- cprofile.accumulate -----------------------------------------------------------------

def test_accumulate_reads_photonsim_millimetres_as_centimetres(tmp_path):
    """Photons emitted evenly over z in [0, 950) mm = [0, 95) cm at cos(theta) = 0.75. The
    profile's 90% point (genhist.cc gsthr) is then 85.5 cm, in the 2.5 cm bin [85, 87.5), centre
    86.25 cm; at a tenth of the scale it would be 8.75 cm, read as cm off the axis entirely.
    All the light sits in the cos(theta) bin holding 0.75, and the yield is photons per event."""
    n_ev, per_ev = 4, 2500
    z = [(np.arange(per_ev) + 0.5) * 950.0 / per_ev for _ in range(n_ev)]
    zeros = [np.zeros(per_ev) for _ in range(n_ev)]
    sin = np.sqrt(1 - 0.75 ** 2)
    path = tmp_path / "photonsim.root"
    import awkward as ak
    with uproot.recreate(path) as f:
        f["OpticalPhotons"] = {"NOpticalPhotons": np.full(n_ev, per_ev, np.int32)}
        f["OpticalPhotonsRaw"] = {
            "PhotonPosX": ak.Array(zeros), "PhotonPosY": ak.Array(zeros),
            "PhotonPosZ": ak.Array(z),
            "PhotonDirX": ak.Array([np.full(per_ev, sin)] * n_ev),
            "PhotonDirY": ak.Array(zeros),
            "PhotonDirZ": ak.Array([np.full(per_ev, 0.75)] * n_ev)}

    cell = cprofile.accumulate(path, pdg=13, momentum_mev=500.0)
    assert cell.s_max_cm == pytest.approx(86.25)
    assert cell.n_photons == per_ev and cell.n_events == n_ev

    s_c = 0.5 * (cell.s_edges[1:] + cell.s_edges[:-1])
    ds = cell.s_edges[1] - cell.s_edges[0]
    dc = cell.costh_edges[1] - cell.costh_edges[0]
    marginal = cell.density.sum(axis=1) * dc * ds                 # probability per s bin
    assert marginal.sum() == pytest.approx(1.0)
    assert marginal[(s_c > 0) & (s_c < 95)].sum() == pytest.approx(1.0)
    ic = np.searchsorted(cell.costh_edges, 0.75, side="right") - 1
    assert (cell.density.sum(axis=0) * dc * ds)[ic] == pytest.approx(1.0)
