#!/usr/bin/env python3
"""Direct-light digits for the fiTQun time PDF (muon by default), from LUCiD's own propagation.

    tpdf_direct.py <photonsim.root> <out-dir> --kinetic-mev KE --seed S [--pdg 13] [--all-light]

fiTQun's time likelihood treats direct and scattered light separately: the time
PDF (tpdfpar) describes direct light only, and scattered light enters through the
6D table and its own timing term. The reference trains the time PDF on MC with
scattered and reflected photons killed (timePDF_base.mac: /opInfo/KillScatterRef
true). LUCiD already labels every propagated photon direct or indirect (the
`indirect` flag of photon_step, the label the 6D table is built from), so the
direct light is taken from LUCiD's own propagation rather than from a modified
simulation:

  * photons, creation times and wavelengths from PhotonSim, read with
    production's reader (root_reader.read_event_raw);
  * placement as production: vertex uniform in r <= 0.9R, |z| <= 0.9H/2
    (placement.sample_vertex), t0 uniform in +-250 ns;
  * production-equivalent per-photon propagation: hard sensor edges, leg bound,
    no extra photon-time spread; arrival = creation time + travel time;
  * keep the photons not flagged indirect (all detected photons with --all-light);
  * production digitiser (ski, dark noise off) and production trigger, keeping
    window 1 only.

Writes sensor.h5 (event_NNN/{PE,T,sensor_idx} and config/sensor_positions in the
propagator's sensor order), gates.h5 (window-1 digit_offsets) and truth.txt, the
inputs lucid_to_wcsim reads.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

import importlib.util
import os

from lucid.pdg import PDG_MASS_MEV

MASS = {pdg: PDG_MASS_MEV[pdg] for pdg in (11, 13, 211)}   # MeV, by |PDG|
# LUCiD's config/ next to the lucid package that is imported (LUCID_CONFIG overrides it)
CONFIG = Path(os.environ.get("LUCID_CONFIG") or Path(importlib.util.find_spec("lucid").origin).parents[1] / "config")
DETECTOR_CONFIG = str(CONFIG / "SK_WAND_geom_config.json")
PHYSICS_CONFIG = str(CONFIG / "SK_WAND_physics_config.json")
GEOMETRY_NPZ = str(CONFIG / "sk_geometry.npz")


def primary_direction(tree, entry, pdg_abs):
    """Unit direction of the parentless primary (|PDG| = pdg_abs) of one PhotonSim event."""
    b = tree.arrays(["TrackInfo_ParentTrackID", "TrackInfo_PDG", "TrackInfo_DirX",
                     "TrackInfo_DirY", "TrackInfo_DirZ"],
                    entry_start=entry, entry_stop=entry + 1, library="np")
    parent = np.asarray(b["TrackInfo_ParentTrackID"][0])
    pdg = np.asarray(b["TrackInfo_PDG"][0])
    sel = np.flatnonzero((parent == 0) & (np.abs(pdg) == pdg_abs))
    if sel.size != 1:
        raise ValueError(f"entry {entry}: expected one primary of |PDG| {pdg_abs}, found {sel.size}")
    i = int(sel[0])
    d = np.array([b["TrackInfo_DirX"][0][i], b["TrackInfo_DirY"][0][i],
                  b["TrackInfo_DirZ"][0][i]], dtype=np.float64)
    return d / np.linalg.norm(d)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("photonsim_root")
    p.add_argument("out_dir", type=Path)
    p.add_argument("--kinetic-mev", type=float, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--pdg", type=int, default=13, choices=sorted(MASS))
    p.add_argument("--all-light", action="store_true",
                   help="keep indirect photons too (validation against production)")
    p.add_argument("--n-photons", type=int, default=20000)
    p.add_argument("--batch", type=int, default=8)
    a = p.parse_args()

    import h5py
    import jax
    import uproot
    from lucid.geometry import generate_detector
    from lucid.simulation.digitizer import digitize_and_decompose, resolve_model_config
    from lucid.simulation.shotgun import setup_shotgun_simulator
    from lucid.simulation.trigger import TriggerConfig, assign_windows, find_trigger_gates
    from lucid.sources.root_reader import read_event_raw
    from lucid.sources.shotgun_source import shotgun_source, stack_shotgun_sources
    from lucid.sources.placement import sample_vertex

    phys = json.loads(Path(PHYSICS_CONFIG).read_text())
    model = resolve_model_config({**phys["digitizer"], "dark_rate_khz": 0.0})
    trig = TriggerConfig.from_block(phys["trigger"])
    g = np.load(GEOMETRY_NPZ)
    bounds = {"type": "cylinder", "radius": float(g["radius"]), "height": float(g["height"])}

    nph = a.n_photons
    sim = setup_shotgun_simulator(
        DETECTOR_CONFIG, physics_config=PHYSICS_CONFIG, n_photons=nph,
        hits="photon_hits", K=12, tts_sigma_ns=0.0,
        wavelength_sampling="cherenkov")
    sensor_pos = np.asarray(generate_detector(DETECTOR_CONFIG).all_points, dtype=np.float32)
    n_sensors = sensor_pos.shape[0]

    rng = np.random.default_rng(a.seed)
    key = jax.random.PRNGKey(a.seed)
    momentum = math.sqrt(a.kinetic_mev ** 2 + 2.0 * a.kinetic_mev * MASS[a.pdg])
    print(f"bounds {bounds}; trigger {trig}; model ski, dark off; "
          f"{'ALL light' if a.all_light else 'DIRECT light only'}; p = {momentum:.3f} MeV/c")

    a.out_dir.mkdir(parents=True, exist_ok=True)
    fs = h5py.File(a.out_dir / "sensor.h5", "w")
    fg = h5py.File(a.out_dir / "gates.h5", "w")
    fs.create_dataset("config/sensor_positions", data=sensor_pos)
    truth_lines = []
    n_kept = n_untrig = 0
    stats = []

    rf = uproot.open(a.photonsim_root)
    tree = rf["OpticalPhotons"]
    for entry in range(tree.num_entries):
        raw = read_event_raw(a.photonsim_root, entry, opened_file=rf)
        org = np.asarray(raw["photon_origins"], dtype=np.float32)
        dirs = np.asarray(raw["photon_directions"], dtype=np.float32)
        t_emit = np.asarray(raw["photon_times"], dtype=np.float64)
        wl = np.asarray(raw["photon_wavelengths"], dtype=np.float32)
        u = primary_direction(tree, entry, a.pdg)
        t0 = float(rng.uniform(-250.0, 250.0))
        vtx = sample_vertex(bounds, rng)
        org = org + vtx[None, :]
        n = org.shape[0]

        # Fixed-size groups for the kernel; the last is padded and its padding dropped.
        n_grp = max(1, math.ceil(n / nph))
        pad = n_grp * nph - n
        idx = np.concatenate([np.arange(n), np.arange(pad) % max(n, 1)])
        det = np.zeros(n_grp * nph, bool); sid = np.zeros(n_grp * nph, np.int64)
        ht = np.zeros(n_grp * nph); ind = np.zeros(n_grp * nph, bool)
        for lo in range(0, n_grp, a.batch):
            hi = min(lo + a.batch, n_grp)
            srcs = [shotgun_source(org[idx[k * nph:(k + 1) * nph]], dirs[idx[k * nph:(k + 1) * nph]],
                                   n_photons=nph, wavelength=wl[idx[k * nph:(k + 1) * nph]])
                    for k in range(lo, hi)]
            key, sub = jax.random.split(key)
            d_, s_, h_, i_ = sim.batch(stack_shotgun_sources(srcs), jax.random.split(sub, hi - lo))
            sl = slice(lo * nph, hi * nph)
            det[sl] = np.asarray(d_).reshape(-1).astype(bool)
            sid[sl] = np.asarray(s_).reshape(-1)
            ht[sl] = np.asarray(h_).reshape(-1)
            ind[sl] = np.asarray(i_).reshape(-1).astype(bool)
        det, sid, ht, ind = det[:n], sid[:n], ht[:n], ind[:n]
        keep = det & (True if a.all_light else ~ind)
        times = t_emit[keep] + ht[keep]
        m = int(keep.sum())
        z = np.zeros(m, np.int64)
        digits, _, _ = digitize_and_decompose(
            sensor_idx=sid[keep], charge=np.ones(m), t_true=times, t_reco=times,
            particle_idx=z, segment_idx=z, emission_process=z, n_sensors=n_sensors,
            model=model, rng=rng, dark_rate_khz=0.0, apply_resolution=True)
        T = np.asarray(digits["T"], dtype=np.float64) + t0
        gates = find_trigger_gates(T, trig)
        if gates.shape[0] == 0:
            n_untrig += 1
            continue
        w1 = assign_windows(T, gates) == 0
        s1 = np.asarray(digits["sensor_idx"])[w1]
        q1 = np.asarray(digits["PE"], dtype=np.float32)[w1]
        t1 = T[w1]
        o = np.lexsort((t1, s1))
        name = f"event_{n_kept:03d}"
        fs.create_dataset(f"{name}/PE", data=q1[o])
        fs.create_dataset(f"{name}/T", data=t1[o])
        fs.create_dataset(f"{name}/sensor_idx", data=s1[o].astype(np.uint16))
        fg.create_dataset(f"{name}/per_window/digit_offsets", data=np.array([0, o.size], np.int32))
        fg.create_dataset(f"{name}/per_window/window_start", data=gates[:1, 0])
        fg.create_dataset(f"{name}/per_window/window_end", data=gates[:1, 1])
        v_cm = np.asarray(vtx, dtype=np.float64) * 100.0
        truth_lines.append(f"track {n_kept} {a.pdg} {v_cm[0]:.6f} {v_cm[1]:.6f} {v_cm[2]:.6f} {t0:.6f} "
                           f"{u[0]:.8f} {u[1]:.8f} {u[2]:.8f} {momentum:.6f}")
        stats.append((n, int(det.sum()), int((det & ind).sum()), o.size, float(q1.sum())))
        n_kept += 1

    fs.close(); fg.close()
    (a.out_dir / "truth.txt").write_text("\n".join(truth_lines) + "\n")
    s = np.asarray(stats, dtype=np.float64) if stats else np.zeros((0, 5))
    print(f"kept {n_kept} events ({n_untrig} untriggered dropped)")
    if len(s):
        print(f"per event: photons {s[:,0].mean():.0f}, detected {s[:,1].mean():.0f} "
              f"(indirect {100*s[:,2].sum()/max(s[:,1].sum(),1):.1f}%), window-1 digits "
              f"{s[:,3].mean():.1f} +- {s[:,3].std()/math.sqrt(len(s)):.1f}, "
              f"window-1 charge {s[:,4].mean():.1f} +- {s[:,4].std()/math.sqrt(len(s)):.1f} pe")
    print("TPDF_DIRECT_DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
