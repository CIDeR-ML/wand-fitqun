#!/usr/bin/env python3
"""Water transparency for fiTQun's T(R), measured the way the reference measures it.

    transparency.py <photonsim.root> <out.npz> --seed S

fiTQun predicts every charge through J = Omega(R) T(R) eps(cos eta) with
T = exp(-R/L) (fiTQun.cc:776-812), and its scattered prediction is J times the 6D
table, which is indirect light relative to the simulated direct light
(makeScatTable.C). So L has to reproduce the simulated direct light versus
distance. The reference measures exactly that in scatTableLooper.C
(MakeTransHistos, Write): direct photons on barrel PMTs whose source sits in front
of the PMT -- |r_s sin(dphi)| < r_PMT and |z_s - z_PMT| < r_PMT, so cos(eta) ~ 1 and
eps is constant -- histogrammed in source-to-PMT distance and fitted with
a exp(-R/L) / (R^2 + 645.16) over 500-3000 cm (fit_transparency.py).

A second histogram replaces that fixed-width tube by a fixed-angle cone around each
barrel PMT's normal. Not part of the reference, but far more efficient: the sources
in a cone shell grow as R^2 and cancel the 1/R^2 solid angle, so photons no longer
pile up next to the PMT. The tube histogram stays as the literal cross-check.

The photons are the 6D-table electron bomb, propagated as
`lucid_fitqun.fitqun sample run` propagates them (same macro, per-event
vertex, physics config and kernel settings); only the reduction differs.
"""
from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path

import numpy as np

# LUCiD's config/ next to the lucid package that is imported (LUCID_CONFIG overrides it)
CONFIG = Path(os.environ.get("LUCID_CONFIG") or Path(importlib.util.find_spec("lucid").origin).parents[1] / "config")
DETECTOR_CONFIG = str(CONFIG / "SK_WAND_geom_config.json")
PHYSICS_CONFIG = str(CONFIG / "SK_WAND_physics_config.json")
# scatTableLooper.C: hphotdist binning (line 270) and the barrel/cap split (line 354).
EDGES_CM = np.linspace(0.0, 3500.0, 1001)
CAP_THR = 0.99
# Cone selection (not in the reference): incidence within a fixed angle of the PMT
# normal. With sources uniform in volume, the sources in a cone shell grow as R^2 and
# cancel the PMT's 1/R^2 solid angle. Binned in cos(eta) and |z_PMT| so the cone angle,
# and the |z| cut that keeps the cone inside the source volume, are chosen at fit time.
CONE_R_EDGES_CM = np.linspace(0.0, 3500.0, 351)
CONE_COS_EDGES = np.linspace(0.95, 1.0, 11)
CONE_ZABS_EDGES_CM = np.linspace(0.0, 1900.0, 20)


def head_on_distances(src_cm, pmt_cm, *, cyl_radius_cm, halflength_cm, tube_radius_cm):
    """Source-to-PMT distances of the photons MakeTransHistos keeps (scatTableLooper.C:430-450)."""
    zs, zt = src_cm[:, 2], pmt_cm[:, 2]
    rs = np.hypot(src_cm[:, 0], src_cm[:, 1])
    dphi = np.arctan2(pmt_cm[:, 1], pmt_cm[:, 0]) - np.arctan2(src_cm[:, 1], src_cm[:, 0])
    barrel = np.abs(zt) <= CAP_THR * halflength_cm
    keep = (barrel & (np.abs(np.sin(dphi) * rs) < tube_radius_cm)
            & (np.abs(zs - zt) < tube_radius_cm))
    # Measured to the PMT's point on the cylinder radius, as the reference does.
    rsq = (zt ** 2 + cyl_radius_cm ** 2 + zs ** 2 + rs ** 2
           - 2.0 * (zt * zs + cyl_radius_cm * rs * np.cos(dphi)))
    return np.sqrt(np.maximum(rsq[keep], 0.0))


def cone_coordinates(src_cm, pmt_cm, pmt_dir, *, cyl_radius_cm, halflength_cm):
    """(R, cos eta, |z_PMT|) of barrel-PMT photons inside the widest stored cone.

    Measured from the PMT's point on the cylinder radius, the tube selection's apex;
    eta is the angle to the PMT's own facing direction."""
    zt = pmt_cm[:, 2]
    phi = np.arctan2(pmt_cm[:, 1], pmt_cm[:, 0])
    apex = np.stack([cyl_radius_cm * np.cos(phi), cyl_radius_cm * np.sin(phi), zt], axis=1)
    d = src_cm - apex
    r = np.linalg.norm(d, axis=1)
    cos_eta = np.einsum("ij,ij->i", d, pmt_dir) / np.maximum(r, 1e-9)
    keep = (np.abs(zt) <= CAP_THR * halflength_cm) & (cos_eta >= CONE_COS_EDGES[0])
    return r[keep], cos_eta[keep], np.abs(zt[keep])


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("photonsim_root")
    p.add_argument("out", type=Path)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--geometry", required=True,
                   help="sensor geometry .npz in the propagator's sensor order (e.g. sk_geometry_detorder.npz)")
    p.add_argument("--n-photons", type=int, default=20000)
    p.add_argument("--batch", type=int, default=8)
    a = p.parse_args()

    import jax
    from lucid.geometry import generate_detector
    from lucid_fitqun.fitqun import isotropic_sample
    from lucid.simulation.shotgun import setup_shotgun_simulator
    from lucid.sources.shotgun_source import shotgun_source, stack_shotgun_sources

    g = np.load(a.geometry)
    pmt_cm = g["positions_mm"] / 10.0
    cyl_r = float(g["radius"]) * 100.0
    half = float(g["height"]) * 100.0 / 2.0
    tube_r = float(g["sensor_radius"]) * 100.0
    pmt_dir = g["directions"].astype(np.float64)

    # Kernel settings of sample_propagate.propagate_and_reduce, which built the 6D table.
    sim = setup_shotgun_simulator(
        DETECTOR_CONFIG, physics_config=PHYSICS_CONFIG, n_photons=a.n_photons,
        hits="photon_hits", K=12, tts_sigma_ns=1.0,
        wavelength_sampling="cherenkov")
    dev = float(np.abs(np.asarray(generate_detector(DETECTOR_CONFIG).all_points, dtype=np.float64) * 100.0 - pmt_cm).max())
    print(f"geometry vs propagator sensor positions: max |diff| = {dev:.3f} cm")
    # The propagator snaps sensors onto the wall (< 3 cm); a wrong order is off by metres.
    if dev > 10.0:
        raise SystemExit("geometry file is not in the propagator's sensor order")

    h = np.zeros(len(EDGES_CM) - 1)
    hc = np.zeros((len(CONE_R_EDGES_CM) - 1, len(CONE_COS_EDGES) - 1, len(CONE_ZABS_EDGES_CM) - 1))
    n = {"photons": 0, "detected": 0, "direct": 0, "selected": 0, "cone": 0}
    events = set()
    rng = np.random.default_rng(a.seed)
    key = jax.random.PRNGKey(a.seed)

    def flush(pending, key):
        if not pending:
            return key
        key, sub = jax.random.split(key)
        batched = stack_shotgun_sources(pending)
        det, sid, _, ind = sim.batch(batched, jax.random.split(sub, len(pending)))
        det = np.asarray(det).reshape(-1).astype(bool)
        ind = np.asarray(ind).reshape(-1).astype(bool)
        sid = np.asarray(sid).reshape(-1)
        direct = det & ~ind
        src = np.asarray(batched.origins).reshape(-1, 3)[direct] * 100.0
        r = head_on_distances(src, pmt_cm[sid[direct]], cyl_radius_cm=cyl_r,
                              halflength_cm=half, tube_radius_cm=tube_r)
        h[:] += np.histogram(r, EDGES_CM)[0]
        rc, cc, zc = cone_coordinates(src, pmt_cm[sid[direct]], pmt_dir[sid[direct]],
                                      cyl_radius_cm=cyl_r, halflength_cm=half)
        hc[:] += np.histogramdd(np.stack([rc, cc, zc], axis=1),
                                bins=(CONE_R_EDGES_CM, CONE_COS_EDGES, CONE_ZABS_EDGES_CM))[0]
        n["cone"] += rc.size
        n["photons"] += det.size
        n["detected"] += int(det.sum())
        n["direct"] += int(direct.sum())
        n["selected"] += r.size
        return key

    # Same streaming as sample_propagate: one vertex per event, fixed-size groups,
    # the incomplete tail group dropped.
    carry_o = np.zeros((0, 3), dtype=np.float32)
    carry_d = np.zeros((0, 3), dtype=np.float32)
    pending = []
    for origins_m, directions, event_id in isotropic_sample.load_photons(a.photonsim_root):
        events.update(np.unique(event_id).tolist())
        origins_m = isotropic_sample.translate_uniform(
            origins_m, rng, det_radius_m=float(g["radius"]),
            det_halfheight_m=float(g["height"]) / 2.0,
            pmt_radius_m=float(g["sensor_radius"]), event_id=event_id)
        carry_o = np.concatenate([carry_o, origins_m])
        carry_d = np.concatenate([carry_d, directions])
        used = 0
        for lo in range(0, carry_o.shape[0] - a.n_photons + 1, a.n_photons):
            sl = slice(lo, lo + a.n_photons)
            pending.append(shotgun_source(carry_o[sl], carry_d[sl], n_photons=a.n_photons))
            used += a.n_photons
            if len(pending) == a.batch:
                key = flush(pending, key)
                pending = []
        carry_o, carry_d = carry_o[used:], carry_d[used:]
    flush(pending, key)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out, edges=EDGES_CM, h=h, events=len(events), seed=a.seed,
                        tube_radius_cm=tube_r, cyl_radius_cm=cyl_r, halflength_cm=half,
                        cone_h=hc, cone_r_edges=CONE_R_EDGES_CM, cone_cos_edges=CONE_COS_EDGES,
                        cone_zabs_edges=CONE_ZABS_EDGES_CM,
                        src_radius_cm=cyl_r - tube_r, src_halflength_cm=half - tube_r, **n)
    print(f"{len(events):,} electrons, {n['photons']:,} photons propagated, "
          f"{n['detected']:,} detected ({n['direct']:,} direct), "
          f"{n['selected']:,} head-on direct photons, {n['cone']:,} in the cos(eta) > "
          f"{CONE_COS_EDGES[0]} cone -> {a.out}")
    print("TRANSPARENCY_DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
