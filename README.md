# LUCiD-fitqun

The interface between [LUCiD](https://github.com/CIDeR-ML/LUCiD) and fiTQun: the LUCiD side of
the fiTQun tuning chain (the tuning inputs LUCiD generates, and the conversion of LUCiD output
into fiTQun's input), and the photon-shotgun scan app. Both are applications *of* LUCiD: they
import `lucid` and add nothing to it. fiTQun itself, SK's tuning macros, the tune's parameter
files and the drivers that run fiTQun live with fiTQun, not here.

| path | what it is |
|---|---|
| `lucid_fitqun/fitqun/` | generators for fiTQun's tuning inputs (Cherenkov profile, charge PDF, angular response, scattering tables, time PDF), the tune manifest and its report. See its `README.md` |
| `lucid_fitqun/jobs/` | the HTCondor/Slurm fan-out for those generators, and its configs |
| `lucid_fitqun/photon_shotgun/` | per-photon and waveform scans with hand-specified photons, with three notebooks |
| `tools/` | LUCiD to fiTQun and back: geometry export (`export_geometry.py`), the WCSim-format converter (`lucid_to_wcsim.cc`), the scattering-table converter (`h5_to_scattable.C`), the direct-light time-PDF samples (`tpdf_direct.py`), the water-transparency reduction (`transparency.py`), and scoring of fiTQun output against LUCiD truth (`fq_residuals.py`, `fq_resolution.py`, `qeeff_summary.py`) |
| `configs/tpdf/` | the 41 time-PDF sample configs for `lucid-run-job` |
| `config/` | the two diagnostic SK_WAND physics configs (`_noreflect`, `_scat10x`) and `materials/water_scat10x.json`. They name LUCiD's `materials/water.json` and `pmt/SK_QE.json` relative to their own directory, so copy them into a LUCiD checkout's `config/` to use them |

## Origin

Split from LUCiD commit `910a9a8a` (branch `tidy/F-051`) with `git filter-repo`,
keeping the history of every path above. Before the split these lived at
`lucid/production/fitqun/`, `lucid/production/photon_shotgun/`,
`lucid/production/jobs/fitqun/`, `lucid/production/configs/GeV/tpdf/`,
`tools/fitqun/` (earlier `tools/sadqun/`) and `tests/test_fitqun_*.py`,
`tests/test_shotgun_io.py`; the commit after the split only moves them into
this layout.

## Install and test

It depends on LUCiD and its environment (JAX, h5py, uproot); use a LUCiD
checkout installed with `pip install -e /path/to/LUCiD`, or put that checkout
on `PYTHONPATH`. `lucid_fitqun` itself is imported from this directory:

```bash
cd /path/to/LUCiD-fitqun
PYTHONPATH=/path/to/LUCiD python -m pytest -q tests
```

(`python -m pytest` puts the current directory on `sys.path`, which is what
makes `lucid_fitqun` importable.)

Cluster jobs need the same on the worker: the generated commands run
`python -m lucid_fitqun.fitqun ...`, and LUCiD's HTCondor adapter exports only
`PYTHONPATH=$LUCID_DEV_PATH`. Until the jobs carry this repository's path
themselves, install `lucid_fitqun` into the image or extend that export. The
job generators read `lucid_fitqun/jobs/user_paths.sh`, copied from one of
LUCiD's `lucid/production/jobs/user_paths.*.sh.template`.

## Sensor geometry

Every table and every geometry handed to fiTQun uses the sensors as the simulator sees
them: `lucid_fitqun/fitqun/sensors.py` takes their positions and normals from the
detector LUCiD builds from its geometry config, and `tools/export_geometry.py` takes
them from the sensor file the hits come with. A geometry file is never read in their
place: `sk_geometry.npz` lists the PMTs in another order (no row is the simulator's
sensor of that index), places half of SK's barrel PMTs 2.87 cm inside the wall the
simulator snaps them onto, and gives flat-panel normals 1-4 deg off the radial ones the
propagation uses. `tests/test_fitqun_sensors.py` fires photons at chosen PMTs and checks
each table files them against the PMT they hit.
