# Sample jobs

`generate_sample_jobs.py` writes, and with `-s` submits, the jobs of the isotropic 3 MeV
electron sample behind the angular-response and indirect-light tables (SLURM or HTCondor; see
LUCiD's `docs/guides/production/cluster-abstraction.md`). Each job runs PhotonSim on a macro
written at submit time, then `python -m lucid_fitqun.fitqun sample run` (propagation and
reduction in one pass), then deletes the ROOT file; the shard is all it keeps.

The WAND tune's sample is the fiTQun fork's `sk_wand/samples/sample_sk_wand_v4.json`. The
fork's `tools/scattable/ebomb_jobs.sh` writes those jobs with this script's `job_command`, so
they run exactly what the script would.

```bash
# 1. Paths, from one of LUCiD's templates
cp $LUCID/lucid/production/platform/templates/user_paths.lxplus.sh.template user_paths.sh   # or .s3df / .nersc
vim user_paths.sh

# 2. One smoke job, then the whole sample
python3 generate_sample_jobs.py -c <sample config> -t -s
python3 generate_sample_jobs.py -c <sample config> -s

# 3. Merge the shards into the tuning inputs
python -m lucid_fitqun.fitqun sample build <out>/job_*/shard.npz -o <out>
```

Re-running skips jobs that already have `shard.npz`; `--no-skip-existing` forces them.
