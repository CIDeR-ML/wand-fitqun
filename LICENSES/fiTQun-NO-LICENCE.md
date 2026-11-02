# fiTQun / WCSimFQTuner vendored C++ — no licence text exists

This file exists so the gap is on the record rather than inferred from an
empty directory.

`tools/reference/` contains **2,773 lines** of upstream fiTQun and
WCSimFQTuner C++ across 12 files. Verified on `origin/main`:

```
$ grep -ril 'copyright\|SPDX\|licen' tools/reference/*.cc \
      tools/reference/*.C tools/reference/*.cxx \
      tools/reference/*.h
(no output)
```

Not one of the 12 files carries a copyright header, an SPDX tag or a licence
notice, and `tools/reference/PATCHES.md:9` states that "anything not
listed below is byte-identical to upstream" — i.e. these are verbatim copies,
not independent work. `PATCHES.md:299` records that the project has no push
access to fiTQun, so the terms cannot be settled by asking upstream through
the repository.

fiTQun is Super-Kamiokande collaboration software. It is not publicly
released, it carries no public licence, and redistributing it — which a public
GitHub repository does on every clone — is not something an absent licence
permits.

**What this blocks**

* Removing these files from the tip is necessary but not sufficient: a public
  git history keeps every blob reachable. Any history rewrite that is run for
  other reasons has to remove `tools/reference/{*.cc,*.C,*.cxx,*.h}` in
  the same pass, or the deletion is cosmetic.
* The 6 unified diffs under `tools/reference/patches/` (349 lines, the
  majority of them upstream context) are a smaller but real derivative of the
  same sources. Keeping only the patches plus `apply.sh`, in a **private**
  repository, is the safe reading under either outcome.
* `tools/reference/fitqun_SK_WAND.parameters.dat` is described in
  `PATCHES.md:313` as "our own tune file, not a patch", but it carries SK-IV
  fitted constants taken from fiTQun's shipped SK parameter file (for example
  `WaterAttenuationLengthWCSim = 7496.46` at line 39, with the comment at
  line 24 saying that "SK-IV's 7496.46 is kept"). Those numbers are
  collaboration output, and the file's own description of itself is wrong.

No LUCiD licence decision changes any of this: the inbound terms are the
constraint, and there are none.
