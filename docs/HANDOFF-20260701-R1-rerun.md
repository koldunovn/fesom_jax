# Handoff 2026-07-01 — R1 CORE2 hindcast RERUN (both bug fixes + canonical output) — for a separate session

**Mandate:** rerun the full CORE2 1958–2020 JAX hindcast (model-paper R1, the fidelity flagship) so it
lands the **two fixes committed today** AND switches to the **new default canonical output**, then
regenerate the paper data/figures and confirm both bugs are gone. The matched **Fortran R2 is UNCHANGED**
(Fortran was already correct) — only the JAX side reruns.

---

## 1. Why rerun — what changed since the current R1

Two bugs were found + fixed today (both committed **locally, not pushed**; the rerun uses the local repo
`/home/a/a270088/port_jax` so they're already in effect):

| commit | fix | changes the… |
|---|---|---|
| `5a61bb0` | **Salinity freshwater-budget leak** — `ice_thermo.py` now bundles `evaporation = evap + subli` so the zstar FW balance cancels sublimation (matches Fortran/C-port). | **MODEL TRAJECTORY** (the ~+0.0044 psu/63yr vol-mean S drift + −0.44 m ⟨SSH⟩ drift). Rerun REQUIRED. |
| `6977c62` | **Diurnal-aliased monthly output** — `run.py`/`integrate_sharded.py` now write a TRUE every-step time-mean (`_MeanStream` + `sample_fn` scan accumulation, split at date-based period boundaries). | **OUTPUT ONLY** (the JAX−Fortran wavenumber-1 SST artifact). Physics byte-identical; the fix removes the k=1 in the *mean*. |

Plus the user's decision: **do the rerun with the new default `'global'` CANONICAL output** (partition-
independent, node-order, the model paper's portable-output showcase), replacing the old **folded** ushow
zarrs. The sbatch passes **no** `--output-layout`, and the default is now `'global'`
(`run_from_config.py:109-110`), so **the rerun gets canonical output automatically — no sbatch change
needed for the format.** (Old R1 used folded because it predates the `'global'` default.)

**See the two resolution handoffs for the physics:** `HANDOFF-20260701-salinity-drift-investigation.md`
(✅ banner) and `HANDOFF-20260701-sst-pattern-investigation.md` (✅ banner); details in
`docs/PORTING_LESSONS.md`.

---

## 2. How to launch (cold start, self-chaining)

The driver `scripts/runs/run_core2_hindcast.sbatch` self-chains: each job resumes the rolling restart, runs a
step budget < 12 h walltime, writes a restart + monthly canonical zarrs, and resubmits until 63 yr
(`TOTAL=1,104,528` steps = 1958-01-01 → 2021-01-01). 1 node / 4 GPU (dist_4; CORE2 anti-scales past 4).
~8 production jobs, ~1.6 GPU-days at the reuse rate.

**Use a NEW `TAG` so the old R1 is preserved for comparison** (and so `CUR=0` ⇒ a clean COLD start from
the PHC IC — both fixes need a fresh trajectory):

```bash
cd /home/a/a270088/port_jax
TAG=core2_hindcast_v2 sbatch --export=ALL,TAG=core2_hindcast_v2 scripts/runs/run_core2_hindcast.sbatch
```

- Cold job = a **1-month CANARY** (`CANARY=1536` steps) — the all-on finiteness gate; non-finite ⇒ chain
  STOPS (inspect). Then it chains `PERJOB=150000`-step jobs. `do_resubmit` propagates `TAG`.
- Output → `/work/ab0995/a270088/port_jax/runs/core2_hindcast_v2/{restart, monthly/}`.
- Guards (unchanged, all in the sbatch): compile-hang self-heal, per-job `--diagnostics` NaN gate,
  rolling `--checkpoint-every 1440` restart, `.jobcount` runaway cap (35).
- `CHUNK=48` is now **only a perf knob** (the mean averages every step regardless) — you MAY raise it
  (e.g. 240) for fewer chunks/less overhead, bounded by forcing memory; leave 48 if you want the proven
  footprint. Bit-identical physics either way (restart-seam invariant).

**Watch the canary first** (`scripts/run_core2_hindcast.<jobid>.out`): confirm `chunk 1/` appears, steps
run, `--diagnostics` reports finite. Then let the chain run.

---

## 3. ⭐ Output format change — canonical `'global'` (the one real new work item)

The rerun's monthly zarrs are **canonical global-node-order** (`fesom_jax/canonical_redist.py` /
`write_global_zarr`), NOT the old folded `[P*Lmax]` ushow layout. Canonical = **partition-independent**
(byte-identical at any device count) and, crucially, **node fields are in FESOM global-node (gid) order —
the SAME order as `fesom.mesh.diag.nc` and the Fortran output.**

**Implication for the paper pipeline (`paper_jax/scripts/`):** the reducers currently read the folded
layout via `common.ushow_to_nodes` + `ushow_lane_map` (the round-6 lon/lat coordinate-match, a documented
fragility — see [[paper-phase3-fidelity]]). For canonical zarrs that machinery is **wrong AND
unnecessary** — read them **directly** like the Fortran (`common.fortran_iter_months` reads native node
order):

- **TODO (do while the rerun runs):** add a canonical reader (e.g. `common.jax_canonical_months(root, …)`
  mirroring `fortran_iter_months`: open each `monthly/<YYYY>_<MM>` zarr, take `temp/salt/ssh/u/v/…`
  node-indexed arrays directly, `[nz, nod2]`/`[nod2]`, pad 3-D to 48 levels). Point
  `reduce/reduce_drift.py` (`:43`) and `reduce/reduce_meanstate.py` (`:45`) at it instead of
  `ushow_to_nodes`. The area weights (`mesh.area2d`/`area3d` from the diag) and the whole
  vol/area-weighted-mean machinery are unchanged — the JAX arrays now line up node-for-node with the
  mesh AND with Fortran, so **the `ushow_to_nodes` coordinate match is eliminated** (a robustness win).
- **First sanity check on the canary output:** open one canonical `monthly/1958_01` zarr and assert its
  `lon`/`lat` (if present) or its node ordering matches `fesom.mesh.diag.nc` node-for-node (and matches
  the Fortran `ssh.fesom.1958.nc` order). If canonical order == diag order (expected — both FESOM global
  order), the direct read is correct. If it does NOT (surprise), fall back to `--output-layout folded` in
  the sbatch to reuse the existing ushow pipeline unchanged, and defer the canonical migration.
- **Attr change:** `n_samples` in each zarr is now the **step count** of the period (~a month × steps/day),
  not the ~30 chunk-snapshots — it's just metadata; the reducers use the mean values, not `n_samples`.

**Escape hatch:** if the canonical migration is more than you want to take on for this rerun, add
`--output-layout folded` to the sbatch srun line — you get the SAME two physics/output fixes with the
old folded zarrs the existing reducers already read. But the user explicitly wants canonical, so prefer
the direct-read migration.

---

## 4. Post-run verification (both fixes) + figure regeneration

Once the chain prints `CORE2_HINDCAST_DONE` (or after enough years to see the trends):

**4.1 Salinity fix — the ⟨SSH⟩/⟨S⟩ trends must be FLAT.** Repeat the master diagnostic (area-weighted
global-mean SSH + vol-mean S vs the SAME sampling): the pre-fix run drifted **⟨SSH⟩ −0.44 m / vol-mean S
+0.0044 psu / 63 yr**; the fixed run should be **flat** (matching Fortran). A scratch script pattern:
area-weighted `ssh` from the canonical monthly zarr + Fortran `ssh.fesom.YYYY.nc`
(`common.fortran_iter_months`), + `vol_weighted_mean(salt)`. (The old script is in this session's
scratch; the SSH-drift recipe is in `HANDOFF-20260701-salinity-drift-investigation.md` §RESOLVED.)

**4.2 SST k=1 fix — the wavenumber-1 harmonic must VANISH.** Regenerate `paper_jax/data/meanstate.nc`
(the canonical-reader `reduce_meanstate.py`) and re-run `scripts/debug/diag_sst_rotation.py` /
`diag_sst_pattern.py`: the pre-fix R²(k=1)=0.88 / amplitude 0.10 °C should drop to noise (Fig 2's
JAX−Fortran column goes ~white). If a residual k=1 remains, it's a REAL (small) SST difference to
investigate — but the diurnal alias should be gone.

**4.3 Regenerate all paper data/figures** off the new run (canonical reader):
`paper_jax/data/{drift.nc, meanstate.nc}` via `reduce/reduce_drift.py` + `reduce/reduce_meanstate.py`;
then the figures (`fig_drift.py`, `fig_meanstate.py`, …). Drop the salinity-drift "one caveat" text and
the k=1 "coordinate/wind-rotation" framing — both are now fixed. Update the model-paper narrative:
JAX↔Fortran now agree on SST **and** salinity.

---

## 5. Gotchas / risks

- **Cold start clears $OUT only when `CUR=0`** (sbatch `:87-88` `rm -rf $OUT`). A NEW `TAG` gives a fresh
  $OUT ⇒ clean cold start AND preserves the old `core2_hindcast` run. Do NOT reuse the old TAG unless you
  intend to wipe it.
- **The fixes are LOCAL commits (not pushed).** The rerun uses the local repo, so it's fine — but if the
  separate session pulls/clones, ensure `5a61bb0` + `6977c62` are present (`git log --oneline -3`).
- **Multi-GPU non-determinism** (roundoff, atomic scatter-adds) is unchanged and fine for a climatology —
  the salinity/SST fixes are structural, far above roundoff.
- **The SST fix is output-only** — if you ever want to check it WITHOUT a full rerun, it only changes the
  monthly means; the model trajectory (hence the restart) is byte-identical to pre-fix. But the salinity
  fix DOES change the trajectory, so the full rerun is needed regardless.
- **`_MeanStream` keys a chunk by its FIRST step** (not the final step, which is the next period's
  boundary) — verified, but if you touch the chunk/period logic, keep that (else a boundary-ending chunk
  is misattributed one period late).
- **Verified before shipping** (single-node CPU, this session): `sample_fn=None` byte-identical
  (`test_forcing_sharded` 4 pass); the accumulation is an exact per-step sum + does not perturb the State
  at npes=1 AND npes=2 (real dist_2); `_period_boundaries` is dt-general (48 steps/day at dt=1800, 480 at
  dt=180). The multi-GPU (dist_4) path is the same machinery — the canary is the end-to-end confirmation.

---

## 6. Inventory

- **Driver:** `scripts/runs/run_core2_hindcast.sbatch` (self-chaining; TAG/PART/PERJOB/TOTAL env-tunable).
- **Config:** `configs/core2_full.yaml` (all-on: zstar+TKE+mEVP+GM, dt=1800, dist_4).
- **Mesh/IC/partition:** `data/mesh_core2`, `data/ic_core2` (PHC cold start),
  `/pool/data/AWICM/FESOM2/MESHES_FESOM2.1/core2/dist_4`.
- **Old R1 (folded, buggy — keep for comparison):** `/work/ab0995/a270088/port_jax/runs/core2_hindcast/`.
- **Fortran R2 (UNCHANGED reference):** `/work/ab0995/a270088/fesom2_core2/` (`{var}.fesom.YYYY.nc`).
- **Paper pipeline:** `paper_jax/scripts/{common.py, reduce/reduce_drift.py, reduce/reduce_meanstate.py,
  fig_drift.py, fig_meanstate.py, diag_sst_*.py}` (nereus env
  `/work/ab0995/a270088/mambaforge/envs/nereus/bin/python`).
- **Fixes:** `fesom_jax/ice_thermo.py` (`5a61bb0`), `fesom_jax/run.py` + `fesom_jax/integrate_sharded.py`
  (`6977c62`).
- **Memory:** [[model-paper-plan]], [[salinity-drift-investigation]] (RESOLVED),
  [[sst-pattern-investigation]] (RESOLVED), [[paper-phase3-fidelity]] (the ushow_to_nodes fragility this
  canonical rerun removes).

**One-line summary:** cold-rerun R1 under a new TAG with today's salinity + SST-output fixes and the new
default canonical output; add a direct canonical reader to the paper reducers (dropping `ushow_to_nodes`),
regenerate `drift.nc`/`meanstate.nc`/figures, and confirm ⟨SSH⟩/⟨S⟩ are flat and the k=1 SST harmonic is
gone — JAX now on-par with Fortran on both temperature and salinity.
