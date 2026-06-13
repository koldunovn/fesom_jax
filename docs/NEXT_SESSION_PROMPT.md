# Next session — Phase 9b: TKE — resume at JT.5 (stability/climate/sharded) + the forcing-gap fix

**JT.0 → JT.4 DONE + committed 2026-06-13** (`d023f75` seam → `b67b3b6` column core →
`60ed77d` driver → `776f3be` step wiring → `da670fa` gradient gates). Column core `cvmix_tke.py`
AND driver `tke.py mixing_tke` are **replay-gated BIT-EXACT** (≤3e-17); TKE is live in the `step.py`
3-way dispatch; `test_tke_step.py` confirms the step runs eager+jit. **JT.4 `TKE_GRAD_GATE_OK`**
(`scripts/core2_tke_grad_gate.py`, A100): FD↔AD plateau on `tke_c_k` (**8.2e-8**) + `tke_cd`
(7.8e-9), masked-NaN `d(SST)/d(T0)` clean, `d(SST)/d(tke-IC)` finite — the ML seam is fully
differentiable. Suites green (OCEAN 559+ / now 563). Plan has JT.0–JT.4 ticked.

**Resume at JT.5** — stability + climate + sharded: 10-day A100 linfs+TKE (KPP swapped out) stable +
physical Kv; year-scale JAX-TKE ↔ the `c_tke_2yr` climate oracle (use **`ic_core2_dist864`** — that
oracle is 864-rank, the per-oracle IC-provenance rule [[zstar-forcing-dump-config-gap]]) ≪ the
TKE↔KPP scheme contrast (the C measured 11–18×); **sharded N-vs-1 (CPU ×4) TKE-ON** — the generic
field loops cover `tke`, and this is what STRESSES the internal node-`tke_Av` exchange in
`_wire_kv_av` (the plan-review MAJOR — omit the exch and N-vs-1 fails on boundary-element Av). Then
**JT.6** close-out (GATE 9b table, move plan to completed/).

**⚠️ ONE TRACKED GAP (do this first or alongside JT.5):** the cdump-matching LIVE forward gate is
**xfail** (`test_tke_step.py::_FORCING_GAP`). The JAX `build_core_forcing` step-1 wind stress at
**dt=1800** differs from the cdump's C-run by ~7e-4 (≈60% of scale), **IC-independent** (identical
under `ic_core2` vs `ic_core2_dist16`) ⇒ a forcing step-1 **time/convention** mismatch, NOT a TKE bug
(the port is bit-exact via the JT.1/JT.2 replay gates; KPP's forcing matched <1e-12 at dt=500 with a
forcing-MATCHED dump). Fix = align `core2_forcing` dt=1800 step-1 forcing time to the C, OR regenerate
a forcing-matched TKE step dump (the KPP-dump pattern). Until then the LIVE forward validation is
open; the bit-exact replay gates carry the port's correctness.

**Resume at JT.4** — gradient gates (the ML seam, GATE 9b core): FD↔AD plateau on `d(loss)/d(tke_c_k)`
+ `d(loss)/d(tke_cd)` (active-region loss; the `test_gradient.py:89-121` pattern); masked-NaN
`d(SST)/d(T0)` finite + 0 on masked lanes TKE-ON; `d(loss)/d(tke-IC)` through the N-step checkpointed
scan (the carry path). The COLUMN-CORE AD is already verified (`test_column_grad_finite`: grad finite,
0 on dry lanes, grad thru a `Params` leaf) + a driver-level grad (`test_mixing_tke_composition_and_grad`).
Then **JT.5** stability/climate/sharded (10-day A100 linfs+TKE vs `c_tke_2yr` using `ic_core2_dist864`;
sharded N-vs-1 stresses the internal Av exch), **JT.6** close-out. Lessons banked: backward min-scan
off-by-one, stale `(float)6.6` cdump (regenerated; stale=`cdump/dump_stale_6.6f`), the JT.2 redundant-tag
+ JT.3 forcing-gap findings.

---
## (original Phase 9b kickoff context — still the source of truth for the ladder)

TKE is the project's **primary hybrid-ML seam** — a prognostic
1-equation closure whose constants (`c_k`, `c_eps`, `cd`, `alpha_tke`) are exactly what Phase 7
tunes/NN-replaces, so `Params`-exposure + differentiability are first-class here.

## Where we are (2026-06-13)

**Phase 9a (zstar) COMPLETE — GATE 9a met, committed, plan in `docs/plans/completed/`.** Suite
OCEAN 529 + ICE 47, 0 fail; zstar behind `ale_cfg=None` (linfs byte-identical). **Three 9a
learnings carry straight into TKE:**

1. **Controlled-replay is the primary gate.** Feed a kernel the C dump's OWN inputs and compare
   outputs — this isolates kernel fidelity (byte-faithful on CPU) from the chained-trajectory FP
   butterfly. In 9a the SSH solve *looked* like it diverged ~mm multi-step; controlled-replay
   proved it byte-identical (7e-16). **The TKE `cdump` carries the 5 column inputs AND all outputs
   per step ⇒ replay is THE gate (plan §3) from JT.1.** (`test_jz7_ssh_solve_controlled_replay` is
   the template.)
2. **IC-partition provenance is PER-ORACLE** (`[[zstar-forcing-dump-config-gap]]`). The C land-fill
   IC depends on the MPI partition. **For TKE the two ICs already exist:** the `cdump` replay oracle
   is **16r ⇒ `data/ic_core2_dist16`**; the climate oracle `c_tke_2yr` is **864r ⇒
   `data/ic_core2_dist864`** (both built this session). Use the matching one per gate — a
   wrong-partition IC inflated the 9a climate SSS 41× before I caught it.
3. **Byte-faithful kernels + chained-trajectory chaos is normal** — validate multi-step by climate
   RMS, not bit-identity. The C's TKE s2/s3 *live* diffs were threshold-flips at the PHC noise floor
   (44 854 columns) — replay sidesteps them; don't chase flips (C lesson).

## The task — JT.0 first (scaffolding, NO behavior change), plan §JT.0

- **`TkeConfig(NamedTuple)`** (static/hashable) — raises on un-ported combos (Dirichlet BCs /
  `mxl_choice≠2` / IDEMIX / Langmuir; the C `fesom_tke_alloc:247-253` abort parity). Variant = the
  **classical Gaspar (1990) only**, `tke_mxl_choice=2`, `only_tke`, **Neumann** surface+bottom BCs.
- **3-way mixing dispatch** at `step.py:177-190`: `tke_cfg` ⇒ TKE; **raise if `kpp_cfg` AND
  `tke_cfg` both set** (the C runs ONE scheme/process — fail loudly); raise on the pi path (needs
  `stress_node_surf`, the KPP precedent). Thread `tke_cfg` through step/integrate/sharded +
  static_argnames (the `kpp_cfg` pattern).
- **`State.tke [nod2D, nl]`** (interface-indexed) joins State + the scan carry — the STRUCTURAL
  difference from KPP/PP/GM (all stateless). Walk the **9-point checklist** (plan §2): state.py decl
  (+ rest/zeros: **tke IC = 0**, cold start; step-1 floors wet column to `tke_min`, Kv/Av≈0 — exact
  C mirror); step.py conditional `dataclasses.replace(..., tke=...)` (ice precedent — None path never
  touches it); integrate.py + integrate_sharded.py thread `tke_cfg`; shard_mesh generic (no change);
  `zarr_output.DEFAULT_FIELDS += "tke"`; the halo TWO FACTS (below); ICs inherit 0;
  **`test_state.py:_expected_shapes` tripwire += `"tke": (nod2D, nl)`**.
- **`Params` += `tke_c_k=0.1, tke_c_eps=0.7, tke_cd=3.75, tke_alpha=30.0`** (the `k_gm`
  `default_factory` pattern — trainable from day one). ⚠️ ALSO update `register_dataclass`
  data_fields (`params.py:76`) AND `Params.defaults()` (`:67-72`). The Pr-law `6.6` stays static
  (`TKE_C66`, a DOUBLE).
- `TKE_TAGS` (20 tags) on the shared `io_dump.read_gid_table` (generalized in 9a's JZ.0); audit
  `/work/ab0995/a270088/port/tke/{cdump,replay}`.
- Tests: `tke_cfg=None` ⇒ bit-identical step; reader round-trip; `pytest.raises` on the un-ported
  `TkeConfig` combos; suite green.

Then **JT.1 column core `cvmix_tke.py` (replay-gated, 13 core tags ≤1e-13) → JT.2 driver `tke.py` +
the Av exchange (kv/av wired tags) → JT.3 step wiring + assembled live gate → JT.4 gradient gates
(the ML seam) → JT.5 stability/climate/sharded → JT.6 close-out.**

## Top landmines (plan §1/§2/§4)

- ⚠️ **All Fortran default-real literals are DOUBLE** (`-r8` build): `6.6, 0.5, √2, …` — the ONE
  real bug the C port had, caught by replay. Port every literal as float64.
- ⚠️ **`tke_cd=3.75`** (namelist beats module default 1.0); despite the Fortran comment
  "3.75=Dirichlet", the **Neumann** BC branch executes in this config.
- ⚠️ **The internal node-`tke_Av` halo exchange** (plan-review MAJOR): the driver exchanges node
  `tke_Av` (`fesom_tke.c:491`) BEFORE the node→elem 3-vertex Av mean — boundary OWNED elements have
  HALO vertices (the exact `kpp.py:787-789` idiom). So **`mixing_tke(..., exch=_exch)` is REQUIRED**;
  omit it and the port passes eager/1-device but FAILS the sharded N-vs-1 on boundary-element Av.
  Separately: **the `tke` FIELD is never exchanged** (each column self-contained on owned data) —
  document a comment row near `halo_points.OCEAN_SCHEDULE`, do NOT add an exchange.
- **mxl min-scans** (Blanke–Delecluse): two directional min-scans (`lax.scan`/assoc cummin) — pad
  dry/below-bottom levels with +∞-like nominal BEFORE the scan, mask after (else the running min is
  contaminated). `alpha_tke` does NOT appear in mxl (only the TKE-diffusivity multiplier).
- `bvfreq2` zeroed at surface + bottom interfaces (nonzero only `[nzmin+1, nzmax−1]`,
  `fesom_tke.c:362-364`) — a naive slice leaks a nonzero surface value.
- `dz_trr` interface spacings with **hnode/2 end caps**; ALL geometry via the **§2 seam inputs**
  (derived-from-hnode at the step.py call site) ⇒ zstar+TKE is free (linfs=static=today; zstar
  ⇒ `live_geometry`, which 9a now provides).
- **Budget closure** `Ttot ≈ Σ(7 terms)` ≤1e-14 rel — a FREE internal oracle; a standing test.
- AD: `sqrttke = √max(0,tke)` safe-sqrt; `|stress|` safe-norm; the `tke_min` floor kills gradients
  in the quiescent ocean (keep the exact clamp — pick gradient-gate losses in ACTIVE regions, e.g.
  mixed-layer Kv or SST). Tridiagonal = the `tracer_diff` Thomas-scan class (padded rows = identity).

## Sources + conventions

- **C source of truth:** `port2/fesom2_port_zstar` (branch `mevp`, tag `tke-validated-2026-06-11`),
  `src/fesom_cvmix_tke.c` (404-LOC pure column core) + `src/fesom_tke.c` (534-LOC driver). C plan +
  26 lessons: `port2/.../docs/plans/completed/20260610-tke-vertical-mixing.md`. C validated vs
  Fortran at SST/SSS RMS 0.0049/0.0028 (yr 1), 11–18× scheme contrast to KPP.
- **Oracles** `/work/ab0995/a270088/port/tke/`: `cdump` (20 tags, 16r, 3 steps, dt=1800, linfs —
  the replay gate), `replay` (Fortran column-input set), `c_tke_2yr` (864r linfs+TKE monthly — the
  climate gate, use `ic_core2_dist864`), `fortran_linfs_tke`, + combined `c_zstar_tke_2yr/5yr`,
  `fortran_zstar_tke` (if you also exercise zstar+TKE in JT.5).
- Oracle-first: replay/dump-gate each kernel WITH the kernel; full suite green before the next task;
  `tke_cfg=None` byte-identity throughout. **Append one PORTING_LESSONS entry per task.**
- Compute: `scripts/run_suite.sbatch` (CPU, `JAX_PLATFORMS=cpu`); A100 (`-A ab0995_gpu`, `-p gpu
  --gres=gpu:1`) for stability/gradients/climate (~0.12 s/step, a 1-yr run ≈ 35 min); C dump regen
  `-p compute --time=30:00`. Env python `/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python`.
  Login node = edit/grep only (heavy tests via sbatch). GATE 9b table is the acceptance bar.
- Everything is committed on `main` (Phase 9a). First action: `git log --oneline -3` to confirm,
  then start JT.0.
