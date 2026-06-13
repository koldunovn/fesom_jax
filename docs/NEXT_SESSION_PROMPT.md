# Next session — Phase 9b (TKE) is COMPLETE; one OPEN forcing puzzle to diagnose

**Phase 9b (CVMix classical-TKE) DONE — GATE 9b MET 2026-06-13.** Plan in
`docs/plans/completed/20260611-fesom-jax-tke.md`. TKE is the project's first FULLY-DIFFERENTIABLE
prognostic mixing scheme, behind `tke_cfg` (`tke_cfg=None` byte-identical). 16+ commits
(`d023f75` → `517e112`). Gates green:

- Column core `cvmix_tke.py` + driver `tke.py` — controlled-replay **BIT-EXACT ≤3e-17**
- **`TKE_GRAD_GATE_OK`** — FD↔AD `tke_c_k` 8.2e-8 / `tke_cd` 7.8e-9, masked-NaN clean, tke-IC finite
- Sharded N-vs-1 (the internal `tke_Av` exch), stable, budget closure ≤4e-19
- **Year-scale climate** JAX-TKE ↔ `c_tke_2yr` **SST 4.68e-3 / SSS 2.74e-3 ≈ the C↔Fortran floor**
  (0.0049/0.0028), ≪ the TKE↔KPP 0.43 °C contrast (`TKE_CLIMATE_OK`)

## ⚠️ ONE OPEN PUZZLE (not a blocker, but UNDIAGNOSED — don't relabel it)

`test_tke_step.py::_FORCING_GAP` is xfail. The JAX `build_core_forcing` step-1 wind stress differs
from the TKE cdump by ~7e-4 at ~10% of OPEN-WATER LOW-WIND nodes (~90% match exactly). I twice
mischaracterized this (first "gustiness", then "transient") and was twice wrong — **it is a
PERSISTENT per-node difference, root cause STILL OPEN.** What's RULED OUT: time/date, IC, ice;
gustiness (no such term in `fesom_bulk.c`); the `BULK_U10MIN=0.3` floor (the JAX ports it faithfully,
`forcing.py:151,195`); a C-bulk change (`fesom_bulk.c` UNCHANGED across the KPP & TKE C builds per
`git log`). The JAX matches the **KPP-branch** dumps <1e-12 but not the **TKE cdump** — so it is NOT
the formula; it must be the **INPUTS** the C fed its bulk when generating the cdump (the JRA55 wind
field, or the node's surface state).

**Climate impact is SMALL** (low wind ⇒ tiny absolute stress ⇒ the 1-yr climate matches at the
C↔Fortran floor), so TKE is climate-faithful and the gates legitimately pass — but a year-scale match
proves *small impact*, NOT *no difference*. **To diagnose definitively:** a C-side instrumented re-run
that dumps the bulk INPUTS (`u_wind/v_wind/T_oc/u_w`) **and** the computed `cd` at the max-Δ node
(node 56098, 1958) — compare to the JAX `forcing.ncar_ocean_fluxes_mode` there. That pins down whether
it's the wind data or the surface state. Until then: log it OPEN, do not dress it in a tidy word.

## Optional follow-ups

- Diagnose the forcing puzzle above (the right next step if anyone cares about the live-step-1 gate).
- The dt=1800 stability re-run for a clean "10-day" label (the smoke ran 480 steps at dt=500 = 2.78 d).
- Phase 9c (mEVP, `docs/plans/20260611-fesom-jax-mevp.md`) is the remaining Phase-9 sub-plan.

## Provenance + conventions (carry-over)

- Oracles on `/work/ab0995/a270088/port/tke/`: `cdump` (regenerated — the original was stale with the
  `(float)6.6` bug; stale preserved as `cdump/dump_stale_6.6f`), `c_tke_2yr` (864r, use
  `data/ic_core2_dist864`), `fortran_linfs_tke`. IC-partition provenance is per-oracle
  ([[zstar-forcing-dump-config-gap]]): replay/cdump = 16r `ic_core2_dist16`; climate = 864r `dist864`.
- Env python `/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python`; suite via
  `scripts/run_suite.sbatch` (CPU); A100 (`-A ab0995_gpu -p gpu --gres=gpu:1`) for climate/gradients.
- 10 porting lessons banked (`docs/PORTING_LESSONS.md`), incl. two real bugs (the backward min-scan
  off-by-one; the stale `(float)6.6` oracle) and the two-strikes forcing-gap mischaracterization.
