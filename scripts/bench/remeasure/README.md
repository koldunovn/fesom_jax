# Paper RE-MEASURE envelope (M7-levers session 3, 2026-07-16)

Protocol — uniform across every point, NEVER mix in a ratio:
- Code: `perf/kokkos-m7-levers` head (fusions IN; `forcing.on_device` OFF default;
  CGPOLY OFF default — protocol-consistent with the 63-yr production figures).
- `--full 1` = model `FULL(ice+kpp+gm+JRA1958)` — the reducer's `is_full` filter and the
  figure's model pin ONLY match this string (a `--mevp 1` row is silently excluded!).
- `--steps 150` (past the CG spin-up ~step 30; comparable with the Kokkos 150-step protocol),
  ×2 reps per point, `--constraint=a100_80`.
- Transport per point = the banked winner (20260714 §5 + Phase 8d):
  padded for core2 2/4/8 + farc 4/8/16 + dars 8; ragged for dars 16/32/64 + forca20 16/32;
  coloured for ng5 32/64. ngpu=1 is transport-exempt (zero halo).
- dars: dt=120 EVERYWHERE (the dist_32 dt-180 blowup is partition-specific but 150-step
  horizons at 180 are unverified; one uniform caption note; SYPD uses DT_PROD=120 anyway).
- **ALWAYS read the bench-finite line before banking a row** (a NaN ocean times a fake step).
- Logs land in `scripts/logs/bench_rm_*.%j.out` = inside the paper reducer's glob.
  ⚠️ BEFORE running `make data` in paper_jax: ARCHIVE (move, never delete) the 77 old
  `scripts/logs/bench_*.out` files (old contaminated protocol) to
  `scripts/logs/pre_m7_protocol/`, or best-per-point selection will quote stale rows.
  In the same paper pass: drop the `halo=ragged` pin in fig_scaling/make_numbers
  (transport-envelope decision) and re-scope the "JAX ≈ Kokkos on GPU" claim.
- dars-128 (32 nodes) intentionally NOT scripted — optional point, decide by queue reality.
