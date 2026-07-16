# On-device JRA55 forcing interpolation — design sketch (item 4 of the M7-transfer ladder)

*2026-07-15, branch `perf/kokkos-m7-levers`. Status: PLAN ONLY — items 1–3 (clamp fix, trig cache,
a100_80 pins, EVP/CG collective fusion) are implemented on this branch; this is the next, bigger
lever. Deferred once already (`FORCING_INTERP_75PCT_NETCDF_THREAD_WALL` in PORTING_LESSONS) because
R1 ran fine in the background; the port_kokkos M7 campaign upgrades its priority with measured
evidence: their D.1 (forcing→device) plus L84 — host levers HOLD their value in the comm-bound
regime while GPU-kernel levers decay, and per-rank serial host work is structurally amplified on
GPU configs (few ranks/node, big shards).*

## Why (measured, ours)

- CORE2 hindcast after the reuse fix: **125 ms/step = 92 ms GPU + ~30 ms host forcing** (~24 % of
  the step). Split: `forcing.stack` (host bilinear interp per step) 75 %, `partition_step_forcing`
  (scatter + device_put) 25 %. NG5 measured 84/16.
- The two in-loop hiding attempts FAILED (device consumes the forcing → it is on the critical
  path; prefetch backpressures the compute stream) and threading hit the netCDF4/HDF5
  thread-unsafety wall. The clean win is to move the *interpolation* on device and shrink the
  transfer, not to hide the host work.

## What moves where

Today (per chunk): host loops the reader `step()` per model step → 8 fields × n_steps × [n_local]
stacked on host → one big `device_put` → the jitted chunk consumes `seq[i]`.

Target: the reader's per-step work — `rdate·coef_a + coef_b` (time interp), the wind `g2r`
rotation, and unit conversions — becomes device code inside the chunk `scan`; the host only
supplies the **bracketing coefficient fields** when a bracket rolls.

1. **Device-resident constants (once per run):** the 4-corner stencil is already applied on host;
   instead ship `idx4 [n,4]`, `dx4·dy4/denom` collapsed to `w4 [n,4]`… — NO. ⚠️ Keep the host
   `_gather` exactly as is (bit-exactness note in `test_jra55.py`: folding `1/denom` into weights
   breaks the C cancellation). Ship instead the **mesh-interpolated slice pair** `d1, d2 [n_local]`
   per field (computed on host by the existing `_gather`, unchanged → bit-identical inputs), i.e.
   `coef_a, coef_b [8, n_local]` per bracket. The rotation trig table `_g2r_trig` (already cached,
   this branch) and `M` go up once at init.
2. **Per-chunk host work collapses** from `8 × n_steps × _gather + n_steps × rotation` to
   `_getcoeffld` only at bracket rolls (3-hourly: once per 60 steps at dt=180) — the host cost and
   the H2D volume drop ~n_steps-fold (60-step chunk: 8×60×[n] → 8×2×[n] per roll).
3. **In-scan device code:** `field_i = rdate_i · coef_a + coef_b` (rdate per step is a scan input),
   then `_vector_g2r` on device (same expression order, trig from the table), Tair−273.15,
   prec/1000. All elementwise — negligible GPU cost, fuses into the step.
4. **Bracket rolls inside a chunk:** a chunk may cross bracket boundaries (60-step chunk at dt=180
   crosses one 3-hourly boundary; prra/prsn roll daily). Options:
   a. **Chunk-aligned coef sequence (preferred, simple):** host precomputes per-STEP
      `(coef_a, coef_b)` INDICES into a small per-chunk bracket table `[n_brackets, 8, n_local]`
      (n_brackets ≤ 3 per chunk) — the scan gathers its bracket by index. Static shapes, no
      in-scan host callback, still ~30× less H2D than today.
   b. Split chunks at bracket boundaries — simpler still, but perturbs chunk sizes (recompile
      risk via n_steps changes — the FESOM_REUSE_EXE cache keys on n_steps).
5. **Fidelity gate:** the C-dump tests (`test_jra55.py`, `test_forcing.py`) stay on the host
   reader; add a device-vs-host equality test (same `coef_a/b` in, device combine vs host combine —
   expected bit-identical: same expression order, elementwise). The step-level gate is the existing
   CORE2 dense-step dump compare; sharded via `test_forcing_sharded.py`.
6. **Sharded path:** `coef_a/b` are per-shard (`_SubMesh` readers already shard the gather —
   unchanged); `partition_step_forcing`'s scatter shrinks by the same ~30×.

## Non-goals

- No change to the bilinear stencil or `_gather` arithmetic (bit-exactness).
- No attempt to read netCDF on device or in threads (the HDF5 wall stands).
- The SSS/runoff/chl monthly climatologies stay host-side (they roll monthly — negligible).

## Expected payoff

CORE2 hindcast: ~−20 % step time (30 → ~5 ms host+transfer). FORCA20/dars production chains:
similar share (single-process, full-mesh reader — the L84-amplified case). NG5 multi-GPU: removes
the 84 %-interp host component from chunk assembly; at 64 GPUs the shards are small, so the
absolute win is smaller but stays on the critical path (host lever ⇒ holds at scale, per M7 L84).

## Order of work

1. Refactor `JRA55Reader.step` to expose `(coef_a, coef_b, rdate)` per bracket (pure host, tested
   vs the existing step()).
2. Device combine + rotation kernel; equality test vs host (bit).
3. Chunk driver: bracket table + per-step index; wire into `integrate_sharded` behind a config
   flag (`forcing_on_device`, default OFF ⇒ byte-identical).
4. CORE2 dense gate → sharded gate → A/B the CORE2 hindcast config (same-allocation, ≥150 steps,
   a100_80) → flip the default if green.
   *(1–4 DONE in session 2 — −12 % CORE2 all-on, 1-yr climate cert PASS; default stays OFF by
   user decision — VALUE-equivalent only, see PORTING_LESSONS `ONDEVICE_FORCING_XLA_CPU_DIVIDE`.)*
5. **NG5 `--local-forcing` increment (DONE 2026-07-16):** `LocalForcing.stack_tables_partitioned`
   + `forcing_const_partitioned` build the bracket tables/trig on the LOCAL sub-mesh (the same
   ~16× interpolation saving as `stack_partitioned` — `bracket_schedule`'s `_gather` is per-node
   ⇒ sub-mesh rows bit-identical to the global tables' local shards) and scatter `[P, …, Lmax]`;
   `run.py` routes `on_device` + `local_forcing` through them instead of raising. Gates: byte-id
   pytest (`test_local_forcing_tables_equal_global_partition`) + driver smoke
   (`ondevice_local_forcing_smoke.sbatch`: local-tables ≡ global-tables restarts BIT-identical,
   host-vs-tables within the FMA-seed budget).
