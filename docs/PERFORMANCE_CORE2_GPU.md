# CORE2 GPU speed ledger

Measurements for `docs/plans/20261020-core2-gpu-speed.md`. One row per measured configuration; nothing is entered
without a job id or a cited source.

## Protocol

- **Steady ms/step**: compile excluded, the executable reused, a long chunk, wall clock with one wait at the end
  (`scripts/perf/gpu_profile.py time`, Task 1).
- **Production-loop ms/step**: `scripts/run_from_config.py --progress`, read from chunk 3 on (two executables
  compile; chunk 2 is inflated).
- A100 jobs pinned to `--constraint=a100_80`; at least 2 repeats inside one allocation; only runs whose final
  State is finite count.
- SYPD at the dt actually run: SYPD = 86400 / (steps per year × s/step); 17,520 steps/yr at dt 1800, 11,680 at
  dt 2700. 1000 SYPD = 4.93 ms/step at dt 1800, 7.40 ms/step at dt 2700.
- Jobs run from the committed HEAD of `wt_perf` through the import-guarded wrappers (plan, Development Approach).
- GPU jobs run from a frozen snapshot worktree of that commit (`wt_perf_<sha>`), so a later commit cannot change a
  running job's code.
- **Pair test for round-off-only changes (`scripts/perf/noise_ab.py`, 2026-10-20).** The single-draw judge fails
  a program that only reorders sums when the noise holds rare discrete events (the mEVP wide halo on 4 A100: 5 of
  7 draws above 2x, one at 10x at a single Hudson Bay node under ice). With 4 runs of each program, compare the
  cross pairs with the within pairs per field on owned lanes: round-off only ⇒ median ratio ~1 and few cross
  pairs above the within maximum (wide halo on GH200: 1.30 and ≤ 2/16, job 28290759).
- **Noise judge, calibrated 2026-10-20 (job 28286319).** The floor is the largest relative difference per field
  over all pairs of 4 repeat runs inside one job. A same-code run from ANOTHER job lands above that floor in every
  leave-one-out draw: worst field 1.12-1.85x (median 1.45x) over 20 draws, and the prefetch job's single-chunk
  noise run (prefetch inactive, so the same program) sits at ~1.75x in `ssh_rhs_old` against all four base runs.
  Runs inside one job are more alike than runs in different jobs. So a single draw from another job passes at
  factor 2 (`compare --factor 2`), not 1, and a change that is round-off only should look like that A/A draw:
  no field consistently above ~1.8x against every base run.

## Baselines (before this plan)

CORE2 production physics (zstar + cvmix-TKE + mEVP + GM), dt 1800, double precision, speed options off unless
noted.

| machine | GPUs | instrument | ms/step | SYPD | source |
|---|---|---|---|---|---|
| JUPITER GH200 | 1 | kernel bench, best transport, constant forcing | 93.2 | 52.9 | `JUPITER_SCALING_FINDINGS.md` §6 |
| JUPITER GH200 | 4 | kernel bench, best transport (padded), constant forcing | 43.7 | 112.8 | `JUPITER_SCALING_FINDINGS.md` §6 |
| Levante A100 | 4 | kernel bench (fig10 v2 production campaign) | 88.0 | 56.0 | `JUPITER_SCALING_FINDINGS.md` §5 (core2/4 ratio) |
| Levante A100 | 4 | production loop, July code (pre drift fixes) | 112.5 | 43.8 | job 26278428 (on-device forcing A/B, OFF leg) |
| Levante A100 | 4 | production loop, July code, on-device forcing ON | 99.0 | 49.8 | job 26278428 (ON leg) |

dolpung GH200 gets its own baseline in Task 2 (the JUPITER rows are another machine and the kernel bench).

## fesom3-jax, for comparison

CORE2, 4 GH200 (one dolpung node), from `~/fesom3_jax/docs/PERFORMANCE.md` (read-only):

| configuration | ms/step | SYPD |
|---|---|---|
| double precision, CG, dt 1800 | 20.2 | 244 |
| the same, SSH solve gathered onto every GPU | 16.9 | 292 |
| dt 2700, split-explicit + wide halo, unrolled subcycles, double precision | 12.7 | 583 |
| + single precision | 8.5 | 867 |
| + ice every second step, Pallas SE/EVP kernels, T/S batched, chunk 192 | 7.26-7.4 | ~1000-1020 |

## Measurements

| date | commit | job | machine | GPUs | dt | config / change | steady ms/step | loop ms/step | SYPD | GPU-h | note |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-10-20 | 4a9fc43 | 28282325 | Levante A100 | 4 | 1800 | core2_full_bench, all_gather halo (production default), options OFF | 107.79 | — | 45.8 | 0.75 | kernels 79.5 ms/GPU, NCCL 39.1 (555 collectives, cost 30.6 + wait 8.5); host between chunks 28.7 not overlapped; 5036 launches, 97 syncs/step |
| 2026-10-20 | c72b0d2 | 28282453 | Levante A100 | 4 | 1800 | + --cheb-degree 3 --forcing-on-device (= core2_gpu.yaml options) | 84.38 | — | 58.4 | 0.57 | kernels 69.6 ms/GPU, NCCL 33.5 (372 collectives, cost 27.8 + wait 5.8); host between chunks 13.2 not overlapped; 3829 launches, 34 syncs/step |
| 2026-10-20 | c72b0d2 | 28283046 | Levante A100 | 4 | 1800 | options ON + --padded halo | 84.37 | — | 58.5 | 0.54 | NCCL 10.0 ms (389 coll., cost 4.3 + wait 5.7) but ops helpers 15.4 (+4.4), halo packing 4.65 (+2.9): kernels 67.1, step unchanged |
| 2026-10-20 | c72b0d2 | 28283047 | Levante A100 | 4 | 1800 | options ON + --coloured halo | 93.51 | — | 52.7 | 0.64 | NCCL 27.2 ms, 910 collectives/step (cost 9.8 + wait 17.5); kernels 88.5 |
| 2026-10-20 | c348ebd | 28283307 | Levante A100 | 4 | 1800 | options ON + --ssh-gathered (all_gather halo) | 81.09 | — | 60.8 | 0.41 | -3.9 %; 172 collectives (NCCL 27.6 ms, median 51 us: the 3-D all_gather exchanges); per-GPU busy 66.4 of 71.3 ms in the call; noise compare vs 28282453: uv 1.54x floor (one draw), 6 fields at 1.00-1.01x |
| 2026-10-20 | 1c82624 | 28284314 | Levante A100 | 4 | 1800 | options ON + forcing prefetch (A6, default on) | 73.92 | — | 66.7 | 0.50 | -12.4 % vs 28282453; host between chunks 13.21 -> 0.08 ms/step, inside the calls 73.78 (74.35 before); kernels 70.0 ms/GPU, per-GPU busy 70.0 = 95 % of the call; bitwise the serial build on CPU (test_forcing_prefetch_bitwise) |
| 2026-10-20 | 56a03aa | 28286309 | Levante A100 | 4 | 1800 | prefetch + --padded + FESOM_HALO_BARRIER=out | 77.03 | — | 64.0 | 0.38 | +4.2 % vs 28284314; NCCL 10.3 ms but the other kernels 59.3 ms (36.0 under all_gather): the build is re-laid level-major (f64[48,32073] 4610x vs 1059x, 205 transposes vs 70), so one-launch 3-D kernels slow 5-10x (impl_vert_visc 2.47 ms, diff_part_hor_redi 2.33) |
| 2026-10-20 | 56a03aa | 28286310 | Levante A100 | 4 | 1800 | prefetch + --padded + FESOM_HALO_BARRIER=both | 79.09 | — | 62.4 | 0.38 | +7.0 % |
| 2026-10-20 | 56a03aa | 28286311 | Levante A100 | 4 | 1800 | prefetch + all_gather + FESOM_HALO_BARRIER=both | 77.09 | — | 64.0 | 0.38 | +4.3 %: the barrier costs, it does not fix; it stays off |
| 2026-10-20 | bb33f0b | 28286769 | Levante A100 | 4 | 1800 | prefetch + FESOM_HALO_COMPACT=1 (all_gather of only the lanes others read) | 70.74 | — | 69.7 | 0.47 | -4.3 %; NCCL 34.0 -> 13.8 ms (cost 6.7, median 14.7 us) but the build flips level-major again (293 transposes): non-NCCL kernels 36.0 -> 49.8; noise compare within 2x floor; CPU: bitwise the all_gather step |
| 2026-10-20 | 65b528f | 28287270 | Levante A100 | 4 | 1800 | + FESOM_COLUMN_UNROLL=1 (level sweeps unrolled, no transpose) | **56.89** | — | **86.7** | 0.67 | -23 % vs 28284314; kernels 52.8 (NCCL 12.3), 2275 launches/step (3829); ops helpers 13.1, halo packing 5.9, TKE 4.9 in 17 launches (3.0 in 1075 before); noise compare within 2x floor; CPU: C-replay gates pass with it, full step at round-off (1 ulp) |
| 2026-10-20 | 65b528f | 28287271 | Levante A100 | 4 | 1800 | prefetch + FESOM_COLUMN_UNROLL=1 alone (all_gather halo) | 74.99 | — | 65.8 | 0.6 | no gain alone (73.92 without): the unroll pays only with the compact transport |
| 2026-10-20 | 31b7fe6 | 28288668 | dolpung GH200 | 4 | 1800 | core2_gpu options + compact + unroll (= core2_gpu.yaml) | **34.68** | — | **142.2** | 0.7 | first steady GH200 number of this plan (JUPITER levers-off kernel bench: 43.7); 4-run GH200 noise floor in runs/perf/28288668-gh200x4_cu/noise |
| 2026-10-20 | 31b7fe6 | 28288670 | dolpung GH200 | 4 | 1800 | core2_gpu options without compact/unroll (prefetch on) | 40.20 | — | 122.7 | 0.1 | the GH200 reference for A5b: compact + unroll is -13.7 % here (-23 % on A100) |
| 2026-10-20 | 31b7fe6 | 28288669 | dolpung GH200 | 4 | 1800 | core2_gpu.yaml + --ssh-gathered | **31.90** | — | **154.6** | 0.26 | -8.0 % vs 28288668; noise compare vs the GH200 4-run floor (factor 2): within |
| 2026-10-20 | 31b7fe6 | 28288671 | dolpung GH200 | 4 | 1800 | core2_gpu.yaml + --xla_gpu_enable_latency_hiding_scheduler=true | 34.77 | — | 141.8 | 0.16 | no gain (fesom3-jax: -1.3 % on 4 A100); not adopted |
| 2026-10-20 | 7fecd74 | 28290739 | dolpung GH200 | 4 | 1800 | core2_gpu.yaml + FESOM_MEVP_WIDE=8, 4 noise runs | 31.17 | — | 158.2 | 0.5 | -10.1 % vs 28288668; owned-lane pair test against 28288668's 4 runs (noise_ab.py, job 28290759): cross pairs distributed like within pairs (largest median ratio 1.30, at most 2/16 cross pairs above the within maximum) ⇒ round-off only |
| 2026-10-20 | 7fecd74 | 28290740 | dolpung GH200 | 4 | 1800 | + FESOM_MEVP_WIDE=12 | **30.89** | — | **159.7** | 0.15 | -10.9 % |
| 2026-10-20 | 8980bf8 | 28291187 | dolpung GH200 | 4 | 1800 | core2_gpu.yaml as of 8980bf8 (compact + unroll + mEVP wide 12 + gathered SSH) | **28.28** | — | **174.4** | 0.7 | kernels: scatters 8.3 ms (scatter_add 5.4, SSH matvec 1.3), concatenate fusions 4.0 (the compact receive), NCCL 4.4 (cost 1.0, waits 3.4) |
| 2026-10-20 | 0d10cba | 28291287 | Levante A100 | 4 | 1800 | the same config | 48.68 | — | 101.3 | 0.6 | NCCL 4.1 ms, 1568 launches/step |
| 2026-10-20 | 0d10cba | 28291288 | Levante A100 | 4 | 1800 | + FESOM_SSH_ELL=1 (SSH matvecs on the ELL layout, no atomics) | 46.67 | — | 105.7 | 0.6 | -4.1 %; round-off (FMA in the add chain), not bitwise |
| 2026-10-20 | ea4f6a4 | 28291670 | Levante A100 | 4 | 1800 | + compact receive in one elementwise pass (no concatenated copy) | 46.02 | — | 107.2 | 0.5 | -1.4 % vs 28291288; bitwise the same values (test_halo, compact step test) |
| 2026-10-20 | e97d482 | 28291789-91 | Levante A100 | 4 | 1800 | + FESOM_CG_BATCH = 1 / 4 / 8 (masked CG iterations per while trip) | 45.75 / 45.55 / 45.48 | — | 108.4 | 1.2 | -0.6 % at k = 8: the A100 host syncs are cheap; the same iterate and count (test_ssh) |
| 2026-10-20 | 84d5068 | 28291809 | Levante A100 | 4 | 1800 | + FESOM_CG_BATCH=4 + FESOM_MEVP_UNROLL=12 | 45.05 | — | 109.5 | 0.4 | -1.1 % vs CG batch 4 alone; round-off (the wide-vs-per-subcycle bitwise tests differ by ~1 ulp with the unroll on, job 28291808; they pass without it) |
| 2026-10-20 | d7947da | 28291852 | Levante A100 | 4 | 1800 | + FESOM_COLUMN_KERNEL=1 (Pallas column kernels for tdma and the TKE sweeps; with CG batch 4, no mEVP unroll) | **37.85** | — | **130.3** | 0.6 | -16.9 % vs 45.55 (CG batch 4); the O(K²) unrolled concatenates gone; syncs 34 -> 10/step; kernels 34.4 (scatter_add 8.75, SSH 5.5, mEVP 3.5, NCCL 3.6); tests: CPU interpret 33 passed, GPU 8 passed (jobs 28291850/55) |
| 2026-10-20 | d7947da | 28292125 | dolpung GH200 | 4 | 1800 | the night set: compact + unroll + mEVP wide 12 + gathered SSH + ELL + CG batch 4 + mEVP unroll 12 + column kernels | **21.29** | — | **231.6** | 0.3 | -25 % vs 28.28 (28291187); at dt 2700 this step is ~347 SYPD |
| 2026-10-20 | 8f4275f | 28292208 | Levante A100 | 4 | 1800 | the night set + FESOM_SSH_INC_ELL=1 (zstar increment as explicit entries, once per solve) | **35.98** | — | **137.1** | 0.5 | ~-3.5 % vs the night set (~37.3); launches 1178/step; solver-tolerance equivalent (test_ssh_ell, job 28292206) |
| 2026-10-21 | 49bb99a | 28293313 | dolpung GH200 | 4 | 1800 | core2_gpu.yaml set (night + explicit increment), A/B on one node: compact 1 vs compact 2 (scatter receive) | **20.37** / 20.45 | — | **242.1** | 0.5 | A B A B, reps agree to 0.05 ms; mode 2 no gain (its saved halo pass reappears in the consumers, A100 job 28292865); at dt 2700 this step is ~363 SYPD |
| 2026-10-21 | fb4c039 | 28295452 | dolpung GH200 | 4 | 1800 | core2_gpu.yaml set, A/B on one node: + FESOM_ORDERED_SUMS=1 (ordered gathers instead of atomic scatter-adds, mode 1 = barrier on the sums) | 20.39 / 20.33 vs **23.27 / 23.24** | — | 242 vs 212 | 0.5 | ordered sums are SLOWER in double precision on GH200 (+14 %), where they are -13.7 % on 4 A100 (39.6 vs 45.9, the ordered-sums branch, job 28293421): Hopper's f64 atomics are fast |
| 2026-10-21 | e6beb67 (perf/core2-sp) | 28295508 | dolpung GH200 | 4 | 1800 | the same set, one node, A B C D x2: DP / DP + ordered sums / **SP (#1207)** / **SP + ordered sums** | 20.41, 20.44 / 23.29, 23.23 / **12.90, 12.92** / **10.78, 10.69** | — | 242 / 212 / 382 / **459** | 0.6 | SP -37 % against DP; ordered sums -16 % more in SP (its WP_full tracer accumulations lose their f64 atomics too); at dt 2700 the SP + ordered-sums step is ~690 SYPD steady. SP is opt-in and NOT cleared for long runs (STOP 4: conservation verdict pending) |
| 2026-10-21 | e6beb67 | 28295800 | dolpung GH200 | 4 | 1800 | SP + ordered sums, nsys profile | 10.78 | — | 457 | 0.2 | kernels 12.41 ms/GPU under nsys: NCCL 3.66 (cost 0.59, **waiting for the slowest GPU 3.07**, the last GPU varies), ops helpers 2.09 (ordered_sum 1.07), mEVP 1.75, SSH 1.39, tracer adv 0.93; 1099 launches/step |
| 2026-10-21 | 67e22ea | 28295889 | dolpung GH200 | 4 | 1800 | SP + ordered sums, command buffers off / on (`XLA_FLAGS=` for B) | 10.81, 10.67 / **10.38, 10.33** | — | 459 / **476** | 0.3 | -3.6 % in single precision (-1 % in double, job 28292305): with the step halved the launches matter more |
| 2026-10-21 | 67e22ea | 28295867 | dolpung GH200 | 4 | **2700** | SP + ordered sums from the v3 1990 restart, chunk 32 / 96 (no output) | 12.99, 13.62 / 11.61, 11.54 | — | 555 / 640 | 0.3 | a fixed ~67 ms per chunk call INSIDE the call (32 x 12.95 = 414 ms vs 96 x 11.55 = 1109 ms): every call re-folded, re-cast and re-placed the mesh, operator and halo arrays on the host |
| 2026-10-21 | c3cea7a | 28296064 | dolpung GH200 | 4 | 2700 | the same, FESOM_PLACE_CACHE 0 / 1 at chunk 32, 1 at chunk 96 | 14.14, 12.97 / 11.55, 12.25 / 11.35, 11.13 | — | 555 / 620 / 655 | 0.4 | inside the calls 13.11, 12.91 -> **11.06, 11.07** (-15 %) -> 11.03, 10.87: the placed-constants cache removes most of the chunk-length penalty; host gaps between chunk-32 calls 0.04-1.5 ms/step (forcing prefetch) |
| 2026-10-21 | e6beb67 / c3cea7a / 3da2c0d | 28295760 / 28296470 / 28296545 | dolpung GH200 | 4 | **2700** | **production loop**, SP + ordered sums, 3 months from the v3 1990 restart, monthly output, chunk 32: no cache / cache / cache + command buffers | — | **13.5 / 11.7 / 11.7** | **540 / 630 / 630** (steady) | 0.6 | loop time of chunks 3-90 at 1 s resolution (38 / 33 / 33 s for 2816 steps); 2 compiles of ~45 s per job; a 10-year job would run ~590 SYPD. THROUGHPUT ONLY: SP awaits STOP 4 |
| 2026-10-21 | 3da2c0d | 28296788 | dolpung GH200 | 4 | **2700** | production loop, DOUBLE precision core2_gpu.yaml set + the placed-constants cache, 3 months from the v3 1990 restart, monthly output, chunk 32 | — | 21.3 | 347 (steady) | 0.2 | against 21.9 without the cache (job 28293434, one year): -3 % in double precision, where the per-chunk host cost was a smaller share |
| 2026-10-21 | 2c063da | 28296535 / 28296536 | dolpung GH200 | 4 | 1800 | SP + ordered sums, 3 noise runs of 96 steps in 3 chunks, cache off / on | 10.71 / 10.26 | — | — | 0.4 | both deterministic (floor 0 in every field) and cache on == cache off BITWISE; in the production loop the new code with and without the cache and a rerun of e6beb67 are bitwise equal too (jobs 28296723, 28296470, 28296742), only the first e6beb67 run (28295760) differs (T up to 0.86 C after 3 months): one of four same-program runs was not reproducible |
| 2026-10-21 | b031663 | 28294992 | dolpung GH200 | 4 | 1800 | core2_gpu.yaml set (= 49bb99a + the cache-key fix), nsys profile | 20.67 | — | 238.6 | 0.3 | kernels 20.45 ms/GPU: scatter_add 5.47, NCCL 3.39 (cost 1.26 + waiting 2.13), mEVP 2.23, SSH 1.72, tracer adv 1.68, momentum 1.55; GPU 0 is last at the first collectives 96/96: its atomic scatter fusions run 40-50 % longer than on GPUs 1-3 (non-NCCL kernels 18.4 vs 16.0-17.3 ms/step), see the GH200 table |
| 2026-10-21 | 49bb99a | 28293434 | dolpung GH200 | 4 | **2700** | production loop, core2_gpu.yaml (full set), 1 year from the v3 1990 restart, monthly output, chunk 32 | — | 21.9 | **338** (steady) / 259 (1-yr job) | 0.4 | 365 chunks: 2 compiles of ~36 s (the AB2-bootstrap chunk and the steady one), then 0.7 s per 32 steps; loop 334 s + setup 11 s; a 10-year job would run ~330 SYPD |
| 2026-10-21 | 49bb99a | 28293527/28 | dolpung GH200 | 4 | 2700 | the same with chunk 96, with / without monthly output | — | — | — | 0.8 | loop 360 / 350 s: chunk 96 is SLOWER here (the month splits give several chunk lengths, each a compile); monthly output costs ~10 s per model year (3 %) |
| 2026-10-20 | d7947da | 28292122/23 | Levante A100 | 4 | 1800 | the night set without / with the column kernels, 4 noise runs each | 45.41 / 40.09 | — | 123.0 | 1.2 | different nodes than 28291852 (37.85): A100 node spread ~5 %; owned-lane pair test (job 28292399): largest median ratio 1.39 ⇒ round-off only |
| 2026-10-20 | 8980bf8 | 28291189 | dolpung GH200 | 4 | **2700** | B2 continuation 1993-1999 (core2_gpu.yaml as of 8980bf8: + mEVP wide 12 + gathered SSH), monthly output, chunk 32 | — | 31.0 | 238 | 2.9 | 10 model years at dt 2700 (1990-1999) without a failure; 81792 steps in 2533 s |
| 2026-10-20 | 0ec2903 | 28288574 | dolpung GH200 | 4 | **2700** | B2: core2_gpu.yaml + dt_ramp to 2700 from the v3 1990-01-01 restart, 3 years, monthly output, chunk 32 | — | 40.0 | **185** | 1.6 | production loop incl. monthly means (1404 s for 35072 steps; job 24 min incl. compile); STABLE 3 years, see Tier B below |
| 2026-10-20 | f997148 | 28289821 | Levante A100 | 4 | 1800 | compact + unroll on perf/core2-mevp-wide, FESOM_MEVP_WIDE=0 (A/A of 28287270) | 57.11 | — | 86.4 | 0.63 | 344 collectives/step, kernels 52.57 (NCCL 12.18), 2275 launches; noise compare vs 28282453 within 2x (worst 1.00x folded, 1.21x owned-only) |
| 2026-10-20 | f997148 | 28289822 | Levante A100 | 4 | 1800 | + FESOM_MEVP_WIDE=4 (mEVP wide halo, one exchange every 4 subcycles) | 51.96 | — | 94.9 | 0.63 | -9.0 % vs 28289821; 255 collectives, kernels 49.14 (NCCL 10.12, median 21.9 us), 1827 launches; noise: OUTSIDE (owned lanes: ssh_rhs 4.2x vs every base run) |
| 2026-10-20 | f997148 | 28289823 | Levante A100 | 4 | 1800 | + FESOM_MEVP_WIDE=8 | 51.77 | — | 95.3 | 0.63 | -9.3 %; 240 collectives, kernels 49.43 (NCCL 10.57), 1752 launches; noise: OUTSIDE (owned: uv_rhs 3.3x, uv_rhsAB 3.3x, uvnode 3.1x, uv 2.7x) |
| 2026-10-20 | f997148 | 28289824 | Levante A100 | 4 | 1800 | + FESOM_MEVP_WIDE=12 | **51.22** | — | **96.3** | 0.61 | -10.3 %; 235 collectives, kernels 48.34 (NCCL 9.54), 1727 launches; noise: OUTSIDE (owned: Kv 3.5x) |
| 2026-10-20 | f997148 | 28290233 | Levante A100 | 4 | 1800 | FESOM_MEVP_WIDE=8, NSYS=0, NOISE=4 | 51.60 | — | 95.6 | 0.69 | timing repeat of 28289823 (-0.3 %); 4 noise reps: owned-lane worst vs every base run 10.0x (density, rep 0), 2.1-2.2x (hbar_old, rep 1), 1.35-1.80x (rep 2), 1.27-1.32x (rep 3); see the Tier A bullet |
| 2026-10-21 | 61e9e71 | 28292255 | Levante A100 | 4 | 1800 | core2_gpu options by env (compact + unroll + mEVP wide 12 + SSH ELL, cheb 3, on-device forcing, gathered SSH), FESOM_ORDERED_SUMS=0, NOISE=4 | 45.89 | — | 107.5 | 0.91 | the OFF reference for the ordered sums (node l50018); kernels 42.16: scatter_add 8.87 ms in 52 launches, SSH stiff increment scatter 1.06 ms in 34 |
| 2026-10-21 | 61e9e71 | 28292256 | Levante A100 | 4 | 1800 | + ordered sums without a barrier (today FESOM_ORDERED_SUMS=2), NOISE=4 | 43.19 | — | 114.2 | 1.02 | -5.9 %, same node; one 8.57 ms multi-output fusion (pp.py:68 compute_vel_nodes, ~90 gathers merged with the PP/TKE column work); 4 noise runs identical (floor 0 in every field) |
| 2026-10-21 | 4ee4506 | 28293421 | Levante A100 | 4 | 1800 | + ordered sums, barrier on the sums (= FESOM_ORDERED_SUMS=1 from b4ef89d, `gpu.ordered_sums: true`) | **39.62** | — | **124.5** | 0.61 | **-13.7 %** vs 28292255 (node l50112); kernels 36.08: ordered_sum 6.33 ms in 53 launches (incl. fused producers), SSH stiff increment 0.62, NCCL 2.32; still one 2.07 ms single-launch fusion labelled forcing.py:354 |
| 2026-10-21 | 4ee4506 | 28293422 | Levante A100 | 4 | 1800 | + ordered sums, barrier on the values and the sums (FESOM_ORDERED_SUMS=3) | 39.88 | — | 123.7 | 0.59 | -13.1 %; ordered_sum 2.58 ms (producers in their own kernels) |
| 2026-10-21 | b4ef89d | 28294896 | Levante A100 | 4 | 1800 | FESOM_ORDERED_SUMS=1, NSYS=0, NOISE=4 | 39.71 | — | 124.2 | 0.71 | timing repeat of 28293421 on the same node (+0.2 %); 4 noise runs identical (floor 0) ⇒ deterministic on the GPU; owned-lane pair test against 28292255's 4 runs: cross pairs inside OFF's own within pairs (largest median ratio 1.30, hbar_old) ⇒ round-off only |
| 2026-10-21 | b5a38fa | 28295418 | Levante A100 | 4 | 1800 | B1: ice_ave_steps 1 vs 2 (sea ice every second step, FESOM2's scheme, lax.cond), A B A B on one node; the core2_gpu.yaml env set + --cheb-degree 3 --forcing-on-device --ssh-gathered | 38.56 / **35.62** | — | 127.9 / **138.4** | 0.56 | -7.6 % (-2.94 ms/step; reps 38.65/38.46 vs 35.74/35.50): the ice update (mEVP + FCT + cut-off + thermodynamics) costs ~5.9 ms = ~15 % of the A100 step, half of it saved; no sign of a lax.cond cost, so no static schedule was built; opt-in, not in core2_gpu.yaml (STOP 3) |
| 2026-10-21 | b5a38fa | 28295449 | Levante A100 | 4 | 1800 | B1 fidelity: core2_gpu.yaml + ice_ave_steps 2 (`runs/ice2/core2_gpu_ice2.yaml`), 1990 from the v3 1990-01-01 restart, monthly output, chunk 48 | — | 34.9 | 132 (1 year in a 653 s loop incl. compile) | 0.77 | all 12 months finite; climate: see the Tier B bullet (more sea ice, SH volume +36 % by December) |
| 2026-10-21 | b5a38fa | 28295530 | Levante A100 | 4 | 1800 | the same with ice_ave_steps 1 (`core2_gpu_ice1.yaml`), Jan-Jun 1990, the same-machine reference | — | 40.2 | 109 (half a year, 394 s loop) | 0.47 | at the noise of the GH200 twin 28291192 (area-rms SST 0.005 °C in June, ice area within 0.002 x 1e12 m²); different node from 28295449 (steady ice-1 vs ice-2 is the one-node A/B 28295418) |

## Where a CORE2 step goes (4 A100, Stage 0, 2026-10-20)

`gpu_profile.py` steady window = the production loop (`run_from_config` main, 2 warm + 4 timed chunks of 96);
nsys with `--cuda-graph-trace=node` over one timed chunk; `hlo_kernel_map.py --by area`; `nccl_waits.py`.
Results in `/work/ab0995/a270088/fesom_jax_dev/runs/perf/<job>-<label>/`. Kernel ms per GPU per step:

| area | options off (28282325) | options on (28282453) |
|---|---|---|
| NCCL collectives | 39.14 (556 launches) | 33.53 (372) |
| ops helpers once per step (scatters, sums, halos) | 12.06 (663) | 10.97 (288) |
| SSH (CG solve, rhs) | 6.57 (872) | 4.18 (475) |
| sea ice dynamics (mEVP) | 3.87 (581) | 3.73 (518) |
| tracer advection | 3.55 (30) | 3.55 (30) |
| TKE | 3.11 (1138) | 2.99 (1075) |
| sharded driver | 2.53 (635) | 2.52 (635) |
| halo exchange (packing) | 2.30 (463) | 1.77 (338) |
| GM/Redi | 1.55 | 1.53 |
| momentum | 1.35 | 1.35 |
| other areas together | 3.50 | 3.49 |
| **sum of kernels** | **79.50** | **69.61** |
| steady step | 107.79 | 84.38 |
| host time between chunk calls | 28.71 | 13.21 |

Reading: half the kernel time is NCCL (genuine cost ~23 us per collective, waits small); the host builds each
chunk's forcing only after the previous chunk returns, because the chunk call blocks (the CG's while loop syncs
the host every iteration). Tracer vertical diffusion (the tridiagonal solves) is 0.29 ms. Compute without NCCL
~36 ms/step vs fesom3-jax's ~25 ms on 4 A100: the gap to fesom3-jax is mostly communication and host
serialisation. ⚠️ The nsys "busy" number in profile.json merges all GPUs' intervals into one timeline (wrong for
P > 1; fix pending); kernel sums and NCCL matching are per GPU and correct.

Noise floors (4 runs of 96 steps from the cold start, max relative difference per field over all pairs):
`runs/perf/28282325-a100x4_off/noise/noise.json`, `runs/perf/28282453-a100x4_on/noise/noise.json`.

## Where a CORE2 step goes (4 GH200, the core2_gpu.yaml set, 2026-10-21, job 28294992)

Kernel ms per GPU per step (`runs/perf/28294992-gh200x4_full_inc/`), steady step 20.67 ms, kernels 20.45:

| area | ms | launches |
|---|---|---|
| ops helpers (scatter_add 5.47 of it) | 6.67 | 131 |
| NCCL (cost 1.26, waiting for the slowest GPU 2.13) | 3.39 | 36 |
| sea ice dynamics (mEVP) | 2.23 | 274 |
| SSH (CG solve, rhs) | 1.72 | 458 |
| tracer advection | 1.68 | 23 |
| momentum (impl_vert_visc set-up fusion 0.67) | 1.55 | 19 |
| GM/Redi | 0.85 | 19 |
| halo packing | 0.59 | 66 |
| sea ice thermo, advection, coupling | 0.56 | 37 |
| everything else | 1.19 | — |

Reading: the atomic scatter-adds are a quarter of the step AND the main load imbalance. Per GPU over the profiled
chunk, the non-NCCL kernels take 1767 / 1656 / 1588 / 1540 ms on GPUs 0-3, and the difference sits almost entirely in
the `input_scatter_fusion_*` kernels (40-50 % longer on GPU 0). The padded shapes are equal on all GPUs, but a
scatter drops its pad-lane updates, so the GPU with the most real elements does the most atomics; the others wait
for it at the next collective (2.13 ms/step). Ordered sums (gathers over a padded inverse map: the same work on every
GPU) should remove both the scatter time and most of that wait.

## Tier A results so far (2026-10-20)

- **A5 halo transport (4 A100): no gain so far.** Padded ties with all_gather (84.37 vs 84.38 ms); it removes
  23 ms of NCCL time and loses it all to slower compute kernels. Cause (jobs 28286309-11): XLA's layout
  assignment re-lays the padded build level-major (node arrays `f64[48,Lmax]`, 205 transposes against 70), so
  one-launch 3-D kernels slow 5-10x. An `optimization_barrier` around the exchange does not stop it and costs
  +4 % on its own. The all_gather transport's NCCL time is bandwidth: it ships each device's WHOLE local array
  (3-D exchanges ~300 us each, mEVP's 120 subcycle exchanges ~51 us each). Coloured is 11 % slower
  (910 collectives a step). The production default stays all_gather; no config change.
- **A4 gathered SSH solve (opt-in, off by default): -3.9 %** (81.09 ms). Fidelity against the 4-run floor:
  every field at or below 1.01x except uv at 1.54x (median over the five same-code runs). That is inside the
  calibrated A/A envelope (a same-code run from another job reaches 1.85x, Protocol), so by this test the
  gathered solve is indistinguishable from a rerun; a 1-year check is still needed before any production use.
- **A5b compact halo + unrolled level sweeps (`gpu:` block of `core2_gpu.yaml`): -23 %** (56.89 ms, 86.7
  SYPD). The compact all_gather cuts NCCL from 34 to 12-14 ms; the unrolled sweeps remove the transposes
  that let XLA lay the step out level-major once the full-array all_gathers no longer pin it node-major.
- **mEVP wide halo (opt-in `gpu.mevp_wide: K` / `FESOM_MEVP_WIDE=K`, OFF in `core2_gpu.yaml`): -9 to -10 %**
  (K = 4/8/12: 51.96/51.77/51.22 ms against 57.11 at K = 0, jobs 28289821-24; collectives 344 -> 255/240/235).
  The sharded mEVP runs K subcycles per exchange on K-ring ice lanes (`fesom_jax/ice_wide.py`). On CPU it is
  BITWISE the per-subcycle exchange on every owned lane of every State field (unit gates at K = 2..60; three
  forced steps; and 96 steps of the full `core2_full_bench` physics with cheb 3 + on-device forcing, compact +
  unroll, K = 8 vs K = 0, jobs 28290236/37). On GPU the noise judge FAILS: 5 of 7 single draws land above 2x
  the 28282453 floor on owned lanes (worst 10x in density/S at one Hudson Bay node under 95 % ice, 57.3N
  77.8W level 5; the others 3-4x in ssh_rhs, uv_rhs, Kv), 2 of 7 within 1.8x; no hotspot sits on a partition
  seam (the place a refresh bug would show), several sit under ice, one (Kv) in the deep tropical Indian
  Ocean. The default stays off; whether that GPU spread is acceptable (the same arithmetic in another order,
  deterministic in the ice, against an atomic-scatter base) is the user's decision. ⚠️ The folded noise compare
  also flags sigma11/12/22 at 15-25x for every K: those are pad and extended-halo element lanes, whose
  per-subcycle values are computed from a sentinel vertex; on owned lanes sigma is within the floor.
- **Ordered sums (opt-in `gpu.ordered_sums: true` / `FESOM_ORDERED_SUMS=1`, OFF in `core2_gpu.yaml`): -13.7 %**
  (39.62 / 39.71 ms against 45.89, jobs 28293421 / 28294896 / 28292255; the OFF and ON nodes differ, the ON
  repeat on one node is within 0.2 %, kernel time 42.16 -> 36.08 ms). The element/edge -> node/element
  assemblies (`ops.scatter_add` on `elem_nodes` / `edges` / `edge_tri`, the SSH rhs and stiff increment) add
  each target's entries in ascending entry order by gathers over per-device inverse maps (`ops.ordered_sum`)
  instead of atomic scatter-adds. An optimization barrier on each sum is required: without it XLA merged ~90
  gather chains with the PP/TKE column work into one 8.6 ms kernel (43.19 ms, job 28292256). Deterministic on
  the GPU (4 runs identical in every field), and the pair test against 4 OFF runs is round-off only. On CPU
  every assembly is bitwise the `segment_sum` in isolation; inside the forced step the program differs by a
  few ulp at step 1 (rel <= 5e-16 in the prognostic fields, <= 3e-13 in the SSH outputs), from the order XLA
  gives the default program (it folds `x + scatter(0, ...)` into a scatter started from `x`). Option off: the
  lowered program is unchanged (dense step and sharded forced run hash the same as on 4682e02).
- **A6 forcing prefetch (default on, `FESOM_FORCING_PREFETCH=0` turns it off): -12.4 %** (73.92 ms,
  66.7 SYPD). The next chunk's forcing is built on one worker thread while the current chunk runs; the
  host gap between chunks fell from 13.2 to 0.08 ms/step and the GPUs are now busy 95 % of the step.
- **Inside a chunk call the GPUs are busy ~93 % of the time** (per-GPU busy, fixed metric); the serial host
  forcing build between chunks (12.6-12.9 ms/step) is the largest single removable item (A6).

## Tier B results so far

- **B2 dt 2700 (45 min), 4 GH200, 1990-1992 from the v3 hindcast's 1990-01-01 restart (job 28288574): stable,
  and close to the dt 1800 hindcast.** All 36 monthly means finite; against v3 (dt 1800, core2_full, so the A1
  options differ too) month by month (`scripts/perf/compare_monthly.py`, job 28289322, `runs/dt2700/compare_v3.json`):
  volume-mean T within 2.4e-5 °C, volume-mean S equal to 1e-6 psu, global SST within 2.2e-3 °C, sea-ice area within
  0.015 (NH) / 0.05 (SH) x 1e12 m², extremes of T, S, SSH and surface speed the same; area-rms SST difference
  0.004 °C in the first month growing to 0.037 °C after 3 years (chaotic divergence plus the dt and A1 changes).
  **Extended to 10 years (1993-1999 continuation, job 28291189; compare job 28292311, `runs/dt2700/compare_v3_1993.json`):
  no failure, all 84 months finite;** against v3 month by month the volume-mean T stays within 7e-4 °C (-1e-5 at the
  end), S within 3.7e-6 psu, global SST within 0.024 °C, NH ice area within 0.2 and volume within 1.1 x 1e12 (a few
  % in single months), the T/S/speed extremes the same; area-rms SST difference 0.033 -> 0.057 °C (chaotic). Not
  yet: the interannual-variability yardstick. The user's caveat: FESOM2 is usually unstable at 45 min on CORE2 —
  fesom-jax was not, over these 10 years.
- **dt 1800 twin with the GPU code, 1990-1992 (job 28291192; compares 28292813/14):** dt 2700 vs the twin (the dt
  effect) and the twin vs v3 (the GPU options' round-off / solver-tolerance effect at the same dt) diverge alike:
  area-rms SST 0.005 -> 0.038 °C and 0.002 -> 0.039 °C after 3 years; volume-mean T within 1.6e-5 / 1.8e-5 °C,
  S within 1.9e-7 / 1.0e-7 psu. Sea-ice area reacts more to dt than to the options (mean |d| NH 0.008 vs 0.0016, SH
  0.020 vs 0.0024 x 1e12 m², i.e. ~0.1-0.3 %). By these measures dt 2700 is inside the chaotic spread.
- **B1 sea ice every second step (`ice_ave_steps = 2`, FESOM2's scheme with its `u_w` quirk, Task 12): -7.6 % on 4
  A100 (38.56 -> 35.62 ms/step, one-node A/B 28295418), but NOT inside the noise.** Same machine, same code, Jan-Jun
  1990 from the v3 restart (28295449 vs 28295530; `runs/ice2/compare_ice2_vs_ice1_half1990.json`): area-rms SST
  0.036 -> 0.18 °C (Jan -> Jun) against 0.000 -> 0.005 °C for the ice-1 A100 run vs the GH200 twin; NH ice area
  +0.51 x 1e12 m² (+3.4 %) in April, SH volume +0.35 x 1e12 m³ (+3.5 %) in June; volume-mean T +3.3e-4 °C, S
  -1.7e-6 psu in June. Over the year against the ice-1 twin (`compare_ice2_vs_twin1990.json`) the SH ice volume
  grows +5 % (Jul) -> +36 % (Dec) and the SH area is +32 % in December (the melt is delayed), NH area +1-3 %,
  global SST -0.02 °C in December, volume-mean T +7.5e-4 °C (the twin vs v3 spread: 6e-6). No supercooling beyond
  the ice-1 runs (T_min the same to 0.04 °C). Not yet separated: how much comes from the doubled ice step and the
  held fluxes and how much from the halved `u_w` (the quirk). Not in core2_gpu.yaml (STOP 3).
- **B1 Fortran pair: FESOM2 itself moves the same way, by the same amount.** The R2 binary, namelist only
  (`ice_ave_steps` 1 vs 2, plus run length / paths / yearly restart / a shorter output list), one year 1990 from a
  copy of the R2 1989 end-of-year restart, 512 ranks (jobs 28296443/44, 14.8 + 13.8 min on 4 nodes = 1.9 node-h;
  `runs/ice2/fortran_pair/`, table `pair_table.json`). Fortran (ice-2 - ice-1) vs JAX (ice-2 - ice-1; JAX ice-1 =
  the A100 run for Jan-Jun, the GH200 twin for Jul-Dec), ice in 1e12 m² / 1e12 m³:

  | month | NH area F / J | NH vol F / J | SH area F / J | SH vol F / J | SST °C F / J | vol-mean T °C F / J | rms SST F / J |
  |---|---|---|---|---|---|---|---|
  | Jan | +0.102 / +0.103 | +0.096 / +0.095 | +0.028 / +0.028 | -0.008 / -0.011 | -2.1e-3 / -2.2e-3 | +1.7e-5 / +1.7e-5 | 0.037 / 0.036 |
  | Mar | +0.432 / +0.428 | +0.424 / +0.416 | -0.009 / -0.007 | -0.032 / -0.034 | -4.5e-3 / -4.8e-3 | +1.6e-4 / +1.6e-4 | 0.106 / 0.106 |
  | Apr | +0.507 / +0.508 (+3.4 %) | +0.506 / +0.500 | +0.113 / +0.109 | +0.050 / +0.033 | -3.9e-3 / -4.2e-3 | +2.2e-4 / +2.2e-4 | 0.118 / 0.118 |
  | Jun | +0.278 / +0.260 | +0.338 / +0.318 | +0.220 / +0.219 | +0.390 / +0.345 | -1.2e-2 / -1.2e-2 | +3.2e-4 / +3.3e-4 | 0.186 / 0.182 |
  | Sep | +0.164 / +0.163 | +0.261 / +0.270 | +0.178 / +0.183 | +1.345 / +1.279 (+9 %) | -6.1e-3 / -7.9e-3 | +6.2e-4 / +6.2e-4 | 0.094 / 0.096 |
  | Nov | +0.078 / +0.077 | +0.244 / +0.250 | +1.193 / +1.186 | +2.336 / +2.203 (+17 %) | -6.8e-3 / -7.6e-3 | +7.7e-4 / +7.9e-4 | 0.074 / 0.082 |
  | Dec | +0.135 / +0.132 | +0.264 / +0.269 | +2.881 / +2.730 (+33 / +32 %) | +3.398 / +3.087 (+38 / +36 %) | -2.0e-2 / -2.1e-2 | +7.4e-4 / +7.5e-4 | 0.173 / 0.184 |

  All 12 months are in `pair_table.json`; the Fortran ice-1 year itself sits within ~1-2 % of the JAX ice-1 one
  (Jan NH area 14.85 vs 14.84, SH volume 3.41 vs 3.38). So the SH ice change is FESOM2's `ice_ave_steps = 2`, not
  the port; the port reproduces the scheme's response to a few % in every month. The R2 binary carries the
  uncommitted edge-based ice FCT (the JAX the standard one), which does not change this.
- Command buffers (`CMDBUF=on`) on 4 GH200 with the night set: 21.09 vs 21.29 ms/step (job 28292305, -1 %): launches
  drop from ~1400 to ~80 + 40 graphs per step, but the step barely moves — the host is not the limiter. Kept off.

## Single precision (branch perf/core2-sp, sub-plan docs/plans/20261020-core2-single-precision.md)

FESOM2 #1207 policy (`precision: single`); runs from frozen snapshots of perf/core2-sp.

| date | commit | job | machine | GPUs | dt | config / change | steady ms/step | SYPD | GPU-h | note |
|---|---|---|---|---|---|---|---|---|---|---|
| 2026-10-20 | f87b3c5 | 28288484 | Levante A100 | 1 | — | convert-pair micro-experiment (Task 1) | — | — | 0.01 | default flags fold f64→f32→f64 pairs (carry 0); per-jit `xla_allow_excess_precision=False` exact |
| 2026-10-21 | f458d60 | 28292758 | Levante A100 | 4 | 1800 | full GPU set, DP vs SP | 36.0 / 470 | — | 0.9 | SP INVALID: every chunk recompiled (the SP params cast missed the id()-keyed executable cache) |
| 2026-10-21 | 1e4f8c4 | 28295120 | Levante A100 | 4 | 1800 | full GPU set (compact, unroll, mEVP wide 12 + unroll 12, ELL + inc ELL, CG batch 4, Pallas columns, Chebyshev 3, on-device forcing, gathered SSH), DP vs SP alternating on one node | DP 35.85 / 36.00, SP 25.11 / 24.72 | DP 137, SP 197 | 0.8 | **SP −31 %** (sp/dp 0.69); finite |
| 2026-10-21 | e6beb67 | 28295508 (coordinator) | dolpung GH200 | 4 | 1800 | same set on one node, DP / DP+ordered sums / SP / SP+ordered sums, 2 reps | 20.41/20.44, 23.29/23.23, 12.90/12.92, 10.78/10.69 | SP+osum 459 | — | SP −37 %; ordered sums slow DP on Hopper (fast f64 atomics) but help SP: SP+osum −48 % vs DP |

Conservation gate (sub-plan T1/T2; `scripts/perf/compare_precision.py`; DP-f32IC = DP from the float32-rounded IC):

| run (4 A100, core2_gpu.yaml, cold start 1958-01-01) | S̄ SP−DP | T̄ SP−DP | global-mean SSH SP−DP | rms SST / SSS / SSH (SP−DP vs DP-f32IC−DP) | verdict |
|---|---|---|---|---|---|
| 30 days, job 28292762 (f458d60) | +5.8e-9 psu | −8.5e-8 °C | +2.6e-8 m | 5.9e-3 / 2.2e-3 / 1.4e-4 vs 1.6e-3 / 6.2e-4 / 5.9e-6 | T1 pass |
| 1 year, job 28295391 (d5e7b13), 3 × 38 min | +6.2e-8 psu (≤1.5e-7) | −3.8e-7 °C (≤2.5e-6) | +3.4e-8 m (≤5e-7) | 2.5e-2 / 1.7e-2 / 1.1e-3 vs 2.0e-2 / 1.3e-2 / 6.9e-4 (≤3×) | **T2 pass** |
| FESOM2 zstar Fortran pair, 1 year (Task 1b) | +3.1e-8 psu | −2.6e-7 °C | +9.6e-8 m | 2.2e-2 / 1.4e-2 / 7.4e-4 | — |

Budget diagnostic, 1 year (not pass/fail; the State-based budget does not close in DP): ocean volume change SP +1.18e7
m³ (DP 0; Fortran SP CONS residual 3.7e7 m³), total-salt change SP 2.18e11 vs DP 8.47e10 psu m³.

## The production stack on 4 GH200, 2026-10-21 morning (branch perf/core2-combo, merged into perf/core2-gpu)

CORE2 at dt 2700 from the v3 1990-01-01 restart, `runs/dt2700/core2_gpu_sp_osum_dt2700_1990.yaml` (core2_gpu.yaml +
`precision: single` + `gpu.ordered_sums`), executable reuse and the placed-constants cache on, one dolpung node.
Steady = `ab_time_gh200.sbatch` (K 96, 4 K steps); loop = `run_core2_gpu_gh200.sbatch`, 3 months with monthly output,
ms/step after the leading compile chunks without the later compile chunks (`scripts/perf/loop_ms.py`).

| commit | job | change | steady ms/step | loop ms/step | SYPD | GPU-h | note |
|---|---|---|---|---|---|---|---|
| 0a0acb5 | 28298126 | ice-1 (the night's 630-SYPD set, repeat) | — | 11.36 (c32) | 651 | 0.23 | same node as the next rows |
| 0a0acb5 | 28298127-29 | + ice_ave_steps 2 | — | — | — | 0.1 | FAILED: `lax.cond` dtypes, the running-mean weights were float64 (fixed 3b9c354: `real(k, WP)` as upstream) |
| 3b9c354 | 28298369 | ice-1 / ice-2, ABAB | 11.16, 11.28 / 10.97, 10.94 | — | 660 / 675 | 0.53 | inside the calls 10.85 → 9.70 (−10.6 %), but the host between chunks 0.45 → 1.6 ms/step: the main loop waited for the forcing prefetch |
| 3b9c354 / 1f980ca | 28298486 | host forcing tables, one Grace thread, K 96 | — | — | — | 0.05 | per step 10.7 → 7.8 ms: 384 → 192 slice reads, gathers 470 → 240 ms per chunk (the reader read and gathered both bracket ends at every roll) |
| 1f980ca | 28298487 | ice-1 / ice-2 with the slice reuse, ABAB | 11.11, 10.89 / **9.74, 9.77** | — | 672 / **758** | 0.52 | host between chunks 0.03 ms/step |
| 1f980ca | 28298758 | ice-2, production loop, chunk 32 | — | 10.65 | 694 | 0.17 | |
| 1f980ca | 28298759 | ice-2, production loop, chunk 96 | — | **9.83** | **753** | 0.22 | + one 46 s compile of the 32-step month-end chunk |
| 230f178 / d1d4ccb | 28298963 | ice-2 / + `--lane-order rcm` / + rcm + `FESOM_FORCING_PROCS=8`, ABCABC | 9.76, 9.85 / **9.61, 9.57** / 9.61, 9.58 | — | 755 / **771** / 771 | 0.75 | RCM lanes −2.1 % (median edge lane distance per shard 6164/4387/5731/5175 → 149/66/103/117); the read-ahead changes nothing while the host forcing (7.8 ms/step) is below the step |
| d1d4ccb | 28299144 | rcm + ice-2, command buffers off / on (`XLA_FLAGS=`), ABAB | 9.56, 9.62 / **9.07, 9.07** | — | 771 / **815** | 0.5 | −5.4 %; the earlier "no gain in the loop" (28298250) was the forcing-bound loop |
| d1d4ccb | 28299142 | rcm + ice-2, production loop, chunk 96 | — | ~9.8-10.3 | — | 0.22 | loop timings at 1 s resolution (±4 %): not resolvable against 28298759; progress now prints ms (658cda9) |
| 658cda9 | 28299375 | rcm + ice-2 + command buffers, production loop, chunk 96 | — | **9.76** (9.05 between month ends) | **758** | 0.22 | the month-end writes ~0.6 s each → gpu.output_async (6e38d73) |
| 6e38d73 | 28299731 | + FCT factor-exchange fix (af5429a) + `gpu.output_async`, production loop, chunk 96 | — | 9.47 | 781 | 0.24 | month-end writes off the loop; the next 96-chunk after a month end's 32-chunk still waited for its forcing (1.1 s over 3 months) |
| 103b11d | 28299884 | + `gpu.forcing_ahead: 2`, production loop, chunk 96 | — | **9.05** | **817** | 0.22 | host waits 1.1 → 0.055 s; the loop runs at the steady speed (28299144: 9.07) |

Correctness note (2026-10-21): every row above ran with the sharded FCT limiter exchanging the antidiffusive sums
instead of the limiter factors (fixed af5429a, docs/PORTING_LESSONS.md); the fix replaces one exchange by one paired
exchange, so the timings stand to within noise until re-measured.

## Where a CORE2 step goes at 817 SYPD (4 GH200, 2026-10-21, job 28303935)

The production stack of the table above (6d4f8cb: SP, ordered sums, ice-2, rcm lanes, command buffers on, the FCT
fix), `gpu_profile_gh200.sbatch` from the v3 1990-01-01 restart at dt 2700: steady 9.07 ms/step (815.7 SYPD, = job
28299144); nsys `--cuda-graph-trace=node` kernels 8.97 ms per GPU per step, so the step is compute-bound
(`runs/perf/28303935-gh200x4_sp_ice2_817/`). Kernel ms per GPU per step:

| area | ms | share | launches |
|---|---|---|---|
| SSH (gathered CG solve, Chebyshev 3: precond 0.92, CG vector ops 0.34, set-up 0.37, rhs) | 1.75 | 19.5 % | 526 |
| ops helpers (`ordered_sum` 0.78 of it) | 1.75 | 19.5 % | 113 |
| NCCL (cost 0.49, waiting for the slowest GPU 0.71) | 1.20 | 13.4 % | 28 |
| tracer advection (Zalesak limiter 0.58 of it) | 0.96 | 10.7 % | 19 |
| sea ice dynamics (mEVP) | 0.75 | 8.3 % | 141 |
| halo packing | 0.40 | 4.5 % | 66 |
| GM/Redi | 0.40 | 4.4 % | 16 |
| pressure, density | 0.32 | 3.5 % | 12 |
| precision casts (`precision.wp`) | 0.30 | 3.3 % | 93 |
| momentum | 0.28 | 3.1 % | 14 |
| everything else | 0.86 | 9.6 % | — |

Reading against fesom3-jax at the same settings (SP, dt 2700, ice every second step, 4 GH200: 7.26-7.4 ms/step,
~1000-1020 SYPD; ours 9.07): the largest single difference is the SSH solve. fesom3-jax runs split-explicit (its
substeps were 0.9 ms/step in DP before its Pallas SE kernels), we run the CG, 1.75 ms here and the only area with a
host-side loop (the while loop's condition; command buffers gave us -5.4 % and fesom3-jax nothing). The SSH share is
above the 15 % mark of plan Task 14 (B4 split-explicit, FESOM2's `oce_ale_ssh_splitexpl_subcycl.F90`). Of the NCCL
waits, two collectives a step (after `input_concatenate_fusion.52`, an f32[32073,48] node field) average 130-140 us
of waiting with up to 2.3 ms, the last GPU 0 or 3: a load imbalance before them, not the exchange.

Command buffers for the CG's while loop (the condition on the device; fesom3-jax: one GH200 CG 38.6 -> 34.6 ms/step):
4 A100, the 817-SYPD stack (SP, dt 2700, ice-2, rcm), A/B on one node, `XLA_FLAGS=` (default command buffers) vs
`--xla_gpu_enable_command_buffer=+WHILE,+CONDITIONAL`: 16.73 / 16.76 vs 16.69 / 16.72 ms/step (job 28304485, -0.2 %,
noise); `+WHILE` alone 17.59 / 17.62 vs 17.73 / 17.42 (job 28304173, another node). No gain: the CG's host round
trips (4 iterations per trip, `gpu.cg_batch`) are not what the SSH costs here; the 1.75 ms are its kernels.

## STOP 4 pair: 10 years SP vs DP at dt 2700 (2026-10-21, jobs 28304114 / 28304115)

The 817-SYPD stack (6d4f8cb: ordered sums, ice-2, rcm, command buffers, async output, forcing two ahead, the FCT
fix), 1990-1999 from the v3 1990-01-01 restart, `runs/sp10/core2_fast_{single,double}_1990.yaml` (only `precision`
differs), monthly output and yearly archive restarts, one dolpung node each. Both finite over all 120 months.

| | SP | DP |
|---|---|---|
| job wall / loop | 24 min / 1354 s | 46 min / 2767 s |
| loop average (`loop_ms.py`, compile chunks of later month lengths excluded) | 10.00 ms/step, 739 SYPD | 22.18, 334 |
| device time of the 96-step chunks, years 1991-1999 | **8.65-8.71 ms/step (~850 SYPD)** | |

The 10-year average is above the 3-month 9.05 by the first year's compiles (bootstrap, 96, 32 and 64 steps,
~45 s each) and the ten synchronous archive restarts (6.8-12.9 s each, ~100 s): an async archive write would give
~7 % on a 10-year job.

SP - DP at the yearly restarts (`runs/sp10/compare/yearly.py`, job 28305132; monthly means job 28305133,
`runs/sp10/compare/monthly_sp_vs_dp.json`):

| Jan 1 of | volume-mean S [psu] | volume-mean T [°C] | global-mean SSH [m] | ocean volume [m³] | rms SST / SSS / SSH (restart) |
|---|---|---|---|---|---|
| 1991 | +1.1e-10 | -1.6e-6 | -1.8e-8 | -6.6e6 | 0.031 / 0.011 / 1.3e-3 |
| 1993 | +4.9e-8 | -2.0e-5 | +6.6e-8 | +2.4e7 | 0.053 / 0.021 / 2.8e-3 |
| 1995 | +1.0e-7 | -7.9e-6 | -9.4e-8 | -3.4e7 | 0.128 / 0.037 / 5.3e-3 |
| 1997 | +1.5e-7 | +8.1e-5 | -1.0e-7 | -3.7e7 | 0.119 / 0.032 / 6.2e-3 |
| 2000 | **+2.0e-7** | +7.6e-5 | **-1.5e-7** | -5.4e7 | 0.130 / 0.035 / 6.7e-3 |

- Salt: a steady +2e-8 psu a year, +2.0e-7 after 10 years. Yardsticks: the FESOM2 zstar Fortran SP-DP pair +3.1e-8
  in one year (sub-plan Task 1b), the sub-plan's 1-year bound 1.5e-7, fesom3-jax's fixed SP +1.5e-7 a year.
- Volume / mean SSH: -1.5e-7 m in 10 years (the 1-year bound 5e-7 m); FESOM2 SP's CONS residual is 3.7e7 m³ in one
  year, ours 5.4e7 m³ in ten.
- Temperature changes sign (-2e-5 ... +8e-5 °C): chaotic, not a drift; there is no 10-year DP-DP envelope to set a
  bound. Monthly-mean area-rms SST 0.02-0.07 °C (1998-1999), the size of the dt-1800 twin vs v3 (0.039 after 3 years)
  and dt 2700 vs v3 (0.057 after 10). Sea-ice area and volume within ~0.1 % in both hemispheres.

## Archival restarts on the writer thread (`gpu.restart_async`, 37c4655)

The calendar archival restarts go to the output thread (`restart.latest` moves after the store is complete; the
end-of-run restarts wait for the thread). Check: the first 3 years of the STOP 4 SP run again (1990-1992, snapshot
d96ab32, job 28311554 on l50426; the 10-year run 28304114 ran on l50413), everything else identical:

| | 10-year run, sync (28304114) | restart_async (28311554) |
|---|---|---|
| loop pause after the 1991 / 1992 archive | 6.75 / 12.9 s | **0.0 / 0.01 s** (the writes, 20.2 / 20.9 s, ran behind the loop) |
| 96-step chunks in the 29 chunks after each archive vs before | +0.03 / +0.18 s | -0.01 / +0.14 s |
| device time of the 96-step chunks, 1990-1992 | 8.74 ms/step | 8.81 (the other node; +0.7 %) |
| the three archive stores (1991, 1992, 1993) | | **bitwise the 10-year run's**, all 57 arrays and attributes |

The write does not slow the chunks running beside it. Over a 10-year job it takes the ~100 s of archive pauses out
of the loop (~7 % of the 24-min job). The writes themselves took 20-22 s on this node against 7-16 s in the 10-year
job (the synchronous end-of-run restart too: 22.2 vs 11.1 s), so the file system was slower during this job.

## Split-explicit SSH: speed (plan 20261021-core2-split-explicit.md, Task 6)

One dolpung node, A/B in one allocation (`ab_time.sbatch`, 2 warm + 4 timed chunks of 96, two reps), the 817-SYPD stack
(SP, ordered sums, ice-2, rcm lanes, command buffers, dt 2700 from the v3 1990-01-01 restart); SE = Sergey's SM
AB3-AM4, M 60, chi 0.1 (the CFL guard's M_min at dt 2700), no drag, in place of the gathered Chebyshev CG.

| variant | job | ms/step (rep 1 / 2) | SYPD |
|---|---|---|---|
| CG (the 817 stack) | 28313318 / 28315580 | 9.07, 9.07 / 9.08, 9.56 | 816 |
| SE, two exchanges per substep | 28313318 | 9.73, 9.74 | 760 |
| SE + `gpu.se_wide` 10 | 28313318 | 8.54, 8.73 | 848-866 |
| SE + se_wide 10 + `gpu.se_unroll` 10 | 28313318 / 28314647 / 28315580 | 8.46, 8.65 / 8.68, 8.37 / 8.60, 8.47 | **855-884** |
| SE + se_wide 20 + se_unroll 20 | 28314647 | 8.63, 8.59 | 857-861 |
| SE + se_wide 30 + se_unroll 30 | 28314647 | 8.55, 8.54 | 865-867 |
| + COLLECTIVES in command buffers | 28314647 | XLA error "Traced CUDA graph is empty" | — |
| CG + `gpu.step_unroll` 2 | 28315580 | 9.58, 9.93 | 745-772 |
| SE (w10 u10) + step_unroll 2 | 28315580 | 8.79, 8.95 | 826-842 |

nsys of SE + w10 + u10 (job 28314373, `--cuda-graph-trace=node`; the profiled chunk ran at 8.57 ms/step, as unprofiled):
- kernels 7.78 ms/step (CG stack 8.97): `ssh_se.py` 0.69 ms (the CG was 1.75), everything else as before; compute without
  NCCL balanced across the GPUs (6.41-6.53 ms/step);
- the GPUs are busy 7.5-8.2 of 8.57 ms/step: ~1.0-1.2 ms/step idle, half of it in gaps of > 50 us between an NCCL
  all-gather and the next CUDA graph, the rest in short gaps; 16 % of the launches start < 20 us after their API call
  returns and the median launch waits only 250 us in the queue, so the host is barely ahead of the GPUs;
- per GPU thread and step: 32 `cuGraphLaunch` taking 2.84 ms (~89 us each), 95 plain launches 0.74 ms (66 small
  packing/slicing fusions next to the 30 all-gathers, which stay outside the graphs), one stream sync 2.6 ms.

So split-explicit removes ~1.1 ms of kernels but the step becomes host-bound at the launch boundaries: ~-5 % (CG 9.07
→ 8.4-8.7 ms/step), less than the -1.0 to -1.3 ms estimated. Wider windows (fewer all-gathers, more redundant ring
lanes) change nothing; two steps per loop trip (fewer predicate syncs) make both stacks slower. Without the idle the
SE step would be ~7.6-7.8 ms/step (~950 SYPD): the next levers are fewer launch boundaries (merged halo exchanges)
and cheaper graph launches, not the substeps.

## Split-explicit SSH: stability screens (plan Task 7, job 28315629)

SP, ice-2, rcm lanes, command buffers, se_wide 10 + se_unroll 10, 3000 steps each with `--chunk-diagnostics` (chunk
32), one GH200 node (`scripts/perf/se_screen.sbatch`, configs `runs/se_screen/`); M = the CFL guard's M_min and 1.5x.

| start | chi | M | no drag | with `bt_drag` |
|---|---|---|---|---|
| v3 1990-01-01 restart, dt 2700 (94 days) | 0.1 | 60 | **NaN at step 864** (day 27) | finite, max\|eta\| 2.02 m |
| | 0.1 | 90 | **NaN at step 544** (day 17) | finite, 2.02 |
| | 0.2 | 68 | finite, 2.04 | finite, 2.02 |
| | 0.2 | 102 | **NaN at step 960** (day 30) | finite, 2.02 |
| CG control | | | finite, max\|eta\| 1.87 m, max\|uv\| 2.38 m/s | |
| 1958 cold start, dt 1800 (62 days) | 0.1 | 40 / 60 | finite, 3.80 / 3.81 | finite, 3.94 / 3.96 |
| | 0.2 | 46 / 69 | finite, 3.77 / 3.79 | finite, 3.90 / 3.93 |
| CG control | | | finite, max\|eta\| 2.09 m | |

- Without the barotropic drag dt 2700 blows up within a month in three of four settings, sooner with more substeps
  (the SM substeps' damping of the fast mode goes as 1/M; FESOM_F measured the same, M 100 worse than 50); with the
  drag (Sergey's answer of 2026-10-18) every setting holds the 94 days, max\|uv\| and the vertical CFL as the CG's.
- The cold start's barotropic adjustment reaches max\|eta\| 3-4 m in the first ~8 days under SE (2.1 m under the CG's
  implicit damping), then both settle at 1.9 m (the Kokkos port saw 3 m against 2 m too).
- Not yet run: DP, a full year (FESOM_F's zstar event at chi 0.05 came on day 229).

## Split-explicit SSH: the 10-year twin (plan Task 8, job 28318591)

User 2026-10-21: chi 0.1 + `bt_drag` + M 60. SE (se_wide 10, se_unroll 10) against the STOP 4 SP run (CG, job
28304114), 1990-1999 at dt 2700 from the v3 1990 restart, SP + ice-2, everything else identical (`runs/se10/`):
all 120 months finite.

| | CG (28304114) | SE (28318591) |
|---|---|---|
| device time of the 96-step chunks (median; --progress syncs after every chunk) | 8.69 ms/step | **7.79** (7.76-7.80 every year) |
| compiles (bootstrap 96, 96, 32, 64) | ~46 s each | ~67 s each |
| host waiting for the forcing, 10 years | 25 s | **168 s** |
| loop (`loop_ms.py`) | 10.00 ms/step, 739 SYPD | 9.26, 799 |

SE - CG (`runs/se10/compare/`, jobs 28319446/7; the SP - DP pair of the STOP 4 section as the chaotic yardstick):
- global-mean SSH at the yearly restarts +5.0e-6 m (1991) ... +4.7e-5 m (2000): the CG runs lose -6e-6 m a year
  (SP and DP alike: -5.95e-5 m by 2000), SE -1.2e-6 m (the explicit eta update is exactly compatible with the
  transport divergence; the CG stops at its tolerance); volume-mean S within +-3e-8 psu, no trend; T chaotic;
- monthly-mean area-rms SE - CG: SST 0.0028 °C / SSS 0.0019 / SSH 0.1 mm in January 1990 (SP - DP 0.0004 / 0.0002 /
  0.0), then the size of SP - DP: December 1990 0.027 / 0.0074 / 1.2 mm (SP - DP 0.022 / 0.006 / 0.9), December 1999
  0.073 / 0.021 / 3.7 mm (0.068 / 0.024 / 3.7); sea-ice area within 0.008e12 m^2.

So the SE solution is the CG's within the chaos after a few months, and the device step is 0.9 ms faster; the
production loop does not show it yet because the host forcing build (one worker) and the longer compiles now lead.

## The production loop with split-explicit: forcing workers and the static ice branch (jobs 28319135, 28319459)

`scripts/perf/loop_ab.sbatch` (5c7ebab): `run_from_config --progress`, 2880 steps (January-March 1990) in chunks of
96 with monthly output, timed by `loop_ms.py` (the host's forcing wait included, later compiles excluded), two reps;
SE = chi 0.1 + drag + M 60, se_wide 10, se_unroll 10, the rest as the 10-year twin:

| SE + | ms/step | SYPD |
|---|---|---|
| nothing | 8.34, 8.44 | 876-886 |
| **`gpu.forcing_procs` 8** (JRA decode in 8 worker processes, bitwise) | **7.93, 7.90** | **933-936** |
| `gpu.ice_static` | 8.36, 8.35 | 885-886 |
| both | 8.11, 8.49 | 871-912 |

- The host's wait for the forcing is 2-4 s over the 2880 steps in every variant, yet the read-ahead processes take
  0.5 ms/step off: the forcing worker thread decoding JRA (HDF5, under the GIL) slows the main thread's work around
  the chunk calls, not only the hand-over. The CG stack never showed it (its chunks were long enough).
- `gpu.ice_static` removes the per-step host sync of the ice `lax.cond` (`gpu_profile` "inside the calls" 7.61 ->
  7.15-7.19 ms/step, job 28319135) but not device time: the loop is unchanged (8.35 against 8.34-8.44) and with the
  forcing workers worse. Off; kept as an option (CPU: the `lax.cond` run to 1e-12, Kv 1.4e-10). On the CG stack it
  costs 2-10 % (9.44 / 10.18 against 9.20-9.22, the CG loop syncs anyway).
- Halo transports: `--coloured` / `--padded` refuse `gpu.halo_compact` (the compact all_gather); measured with
  compact off in job 28320507.

## Halo transport and forcing workers in the SE production loop (job 28320507)

`loop_ab.sbatch`, the 28319459 setup with `gpu.forcing_procs` 8 as the base (2880 steps, chunk 96, monthly output):

| SE + forcing_procs 8 + | ms/step | SYPD |
|---|---|---|
| the compact all_gather halo (`gpu.halo_compact`) | 8.20, 7.86 | 902-941 |
| `--coloured` (ppermute rounds; compact off) | 8.69, 8.70 | 851 |
| **`--padded`** (slot-padded all_to_all; compact off) | **7.77, 7.78** | **951-952** |
| forcing_procs 16 instead of 8 | 7.90, 7.86 | 937-941 |

The padded transport is now the fastest (A5 found it tied with all_gather on the old A100 DP stack); the coloured one
costs a round per colour. More read-ahead workers than 8 give nothing.

## The best stack over 10 years and over one year; the host I/O now leads (jobs 28321119, 28321933)

Best stack = SE (chi 0.1, drag, M 60, se_wide 10, se_unroll 10) + `--padded` (compact off) + forcing_procs 8 +
`gpu.chunk_tail` 32 + async output and restarts, SP, ice-2, rcm lanes, command buffers.

| | 10 years, 28321119 | 1 year, monthly archive restarts, 28321933 A / B (same job) |
|---|---|---|
| device time of the 96-step chunks (median) | 7.71 ms/step | **7.43** / 7.66 |
| compiles | 3 x ~67 s (the 64-step one gone) | 3 x ~65 s |
| host wait for the forcing | 139 s | 23 s / 2 s |
| main loop held by the writers | 136 s (10-28 s at every year end) | 0.6 s / 81 s |
| background restart store | 22-40 s | 3-7 s / 15-20 s |
| loop (`loop_ms.py`) | 10.05 ms/step | 8.93 / 14.73 |

- The device step is at the 1000-SYPD mark (7.40 ms at dt 2700); the loop is decided by /work: a restart store takes
  2-40 s depending on the hour (the same code: 2.0 s at 22:30, 17.1 s at 00:30 for the final synchronous one).
- The 10-year run's year-end holds: the restart and the monthly output shared one writer thread with two writes in
  flight, so December's output queued behind a 22-40 s restart and the loop waited. Fixed: the restarts have their
  own thread (combo ecbf10b; both async tests green, jobs 28321713/7). The 1-year B run's 81 s are the monthly
  archive restarts (a stress setting) backing up behind a slow file system; yearly ones never queue.
- The host's forcing build per chunk (`scripts/perf/forcing_host_time.py`, a Levante CPU node, 96 steps of 2700 s,
  SP tables): 1304 ms of tables + 332 ms of partition without read-ahead, 736 + 355 ms with 8 processes. Now:
  the tables are partitioned straight into the runner's folded layout (one copy instead of four, the last of which
  ran on the main thread; combo 885144d, end-to-end restarts bitwise on pi, job 28321825) and built in float32
  (combo f55c061: 589 ms). Interpolating the eight fields on threads made it slower (1884 ms, job 28321972:
  the threads starve the process pool's result thread) and was reverted.

## SE substeps unrolled: compile against speed (job 28322172)

`loop_ab.sbatch`, the best stack (ae152b6), 2880 steps, two reps:

| `gpu.se_unroll` (se_wide 10) | loop ms/step | SYPD | 96-chunk device | compile per chunk length |
|---|---|---|---|---|
| **10** | **7.59, 7.59** | **974-975** | 7.45 | 61-65 s |
| 5 | 8.03, 7.97 | 921-928 | 7.54 | 58-65 s |
| 1 | 8.68, 8.68 | 852 | 8.53 | 53-56 s |

The unroll takes 1.1 ms/step off for ~9 s more per compile: a 10-year job (3 compiles) gains ~120 s for ~27 s.

## SE window width (job 28322503, e3c23eb)

`loop_ab.sbatch`, 2880 steps, two reps, se_unroll = se_wide:

| `gpu.se_wide` / `se_unroll` | loop ms/step | 96-chunk device | compile per length |
|---|---|---|---|
| **10** | 8.49 (a 5.6-s forcing stall), **7.58** | 7.39-7.44 | 60-65 s |
| 15 | 7.62, 7.66 | 7.48-7.50 | 68-73 s |
| 20 | 7.71, 7.71 | 7.54-7.55 | 74-81 s |
| 30 | 7.78, 7.77 | 7.58-7.60 | 87-97 s |

Wider windows trade fewer exchanges for more ring lanes and longer compiles; 10 stays.

## The final stack over 10 years (job 28323053, e3c23eb)

SE (chi 0.1, drag, M 60, se_wide 10, se_unroll 10) + `--padded` (compact off) + forcing_procs 8 + chunk_tail 32 + output
and restarts on their own threads + the folded float32 forcing tables; SP, ice-2, rcm lanes, command buffers; 1990-1999
from the v3 1990 restart, monthly output, yearly archive restarts (3-9 s each in the background):

| | CG stack (28304114) | first SE (28318591) | **final** |
|---|---|---|---|
| loop for 10 years (compiles included) | 1354 s (638 SYPD) | 1352 s | **1161 s (744 SYPD)** |
| `loop_ms.py` (later compiles excluded) | 10.00 ms/step, 739 SYPD | 9.26, 799 | **8.25, 897** |
| 96-step chunks, device (median) | 8.69 | 7.79 | **7.43** (~1000 SYPD) |
| host wait for the forcing | 25 s | 168 s | 92 s |
| loop held by the writers | 128 s | 4 s | 6.5 s |
| compiles | 4 x 46 s | 4 x 67 s | 3 x 65 s |

The whole job (setup, compiles, the 10 years, the final restarts) took 21 min. What separates the loop from the device
step now: the compiles (~17 % of a 10-year job) and the host's forcing (~8 %, stalls of the /work reads).

The final run's ten yearly restarts are **bitwise** the first SE twin's (28318591; every array of 1991-2000, job
28323392 and a direct check): the padded transport, the folded float32 forcing tables, chunk_tail, the writer threads
and the read-ahead processes change no bit on the GPU (the ordered sums make the runs deterministic). Both differ
from the CG run pointwise by up to 8.4 °C in T (chaos, see the twin section).

## Chunk length and forcing prefetch over one year (job 28323381)

`loop_ab.sbatch`, the final stack, 11680 steps (1990), monthly output, one rep each:

| | `loop_ms` (later compiles excluded) | the whole loop, compiles included | compiles | host wait |
|---|---|---|---|---|
| chunk 96, forcing_ahead 2 | 8.58 ms/step | 304 s | 3 | 16.1 s |
| **chunk 32**, forcing_ahead 2 | 7.96 (930 SYPD) | **223 s** | **2** (bootstrap 32, 32) | 1.5 s |
| chunk 96, **forcing_ahead 4** | **7.56 (978 SYPD)** | 284 s | 3 | 2.0 s |

- Chunk 32 is the only length a CORE2 month needs at dt 2700 (32 steps a day): one program fewer, ~0.4 ms/step of
  per-chunk overhead more. For jobs up to a few years it is the faster choice; over 10 years the two are even
  (96: 3 x 65 s + 116864 x 7.56 ms = 1078 s, 32: 2 x 65 s + 116864 x 7.96 ms = 1060 s).
- Four chunks of forcing prefetch absorb the /work read stalls (bitwise; now in core2_gpu_fast.yaml).

## The final stack, steady and under nsys (job 28323629, 790ca2e)

`gpu_profile.py time` (2 warm + 4 timed chunks of 96), 4 GH200, the final stack with forcing_ahead 4: **7.39 ms/step =
1001.5 SYPD at dt 2700**, host between chunks 0.03 ms/step (the forcing fully hidden), finite. Under nsys (node
trace) the step stretches to 9.12 ms: the padded transport's 50 SendRecv a step (+12 all-gathers of the wide halos)
are traced one by one and the host falls behind (26.8 host-late launches a step); kernels by area as before (SE
substeps 0.68 ms, mEVP 0.76, tracer advection 0.91, ordered sums/scatters 1.86). A per-area comparison with fesom3's
trace needs an untraced kernel timing (ncu or `--cuda-graph-trace=graph` sums).

## The sharded FCT seam bug and the salt budget (jobs 28329875, 28330443/4)

`--budget` (float64 global volume, ocean + ice salt, heat after every chunk), before the fix (658cda9) and with it
(af5429a), the same configuration (core2_gpu.yaml of that day, DP):

| | change of the total salt (ocean + ice) | volume |
|---|---|---|
| pi, 1 device (no seams), 432 steps | -2.5e-11 relative | |
| pi, 4 devices, before the fix, 432 steps | **+3.4e-9 relative** | 0 |
| pi, 4 devices, with the fix, 96 steps (time limit) | -3.7e-12 relative | 0 |
| CORE2, 4 GH200, 1990 (17424 steps), before the fix | **+2.2e-7 psu** (volume mean) | 0 |
| CORE2, the same, with the fix | -1.3e-7 psu | 0 |

The seam bug broke the salt budget: the two devices limited the same seam edge with different factors, so what
left a node was not what entered its neighbour. On CORE2 it moved the yearly salt change by ~+3.5e-7 psu, the order
of the CORE2 v3 hindcast's excess over the Fortran one (+5e-5 against +2e-5 psu in 62 years, ~5e-7 psu a year): v3
ran on 4 GPUs, so part of that excess may be this bug (one year each, not proven). The -1.3e-7 psu a year left with
the fix has another source, not yet looked for.

## The year-window adjoint on the fast stack (plan 20261022-core2-fast-adjoint.md)

The Y4raw7 adjoint (J = the window-mean MLD over natl_subpolar; c_k, c_eps, cd, alpha, k_gm) on GH200, judged by its
own central FD of the same configuration. s/step is `run_s / n_steps` (without the compile; `reverse_seconds`
includes it). 1-GPU rows ran four processes on a node.

| configuration | GPUs | window | vs own FD | s/step | s per model day | h per year |
|---|---|---|---|---|---|---|
| reference (DP, CG, ice every step, dt 1800) | 1 | 30 d Jan 2014 | 0.3-3.5 % | 0.607 | 29.1 | 2.96 |
| ice2 | 1 | 30 d | 0.3-2.7 % | 0.547 | 26.3 | 2.66 |
| ice2 + se (M 60) | 1 | 30 d | 0.4-2.2 % | 0.643 | 30.9 | 3.13 |
| ice2 + se + dt 2700 | 1 | 45 d | 0.0-1.2 % | 0.650 | 20.8 | 2.11 |
| ice2 + se (M 60) | 4 | 30 d | = 1 GPU | 0.313 | 15.0 | 1.52 |
| ice2 (CG) | 4 | 30 d | = 1 GPU | 0.338 | 16.2 | 1.65 |
| ice2 + se + dt 2700 + sp | 4 | 30 d | 0.6-1.1 % | 0.244 | 7.8 | 0.79 |
| ice2 + se + dt 2700 + sp | 4 | **365 d Jan 2013** | **blows up** (x10/15 d from ~Sep backward) | 0.273 | | |
| **ice2 + se (M 40) + sp, dt 1800** | 4 | **365 d Jan 2013** | **0.2-7.0 %** | 0.213 | 10.2 | 1.04 |
| **ice2 + se (M 40) + sp, dt 1800** | 4 | **365 d Jul 2013** | **2.1-9.0 %** | 0.216 | 10.4 | 1.05 |

dt 2700 is dropped for the adjoint (Nikolay 2026-10-22); the SE substep count follows dt (M 40 at dt 1800). The GPU
options on one chunk (48 steps, dt 1800, fwd + bwd ms/step, job 28355056): none 216.5; column kernel + column unroll +
mEVP unroll 12 + ordered sums + SE unroll 10: 132.8 (-39 %; leaving out the column kernel costs ~10 %, the SE unroll
~5 %, the ordered sums ~2 %, the other two ~0); + the SE wide halo K 10: 133.9 (forward -12 %, reverse +1 %, job
28355248); without the ice derivative 104.9 (the ice adjoint ~21 %, job 28355270). The reverse program is ~11x the
forward; the profile without options (job 28350488): NCCL 54 % of the kernel time (950 launches a step), the SE
divergence's gather transpose (an atomic scatter) the largest compute item (25 ms of 352). The year rows above ran
without the options: with them the year is ~0.65 h (one chunk measured, not a year).

## REcoM: the BGC tracer stage, 31 passive tracers (plan 20261025-recom-port.md, Task 2)

31 passive tracers (REcoM's bin_2p3z2d count) through the BGC tracer stage (FCT, Redi + K33, implicit vertical
diffusion; no sources or sinks), `configs/core2_gpu_fast_bgc31.yaml` = the fast stack + `recom: {passive_tracers:
31}`, from the v3 1990 restart, dt 2700, 4 GPUs, `ab_time` (2 warm + 4 timed chunks of 96 steps).

| date | commit | job | machine | variant | steady ms/step | SYPD | per added tracer | note |
|---|---|---|---|---|---|---|---|---|
| 2026-10-25 | 9a4a5e8 | 28417245 | GH200 | SP fast stack (no BGC) | 7.48 | 988 | — | the reference |
| 2026-10-25 | 9a4a5e8 | 28417245 | GH200 | DP, ordered sums off (no BGC) | 16.72 | 442 | — | |
| 2026-10-25 | 9a4a5e8 | 28417245 | GH200 | DP + 31, one tracer at a time (`loop`) | 137.49 | 54 | 3.9 ms | peak 11.0 GB/GPU |
| 2026-10-25 | 9a4a5e8 | 28417245 | GH200 | DP + 31, batched (`vmap`) | 84.42 | 88 | 2.2 ms | peak 13.9 GB/GPU |
| 2026-10-25 | 9a4a5e8 | 28417245 | GH200 | SP + 31 (`loop`, `vmap`) | — | — | — | failed: the Pallas column kept a float64 result in a float32 loop (fixed c7f4863, b02b100) |
| 2026-10-25 | c7f4863 | 28417425 | A100 | DP, no BGC / + 31 vmap / + 31 vmap, column kernel off / + 8 vmap | 29.91 / 156.61 / 175.14 / 68.06 | 247 / 47 / 42 / 109 | 4.1 / 4.7 / 4.8 ms | linear in the tracer count; the Pallas column kernel helps (+12 % without) |
| 2026-10-25 | b02b100 | 28417702 | GH200 | SP fast stack (no BGC) | 7.63 | 969 | — | same node as the next three |
| 2026-10-25 | b02b100 | 28417702 | GH200 | **SP + 31 vmap, ordered sums on** (the fast stack's setting) | **47.34** | **156** | 1.28 ms | SP cost ESTIMATE (the rounding carry dropped); peak 9.5 GB/GPU |
| 2026-10-25 | b02b100 | 28417702 | GH200 | SP + 31 vmap, ordered sums off | 51.32 | 144 | 1.41 ms | |
| 2026-10-25 | b02b100 | 28417702 | GH200 | **DP + 31 vmap, ordered sums on** | **67.18** | **110** | 1.6 ms vs DP 16.7 | -20 % against ordered sums off (84.42): the batched sums favour the ordered gathers in DP too |

nsys of the GH200 DP + 31 vmap step (job 28417424, kernels 88.4 ms/step): ops helpers (the FCT and Redi scatter-adds,
sums, halo glue) 33.8 ms, tracer advection 28.4 ms (0.92 ms per tracer, as for T and S), GM/Redi 9.0 ms, NCCL 8.0 ms
(68 launches), halo packing 2.4 ms, vertical diffusion 0.6 ms. No pathology: a BGC tracer costs what a T/S tracer
costs, the batched pass saves ~45 % against one at a time, and the step is memory- and scatter-bound.

BGC every k-th step (Task 2b, `recom.bgc_every`, the k-step mean transport; job 28420153, 742d5c7, one GH200
node, vmap, ordered sums on): SP + 31 k = 1 / 2 / 4: 47.32 / 26.78 / 17.22 ms/step = 156 / 276 / 430 SYPD; DP + 31
k = 2 / 4: 44.02 / 32.56 ms/step = 168 / 227 SYPD (k = 1: 67.18 = 110). Each step costs 7.6 + 39.7/k ms in SP, as
the k = 1 split predicts: the interval sums and the lax.cond cost nothing visible.

STOP 1 reading (2026-10-25): the transport of REcoM's 31 tracers alone takes CORE2 on 4 GH200 at dt 2700 from 969 to
~156 SYPD in SP (estimate) and to ~110 SYPD in DP with ordered sums, before the sources and sinks. The plan's guess of
350-450 SP assumed the batched pass would amortise far more than the measured ~45 %.

## REcoM: full REcoM on CORE2 (plan 20261025-recom-port.md, Task 15)

`configs/core2_gpu_recom.yaml` = the fast stack in double precision (full REcoM refuses single precision) + full
REcoM (bin_2p3z2d, 31 tracers, BGC from the 1-degree IC) + `forcing.mslp` + the Albani dust; 4 GH200, dt 2700, the
v3 1990 restart.

| date | commit | job | variant | ms/step | SYPD | note |
|---|---|---|---|---|---|---|
| 2026-10-26 | 77c9cd4 | 28424908 | full REcoM, k = 1 | **120.06** | **61.6** | peak 14.8 GB/GPU |
| 2026-10-26 | 77c9cd4 | 28424908 | full REcoM, `bgc_every` 4 | 47.08 | 157.1 | the every-k deviation |
| 2026-10-26 | 77c9cd4 | 28424908 | 31 passive tracers, DP (reference) | 67.09 | 110.3 | = STOP 1 |
| 2026-10-26 | b8ff45c | 28430160 | full REcoM, k = 1, one year in the production loop | 137.9 | 53.6 | 1611 s for 11680 steps with monthly output, CMDBUF on; finite; forcing year 2006 (see below) |
| 2026-10-26 | 91b8210 | 28431153 | the same year with the dt ramp (1990 forcing) | 146.2 | 50.6 | 1708 s for 11680 steps, CMDBUF on; finite |
| 2026-10-26 | 73e2171 | 28439576 | full REcoM, k = 1, DP (A of the A/B, same node) | 120.43 / 119.84 | 61.4 / 61.7 | 2 reps; peak 14.8 GB/GPU |
| 2026-10-26 | 73e2171 | 28439576 | full REcoM, k = 1, **single precision** (opt-in deviation, plan 20261026-recom-lane-sp) | **76.49 / 76.69** | **96.7 / 96.5** | 2 reps alternating with DP; SP/DP 0.64; peak 9.3 GB/GPU |
| 2026-10-26 | a4c4051 | 28446572 | **two nodes, 8 GH200** (dist_8, one process per node, `scripts/perf/time_2node_gh200.sbatch`, CMDBUF on, async writers off): physics only (`core2_gpu_fast.yaml`), padded | 7.36 | 1004.7 | against 7.39 on one node: the second node buys nothing; first try 28446432 7.49 |
| 2026-10-26 | a4c4051 | 28446572 | the same, coloured exchange | 7.95 | 930.7 | slower than padded across two nodes |
| 2026-10-26 | a4c4051 | 28446572 | two nodes, full REcoM, k = 1, DP, padded | **67.13** | **110.2** | 1.79x the one-node 120.1 ms; peak 7.3 GB/GPU; coloured 67.79 (109.1) |
| 2026-10-26 | a4c4051 | 28446572 | two nodes, full REcoM, k = 1, SP, padded | **43.09** | **171.7** | 1.78x the one-node 76.6 ms; peak 4.7 GB/GPU; coloured 43.16 (171.4) |
| 2026-10-26 | cd3ee9c | 28447462 | **one GH200 node**, one process per GPU pinned to its Grace (`proc_per_gpu.sh`) against one process: physics only | 7.13 / 7.30 | 1037 / 1013 | -2.3 %; `nvidia-smi topo`: GPU i on cores 72i..72i+71 |
| 2026-10-26 | cd3ee9c | 28447462 | the same, full REcoM DP | 120.05 / 119.95 | 61.6 / 61.7 | no gain |
| 2026-10-26 | cd3ee9c | 28447462 | one GH200 node without halo exchanges (`FESOM_HALO_NONE=1`, timing only): physics only / full REcoM DP | 6.73 / 117.08 | 1099 / 63.2 | the exchanges cost 0.57 ms (8 %) / 2.9 ms (2.4 %) on NVLink: the one-node REcoM step is computation |
| 2026-10-26 | 9373786 | 28447542-4 | **A100 scaling** (`time_nodes_a100.sbatch`, padded, one process a node, CMDBUF on), 1 / 2 / 4 nodes = 4 / 8 / 16 A100: physics only | 14.14 / 14.04 / 13.93 | 523 / 527 / 531 | without exchanges 12.37 / 8.41 / 5.36: the exchanges take 1.8 / 5.6 / 8.6 ms |
| 2026-10-26 | 9373786 | 28447542-4 | the same, full REcoM DP | 233.7 / 129.2 / 80.6 | 31.7 / 57.3 / 91.8 | without exchanges 226.5 / 117.8 / 55.6 (compute 1.92x / 4.08x); exchanges 7 / 11 / 25 ms |
| 2026-10-26 | 9373786 | 28447542-4 | the same, full REcoM SP | 187.0 / 101.7 / 61.2 | 39.6 / 72.7 / 120.9 | SP/DP 0.80 on one A100 node (0.64 on GH200) |
| 2026-10-26 | 10d28d6 | 28456284 | **one GH200 node, full REcoM**: REcoM's tracer-loop terms as per-slot columns (no full-stack rewrite per group) + the FCT limiter's node-cluster max/min as gathers (`ops.segment_extreme`, no atomic float64 max/min) | **105.3 / 105.4** | **70.2** | DP, against 119.8 (job 28455642); peak 11.9 GB/GPU (14.8); SP 66.4 (111.4 SYPD) |
| 2026-10-26 | 0445a5f | 28458068 | the same + **the BGC transport tracer-major in memory** (`FESOM_BGC_LAYOUT=major`: the stack moved to [ntr, entity, level] under a row-major `with_layout_constraint`) | **67.3 / 65.9** | **110-112** | DP; the same job's old layout 106.9 / 107.1; peak 12.6 GB/GPU |
| 2026-10-26 | 0445a5f | 28458068 | the same, single precision | **39.5 / 39.5** | **187.3** | against 76.6 at the start of the day (job 28439576): 5.3x the physics-only step |
| 2026-10-26 | 0445a5f | 28457954 | ncu on ONE GH200 (`ncu_gh200.sbatch`, P=1, DP): old layout (job 28456722) against tracer-major | kernels 530.6 -> 232.4 | | DRAM 1099 -> 584 GB a step; the HO-LO advection flux 106 -> 14.5 ms, ordered sums 91 -> 30, horizontal Redi 65 -> 22, limiter max/min 55 -> 16; the transposes themselves 19 ms at 0.9 TB/s |
| 2026-10-27 | a70b24a | 28470089-91 | **A100 scaling again** with the three fixes (1 / 2 / 4 nodes = 4 / 8 / 16 A100, as 28447542-4): physics only | 12.61 / 13.27 / 13.51 | 587 / 558 / 548 | without exchanges 11.15 / 7.21 / 5.04: the exchanges take 1.5 / 6.1 / 8.5 ms; one node -11 % (the limiter max/min) |
| 2026-10-27 | a70b24a | 28470089-91 | the same, full REcoM DP | **103.5 / 64.5 / 50.5** | **71.5 / 114.7 / 146.6** | against 233.7 / 129.2 / 80.6; without exchanges 98.8 / 51.4 / 26.5: the exchanges take 5 / 13 / 24 ms, half the 4-node step |
| 2026-10-27 | a70b24a | 28470089-91 | the same, full REcoM SP | **80.0 / 48.3 / 36.0** | **92.5 / 153.3 / 205.4** | against 187.0 / 101.7 / 61.2 |
| 2026-10-27 | 6037ca2 | 28470553 | **one GH200 node, full REcoM**: stage 3 holds the stage-1 stack and each group's T* input tracer-major (`transport.hold_major`, `FESOM_BGC_HOLD`), stage 1 reads `State.bgc` tracer-major | **62.34 / 62.39** | **118.7** | DP, against 65.41 / 65.47 in the same job (`FESOM_BGC_HOLD=0`); peak 11.5 GB/GPU (12.6) |
| 2026-10-27 | 6037ca2 | 28470553 | the same, single precision | **35.36 / 35.36** | **209.2** | against 36.28 / 36.20 (hold off; the SP carry assembled once from its columns is in both: 39.5 yesterday) |
| 2026-10-27 | 6037ca2 | 28470554 | ncu on ONE GH200 (P=1, DP) with the holds | kernels 232.4 -> 218.1 | | the 23-tracer T* copy (13.0 + 4.3 ms at 0.68 TB/s) is gone; the T* fusion 6.6 ms at 2.66 TB/s (it reads the advection and both Redi terms fused); stage 1's result stack still tracer-minor, 14.0 ms at 0.42 TB/s, then transposed |
| 2026-10-27 | 009ab2a | 28471929 | + stage 1's result stack held tracer-major (inside the wet select) | 62.36 / 62.34 | 118.7 | no change (A 65.49 / 65.43); ncu 28473168: the stack fusion still `f64[126858,47,31]{2,1,0}`, 14.0 ms at 0.42 TB/s: the constraint is a copy after it, its layout does not reach the producer |
| 2026-10-27 | b9a916e | 28473598 | instead: stage 1's 31 updated columns each its own kernel before the stack (`optimization_barrier`, with the holds) | **60.05 / 60.06** | **123.2** | DP, against 65.73 / 65.78 in the same job (`FESOM_BGC_HOLD=0`); peak 11.5 GB/GPU |
| 2026-10-27 | b9a916e | 28473598 | the same, single precision | **32.54 / 32.61** | **227.1** | against 36.32 / 36.39; 4.5x the physics-only step (7.2 ms) |
| 2026-10-27 | b9a916e | 28473599 | ncu on ONE GH200 (P=1, DP) | kernels 218.1 -> 208.3 | | stage 1's stack 14.0 -> 6.2 ms (the columns 3.7 at 1.4-1.8 TB/s, the stack 2.5); left on top: ordered sums 30.1, horizontal Redi 21.9, limiter max/min 16.2, HO-LO flux 14.4 (all at 2.6-3.0 TB/s) |
| 2026-10-27 | caf9c2d | 28476249 | full REcoM with the ordered sums OFF (`FESOM_ORDERED_SUMS=0`: atomic scatter-adds, and the limiter's max/min back to atomic `segment_max/min`) | 90.18 / 90.18 | 82.0 | DP +50 % against 60.01 / 60.06 in the same job; SP 46.33 / 46.41 against 32.57 / 32.57 (+42 %): unlike the physics in DP (+14 % for the ordered sums, job 28295452) they stay on for REcoM; the two parts not separated |
| 2026-10-27 | c4b3b18 | 28486583 | **16 A100 (4 nodes), halo transports** (`time_nodes_a100.sbatch`, one allocation): full REcoM DP, padded / coloured / no exchanges | 47.71 / **40.17** / 24.41 | 155.0 / 184.1 / 303 | the exchanges cost 23.3 ms padded, 15.8 coloured: the BGC stack's exchanges (31 tracers a call) are bound by their bytes, and padded ships 1648 rows a shard per nod exchange where coloured ships 323 |
| 2026-10-27 | c4b3b18 | 28486583 | the same, physics only, padded / coloured | **13.65** / 19.50 | 542 / 379 | the physics' small exchanges are bound by their latency: coloured's 6 rounds per exchange (6 ppermutes) lose to one all_to_all |
| 2026-10-27 | 537339e | 28487014 | **the grouped transport** (`halo_transport: grouped`, `fesom_jax/halo_grouped.py`: the coloured rounds in one NCCL group, real rows only; fesom3_jax's), 16 A100, one allocation: full REcoM DP, padded / grouped / coloured | 47.59 / **32.72** / 39.66 | 155.4 / **226.1** / 186.5 | -31 % against padded; bitwise the coloured exchange forward on 16 A100 (`scripts/perf/grouped_check.py`; its vjp to 8.9e-16, the atomic scatter-add) |
| 2026-10-27 | 537339e | 28487014 | the same, full REcoM SP, grouped | **30.53** | **242.3** | (padded SP 36.0 in job 28470091 at a70b24a) |
| 2026-10-27 | 537339e | 28487014 | the same, physics only, padded / grouped | **13.60** / 15.06 | 544 / 491 | grouped +11 %: still slower than padded on the physics' small exchanges (coloured 19.5) |
| 2026-10-27 | 81e7949 | 28489084 | 16 A100, one allocation: full REcoM DP, padded / grouped for the BGC exchanges only (`gpu.bgc_halo: grouped`) / grouped | 47.53 / 37.37 / **33.34** | 155.6 / 197.9 / **221.9** | in DP the physics' own exchanges (twice SP's bytes) gain from grouped too: every exchange grouped is best |
| 2026-10-27 | 81e7949 | 28489084 | the same, full REcoM SP, padded / grouped for the BGC exchanges only | 32.92 / **28.72** | 224.7 / **257.6** | all grouped 30.53 (job 28487014): in SP the physics' small exchanges stay faster padded |
| 2026-10-27 | 81e7949 | 28490129 | **one GH200 node** (NVLink, 1990 restart), A/B in one allocation: full REcoM DP, padded / grouped | **60.47 / 60.10** / 62.28 / 61.63 | 122.3 / 123.1 / 118.8 / 120.0 | grouped +2.8 % on one node: padded stays the one-node DP setting |
| 2026-10-27 | 81e7949 | 28490129 | the same, SP: padded / grouped for the BGC exchanges only / grouped | 32.76 / 32.69, **32.15 / 32.09**, 32.62 / 32.54 | 225.8 / 226.3, **230.1 / 230.5**, 226.8 / 227.3 | BGC-only grouped -1.8 % on one node too; the library builds and runs on aarch64 |
| 2026-10-26 | 73e2171 | 28439582 | the 1990 year with the dt ramp, DP (the SP twin's reference) | 143.8 | 51.5 | 1679 s for 11680 steps, monthly output, CMDBUF on; repeat 28439750: 1677 s, physics bitwise |
| 2026-10-26 | 73e2171 | 28439583 | the same year, **single precision** | **102.4** | **72.3** | 1196 s; SP/DP 0.71 in the production loop |

nsys of the k = 1 step (job 28430158, b8ff45c; kernels 122.3 ms/step per GPU, against 88.4 for the 31 passive
tracers, job 28417424): ops helpers 40.0 ms (the ordered sums `ops.py:139 _ordered_chain` 31.6), tracer advection
35.2 (the Zalesak limiter 24.4 of it), GM/Redi 15.4, `recom/sinking.py` 9.2 (`recom_tracer_loop_terms` 6.4),
NCCL 6.0, `recom/sms.py` 4.2, halo packing 3.6, `recom/transport.py` 2.0, `recom/wiring.py` 0.8,
`recom/carbonate.py` 0.4; an unmapped concatenate-transpose fusion 4.0. Reading: the carbonate system is cheap
(0.4 ms); REcoM's own kernels (sources/sinks, sinking, benthos, glue) are ~17 ms; the rest of the increment over the
passive profile is in the transport areas (advection +6.8, Redi +6.4, helpers +6.2; not yet attributed). The step
stays transport-bound: the levers are `bgc_every` (k = 4: 157 SYPD) and the single-precision opt-in (Task 16), not
the biology. ⚠️ The one-year job 28430160 ran with dt 2700 from the dt-1800 restart without `dt_ramp`, so its calendar
was n*2700 s from 1958 (forcing year 2006, not 1990); its timing stands, its dates do not (rerun 28430951 with
`dt: 1800` + `dt_ramp: {after_step: 561024, dt: 2700}`, as the se10 runs).
