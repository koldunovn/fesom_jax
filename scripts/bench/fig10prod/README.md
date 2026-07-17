# fig10 v2 — PRODUCTION-PHYSICS kernel scaling campaign (user go 2026-07-17)

Replaces the uniform-benchmark-model fig10 data: each mesh now runs its 63-year-class
production physics, on the current code (all BIT-IDENTICAL optimizations in: clamp fix,
trig cache, fused exchanges/reductions), with the non-bit-identical levers OFF
(on_device forcing OFF, CGPOLY OFF — same class as the 63-yr runs).

Physics per mesh (bench_forward_scaling flags):
- core2:  --ice 1 --mevp 1 --tke 1 --kpp 0 --gm 1 --zstar 1  (dt=1800) = the exact hindcast set
- farc/dars/forca20/ng5: --ice 1 --mevp 1 --tke 1 --kpp 0 --gm 0 --zstar 1 (GM off in their
  production configs; NG5 mixing = TKE per user 2026-07-17 — ng5.yaml's KPP was stale)
- dt: farc 900 · dars 120 (verified-finite bench regime) · forca20 240 · ng5 180

Protocol otherwise = scripts/bench/remeasure/README.md: 150 steps, x2 reps, a100_80,
per-point banked-best transport, bench-finite gate per row, logs to scripts/logs/bench_pr_*.out
(inside the reducer glob). Model strings become FULL(mevp+tke[+gm]+zstar+JRA1958) —
reducer/figure selection must switch from the ice+kpp+gm filter to these (paper pass).

The UNIFORM-model envelope (bench_rm_*, 2026-07-16) stays banked: it is the like-for-like
basis for the Kokkos code-twin comparison (their harness runs their uniform config too);
where that comparison lives in the paper is decided when their post-M7 data arrives.
The pending uniform far points (26312903/05/81) were CANCELLED unstarted; far points rerun
here under production physics.
