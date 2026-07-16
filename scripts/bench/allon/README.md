# ALL-LEVERS-ON production-loop envelope (Gate 2 of docs/plans/20260716-levers-on-scaling-figure.md)

Certified config (pooled 1-yr cert PASS, jobs 26312994 + 26314418): on-device forcing +
CGPOLY k=3/κ=30. Instrument: run_from_config production loop — the kernel bench cannot see
the forcing lever (host build + H2D live outside its timed window).

Protocol per point:
- 240 steps / 48-step chunks / --progress; timing = mean ms/step over chunks 2–5 per leg
  (chunk 1 carries the XLA compile), parsed from the `[run.progress] ... host= device=` lines.
- Legs interleaved ×2: off_1, on_1, off_2, on_2. ON = `--forcing-on-device --cheb-degree 3`.
  NG5 additionally passes `--local-forcing` in BOTH legs (bit-identical host lever).
- Configs: `configs/<mesh>_full_bench.yaml` — all-on physics twins with ALL periodic output
  OFF and restart_out null (a bench must never write into a production run directory).
- dars at dt=120 (the verified-finite regime; dt=180 cold blows up within ~200 steps).
- a100_80 everywhere. Grid = the baseline envelope's (far 64/128 points join after the
  baseline far points prove healthy).
- FAIRNESS NOTE: both legs pay the same physics; the delta is host-forcing + CG solver.
