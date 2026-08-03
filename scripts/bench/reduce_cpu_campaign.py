"""Reduce the CPU strong-scaling campaign logs to one table per mesh.

    python scripts/bench/reduce_cpu_campaign.py [logdir]

Reads ``cpu_camp_*.out`` (and the gate/sweep logs, which share the ``[bench]`` line format),
pairing each result with the allocation recorded in its campaign header. Rows that died —
OOM, gloo abort, walltime — are reported as such rather than silently dropped, so a table is
never mistaken for full coverage.
"""
from __future__ import annotations

import re
import sys
from collections import defaultdict
from pathlib import Path

LOGS = Path(sys.argv[1] if len(sys.argv) > 1 else "/home/a/a270088/port_jax/scripts/logs")

HDR = re.compile(r"CPU CAMPAIGN\s+mesh=(\S+)\s+npes=(\d+)\s+nodes=(\d+)\s+(\d+)procs/node x (\d+)cores")
# plat=cpu is load-bearing: scripts/logs also holds the GPU campaigns, whose [bench] lines
# are otherwise identical in format.
RES = re.compile(r"\[bench\] mesh=(\S+)\s+nod2D=(\d+)\s+nl=(\d+)\s+npes=(\d+)\s+halo=(\S+)\s+"
                 r".*?per_step=\s*([\d.]+) ms\s+throughput=\s*([\d.]+) Mnodlev/s\s+compile=\s*([\d.]+)s"
                 r".*?plat=cpu")
FIN = re.compile(r"nonfinite_T=(\d+) nonfinite_uv=(\d+) max_uv=([\d.]+)")
DEAD = {"oom": re.compile(r"OOM|oom_kill|signal 9|Out Of Memory", re.I),
        "gloo": re.compile(r"Gloo collective permute failed"),
        "mpi": re.compile(r"MPI: Communicator requested"),
        "time": re.compile(r"DUE TO TIME LIMIT|CANCELLED AT")}

# Only this campaign's logs. scripts/logs also holds the 2026-07-13 CPU study (bench_cpu_*,
# bench_multiproc_*, …), whose [bench] lines are plat=cpu too and would silently merge in.
PREFIXES = ("cpu_camp_", "gate_cpu_", "cpu_topo_sweep")

# Each campaign leg writes twice: the job output (header + last lines) and a full log keyed by
# the same job id. Read the full log where it exists — the job output's tail keeps the abort
# stack but not the error message, which misfiles gloo failures as "no result" — and never
# count both, or every result appears twice.
FULL_BY_JOB = {p.name.rsplit(".", 2)[-2]: p for p in LOGS.glob("full_*.out")}

rows, dead = [], []
for f in sorted(p for pre in PREFIXES for p in LOGS.glob(f"{pre}*.out")):
    head_txt = f.read_text(errors="replace")
    job = f.name.rsplit(".", 2)[-2]
    # The header only ever appears in the job output; results and errors come from the full
    # log when there is one (the job output holds a truncated copy of the same lines).
    txt = FULL_BY_JOB[job].read_text(errors="replace") if job in FULL_BY_JOB else head_txt
    hdr = HDR.search(head_txt)
    mesh_h, nodes, ppn, cores = (hdr.group(1), int(hdr.group(3)), int(hdr.group(4)),
                                 int(hdr.group(5))) if hdr else (None, None, None, None)
    hits = list(RES.finditer(txt))
    for m in hits:
        fin = FIN.search(txt[m.start():m.start() + 2000]) or FIN.search(txt)
        rows.append(dict(mesh=m.group(1), nod2D=int(m.group(2)), npes=int(m.group(4)),
                         halo=m.group(5), ms=float(m.group(6)), mnl=float(m.group(7)),
                         compile_s=float(m.group(8)), nodes=nodes, ppn=ppn, cores=cores,
                         maxuv=(fin.group(3) if fin else "?"), job=f.name))
    if not hits and hdr:
        why = next((k for k, r in DEAD.items() if r.search(txt)), None)
        if why is None:
            why = "running" if "=== done" not in txt else "no-result"
        dead.append(dict(mesh=mesh_h, nodes=nodes, ppn=ppn, why=why, job=f.name))

by_mesh = defaultdict(list)
for r in rows:
    by_mesh[r["mesh"]].append(r)

DAY = 86400.0
for mesh in sorted(by_mesh, key=lambda m: by_mesh[m][0]["nod2D"]):
    rs = sorted(by_mesh[mesh], key=lambda r: (r["npes"], r["halo"]))
    print(f"\n### {mesh}  (nod2D={rs[0]['nod2D']})")
    print(f"{'nodes':>5} {'ranks':>6} {'r/node':>7} {'cores':>6} {'halo':>9} "
          f"{'ms/step':>9} {'Mnodlev/s':>10} {'SYPD':>7} {'effic':>8} {'compile':>8} {'max_uv':>7}")
    base = None
    for r in rs:
        # SYPD = simulated years per wall day: dt seconds of model per step, so
        # dt / step_seconds sim-seconds per wall second -> / 365.25 gives years per day.
        dt = {"core2": 1800, "farc": 900}.get(mesh, 180)   # dars/ng5 run dt=180
        sypd = dt / (r["ms"] / 1000.0) / 365.25
        if r["halo"] == "coloured" and base is None and r["nodes"]:
            base = (r["nodes"], r["ms"])
        sp = f"{base[1] / r['ms'] * base[0] / r['nodes']:.2f}x" if (base and r["nodes"]) else "-"
        print(f"{str(r['nodes'] or '?'):>5} {r['npes']:>6} {str(r['ppn'] or '?'):>7} "
              f"{str(r['cores'] or '?'):>6} {r['halo']:>9} {r['ms']:>9.1f} {r['mnl']:>10.1f} "
              f"{sypd:>7.2f} {sp:>8} {r['compile_s']:>7.0f}s {r['maxuv']:>7}")
    print("  speedup = parallel efficiency vs the smallest coloured rung (1.00x = ideal scaling)")

if dead:
    print("\n### legs that did NOT produce a number")
    for d in dead:
        print(f"  {str(d['mesh']):>6} nodes={str(d['nodes']):>3} ranks/node={str(d['ppn']):>3} "
              f"-> {d['why']:<8} ({d['job']})")
print(f"\n{len(rows)} results, {len(dead)} failed legs, from {LOGS}")
