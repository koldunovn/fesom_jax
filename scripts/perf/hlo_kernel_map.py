#!/usr/bin/env python
"""CORE2 GPU speed plan, Task 1: map the kernels of an nsys (or ncu) kernel summary to the fesom_jax
source lines and model areas that made them (adapted from fesom3-jax's scripts/hlo_kernel_map.py).

    hlo_kernel_map.py <after_optimizations.txt> <kern_sum.csv> [--steps S] [--by kernel|line|function|file|area]
                      [--cg-iters N] [--mevp-sub 120] [--top 30]
    hlo_kernel_map.py <after_optimizations.txt> <ncu raw.csv> --ncu --steps 1 --by area

The optimised HLO (``--xla_dump_to=<dir> --xla_dump_hlo_as_text``, the *_gpu_after_optimizations.txt
of the chunk) names each fusion like its kernel (``input_scatter_fusion.33`` -> kernel
``input_scatter_fusion_33``) and carries a stack_frame_id; its FileNames / FunctionNames /
FileLocations / StackFrames tables give the Python call chain. nsys sums the kernels of every GPU,
so ``--steps`` = steps x GPUs gives one GPU's share per step. Profile with
``--cuda-graph-trace=node``, or the kernels inside CUDA graphs are missing from the summary.

Kernels made inside the ``ops.py`` helpers (``tdma``, scatters, sums) are credited to the model
function that called the helper. A helper kernel with no model frame at all (JAX traces custom
derivatives outside the caller's stack) is placed by its launches per step: a multiple of the CG
iterations (``--cg-iters``) is the SSH solve, a multiple of the mEVP subcycles (``--mevp-sub``) the
sea-ice dynamics, anything else "ops helpers once per step".
"""

from __future__ import annotations

import argparse
import collections
import csv
import re
from dataclasses import dataclass, field

# model file -> area (first match wins)
AREAS = (
    ("ssh.py", "SSH (CG solve, rhs)"),
    ("ice_mevp.py", "sea ice dynamics (mEVP)"), ("ice_evp.py", "sea ice dynamics (mEVP)"),
    ("ice_", "sea ice thermo, advection, coupling"), ("ice.py", "sea ice thermo, advection, coupling"),
    ("tracer_adv.py", "tracer advection"),
    ("tracer_diff.py", "tracer vertical diffusion"),
    ("gm.py", "GM/Redi"), ("gm_redi.py", "GM/Redi"),
    ("cvmix_tke.py", "TKE"), ("tke.py", "TKE"),
    ("kpp.py", "KPP/PP mixing"), ("pp.py", "KPP/PP mixing"),
    ("momentum.py", "momentum"),
    ("pgf.py", "pressure, density"), ("eos.py", "pressure, density"),
    ("ale.py", "ALE (zstar), vertical velocity"),
    ("jra55.py", "forcing, fluxes"), ("forcing", "forcing, fluxes"), ("surface_forcing.py", "forcing, fluxes"),
    ("sss_runoff.py", "forcing, fluxes"),
    ("halo", "halo exchange (packing)"), ("shard_mesh.py", "halo exchange (packing)"),
    ("integrate", "sharded driver"),
    ("step.py", "step (other)"),
)
HELPERS = ("ops.py", "reductions.py")          # credited to their caller
# Pallas kernels by function name (Task 4 adds its kernels here)
PALLAS: dict = {}

_FUSION_RE = re.compile(r"^\s*(?:ROOT )?%([\w.\-]+) = (.*?) (fusion|custom-call)\(.*?stack_frame_id=(\d+)")
_DEDUP_RE = re.compile(r'deduplicated_name="([^"]+)"')


@dataclass
class Tables:
    files: dict = field(default_factory=dict)
    funcs: dict = field(default_factory=dict)
    locs: dict = field(default_factory=dict)
    frames: dict = field(default_factory=dict)
    inst: dict = field(default_factory=dict)        # HLO instruction -> (shape, stack frame)
    dedup: dict = field(default_factory=lambda: collections.defaultdict(list))
    pallas: list = field(default_factory=list)


def parse_hlo(path) -> Tables:
    t, table = Tables(), None
    with open(path) as f:
        for line in f:
            s = line.rstrip("\n")
            if s in ("FileNames", "FunctionNames", "FileLocations", "StackFrames"):
                table = s
                continue
            if table and (not s or not s[0].isdigit()):
                table = None
            if table == "FileNames":
                i, name = s.split(" ", 1)
                t.files[int(i)] = name.strip('"')
            elif table == "FunctionNames":
                i, name = s.split(" ", 1)
                t.funcs[int(i)] = name.strip('"')
            elif table == "FileLocations":
                i, rest = s.split(" ", 1)
                kv = dict(re.findall(r"(\w+)=(\d+)", rest))
                t.locs[int(i)] = (int(kv["file_name_id"]), int(kv["function_name_id"]), int(kv["line"]))
            elif table == "StackFrames":
                i, rest = s.split(" ", 1)
                kv = dict(re.findall(r"(\w+)=(\d+)", rest))
                t.frames[int(i)] = (int(kv["file_location_id"]), int(kv["parent_frame_id"]))
            elif table is None and (" fusion(" in s or " custom-call(" in s):
                m = _FUSION_RE.match(s)
                if m:
                    t.inst[m.group(1)] = (m.group(2), int(m.group(4)))
                    d = _DEDUP_RE.search(s)
                    if d:
                        t.dedup[d.group(1)].append(m.group(1))
                    if m.group(3) == "custom-call":
                        t.pallas.append(m.group(1))
    return t


def chain(t: Tables, fid) -> list:
    """The fesom_jax frames of a stack, innermost first, as ``file.py:line function``."""
    out, seen = [], set()
    while fid in t.frames and fid not in seen:
        seen.add(fid)
        loc, parent = t.frames[fid]
        fi, fu, ln = t.locs[loc]
        path = t.files.get(fi, "?")
        if "/fesom_jax/" in path:
            out.append(f"{path.rsplit('/fesom_jax/', 1)[1]}:{ln} {t.funcs.get(fu, '?')}")
        if parent == fid:
            break
        fid = parent
    return out


def area_of(fname: str) -> str:
    return next((label for pre, label in AREAS if fname.startswith(pre)), fname)


def caller(ch: list, by: str) -> str:
    """The model function that made a kernel: the innermost frame outside the helper modules."""
    for fr in ch:
        where, fn = fr.split(" ", 1)
        f = where.split(":")[0]
        if f in HELPERS:
            continue
        if by == "file":
            return f
        if by == "area":
            return area_of(f)
        return f"{f} {fn}"
    return "(ops only) " + (ch[0] if ch else "?")


def by_count(key, name, n, cg_iters, mevp_sub) -> str:
    stem = name.rstrip("_0123456789")
    if stem in PALLAS:
        return PALLAS[stem]
    if not (key.startswith("(ops only)") or key.startswith("(unmapped")):
        return key
    for iters, label in ((cg_iters, "SSH (CG solve, rhs)"), (mevp_sub, "sea ice dynamics (mEVP)")):
        if iters and n >= 0.8 * iters and abs(n / iters - round(n / iters)) < 0.08:
            return label
    return "ops helpers once per step (scatters, sums, halos)"


def aggregate(rows, t: Tables, steps, by, cg_iters=None, mevp_sub=None) -> dict:
    """key -> [ms per step, launches per step, kernel names, DRAM bytes per step]. A kernel made at
    several places has its time split evenly among them."""
    agg = collections.defaultdict(lambda: [0.0, 0.0, set(), 0.0])
    for r in rows:
        name = r["Name"]
        ms = float(r["Total Time (ns)"]) / 1e6 / steps
        n = int(r["Instances"]) / steps
        by_bytes = float(r.get("Bytes", 0.0)) / steps
        hlo_name = re.sub(r"_(\d+)$", r".\1", name)
        if by == "kernel":
            shape, fid = t.inst.get(hlo_name, t.inst.get(name, ("?", None)))
            ch = chain(t, fid) if fid is not None else []
            keys = [ch[0] if ch else f"(no frame) {name}"]
        elif name.startswith("nccl"):
            keys = ["(NCCL)"]
        else:
            cands = [hlo_name] if hlo_name in t.inst else t.dedup.get(name, []) or t.dedup.get(hlo_name, [])
            if not cands and not name.startswith(("loop_", "input_", "wrapped_", "copy", "transpose")):
                stem = name.rstrip("_0123456789")       # a Pallas kernel: its custom-calls by frame
                cands = [c for c in t.pallas if any(f" {stem}" in fr for fr in chain(t, t.inst[c][1]))]
            if by == "line":
                keys = [(chain(t, t.inst[c][1]) or ["?"])[0] for c in cands]
            else:
                keys = [caller(chain(t, t.inst[c][1]), by) for c in cands]
            keys = keys or ["(unmapped: " + re.sub(r"[_.\d]+$", "", name) + ")"]
            if by == "area":
                keys = [by_count(k, name, n, cg_iters, mevp_sub) for k in keys]
        for k in keys:
            agg[k][0] += ms / len(keys)
            agg[k][1] += n / len(keys)
            agg[k][2].add(name)
            agg[k][3] += by_bytes / len(keys)
    return dict(agg)


def ncu_rows(path):
    """Nsight Compute's raw CSV (row 0 metric names, row 1 units, then one row per launch) as
    nsys-style rows per kernel name, with the DRAM bytes ("Bytes") summed over the launches."""
    raw = list(csv.reader(open(path)))
    head, units = raw[0], raw[1]
    ix = {k: i for i, k in enumerate(head)}
    scale = {"ns": 1.0, "nsecond": 1.0, "us": 1e3, "usecond": 1e3, "ms": 1e6, "msecond": 1e6, "byte": 1.0,
             "Kbyte": 1e3, "Mbyte": 1e6, "Gbyte": 1e9, "Tbyte": 1e12}

    def val(r, k):
        x = r[ix[k]].replace(",", "")
        return float(x) * scale[units[ix[k]]] if x else 0.0
    agg = collections.defaultdict(lambda: [0.0, 0, 0.0])
    for r in raw[2:]:
        g = agg[r[ix["Kernel Name"]]]
        g[0] += val(r, "gpu__time_duration.sum")
        g[1] += 1
        g[2] += val(r, "dram__bytes_read.sum") + val(r, "dram__bytes_write.sum")
    return [{"Name": n, "Total Time (ns)": t, "Instances": c, "Bytes": b} for n, (t, c, b) in agg.items()]


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("hlo")
    p.add_argument("kern_csv")
    p.add_argument("--steps", type=int, default=96, help="profiled steps x GPUs")
    p.add_argument("--top", type=int, default=30)
    p.add_argument("--by", choices=("kernel", "line", "function", "file", "area"), default="area")
    p.add_argument("--cg-iters", type=float, default=None, help="--by area: SSH CG iterations per step")
    p.add_argument("--mevp-sub", type=float, default=120, help="--by area: mEVP subcycles per step")
    p.add_argument("--ncu", action="store_true", help="kern_csv is an Nsight Compute raw CSV")
    a = p.parse_args()
    t = parse_hlo(a.hlo)
    rows = ncu_rows(a.kern_csv) if a.ncu else list(csv.DictReader(open(a.kern_csv)))
    tot = sum(float(r["Total Time (ns)"]) for r in rows) / 1e6 / a.steps
    print(f"kernels {tot:.2f} ms/step; {len(t.inst)} fusions with a stack frame in the HLO")
    if a.ncu:
        gb = sum(r["Bytes"] for r in rows) / 1e9 / a.steps
        big = [r["Bytes"] / r["Total Time (ns)"] for r in rows if r["Total Time (ns)"] / r["Instances"] > 50e3]
        print(f"DRAM {gb:.2f} GB/step, {gb / tot:.2f} TB/s on average; best kernel over 50 us: "
              f"{(max(big) / 1e3 if big else float('nan')):.2f} TB/s")
    agg = aggregate(rows, t, a.steps, a.by, a.cg_iters, a.mevp_sub)
    for key, (ms, n, names, by) in sorted(agg.items(), key=lambda kv: -kv[1][0])[:a.top]:
        bw = f" {by / 1e9:6.2f} GB {by / (ms * 1e9) if ms else 0:5.2f} TB/s" if a.ncu else ""
        print(f"{ms:7.2f} ms ({100 * ms / tot:4.1f} %) {n:7.0f} launches/step {len(names):3d} kernels{bw}  {key}")


if __name__ == "__main__":
    main()
