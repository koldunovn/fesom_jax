"""DISCRIMINATOR for the pre-existing `test_forced_multistep_owned_matches[2]` failure
(uv owned max|Δ| = 8.969e-03 vs the test's 1e-7 clean budget; identical value on
main@ceea184 / b296ee1 / the m7-levers branch — jobs 26275895/26275933/26275876,
docs/plans/20260715-m7levers-session2-FINDINGS.md §4).

Two competing readings:
  (A) STALE CALIBRATION — the documented FCT upwind-flip floor (accepted, climate-close)
      leaks into T/S, then density -> hpressure -> pgf -> uv, and AMPLIFIES over the 3
      forced steps. uv only fails because it sits in `_owned_match`'s CLEAN (1e-7) set
      while T/S sit in the soft set.
  (B) REAL DEFECT — a halo exchange is wrong on the forced multi-step sharded path, in
      which case every multi-GPU production run is affected.

Discriminators printed here:
  1. GROWTH CURVE. Same comparison at n_steps = 1, 2, 3. (A) predicts uv at the clean
     reassociation floor (~1e-9 CPU) after ONE step and orders-of-magnitude growth per
     step. (B) predicts uv already off after one step.
  2. npes=1 CONTROL at each n. The sharded machinery collapsed to a single device must
     stay at the byte-identity floor for all n — isolating "sharding" from "shards".
  3. LOCALIZATION. How many owned entities exceed 1e-7, and the minimum hop distance
     from each offending node to a NON-owned (halo) node. A halo-wiring defect lives on
     the partition boundary; the FCT flip floor is scattered, including deep interior.
  4. CONTIGUOUS-DATE control. The test deliberately uses Jan 15 / Jul 15 / Oct 15 as
     CONSECUTIVE timesteps (so a mis-wired per-step fold is detectable), which is a
     violent forcing shock and maximises FCT flips. Re-run with three genuinely
     consecutive half-hour steps: under (A) the diffs must collapse, under (B) not.

Absolute AND per-field relative diffs are reported: `_owned_match` compares absolute
across fields whose scales span ~9 orders (ssh_rhs is O(1e6)), which is why several
fields breach their budgets while being relatively tiny.

    XLA_FLAGS=--xla_force_host_platform_device_count=4 python scripts/debug/forced_owned_step_growth.py
"""
from __future__ import annotations

import dataclasses
from collections import deque
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from fesom_jax import surface_forcing, partit, shard_mesh, ssh
from fesom_jax import integrate as integ
from fesom_jax import integrate_sharded as ish
from fesom_jax.kpp import KppConfig
from fesom_jax.mesh import load_mesh
from fesom_jax.phc_ic import phc_initial_state
from fesom_jax.state import State

ROOT = Path(__file__).resolve().parents[2]
CORE2_MESH = ROOT / "data" / "mesh_core2"
IC_DIR = ROOT / "data" / "ic_core2"
CORE2_DIST = Path("/pool/data/AWICM/FESOM2/MESHES_FESOM2.1/core2")
DT = 1800.0
YEAR = 1958
NPES = 2

# the test's fields classes (test_step_sharded._FCT_FIELDS / _DIAG_FIELDS)
_FCT_FIELDS = {"T", "S", "T_old", "S_old", "ssh_rhs", "ssh_rhs_old", "del_ttf"}
_DIAG_FIELDS = {"sigma11", "sigma12", "sigma22"}

SEPARATED = [(YEAR, 15, 0.0, 1), (YEAR, 196, 0.0, 7), (YEAR, 288, 0.0, 10)]
CONTIGUOUS = [(YEAR, 15, i * DT, 1) for i in range(3)]


# ----------------------------------------------------------------- mesh topology
def _hop_distance_to_halo(mesh, part, dev):
    """For device `dev`: BFS hop distance (over mesh-edge adjacency) from every OWNED
    global node to the nearest node this device does NOT own. 0 = the node itself is on the
    owned/halo interface. Returns dict {global node -> hops}, only for owned nodes."""
    ed = np.asarray(mesh.edges)                             # [edge2D, 2] node ids
    md = int(part.myDim_nod2D[dev])
    owned = set(int(x) for x in part.myList_nod2D[dev][:md])
    nbr = [[] for _ in range(mesh.nod2D)]
    for a, b in ed:
        a, b = int(a), int(b)
        nbr[a].append(b)
        nbr[b].append(a)
    dist = {}
    dq = deque()
    for n in owned:                       # seed: owned nodes touching a non-owned node
        if any(m not in owned for m in nbr[n]):
            dist[n] = 0
            dq.append(n)
    while dq:
        n = dq.popleft()
        for m in nbr[n]:
            if m in owned and m not in dist:
                dist[m] = dist[n] + 1
                dq.append(m)
    return dist


# ----------------------------------------------------------------- diff reporting
def _owned_pairs(st_dense, st_N, mesh, part, npes, fld):
    """(dense_owned, sharded_owned, per-device global index arrays) for one field."""
    a = np.asarray(getattr(st_dense, fld))
    B = np.asarray(getattr(st_N, fld))
    if a.shape[0] == mesh.nod2D:
        mydim, myl, kind = part.myDim_nod2D, part.myList_nod2D, "nod"
    elif a.shape[0] == mesh.elem2D:
        mydim, myl, kind = part.myDim_elem2D, part.myList_elem2D, "elem"
    else:
        return None
    da, db, idx = [], [], []
    for d in range(npes):
        md = int(mydim[d])
        if not md:
            continue
        gi = np.asarray(myl[d][:md])
        da.append(a[gi])
        db.append(B[d, :md])
        idx.append((d, gi))
    return np.concatenate(da), np.concatenate(db), idx, kind


def _report(st_dense, st_N, mesh, part, npes, tag):
    """Absolute + relative owned max|Δ| per field, with the test's budget verdict."""
    print(f"\n[{tag}] per-field owned diffs (npes={npes}):")
    print(f"   {'':4s}{'field':14s} {'abs max|Δ|':>11s} {'rel':>10s} {'scale':>11s}  budget")
    rows = []
    for f in dataclasses.fields(State):
        got = _owned_pairs(st_dense, st_N, mesh, part, npes, f.name)
        if got is None:
            continue
        a, b, _, _ = got
        if not a.size:
            continue
        ad = float(np.max(np.abs(a - b)))
        scale = float(np.max(np.abs(a))) or 1.0
        rows.append((f.name, ad, ad / scale, scale))
    for name, ad, rel, scale in sorted(rows, key=lambda r: -r[1]):
        if ad == 0.0:
            continue
        cls = "DIAG" if name in _DIAG_FIELDS else ("FCT " if name in _FCT_FIELDS else "    ")
        budget = 1e-1 if name in _FCT_FIELDS else 1e-7           # the test's own numbers
        verdict = "ok" if (ad < budget or name in _DIAG_FIELDS) else "OVER"
        print(f"   {cls}{name:14s} {ad:11.3e} {rel:10.2e} {scale:11.3e}  {verdict}")
    return {n: (ad, rel) for n, ad, rel, _ in rows}


def _localize(st_dense, st_N, mesh, part, npes, fld, hopmaps, atol=1e-7):
    """Where do the >atol entries of `fld` live relative to the partition boundary?"""
    got = _owned_pairs(st_dense, st_N, mesh, part, npes, fld)
    if got is None:
        return
    a, b, idx, kind = got
    diff = np.abs(a - b)
    if diff.ndim > 1:
        diff = diff.reshape(diff.shape[0], -1).max(axis=1)
    bad = diff > atol
    n_tot, n_bad = diff.size, int(bad.sum())
    print(f"   [{fld}] {n_bad}/{n_tot} owned entities over {atol:.0e} "
          f"({100.0 * n_bad / max(n_tot, 1):.3f} %)")
    if n_bad == 0:
        return
    # element fields inherit the hop depth of their SHALLOWEST vertex (an element is as
    # close to the halo as its nearest boundary node)
    en = np.asarray(mesh.elem_nodes) if kind == "elem" else None
    off = 0
    hops_bad = []
    for d, gi in idx:
        seg = bad[off:off + len(gi)]
        off += len(gi)
        hm = hopmaps[d]
        for g in gi[seg]:
            if en is None:
                hops_bad.append(hm.get(int(g), -1))
            else:
                hs = [hm.get(int(v), -1) for v in en[int(g)]]
                hs = [h for h in hs if h >= 0]
                hops_bad.append(min(hs) if hs else -1)
    hops_bad = np.array(hops_bad)
    if hops_bad.size:
        uniq, cnt = np.unique(np.clip(hops_bad, -1, 8), return_counts=True)
        hist = " ".join(f"h{int(u)}:{int(c)}" for u, c in zip(uniq, cnt))
        print(f"      hop-to-halo histogram (h8 = 8 or deeper): {hist}")
        print(f"      max hop depth of an offending node: {int(hops_bad.max())}")


# ----------------------------------------------------------------- runs
def _dense(fx, stack, n):
    mesh = fx["mesh"]
    return integ.integrate(fx["state"], mesh, fx["op"], jnp.zeros((mesh.elem2D, 2)), n,
                           dt=DT, step_forcings=stack, forcing_static=fx["fs"],
                           kpp_cfg=KppConfig())


def _sharded(fx, part, npes, stack, n):
    mesh = fx["mesh"]
    sm = shard_mesh.build_sharded_mesh(mesh, part)
    state_p = shard_mesh.partition_state(fx["state"], part)
    sop = ssh.partition_ssh_operator(fx["op"], part)
    seq_p = shard_mesh.partition_step_forcing(stack, part)
    fs_p = shard_mesh.partition_forcing_static(fx["fs"], part)
    stress_p = jnp.zeros((npes, sm.Lmax["elem"], 2))
    return ish.run_steps_sharded_forced(sm, state_p, sop, stress_p, seq_p, fs_p, n,
                                        dt=DT, npes=npes, kpp_cfg=KppConfig())


def main():
    print(f"devices: {jax.devices()}")
    mesh = load_mesh(CORE2_MESH)
    state = phc_initial_state(mesh, IC_DIR)
    op = ssh.build_ssh_operator(mesh, dt=DT)
    cf = surface_forcing.build_surface_forcing(mesh, YEAR,
                                               sst_ic=np.asarray(state.T[:, 0]))
    fx = dict(mesh=mesh, state=state, op=op, fs=cf.static)

    part2 = partit.read_partition(CORE2_DIST, NPES)
    ser = partit.synth_serial(mesh.nod2D, mesh.elem2D, mesh.edge2D)
    print("building hop-to-halo maps ...", flush=True)
    hopmaps = {d: _hop_distance_to_halo(mesh, part2, d) for d in range(NPES)}
    for d in range(NPES):
        hm = hopmaps[d]
        print(f"   dev {d}: {len(hm)} owned nodes, max interior depth "
              f"{max(hm.values()) if hm else 0}")

    for label, dates in (("separated(Jan/Jul/Oct)", SEPARATED),
                         ("contiguous(3x30min)", CONTIGUOUS)):
        stack_full = cf.stack(dates, xp=np)
        for n in (1, 2, 3):
            stack = jax.tree.map(lambda x: jnp.asarray(x[:n]), stack_full)
            d = _dense(fx, stack, n)
            print(f"\n{'=' * 78}\n=== {label}  n_steps={n}\n{'=' * 78}", flush=True)

            s1 = _sharded(fx, ser, 1, stack, n)
            _report(d, s1, mesh, ser, 1, f"{label} n={n} npes=1 CONTROL")

            s2 = _sharded(fx, part2, NPES, stack, n)
            r = _report(d, s2, mesh, part2, NPES, f"{label} n={n} npes=2")
            for fld in ("uv", "uvnode", "T", "S", "density"):
                _localize(d, s2, mesh, part2, NPES, fld, hopmaps)
            print(f"   >>> SUMMARY {label} n={n}: "
                  f"uv abs={r['uv'][0]:.3e} rel={r['uv'][1]:.2e} | "
                  f"T abs={r['T'][0]:.3e} rel={r['T'][1]:.2e}", flush=True)


if __name__ == "__main__":
    main()
