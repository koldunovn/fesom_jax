#!/usr/bin/env python
"""JZ.7 diagnostic — CONTROLLED-REPLAY of the SSH solve vs z2_cdump (steps 1 & 2).

The multi-step CHAINED gate showed d_eta/hbar diverging ~mm at step ≥2. This isolates WHY: is
the SSH solve (incl. the zstar D2 stiffness increment) faithful with IDENTICAL inputs, or buggy?

Controlled replay = feed the C's OWN dumped inputs (not the JAX chained state):
  * step 1: solve_ssh(op, C_ssh_rhs[1], x0=0,           hbar=0)        vs C_d_eta[1]  (D2 no-op)
  * step 2: solve_ssh(op, C_ssh_rhs[2], x0=C_d_eta[1],  hbar=C_hbar[1]) vs C_d_eta[2]  (D2 LIVE)

If BOTH match to the SAME CG floor (~5e-7, the iterated near-null-space barotropic-solve
reassociation, established since Phase 2 — NOT byte-identical, by design), then the solve + the
D2 increment are FAITHFUL with identical inputs, and the chained ~mm divergence is purely the
amplification of the upstream velocity/ssh_rhs reassociation through the near-cancelling SSH
machinery (dx·helem~1e7 / areasvol·dt) — chaos, not a bug. If step 2 diverges to ~mm even with
identical inputs, the D2 increment IS wrong.

CPU (the dump is a 16-rank CPU C run). Usage: python scripts/debug/jz7_ssh_replay_check.py
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np

from fesom_jax import io_dump, ssh
from fesom_jax.mesh import load_mesh

ROOT = Path(__file__).resolve().parents[2]
MESH_DIR = ROOT / "data" / "mesh_core2"
ZORACLE = Path("/work/ab0995/a270088/port/zstar/z2_cdump/dump")
NG5_NOD2D = 126858
DT = 1800.0


def load(step, mesh):
    fsh, _ = io_dump.load_ale_dump(ZORACLE, ["sshsolve", "hbar"], step=step, n_nod=NG5_NOD2D)
    return (io_dump.ale_component(fsh, "sshsolve", "ssh_rhs").astype(np.float64),
            io_dump.ale_component(fsh, "sshsolve", "d_eta").astype(np.float64),
            io_dump.ale_component(fsh, "hbar", "hbar").astype(np.float64))


def report(tag, jx, c, clean):
    d = np.abs(np.asarray(jx, np.float64) - c)[clean]
    print(f"[replay] {tag:32s} |Δ| p50={np.percentile(d,50):.2e} p99={np.percentile(d,99):.2e} "
          f"p99.9={np.percentile(d,99.9):.2e} max={d.max():.2e}", flush=True)
    return d


def main():
    mesh = load_mesh(MESH_DIR)
    op = ssh.build_ssh_operator(mesh, dt=DT)
    nlm0 = np.asarray(mesh.node_layer_mask)[:, 0]            # wet surface nodes
    z = jnp.zeros(mesh.nod2D)

    rhs1, deta1, hbar1 = load(1, mesh)
    rhs2, deta2, hbar2 = load(2, mesh)

    # --- step 1: x0=0, hbar=0 (D2 increment is a cold-start no-op) ---
    jx1 = np.asarray(ssh.solve_ssh(op, jnp.asarray(rhs1), x0=z, mesh=mesh, hbar=z, dt=DT))
    d1 = report("step1  (x0=0, hbar=0)            ", jx1, deta1, nlm0)

    # --- step 2: x0=C_d_eta[1], hbar=C_hbar[1] (the D2 increment is LIVE) ---
    jx2 = np.asarray(ssh.solve_ssh(op, jnp.asarray(rhs2), x0=jnp.asarray(deta1),
                                   mesh=mesh, hbar=jnp.asarray(hbar1), dt=DT))
    d2 = report("step2  (x0=C_d_eta1, hbar=C_hbar1)", jx2, deta2, nlm0)

    # control: the CHAINED step-2 d_eta diff was ~7e-3 (the assembled multi-step). If the
    # controlled-replay step-2 matches at the step-1 floor, the solve+D2 are faithful and the
    # chained mm divergence is the input (velocity/ssh_rhs) amplification — not a solve bug.
    same_floor = d2.max() < 20.0 * d1.max()
    print(f"\nstep1 max={d1.max():.2e}  step2(replay) max={d2.max():.2e}  "
          f"chained step2 was ~7e-3", flush=True)
    print(f"{'PASS' if same_floor else 'FAIL'}: controlled-replay step2 is "
          f"{'at the SAME CG floor as step1 ⇒ the solve + D2 increment are FAITHFUL with identical inputs; the chained mm divergence is upstream reassociation amplified by the ill-conditioned SSH machinery (not a bug)' if same_floor else 'much larger than step1 even with identical inputs ⇒ the D2 increment / solve is WRONG'}",
          flush=True)
    return 0 if same_floor else 1


if __name__ == "__main__":
    raise SystemExit(main())
