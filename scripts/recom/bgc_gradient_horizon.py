"""Phase 1 of docs/plans/20261105-recom-parameter-estimation.md: how far does the REcoM gradient reach?

Physics reads no BGC tracer, so the day-scale limit of the physics adjoint (docs/ADJOINT_HORIZON.md) need not apply to
REcoM parameters. This measures it. For each window of D days from a pi restart (``--restart``, ``--start-day``; by
default the 30-day spectral restart, a young ecosystem), with a cost taken on the LAST day only (so every window tests sensitivity across its whole length):

- ``jax.grad`` with respect to the parameters against central differences at two step sizes;
- the noise of J: the same forward repeated. The GPU forward is not bit-reproducible, and the noise bar of a central
  difference is noise / (2 h p), printed beside it. ``--xla_gpu_deterministic_ops`` is impractical (ADJOINT_HORIZON.md);
  a CPU run gives clean differences at a higher cost;
- the reverse field at the start of the window, raw ``dJ/dx0`` and normalised by the reference ``x0 * dJ/dx0`` (the
  plan's normalisation, x = xbar * xt), per tracer with the location of the raw peak. Raw growth that collapses when
  scaled is a magnitude effect at a tracer near zero; growth that survives is a real amplification;
- forward and gradient wall time.

Costs: ``chl`` area-mean surface Chl of the four groups, ``din`` area-mean surface DIN, ``npp`` area-integrated NPP.
The time loop is a two-level checkpointed scan (about 2 sqrt(N) stored states). Windows must end inside 1958 (<= 330
days), the year of the forcing stack. Appends one JSON line per (window, cost) to OUT/horizon.jsonl.

    sbatch scripts/albedo/run_bgc_horizon.sbatch --days 1 5 30 --cost chl
"""
from __future__ import annotations

import argparse
import dataclasses
import functools
import json
import sys
import time
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from fesom_jax import step as stepmod  # noqa: E402
from fesom_jax import surface_forcing  # noqa: E402
from fesom_jax.params import Params  # noqa: E402
from fesom_jax.recom import wiring  # noqa: E402
from fesom_jax.tests import test_recom_step_fork as TF  # noqa: E402

NAMES = ("expon_phy", "graz_max", "rho_N", "deg_Chl", "cdomcoeff", "k_Fe_d")   # fast and slow, light, loss, iron
RUN = "pi_1d_spectral_jaxphys"
STEPS_PER_DAY = 32
CHL = ("PhyChl", "DiaChl", "CoccoChl", "PhaeoChl")


def split(n: int) -> tuple[int, int]:
    """``(segments, length)`` with segments * length = n and length the divisor of n nearest sqrt(n)."""
    divs = [d for d in range(1, n + 1) if n % d == 0]
    length = min(divs, key=lambda d: abs(d - n ** 0.5))
    return n // length, length


def make_cost(fx, kind: str, n: int):
    mesh, dt, lay = fx["mesh"], fx["cfg"].dt, fx["lay"]
    zero = jnp.zeros((mesh.elem2D, 2))
    _, p0 = fx["cfg"].recom.load()
    a2 = mesh.areasvol[:, 0]
    k_npp = wiring.DIAG2D.index("npp")
    chl = [lay.names.index(c) for c in CHL]
    din = lay.names.index("DIN")
    n_seg, n_len = split(n)

    def metric(st):
        if kind == "npp":
            return jnp.sum(st.bgc_diag[:, k_npp] * a2)
        x = jnp.sum(st.bgc[:, 0, jnp.array(chl)], axis=-1) if kind == "chl" else st.bgc[:, 0, din]
        return jnp.sum(x * a2) / jnp.sum(a2)

    def cost(theta, bgc0, st0, stack):
        params = dataclasses.replace(Params.defaults(), recom=p0._replace(**dict(zip(NAMES, theta))))

        def one(c, xs):
            st, acc = c
            sf, w = xs
            st = stepmod.step(st, mesh, fx["op"], zero, params, dt=dt, is_first_step=False, step_forcing=sf,
                              forcing_static=fx["fs"], **fx["phys"])
            return (st, acc + w * metric(st)), None

        def segment(c, xs):
            return jax.lax.scan(jax.checkpoint(one), c, xs)[0], None

        w = jnp.where(jnp.arange(n) >= n - STEPS_PER_DAY, 1.0 / STEPS_PER_DAY, 0.0)
        xs = jax.tree.map(lambda a: a.reshape((n_seg, n_len) + a.shape[1:]), (stack, w))
        (_, j), _ = jax.lax.scan(jax.checkpoint(segment), (dataclasses.replace(st0, bgc=bgc0), jnp.zeros(())), xs)
        return j

    return cost, jnp.array([getattr(p0, k) for k in NAMES], jnp.float64)


def tangent_check(fx, st0, cf, day):
    """Is the physics out of the linearisation? Linearise one step with respect to (parameters, BGC state): the
    tangent of every State field, exact zero meaning no dependence, and one forward against one tangent evaluation."""
    mesh, dt = fx["mesh"], fx["cfg"].dt
    zero = jnp.zeros((mesh.elem2D, 2))
    _, p0 = fx["cfg"].recom.load()
    sf = jax.tree.map(lambda x: x[0], cf.stack(surface_forcing.dates_for_steps(1958, dt, 1, start_day=day)))
    theta = jnp.array([getattr(p0, k) for k in NAMES], jnp.float64)

    def f(th, bgc):
        params = dataclasses.replace(Params.defaults(), recom=p0._replace(**dict(zip(NAMES, th))))
        return stepmod.step(dataclasses.replace(st0, bgc=bgc), mesh, fx["op"], zero, params, dt=dt, is_first_step=False,
                            step_forcing=sf, forcing_static=fx["fs"], **fx["phys"])

    fj = jax.jit(f)
    jax.block_until_ready(fj(theta, st0.bgc))
    _, t_f = timed(fj, theta, st0.bgc)
    _, lin = jax.linearize(f, theta, st0.bgc)
    linj = jax.jit(lin)
    th_t, x_t = 0.01 * theta, 0.01 * st0.bgc
    out = jax.block_until_ready(linj(th_t, x_t))
    _, t_l = timed(linj, th_t, x_t)
    n_eqn = len(jax.make_jaxpr(lin)(th_t, x_t).eqns)
    live, dead = [], []
    for path, v in jax.tree_util.tree_flatten_with_path(out)[0]:
        name = jax.tree_util.keystr(path)
        if v.dtype == jax.dtypes.float0:                 # integer fields have no tangent
            continue
        (live if float(jnp.max(jnp.abs(jnp.asarray(v, jnp.float64)))) > 0 else dead).append(name)
    print(f"tangent check: forward {t_f:.3f} s, tangent {t_l:.3f} s, {n_eqn} top-level eqns\n  nonzero: {live}\n"
          f"  exactly zero: {dead}", flush=True)


def backward_scan(fx, st0, cf, day, kind, end_day, snapshots, names):
    """When does the reverse field blow up? The cost stays on the last day before ``end_day``; the gradient is taken from
    snapshots of the forward run at later and later start days, so ``|dJ/dx(s)|`` traces the reverse field back through
    the window. The first snapshot (going backward) with a huge value brackets the event."""
    mesh, dt = fx["mesh"], fx["cfg"].dt
    zero = jnp.zeros((mesh.elem2D, 2))
    n_end = end_day * STEPS_PER_DAY
    full = cf.stack(surface_forcing.dates_for_steps(1958, dt, n_end, start_day=day))
    params = Params.defaults()._replace(recom=fx["cfg"].recom.load()[1]) if hasattr(Params, "_replace") else \
        dataclasses.replace(Params.defaults(), recom=fx["cfg"].recom.load()[1])

    @jax.jit
    def advance(st, stack):
        def one(s, sf):
            return stepmod.step(s, mesh, fx["op"], zero, params, dt=dt, is_first_step=False, step_forcing=sf,
                                forcing_static=fx["fs"], **fx["phys"]), None
        return jax.lax.scan(one, st, stack)[0]

    st, done = st0, 0
    for s in snapshots:
        k = s * STEPS_PER_DAY
        st = advance(st, jax.tree.map(lambda a: a[done:k], full))
        done = k
        n = n_end - k
        stack = jax.tree.map(lambda a: a[k:], full)
        cost, theta = make_cost(fx, kind, n)
        (g, gx) = jax.block_until_ready(jax.jit(jax.grad(cost, argnums=(0, 1)))(theta, st.bgc, st, stack))
        print(f"from day {s:4d} to {end_day}: |dJ/dx| {float(jnp.sqrt(jnp.sum(gx * gx))):.3e}  scaled "
              f"{float(jnp.sqrt(jnp.sum((st.bgc * gx) ** 2))):.3e}  dJ/dlog p {np.asarray(g * theta)}", flush=True)
        tracer_report(names, st.bgc, gx, top=3)


def column_jacobian(fx, st0, cf, day, jac_days, steps_in_day, nodes, names, out, batch=32, hs=(1e-2, 1e-4, 1e-6)):
    """Which term expands the reverse field? At each (day, step of that day) of the forward run and each node: the
    one-step Jacobian of the node's own column, J = d bgc_new[n] / d bgc[n] over its wet levels and all tracers (the
    full step: sources and sinks, sinking, vertical mixing, the column's own advection; the coupling to neighbours is
    left out). Reported raw and normalised by the column itself, Jt = diag(1/xb) J diag(xb), xb = max(|x|, floor):
    spectral radius (asymptotic growth per step), largest singular value (one-step growth, non-normal), the leading
    singular vector by tracer and level, the tracer-pair blocks of Jt - I with the largest norms (which tracer drives
    which), and a central difference of the step along the leading scaled singular vector at several sizes: FD close
    to sigma means the expansion is in the model's response, FD far below sigma means a local derivative (a floor, a
    switch, a ratio near zero). Matrices go to OUT/jac_d<day>_s<step>_n<node>.npz, a summary line to OUT/jacobian.jsonl."""
    mesh, dt = fx["mesh"], fx["cfg"].dt
    zero = jnp.zeros((mesh.elem2D, 2))
    params = dataclasses.replace(Params.defaults(), recom=fx["cfg"].recom.load()[1])
    nzmax = np.asarray(mesh.nlevels_nod2D) - 1
    ntr = st0.bgc.shape[-1]
    last = max(jac_days) * STEPS_PER_DAY + max(steps_in_day) + 1
    full = cf.stack(surface_forcing.dates_for_steps(1958, dt, last, start_day=day))

    def one(s, sf):
        return stepmod.step(s, mesh, fx["op"], zero, params, dt=dt, is_first_step=False, step_forcing=sf,
                            forcing_static=fx["fs"], **fx["phys"]), None

    advance = jax.jit(lambda st, stack: jax.lax.scan(one, st, stack)[0])
    step_bgc = lambda st, bgc, sf: one(dataclasses.replace(st, bgc=bgc), sf)[0].bgc
    stepf = jax.jit(step_bgc)

    @functools.partial(jax.jit, static_argnames=("L",))
    def tangents(st, x, sf, E, n, L):                    # E: (batch, L * ntr) column tangents -> column responses
        def tan(e):
            t = jnp.zeros_like(x).at[n, :L, :].set(e.reshape(L, ntr))
            return jax.jvp(lambda b: step_bgc(st, b, sf), (x,), (t,))[1][n, :L, :].reshape(-1)
        return jax.vmap(tan)(E)
    st, done = st0, 0
    for d in jac_days:
        for s in steps_in_day:
            k = d * STEPS_PER_DAY + s
            st = advance(st, jax.tree.map(lambda a: a[done:k], full))
            done = k
            sf = jax.tree.map(lambda a: a[k], full)
            x = st.bgc
            for n in nodes:
                L = int(nzmax[n])
                m = L * ntr
                eye = np.eye(m, m + (-m) % batch).T             # padded to whole batches: one compile per L
                J = np.concatenate([np.asarray(tangents(st, x, sf, jnp.asarray(eye[i:i + batch]), n, L))
                                    for i in range(0, eye.shape[0], batch)])[:m].T   # J[out, in]
                xc = np.asarray(x[n, :L, :])
                xb = np.maximum(np.abs(xc), 1e-12 * np.maximum(np.abs(xc).max(axis=0, keepdims=True), 1e-30)).reshape(m)
                Jt = J * (xb[None, :] / xb[:, None])
                rho, rho_t = float(np.abs(np.linalg.eigvals(J)).max()), float(np.abs(np.linalg.eigvals(Jt)).max())
                U, S, Vh = np.linalg.svd(Jt)
                v = Vh[0].reshape(L, ntr)
                comp = (v ** 2).sum(axis=0)
                lev = (v ** 2).sum(axis=1)
                D = (Jt - np.eye(m)).reshape(L, ntr, L, ntr)
                blocks = np.sqrt((D ** 2).sum(axis=(0, 2)))   # [out tracer, in tracer]
                top = np.dstack(np.unravel_index(np.argsort(-blocks, axis=None)[:8], blocks.shape))[0]
                fd = {}
                y = np.asarray(x[n, :L, :])
                for h in hs:
                    dx = jnp.zeros_like(x).at[n, :L, :].set(jnp.asarray((h * xb * Vh[0]).reshape(L, ntr)))
                    dy = (np.asarray(stepf(st, x + dx, sf)[n, :L, :]) - np.asarray(stepf(st, x - dx, sf)[n, :L, :])).reshape(m)
                    fd[str(h)] = float(np.linalg.norm(dy / xb) / (2 * h))
                rec = dict(day=day + d, step=s, node=int(n), levels=L, rho=rho, rho_scaled=rho_t, sigma_scaled=S[:4].tolist(),
                           sigma_raw=float(np.linalg.norm(J, 2)), fd_along_v1=fd,
                           v1_tracers={names[i]: float(comp[i]) for i in np.argsort(-comp)[:6]},
                           v1_levels={int(i): float(lev[i]) for i in np.argsort(-lev)[:4]},
                           blocks=[(names[i], names[j], float(blocks[i, j])) for i, j in top],
                           x_v1={names[i]: float(np.abs(xc[:, i]).max()) for i in np.argsort(-comp)[:6]})
                np.savez_compressed(out / f"jac_d{day + d}_s{s}_n{n}.npz", J=J.astype(np.float32), xb=xb, x=xc)
                with open(out / "jacobian.jsonl", "a") as f:
                    f.write(json.dumps(rec) + "\n")
                print(f"day {day + d:3d} step {s:2d} node {n:4d} ({L} levels): rho {rho:.4f}  rho~ {rho_t:.4f}  "
                      f"sigma~ {S[0]:.3e}  FD along v1 " + "  ".join(f"{h}: {fd[h]:.3e}" for h in fd), flush=True)
                print("    v1: " + "  ".join(f"{k} {w:.2f}" for k, w in rec["v1_tracers"].items())
                      + "  | levels " + "  ".join(f"{k}:{w:.2f}" for k, w in rec["v1_levels"].items()), flush=True)
                print("    largest blocks of Jt - I (out <- in): " + "  ".join(f"{a}<-{b} {w:.2e}" for a, b, w in rec["blocks"]),
                      flush=True)


def tracer_report(names, bgc0, gx, top=6):
    """Per tracer: raw and reference-normalised reverse-field norms, the raw peak and the tracer's value there."""
    x, g = np.asarray(bgc0), np.asarray(gx)
    rows = []
    for i, nm in enumerate(names):
        gi, xi = g[..., i], x[..., i]
        k = np.unravel_index(int(np.argmax(np.abs(gi))), gi.shape)
        rows.append(dict(tracer=nm, raw=float(np.linalg.norm(gi)), scaled=float(np.linalg.norm(xi * gi)),
                         raw_max=float(np.abs(gi[k])), node=int(k[0]), level=int(k[1]), x_at_max=float(xi[k])))
    for r in sorted(rows, key=lambda r: -r["raw"])[:top]:
        print(f"    raw {r['raw']:.3e} scaled {r['scaled']:.3e}  {r['tracer']:>9}: peak {r['raw_max']:.3e} at node "
              f"{r['node']} level {r['level']}, x there {r['x_at_max']:.3e}", flush=True)
    return rows


def timed(fn, *a):
    t = time.time()
    r = jax.block_until_ready(fn(*a))
    return r, time.time() - t


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--days", type=int, nargs="+", default=[1, 5, 30, 90])
    ap.add_argument("--cost", nargs="+", choices=("chl", "din", "npp"), default=["chl"])
    ap.add_argument("--h", type=float, nargs="+", default=[1e-2, 3e-3], help="relative FD steps")
    ap.add_argument("--backward", type=int, nargs="+", default=None, metavar="DAY",
                    help="backward scan: END_DAY then the snapshot days (ascending); no windows are run")
    ap.add_argument("--column-jacobian", type=int, nargs="+", default=None, metavar="NODE",
                    help="one-step column Jacobians at these nodes on --jac-days (from the start); no windows are run")
    ap.add_argument("--jac-days", type=int, nargs="+", default=[150, 180, 210, 240, 270])
    ap.add_argument("--jac-steps", type=int, nargs="+", default=[0, 16], help="steps of the day (32 a day; 16 = noon)")
    ap.add_argument("--batch", type=int, default=32, help="column tangents per vmap batch")
    ap.add_argument("--repeats", type=int, default=4, help="repeats of the forward, for the noise of J")
    ap.add_argument("--restart", default=TF.RESTART, help="restart to start from")
    ap.add_argument("--start-day", type=int, default=None, help="day of 1958 of the start (default: from the restart)")
    ap.add_argument("--no-fd", action="store_true")
    ap.add_argument("--tangent-check", action="store_true", help="one-step physics-tangent check first")
    ap.add_argument("--out", type=Path,
                    default=Path("/albedo/work/projects/p_specbic/sloza/fesom_jax_runs/bgc_gradient_horizon"))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    fx = TF._setup(RUN, TF.RUN_CFG)
    st0, cf, day = TF.warm_start(fx, a.restart)
    if day <= 1:
        raise SystemExit(f"no warm restart at {a.restart}: the windows need a grown ecosystem")
    day = a.start_day or day
    print(f"start day {day} of 1958 from {a.restart}, backend {jax.default_backend()}, bgc {st0.bgc.shape}", flush=True)
    b0 = np.asarray(st0.bgc)
    print("surface means: " + "  ".join(f"{n} {b0[:, 0, i].mean():.3g}" for i, n in enumerate(fx["lay"].names)), flush=True)
    if a.tangent_check:
        tangent_check(fx, st0, cf, day)
    if a.column_jacobian:
        column_jacobian(fx, st0, cf, day, a.jac_days, a.jac_steps, a.column_jacobian, fx["lay"].names, a.out, a.batch)
        return
    if a.backward:
        backward_scan(fx, st0, cf, day, a.cost[0], a.backward[0], a.backward[1:], fx["lay"].names)
        return
    for d in a.days:
        if day + d > 361:
            raise SystemExit(f"window of {d} days from day {day} leaves 1958")
        n = d * STEPS_PER_DAY
        stack = cf.stack(surface_forcing.dates_for_steps(1958, fx["cfg"].dt, n, start_day=day))
        for kind in a.cost:
            cost, theta = make_cost(fx, kind, n)
            fwd = jax.jit(cost)
            grad = jax.jit(jax.grad(cost, argnums=(0, 1)))
            t = time.time()
            fwd_c = fwd.lower(theta, st0.bgc, st0, stack).compile()
            grad_c = grad.lower(theta, st0.bgc, st0, stack).compile()
            t_compile = time.time() - t
            j0, t_fwd = timed(fwd_c, theta, st0.bgc, st0, stack)
            (g, gx), t_grad = timed(grad_c, theta, st0.bgc, st0, stack)
            g = np.asarray(g)
            reps = [float(fwd_c(theta, st0.bgc, st0, stack)) for _ in range(a.repeats)]
            noise = float(np.std(reps + [float(j0)], ddof=1)) if a.repeats else 0.0
            rec = dict(days=d, steps=n, cost=kind, backend=jax.default_backend(), J=float(j0), names=NAMES,
                       theta=[float(x) for x in theta], grad=g.tolist(), finite=bool(np.all(np.isfinite(g))),
                       x0_norm=float(jnp.sqrt(jnp.sum(gx * gx))), x0_max=float(jnp.max(jnp.abs(gx))),
                       x0_scaled_norm=float(jnp.sqrt(jnp.sum((st0.bgc * gx) ** 2))), noise=noise, reps=reps,
                       t_compile=t_compile, t_fwd=t_fwd, t_grad=t_grad, split=split(n), fd={})
            if not a.no_fd:
                for h in a.h:
                    fd = []
                    for i in range(len(NAMES)):
                        e = jnp.zeros(len(NAMES)).at[i].set(h * theta[i])
                        fd.append((float(fwd_c(theta + e, st0.bgc, st0, stack))
                                   - float(fwd_c(theta - e, st0.bgc, st0, stack))) / (2 * float(e[i])))
                    rec["fd"][str(h)] = fd
            print(f"{d:4d} d {kind}: J {rec['J']:.6e}  noise {noise:.2e}  |dJ/dx0| {rec['x0_norm']:.3e}  "
                  f"|x0 dJ/dx0| {rec['x0_scaled_norm']:.3e}  fwd {t_fwd:.1f}s  grad {t_grad:.1f}s  "
                  f"compile {t_compile:.0f}s", flush=True)
            rec["tracers"] = tracer_report(fx["lay"].names, st0.bgc, gx)
            with open(a.out / "horizon.jsonl", "a") as f:
                f.write(json.dumps(rec) + "\n")
            for i, k in enumerate(NAMES):
                fds = "  ".join(f"FD({h}) {rec['fd'][h][i]: .6e} +- {noise / (2 * float(h) * float(theta[i])):.1e}"
                                for h in rec["fd"])
                print(f"    {k:>10}: dJ/dlog p {g[i] * float(theta[i]): .4e}  adjoint {g[i]: .6e}  {fds}", flush=True)


if __name__ == "__main__":
    main()
