"""REcoM plan Task 2b on CORE2: the error of ``recom.bgc_every = k`` against k = 1 over one month.

Runs from the same restart and forcing under one base directory, compared in named pairs ``name:A:B`` (A against
B), e.g. ``floor:k1rep:k1`` (k = 1 twice: the GPU's run-to-run floor) and ``kerr:k4:k1``. For the monthly means
(``<run>/monthly/<YYYY_MM>``) and the end states (``<run>/restart``, the canonical global layout): per field the
relative difference of the global integral (inventory or flux) and the pointwise ``max |a - b| / max |b|`` and
``rms(a - b) / rms(b)`` over the wet cells; T and S as the physics check (the BGC tracers do not feed back, so the
physics of a k = 4 run equals the k = 1 run's up to the floor).

    python scripts/recom/kerr_compare.py <base> floor:k1rep:k1 kerr:k4:k1new [--mesh data/mesh_core2]

Writes ``<base>/kerr.json``; prints the table.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
F3 = ("DIC", "Alk", "DIN", "DSi", "DFe", "O2", "Chl", "temp", "salt")
F2 = ("co2flux", "pco2surf", "npp", "export100")
# the end-state BGC slots (fesom_jax.recom.tracers.NAMES order) reported, and T/S
END = {"DIN": 0, "DIC": 1, "Alk": 2, "PhyC": 4, "DetC": 7, "DiaC": 13, "DSi": 17, "DFe": 18, "O2": 21,
       "Zoo2C": 23, "DetZ2C": 25}


def _stats(a, b, w, wet):
    a, b = np.where(wet, a, 0.0), np.where(wet, b, 0.0)
    ia, ib = float(np.sum(a * w)), float(np.sum(b * w))
    d = a - b
    return {"integral": ia / ib - 1.0 if ib != 0 else float("nan"),
            "max": float(np.max(np.abs(d)) / max(np.max(np.abs(b)), 1e-300)),
            "rms": float(np.sqrt(np.sum(d * d * w) / max(np.sum(b * b * w), 1e-300)))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base", type=Path)
    ap.add_argument("pairs", nargs="+", help="name:A:B, A compared against B")
    ap.add_argument("--mesh", type=Path, default=REPO / "data" / "mesh_core2")
    args = ap.parse_args()
    import zarr

    m = args.mesh
    area = np.load(m / "area.npy")[:, 0]
    areasvol = np.load(m / "areasvol.npy")
    zbar = np.load(m / "zbar.npy")
    nlev = np.load(m / "nlevels_nod2D.npy").astype(int)
    dz = zbar[:-1] - zbar[1:]
    nl1 = dz.size
    wet3 = np.arange(nl1)[None, :] < (nlev - 1)[:, None]
    vol = dz[None, :] * areasvol[:, :nl1]
    pairs = [tuple(p.split(":")) for p in args.pairs]
    names = sorted({r for _, a, b in pairs for r in (a, b)})
    out = {"monthly": {}, "end": {}}

    month = sorted(p.name for p in (args.base / pairs[0][2] / "monthly").iterdir())[-1]
    g = {r: zarr.open_group(str(args.base / r / "monthly" / month), mode="r") for r in names}
    rd = lambda grp, f, two: (np.asarray(grp[f][0], dtype=np.float64) if two                   # noqa: E731
                              else np.asarray(grp[f][0], dtype=np.float64).T[:, :nl1])
    for f in F3 + F2:
        two = f in F2
        w, wet = (area, np.ones_like(area, bool)) if two else (vol, wet3)
        out["monthly"][f] = {n: _stats(rd(g[a], f, two), rd(g[b], f, two), w, wet) for n, a, b in pairs}

    rs = {r: zarr.open_group(str(args.base / r / "restart"), mode="r") for r in names}
    X = {r: np.asarray(rs[r]["bgc"][:], dtype=np.float64)[:, :nl1] for r in names}
    for name, s in END.items():
        out["end"][name] = {n: _stats(X[a][..., s], X[b][..., s], vol, wet3) for n, a, b in pairs}
    for f in ("T", "S"):
        Y = {r: np.asarray(rs[r][f][:], dtype=np.float64)[:, :nl1] for r in names}
        out["end"][f] = {n: _stats(Y[a], Y[b], vol, wet3) for n, a, b in pairs}

    for part in ("monthly", "end"):
        print(f"== {part} ({month if part == 'monthly' else 'end state'}): "
              + " | ".join(f"{n} = {a} - {b}" for n, a, b in pairs))
        print(f"{'field':>10s} " + " | ".join(f"{'d_int':>9s} {'max':>9s} {'rms':>9s}" for _ in pairs))
        for f, row in out[part].items():
            print(f"{f:>10s} " + " | ".join(f"{row[n]['integral']:9.2e} {row[n]['max']:9.2e} {row[n]['rms']:9.2e}"
                                            for n, _, _ in pairs))
    (args.base / "kerr.json").write_text(json.dumps({"pairs": pairs, **out}, indent=1))


if __name__ == "__main__":
    main()
