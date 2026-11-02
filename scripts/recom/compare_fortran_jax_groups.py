"""A fesom-jax REcoM run against a FESOM2-REcoM Fortran run group by group (the 4-PFT spectral fork: small
phytoplankton, diatoms, coccolithophores, Phaeocystis), month by month; the companion of ``compare_fortran_jax.py``
(lane W of docs/plans/20261028-recom-spectral-port.md).

For every month both have and every group:

* the global NPP [Pg C yr-1]: fesom-jax ``npp_phy``/``npp_dia``/``npp_cocco``/``npp_phaeo`` (``recom.output``, the
  per-group outputs of b178eb6) against the Fortran ``NPPn``/``NPPd``/``NPPc``/``NPPp`` streams, where the Fortran run
  wrote one (the user's kdind15* runs have no ``NPPc``), and its area-weighted pattern correlation;
* the global carbon and chlorophyll inventories (``<G>C``, ``<G>Chl``, ``sum X dz areasvol`` over the wet layers)
  and the pattern correlation of their surface fields.

It also checks that fesom-jax ``npp`` is the sum of its four groups, and prints the NPP of the groups both sides have
(the like-for-like total). Writes ``groups_summary.json`` (monthly records and the annual means per year) into
``--out``.

    python scripts/recom/compare_fortran_jax_groups.py --jax <run>/monthly --fortran <ResultPath> --mesh <bundle> \
        --out <dir> [--months 1998_01 ...]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_fortran_jax import MMOL_D_TO_PGC_YR, Fortran, Mesh, jax_month, wcorr  # noqa: E402

# (fesom-jax npp suffix, tracer prefix, Fortran NPP stream)
GROUPS = (("phy", "Phy", "NPPn"), ("dia", "Dia", "NPPd"), ("cocco", "Cocco", "NPPc"), ("phaeo", "Phaeo", "NPPp"))


def compare_month(m: Mesh, J: dict, F: Fortran, year: int, month: int) -> dict:
    A, nl1 = m.area, len(m.dz)
    r = {"month": f"{year:04d}_{month:02d}"}
    r["npp_total_jax"] = float(np.sum(J["npp"] * A) * MMOL_D_TO_PGC_YR)
    r["npp_sum_groups_jax"] = float(np.sum(sum(J[f"npp_{g}"] for g, _, _ in GROUPS) * A) * MMOL_D_TO_PGC_YR)
    for g, G, fv in GROUPS:
        a = J[f"npp_{g}"]
        r[f"npp_{g}_jax"] = float(np.sum(a * A) * MMOL_D_TO_PGC_YR)
        if F.has(fv, year):
            b = F.field(fv, year, month)
            r[f"npp_{g}_fortran"] = float(np.sum(b * A) * MMOL_D_TO_PGC_YR)
            r[f"npp_{g}_corr"] = wcorr(a, b, A)
        for kind in ("C", "Chl"):
            xj = np.asarray(J[f"{G}{kind}"])[:nl1].T                                # (nod, nl-1)
            xf = F.field(f"{G}{kind}", year, month).T
            r[f"{kind}_{g}_inv_jax_mmol"] = float(np.sum(xj * m.vol))
            r[f"{kind}_{g}_inv_fortran_mmol"] = float(np.sum(xf * m.vol))
            r[f"{kind}_{g}_surf_corr"] = wcorr(xj[:, 0], xf[:, 0], A)
    both = [g for g, _, _ in GROUPS if f"npp_{g}_fortran" in r]
    r["npp_common_groups"] = both
    r["npp_common_jax"] = sum(r[f"npp_{g}_jax"] for g in both)
    r["npp_common_fortran"] = sum(r[f"npp_{g}_fortran"] for g in both)
    return r


def pct(a, b):
    return 100.0 * (a / b - 1.0) if b else float("nan")


def report(r: dict):
    print(f"[groups] {r['month']}: NPP fesom-jax {r['npp_total_jax']:.2f} (sum of groups {r['npp_sum_groups_jax']:.2f}); "
          f"common groups ({'+'.join(r['npp_common_groups'])}) {r['npp_common_jax']:.2f} vs {r['npp_common_fortran']:.2f} "
          f"Pg C/yr ({pct(r['npp_common_jax'], r['npp_common_fortran']):+.2f} %)")
    for g, _, _ in GROUPS:
        s = f"[groups]   {g:6s} NPP {r[f'npp_{g}_jax']:7.3f}"
        if f"npp_{g}_fortran" in r:
            s += (f" vs {r[f'npp_{g}_fortran']:7.3f} ({pct(r[f'npp_{g}_jax'], r[f'npp_{g}_fortran']):+6.2f} %, "
                  f"corr {r[f'npp_{g}_corr']:.4f})")
        else:
            s += " (no Fortran stream)" + " " * 25
        for kind in ("C", "Chl"):
            s += (f" | {kind} inv {pct(r[f'{kind}_{g}_inv_jax_mmol'], r[f'{kind}_{g}_inv_fortran_mmol']):+7.3f} % "
                  f"corr {r[f'{kind}_{g}_surf_corr']:.4f}")
        print(s)


def annual(rows: list[dict]) -> dict:
    out = {}
    for y in sorted({r["month"][:4] for r in rows}):
        ys = [r for r in rows if r["month"][:4] == y]
        keys = [k for k, v in ys[0].items() if isinstance(v, float) and all(k in r for r in ys)]
        out[y] = {"n_months": len(ys), **{k: float(np.mean([r[k] for r in ys])) for k in keys}}
        a = out[y]
        print(f"[groups] {y} mean ({len(ys)} months): NPP fesom-jax {a['npp_total_jax']:.2f}, common groups "
              f"{a['npp_common_jax']:.2f} vs {a['npp_common_fortran']:.2f} Pg C/yr "
              f"({pct(a['npp_common_jax'], a['npp_common_fortran']):+.2f} %); "
              + ", ".join(f"{g} {a[f'npp_{g}_jax']:.2f}" + (f"/{a[f'npp_{g}_fortran']:.2f}"
                                                            if f"npp_{g}_fortran" in a else "")
                          for g, _, _ in GROUPS))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--jax", type=Path, required=True, help="the JAX run's monthly directory")
    ap.add_argument("--fortran", type=Path, required=True, help="the Fortran run's ResultPath")
    ap.add_argument("--mesh", type=Path, required=True, help="the mesh bundle")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--months", nargs="*", default=None, help="YYYY_MM (default: every month both have)")
    a = ap.parse_args()
    m, F = Mesh(a.mesh), Fortran(a.fortran)
    names = a.months or sorted(p.name for p in a.jax.iterdir() if p.is_dir())
    rows = []
    for name in names:
        year, month = map(int, name.split("_"))
        if not F.has("PhyC", year):
            continue
        r = compare_month(m, jax_month(a.jax / name), F, year, month)
        report(r)
        rows.append(r)
    out = {"monthly": rows, "annual": annual(rows) if rows else {}}
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "groups_summary.json").write_text(json.dumps(out, indent=1))
    print(f"[groups] wrote {a.out / 'groups_summary.json'} ({len(rows)} months)")


if __name__ == "__main__":
    main()
