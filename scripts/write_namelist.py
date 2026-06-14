#!/usr/bin/env python
"""Scalar → Fortran-namelist transfer (Paper-experiments Task A6 / §2 Fortran transfer).

The "killer app" of the calibration pillar: a tuned **scalar** physics constant transfers to the
operational Fortran FESOM2 with **zero Fortran code** — just patch ``namelist.oce`` /
``namelist.cvmix`` and re-run. This module is that patch mechanic: a **format-preserving** writer
that replaces the value of named keys in a real FESOM namelist (keeping every comment, blank line,
column, and untouched key byte-for-byte), so the diff is exactly the tuned constants.

The three transfer tiers (the paper's "how the optimum reaches the operational model"):
  1. **scalar / profile → namelist** — ZERO code (this module). ``K_GM_max``, ``Redi_Kmax``,
     ``tke_c_k``/``tke_c_eps``/``tke_cd``/``tke_alpha``.
  2. **2-D / 3-D field → netCDF** — tens of lines (a field the Fortran reads as a forcing).
  3. **NN → Fortran inference** — the ``tke_nn`` MLP re-implemented as ~150 lines of Fortran
     (the structure-preserving closure; weights from the trained pytree).

The auto-sync rule (``namelist.oce``): ``Redi_Kmax <= 0`` ⇒ the Fortran synchronizes
``Redi_Kmax = K_GM_max``. The GM perfect-model twin tunes a SINGLE scalar θ driving BOTH
(``k_gm = redi_kmax = θ``), so :func:`params_to_namelist` writes both to θ by default — honoring
the namelist's own "recommend to be the same" note explicitly rather than relying on the ≤0 sync.

Real FESOM key locations (verified against ``fesom2/work_all3``):
  * ``namelist.oce`` ``&oce_dyn``: ``K_GM_max``, ``Redi_Kmax``
  * ``namelist.cvmix`` ``&param_tke``: ``tke_c_k``, ``tke_c_eps``, ``tke_cd``, ``tke_alpha``

Usage:
    python scripts/write_namelist.py --template namelist.oce --out namelist.oce.tuned \\
        --set K_GM_max=1500 --set Redi_Kmax=1500
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

# Params-leaf → FESOM-namelist-key map (the tunables the §2 calibration writes back).
TUNABLE_KEYMAP = {
    "k_gm": "K_GM_max",
    "redi_kmax": "Redi_Kmax",
    "tke_c_k": "tke_c_k",
    "tke_c_eps": "tke_c_eps",
    "tke_cd": "tke_cd",
    "tke_alpha": "tke_alpha",
}

# A Fortran namelist assignment line:  [indent]KEY = VALUE [! comment]
_LINE = re.compile(
    r"^(?P<indent>\s*)(?P<key>[A-Za-z]\w*)(?P<eq>\s*=\s*)"
    r"(?P<val>[^!\n]*?)(?P<trail>\s*)(?P<comment>!.*)?$")


def _fmt(v) -> str:
    """Format a value as a Fortran-parseable literal. Python ``repr(float)`` always carries a
    ``.`` or ``e`` (e.g. ``1500.0``, ``0.1``, ``1e-05``), which Fortran reads as a real."""
    if isinstance(v, bool):
        return ".true." if v else ".false."
    if isinstance(v, str):
        return v
    return repr(float(v))


def params_to_namelist(params, *, sync_redi: bool = True) -> dict:
    """Map tuned parameters → ``{namelist_key: value}`` (the namelist keys, not Params leaves).

    ``params`` is a ``dict`` of Params-leaf names (preferred — the tuned subset, e.g.
    ``{'k_gm': 1500.}``) OR a ``Params`` object (then the GM+TKE tunable leaves are written).
    Only keys in :data:`TUNABLE_KEYMAP` are mapped. ``sync_redi`` (default) writes
    ``Redi_Kmax = K_GM_max`` whenever ``k_gm`` is tuned and ``redi_kmax`` is not explicitly given
    (the GM twin's single-scalar convention)."""
    if isinstance(params, dict):
        src = params
    else:
        src = {leaf: getattr(params, leaf) for leaf in TUNABLE_KEYMAP
               if getattr(params, leaf, None) is not None}
    out = {}
    for leaf, val in src.items():
        if leaf not in TUNABLE_KEYMAP:
            raise ValueError(f"params_to_namelist: unknown tunable leaf {leaf!r}; "
                             f"valid: {sorted(TUNABLE_KEYMAP)}")
        out[TUNABLE_KEYMAP[leaf]] = float(val)
    return _sync_redi_keys(out, sync_redi)


def _sync_redi_keys(nl: dict, sync_redi: bool) -> dict:
    """Honor the namelist auto-sync on the *namelist-keyed* dict: if ``K_GM_max`` is patched and
    ``Redi_Kmax`` is not explicitly given, set ``Redi_Kmax = K_GM_max`` (the GM twin's single-θ
    convention; the namelist itself recommends they be equal). Idempotent."""
    if sync_redi and "K_GM_max" in nl and "Redi_Kmax" not in nl:
        nl = {**nl, "Redi_Kmax": nl["K_GM_max"]}
    return nl


def patch_namelist_text(text: str, patch: dict):
    """Replace the values of the keys in ``patch`` wherever they appear (case-insensitive, EXACT
    key match — ``K_GM_max`` does not touch ``K_GM_min``). Every other byte is preserved; the
    inline comment is kept (re-padded with two spaces). Returns ``(new_text, patched, not_found)``
    where ``patched`` is the set of keys actually replaced and ``not_found`` the rest (e.g. TKE
    keys when patching ``namelist.oce``)."""
    by_lower = {k.lower(): (k, v) for k, v in patch.items()}
    patched: set[str] = set()
    out_lines = []
    for line in text.splitlines():
        m = _LINE.match(line)
        if m and m.group("key").lower() in by_lower:
            orig_key, val = by_lower[m.group("key").lower()]
            new = f"{m.group('indent')}{m.group('key')}{m.group('eq')}{_fmt(val)}"
            if m.group("comment"):
                new += "  " + m.group("comment")
            out_lines.append(new)
            patched.add(orig_key)
        else:
            out_lines.append(line)
    new_text = "\n".join(out_lines)
    if text.endswith("\n"):
        new_text += "\n"
    not_found = {k for k in patch if k not in patched}
    return new_text, patched, not_found


def write_namelist(params, template, out, *, sync_redi: bool = True) -> dict:
    """Patch the tuned ``params`` into the namelist ``template`` → ``out`` (format-preserving).

    ``params`` is a tuned-leaf ``dict`` / ``Params`` (mapped via :func:`params_to_namelist`) OR an
    already-namelist-keyed ``dict``. Keys not present in *this* template (e.g. the TKE keys when
    writing ``namelist.oce``) are returned under ``'not_found'`` so the caller knows to patch the
    other file too. Returns ``{'patched': {key: value}, 'not_found': {...}}``."""
    template, out = Path(template), Path(out)
    # accept either Params-leaf names or already-namelist keys
    if isinstance(params, dict) and set(params) <= set(TUNABLE_KEYMAP):
        nl = params_to_namelist(params, sync_redi=sync_redi)
    elif isinstance(params, dict):
        # already namelist-keyed (e.g. the CLI --set path) — still honor the Redi auto-sync
        nl = _sync_redi_keys({k: float(v) for k, v in params.items()}, sync_redi)
    else:
        nl = params_to_namelist(params, sync_redi=sync_redi)
    text = template.read_text()
    new_text, patched, not_found = patch_namelist_text(text, nl)
    out.write_text(new_text)
    return {"patched": {k: nl[k] for k in patched}, "not_found": {k: nl[k] for k in not_found}}


def _parse_kv(s: str):
    k, v = s.split("=", 1)
    return k.strip(), float(v)


def main():
    ap = argparse.ArgumentParser(description="Patch tuned scalars into a FESOM namelist.")
    ap.add_argument("--template", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="namelist key to patch (repeatable)")
    ap.add_argument("--no-sync-redi", action="store_true",
                    help="do NOT auto-set Redi_Kmax=K_GM_max")
    args = ap.parse_args()
    patch = dict(_parse_kv(s) for s in args.set)
    res = write_namelist(patch, args.template, args.out, sync_redi=not args.no_sync_redi)
    print(f"patched {sorted(res['patched'])} into {args.out}")
    if res["not_found"]:
        print(f"  (keys not in this template: {sorted(res['not_found'])})")
    print("FORTRAN_TRANSFER_OK")


if __name__ == "__main__":
    main()
