"""Reader for the FESOM2 Fortran reference-dump files — the verification oracle.

Ported from ``/home/a/a270088/port2/inspect_dump.py``. The Fortran shim
(``fesom_dump_shim.F90``) writes, per probe, one record per (step, substep,
field):

Binary record (little-endian, stream, no header)::

    int32    step
    int32    substep_id
    int32    probe_global_id    (Fortran 1-based)
    int32    nlevels
    char[24] field_name         (blank-padded ASCII)
    real64   values[nlevels]

These dumps come from the **Fortran** model and record **NODE fields only**
(per-node columns, already truncated to ``nlevels_nod2D(node)``). Element fields
(``pgf``, ``uv_rhs``, ``uv``, ``Av``) need the Task-0.4 element-dump extension —
see the plan's Context section.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Union

import numpy as np

# The 16 substep IDs — the porting + verification granularity for one ocean step.
SUBSTEP_NAMES: dict[int, str] = {
    0: "init",
    1: "pressure_bv",
    2: "sw_alpha_beta",
    3: "pressure_force",
    4: "mixing",
    5: "vel_rhs",
    6: "viscosity_filter",
    7: "impl_vert_visc",
    8: "ssh_rhs",
    9: "ssh_solve",
    10: "update_vel",
    11: "compute_hbar",
    12: "eta_n",
    13: "ale_step",
    14: "gm_bolus",
    15: "solve_tracers",
    16: "update_thickness",
}
SUBSTEP_IDS: dict[str, int] = {v: k for k, v in SUBSTEP_NAMES.items()}

_HEADER_FMT = "<iiii24s"  # 4× int32 + 24 char => 40 bytes
_HEADER_LEN = struct.calcsize(_HEADER_FMT)


@dataclass(frozen=True, eq=False)
class DumpRecord:
    """One probe-column record from a Fortran reference dump."""

    step: int
    substep: int
    probe_gid: int  # Fortran 1-based global id
    nlevels: int
    field: str
    values: np.ndarray  # shape (nlevels,), float64

    @property
    def substep_name(self) -> str:
        return SUBSTEP_NAMES.get(self.substep, f"?({self.substep})")

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        head = self.values[:3]
        tail = " ..." if self.nlevels > 3 else ""
        vals = " ".join(f"{v:.4e}" for v in head)
        return (
            f"DumpRecord(step={self.step}, sub={self.substep}/{self.substep_name}, "
            f"probe={self.probe_gid}, field={self.field!r}, nlev={self.nlevels}, "
            f"values=[{vals}{tail}])"
        )


def read_records(path: Union[str, Path]) -> Iterator[DumpRecord]:
    """Yield every :class:`DumpRecord` in a dump file, in write order."""
    path = Path(path)
    with open(path, "rb") as fh:
        while True:
            hdr = fh.read(_HEADER_LEN)
            if not hdr:
                return
            if len(hdr) != _HEADER_LEN:
                raise IOError(f"truncated header in {path}: got {len(hdr)} bytes")
            step, sub, gid, nlev, name = struct.unpack(_HEADER_FMT, hdr)
            name = name.decode("ascii", "replace").rstrip()
            payload = fh.read(8 * nlev)
            if len(payload) != 8 * nlev:
                raise IOError(
                    f"truncated payload in {path}: want {8 * nlev} got {len(payload)}"
                )
            values = np.frombuffer(payload, dtype="<f8", count=nlev).astype(np.float64)
            yield DumpRecord(step, sub, gid, nlev, name, values)


def load_records(path: Union[str, Path]) -> list[DumpRecord]:
    """Read all records from a dump file into a list."""
    return list(read_records(path))


def _norm_substep(substep: Union[int, str]) -> int:
    if isinstance(substep, str):
        if substep not in SUBSTEP_IDS:
            raise KeyError(f"unknown substep name {substep!r}; one of {list(SUBSTEP_IDS)}")
        return SUBSTEP_IDS[substep]
    return int(substep)


def find_record(
    source: Union[str, Path, Iterable[DumpRecord]],
    *,
    step: int,
    substep: Union[int, str],
    field: str,
    probe_gid: int | None = None,
) -> DumpRecord:
    """Return the unique record matching the filters.

    ``source`` may be a dump-file path or an iterable of :class:`DumpRecord`.
    ``substep`` may be the numeric id or its name. Raises if no match, or if the
    match is ambiguous (multiple probes) and ``probe_gid`` was not given.
    """
    records: Iterable[DumpRecord]
    if isinstance(source, (str, Path)):
        records = read_records(source)
    else:
        records = source
    sub = _norm_substep(substep)
    matches = [
        r
        for r in records
        if r.step == step
        and r.substep == sub
        and r.field == field
        and (probe_gid is None or r.probe_gid == probe_gid)
    ]
    if not matches:
        raise LookupError(
            f"no record step={step} substep={sub}/{SUBSTEP_NAMES.get(sub, '?')} "
            f"field={field!r} probe={probe_gid}"
        )
    if len(matches) > 1:
        probes = sorted({r.probe_gid for r in matches})
        raise LookupError(
            f"ambiguous match ({len(matches)} records; probes {probes}); "
            f"pass probe_gid="
        )
    return matches[0]


def load_gm_dump(dirpath: Union[str, Path]) -> tuple[dict, dict]:
    """Read a Phase-6B GM/Redi all-node dump (``fesom_gm_dump``, ``gm_meta.txt`` +
    raw ``gm_<field>.f64`` blobs).

    Returns ``(fields, meta)`` where ``meta = {N, E, nl, npes}`` and ``fields`` maps
    each field name to a float64 array reshaped per the C row-major layout:
    ``(rows, levels)`` when ``comp == 1`` (``(rows, 1)`` for the per-node scalars
    ``fer_C``/``fer_scal``), ``(rows, levels, comp)`` when ``comp > 1`` (comp 0=x,
    1=y; for the slopes comp 2=|s|). ``rows`` = ``N`` (node fields) or ``E``
    (``fer_uv``). The JAX gate slices/masks these to its layer convention.
    """
    d = Path(dirpath)
    meta_path = d / "gm_meta.txt"
    if not meta_path.is_file():
        raise FileNotFoundError(f"GM dump meta missing: {meta_path}")
    lines = meta_path.read_text().splitlines()
    header = dict(tok.split("=") for tok in lines[0].split())
    meta = {k: int(header[k]) for k in ("N", "E", "nl", "npes")}

    fields: dict = {}
    for line in lines[1:]:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name, rows, levels, comp = line.split()
        rows, levels, comp = int(rows), int(levels), int(comp)
        raw = np.fromfile(d / f"gm_{name}.f64", dtype="<f8")
        if raw.size != rows * levels * comp:
            raise ValueError(
                f"GM dump {name}: size {raw.size} != {rows}*{levels}*{comp}")
        arr = raw.reshape(rows, levels, comp)
        fields[name] = arr[..., 0] if comp == 1 else arr
    return fields, meta


# ---------------------------------------------------------------------------
# KPP reference dumps (Phase 6C) — the C ``fesom_kpp.c`` dump harness writes
# plain-text files (NOT the binary GM ``.f64`` blobs): one file per (step, tag,
# rank), keyed by 1-based global id. The dump job (``jax_kpp_dump_core2.sh``)
# runs single-rank, so ``*_rank0.txt`` carries every node/element (gid 1..N) and
# row order is the partition's ``myList`` order; we reorder by gid into JAX mesh
# index order (``out[gid-1] = row``) — robust regardless of the partition order,
# and (single-rank) JAX node index ``i`` ↔ global gid ``i+1`` (the GM-gate node
# alignment, made explicit). Three file kinds:
#   * ``kpp_dump_s<step>_<tag>_rank<R>.txt`` — per-kernel node/element columns
#     (``# step=.. tag=.. rank=.. N=.. ncomp=..`` header + ``gid v0 v1 …`` lines).
#   * ``kpp_init_rank0.txt``  — Vtc/cg/deltaz/deltau scalars + the wm/ws lookup
#     table (``i j wmt wst``, row-major in i; K.1 gate).
#   * ``kpp_wscale_rank0.txt`` — the wscale sweep (``i j zehat ustar wm ws``; K.2 gate).
# ---------------------------------------------------------------------------

# tag → ncomp / entity, for reference + selective loading (the per-kernel gates
# request only the few tags they need; the full all-node set is ~GB of text).
KPP_NODE_TAGS = (
    "prestep", "dVsq", "dbsfc",                       # K.5 bldepth inputs (prestep=ustar,Bo)
    "ri_viscA", "ri_diffKt", "ri_diffKs", "ri_bvfreq",  # K.3 ri_iwmix outputs (+ bvfreq input)
    "bldepth",                                         # K.5 outputs: hbl, kbl+1, bfsfc, stable, caseA
    "blmc_m", "blmc_t", "blmc_s", "bl_ghats", "dkm1",   # K.6 blmix outputs (pre-enhance)
    "viscA", "diffKt", "diffKs", "ghats",              # K.7 final (post-combine) node outputs
)
KPP_ELEM_TAGS = ("viscAE",)                            # K.7 final element viscosity (= Av)


def _parse_kpp_header(line: str) -> dict:
    """Parse a ``# step=1 tag=viscA rank=0 N=126858 ncomp=48`` header line."""
    if not line.startswith("#"):
        raise IOError(f"KPP dump: expected '# step=...' header, got {line!r}")
    out: dict = {}
    for tok in line.lstrip("#").split():
        if "=" in tok:
            k, v = tok.split("=", 1)
            out[k] = v
    for k in ("step", "rank", "N", "ncomp"):
        out[k] = int(out[k])
    return out


def read_kpp_table(path: Union[str, Path]) -> tuple[np.ndarray, np.ndarray, dict]:
    """Read one ``kpp_dump_s<step>_<tag>_rank<R>.txt`` → ``(gids, values, meta)``.

    ``gids`` is ``int64[N]`` (1-based global ids, in file/``myList`` order),
    ``values`` is ``float64[N, ncomp]`` (same order), ``meta`` the parsed header.
    Uses ``np.fromstring`` (C parser) for the large all-node files."""
    path = Path(path)
    with open(path) as fh:
        meta = _parse_kpp_header(fh.readline())
        flat = np.fromstring(fh.read(), sep=" ", dtype=np.float64)
    N, ncomp = meta["N"], meta["ncomp"]
    if flat.size != N * (1 + ncomp):
        raise ValueError(
            f"KPP dump {path.name}: {flat.size} numbers != N*(1+ncomp)={N*(1+ncomp)}")
    flat = flat.reshape(N, 1 + ncomp)
    return flat[:, 0].astype(np.int64), np.ascontiguousarray(flat[:, 1:]), meta


def load_kpp_dump(
    dirpath: Union[str, Path], tags: Iterable[str] | None = None,
    *, step: int = 1, rank: int = 0, reorder: bool = True,
) -> tuple[dict, dict]:
    """Read the per-kernel KPP node/element dumps for ``(step, rank)``.

    ``tags`` selects which to load (the per-kernel gate names the few it needs; the
    full set is ~GB of text). ``None`` ⇒ every ``kpp_dump_s<step>_*_rank<rank>.txt``
    present. Returns ``(fields, meta)``: ``fields[tag]`` is ``float64[N, ncomp]`` in
    **JAX mesh index order** (``reorder`` reindexes by gid: ``out[gid-1]=row``) — the
    gate slices/masks per its layer convention. ``meta`` maps each tag to its header
    ``{N, ncomp}`` and records ``gids_identity`` (whether the file was already in
    ``gid==row+1`` order — the single-rank expectation)."""
    d = Path(dirpath)
    if tags is None:
        tags = sorted(p.name.split("_rank")[0].split(f"s{step}_", 1)[1]
                      for p in d.glob(f"kpp_dump_s{step}_*_rank{rank}.txt"))
    fields: dict = {}
    meta: dict = {}
    for tag in tags:
        path = d / f"kpp_dump_s{step}_{tag}_rank{rank}.txt"
        if not path.is_file():
            raise FileNotFoundError(f"KPP dump tag {tag!r} missing: {path}")
        gids, values, hdr = read_kpp_table(path)
        N = hdr["N"]
        identity = bool(np.array_equal(gids, np.arange(1, N + 1)))
        if reorder and not identity:
            out = np.empty_like(values)
            if not np.array_equal(np.sort(gids), np.arange(1, N + 1)):
                raise ValueError(f"KPP dump {tag}: gids are not a permutation of 1..{N}")
            out[gids - 1] = values
            values = out
        fields[tag] = values
        meta[tag] = {"N": N, "ncomp": hdr["ncomp"], "gids_identity": identity}
    return fields, meta


def load_kpp_init(dirpath: Union[str, Path]) -> dict:
    """Read ``kpp_init_rank0.txt`` → the 4 derived scalars + the ``wmt``/``wst``
    lookup tables (each ``float64[nni+2, nnj+2]``, row-major in i). K.1 gate."""
    d = Path(dirpath)
    path = d / "kpp_init_rank0.txt"
    with open(path) as fh:
        out: dict = {}
        # 4 scalar comment lines: "# Vtc <val>", …
        for _ in range(4):
            toks = fh.readline().lstrip("#").split()
            out[toks[0]] = float(toks[1])
        hdr = fh.readline()                          # "# i j wmt wst  (nni=890 nnj=480)"
        nni = int(hdr.split("nni=")[1].split()[0])
        nnj = int(hdr.split("nnj=")[1].split(")")[0])
        body = np.fromstring(fh.read(), sep=" ", dtype=np.float64).reshape(-1, 4)
    out["nni"], out["nnj"] = nni, nnj
    out["wmt"] = body[:, 2].reshape(nni + 2, nnj + 2)
    out["wst"] = body[:, 3].reshape(nni + 2, nnj + 2)
    return out


def load_kpp_wscale_sweep(dirpath: Union[str, Path]) -> dict:
    """Read ``kpp_wscale_rank0.txt`` → ``zehat``/``ustar``/``wm``/``ws`` over the
    fixed sweep grid (each ``float64[NZ, NU]``, row-major in i). K.2 gate."""
    d = Path(dirpath)
    path = d / "kpp_wscale_rank0.txt"
    with open(path) as fh:
        hdr = fh.readline()                          # "# i j zehat ustar wm ws  (sweep 201x101)"
        nz, nu = (int(x) for x in hdr.split("sweep")[1].split(")")[0].strip().split("x"))
        body = np.fromstring(fh.read(), sep=" ", dtype=np.float64).reshape(-1, 6)
    return {
        "NZ": nz, "NU": nu,
        "zehat": body[:, 2].reshape(nz, nu), "ustar": body[:, 3].reshape(nz, nu),
        "wm": body[:, 4].reshape(nz, nu), "ws": body[:, 5].reshape(nz, nu),
    }


def _summary_main(argv: list[str]) -> int:  # pragma: no cover - debug CLI
    if not argv:
        print("usage: python -m fesom_jax.io_dump <dumpfile> [more...]")
        return 2
    n = 0
    for path in argv:
        for r in read_records(path):
            n += 1
            print(
                f"step={r.step:5d}  sub={r.substep:2d}/{r.substep_name:<18s}  "
                f"probe={r.probe_gid:6d}  {r.field:<12s}  nlev={r.nlevels:3d}  "
                + " ".join(f"{v:11.4e}" for v in r.values[:3])
                + (" ..." if r.nlevels > 3 else "")
            )
    print(f"[{n} records]")
    return 0


if __name__ == "__main__":  # pragma: no cover
    import sys

    raise SystemExit(_summary_main(sys.argv[1:]))
