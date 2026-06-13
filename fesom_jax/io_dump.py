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
# Generic gid-keyed text dumps — the SHARED format used by both the KPP harness
# (Phase 6C, ``fesom_kpp.c``) and the ALE/zstar harness (Phase 9a,
# ``fesom_ale_dump.c``): one plain-text file per (step, tag, rank) with a
# ``# step=.. tag=.. rank=.. N=.. ncomp=..`` header + ``gid v0 v1 …`` lines, the
# 1-based gid first. :func:`read_gid_table` parses one such file; :func:`load_kpp_dump`
# (single-rank KPP) and :func:`load_ale_dump` (multi-rank ALE, merge-by-gid) build on it.
#
# KPP file kinds (the dump job ``jax_kpp_dump_core2.sh`` runs single-rank, so
# ``*_rank0.txt`` carries every node/element gid 1..N; we reorder by gid into JAX
# mesh index order ``out[gid-1] = row`` — robust regardless of partition order, and
# single-rank JAX node index ``i`` ↔ global gid ``i+1``). Three file kinds:
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


def _parse_gid_header(line: str) -> dict:
    """Parse a ``# step=1 tag=viscA rank=0 N=126858 ncomp=48`` header line (the shared
    KPP / ALE gid-table header)."""
    if not line.startswith("#"):
        raise IOError(f"gid dump: expected '# step=...' header, got {line!r}")
    out: dict = {}
    for tok in line.lstrip("#").split():
        if "=" in tok:
            k, v = tok.split("=", 1)
            out[k] = v
    for k in ("step", "rank", "N", "ncomp"):
        out[k] = int(out[k])
    return out


def read_gid_table(path: Union[str, Path]) -> tuple[np.ndarray, np.ndarray, dict]:
    """Read one gid-keyed dump table → ``(gids, values, meta)``.

    The shared text format of the KPP (``kpp_dump_s<step>_<tag>_rank<R>.txt``) and ALE
    (``ale_dump_s<step>_<tag>_rank<R>.txt``) harnesses: a ``# step=.. tag=.. rank=.. N=..
    ncomp=..`` header followed by ``N`` lines ``gid v0 v1 … v<ncomp-1>``. ``gids`` is
    ``int64[N]`` (1-based global ids, in file/``myList`` order), ``values`` is
    ``float64[N, ncomp]`` (same order), ``meta`` the parsed header. Uses ``np.fromstring``
    (C parser) for the large all-node files."""
    path = Path(path)
    with open(path) as fh:
        meta = _parse_gid_header(fh.readline())
        flat = np.fromstring(fh.read(), sep=" ", dtype=np.float64)
    N, ncomp = meta["N"], meta["ncomp"]
    if flat.size != N * (1 + ncomp):
        raise ValueError(
            f"gid dump {path.name}: {flat.size} numbers != N*(1+ncomp)={N*(1+ncomp)}")
    flat = flat.reshape(N, 1 + ncomp)
    return flat[:, 0].astype(np.int64), np.ascontiguousarray(flat[:, 1:]), meta


# Backward-compatible aliases (KPP loaders + any scripts predating the Phase-9a rename).
read_kpp_table = read_gid_table
_parse_kpp_header = _parse_gid_header


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
        gids, values, hdr = read_gid_table(path)
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


# ---------------------------------------------------------------------------
# ALE / zstar reference dumps (Phase 9a) — the C ``fesom_ale_dump.c`` harness
# (env ``FESOM_ALE_DUMP_DIR`` / ``FESOM_ALE_DUMP_STEPS``) writes the SAME gid-keyed
# text format as KPP, but **multi-rank**: each rank dumps only its OWNED rows
# (``myDim_nod2D`` / ``myDim_elem2D``, keyed by ``myList_*``). Node partitions are
# disjoint (union = 1..nod2D, no dupes); element partitions overlap on a thin
# boundary ring but the duplicated rows are bit-identical — so a merge-by-gid
# (``out[gid-1] = row``) reconstructs the global field exactly. 12 tags at 6 driver
# sites (``fesom_ale_dump.c:77-154``), the component layout fixed by the C getters.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AleTag:
    """One ALE dump tag's layout (the source of truth = the C ``fesom_ale_dump.c``
    getter that fills it). ``entity`` selects the global size (node vs element);
    ``kind`` selects the level convention for masking in verification:

    * ``"scalar"`` — ``comps`` is a tuple of independent per-(node|elem) 2-D field
      names (``ncomp == len(comps)``; e.g. ``forcing`` packs 4 surface flux fields);
    * ``"layer"`` — a per-layer column, ``ncomp == nl-1`` (valid ``[ulevels-1, nlevels-1)``);
    * ``"iface"`` — a per-interface column, ``ncomp == nl`` (valid ``[ulevels-1, nlevels-1]``).
    """

    name: str
    entity: str                      # "nod" | "elem"
    kind: str                        # "scalar" | "layer" | "iface"
    comps: tuple[str, ...] | None    # field names for "scalar"; None for columns


# The 12 tags (`fesom_ale_dump.c:77-154`), component order matching the C getters
# (`get_2d`'s `s->a[c]` array / `get_col`'s level index).
ALE_TAGS: dict[str, AleTag] = {
    # forcing site (fesom_ale_dump.c:82-85)
    "forcing":   AleTag("forcing", "nod", "scalar",
                        ("water_flux", "virtual_salt", "relax_salt", "real_salt_flux")),
    # pgf site (fesom_ale_dump.c:93-98)
    "pgf_x":     AleTag("pgf_x", "elem", "layer", None),
    "pgf_y":     AleTag("pgf_y", "elem", "layer", None),
    # sshsolve site (fesom_ale_dump.c:106-108)
    "sshsolve":  AleTag("sshsolve", "nod", "scalar", ("ssh_rhs", "d_eta")),
    # hbar site (fesom_ale_dump.c:116-122)
    "hbar":      AleTag("hbar", "nod", "scalar",
                        ("hbar", "hbar_old", "ssh_rhs_old", "eta_n")),
    "dhe":       AleTag("dhe", "elem", "scalar", ("dhe",)),
    # vertvel site (fesom_ale_dump.c:130-135)
    "Wvel":      AleTag("Wvel", "nod", "iface", None),
    "hnode_new": AleTag("hnode_new", "nod", "layer", None),
    # thickness (post-commit) site (fesom_ale_dump.c:142-153)
    "hnode":     AleTag("hnode", "nod", "layer", None),
    "zbar_3d_n": AleTag("zbar_3d_n", "nod", "iface", None),
    "Z_3d_n":    AleTag("Z_3d_n", "nod", "layer", None),
    "helem":     AleTag("helem", "elem", "layer", None),
}


def load_ale_dump(
    dirpath: Union[str, Path], tags: Iterable[str] | None = None,
    *, step: int = 1, n_nod: int | None = None, n_elem: int | None = None,
    strict: bool = True,
) -> tuple[dict, dict]:
    """Read the multi-rank ALE/zstar dump for ``step``, merging ranks by gid.

    ``tags`` selects which to load (``None`` ⇒ every tag present for ``step``). Returns
    ``(fields, meta)``: ``fields[tag]`` is ``float64[Nglobal, ncomp]`` in **JAX mesh index
    order** (``out[gid-1] = row``), with ``Nglobal`` = ``n_nod`` (node tags) / ``n_elem``
    (element tags), or the max gid seen when those are ``None``. ``meta[tag]`` records
    ``{N, ncomp, entity, kind, nranks}``. Gids never written stay ``NaN``; with
    ``strict`` (default) a node tag missing any gid in ``1..N`` raises, and overlapping
    element rows are asserted bit-identical (the boundary-ring invariant)."""
    d = Path(dirpath)
    if tags is None:
        present = set()
        prefix = f"ale_dump_s{step}_"
        for p in d.glob(f"{prefix}*_rank*.txt"):
            present.add(p.name[len(prefix):].rsplit("_rank", 1)[0])
        tags = sorted(present)
    fields: dict = {}
    meta: dict = {}
    for tag in tags:
        paths = sorted(d.glob(f"ale_dump_s{step}_{tag}_rank*.txt"),
                       key=lambda p: int(p.name.rsplit("_rank", 1)[1].split(".")[0]))
        if not paths:
            raise FileNotFoundError(
                f"ALE dump tag {tag!r} step {step} missing: {d}/ale_dump_s{step}_{tag}_rank*.txt")
        chunks = [read_gid_table(p) for p in paths]
        ncomp = chunks[0][2]["ncomp"]
        spec = ALE_TAGS.get(tag)
        entity = spec.entity if spec else None
        kind = spec.kind if spec else None
        gmax = max(int(g.max()) for g, _, _ in chunks)
        Ng = (n_nod if entity == "nod" else n_elem if entity == "elem" else None) or gmax
        out = np.full((Ng, ncomp), np.nan, np.float64)
        seen = np.zeros(Ng, dtype=bool)
        for g, v, hdr in chunks:
            if hdr["ncomp"] != ncomp:
                raise ValueError(f"ALE dump {tag}: rank {hdr['rank']} ncomp "
                                 f"{hdr['ncomp']} != {ncomp}")
            idx = g - 1
            if strict:
                dup = seen[idx]
                if dup.any() and not np.array_equal(out[idx[dup]], v[dup]):
                    raise ValueError(
                        f"ALE dump {tag}: overlapping gids disagree across ranks "
                        f"(stale-halo dump?) — expected the boundary-ring rows bit-identical")
            out[idx] = v
            seen[idx] = True
        if strict and not seen.all():
            raise ValueError(
                f"ALE dump {tag}: {int((~seen).sum())} of {Ng} gids never written "
                f"(incomplete rank set or wrong n_{entity})")
        fields[tag] = out
        meta[tag] = {"N": Ng, "ncomp": ncomp, "entity": entity, "kind": kind,
                     "nranks": len(paths)}
    return fields, meta


def ale_component(fields: dict, tag: str, name: str) -> np.ndarray:
    """Pick one named component column out of a ``"scalar"``-kind ALE tag's
    ``[N, ncomp]`` block (e.g. ``ale_component(f, "forcing", "water_flux")`` →
    ``[N]``). Uses :data:`ALE_TAGS` for the component order (the C getter packing)."""
    spec = ALE_TAGS[tag]
    if spec.comps is None:
        raise ValueError(f"ALE tag {tag!r} is a column ({spec.kind}); index by level, "
                         f"not component name")
    if name not in spec.comps:
        raise KeyError(f"ALE tag {tag!r} has no component {name!r}; one of {spec.comps}")
    return fields[tag][:, spec.comps.index(name)]


# ---------------------------------------------------------------------------
# TKE reference dumps (Phase 9b) — the C ``fesom_tke.c`` harness (env
# ``FESOM_TKE_DUMP_DIR`` / ``FESOM_TKE_DUMP_STEPS``) writes the SAME gid-keyed text format
# as KPP/ALE, **multi-rank**: each rank dumps only its OWNED rows (``myDim_nod2D`` /
# ``myDim_elem2D``, keyed by ``myList_*``). 20 tags per step: 5 column inputs + 3
# column-core outputs + 10 budget/aux diagnostics + 2 driver-wired final fields. Node
# partitions are disjoint (union = 1..nod2D); ``av`` is the only element tag (boundary-ring
# overlap is bit-identical). The replay/cdump oracle is the **16-rank linfs** set (3 steps,
# dt=1800) — use ``data/ic_core2_dist16``; the climate oracle ``c_tke_2yr`` is 864-rank — use
# ``data/ic_core2_dist864`` (IC-partition provenance is per-oracle, [[zstar-forcing-dump-config-gap]]).
# Files: ``tke_dump_s<step>_<tag>_rank<R>.txt``. Reuses :func:`read_gid_table` + the ALE
# merge-by-gid logic.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TkeTag:
    """One TKE dump tag's layout (the source of truth = the C ``fesom_tke.c`` dump call).
    ``entity`` selects the global size (node vs element); ``kind`` the level convention
    (``"iface"`` = per-interface column ``ncomp == nl``; ``"scalar"`` = one per-node value);
    ``role`` groups the tags for selective gating (the JT.1 column-core gate requests the 13
    ``output``+``diag`` tags; ``input`` is the replay-injection set; ``wired`` are the
    driver-level final ``Kv``/``Av`` ⇒ JT.2)."""

    name: str
    entity: str       # "nod" | "elem"
    kind: str         # "iface" | "scalar"
    role: str         # "input" | "output" | "diag" | "wired"


# The 20 tags (`fesom_tke.c:456-509`). Column tags carry the full ``nl`` interfaces; the JAX
# gate slices/masks to the wet column ``[ulevels-1, nlevels-1]``. The 3 outputs map to the
# column core's returns: ``tke`` = tke_new, ``tkeav`` = KappaM, ``tkekv`` = KappaH.
TKE_TAGS: dict[str, TkeTag] = {
    # 5 column inputs (the controlled-replay injection set, fesom_tke.c:456-460)
    "normstress": TkeTag("normstress", "nod", "scalar", "input"),  # |stress|/ρ₀ (forc_tke_surf)
    "vshear2":    TkeTag("vshear2", "nod", "iface", "input"),      # Ssqr
    "bvfreq2":    TkeTag("bvfreq2", "nod", "iface", "input"),      # Nsqr (smoothed)
    "dztrr":      TkeTag("dztrr", "nod", "iface", "input"),        # dzt (hnode/2 end caps)
    "tkeold":     TkeTag("tkeold", "nod", "iface", "input"),       # tke_old (the recurrent state)
    # 3 column-core outputs (fesom_tke.c:461-463)
    "tke":        TkeTag("tke", "nod", "iface", "output"),         # tke_new
    "tkeav":      TkeTag("tkeav", "nod", "iface", "output"),       # KappaM
    "tkekv":      TkeTag("tkekv", "nod", "iface", "output"),       # KappaH
    # 10 budget/aux diagnostics (fesom_tke.c:465-474; only when diag_on)
    "tbpr":       TkeTag("tbpr", "nod", "iface", "diag"),          # buoyancy production
    "tspr":       TkeTag("tspr", "nod", "iface", "diag"),          # shear production
    "tdif":       TkeTag("tdif", "nod", "iface", "diag"),          # vertical diffusion
    "tdis":       TkeTag("tdis", "nod", "iface", "diag"),          # dissipation
    "twin":       TkeTag("twin", "nod", "iface", "diag"),          # wind/surface input
    "tiwf":       TkeTag("tiwf", "nod", "iface", "diag"),          # iw forcing (≡0, only_tke)
    "tbck":       TkeTag("tbck", "nod", "iface", "diag"),          # background (floor) reset
    "ttot":       TkeTag("ttot", "nod", "iface", "diag"),          # total tendency (closure Σ)
    "lmix":       TkeTag("lmix", "nod", "iface", "diag"),          # mixing length mxl
    "pr":         TkeTag("pr", "nod", "iface", "diag"),            # Prandtl number
    # 2 driver-wired final fields (fesom_tke.c:508-509)
    "kv":         TkeTag("kv", "nod", "iface", "wired"),           # aux->Kv (full-slab copy)
    "av":         TkeTag("av", "elem", "iface", "wired"),          # aux->Av (node→elem mean)
}


def load_tke_dump(
    dirpath: Union[str, Path], tags: Iterable[str] | None = None,
    *, step: int = 1, n_nod: int | None = None, n_elem: int | None = None,
    strict: bool = True,
) -> tuple[dict, dict]:
    """Read the multi-rank TKE dump for ``step``, merging ranks by gid — the
    :func:`load_ale_dump` pattern for the ``tke_dump_s<step>_<tag>_rank*.txt`` set.

    ``tags`` selects which to load (``None`` ⇒ every tag present for ``step``). Returns
    ``(fields, meta)``: ``fields[tag]`` is ``float64[Nglobal, ncomp]`` in **JAX mesh index
    order** (``out[gid-1] = row``), ``Nglobal`` = ``n_nod`` (node tags) / ``n_elem``
    (element tags) or the max gid seen when those are ``None``. ``meta[tag]`` records
    ``{N, ncomp, entity, kind, role, nranks}``. Gids never written stay ``NaN``; with
    ``strict`` (default) a node tag missing any gid in ``1..N`` raises and overlapping
    element rows are asserted bit-identical (the boundary-ring invariant)."""
    d = Path(dirpath)
    if tags is None:
        present = set()
        prefix = f"tke_dump_s{step}_"
        for p in d.glob(f"{prefix}*_rank*.txt"):
            present.add(p.name[len(prefix):].rsplit("_rank", 1)[0])
        tags = sorted(present)
    fields: dict = {}
    meta: dict = {}
    for tag in tags:
        paths = sorted(d.glob(f"tke_dump_s{step}_{tag}_rank*.txt"),
                       key=lambda p: int(p.name.rsplit("_rank", 1)[1].split(".")[0]))
        if not paths:
            raise FileNotFoundError(
                f"TKE dump tag {tag!r} step {step} missing: {d}/tke_dump_s{step}_{tag}_rank*.txt")
        chunks = [read_gid_table(p) for p in paths]
        ncomp = chunks[0][2]["ncomp"]
        spec = TKE_TAGS.get(tag)
        entity = spec.entity if spec else None
        gmax = max(int(g.max()) for g, _, _ in chunks)
        Ng = (n_nod if entity == "nod" else n_elem if entity == "elem" else None) or gmax
        out = np.full((Ng, ncomp), np.nan, np.float64)
        seen = np.zeros(Ng, dtype=bool)
        for g, v, hdr in chunks:
            if hdr["ncomp"] != ncomp:
                raise ValueError(f"TKE dump {tag}: rank {hdr['rank']} ncomp "
                                 f"{hdr['ncomp']} != {ncomp}")
            idx = g - 1
            if strict:
                dup = seen[idx]
                if dup.any() and not np.array_equal(out[idx[dup]], v[dup]):
                    raise ValueError(
                        f"TKE dump {tag}: overlapping gids disagree across ranks "
                        f"(stale-halo dump?) — expected the boundary-ring rows bit-identical")
            out[idx] = v
            seen[idx] = True
        if strict and not seen.all():
            raise ValueError(
                f"TKE dump {tag}: {int((~seen).sum())} of {Ng} gids never written "
                f"(incomplete rank set or wrong n_{entity})")
        fields[tag] = out
        meta[tag] = {"N": Ng, "ncomp": ncomp, "entity": entity,
                     "kind": spec.kind if spec else None,
                     "role": spec.role if spec else None, "nranks": len(paths)}
    return fields, meta


# ---------------------------------------------------------------------------
# mEVP reference dumps (Phase 9c) — the C ``fesom_ice_maevp.c`` harness (env
# ``FESOM_EVP_DUMP_DIR``) writes a gid-keyed text format like KPP/ALE/TKE, **multi-rank**:
# each rank dumps only its OWNED rows (``myDim_nod2D`` / ``myDim_elem2D``, keyed by
# ``myList_*``). ⚠️ The header is ``# step=.. point=.. array=.. rank=.. N=..`` — **no
# ``ncomp``** (it varies per point: Q=4, U0/UF=2, F=4, P_node=4, P_elem=1, it*_node=2,
# it*_elem=3), so :func:`read_evp_table` infers it from the row width. Node partitions are
# disjoint (union = 1..nod2D); element partitions overlap on a thin boundary ring but the
# redundantly-computed rows are bit-identical (the per-substep halo exchange keeps the element
# vertex velocities current ⇒ identical strain). Files:
# ``evp_dump_s<step>_<point>_<array>_rank<R>.txt`` (``fesom_ice_maevp.c:52-54``). The
# 16-rank cdump oracle (2 steps, dt=1800) lives on ``/work/ab0995/a270088/port/mevp/cdump_16r``.
# ---------------------------------------------------------------------------

# Per-(point, array) component layout — the dump value order in the C maevp_dump calls
# (``fesom_ice_maevp.c:127-135, 215-219, 331-335, 359-360``). it1/it2/it60/it120 share a layout.
_EVP_IT_NODE = ("u_aux", "v_aux")
_EVP_IT_ELEM = ("sigma11", "sigma12", "sigma22")
EVP_POINT_LAYOUT: dict[tuple[str, str], tuple[str, ...]] = {
    ("Q",  "node"): ("a_ice", "m_ice", "m_snow", "elevation"),       # entry inputs (:127-129)
    ("U0", "node"): ("u_ice", "v_ice"),                              # entry velocity (:130-132)
    ("F",  "node"): ("stress_atmice_x", "stress_atmice_y", "u_w", "v_w"),  # entry forcing (:133-135)
    ("P",  "node"): ("inv_thickness", "mass", "rhs_a", "rhs_m"),     # precompute (:215-217)
    ("P",  "elem"): ("pressure_fac",),                               # precompute (:218-220)
    ("UF", "node"): ("u_ice", "v_ice"),                             # final (:359-361)
}


def evp_point_layout(point: str, array: str) -> tuple[str, ...]:
    """Component names for one mEVP dump (point, array) — the C value order. it* points
    (``it1``/``it2``/``it60``/``it120``) share the iterate layout."""
    if point.startswith("it"):
        return _EVP_IT_NODE if array == "node" else _EVP_IT_ELEM
    return EVP_POINT_LAYOUT[(point, array)]


def _parse_evp_header(line: str) -> dict:
    """Parse a ``# step=1 point=Q array=node rank=0 N=7928`` header (no ``ncomp`` — inferred
    by :func:`read_evp_table` from the data width)."""
    if not line.startswith("#"):
        raise IOError(f"evp dump: expected '# step=...' header, got {line!r}")
    out: dict = {}
    for tok in line.lstrip("#").split():
        if "=" in tok:
            k, v = tok.split("=", 1)
            out[k] = v
    for k in ("step", "rank", "N"):
        out[k] = int(out[k])
    return out


def read_evp_table(path: Union[str, Path]) -> tuple[np.ndarray, np.ndarray, dict]:
    """Read one mEVP gid-keyed dump → ``(gids, values, meta)``. Like :func:`read_gid_table`
    but the header carries no ``ncomp`` — it is inferred from the row width
    (``ncomp = total/N - 1``). ``gids`` is ``int64[N]`` (1-based, file/``myList`` order),
    ``values`` is ``float64[N, ncomp]``; ``meta`` has ``step/point/array/rank/N/ncomp``."""
    path = Path(path)
    with open(path) as fh:
        meta = _parse_evp_header(fh.readline())
        flat = np.fromstring(fh.read(), sep=" ", dtype=np.float64)
    N = meta["N"]
    if N <= 0 or flat.size % N != 0:
        raise ValueError(f"evp dump {path.name}: {flat.size} numbers not divisible by N={N}")
    width = flat.size // N
    ncomp = width - 1
    if ncomp < 1:
        raise ValueError(f"evp dump {path.name}: row width {width} has no value columns")
    flat = flat.reshape(N, width)
    meta["ncomp"] = ncomp
    return flat[:, 0].astype(np.int64), np.ascontiguousarray(flat[:, 1:]), meta


def load_mevp_dump(
    dirpath: Union[str, Path], points: Iterable[str] | None = None,
    *, step: int = 1, array: str = "node", n_nod: int | None = None,
    n_elem: int | None = None, strict: bool = True,
) -> tuple[dict, dict]:
    """Read the multi-rank mEVP dump for ``(step, array)``, merging ranks by gid into JAX mesh
    index order (``out[gid-1] = row``). ``array`` selects node vs elem files (``P`` and the
    ``it*`` points have both). ``points`` selects which to load (``None`` ⇒ every point present
    for ``(step, array)``). Returns ``(fields, meta)``: ``fields[point]`` is
    ``float64[Nglobal, ncomp]`` (``Nglobal`` = ``n_nod``/``n_elem`` or the max gid seen).
    ``meta[point]`` records ``{N, ncomp, array, layout, nranks}``. With ``strict`` a node point
    missing any gid in ``1..N`` raises, and overlapping element rows are asserted bit-identical
    (the boundary-ring invariant — the redundant element compute is identical post-halo)."""
    d = Path(dirpath)
    if points is None:
        present = set()
        for p in d.glob(f"evp_dump_s{step}_*_{array}_rank*.txt"):
            present.add(p.name[len(f"evp_dump_s{step}_"):].rsplit(f"_{array}_rank", 1)[0])
        points = sorted(present)
    Ng_default = n_nod if array == "node" else n_elem
    fields: dict = {}
    meta: dict = {}
    for point in points:
        paths = sorted(d.glob(f"evp_dump_s{step}_{point}_{array}_rank*.txt"),
                       key=lambda p: int(p.name.rsplit("_rank", 1)[1].split(".")[0]))
        if not paths:
            raise FileNotFoundError(
                f"mEVP dump {point!r} step {step} array {array}: "
                f"{d}/evp_dump_s{step}_{point}_{array}_rank*.txt missing")
        chunks = [read_evp_table(p) for p in paths]
        ncomp = chunks[0][2]["ncomp"]
        gmax = max(int(g.max()) for g, _, _ in chunks)
        Ng = Ng_default or gmax
        out = np.full((Ng, ncomp), np.nan, np.float64)
        seen = np.zeros(Ng, dtype=bool)
        for g, v, hdr in chunks:
            if hdr["ncomp"] != ncomp:
                raise ValueError(f"mEVP dump {point}: rank {hdr['rank']} ncomp "
                                 f"{hdr['ncomp']} != {ncomp}")
            idx = g - 1
            if strict:
                dup = seen[idx]
                if dup.any() and not np.array_equal(out[idx[dup]], v[dup]):
                    raise ValueError(
                        f"mEVP dump {point}: overlapping gids disagree across ranks "
                        f"(stale-halo dump?) — boundary-ring rows must be bit-identical")
            out[idx] = v
            seen[idx] = True
        if strict and not seen.all():
            raise ValueError(
                f"mEVP dump {point}: {int((~seen).sum())} of {Ng} gids never written "
                f"(incomplete rank set or wrong n_{'nod' if array == 'node' else 'elem'})")
        fields[point] = out
        meta[point] = {"N": Ng, "ncomp": ncomp, "array": array,
                       "layout": evp_point_layout(point, array), "nranks": len(paths)}
    return fields, meta


def evp_component(fields: dict, meta: dict, point: str, name: str) -> np.ndarray:
    """Pick one named component column out of a loaded mEVP ``point`` block
    (e.g. ``evp_component(f, m, "P", "inv_thickness")`` → ``[Nglobal]``), using the
    layout recorded in ``meta`` (:func:`evp_point_layout`)."""
    layout = meta[point]["layout"]
    if name not in layout:
        raise KeyError(f"mEVP point {point!r} has no component {name!r}; one of {layout}")
    return fields[point][:, layout.index(name)]


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
