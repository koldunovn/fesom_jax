"""Central resolver for the model's **external input data** paths (mesh-scale runs only).

Every path the model reads from outside the repo — JRA55-do forcing, the PHC initial
condition, the SSS-restoring climatology, runoff, chlorophyll, and the mesh/partition root —
used to be a hardcoded DKRZ/Levante ``/pool/data/AWICM/...`` string inside the reader that
consumed it, which made the model unusable off Levante. They now all funnel through here.

Precedence (one rule, everywhere)::

    explicit argument / run-YAML `forcing:` key   >   environment variable   >   Levante default

The env var is read **at call time** (late binding), so setting ``FESOM_JRA_DIR`` in a job
script or a notebook after ``import fesom_jax`` still takes effect.

Nothing here changes numerics: this is pure plumbing. On Levante, with no env vars and no
explicit paths, every ``resolve()`` returns exactly the string that used to be hardcoded.

The bundled **pi** mesh needs NONE of this (it ships with the package, idealized forcing);
only CORE2-and-larger runs read external data. See ``docs/DATA.md`` for the full table.
"""

from __future__ import annotations

import dataclasses
import os
from pathlib import Path

# --------------------------------------------------------------------------
# Levante (DKRZ) defaults — the paths that used to be hardcoded in the readers.
# --------------------------------------------------------------------------
LEVANTE_JRA_DIR = "/pool/data/AWICM/FESOM2/FORCING/JRA55-do-v1.4.0"
LEVANTE_PHC_PATH = "/pool/data/AWICM/FESOM2/INITIAL/phc3.0/phc3.0_winter.nc"
LEVANTE_SSS_PATH = "/pool/data/AWICM/FESOM2/FORCING/JRA55-do-v1.4.0/PHC2_salx.nc"
LEVANTE_RUNOFF_PATH = "/pool/data/AWICM/FESOM2/FORCING/JRA55-do-v1.4.0/CORE2_runoff.nc"
LEVANTE_CHL_PATH = "/pool/data/AWICM/FESOM2/FORCING/Sweeney/Sweeney_2005.nc"
LEVANTE_MESH_ROOT = "/pool/data/AWICM/FESOM2/MESHES_FESOM2.1"

# NOT a /pool path: the repo-relative cache of the interpolated PHC IC (``T_ic.npy`` /
# ``S_ic.npy``, gitignored under ``data/``). Kept in the same resolver so an off-repo /
# scratch cache can be selected with $FESOM_IC_DIR.
DEFAULT_IC_DIR = str(Path(__file__).resolve().parents[1] / "data" / "ic_core2")


@dataclasses.dataclass(frozen=True)
class InputSpec:
    """One external input: what it is, its env var, its run-YAML key, its default."""
    name: str                     # the short resolver key
    env: str                      # environment variable
    default: str                  # Levante (or repo-relative) default
    what: str                     # human description, used in the error message
    yaml_key: str | None = None   # key inside the run-YAML `forcing:` mapping (if any)
    alt: str | None = None        # the non-YAML alternative (CLI flag), for the message


_SPECS: dict[str, InputSpec] = {
    s.name: s for s in (
        InputSpec("jra_dir", "FESOM_JRA_DIR", LEVANTE_JRA_DIR,
                  "JRA55-do forcing", yaml_key="jra_dir"),
        InputSpec("phc_path", "FESOM_PHC_PATH", LEVANTE_PHC_PATH,
                  "PHC3.0 initial condition", alt="--phc-path / the load_phc_ic(path=…) arg"),
        InputSpec("sss_path", "FESOM_SSS_PATH", LEVANTE_SSS_PATH,
                  "SSS restoring climatology (PHC2_salx)", yaml_key="sss_path"),
        InputSpec("runoff_path", "FESOM_RUNOFF_PATH", LEVANTE_RUNOFF_PATH,
                  "CORE2 runoff", yaml_key="runoff_path"),
        InputSpec("chl_path", "FESOM_CHL_PATH", LEVANTE_CHL_PATH,
                  "chlorophyll climatology (Sweeney 2005)", yaml_key="chl_path"),
        InputSpec("mesh_root", "FESOM_MESH_ROOT", LEVANTE_MESH_ROOT,
                  "mesh / dist_<N> partition root", alt="the run-YAML `mesh:` key or --mesh-dir"),
        InputSpec("ic_dir", "FESOM_IC_DIR", DEFAULT_IC_DIR,
                  "cached PHC initial-condition directory", alt="--ic-dir"),
    )
}

NAMES = tuple(_SPECS)


def spec(name: str) -> InputSpec:
    """The :class:`InputSpec` for ``name`` (raises ``KeyError`` on an unknown input)."""
    try:
        return _SPECS[name]
    except KeyError:
        raise KeyError(f"unknown input path {name!r} (known: {sorted(_SPECS)})") from None


def resolve(name: str, explicit=None) -> str:
    """Resolve input ``name`` → a path string, WITHOUT checking that it exists.

    Precedence: ``explicit`` (an argument or run-YAML key) → ``$ENV`` → the Levante default.
    The environment is read here, at call time, not at import."""
    sp = spec(name)
    if explicit is not None:
        return str(explicit)
    env = os.environ.get(sp.env)
    if env:                                  # empty string ⇒ treat as unset
        return env
    return sp.default


def require(name: str, explicit=None) -> str:
    """:func:`resolve` + an existence check, with an ACTIONABLE ``FileNotFoundError``.

    Use this wherever the path is about to be opened (the readers do); use :func:`resolve`
    where the path is only being reported or composed."""
    sp = spec(name)
    path = resolve(name, explicit)
    if not os.path.exists(path):
        raise FileNotFoundError(f"{sp.what} not found: {path} — {_how_to_set(sp)} "
                                f"On Levante the default should work; elsewhere see docs/DATA.md.")
    return path


def _how_to_set(sp: InputSpec) -> str:
    """The 'here is how you fix it' clause of the :func:`require` message."""
    how = f"set ${sp.env}"
    if sp.yaml_key is not None:
        how += f", or the `forcing.{sp.yaml_key}` key in the run YAML"
    if sp.alt is not None:
        how += f", or {sp.alt}"
    return how + "."


def describe() -> list[dict]:
    """Every input as a plain dict (name / env / yaml_key / default / resolved) — the table
    behind ``docs/DATA.md`` and a quick "where is my data coming from?" dump."""
    return [dict(name=s.name, env=s.env, yaml_key=s.yaml_key, what=s.what,
                 default=s.default, resolved=resolve(s.name),
                 exists=os.path.exists(resolve(s.name)))
            for s in _SPECS.values()]
