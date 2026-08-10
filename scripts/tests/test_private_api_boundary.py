"""Which files may reach into `fesom_jax`'s private surface, and which may not.

The NN-GM research code is being split into its own repository (`docs/plans/20260810-
split-nn-eddy-and-promote-adjoint.md`). Once it is, a private import inside it is not untidiness —
it is a dependency on an internal that the library is entitled to rename in a patch release,
across a boundary where nothing would notice until a run failed.

⚠️ **This does NOT assert "no file under scripts/ imports a private symbol".** That was the
original proposal and it cannot pass and should not: an AST census on 2026-08-10 found 23 such
files, and 16 are in-repo tools of the library — `scripts/bench/`, `scripts/debug/` and
`scripts/paper/` reaching for `_shard_along_axis`, `_owner_map`, `_chunk_dates`,
`_all_state_fields` and friends. In-repo, a private import is a coupling the repository owns and
can fix in one commit. It is only a hazard once it spans two repositories.

So the rule is scoped to the files that are LEAVING, and the allowlist below is the exception
list, with a reason each.

⚠️ **AST, not grep.** The first census of this used a regular expression and reported
`from fesom_jax import gm as _gm` as a private import three times. An alias is not an import of a
private name, and a check that cannot tell the difference produces a list nobody trusts.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
LEAVING = ROOT / "scripts" / "nn_eddy"

# Files under scripts/nn_eddy/ that STAY in fesom_jax (docs/plans/20260810-inventory.md): their
# subject is the library's own machinery, so their private imports never cross a boundary.
STAYS = {
    "determinism_probe.py",
    "freeze_equivalence.py",
    "sharded_grad_ab.py",
}

# Private imports that are allowed anyway, with the reason. Empty is the goal; an entry here is a
# decision, not an oversight.
ALLOWED: dict[str, set[str]] = {
    # determinism_probe.py is in STAYS, but name it here too so that if the inventory ever
    # reclassifies it as MOVING, this test fails and the reason gets re-read rather than lost:
    # `_run_steps(segments=0)` is per-step rematerialization with no two-level scheme, which
    # `fesom_jax.adjoint.scan_steps` does not build. The probe asks whether one specific compiled
    # executable reproduces itself bit for bit, and ADJOINT_HORIZON.md cites the answer.
    "determinism_probe.py": {"fesom_jax.integrate._run_steps"},
}


def _private_imports(path: Path) -> set[str]:
    """`from fesom_jax.x import _y` and `fesom_jax.x._y`, but not `import y as _y`."""
    try:
        tree = ast.parse(path.read_text())
    except SyntaxError as exc:                       # a broken file is a different test's problem
        pytest.fail(f"{path.name} does not parse: {exc}")
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[0] == "fesom_jax":
            for a in node.names:
                if a.name.startswith("_"):           # a.name, NOT a.asname
                    found.add(f"{node.module}.{a.name}")
        if isinstance(node, ast.Attribute) and node.attr.startswith("_"):
            v = node.value
            if (isinstance(v, ast.Attribute) and isinstance(v.value, ast.Name)
                    and v.value.id == "fesom_jax"):
                found.add(f"fesom_jax.{v.attr}.{node.attr}")
    return found


def _tracked():
    """Paths git tracks under the leaving directory.

    ⚠️ TRACKED, not globbed. `adjoint_equivalence.sbatch` writes the pre-extraction copy of
    `adjoint_horizon.py` into this directory while it runs — it has to live exactly there, because
    the script derives the repo root as `Path(__file__).parents[2]` and resolves its siblings from
    its own directory, and a copy anywhere else makes `fesom_jax` resolve to the main checkout
    (the script's own guard catches that and refuses to report, which is how this was found). That
    copy is untracked and temporary, it is BY CONSTRUCTION full of private imports, and a globbing
    test reports it as a violation every time the gate runs. An untracked file is not a claim the
    repository is making."""
    import subprocess
    out = subprocess.run(["git", "ls-files", "--", str(LEAVING)],
                         cwd=ROOT, capture_output=True, text=True)
    if out.returncode != 0:
        pytest.skip("not a git checkout, so 'tracked' has no meaning here")
    return {ROOT / line for line in out.stdout.split() if line.endswith(".py")}


def _moving_files():
    if not LEAVING.is_dir():
        pytest.skip(f"{LEAVING} is gone — the split has happened and this test moved with it")
    tracked = _tracked()
    return sorted(p for p in LEAVING.glob("*.py")
                  if p.name not in STAYS and p in tracked)


def test_the_moving_research_code_uses_only_public_fesom_jax():
    offenders = {}
    for p in _moving_files():
        bad = _private_imports(p) - ALLOWED.get(p.name, set())
        if bad:
            offenders[p.name] = sorted(bad)
    assert not offenders, (
        "these files are moving to their own repository and import private fesom_jax symbols, "
        f"which would become a cross-repository dependency on an internal: {offenders}\n"
        "Use the public API, or promote the symbol and say why in its docstring "
        "(fesom_jax.eos.shift_down is the worked example), or add it to ALLOWED with a reason.")


def test_the_public_adjoint_is_what_they_use_instead():
    """The positive half. A file that stopped importing `_run_steps` by deleting the call rather
    than by moving to the public API would pass the test above and mean nothing."""
    users = [p.name for p in _moving_files()
             if "fesom_jax.adjoint" in p.read_text()]
    assert users, ("no moving file imports fesom_jax.adjoint; the private coupling was supposed "
                   "to be replaced by it, not merely removed")
    assert "adjoint_horizon.py" in users, \
        "adjoint_horizon.py is the reference implementation the library function came from"


def test_the_allowlist_has_no_stale_entries():
    """An allowlist that outlives its reason is how an exception becomes a convention."""
    for name, symbols in ALLOWED.items():
        p = LEAVING / name
        if not p.is_file():
            continue
        unused = symbols - _private_imports(p)
        assert not unused, (f"{name} no longer imports {sorted(unused)}; drop the entry from "
                            f"ALLOWED so the exception stops being granted")
