#!/usr/bin/env python
"""Upload the CORE2 one-year data package to Zenodo, via the API.

Adapted from the FESOM uploader (github.com/FESOM/FESOM_examples, `upload_to_zenodo/`), with
three deliberate differences:

  * **No community.** The original pins `communities: [{"identifier": "fesom2_meshes_..."}]`.
    This package is not a plain FESOM mesh — it is a fesom-jax model setup (mesh + initial state +
    a year of forcing) — so it goes to a PERSONAL Zenodo record, not the FESOM meshes community.
  * **An explicit file list**, not a folder walk. The staging directory also holds the ~21 GB of
    loose files the archives were built from; a walk would upload all of them.
  * **Real metadata** — description, creators, license, keywords, and the related identifiers that
    make the record cite FESOM2, the CORE2 mesh, and JRA55-do.

Stdlib only (no requests/click), so it runs in the fesom-jax env as-is.

    export ZENODO_TOKEN=...                     # personal token: zenodo.org/account/settings/applications
    python scripts/upload_to_zenodo.py --stage /work/ab0995/a270088/zenodo_core2 --sandbox --dry-run
    python scripts/upload_to_zenodo.py --stage /work/ab0995/a270088/zenodo_core2 --sandbox
    python scripts/upload_to_zenodo.py --stage /work/ab0995/a270088/zenodo_core2

WHY THE ARCHIVES ARE SPLIT INTO PARTS
-------------------------------------
Zenodo has **no resumable upload** (multipart is only on the InvenioRDM roadmap), and multi-GB PUTs
to it are known to die mid-transfer at a random point (zenodo/zenodo#2328: "randomly stopped anywhere
between 10-99%"). Observed here: the connection is dropped after ~20-30 s regardless of how many
bytes have gone -- so a single 10.4 GB PUT can never finish, and even a 369 MB one failed 3/3.

So we never PUT a whole archive. Each is sliced into `--split-mb` parts (default 128 MB, ~13 s each),
streamed straight from the original file (a seek + bounded read -- no extra disk), and every part is
retried on its own. `MANIFEST.json` on the record records the parts and each archive's sha256;
`scripts/fetch_data.py` reassembles and verifies them, so the split is invisible to whoever downloads
it. Zenodo allows 100 files / 50 GB per record, which bounds how small the parts can be.

RESUMING
--------
Every part is a separate file on the draft, and files already present are skipped -- so if the
upload dies, just re-run it against the same draft and it picks up where it stopped:

    python scripts/upload_to_zenodo.py --stage <STAGE> --deposition <DRAFT_ID>

REHEARSE ON --sandbox FIRST. Same API against sandbox.zenodo.org, its own token, no real DOI.

The record is left as an unpublished DRAFT: nothing is public until you review it and either press
Publish in the web UI or re-run with --publish. Publishing is IRREVERSIBLE -- files in a published
Zenodo record cannot be changed or removed, only superseded by a new version.
"""
from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import os
import ssl
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

PROD = "zenodo.org"
SANDBOX = "sandbox.zenodo.org"

# The two archives, in upload order (small first, so a metadata mistake surfaces before the 10 GB).
ARCHIVES = ("core2_mesh_ic.zip", "core2_forcing_1958.zip")

GITHUB = "https://github.com/koldunovn/fesom_jax"

DESCRIPTION = """\
<p>Everything needed to run the <a href="https://github.com/koldunovn/fesom_jax">fesom-jax</a>
ocean model on the <strong>CORE2</strong> mesh (global, ~1&deg;, 126,858 nodes, 48 levels) for one
year. fesom-jax is a differentiable port of the FESOM2 ocean circulation model to JAX.</p>

<p>Two archives:</p>
<ul>
  <li><strong>core2_mesh_ic.zip</strong> (~370 MB) &mdash; the mesh (as a dense <code>.npy</code>
      bundle and in standard FESOM2 text format), the cached PHC 3.0 initial temperature/salinity,
      the PHC source file, the sea-surface-salinity restoring field, river runoff, and the
      chlorophyll climatology.</li>
  <li><strong>core2_forcing_1958.zip</strong> (~10.4 GB) &mdash; the eight JRA55-do v1.4.0 surface
      fields for 1958 (<code>uas, vas, tas, huss, rsds, rlds, prra, prsn</code>), 3-hourly, on the
      native JRA55 grid.</li>
</ul>

<p>Usage: <code>python scripts/fetch_data.py --dest ~/fesom-data --record &lt;THIS RECORD ID&gt;</code>,
then point the model at it with the printed environment variables. See <code>docs/DATA.md</code> in
the repository. Each archive carries a README and a <code>CHECKSUMS.sha256</code>.</p>

<h4>Note on the CORE2 mesh version</h4>
<p>The CORE2 vertical level files (<code>nlvls.out</code>, <code>elvls.out</code>) were regenerated
upstream on 2026-07-03. <strong>This package ships the earlier version</strong>, because that is what
all fesom-jax results were produced with. The two differ at exactly <strong>2 nodes and 4
elements</strong>, all in the Ross Sea (~154&deg;W, 77&deg;S), where the version here is deeper
(580 m vs 280 m at the largest difference). Both are structurally valid. To match a current FESOM2
run instead, take the level files from the upstream mesh
(<a href="https://gitlab.awi.de/fesom/core2">gitlab.awi.de/fesom/core2</a>) and rebuild the dense
bundle with <code>scripts/prepare_mesh.py</code>.</p>

<h4>Please also cite the underlying model and datasets</h4>
<ul>
  <li>FESOM2 &mdash; Danilov et al. (2017), doi:10.5194/gmd-10-765-2017</li>
  <li>CORE2 mesh &mdash; Wang et al. (2014), doi:10.5194/gmd-7-663-2014</li>
  <li>JRA55-do forcing &mdash; Tsujino et al. (2018), doi:10.1016/j.ocemod.2018.07.002</li>
</ul>
"""

METADATA = {
    "upload_type": "dataset",
    "title": "fesom-jax CORE2 one-year setup: mesh, initial state, and JRA55-do forcing (1958)",
    "description": DESCRIPTION,
    "creators": [
        {"name": "Koldunov, Nikolay",
         "affiliation": "Alfred Wegener Institute, Helmholtz Centre for Polar and Marine Research"},
    ],
    "access_right": "open",
    "license": "cc-by-4.0",
    "version": "1.0",
    "keywords": ["ocean model", "FESOM", "FESOM2", "fesom-jax", "JAX", "CORE2",
                 "differentiable model", "JRA55-do", "ocean forcing", "climate"],
    "related_identifiers": [
        {"identifier": GITHUB, "relation": "isSupplementTo", "scheme": "url"},
        {"identifier": "10.5194/gmd-10-765-2017", "relation": "cites", "scheme": "doi"},
        {"identifier": "10.5194/gmd-7-663-2014", "relation": "cites", "scheme": "doi"},
        {"identifier": "10.1016/j.ocemod.2018.07.002", "relation": "cites", "scheme": "doi"},
    ],
    # NOTE: no "communities" key -- this is a personal record, deliberately NOT the FESOM
    # meshes community (it is a model setup, not a mesh distribution).
}


# ---------------------------------------------------------------- API helpers
def _api(host: str, method: str, path: str, token: str, payload=None) -> tuple[int, dict]:
    url = f"https://{host}{path}"
    sep = "&" if "?" in path else "?"
    url = f"{url}{sep}access_token={urllib.parse.quote(token)}"
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            body = r.read()
            return r.status, (json.loads(body) if body else {})
    except urllib.error.HTTPError as e:
        body = e.read()
        try:
            return e.code, json.loads(body)
        except Exception:  # noqa: BLE001
            return e.code, {"raw": body[:400].decode(errors="replace")}


def _hash_slice(path: Path, offset: int, length: int, algo="md5") -> str:
    h = hashlib.new(algo)
    with open(path, "rb") as f:
        f.seek(offset)
        left = length
        while left:
            chunk = f.read(min(1 << 20, left))
            if not chunk:
                break
            h.update(chunk)
            left -= len(chunk)
    return h.hexdigest()


def sha256sum(path: Path) -> str:
    return _hash_slice(path, 0, path.stat().st_size, "sha256")


class UploadError(RuntimeError):
    pass


def put_slice(bucket_url: str, path: Path, name: str, offset: int, length: int,
              token: str, retries: int = 8) -> dict:
    """PUT `length` bytes of `path` starting at `offset` into the bucket, as file `name`.

    Zenodo has **no resumable upload** (multipart is only on the InvenioRDM roadmap), so a PUT that
    dies at 99% has to start over. Uploads of multi-GB files are also a known Zenodo flakiness
    (zenodo/zenodo#2328: "randomly stopped anywhere between 10-99%") -- which is why we never PUT a
    whole multi-GB archive: we PUT bounded slices, each short enough to land inside the window the
    connection actually survives, and each independently retryable.

    We stream from the ORIGINAL file (a seek + a bounded read), so splitting costs no extra disk.
    """
    parts = urllib.parse.urlparse(bucket_url)
    target = f"{parts.path}/{urllib.parse.quote(name)}?access_token={urllib.parse.quote(token)}"

    for attempt in range(1, retries + 1):
        conn = None
        try:
            conn = http.client.HTTPSConnection(parts.netloc, timeout=180,
                                               context=ssl.create_default_context())
            conn.putrequest("PUT", target)
            conn.putheader("Content-Length", str(length))
            conn.putheader("Content-Type", "application/octet-stream")
            conn.endheaders()

            sent, t0, last = 0, time.time(), 0.0
            with open(path, "rb") as f:
                f.seek(offset)
                while sent < length:
                    chunk = f.read(min(1 << 20, length - sent))
                    if not chunk:
                        raise UploadError(f"short read at {offset + sent}")
                    conn.send(chunk)
                    sent += len(chunk)
                    now = time.time()
                    if now - last > 1 or sent == length:
                        last = now
                        rate = sent / max(now - t0, 1e-6) / 2**20
                        print(f"\r      {100 * sent / length:5.1f}%  {sent / 2**20:6.0f}/"
                              f"{length / 2**20:.0f} MB  {rate:5.1f} MB/s", end="", flush=True)

            resp = conn.getresponse()
            body = resp.read()
            print(f"\r      {'':44s}\r", end="")
            if resp.status not in (200, 201):
                raise UploadError(f"HTTP {resp.status}: {body[:200].decode(errors='replace')}")
            return json.loads(body)

        except (BrokenPipeError, ssl.SSLError, OSError, UploadError, http.client.HTTPException) as e:
            # The peer closed mid-body. It may have said WHY before closing (413, 401, 400...) --
            # a plain "Broken pipe" hides that, so always try to read the response first.
            detail = f"{type(e).__name__}: {str(e)[:110]}"
            if conn is not None and not isinstance(e, UploadError):
                try:
                    r = conn.getresponse()
                    b = r.read()
                    detail = f"server said HTTP {r.status}: {b[:200].decode(errors='replace')}"
                except Exception:  # noqa: BLE001
                    pass                       # nothing to read: a genuine mid-transfer drop
            print(f"\n      attempt {attempt}/{retries} failed: {detail}")
            if attempt == retries:
                raise UploadError(f"{name}: giving up after {retries} attempts -- {detail}") from e
            time.sleep(min(5 * attempt, 30))
        finally:
            if conn is not None:
                conn.close()
    raise UploadError("unreachable")


def plan_parts(path: Path, part_bytes: int) -> list[dict]:
    """Split a file into upload parts. `part_bytes<=0` (or a small file) => one whole-file part."""
    size = path.stat().st_size
    if part_bytes <= 0 or size <= part_bytes:
        return [{"name": path.name, "offset": 0, "length": size}]
    out, offset, i = [], 0, 0
    while offset < size:
        length = min(part_bytes, size - offset)
        out.append({"name": f"{path.name}.part{i:03d}", "offset": offset, "length": length})
        offset += length
        i += 1
    return out


# ---------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", type=Path, required=True,
                    help="directory holding the two .zip archives (built by build_zenodo_package.py)")
    ap.add_argument("--sandbox", action="store_true",
                    help="use sandbox.zenodo.org (rehearsal; needs its own token; no real DOI)")
    ap.add_argument("--dry-run", action="store_true",
                    help="show the files and the metadata, contact Zenodo not at all")
    ap.add_argument("--publish", action="store_true",
                    help="PUBLISH the record. IRREVERSIBLE: a published record's files are permanent")
    ap.add_argument("--deposition", type=int,
                    help="add to / update an EXISTING draft deposition instead of creating one. "
                         "THIS IS HOW YOU RESUME: parts already on the draft are skipped")
    ap.add_argument("--split-mb", type=int, default=128,
                    help="upload files larger than this as N-MB parts (default 128). Zenodo has no "
                         "resumable upload and drops long PUTs, so big files MUST be sliced. 0 = off")
    args = ap.parse_args()

    host = SANDBOX if args.sandbox else PROD
    files = [args.stage / n for n in ARCHIVES]
    missing = [f for f in files if not f.is_file()]
    if missing:
        print(f"error: missing archive(s): {[str(m) for m in missing]}\n"
              f"       build them first:  python scripts/build_zenodo_package.py --out {args.stage}",
              file=sys.stderr)
        return 1

    part_bytes = args.split_mb * 2**20
    plans = {f: plan_parts(f, part_bytes) for f in files}
    n_parts = sum(len(p) for p in plans.values()) + 1        # +1 for MANIFEST.json
    total = sum(f.stat().st_size for f in files)

    print(f"target   : {host}{'  (SANDBOX -- rehearsal, no real DOI)' if args.sandbox else ''}")
    print(f"community: none (personal record)")
    print(f"split    : {args.split_mb} MB parts  ->  {n_parts} files on the record"
          f"{'  !! OVER ZENODO LIMIT OF 100' if n_parts > 100 else '  (limit 100)'}")
    print("files    :")
    for f in files:
        print(f"    {f.name:28s} {f.stat().st_size / 2**30:6.2f} GB  -> {len(plans[f]):3d} part(s)")
    print(f"    {'TOTAL':28s} {total / 2**30:6.2f} GB  (limit 50 GB)")
    if n_parts > 100:
        print(f"\nerror: {n_parts} files exceeds Zenodo's 100-file limit. Raise --split-mb "
              f"(>= {int(total / 95 / 2**20)} MB).", file=sys.stderr)
        return 1

    if args.dry_run:
        print("\n--- metadata that would be attached ---")
        print(json.dumps(METADATA, indent=2)[:1200])
        print("\n(dry run: Zenodo was not contacted)")
        return 0

    token = os.environ.get("ZENODO_TOKEN", "").strip()
    if not token:
        print("error: set ZENODO_TOKEN (a personal token with 'deposit:write' scope).\n"
              f"       create one at https://{host}/account/settings/applications/tokens/new/\n"
              "       export ZENODO_TOKEN=...    (do NOT pass it on the command line)",
              file=sys.stderr)
        return 1

    # 1. deposition ---------------------------------------------------------
    if args.deposition:
        code, dep = _api(host, "GET", f"/api/deposit/depositions/{args.deposition}", token)
        if code != 200:
            print(f"error: cannot open draft {args.deposition}: {dep}", file=sys.stderr)
            return 1
        print(f"\nreusing draft deposition {dep['id']}")
    else:
        code, dep = _api(host, "POST", "/api/deposit/depositions", token, payload={})
        if code != 201:
            print(f"error: could not create deposition (HTTP {code}): {dep}", file=sys.stderr)
            return 1
        print(f"\ncreated draft deposition {dep['id']}")

    dep_id = dep["id"]
    bucket = dep["links"]["bucket"]
    already = {f["filename"] for f in dep.get("files", [])}
    if already:
        print(f"  {len(already)} file(s) already on this draft -- they will be skipped")

    # 2. metadata FIRST -- so a mistake surfaces before 10 GB of upload -------
    code, resp = _api(host, "PUT", f"/api/deposit/depositions/{dep_id}", token,
                      payload={"metadata": METADATA})
    if code != 200:
        print(f"error: metadata rejected (HTTP {code}): {json.dumps(resp)[:600]}", file=sys.stderr)
        print(f"       the draft still exists: https://{host}/deposit/{dep_id}", file=sys.stderr)
        return 1
    print("metadata attached")

    # 3. the manifest -- tells fetch_data.py how to put the parts back together
    print("\nhashing archives (once; needed for the manifest)...", flush=True)
    manifest = {"note": "Large archives are split into .partNNN files because Zenodo has no "
                        "resumable upload. Reassemble with: cat NAME.part* > NAME  "
                        "(scripts/fetch_data.py does this for you, and verifies sha256).",
                "archives": []}
    for f in files:
        entry = {"name": f.name, "size": f.stat().st_size, "sha256": sha256sum(f),
                 "parts": [p["name"] for p in plans[f]]}
        manifest["archives"].append(entry)
        print(f"  {f.name}  sha256 {entry['sha256'][:16]}...  {len(entry['parts'])} part(s)")

    man_path = args.stage / "MANIFEST.json"
    man_path.write_text(json.dumps(manifest, indent=2) + "\n")

    # 4. upload: the manifest, then every part, skipping what is already there
    todo = [("MANIFEST.json", man_path, 0, man_path.stat().st_size)]
    for f in files:
        todo += [(p["name"], f, p["offset"], p["length"]) for p in plans[f]]
    pending = [t for t in todo if t[0] not in already]
    print(f"\nuploading {len(pending)} of {len(todo)} files "
          f"({sum(t[3] for t in pending) / 2**30:.2f} GB to go)")

    done_bytes, t_start = 0, time.time()
    for i, (name, src, offset, length) in enumerate(pending, 1):
        print(f"  [{i:3d}/{len(pending)}] {name}  ({length / 2**20:.0f} MB)")
        info = put_slice(bucket, src, name, offset, length, token)
        remote = (info.get("checksum") or "").replace("md5:", "")
        local = _hash_slice(src, offset, length)
        if remote and remote != local:
            print(f"    !! checksum mismatch on {name} (zenodo {remote}, local {local})",
                  file=sys.stderr)
            return 1
        done_bytes += length
        el = time.time() - t_start
        left = (sum(t[3] for t in pending) - done_bytes) / max(done_bytes / max(el, 1e-6), 1)
        print(f"    ok (md5 {local[:12]}...)   overall {done_bytes / 2**30:.2f} GB, "
              f"~{left / 60:.0f} min left")

    url = f"https://{host}/deposit/{dep_id}"
    if not args.publish:
        print(f"\nDRAFT ready (nothing is public yet): {url}")
        print("Review it, then publish from the web page, or re-run with --publish.")
        print(f"After publishing, the record id for fetch_data.py is: {dep_id}")
        return 0

    # Refuse to publish a half-uploaded record: publishing is irreversible.
    code, dep = _api(host, "GET", f"/api/deposit/depositions/{dep_id}", token)
    on_record = {f["filename"] for f in dep.get("files", [])}
    want = {t[0] for t in todo}
    if not want <= on_record:
        print(f"\nrefusing to publish: {len(want - on_record)} file(s) missing from the draft "
              f"({sorted(want - on_record)[:3]}...).\n"
              f"Re-run to upload the rest:  --deposition {dep_id}", file=sys.stderr)
        return 1

    code, pub = _api(host, "POST", f"/api/deposit/depositions/{dep_id}/actions/publish", token)
    if code != 202:
        print(f"error: publish failed (HTTP {code}): {json.dumps(pub)[:500]}", file=sys.stderr)
        print(f"       the draft is intact: {url}", file=sys.stderr)
        return 1

    rec = pub.get("record_id", dep_id)
    doi = pub.get("doi", "(pending)")
    print(f"\nPUBLISHED  record {rec}   DOI {doi}")
    print(f"  {pub.get('links', {}).get('record_html', f'https://{host}/records/{rec}')}")
    print(f"\nNow wire it up:")
    print(f"  python scripts/fetch_data.py --dest ~/fesom-data --record {rec}")
    print(f"  ...and put record id {rec} into docs/DATA.md + README.md.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
