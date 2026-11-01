"""Merge the partial monthly-mean stores a segment boundary splits into one, weighted by ``n_samples``.

    python scripts/albedo/merge_monthly.py OUT STORE [STORE ...]

A half-year segment of 5840 steps ends 1.5 days into July, so July's mean is written twice (48 + 944 samples, the
two seg*/monthly/YYYY_07 stores). ``link_monthly.sh`` calls this for every month found in more than one segment: OUT
is a copy of the store with the most samples, every field the sample-weighted mean of the inputs (lat, lon and time
are copied), ``n_samples`` the total.
"""
import shutil
import sys

import numpy as np
import zarr

COORDS = ("lat", "lon", "time")


def main(out, stores):
    groups = [zarr.open_group(s, mode="r") for s in stores]
    counts = [int(g.attrs["n_samples"]) for g in groups]
    shutil.copytree(stores[int(np.argmax(counts))], out)
    root = zarr.open_group(out, mode="r+")
    total = sum(counts)
    for name in root.array_keys():
        if name in COORDS:
            continue
        root[name][...] = sum(np.asarray(g[name], dtype=np.float64) * n for g, n in zip(groups, counts)) / total
    root.attrs["n_samples"] = total
    print(f"{out}: n_samples {'+'.join(map(str, counts))} = {total}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
