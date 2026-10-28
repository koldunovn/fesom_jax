"""Read-ahead of the JRA55 slices in worker processes (host side of the on-device forcing, CORE2 GPU speed plan).

Adapted from fesom3-jax (``fesom3_jax/forcing/readahead.py``, c70ad11). The JRA55-do files are netCDF-4 with one
deflate-compressed chunk per record, so decoding a slice is about half of the host forcing table build (CORE2, 96
steps of 2700 s, one Grace host: 300 of 640 ms per chunk). HDF5 is not thread-safe, so the reads ahead run in
spawned processes: :meth:`fesom_jax.jra55.JRA55Reader.bracket_schedule` asks for every record the chunk's brackets
touch, and the reader's slice read takes them decoded. A worker opens the file as the reader does (no auto
mask/scale) and returns the raw slice: the same bytes as a read in the calling thread, so the forcing is bit for bit
the same. Workers: ``$FESOM_FORCING_PROCS`` (``gpu.forcing_procs``), 0 (the default) = no workers, every read in
the calling thread. A spawned worker imports the ``fesom_jax`` package (and so JAX, without touching a device).
"""

from __future__ import annotations

import concurrent.futures
import multiprocessing
import os

import numpy as np

_DS = {}                                   # a worker's open datasets, by path


def read_slice(path: str, var: str, t: int) -> np.ndarray:
    """Record ``t`` (1-based) of ``var`` in ``path``, (lat, lon) as stored."""
    import netCDF4
    ds = _DS.get(path)
    if ds is None:
        ds = netCDF4.Dataset(path)
        for v in ds.variables.values():
            v.set_auto_maskandscale(False)
        _DS[path] = ds
    return np.asarray(ds.variables[var][t - 1, :, :])


def workers() -> int:
    return int(os.environ.get("FESOM_FORCING_PROCS", "0") or 0)


class ReadAhead:
    """A pool of ``n`` spawned processes and the slices asked for: (path, t) -> future."""

    def __init__(self, n: int):
        ctx = multiprocessing.get_context("spawn")
        self.pool = concurrent.futures.ProcessPoolExecutor(n, mp_context=ctx)
        self.pending = {}

    def want(self, path: str, var: str, t: int):
        key = (path, t)
        if key not in self.pending:
            self.pending[key] = self.pool.submit(read_slice, path, var, t)

    def take(self, path: str, t: int):
        """The slice if it was asked for (waits for it), else None (the caller reads it itself)."""
        fut = self.pending.pop((path, t), None)
        return None if fut is None else fut.result()

    def forget(self, path: str):
        """Drop what was asked for from ``path`` (a year's file the reader has left)."""
        for key in [k for k in self.pending if k[0] == path]:
            self.pending.pop(key).cancel()

    def close(self):
        self.pool.shutdown(wait=False, cancel_futures=True)
        self.pending = {}
