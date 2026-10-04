"""Parallel stable sort + per-gap aggregation of a gap sample set."""

from __future__ import annotations

import mmap
import multiprocessing as mp
from dataclasses import dataclass

import numpy as np

_BUCKET_WIDTH = 1000.0
_MAX_BUCKETS = 1 << 22

_STATE: dict = {}


@dataclass
class SortedSamples:
    """Sample set sorted by gap, plus its per-distinct-gap aggregates."""

    gaps: np.ndarray
    mismatch: np.ndarray
    fold: np.ndarray | None
    gap_values: np.ndarray
    trials: np.ndarray
    events: np.ndarray
    trials_fold: np.ndarray | None
    events_fold: np.ndarray | None


def shared_array(n: int, dtype) -> np.ndarray:
    """Zero-filled array in anonymous shared memory (visible to forked children)."""
    dtype = np.dtype(dtype)
    buf = mmap.mmap(-1, max(int(n) * dtype.itemsize, 1))
    return np.frombuffer(buf, dtype=dtype, count=int(n))


def _bucket(g: np.ndarray, gmin, width) -> np.ndarray:
    """Coarse gap bucket; monotone non-decreasing in g and deterministic."""
    dt = g.dtype if g.dtype.kind == "f" else np.dtype(np.float64)
    return np.floor((g - dt.type(gmin)) * dt.type(width)).astype(np.int64)


def _chunk_ranges(n: int, num_chunks: int) -> list[tuple[int, int]]:
    edges = np.linspace(0, n, num_chunks + 1).astype(np.int64)
    return [(int(a), int(b)) for a, b in zip(edges[:-1], edges[1:]) if b > a]


def _minmax_chunk(rng: tuple[int, int]):
    g = _STATE["gaps"][rng[0] : rng[1]]
    return float(g.min()), float(g.max())


def _hist_chunk(args):
    (lo, hi), gmin, width, nb = args
    b = _bucket(_STATE["gaps"][lo:hi], gmin, width)
    return np.bincount(b, minlength=nb)


def _range_ids(lo: int, hi: int, gmin, width, edges: np.ndarray) -> np.ndarray:
    b = _bucket(_STATE["gaps"][lo:hi], gmin, width)
    return (np.searchsorted(edges, b, side="right") - 1).astype(np.int16)


def _count_chunk(args):
    (lo, hi), gmin, width, edges, num_ranges = args
    return np.bincount(_range_ids(lo, hi, gmin, width, edges), minlength=num_ranges)


def _scatter_chunk(args):
    """Write this chunk's row indices, grouped by range, into ``sel``."""
    (lo, hi), gmin, width, edges, offsets, seed = args
    st = _STATE
    r = _range_ids(lo, hi, gmin, width, edges)
    order = np.argsort(r, kind="stable")
    counts = np.bincount(r, minlength=len(offsets))
    starts = np.cumsum(counts) - counts
    sel = st["sel"]
    for w in np.flatnonzero(counts):
        rows = order[starts[w] : starts[w] + counts[w]]
        sel[offsets[w] : offsets[w] + counts[w]] = rows + lo
    st["mismatch_raw"][lo:hi] = np.bitwise_xor(
        st["actual"][lo:hi], st["predicted"][lo:hi]
    ).astype(bool)
    if seed is not None:
        gen = np.random.default_rng(seed)
        gen.bit_generator.advance(lo)
        st["fold_raw"][lo:hi] = gen.random(hi - lo) < 0.5


def _sort_range(args):
    w, base, m = args
    st = _STATE
    sel = st["sel"][base : base + m]
    g_sel = st["gaps"][sel]
    order = np.argsort(g_sel, kind="stable")
    gs = g_sel[order].astype(np.float64)
    ys = st["mismatch_raw"][sel][order]
    st["out_gaps"][base : base + m] = gs
    st["out_mismatch"][base : base + m] = ys
    fold = None
    if st["fold_raw"] is not None:
        fold = st["fold_raw"][sel][order]
        st["out_fold"][base : base + m] = fold

    starts = np.flatnonzero(np.concatenate(([True], gs[1:] != gs[:-1])))
    trials = np.diff(np.append(starts, m)).astype(float)
    events = np.add.reduceat(ys.astype(np.int64), starts).astype(float)
    agg = [gs[starts], trials, events]
    if fold is not None:
        agg.append(np.add.reduceat(fold.astype(np.int64), starts).astype(float))
        agg.append(
            np.add.reduceat((fold & ys).astype(np.int64), starts).astype(float)
        )
    return w, agg


def sort_and_aggregate(
    gaps: np.ndarray,
    actual: np.ndarray,
    predicted: np.ndarray,
    *,
    jobs: int,
    fold_seed: int | None = None,
) -> SortedSamples:
    """Stable-sort a sample set by gap using ``jobs`` forked workers."""
    n = len(gaps)
    jobs = max(1, int(jobs))
    if n == 0:
        empty = np.zeros(0)
        return SortedSamples(
            empty, np.zeros(0, bool), None if fold_seed is None else np.zeros(0, bool),
            empty, empty, empty,
            None if fold_seed is None else empty, None if fold_seed is None else empty,
        )

    global _STATE
    _STATE = {
        "gaps": gaps,
        "actual": actual,
        "predicted": predicted,
        "sel": shared_array(n, np.int64),
        "mismatch_raw": shared_array(n, np.bool_),
        "fold_raw": None if fold_seed is None else shared_array(n, np.bool_),
        "out_gaps": shared_array(n, np.float64),
        "out_mismatch": shared_array(n, np.bool_),
        "out_fold": None if fold_seed is None else shared_array(n, np.bool_),
    }
    try:
        if jobs == 1:
            result = _run(n, jobs, fold_seed, map)
        else:
            pool = mp.get_context("fork").Pool(jobs)
            try:
                result = _run(n, jobs, fold_seed, pool.map)
                pool.close()
            except BaseException:
                pool.terminate()
                raise
            finally:
                pool.join()
    finally:
        _STATE = {}
    return result


def _run(n: int, jobs: int, fold_seed, pmap) -> SortedSamples:
    st = _STATE
    chunks = _chunk_ranges(n, min(n, 4 * jobs))

    mins, maxs = zip(*pmap(_minmax_chunk, chunks))
    gmin, gmax = min(mins), max(maxs)
    width = _BUCKET_WIDTH
    span = max(gmax - gmin, 0.0)
    if span * width > _MAX_BUCKETS:
        width = _MAX_BUCKETS / span
    nb = int(_bucket(np.array([gmax], dtype=st["gaps"].dtype), gmin, width)[0]) + 1
    hist = sum(pmap(_hist_chunk, [(c, gmin, width, nb) for c in chunks]))
    cum = np.cumsum(hist)
    num_ranges = max(1, min(jobs, n // 4096))
    targets = n * np.arange(1, num_ranges) / num_ranges
    edges = np.unique(
        np.concatenate(([0], np.searchsorted(cum, targets, side="left") + 1, [nb]))
    )
    edges = edges[edges <= nb]
    if edges[-1] != nb:
        edges = np.append(edges, nb)
    num_ranges = len(edges) - 1

    counts = np.stack(
        list(pmap(_count_chunk, [(c, gmin, width, edges, num_ranges) for c in chunks]))
    )
    range_sizes = counts.sum(axis=0)
    range_base = np.cumsum(range_sizes) - range_sizes
    before = np.cumsum(counts, axis=0) - counts
    offsets = range_base[None, :] + before
    list(
        pmap(
            _scatter_chunk,
            [
                (c, gmin, width, edges, offsets[i], fold_seed)
                for i, c in enumerate(chunks)
            ],
        )
    )

    tasks = [
        (w, int(range_base[w]), int(range_sizes[w]))
        for w in range(num_ranges)
        if range_sizes[w] > 0
    ]
    parts = sorted(pmap(_sort_range, tasks), key=lambda t: t[0])
    cols = [np.concatenate([agg[i] for _, agg in parts]) for i in range(len(parts[0][1]))]

    return SortedSamples(
        gaps=st["out_gaps"],
        mismatch=st["out_mismatch"],
        fold=st["out_fold"],
        gap_values=cols[0],
        trials=cols[1],
        events=cols[2],
        trials_fold=cols[3] if fold_seed is not None else None,
        events_fold=cols[4] if fold_seed is not None else None,
    )
