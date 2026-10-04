import csv
import fcntl
import io
import json
import multiprocessing
import os
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from zipfile import BadZipFile

import numpy as np
import pymatching

from _correlated_decoder import (
    CorrelatedDecoder,
    apply_reweighted_tables_to_matcher,
    compute_reweighted_tables,
    reset_matcher_edge_weights,
)

SCRIPT_DIR = Path(__file__).resolve().parent.parent
GAP_DIR = Path(__file__).resolve().parents[2] / "data" / "sampled_gap"
_SAMPLE_CACHE: dict[tuple[int, float], dict[str, np.ndarray]] = {}

_GAP_DTYPE = np.float32

PACKED_DTYPE = np.dtype(
    [("gaps", _GAP_DTYPE), ("actual", np.bool_), ("predicted", np.bool_)],
    align=True,
)

MISMATCH_BUCKET_SHIFT = 8

CIRCUIT_METADATA = re.compile(
    r"circuit_(?:\w+?_)?d(?P<d>\d+)(?:xd\d+)*_p(?P<p>[\d.]+)\.stim$"
)

CSV_COLUMNS = [
    "d",
    "p",
    "gap",
    "actual_observable",
    "predicted_observable",
]


def parse_circuit_kind(circuit_path: Path) -> str:
    try:
        relative = circuit_path.resolve().relative_to(SCRIPT_DIR / "circuit")
    except ValueError as exc:
        raise ValueError(
            f"Circuit must live under {SCRIPT_DIR / 'circuit'}/time/ or "
            f"{SCRIPT_DIR / 'circuit'}/space/: {circuit_path}"
        ) from exc

    kind = relative.parts[0]
    if kind not in {"time", "space"}:
        raise ValueError(
            f"Expected circuit under time/ or space/, got {kind!r} in {circuit_path}"
        )
    return kind


def parse_circuit_metadata(circuit_path: Path) -> dict:
    match = CIRCUIT_METADATA.search(circuit_path.name)
    if match is None:
        raise ValueError(f"Could not parse d, p from {circuit_path}")
    return {
        "kind": parse_circuit_kind(circuit_path),
        "d": int(match.group("d")),
        "p": float(match.group("p")),
    }


def default_csv_path(circuit_path: Path) -> Path:
    meta = parse_circuit_metadata(circuit_path)
    return SCRIPT_DIR / "csv" / meta["kind"] / f"gap_d{meta['d']}_p{meta['p']}.csv"


class GapCsvWriter:
    """Append gap-sample rows incrementally, flushing after each batch."""

    def __init__(self, csv_path: Path):
        self.csv_path = csv_path
        self._file = None
        self._writer = None

    def __enter__(self):
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not self.csv_path.exists() or self.csv_path.stat().st_size == 0
        self._file = open(self.csv_path, "a", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=CSV_COLUMNS)
        if write_header:
            self._writer.writeheader()
            self._file.flush()
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._file is not None:
            self._file.close()
        self._file = None
        self._writer = None

    def append_rows(self, rows: list[dict]) -> None:
        for row in rows:
            self._writer.writerow(
                {
                    "d": row["d"],
                    "p": row["p"],
                    "gap": f"{row['gap']:.6f}",
                    "actual_observable": json.dumps(
                        row["actual_observable"], separators=(",", ":")
                    ),
                    "predicted_observable": json.dumps(
                        row["predicted_observable"], separators=(",", ":")
                    ),
                }
            )
        self._file.flush()

    def append_arrays(
        self,
        *,
        d: int,
        p: float,
        gaps: np.ndarray,
        actual: np.ndarray,
        predicted: np.ndarray,
    ) -> None:
        """Write a chunk from compact arrays (avoids per-row dicts)."""
        write = self._file.write
        p_str = str(p)
        d_str = str(d)
        for i in range(gaps.shape[0]):
            act = "true" if actual[i] else "false"
            write(f"{d_str},{p_str},{gaps[i]:.6f},[{act}],[{int(predicted[i])}]\n")
        self._file.flush()


def _gap_for_shot(
    syndrome: np.ndarray,
    *,
    decoder: CorrelatedDecoder,
    membrane_decoder: pymatching.Matching,
    membrane_matcher_edges: dict,
    physical_to_membrane: dict,
    syndrome_membrane: np.ndarray,
    l_indices: list[int],
    membrane_sets: list[list[int]],
    l_to_detector: dict[int, int],
    num_physical_detectors: int,
) -> tuple[np.ndarray, float, list[float]]:
    first_edges = decoder.first_pass_edges(syndrome)
    reweighted = compute_reweighted_tables(decoder.graph.edges, first_edges)
    pred, initial_weight = decoder.decode(syndrome)

    membrane_changed = apply_reweighted_tables_to_matcher(
        membrane_decoder,
        membrane_matcher_edges,
        decoder.graph.edges,
        reweighted,
        edge_map=physical_to_membrane,
    )
    try:
        syndrome_membrane.fill(False)
        syndrome_membrane[:num_physical_detectors] = syndrome
        for l in l_indices:
            syndrome_membrane[l_to_detector[l]] = pred[l]

        gap_list: list[float] = []
        for membrane_set in membrane_sets:
            ca = l_to_detector[membrane_set[0]]
            cb = l_to_detector[membrane_set[1]]
            syndrome_membrane[ca] ^= True
            syndrome_membrane[cb] ^= True
            _, weight = membrane_decoder.decode(syndrome_membrane, return_weight=True)
            syndrome_membrane[ca] ^= True
            syndrome_membrane[cb] ^= True
            gap_list.append(weight - initial_weight)
        return pred, initial_weight, gap_list
    finally:
        reset_matcher_edge_weights(
            membrane_decoder,
            membrane_matcher_edges,
            membrane_changed,
        )


def _npz_path(csv_path: Path) -> Path:
    return csv_path.with_suffix(".npz")


def _mmap_dir(csv_path: Path) -> Path:
    return csv_path.with_suffix(".mmap")


def _normalize_samples(samples: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Canonical in-memory / on-disk dtypes used by Patch/Simulator."""
    return {
        "gaps": np.asarray(samples["gaps"], dtype=_GAP_DTYPE),
        "actual": np.asarray(samples["actual"], dtype=bool),
        "predicted": np.asarray(samples["predicted"], dtype=bool),
    }


def _mmap_is_fresh(mmap_dir: Path, csv_path: Path) -> bool:
    gaps_path = mmap_dir / "gaps.npy"
    if not gaps_path.is_file():
        return False
    if not csv_path.is_file():
        return True
    csv_mtime = csv_path.stat().st_mtime
    required = ("gaps.npy", "actual.npy", "predicted.npy")
    return all(
        (mmap_dir / name).is_file() and (mmap_dir / name).stat().st_mtime >= csv_mtime
        for name in required
    )


def _load_from_mmap(mmap_dir: Path) -> dict[str, np.ndarray]:
    return {
        "gaps": np.load(mmap_dir / "gaps.npy", mmap_mode="r"),
        "actual": np.load(mmap_dir / "actual.npy", mmap_mode="r"),
        "predicted": np.load(mmap_dir / "predicted.npy", mmap_mode="r"),
    }


def _save_mmap(mmap_dir: Path, samples: dict[str, np.ndarray]) -> None:
    samples = _normalize_samples(samples)
    mmap_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = mmap_dir.with_suffix(".mmap.tmp")
    if tmp_dir.exists():
        for path in tmp_dir.glob("*"):
            path.unlink()
        tmp_dir.rmdir()
    tmp_dir.mkdir(parents=True, exist_ok=True)
    try:
        for key in ("gaps", "actual", "predicted"):
            np.save(tmp_dir / f"{key}.npy", samples[key])
        mmap_dir.mkdir(parents=True, exist_ok=True)
        for key in ("gaps", "actual", "predicted"):
            os.replace(tmp_dir / f"{key}.npy", mmap_dir / f"{key}.npy")
    finally:
        if tmp_dir.exists():
            for path in tmp_dir.glob("*"):
                path.unlink()
            tmp_dir.rmdir()


def _load_from_npz(npz_path: Path) -> dict[str, np.ndarray]:
    with np.load(npz_path) as data:
        return _normalize_samples(
            {
                "gaps": data["gaps"],
                "actual": data["actual"],
                "predicted": data["predicted"],
            }
        )


def _try_load_from_npz(npz_path: Path) -> dict[str, np.ndarray] | None:
    try:
        return _load_from_npz(npz_path)
    except (BadZipFile, EOFError, OSError, KeyError, ValueError):
        return None


_MIN_CHUNK_BYTES = 4 << 20


def _csv_jobs(jobs: int | None) -> int:
    """Resolve the worker count: explicit arg > GAP_CSV_JOBS env > all cores."""
    if jobs is None:
        env = os.environ.get("GAP_CSV_JOBS")
        jobs = int(env) if env else (os.cpu_count() or 1)
    if multiprocessing.current_process().daemon:
        return 1
    return max(1, jobs)


def _read_csv_header(csv_path: Path) -> tuple[tuple[int, int, int], int]:
    """Return ((gap_i, actual_i, predicted_i), byte offset of the first data row)."""
    with open(csv_path, "rb") as f:
        line = f.readline()
        data_start = f.tell()
    if not line:
        raise ValueError(f"Empty CSV: {csv_path}")
    header = next(csv.reader([line.decode()]))
    cols = (
        header.index("gap"),
        header.index("actual_observable"),
        header.index("predicted_observable"),
    )
    return cols, data_start


def _csv_chunk_ranges(csv_path: Path, data_start: int, n_chunks: int) -> list[tuple[int, int]]:
    """Split [data_start, EOF) into ``<= n_chunks`` newline-aligned byte ranges."""
    size = csv_path.stat().st_size
    body = size - data_start
    if body <= 0:
        return []
    n_chunks = max(1, min(n_chunks, body // _MIN_CHUNK_BYTES))
    if n_chunks == 1:
        return [(data_start, size)]

    cuts = [data_start]
    with open(csv_path, "rb") as f:
        for k in range(1, n_chunks):
            target = data_start + (body * k) // n_chunks
            if target <= cuts[-1]:
                continue
            f.seek(target)
            f.readline()
            pos = f.tell()
            if pos >= size:
                break
            cuts.append(pos)
    cuts.append(size)
    return [(a, b) for a, b in zip(cuts, cuts[1:]) if b > a]


def _parse_csv_range(
    csv_path: Path, start: int, end: int, cols: tuple[int, int, int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Parse the rows in bytes [start, end) of ``csv_path`` (worker side)."""
    gap_i, actual_i, predicted_i = cols
    with open(csv_path, "rb") as f:
        f.seek(start)
        raw = f.read(end - start)
    n = raw.count(b"\n")
    if raw and not raw.endswith(b"\n"):
        n += 1

    gaps = np.empty(n, dtype=_GAP_DTYPE)
    actual = np.empty(n, dtype=bool)
    predicted = np.empty(n, dtype=bool)

    i = 0
    for row in csv.reader(io.StringIO(raw.decode(), newline="")):
        if not row:
            continue
        gaps[i] = float(row[gap_i])
        actual[i] = "true" in row[actual_i].lower()
        predicted[i] = bool(int(row[predicted_i].strip("[]")))
        i += 1
    return gaps[:i], actual[:i], predicted[:i]


def _read_from_csv_serial(csv_path: Path) -> dict[str, np.ndarray]:
    n = 0
    with open(csv_path, "rb") as f:
        n = max(sum(1 for _ in f) - 1, 0)
    if n == 0:
        raise ValueError(f"No samples found in {csv_path}")

    gaps = np.empty(n, dtype=_GAP_DTYPE)
    actual = np.empty(n, dtype=bool)
    predicted = np.empty(n, dtype=bool)

    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header is None:
            raise ValueError(f"Empty CSV: {csv_path}")
        gap_i = header.index("gap")
        actual_i = header.index("actual_observable")
        predicted_i = header.index("predicted_observable")
        for i, row in enumerate(reader):
            gaps[i] = float(row[gap_i])
            actual[i] = "true" in row[actual_i].lower()
            predicted[i] = bool(int(row[predicted_i].strip("[]")))

    return {"gaps": gaps, "actual": actual, "predicted": predicted}


def _read_from_csv(csv_path: Path, jobs: int | None = None) -> dict[str, np.ndarray]:
    """Parse a gap CSV into arrays, using ``jobs`` worker processes."""
    jobs = _csv_jobs(jobs)
    if jobs == 1:
        return _read_from_csv_serial(csv_path)

    cols, data_start = _read_csv_header(csv_path)
    ranges = _csv_chunk_ranges(csv_path, data_start, jobs * 3)
    if not ranges:
        raise ValueError(f"No samples found in {csv_path}")
    if len(ranges) == 1:
        return _read_from_csv_serial(csv_path)

    with ProcessPoolExecutor(max_workers=min(jobs, len(ranges))) as pool:
        parts = list(
            pool.map(
                _parse_csv_range,
                [csv_path] * len(ranges),
                *zip(*ranges),
                [cols] * len(ranges),
            )
        )

    gaps = np.concatenate([p[0] for p in parts])
    if gaps.size == 0:
        raise ValueError(f"No samples found in {csv_path}")
    return {
        "gaps": gaps,
        "actual": np.concatenate([p[1] for p in parts]),
        "predicted": np.concatenate([p[2] for p in parts]),
    }


def _save_npz(npz_path: Path, samples: dict[str, np.ndarray]) -> None:
    samples = _normalize_samples(samples)
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = npz_path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp_path, **samples)
    os.replace(tmp_path, npz_path)


def _ensure_mmap(csv_path: Path, mmap_dir: Path) -> dict[str, np.ndarray]:
    lock_path = mmap_dir.with_suffix(".mmap.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "w") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        if _mmap_is_fresh(mmap_dir, csv_path):
            return _load_from_mmap(mmap_dir)

        npz_path = _npz_path(csv_path)
        if npz_path.is_file() and (
            not csv_path.is_file()
            or npz_path.stat().st_mtime >= csv_path.stat().st_mtime
        ):
            loaded = _try_load_from_npz(npz_path)
            if loaded is not None:
                _save_mmap(mmap_dir, loaded)
                return _load_from_mmap(mmap_dir)

        if not csv_path.is_file():
            raise FileNotFoundError(f"Gap CSV not found: {csv_path}")
        samples = _read_from_csv(csv_path)
        _save_mmap(mmap_dir, samples)
        return _load_from_mmap(mmap_dir)


def _load_samples(csv_path: Path) -> dict[str, np.ndarray]:
    mmap_dir = _mmap_dir(csv_path)
    if _mmap_is_fresh(mmap_dir, csv_path):
        return _load_from_mmap(mmap_dir)
    return _ensure_mmap(csv_path, mmap_dir)


def get_gap_samples(d: int, p: float) -> dict[str, np.ndarray]:
    key = (d, p)
    if key not in _SAMPLE_CACHE:
        csv_path = GAP_DIR / f"gap_d{d}_p{p}.csv"
        npz_path = _npz_path(csv_path)
        mmap_dir = _mmap_dir(csv_path)
        if (
            not csv_path.is_file()
            and not npz_path.is_file()
            and not _mmap_is_fresh(mmap_dir, csv_path)
        ):
            raise FileNotFoundError(f"Gap CSV not found: {csv_path}")
        _SAMPLE_CACHE[key] = _load_samples(csv_path)
    return _SAMPLE_CACHE[key]


def preload_gap_samples(d: int, p: float) -> dict[str, np.ndarray]:
    """Load samples in the parent process so forked workers share pages."""
    return get_gap_samples(d, p)


def _set_madvise_hugepage(enabled: bool):
    """Toggle numpy's MADV_HUGEPAGE hint; returns the previous setting or None."""
    for mod in ("_core", "core"):
        try:
            fn = getattr(np, mod).multiarray._set_madvise_hugepage
        except AttributeError:
            continue
        return fn(enabled)
    return None


def hugepage_coverage(arr: np.ndarray) -> tuple[int, int] | None:
    """(huge_kB, rss_kB) of the mappings holding ``arr``, from /proc/self/smaps."""
    lo = arr.ctypes.data
    hi = lo + arr.nbytes
    try:
        with open("/proc/self/smaps") as f:
            lines = f.read().splitlines()
    except OSError:
        return None
    inside = False
    huge = rss = 0
    found = False
    for line in lines:
        m = re.match(r"^([0-9a-f]+)-([0-9a-f]+) ", line)
        if m:
            start, end = int(m.group(1), 16), int(m.group(2), 16)
            inside = start < hi and lo < end
            found |= inside
        elif inside:
            if line.startswith("AnonHugePages:"):
                huge += int(line.split()[1])
            elif line.startswith("Rss:"):
                rss += int(line.split()[1])
    return (huge, rss) if found else None


def _load_into_ram(mmap_dir: Path) -> dict[str, np.ndarray]:
    """Read the three sample files into one anonymous, packed array."""
    want_huge = os.environ.get("GAP_SIM_NO_HUGEPAGE", "") not in ("1", "true", "yes")
    restore = _set_madvise_hugepage(False)
    try:
        gaps = np.load(mmap_dir / "gaps.npy")
        actual = np.load(mmap_dir / "actual.npy")
        predicted = np.load(mmap_dir / "predicted.npy")
        if want_huge:
            _set_madvise_hugepage(True)
        packed = np.empty(len(gaps), dtype=PACKED_DTYPE)
        _set_madvise_hugepage(False)
        packed["gaps"] = gaps
        packed["actual"] = actual
        packed["predicted"] = predicted
        del gaps, actual, predicted
        buckets = mismatch_bucket_table(packed)
    finally:
        if restore is not None:
            _set_madvise_hugepage(restore)
    return {
        "gaps": packed["gaps"],
        "actual": packed["actual"],
        "predicted": packed["predicted"],
        "packed": packed,
        "mismatch_buckets": buckets,
    }


def mismatch_bucket_table(
    packed: np.ndarray, shift: int = MISMATCH_BUCKET_SHIFT, chunk: int = 1 << 24
) -> np.ndarray:
    """uint8 table with table[i >> shift] == 1 for every mismatching sample i."""
    n = len(packed)
    table = np.zeros((n >> shift) + 1, dtype=np.uint8)
    actual = packed["actual"]
    predicted = packed["predicted"]
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        hits = np.flatnonzero(actual[start:stop] != predicted[start:stop])
        if hits.size:
            table[(hits + start) >> shift] = 1
    return table


def materialize_gap_samples(d: int, p: float) -> dict[str, np.ndarray]:
    """Replace the memory-mapped sample set with an in-RAM packed copy."""
    samples = get_gap_samples(d, p)
    if "packed" in samples:
        return samples
    csv_path = GAP_DIR / f"gap_d{d}_p{p}.csv"
    in_ram = _load_into_ram(_mmap_dir(csv_path))
    if len(in_ram["gaps"]) != len(samples["gaps"]):
        raise RuntimeError(
            f"gap sample set d={d} p={p} changed on disk while loading"
        )
    samples.update(in_ram)
    return samples
