"""Fast aggregation of sinter result CSVs for plotting."""

import csv
import hashlib
import json
import mmap
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import sinter


CACHE_VERSION = 2
CACHE_DIR_NAME = ".plot_cache"
_CACHE_TAIL_BYTES = 4096
_PARSE_CHUNK_BYTES = 64 << 20
_NL, _COMMA, _DOT, _SPACE, _ZERO = 10, 44, 46, 32, 48
_REQUIRED_COLUMNS = (
    "shots",
    "errors",
    "discards",
    "seconds",
    "strong_id",
    "decoder",
    "json_metadata",
)
_NUMERIC_COLUMNS = ("shots", "errors", "discards", "seconds")


class _UnsupportedLayout(Exception):
    """Row layout the vectorised parser does not handle; use csv.reader."""


def _parse_columns(header_line: str, path: Path) -> dict[str, int]:
    header = [name.strip() for name in header_line.rstrip("\r\n").split(",")]
    columns = {name: index for index, name in enumerate(header)}
    if not set(_REQUIRED_COLUMNS) <= set(columns):
        raise ValueError(
            f"Bad CSV data in {path}. "
            f"Got columns {sorted(columns)!r} but expected "
            f"{sorted(_REQUIRED_COLUMNS)!r}."
        )
    return columns


def _accumulate(
    entries: dict,
    strong_id: str,
    shots,
    errors,
    discards,
    seconds,
    decoder: str,
    metadata_json: str,
) -> None:
    """Add one row (or one pre-summed group of rows) to ``entries``."""
    entry = entries.get(strong_id)
    if entry is None:
        entry = entries[strong_id] = [0, 0, 0, 0.0, decoder, json.loads(metadata_json)]
    entry[0] += int(shots)
    entry[1] += int(errors)
    entry[2] += int(discards)
    entry[3] += float(seconds)


def _aggregate_lines_python(lines, columns: dict[str, int], entries: dict) -> None:
    """Reference path: plain csv.reader over decoded lines (no header)."""
    shots_i = columns["shots"]
    errors_i = columns["errors"]
    discards_i = columns["discards"]
    seconds_i = columns["seconds"]
    strong_id_i = columns["strong_id"]
    decoder_i = columns["decoder"]
    metadata_i = columns["json_metadata"]
    for row in csv.reader(lines):
        if not row:
            continue
        _accumulate(
            entries,
            row[strong_id_i],
            row[shots_i],
            row[errors_i],
            row[discards_i],
            row[seconds_i],
            row[decoder_i],
            row[metadata_i],
        )


def _parse_numeric_prefix(
    prefix: np.ndarray, comma_cols: np.ndarray, columns: dict[str, int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Parse shots/errors/discards/seconds from a contiguous (n, P) uint8 block."""
    width = prefix.shape[1]
    is_digit = (prefix >= _ZERO) & (prefix <= _ZERO + 9)
    is_dot = prefix == _DOT
    is_space = prefix == _SPACE
    is_comma = prefix == _COMMA
    if not (is_digit | is_dot | is_space | is_comma).all():
        raise _UnsupportedLayout
    if (
        not is_comma[:, comma_cols].all()
        or (is_comma.sum(axis=1) != comma_cols.size).any()
    ):
        raise _UnsupportedLayout
    if (is_space[:, 1:] & (is_digit | is_dot)[:, :-1]).any():
        raise _UnsupportedLayout

    def span(index: int) -> tuple[int, int]:
        start = int(comma_cols[index - 1]) + 1 if index > 0 else 0
        return start, int(comma_cols[index])

    fields = [span(columns[name]) for name in _NUMERIC_COLUMNS]
    weights = np.zeros((width, len(fields)), dtype=np.float64)
    for k, (a, b) in enumerate(fields):
        if b - a == 0 or b - a > 15:
            raise _UnsupportedLayout
        weights[a:b, k] = 10.0 ** np.arange(b - a - 1, -1, -1)
    int_spans = fields[:3]
    if any(is_dot[:, a:b].any() for a, b in int_spans):
        raise _UnsupportedLayout
    digits = np.where(is_digit, prefix, _ZERO).astype(np.float64) - _ZERO
    values = digits @ weights
    ints = values[:, :3]
    if (ints >= 2.0**53).any():
        raise _UnsupportedLayout

    a, b = fields[3]
    sec_width = b - a
    dot_marker = np.zeros(width, dtype=np.float64)
    dot_marker[a:b] = np.arange(1, sec_width + 1)
    dot_pos = is_dot.astype(np.float64) @ dot_marker
    if (is_dot[:, a:b].sum(axis=1) > 1).any():
        raise _UnsupportedLayout
    raw = values[:, 3]
    frac_scale = 10.0 ** (sec_width - dot_pos)
    left = np.floor(raw / (10.0 * frac_scale))
    frac = raw - left * 10.0 * frac_scale
    seconds = np.where(dot_pos > 0, (left * frac_scale + frac) / frac_scale, raw)
    return (
        ints[:, 0].astype(np.int64),
        ints[:, 1].astype(np.int64),
        ints[:, 2].astype(np.int64),
        seconds,
    )


def _group_sums(inverse: np.ndarray, values: np.ndarray, n_groups: int) -> np.ndarray:
    if values.dtype.kind == "f":
        return np.bincount(inverse, weights=values, minlength=n_groups)
    if values.size and int(values.max()) * values.size < 2**53:
        return np.bincount(inverse, weights=values, minlength=n_groups).astype(np.int64)
    out = np.zeros(n_groups, dtype=np.int64)
    np.add.at(out, inverse, values)
    return out


def _group_rows_by_rest(rest: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Group identical rows of a contiguous (n, R) uint8 array."""
    n_rows, rest_len = rest.shape
    pad = (-rest_len) % 4
    if pad:
        rest = np.concatenate([rest, np.zeros((n_rows, pad), dtype=np.uint8)], axis=1)
    words = rest.view(np.uint32).astype(np.float64)
    mix = (
        np.cos(np.arange(words.shape[1], dtype=np.float64) * 12.9898 + 78.233)
        * 43758.5453
    )
    hashes = words @ mix
    _, first_index, inverse = np.unique(hashes, return_index=True, return_inverse=True)
    inverse = inverse.ravel()
    if (rest == rest[first_index][inverse]).all():
        return first_index, inverse
    rest_void = np.ascontiguousarray(rest).view(f"V{rest.shape[1]}").ravel()
    _, first_index, inverse = np.unique(
        rest_void, return_index=True, return_inverse=True
    )
    return first_index, inverse.ravel()


def _aggregate_rows_numpy(
    rows: np.ndarray, columns: dict[str, int], entries: dict
) -> None:
    """Aggregate an (n, row_len) uint8 array of equal-length CSV rows."""
    n_rows, row_len = rows.shape
    if n_rows == 0:
        return
    metadata_i = columns["json_metadata"]
    rest_field = min(columns["decoder"], columns["strong_id"], metadata_i)
    if max(columns[name] for name in _NUMERIC_COLUMNS) >= rest_field:
        raise _UnsupportedLayout
    comma_cols = np.flatnonzero(rows[0] == _COMMA)[:rest_field]
    if comma_cols.size < rest_field:
        raise _UnsupportedLayout
    rest_start = int(comma_cols[-1]) + 1
    prefix = np.ascontiguousarray(rows[:, :rest_start])
    shots, errors, discards, seconds = _parse_numeric_prefix(
        prefix, comma_cols, columns
    )

    rest = np.ascontiguousarray(rows[:, rest_start:])
    first_index, inverse = _group_rows_by_rest(rest)
    order = np.argsort(first_index, kind="stable")
    first_index = first_index[order]
    inverse = np.argsort(order, kind="stable")[inverse]
    representatives = rest[first_index]
    if (representatives[:, -1] != _NL).any() or (representatives[:, :-1] == _NL).any():
        raise _UnsupportedLayout
    n_groups = first_index.size
    group_shots = _group_sums(inverse, shots, n_groups)
    group_errors = _group_sums(inverse, errors, n_groups)
    group_discards = _group_sums(inverse, discards, n_groups)
    group_seconds = _group_sums(inverse, seconds, n_groups)

    lines = [rows[i].tobytes().decode() for i in first_index]
    strong_id_i = columns["strong_id"]
    decoder_i = columns["decoder"]
    for g, row in enumerate(csv.reader(lines)):
        _accumulate(
            entries,
            row[strong_id_i],
            int(group_shots[g]),
            int(group_errors[g]),
            int(group_discards[g]),
            float(group_seconds[g]),
            row[decoder_i],
            row[metadata_i],
        )


def _aggregate_block(
    mm, start: int, end: int, columns: dict[str, int], entries: dict
) -> None:
    """Aggregate mm[start:end], a run of complete lines ending in a newline."""
    buf = np.frombuffer(mm, dtype=np.uint8, offset=start, count=end - start)
    row_len = mm.find(b"\n", start, end) - start + 1
    n_rows = (end - start) // row_len
    if n_rows * row_len == end - start:
        try:
            _aggregate_rows_numpy(buf.reshape(n_rows, row_len), columns, entries)
            return
        except _UnsupportedLayout:
            pass
    newlines = np.flatnonzero(buf == _NL)
    starts = np.empty(newlines.size, dtype=np.int64)
    starts[0] = 0
    starts[1:] = newlines[:-1] + 1
    lengths = newlines + 1 - starts
    for length in np.unique(lengths):
        class_starts = starts[lengths == length]
        rows = buf[class_starts[:, None] + np.arange(int(length))]
        try:
            _aggregate_rows_numpy(rows, columns, entries)
        except _UnsupportedLayout:
            lines = [
                mm[start + s : start + s + int(length)].decode().rstrip("\r\n")
                for s in class_starts.tolist()
            ]
            _aggregate_lines_python(lines, columns, entries)


def _aggregate_from_offset(
    f, offset: int, size: int, columns: dict[str, int], entries: dict
) -> int:
    """Aggregate complete lines of f[offset:size]; return the offset consumed."""
    if size <= offset:
        return offset
    mm = mmap.mmap(f.fileno(), size, access=mmap.ACCESS_READ)
    try:
        consumed = offset
        while consumed < size:
            end = min(size, consumed + _PARSE_CHUNK_BYTES)
            cut = mm.rfind(b"\n", consumed, end) + 1
            if cut <= consumed:
                if end == size:
                    break
                cut = mm.find(b"\n", end, size) + 1
                if cut == 0:
                    break
            _aggregate_block(mm, consumed, cut, columns, entries)
            consumed = cut
    finally:
        mm.close()
    return consumed


def _cache_path(path: Path) -> Path:
    return path.parent / CACHE_DIR_NAME / (path.name + ".json")


def _tail_digest(f, offset: int) -> str:
    start = max(0, offset - _CACHE_TAIL_BYTES)
    f.seek(start)
    return hashlib.sha256(f.read(offset - start)).hexdigest()


def _load_cache(
    path: Path, f, header_line: bytes, size: int
) -> tuple[int, dict] | None:
    try:
        raw = json.loads(_cache_path(path).read_text())
        if (
            raw["version"] != CACHE_VERSION
            or raw["header"] != header_line.decode(errors="replace")
            or not isinstance(raw["offset"], int)
            or raw["offset"] > size
        ):
            return None
        offset = raw["offset"]
        if _tail_digest(f, offset) != raw["tail_digest"]:
            return None
        entries = {
            strong_id: list(entry) for strong_id, entry in raw["entries"].items()
        }
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return None
    return offset, entries


def _store_cache(path: Path, f, header_line: bytes, offset: int, entries: dict) -> None:
    cache_path = _cache_path(path)
    payload = {
        "version": CACHE_VERSION,
        "header": header_line.decode(errors="replace"),
        "offset": offset,
        "tail_digest": _tail_digest(f, offset),
        "entries": entries,
    }
    tmp_name = None
    try:
        cache_path.parent.mkdir(exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(dir=cache_path.parent, prefix=cache_path.name)
        with os.fdopen(fd, "w") as tmp:
            json.dump(payload, tmp)
        os.replace(tmp_name, cache_path)
    except OSError as exc:
        print(f"warning: could not write {cache_path}: {exc}", flush=True)
        if tmp_name is not None:
            try:
                os.unlink(tmp_name)
            except OSError:
                pass


def _aggregate_csv_file(
    path: Path, *, use_cache: bool = True
) -> dict[str, sinter.TaskStats]:
    """Aggregate one sinter CSV per strong_id (vectorised, incrementally cached)."""
    with path.open("rb") as f:
        header_line = f.readline()
        columns = _parse_columns(header_line.decode(errors="replace"), path)
        size = path.stat().st_size
        cached = _load_cache(path, f, header_line, size) if use_cache else None
        if cached is None:
            offset, entries = len(header_line), {}
        else:
            offset, entries = cached
        consumed = _aggregate_from_offset(f, offset, size, columns, entries)
        if use_cache and (cached is None or consumed != offset):
            _store_cache(path, f, header_line, consumed, entries)

    return {
        strong_id: sinter.TaskStats(
            shots=shots,
            errors=errors,
            discards=discards,
            seconds=seconds,
            strong_id=strong_id,
            decoder=decoder,
            json_metadata=metadata,
        )
        for strong_id, (
            shots,
            errors,
            discards,
            seconds,
            decoder,
            metadata,
        ) in entries.items()
    }


def fast_read_stats_from_csv_files(
    *paths: str | Path, use_cache: bool = True
) -> list[sinter.TaskStats]:
    """Read and merge sinter stats from CSV files (files parsed in parallel)."""
    paths = [Path(path) for path in paths]
    workers = max(1, min(4, os.cpu_count() or 1, len(paths)))
    merged: dict[str, sinter.TaskStats] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        per_file = pool.map(
            lambda p: _aggregate_csv_file(p, use_cache=use_cache), paths
        )
        for file_stats in per_file:
            for strong_id, stat in file_stats.items():
                existing = merged.get(strong_id)
                merged[strong_id] = existing + stat if existing is not None else stat
    return list(merged.values())
