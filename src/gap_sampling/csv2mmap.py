"""Build memory-mapped .npy caches from the sampled gap CSVs."""

import argparse
import os
import shutil
import time
from pathlib import Path

GAP_DIR = Path(__file__).resolve().parents[2] / "data" / "sampled_gap"

from _gap_samples import (  # noqa: E402
    _mmap_dir,
    _mmap_is_fresh,
    _read_from_csv,
    _save_mmap,
)


def _dir_size(path: Path) -> int:
    return sum(f.stat().st_size for f in path.glob("*") if f.is_file())


def _free_bytes(path: Path) -> int:
    return shutil.disk_usage(path).free


def build_one(csv_path: Path, *, force: bool = False, jobs: int = 1) -> bool:
    csv_path = csv_path.resolve()
    if not csv_path.is_file():
        print(f"[skip]  not a file: {csv_path}")
        return False

    mmap_dir = _mmap_dir(csv_path)
    if not force and _mmap_is_fresh(mmap_dir, csv_path):
        print(f"[fresh] {mmap_dir}  (up to date; use --force to rebuild)")
        return True

    csv_gb = csv_path.stat().st_size / 1e9
    free_gb = _free_bytes(csv_path.parent) / 1e9
    print(
        f"[build] {csv_path}  ({csv_gb:.2f} GB csv, {free_gb:.2f} GB free, {jobs} workers)",
        flush=True,
    )

    t = time.perf_counter()
    samples = _read_from_csv(csv_path, jobs=jobs)
    t_parse = time.perf_counter() - t
    _save_mmap(mmap_dir, samples)
    n = len(samples["gaps"])
    out_gb = _dir_size(mmap_dir) / 1e9
    print(
        f"[done]  {n:,} samples -> {mmap_dir} ({out_gb:.2f} GB) "
        f"in {time.perf_counter() - t:.1f}s (parse {t_parse:.1f}s)"
    )
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build memory-mapped .npy caches from gap-sample CSVs."
    )
    parser.add_argument(
        "csv",
        nargs="*",
        type=Path,
        help="CSV file path(s) (default: every gap_*.csv in data/sampled_gap).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild even if the mmap is already fresh.",
    )
    parser.add_argument(
        "--jobs",
        "-j",
        type=int,
        default=os.cpu_count() or 1,
        help="Worker processes used to parse each CSV (default: all cores; 1 = serial).",
    )
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be >= 1")

    targets: list[Path] = list(args.csv) or sorted(GAP_DIR.glob("gap_d*_p*.csv"))
    if not targets:
        parser.error(f"no gap_*.csv found in {GAP_DIR}")

    seen: set[Path] = set()
    ok = 0
    total = 0
    for csv_path in targets:
        rp = csv_path.resolve()
        if rp in seen:
            continue
        seen.add(rp)
        total += 1
        if build_one(rp, force=args.force, jobs=args.jobs):
            ok += 1
    print(f"\n{ok}/{total} datasets ready.")


if __name__ == "__main__":
    main()
