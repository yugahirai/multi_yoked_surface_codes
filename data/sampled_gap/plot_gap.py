import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CSV_DIR = SCRIPT_DIR
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "plots"

FIG_SIZE = (8, 9)
FONT_SIZE_TICK = 18
FONT_SIZE_LABEL = 20
FONT_SIZE_LEGEND = 16
FONT_SIZE_TITLE = 16

DEFAULT_BINS = 100

# x-axis range as (lo, hi), or None for automatic.
XLIM: tuple[float, float] | None = None
# y-axis range as (lo, hi), or None for automatic.
YLIM: tuple[float, float] | None = None

# Legend display names, keyed by dataset name (the CSV stem without the
# ``_dist`` suffix). Files not listed here fall back to their name.
LEGEND_LABELS: dict[str, str] = {
    "gap_d5_p0.001": "d = 5",
    "gap_d6_p0.001": "d = 6",
    "gap_d7_p0.001": "d = 7",
    "gap_d8_p0.001": "d = 8",
    "gap_d9_p0.001": "d = 9",
    "gap_d10_p0.001": "d = 10",
    "gap_d11_p0.001": "d = 11",
}

DIST_SUFFIX = "_dist.csv"


class GapHistogram:
    """Pre-binned signed-gap histogram as written by ``map2dist.py``.

    ``lo``/``hi`` are the bin edges of each row, ``counts`` the number of shots
    in that bin. Negative gaps are shots where the decoder failed
    (``actual XOR predicted == 1``).
    """

    def __init__(self, lo: np.ndarray, hi: np.ndarray, counts: np.ndarray):
        self.lo = lo
        self.hi = hi
        self.counts = counts

    @property
    def centers(self) -> np.ndarray:
        return (self.lo + self.hi) / 2

    @property
    def bin_width(self) -> float:
        """Width of the finest bin in the file (bins in a dist CSV are uniform)."""
        return float(np.min(self.hi - self.lo))

    def min(self) -> float:
        return float(self.lo.min())

    def max(self) -> float:
        return float(self.hi.max())


def dataset_name(path: Path) -> str:
    """``gap_d5_p0.001_dist.csv`` -> ``gap_d5_p0.001``."""
    return path.name.removesuffix(DIST_SUFFIX).removesuffix(".csv")


def load_gaps(csv_path: Path) -> GapHistogram:
    """Load a ``*_dist.csv`` (columns ``gap_lo,gap_hi,count``).

    The sign of the gap encodes decoder success: bins with negative gap hold
    shots where ``actual_observable XOR predicted_observable`` is 1.
    """
    if not csv_path.is_file():
        raise FileNotFoundError(f"Gap distribution CSV not found: {csv_path}")
    data = np.loadtxt(csv_path, delimiter=",", skiprows=1, ndmin=2)
    if data.size == 0:
        raise ValueError(f"No gap bins found in {csv_path}")
    lo, hi, counts = data[:, 0], data[:, 1], data[:, 2]
    if counts.sum() == 0:
        raise ValueError(f"No gap values found in {csv_path}")
    return GapHistogram(lo, hi, counts)


def compute_bin_edges(all_hists: list[GapHistogram], bins: int | str) -> np.ndarray:
    """Shared bin edges across all series.

    The input histograms are already binned, so the output edges are snapped
    to whole multiples of the finest input bin width: each output bin merges an
    integer number of input bins and no count is split between two bins.
    ``bins="auto"`` keeps the input binning as-is.
    """
    width = min(h.bin_width for h in all_hists)
    lo = min(h.min() for h in all_hists)
    hi = max(h.max() for h in all_hists)
    if not hi > lo:
        hi = lo + width
    if bins == "auto":
        step = 1
    else:
        step = max(1, int(np.ceil((hi - lo) / (int(bins) * width))))
    coarse = step * width
    # Align to a multiple of the coarse width so 0 stays on a bin edge and the
    # error/success sides never share a bin.
    start = np.floor(lo / coarse) * coarse
    n = int(np.ceil((hi - start) / coarse - 1e-9))
    return start + coarse * np.arange(n + 1)


def plot_gap_distribution(
    hist: GapHistogram,
    *,
    ax: plt.Axes,
    label: str,
    bin_edges: np.ndarray,
    color: str | None = None,
) -> None:
    # Re-bin the pre-binned counts onto the shared edges (each input bin's
    # center lands in exactly one output bin because the edges are aligned).
    counts, _ = np.histogram(hist.centers, bins=bin_edges, weights=hist.counts)
    total = counts.sum()
    prob = counts / total if total else counts.astype(float)
    centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    # Only draw bins that actually have samples; empty bins would otherwise
    # break the log-scale y axis with zeros.
    mask = prob > 0
    ax.plot(
        centers[mask],
        prob[mask],
        marker="o",
        markersize=4,
        linestyle="none",
        label=label,
        color=color,
    )


def discover_csvs(csv_dir: Path, kind: str, *, required: bool = True) -> list[Path]:
    subdir = csv_dir / kind if kind != "all" else csv_dir
    if not subdir.is_dir():
        if required:
            raise FileNotFoundError(f"CSV directory not found: {subdir}")
        return []
    csv_paths = sorted(subdir.glob(f"*{DIST_SUFFIX}"))
    if not csv_paths:
        if required:
            raise FileNotFoundError(f"No gap distributions (*{DIST_SUFFIX}) found under {subdir}")
        return []
    return csv_paths


def csv_kind(csv_path: Path) -> str:
    if "time" in csv_path.parts:
        return "time"
    if "space" in csv_path.parts:
        return "space"
    return "all"


def group_csvs_by_kind(csv_paths: list[Path]) -> dict[str, list[Path]]:
    grouped: dict[str, list[Path]] = {}
    for csv_path in csv_paths:
        grouped.setdefault(csv_kind(csv_path), []).append(csv_path)
    return grouped


_DISTANCE_RE = re.compile(r"_d(\d+)(?:_|$)")


def series_sort_key(csv_path: Path) -> tuple[int, int, str]:
    """Sort series by code distance (numeric), so d=10 follows d=9.

    Names without a ``_d<N>`` token sort after the numbered ones, by name.
    """
    name = dataset_name(csv_path)
    match = _DISTANCE_RE.search(name)
    if match:
        return (0, int(match.group(1)), name)
    return (1, 0, name)


def output_path_for_kind(output: Path | None, kind: str) -> Path:
    if output is None:
        return DEFAULT_OUTPUT_DIR / f"gap_distribution_{kind}.pdf"
    if output.suffix:
        return output.with_name(f"{output.stem}_{kind}{output.suffix}")
    return output / f"gap_distribution_{kind}.pdf"


def plot_kind(
    *,
    kind: str,
    csv_paths: list[Path],
    bins: int | str,
    output: Path | None,
    show: bool,
) -> Path:
    csv_paths = sorted(csv_paths, key=series_sort_key)
    gap_series = [(csv_path, load_gaps(csv_path)) for csv_path in csv_paths]
    bin_edges = compute_bin_edges([hist for _, hist in gap_series], bins)

    fig, ax = plt.subplots(figsize=FIG_SIZE)
    colors = plt.cm.tab10.colors

    for i, (csv_path, hist) in enumerate(gap_series):
        name = dataset_name(csv_path)
        label = LEGEND_LABELS.get(name, name)
        plot_gap_distribution(
            hist,
            ax=ax,
            label=label,
            bin_edges=bin_edges,
            color=colors[i % len(colors)],
        )

    ax.set_xlabel(
        "Signed gap (log-likelihood ratio)",
        fontsize=FONT_SIZE_LABEL,
    )
    ax.set_ylabel("Probability", fontsize=FONT_SIZE_LABEL)
    ax.set_yscale("log")
    if XLIM is not None:
        ax.set_xlim(*XLIM)
    if YLIM is not None:
        ax.set_ylim(*YLIM)
    ax.tick_params(labelsize=FONT_SIZE_TICK)
    ax.legend(fontsize=FONT_SIZE_LEGEND)
    ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()

    output_path = output_path_for_kind(output, kind)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    print(f"Saved plot to {output_path}")
    if output_path.suffix != ".png":
        png_path = output_path.with_suffix(".png")
        fig.savefig(png_path, dpi=150)
        print(f"Saved plot to {png_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Plot the probability distribution of the signed gap (gap negated "
            "where the decoder failed) from the *_dist.csv histograms written "
            "by map2dist.py."
        )
    )
    parser.add_argument(
        "--csv",
        type=Path,
        action="append",
        default=None,
        help="Path to a *_dist.csv (gap_lo,gap_hi,count). Can be passed multiple times.",
    )
    parser.add_argument(
        "--csv-dir",
        type=Path,
        default=DEFAULT_CSV_DIR,
        help=(
            "Directory containing *_dist.csv files, or time/ and space/ "
            "subfolders of them."
        ),
    )
    parser.add_argument(
        "--kind",
        choices=["time", "space", "both", "all"],
        default="both",
        help=(
            "Which subfolder(s) to load when --csv is not set. 'both' uses "
            "time/ and space/ if present, else falls back to --csv-dir itself."
        ),
    )
    parser.add_argument(
        "--bins",
        default=str(DEFAULT_BINS),
        help=(
            f"Approximate number of histogram bins (input bins are merged in "
            f"whole multiples), or 'auto' to keep the CSV's own binning "
            f"(default: {DEFAULT_BINS})."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Path to save the plot. With --kind both, writes "
            "{stem}_time{suffix} and {stem}_space{suffix}."
        ),
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show the plot interactively.",
    )
    args = parser.parse_args()

    bins: int | str
    if args.bins != "auto":
        bins = int(args.bins)
    else:
        bins = "auto"

    csv_dir = args.csv_dir.resolve()
    if args.csv:
        csv_paths = [path.resolve() for path in args.csv]
        grouped = group_csvs_by_kind(csv_paths)
    elif args.kind == "both":
        grouped = {}
        for kind in ("time", "space"):
            csv_paths = discover_csvs(csv_dir, kind, required=False)
            if csv_paths:
                grouped[kind] = csv_paths
        if not grouped:
            grouped = {"all": discover_csvs(csv_dir, "all")}
    else:
        grouped = {args.kind: discover_csvs(csv_dir, args.kind)}

    for kind, kind_csv_paths in grouped.items():
        plot_kind(
            kind=kind,
            csv_paths=kind_csv_paths,
            bins=bins,
            output=args.output.resolve() if args.output is not None else None,
            show=args.show,
        )


if __name__ == "__main__":
    main()
