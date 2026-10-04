import argparse
import math
import os
import re
import string
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

# The CSV loader (src/gap_simulator/util/_sinter_csv.py) feeds numpy small, skinny matrix
# products.  With the default thread pool OpenBLAS is slower on them and burns
# ~60x the CPU on a machine that is already running simulations, so pin BLAS
# to one thread before numpy is imported (unless the caller chose otherwise).
for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.ticker
import numpy as np
import sinter
from sinter._plotting import plot_custom as sinter_plot_custom

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent / "src" / "gap_simulator"))

from util._sinter_csv import CACHE_DIR_NAME, fast_read_stats_from_csv_files

# --- Configuration (edit these) ---
# Results of src/gap_simulator/sim_two_yoke.py: one csv per code.
CSVS = sorted((SCRIPT_DIR / "simulated").glob("two_yoke_*.csv"))
PROTOCOL = None  # e.g. "two_yoke_q48_30_4"; None = plot all protocols in CSVS
# Which parameter runs along the x axis; every other parameter is either fixed
# by the filters below (or the matching CLI flags) or becomes a separate curve.
X_AXIS = "d"  # one of AXES: "d", "t" (t_interval), "r" (l1_check_rate), "p", "tr" (t/r)
T_INTERVAL = (
    None  # None = infer from CSV metadata (must be unique unless X_AXIS uses t)
)
CHECK_RATE = None  # l1_check_rate value(s) to keep; None = all
DISTANCE = None  # d value(s) to keep; None = all
D_MIN = None  # None = no lower bound on the plotted distance; --d-min
D_MAX = None  # None = no upper bound on the plotted distance; --d-max
PHYSICAL_P = None  # p value(s) to keep; None = all
T_OVER_R = None  # t_interval / l1_check_rate value(s) to keep; None = all
OUTPUT = None  # explicit path (or --output); None = OUTPUT_DIR / OUTPUT_NAME
OUTPUT_DIR = SCRIPT_DIR / "plots"
# Filename template used when OUTPUT is None.  Placeholders:
#   {protocols}  protocol names joined by "+"  (e.g. "two_yoke_q48_30_4+two_yoke_q128_104_4")
#   {protocol}   the first protocol name only
#   {x}          the x-axis short name ("d", "t", "r", "p", "tr")
#   {x_suffix}   "" when x is "d", otherwise "_x{x}"
#   {params}     every parameter fixed across the plot as "_t144d_r2..." ("" if none)
#   {d} {p} {t} {r} {tr}  the fixed value of that axis ("" when it varies),
#                written as in the legend (PARAM_VALUE_TEXT: t=12 -> "144d")
OUTPUT_NAME = "{protocols}{x_suffix}{params}.png"
# File formats to write, e.g. ["png", "pdf"]; --formats.  Each format is saved
# next to the others with its own extension (the extension in OUTPUT_NAME or
# --output is replaced).
OUTPUT_FORMATS = ["png"]
# Substrings removed from protocol names before they go into the filename.
OUTPUT_NAME_STRIP = []

PROTOCOL_TITLES = {
    "two_yoke_q48_30_4": "Two-yoke [[48, 30, 4]]",
    "two_yoke_q128_104_4": "Two-yoke [[128, 104, 4]]",
}

# --- Wording (edit these to change the sentences in the plot) ---
# Templates are lists of segments joined by PARAM_SEP.  A segment is dropped
# when any placeholder it uses is empty (no slope, no shared parameter...).
# Placeholders available in every template:
#   {params}      the parameters written with PARAM_TEXT and joined by PARAM_SEP
#   {d} {p} {t} {r} {tr}  the raw value of that axis ("" when it is not fixed)
#   {slope_name}  SLOPE_NAMES["slope"] or SLOPE_NAMES["exp"], see LOG_X_FIT_AXES
#   {slopes}      fitted slopes, one per level, joined by SLOPE_SEP (entries)
#   {levels}      level names in linestyle order, joined by SLOPE_SEP (title)
#   {x_name}      the x-axis name from AXIS_NAMES
LEVEL_TITLES = {  # legend name per level (CSV "level" metadata)
    "inner": "inner",
    "outer": "outer",
}
AXIS_NAMES = {  # how each axis / derived parameter is called in legend text
    "d": "d",
    "p": "p",
    "t": "t_2",
    "r": "r_1",
    "tr": "t/r",
    "t1": "t_1",
}
# Extra legend parameters computed from the fixed axis values of a curve.
# They behave like any other parameter: shown in {params} (named via
# AXIS_NAMES, in this order after the real axes), usable as {t1} etc., put in
# the entries when they differ between curves and in the legend title when
# shared.  Each function gets {axis: value} (missing = not fixed for that
# curve) and returns the value, or None when it cannot be computed.
DERIVED_PARAMS = {
    "t1": lambda v: (  # L1 check interval = t_2 // l1_check_rate
        v["t"] // v["r"] if v.get("t") is not None and v.get("r") else None
    ),
}
# How the value of a parameter is written (default: the plain number).  The
# t_interval values count in units of d rounds, so t_2 and t_1 are shown as
# T_SCALE * value followed by a literal "d" (e.g. t_2=12 -> "144d").
T_SCALE = 12
PARAM_VALUE_TEXT = {
    "t": lambda value: f"{value * T_SCALE:g}d",
    "t1": lambda value: f"{value * T_SCALE:g}d",
}
# Parameters never written in {params} (legend entries/title, figure title).
# Their {p}-style placeholders still work in explicit templates.
HIDDEN_PARAMS = {"p", "r"}
PARAM_TEXT = "{name}={value}"  # one parameter, e.g. "r=2"
PARAM_SEP = ", "  # between parameters and between template segments
LEGEND_ENTRY_TEXT = ["{params}", "{slope_name}={slopes}"]  # one colour/marker entry
LEGEND_TITLE_TEXT = ["{params}", "{slope_name}: {levels}"]  # legend title; [] = none
FIGURE_TITLE_TEXT: list[str] = []  # e.g. ["Two-yoke protocols ({params})"]; [] = none
FIT_ENTRY_TEXT = "fit (extrapolated)"  # dotted-line legend entry (only with fit lines)
SLOPE_NAMES = {"slope": "sl", "exp": "exp"}  # exponential-in-x / power-law fit
SLOPE_VALUE_TEXT = "{:.2f}"  # one fitted slope
SLOPE_MISSING_TEXT = "n/a"  # slope of a level with too few points
SLOPE_SEP = " / "  # between the per-level slopes / level names
Y_LABEL_TEXT = "Logical error rate"
Y_LABEL_NORMALIZED_TEXT = "Logical error rate per inner round"
# x-axis labels live in AXES below (second tuple element).
LEGEND_LOC = "best"  # matplotlib legend location, e.g. "lower left"
LEGEND_HANDLE_LENGTH = 1.2  # length of the legend line samples, in font-size units (matplotlib default 2.0)

# Plot axes: short name -> (json_metadata key, axis label).  The short names
# double as the filename tokens (``_d10_p0.001_t12_r2.csv``) and CLI flags.
AXES = {
    "d": ("d", "Inner distance (d)"),
    "p": ("p", "Physical error rate (p)"),
    "t": ("t_interval", "t_interval"),
    "r": ("l1_check_rate", "l1_check_rate"),
    "tr": ("t_over_r", "t_interval / l1_check_rate"),
}
# Primary axes: the parameters a curve can be grouped by.  Derived axes are
# computed from primary ones and, when used as the x axis, release every
# primary axis they depend on so those may vary along the curve.
AXIS_ORDER = ("d", "p", "t", "r")
DERIVED_AXES = {"tr": ("t", "r")}
# For a derived x axis, primary axes that still split curves even though they
# vary along x: with --x tr each t_interval gets its own curve (r varies along
# it), so (t=24, r=4) and (t=48, r=8) at t/r=6 are compared across curves
# instead of being joined into one zig-zag line.  Pin --t to get one curve.
DERIVED_CURVE_AXES = {"tr": ("t",)}
FILTER_AXES = AXIS_ORDER + tuple(DERIVED_AXES)
FLOAT_AXES = {"p", "tr"}
LOG_X_AXES = {"p", "tr"}  # drawn on a log x scale by default (--log-x forces it)
# Fit log10(rate) against log10(x) (power law, legend shows "exp=") for these
# axes; against x itself (exponential in d, legend shows "slope=") otherwise.
LOG_X_FIT_AXES = {"p", "t", "r", "tr"}

# Marker per curve pair (protocol + fixed parameters); linestyle per level --
# the same scheme as plot_one_yoke_all.py: inner and outer curves of one
# protocol share colour and marker and are told apart by linestyle only.
PAIR_MARKERS = "osD^v<>Pp*Xh"
LEVEL_LINESTYLES = {
    "inner": "--",
    "outer": "-",
}
FIT_LINESTYLE = ":"  # extrapolated fit lines


FIG_SIZE = (9, 8)
FONT_SIZE_TICK = 20
FONT_SIZE_LABEL = 24
FONT_SIZE_LEGEND = 17
FONT_SIZE_TITLE = 16

DEFAULT_HIGHLIGHT_FACTOR = 100
DEFAULT_Y_PAD_FRACTION = 0.1
EXTRAPOLATED_D: list[int] = (
    []
)  # fit lines off by default; e.g. [9, 11] or --extrapolate 9 11 to draw them
EXTRAPOLATED_X = None  # for --x t/r/p: x values to extend fit lines to; None = none

_PROTOCOL_FILENAME_RE = re.compile(r"^(.+?)_d\d+_p")
_FILENAME_TOKEN_RE = re.compile(r"^([a-z])([0-9][0-9.eE+-]*)$")
USE_CACHE = True  # per-file aggregation cache, see src/gap_simulator/util/_sinter_csv.py; --no-cache

# Filters: axis -> set of allowed values (None entries mean "keep everything").
Filters = dict[str, set]
# One plotted curve: (protocol variant, ((axis, value), ...) of the fixed axes, level)
GroupKey = tuple[str, tuple[tuple[str, object], ...], str]


def _resolve_path(path: Path) -> Path:
    path = Path(path)
    if path.exists():
        return path.resolve()
    script_relative = (SCRIPT_DIR / path).resolve()
    if script_relative.exists():
        return script_relative
    return path.resolve()


def discover_csvs(path: Path) -> list[Path]:
    """CSV files under ``path``; a missing path or an empty directory is skipped."""
    path = _resolve_path(path)
    if path.is_file():
        return [path]
    if path.is_dir():
        found = sorted(path.glob("*.csv"))
        if not found:
            print(f"skipping {path}: no CSV files", flush=True)
        return found
    print(f"skipping {path}: not found", flush=True)
    return []


def _t_interval(stat: sinter.TaskStats) -> int:
    return stat.json_metadata.get("t_interval", 1)


def _derive(axis: str, values: dict[str, object]):
    """Value of a derived axis from primary values; None when one is missing."""
    if axis == "tr":
        t_val, r_val = values.get("t"), values.get("r")
        if t_val is None or r_val in (None, 0):
            return None
        return t_val / r_val
    raise KeyError(axis)


def axis_value(stat: sinter.TaskStats, axis: str):
    """Value of ``axis`` for one stat; None when the metadata lacks it."""
    if axis in DERIVED_AXES:
        return _derive(axis, {dep: axis_value(stat, dep) for dep in DERIVED_AXES[axis]})
    if axis == "t":
        return _t_interval(stat)
    return stat.json_metadata.get(AXES[axis][0])


def _axis_components(axis: str) -> tuple[str, ...]:
    """Primary axes that vary when ``axis`` runs along x."""
    return DERIVED_AXES.get(axis, (axis,))


def _parse_axis_value(axis: str, text: str):
    return float(text) if axis in FLOAT_AXES else int(text)


def _value_in(value, allowed: set) -> bool:
    """Membership test that compares floats with a tolerance (e.g. 24/4 vs 6)."""
    if value is None:
        return False
    if isinstance(value, float):
        return any(
            isinstance(item, (int, float)) and math.isclose(value, item)
            for item in allowed
        )
    return value in allowed


def protocol_from_filename(path: Path) -> str | None:
    match = _PROTOCOL_FILENAME_RE.match(path.name)
    if match is None:
        return None
    return match.group(1)


def params_from_filename(path: Path) -> dict[str, object]:
    """Axis values encoded in the filename, e.g. ``..._d10_p0.001_t12_r2.csv``."""
    match = _PROTOCOL_FILENAME_RE.match(path.name)
    if match is None:
        return {}
    rest = path.name[match.end(1) :]
    if rest.endswith(".csv"):
        rest = rest[: -len(".csv")]
    params: dict[str, object] = {}
    for token in rest.split("_"):
        token_match = _FILENAME_TOKEN_RE.match(token)
        if token_match is None:
            continue
        axis, text = token_match.groups()
        if axis in AXES and axis not in params:
            try:
                params[axis] = _parse_axis_value(axis, text)
            except ValueError:
                pass
    for axis in DERIVED_AXES:
        derived = _derive(axis, params)
        if derived is not None:
            params[axis] = derived
    return params


def t_interval_from_filename(path: Path) -> int | None:
    return params_from_filename(path).get("t")


def _available_from_filenames(csv_paths: list[Path], axis: str) -> set:
    return {
        value
        for path in csv_paths
        if (value := params_from_filename(path).get(axis)) is not None
    }


def infer_t_interval_from_paths(csv_paths: list[Path]) -> int | None:
    """The unique t_interval named in the CSV filenames, or None when the
    filenames carry no parameters (one csv per code holds every parameter set:
    all t_interval values are then kept, one curve each)."""
    intervals = _available_from_filenames(csv_paths, "t")
    if len(intervals) == 1:
        return intervals.pop()
    if len(intervals) > 1:
        raise ValueError(
            f"Multiple t_interval values found in filenames: {sorted(intervals)}. "
            "Specify --t-interval (one or more values), or use --x t to put "
            "t_interval on the x axis."
        )
    return None


def filter_csv_paths(
    csv_paths: list[Path],
    *,
    protocol: str | None = None,
    filters: Filters | None = None,
) -> list[Path]:
    """Drop files whose name already says they cannot match the filters."""
    filters = filters or {}
    filtered: list[Path] = []
    for path in csv_paths:
        if protocol is not None:
            file_protocol = protocol_from_filename(path)
            if file_protocol is not None and file_protocol != protocol:
                continue
        params = params_from_filename(path)
        if any(
            axis in params and not _value_in(params[axis], allowed)
            for axis, allowed in filters.items()
            if allowed is not None
        ):
            continue
        filtered.append(path)
    return filtered


def _matches_filters(stat: sinter.TaskStats, filters: Filters) -> bool:
    return all(
        allowed is None or _value_in(axis_value(stat, axis), allowed)
        for axis, allowed in filters.items()
    )


def resolve_filters(
    csv_paths: list[Path],
    *,
    x_axis: str,
    t_interval: list[int] | None = None,
    check_rate: list[int] | None = None,
    distance: list[int] | None = None,
    physical_p: list[float] | None = None,
    t_over_r: list[float] | None = None,
) -> Filters:
    """Turn the CLI/config selections into per-axis value sets.

    ``t_interval`` keeps its historical behaviour: when it is not given and
    t_interval does not vary along the x axis (``--x t`` / ``--x tr``), the
    (unique) value is inferred from the filenames when they name one; a single
    csv per code names none, and every t_interval is kept.  Every other axis defaults
    to "all values" (separate curves).
    """
    requested = {
        "t": t_interval,
        "r": check_rate,
        "d": distance,
        "p": physical_p,
        "tr": t_over_r,
    }
    varying = _axis_components(x_axis)
    filters: Filters = {}
    for axis in FILTER_AXES:
        values = requested[axis]
        if values is not None and axis == x_axis:
            print(
                f"note: --{axis} filter is applied although {axis} is the x axis",
                flush=True,
            )
        if values is None:
            if axis == "t" and "t" not in varying:
                inferred = infer_t_interval_from_paths(csv_paths)
                filters[axis] = None if inferred is None else {inferred}
            else:
                filters[axis] = None
            continue
        values_set = set(values)
        available = _available_from_filenames(csv_paths, axis)
        if available and not any(_value_in(v, values_set) for v in available):
            raise ValueError(
                f"No CSV files with {AXES[axis][0]} in {sorted(values_set)} found. "
                f"Available values: {sorted(available)}."
            )
        filters[axis] = values_set
    return filters


def load_stats(
    csv_paths: list[Path],
    *,
    protocol: str | None = None,
    filters: Filters,
) -> tuple[list[sinter.TaskStats], list[Path]]:
    csv_paths = filter_csv_paths(csv_paths, protocol=protocol, filters=filters)
    if not csv_paths:
        raise ValueError("No CSV files matched the requested protocol/axis filters.")

    stats = fast_read_stats_from_csv_files(*csv_paths, use_cache=USE_CACHE)
    if protocol is not None:
        stats = [
            stat for stat in stats if stat.json_metadata.get("protocol") == protocol
        ]
        if not stats:
            raise ValueError(
                f"No {protocol!r} sinter stats found in the provided CSV file(s)."
            )
    stats = [stat for stat in stats if _matches_filters(stat, filters)]
    if not stats:
        active = {
            AXES[axis][0]: sorted(allowed)
            for axis, allowed in filters.items()
            if allowed is not None
        }
        raise ValueError(
            f"No stats matching {active} found in the provided CSV file(s)."
        )
    return stats, csv_paths


def filter_stats_by_distance(
    stats: list[sinter.TaskStats],
    d_min: int | None,
    d_max: int | None,
) -> list[sinter.TaskStats]:
    """Keep only stats with d_min <= d <= d_max (None = unbounded)."""
    if d_min is None and d_max is None:
        return stats
    filtered = [
        stat
        for stat in stats
        if (d_min is None or stat.json_metadata["d"] >= d_min)
        and (d_max is None or stat.json_metadata["d"] <= d_max)
    ]
    if not filtered:
        raise ValueError(f"No stats with distance in [{d_min}, {d_max}] found.")
    return filtered


def _variant(stat: sinter.TaskStats) -> str:
    """Protocol name plus any check rate that is not a plot axis.

    l1_check_rate is an axis (see AXES) and is handled by the group key;
    rows that differ only in l2_check_rate are still different protocols
    (different strong_ids), so they are kept apart here.
    """
    meta = stat.json_metadata
    parts = [meta["protocol"]]
    if "l2_check_rate" in meta:
        parts.append(f"r2{meta['l2_check_rate']}")
    return "|".join(parts)


def protocol_title(protocol: str) -> str:
    base, *rates = protocol.split("|")
    title = PROTOCOL_TITLES.get(base, base.replace("_", " ").title())
    if rates:
        title = f"{title} ({', '.join(rates)})"
    return title


def level_title(level: str) -> str:
    return LEVEL_TITLES.get(level, level)


def axis_name(axis: str) -> str:
    if axis in AXIS_NAMES:
        return AXIS_NAMES[axis]
    return AXES[axis][0] if axis in AXES else axis


def _fill_text(segments: list[str], **values) -> str:
    """Format the wording templates: join segments, dropping the empty ones."""
    values = {key: "" if value is None else value for key, value in values.items()}
    filled: list[str] = []
    for segment in segments:
        fields = [
            field for _, field, _, _ in string.Formatter().parse(segment) if field
        ]
        if any(values.get(field, "") == "" for field in fields):
            continue
        filled.append(segment.format(**values))
    return PARAM_SEP.join(part for part in filled if part)


def _with_derived_params(axis_items) -> list[tuple[str, object]]:
    """Append the DERIVED_PARAMS that can be computed from ``axis_items``."""
    known = {axis: value for axis, value in axis_items if value is not None}
    items = list(axis_items)
    for name, func in DERIVED_PARAMS.items():
        value = func(known)
        if value is not None:
            items.append((name, value))
    return items


def _format_param_value(axis: str, value) -> str:
    """Value of one parameter as written in the plot (see PARAM_VALUE_TEXT)."""
    if axis in PARAM_VALUE_TEXT:
        return PARAM_VALUE_TEXT[axis](value)
    return _format_value(value)


def _text_values(axis_items, *, x_axis: str, **extra) -> dict:
    """Placeholder values shared by every template (see the wording block)."""
    values = {axis: "" for axis in (*AXES, *DERIVED_PARAMS)}
    values.update(
        {
            axis: _format_param_value(axis, value)
            for axis, value in axis_items
            if value is not None
        }
    )
    values["params"] = PARAM_SEP.join(
        _format_axis(axis, value)
        for axis, value in axis_items
        if value is not None and axis not in HIDDEN_PARAMS
    )
    values["x_name"] = axis_name(x_axis)
    values.update(extra)
    return values


def normalization_denominator(meta: dict) -> int:
    """Per-level normalization from CSV metadata (prefers num_ticks)."""
    if "num_ticks" in meta:
        return int(meta["num_ticks"])
    d_val = meta["d"]
    t_interval_val = meta["t_interval"]
    outer_rounds = meta["outer_rounds"]
    base = (1 + t_interval_val * (outer_rounds - 1)) * d_val
    if meta["level"] == "outer":
        return base * 4
    return base


def _normalization_denominator(stat: sinter.TaskStats) -> int:
    return normalization_denominator(stat.json_metadata)


def _axis_items(stat: sinter.TaskStats, axes) -> tuple[tuple[str, object], ...]:
    return tuple((axis, axis_value(stat, axis)) for axis in axes)


def _point_key(stat: sinter.TaskStats) -> tuple:
    """Identity of one plotted data point (variant, level, every axis value)."""
    return (_variant(stat), stat.json_metadata["level"], _axis_items(stat, AXIS_ORDER))


def _sortable(value):
    """Sort key that tolerates None (missing metadata) mixed with numbers."""
    if isinstance(value, tuple):
        return tuple(_sortable(item) for item in value)
    if value is None:
        return (0, 0.0)
    if isinstance(value, (int, float)):
        return (1, float(value))
    return (2, str(value))


def _format_value(value) -> str:
    return f"{value:g}" if isinstance(value, float) else str(value)


def _format_axis(axis: str, value) -> str:
    return PARAM_TEXT.format(
        name=axis_name(axis), value=_format_param_value(axis, value)
    )


def aggregate_stats_renormalized_by_num_ticks(
    stats: list[sinter.TaskStats],
) -> list[sinter.TaskStats]:
    """Merge stats that share the same plot point, weighting by each num_ticks.

    Combined per-tick rate is ``sum(errors) / sum((shots - discards) * num_ticks)``.
    When contributors have different ``num_ticks``, the merge is stored as an
    equivalent TaskStats with ``num_ticks=1`` and ``shots`` equal to the total
    tick-exposure so later ``/ num_ticks`` normalization stays correct.
    """
    grouped: dict[tuple, list[sinter.TaskStats]] = defaultdict(list)
    for stat in stats:
        grouped[_point_key(stat)].append(stat)

    merged: list[sinter.TaskStats] = []
    for key, group in grouped.items():
        if len(group) == 1:
            merged.append(group[0])
            continue

        ticks = [_normalization_denominator(stat) for stat in group]
        if len(set(ticks)) == 1:
            total = group[0]
            for stat in group[1:]:
                total = total + stat
            merged.append(total)
            continue

        total_errors = sum(stat.errors for stat in group)
        total_seconds = sum(stat.seconds for stat in group)
        exposure = sum(
            (stat.shots - stat.discards) * tick for stat, tick in zip(group, ticks)
        )
        protocol, level, axis_items = key
        meta = dict(group[0].json_metadata)
        meta["num_ticks"] = 1
        meta["aggregated_num_ticks"] = sorted(set(ticks))
        axis_tag = ":".join(f"{axis}{value}" for axis, value in axis_items)
        merged.append(
            sinter.TaskStats(
                shots=exposure,
                errors=total_errors,
                discards=0,
                seconds=total_seconds,
                strong_id=f"renorm:{protocol}:{axis_tag}:{level}",
                decoder=group[0].decoder,
                json_metadata=meta,
            )
        )
    return merged


def normalized_error_rate(stat: sinter.TaskStats) -> float:
    shots = stat.shots - stat.discards
    if shots == 0:
        return 0.0
    return stat.errors / shots / _normalization_denominator(stat)


def normalization_factor(stat: sinter.TaskStats) -> float:
    return _normalization_denominator(stat)


def fit_error_rate(
    stat: sinter.TaskStats,
    *,
    highlight_max_likelihood_factor: float,
) -> sinter.Fit:
    return sinter.fit_binomial(
        num_shots=stat.shots - stat.discards,
        num_hits=stat.errors,
        max_likelihood_factor=highlight_max_likelihood_factor,
    )


def fit_plot_rate(
    stat: sinter.TaskStats,
    *,
    highlight_max_likelihood_factor: float,
    normalize: bool,
    fit_cache: dict[str, sinter.Fit] | None = None,
) -> sinter.Fit:
    cache_key = None
    if fit_cache is not None:
        cache_key = (
            f"{stat.strong_id}:{highlight_max_likelihood_factor}:" f"{int(normalize)}"
        )
        cached = fit_cache.get(cache_key)
        if cached is not None:
            return cached

    fit = fit_error_rate(
        stat, highlight_max_likelihood_factor=highlight_max_likelihood_factor
    )
    if normalize:
        factor = normalization_factor(stat)
        fit = sinter.Fit(
            low=fit.low / factor,
            best=fit.best / factor,
            high=fit.high / factor,
        )
    if fit_cache is not None and cache_key is not None:
        fit_cache[cache_key] = fit
    return fit


@dataclass
class _PlotPrep:
    x_axis: str
    grouped: dict[GroupKey, list[sinter.TaskStats]]
    group_labels: dict[GroupKey, str]
    group_xs_rates: dict[GroupKey, tuple[list[float], list[float]]]
    highlight_max_likelihood_factor: float
    normalize: bool
    fit_cache: dict[str, sinter.Fit] = field(default_factory=dict)

    @property
    def log_x_fit(self) -> bool:
        """Power-law fit (log-log) instead of exponential-in-x (log-linear)."""
        return self.x_axis in LOG_X_FIT_AXES

    def slope_fit(self, xs, rates):
        return fit_log10_slope(xs, rates, log_x=self.log_x_fit)

    def predict(self, x, slope, intercept):
        return predict_rate(x, slope, intercept, log_x=self.log_x_fit)

    @property
    def slope_name(self) -> str:
        return SLOPE_NAMES["exp" if self.log_x_fit else "slope"]


def _fixed_axes(x_axis: str) -> tuple[str, ...]:
    """Primary axes that identify a curve (everything not running along x)."""
    varying = set(_axis_components(x_axis)) - set(DERIVED_CURVE_AXES.get(x_axis, ()))
    return tuple(axis for axis in AXIS_ORDER if axis not in varying)


def _group_key(stat: sinter.TaskStats, x_axis: str) -> GroupKey:
    return (
        _variant(stat),
        _axis_items(stat, _fixed_axes(x_axis)),
        stat.json_metadata["level"],
    )


def _group_stats(
    stats: list[sinter.TaskStats], x_axis: str
) -> dict[GroupKey, list[sinter.TaskStats]]:
    grouped: dict[GroupKey, list[sinter.TaskStats]] = defaultdict(list)
    for stat in stats:
        grouped[_group_key(stat, x_axis)].append(stat)
    return grouped


def _sorted_items(mapping: dict):
    return sorted(mapping.items(), key=lambda item: _sortable(item[0]))


def _group_label(key: GroupKey, slope: float | None, slope_name: str) -> str:
    protocol, axis_items, level = key
    parts = [protocol_title(protocol), level]
    parts.extend(_format_axis(axis, value) for axis, value in axis_items)
    if slope is not None:
        parts.append(f"{slope_name}={slope:.2f}")
    return ", ".join(parts)


def _prepare_plot(
    stats: list[sinter.TaskStats],
    *,
    x_axis: str,
    highlight_max_likelihood_factor: float,
    normalize: bool,
) -> _PlotPrep:
    grouped = _group_stats(stats, x_axis)
    fit_cache: dict[str, sinter.Fit] = {}
    group_xs_rates: dict[GroupKey, tuple[list[float], list[float]]] = {}
    group_labels: dict[GroupKey, str] = {}
    prep = _PlotPrep(
        x_axis=x_axis,
        grouped=grouped,
        group_labels=group_labels,
        group_xs_rates=group_xs_rates,
        highlight_max_likelihood_factor=highlight_max_likelihood_factor,
        normalize=normalize,
        fit_cache=fit_cache,
    )

    for key, group in _sorted_items(grouped):
        xs, rates = _plot_rates_for_group(
            group,
            x_axis=x_axis,
            highlight_max_likelihood_factor=highlight_max_likelihood_factor,
            normalize=normalize,
            fit_cache=fit_cache,
        )
        group_xs_rates[key] = (xs, rates)
        slope_fit = prep.slope_fit(xs, rates)
        slope = None if slope_fit is None else slope_fit[0]
        group_labels[key] = _group_label(key, slope, prep.slope_name)

    return prep


def print_fit_summary(
    stats: list[sinter.TaskStats],
    *,
    highlight_max_likelihood_factor: float,
    normalize: bool,
    fit_cache: dict[str, sinter.Fit] | None = None,
) -> None:
    for stat in sorted(stats, key=lambda s: _sortable(_point_key(s))):
        fit = fit_plot_rate(
            stat,
            highlight_max_likelihood_factor=highlight_max_likelihood_factor,
            normalize=normalize,
            fit_cache=fit_cache,
        )
        meta = stat.json_metadata
        raw_rate = stat.errors / stat.shots if stat.shots else 0.0
        print(
            f"protocol={_variant(stat)} level={meta['level']} d={meta['d']} "
            f"p={meta['p']} t_interval={_t_interval(stat)} "
            f"l1_check_rate={axis_value(stat, 'r')} "
            f"t_over_r={_format_value(axis_value(stat, 'tr'))} "
            f"num_ticks={_normalization_denominator(stat)}: "
            f"{stat.shots} shots, {stat.errors} errors, "
            f"rate={raw_rate:.6g}, "
            f"plot_rate={fit.best:.6g}, "
            f"normalized_rate={normalized_error_rate(stat):.6g}, "
            f"CI=[{fit.low:.6g}, {fit.high:.6g}]"
        )


def _collect_plot_y_values(
    stats: list[sinter.TaskStats],
    prep: _PlotPrep,
    *,
    extrapolated_x: list[float],
) -> list[float]:
    values: list[float] = []
    for stat in stats:
        fit = fit_plot_rate(
            stat,
            highlight_max_likelihood_factor=prep.highlight_max_likelihood_factor,
            normalize=prep.normalize,
            fit_cache=prep.fit_cache,
        )
        for value in (fit.low, fit.best, fit.high):
            if value > 0 and not np.isnan(value):
                values.append(float(value))

    for xs, rates in prep.group_xs_rates.values():
        for rate in rates:
            if rate > 0 and not np.isnan(rate):
                values.append(float(rate))
        if not extrapolated_x:
            continue
        slope_fit = prep.slope_fit(xs, rates)
        if slope_fit is None:
            continue
        slope, intercept = slope_fit
        x_line = _fit_line_x(xs, extrapolated_x, prep.log_x_fit, 100)
        rate_line = prep.predict(x_line, slope, intercept)
        values.extend(
            float(value) for value in rate_line if value > 0 and not np.isnan(value)
        )
    return values


def _fit_line_x(
    xs: list[float], extrapolated_x: list[float], log_x: bool, n: int
) -> np.ndarray:
    x_min = min(xs)
    x_max = max([*xs, *extrapolated_x])
    if log_x:
        return np.geomspace(x_min, x_max, n)
    return np.linspace(x_min, x_max, n)


def set_log_y_limits(
    ax: plt.Axes,
    y_values: list[float],
    *,
    pad_fraction: float,
) -> None:
    if not y_values:
        return

    y_min = min(y_values)
    y_max = max(y_values)
    log_min = np.log10(y_min)
    log_max = np.log10(y_max)
    if log_max <= log_min:
        log_max = log_min + 1.0

    pad = pad_fraction * (log_max - log_min)
    ax.set_ylim(10 ** (log_min - pad), 10 ** (log_max + pad))


def set_log_y_ticks(ax: plt.Axes) -> None:
    """Show every power-of-10 tick (10^-1 spacing in exponent) within y limits."""
    ymin, ymax = ax.get_ylim()
    if ymin <= 0 or ymax <= 0:
        return
    log_min = int(np.floor(np.log10(ymin)))
    log_max = int(np.ceil(np.log10(ymax)))
    ax.set_yticks([10.0**exp for exp in range(log_min, log_max + 1)])
    # Minor ticks at 2..9 x 10^k in every decade.  Matplotlib's default
    # LogLocator drops them once the axis spans more than ~8 decades, which
    # made the fine grid lines appear on some plots and not others.
    ax.yaxis.set_minor_locator(
        matplotlib.ticker.LogLocator(base=10.0, subs=np.arange(2, 10), numticks=100)
    )
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())


def fit_log10_slope(
    xs: list[float], rates: list[float], *, log_x: bool = False
) -> tuple[float, float] | None:
    """Fit log10(rate) = slope * x + intercept (straight on log-y vs linear-x).

    With ``log_x`` the fit is against log10(x) instead, i.e. a power law
    rate ~ x**slope (straight on a log-log plot).
    """
    positive = [
        (x, rate)
        for x, rate in zip(xs, rates)
        if x is not None and x > 0 and rate > 0 and not np.isnan(rate)
    ]
    if len(positive) < 2 or len({x for x, _ in positive}) < 2:
        return None
    x_arr = np.array([x for x, _ in positive], dtype=float)
    if log_x:
        x_arr = np.log10(x_arr)
    log_r = np.log10([rate for _, rate in positive])
    slope, intercept = np.polyfit(x_arr, log_r, 1)
    return float(slope), float(intercept)


def predict_rate(
    x: float | np.ndarray, slope: float, intercept: float, *, log_x: bool = False
) -> np.ndarray:
    x_arr = np.asarray(x, dtype=float)
    if log_x:
        x_arr = np.log10(x_arr)
    return 10 ** (slope * x_arr + intercept)


def _plot_rates_for_group(
    group: list[sinter.TaskStats],
    *,
    x_axis: str,
    highlight_max_likelihood_factor: float,
    normalize: bool,
    fit_cache: dict[str, sinter.Fit] | None = None,
) -> tuple[list[float], list[float]]:
    # For a derived axis several (t, r) pairs can share one x (24/4 == 48/8);
    # sorting by the components too keeps the drawn line deterministic.
    group = sorted(
        group,
        key=lambda stat: _sortable(
            (axis_value(stat, x_axis),)
            + tuple(axis_value(stat, axis) for axis in _axis_components(x_axis))
        ),
    )
    xs = [axis_value(stat, x_axis) for stat in group]
    rates = [
        fit_plot_rate(
            stat,
            highlight_max_likelihood_factor=highlight_max_likelihood_factor,
            normalize=normalize,
            fit_cache=fit_cache,
        ).best
        for stat in group
    ]
    return xs, rates


def _plot_extrapolated_fit_lines(
    ax: plt.Axes,
    prep: _PlotPrep,
    *,
    colors: list,
    pair_index: dict[tuple, int],
    extrapolated_x: list[float],
) -> None:
    """Extend fit lines to x values without measured results."""
    if not extrapolated_x:
        return
    for key, (xs, rates) in _sorted_items(prep.group_xs_rates):
        slope_fit = prep.slope_fit(xs, rates)
        if slope_fit is None:
            continue
        slope, intercept = slope_fit
        x_line = _fit_line_x(xs, extrapolated_x, prep.log_x_fit, 200)
        rate_line = prep.predict(x_line, slope, intercept)
        index = pair_index[key[:2]]
        ax.plot(
            x_line,
            rate_line,
            linestyle=FIT_LINESTYLE,
            color=colors[index % len(colors)],
            linewidth=1,
            zorder=1,
        )


def fit_equation_text(
    slope: float, intercept: float, *, x_name: str, log_x: bool
) -> str:
    """The fitted line as an equation, in log10 form and in rate form.

    Linear x:  log10(rate) = slope * x + intercept   <=>  rate = A * 10**(slope * x)
    Log x:     log10(rate) = slope * log10(x) + intercept  <=>  rate = A * x**slope
    with A = 10**intercept.
    """
    prefactor = 10.0**intercept
    sign = "+" if intercept >= 0 else "-"
    if log_x:
        return (
            f"log10(rate) = {slope:.4f} * log10({x_name}) {sign} {abs(intercept):.4f}"
            f"  <=>  rate = {prefactor:.4g} * {x_name}^({slope:.4f})"
        )
    return (
        f"log10(rate) = {slope:.4f} * {x_name} {sign} {abs(intercept):.4f}"
        f"  <=>  rate = {prefactor:.4g} * 10^({slope:.4f} * {x_name})"
    )


def print_slope_summary(
    prep: _PlotPrep,
) -> None:
    x_name = AXES[prep.x_axis][0]
    if prep.log_x_fit:
        print(f"fit: log10(rate) vs log10({x_name}) (power law, exponent)")
    else:
        print(f"fit: log10(rate) vs {x_name} (slope per unit {x_name})")
    for key, (xs, rates) in _sorted_items(prep.group_xs_rates):
        protocol, axis_items, level = key
        fixed = " ".join(
            f"{AXES[axis][0]}={_format_value(value)}" for axis, value in axis_items
        )
        slope_fit = prep.slope_fit(xs, rates)
        if slope_fit is None:
            print(
                f"protocol={protocol} level={level} {fixed}: "
                f"{prep.slope_name}=n/a (need >=2 {x_name} values)"
            )
        else:
            slope, intercept = slope_fit
            equation = fit_equation_text(
                slope, intercept, x_name=x_name, log_x=prep.log_x_fit
            )
            print(
                f"protocol={protocol} level={level} {fixed}: "
                f"{prep.slope_name}={slope:.3f}  {equation}"
            )


def _single_valued_axes(
    stats: list[sinter.TaskStats], x_axis: str
) -> list[tuple[str, object]]:
    """Fixed axes that take exactly one value over ``stats`` (for title/filename)."""
    single: list[tuple[str, object]] = []
    for axis in _fixed_axes(x_axis):
        values = {axis_value(stat, axis) for stat in stats}
        values.discard(None)
        if len(values) == 1:
            single.append((axis, values.pop()))
    return single


def plot_title_prefix(stats: list[sinter.TaskStats], x_axis: str) -> str:
    title_prefix = "Two-yoke protocols"
    fixed = [
        f"{AXES[axis][0]}={_format_value(value)}"
        for axis, value in _single_valued_axes(stats, x_axis)
        if not (axis == "t" and value == 1)
    ]
    if fixed:
        title_prefix = f"{title_prefix} ({', '.join(fixed)})"
    return title_prefix


def default_output_path(stats: list[sinter.TaskStats], x_axis: str) -> Path:
    """OUTPUT_DIR / OUTPUT_NAME with the template placeholders filled in."""
    protocols = sorted({stat.json_metadata.get("protocol") for stat in stats})
    for token in OUTPUT_NAME_STRIP:
        protocols = [name.replace(token, "") for name in protocols]
    fixed = _single_valued_axes(stats, x_axis)
    values = {axis: "" for axis in AXES}
    # Values are written as in the legend (PARAM_VALUE_TEXT): t12 -> "144d".
    values.update({axis: _format_param_value(axis, value) for axis, value in fixed})
    values.update(
        protocols="+".join(protocols),
        protocol=protocols[0] if protocols else "",
        x=x_axis,
        x_suffix="" if x_axis == "d" else f"_x{x_axis}",
        params="".join(f"_{axis}{values[axis]}" for axis, _ in fixed),
    )
    return OUTPUT_DIR / OUTPUT_NAME.format(**values)


def _build_compact_legend(
    prep: _PlotPrep,
    *,
    pair_index: dict[tuple, int],
    colors: list,
    show_fit_entry: bool = False,
) -> tuple[list[Line2D], str | None]:
    """Legend that explains the encoding instead of listing every curve.

    Linestyle entries (black) name the level; one colour+marker entry per
    (protocol, fixed parameters) pair names the pair by whatever differs
    between pairs and carries its fitted slopes, one per level in the order
    of the linestyle entries.  Parameters shared by every pair move into the
    legend title.
    """
    slopes: dict[GroupKey, float | None] = {}
    for key, (xs, rates) in prep.group_xs_rates.items():
        fit = prep.slope_fit(xs, rates)
        slopes[key] = None if fit is None else fit[0]
    levels = sorted(
        {key[2] for key in slopes},
        key=lambda level: (
            list(LEVEL_LINESTYLES).index(level)
            if level in LEVEL_LINESTYLES
            else len(LEVEL_LINESTYLES)
        ),
    )
    pair_keys = sorted(pair_index, key=pair_index.get)
    pair_items = {key: _with_derived_params(key[1]) for key in pair_keys}
    axes: list[str] = []
    for items in pair_items.values():
        axes.extend(axis for axis, _ in items if axis not in axes)
    axis_values = {
        axis: {dict(items).get(axis) for items in pair_items.values()} for axis in axes
    }
    varying_axes = [axis for axis in axes if len(axis_values[axis]) > 1]

    handles: list[Line2D] = []
    for level in levels:
        handles.append(
            Line2D(
                [],
                [],
                color="black",
                linestyle=LEVEL_LINESTYLES.get(level, "-"),
                linewidth=2,
                label=level_title(level),
            )
        )
    if show_fit_entry:
        handles.append(
            Line2D(
                [],
                [],
                color="black",
                linestyle=FIT_LINESTYLE,
                linewidth=1,
                label=FIT_ENTRY_TEXT,
            )
        )

    # Protocol names are deliberately left out of the legend.  {params} only
    # lists the parameters that differ between pairs; the shared ones go into
    # the legend title (so an entry may carry nothing but its slopes).
    for key in pair_keys:
        axis_items = pair_items[key]
        entry_items = [
            (axis, value) for axis, value in axis_items if axis in varying_axes
        ]
        pair_slopes = [slopes.get((*key, level)) for level in levels]
        slopes_text = ""
        if any(slope is not None for slope in pair_slopes):
            slopes_text = SLOPE_SEP.join(
                SLOPE_MISSING_TEXT if slope is None else SLOPE_VALUE_TEXT.format(slope)
                for slope in pair_slopes
            )
        label = _fill_text(
            LEGEND_ENTRY_TEXT,
            **_text_values(
                entry_items,
                x_axis=prep.x_axis,
                slope_name=prep.slope_name,
                slopes=slopes_text,
                levels="",
            ),
        )
        index = pair_index[key]
        handles.append(
            Line2D(
                [],
                [],
                color=colors[index % len(colors)],
                marker=PAIR_MARKERS[index % len(PAIR_MARKERS)],
                linestyle="none",
                markersize=8,
                label=label,
            )
        )

    shared_items = [
        (axis, next(iter(axis_values[axis])))
        for axis in axes
        if axis not in varying_axes
    ]
    levels_text = ""
    if len(levels) > 1 and any(slope is not None for slope in slopes.values()):
        levels_text = SLOPE_SEP.join(level_title(level) for level in levels)
    title = _fill_text(
        LEGEND_TITLE_TEXT,
        **_text_values(
            shared_items,
            x_axis=prep.x_axis,
            slope_name=prep.slope_name,
            slopes="",
            levels=levels_text,
        ),
    )
    return handles, title or None


def plot_two_yoke_both_error_rate(
    stats: list[sinter.TaskStats],
    *,
    x_axis: str,
    output: Path | None,
    output_formats: list[str] | None = None,
    show: bool,
    highlight_max_likelihood_factor: float,
    y_pad_fraction: float,
    normalize: bool,
    extrapolated_x: list[float],
    log_x: bool = False,
    prep: _PlotPrep | None = None,
) -> None:
    if prep is None:
        prep = _prepare_plot(
            stats,
            x_axis=x_axis,
            highlight_max_likelihood_factor=highlight_max_likelihood_factor,
            normalize=normalize,
        )
    fig, ax = plt.subplots(figsize=FIG_SIZE, constrained_layout=True)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    # Inner and outer curves of the same (protocol, fixed parameters) share a
    # colour and marker; the level is told apart by linestyle only.
    pair_keys = sorted({_group_key(stat, x_axis)[:2] for stat in stats}, key=_sortable)
    pair_index = {key: index for index, key in enumerate(pair_keys)}

    def _plot_args(_index, _group_key_value, group_stats) -> dict:
        key = _group_key(group_stats[0], x_axis)
        index = pair_index[key[:2]]
        return {
            "color": colors[index % len(colors)],
            "marker": PAIR_MARKERS[index % len(PAIR_MARKERS)],
            "linewidth": 2,
            "markersize": 8,
            "linestyle": LEVEL_LINESTYLES.get(key[2], "-"),
        }

    # Use simple errors/(shots*num_ticks), not sinter's XOR piece conversion.
    # XOR blows up when the per-shot error rate is near/above 50% (common at
    # small d), which made three_yoke_72q d=5 look like ~1 instead of ~1e-4.
    def _y_func(stat: sinter.TaskStats) -> sinter.Fit:
        fit = fit_plot_rate(
            stat,
            highlight_max_likelihood_factor=highlight_max_likelihood_factor,
            normalize=normalize,
            fit_cache=prep.fit_cache,
        )
        if stat.errors == 0:
            return sinter.Fit(low=fit.low, high=fit.high, best=float("nan"))
        return fit

    sinter_plot_custom(
        ax=ax,
        stats=stats,
        x_func=lambda stat: axis_value(stat, x_axis),
        y_func=_y_func,
        group_func=lambda stat: {
            "label": prep.group_labels[_group_key(stat, x_axis)],
            "sort": _sortable(_group_key(stat, x_axis)),
        },
        plot_args_func=_plot_args,
    )
    _plot_extrapolated_fit_lines(
        ax,
        prep,
        colors=colors,
        pair_index=pair_index,
        extrapolated_x=extrapolated_x,
    )

    ax.set_yscale("log")
    set_log_y_limits(
        ax,
        _collect_plot_y_values(
            stats,
            prep,
            extrapolated_x=extrapolated_x,
        ),
        pad_fraction=y_pad_fraction,
    )
    set_log_y_ticks(ax)
    xs = sorted({axis_value(stat, x_axis) for stat in stats} - {None})
    if xs:
        x_ticks = sorted(set(xs) | set(extrapolated_x))
        x_min = min(x_ticks)
        x_max = max(x_ticks)
        if log_x or x_axis in LOG_X_AXES:
            ax.set_xscale("log")
            ax.set_xlim(x_min / 1.3, x_max * 1.3)
            ax.set_xticks(x_ticks)
            ax.set_xticklabels([_format_param_value(x_axis, x) for x in x_ticks])
            ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        else:
            pad = max(0.5, 0.05 * (x_max - x_min))
            ax.set_xlim(x_min - pad, x_max + pad)
            ax.set_xticks(x_ticks)
            if x_axis in PARAM_VALUE_TEXT:
                ax.set_xticklabels([_format_param_value(x_axis, x) for x in x_ticks])
    ax.set_xlabel(AXES[x_axis][1], fontsize=FONT_SIZE_LABEL)
    ax.set_ylabel(
        Y_LABEL_NORMALIZED_TEXT if normalize else Y_LABEL_TEXT,
        fontsize=FONT_SIZE_LABEL,
    )
    figure_title = _fill_text(
        FIGURE_TITLE_TEXT,
        **_text_values(
            _with_derived_params(_single_valued_axes(stats, x_axis)),
            x_axis=x_axis,
            slope_name=prep.slope_name,
            slopes="",
            levels=SLOPE_SEP.join(
                level_title(level)
                for level in sorted({stat.json_metadata["level"] for stat in stats})
            ),
        ),
    )
    if figure_title:
        ax.set_title(figure_title, fontsize=FONT_SIZE_TITLE)
    ax.tick_params(labelsize=FONT_SIZE_TICK)
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    legend_handles, legend_title = _build_compact_legend(
        prep,
        pair_index=pair_index,
        colors=colors,
        show_fit_entry=bool(extrapolated_x),
    )
    ax.legend(
        handles=legend_handles,
        title=legend_title,
        loc=LEGEND_LOC,
        fontsize=FONT_SIZE_LEGEND,
        title_fontsize=FONT_SIZE_LEGEND,
        handlelength=LEGEND_HANDLE_LENGTH,
    )

    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        for fmt in output_formats or [output.suffix.lstrip(".") or "png"]:
            path = output.with_suffix(f".{fmt.lstrip('.')}")
            fig.savefig(path, dpi=150, bbox_inches="tight")
            print(f"Saved plot to {path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def discover_csvs_from_paths(paths: list[Path]) -> list[Path]:
    csv_paths: list[Path] = []
    for path in paths:
        csv_paths.extend(discover_csvs(path))
    if not csv_paths:
        raise FileNotFoundError(
            f"No CSV files found in any of the {len(paths)} --csv path(s)."
        )
    return csv_paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Plot inner/outer logical error rates from two-yoke sinter CSV results "
            "in a single figure.  Pick the x axis with --x and pin the other "
            "parameters with --t/--r/--d/--p; unpinned parameters become "
            "separate curves."
        )
    )
    parser.add_argument(
        "--csv",
        type=Path,
        nargs="+",
        default=CSVS,
        help="Sinter CSV file(s) or director(ies) of CSV files.",
    )
    parser.add_argument(
        "--protocol",
        type=str,
        default=PROTOCOL,
        help="Optional protocol filter (default: plot all protocols in --csv).",
    )
    parser.add_argument(
        "--x",
        "--x-axis",
        choices=sorted(AXES),
        default=X_AXIS,
        dest="x_axis",
        help=(
            "Parameter on the x axis: d (distance), t (t_interval), "
            "r (l1_check_rate), p (physical error rate) or tr "
            f"(t_interval / l1_check_rate). Default: {X_AXIS}."
        ),
    )
    parser.add_argument(
        "--t-interval",
        "--t",
        type=int,
        nargs="+",
        default=T_INTERVAL,
        dest="t_interval",
        help=(
            "t_interval value(s) to plot (default: infer the unique value from "
            "CSV filenames; with --x t or --x tr every value is used)."
        ),
    )
    parser.add_argument(
        "--check-rate",
        "--r",
        type=int,
        nargs="+",
        default=CHECK_RATE,
        dest="check_rate",
        help="l1_check_rate value(s) to keep (default: all, as separate curves).",
    )
    parser.add_argument(
        "--d",
        "--distance",
        type=int,
        nargs="+",
        default=DISTANCE,
        dest="distance",
        help="Distance value(s) to keep (default: all).",
    )
    parser.add_argument(
        "--d-min",
        type=int,
        default=D_MIN,
        help="Smallest distance to plot (default: no lower bound).",
    )
    parser.add_argument(
        "--d-max",
        type=int,
        default=D_MAX,
        help="Largest distance to plot (default: no upper bound).",
    )
    parser.add_argument(
        "--p",
        type=float,
        nargs="+",
        default=PHYSICAL_P,
        dest="physical_p",
        help="Physical error rate value(s) to keep (default: all).",
    )
    parser.add_argument(
        "--tr",
        "--t-over-r",
        type=float,
        nargs="+",
        default=T_OVER_R,
        dest="t_over_r",
        help=(
            "t_interval / l1_check_rate value(s) to keep (default: all). "
            "Matched with a float tolerance, e.g. --tr 6 keeps t24/r4 and t48/r8."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT,
        help="Path to save the plot image.",
    )
    parser.add_argument(
        "--formats",
        type=str,
        nargs="+",
        default=OUTPUT_FORMATS,
        help=(
            "File format(s) to write, e.g. --formats png pdf (default: "
            f"{' '.join(OUTPUT_FORMATS)}). Each is saved with its own extension."
        ),
    )
    parser.add_argument(
        "--highlight-factor",
        type=float,
        default=DEFAULT_HIGHLIGHT_FACTOR,
        help="Likelihood-ratio factor for uncertainty bands.",
    )
    parser.add_argument(
        "--y-pad-fraction",
        type=float,
        default=DEFAULT_Y_PAD_FRACTION,
        help="Log-scale y-axis padding as a fraction of the data span (default: 0.08).",
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Plot raw logical error rate instead of dividing by num_ticks.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show the plot interactively.",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help=(
            "Re-parse every CSV from scratch instead of using/updating the "
            f"per-file aggregation cache in {CACHE_DIR_NAME}/."
        ),
    )
    parser.add_argument(
        "--extrapolate",
        "--extrapolate-d",
        type=float,
        nargs="+",
        default=None,
        dest="extrapolate",
        help=(
            "x values to extend fit lines to (default: none, so no fit lines "
            "are drawn; e.g. --extrapolate 9 11 with --x d)."
        ),
    )
    parser.add_argument(
        "--log-x",
        action="store_true",
        help="Draw the x axis on a log scale (default for --x p).",
    )
    args = parser.parse_args()
    if args.extrapolate is None:
        extrapolated_x = (
            EXTRAPOLATED_D if args.x_axis == "d" else (EXTRAPOLATED_X or [])
        )
    else:
        extrapolated_x = [x for x in args.extrapolate if x > 0]
    if args.x_axis not in FLOAT_AXES:
        extrapolated_x = [int(x) for x in extrapolated_x]
    USE_CACHE = not args.no_cache

    csv_paths = discover_csvs_from_paths(args.csv)
    filters = resolve_filters(
        csv_paths,
        x_axis=args.x_axis,
        t_interval=args.t_interval,
        check_rate=args.check_rate,
        distance=args.distance,
        physical_p=args.physical_p,
        t_over_r=args.t_over_r,
    )
    stats, loaded_csv_paths = load_stats(
        csv_paths,
        protocol=args.protocol,
        filters=filters,
    )
    stats = filter_stats_by_distance(stats, args.d_min, args.d_max)
    n_before = len(stats)
    stats = aggregate_stats_renormalized_by_num_ticks(stats)
    protocols = sorted({stat.json_metadata.get("protocol") for stat in stats})
    output = args.output
    if output is None:
        output = default_output_path(stats, args.x_axis)
    active_filters = {
        AXES[axis][0]: sorted(allowed)
        for axis, allowed in filters.items()
        if allowed is not None
    }
    print(
        f"Loaded {n_before} strong_id stat(s), "
        f"renormalized/merged to {len(stats)} point(s) "
        f"for protocol(s)={protocols} x_axis={AXES[args.x_axis][0]} "
        f"filters={active_filters} "
        f"from {len(loaded_csv_paths)} CSV file(s)."
    )
    normalize = not args.no_normalize
    prep = _prepare_plot(
        stats,
        x_axis=args.x_axis,
        highlight_max_likelihood_factor=args.highlight_factor,
        normalize=normalize,
    )
    print_fit_summary(
        stats,
        highlight_max_likelihood_factor=args.highlight_factor,
        normalize=normalize,
        fit_cache=prep.fit_cache,
    )
    print_slope_summary(prep)
    plot_two_yoke_both_error_rate(
        stats,
        x_axis=args.x_axis,
        output=output.resolve(),
        output_formats=args.formats,
        show=args.show,
        highlight_max_likelihood_factor=args.highlight_factor,
        y_pad_fraction=args.y_pad_fraction,
        normalize=normalize,
        extrapolated_x=extrapolated_x,
        log_x=args.log_x,
        prep=prep,
    )
