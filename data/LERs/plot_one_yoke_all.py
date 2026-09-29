import argparse
import math
import string
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.ticker
import numpy as np
import sinter
from sinter._plotting import plot_custom as sinter_plot_custom

SCRIPT_DIR = Path(__file__).resolve().parent

# --- Configuration (edit these) ---
# Written by combine_oneyoke.py (protocols renamed to "[[16,14,2]]" / "[[8,6,2]]").
CSV = SCRIPT_DIR / "one_yoke_combined.csv"
PROTOCOL = None  # e.g. "[[16,14,2]]"; None = plot all protocols found
T_INTERVAL = (
    None  # t_interval value(s) to keep, e.g. [1, 2]; None = all (one curve each)
)
DISTANCE = None  # d value(s) to keep; None = all
D_MIN = 7  # None = no lower bound on the plotted distance; --d-min
D_MAX = None  # None = no upper bound on the plotted distance; --d-max
PHYSICAL_P = None  # p value(s) to keep; None = all
OUTPUT = None  # explicit path (or --output); None = OUTPUT_DIR / OUTPUT_NAME
OUTPUT_DIR = SCRIPT_DIR / "plots"
# Filename template used when OUTPUT is None.  Placeholders:
#   {protocols}  protocol names joined by "+"  (e.g. "16_14_2+8_6_2")
#   {protocol}   the first protocol name only
#   {params}     every parameter fixed across the plot as "_p0.001_t144d" ("" if none)
#   {t} {p}      the fixed value of that axis ("" when it varies), written as
#                in the legend (PARAM_VALUE_TEXT: t=12 -> "144d")
#   {t_suffix}   "" when t varies, otherwise "_t{t}"
OUTPUT_NAME = "{protocols}{t_suffix}.png"
# File formats to write, e.g. ["png", "pdf"]; --formats.  Each format is saved
# next to the others with its own extension (the extension in OUTPUT_NAME or
# --output is replaced).
OUTPUT_FORMATS = ["png"]
# Substrings replaced in protocol names before they go into the filename, so
# "[[16,14,2]]" becomes "16_14_2".
OUTPUT_NAME_REPLACE = {"[": "", "]": "", ",": "_"}

PROTOCOL_TITLES = {
    "[[16,14,2]]": "[[16,14,2]]",
    "[[8,6,2]]": "[[8,6,2]]",
}

# --- Wording (edit these to change the sentences in the plot) ---
# Templates are lists of segments joined by PARAM_SEP.  A segment is dropped
# when any placeholder it uses is empty (no slope, no shared parameter...).
# Placeholders available in every template:
#   {params}      the parameters written with PARAM_TEXT and joined by PARAM_SEP
#   {p} {t}       the raw value of that axis ("" when it is not fixed)
#   {protocol}    the protocol title (entries: only when several are plotted;
#                 title: only when a single one is plotted)
#   {slope_name}  SLOPE_NAME
#   {slopes}      fitted slopes, one per level, joined by SLOPE_SEP (entries)
#   {levels}      level names in linestyle order, joined by SLOPE_SEP (title)
LEVEL_TITLES = {  # legend name per level (CSV "level" metadata)
    "inner": "inner",
    "outer": "outer",
}
AXIS_NAMES = {  # how each axis is called in legend text
    "p": "p",
    "t": "t_1",
}
# How the value of a parameter is written (default: the plain number).  The
# t_interval values count in units of d rounds, so the interval is shown as
# T_INTERVAL_SCALE * value followed by a literal "d" (e.g. t=12 -> "144d").
T_INTERVAL_SCALE = 12  # display interval = T_INTERVAL_SCALE * t_interval from CSV
PARAM_VALUE_TEXT = {
    "t": lambda value: f"{value * T_INTERVAL_SCALE:g}d",
}
# Parameters never written in {params} (legend entries/title, figure title).
# Their {p}-style placeholders still work in explicit templates.
HIDDEN_PARAMS = {"p"}
PARAM_TEXT = "{name}={value}"  # one parameter, e.g. "interval=144d"
PARAM_SEP = ", "  # between parameters and between template segments
LEGEND_ENTRY_TEXT = [
    "{protocol}",
    "{params}",
    "{slope_name}={slopes}",
]  # one colour/marker entry
LEGEND_TITLE_TEXT = ["{params}", "{slope_name}: {levels}"]  # legend title; [] = none
FIGURE_TITLE_TEXT: list[str] = []  # e.g. ["{protocol} ({params})"]; [] = none
SLOPE_NAME = "sl"  # how the fitted slope is called in the legend ("sl=-0.46 / -0.85")
SLOPE_VALUE_TEXT = "{:.2f}"  # one fitted slope
SLOPE_MISSING_TEXT = "n/a"  # slope of a level with too few points
SLOPE_SEP = " / "  # between the per-level slopes / level names
X_LABEL_TEXT = "Inner distance (d)"
Y_LABEL_TEXT = "Logical error rate"
Y_LABEL_NORMALIZED_TEXT = "Logical error rate per inner round"
LEGEND_LOC = "best"  # matplotlib legend location, e.g. "lower left"
LEGEND_HANDLE_LENGTH = 1.2  # length of the legend line samples, in font-size units (matplotlib default 2.0)

# Marker per (protocol, p, t_interval) pair; linestyle per level.
PAIR_MARKERS = "osD^v<>Pp*Xh"
LEVEL_LINESTYLES = {
    "inner": "--",
    "outer": "-",
}

FIG_SIZE = (9, 8)
FONT_SIZE_TICK = 20
FONT_SIZE_LABEL = 24
FONT_SIZE_LEGEND = 16
FONT_SIZE_TITLE = 16

DEFAULT_HIGHLIGHT_FACTOR = 100
DEFAULT_Y_PAD_FRACTION = 0.08


def _resolve_path(path: Path) -> Path:
    path = Path(path)
    if path.exists():
        return path.resolve()
    script_relative = (SCRIPT_DIR / path).resolve()
    if script_relative.exists():
        return script_relative
    return path.resolve()


def discover_csvs(paths: Path | list[Path]) -> list[Path]:
    """CSV files under ``paths``; missing files/dirs and empty dirs are skipped.

    Raises only when nothing at all was found.
    """
    if isinstance(paths, Path):
        paths = [paths]
    csv_paths: list[Path] = []
    for path in paths:
        path = _resolve_path(path)
        if path.is_file():
            csv_paths.append(path)
            continue
        if path.is_dir():
            found = sorted(path.glob("*.csv"))
            if not found:
                print(f"skipping {path}: no CSV files", flush=True)
            csv_paths.extend(found)
            continue
        print(f"skipping {path}: not found", flush=True)
    if not csv_paths:
        raise FileNotFoundError(f"No CSV files found in any of: {list(paths)}")
    return csv_paths


def _t_interval(stat: sinter.TaskStats) -> int:
    return stat.json_metadata.get("t_interval", 1)


def _display_interval(t_interval: int) -> int:
    return T_INTERVAL_SCALE * t_interval


def _format_value(value) -> str:
    return f"{value:g}" if isinstance(value, float) else str(value)


def _format_param_value(axis: str, value) -> str:
    """Value of one parameter as written in the plot (see PARAM_VALUE_TEXT)."""
    if axis in PARAM_VALUE_TEXT:
        return PARAM_VALUE_TEXT[axis](value)
    return _format_value(value)


def _format_axis(axis: str, value) -> str:
    return PARAM_TEXT.format(
        name=AXIS_NAMES.get(axis, axis), value=_format_param_value(axis, value)
    )


def _axis_value(stat: sinter.TaskStats, axis: str):
    return _t_interval(stat) if axis == "t" else stat.json_metadata["p"]


# Parameters a curve pair is identified by, in the order they are written.
PARAM_AXES = ("p", "t")


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


def _text_values(axis_items, **extra) -> dict:
    """Placeholder values shared by every template (see the wording block)."""
    values = {axis: "" for axis in PARAM_AXES}
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
    values["slope_name"] = SLOPE_NAME
    values.update(extra)
    return values


def _single_valued_axes(stats: list[sinter.TaskStats]) -> list[tuple[str, object]]:
    """(axis, value) for every parameter that takes one value across ``stats``."""
    fixed: list[tuple[str, object]] = []
    for axis in PARAM_AXES:
        values = {_axis_value(stat, axis) for stat in stats}
        if len(values) == 1:
            fixed.append((axis, next(iter(values))))
    return fixed


def default_output_path(protocols: list[str], stats: list[sinter.TaskStats]) -> Path:
    """OUTPUT_DIR / OUTPUT_NAME with the template placeholders filled in."""
    names = list(protocols)
    for old, new in OUTPUT_NAME_REPLACE.items():
        names = [name.replace(old, new) for name in names]
    fixed = _single_valued_axes(stats)
    values = {axis: "" for axis in PARAM_AXES}
    # Values are written as in the legend (PARAM_VALUE_TEXT): t12 -> "144d".
    values.update({axis: _format_param_value(axis, value) for axis, value in fixed})
    values.update(
        protocols="+".join(names),
        protocol=names[0] if names else "",
        params="".join(f"_{axis}{values[axis]}" for axis, _ in fixed),
        t_suffix=f"_t{values['t']}" if values["t"] else "",
    )
    return OUTPUT_DIR / OUTPUT_NAME.format(**values)


def discover_t_intervals(stats: list[sinter.TaskStats]) -> list[int]:
    return sorted({_t_interval(stat) for stat in stats})


def discover_protocols(stats: list[sinter.TaskStats]) -> list[str]:
    return sorted(
        {
            protocol
            for protocol in (stat.json_metadata.get("protocol") for stat in stats)
            if protocol is not None
        }
    )


def filter_stats_by_axis(
    stats: list[sinter.TaskStats],
    axis: str,
    values: list | None,
) -> list[sinter.TaskStats]:
    """Keep only stats whose ``axis`` value is in ``values`` (None = keep all)."""
    if values is None:
        return stats
    allowed = set(values)
    filtered = [
        stat
        for stat in stats
        if any(math.isclose(_axis_value(stat, axis), v) for v in allowed)
    ]
    if not filtered:
        available = sorted({_axis_value(stat, axis) for stat in stats})
        raise ValueError(
            f"No stats with {AXIS_NAMES.get(axis, axis)} in {sorted(allowed)} found. "
            f"Available values: {available}."
        )
    return filtered


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


def filter_stats_by_protocol(
    stats: list[sinter.TaskStats],
    protocol: str,
) -> list[sinter.TaskStats]:
    filtered = [
        stat for stat in stats if stat.json_metadata.get("protocol") == protocol
    ]
    if not filtered:
        raise ValueError(f"No stats with protocol={protocol!r} found.")
    return filtered


def load_base_stats(
    csv_paths: list[Path],
    *,
    protocol: str | None = None,
) -> tuple[list[sinter.TaskStats], list[str]]:
    stats = sinter.read_stats_from_csv_files(*csv_paths)
    protocols = discover_protocols(stats)
    if not protocols:
        raise ValueError("No protocol metadata found in the provided CSV file(s).")
    if protocol is not None:
        stats = filter_stats_by_protocol(stats, protocol)
        return stats, [protocol]
    return stats, protocols


def protocol_title(protocol: str) -> str:
    return PROTOCOL_TITLES.get(protocol, protocol.replace("_", " ").title())


def combined_protocol_title(protocols: list[str]) -> str:
    if len(protocols) == 1:
        return protocol_title(protocols[0])
    return " / ".join(protocol_title(protocol) for protocol in protocols)


def level_title(level: str) -> str:
    return LEVEL_TITLES.get(level, level.replace("_", " ").title())


def num_ticks_from_metadata(meta: dict) -> int:
    if "num_ticks" in meta:
        return meta["num_ticks"]
    d = meta["d"]
    if "t_interval" in meta and "ticks_per_shot" in meta:
        return (1 + meta["t_interval"] * meta["ticks_per_shot"]) * d
    if "ticks_per_shot" in meta:
        return meta["ticks_per_shot"] * d
    raise KeyError(
        "CSV metadata must include num_ticks or ticks_per_shot for normalization."
    )


def _num_ticks(stat: sinter.TaskStats) -> int:
    return num_ticks_from_metadata(stat.json_metadata)


def normalized_error_rate(stat: sinter.TaskStats) -> float:
    shots = stat.shots - stat.discards
    if shots == 0:
        return 0.0
    return stat.errors / shots / _num_ticks(stat)


def normalization_factor(stat: sinter.TaskStats) -> float:
    return _num_ticks(stat)


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


_fit_cache: dict[tuple[str, float, bool], sinter.Fit] = {}


def fit_plot_rate(
    stat: sinter.TaskStats,
    *,
    highlight_max_likelihood_factor: float,
    normalize: bool,
) -> sinter.Fit:
    # The same fit is requested several times per stat (summary, labels,
    # slopes, axis limits, plot); memoise the binomial likelihood search.
    cache_key = (stat.strong_id, highlight_max_likelihood_factor, normalize)
    fit = _fit_cache.get(cache_key)
    if fit is not None:
        return fit
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
    _fit_cache[cache_key] = fit
    return fit


def print_fit_summary(
    stats: list[sinter.TaskStats],
    *,
    highlight_max_likelihood_factor: float,
    normalize: bool,
) -> None:
    for stat in stats:
        fit = fit_plot_rate(
            stat,
            highlight_max_likelihood_factor=highlight_max_likelihood_factor,
            normalize=normalize,
        )
        meta = stat.json_metadata
        raw_rate = stat.errors / stat.shots if stat.shots else 0.0
        print(
            f"protocol={protocol_title(meta['protocol'])} "
            f"level={level_title(meta['level'])} d={meta['d']} p={meta['p']} "
            f"interval={_display_interval(_t_interval(stat))} num_ticks={_num_ticks(stat)}: "
            f"{stat.shots} shots, {stat.errors} errors, "
            f"rate={raw_rate:.6g}, "
            f"plot_rate={fit.best:.6g}, "
            f"normalized_rate={normalized_error_rate(stat):.6g}, "
            f"CI=[{fit.low:.6g}, {fit.high:.6g}]"
        )


def _collect_plot_y_values(
    stats: list[sinter.TaskStats],
    *,
    highlight_max_likelihood_factor: float,
    normalize: bool,
) -> list[float]:
    values: list[float] = []
    for stat in stats:
        fit = fit_plot_rate(
            stat,
            highlight_max_likelihood_factor=highlight_max_likelihood_factor,
            normalize=normalize,
        )
        if fit.best > 0 and not np.isnan(fit.best):
            values.append(float(fit.best))

    for group in _group_stats(stats).values():
        ds, rates = _plot_rates_for_group(
            group,
            highlight_max_likelihood_factor=highlight_max_likelihood_factor,
            normalize=normalize,
        )
        for rate in rates:
            if rate > 0 and not np.isnan(rate):
                values.append(float(rate))
        slope_fit = fit_log10_slope(ds, rates)
        if slope_fit is None:
            continue
        slope, intercept = slope_fit
        d_line = np.linspace(min(ds), max(ds), 100)
        rate_line = predict_rate(d_line, slope, intercept)
        values.extend(
            float(value) for value in rate_line if value > 0 and not np.isnan(value)
        )
    return values


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


def fit_log10_slope(ds: list[float], rates: list[float]) -> tuple[float, float] | None:
    """Fit log10(rate) = slope * d + intercept (straight on log-y vs linear-x)."""
    if len(ds) < 2 or len(set(ds)) < 2:
        return None
    positive = [(d, rate) for d, rate in zip(ds, rates) if d > 0 and rate > 0]
    if len(positive) < 2 or len({d for d, _ in positive}) < 2:
        return None
    d_arr = np.array([d for d, _ in positive])
    log_r = np.log10([rate for _, rate in positive])
    slope, intercept = np.polyfit(d_arr, log_r, 1)
    return float(slope), float(intercept)


def predict_rate(d: float | np.ndarray, slope: float, intercept: float) -> np.ndarray:
    return 10 ** (slope * np.asarray(d) + intercept)


def _group_key(stat: sinter.TaskStats) -> tuple[str, float, int, str]:
    meta = stat.json_metadata
    return (meta["protocol"], meta["p"], _t_interval(stat), meta["level"])


def _failure_units_per_shot(stat: sinter.TaskStats, *, normalize: bool) -> float:
    if not normalize:
        return 1
    return _num_ticks(stat)


def _group_stats(
    stats: list[sinter.TaskStats],
) -> dict[tuple[str, float, int, str], list[sinter.TaskStats]]:
    grouped: dict[tuple[str, float, int, str], list[sinter.TaskStats]] = defaultdict(
        list
    )
    for stat in stats:
        grouped[_group_key(stat)].append(stat)
    return grouped


def _plot_rates_for_group(
    group: list[sinter.TaskStats],
    *,
    highlight_max_likelihood_factor: float,
    normalize: bool,
) -> tuple[list[float], list[float]]:
    group = sorted(group, key=lambda stat: stat.json_metadata["d"])
    ds = [stat.json_metadata["d"] for stat in group]
    rates = [
        fit_plot_rate(
            stat,
            highlight_max_likelihood_factor=highlight_max_likelihood_factor,
            normalize=normalize,
        ).best
        for stat in group
    ]
    return ds, rates


def _build_group_labels(
    stats: list[sinter.TaskStats],
    *,
    highlight_max_likelihood_factor: float,
    normalize: bool,
    include_protocol: bool,
) -> dict[tuple[str, float, int, str], str]:
    labels: dict[tuple[str, float, int, str], str] = {}
    for key, group in sorted(_group_stats(stats).items()):
        protocol, p, t_interval, level = key
        ds, rates = _plot_rates_for_group(
            group,
            highlight_max_likelihood_factor=highlight_max_likelihood_factor,
            normalize=normalize,
        )
        slope_fit = fit_log10_slope(ds, rates)
        parts = [protocol_title(protocol)] if include_protocol else []
        parts.append(level_title(level))
        parts.extend(
            _format_axis(axis, value) for axis, value in (("p", p), ("t", t_interval))
        )
        if slope_fit is not None:
            parts.append(f"{SLOPE_NAME}={slope_fit[0]:.2f}")
        labels[key] = ", ".join(parts)
    return labels


def print_slope_summary(
    stats: list[sinter.TaskStats],
    *,
    highlight_max_likelihood_factor: float,
    normalize: bool,
) -> None:
    for key, group in sorted(_group_stats(stats).items()):
        protocol, p, t_interval, level = key
        ds, rates = _plot_rates_for_group(
            group,
            highlight_max_likelihood_factor=highlight_max_likelihood_factor,
            normalize=normalize,
        )
        slope_fit = fit_log10_slope(ds, rates)
        level_label = level_title(level)
        prefix = f"protocol={protocol_title(protocol)} "
        if slope_fit is None:
            print(
                f"{prefix}level={level_label} p={p} "
                f"interval={_display_interval(t_interval)}: "
                "slope=n/a (need >=2 distances)"
            )
        else:
            slope, _ = slope_fit
            print(
                f"{prefix}level={level_label} p={p} "
                f"interval={_display_interval(t_interval)}: "
                f"slope={slope:.3f}"
            )


def _group_slopes(
    stats: list[sinter.TaskStats],
    *,
    highlight_max_likelihood_factor: float,
    normalize: bool,
) -> dict[tuple[str, float, int, str], float | None]:
    slopes: dict[tuple[str, float, int, str], float | None] = {}
    for key, group in _group_stats(stats).items():
        ds, rates = _plot_rates_for_group(
            group,
            highlight_max_likelihood_factor=highlight_max_likelihood_factor,
            normalize=normalize,
        )
        fit = fit_log10_slope(ds, rates)
        slopes[key] = None if fit is None else fit[0]
    return slopes


def _build_compact_legend(
    stats: list[sinter.TaskStats],
    *,
    pair_index: dict[tuple[str, float, int], int],
    colors: list,
    highlight_max_likelihood_factor: float,
    normalize: bool,
    include_protocol: bool,
) -> tuple[list[Line2D], str | None]:
    """Legend that explains the encoding instead of listing every curve.

    Linestyle entries (black) name the level; one colour+marker entry per
    (protocol, p, t_interval) pair names the pair by whatever differs between
    pairs (LEGEND_ENTRY_TEXT) and carries its fitted slopes, one per level in
    the order of the linestyle entries.  Parameters shared by every pair move
    into the legend title (LEGEND_TITLE_TEXT).
    """
    slopes = _group_slopes(
        stats,
        highlight_max_likelihood_factor=highlight_max_likelihood_factor,
        normalize=normalize,
    )
    levels = sorted(
        {key[3] for key in slopes},
        key=lambda level: (
            list(LEVEL_LINESTYLES).index(level)
            if level in LEVEL_LINESTYLES
            else len(LEVEL_LINESTYLES)
        ),
    )
    pair_keys = sorted(pair_index, key=pair_index.get)
    protocols = {key[0] for key in pair_keys}
    axis_values = {
        axis: {key[1 + PARAM_AXES.index(axis)] for key in pair_keys}
        for axis in PARAM_AXES
    }
    varying_axes = [axis for axis in PARAM_AXES if len(axis_values[axis]) > 1]

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

    for key in pair_keys:
        protocol, p, t_interval = key
        axis_items = [
            (axis, value)
            for axis, value in (("p", p), ("t", t_interval))
            if axis in varying_axes
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
                axis_items,
                protocol=(
                    protocol_title(protocol)
                    if include_protocol and len(protocols) > 1
                    else ""
                ),
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
        for axis in PARAM_AXES
        if axis not in varying_axes
    ]
    levels_text = ""
    if len(levels) > 1 and any(slope is not None for slope in slopes.values()):
        levels_text = SLOPE_SEP.join(level_title(level) for level in levels)
    title = _fill_text(
        LEGEND_TITLE_TEXT,
        **_text_values(
            shared_items,
            protocol=(
                protocol_title(next(iter(protocols)))
                if include_protocol and len(protocols) == 1
                else ""
            ),
            slopes="",
            levels=levels_text,
        ),
    )
    return handles, title or None


def plot_one_yoke_error_rate(
    stats: list[sinter.TaskStats],
    *,
    protocols: list[str],
    output: Path | None,
    output_formats: list[str] | None = None,
    show: bool,
    highlight_max_likelihood_factor: float,
    y_pad_fraction: float,
    normalize: bool,
) -> None:
    fig, ax = plt.subplots(figsize=FIG_SIZE, constrained_layout=True)
    group_labels = _build_group_labels(
        stats,
        highlight_max_likelihood_factor=highlight_max_likelihood_factor,
        normalize=normalize,
        include_protocol=len(protocols) > 1,
    )
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    # Zero-yoke (inner) and one-yoke (outer) curves of the same
    # (protocol, p, t_interval) share a colour and marker; the level is told
    # apart by linestyle only.
    pair_keys = sorted({_group_key(stat)[:3] for stat in stats})
    pair_index = {key: index for index, key in enumerate(pair_keys)}

    def _plot_args(_index, _group_key_value, group_stats) -> dict:
        key = _group_key(group_stats[0])
        index = pair_index[key[:3]]
        return {
            "color": colors[index % len(colors)],
            "marker": PAIR_MARKERS[index % len(PAIR_MARKERS)],
            "linewidth": 2,
            "markersize": 8,
            "linestyle": LEVEL_LINESTYLES.get(key[3], "-"),
        }

    # Use errors/(shots*num_ticks), not sinter's XOR piece conversion.
    # XOR blows up when the per-shot error rate is near/above 50% (common at
    # small d), which makes confidence bands span up to ~1 and ruins the plot.
    def _y_func(stat: sinter.TaskStats) -> sinter.Fit:
        fit = fit_plot_rate(
            stat,
            highlight_max_likelihood_factor=highlight_max_likelihood_factor,
            normalize=normalize,
        )
        if stat.errors == 0:
            return sinter.Fit(low=fit.low, high=fit.high, best=float("nan"))
        return fit

    stats = sorted(stats, key=lambda stat: (*_group_key(stat), stat.json_metadata["d"]))

    sinter_plot_custom(
        ax=ax,
        stats=stats,
        x_func=lambda stat: stat.json_metadata["d"],
        y_func=_y_func,
        group_func=lambda stat: {
            "label": group_labels[_group_key(stat)],
            "sort": _group_key(stat),
        },
        plot_args_func=_plot_args,
    )

    ax.set_yscale("log")
    set_log_y_limits(
        ax,
        _collect_plot_y_values(
            stats,
            highlight_max_likelihood_factor=highlight_max_likelihood_factor,
            normalize=normalize,
        ),
        pad_fraction=y_pad_fraction,
    )
    set_log_y_ticks(ax)
    ds = sorted({stat.json_metadata["d"] for stat in stats})
    if ds:
        pad = max(0.5, 0.05 * (max(ds) - min(ds)))
        ax.set_xlim(min(ds) - pad, max(ds) + pad)
        ax.set_xticks(ds)
    ax.set_xlabel(X_LABEL_TEXT, fontsize=FONT_SIZE_LABEL)
    ax.set_ylabel(
        Y_LABEL_NORMALIZED_TEXT if normalize else Y_LABEL_TEXT,
        fontsize=FONT_SIZE_LABEL,
    )
    figure_title = _fill_text(
        FIGURE_TITLE_TEXT,
        **_text_values(
            _single_valued_axes(stats),
            protocol=combined_protocol_title(protocols),
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
        stats,
        pair_index=pair_index,
        colors=colors,
        highlight_max_likelihood_factor=highlight_max_likelihood_factor,
        normalize=normalize,
        include_protocol=len(protocols) > 1,
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Plot inner/outer logical error rates from one-yoke sinter CSV results "
            "(all protocols/CSVs in one figure by default)."
        )
    )
    parser.add_argument(
        "--csv",
        type=Path,
        nargs="+",
        default=CSV,
        help="Sinter CSV file(s) or directory(ies) of CSV files.",
    )
    parser.add_argument(
        "--protocol",
        type=str,
        default=PROTOCOL,
        help=(
            "Protocol to plot (default: all protocols found in CSV metadata, "
            "e.g. '[[8,6,2]]')."
        ),
    )
    parser.add_argument(
        "--t-interval",
        type=int,
        nargs="+",
        default=T_INTERVAL,
        help=(
            "t_interval value(s) to plot (default: every t_interval found, as "
            "separate curves)."
        ),
    )
    parser.add_argument(
        "--d",
        type=int,
        nargs="+",
        default=DISTANCE,
        dest="distance",
        help="Distance value(s) to keep (default: all).",
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
    args = parser.parse_args()

    csv_paths = discover_csvs(args.csv)
    stats, protocols = load_base_stats(csv_paths, protocol=args.protocol)
    stats = filter_stats_by_axis(stats, "t", args.t_interval)
    stats = filter_stats_by_axis(stats, "p", args.physical_p)
    if args.distance is not None:
        allowed_d = set(args.distance)
        stats = [stat for stat in stats if stat.json_metadata["d"] in allowed_d]
        if not stats:
            raise ValueError(f"No stats with d in {sorted(allowed_d)} found.")
    stats = filter_stats_by_distance(stats, args.d_min, args.d_max)

    output = args.output
    if output is None:
        output = default_output_path(protocols, stats)

    t_label = f"interval={[_display_interval(t) for t in discover_t_intervals(stats)]}"
    p_label = f"p={sorted({stat.json_metadata['p'] for stat in stats})}"
    print(
        f"Loaded {len(stats)} aggregated stat(s) "
        f"for protocol={protocols} {p_label} {t_label} "
        f"from {len(csv_paths)} CSV file(s)."
    )
    print_fit_summary(
        stats,
        highlight_max_likelihood_factor=args.highlight_factor,
        normalize=not args.no_normalize,
    )
    print_slope_summary(
        stats,
        highlight_max_likelihood_factor=args.highlight_factor,
        normalize=not args.no_normalize,
    )
    plot_one_yoke_error_rate(
        stats,
        protocols=protocols,
        output=output.resolve(),
        output_formats=args.formats,
        show=args.show,
        highlight_max_likelihood_factor=args.highlight_factor,
        y_pad_fraction=args.y_pad_fraction,
        normalize=not args.no_normalize,
    )
