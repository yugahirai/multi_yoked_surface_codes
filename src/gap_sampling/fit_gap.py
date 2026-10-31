"""Fit the gap -> llr calibration used by the outer decoder."""

import argparse
import contextlib
import io
import json
import multiprocessing as mp
import os
import re
import time
import traceback
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from _calibration import (
    DEFAULT_OFFSET,
    DEFAULT_SCALE,
    GapCalibration,
    calibration_path,
    fit_logistic_llr,
    fit_piecewise_llr,
    log_loss,
    log_loss_map,
)
from _gap_samples import GAP_DIR, get_gap_samples
from _sorted_samples import sort_and_aggregate

GAP_FILE = re.compile(r"gap_d(?P<d>\d+)_p(?P<p>[\d.]+?)\.(?:csv|npz|mmap)$")
FORK = mp.get_context("fork")

_SORTED = None


def discover_sample_sets():
    matches = (GAP_FILE.search(path.name) for path in GAP_DIR.glob("gap_d*"))
    return sorted({(int(m["d"]), float(m["p"])) for m in matches if m})


def binned_stats(gs, ys, num_bins=12):
    """Per gap-quantile bin: (lo, hi, n, empirical mismatch rate, mean gap)."""
    n_all = len(gs)
    q = np.linspace(0.0, 1.0, num_bins + 1)
    edges = np.unique(np.interp(q * (n_all - 1), np.arange(n_all), gs))
    cum_y = np.concatenate(([0.0], np.cumsum(ys, dtype=np.float64)))
    cum_g = np.concatenate(([0.0], np.cumsum(gs, dtype=np.float64)))
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        i0 = int(np.searchsorted(gs, lo, side="left"))
        i1 = int(np.searchsorted(gs, hi, side="right" if hi == edges[-1] else "left"))
        n = i1 - i0
        if n > 0:
            rows.append((float(lo), float(hi), n, float((cum_y[i1] - cum_y[i0]) / n), float((cum_g[i1] - cum_g[i0]) / n)))
    return rows


def format_binned_table(rows, maps):
    """Empirical vs fitted mismatch probability per bin; ``maps`` is a list of (name, gap->llr)."""
    lines = ["    gap bin           n        empirical" + "".join(f"  {name:>11}" for name, _ in maps)]
    for lo, hi, n, emp, center in rows:
        fits = "".join(f"  {1.0 / (1.0 + np.exp(float(llr_map(center)))):11.3e}" for _, llr_map in maps)
        lines.append(f"    [{lo:6.2f},{hi:6.2f})  {n:9d}  {emp:11.3e}" + fits)
    return "\n".join(lines)


def _fit_piecewise(gaps, mismatch, scale, offset, min_events, shrink_z):
    """Knots of the piecewise correction to an affine fit, and the resulting map."""
    knots_gap, knots_llr = fit_piecewise_llr(
        gaps, mismatch, scale, offset, min_events=min_events, shrink_z=shrink_z, assume_sorted=True
    )
    return knots_gap, knots_llr, GapCalibration(scale, offset, knots_gap, knots_llr, tail_scale=scale)


def _fit_full(want_piecewise, min_events, shrink_z):
    """Fit on the full set.  Logistic fits and log-losses only need the per-distinct-gap counts."""
    s = _SORTED
    g_u, n_u, k_u = s.gap_values, s.trials, s.events
    scale, offset = fit_logistic_llr(g_u, k_u, counts=n_u)
    out = {
        "scale": scale,
        "offset": offset,
        "fitted_log_loss": log_loss(g_u, k_u, scale, offset, counts=n_u),
        "default_log_loss": log_loss(g_u, k_u, DEFAULT_SCALE, DEFAULT_OFFSET, counts=n_u),
    }
    if want_piecewise:
        knots_gap, knots_llr, piecewise = _fit_piecewise(s.gaps, s.mismatch, scale, offset, min_events, shrink_z)
        out.update(
            knots_gap=knots_gap,
            knots_llr=knots_llr,
            piecewise_log_loss=log_loss_map(g_u, k_u, piecewise, counts=n_u),
        )
    return out


def _fit_fold(train_is_fold, min_events, shrink_z):
    """One holdout fold: fit affine + knots on the train half, return the
    (affine, piecewise) log-losses on the test half."""
    s = _SORTED
    train = s.fold if train_is_fold else ~s.fold
    n_tr, k_tr = s.trials_fold, s.events_fold
    n_te, k_te = s.trials - n_tr, s.events - k_tr
    if not train_is_fold:
        n_tr, k_tr, n_te, k_te = n_te, k_te, n_tr, k_tr
    tr, te = n_tr > 0, n_te > 0
    scale, offset = fit_logistic_llr(s.gap_values[tr], k_tr[tr], counts=n_tr[tr])
    _, _, piecewise = _fit_piecewise(s.gaps[train], s.mismatch[train], scale, offset, min_events, shrink_z)
    return (
        log_loss(s.gap_values[te], k_te[te], scale, offset, counts=n_te[te]),
        log_loss_map(s.gap_values[te], k_te[te], piecewise, counts=n_te[te]),
    )


def _table_rows(num_bins):
    return binned_stats(_SORTED.gaps, _SORTED.mismatch, num_bins)


def _call(task):
    fn, args = task
    return fn(*args)


def fit_one(d, p, *, min_events=25, shrink_z=1.0, affine_only=False, seed=0, jobs=None):
    global _SORTED
    jobs = max(1, int(jobs or os.cpu_count() or 1))
    t_start = time.perf_counter()
    samples = get_gap_samples(d, p)
    num_samples = len(samples["gaps"])

    _SORTED = sort_and_aggregate(
        samples["gaps"], samples["actual"], samples["predicted"],
        jobs=jobs, fold_seed=None if affine_only else seed,
    )  # fmt: skip
    num_mismatch = int(_SORTED.events.sum())
    print(f"== d={d} p={p}: n={num_samples}, mismatch={num_mismatch}")
    if num_mismatch == 0:
        print("   no mismatches; skipping (defaults will be used).")
        return
    want_piecewise = not affine_only and num_mismatch >= 4 * min_events

    tasks = [(_fit_full, (want_piecewise, min_events, shrink_z)), (_table_rows, (12,))]
    if want_piecewise:
        tasks += [(_fit_fold, (True, min_events, shrink_z)), (_fit_fold, (False, min_events, shrink_z))]
    if jobs > 1:
        with FORK.Pool(len(tasks)) as pool:
            full, rows, *folds = pool.map(_call, tasks, chunksize=1)
    else:
        full, rows, *folds = map(_call, tasks)
    _SORTED = None

    scale, offset = full["scale"], full["offset"]
    print(
        f"   scale={scale:.6f} offset={offset:.6f} (default {DEFAULT_SCALE}/{DEFAULT_OFFSET})\n"
        f"   log-loss: fitted={full['fitted_log_loss']:.6e} default={full['default_log_loss']:.6e}"
    )
    payload = {
        "d": d,
        "p": p,
        "scale": scale,
        "offset": offset,
        "num_samples": num_samples,
        "num_mismatch": num_mismatch,
        "fitted_log_loss": full["fitted_log_loss"],
        "default_log_loss": full["default_log_loss"],
    }
    maps = [("affine", GapCalibration(scale, offset))]

    if want_piecewise:
        holdout_affine = sum(0.5 * fold[0] for fold in folds)
        holdout_piecewise = sum(0.5 * fold[1] for fold in folds)
        piecewise_wins = holdout_piecewise < holdout_affine
        print(
            f"   holdout log-loss: affine={holdout_affine:.6e} piecewise={holdout_piecewise:.6e}"
            + ("  (piecewise wins)" if piecewise_wins else "  (affine wins)")
        )
        payload.update(holdout_affine_log_loss=holdout_affine, holdout_piecewise_log_loss=holdout_piecewise)
        if piecewise_wins:
            knots_gap, knots_llr = full["knots_gap"], full["knots_llr"]
            maps.append(("piecewise", GapCalibration(scale, offset, knots_gap, knots_llr, tail_scale=scale)))
            print(
                f"   piecewise: {len(knots_gap)} knots (min_events={min_events}), "
                f"in-sample log-loss={full['piecewise_log_loss']:.6e}"
            )
            payload.update(
                knots_gap=[float(v) for v in knots_gap],
                knots_llr=[float(v) for v in knots_llr],
                tail_scale=scale,
                min_events=min_events,
                shrink_z=shrink_z,
                piecewise_log_loss=full["piecewise_log_loss"],
            )
        else:
            print("   piecewise lost the holdout; writing affine only.")
    elif not affine_only:
        print(f"   too few mismatches for a piecewise fit (need {4 * min_events}); writing affine only.")
    print(format_binned_table(rows, maps))

    out_path = calibration_path(d, p)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    print(f"   wrote {out_path}  ({time.perf_counter() - t_start:.1f}s, jobs={jobs})")


def _fit_one_captured(kwargs):
    """fit_one with its report captured, for running several sets at once."""
    buf = io.StringIO()
    ok = True
    with contextlib.redirect_stdout(buf):
        try:
            fit_one(**kwargs)
        except Exception:
            ok = False
            traceback.print_exc(file=buf)
    return buf.getvalue(), ok


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--d", type=int)
    parser.add_argument("--p", type=float)
    parser.add_argument("--min-events", type=int, default=25, help="Mismatch events per adaptive bin of the isotonic fit.")
    parser.add_argument("--shrink-z", type=float, default=1.0, help="Soft-threshold each correction by this many standard errors.")
    parser.add_argument("--affine-only", action="store_true", help="Skip the isotonic fit and write only the affine calibration.")
    parser.add_argument("--jobs", type=int, default=os.cpu_count() or 1, help="Worker processes in total (default: all cores; 1 = serial).")
    parser.add_argument("--set-jobs", type=int, default=1, help="Sample sets fitted concurrently; each gets --jobs / --set-jobs workers.")
    args = parser.parse_args()

    if (args.d is None) != (args.p is None):
        parser.error("Specify both --d and --p, or neither to fit everything.")
    targets = [(args.d, args.p)] if args.d is not None else discover_sample_sets()
    if not targets:
        raise SystemExit(f"No gap sample sets found in {GAP_DIR}.")

    set_jobs = max(1, min(args.set_jobs, len(targets)))
    common = dict(
        min_events=args.min_events,
        shrink_z=args.shrink_z,
        affine_only=args.affine_only,
        jobs=max(1, args.jobs // set_jobs),
    )
    if set_jobs == 1:
        for d, p in targets:
            fit_one(d, p, **common)
    else:
        failed = 0
        with ProcessPoolExecutor(set_jobs, mp_context=FORK) as ex:
            for report, ok in ex.map(_fit_one_captured, [dict(d=d, p=p, **common) for d, p in targets]):
                print(report, end="", flush=True)
                failed += not ok
        if failed:
            raise SystemExit(f"{failed} sample set(s) failed.")
