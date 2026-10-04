"""Gap -> log-likelihood-ratio calibration shared by Patch/Circuit/Decoder."""

from __future__ import annotations

import ctypes
import json
import os
import subprocess
import tempfile
import warnings
from functools import lru_cache
from pathlib import Path

import numpy as np

DEFAULT_SCALE = 0.95
DEFAULT_OFFSET = 0.0

CALIBRATION_DIR = Path(__file__).resolve().parents[2] / "data" / "calibration"


def calibration_path(d: int, p: float) -> Path:
    return CALIBRATION_DIR / f"gap_d{d}_p{p}.json"


@lru_cache(maxsize=None)
def get_calibration(d: int, p: float) -> tuple[float, float]:
    """Return (scale, offset) of the affine gap->llr map for the sample set."""
    path = calibration_path(d, p)
    if path.is_file():
        with open(path) as f:
            data = json.load(f)
        return float(data["scale"]), float(data["offset"])
    warnings.warn(
        f"No fitted gap calibration at {path}; using default "
        f"scale={DEFAULT_SCALE}, offset={DEFAULT_OFFSET}. "
        "Run fit_gap.py to fit one.",
        RuntimeWarning,
    )
    return DEFAULT_SCALE, DEFAULT_OFFSET


class GapCalibration:
    """Monotone gap -> llr map, callable on scalars or arrays of gaps."""

    def __init__(
        self,
        scale: float,
        offset: float,
        knots_gap=None,
        knots_llr=None,
        tail_scale: float | None = None,
        num_samples: int = 0,
    ):
        self.scale = float(scale)
        self.offset = float(offset)
        self.num_samples = int(num_samples)
        if knots_gap is not None:
            self.knots_gap = np.asarray(knots_gap, dtype=float)
            self.knots_llr = np.asarray(knots_llr, dtype=float)
            if len(self.knots_gap) < 2:
                raise ValueError("Piecewise calibration needs at least 2 knots.")
        else:
            self.knots_gap = None
            self.knots_llr = None
        self.tail_scale = float(tail_scale) if tail_scale is not None else self.scale

    @property
    def label(self) -> str:
        return "affine" if self.knots_gap is None else "piecewise"

    def __call__(self, gaps):
        g = np.asarray(gaps, dtype=float)
        if self.knots_gap is None:
            return self.scale * g + self.offset
        out = np.interp(g, self.knots_gap, self.knots_llr)
        right = g > self.knots_gap[-1]
        if np.any(right):
            out = np.where(
                right,
                self.knots_llr[-1] + self.tail_scale * (g - self.knots_gap[-1]),
                out,
            )
        return out


@lru_cache(maxsize=None)
def get_calibration_map(d: int, p: float) -> GapCalibration:
    """Return the fitted gap->llr map: piecewise if available, else affine."""
    path = calibration_path(d, p)
    if path.is_file():
        with open(path) as f:
            data = json.load(f)
        return GapCalibration(
            scale=data["scale"],
            offset=data["offset"],
            knots_gap=data.get("knots_gap"),
            knots_llr=data.get("knots_llr"),
            tail_scale=data.get("tail_scale"),
            num_samples=data.get("num_samples", 0),
        )
    warnings.warn(
        f"No fitted gap calibration at {path}; using default affine "
        f"scale={DEFAULT_SCALE}, offset={DEFAULT_OFFSET}. "
        "Run fit_gap.py to fit one.",
        RuntimeWarning,
    )
    return GapCalibration(DEFAULT_SCALE, DEFAULT_OFFSET)


def fit_piecewise_llr(
    gaps: np.ndarray,
    mismatch: np.ndarray,
    scale: float,
    offset: float,
    min_events: int = 10,
    prior: float = 0.5,
    shrink_z: float = 1.0,
    assume_sorted: bool = False,
) -> tuple[list[float], list[float]]:
    """Fit a monotone piecewise-linear correction of the affine gap->llr map."""
    g = np.asarray(gaps, dtype=float)
    y = np.asarray(mismatch, dtype=bool)
    total = int(y.sum())
    if total < 2 * min_events:
        raise ValueError(
            f"Need at least {2 * min_events} mismatches to fit a piecewise "
            f"calibration, got {total}."
        )
    if assume_sorted:
        gs, ys = g, y
    else:
        order = np.argsort(g, kind="stable")
        gs = g[order]
        ys = y[order]
    p_affine = 1.0 / (1.0 + np.exp(scale * gs + offset))
    ck = np.cumsum(ys, dtype=np.float64)
    ce = np.cumsum(p_affine)
    cg = np.cumsum(gs * p_affine)
    num = len(gs)
    cuts = [0]
    while cuts[-1] < num:
        start = cuts[-1]
        base_k = ck[start - 1] if start else 0.0
        base_e = ce[start - 1] if start else 0.0
        i_k = int(np.searchsorted(ck, base_k + min_events))
        i_e = int(np.searchsorted(ce, base_e + min_events))
        cuts.append(min(min(i_k, i_e) + 1, num))

    def _seg(cum: np.ndarray, lo: int, hi: int) -> float:
        return float(cum[hi - 1] - (cum[lo - 1] if lo else 0.0))

    k = np.array([_seg(ck, lo, hi) for lo, hi in zip(cuts[:-1], cuts[1:])])
    e = np.array([_seg(ce, lo, hi) for lo, hi in zip(cuts[:-1], cuts[1:])])
    gsum = np.array([_seg(cg, lo, hi) for lo, hi in zip(cuts[:-1], cuts[1:])])
    if len(k) >= 2 and max(k[-1], e[-1]) < min_events / 2.0:
        k[-2] += k[-1]
        e[-2] += e[-1]
        gsum[-2] += gsum[-1]
        k, e, gsum = k[:-1], e[:-1], gsum[:-1]

    def knot(block: list[float]) -> tuple[float, float]:
        kb, eb, gb = block
        g_star = gb / eb
        shift = float(np.log((kb + prior) / (eb + prior)))
        se = shrink_z / np.sqrt(kb + prior)
        shift = np.sign(shift) * max(0.0, abs(shift) - se)
        return g_star, scale * g_star + offset - shift

    blocks: list[list[float]] = [
        [kb, eb, gb] for kb, eb, gb in zip(k.tolist(), e.tolist(), gsum.tolist())
    ]
    merged = True
    while merged:
        merged = False
        pooled: list[list[float]] = []
        for block in blocks:
            pooled.append(block)
            while len(pooled) >= 2 and knot(pooled[-2])[1] >= knot(pooled[-1])[1]:
                kb2, eb2, gb2 = pooled.pop()
                pooled[-1][0] += kb2
                pooled[-1][1] += eb2
                pooled[-1][2] += gb2
                merged = True
        blocks = pooled

    knots_gap: list[float] = []
    knots_llr: list[float] = []
    for block in blocks:
        g_star, llr = knot(block)
        knots_gap.append(float(g_star))
        knots_llr.append(float(llr))
    g0 = min(0.0, float(gs[0]))
    if knots_gap[0] > g0:
        first_shift = scale * knots_gap[0] + offset - knots_llr[0]
        knots_gap.insert(0, g0)
        knots_llr.insert(0, scale * g0 + offset - first_shift)
    return knots_gap, knots_llr


def _load_boxplus():
    """Compile (if needed) and load the C box-plus kernel; None on failure."""
    if not os.environ.get("GAP_SIM_C_BOXPLUS"):
        return None
    here = Path(__file__).resolve().parent
    src = here / "_boxplus.c"
    so = here / "_boxplus.so"
    if not src.is_file():
        return None
    try:
        if not so.is_file() or so.stat().st_mtime < src.stat().st_mtime:
            fd, tmp = tempfile.mkstemp(prefix="_boxplus.", suffix=".so", dir=here)
            os.close(fd)
            subprocess.run(
                ["gcc", "-O2", "-shared", "-fPIC", "-ffp-contract=off",
                 "-o", tmp, str(src), "-lm"],
                check=True,
                capture_output=True,
            )
            os.replace(tmp, so)
        lib = ctypes.CDLL(str(so))
    except Exception as exc:
        warnings.warn(
            f"C box-plus kernel unavailable ({exc!r}); using the numpy path.",
            RuntimeWarning,
        )
        return None
    fn = lib.boxplus_rows
    fn.restype = None
    fn.argtypes = (
        ctypes.c_void_p,
        ctypes.c_ssize_t,
        ctypes.c_ssize_t,
        ctypes.c_void_p,
    )
    return fn


_BOXPLUS_ROWS = _load_boxplus()


def _boxplus_rows_positive(flat: np.ndarray):
    """Box-plus fold of ``flat`` (rows of positive llrs), or None."""
    n, k = flat.shape
    cols = np.ascontiguousarray(flat.T)
    out = cols[0].copy()
    mag = np.empty(n)
    t_plus = np.empty(n)
    t_minus = np.empty(n)
    for j in range(1, k):
        b = cols[j]
        np.minimum(out, b, out=mag)
        np.add(out, b, out=t_plus)
        np.negative(t_plus, out=t_plus)
        np.exp(t_plus, out=t_plus)
        np.log1p(t_plus, out=t_plus)
        np.subtract(out, b, out=t_minus)
        np.abs(t_minus, out=t_minus)
        np.negative(t_minus, out=t_minus)
        np.exp(t_minus, out=t_minus)
        np.log1p(t_minus, out=t_minus)
        np.add(mag, t_plus, out=out)
        np.subtract(out, t_minus, out=out)
        if not (out > 0.0).all():
            return None
    return out


def combine_llrs(llrs: np.ndarray) -> np.ndarray:
    """XOR-combine independent error events along the last axis."""
    llrs = np.asarray(llrs, dtype=float)
    k = llrs.shape[-1] if llrs.ndim else 0
    if _BOXPLUS_ROWS is not None and k >= 1:
        x = np.ascontiguousarray(llrs)
        out = np.empty(llrs.shape[:-1], dtype=float)
        _BOXPLUS_ROWS(x.ctypes.data, out.size, k, out.ctypes.data)
        return out
    if k <= 1:
        return llrs[..., 0]
    flat = np.ascontiguousarray(llrs).reshape(-1, k)
    if (flat > 0.0).all():
        out = _boxplus_rows_positive(flat)
        if out is not None:
            return out.reshape(llrs.shape[:-1])
    n = flat.shape[0]
    out = flat[:, 0].copy()
    sgn = np.empty(n)
    mag = np.empty(n)
    t_plus = np.empty(n)
    t_minus = np.empty(n)
    for j in range(1, k):
        b = flat[:, j]
        np.multiply(np.sign(out), np.sign(b), out=sgn)
        np.minimum(np.abs(out), np.abs(b), out=mag)
        np.multiply(sgn, mag, out=mag)
        np.add(out, b, out=t_plus)
        np.abs(t_plus, out=t_plus)
        np.negative(t_plus, out=t_plus)
        np.exp(t_plus, out=t_plus)
        np.log1p(t_plus, out=t_plus)
        np.subtract(out, b, out=t_minus)
        np.abs(t_minus, out=t_minus)
        np.negative(t_minus, out=t_minus)
        np.exp(t_minus, out=t_minus)
        np.log1p(t_minus, out=t_minus)
        np.add(mag, t_plus, out=out)
        np.subtract(out, t_minus, out=out)
    return out.reshape(llrs.shape[:-1])


def aggregate_gap_samples(
    gaps: np.ndarray, mismatch: np.ndarray, assume_sorted: bool = False
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Collapse per-sample (gap, mismatch) rows into per-unique-gap counts."""
    g = np.asarray(gaps, dtype=float)
    y = np.asarray(mismatch, dtype=bool)
    if not assume_sorted:
        order = np.argsort(g, kind="stable")
        g = g[order]
        y = y[order]
    if g.size == 0:
        return g, np.zeros(0), np.zeros(0)
    starts = np.flatnonzero(np.concatenate(([True], g[1:] != g[:-1])))
    trials = np.diff(np.append(starts, g.size)).astype(float)
    events = np.add.reduceat(y.astype(np.int64), starts).astype(float)
    return g[starts], trials, events


def fit_logistic_llr(
    gaps: np.ndarray,
    mismatch: np.ndarray,
    max_iter: int = 100,
    tol: float = 1e-10,
    counts: np.ndarray | None = None,
) -> tuple[float, float]:
    """Fit P(mismatch | gap) = 1 / (1 + exp(scale * gap + offset)) by Newton."""
    if counts is None:
        g, n, k = aggregate_gap_samples(gaps, mismatch)
    else:
        g = np.asarray(gaps, dtype=float)
        n = np.asarray(counts, dtype=float)
        k = np.asarray(mismatch, dtype=float)
    k_total = k.sum()
    if k_total == 0 or k_total == n.sum():
        raise ValueError("Cannot fit calibration: mismatch labels are constant.")
    for w0 in ((-DEFAULT_OFFSET, -DEFAULT_SCALE), (0.0, 0.0)):
        w = np.array(w0, dtype=float)
        for _ in range(max_iter):
            eta = w[0] + w[1] * g
            mu = 1.0 / (1.0 + np.exp(-eta))
            resid = k - n * mu
            grad = np.array([resid.sum(), (g * resid).sum()])
            weight = n * mu * (1.0 - mu)
            wg = weight * g
            h01 = wg.sum()
            hessian = np.array([[weight.sum(), h01], [h01, (wg * g).sum()]])
            try:
                delta = np.linalg.solve(hessian, grad)
            except np.linalg.LinAlgError:
                break
            if not np.all(np.isfinite(delta)):
                break
            w += delta
            if np.max(np.abs(delta)) < tol:
                return -w[1], -w[0]
    raise RuntimeError("Logistic calibration fit did not converge.")


def log_loss(
    gaps: np.ndarray,
    mismatch: np.ndarray,
    scale: float,
    offset: float,
    counts: np.ndarray | None = None,
) -> float:
    """Mean negative log-likelihood of the mismatch labels under (scale, offset)."""
    return log_loss_map(gaps, mismatch, GapCalibration(scale, offset), counts=counts)


def log_loss_map(
    gaps: np.ndarray,
    mismatch: np.ndarray,
    llr_map,
    counts: np.ndarray | None = None,
) -> float:
    """Mean negative log-likelihood of the mismatch labels under a gap->llr map."""
    llr = np.asarray(llr_map(np.asarray(gaps, dtype=float)), dtype=float)
    if counts is None:
        y = np.asarray(mismatch, dtype=bool)
        signed = np.where(y, llr, -llr)
        return float(np.mean(np.logaddexp(0.0, signed)))
    n = np.asarray(counts, dtype=float)
    k = np.asarray(mismatch, dtype=float)
    total = k * np.logaddexp(0.0, llr) + (n - k) * np.logaddexp(0.0, -llr)
    return float(total.sum() / n.sum())
