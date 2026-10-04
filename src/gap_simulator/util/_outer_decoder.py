import os
import sys
import time
import warnings
from collections import deque

import numpy as np

import stim
import scipy.sparse
import pymatching
from tesseract_decoder import common, tesseract

_SLOW_DECODE_SEC = float(os.environ.get("GAP_SIM_SLOW_DECODE_SEC", "60"))

_BOUND_EPS = 1e-12


TESSERACT_CONFIG = dict(
    det_beam=200,
    beam_climbing=False,
    no_revisit_dets=False,
    pqlimit=10000000000,
)


TESSERACT_FALLBACK_CONFIGS = (
    dict(TESSERACT_CONFIG, beam_climbing=True, det_beam=tesseract.INF_DET_BEAM),
    dict(TESSERACT_CONFIG, det_beam=tesseract.INF_DET_BEAM, no_revisit_dets=False),
)

BOUNDED_TESSERACT_CONFIG = dict(
    TESSERACT_CONFIG,
    det_beam=tesseract.INF_DET_BEAM,
    no_revisit_dets=False,
    pqlimit=10000000,
)

PAIR_CERT_RUNGS = ((4.0, 100000), (8.0, 300000))

PAIR_CERT_MAX_CLASSES = 320
PAIR_CERT_FINAL_PQLIMIT = 300000

PAIR_ISD_ITERS = 120
PAIR_ISD_MAX_CLASSES = 600
PAIR_ISD_WIDEN = (8.0, 12.0, 16.0, 24.0, float("inf"))


_SET_LIKELIHOOD_COST = common.Error.likelihood_cost.fset
_HAS_BULK_COSTS = hasattr(tesseract.TesseractDecoder, "set_likelihood_costs")


_HAS_BATCH_DECODE = hasattr(tesseract.TesseractDecoder, "decode_batch_with_costs")
REBUILD_EVERY_SHOT = True
_WARNED_STALE_HEURISTIC = False


def _write_costs(decoder, errors, costs) -> None:
    """Load per-error likelihood costs into a compiled tesseract decoder."""
    if _HAS_BULK_COSTS:
        decoder.set_likelihood_costs(np.ascontiguousarray(costs, dtype=float))
    else:
        global _WARNED_STALE_HEURISTIC
        if not _WARNED_STALE_HEURISTIC:
            _WARNED_STALE_HEURISTIC = True
            warnings.warn(
                "tesseract module without set_likelihood_costs: per-shot costs "
                "are written error by error and the search heuristic keeps the "
                "compile-time costs, so decoding is suboptimal and depends on "
                "the build shot. Install the patched +shotcosts2 wheel.",
                RuntimeWarning,
            )
        if not isinstance(costs, list):
            costs = np.asarray(costs, dtype=float).tolist()
        deque(map(_SET_LIKELIHOOD_COST, errors, costs), maxlen=0)


class Decoder:
    def __init__(self, matrix: np.ndarray):
        self.matrix = scipy.sparse.csc_matrix(matrix)
        self.gap_list = np.zeros(self.matrix.shape[1], dtype=float)
        self.observables = None
        self.matching = None
        self.predicted_observables = None
        self.dem = stim.DetectorErrorModel()
        self._error_suffixes: list[str] | None = None
        self._detector_block: str | None = None
        self._det_windows: tuple[np.ndarray, np.ndarray] | None = None
        self._tesseract = None
        self._tesseract_errors: list | None = None
        self._reuse_tesseract = True
        self._tesseract_fallbacks: list = [None] * len(TESSERACT_FALLBACK_CONFIGS)
        self._presolve_rng = np.random.default_rng(0xC0FFEE)
        self.low_confidence = False
        self.low_confidence_shots: np.ndarray = np.zeros(0, dtype=bool)
        self.low_confidence_primary_count = 0
        self.low_confidence_final_count = 0
        self.audit_complete = False
        self.audit_shots = 0
        self.audit_disagreements = 0
        self._presolve_l1_rows: np.ndarray | None = None
        self._presolve_supports: list[np.ndarray] | None = None
        self._presolve_obs: tuple[np.ndarray, np.ndarray] | None = None
        self._observable_block: str | None = None
        self.presolve_trivial_shots = 0
        self.presolve_decoded_shots = 0
        self.presolve_fallback_shots = 0
        self.presolve_slack_used: list[float] = []
        self.presolve_kept_cols: list[int] = []
        self.presolve_cert_rung_shots = [0] * (len(PAIR_CERT_RUNGS) + 1)
        self.presolve_uncertified_shots = 0

    def set_gap_list(self, gap_index: int, gap_value: float):
        self.gap_list[gap_index] = gap_value

    def set_observables(self, observables):
        self.observables = scipy.sparse.csr_matrix(observables)
        self._error_suffixes = None
        self._detector_block = None
        self._tesseract = None
        self._tesseract_errors = None
        self._reuse_tesseract = True
        self._tesseract_fallbacks = [None] * len(TESSERACT_FALLBACK_CONFIGS)

    def get_observables(self):
        return self.observables

    def set_matching(self, weights: np.ndarray):
        self.matching = pymatching.Matching.from_check_matrix(
            self.matrix, weights=weights, faults_matrix=self.observables
        )

    def decode(self, syndrome: np.ndarray, return_weight=False):
        self.predicted_observables, weight = self.matching.decode(
            syndrome, return_weight=True
        )
        if return_weight:
            return self.predicted_observables, weight
        else:
            return self.predicted_observables

    def decode_tesseract(self, syndrome: np.ndarray, return_weight=False):
        syndrome = np.asarray(syndrome, dtype=bool)
        num_observables = (
            self.observables.shape[0] if self.observables is not None else 1
        )
        self.low_confidence = False
        if not syndrome.any():
            return np.zeros(num_observables, dtype=bool)

        costs = np.ascontiguousarray(self.gap_list, dtype=float)
        self._stage_times = []
        t_stage = time.perf_counter()
        decoder = self._tesseract
        if decoder is None and self._reuse_tesseract:
            decoder = self._compile_tesseract()
        if decoder is None:
            decoder = self._compile_per_shot(TESSERACT_CONFIG)
            result = decoder.decode(syndrome)
        else:
            _write_costs(decoder, self._tesseract_errors, costs)
            result = decoder.decode(syndrome)
        self._stage_times.append(("primary", time.perf_counter() - t_stage))

        if decoder.low_confidence_flag:
            self.low_confidence_primary_count += 1
            self.low_confidence = True
            result = self._decode_tesseract_fallback(syndrome, costs, result)
            if self.low_confidence:
                self.low_confidence_final_count += 1
        if self.audit_complete:
            complete = self._fallback_decoder(
                len(TESSERACT_FALLBACK_CONFIGS) - 1, costs
            ).decode(syndrome)
            self.audit_shots += 1
            if not np.array_equal(np.asarray(result), np.asarray(complete)):
                self.audit_disagreements += 1
        if _SLOW_DECODE_SEC:
            total = sum(dt for _, dt in self._stage_times)
            if total >= _SLOW_DECODE_SEC:
                breakdown = "  ".join(
                    f"{name} {dt:.1f}s" for name, dt in self._stage_times
                )
                print(
                    f"[slow decode] {total:.1f}s  dets {int(syndrome.sum())}"
                    f"/{syndrome.size}  {breakdown}",
                    file=sys.stderr,
                    flush=True,
                )
        return result

    def decode_tesseract_batch(self, syndromes, costs) -> np.ndarray:
        """Decode many shots, each with its own costs, in one tesseract call."""
        syndromes = np.asarray(syndromes, dtype=bool)
        costs = np.asarray(costs, dtype=float)
        num_shots = syndromes.shape[0]
        if syndromes.ndim != 2 or syndromes.shape[1] != self.matrix.shape[0]:
            raise ValueError(
                f"syndromes must have shape (num_shots, {self.matrix.shape[0]}), "
                f"got {syndromes.shape}"
            )
        if costs.shape != (num_shots, self.matrix.shape[1]):
            raise ValueError(
                f"costs must have shape ({num_shots}, {self.matrix.shape[1]}), "
                f"got {costs.shape}"
            )
        num_observables = (
            self.observables.shape[0] if self.observables is not None else 1
        )
        result = np.zeros((num_shots, num_observables), dtype=bool)
        flags = np.zeros(num_shots, dtype=bool)
        self.low_confidence = False
        fired = np.flatnonzero(syndromes.any(axis=1))
        if fired.size == 0:
            self.low_confidence_shots = flags
            return result

        decoder = self._tesseract
        if decoder is None and self._reuse_tesseract:
            decoder = self._compile_tesseract()
        if decoder is None or not _HAS_BATCH_DECODE or self.audit_complete:
            for i in fired.tolist():
                self.gap_list = costs[i]
                result[i] = np.asarray(self.decode_tesseract(syndromes[i]), dtype=bool)
                flags[i] = self.low_confidence
            self.low_confidence = bool(flags.any())
            self.low_confidence_shots = flags
            return result

        t_batch = time.perf_counter()
        preds, low = decoder.decode_batch_with_costs(
            np.ascontiguousarray(syndromes[fired]), np.ascontiguousarray(costs[fired])
        )
        result[fired] = preds
        batch_seconds = time.perf_counter() - t_batch
        for k in np.flatnonzero(low).tolist():
            i = int(fired[k])
            self.low_confidence_primary_count += 1
            self.low_confidence = True
            self.gap_list = costs[i]
            self._stage_times = [("primary", batch_seconds / fired.size)]
            res = self._decode_tesseract_fallback(
                syndromes[i], np.ascontiguousarray(costs[i]), preds[k]
            )
            result[i] = np.asarray(res, dtype=bool)
            if self.low_confidence:
                self.low_confidence_final_count += 1
                flags[i] = True
        self.low_confidence = bool(flags.any())
        self.low_confidence_shots = flags
        if _SLOW_DECODE_SEC and batch_seconds >= _SLOW_DECODE_SEC:
            print(
                f"[slow decode] batch of {fired.size} shots {batch_seconds:.1f}s",
                file=sys.stderr,
                flush=True,
            )
        return result

    def warm_compile(self) -> None:
        """Compile the primary decoder and every fallback rung from the current gaps."""
        if REBUILD_EVERY_SHOT:
            return
        self._compile_tesseract()
        costs = np.clip(np.asarray(self.gap_list, dtype=float), -300.0, 300.0)
        for rung in range(len(TESSERACT_FALLBACK_CONFIGS)):
            self._fallback_decoder(rung, costs)

    def prepare_l1_presolve(self, l1_patch_supports, num_patches: int) -> None:
        """Precompute the structures for the trivial L1 pre-decode."""
        num_det, num_err = self.matrix.shape
        if num_err % num_patches:
            raise ValueError(
                f"matrix has {num_err} columns, not a multiple of "
                f"num_patches={num_patches}"
            )
        wanted = {frozenset(s) for s in l1_patch_supports}
        csr = self.matrix.tocsr()
        indptr, indices = csr.indptr, csr.indices
        l1_rows: list[int] = []
        supports: list[np.ndarray] = []
        for det in range(num_det):
            cols = indices[indptr[det] : indptr[det + 1]]
            if cols.size == 0:
                continue
            if cols[0] // num_patches != cols[-1] // num_patches:
                continue
            if frozenset((cols % num_patches).tolist()) not in wanted:
                continue
            l1_rows.append(det)
            supports.append(np.asarray(cols, dtype=np.int64).copy())
        coverage = np.zeros(num_err, dtype=np.int8)
        for cols in supports:
            coverage[cols] += 1
        if not np.all(coverage == 1):
            raise ValueError(
                "L1 presolve requires every error column to belong to exactly "
                "one L1 detector; found columns covered "
                f"{sorted(set(coverage.tolist()))} times."
            )
        if min(len(cols) for cols in supports) < 2:
            raise ValueError(
                "L1 presolve needs every block to hold at least 2 columns; "
                "a size-1 block cannot be parity-corrected."
            )
        self._presolve_l1_rows = np.asarray(l1_rows, dtype=np.int64)
        self._presolve_supports = supports
        widths = {len(cols) for cols in supports}
        if len(widths) != 1:
            raise ValueError(
                f"L1 presolve expects uniform block widths, got {sorted(widths)}."
            )
        self._presolve_block_cols = np.stack(supports)
        obs_csc = self.observables.tocsc()
        self._presolve_obs = (obs_csc.indptr, obs_csc.indices)
        self._ensure_dem_fragments()
        self._build_pair_classes()
        self.presolve_trivial_shots = 0
        self.presolve_decoded_shots = 0
        self.presolve_fallback_shots = 0
        self.presolve_slack_used = []
        self.presolve_kept_cols = []
        self.presolve_cert_rung_shots = [0] * (len(PAIR_CERT_RUNGS) + 1)
        self.presolve_uncertified_shots = 0

    def _build_pair_classes(self) -> None:
        """Precompute the within-block pair classes of the residual decode."""
        num_det, num_err = self.matrix.shape
        indptr, indices = self.matrix.indptr, self.matrix.indices
        o_indptr, o_indices = self._presolve_obs
        n_words = (num_det + 63) // 64
        col_masks = np.zeros((num_err, n_words), dtype=np.uint64)
        for c in range(num_err):
            for d in indices[indptr[c] : indptr[c + 1]].tolist():
                col_masks[c, d >> 6] |= np.uint64(1) << np.uint64(d & 63)
        col_obs = [
            frozenset(o_indices[o_indptr[c] : o_indptr[c + 1]].tolist())
            for c in range(num_err)
        ]
        classes: dict = {}
        for block in self._presolve_block_cols:
            cols = block.tolist()
            for a in range(len(cols)):
                for b in range(a + 1, len(cols)):
                    i, j = cols[a], cols[b]
                    mask = col_masks[i] ^ col_masks[j]
                    if not mask.any():
                        continue
                    key = (mask.tobytes(), col_obs[i] ^ col_obs[j])
                    classes.setdefault(key, []).append((i, j))
        keys = sorted(classes.keys(), key=lambda k: (k[0], tuple(sorted(k[1]))))
        pair_i: list[int] = []
        pair_j: list[int] = []
        pair_class: list[int] = []
        masks = []
        suffixes: list[str] = []
        for ci, key in enumerate(keys):
            mask_bytes, obs = key
            for i, j in classes[key]:
                pair_i.append(i)
                pair_j.append(j)
                pair_class.append(ci)
            mask = np.frombuffer(mask_bytes, dtype=np.uint64)
            masks.append(mask)
            dets = []
            for w in range(n_words):
                v = int(mask[w])
                while v:
                    low = v & -v
                    dets.append((w << 6) + low.bit_length() - 1)
                    v ^= low
            parts = [f"D{d}" for d in dets] + [f"L{o}" for o in sorted(obs)]
            suffixes.append(" " + " ".join(parts))
        self._pair_i = np.asarray(pair_i, dtype=np.int64)
        self._pair_j = np.asarray(pair_j, dtype=np.int64)
        self._pair_class = np.asarray(pair_class, dtype=np.int64)
        self._pair_class_masks = (
            np.stack(masks) if masks else np.zeros((0, n_words), dtype=np.uint64)
        )
        self._pair_suffixes = suffixes
        self._pair_num_classes = len(keys)
        self._pair_mask_words = n_words

    def _l1_block_bounds(self, syndrome: np.ndarray, gaps: np.ndarray):
        """Solve the L1 level exactly, block by block."""
        cols = self._presolve_block_cols
        block = gaps[cols]
        parity = syndrome[self._presolve_l1_rows].astype(np.int64)[:, None]

        neg = block < 0.0
        n_neg = neg.sum(axis=1, keepdims=True)
        neg_sum = np.where(neg, block, 0.0).sum(axis=1, keepdims=True)
        absb = np.abs(block)
        two = np.argpartition(absb, 1, axis=1)[:, :2]
        a0 = np.take_along_axis(absb, two[:, :1], axis=1)
        a1 = np.take_along_axis(absb, two[:, 1:2], axis=1)
        swap = a0 > a1
        m1 = np.where(swap, two[:, 1:2], two[:, :1])
        small1 = np.minimum(a0, a1)
        small2 = np.maximum(a0, a1)

        need_fix = (n_neg & 1) != parity
        base = neg_sum + np.where(need_fix, small1, 0.0)
        chosen = neg.copy()
        flip = np.zeros_like(chosen)
        np.put_along_axis(flip, m1, need_fix, axis=1)
        chosen ^= flip

        with_c = np.where(neg, neg_sum, neg_sum + block)
        card = np.where(neg, n_neg, n_neg + 1)
        is_m1 = np.zeros_like(neg)
        np.put_along_axis(is_m1, m1, True, axis=1)
        fix = np.where(is_m1, small2, small1)
        cost_c = with_c + np.where((card & 1) != parity, fix, 0.0)

        excess = np.empty_like(gaps)
        excess[cols] = cost_c - base
        np.maximum(excess, 0.0, out=excess)
        return float(base.sum()), excess, cols[chosen]

    def _signature(self, cols: np.ndarray):
        """XOR the detector and observable signatures of ``cols``."""
        dets = np.zeros(self.matrix.shape[0], dtype=bool)
        obs = np.zeros(self.observables.shape[0], dtype=bool)
        m_indptr, m_indices = self.matrix.indptr, self.matrix.indices
        o_indptr, o_indices = self._presolve_obs
        for c in cols.tolist():
            dets[m_indices[m_indptr[c] : m_indptr[c + 1]]] ^= True
            obs[o_indices[o_indptr[c] : o_indptr[c + 1]]] ^= True
        return dets, obs

    def _l1_block_structure(self, l1_patch_supports, num_patches: int):
        """Recognize the L1 detector rows and their error-column blocks."""
        num_det, num_err = self.matrix.shape
        if num_err % num_patches:
            raise ValueError(
                f"matrix has {num_err} columns, not a multiple of "
                f"num_patches={num_patches}"
            )
        wanted = {frozenset(s) for s in l1_patch_supports}
        csr = self.matrix.tocsr()
        indptr, indices = csr.indptr, csr.indices
        l1_rows: list[int] = []
        supports: list[np.ndarray] = []
        for det in range(num_det):
            cols = indices[indptr[det] : indptr[det + 1]]
            if cols.size == 0:
                continue
            if cols[0] // num_patches != cols[-1] // num_patches:
                continue
            if frozenset((cols % num_patches).tolist()) not in wanted:
                continue
            l1_rows.append(det)
            supports.append(np.asarray(cols, dtype=np.int64).copy())
        coverage = np.zeros(num_err, dtype=np.int8)
        for cols in supports:
            coverage[cols] += 1
        if not np.all(coverage == 1):
            raise ValueError(
                "L1 blocks must cover every error column exactly once; found "
                f"columns covered {sorted(set(coverage.tolist()))} times."
            )
        widths = {len(cols) for cols in supports}
        if len(widths) != 1 or min(widths) < 2:
            raise ValueError(
                f"L1 blocks must have one uniform width >= 2, got {sorted(widths)}."
            )
        return np.asarray(l1_rows, dtype=np.int64), np.stack(supports)

    def prepare_l1_trivial(self, l1_patch_supports, num_patches: int) -> None:
        """Precompute the L1 block structure for ``decode_tesseract_trivial_first``."""
        l1_rows, block_cols = self._l1_block_structure(l1_patch_supports, num_patches)
        self._trivial_l1_rows = l1_rows
        self._trivial_block_cols = block_cols
        obs_csc = self.observables.tocsc()
        self._trivial_obs = (obs_csc.indptr, obs_csc.indices)
        self.trivial_first_shots = 0
        self.trivial_first_decoded_shots = 0

    def decode_tesseract_trivial_first(self, syndrome: np.ndarray):
        """Plain tesseract decode with a certified-optimal shortcut in front."""
        if getattr(self, "_trivial_l1_rows", None) is None:
            raise RuntimeError("call prepare_l1_trivial() first")
        syndrome = np.asarray(syndrome, dtype=bool)
        self.low_confidence = False
        obs = self._trivial_first_shortcut(
            syndrome, np.asarray(self.gap_list, dtype=float)
        )
        if obs is not None:
            self.trivial_first_shots += 1
            return obs
        self.trivial_first_decoded_shots += 1
        return self.decode_tesseract(syndrome)

    def _trivial_first_shortcut(self, syndrome: np.ndarray, gaps: np.ndarray):
        """The certified answer of ``decode_tesseract_trivial_first``, or None."""
        num_observables = self.observables.shape[0]
        if not syndrome.any():
            return np.zeros(num_observables, dtype=bool)
        fired = np.flatnonzero(syndrome[self._trivial_l1_rows])
        if fired.size and gaps.min() > 0.0:
            blocks = self._trivial_block_cols[fired]
            pick = np.argmin(gaps[blocks], axis=1)
            base_cols = blocks[np.arange(fired.size), pick]
            dets = np.zeros(self.matrix.shape[0], dtype=bool)
            obs = np.zeros(num_observables, dtype=bool)
            m_indptr, m_indices = self.matrix.indptr, self.matrix.indices
            o_indptr, o_indices = self._trivial_obs
            for c in base_cols.tolist():
                dets[m_indices[m_indptr[c] : m_indptr[c + 1]]] ^= True
                obs[o_indices[o_indptr[c] : o_indptr[c + 1]]] ^= True
            if np.array_equal(dets, syndrome):
                return obs
        return None

    def decode_tesseract_trivial_first_batch(self, syndromes, costs) -> np.ndarray:
        """``decode_tesseract_trivial_first`` over a batch of shots."""
        if getattr(self, "_trivial_l1_rows", None) is None:
            raise RuntimeError("call prepare_l1_trivial() first")
        syndromes = np.asarray(syndromes, dtype=bool)
        costs = np.asarray(costs, dtype=float)
        num_shots = syndromes.shape[0]
        result = np.zeros((num_shots, self.observables.shape[0]), dtype=bool)
        rest = []
        for i in range(num_shots):
            obs = self._trivial_first_shortcut(syndromes[i], costs[i])
            if obs is None:
                rest.append(i)
            else:
                result[i] = obs
        self.trivial_first_shots += num_shots - len(rest)
        self.trivial_first_decoded_shots += len(rest)
        flags = np.zeros(num_shots, dtype=bool)
        self.low_confidence = False
        if rest:
            rest = np.asarray(rest)
            result[rest] = self.decode_tesseract_batch(syndromes[rest], costs[rest])
            flags[rest] = self.low_confidence_shots
        self.low_confidence_shots = flags
        self.low_confidence = bool(flags.any())
        return result

    def _compile_pair_dem(self, kept: np.ndarray, costs: np.ndarray, pqlimit: int):
        """Compile a complete-search decoder over the kept pair classes."""
        probs = 1.0 / (1.0 + np.exp(np.clip(costs, -300.0, 300.0)))
        suffixes = self._pair_suffixes
        body = "\n".join(
            [self._detector_block, self._observable_block]
            + [
                f"error({p}){suffixes[ci]}"
                for p, ci in zip(probs.tolist(), kept.tolist())
            ]
        )
        config = tesseract.TesseractConfig(
            dem=stim.DetectorErrorModel(body),
            det_orders=self._det_orders(),
            **dict(BOUNDED_TESSERACT_CONFIG, pqlimit=pqlimit),
        )
        return config.compile_decoder()

    def _pair_class_costs(self, t_cost: np.ndarray):
        """Per-shot cost and cheapest member of every pair class."""
        pcost = t_cost[self._pair_i] + t_cost[self._pair_j]
        order = np.lexsort((pcost, self._pair_class))
        cls_sorted = self._pair_class[order]
        first = np.searchsorted(
            cls_sorted, np.arange(self._pair_num_classes), side="left"
        )
        members = order[first]
        return pcost[members], members

    def _gf2_feasible(self, kept: np.ndarray, r_mask: np.ndarray) -> bool:
        """Can the kept classes produce the residual syndrome at all?"""
        A = self._pair_class_masks[kept].copy()
        r = r_mask.copy()
        used = np.zeros_like(r_mask)
        for k in range(len(A)):
            v = A[k]
            free = v & ~used
            if not free.any():
                continue
            w = int(np.flatnonzero(free)[0])
            b = np.uint64(int(free[w]) & -int(free[w]))
            hit = (A[:, w] & b) != 0
            hit[k] = False
            if hit.any():
                A[hit] ^= v
            if r[w] & b:
                r = r ^ v
            used[w] |= b
        return not r.any()

    def _pair_isd(self, class_cost: np.ndarray, r_mask: np.ndarray):
        """Randomized cost-biased information-set decoding on the classes."""
        n_words = self._pair_mask_words
        best_cost = np.inf
        best_sol = None
        for level in PAIR_ISD_WIDEN:
            kept = np.flatnonzero(class_cost <= level)
            if kept.size == 0:
                continue
            iters = PAIR_ISD_ITERS
            if np.isinf(level):
                iters = max(6, PAIR_ISD_ITERS // 6)
            elif kept.size > PAIR_ISD_MAX_CLASSES:
                kept = kept[
                    np.argsort(class_cost[kept], kind="stable")[:PAIR_ISD_MAX_CLASSES]
                ]
            kcost = class_cost[kept]
            kmasks = self._pair_class_masks[kept]
            scale = 0.3 + 0.2 * float(np.median(kcost))
            n_kept = kept.size
            id_words = (n_kept + 63) // 64
            rows = np.arange(n_kept)
            found = False
            for it in range(iters):
                if it == 0:
                    order = np.argsort(kcost, kind="stable")
                else:
                    order = np.argsort(
                        kcost + self._presolve_rng.gumbel(size=n_kept) * scale,
                        kind="stable",
                    )
                aug = np.zeros((n_kept, n_words + id_words), dtype=np.uint64)
                aug[:, :n_words] = kmasks[order]
                aug[rows, n_words + (rows >> 6)] |= np.uint64(1) << (rows & 63).astype(
                    np.uint64
                )
                r = r_mask.copy()
                r_comb = np.zeros(id_words, dtype=np.uint64)
                used = np.zeros_like(r_mask)
                for k in range(n_kept):
                    v = aug[k]
                    free = v[:n_words] & ~used
                    if not free.any():
                        continue
                    w = int(np.flatnonzero(free)[0])
                    b = np.uint64(int(free[w]) & -int(free[w]))
                    hit = (aug[:, w] & b) != 0
                    hit[k] = False
                    if hit.any():
                        aug[hit] ^= v
                    if r[w] & b:
                        r = r ^ v[:n_words]
                        r_comb = r_comb ^ v[n_words:]
                    used[w] |= b
                if r.any():
                    continue
                found = True
                picked = []
                for w in range(id_words):
                    v = int(r_comb[w])
                    while v:
                        low = v & -v
                        picked.append((w << 6) + low.bit_length() - 1)
                        v ^= low
                sol = kept[order[np.asarray(picked, dtype=np.int64)]]
                cost = float(class_cost[sol].sum()) if sol.size else 0.0
                if cost < best_cost:
                    best_cost = cost
                    best_sol = sol
            if found:
                break
        return best_cost, best_sol

    def _pair_answer(self, chosen_classes, members, t_cost, r, base_obs):
        """Turn chosen pair classes into (observables, true_cost, clean)."""
        counts = np.zeros(self.matrix.shape[1], dtype=np.int64)
        sel = members[chosen_classes]
        np.add.at(counts, self._pair_i[sel], 1)
        np.add.at(counts, self._pair_j[sel], 1)
        y_cols = np.flatnonzero(counts & 1)
        y_dets, y_obs = self._signature(y_cols)
        if not np.array_equal(y_dets, r):
            return None, np.inf, False
        true_cost = float(t_cost[y_cols].sum())
        clean = bool((counts <= 1).all())
        return base_obs ^ y_obs, true_cost, clean

    def decode_tesseract_l1_presolve(self, syndrome: np.ndarray):
        """Level-by-level decode: L1 in closed form, then the pair residual."""
        if self._presolve_l1_rows is None:
            raise RuntimeError("call prepare_l1_presolve() first")
        syndrome = np.asarray(syndrome, dtype=bool)
        num_observables = self.observables.shape[0]
        self.low_confidence = False
        if not syndrome.any():
            self.presolve_trivial_shots += 1
            return np.zeros(num_observables, dtype=bool)

        gaps = np.asarray(self.gap_list, dtype=float)
        lower, excess, base_cols = self._l1_block_bounds(syndrome, gaps)

        base_dets, base_obs = self._signature(base_cols)
        if np.array_equal(base_dets, syndrome):
            self.presolve_trivial_shots += 1
            return base_obs

        r = base_dets ^ syndrome
        in_base = np.zeros(self.matrix.shape[1], dtype=bool)
        in_base[base_cols] = True
        t_cost = np.where(in_base, -gaps, gaps)
        class_cost, members = self._pair_class_costs(t_cost)
        r_mask = np.zeros(self._pair_mask_words, dtype=np.uint64)
        for d in np.flatnonzero(r).tolist():
            r_mask[d >> 6] |= np.uint64(1) << np.uint64(d & 63)

        stages: list[tuple[str, float]] = []
        best_obs = None
        best_cost = np.inf
        certified = False
        cert_rung = -1

        for rung, (u_limit, pqlimit) in enumerate(PAIR_CERT_RUNGS):
            kept = np.flatnonzero(class_cost <= u_limit + _BOUND_EPS)
            if kept.size == 0 or not self._gf2_feasible(kept, r_mask):
                continue
            t_stage = time.perf_counter()
            pair_dec = self._compile_pair_dem(kept, class_cost[kept], pqlimit)
            errs = pair_dec.decode_to_errors(r)
            stages.append((f"pair{rung}", time.perf_counter() - t_stage))
            if pair_dec.low_confidence_flag:
                continue
            pairsum = pair_dec.cost_from_errors(errs)
            obs, true_cost, clean = self._pair_answer(
                kept[np.asarray(errs, dtype=np.int64)],
                members,
                t_cost,
                r,
                base_obs,
            )
            if obs is None:
                continue
            if true_cost < best_cost:
                best_obs, best_cost = obs, true_cost
            if pairsum <= u_limit + _BOUND_EPS and clean:
                certified = True
                cert_rung = rung
                break

        if not certified:
            t_stage = time.perf_counter()
            isd_cost, isd_sol = self._pair_isd(class_cost, r_mask)
            stages.append(("isd", time.perf_counter() - t_stage))
            if isd_sol is not None and isd_cost < best_cost:
                obs, true_cost, _ = self._pair_answer(
                    isd_sol, members, t_cost, r, base_obs
                )
                if obs is not None and true_cost < best_cost:
                    best_obs, best_cost = obs, true_cost

        if not certified and np.isfinite(best_cost):
            kept = np.flatnonzero(class_cost <= best_cost + _BOUND_EPS)
            u_eff = best_cost
            if kept.size > PAIR_CERT_MAX_CLASSES:
                cheapest = np.argsort(class_cost[kept], kind="stable")[
                    :PAIR_CERT_MAX_CLASSES
                ]
                kept = kept[cheapest]
                u_eff = float(class_cost[kept].max())
            if self._gf2_feasible(kept, r_mask):
                t_stage = time.perf_counter()
                pair_dec = self._compile_pair_dem(
                    kept, class_cost[kept], PAIR_CERT_FINAL_PQLIMIT
                )
                errs = pair_dec.decode_to_errors(r)
                stages.append(("cert", time.perf_counter() - t_stage))
                if not pair_dec.low_confidence_flag:
                    pairsum = pair_dec.cost_from_errors(errs)
                    obs, true_cost, clean = self._pair_answer(
                        kept[np.asarray(errs, dtype=np.int64)],
                        members,
                        t_cost,
                        r,
                        base_obs,
                    )
                    if obs is not None:
                        if true_cost <= best_cost + _BOUND_EPS:
                            best_obs, best_cost = obs, true_cost
                        if clean and pairsum <= u_eff + _BOUND_EPS:
                            certified = True
                            cert_rung = len(PAIR_CERT_RUNGS)

        if best_obs is not None:
            if certified:
                self.presolve_cert_rung_shots[cert_rung] += 1
            else:
                self.presolve_uncertified_shots += 1
                self.low_confidence = True
            self.presolve_decoded_shots += 1
            self.presolve_slack_used.append(best_cost)
            self.presolve_kept_cols.append(int(r.sum()))
            result = best_obs
        else:
            self.presolve_fallback_shots += 1
            result = np.asarray(self.decode_tesseract(syndrome), dtype=bool)
        if _SLOW_DECODE_SEC and stages:
            total = sum(dt for _, dt in stages)
            if total >= _SLOW_DECODE_SEC:
                breakdown = "  ".join(f"{name} {dt:.1f}s" for name, dt in stages)
                print(
                    f"[slow presolve] {total:.1f}s  dets {int(syndrome.sum())}"
                    f"/{syndrome.size}  {breakdown}",
                    file=sys.stderr,
                    flush=True,
                )
        return result

    def _fallback_decoder(self, rung: int, costs):
        """Return the given escalation rung's decoder, loaded with ``costs``."""
        config_kwargs = TESSERACT_FALLBACK_CONFIGS[rung]
        if REBUILD_EVERY_SHOT:
            return self._compile_per_shot(
                config_kwargs, self._det_orders_fallback(), gaps=costs
            )
        entry = self._tesseract_fallbacks[rung]
        if entry is None:
            decoder, errors = self._compile_reusable(
                config_kwargs, self._det_orders_fallback()
            )
            entry = (decoder, errors) if decoder is not None else False
            self._tesseract_fallbacks[rung] = entry
        if entry is False:
            return self._compile_per_shot(config_kwargs, self._det_orders_fallback())
        decoder, errors = entry
        _write_costs(decoder, errors, costs)
        return decoder

    def _decode_tesseract_fallback(self, syndrome, costs, result):
        """Re-decode a low-confidence shot with the escalation ladder."""
        for rung in range(len(TESSERACT_FALLBACK_CONFIGS)):
            t_stage = time.perf_counter()
            decoder = self._fallback_decoder(rung, costs)
            result = decoder.decode(syndrome)
            self._stage_times.append((f"rung{rung}", time.perf_counter() - t_stage))
            if not decoder.low_confidence_flag:
                self.low_confidence = False
                return result
        return result

    def _compile_per_shot(self, config_kwargs, det_orders=None, gaps=None):
        """Compile a throwaway decoder whose DEM carries this shot's gaps
        (``gaps``, or ``self.gap_list`` when not given)."""
        if gaps is None:
            gaps = self.gap_list
        gaps = np.clip(np.asarray(gaps, dtype=float), -300.0, 300.0)
        config = tesseract.TesseractConfig(
            dem=self.construct_dem(gaps),
            det_orders=det_orders if det_orders is not None else self._det_orders(),
            **config_kwargs,
        )
        return config.compile_decoder()

    def _compile_tesseract(self):
        """Compile the tesseract decoder once, using the current gaps as base."""
        if REBUILD_EVERY_SHOT:
            self._reuse_tesseract = False
            return None
        decoder, errors = self._compile_reusable(TESSERACT_CONFIG)
        if decoder is None:
            self._reuse_tesseract = False
            warnings.warn(
                "tesseract error list does not match the check matrix columns; "
                "falling back to rebuilding the decoder every shot.",
                RuntimeWarning,
            )
            return None
        self._tesseract = decoder
        self._tesseract_errors = errors
        return decoder

    def _compile_reusable(self, config_kwargs, det_orders=None):
        """Compile a decoder whose error list mirrors the check-matrix columns."""
        base_gaps = np.clip(np.asarray(self.gap_list, dtype=float), -300.0, 300.0)
        config = tesseract.TesseractConfig(
            dem=self.construct_dem(base_gaps),
            det_orders=det_orders if det_orders is not None else self._det_orders(),
            **config_kwargs,
        )
        decoder = config.compile_decoder()
        errors = decoder.errors
        if not self._errors_match_columns(errors):
            return None, None
        return decoder, errors

    def _detector_windows(self) -> tuple[np.ndarray, np.ndarray]:
        """Per-detector support window in error-column (time) space."""
        if self._det_windows is None:
            csr = self.matrix.tocsr()
            indptr, indices = csr.indptr, csr.indices
            num_det = csr.shape[0]
            starts = np.zeros(num_det, dtype=np.int64)
            ends = np.zeros(num_det, dtype=np.int64)
            for det in range(num_det):
                row = indices[indptr[det] : indptr[det + 1]]
                if row.size:
                    starts[det] = row[0]
                    ends[det] = row[-1]
            self._det_windows = (starts, ends)
        return self._det_windows

    def _det_orders(self) -> list[list[int]]:
        """Single increasing-index detector order for the A* search."""
        return [list(range(self.matrix.shape[0]))]

    def _det_orders_fallback(self) -> list[list[int]]:
        """Detector orders for the low-confidence escalation rungs."""
        starts, ends = self._detector_windows()
        wide_first = sorted(
            range(self.matrix.shape[0]),
            key=lambda det: (starts[det], starts[det] - ends[det], det),
        )
        return self._det_orders() + [wide_first]

    def _errors_match_columns(self, errors) -> bool:
        if len(errors) != self.matrix.shape[1]:
            return False
        indptr = self.matrix.indptr
        indices = self.matrix.indices
        obs = self.observables
        if obs is not None:
            obs_csc = obs.tocsc()
            obs_indptr, obs_indices_arr = obs_csc.indptr, obs_csc.indices
        for e, error in enumerate(errors):
            dets = sorted(int(d) for d in indices[indptr[e] : indptr[e + 1]])
            if sorted(error.symptom.detectors) != dets:
                return False
            if obs is not None:
                obs_indices = sorted(
                    int(o) for o in obs_indices_arr[obs_indptr[e] : obs_indptr[e + 1]]
                )
                if sorted(error.symptom.observables) != obs_indices:
                    return False
        return True

    def _ensure_dem_fragments(self) -> None:
        if self._error_suffixes is not None:
            return

        num_detectors = self.matrix.shape[0]
        num_errors = self.matrix.shape[1]
        self._detector_block = "\n".join(
            f"detector D{det}" for det in range(num_detectors)
        )
        num_obs = self.observables.shape[0] if self.observables is not None else 0
        self._observable_block = "\n".join(
            f"logical_observable L{o}" for o in range(num_obs)
        )

        suffixes: list[str] = []
        indptr = self.matrix.indptr
        indices = self.matrix.indices
        obs = self.observables
        if obs is not None:
            obs_csc = obs.tocsc()
            obs_indptr, obs_indices = obs_csc.indptr, obs_csc.indices
        for e in range(num_errors):
            parts = [f"D{int(d)}" for d in indices[indptr[e] : indptr[e + 1]]]
            if obs is not None:
                parts.extend(
                    f"L{int(o)}" for o in obs_indices[obs_indptr[e] : obs_indptr[e + 1]]
                )
            suffixes.append((" " + " ".join(parts)) if parts else "")
        self._error_suffixes = suffixes

    def construct_dem(self, gap_list=None):
        self._ensure_dem_fragments()
        if gap_list is None:
            gap_list = self.gap_list
        probs = 1.0 / (1.0 + np.exp(np.asarray(gap_list, dtype=float)))
        suffixes = self._error_suffixes
        body = "\n".join(
            [self._detector_block]
            + [f"error({p}){s}" for p, s in zip(probs.tolist(), suffixes)]
        )
        self.dem = stim.DetectorErrorModel(body)
        return self.dem


class RoundDecoder:
    """Decode a block-diagonal check matrix one outer round at a time."""

    def __init__(self, matrix, observables, col_blocks) -> None:
        matrix = scipy.sparse.csr_matrix(matrix)
        observables = scipy.sparse.csr_matrix(observables)
        num_det, num_err = matrix.shape
        col_blocks = [int(n) for n in col_blocks]
        col_bounds = np.cumsum([0] + col_blocks)
        if col_bounds[-1] != num_err:
            raise ValueError(
                f"col_blocks sum to {col_bounds[-1]} columns, matrix has {num_err}"
            )
        if observables.shape[1] != num_err:
            raise ValueError("observables must have one column per error column")

        indptr, indices = matrix.indptr, matrix.indices
        row_block = np.full(num_det, -1, dtype=np.int64)
        for det in range(num_det):
            cols = indices[indptr[det] : indptr[det + 1]]
            if cols.size == 0:
                raise ValueError(f"detector {det} has no error column")
            first = int(np.searchsorted(col_bounds, cols.min(), side="right")) - 1
            last = int(np.searchsorted(col_bounds, cols.max(), side="right")) - 1
            if first != last:
                raise ValueError(
                    f"detector {det} couples to columns of blocks {first} and "
                    f"{last}: the matrix is not block diagonal per round"
                )
            row_block[det] = first

        self.matrix = scipy.sparse.csc_matrix(matrix)
        self.observables = observables
        self.num_observables = observables.shape[0]
        self.gap_list = np.zeros(num_err, dtype=float)
        self.blocks: list[tuple] = []
        self.decoders: list[Decoder] = []
        structures: dict = {}
        for b, (c0, c1) in enumerate(zip(col_bounds[:-1], col_bounds[1:])):
            rows = np.flatnonzero(row_block == b)
            if rows.size == 0:
                raise ValueError(f"block {b} has no detector")
            col_slice = slice(int(c0), int(c1))
            sub = matrix[rows][:, col_slice]
            sub_obs = observables[:, col_slice]
            key = (
                sub.shape,
                sub.indptr.tobytes(),
                sub.indices.tobytes(),
                sub_obs.indptr.tobytes(),
                sub_obs.indices.tobytes(),
            )
            dec_idx = structures.get(key)
            if dec_idx is None:
                decoder = Decoder(sub)
                decoder.set_observables(sub_obs)
                dec_idx = len(self.decoders)
                self.decoders.append(decoder)
                structures[key] = dec_idx
            contiguous = rows[-1] - rows[0] + 1 == rows.size
            row_sel = slice(int(rows[0]), int(rows[-1]) + 1) if contiguous else rows
            self.blocks.append((row_sel, col_slice, dec_idx))
        self.low_confidence = False
        self.low_confidence_shots: np.ndarray = np.zeros(0, dtype=bool)
        self._lazy_l1: tuple | None = None

    @property
    def low_confidence_primary_count(self) -> int:
        return sum(d.low_confidence_primary_count for d in self.decoders)

    @property
    def low_confidence_final_count(self) -> int:
        return sum(d.low_confidence_final_count for d in self.decoders)

    def _first_gaps(self, dec_idx: int) -> np.ndarray:
        """Costs of the first block decoded by decoder ``dec_idx``."""
        gaps = np.asarray(self.gap_list, dtype=float)
        for _, col_slice, idx in self.blocks:
            if idx == dec_idx:
                return gaps[col_slice]
        raise RuntimeError("decoder without a block")

    def _ensure_dem_fragments(self) -> None:
        for decoder in self.decoders:
            decoder._ensure_dem_fragments()

    def warm_compile(self) -> None:
        """Compile every block decoder from the current (whole-shot) gaps."""
        for idx, decoder in enumerate(self.decoders):
            decoder.gap_list = self._first_gaps(idx)
            decoder.warm_compile()

    def prepare_l1_presolve(self, l1_patch_supports, num_patches: int) -> None:
        for decoder in self.decoders:
            decoder.prepare_l1_presolve(l1_patch_supports, num_patches)

    def prepare_l1_trivial(self, l1_patch_supports, num_patches: int) -> None:
        for decoder in self.decoders:
            decoder.prepare_l1_trivial(l1_patch_supports, num_patches)
        num_det = self.matrix.shape[0]
        all_rows = np.arange(num_det)
        rows: list[np.ndarray] = []
        cols: list[np.ndarray] = []
        rounds: list[np.ndarray] = []
        for r, (row_sel, col_slice, dec_idx) in enumerate(self.blocks):
            decoder = self.decoders[dec_idx]
            rows.append(all_rows[row_sel][decoder._trivial_l1_rows])
            cols.append(decoder._trivial_block_cols + col_slice.start)
            rounds.append(np.full(len(decoder._trivial_l1_rows), r, dtype=np.int64))
        obs_csc = self.observables.tocsc()
        self._lazy_l1 = (
            np.concatenate(rows),
            np.concatenate(cols),
            np.concatenate(rounds),
            (self.matrix.indptr, self.matrix.indices),
            (obs_csc.indptr, obs_csc.indices),
        )

    def _decode_blocks(self, method: str, syndrome) -> np.ndarray:
        syndrome = np.asarray(syndrome, dtype=bool)
        if syndrome.size != self.matrix.shape[0]:
            raise ValueError(
                f"syndrome has {syndrome.size} detectors, matrix {self.matrix.shape[0]}"
            )
        gaps = np.asarray(self.gap_list, dtype=float)
        obs = np.zeros(self.num_observables, dtype=bool)
        self.low_confidence = False
        for row_sel, col_slice, dec_idx in self.blocks:
            s = syndrome[row_sel]
            if not s.any():
                continue
            decoder = self.decoders[dec_idx]
            decoder.gap_list = gaps[col_slice]
            obs ^= np.asarray(getattr(decoder, method)(s), dtype=bool)
            self.low_confidence |= decoder.low_confidence
        return obs

    def decode_lazy(
        self, method: str, syndrome, costs_of, closed_form: bool = False
    ) -> np.ndarray:
        """Per-round decode that fetches its costs on demand."""
        syndrome = np.asarray(syndrome, dtype=bool)
        if syndrome.size != self.matrix.shape[0]:
            raise ValueError(
                f"syndrome has {syndrome.size} detectors, matrix {self.matrix.shape[0]}"
            )
        obs = np.zeros(self.num_observables, dtype=bool)
        self.low_confidence = False
        if not syndrome.any():
            return obs
        if closed_form and method != "decode_tesseract_trivial_first":
            raise ValueError(
                "closed_form needs method='decode_tesseract_trivial_first'"
            )
        base = None
        if closed_form:
            if self._lazy_l1 is None:
                raise RuntimeError("call prepare_l1_trivial() first")
            (
                l1_rows,
                block_cols,
                block_round,
                (m_indptr, m_indices),
                (o_indptr, o_indices),
            ) = self._lazy_l1
            fired = np.flatnonzero(syndrome[l1_rows])
            if fired.size:
                blocks = block_cols[fired]
                costs = np.asarray(costs_of(blocks.ravel()), dtype=float).reshape(
                    blocks.shape
                )
                if costs.min() > 0.0:
                    base = blocks[np.arange(fired.size), np.argmin(costs, axis=1)]
                    base_round = block_round[fired]
                    dets = np.zeros(syndrome.size, dtype=bool)
                    for c in base.tolist():
                        dets[m_indices[m_indptr[c] : m_indptr[c + 1]]] ^= True
        for r, (row_sel, col_slice, dec_idx) in enumerate(self.blocks):
            s = syndrome[row_sel]
            if not s.any():
                continue
            decoder = self.decoders[dec_idx]
            if base is not None and np.array_equal(dets[row_sel], s):
                decoder.low_confidence = False
                decoder.trivial_first_shots += 1
                for c in base[base_round == r].tolist():
                    obs[o_indices[o_indptr[c] : o_indptr[c + 1]]] ^= True
                continue
            decoder.gap_list = np.asarray(
                costs_of(np.arange(col_slice.start, col_slice.stop)), dtype=float
            )
            obs ^= np.asarray(getattr(decoder, method)(s), dtype=bool)
            self.low_confidence |= decoder.low_confidence
        return obs

    def _decode_blocks_batch(self, method: str, syndromes, costs) -> np.ndarray:
        """Per-round decode of a batch of shots (see ``Decoder.decode_tesseract_batch``)."""
        syndromes = np.asarray(syndromes, dtype=bool)
        costs = np.asarray(costs, dtype=float)
        num_shots = syndromes.shape[0]
        if syndromes.ndim != 2 or syndromes.shape[1] != self.matrix.shape[0]:
            raise ValueError(
                f"syndromes must have shape (num_shots, {self.matrix.shape[0]}), "
                f"got {syndromes.shape}"
            )
        if costs.shape != (num_shots, self.matrix.shape[1]):
            raise ValueError(
                f"costs must have shape ({num_shots}, {self.matrix.shape[1]}), "
                f"got {costs.shape}"
            )
        obs = np.zeros((num_shots, self.num_observables), dtype=bool)
        flags = np.zeros(num_shots, dtype=bool)
        for row_sel, col_slice, dec_idx in self.blocks:
            s = syndromes[:, row_sel]
            fired = np.flatnonzero(s.any(axis=1))
            if fired.size == 0:
                continue
            decoder = self.decoders[dec_idx]
            obs[fired] ^= getattr(decoder, method)(s[fired], costs[fired, col_slice])
            flags[fired] |= decoder.low_confidence_shots
        self.low_confidence_shots = flags
        self.low_confidence = bool(flags.any())
        return obs

    def decode_tesseract(self, syndrome) -> np.ndarray:
        return self._decode_blocks("decode_tesseract", syndrome)

    def decode_tesseract_l1_presolve(self, syndrome) -> np.ndarray:
        return self._decode_blocks("decode_tesseract_l1_presolve", syndrome)

    def decode_tesseract_trivial_first(self, syndrome) -> np.ndarray:
        return self._decode_blocks("decode_tesseract_trivial_first", syndrome)

    def decode_tesseract_batch(self, syndromes, costs) -> np.ndarray:
        return self._decode_blocks_batch("decode_tesseract_batch", syndromes, costs)

    def decode_tesseract_trivial_first_batch(self, syndromes, costs) -> np.ndarray:
        return self._decode_blocks_batch(
            "decode_tesseract_trivial_first_batch", syndromes, costs
        )
