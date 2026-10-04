from collections import deque

from _calibration import combine_llrs
from _gap_samples import MISMATCH_BUCKET_SHIFT, materialize_gap_samples
from ._patch import Patch
from ._simulator import Simulator
import numpy as np
from . import _const_matrix
from ._format_matrix import format_matrix


class Circuit:
    circuit_matrix: np.ndarray
    patches: list[Patch]

    def __init__(
        self,
        d: int,
        elementary_matrix: list[list[int]],
        elementary_observables: list[list[int]],
        num_patches: int,
    ):
        self.d = d
        self.circuit_matrix: np.ndarray
        self.patches = [Patch(d) for _ in range(num_patches)]
        self.simulator = Simulator(d)
        self.elementary_matrix = np.array(elementary_matrix)
        self.elementary_observables = np.array(elementary_observables)
        self.observables = np.array(elementary_observables)
        self._rng = np.random.default_rng()
        self._idle_cache = None
        self._idle_plan = None
        self._plan_ticks = 0
        self._actual_vec = None
        self._predicted_vec = None
        self._plan_specs: dict = {}
        self._record_blocks: list = []

    def get_circuit_matrix(
        self, elementary_matrix: list[list[int]], dem_matrix: list[list[int]]
    ) -> list[list[int]]:
        col_width = len(elementary_matrix[0]) + 1
        num_cols = len(dem_matrix[0]) - 1
        return [
            [row[c : min(c + col_width, num_cols)] for row in dem_matrix]
            for c in range(0, num_cols, col_width)
        ]

    def initialize(self) -> None:
        for patch in self.patches:
            patch.initialize()
        self.simulator.initialize()
        self.circuit_matrix = None
        self.observables = np.array(self.elementary_observables)
        self._idle_plan = None
        self._plan_ticks = 0
        self._actual_vec = None
        self._predicted_vec = None
        self._record_blocks = []

    def materialize_samples(self) -> None:
        """Pin the patches' gap sample set in RAM (see materialize_gap_samples)."""
        seen = set()
        for patch in self.patches:
            key = (patch.d, patch.p)
            if key not in seen:
                materialize_gap_samples(*key)
                seen.add(key)
            patch._bind_samples()
        self._idle_cache = None

    def _idle_arrays(self):
        cache = self._idle_cache
        if cache is None:
            p0 = self.patches[0]
            cache = (
                p0._gaps,
                p0._actual,
                p0._predicted,
                p0._n_samples,
                p0._llr_map,
                p0._packed,
            )
            self._idle_cache = cache
        return cache

    def mismatch_buckets(self):
        """``(table, shift)`` of the samples' mismatch bucket table, or None."""
        table = getattr(self.patches[0], "_mismatch_buckets", None)
        if table is None:
            return None
        return table, MISMATCH_BUCKET_SHIFT

    @staticmethod
    def _gather(idx, gaps_arr, actual_arr, predicted_arr, packed):
        """Fetch the samples at idx as (gaps, actual, predicted) arrays."""
        if packed is not None:
            rec = packed[idx]
            return rec["gaps"], rec["actual"], rec["predicted"]
        return gaps_arr[idx], actual_arr[idx], predicted_arr[idx]

    def _sync_patches_from_vec(self) -> None:
        """Write the mirror arrays back into the Patch attributes."""
        if self._actual_vec is None:
            return
        a_list = self._actual_vec.tolist()
        p_list = self._predicted_vec.tolist()
        for patch, a, p in zip(self.patches, a_list, p_list):
            patch.actual_obs = a
            patch.predicted_obs = p

    def _flush_idle_plan(self) -> None:
        """Drop the current plan, settling the deferred per-patch state."""
        self._sync_patches_from_vec()
        if self._plan_ticks:
            n = self._plan_ticks
            for patch in self.patches:
                patch._tick += n
            self._plan_ticks = 0
        self._idle_plan = None

    def plan_idling(self, lengths) -> None:
        """Pre-draw every idling interval of the coming shot in one batch."""
        if self._idle_plan is not None:
            self._flush_idle_plan()
        patches = self.patches
        num_patches = len(patches)
        gaps_arr, actual_arr, predicted_arr, n_samples, llr_map, packed = (
            self._idle_arrays()
        )

        self._actual_vec = np.fromiter(
            (p.actual_obs for p in patches), dtype=bool, count=num_patches
        )
        self._predicted_vec = np.fromiter(
            (p.predicted_obs for p in patches), dtype=bool, count=num_patches
        )

        key = tuple(int(n) for n in lengths)
        spec = self._plan_specs.get(key)
        if spec is None:
            if any(n <= 0 for n in key):
                raise ValueError("plan_idling intervals must be positive.")
            offsets = np.concatenate(([0], np.cumsum(key)[:-1])).astype(np.int64)
            by_len: dict[int, list[int]] = {}
            for pos, n in enumerate(key):
                by_len.setdefault(n, []).append(pos)
            groups = []
            for n, positions in by_len.items():
                cols = offsets[positions][:, None] + np.arange(n, dtype=np.int64)
                groups.append((n, positions, cols))
            spec = (int(sum(key)), groups)
            self._plan_specs[key] = spec
        total, groups = spec

        idx = self._rng.integers(n_samples, size=(num_patches, total))
        g_all, a_all, p_all = self._gather(
            idx, gaps_arr, actual_arr, predicted_arr, packed
        )
        results: list = [None] * len(key)
        for n, positions, cols in groups:
            g_sel = g_all[:, cols].transpose(1, 0, 2)
            gap_llr = combine_llrs(llr_map(g_sel))
            apar = (np.count_nonzero(a_all[:, cols], axis=2).T & 1).astype(bool)
            ppar = (np.count_nonzero(p_all[:, cols], axis=2).T & 1).astype(bool)
            for row, pos in enumerate(positions):
                results[pos] = (n, gap_llr[row], apar[row], ppar[row])
        self._plan_ticks = 0
        self._idle_plan = deque(results)

    def idling(self, t_interval: int, as_array: bool = False):
        """Idle every patch for t_interval ticks; return the interval llrs."""
        patches = self.patches
        n = t_interval
        if n <= 0:
            if as_array:
                return np.full(len(patches), np.inf)
            return [float("inf")] * len(patches)
        plan = self._idle_plan
        if plan is not None:
            if plan and plan[0][0] == n:
                _, gap_llr, apar, ppar = plan.popleft()
                self._actual_vec ^= apar
                self._predicted_vec ^= ppar
                self._plan_ticks += n
                if not plan:
                    self._flush_idle_plan()
                return gap_llr if as_array else gap_llr.tolist()
            self._flush_idle_plan()
        gaps_arr, actual_arr, predicted_arr, n_samples, llr_map, packed = (
            self._idle_arrays()
        )

        idx = self._rng.integers(n_samples, size=(len(patches), n))
        g_sel, a_sel, p_sel = self._gather(
            idx, gaps_arr, actual_arr, predicted_arr, packed
        )
        gap_llr = combine_llrs(llr_map(g_sel))
        actual_par = np.count_nonzero(a_sel, axis=1) & 1
        predicted_par = np.count_nonzero(p_sel, axis=1) & 1
        if self._actual_vec is not None:
            self._actual_vec ^= actual_par.astype(bool)
            self._predicted_vec ^= predicted_par.astype(bool)
        for i in np.flatnonzero(actual_par).tolist():
            patch = patches[i]
            patch.actual_obs = not patch.actual_obs
        for i in np.flatnonzero(predicted_par).tolist():
            patch = patches[i]
            patch.predicted_obs = not patch.predicted_obs
        for patch in patches:
            patch._tick += n
        return gap_llr if as_array else gap_llr.tolist()

    def measure_patches(
        self, check_row: list[int], level: int, measure_error=True
    ) -> float:
        if self._idle_plan is not None:
            self._sync_patches_from_vec()
        num_patches = len(self.patches)
        qubits_to_measure = check_row[:num_patches]
        cache = getattr(self, "_measure_cache", None)
        if cache is None:
            cache = {}
            self._measure_cache = cache
        key = (id(check_row), level, num_patches)
        cached = cache.get(key)
        if cached is None:
            patch_indices = tuple(i for i, v in enumerate(qubits_to_measure) if v == 1)
            if not patch_indices:
                cached = (0, ())
            else:
                distance = patch_indices[-1] - patch_indices[0] + 1
                cached = (distance, patch_indices)
            cache[key] = cached
        distance, patch_indices = cached
        if not patch_indices:
            return self.simulator.measure_patches([], 0)
        patches_to_measure = [self.patches[i] for i in patch_indices]
        return self.simulator.measure_patches(
            patches_to_measure, distance, measure_error
        )

    def measure_patch_block(self, elementary_matrix) -> None:
        """Append the records of every row of ``elementary_matrix`` at once."""
        cache = getattr(self, "_block_cache", None)
        if cache is None:
            cache = {}
            self._block_cache = cache
        key = id(elementary_matrix)
        block = cache.get(key)
        if block is None:
            block = (
                np.asarray(elementary_matrix)[:, : len(self.patches)] != 0
            ).astype(np.uint8)
            cache[key] = block
        if self._actual_vec is not None:
            err = (self._actual_vec ^ self._predicted_vec).view(np.uint8)
        else:
            err = np.fromiter(
                (p.actual_obs != p.predicted_obs for p in self.patches),
                dtype=np.uint8,
                count=len(self.patches),
            )
        rec = (block @ err) & 1
        self._record_blocks.append(rec)
        self.simulator.records.extend(rec.astype(bool).tolist())

    def get_record(self, tick: int) -> int:
        records = self.simulator.records
        return records[len(records) + tick]

    def record_array(self, sentinel: bool = False) -> np.ndarray:
        """All records of this shot as a uint8 array, in get_record order."""
        blocks = self._record_blocks
        if sentinel:
            blocks = blocks + [np.zeros(1, dtype=np.uint8)]
        if not blocks:
            return np.zeros(0, dtype=np.uint8)
        return np.concatenate(blocks)

    def get_patch_error_vec(self) -> np.ndarray:
        """actual_obs ^ predicted_obs of every patch as a uint8 vector."""
        if self._actual_vec is not None:
            return (self._actual_vec ^ self._predicted_vec).view(np.uint8)
        return np.fromiter(
            (p.actual_obs != p.predicted_obs for p in self.patches),
            dtype=np.uint8,
            count=len(self.patches),
        )

    def get_patch_observables(self) -> list[bool]:
        if self._actual_vec is not None:
            return (self._actual_vec ^ self._predicted_vec).tolist()
        return [
            patch.get_actual_obs() ^ patch.get_predicted_obs() for patch in self.patches
        ]

    def extend_circuit_matrix(
        self,
        initialize: bool = False,
        only_level_1_checks: bool = False,
        num_l1_checks: int = 0,
        tick_0: bool = False,
        last_check: bool = False,
        first_check: bool = False,
        l1_check_rate: int = 0,
        l2_check_rate: int = 0,
        only_l1=False,
        measure_error=True,
        elementary_matrix=None,
        elementary_observables=None,
        L1_elementary_matrix=None,
        L2_elementary_matrix=None,
        L3_elementary_matrix=None,
    ):
        if measure_error:
            if only_l1:
                self.circuit_matrix, self.observables = (
                    _const_matrix.extend_circuit_matrix_with_only_l1(
                        self.circuit_matrix,
                        self.observables,
                        elementary_matrix=self.elementary_matrix,
                        elementary_observables=self.elementary_observables,
                        num_l1_checks=num_l1_checks,
                    )
                )

            else:
                if initialize:
                    self.circuit_matrix = self.elementary_matrix[:num_l1_checks]
                    self.observables = self.elementary_observables

                elif last_check:
                    self.circuit_matrix, self.observables = (
                        _const_matrix.extend_circuit_matrix_with_last_check(
                            self.circuit_matrix,
                            self.observables,
                            elementary_matrix=self.elementary_matrix,
                            elementary_observables=self.elementary_observables,
                            num_l1_checks=num_l1_checks,
                        )
                    )
                elif only_level_1_checks:
                    self.circuit_matrix, self.observables = (
                        _const_matrix.extend_circuit_matrix_with_level_1_checks(
                            self.circuit_matrix,
                            self.observables,
                            elementary_matrix=self.elementary_matrix,
                            elementary_observables=self.elementary_observables,
                            num_l1_checks=num_l1_checks,
                            first_check=first_check,
                        )
                    )
                elif not only_level_1_checks:
                    self.circuit_matrix, self.observables = (
                        _const_matrix.extend_circuit_matrix_with_all(
                            self.circuit_matrix,
                            self.observables,
                            elementary_matrix=self.elementary_matrix,
                            elementary_observables=self.elementary_observables,
                            num_l1_checks=num_l1_checks,
                            tick_0=tick_0,
                            l1_check_rate=l1_check_rate,
                        )
                    )
        else:
            if L2_elementary_matrix is not None and L3_elementary_matrix is None:
                self.circuit_matrix, self.observables = (
                    _const_matrix.extend_circuit_matrix_with_all_without_measure_error(
                        self.circuit_matrix,
                        self.observables,
                        l1_check_rate=l1_check_rate,
                        elementary_matrix=elementary_matrix,
                        elementary_observables=self.elementary_observables,
                        l1_elementary_matrix=L1_elementary_matrix,
                        l2_elementary_matrix=L2_elementary_matrix,
                    )
                )
            elif L3_elementary_matrix is not None:
                self.circuit_matrix, self.observables = (
                    _const_matrix.extend_circuit_matrix_with_all_without_measure_error_for_three_yoke(
                        self.circuit_matrix,
                        self.observables,
                        l1_check_rate=l1_check_rate,
                        l2_check_rate=l2_check_rate,
                        elementary_matrix=elementary_matrix,
                        elementary_observables=self.elementary_observables,
                        l1_elementary_matrix=L1_elementary_matrix,
                        l2_elementary_matrix=L2_elementary_matrix,
                        l3_elementary_matrix=L3_elementary_matrix,
                    )
                )
            else:
                self.circuit_matrix, self.observables = (
                    _const_matrix.extend_circuit_matrix_without_measure_error(
                        self.circuit_matrix,
                        self.observables,
                        elementary_matrix=elementary_matrix,
                        elementary_observables=self.elementary_observables,
                    )
                )
