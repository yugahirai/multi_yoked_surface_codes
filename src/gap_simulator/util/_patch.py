import numpy as np

from _calibration import combine_llrs, get_calibration_map
from _gap_samples import get_gap_samples


class Patch:
    gap_values_map: dict[int, float]

    def __init__(self, d: int, p: float = 0.001):
        self.d = d
        self.p = p
        self.actual_obs = False
        self.predicted_obs = False
        self._tick = 0
        self.gap_values_map: dict[int, float] = {}
        self.actual_observable_map: dict[int, bool] = {}
        self.predicted_observable_map: dict[int, int] = {}
        self._samples = get_gap_samples(d, p)
        self._llr_map = get_calibration_map(d, p)
        self._bind_samples()
        self._rng = np.random.default_rng()

    def _bind_samples(self) -> None:
        """(Re)bind the shared sample arrays; call after materialize_gap_samples."""
        samples = self._samples
        self._n_samples = len(samples["gaps"])
        self._gaps = samples["gaps"].view(np.ndarray)
        self._actual = samples["actual"].view(np.ndarray)
        self._predicted = samples["predicted"].view(np.ndarray)
        self._packed = samples.get("packed")
        self._mismatch_buckets = samples.get("mismatch_buckets")

    def initialize(self):
        self.actual_obs = False
        self.predicted_obs = False
        self._tick = 0
        self.gap_values_map = {}
        self.actual_observable_map = {}
        self.predicted_observable_map = {}

    def get_actual_obs(self) -> bool:
        return self.actual_obs

    def get_predicted_obs(self) -> bool:
        return self.predicted_obs

    def get_tick(self) -> int:
        return self._tick

    def get_gap_value(self, tick: int) -> float:
        return self.gap_values_map[tick]

    def get_actual_observable(self, tick: int) -> list[bool]:
        return [self.actual_observable_map[tick]]

    def get_predicted_observable(self, tick: int) -> list[int]:
        return [self.predicted_observable_map[tick]]

    def tick(self) -> dict:
        i = int(self._rng.integers(self._n_samples))
        tick = self._tick
        gap = float(self._llr_map(self._gaps[i]))
        actual = bool(self._actual[i])
        predicted = bool(self._predicted[i])

        self.actual_obs ^= actual
        self.predicted_obs ^= predicted
        self.gap_values_map[tick] = gap
        self.actual_observable_map[tick] = actual
        self.predicted_observable_map[tick] = predicted
        self._tick += 1
        return {
            "tick": tick,
            "gap": gap,
            "actual_observable": [actual],
            "predicted_observable": [predicted],
        }

    def idle_ticks(self, n: int) -> float:
        """Apply n idling ticks; return the interval's combined llr."""
        if n <= 0:
            return float("inf")
        idx = self._rng.integers(self._n_samples, size=n)
        self.actual_obs ^= bool(np.count_nonzero(self._actual[idx]) & 1)
        self.predicted_obs ^= bool(np.count_nonzero(self._predicted[idx]) & 1)
        self._tick += n
        llrs = self._llr_map(self._gaps[idx])
        return float(combine_llrs(llrs))

    def set_gap_value(self, tick: int, value: float):
        self.gap_values_map[tick] = value
