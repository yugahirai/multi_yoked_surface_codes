import numpy as np

from _calibration import combine_llrs, get_calibration_map
from _gap_samples import get_gap_samples
from ._patch import Patch


class Simulator:
    records: list[dict]
    gap_values: list[float]

    def __init__(self, d: int, p: float = 0.001):
        self.records = []
        self.gap_values = []
        self.d = d
        self.p = p
        self.samples = None
        self._n_samples = 0
        self._gaps = None
        self._rng = np.random.default_rng()

    def _ensure_samples(self) -> None:
        if self.samples is None:
            self.samples = get_gap_samples(self.d, self.p)
            self._n_samples = len(self.samples["gaps"])
            self._gaps = self.samples["gaps"].view(np.ndarray)
            self._llr_map = get_calibration_map(self.d, self.p)

    def add_record(self, record: dict):
        self.records.append(record)

    def get_records(self):
        return self.records

    def initialize(self):
        self.records = []
        self.gap_values = []

    def get_gap_value(self, tick: int) -> float:
        return self.gap_values[tick]

    def measure_patches(
        self, patches: list[Patch], distance: int, measure_error=True
    ) -> float:
        record = False
        for patch in patches:
            record ^= patch.get_actual_obs() ^ patch.get_predicted_obs()

        if distance > 0 and measure_error:
            self._ensure_samples()
            idx = self._rng.integers(self._n_samples, size=distance)
            llrs = self._llr_map(self._gaps[idx])
            gap = float(combine_llrs(llrs))
        else:
            gap = float("inf")
        if measure_error:
            self.gap_values.append(gap)
        self.records.append(record)
        if measure_error:
            return gap
