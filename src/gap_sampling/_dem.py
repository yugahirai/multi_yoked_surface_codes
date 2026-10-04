import stim
import pymatching


class Dem:
    def __init__(self, dem: stim.DetectorErrorModel):
        self.dem = dem.flattened()
        self.membrane_dem = stim.DetectorErrorModel()
        self.detector_set = set()
        self.error_list = []
        self.detector_values = []
        self.obs_values_predicted = []
        self.obs_values = []
        self.parse_dem()

    def get_dem(self):
        return self.dem

    def parse_dem(self):
        for inst in self.dem:
            if inst.type == "error":
                self.error_list.append(inst)
                for t in inst.targets_copy():
                    if t.is_relative_detector_id():
                        self.detector_set.add(t)

    def get_detector_set(self):
        return self.detector_set

    def get_error_list(self):
        return self.error_list

    def decode(self, syndrome):
        matcher = pymatching.Matching.from_detector_error_model(
            self.dem, enable_correlations=True
        )
        observable, weight = matcher.decode(
            syndrome, return_weight=True, enable_correlations=True
        )
        self.obs_values_predicted = observable.tolist()
        return observable, weight

    def decode_membrane(self, syndrome_membrane):
        matcher = pymatching.Matching.from_detector_error_model(
            self.membrane_dem, enable_correlations=True
        )
        observable, weight = matcher.decode(
            syndrome_membrane, return_weight=True, enable_correlations=True
        )
        return observable, weight

    def construct_dem(self) -> stim.DetectorErrorModel:
        membrane_dem = stim.DetectorErrorModel()
        for detector in sorted(self.detector_set, key=lambda t: t.val):
            membrane_dem.append(stim.DemInstruction("detector", targets=[detector]))
        for inst in self.error_list:
            membrane_dem.append(inst)
        self.membrane_dem = membrane_dem.flattened()
        return self.membrane_dem

    def set_membranes(self, l_indices: list[int]):
        base_detector_id = max((t.val for t in self.detector_set), default=-1) + 1
        l_to_detector = {l: base_detector_id + i for i, l in enumerate(l_indices)}

        for detector_id in l_to_detector.values():
            self.detector_set.add(stim.target_relative_detector_id(detector_id))

        new_error_list = []
        for error in self.error_list:
            new_targets = []
            for t in error.targets_copy():
                if t.is_logical_observable_id() and t.val in l_to_detector:
                    new_targets.append(
                        stim.target_relative_detector_id(l_to_detector[t.val])
                    )
                else:
                    new_targets.append(t)
            new_error_list.append(
                stim.DemInstruction(
                    error.type,
                    error.args_copy(),
                    new_targets,
                    tag=error.tag,
                )
            )
        self.error_list = new_error_list
        self.membrane_dem = self.construct_dem()
        return self.membrane_dem, l_to_detector
