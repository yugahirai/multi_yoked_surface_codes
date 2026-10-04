"""Correlated matching reweighting for Stim detector error models."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import stim

BOUNDARY: None = None
EdgeKey = tuple[int, int | None]


def normalize_edge(u: int, v: int | None) -> EdgeKey:
    """Return a canonical (min, max) edge key; ``None`` marks the boundary."""
    if v is None or v < 0:
        return (u, BOUNDARY)
    if u <= v:
        return (u, v)
    return (v, u)


def to_weight(probability: float) -> float:
    """Convert an error probability to a matching-graph edge weight."""
    if not 0.0 < probability < 1.0:
        raise ValueError(f"Probability must be in (0, 1), got {probability}")
    return math.log((1.0 - probability) / probability)


def to_probability(weight: float) -> float:
    """Convert a matching-graph edge weight back to an error probability."""
    return 1.0 / (1.0 + math.exp(weight))


def merge_weights(a: float, b: float) -> float:
    """Merge two independent edge weights (PyMatching INDEPENDENT merge rule)."""
    sign = math.copysign(1.0, a) * math.copysign(1.0, b)
    signed_min = sign * min(abs(a), abs(b))
    return (
        signed_min
        + math.log(1.0 + math.exp(-abs(a + b)))
        - math.log(1.0 + math.exp(-abs(a - b)))
    )


def bernoulli_xor(p1: float, p2: float) -> float:
    """Probability that exactly one of two independent Bernoulli events occurs."""
    return p1 * (1.0 - p2) + p2 * (1.0 - p1)


@dataclass
class GraphEdge:
    """A matching-graph edge extracted from a detector error model."""

    detectors: EdgeKey
    observables: tuple[int, ...]
    weight: float
    probability: float
    implied_weights: list[tuple[EdgeKey, float]] = field(default_factory=list)


@dataclass
class CorrelationGraph:
    """Matching graph plus correlation reweight rules derived from a DEM."""

    edges: dict[EdgeKey, GraphEdge]
    detector_ids: set[int]

    def reweight_for_edges(
        self,
        first_pass_edges: list[EdgeKey] | list[tuple[int, int | None]],
    ) -> dict[EdgeKey, GraphEdge]:
        """Return edge table with correlated reweights applied (does not mutate)."""
        updated = {
            key: GraphEdge(
                detectors=edge.detectors,
                observables=edge.observables,
                weight=edge.weight,
                probability=edge.probability,
                implied_weights=list(edge.implied_weights),
            )
            for key, edge in self.edges.items()
        }
        for raw in first_pass_edges:
            u, v = raw
            causal = normalize_edge(u, v)
            source = self.edges.get(causal)
            if source is None:
                continue
            for affected_key, implied_weight in source.implied_weights:
                affected = updated.get(affected_key)
                if affected is None:
                    continue
                if implied_weight < affected.weight:
                    affected.weight = implied_weight
                    affected.probability = to_probability(implied_weight)
        return updated


def _edge_key_from_detectors(detectors: list[int]) -> EdgeKey:
    if len(detectors) == 1:
        return (detectors[0], BOUNDARY)
    if len(detectors) == 2:
        return normalize_edge(detectors[0], detectors[1])
    raise ValueError(f"Expected 1 or 2 detectors, got {detectors}")


def _merge_edge(
    edges: dict[EdgeKey, GraphEdge],
    detectors: EdgeKey,
    observables: tuple[int, ...],
    probability: float,
) -> None:
    weight = to_weight(probability)
    if detectors not in edges:
        edges[detectors] = GraphEdge(
            detectors=detectors,
            observables=observables,
            weight=weight,
            probability=probability,
        )
        return

    existing = edges[detectors]
    new_weight = merge_weights(weight, existing.weight)
    new_probability = bernoulli_xor(probability, existing.probability)
    existing.weight = new_weight
    existing.probability = new_probability


def _decompose_error_instruction(
    instruction: stim.DemInstruction,
) -> list[tuple[list[int], tuple[int, ...]]]:
    """Split a (possibly decomposed) error instruction into graphlike components."""
    if instruction.type != "error":
        raise ValueError(f"Expected an error instruction, got {instruction.type}")

    components: list[tuple[list[int], tuple[int, ...]]] = []
    detectors: list[int] = []
    observables: list[int] = []

    for target in instruction.targets_copy():
        if target.is_relative_detector_id():
            detectors.append(target.val)
            if len(detectors) > 2:
                raise ValueError(
                    "Encountered an undecomposed error with 3+ detectors. "
                    "Use decompose_errors=True when building the DEM."
                )
        elif target.is_logical_observable_id():
            observables.append(target.val)
        elif target.is_separator():
            if not detectors:
                raise ValueError(
                    "Decomposed error contains an undetectable component (0 detectors)."
                )
            components.append((detectors, tuple(observables)))
            detectors = []
            observables = []
        else:
            raise ValueError(f"Unsupported DEM target: {target}")

    if not detectors and components:
        raise ValueError(
            "Decomposed error contains an undetectable component (0 detectors)."
        )
    if detectors:
        components.append((detectors, tuple(observables)))

    return components


def _add_decomposed_error_to_joint_probabilities(
    probability: float,
    components: list[tuple[list[int], tuple[int, ...]]],
    joint_probabilities: dict[EdgeKey, dict[EdgeKey, float]],
) -> None:
    edge_keys = [_edge_key_from_detectors(dets) for dets, _ in components]

    if len(edge_keys) > 1:
        for i, e0 in enumerate(edge_keys):
            for e1 in edge_keys[i + 1 :]:
                joint_probabilities.setdefault(e0, {})[e1] = bernoulli_xor(
                    joint_probabilities.get(e0, {}).get(e1, 0.0), probability
                )
                joint_probabilities.setdefault(e1, {})[e0] = bernoulli_xor(
                    joint_probabilities.get(e1, {}).get(e0, 0.0), probability
                )

    for edge_key in edge_keys:
        inner = joint_probabilities.setdefault(edge_key, {})
        inner[edge_key] = bernoulli_xor(inner.get(edge_key, 0.0), probability)


def _populate_implied_edge_weights(
    edges: dict[EdgeKey, GraphEdge],
    joint_probabilities: dict[EdgeKey, dict[EdgeKey, float]],
) -> None:
    for causal_key, edge in edges.items():
        joint_map = joint_probabilities.get(causal_key)
        if not joint_map:
            continue
        marginal = joint_map.get(causal_key, 0.0)
        if marginal == 0.0:
            continue
        for affected_key, joint_p in joint_map.items():
            if affected_key == causal_key:
                continue
            implied_p = min(0.5, joint_p / marginal)
            edge.implied_weights.append((affected_key, to_weight(implied_p)))


def build_correlation_graph(dem: stim.DetectorErrorModel) -> CorrelationGraph:
    """Parse a flattened DEM into a graph with correlation reweight rules."""
    dem = dem.flattened()
    edges: dict[EdgeKey, GraphEdge] = {}
    joint_probabilities: dict[EdgeKey, dict[EdgeKey, float]] = {}
    detector_ids: set[int] = set()

    for instruction in dem:
        if instruction.type == "detector":
            for target in instruction.targets_copy():
                if target.is_relative_detector_id():
                    detector_ids.add(target.val)
            continue
        if instruction.type != "error":
            continue

        probability = instruction.args_copy()[0]
        if probability == 0.0:
            continue
        if probability > 0.5:
            raise ValueError(
                "Errors with probability greater than 0.5 are not supported "
                "with correlated matching."
            )

        components = _decompose_error_instruction(instruction)
        for detectors, observables in components:
            edge_key = _edge_key_from_detectors(detectors)
            for det in detectors:
                detector_ids.add(det)
            _merge_edge(edges, edge_key, observables, probability)

        _add_decomposed_error_to_joint_probabilities(
            probability, components, joint_probabilities
        )

    _populate_implied_edge_weights(edges, joint_probabilities)
    return CorrelationGraph(edges=edges, detector_ids=detector_ids)


def _edge_sort_key(edge_key: EdgeKey) -> tuple[int, int]:
    d0, d1 = edge_key
    return (d0, -1 if d1 is BOUNDARY else d1)


def edges_to_dem(
    edges: dict[EdgeKey, GraphEdge],
    *,
    detector_ids: set[int] | None = None,
) -> stim.DetectorErrorModel:
    """Build a flattened Stim DEM from a matching-graph edge table."""
    if detector_ids is None:
        detector_ids = set()
        for d0, d1 in edges:
            detector_ids.add(d0)
            if d1 is not None:
                detector_ids.add(d1)

    reweighted_dem = stim.DetectorErrorModel()
    for detector_id in sorted(detector_ids):
        reweighted_dem.append(
            stim.DemInstruction(
                "detector",
                [],
                [stim.target_relative_detector_id(detector_id)],
            )
        )

    for (_d0, _d1), edge in sorted(
        edges.items(), key=lambda item: _edge_sort_key(item[0])
    ):
        targets: list[stim.DemTarget] = []
        if edge.detectors[1] is BOUNDARY:
            targets.append(stim.target_relative_detector_id(edge.detectors[0]))
        else:
            d0, d1 = edge.detectors
            assert d1 is not None
            targets.extend(
                [
                    stim.target_relative_detector_id(d0),
                    stim.target_relative_detector_id(d1),
                ]
            )
        for obs in edge.observables:
            targets.append(stim.target_logical_observable_id(obs))
        reweighted_dem.append(stim.DemInstruction("error", [edge.probability], targets))

    return reweighted_dem.flattened()


def first_pass_edges_from_syndrome(
    dem: stim.DetectorErrorModel,
    syndrome: np.ndarray,
    *,
    matcher: "pymatching.Matching | None" = None,
) -> tuple[list[EdgeKey], float]:
    """Run uncorrelated MWPM and return flipped graph edges (PyMatching pass 1)."""
    import pymatching

    if matcher is None:
        matcher = pymatching.Matching.from_detector_error_model(
            dem.flattened(), enable_correlations=True
        )

    edge_array = matcher.decode_to_edges_array(syndrome, enable_correlations=False)
    return [
        normalize_edge(int(u), None if int(v) < 0 else int(v)) for u, v in edge_array
    ], 1.0


def build_matcher_edge_table(
    matcher: "pymatching.Matching",
) -> dict[EdgeKey, dict[str, float | frozenset[int]]]:
    """Snapshot matcher edge weights for later in-place restore."""
    table: dict[EdgeKey, dict[str, float | frozenset[int]]] = {}
    for u, v, data in matcher.edges():
        key = normalize_edge(u, v)
        table[key] = {
            "weight": data["weight"],
            "error_probability": data["error_probability"],
            "fault_ids": frozenset(data["fault_ids"]),
        }
    return table


def build_physical_to_membrane_edge_map(
    edges: dict[EdgeKey, GraphEdge],
    l_to_detector: dict[int, int],
) -> dict[EdgeKey, EdgeKey]:
    """Map physical correlation-graph edges to membrane matcher edges."""
    edge_map: dict[EdgeKey, EdgeKey] = {}
    for key, edge in edges.items():
        d0, d1 = key
        if d1 is BOUNDARY and len(edge.observables) == 1:
            logical = edge.observables[0]
            if logical in l_to_detector:
                edge_map[key] = normalize_edge(d0, l_to_detector[logical])
                continue
        edge_map[key] = key
    return edge_map


def compute_reweighted_tables(
    edges: dict[EdgeKey, GraphEdge],
    first_pass_edges: list[EdgeKey] | list[tuple[int, int | None]],
) -> dict[EdgeKey, float]:
    """Return only edges whose correlated weight is strictly below the base weight."""
    reweighted: dict[EdgeKey, float] = {}
    for raw in first_pass_edges:
        u, v = raw
        causal = normalize_edge(u, v)
        source = edges.get(causal)
        if source is None:
            continue
        for affected_key, implied_weight in source.implied_weights:
            base = edges.get(affected_key)
            if base is None:
                continue
            current = reweighted.get(affected_key, base.weight)
            if implied_weight < current:
                reweighted[affected_key] = implied_weight
    return reweighted


def _set_matcher_edge(
    matcher: "pymatching.Matching",
    edge_key: EdgeKey,
    *,
    weight: float,
    error_probability: float,
    fault_ids: frozenset[int] | set[int],
) -> None:
    u, v = edge_key
    if v is BOUNDARY:
        matcher.add_boundary_edge(
            u,
            fault_ids=fault_ids,
            weight=weight,
            error_probability=error_probability,
            merge_strategy="replace",
        )
        return

    assert v is not None
    matcher.add_edge(
        u,
        v,
        fault_ids=fault_ids,
        weight=weight,
        error_probability=error_probability,
        merge_strategy="replace",
    )


def apply_reweighted_tables_to_matcher(
    matcher: "pymatching.Matching",
    edge_table: dict[EdgeKey, dict[str, float | frozenset[int]]],
    base_edges: dict[EdgeKey, GraphEdge],
    reweighted: dict[EdgeKey, float],
    *,
    edge_map: dict[EdgeKey, EdgeKey] | None = None,
) -> set[EdgeKey]:
    """Apply correlated reweights in-place; return changed matcher edge keys."""
    changed: set[EdgeKey] = set()
    for phys_key, new_weight in reweighted.items():
        edge = base_edges.get(phys_key)
        if edge is None or new_weight >= edge.weight:
            continue

        mem_key = edge_map.get(phys_key, phys_key) if edge_map else phys_key
        original = edge_table.get(mem_key)
        if original is None:
            continue

        _set_matcher_edge(
            matcher,
            mem_key,
            weight=new_weight,
            error_probability=to_probability(new_weight),
            fault_ids=original["fault_ids"],  # type: ignore[arg-type]
        )
        changed.add(mem_key)
    return changed


def reset_matcher_edge_weights(
    matcher: "pymatching.Matching",
    edge_table: dict[EdgeKey, dict[str, float | frozenset[int]]],
    changed: set[EdgeKey],
) -> None:
    """Restore matcher edges changed by ``apply_reweighted_tables_to_matcher``."""
    for key in changed:
        original = edge_table[key]
        _set_matcher_edge(
            matcher,
            key,
            weight=float(original["weight"]),
            error_probability=float(original["error_probability"]),
            fault_ids=original["fault_ids"],  # type: ignore[arg-type]
        )


def reweight_dem(
    dem: stim.DetectorErrorModel,
    first_pass_edges: list[EdgeKey] | list[tuple[int, int]],
) -> stim.DetectorErrorModel:
    """Apply correlated reweighting and return an updated Stim DEM."""
    graph = build_correlation_graph(dem)
    normalized = [
        normalize_edge(u, None if (isinstance(v, int) and v < 0) else v)
        for u, v in first_pass_edges
    ]
    reweighted_edges = graph.reweight_for_edges(normalized)
    return edges_to_dem(reweighted_edges, detector_ids=graph.detector_ids)


class CorrelatedDecoder:
    """Two-pass correlated decoder with explicit reweighted DEM export."""

    def __init__(self, dem: stim.DetectorErrorModel):
        import pymatching

        self.base_dem = dem.flattened()
        self.graph = build_correlation_graph(self.base_dem)
        self.decibels_per_w = 1
        self._matcher = pymatching.Matching.from_detector_error_model(
            self.base_dem, enable_correlations=True
        )
        self._matcher_edges = build_matcher_edge_table(self._matcher)

    def first_pass_edges(self, syndrome: np.ndarray) -> list[EdgeKey]:
        first_edges, decibels_per_w = first_pass_edges_from_syndrome(
            self.base_dem, syndrome, matcher=self._matcher
        )
        self.decibels_per_w = decibels_per_w
        return first_edges

    def reweighted_dem(
        self, first_pass_edges: list[EdgeKey]
    ) -> stim.DetectorErrorModel:
        reweighted_edges = self.graph.reweight_for_edges(first_pass_edges)
        return edges_to_dem(reweighted_edges, detector_ids=self.graph.detector_ids)

    def decode(
        self,
        syndrome: np.ndarray,
        dem: stim.DetectorErrorModel | None = None,
        *,
        first_edges: list[EdgeKey] | None = None,
        reweighted: dict[EdgeKey, float] | None = None,
    ) -> tuple[np.ndarray, float]:
        """Two-pass correlated decode (native PyMatching correlations)."""
        import pymatching

        if dem is not None:
            matcher = pymatching.Matching.from_detector_error_model(dem)
            return matcher.decode(syndrome, return_weight=True)

        _ = first_edges, reweighted
        return self._matcher.decode(
            syndrome, return_weight=True, enable_correlations=True
        )
