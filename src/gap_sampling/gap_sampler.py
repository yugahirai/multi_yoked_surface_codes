"""Multiprocessed gap sampling with cached correlated decoders."""

from __future__ import annotations

import shutil
import signal
import time
from multiprocessing import get_context
from os import cpu_count
from pathlib import Path

import numpy as np
import pymatching
import stim
from tqdm import tqdm

REPO_DIR = Path(__file__).resolve().parents[2]

from _correlated_decoder import (
    BOUNDARY,
    CorrelatedDecoder,
    build_matcher_edge_table,
    build_physical_to_membrane_edge_map,
    normalize_edge,
)
from _dem import Dem
from _gap_samples import GapCsvWriter, _gap_for_shot

D = 7
P = 0.001
NUM_SHOTS = 10000000
CHUNK_SIZE = 100
N_PROCESSES = None
CIRCUIT_PATH = REPO_DIR / f"circuits/SI1000/circuit_12dxdxd_d{D}_p{P}.stim"
CSV_PATH = REPO_DIR / f"data/sampled_gap/gap_d{D}_p{P}.csv"


def _fast_legacy_available() -> bool:
    """True when pymatching exposes decode_with_reweight_edges."""
    import pymatching._cpp_pymatching as _cpp

    return hasattr(_cpp.MatchingGraph, "decode_with_reweight_edges")


_SHARED: dict = {}
_WORKER: dict = {}


def _build_shared(circuit_path: str, l_indices: list[int]) -> dict:
    circuit = stim.Circuit.from_file(circuit_path)
    base_dem = Dem(circuit.detector_error_model(decompose_errors=True))
    membrane_dem, l_to_detector = base_dem.set_membranes(l_indices)

    shared: dict = {
        "circuit": circuit,
        "l_to_detector": l_to_detector,
    }

    decoder = CorrelatedDecoder(base_dem.dem)
    shared["decoder"] = decoder
    shared["decibels_per_w"] = decoder.decibels_per_w
    physical_to_membrane = build_physical_to_membrane_edge_map(
        decoder.graph.edges,
        l_to_detector,
    )
    shared["physical_to_membrane"] = physical_to_membrane
    shared["fast_legacy"] = _fast_legacy_available()
    if shared["fast_legacy"]:
        membrane_decoder = pymatching.Matching.from_detector_error_model(
            membrane_dem, enable_correlations=True
        )
        shared["membrane_decoder"] = membrane_decoder
        membrane_keys = {normalize_edge(u, v) for u, v, _ in membrane_decoder.edges()}
        translate: dict = {}
        for key in decoder.graph.edges:
            mem_key = physical_to_membrane.get(key, key)
            if mem_key in membrane_keys:
                u, v = mem_key
                pair = (u, -1 if v is BOUNDARY else v)
                pu, pv = key
                if pv is BOUNDARY:
                    translate[(pu, -1)] = pair
                else:
                    translate[(pu, pv)] = pair
                    translate[(pv, pu)] = pair
        stride = decoder._matcher.num_nodes + 2
        codes = np.fromiter(
            (u * stride + v + 1 for u, v in translate),
            dtype=np.int64,
            count=len(translate),
        )
        pairs = np.array(list(translate.values()), dtype=np.int64)
        order = np.argsort(codes)
        shared["fp_translate_stride"] = stride
        shared["fp_translate_codes"] = codes[order]
        shared["fp_translate_pairs"] = pairs[order]
    else:
        membrane_decoder = pymatching.Matching.from_detector_error_model(membrane_dem)
        shared["membrane_decoder"] = membrane_decoder
        shared["membrane_matcher_edges"] = build_matcher_edge_table(membrane_decoder)

    return shared


def _init_worker(l_indices: list[int], membrane_sets: list[list[int]]) -> None:
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    _WORKER.update(_SHARED)
    _WORKER["sampler"] = _SHARED["circuit"].compile_detector_sampler()
    _WORKER["l_indices"] = l_indices
    _WORKER["membrane_sets"] = membrane_sets
    _WORKER["syndrome_membrane"] = np.zeros(
        _SHARED["membrane_decoder"].num_detectors, dtype=bool
    )


def _run_chunk_legacy_fast(n_shots: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Legacy gaps via the patched decode_with_reweight_edges (no rebuilds)."""
    sampler = _WORKER["sampler"]
    decoder = _WORKER["decoder"]
    decibels_per_w = _WORKER["decibels_per_w"]
    physical_graph = decoder._matcher._matching_graph
    membrane_graph = _WORKER["membrane_decoder"]._matching_graph
    stride = _WORKER["fp_translate_stride"]
    translate_codes = _WORKER["fp_translate_codes"]
    translate_pairs = _WORKER["fp_translate_pairs"]
    l_indices = _WORKER["l_indices"]
    membrane_sets = _WORKER["membrane_sets"]
    l_to_detector = _WORKER["l_to_detector"]

    syndromes, observables_actual = sampler.sample(n_shots, separate_observables=True)

    ca = l_to_detector[membrane_sets[0][0]]
    cb = l_to_detector[membrane_sets[0][1]]

    gaps = np.empty(n_shots, dtype=np.float64)
    predicted = np.empty(n_shots, dtype=bool)
    empty_reweights = np.empty(0, dtype=np.int64)
    last_code = translate_codes.size - 1
    zero_gap = None

    for i in range(n_shots):
        s = syndromes[i]
        detection_events = np.flatnonzero(s).astype(np.uint64)
        if detection_events.size == 0:
            if zero_gap is None:
                mem_dets = np.array(sorted((ca, cb)), dtype=np.uint64)
                _, w0 = membrane_graph.decode_with_reweight_edges(
                    mem_dets, empty_reweights
                )
                zero_gap = w0 * decibels_per_w
            gaps[i] = zero_gap
            predicted[i] = False
            continue

        first_edges = physical_graph.decode_to_edges_array(detection_events)
        pred, initial_weight = physical_graph.decode_with_reweight_edges(
            detection_events, first_edges.ravel()
        )

        codes = first_edges[:, 0] * stride + first_edges[:, 1] + 1
        idx = np.minimum(np.searchsorted(translate_codes, codes), last_code)
        reweight_edges = translate_pairs[idx[translate_codes[idx] == codes]].ravel()

        membrane_bits = {l_to_detector[l]: bool(pred[l]) for l in l_indices}
        membrane_bits[ca] = not membrane_bits.get(ca, False)
        membrane_bits[cb] = not membrane_bits.get(cb, False)
        extra = sorted(d for d, bit in membrane_bits.items() if bit)
        mem_dets = np.concatenate(
            (detection_events, np.asarray(extra, dtype=np.uint64))
        )
        _, weight = membrane_graph.decode_with_reweight_edges(mem_dets, reweight_edges)

        gaps[i] = (weight - initial_weight) * decibels_per_w
        predicted[i] = bool(pred[0])

    return gaps, observables_actual[:, 0].copy(), predicted


def _run_chunk_legacy(n_shots: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sampler = _WORKER["sampler"]
    decoder = _WORKER["decoder"]
    decibels_per_w = _WORKER["decibels_per_w"]
    membrane_decoder = _WORKER["membrane_decoder"]
    membrane_matcher_edges = _WORKER["membrane_matcher_edges"]
    physical_to_membrane = _WORKER["physical_to_membrane"]
    syndrome_membrane = _WORKER["syndrome_membrane"]
    l_indices = _WORKER["l_indices"]
    membrane_sets = _WORKER["membrane_sets"]
    l_to_detector = _WORKER["l_to_detector"]

    syndromes, observables_actual = sampler.sample(n_shots, separate_observables=True)
    num_physical_detectors = syndromes.shape[1]

    gaps = np.empty(n_shots, dtype=np.float64)
    predicted = np.empty(n_shots, dtype=bool)

    for i in range(n_shots):
        pred, _, gap_list = _gap_for_shot(
            syndromes[i],
            decoder=decoder,
            membrane_decoder=membrane_decoder,
            membrane_matcher_edges=membrane_matcher_edges,
            physical_to_membrane=physical_to_membrane,
            syndrome_membrane=syndrome_membrane,
            l_indices=l_indices,
            membrane_sets=membrane_sets,
            l_to_detector=l_to_detector,
            num_physical_detectors=num_physical_detectors,
        )
        gaps[i] = gap_list[0] * decibels_per_w
        predicted[i] = bool(pred[0])

    return gaps, observables_actual[:, 0].copy(), predicted


def _run_chunk(n_shots: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if _WORKER.get("fast_legacy"):
        return _run_chunk_legacy_fast(n_shots)
    return _run_chunk_legacy(n_shots)


def _format_progress_stats(
    *,
    completed_shots: int,
    decode_errors: int,
    mean_gap: float,
    n_processes: int,
) -> str:
    decode_rate = decode_errors / completed_shots if completed_shots else 0.0
    return (
        f"procs={n_processes} "
        f"decode={decode_rate:.6g} ({decode_errors} err) "
        f"mean_gap={mean_gap:.4f}"
    )


if __name__ == "__main__":
    circuit_path = CIRCUIT_PATH
    csv_path = CSV_PATH

    n_processes = N_PROCESSES
    if n_processes is None:
        n_processes = max(1, (cpu_count() or 2) - 1)
    n_processes = max(1, min(n_processes, NUM_SHOTS))

    n_full, rem = divmod(NUM_SHOTS, CHUNK_SIZE)
    chunks = [CHUNK_SIZE] * n_full + ([rem] if rem else [])

    l_indices = [0, 1]
    membrane_sets = [[0, 1]]
    start = time.perf_counter()

    build_start = time.perf_counter()
    _SHARED.update(_build_shared(str(circuit_path), l_indices))

    print(
        f"Circuit: {circuit_path.name}\n"
        f"Output:  {csv_path}\n"
        f"Shots:   {NUM_SHOTS:,}  chunk_size={CHUNK_SIZE}  "
        f"chunks={len(chunks)}  processes={n_processes}\n"
        f"Decoder: "
        f"{'legacy reweighting (fast C++ path)' if _SHARED.get('fast_legacy') else 'legacy reweighting (python add_edge path)'}"
        f"  (setup {time.perf_counter() - build_start:.1f}s, shared via fork)"
    )

    completed_shots = 0
    decode_errors = 0
    sum_gap = 0.0
    term_cols = shutil.get_terminal_size(fallback=(120, 24)).columns
    pbar = tqdm(
        total=NUM_SHOTS,
        desc="Sampling gaps",
        unit="shot",
        position=0,
        ncols=term_cols,
        bar_format=(
            "{desc}: {n_fmt}/{total_fmt} ({percentage:5.2f}%)|{bar}| "
            "[{elapsed}<{remaining}, {rate_fmt}]"
        ),
    )
    rates_bar = tqdm(
        total=1,
        bar_format="{desc}",
        position=1,
        ncols=term_cols,
        leave=False,
    )
    rates_bar.set_description(
        _format_progress_stats(
            completed_shots=0,
            decode_errors=0,
            mean_gap=0.0,
            n_processes=n_processes,
        )
    )

    with GapCsvWriter(csv_path) as writer:
        with get_context("fork").Pool(
            processes=n_processes,
            initializer=_init_worker,
            initargs=(l_indices, membrane_sets),
        ) as pool:
            for gaps, actual, predicted in pool.imap_unordered(_run_chunk, chunks):
                writer.append_arrays(
                    d=D,
                    p=P,
                    gaps=gaps,
                    actual=actual,
                    predicted=predicted,
                )
                completed_shots += gaps.shape[0]
                decode_errors += int(np.count_nonzero(actual != predicted))
                sum_gap += float(gaps.sum())
                pbar.update(gaps.shape[0])
                rates_bar.set_description(
                    _format_progress_stats(
                        completed_shots=completed_shots,
                        decode_errors=decode_errors,
                        mean_gap=sum_gap / completed_shots,
                        n_processes=n_processes,
                    )
                )

    pbar.close()
    rates_bar.close()

    elapsed = time.perf_counter() - start
    mean_gap = sum_gap / completed_shots if completed_shots else 0.0
    print(
        f"Wrote {completed_shots} rows to {csv_path} "
        f"in {elapsed:.1f}s ({completed_shots / elapsed:.0f} shots/s), "
        f"{_format_progress_stats(completed_shots=completed_shots, decode_errors=decode_errors, mean_gap=mean_gap, n_processes=n_processes)}"
    )
