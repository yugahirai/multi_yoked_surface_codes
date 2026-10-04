import argparse
import hashlib
import json
import sys
import time
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np
import sinter
from tqdm import tqdm

REPO_DIR = Path(__file__).resolve().parents[2]
sys.path[:0] = [
    str(REPO_DIR / "src" / "gap_sampling"),
    str(REPO_DIR / "src" / "code_maker"),
]

from _calibration import combine_llrs, get_calibration_map
from _extend import read_checks
from _gap_samples import hugepage_coverage
from _qubit_identifier import logical_qubits
from util._circuit import Circuit
from util._outer_decoder import Decoder, RoundDecoder
from util._parse_matrix import parse_matrix

MATRIX_PATH = REPO_DIR / "ecc" / "chain_codes" / "2_yoke" / "q48_30_4.txt"
L1_ELEMENTARY_MATRIX, L2_ELEMENTARY_MATRIX, _ = parse_matrix(MATRIX_PATH, n_l1=6)
OBSERVABLES = [[int(v) for v in logical_qubits(*read_checks(MATRIX_PATH))[0][0]]]
ELEMENTARY_MATRIX = L1_ELEMENTARY_MATRIX + L2_ELEMENTARY_MATRIX

N_L1 = len(L1_ELEMENTARY_MATRIX)
N_L2 = len(L2_ELEMENTARY_MATRIX)
N_ALL = N_L1 + N_L2

CHECK_SUPPORT = [
    tuple(i for i, v in enumerate(row) if v == 1) for row in ELEMENTARY_MATRIX
]
OBSERVABLE_SUPPORT = tuple(i for i, v in enumerate(OBSERVABLES[0]) if v == 1)
NUM_PATCHES = len(ELEMENTARY_MATRIX[0])
CHECK_BLOCK = (np.asarray(ELEMENTARY_MATRIX)[:, :NUM_PATCHES] != 0).astype(np.int32)
OBSERVABLE_VEC = (np.asarray(OBSERVABLES[0])[:NUM_PATCHES] != 0).astype(np.int32)
CHECK_BLOCK_F32 = CHECK_BLOCK.astype(np.float32)


D = 7
P = 0.001
T_INTERVAL = 48
SHOTS = 100000
OUTER_ROUNDS = 11
L1_CHECK_RATE = 2
CHUNK_SIZE = 100
DEFAULT_PROCESSES = 111
BATCH_SHOTS = 0

PROTOCOL = f"two_yoke_{MATRIX_PATH.stem}"

_WORKER: dict = {}


def num_ticks(d_val: int, t_interval_val: int) -> int:
    return (1 + t_interval_val * (OUTER_ROUNDS - 1)) * d_val * 12


def task_strong_id(metadata: dict) -> str:
    payload = json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def write_sinter_stats(csv_path: Path, stats: sinter.TaskStats) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    with open(csv_path, "a") as f:
        if write_header:
            f.write(sinter.CSV_HEADER + "\n")
        f.write(stats.to_csv_line() + "\n")


def existing_errors(csv_path: Path, strong_id: str) -> int:
    """Errors already recorded in ``csv_path`` under ``strong_id`` (0 if none)."""
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return 0
    return sum(
        stat.errors
        for stat in sinter.read_stats_from_csv_files(str(csv_path))
        if stat.strong_id == strong_id
    )


def _metadata(
    d_val: int,
    p_val: float,
    t_interval_val: int,
    l1_check_rate_val: int,
    *,
    level: str,
) -> dict:
    return {
        "protocol": PROTOCOL,
        "d": d_val,
        "p": p_val,
        "t_interval": t_interval_val,
        "l1_check_rate": l1_check_rate_val,
        "outer_rounds": OUTER_ROUNDS,
        "num_ticks": num_ticks(d_val, t_interval_val),
        "level": level,
        "calibration": get_calibration_map(d_val, p_val).label,
    }


def _write_chunk_stats(
    csv_path: Path,
    *,
    d_val: int,
    p_val: float,
    t_interval_val: int,
    l1_check_rate_val: int,
    shots: int,
    inner_errors: int,
    outer_errors: int,
    seconds: float,
) -> None:
    for level, errors in (("inner", inner_errors), ("outer", outer_errors)):
        metadata = _metadata(
            d_val, p_val, t_interval_val, l1_check_rate_val, level=level
        )
        write_sinter_stats(
            csv_path,
            sinter.TaskStats(
                strong_id=task_strong_id(metadata),
                decoder=f"{PROTOCOL}_{level}",
                json_metadata=metadata,
                shots=shots,
                errors=errors,
                discards=0,
                seconds=seconds,
            ),
        )


def default_csv_path() -> Path:
    """One csv for every parameter set: the rows carry the parameters in their metadata."""
    return REPO_DIR / "data" / "simulated" / f"{PROTOCOL}.csv"


def _make_circuit(d_val: int) -> Circuit:
    return Circuit(d_val, ELEMENTARY_MATRIX, OBSERVABLES, num_patches=NUM_PATCHES)


def _syndrome_plan(l1_check_rate: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Index tables that turn a shot's record stream into its syndrome."""
    n_rec = 0
    cur: list[int] = []
    back: list[int] = []

    def emit(count: int, cur_back: int, partner_back: int | None) -> None:
        for r in range(count):
            cur.append(n_rec + r - cur_back)
            back.append(-1 if partner_back is None else n_rec + r - partner_back)

    for tick in range(OUTER_ROUNDS - 1):
        for in_interval in range(l1_check_rate - 1):
            n_rec += N_L1
            if in_interval > 0:
                partner = 2 * N_L1
            elif tick > 0:
                partner = 2 * N_L1 + N_L2
            else:
                partner = None
            emit(N_L1, N_L1, partner)
        n_rec += N_ALL
        if l1_check_rate > 1:
            partner = 2 * N_L1 + N_L2
        elif tick > 0:
            partner = 2 * N_ALL
        else:
            partner = None
        emit(N_L1, N_L1 + N_L2, partner)
        emit(N_L2, N_L2, (l1_check_rate * N_L1 + 2 * N_L2) if tick > 0 else None)
    last = [n_rec - N_ALL + c for c in range(N_ALL)]

    cur_arr = np.asarray(cur, dtype=np.int64)
    back_arr = np.asarray(back, dtype=np.int64)
    last_arr = np.asarray(last, dtype=np.int64)
    if (cur_arr < 0).any() or (back_arr < -1).any() or (last_arr < 0).any():
        raise RuntimeError("syndrome plan references records before the shot")
    back_arr[back_arr < 0] = n_rec
    return cur_arr, back_arr, last_arr


def _shot_tables(t_interval: int, l1_check_rate: int) -> dict:
    """Index tables of the vectorized shot (see _sample_shot_fast)."""
    cur_idx, back_idx, last_idx = _syndrome_plan(l1_check_rate)
    idle_len = t_interval // l1_check_rate
    n_int = l1_check_rate * (OUTER_ROUNDS - 1)
    n_cols = n_int + 1
    rows: list[int] = []
    blocks: list[int] = []
    b = 0
    for tick in range(OUTER_ROUNDS - 1):
        for in_interval in range(l1_check_rate - 1):
            rows.extend(range(N_L1))
            blocks.extend([b] * N_L1)
            b += 1
        rows.extend(range(N_ALL))
        blocks.extend([b] * N_ALL)
        b += 1
    if b != n_int:
        raise RuntimeError("block sequence does not match the idling plan")
    rec_flat = np.asarray(rows, dtype=np.int64) * n_cols + np.asarray(
        blocks, dtype=np.int64
    )
    n_rec = len(rec_flat)
    if (
        (cur_idx >= n_rec).any()
        or (back_idx > n_rec).any()
        or (last_idx >= n_rec).any()
    ):
        raise RuntimeError("syndrome plan references records the block sequence lacks")
    sentinel = N_ALL * n_cols
    rec_ext = np.append(rec_flat, sentinel)
    fold_idx = np.arange(N_ALL, dtype=np.int64) * n_cols + n_int
    left = np.concatenate((rec_ext[cur_idx], rec_ext[last_idx]))
    right = np.concatenate((rec_ext[back_idx], fold_idx))
    return {
        "idle_len": idle_len,
        "n_int": n_int,
        "total_ticks": n_int * idle_len + 1,
        "left": left,
        "right": right,
        "full_ext": np.zeros(sentinel + 1, dtype=np.uint8),
    }


def _build_shared(
    d_val: int,
    t_interval_val: int,
    l1_check_rate_val: int,
    l1_presolve: bool = False,
    ram_samples: bool = True,
    trivial_first: bool = False,
    per_round: bool = True,
    dense_shot: bool = False,
    batch_shots: int = BATCH_SHOTS,
) -> None:
    """Build the shared deterministic state once, in the current process."""
    _WORKER.clear()
    circuit = _make_circuit(d_val)
    if ram_samples:
        circuit.materialize_samples()
    _WORKER["circuit"] = circuit
    _WORKER["t_interval"] = t_interval_val
    _WORKER["l1_check_rate"] = l1_check_rate_val
    _WORKER["l1_presolve"] = l1_presolve
    _WORKER["ram_samples"] = ram_samples
    _WORKER["trivial_first"] = trivial_first
    _WORKER["per_round"] = per_round
    _WORKER["dense_shot"] = dense_shot
    _WORKER["batch_shots"] = batch_shots
    _run_shot(_WORKER["circuit"], t_interval_val, l1_check_rate_val)
    _WORKER["shot_tables"] = _shot_tables(t_interval_val, l1_check_rate_val)
    _WORKER["lazy_tables"] = _lazy_tables(circuit, _WORKER)


def _init_worker(
    worker_d: int,
    worker_t_interval: int,
    worker_l1_check_rate: int,
    worker_l1_presolve: bool = False,
    worker_ram_samples: bool = True,
    worker_trivial_first: bool = False,
    worker_per_round: bool = True,
    worker_dense_shot: bool = False,
    worker_batch_shots: int = BATCH_SHOTS,
) -> None:
    if _WORKER.get("circuit_matrix") is None:
        _build_shared(
            worker_d,
            worker_t_interval,
            worker_l1_check_rate,
            worker_l1_presolve,
            worker_ram_samples,
            worker_trivial_first,
            worker_per_round,
            worker_dense_shot,
            worker_batch_shots,
        )
    circuit = _WORKER["circuit"]
    circuit._rng = np.random.default_rng()
    circuit.simulator._rng = np.random.default_rng()
    for patch in circuit.patches:
        patch._rng = np.random.default_rng()


def _run_shot(
    circuit: Circuit,
    t_interval: int,
    l1_check_rate: int,
    cache: dict | None = None,
) -> tuple[bool, bool]:
    if cache is None:
        cache = _WORKER
    build = cache.get("circuit_matrix") is None
    plan = cache.get("syndrome_plan")
    if plan is None:
        plan = _syndrome_plan(l1_check_rate)
        cache["syndrome_plan"] = plan
    cur_idx, back_idx, last_idx = plan
    gap_rows = []
    idle_len = t_interval // l1_check_rate
    circuit.initialize()
    circuit.plan_idling([idle_len] * (l1_check_rate * (OUTER_ROUNDS - 1)) + [1])
    for tick in range(OUTER_ROUNDS - 1):
        for in_interval in range(l1_check_rate - 1):
            gap_rows.append(circuit.idling(idle_len, as_array=True))
            circuit.measure_patch_block(L1_ELEMENTARY_MATRIX)
            if build:
                circuit.extend_circuit_matrix(
                    measure_error=False, elementary_matrix=L1_ELEMENTARY_MATRIX
                )

        gap_rows.append(circuit.idling(idle_len, as_array=True))
        circuit.measure_patch_block(ELEMENTARY_MATRIX)
        if build:
            circuit.extend_circuit_matrix(
                measure_error=False,
                l1_check_rate=l1_check_rate,
                elementary_matrix=ELEMENTARY_MATRIX,
                L1_elementary_matrix=L1_ELEMENTARY_MATRIX,
                L2_elementary_matrix=L2_ELEMENTARY_MATRIX,
            )

    gap_rows.append(circuit.idling(1, as_array=True))
    if build:
        circuit.extend_circuit_matrix(
            measure_error=False, elementary_matrix=ELEMENTARY_MATRIX
        )

    records = circuit.record_array(sentinel=True)
    err_vec = circuit.get_patch_error_vec()
    fold = ((CHECK_BLOCK @ err_vec) & 1).astype(np.uint8)
    syndrome = np.concatenate(
        (records[cur_idx] ^ records[back_idx], records[last_idx] ^ fold)
    ).astype(bool)
    gap_list = np.concatenate(gap_rows)

    if build:
        cache["circuit_matrix"] = circuit.circuit_matrix
        cache["observables"] = circuit.observables
        decoder = _make_decoder(
            circuit.circuit_matrix,
            circuit.observables,
            l1_check_rate,
            per_round=cache.get("per_round", True),
        )
        decoder._ensure_dem_fragments()
        if cache.get("l1_presolve"):
            decoder.prepare_l1_presolve(
                CHECK_SUPPORT[:N_L1], num_patches=len(circuit.patches)
            )
        if cache.get("trivial_first"):
            decoder.prepare_l1_trivial(
                CHECK_SUPPORT[:N_L1], num_patches=len(circuit.patches)
            )
        decoder.gap_list = gap_list
        decoder.warm_compile()
        cache["decoder"] = decoder

    decoder = cache["decoder"]
    decoder.gap_list = gap_list
    predicted_observables = _decode(decoder, syndrome, cache)

    patch_observable = int(OBSERVABLE_VEC @ err_vec) & 1

    inner_error = (
        circuit.patches[0].get_predicted_obs() != circuit.patches[0].get_actual_obs()
    )
    outer_error = bool(predicted_observables[0]) != bool(patch_observable)
    return inner_error, outer_error


def _make_decoder(circuit_matrix, observables, l1_check_rate: int, *, per_round: bool):
    """The outer decoder of the shot: per-round blocks, or the whole DEM."""
    if not per_round:
        decoder = Decoder(circuit_matrix)
        decoder.set_observables(observables)
        return decoder
    col_blocks = [l1_check_rate * NUM_PATCHES] * (OUTER_ROUNDS - 1) + [NUM_PATCHES]
    return RoundDecoder(circuit_matrix, observables, col_blocks)


def _decode(decoder, syndrome, cache: dict):
    if cache.get("l1_presolve"):
        return decoder.decode_tesseract_l1_presolve(syndrome)
    if cache.get("trivial_first"):
        return decoder.decode_tesseract_trivial_first(syndrome)
    return decoder.decode_tesseract(syndrome)


def _batch_shots(cache: dict) -> int:
    """Shots per decoder call on the dense path, 0 when it decodes shot by shot."""
    if cache.get("l1_presolve") or cache.get("lazy_tables") is not None:
        return 0
    return max(int(cache.get("batch_shots") or 0), 0)


def _decode_batch(decoder, syndromes, costs, cache: dict):
    """``_decode`` over a batch: rows of ``syndromes``/``costs`` are shots."""
    if cache.get("trivial_first"):
        return decoder.decode_tesseract_trivial_first_batch(syndromes, costs)
    return decoder.decode_tesseract_batch(syndromes, costs)


def _run_shot_fast(circuit: Circuit, cache: dict | None = None) -> tuple[bool, bool]:
    """One shot with the same draws, llrs, syndrome and decode as _run_shot."""
    if cache is None:
        cache = _WORKER
    syndrome, gap_list, err_vec = _sample_shot_fast(circuit, cache)
    decoder = cache["decoder"]
    decoder.gap_list = gap_list
    predicted_observables = _decode(decoder, syndrome, cache)

    patch_observable = int(OBSERVABLE_VEC @ err_vec) & 1
    inner_error = bool(err_vec[0])
    outer_error = bool(predicted_observables[0]) != bool(patch_observable)
    return inner_error, outer_error


def _sample_shot_fast(
    circuit: Circuit, cache: dict
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The draws of one _run_shot_fast shot: (syndrome, gap_list, err_vec)."""
    tables = cache["shot_tables"]
    idle_len = tables["idle_len"]
    n_int = tables["n_int"]
    n_idle = n_int * idle_len
    gaps_arr, actual_arr, predicted_arr, n_samples, llr_map, packed = (
        circuit._idle_arrays()
    )
    num_patches = len(circuit.patches)

    idx = circuit._rng.integers(n_samples, size=(num_patches, tables["total_ticks"]))
    if packed is not None:
        rec = np.take(packed.view(np.uint64), idx).view(packed.dtype)
        g_all, a_all, p_all = rec["gaps"], rec["actual"], rec["predicted"]
    else:
        g_all, a_all, p_all = gaps_arr[idx], actual_arr[idx], predicted_arr[idx]

    llr_int = combine_llrs(
        llr_map(g_all[:, :n_idle].reshape(num_patches, n_int, idle_len))
    )
    gap_list = np.concatenate((llr_int.T.ravel(), llr_map(g_all[:, n_idle])))

    x = a_all ^ p_all
    x_int = x[:, :n_idle].reshape(num_patches, n_int, idle_len)
    par = x_int[:, :, 0].copy()
    for j in range(1, idle_len):
        par ^= x_int[:, :, j]
    state = np.empty((num_patches, n_int + 1), dtype=bool)
    state[:, :n_int] = np.bitwise_xor.accumulate(par, axis=1)
    state[:, n_int] = state[:, n_int - 1] ^ x[:, n_idle]

    full_ext = tables["full_ext"]
    full_ext[:-1] = (CHECK_BLOCK_F32 @ state.view(np.uint8).astype(np.float32)).astype(
        np.uint8
    ).ravel() & 1
    syndrome = (full_ext[tables["left"]] ^ full_ext[tables["right"]]).astype(bool)
    err_vec = state[:, n_int].view(np.uint8)
    return syndrome, gap_list, err_vec


def _llr_map_positive(llr_map, gaps: np.ndarray) -> bool:
    """True when the calibration map is > 0 on every gap of the sample set."""
    gmin = float(gaps.min())
    gmax = float(gaps.max())
    if llr_map.knots_gap is not None and not (
        np.all(llr_map.knots_llr > 0.0) and llr_map.tail_scale >= 0.0
    ):
        return False
    return float(llr_map(gmin)) > 0.0 and float(llr_map(gmax)) > 0.0


def _lazy_tables(circuit: Circuit, cache: dict) -> dict | None:
    """Tables of _run_shot_lazy, or None when the dense path is the right one."""
    if (
        cache.get("dense_shot")
        or cache.get("l1_presolve")
        or not cache.get("trivial_first")
        or not cache.get("per_round", True)
    ):
        return None
    buckets = circuit.mismatch_buckets()
    if buckets is None:
        return None
    table, shift = buckets
    packed = circuit._idle_arrays()[5]
    llr_map = circuit._idle_arrays()[4]
    if not _llr_map_positive(llr_map, packed["gaps"]):
        print(
            "calibration map is not positive on the sample set: workers run "
            "the dense shot path",
            flush=True,
        )
        return None
    idle_len = cache["shot_tables"]["idle_len"]
    return {
        "table": table,
        "shift": shift,
        "closed_form": True,
        "ticks": np.arange(idle_len, dtype=np.int64),
    }


def _decode_lazy(decoder, syndrome, costs_of, cache: dict):
    if cache.get("l1_presolve"):
        method = "decode_tesseract_l1_presolve"
    elif cache.get("trivial_first"):
        method = "decode_tesseract_trivial_first"
    else:
        method = "decode_tesseract"
    return decoder.decode_lazy(
        method, syndrome, costs_of, closed_form=cache["lazy_tables"]["closed_form"]
    )


def _run_shot_lazy(circuit: Circuit, cache: dict | None = None) -> tuple[bool, bool]:
    """One shot with the draws, syndrome and decode of _run_shot_fast, minus
    the llrs nobody reads."""
    if cache is None:
        cache = _WORKER
    tables = cache["shot_tables"]
    lazy = cache["lazy_tables"]
    idle_len = tables["idle_len"]
    n_int = tables["n_int"]
    total_ticks = tables["total_ticks"]
    n_samples, llr_map, packed = circuit._idle_arrays()[3:]
    num_patches = len(circuit.patches)
    u64 = packed.view(np.uint64)

    idx = circuit._rng.integers(n_samples, size=(num_patches, total_ticks))
    flat = idx.reshape(-1)
    cand = np.flatnonzero(np.take(lazy["table"], flat >> lazy["shift"]))
    state = np.zeros((num_patches, n_int + 1), dtype=bool)
    if cand.size:
        rec = np.take(u64, flat[cand]).view(packed.dtype)
        hit = cand[rec["actual"] != rec["predicted"]]
        if hit.size:
            np.bitwise_xor.at(
                state,
                (hit // total_ticks, np.minimum(hit % total_ticks // idle_len, n_int)),
                True,
            )
            state = np.bitwise_xor.accumulate(state, axis=1)

    full_ext = tables["full_ext"]
    full_ext[:-1] = (CHECK_BLOCK_F32 @ state.view(np.uint8).astype(np.float32)).astype(
        np.uint8
    ).ravel() & 1
    syndrome = (full_ext[tables["left"]] ^ full_ext[tables["right"]]).astype(bool)
    err_vec = state[:, n_int].view(np.uint8)

    ticks = lazy["ticks"]
    rec_dtype = packed.dtype

    def costs_of(cols: np.ndarray) -> np.ndarray:
        cols = np.asarray(cols, dtype=np.int64)
        patch = cols % num_patches
        interval = cols // num_patches
        final = interval >= n_int
        out = np.empty(cols.size)
        if final.any():
            sel = ~final
            rows = idx[patch[sel][:, None], interval[sel][:, None] * idle_len + ticks]
            out[sel] = combine_llrs(llr_map(np.take(u64, rows).view(rec_dtype)["gaps"]))
            fin = np.take(u64, idx[patch[final], n_int * idle_len]).view(rec_dtype)
            out[final] = llr_map(fin["gaps"])
        else:
            rows = idx[patch[:, None], interval[:, None] * idle_len + ticks]
            out[:] = combine_llrs(llr_map(np.take(u64, rows).view(rec_dtype)["gaps"]))
        return out

    predicted_observables = _decode_lazy(cache["decoder"], syndrome, costs_of, cache)

    patch_observable = int(OBSERVABLE_VEC @ err_vec) & 1
    inner_error = bool(err_vec[0])
    outer_error = bool(predicted_observables[0]) != bool(patch_observable)
    return inner_error, outer_error


def _run_chunk(n_shots: int) -> tuple[int, int, int, float]:
    circuit = _WORKER["circuit"]
    batch = _batch_shots(_WORKER)
    if batch > 1:
        return _run_chunk_batched(circuit, n_shots, batch, _WORKER)
    shot = _run_shot_lazy if _WORKER.get("lazy_tables") is not None else _run_shot_fast
    inner_errors = 0
    outer_errors = 0
    t0 = time.perf_counter()
    for _ in range(n_shots):
        inner_error, outer_error = shot(circuit)
        if inner_error:
            inner_errors += 1
        if outer_error:
            outer_errors += 1
    return inner_errors, outer_errors, n_shots, time.perf_counter() - t0


def _run_chunk_batched(
    circuit: Circuit, n_shots: int, batch: int, cache: dict
) -> tuple[int, int, int, float]:
    """The dense chunk with ``batch`` shots per decoder call."""
    decoder = cache["decoder"]
    num_det = cache["circuit_matrix"].shape[0]
    num_err = cache["circuit_matrix"].shape[1]
    num_patches = len(circuit.patches)
    inner_errors = 0
    outer_errors = 0
    t0 = time.perf_counter()
    for start in range(0, n_shots, batch):
        b = min(batch, n_shots - start)
        syndromes = np.empty((b, num_det), dtype=bool)
        costs = np.empty((b, num_err), dtype=float)
        err_vecs = np.empty((b, num_patches), dtype=np.uint8)
        for k in range(b):
            syndromes[k], costs[k], err_vecs[k] = _sample_shot_fast(circuit, cache)
        predicted = _decode_batch(decoder, syndromes, costs, cache)
        patch_observables = (err_vecs.astype(np.int64) @ OBSERVABLE_VEC) & 1
        inner_errors += int(err_vecs[:, 0].sum())
        outer_errors += int(
            (predicted[:, 0].astype(bool) != patch_observables.astype(bool)).sum()
        )
    return inner_errors, outer_errors, n_shots, time.perf_counter() - t0


def run(
    *,
    d_val: int,
    t_interval_val: int,
    l1_check_rate_val: int,
    shots: int,
    csv_path: Path,
    p_val: float = P,
    n_processes: int | None = None,
    chunk_size: int = CHUNK_SIZE,
    l1_presolve: bool = False,
    ram_samples: bool = True,
    trivial_first: bool = False,
    per_round: bool = True,
    dense_shot: bool = False,
    batch_shots: int = BATCH_SHOTS,
    max_outer_errors: int | None = None,
) -> tuple[int, int, int]:
    """Sample ``shots`` shots (at most) and append the chunk stats to ``csv_path``."""
    if t_interval_val % l1_check_rate_val:
        raise ValueError("t_interval must be a multiple of l1_check_rate.")

    if n_processes is None:
        n_processes = max(1, min(DEFAULT_PROCESSES, cpu_count() - 1))
    n_processes = max(1, min(n_processes, shots))

    n_full, rem = divmod(shots, chunk_size)
    chunks = [chunk_size] * n_full + ([rem] if rem else [])

    ticks = num_ticks(d_val, t_interval_val)
    inner_errors = 0
    outer_errors = 0
    processed_shots = 0

    print("building shared circuit matrix and decoder ...", flush=True)
    t_build = time.perf_counter()
    _build_shared(
        d_val,
        t_interval_val,
        l1_check_rate_val,
        l1_presolve,
        ram_samples,
        trivial_first,
        per_round,
        dense_shot,
        batch_shots,
    )
    print(f"build done in {time.perf_counter() - t_build:.1f}s", flush=True)
    if ram_samples:
        packed = _WORKER["circuit"]._idle_arrays()[5]
        cov = hugepage_coverage(packed) if packed is not None else None
        if cov is not None:
            huge_kb, rss_kb = cov
            print(
                f"sample array: {rss_kb / 2**20:.2f} GiB resident, "
                f"{100 * huge_kb / max(rss_kb, 1):.0f}% on huge pages",
                flush=True,
            )

    print(
        f"launching {n_processes} workers (chunk_size={chunk_size}, shots={shots}"
        f"{', l1-presolve' if l1_presolve else ''}"
        f"{', trivial-first' if trivial_first else ''}"
        f"{', per-round decode' if per_round else ', whole-shot decode'}"
        f"{'' if ram_samples else ', mmap samples'}"
        f"{', lazy llrs' if _WORKER.get('lazy_tables') is not None else ', dense shot'}"
        f"{f', batch decode x{_batch_shots(_WORKER)}' if _batch_shots(_WORKER) > 1 else ''}) ...",
        flush=True,
    )
    existing_outer = 0
    if max_outer_errors is not None:
        existing_outer = existing_errors(
            csv_path,
            task_strong_id(
                _metadata(
                    d_val, p_val, t_interval_val, l1_check_rate_val, level="outer"
                )
            ),
        )
        print(
            f"stopping at {max_outer_errors} outer errors in the csv "
            f"({existing_outer} already recorded)",
            flush=True,
        )
        if existing_outer >= max_outer_errors:
            print("target already reached, nothing to do", flush=True)
            return 0, 0, 0
    pbar = tqdm(total=shots, desc="simulating", unit="shot")
    with Pool(
        processes=n_processes,
        initializer=_init_worker,
        initargs=(
            d_val,
            t_interval_val,
            l1_check_rate_val,
            l1_presolve,
            ram_samples,
            trivial_first,
            per_round,
            dense_shot,
            batch_shots,
        ),
    ) as pool:
        for chunk_inner, chunk_outer, chunk_shots, chunk_seconds in pool.imap_unordered(
            _run_chunk, chunks, chunksize=1
        ):
            inner_errors += chunk_inner
            outer_errors += chunk_outer
            processed_shots += chunk_shots
            _write_chunk_stats(
                csv_path,
                d_val=d_val,
                p_val=p_val,
                t_interval_val=t_interval_val,
                l1_check_rate_val=l1_check_rate_val,
                shots=chunk_shots,
                inner_errors=chunk_inner,
                outer_errors=chunk_outer,
                seconds=chunk_seconds,
            )
            pbar.update(chunk_shots)
            pbar.set_postfix_str(
                f"inner {inner_errors:>6} err ({inner_errors / (processed_shots * ticks):.2e}) | "
                f"outer {outer_errors:>6} err ({outer_errors / (processed_shots * ticks):.2e})"
            )
            if (
                max_outer_errors is not None
                and existing_outer + outer_errors >= max_outer_errors
            ):
                pbar.write(
                    f"reached {existing_outer + outer_errors} outer errors "
                    f"(target {max_outer_errors}) after {processed_shots} shots: "
                    "stopping"
                )
                break
    pbar.close()
    return inner_errors, outer_errors, processed_shots


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sample two-yoke (without measure error) inner/outer error rates."
    )
    parser.add_argument("--d", type=int, default=D)
    parser.add_argument("--p", type=float, default=P)
    parser.add_argument("--t-interval", type=int, default=T_INTERVAL)
    parser.add_argument("--l1-check-rate", type=int, default=L1_CHECK_RATE)
    parser.add_argument("--shots", type=int, default=SHOTS)
    parser.add_argument(
        "--processes",
        type=int,
        default=None,
        help=f"Worker processes (default: min({DEFAULT_PROCESSES}, cpu_count - 1)).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=CHUNK_SIZE,
        help="Shots per worker task.",
    )
    parser.add_argument(
        "--l1-presolve",
        action="store_true",
        help=(
            "Trivially pre-decode every fired L1 detector (cheapest column of "
            "its block) before running tesseract on the residual. Shots whose "
            "corrections explain the whole syndrome skip tesseract entirely. "
            "Results are written to the same CSV with the same strong_id as a "
            "run without the flag, so the rows merge."
        ),
    )
    parser.add_argument(
        "--trivial-first",
        action="store_true",
        help=(
            "Before each tesseract decode, solve the L1 relaxation in closed "
            "form (cheapest column of every fired L1 block); when that error "
            "set already explains the whole syndrome it is a certified global "
            "optimum and tesseract is skipped. All other shots decode exactly "
            "as without the flag. Same strong_id, so the rows merge."
        ),
    )
    parser.add_argument(
        "--mmap-samples",
        action="store_true",
        help=(
            "Keep the gap sample set memory-mapped from disk instead of copying "
            "it into RAM before forking the workers. Saves ~1.4 GB of RAM at the "
            "cost of every worker faulting the mapping in on its first chunk."
        ),
    )
    parser.add_argument(
        "--whole-shot",
        action="store_true",
        help=(
            "Decode the whole shot on one DEM instead of one outer round at a "
            "time. The checks are measured without error, so the rounds "
            "decouple and the per-round decode (the default) finds the same "
            "most likely error on a DEM a round in size. Same strong_id, so "
            "the rows merge."
        ),
    )
    parser.add_argument(
        "--dense-shot",
        action="store_true",
        help=(
            "Run the workers on the dense shot path (gather every drawn "
            "sample, compute every llr) instead of the lazy one that fetches "
            "only the mismatching samples and the llrs the decoder reads. "
            "Same draws and results; for timing comparisons."
        ),
    )
    parser.add_argument(
        "--batch-shots",
        type=int,
        default=BATCH_SHOTS,
        help=(
            "Decode this many shots per tesseract call on the dense shot path "
            "(implies --dense-shot when combined with --trivial-first): every "
            "shot keeps its own llrs, written inside the one C++ loop of the "
            "patched tesseract module. 0 or 1 decodes shot by shot. Not "
            "applied with --l1-presolve. Same draws and results."
        ),
    )
    parser.add_argument(
        "--max-outer-errors",
        type=int,
        default=None,
        help=(
            "Stop once the CSV holds this many outer error events for this "
            "task (errors recorded by earlier runs with the same parameters "
            "count). --shots is then only an upper bound."
        ),
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help=(
            "Output CSV path "
            f"(default: data/simulated/{PROTOCOL}.csv, shared by every parameter set)."
        ),
    )
    args = parser.parse_args()

    csv_path = args.csv or default_csv_path()
    inner_errors, outer_errors, processed_shots = run(
        d_val=args.d,
        t_interval_val=args.t_interval,
        l1_check_rate_val=args.l1_check_rate,
        shots=args.shots,
        csv_path=csv_path.resolve(),
        p_val=args.p,
        n_processes=args.processes,
        chunk_size=args.chunk_size,
        l1_presolve=args.l1_presolve,
        ram_samples=not args.mmap_samples,
        trivial_first=args.trivial_first,
        per_round=not args.whole_shot,
        dense_shot=args.dense_shot or args.batch_shots > 1,
        batch_shots=args.batch_shots,
        max_outer_errors=args.max_outer_errors,
    )

    ticks = num_ticks(args.d, args.t_interval)
    if processed_shots:
        print(f"Error rate inner: {inner_errors / (processed_shots * ticks)}")
        print(f"Error rate outer: {outer_errors / (processed_shots * ticks)}")
    print(f"Saved results to {csv_path}")
