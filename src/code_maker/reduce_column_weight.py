"""Reduce the column weight of a level 3 code made by l3_code_maker.py."""

import math
import multiprocessing as mp
import random
from itertools import combinations

import numpy as np

from _extend import read_checks

EXACT_MAX_L3_ROWS = 22
MAX_L3_COMBO = 6
ROUNDS = 10
MAX_COLUMN_WEIGHT = 11
ANNEAL_SEEDS = 1000
ANNEAL_ITERS = 400_000
ANNEAL_REPEATS = 100
SOLVE_SEED = None

_TIE_RNG = random.Random(SOLVE_SEED)
_EXACT = {}
Pool = mp.get_context("fork").Pool


def levels(H):
    """(n_l1, n_l2) of our construction: the L1 rows are disjoint and the L3 rows are [M, M]."""
    half = H.shape[1] // 2
    n_l1 = int(np.argmax(np.cumsum(H, axis=0).max(axis=1) > 1))
    n_l3 = int((H[:, :half] == H[:, half:]).all(axis=1).sum())
    return n_l1, len(H) - n_l1 - n_l3


def to_bits(row):
    return int("".join(map(str, row))[::-1], 2)


def to_str(bits, n_cols):
    return format(bits, f"0{n_cols}b")[::-1]


def subset_xors(rows):
    """XOR of every subset of rows (2**len(rows) values, starts with 0)."""
    xors = [0]
    for r in rows:
        xors += [x ^ r for x in xors]
    return xors


def l1_blocks(l1, n_cols):
    """[(mask, size)] of the L1 blocks; columns outside L1 form a block that is never folded."""
    blocks = [(m, m.bit_count()) for m in l1]
    uncovered = ((1 << n_cols) - 1) ^ sum(l1)
    return blocks + [(uncovered, 2 * n_cols)] * bool(uncovered)


def fold(row, blocks):
    """The lightest row + (L1 rows): add an L1 row iff the row covers more than half of its block."""
    for mask, size in blocks:
        if 2 * (row & mask).bit_count() > size:
            row ^= mask
    return row


def _exact_weight_chunk(bounds):
    """Minimum folded weight over every lower-level subset, for the cosets lo..hi."""
    lo, hi = bounds
    arr, lut = _EXACT["arr"][lo:hi], _EXACT["lut"]
    best = np.full(hi - lo, 65535, dtype=np.uint16)
    w = np.empty(hi - lo, dtype=np.uint16)
    wb = np.empty(hi - lo, dtype=np.uint16)
    for x in _EXACT["xors"]:
        y = arr ^ x
        w.fill(0)
        for size, words in _EXACT["blocks"]:
            (j, m), *rest = words
            np.copyto(wb, lut[y[:, j] & m])
            for j, m in rest:
                wb += lut[y[:, j] & m]
            w += np.minimum(wb, size - wb)
        np.minimum(best, w, out=best)
    return best


def exact_min_rows(blocks, lower_rows, target_rows, n_cols):
    """Minimum-total-weight basis of span(all) / span(L1 + lower_rows)."""
    n_words, k = -(-n_cols // 16), len(target_rows)
    pack = lambda r: np.array([(r >> (16 * j)) & 0xFFFF for j in range(n_words)], dtype=np.uint16)
    arr = np.zeros((1 << k, n_words), dtype=np.uint16)
    for b, r in enumerate(target_rows):
        arr[1 << b : 2 << b] = arr[: 1 << b] ^ pack(r)
    lower_xors = subset_xors(lower_rows)
    _EXACT.update(
        arr=arr,
        lut=np.array([i.bit_count() for i in range(1 << 16)], dtype=np.uint16),
        xors=[pack(x) for x in lower_xors],
        blocks=[(size, [(j, w) for j, w in enumerate(pack(mask)) if w]) for mask, size in blocks],
    )
    n_rows = 1 << k
    if n_rows < 1 << 16:
        weights = _exact_weight_chunk((0, n_rows))
    else:
        step = -(-n_rows // (2 * mp.cpu_count()))
        with Pool() as pool:
            parts = pool.map(_exact_weight_chunk, [(lo, min(lo + step, n_rows)) for lo in range(0, n_rows, step)])
        weights = np.concatenate(parts)

    tie_keys = np.frombuffer(_TIE_RNG.randbytes(8 * (n_rows - 1)), dtype=np.uint64)
    basis, picked = {}, []
    for idx in np.lexsort((tie_keys, weights[1:])) + 1:
        v = int(idx)
        while v:
            lead = v.bit_length() - 1
            if lead not in basis:
                basis[lead] = v
                picked.append(int(idx))
                break
            v ^= basis[lead]
        if len(picked) == k:
            break
    assert len(picked) == k, "target rows are linearly dependent over the lower levels"

    new_rows = []
    for idx in picked:
        v = 0
        for b in range(k):
            if idx >> b & 1:
                v ^= target_rows[b]
        reps = [fold(v ^ x, blocks) for x in lower_xors]
        new_rows.append(_TIE_RNG.choice([r for r in reps if r.bit_count() == weights[idx]]))
    return new_rows


def _eval_l3_chunk(args):
    """Best candidate of one chunk of the greedy L3 scan: (weight, index, row)."""
    blocks, base, x3_chunk, x3_offset, l2_xors = args
    best = (base.bit_count(), -1, base)
    for k, x3 in enumerate(x3_chunk):
        partial = base ^ x3
        for j, x2 in enumerate(l2_xors):
            r = fold(partial ^ x2, blocks)
            if r.bit_count() < best[0]:
                best = (r.bit_count(), (x3_offset + k) * len(l2_xors) + j, r)
    return best


def reduce_levels(blocks, l2, l3):
    """Greedy local search, used when there are too many L3 rows for the exact solver."""
    l2 = [fold(r, blocks) for r in l2]
    l3 = [fold(r, blocks) for r in l3]
    n_chunks = 4 * mp.cpu_count()
    with Pool() as pool:
        for round_index in range(1, ROUNDS + 1 if ROUNDS else 10**9):
            improved = False
            for i in range(len(l2)):
                best = min((fold(l2[i] ^ x, blocks) for x in subset_xors(l2[:i] + l2[i + 1 :])), key=int.bit_count)
                if best.bit_count() < l2[i].bit_count():
                    l2[i], improved = best, True
            l2_xors = subset_xors(l2)
            for i in range(len(l3)):
                l3_xors = [0]
                for k in range(1, MAX_L3_COMBO + 1):
                    for combo in combinations(l3[:i] + l3[i + 1 :], k):
                        x = 0
                        for r in combo:
                            x ^= r
                        l3_xors.append(x)
                chunk = -(-len(l3_xors) // n_chunks)
                tasks = [(blocks, l3[i], l3_xors[k : k + chunk], k, l2_xors) for k in range(0, len(l3_xors), chunk)]
                weight, _, best = min(pool.map(_eval_l3_chunk, tasks))
                if weight < l3[i].bit_count():
                    l3[i], improved = best, True
            print(f"  round {round_index}: L2 + L3 weight {sum(r.bit_count() for r in l2 + l3)}", flush=True)
            if not improved:
                break
    return l2, l3


def _min_weight_reps(row, blocks, lower_xors, cap=4096):
    """Every minimum-weight representative of the row's coset (at most cap of them)."""
    best_w, reps = None, set()
    for x in lower_xors:
        base = fold(row ^ x, blocks)
        w = base.bit_count()
        if best_w is None or w < best_w:
            best_w, reps = w, set()
        if w == best_w and len(reps) < cap:
            tie_masks = [mask for mask, size in blocks if 2 * (base & mask).bit_count() == size]
            for k in range(len(tie_masks) + 1):
                for combo in combinations(tie_masks, k):
                    reps.add(base ^ sum(combo))
                    if len(reps) >= cap:
                        break
                if len(reps) >= cap:
                    break
    return sorted(reps)


def flatten_column_max(l1, blocks, l2, l3, n_cols):
    """Swap rows for equal-weight representatives of the same coset to lexicographically
    minimize the sorted-descending column weights (local search)."""
    pools = [_min_weight_reps(r, blocks, [0]) for r in l2]
    pools += [_min_weight_reps(r, blocks, subset_xors(l2)) for r in l3]
    rows = [_TIE_RNG.choice(pool) for pool in pools]
    visit = list(range(len(pools)))

    weights = _cap_col_weights(l1, rows, n_cols)
    hist = [0] * (len(l1) + len(rows) + 2)
    for w in weights:
        hist[w] += 1
    best_key = hist[::-1]
    improved = True
    while improved:
        improved = False
        _TIE_RNG.shuffle(visit)
        for i in visit:
            pool = pools[i]
            if len(pool) == 1:
                continue
            for cand in _TIE_RNG.sample(pool, len(pool)):
                cur = rows[i]
                if cand == cur:
                    continue
                diff = _bit_positions(cand ^ cur)
                h = hist.copy()
                for c in diff:
                    w = weights[c]
                    h[w] -= 1
                    h[w + 1 if (cand >> c) & 1 else w - 1] += 1
                key = h[::-1]
                if key < best_key:
                    for c in diff:
                        weights[c] += 1 if (cand >> c) & 1 else -1
                    hist, best_key, rows[i], improved = h, key, cand, True
    return rows[: len(l2)], rows[len(l2) :]


def _bit_positions(v):
    out = []
    while v:
        low = v & -v
        out.append(low.bit_length() - 1)
        v ^= low
    return out


def _cap_col_weights(l1, current, n_cols):
    weights = [0] * n_cols
    for r in l1 + current:
        for c in _bit_positions(r):
            weights[c] += 1
    return weights


def _anneal_seed(args):
    """One annealing chain; returns (excess energy, seed, rows)."""
    seed, start, l1, n_l2, n_cols, target, iters = args
    rng = random.Random(seed)
    current = list(start)
    n_rows = len(current)
    weights = _cap_col_weights(l1, current, n_cols)
    pen = [(w - target) ** 2 if w > target else 0 for w in range(len(l1) + n_rows + 2)]
    energy = sum(pen[w] for w in weights)
    if energy == 0:
        return 0, seed, current

    n_l1 = len(l1)
    l1_bits = [_bit_positions(m) for m in l1]
    col_l1 = {c: k for k, bits in enumerate(l1_bits) for c in bits}
    row_bits = [_bit_positions(r) for r in current]
    col_rows = [set() for _ in range(n_cols)]
    for i, bits in enumerate(row_bits):
        for c in bits:
            col_rows[c].add(i)
    right = lambda r: r & ((1 << n_cols // 2) - 1) == 0
    gen_idx = [
        [
            k
            for k, g in enumerate(l1 + current[: n_l2 if i < n_l2 else n_rows])
            if k != n_l1 + i and (i >= n_l2 or right(g) == right(current[i]))
        ]
        for i in range(n_rows)
    ]
    over = [c for c, w in enumerate(weights) if w > target]
    over_pos = {c: p for p, c in enumerate(over)}

    temp, t_lo = 3.0, 0.02
    decay = (t_lo / temp) ** (1.0 / iters)
    rnd, randrange, exp = rng.random, rng.randrange, math.exp
    for _ in range(iters):
        temp *= decay
        k = -1
        if over and rnd() < 0.7:
            c = over[randrange(len(over))]
            covering = col_rows[c]
            if covering:
                i = rng.choice(tuple(covering))
                cands = [col_l1[c]] if c in col_l1 else []
                for j in covering:
                    if j != i and (i >= n_l2 or j < n_l2):
                        cands.append(n_l1 + j)
                if cands:
                    k = cands[randrange(len(cands))]
        if k < 0:
            i = randrange(n_rows)
            gens = gen_idx[i]
            k = gens[randrange(len(gens))]
        if k < n_l1:
            g, g_bits = l1[k], l1_bits[k]
        else:
            g, g_bits = current[k - n_l1], row_bits[k - n_l1]
        r = current[i]
        delta = 0
        for c2 in g_bits:
            w = weights[c2]
            delta += pen[w - 1 if (r >> c2) & 1 else w + 1] - pen[w]
        if delta <= 0 or rnd() < exp(-delta / temp):
            for c2 in g_bits:
                if (r >> c2) & 1:
                    w = weights[c2] - 1
                    weights[c2] = w
                    col_rows[c2].discard(i)
                    if w == target:
                        p = over_pos.pop(c2)
                        last = over.pop()
                        if last != c2:
                            over[p] = last
                            over_pos[last] = p
                else:
                    w = weights[c2] + 1
                    weights[c2] = w
                    col_rows[c2].add(i)
                    if w == target + 1:
                        over_pos[c2] = len(over)
                        over.append(c2)
            current[i] = r ^ g
            row_bits[i] = _bit_positions(current[i])
            energy += delta
            if energy == 0:
                break
    return energy, seed, current


def cap_column_weight_section(l1, rows, n_l2, n_cols, target, name):
    """Anneal the L2 + L3 rows until no column weighs more than target, then greedily
    win back total weight under that cap.  Best effort if the target is not reached."""
    total = sum(r.bit_count() for r in l1 + rows)
    print(f"  {name}: total weight {total}, so no valid matrix has max column weight below {-(-total // n_cols)}")
    entropy = random.SystemRandom()
    for rep in range(ANNEAL_REPEATS):
        tasks = [(entropy.randrange(1 << 62), rows, l1, n_l2, n_cols, target, ANNEAL_ITERS) for _ in range(ANNEAL_SEEDS)]
        with Pool() as pool:
            energy, seed, rows = _first_or_best(pool.imap_unordered(_anneal_seed, tasks))
        print(f"  {name}: repeat {rep + 1}: excess energy {energy}", flush=True)
        if energy == 0:
            break
    else:
        print(f"  {name}: max column weight {target} not reached; keeping the best state")
        return rows

    rng = random.Random(seed + 1_000_003)
    weights = _cap_col_weights(l1, rows, n_cols)
    for _ in range(ANNEAL_ITERS // 2):
        i = rng.randrange(len(rows))
        same_level = range(n_l2) if i < n_l2 else range(len(rows))
        g = rng.choice(l1 + [rows[j] for j in same_level if j != i])
        r = rows[i]
        bits = _bit_positions(g)
        if all((r >> c) & 1 or weights[c] < target for c in bits) and 2 * (r & g).bit_count() > len(bits):
            for c in bits:
                weights[c] += -1 if (r >> c) & 1 else 1
            rows[i] = r ^ g
    return rows


def _first_or_best(results):
    """The first chain that reaches energy 0, or the best chain if none does."""
    best = None
    for res in results:
        if best is None or res[0] < best[0]:
            best = res
        if res[0] == 0:
            break
    return best


def gf2_rank(rows):
    basis = {}
    for v in rows:
        while v:
            lead = v.bit_length() - 1
            if lead not in basis:
                basis[lead] = v
                break
            v ^= basis[lead]
    return len(basis)


def assert_same_span(old_rows, new_rows):
    assert gf2_rank(old_rows) == gf2_rank(new_rows) == gf2_rank(old_rows + new_rows), "span changed"


def reduce_section(H, name):
    """Reduce one check matrix (Z or X); returns the new rows as 0/1 strings."""
    n_cols = H.shape[1]
    n_l1, n_l2 = levels(H)
    rows = [to_bits(r) for r in H]
    l1, l2, l3 = rows[:n_l1], rows[n_l1 : n_l1 + n_l2], rows[n_l1 + n_l2 :]
    blocks = l1_blocks(l1, n_cols)
    print(f"{name}: {len(l1)} L1 + {len(l2)} L2 + {len(l3)} L3 rows")
    if len(l3) <= EXACT_MAX_L3_ROWS:
        l2 = exact_min_rows(blocks, [], l2, n_cols)
        l3 = exact_min_rows(blocks, l2, l3, n_cols)
        l2, l3 = flatten_column_max(l1, blocks, l2, l3, n_cols)
    else:
        print(f"  more than {EXACT_MAX_L3_ROWS} L3 rows: using the greedy solver")
        l2, l3 = reduce_levels(blocks, l2, l3)
    half = n_cols // 2
    l2.sort(key=lambda r: r >> half != 0)
    if MAX_COLUMN_WEIGHT is not None:
        capped = cap_column_weight_section(l1, l2 + l3, n_l2, n_cols, MAX_COLUMN_WEIGHT, name)
        l2, l3 = capped[:n_l2], capped[n_l2:]
    new_rows = l1 + l2 + l3

    assert all(r >> half == 0 for r in l2[: n_l2 // 2]) and all(r & ((1 << half) - 1) == 0 for r in l2[n_l2 // 2 :]), "L2 is not [[M_1, O], [O, M_2]]"
    assert_same_span(rows[: n_l1 + n_l2], l1 + l2)
    assert_same_span(rows, new_rows)
    for label, rs in [("before", rows), ("after", new_rows)]:
        weights = _cap_col_weights([], rs, n_cols)
        print(f"  {label}: total weight {sum(weights)}, column weight mean {sum(weights) / n_cols:.2f}, max {max(weights)}")
    return [to_str(r, n_cols) for r in new_rows]


def reduce_file(path):
    """Reduce the code in path and save it next to it as <name>_reduced.txt."""
    _TIE_RNG.seed(SOLVE_SEED)
    Hz, Hx = read_checks(path)
    z_rows, x_rows = reduce_section(Hz, "Z checks"), reduce_section(Hx, "X checks")
    assert all((to_bits(z) & to_bits(x)).bit_count() % 2 == 0 for z in z_rows for x in x_rows), "CSS condition violated"
    out_path = path.replace(".txt", "_reduced.txt")
    with open(out_path, "w") as f:
        f.write("\n".join(["Z checks", *z_rows, "", "X checks", *x_rows]) + "\n")
    print(f"saved to {out_path}")


if __name__ == "__main__":
    reduce_file("ecc/chain_codes/3_yoke/q112_40_8.txt")
