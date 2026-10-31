"""Exact distance of a CSS code by meet-in-the-middle search."""

import multiprocessing as mp
from itertools import combinations
from math import comb

import galois
import numpy as np

from _qubit_identifier import logical_qubits

TABLE_MAX = 300_000_000
CHUNK = 4_000_000
_T = {}


def _keys(M):
    """Each column of M as a uint64: the column itself if M has <= 64 rows, else a random XOR-linear hash."""
    m = len(M)
    bits = np.random.default_rng(0).integers(0, 2**64, m, dtype=np.uint64)
    if m <= 64:
        bits = np.uint64(1) << np.arange(m, dtype=np.uint64)
    return np.bitwise_xor.reduce(np.where(M.T.astype(bool), bits, np.uint64(0)), axis=1)


def _subset_xors(keys, r):
    """XOR of keys over every r-subset of columns, in colex order (the subsets of range(j) come first)."""
    if r == 0:
        return np.zeros(1, np.uint64)
    prev = _subset_xors(keys, r - 1)
    return np.concatenate([prev[: comb(j, r - 1)] ^ keys[j] for j in range(len(keys))])


def _subset(i, r):
    """The i-th r-subset in colex order."""
    out = []
    for r in range(r, 0, -1):
        j = r - 1
        while comb(j + 1, r) <= i:
            j += 1
        out.append(j)
        i -= comb(j, r)
    return out


def _query(task):
    """Is there a logical made of a table subset and (a table subset in [a, a + CHUNK)) + extra?"""
    extra, a = task
    H, L, h, syn, lsyn = _T["H"], _T["L"], _T["h"], _T["syn"], _T["lsyn"]
    b = min(
        a + CHUNK, comb(extra[0], h) if extra else len(syn)
    )
    qk = syn[a:b] ^ np.bitwise_xor.reduce(_T["keys"][list(extra)])
    ql = lsyn[a:b] ^ np.bitwise_xor.reduce(_T["lkeys"][list(extra)])
    pos = np.minimum(np.searchsorted(_T["ukeys"], qk), len(_T["ukeys"]) - 1)
    hit = (_T["ukeys"][pos] == qk) & ((_T["lmin"][pos] != ql) | (_T["lmax"][pos] != ql))
    for i in np.flatnonzero(hit):
        for t in np.flatnonzero((syn == qk[i]) & (lsyn != ql[i])):
            support = _subset(a + i, h) + list(extra) + _subset(t, h)
            v = np.bincount(support, minlength=H.shape[1]) % 2
            if not (H @ v % 2).any() and (L @ v % 2).any():
                return True
    return False


def _has_logical(H, L, keys, lkeys, w):
    """Is there a weight-w logical, given that there is none of lower weight?"""
    n = H.shape[1]
    h = w // 2
    while comb(n, h) > TABLE_MAX:
        h -= 1
    syn, lsyn = _subset_xors(keys, h), _subset_xors(lkeys, h)
    order = np.argsort(syn)
    s, l = syn[order], lsyn[order]
    starts = np.flatnonzero(np.r_[True, s[1:] != s[:-1]])
    lmin, lmax = np.minimum.reduceat(l, starts), np.maximum.reduceat(l, starts)
    if w == 2 * h and (lmin == lmax).all():
        return False
    _T.update(
        H=H,
        L=L,
        h=h,
        keys=keys,
        lkeys=lkeys,
        syn=syn,
        lsyn=lsyn,
        ukeys=s[starts],
        lmin=lmin,
        lmax=lmax,
    )
    tasks = (
        (extra, a)
        for extra in combinations(range(n), w - 2 * h)
        for a in range(0, comb(extra[0], h) if extra else len(syn), CHUNK)
    )
    with mp.get_context("fork").Pool() as pool:
        return any(pool.imap_unordered(_query, tasks))


def min_weight(H, L, bound):
    """Minimum weight of v with H @ v == 0 and L @ v != 0 (mod 2), given a known one of weight `bound`."""
    keys, lkeys = _keys(H), _keys(L)
    rank = lambda M: np.linalg.matrix_rank(galois.GF2(M))
    step = 2 if rank(np.vstack([H, np.ones_like(H[:1])])) == rank(H) else 1
    for w in range(step, bound, step):
        if _has_logical(H, L, keys, lkeys, w):
            return w
    return bound


def distance(Hz, Hx):
    """Return (dz, dx): the minimum weights of a Z-type and an X-type logical operator."""
    Lz, Lx = logical_qubits(Hz, Hx)
    dz = min_weight(Hx, Lx, int(Lz.sum(1).min()))
    dx = min_weight(Hz, Lz, int(Lx.sum(1).min()))
    return dz, dx


if __name__ == "__main__":
    from _code_concatenator import concatenate, level_3
    from _extend import read_checks

    l_1, l_2 = 7, 3
    Hz, Hx = read_checks("ecc/chain_codes/3_yoke/q256_166_8_reduced.txt")
    dz, dx = distance(Hz, Hx)
    print(
        f"[[n, k, d]] = [[{Hz.shape[1]}, {len(logical_qubits(Hz, Hx)[0])}, {min(dz, dx)}]]  (dz = {dz}, dx = {dx})"
    )
