"""Minimum distance of a classical binary linear code.

The parity-check matrix is read from a text file containing a ``Z checks``
section: one 0/1 row per check, one column per bit.  For a code with check
matrix ``H`` over GF(2):

  * codewords  C = ker(H) = {c : H @ c = 0 (mod 2)}
  * dimension  k = n - rank(H)
  * distance   d = min Hamming weight of a nonzero codeword
"""

from __future__ import annotations

import os
import sys

import numpy as np

from distance_check import coset_min_weight, rank_gf2


def load_parity_check(path: str) -> np.ndarray:
    """Load the parity-check matrix.

    Accepts either a bare 0/1 matrix (one row per line) or a file with a
    ``Z checks`` header, in which case only that section is read.
    """
    rows: list[list[int]] = []
    in_section = True  # bare matrix: every 0/1 line counts

    with open(path) as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            lowered = line.lower()
            if lowered.startswith("z check"):
                in_section = True
                rows = []
            elif lowered.startswith("x check"):
                in_section = False
            elif set(line) <= {"0", "1"} and in_section:
                rows.append([int(c) for c in line])

    if not rows:
        raise ValueError(f"{path}: no parity-check matrix found")
    return np.array(rows, dtype=np.int8)


def save_parity_check(path: str, H: np.ndarray) -> None:
    """Write ``H`` as bare 0/1 rows with no section header."""
    with open(path, "w") as fh:
        for row in H:
            fh.write("".join(str(int(b)) for b in row) + "\n")


def main() -> None:
    fname = sys.argv[1] if len(sys.argv) > 1 else "c216_188_8_classical.txt"
    path = (
        fname
        if os.path.isabs(fname)
        else os.path.join(os.path.dirname(__file__), fname)
    )

    H = load_parity_check(path)
    n = H.shape[1]
    k = n - rank_gf2(H)
    # Classical distance: min-weight nonzero vector in ker(H).
    d = coset_min_weight(H, np.zeros((0, n), dtype=np.int8))

    print(f"[n, k, d] = [{n}, {k}, {d}]")

    # Rewrite the input as c{n}_{k}_{d}.txt with the "Z checks" header removed.
    new_name = f"c{n}_{k}_{d}.txt"
    new_path = os.path.join(os.path.dirname(path), new_name)
    if os.path.basename(path) != new_name and os.path.exists(new_path):
        print(f"not renamed: {new_name} already exists")
        return
    save_parity_check(new_path, H)
    if os.path.basename(path) != new_name:
        os.remove(path)
        print(f"renamed {os.path.basename(path)} -> {new_name}")


if __name__ == "__main__":
    main()
