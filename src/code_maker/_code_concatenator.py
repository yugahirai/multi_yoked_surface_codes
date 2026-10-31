import numpy as np

from _extend import extend, read_checks
from _qubit_identifier import logical_qubits


def concatenate(C, L):
    """Rows of C (classical checks) written as products of the logicals L. Inner code not included."""
    return C @ L % 2


def level_3(H, M, l_1, l_2):
    """Extended matrix from _extend.py with [M, M] merged below its last row."""
    return np.vstack([extend(H, l_1, l_2), np.hstack([M, M])])


if __name__ == "__main__":
    l_1, l_2 = 7, 3
    Hz, Hx = read_checks("ecc/chain_codes/2_yoke/q56_36_4.txt")
    C = np.array([[int(c) for c in row] for row in open("ecc/seed_codes/class_candidates_level_3/c36_20_7.txt").read().split()])
    Lz, Lx = logical_qubits(Hz, Hx)
    for name, H, L in [("Z checks", Hz, Lz), ("X checks", Hx, Lx)]:
        print(name)
        for row in level_3(H, concatenate(C, L), l_1, l_2):
            print("".join(map(str, row)))
        print()
