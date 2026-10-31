import numpy as np

from _code_concatenator import concatenate, level_3
from _distance_checker import min_weight
from _extend import read_checks
from _qubit_identifier import logical_qubits


def reaches(Hz, Hx, d):
    """True if the CSS code has no logical operator of weight below d."""
    Lz, Lx = logical_qubits(Hz, Hx)
    return min_weight(Hx, Lx, d) == d and min_weight(Hz, Lz, d) == d


def save_checks(path, Hz, Hx):
    with open(path, "w") as f:
        for name, H in [("Z checks", Hz), ("X checks", Hx)]:
            f.write(
                name + "\n" + "\n".join("".join(map(str, row)) for row in H) + "\n\n"
            )


def make_l3(inner_path, classical_path, l_1, l_2, d=8, seed=0):
    """Permute the columns of the classical code until the level 3 code reaches distance d, then save it."""
    rng = np.random.default_rng(seed)
    Hz, Hx = read_checks(inner_path)
    C = np.array([[int(c) for c in row] for row in open(classical_path).read().split()])
    Lz, Lx = logical_qubits(Hz, Hx)
    tries = 1
    while True:
        Hz3, Hx3 = level_3(Hz, concatenate(C, Lz), l_1, l_2), level_3(
            Hx, concatenate(C, Lx), l_1, l_2
        )
        if reaches(Hz3, Hx3, d):
            break
        print(f"try {tries}: distance below {d}, permuting the classical code")
        C = C[:, rng.permutation(C.shape[1])]
        tries += 1
    assert not (
        Hx3 @ Hz3.T % 2
    ).any(), "CSS condition violated: X and Z checks do not commute"
    n, k = Hz3.shape[1], len(logical_qubits(Hz3, Hx3)[0])
    path = f"ecc/chain_codes/3_yoke/q{n}_{k}_{d}.txt"
    save_checks(path, Hz3, Hx3)
    print(f"try {tries}: distance {d} reached, saved to {path}")


if __name__ == "__main__":
    make_l3(
        "ecc/chain_codes/2_yoke/q256_216_4.txt",
        "ecc/seed_codes/class_candidates_level_3/c216_188_8.txt",
        l_1=16,
        l_2=4,
    )
