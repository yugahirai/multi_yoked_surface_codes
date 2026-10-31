import numpy as np
from qldpc.codes import CSSCode
from qldpc.objects import Pauli

from _extend import extend, read_checks


def logical_qubits(Hz, Hx):
    """Return (Lz, Lx): row i of Lz and row i of Lx are the logical Z and X of logical qubit i."""
    code = CSSCode(Hx, Hz)
    return np.array(code.get_logical_ops(Pauli.Z)), np.array(
        code.get_logical_ops(Pauli.X)
    )


if __name__ == "__main__":
    l_1, l_2 = 4, 4
    Hz, Hx = read_checks("ecc/chain_codes/2_yoke/q32_18_4.txt")
    Lz, Lx = logical_qubits(Hz, Hx)
    print(f"{len(Lz)} logical qubits")
    for name, L in [("Z logicals", Lz), ("X logicals", Lx)]:
        print(name)
        for row in L:
            print("".join(map(str, row)))
        print()
