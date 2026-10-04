import numpy as np


def read_checks(path):
    z_text, x_text = open(path).read().replace("Z checks", "").split("X checks")
    return [
        np.array([[int(c) for c in row] for row in text.split()])
        for text in (z_text, x_text)
    ]


def extend(H, l_1, l_2):
    level_1 = np.kron(np.eye(2, dtype=int), H[:l_1])
    level_2 = np.kron(np.eye(2, dtype=int), H[l_1 : l_1 + l_2])
    return np.vstack([level_1, level_2])


if __name__ == "__main__":
    Hz, Hx = read_checks("ecc/chain_codes/2_yoke/q24_8_4.txt")
    for name, H in [("Z checks", Hz), ("X checks", Hx)]:
        print(name)
        for row in extend(H, l_1=4, l_2=4):
            print("".join(map(str, row)))
        print()
