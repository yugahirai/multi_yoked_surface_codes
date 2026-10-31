from pathlib import Path

REST = "rest"


def read_sections(path):
    """Return (z_rows, obs_rows) from a check-matrix text file (X checks ignored)."""
    z_rows = []
    obs_rows = []
    section = None
    for line in Path(path).read_text().splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        low = stripped.lower()
        if low == "z checks":
            section = "z"
            continue
        if low == "x checks":
            section = "x"
            continue
        if low == "observable" or low == "observables":
            section = "obs"
            continue
        if set(stripped) > {"0", "1"}:
            continue
        row = [int(c) for c in stripped]
        if section == "z":
            z_rows.append(row)
        elif section == "obs":
            obs_rows.append(row)
    return z_rows, obs_rows


def parse_matrix(path, n_l1=None, n_l2=None, n_l3=None, *, n_l0=None):
    """Parse a check-matrix text file into level matrices and observables."""
    if n_l0 is not None:
        if n_l1 is not None:
            raise TypeError("Pass either n_l0 or n_l1, not both.")
        n_l1 = n_l0
    if n_l1 is None:
        n_l1 = 2

    z_rows, obs_rows = read_sections(path)

    three_levels = n_l3 is not None
    if three_levels:
        if n_l2 is None:
            raise TypeError("n_l2 is required when n_l3 is given.")
        sizes = [n_l1, n_l2, n_l3]
    else:
        sizes = [n_l1, n_l2]

    if any(size is None or size == REST for size in sizes[:-1]):
        raise TypeError("Only the last level may be None/REST.")

    if sizes[-1] is None or sizes[-1] == REST:
        sizes[-1] = max(0, len(z_rows) - sum(sizes[:-1]))

    total = sum(sizes)
    if total > len(z_rows):
        raise ValueError(
            f"{path} has {len(z_rows)} Z-check rows but the requested level sizes "
            f"{sizes} need {total}."
        )
    if total < len(z_rows):
        rest = len(z_rows) - sum(sizes[:-1])
        raise ValueError(
            f"{path} has {len(z_rows)} Z-check rows but the requested level sizes "
            f"{sizes} only cover {total}. Set the last level to {rest} or leave it "
            "as None to take all remaining rows."
        )

    levels = []
    start = 0
    for size in sizes:
        levels.append(z_rows[start : start + size])
        start += size

    return (*levels, obs_rows)


if __name__ == "__main__":
    import sys

    matrix_path = (
        sys.argv[1]
        if len(sys.argv) > 1
        else (Path(__file__).resolve().parent.parent / "matrices" / "40q.txt")
    )
    def _size(arg):
        return REST if arg.lower() == REST else int(arg)

    n_l1 = _size(sys.argv[2]) if len(sys.argv) > 2 else 2
    n_l2 = _size(sys.argv[3]) if len(sys.argv) > 3 else None
    n_l3 = _size(sys.argv[4]) if len(sys.argv) > 4 else None
    *levels, obs = parse_matrix(matrix_path, n_l1=n_l1, n_l2=n_l2, n_l3=n_l3)

    def _fmt(name, mat):
        lines = [f"{name} = ["]
        for row in mat:
            lines.append("    [" + ", ".join(str(v) for v in row) + "],")
        lines.append("]")
        return "\n".join(lines)

    for index, level in enumerate(levels, start=1):
        print(_fmt(f"L{index}_ELEMENTARY_MATRIX", level))
        print()
    print(_fmt("OBSERVABLES", obs))
