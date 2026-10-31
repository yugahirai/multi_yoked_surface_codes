import numpy as np


def format_matrix(
    matrix: np.ndarray,
    *,
    name: str = "matrix",
    group_size: int = 4,
) -> str:
    arr = np.asarray(matrix, dtype=int)
    n_rows, n_cols = arr.shape
    cell_width = max(1, len(str(int(arr.max()))) if arr.size else 1)

    def _format_row(row: np.ndarray) -> str:
        chunks = []
        for start in range(0, len(row), group_size):
            chunk = row[start : start + group_size]
            chunks.append("".join(f"{v:>{cell_width}d}" for v in chunk))
        return "".join(chunks)

    lines = [f"{name}  shape=({n_rows}, {n_cols})"]
    for i, row in enumerate(arr):
        lines.append(f"  r{i:02d}  [{_format_row(row)}]")
    return "\n".join(lines)
