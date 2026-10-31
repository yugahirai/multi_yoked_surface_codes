import fcntl
import os
import shutil
from pathlib import Path

from tqdm import tqdm

_LOCK_PATH = Path(os.environ.get("GAP_SIM_TQDM_LOCK", "/tmp/gap_sim_tqdm.lock"))
_lock_fd = None


def _acquire_tqdm_lock() -> None:
    global _lock_fd
    _lock_fd = open(_LOCK_PATH, "w")
    fcntl.flock(_lock_fd.fileno(), fcntl.LOCK_EX)


def _release_tqdm_lock() -> None:
    fcntl.flock(_lock_fd.fileno(), fcntl.LOCK_UN)
    _lock_fd.close()


def shot_progress_bar(*, desc: str, total: int) -> tqdm:
    term_cols = shutil.get_terminal_size(fallback=(120, 24)).columns
    position = int(os.environ.get("GAP_SIM_TQDM_BASE", "0"))
    return tqdm(
        total=total,
        desc=desc,
        unit="shot",
        position=position,
        ncols=term_cols,
        leave=True,
        lock_args=(_acquire_tqdm_lock, _release_tqdm_lock),
        bar_format=(
            "{desc}: {n_fmt}/{total_fmt} ({percentage:5.2f}%)|{bar}| "
            "[{elapsed}<{remaining}, {rate_fmt}] {postfix}"
        ),
    )
