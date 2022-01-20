import os


def _ensure_exists(path_out: str) -> None:
    if os.path.exists(path_out):
        return
    os.makedirs(path_out)
