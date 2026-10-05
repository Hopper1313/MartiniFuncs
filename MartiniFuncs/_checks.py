"""Argument validators and JSON conversion shared across the package."""
from pathlib import Path

import numpy as np


def integer(value, name, minimum=1):
    """Return value as int if it is an integer >= minimum (bools rejected), else raise ValueError."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        kind = {0: "a nonnegative integer", 1: "a positive integer"}.get(minimum, f"an integer >= {minimum}")
        raise ValueError(f"{name} must be {kind}")
    return int(value)


def positive(value, name):
    """Return value as float if it is finite and > 0, else raise ValueError."""
    if value is None or not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return float(value)


def plain(value, fallback=None):
    """Convert scientific metadata to JSON types (arrays, NumPy scalars, paths, tuples, sets).

    Objects of any other type are passed to fallback (e.g. str), or returned
    unchanged when fallback is None so json.dumps reports them.
    """
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): plain(v, fallback) for k, v in value.items()}
    if isinstance(value, set):
        return [plain(v, fallback) for v in sorted(value)]
    if isinstance(value, (list, tuple)):
        return [plain(v, fallback) for v in value]
    if value is None or isinstance(value, (str, int, float, bool)) or fallback is None:
        return value
    return fallback(value)
