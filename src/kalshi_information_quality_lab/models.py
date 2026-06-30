"""Probability validation shared by the empirical evaluation paths."""

import math
from collections.abc import Mapping


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{name} must be a finite number")
    try:
        result = float(value)
    except OverflowError:
        raise ValueError(f"{name} must be a finite number") from None
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


def assert_simplex(vector: Mapping[str, float], tol: float = 1e-9) -> None:
    """Reject empty, nonfinite, out-of-range or non-unit probability vectors."""
    tolerance = _finite(tol, "tol")
    if tolerance < 0:
        raise ValueError("tol must be nonnegative")
    if not vector:
        raise ValueError("probability vector must not be empty")
    values: list[float] = []
    for bin_id, probability in vector.items():
        if not isinstance(bin_id, str) or not bin_id.strip():
            raise ValueError("probability bin IDs must be nonempty strings")
        value = _finite(probability, f"probability[{bin_id}]")
        if not 0 <= value <= 1:
            raise ValueError(f"probability[{bin_id}] must be in [0, 1]")
        values.append(value)
    total = math.fsum(values)
    if abs(total - 1) > tolerance:
        raise ValueError(f"probability sum must be one within tol={tolerance}; got {total}")
