"""Scoring and calibration calculations used by the empirical evaluations."""

import math
from bisect import bisect_right
from collections.abc import Iterable, Mapping
from itertools import pairwise
from typing import TypedDict

from .models import assert_simplex


def _number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{name} must be a finite number")
    try:
        result = float(value)
    except OverflowError:
        raise ValueError(f"{name} must be a finite number") from None
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


def _winner(probabilities: Mapping[str, float], winning_bin_id: str) -> None:
    assert_simplex(probabilities)
    if not isinstance(winning_bin_id, str) or not winning_bin_id.strip():
        raise ValueError("winning_bin_id must be a nonempty string")
    if winning_bin_id not in probabilities:
        raise ValueError("winning_bin_id must occur in the probability vector")


def multiclass_brier(probabilities: Mapping[str, float], winning_bin_id: str) -> float:
    """Return the sum of squared errors across all bins."""
    _winner(probabilities, winning_bin_id)
    return math.fsum(
        (probabilities[bin_id] - int(bin_id == winning_bin_id)) ** 2
        for bin_id in sorted(probabilities)
    )


def log_loss(
    probabilities: Mapping[str, float],
    winning_bin_id: str,
    *,
    epsilon: float = 1e-6,
) -> float:
    """Return natural-log loss with a declared winner-probability floor."""
    floor = _number(epsilon, "epsilon")
    if not 0 < floor <= 1:
        raise ValueError("epsilon must be in (0, 1]")
    _winner(probabilities, winning_bin_id)
    return -math.log(max(probabilities[winning_bin_id], floor))


class ReliabilityBin(TypedDict):
    lower: float
    upper: float
    count: int
    mean_probability: float | None
    observed_frequency: float | None


def _binary_inputs(
    probabilities: Iterable[float],
    outcomes: Iterable[bool],
) -> tuple[tuple[float, ...], tuple[bool, ...]]:
    try:
        raw_probabilities = tuple(probabilities)
        labels = tuple(outcomes)
    except TypeError as exc:
        raise ValueError("probabilities and outcomes must be iterable") from exc
    if not raw_probabilities or len(raw_probabilities) != len(labels):
        raise ValueError("probabilities and outcomes must be nonempty and have the same length")
    values = tuple(_number(value, "probability") for value in raw_probabilities)
    if any(not 0 <= value <= 1 for value in values):
        raise ValueError("probabilities must be in [0, 1]")
    if any(type(label) is not bool for label in labels):
        raise ValueError("outcomes must be boolean")
    return values, labels


def reliability_bins(
    probabilities: Iterable[float],
    outcomes: Iterable[bool],
    *,
    edges: tuple[float, ...],
) -> list[ReliabilityBin]:
    """Return descriptive binary calibration bins with explicit empty bins."""
    values, labels = _binary_inputs(probabilities, outcomes)
    if not isinstance(edges, tuple) or len(edges) < 2:
        raise ValueError("edges must be a tuple containing at least 0 and 1")
    boundaries = tuple(_number(edge, "edge") for edge in edges)
    if boundaries[0] != 0 or boundaries[-1] != 1:
        raise ValueError("edges must start at 0 and end at 1")
    if any(right <= left for left, right in pairwise(boundaries)):
        raise ValueError("edges must be strictly increasing")

    bin_probabilities: list[list[float]] = [[] for _ in boundaries[:-1]]
    bin_outcomes: list[list[bool]] = [[] for _ in boundaries[:-1]]
    for probability, outcome in zip(values, labels, strict=True):
        index = min(bisect_right(boundaries, probability) - 1, len(boundaries) - 2)
        bin_probabilities[index].append(probability)
        bin_outcomes[index].append(outcome)

    result: list[ReliabilityBin] = []
    for index, (lower, upper) in enumerate(pairwise(boundaries)):
        probabilities_in_bin = bin_probabilities[index]
        outcomes_in_bin = bin_outcomes[index]
        count = len(probabilities_in_bin)
        result.append(
            ReliabilityBin(
                lower=lower,
                upper=upper,
                count=count,
                mean_probability=math.fsum(probabilities_in_bin) / count if count else None,
                observed_frequency=math.fsum(outcomes_in_bin) / count if count else None,
            )
        )
    return result
