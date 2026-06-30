"""Calendar bootstrap, interval accounting, and disagreement support helpers."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from functools import lru_cache
from numbers import Real
from typing import Any, Literal, Protocol

import numpy as np

SplitBoundary = tuple[str, date, date]
Group = Literal["low", "high"]


class IntegerGenerator(Protocol):
    def integers(self, low: int, high: int) -> Any: ...


@dataclass(frozen=True)
class _CalendarStratum:
    name: str
    year: int
    month: int
    days: tuple[date, ...]
    blocks: tuple[tuple[date, ...], ...]


def _calendar_days(start: date, end: date) -> tuple[date, ...]:
    return tuple(start + timedelta(days=offset) for offset in range((end - start).days + 1))


@lru_cache(maxsize=64)
def _calendar_strata(
    start: date,
    end: date,
    split_boundaries: tuple[SplitBoundary, ...],
    block_days: int,
) -> tuple[_CalendarStratum, ...]:
    if start > end:
        raise ValueError("start must not follow end")
    if type(block_days) is not int or block_days <= 0:
        raise ValueError("block_days must be a positive integer")
    for name, first, last in split_boundaries:
        if not isinstance(name, str) or not name or first > last:
            raise ValueError("split boundaries require a name and ordered dates")

    grouped: dict[tuple[int, str, int, int], list[date]] = defaultdict(list)
    for day in _calendar_days(start, end):
        containing = [
            (index, name)
            for index, (name, first, last) in enumerate(split_boundaries)
            if first <= day <= last
        ]
        if len(containing) != 1:
            raise ValueError("split boundaries must cover each calendar date exactly once")
        boundary_index, name = containing[0]
        grouped[(boundary_index, name, day.year, day.month)].append(day)

    strata = []
    for (_, name, year, month), values in sorted(grouped.items(), key=lambda item: item[1][0]):
        days = tuple(values)
        possible_starts = range(max(1, len(days) - block_days + 1))
        blocks = tuple(days[index : index + block_days] for index in possible_starts)
        strata.append(
            _CalendarStratum(
                name=name,
                year=year,
                month=month,
                days=days,
                blocks=blocks,
            )
        )
    return tuple(strata)


def sample_calendar_weights(
    start: date,
    end: date,
    *,
    split_boundaries: Sequence[SplitBoundary],
    block_days: int,
    rng: IntegerGenerator,
) -> dict[str, int]:
    """Draw split-by-month nonwrapping blocks over every calendar position."""
    boundaries = tuple(split_boundaries)
    strata = _calendar_strata(start, end, boundaries, block_days)
    weights = {day.isoformat(): 0 for day in _calendar_days(start, end)}
    for stratum in strata:
        sampled: list[date] = []
        while len(sampled) < len(stratum.days):
            block_index = int(rng.integers(0, len(stratum.blocks)))
            sampled.extend(stratum.blocks[block_index])
        for day in sampled[: len(stratum.days)]:
            weights[day.isoformat()] += 1
    return weights


@dataclass(frozen=True)
class BootstrapInterval:
    confidence: float
    attempted: int
    valid: int
    failed: int
    valid_fraction: float
    min_valid_fraction: float
    interval: tuple[float, float] | None
    unavailable_reason: str | None

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        if self.interval is not None:
            result["interval"] = list(self.interval)
        return result


@dataclass(frozen=True)
class BootstrapIntervalSummary:
    attempted: int
    valid: int
    failed: int
    valid_fraction: float
    min_valid_fraction: float
    intervals: tuple[BootstrapInterval, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "attempted": self.attempted,
            "valid": self.valid,
            "failed": self.failed,
            "valid_fraction": self.valid_fraction,
            "min_valid_fraction": self.min_valid_fraction,
            "intervals": [interval.to_dict() for interval in self.intervals],
        }


def _finite_replicates(values: Sequence[float | None]) -> list[float]:
    return [
        float(value)
        for value in values
        if isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(float(value))
    ]


def percentile_interval(
    values: Sequence[float | None],
    *,
    attempted: int | None = None,
    confidence: float = .95,
    min_valid_fraction: float = .95,
) -> BootstrapInterval:
    """Return a central percentile interval or a support-accounting suppression."""
    attempts = len(values) if attempted is None else attempted
    if type(attempts) is not int or attempts < 1 or attempts < len(values):
        raise ValueError("attempted must be a positive integer at least len(values)")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be in (0, 1)")
    if not 0 <= min_valid_fraction <= 1:
        raise ValueError("min_valid_fraction must be in [0, 1]")
    finite = _finite_replicates(values)
    valid = len(finite)
    fraction = valid / attempts
    interval: tuple[float, float] | None = None
    reason: str | None = None
    if valid == 0:
        reason = "no_finite_replicates"
    elif fraction < min_valid_fraction:
        reason = f"valid_fraction_below_{min_valid_fraction:g}"
    else:
        alpha = (1 - confidence) / 2
        quantiles = np.quantile(np.asarray(finite), [alpha, 1 - alpha])
        interval = (float(quantiles[0]), float(quantiles[1]))
    return BootstrapInterval(
        confidence=confidence,
        attempted=attempts,
        valid=valid,
        failed=attempts - valid,
        valid_fraction=fraction,
        min_valid_fraction=min_valid_fraction,
        interval=interval,
        unavailable_reason=reason,
    )


def summarize_percentile_intervals(
    values: Sequence[float | None],
    *,
    attempted: int | None = None,
    levels: Sequence[float] = (.95, .975),
    min_valid_fraction: float = .95,
) -> BootstrapIntervalSummary:
    """Summarize shared support and one or more central percentile intervals."""
    if not levels:
        raise ValueError("levels must not be empty")
    intervals = tuple(
        percentile_interval(
            values,
            attempted=attempted,
            confidence=level,
            min_valid_fraction=min_valid_fraction,
        )
        for level in levels
    )
    first = intervals[0]
    return BootstrapIntervalSummary(
        attempted=first.attempted,
        valid=first.valid,
        failed=first.failed,
        valid_fraction=first.valid_fraction,
        min_valid_fraction=min_valid_fraction,
        intervals=intervals,
    )


def weighted_median(values: Sequence[float], weights: Sequence[int]) -> float:
    """Frequency-weighted median, including the even-total central midpoint."""
    if len(values) != len(weights) or not values:
        raise ValueError("values and weights must be nonempty and equal length")
    pairs: list[tuple[float, int]] = []
    for value, weight in zip(values, weights, strict=True):
        number = float(value)
        frequency = float(weight)
        if not math.isfinite(number) or not math.isfinite(frequency) or not frequency.is_integer():
            raise ValueError("weighted median requires finite values and integer frequencies")
        if frequency < 0:
            raise ValueError("weighted median frequencies must be nonnegative")
        if frequency:
            pairs.append((number, int(frequency)))
    if not pairs:
        raise ValueError("weighted median requires positive total frequency")
    pairs.sort()
    total = sum(weight for _, weight in pairs)
    lower_index = (total - 1) // 2
    upper_index = total // 2

    def at(index: int) -> float:
        cumulative = 0
        for value, weight in pairs:
            cumulative += weight
            if cumulative > index:
                return value
        raise AssertionError("frequency index must be reachable")

    return (at(lower_index) + at(upper_index)) / 2


def classify_disagreement(values: Sequence[float], threshold: float) -> tuple[Group, ...]:
    """Classify equality with the low group; only strict exceedance is high."""
    boundary = float(threshold)
    if not math.isfinite(boundary):
        raise ValueError("disagreement threshold must be finite")
    groups: list[Group] = []
    for value in values:
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("disagreement values must be finite")
        groups.append("high" if number > boundary else "low")
    return tuple(groups)


def _as_date(value: date | str) -> date:
    if isinstance(value, date):
        return value
    try:
        parsed = date.fromisoformat(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("condition dates must use YYYY-MM-DD") from exc
    if parsed.isoformat() != value:
        raise ValueError("condition dates must use YYYY-MM-DD")
    return parsed


def condition_support(
    dates: Sequence[date | str],
    groups: Sequence[Group],
    *,
    anchor: date,
    block_days: int = 7,
    min_dates: int = 20,
    min_blocks: int = 5,
) -> dict[str, Any]:
    """Check original observed group support in anchored nonoverlapping blocks."""
    if len(dates) != len(groups):
        raise ValueError("condition dates and groups must have equal length")
    for name, value in (
        ("block_days", block_days),
        ("min_dates", min_dates),
        ("min_blocks", min_blocks),
    ):
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    grouped_dates: dict[Group, set[date]] = {"low": set(), "high": set()}
    for raw_day, group in zip(dates, groups, strict=True):
        if group not in ("low", "high"):
            raise ValueError("condition group must be low or high")
        day = _as_date(raw_day)
        if day < anchor:
            raise ValueError("condition date precedes calendar-block anchor")
        grouped_dates[group].add(day)
    results: dict[str, dict[str, int | bool]] = {}
    for group in ("low", "high"):
        unique_dates = grouped_dates[group]
        occupied = {(day - anchor).days // block_days for day in unique_dates}
        results[group] = {
            "dates": len(unique_dates),
            "calendar_blocks": len(occupied),
            "supported": len(unique_dates) >= min_dates and len(occupied) >= min_blocks,
        }
    return {
        "supported": all(bool(result["supported"]) for result in results.values()),
        "requirements": {
            "min_dates": min_dates,
            "min_calendar_blocks": min_blocks,
            "block_days": block_days,
            "anchor": anchor.isoformat(),
        },
        "groups": results,
    }


def weighted_high_minus_low(
    values: Sequence[float], groups: Sequence[Group], weights: Sequence[float]
) -> float:
    """Compute the direct weighted high-minus-low condition contrast."""
    if not (len(values) == len(groups) == len(weights)) or not values:
        raise ValueError("condition values, groups and weights must be nonempty and equal length")
    totals = {"low": 0.0, "high": 0.0}
    weighted = {"low": 0.0, "high": 0.0}
    for value, group, weight in zip(values, groups, weights, strict=True):
        if group not in ("low", "high"):
            raise ValueError("condition group must be low or high")
        number = float(value)
        frequency = float(weight)
        if not math.isfinite(number) or not math.isfinite(frequency) or frequency < 0:
            raise ValueError("condition values and weights must be finite with nonnegative weights")
        totals[group] += frequency
        weighted[group] += number * frequency
    if totals["low"] <= 0 or totals["high"] <= 0:
        raise ValueError("condition contrast requires positive weight in both groups")
    return weighted["high"] / totals["high"] - weighted["low"] / totals["low"]
