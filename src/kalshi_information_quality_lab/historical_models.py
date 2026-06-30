"""Frozen public-weather models and chronological out-of-fold predictions."""

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime, time, timedelta
from typing import Any, Literal

import numpy as np

PublicModelSpec = Literal["p0", "p1"]
RIDGE_GRID = (.1, 1.0, 10.0)
CASE_WEIGHT_POLICY = "date multiplicity times 2 / distinct eligible horizons on outcome date"


def _instant(value: object, name: str) -> datetime:
    if isinstance(value, datetime):
        result = value
    elif isinstance(value, str):
        try:
            result = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError(f"{name} must be an ISO timestamp") from exc
    else:
        raise ValueError(f"{name} must be an ISO timestamp")
    if result.utcoffset() is None:
        raise ValueError(f"{name} must include a timezone")
    return result.astimezone(UTC)


def _day(value: object, name: str = "outcome_date") -> date:
    if not isinstance(value, str):
        raise ValueError(f"{name} must use YYYY-MM-DD")
    try:
        result = date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{name} must use YYYY-MM-DD") from exc
    if result.isoformat() != value:
        raise ValueError(f"{name} must use YYYY-MM-DD")
    return result


def _number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{name} must be finite numeric data")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite numeric data")
    return result


def _case_id(row: Mapping[str, Any], index: int) -> str:
    value = row.get("case_id", f"row-{index}")
    if not isinstance(value, str) or not value:
        raise ValueError("case_id must be a nonempty string")
    return value


def _multiplicity(day: str, date_weights: Mapping[str, float] | None) -> float:
    if date_weights is None:
        return 1.0
    value = date_weights.get(day, 0)
    result = _number(value, f"date weight for {day}")
    if result < 0:
        raise ValueError("date weights must be nonnegative")
    return result


def _validate_date_weights(date_weights: Mapping[str, float] | None) -> None:
    if date_weights is None:
        return
    for day, value in date_weights.items():
        _day(day, "date weight key")
        if _number(value, f"date weight for {day}") < 0:
            raise ValueError("date weights must be nonnegative")


@dataclass(frozen=True)
class PublicModelFit:
    """A fitted residual-mean model with immutable training lineage."""

    spec: PublicModelSpec
    cutoff_at: str
    feature_names: tuple[str, ...]
    coefficients: tuple[float, ...]
    max_t_mean_f: float | None
    max_t_std_f: float | None
    residual_sample_std_f: float
    predictive_std_f: float | None
    ridge_lambda: float | None
    training_case_ids: tuple[str, ...]
    excluded_case_ids: tuple[str, ...]
    training_dates: tuple[str, ...]
    total_frequency_weight: float
    case_weight_policy: str = CASE_WEIGHT_POLICY

    def predict_mean(self, row: Mapping[str, Any]) -> float:
        forecast = _number(row.get("forecast_f"), "forecast_f")
        if self.spec == "p0":
            return forecast + self.coefficients[0]
        outcome_day = _day(row.get("outcome_date"))
        horizon = row.get("horizon_hours")
        if type(horizon) is not int:
            raise ValueError("horizon_hours must be an integer")
        assert self.max_t_mean_f is not None and self.max_t_std_f is not None
        angle = 2 * math.pi * (outcome_day.timetuple().tm_yday - 1) / 365.25
        features = (
            1,
            float(horizon == 6),
            (forecast - self.max_t_mean_f) / self.max_t_std_f,
            math.sin(angle),
            math.cos(angle),
        )
        return forecast + math.fsum(
            coefficient * feature
            for coefficient, feature in zip(self.coefficients, features, strict=True)
        )

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        for key in ("feature_names", "coefficients", "training_case_ids", "excluded_case_ids"):
            result[key] = list(result[key])
        result["training_dates"] = list(result["training_dates"])
        return result


@dataclass(frozen=True)
class OofPrediction:
    case_id: str
    outcome_date: str
    horizon_hours: int
    cutoff_at: str
    mean_f: float
    error_f: float
    outcome_available_at: str | None
    std_f: float | None
    probabilities: tuple[float, ...] | None

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        if self.probabilities is not None:
            result["probabilities"] = list(self.probabilities)
        return result


@dataclass(frozen=True)
class OofResult:
    spec: PublicModelSpec
    start_month: str
    end_month: str
    fits: tuple[PublicModelFit, ...]
    predictions: tuple[OofPrediction, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "spec": self.spec,
            "start_month": self.start_month,
            "end_month": self.end_month,
            "fits": [fit.to_dict() for fit in self.fits],
            "predictions": [prediction.to_dict() for prediction in self.predictions],
        }


def _eligible_training(
    rows: Sequence[Mapping[str, Any]],
    cutoff: datetime,
    date_weights: Mapping[str, float] | None,
) -> tuple[list[tuple[Mapping[str, Any], str, float]], tuple[str, ...]]:
    candidates: list[tuple[Mapping[str, Any], str, float]] = []
    excluded: list[str] = []
    for index, row in enumerate(rows):
        identity = _case_id(row, index)
        outcome_day = _day(row.get("outcome_date"))
        if type(row.get("horizon_hours")) is not int:
            raise ValueError("horizon_hours must be an integer")
        multiplicity = _multiplicity(outcome_day.isoformat(), date_weights)
        available = row.get("outcome_available_at")
        try:
            availability_eligible = _instant(available, "outcome_available_at") <= cutoff
        except ValueError:
            availability_eligible = False
        if outcome_day >= cutoff.date() or not availability_eligible or multiplicity <= 0:
            excluded.append(identity)
            continue
        candidates.append((row, identity, multiplicity))
    candidates.sort(
        key=lambda item: (
            str(item[0]["outcome_date"]),
            str(item[0].get("event_id", "")),
            int(item[0]["horizon_hours"]),
            item[1],
        )
    )
    return candidates, tuple(sorted(excluded))


def fit_public_model(
    rows: Sequence[Mapping[str, Any]],
    *,
    spec: PublicModelSpec,
    cutoff_at: datetime | str,
    ridge_lambda: float | None = None,
    date_weights: Mapping[str, float] | None = None,
    std_floor_f: float = 1,
) -> PublicModelFit:
    """Fit P0 or P1 using only labels available before an aware UTC cutoff."""
    cutoff = _instant(cutoff_at, "cutoff_at")
    floor = _number(std_floor_f, "std_floor_f")
    if floor <= 0:
        raise ValueError("std_floor_f must be positive")
    if spec not in ("p0", "p1"):
        raise ValueError("spec must be p0 or p1")
    if spec == "p1" and ridge_lambda not in RIDGE_GRID:
        raise ValueError(f"ridge_lambda must be one of {RIDGE_GRID}")
    if spec == "p0" and ridge_lambda is not None:
        raise ValueError("ridge_lambda applies only to p1")
    _validate_date_weights(date_weights)
    eligible, excluded = _eligible_training(rows, cutoff, date_weights)
    if not eligible:
        raise ValueError("No positive-weight labels are available before fit cutoff")

    horizons_by_date: dict[str, set[int]] = {}
    for row, _, _ in eligible:
        day = str(row["outcome_date"])
        horizon = row.get("horizon_hours")
        if type(horizon) is not int:
            raise ValueError("horizon_hours must be an integer")
        horizons_by_date.setdefault(day, set()).add(horizon)
    weights = np.asarray(
        [
            multiplicity * 2 / len(horizons_by_date[str(row["outcome_date"])])
            for row, _, multiplicity in eligible
        ],
        dtype=float,
    )
    total_weight = float(weights.sum())
    if not math.isfinite(total_weight) or total_weight <= 0:
        raise ValueError("Public model requires positive frequency weight")
    if spec == "p0" and total_weight <= 1:
        raise ValueError("P0 requires frequency weight greater than one")
    forecasts = np.asarray(
        [_number(row.get("forecast_f"), "forecast_f") for row, _, _ in eligible], dtype=float
    )
    outcomes = np.asarray(
        [_number(row.get("outcome_f"), "outcome_f") for row, _, _ in eligible], dtype=float
    )
    residuals = outcomes - forecasts

    max_t_mean: float | None = None
    max_t_std: float | None = None
    if spec == "p0":
        coefficients = np.asarray([np.average(residuals, weights=weights)], dtype=float)
        fitted = np.full(len(eligible), coefficients[0])
        feature_names = ("intercept",)
    else:
        max_t_mean = float(np.average(forecasts, weights=weights))
        variance = float(np.average((forecasts - max_t_mean) ** 2, weights=weights))
        max_t_std = math.sqrt(variance) if variance > 0 else 1.0
        dates = [_day(row.get("outcome_date")) for row, _, _ in eligible]
        angles = np.asarray(
            [2 * math.pi * (item.timetuple().tm_yday - 1) / 365.25 for item in dates]
        )
        design = np.column_stack(
            (
                np.ones(len(eligible)),
                [float(row.get("horizon_hours") == 6) for row, _, _ in eligible],
                (forecasts - max_t_mean) / max_t_std,
                np.sin(angles),
                np.cos(angles),
            )
        )
        weighted_design = design * weights[:, None]
        penalty = np.diag([0, 1, 1, 1, 1])
        assert ridge_lambda is not None
        coefficients = np.linalg.solve(
            design.T @ weighted_design + total_weight * ridge_lambda * penalty,
            design.T @ (weights * residuals),
        )
        fitted = design @ coefficients
        feature_names = (
            "intercept",
            "horizon_6h",
            "max_t_standardized",
            "annual_sin",
            "annual_cos",
        )

    sample_variance = (
        float(np.sum(weights * (residuals - fitted) ** 2) / (total_weight - 1))
        if total_weight > 1
        else 0
    )
    residual_std = math.sqrt(max(0, sample_variance))
    if not all(math.isfinite(float(value)) for value in coefficients) or not math.isfinite(
        residual_std
    ):
        raise ValueError("Public model fit is not finite")
    return PublicModelFit(
        spec=spec,
        cutoff_at=cutoff.isoformat(),
        feature_names=feature_names,
        coefficients=tuple(float(value) for value in coefficients),
        max_t_mean_f=max_t_mean,
        max_t_std_f=max_t_std,
        residual_sample_std_f=residual_std,
        predictive_std_f=max(floor, residual_std) if spec == "p0" else None,
        ridge_lambda=ridge_lambda,
        training_case_ids=tuple(identity for _, identity, _ in eligible),
        excluded_case_ids=excluded,
        training_dates=tuple(sorted(horizons_by_date)),
        total_frequency_weight=total_weight,
    )


def normal_bin_probabilities(
    mean_f: float, std_f: float, bins: Sequence[Mapping[str, Any]]
) -> list[float]:
    """Integrate a Normal distribution over integer bins using half-degree edges."""
    mean = _number(mean_f, "mean_f")
    std = _number(std_f, "std_f")
    if std <= 0:
        raise ValueError("std_f must be positive")
    distribution = statistics.NormalDist(mean, std)
    result = []
    for bin_ in bins:
        lower = bin_.get("lower")
        upper = bin_.get("upper")
        if lower is not None and type(lower) is not int:
            raise ValueError("bin lower bounds must be integers or None")
        if upper is not None and type(upper) is not int:
            raise ValueError("bin upper bounds must be integers or None")
        result.append(
            (1.0 if upper is None else distribution.cdf(upper + .5))
            - (0.0 if lower is None else distribution.cdf(lower - .5))
        )
    if not result or not math.isclose(math.fsum(result), 1, abs_tol=1e-12):
        raise ValueError("bins must form a complete probability partition")
    return result


def _next_month(month: date) -> date:
    return date(month.year + (month.month == 12), month.month % 12 + 1, 1)


def _month_cutoff(month: date) -> datetime:
    return datetime.combine(month - timedelta(days=1), time(17), UTC)


def monthly_oof(
    rows: Sequence[Mapping[str, Any]],
    *,
    spec: PublicModelSpec,
    start_month: date,
    end_month: date,
    ridge_lambda: float | None = None,
    date_weights: Mapping[str, float] | None = None,
    min_error_dates: int = 20,
    require_original_support: bool = True,
    std_floor_f: float = 1,
) -> OofResult:
    """Build expanding monthly mean predictions and August-onward probabilities."""
    if start_month.day != 1 or end_month.day != 1 or end_month < start_month:
        raise ValueError("OOF month bounds must be ordered first-of-month dates")
    if type(min_error_dates) is not int or min_error_dates < 1:
        raise ValueError("min_error_dates must be a positive integer")
    floor = _number(std_floor_f, "std_floor_f")
    if floor <= 0:
        raise ValueError("std_floor_f must be positive")
    _validate_date_weights(date_weights)
    source = list(rows)
    fits: list[PublicModelFit] = []
    predictions: list[OofPrediction] = []
    month = start_month
    while month <= end_month:
        cutoff = _month_cutoff(month)
        fit = fit_public_model(
            source,
            spec=spec,
            cutoff_at=cutoff,
            ridge_lambda=ridge_lambda,
            date_weights=date_weights,
            std_floor_f=floor,
        )
        fits.append(fit)
        targets = [
            (index, candidate)
            for index, candidate in enumerate(source)
            if _day(candidate.get("outcome_date")).replace(day=1) == month
            and _multiplicity(str(candidate["outcome_date"]), date_weights) > 0
        ]
        if any(type(candidate.get("horizon_hours")) is not int for _, candidate in targets):
            raise ValueError("horizon_hours must be an integer")
        targets.sort(
            key=lambda item: (
                str(item[1]["outcome_date"]),
                str(item[1].get("event_id", "")),
                int(item[1]["horizon_hours"]),
                _case_id(item[1], item[0]),
            )
        )
        # cases in a month share a cutoff, so reuse the scale for each horizon
        monthly_scales: dict[int, float] = {}
        for index, candidate in targets:
            identity = _case_id(candidate, index)
            horizon = candidate.get("horizon_hours")
            if type(horizon) is not int:
                raise ValueError("horizon_hours must be an integer")
            mean = fit.predict_mean(candidate)
            error = _number(candidate.get("outcome_f"), "outcome_f") - mean
            std: float | None = None
            probabilities: tuple[float, ...] | None = None
            if month > start_month:
                if horizon in monthly_scales:
                    std = monthly_scales[horizon]
                elif spec == "p0":
                    assert fit.predictive_std_f is not None
                    std = fit.predictive_std_f
                else:
                    earlier = [
                        prediction
                        for prediction in predictions
                        if prediction.horizon_hours == horizon
                        and prediction.outcome_date < month.isoformat()
                        and prediction.outcome_available_at is not None
                        and _instant(prediction.outcome_available_at, "outcome_available_at")
                        <= cutoff
                    ]
                    distinct_dates = {prediction.outcome_date for prediction in earlier}
                    if require_original_support and len(distinct_dates) < min_error_dates:
                        raise ValueError(
                            f"Need at least {min_error_dates} distinct earlier OOF error dates "
                            f"for horizon {horizon}"
                        )
                    weighted_errors = [
                        (
                            prediction.error_f,
                            _multiplicity(prediction.outcome_date, date_weights),
                        )
                        for prediction in earlier
                    ]
                    total = math.fsum(weight for _, weight in weighted_errors)
                    if total <= 0:
                        raise ValueError("OOF scale requires positive eligible error weight")
                    rms = math.sqrt(
                        math.fsum(
                            error_value * error_value * weight
                            for error_value, weight in weighted_errors
                        )
                        / total
                    )
                    if not math.isfinite(rms):
                        raise ValueError("OOF scale must be finite")
                    std = max(floor, rms)
                monthly_scales[horizon] = std
                bins = candidate.get("bins")
                if not isinstance(bins, list):
                    raise ValueError("OOF probability rows require bins")
                probabilities = tuple(normal_bin_probabilities(mean, std, bins))
            availability = candidate.get("outcome_available_at")
            try:
                normalized_availability = _instant(availability, "outcome_available_at").isoformat()
            except ValueError:
                normalized_availability = None
            predictions.append(
                OofPrediction(
                    case_id=identity,
                    outcome_date=str(candidate["outcome_date"]),
                    horizon_hours=horizon,
                    cutoff_at=cutoff.isoformat(),
                    mean_f=mean,
                    error_f=error,
                    outcome_available_at=normalized_availability,
                    std_f=std,
                    probabilities=probabilities,
                )
            )
        month = _next_month(month)
    return OofResult(
        spec=spec,
        start_month=start_month.isoformat(),
        end_month=end_month.isoformat(),
        fits=tuple(fits),
        predictions=tuple(predictions),
    )
