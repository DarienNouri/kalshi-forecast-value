"""Frozen replication: chronological selection, OOF blending, and full-refit inference.

This module reads no files and performs no collection. Calendar multiplicities retain
the original chronology throughout every bootstrap fit; failed attempts are counted,
never replaced. Development summaries describe the final refit and are in-sample;
candidate validation records retain the separately frozen validation predictions.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from collections.abc import Callable, Iterator, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from itertools import islice
from typing import Any

import numpy as np

from .evaluation import log_loss, multiclass_brier, reliability_bins
from .historical_bootstrap import (
    condition_support,
    sample_calendar_weights,
    summarize_percentile_intervals,
    weighted_high_minus_low,
    weighted_median,
)
from .historical_evaluation import _validated_cases
from .historical_models import (
    OofPrediction,
    OofResult,
    PublicModelFit,
    fit_public_model,
    monthly_oof,
    normal_bin_probabilities,
)
from .models import assert_simplex

METHODS = ("public_only", "market_only", "equal_blend", "trained_blend", "p0_reference")
METRICS = ("brier", "log_loss")
DateWeights = Mapping[str, int]


def _semantic_hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _instant(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.utcoffset() is None:
        raise ValueError("replication cutoff requires a timezone")
    return parsed.astimezone(UTC)


def _available(row: Mapping[str, Any], cutoff: str) -> bool:
    instant = _instant(cutoff)
    value = row.get("outcome_available_at")
    return (
        isinstance(value, str)
        and row["outcome_date"] < instant.date().isoformat()
        and _instant(value) <= instant
    )


def _frequency(day: str, weights: DateWeights | None) -> int:
    return 1 if weights is None else weights.get(day, 0)


def _case_weights(rows: Sequence[dict], weights: DateWeights | None) -> list[float]:
    horizons: dict[str, set[int]] = {}
    for row in rows:
        horizons.setdefault(row["outcome_date"], set()).add(row["horizon_hours"])
    return [
        _frequency(row["outcome_date"], weights) / len(horizons[row["outcome_date"]])
        for row in rows
    ]


def _mean(values: Sequence[float], weights: Sequence[float]) -> float:
    total = math.fsum(weights)
    if total <= 0:
        raise ValueError("replication statistic requires positive eligible weight")
    result = math.fsum(v * w for v, w in zip(values, weights, strict=True)) / total
    if not math.isfinite(result):
        raise ValueError("replication statistic is not finite")
    return result


def _candidate(candidate: str) -> dict[str, Any]:
    if candidate == "p0":
        return {"spec": "p0"}
    if candidate.startswith("p1_lambda_"):
        return {"spec": "p1", "ridge_lambda": float(candidate.removeprefix("p1_lambda_"))}
    raise ValueError("unknown replication model candidate")


def _choose_candidate(scores: Mapping[str, float], protocol: dict) -> str:
    if not scores or any(not math.isfinite(score) for score in scores.values()):
        raise ValueError("all replication candidate scores must be finite")
    best = min(scores.values())
    tolerance = protocol["models"]["tie_tolerance"]
    for candidate in protocol["models"]["tie_order"]:
        if candidate in scores and scores[candidate] <= best + tolerance:
            return candidate
    raise ValueError("replication tie order omits all candidates")


def _oof(
    rows: list[dict],
    candidate: str,
    protocol: dict,
    weights: DateWeights | None,
    original_support: bool,
) -> OofResult:
    settings = protocol["oof"]
    return monthly_oof(
        rows,
        **_candidate(candidate),
        start_month=date.fromisoformat(settings["start_month"]),
        end_month=date.fromisoformat(settings["end_month"]),
        date_weights=weights,
        min_error_dates=protocol["models"]["min_original_error_dates"],
        require_original_support=original_support,
        std_floor_f=protocol["models"]["std_floor_f"],
    )


def _scales(
    fit: PublicModelFit,
    predictions: Sequence[OofPrediction],
    cutoff: str,
    protocol: dict,
    weights: DateWeights | None,
    original_support: bool,
) -> dict[int, float]:
    if fit.spec == "p0":
        assert fit.predictive_std_f is not None
        return {h: fit.predictive_std_f for h in protocol["collection"]["horizon_hours"]}
    result = {}
    for horizon in protocol["collection"]["horizon_hours"]:
        eligible = [
            prediction
            for prediction in predictions
            if prediction.horizon_hours == horizon
            and _available(prediction.to_dict(), cutoff)
            and _frequency(prediction.outcome_date, weights) > 0
        ]
        if (
            original_support
            and len({p.outcome_date for p in eligible})
            < protocol["models"]["min_original_error_dates"]
        ):
            raise ValueError(f"insufficient original OOF error dates for horizon {horizon}")
        rms = math.sqrt(
            _mean(
                [p.error_f**2 for p in eligible],
                [_frequency(p.outcome_date, weights) for p in eligible],
            )
        )
        result[horizon] = max(protocol["models"]["std_floor_f"], rms)
    return result


def _probabilities(fit: PublicModelFit, scales: Mapping[int, float], row: dict) -> list[float]:
    return normal_bin_probabilities(
        fit.predict_mean(row), scales[row["horizon_hours"]], row["bins"]
    )


def _brier(probabilities: Sequence[float], winner: int) -> float:
    return multiclass_brier({str(i): p for i, p in enumerate(probabilities)}, str(winner))


def _fit_blend(
    rows: Sequence[dict],
    probabilities: Mapping[str, Sequence[float]],
    date_weights: DateWeights | None = None,
) -> float:
    numerator, denominator = [], []
    weights = _case_weights(rows, date_weights)
    if math.fsum(weights) <= 0:
        raise ValueError("blend requires positive eligible OOF weight")
    for row, weight in zip(rows, weights, strict=True):
        public = probabilities[row["case_id"]]
        market = row["market_probabilities"]
        for index, (p, m) in enumerate(zip(public, market, strict=True)):
            difference = m - p
            numerator.append(weight * difference * (float(index == row["winner_index"]) - p))
            denominator.append(weight * difference**2)
    total = math.fsum(denominator)
    return min(1.0, max(0.0, math.fsum(numerator) / total)) if total else 0.0


@dataclass(frozen=True)
class ReplicationFit:
    selected_candidate: str
    public_model: PublicModelFit
    p0_reference_model: PublicModelFit
    predictive_std_by_horizon: dict[int, float]
    trained_market_weight: float
    disagreement_l1_median_by_horizon: dict[int, float]
    candidate_validation: tuple[dict, ...]
    oof: OofResult
    blend_training_case_ids: tuple[str, ...]
    selection_excluded_case_ids: tuple[str, ...]
    oof_excluded_case_ids: tuple[str, ...]

    def to_dict(self) -> dict:
        return {
            "selected_candidate": self.selected_candidate,
            "validation_fit_at": self.candidate_validation[0]["fit"]["cutoff_at"],
            "final_fit_at": self.public_model.cutoff_at,
            "public_model": self.public_model.to_dict(),
            "p0_reference_model": self.p0_reference_model.to_dict(),
            "predictive_std_by_horizon": {
                str(h): v for h, v in self.predictive_std_by_horizon.items()
            },
            "trained_market_weight": self.trained_market_weight,
            "disagreement_l1_median_by_horizon": {
                str(h): v for h, v in self.disagreement_l1_median_by_horizon.items()
            },
            "oof": self.oof.to_dict(),
            "blend_training_case_ids": list(self.blend_training_case_ids),
            "selection_excluded_case_ids": list(self.selection_excluded_case_ids),
            "oof_excluded_case_ids": list(self.oof_excluded_case_ids),
            "label_cutoff_policy": "outcome date before cutoff date and label available by cutoff",
        }


def _fit_pipeline(
    rows: list[dict],
    protocol: dict,
    date_weights: DateWeights | None = None,
    *,
    original_support: bool = True,
) -> ReplicationFit:
    final_cutoff = protocol["final_fit_at"]
    validation_all = [r for r in rows if r["split"] == "validation"]
    validation = [r for r in validation_all if _available(r, final_cutoff)]
    validation_weights = _case_weights(validation, date_weights)
    candidates, oof_results, scores = [], {}, {}
    for candidate in protocol["models"]["candidates"]:
        oof = _oof(rows, candidate, protocol, date_weights, original_support)
        oof_results[candidate] = oof
        fit = fit_public_model(
            rows,
            **_candidate(candidate),
            cutoff_at=protocol["validation_fit_at"],
            date_weights=date_weights,
            std_floor_f=protocol["models"]["std_floor_f"],
        )
        scales = _scales(
            fit,
            oof.predictions,
            protocol["validation_fit_at"],
            protocol,
            date_weights,
            original_support,
        )
        predictions = [_probabilities(fit, scales, row) for row in validation]
        losses = [
            _brier(p, row["winner_index"]) for row, p in zip(validation, predictions, strict=True)
        ]
        score = _mean(losses, validation_weights)
        scores[candidate] = score
        candidates.append(
            {
                "candidate": candidate,
                "mean_brier": score,
                **_counts(validation),
                "fit": fit.to_dict(),
                "predictive_std_by_horizon": {str(h): v for h, v in scales.items()},
                "case_ids": [r["case_id"] for r in validation],
                "case_brier": dict(zip([r["case_id"] for r in validation], losses, strict=True)),
                "label_cutoff_at": final_cutoff,
                "prediction_policy": "parameters and scale fixed before validation",
            }
        )
    selected = _choose_candidate(scores, protocol)
    fit = fit_public_model(
        rows,
        **_candidate(selected),
        cutoff_at=final_cutoff,
        date_weights=date_weights,
        std_floor_f=protocol["models"]["std_floor_f"],
    )
    p0 = (
        fit
        if selected == "p0"
        else fit_public_model(
            rows,
            spec="p0",
            cutoff_at=final_cutoff,
            date_weights=date_weights,
            std_floor_f=protocol["models"]["std_floor_f"],
        )
    )
    selected_oof = oof_results[selected]
    scales = _scales(
        fit, selected_oof.predictions, final_cutoff, protocol, date_weights, original_support
    )
    lookup = {r["case_id"]: r for r in rows}
    probability_start = protocol["oof"]["probabilities_start_month"]
    probability_candidates = [
        p
        for p in selected_oof.predictions
        if p.outcome_date >= probability_start and p.probabilities is not None
    ]
    eligible_predictions = [
        p for p in probability_candidates if _available(p.to_dict(), final_cutoff)
    ]
    blend_rows = [lookup[p.case_id] for p in eligible_predictions]
    probabilities = {p.case_id: p.probabilities for p in eligible_predictions}
    blend = _fit_blend(blend_rows, probabilities, date_weights)
    thresholds = {}
    for horizon in protocol["collection"]["horizon_hours"]:
        subset = [p for p in eligible_predictions if p.horizon_hours == horizon]
        disagreements = []
        for prediction in subset:
            distance = math.fsum(
                abs(p - m)
                for p, m in zip(
                    prediction.probabilities,
                    lookup[prediction.case_id]["market_probabilities"],
                    strict=True,
                )
            )
            disagreements.append(distance)
        thresholds[horizon] = weighted_median(
            disagreements,
            [_frequency(p.outcome_date, date_weights) for p in subset],
        )
    # omit labels that were unavailable at prediction time from fitted residuals
    eligible_oof = OofResult(
        selected_oof.spec,
        selected_oof.start_month,
        selected_oof.end_month,
        selected_oof.fits,
        tuple(p for p in selected_oof.predictions if _available(p.to_dict(), final_cutoff)),
    )
    return ReplicationFit(
        selected,
        fit,
        p0,
        scales,
        blend,
        thresholds,
        tuple(candidates),
        eligible_oof,
        tuple(r["case_id"] for r in blend_rows),
        tuple(r["case_id"] for r in validation_all if not _available(r, final_cutoff)),
        tuple(
            p.case_id for p in probability_candidates if not _available(p.to_dict(), final_cutoff)
        ),
    )


def _score_rows(rows: list[dict], fit: ReplicationFit, protocol: dict) -> list[dict]:
    scored = []
    p0_std = fit.p0_reference_model.predictive_std_f
    assert p0_std is not None
    training = set(fit.public_model.training_case_ids)
    for row in rows:
        public = _probabilities(fit.public_model, fit.predictive_std_by_horizon, row)
        market = row["market_probabilities"]
        weight = fit.trained_market_weight
        vectors = {
            "public_only": public,
            "market_only": list(market),
            "equal_blend": [(p + m) / 2 for p, m in zip(public, market, strict=True)],
            "trained_blend": [
                (1 - weight) * p + weight * m for p, m in zip(public, market, strict=True)
            ],
            "p0_reference": normal_bin_probabilities(
                fit.p0_reference_model.predict_mean(row), p0_std, row["bins"]
            ),
        }
        scores = {}
        for method, values in vectors.items():
            vector = {str(i): p for i, p in enumerate(values)}
            scores[method] = {
                "brier": multiclass_brier(vector, str(row["winner_index"])),
                "log_loss": log_loss(
                    vector,
                    str(row["winner_index"]),
                    epsilon=protocol["metrics"]["log_loss_epsilon"],
                ),
            }
        disagreement = math.fsum(abs(p - m) for p, m in zip(public, market, strict=True))
        scored.append(
            {
                **row,
                "probabilities": vectors,
                "scores": scores,
                "public_mean_f": fit.public_model.predict_mean(row),
                "public_std_f": fit.predictive_std_by_horizon[row["horizon_hours"]],
                "disagreement_l1": disagreement,
                "disagreement_group": "high"
                if disagreement > fit.disagreement_l1_median_by_horizon[row["horizon_hours"]]
                else "low",
                "used_for_fit": row["case_id"] in training,
                "prediction_role": "held_out_fixed_final_fit"
                if row["split"] == "holdout"
                else "development_refit_in_sample",
            }
        )
    return scored


def _counts(rows: Sequence[dict]) -> dict[str, int]:
    return {
        "cases": len(rows),
        "events": len({r["event_id"] for r in rows}),
        "dates": len({r["outcome_date"] for r in rows}),
    }


def _validate_inventory(panel: dict, rows: list[dict], protocol: dict) -> dict[str, int]:
    collection = protocol["collection"]
    first = date.fromisoformat(collection["start_date"])
    last = date.fromisoformat(collection["end_date"])
    days = (last - first).days + 1
    expected = set()
    for idx in range(days):
        day = (first + timedelta(days=idx)).isoformat()
        for horizon in collection["horizon_hours"]:
            expected.add((day, horizon))
    exclusions = panel.get("exclusions")
    if not isinstance(exclusions, list):
        raise ValueError("replication exclusions must be an inventory list")
    slots = Counter((r["outcome_date"], r["horizon_hours"]) for r in rows)
    for exclusion in exclusions:
        if (
            not isinstance(exclusion, dict)
            or not isinstance(exclusion.get("outcome_date"), str)
            or type(exclusion.get("horizon_hours")) is not int
            or not isinstance(exclusion.get("reason"), str)
            or not exclusion["reason"]
        ):
            raise ValueError("replication exclusion requires date, horizon, and reason")
        slots[(exclusion["outcome_date"], exclusion["horizon_hours"])] += 1
    if set(slots) != expected or any(count != 1 for count in slots.values()):
        raise ValueError("replication inventory must contain each requested date/horizon once")
    return {
        "requested_dates": days,
        "potential_cases": len(expected),
        "eligible_cases": len(rows),
        "excluded_cases": len(exclusions),
    }


def _point_summaries(rows: list[dict], protocol: dict) -> list[dict]:
    summaries = []
    for split in ("train", "validation", "holdout"):
        for horizon in protocol["collection"]["horizon_hours"]:
            subset = [r for r in rows if r["split"] == split and r["horizon_hours"] == horizon]
            if not subset:
                continue
            weights = _case_weights(subset, None)
            calibration = {}
            for method in METHODS:
                probabilities, outcomes = [], []
                for row in subset:
                    probabilities.extend(row["probabilities"][method])
                    outcomes.extend(i == row["winner_index"] for i in range(len(row["bins"])))
                calibration[method] = reliability_bins(
                    probabilities, outcomes, edges=tuple(i / 10 for i in range(11))
                )
            scores = {}
            for method in METHODS:
                scores[method] = {}
                for metric in METRICS:
                    values = [row["scores"][method][metric] for row in subset]
                    scores[method][metric] = _mean(values, weights)
            summaries.append(
                {
                    "split": split,
                    "horizon_hours": horizon,
                    **_counts(subset),
                    "scores": scores,
                    "calibration": calibration,
                    "paired_comparisons": [],
                    "conditions": {},
                    "prediction_role": subset[0]["prediction_role"],
                }
            )
    return summaries


def _statistics(
    rows: list[dict], protocol: dict, weights: DateWeights | None
) -> dict[str, float | None]:
    result = {}
    for horizon in protocol["collection"]["horizon_hours"]:
        subset = [r for r in rows if r["split"] == "holdout" and r["horizon_hours"] == horizon]
        frequencies = _case_weights(subset, weights)
        for method in METHODS[1:]:
            for metric in METRICS:
                differences = [
                    r["scores"][method][metric] - r["scores"]["public_only"][metric] for r in subset
                ]
                result[f"{horizon}|{method}|{metric}"] = _mean(differences, frequencies)
        differences = [
            r["scores"]["trained_blend"]["brier"] - r["scores"]["public_only"]["brier"]
            for r in subset
        ]
        groups = [r["disagreement_group"] for r in subset]
        for group in ("low", "high"):
            selected = [
                (v, w)
                for v, w, g in zip(differences, frequencies, groups, strict=True)
                if g == group
            ]
            try:
                result[f"{horizon}|condition|{group}"] = _mean(
                    [v for v, _ in selected], [w for _, w in selected]
                )
            except ValueError:
                result[f"{horizon}|condition|{group}"] = None
        try:
            result[f"{horizon}|condition|interaction"] = weighted_high_minus_low(
                differences, groups, frequencies
            )
        except ValueError:
            result[f"{horizon}|condition|interaction"] = None
    return result


def _interval_record(values: list[float | None], protocol: dict, supported: bool = True) -> dict:
    settings = protocol["bootstrap"]
    summary = summarize_percentile_intervals(
        values,
        attempted=settings["attempts"],
        levels=settings["interval_levels"],
        min_valid_fraction=settings["min_valid_fraction"],
    ).to_dict()
    intervals = {item["confidence"]: item for item in summary["intervals"]}
    reason = intervals[.95]["unavailable_reason"]
    if not supported:
        reason = "insufficient_original_condition_group_support"
        for interval in summary["intervals"]:
            interval["interval"] = None
            interval["unavailable_reason"] = reason
    return {
        "ci95": intervals[.95]["interval"] if reason is None else None,
        "ci975": intervals[.975]["interval"] if reason is None else None,
        "valid_replicates": summary["valid"],
        "undefined_replicates": summary["failed"],
        "interval_unavailable_reason": reason,
        "interval_support": summary,
    }


def _bootstrap_summary(
    point_rows: list[dict],
    values: Mapping[str, list[float | None]],
    protocol: dict,
    *,
    block: int,
    selection_counts: Counter,
    errors: Counter,
    mode: str,
) -> dict:
    point = _statistics(point_rows, protocol, None)
    comparisons, conditions = [], []
    settings = protocol["conditions"]
    anchor = date.fromisoformat(protocol["collection"]["validation_end_date"]) + timedelta(days=1)
    for horizon in protocol["collection"]["horizon_hours"]:
        subset = [
            r for r in point_rows if r["split"] == "holdout" and r["horizon_hours"] == horizon
        ]
        support = condition_support(
            [r["outcome_date"] for r in subset],
            [r["disagreement_group"] for r in subset],
            anchor=anchor,
            block_days=settings["support_block_days"],
            min_dates=settings["min_observed_group_dates"],
            min_blocks=settings["min_observed_group_blocks"],
        )
        for method in METHODS[1:]:
            for metric in METRICS:
                key = f"{horizon}|{method}|{metric}"
                comparisons.append(
                    {
                        "horizon_hours": horizon,
                        "method": method,
                        "reference": "public_only",
                        "metric": metric,
                        "mean_difference": point[key],
                        "primary": method == "trained_blend" and metric == "brier",
                        **_interval_record(values[key], protocol),
                    }
                )
        groups = {}
        for group in ("low", "high"):
            key = f"{horizon}|condition|{group}"
            group_rows = [r for r in subset if r["disagreement_group"] == group]
            groups[group] = {
                **_counts(group_rows),
                **support["groups"][group],
                "paired_comparisons": [
                    {
                        "method": "trained_blend",
                        "reference": "public_only",
                        "metric": "brier",
                        "mean_difference": point[key],
                        **_interval_record(values[key], protocol, support["supported"]),
                    }
                ],
            }
        key = f"{horizon}|condition|interaction"
        conditions.append(
            {
                "horizon_hours": horizon,
                "support": support,
                "groups": groups,
                "interactions": [
                    {
                        "method": "trained_blend",
                        "metric": "brier",
                        "contrast": "high_minus_low_of_trained_minus_public_brier",
                        "mean_difference": point[key],
                        **_interval_record(values[key], protocol, support["supported"]),
                    }
                ],
            }
        )
    return {
        "block_days": block,
        "attempted": protocol["bootstrap"]["attempts"],
        "valid": protocol["bootstrap"]["attempts"] - sum(errors.values()),
        "failed": sum(errors.values()),
        "errors": dict(sorted(errors.items())),
        "selection_counts": dict(sorted(selection_counts.items())),
        "mode": mode,
        "comparisons": comparisons,
        "conditions": conditions,
    }


_WORKER_INPUTS: tuple[list[dict], dict, list[dict]] | None = None


def _initialize_worker(rows: list[dict], protocol: dict, point_rows: list[dict]) -> None:
    global _WORKER_INPUTS
    _WORKER_INPUTS = rows, protocol, point_rows


def _replicate(inputs: tuple[list[dict], dict, list[dict]], weights: DateWeights) -> dict:
    rows, protocol, point_rows = inputs
    try:
        fit = _fit_pipeline(rows, protocol, weights, original_support=False)
        heldout = [r for r in rows if r["split"] == "holdout"]
        values = _statistics(_score_rows(heldout, fit, protocol), protocol, weights)
        result = {"candidate": fit.selected_candidate, "statistics": values, "error": None}
    except (ValueError, ArithmeticError, np.linalg.LinAlgError) as exc:
        result = {"candidate": None, "statistics": {}, "error": f"{type(exc).__name__}: {exc}"}
    try:
        result["fixed_statistics"] = _statistics(point_rows, protocol, weights)
        result["fixed_error"] = None
    except (ValueError, ArithmeticError) as exc:
        result["fixed_statistics"] = {}
        result["fixed_error"] = f"{type(exc).__name__}: {exc}"
    return result


def _worker_replicate(weights: DateWeights) -> dict:
    if _WORKER_INPUTS is None:
        raise RuntimeError("replication worker not initialized")
    return _replicate(_WORKER_INPUTS, weights)


def _parallel_draws(
    executor: ProcessPoolExecutor, weights: Iterator[dict[str, int]], workers: int
) -> Iterator[dict]:
    # Python 3.13 submits map inputs eagerly; batches limit memory use
    while batch := list(islice(weights, workers * 8)):
        yield from executor.map(_worker_replicate, batch, chunksize=4)


def _bootstrap(
    rows: list[dict],
    point_rows: list[dict],
    protocol: dict,
    *,
    workers: int,
    progress: Callable[[str], None] | None,
) -> dict:
    collection, settings = protocol["collection"], protocol["bootstrap"]
    first, train_end, validation_end, last = [
        date.fromisoformat(collection[key])
        for key in ("start_date", "train_end_date", "validation_end_date", "end_date")
    ]
    boundaries = [
        ("train", first, train_end),
        ("validation", train_end + timedelta(days=1), validation_end),
        ("holdout", validation_end + timedelta(days=1), last),
    ]
    keys = list(_statistics(point_rows, protocol, None))
    results = {}
    for block in [settings["block_days"], *settings["sensitivity_block_days"]]:
        rng = np.random.default_rng(settings["seed"])
        weights = (
            sample_calendar_weights(
                first, last, split_boundaries=boundaries, block_days=block, rng=rng
            )
            for _ in range(settings["attempts"])
        )
        values = {key: [] for key in keys}
        fixed_values = {key: [] for key in keys}
        choices, errors, fixed_errors = Counter(), Counter(), Counter()
        executor = None
        try:
            if workers > 1:
                executor = ProcessPoolExecutor(
                    max_workers=workers,
                    initializer=_initialize_worker,
                    initargs=(rows, protocol, point_rows),
                )
                draws = _parallel_draws(executor, weights, workers)
            else:
                draws = (_replicate((rows, protocol, point_rows), weight) for weight in weights)
            for index, result in enumerate(draws, 1):
                if result["error"]:
                    errors[result["error"]] += 1
                else:
                    choices[result["candidate"]] += 1
                if result["fixed_error"]:
                    fixed_errors[result["fixed_error"]] += 1
                for key in keys:
                    values[key].append(result["statistics"].get(key))
                    fixed_values[key].append(result["fixed_statistics"].get(key))
                if progress is not None and (
                    index == 1 or index % 100 == 0 or index == settings["attempts"]
                ):
                    progress(
                        f"{block}-day bootstrap: {index}/{settings['attempts']} attempts, "
                        f"{sum(errors.values())} failed"
                    )
        finally:
            if executor is not None:
                executor.shutdown(wait=True, cancel_futures=True)
        results[block] = _bootstrap_summary(
            point_rows,
            values,
            protocol,
            block=block,
            selection_counts=choices,
            errors=errors,
            mode="full_refit_including_selection_oof_scale_blend_conditions",
        )
        if block == settings["block_days"]:
            results["fixed_fit"] = _bootstrap_summary(
                point_rows,
                fixed_values,
                protocol,
                block=block,
                selection_counts=Counter(),
                errors=fixed_errors,
                mode="fixed_fit_holdout_resampling_only",
            )
    return {
        "primary": results[settings["block_days"]],
        "sensitivity_by_block_days": {
            str(block): results[block] for block in settings["sensitivity_block_days"]
        },
        "fixed_fit": results["fixed_fit"],
        "limitations": list(settings["limitations"]),
        "family_coverage": settings["primary_family"],
    }


def _project_simplex(values: Sequence[float]) -> list[float]:
    if not values or any(not math.isfinite(v) or v < 0 for v in values):
        raise ValueError("simplex projection requires finite nonnegative raw midpoints")
    ordered = sorted(values, reverse=True)
    cumulative, threshold = 0, 0
    for index, value in enumerate(ordered, 1):
        cumulative += value
        candidate = (cumulative - 1) / index
        if value > candidate:
            threshold = candidate
    projected = [max(0.0, value - threshold) for value in values]
    assert_simplex({str(i): v for i, v in enumerate(projected)})
    return projected


def _sensitivities(panel: dict, protocol: dict, primary: list[dict]) -> dict:
    result = {}
    label = protocol["sensitivities"]["label_substitution"]
    variants = {"label_substitution": deepcopy(panel), "normalization_projection": deepcopy(panel)}
    changed_ids = []
    for row in variants["label_substitution"]["cases"]:
        if row["outcome_date"] == label["outcome_date"]:
            if not _available(row, protocol["final_fit_at"]):
                raise ValueError("label sensitivity cannot change unavailable development labels")
            row["outcome_f"] = label["temperature_f"]
            changed_ids.append(row["case_id"])
    for name, variant in variants.items():
        try:
            if name == "normalization_projection":
                for row in variant["cases"]:
                    raw = [observation["probability"] for observation in row["market_observations"]]
                    row["market_probabilities"] = _project_simplex(raw)
            fit = _fit_pipeline(_validated_cases(variant, protocol["collection"]), protocol)
            rows = _score_rows(_validated_cases(variant, protocol["collection"]), fit, protocol)
            summaries = _point_summaries(rows, protocol)
            primary_by_key = {(s["split"], s["horizon_hours"]): s for s in primary}
            deltas = []
            for summary in summaries:
                if summary["split"] != "holdout":
                    continue
                horizon = summary["horizon_hours"]
                ref = primary_by_key[("holdout", horizon)]
                for method in METHODS:
                    diff = summary["scores"][method]["brier"] - ref["scores"][method]["brier"]
                    deltas.append(
                        {
                            "horizon_hours": horizon,
                            "method": method,
                            "brier_difference_from_primary": diff,
                        }
                    )
            result[name] = {
                "status": "evaluated",
                "inference": "prespecified point sensitivity; no separate bootstrap intervals",
                "fit": fit.to_dict(),
                "candidate_validation": list(fit.candidate_validation),
                "summaries": summaries,
                "holdout_deltas": deltas,
                "changed_case_ids": changed_ids
                if name == "label_substitution"
                else [r["case_id"] for r in variant["cases"]],
                "input_panel_sha256": _semantic_hash(variant),
            }
        except (ValueError, KeyError, TypeError) as exc:
            result[name] = {"status": "unavailable", "reason": f"{type(exc).__name__}: {exc}"}
    return result


def evaluate_replication(
    panel: dict, protocol: dict, *, workers: int = 1, progress: Callable[[str], None] | None = None
) -> dict:
    """Evaluate a complete frozen replication without changing inputs or writing files."""
    from .historical_replication_inputs import validate_replication_protocol

    frozen = validate_replication_protocol(protocol)
    if not isinstance(panel, dict) or panel.get("complete_requested_dates") is not True:
        raise ValueError("historical replication collection incomplete")
    if (
        panel.get("replication_schema_version") != "historical_replication_panel_v1"
        or not isinstance(panel.get("target_continuity"), dict)
        or panel["target_continuity"].get("analysis_ready") is not True
    ):
        raise ValueError("historical replication target continuity not ready")
    if type(workers) is not int or not 1 <= workers <= 8:
        raise ValueError("replication workers must be an integer from 1 to 8")
    rows = _validated_cases(panel, frozen["collection"])
    coverage = _validate_inventory(panel, rows, frozen)
    for split in ("train", "validation", "holdout"):
        if not any(row["split"] == split for row in rows):
            raise ValueError("replication requires eligible cases in every split")
    fit = _fit_pipeline(rows, frozen)
    scored = _score_rows(rows, fit, frozen)
    summaries = _point_summaries(scored, frozen)
    bootstrap = _bootstrap(rows, scored, frozen, workers=workers, progress=progress)
    for summary in summaries:
        if summary["split"] == "holdout":
            horizon = summary["horizon_hours"]
            summary["paired_comparisons"] = [
                r for r in bootstrap["primary"]["comparisons"] if r["horizon_hours"] == horizon
            ]
            summary["conditions"] = next(
                r for r in bootstrap["primary"]["conditions"] if r["horizon_hours"] == horizon
            )
    exclusions = deepcopy(panel.get("exclusions", []))
    return {
        "schema_version": "historical_replication_evaluation_v1",
        "data_origin": "empirical_historical",
        "analysis_status": "exploratory_historical_replication",
        "complete_requested_dates": True,
        "protocol": frozen,
        "input_panel_sha256": _semantic_hash(panel),
        "input_protocol_sha256": _semantic_hash(protocol),
        "protocol_sha256": _semantic_hash(frozen),
        "fit": fit.to_dict(),
        "candidate_validation": list(fit.candidate_validation),
        "cohort": _counts(scored),
        "coverage": coverage,
        "target_continuity": deepcopy(panel["target_continuity"]),
        "summaries": summaries,
        "case_results": scored,
        "exclusions": exclusions,
        "exclusion_reason_counts": dict(
            Counter(r.get("reason", "unspecified") for r in exclusions)
        ),
        "provenance": deepcopy(panel.get("provenance", {})),
        "bootstrap": bootstrap,
        "sensitivities": _sensitivities(panel, frozen, summaries),
        "interpretation": {
            "negative_paired_difference": "favors the named method over public_only",
            "development_scores": (
                "Final-refit train and validation summaries are in-sample; only "
                "candidate_validation uses the frozen prevalidation models."
            ),
            "calibration": (
                "Descriptive pooled bin probabilities; bins sharing an event "
                "are not independent observations."
            ),
            "label_cutoffs": (
                "Selection and all fitted OOF inputs exclude unknown or later labels "
                "and cutoff-date outcomes, including March31 validation labels."
            ),
            "limitations": [
                *frozen["collection"]["limitations"],
                *frozen["bootstrap"]["limitations"],
                "Ridge calibration does not add overnight information or erase "
                "the grid/station target mismatch.",
                "Condition support guards are not statistical power guarantees.",
                "Forecast skill is not trading performance; no executable-fill or causal claim.",
            ],
        },
    }
