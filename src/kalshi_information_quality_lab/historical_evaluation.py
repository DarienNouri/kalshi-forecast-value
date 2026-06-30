"""Frozen chronological evaluation of a reconstructed historical weather panel.

Inputs are supplied by the collector; this module performs no network requests.
The output is exploratory empirical evidence, not a trading or execution backtest.
"""

import hashlib
import json
import math
import random
import statistics
from collections import Counter
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

from .evaluation import log_loss, multiclass_brier, reliability_bins
from .historical_inputs import (
    COLLECTION_PROTOCOL_FIELDS,
    HistoricalInputError,
    validate_protocol,
)
from .models import assert_simplex

METHODS = ("public_only", "market_only", "equal_blend", "trained_blend")
METRICS = ("brier", "log_loss")


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{name} must be finite numeric data")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _instant(value: Any, name: str) -> datetime:
    if not isinstance(value, str):
        raise HistoricalInputError(f"{name} must be an ISO timestamp")
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise HistoricalInputError(f"{name} must be an ISO timestamp") from exc
    if result.utcoffset() is None:
        raise HistoricalInputError(f"{name} must include a timezone")
    return result.astimezone(UTC)


def _identifier(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _protocol(raw: object) -> dict:
    return validate_protocol(raw)


def _split(day: str, protocol: dict) -> str:
    if day <= protocol["train_end_date"]:
        return "train"
    if day <= protocol["validation_end_date"]:
        return "validation"
    return "holdout"


def _validated_cases(panel: object, protocol: dict) -> list[dict]:
    if not isinstance(panel, dict):
        raise HistoricalInputError("historical panel must be an object")
    if panel.get("schema_version") != "historical_panel_v1":
        raise HistoricalInputError("Expected historical_panel_v1")
    if panel.get("data_origin") != "empirical_historical":
        raise HistoricalInputError("Expected empirical_historical origin")
    if not isinstance(panel.get("cases"), list) or not panel["cases"]:
        raise HistoricalInputError("Historical panel has no cases")
    if not isinstance(panel.get("protocol"), dict):
        raise HistoricalInputError("Historical panel is missing its collection protocol")
    collected = validate_protocol(panel["protocol"], collection=True)
    for key in COLLECTION_PROTOCOL_FIELDS:
        if collected.get(key) != protocol.get(key):
            raise HistoricalInputError(f"Collection protocol mismatch: {key}")
    rows, identities, event_horizons, event_definitions = [], set(), set(), {}
    for source in panel["cases"]:
        if not isinstance(source, dict):
            raise HistoricalInputError("historical case must be an object")
        row = dict(source)
        required = (
            "case_id",
            "event_id",
            "outcome_date",
            "horizon_hours",
            "as_of_at",
            "outcome_window_start_at",
            "outcome_window_end_at",
            "forecast_available_at",
            "forecast",
            "forecast_f",
            "outcome_f",
            "outcome_available_at",
            "bins",
            "market_probabilities",
        )
        missing = next((key for key in required if key not in row), None)
        if missing is not None:
            raise HistoricalInputError(f"missing required case timing/data field: {missing}")
        identity = _identifier(row["case_id"], "case_id")
        event = _identifier(row["event_id"], "event_id")
        day = row["outcome_date"]
        if not isinstance(day, str):
            raise HistoricalInputError("outcome_date must use YYYY-MM-DD")
        try:
            outcome_day = date.fromisoformat(day)
        except ValueError as exc:
            raise HistoricalInputError("outcome_date must use YYYY-MM-DD") from exc
        if outcome_day.isoformat() != day:
            raise HistoricalInputError("outcome_date must use YYYY-MM-DD")
        if not protocol["start_date"] <= day <= protocol["end_date"]:
            raise ValueError(f"{identity}: date outside frozen cohort")
        horizon = row["horizon_hours"]
        if type(horizon) is not int or horizon not in protocol["horizon_hours"]:
            raise ValueError(f"{identity}: undeclared horizon")
        if identity in identities or (event, horizon) in event_horizons:
            raise ValueError("Duplicate case or event/horizon")
        identities.add(identity)
        event_horizons.add((event, horizon))
        as_of = _instant(row["as_of_at"], "as_of_at")
        start = _instant(row["outcome_window_start_at"], "outcome_window_start_at")
        end = _instant(row["outcome_window_end_at"], "outcome_window_end_at")
        expected_start = datetime(
            outcome_day.year,
            outcome_day.month,
            outcome_day.day,
            protocol["outcome_window_start_utc_hour"],
            tzinfo=UTC,
        )
        if start != expected_start or end != start + timedelta(days=1):
            raise HistoricalInputError(f"{identity}: outcome window disagrees with protocol")
        if start - as_of != timedelta(hours=horizon):
            raise ValueError(f"{identity}: as_of and horizon disagree")

        forecast = row["forecast"]
        if not isinstance(forecast, dict):
            raise HistoricalInputError(f"{identity}: forecast timing must be an object")
        forecast_fields = ("issued_at", "available_at", "valid_start_at", "valid_end_at")
        missing_forecast = next((key for key in forecast_fields if key not in forecast), None)
        if missing_forecast is not None:
            raise HistoricalInputError(
                f"{identity}: missing forecast timing field: {missing_forecast}"
            )
        forecast_available = _instant(row["forecast_available_at"], "forecast_available_at")
        nested_available = _instant(forecast["available_at"], "forecast.available_at")
        if forecast_available != nested_available:
            raise HistoricalInputError(
                f"{identity}: forecast availability/available_at fields disagree"
            )
        for top_level, nested in (
            ("forecast_issued_at", "issued_at"),
            ("forecast_valid_start_at", "valid_start_at"),
            ("forecast_valid_end_at", "valid_end_at"),
        ):
            if top_level in row and _instant(row[top_level], top_level) != _instant(
                forecast[nested], f"forecast.{nested}"
            ):
                raise HistoricalInputError(f"{identity}: forecast timing fields disagree")
        issued = _instant(forecast["issued_at"], "forecast.issued_at")
        if issued > forecast_available or forecast_available > as_of:
            raise HistoricalInputError(
                f"{identity}: forecast timing later than prediction checkpoint"
            )
        valid_start = _instant(forecast["valid_start_at"], "forecast.valid_start_at")
        valid_end = _instant(forecast["valid_end_at"], "forecast.valid_end_at")
        expected_valid_start = datetime(
            outcome_day.year,
            outcome_day.month,
            outcome_day.day,
            protocol["forecast_valid_start_utc_hour"],
            tzinfo=UTC,
        )
        if valid_start != expected_valid_start or valid_end != valid_start + timedelta(
            hours=protocol["forecast_valid_hours"]
        ):
            raise HistoricalInputError(f"{identity}: forecast valid window disagrees with protocol")
        outcome_available = _instant(row["outcome_available_at"], "outcome_available_at")
        if outcome_available < end:
            raise HistoricalInputError(f"{identity}: outcome availability precedes outcome window")
        if (
            row.get("market_available_at") is not None
            and _instant(row["market_available_at"], "market_available_at") > as_of
        ):
            raise HistoricalInputError(
                f"{identity}: market_available_at later than prediction checkpoint"
            )
        if not isinstance(row["bins"], list):
            raise HistoricalInputError(f"{identity}: bins must be a list")
        if "market_observations" in row or collected.get("market_price_policy"):
            observations = row.get("market_observations")
            if not isinstance(observations, list) or len(observations) != len(row["bins"]):
                raise ValueError(f"{identity}: incomplete market observations")
            tolerance = _number(
                protocol.get("market_max_candle_age_minutes", 60), "market_max_candle_age_minutes"
            )
            if tolerance < 0:
                raise ValueError("market_max_candle_age_minutes must be nonnegative")
            for observation in observations:
                if not isinstance(observation, dict) or not isinstance(
                    observation.get("candle_end_at"), str
                ):
                    raise ValueError(f"{identity}: missing candle_end_at")
                candle_end = _instant(observation["candle_end_at"], "candle_end_at")
                age = (as_of - candle_end).total_seconds() / 60
                if age < 0 or age > tolerance:
                    raise ValueError(
                        f"{identity}: candle_end_at outside prediction checkpoint tolerance"
                    )
        row["forecast_f"] = _number(row["forecast_f"], "forecast_f")
        row["outcome_f"] = _number(row["outcome_f"], "outcome_f")
        if not row["outcome_f"].is_integer():
            raise ValueError("outcome_f must already be rounded to integer Fahrenheit")
        bins = row["bins"]
        if len(bins) < 2 or bins[0]["lower"] is not None or bins[-1]["upper"] is not None:
            raise ValueError("Bins require ordered complete lower and upper tails")
        tickers = [_identifier(b["ticker"], "ticker") for b in bins]
        if len(set(tickers)) != len(tickers):
            raise ValueError("Bin tickers must be unique")
        for i, bin_ in enumerate(bins):
            low, high = bin_["lower"], bin_["upper"]
            for bound in (low, high):
                if bound is not None and type(bound) is not int:
                    raise ValueError("Bin bounds must be integers or None")
            if (low is None and i != 0) or (high is None and i != len(bins) - 1):
                raise ValueError("Only exterior bins may have open bounds")
            if low is not None and high is not None and low > high:
                raise ValueError("Empty bin")
            if i and low != bins[i - 1]["upper"] + 1:
                raise ValueError("Bins overlap or have a gap")
        definition = (
            day,
            row["outcome_f"],
            tuple((b["ticker"], b["lower"], b["upper"]) for b in bins),
        )
        if event in event_definitions and event_definitions[event] != definition:
            raise ValueError(
                "An event must retain its date, outcome and bin definition across horizons"
            )
        event_definitions[event] = definition
        probabilities = row["market_probabilities"]
        if len(probabilities) != len(bins):
            raise ValueError("Market vector and bins have different lengths")
        assert_simplex(dict(zip(tickers, probabilities, strict=True)))
        row["market_probabilities"] = list(probabilities)
        row["split"] = _split(day, protocol)
        row["winner_index"] = next(
            i
            for i, b in enumerate(bins)
            if (b["lower"] is None or row["outcome_f"] >= b["lower"])
            and (b["upper"] is None or row["outcome_f"] <= b["upper"])
        )
        rows.append(row)
    return sorted(rows, key=lambda c: (c["outcome_date"], c["event_id"], c["horizon_hours"]))


def _public_probabilities(row: dict, mean: float, std: float) -> list[float]:
    distribution = statistics.NormalDist(row["forecast_f"] + mean, std)
    result = [
        (1.0 if b["upper"] is None else distribution.cdf(b["upper"] + .5))
        - (0.0 if b["lower"] is None else distribution.cdf(b["lower"] - .5))
        for b in row["bins"]
    ]
    assert_simplex({str(i): p for i, p in enumerate(result)})
    return result


def _fit(rows: list[dict], protocol: dict) -> dict:
    later = [r for r in rows if r["split"] != "train"]
    if not later:
        raise ValueError("Need validation or holdout cases to establish fit cutoff")
    earliest_prediction = min(_instant(r["as_of_at"], "as_of_at") for r in later)
    fit_at = (
        _instant(protocol["fit_at"], "fit_at") if protocol.get("fit_at") else earliest_prediction
    )
    if fit_at > earliest_prediction:
        raise ValueError("fit_at must precede every validation and holdout prediction")
    candidates = [r for r in rows if r["split"] == "train"]
    excluded = []
    for row in candidates:
        available = row.get("outcome_available_at")
        try:
            eligible = (
                isinstance(available, str) and _instant(available, "outcome_available_at") <= fit_at
            )
        except ValueError:
            eligible = False
        if not eligible:
            excluded.append(row["case_id"])
    training = [r for r in candidates if r["case_id"] not in excluded]
    if len({r["outcome_date"] for r in training}) < 2:
        raise ValueError("Need at least two training outcome dates available before fit")
    residuals = [r["outcome_f"] - r["forecast_f"] for r in training]
    mean = statistics.fmean(residuals)
    raw_std = statistics.stdev(residuals)
    std = max(raw_std, protocol["residual_std_floor_f"])
    numerator, denominator, disagreements = [], [], []
    for row in training:
        public = _public_probabilities(row, mean, std)
        market = row["market_probabilities"]
        difference = [m - p for p, m in zip(public, market, strict=True)]
        numerator.extend(
            d * (float(i == row["winner_index"]) - p)
            for i, (d, p) in enumerate(zip(difference, public, strict=True))
        )
        denominator.extend(d * d for d in difference)
        disagreements.append(math.fsum(abs(d) for d in difference))
    denominator_sum = math.fsum(denominator)
    weight = min(1.0, max(0.0, math.fsum(numerator) / denominator_sum)) if denominator_sum else 0.0
    missing_availability: list[str] = []
    return {
        "fit_at": fit_at.isoformat(),
        "training_cases": len(training),
        "training_events": len({r["event_id"] for r in training}),
        "training_dates": len({r["outcome_date"] for r in training}),
        "training_case_ids": [r["case_id"] for r in training],
        "excluded_training_case_ids": excluded,
        "training_exclusion_reason": "training_label_not_available_at_fit",
        "training_label_availability": (
            "unverified_for_some_fitted_cases"
            if missing_availability
            else "verified_for_all_fitted_cases"
        ),
        "training_missing_label_availability_case_ids": missing_availability,
        "residual_mean_f": mean,
        "residual_sample_std_f": raw_std,
        "residual_std_f": std,
        "residual_std_floor_f": protocol["residual_std_floor_f"],
        "residual_estimator": "pooled case residual sample mean and sample standard deviation",
        "trained_market_weight": weight,
        "blend_objective": "training multiclass Brier; closed-form convex weight clipped to [0,1]",
        "disagreement_l1_median": statistics.median(disagreements),
        "condition_rule": "high iff L1(market-public) > training median; otherwise low",
    }


def _score_cases(rows: list[dict], fit: dict, protocol: dict) -> list[dict]:
    results = []
    for row in rows:
        public = _public_probabilities(row, fit["residual_mean_f"], fit["residual_std_f"])
        market = row["market_probabilities"]
        weight = fit["trained_market_weight"]
        vectors = {
            "public_only": public,
            "market_only": market,
            "equal_blend": [(p + m) / 2 for p, m in zip(public, market, strict=True)],
            "trained_blend": [
                (1 - weight) * p + weight * m for p, m in zip(public, market, strict=True)
            ],
        }
        winner = str(row["winner_index"])
        scores = {}
        for method, probabilities in vectors.items():
            vector = {str(i): p for i, p in enumerate(probabilities)}
            scores[method] = {
                "brier": multiclass_brier(vector, winner),
                "log_loss": log_loss(vector, winner, epsilon=protocol["log_loss_epsilon"]),
            }
        disagreement = math.fsum(abs(m - p) for p, m in zip(public, market, strict=True))
        public_mean = row["forecast_f"] + fit["residual_mean_f"]
        distribution = statistics.NormalDist(public_mean, fit["residual_std_f"])
        public_interval = [distribution.inv_cdf(q) for q in (.05, .95)]
        results.append(
            {
                **row,
                "probabilities": vectors,
                "scores": scores,
                "public_mean_f": public_mean,
                "public_predictive_interval90_f": public_interval,
                "public_interval90_contains_outcome": (
                    public_interval[0] <= row["outcome_f"] <= public_interval[1]
                ),
                "disagreement_l1": disagreement,
                "disagreement_group": "high"
                if disagreement > fit["disagreement_l1_median"]
                else "low",
                "used_for_fit": row["case_id"] in fit["training_case_ids"],
            }
        )
    return results


def _draw_dates(dates: list[str], block: int, replicates: int, seed: int) -> list[Counter]:
    """Circular moving blocks of ordered observed dates; all records travel together."""
    rng = random.Random(seed)
    n = len(dates)
    result = []
    for _ in range(replicates):
        sampled = []
        while len(sampled) < n:
            start = rng.randrange(n)
            sampled.extend(dates[(start + j) % n] for j in range(min(block, n)))
        result.append(Counter(sampled[:n]))
    return result


def _quantile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    at = (len(ordered) - 1) * q
    left = math.floor(at)
    fraction = at - left
    return ordered[left] * (1 - fraction) + ordered[min(left + 1, len(ordered) - 1)] * fraction


def _date_totals(rows: list[dict], method: str, metric: str) -> dict[str, tuple[float, int]]:
    buckets: dict[str, list[float]] = {}
    for row in rows:
        buckets.setdefault(row["outcome_date"], []).append(
            row["scores"][method][metric] - row["scores"]["public_only"][metric]
        )
    return {day: (math.fsum(values), len(values)) for day, values in buckets.items()}


def _weighted_mean(totals: dict[str, tuple[float, int]], weights: Counter) -> float | None:
    count = sum(n * weights[day] for day, (_, n) in totals.items())
    return (
        math.fsum(value * weights[day] for day, (value, _) in totals.items()) / count
        if count
        else None
    )


def _interval(values: list[float], replicates: int, enough_dates: bool) -> dict:
    # skip conditional intervals when too many draws lose a group
    if not enough_dates:
        reason = "fewer_than_two_dates_in_a_required_group"
    elif len(values) < max(20, math.ceil(.95 * replicates)):
        reason = "insufficient_valid_bootstrap_replicates"
    else:
        reason = None
    return {
        "ci95": None if reason else [_quantile(values, .025), _quantile(values, .975)],
        "valid_replicates": len(values),
        "undefined_replicates": replicates - len(values),
        "interval_unavailable_reason": reason,
    }


def _contrast(rows: list[dict], method: str, metric: str, draws: list[Counter]) -> dict:
    totals = _date_totals(rows, method, metric)
    original = _weighted_mean(totals, Counter(dict.fromkeys(totals, 1)))
    values = [value for weights in draws if (value := _weighted_mean(totals, weights)) is not None]
    return {
        "method": method,
        "reference": "public_only",
        "metric": metric,
        "mean_difference": original,
        **_interval(values, len(draws), len(totals) >= 2),
    }


def _counts(rows: list[dict]) -> dict:
    return {
        "cases": len(rows),
        "events": len({r["event_id"] for r in rows}),
        "dates": len({r["outcome_date"] for r in rows}),
    }


def _conditions(rows: list[dict], draws: list[Counter]) -> dict:
    groups = {"low": [], "high": []}
    for row in rows:
        for group in groups:
            if row["disagreement_group"] == group:
                groups[group].append(row)
    result: dict = {"groups": {}, "interactions": []}
    for group, subset in groups.items():
        comparisons = []
        for method in METHODS[1:]:
            for metric in METRICS:
                comparisons.append(_contrast(subset, method, metric, draws))
        result["groups"][group] = {
            **_counts(subset),
            "paired_comparisons": comparisons,
        }
    for method in METHODS[1:]:
        for metric in METRICS:
            totals = {g: _date_totals(v, method, metric) for g, v in groups.items()}
            original = {
                g: _weighted_mean(v, Counter(dict.fromkeys(v, 1))) for g, v in totals.items()
            }
            values = []
            for draw in draws:
                low = _weighted_mean(totals["low"], draw)
                high = _weighted_mean(totals["high"], draw)
                if low is not None and high is not None:
                    values.append(high - low)
            low, high = original["low"], original["high"]
            result["interactions"].append(
                {
                    "method": method,
                    "metric": metric,
                    "contrast": "(method-public) high minus (method-public) low disagreement",
                    "mean_difference": high - low if high is not None and low is not None else None,
                    **_interval(values, len(draws), all(len(t) >= 2 for t in totals.values())),
                }
            )
    return result


def _summaries(rows: list[dict], protocol: dict) -> list[dict]:
    summaries = []
    blocks = list(
        dict.fromkeys(
            [protocol["bootstrap_block_dates"], *protocol["bootstrap_sensitivity_block_dates"]]
        )
    )
    for split in ("train", "validation", "holdout"):
        split_rows = [r for r in rows if r["split"] == split]
        dates = sorted({r["outcome_date"] for r in split_rows})
        if not dates:
            continue
        draws_by_block = {
            block: _draw_dates(
                dates, block, protocol["bootstrap_replicates"], protocol["bootstrap_seed"]
            )
            for block in blocks
        }
        draws = draws_by_block[protocol["bootstrap_block_dates"]]
        for horizon in protocol["horizon_hours"]:
            subset = [r for r in split_rows if r["horizon_hours"] == horizon]
            if not subset:
                continue
            contrasts = []
            for method in METHODS[1:]:
                for metric in METRICS:
                    contrast = _contrast(subset, method, metric, draws)
                    contrast["block_sensitivity"] = {
                        str(b): _contrast(subset, method, metric, draws_by_block[b])
                        for b in protocol["bootstrap_sensitivity_block_dates"]
                    }
                    contrasts.append(contrast)
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
                    score = statistics.fmean(row["scores"][method][metric] for row in subset)
                    scores[method][metric] = score
            summaries.append(
                {
                    "split": split,
                    "horizon_hours": horizon,
                    **_counts(subset),
                    "date_block_count_floor": len(dates) // protocol["bootstrap_block_dates"],
                    "scores": scores,
                    "public_interval90_observed_coverage": statistics.fmean(
                        r["public_interval90_contains_outcome"] for r in subset
                    ),
                    "public_mean_error_f": statistics.fmean(
                        r["outcome_f"] - r["public_mean_f"] for r in subset
                    ),
                    "paired_comparisons": contrasts,
                    "calibration": calibration,
                    "conditions": _conditions(subset, draws),
                }
            )
    return summaries


def evaluate_historical_panel(panel: dict, protocol: dict) -> dict:
    """Validate, fit on available training labels, score and resample whole dates."""
    frozen = _protocol(protocol)
    rows = _validated_cases(panel, frozen)
    fit = _fit(rows, frozen)
    scored = _score_cases(rows, fit, frozen)
    inventory = []
    for split in ("train", "validation", "holdout"):
        for horizon in frozen["horizon_hours"]:
            subset = []
            for row in scored:
                if row["split"] == split and row["horizon_hours"] == horizon:
                    subset.append(row)
            inventory.append({"split": split, "horizon_hours": horizon, **_counts(subset)})
    return {
        "schema_version": "historical_evaluation_v1",
        "data_origin": "empirical_historical",
        "analysis_status": "exploratory_historical_reconstruction",
        "protocol": frozen,
        "input_protocol_sha256": hashlib.sha256(
            json.dumps(protocol, sort_keys=True, allow_nan=False).encode()
        ).hexdigest(),
        "protocol_sha256": hashlib.sha256(
            json.dumps(frozen, sort_keys=True, allow_nan=False).encode()
        ).hexdigest(),
        "input_panel_sha256": hashlib.sha256(
            json.dumps(panel, sort_keys=True, allow_nan=False).encode()
        ).hexdigest(),
        "fit": fit,
        "cohort": _counts(scored),
        "split_horizon_inventory": inventory,
        "summaries": _summaries(scored, frozen),
        "case_results": scored,
        "exclusions": panel.get("exclusions", []),
        "exclusion_reason_counts": dict(
            Counter(r.get("reason", "unspecified") for r in panel.get("exclusions", []))
        ),
        "provenance": panel.get("provenance", {}),
        "interpretation": {
            "negative_paired_difference": "favors the named method over public_only",
            "bootstrap": (
                "circular moving blocks of sorted observed dates, all events/horizons retained; "
                "shared draws across methods and conditions"
            ),
            "bootstrap_calendar_gaps": (
                "blocks count observed dates and may span missing calendar dates"
            ),
            "calibration": (
                "descriptive pooled bin probabilities; bin observations are not independent events"
            ),
            "training_scores": "in-sample, not evidence of later forecasting performance",
            "multiplicity": (
                "intervals are marginal 95% intervals, not adjusted for multiple comparisons"
            ),
            "historical_availability": (
                "historical reconstruction follows input provenance; "
                "retrieved-now timestamps alone do not prove past availability"
            ),
            "scope": "bounded cohort; final labels; no causal or executable trading claim",
        },
    }


def write_historical_report(results: dict, output_dir: Path) -> dict:
    """Compatibility entry point; rendering lives outside scientific evaluation."""
    from .historical_reporting import write_historical_report as write

    return write(results, output_dir)
