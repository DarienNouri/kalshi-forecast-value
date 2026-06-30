"""Shared validation for historical collection and evaluation protocols."""

from __future__ import annotations

import math
from copy import deepcopy
from datetime import date
from typing import Any


class HistoricalInputError(ValueError):
    """A historical protocol or empirical input violates the study contract."""


EVALUATION_DEFAULTS = {
    "horizon_hours": [6, 12],
    "residual_std_floor_f": 1.0,
    "bootstrap_replicates": 2000,
    "bootstrap_seed": 20250914,
    "bootstrap_block_dates": 2,
    "bootstrap_sensitivity_block_dates": [1, 3, 7],
    "log_loss_epsilon": 1e-6,
}

COLLECTION_PROTOCOL_FIELDS = (
    "start_date",
    "end_date",
    "series_ticker",
    "station_id",
    "station_latitude",
    "station_longitude",
    "horizon_hours",
    "outcome_window_start_utc_hour",
    "forecast_valid_start_utc_hour",
    "forecast_valid_hours",
    "forecast_min_lead_minutes",
    "forecast_max_age_hours",
    "market_candle_interval_minutes",
    "market_max_candle_age_minutes",
    "market_price_policy",
    "availability_policy",
    "target_policy",
)


def _required(protocol: dict, key: str) -> Any:
    if key not in protocol:
        raise HistoricalInputError(f"protocol is missing required field: {key}")
    return protocol[key]


def _date(value: Any, name: str) -> date:
    if not isinstance(value, str):
        raise HistoricalInputError(f"{name} must use YYYY-MM-DD")
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise HistoricalInputError(f"{name} must use YYYY-MM-DD") from exc
    if parsed.isoformat() != value:
        raise HistoricalInputError(f"{name} must use YYYY-MM-DD")
    return parsed


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise HistoricalInputError(f"{name} must be finite numeric data")
    result = float(value)
    if not math.isfinite(result):
        raise HistoricalInputError(f"{name} must be finite numeric data")
    return result


def _integer(value: Any, name: str, *, minimum: int, maximum: int | None = None) -> int:
    if type(value) is not int or value < minimum or (maximum is not None and value > maximum):
        bounds = f" between {minimum} and {maximum}" if maximum is not None else f" >= {minimum}"
        raise HistoricalInputError(f"{name} must be an integer{bounds}")
    return value


def _positive_integer_list(value: Any, name: str) -> list[int]:
    if not isinstance(value, list) or not value:
        raise HistoricalInputError(f"{name} must contain positive integers")
    if any(type(item) is not int or item <= 0 for item in value):
        raise HistoricalInputError(f"{name} must contain positive integers")
    if len(set(value)) != len(value):
        raise HistoricalInputError(f"{name} must not contain duplicates")
    return value


def _nonempty_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise HistoricalInputError(f"{name} must be a nonempty string")
    return value


def validate_protocol(raw: object, *, collection: bool = False) -> dict:
    """Validate a v1 protocol, adding defaults only for evaluation.

    Collection returns the supplied payload values without inserting analysis
    defaults so the persisted v1 protocol and its content hash remain stable.
    """
    if not isinstance(raw, dict):
        raise HistoricalInputError("protocol must be an object")
    protocol: dict[str, Any] = deepcopy(raw)
    if not collection:
        protocol = {**deepcopy(EVALUATION_DEFAULTS), **protocol}

    if _required(protocol, "schema_version") != "historical_protocol_v1":
        raise HistoricalInputError("Expected historical_protocol_v1")

    if collection:
        for key in COLLECTION_PROTOCOL_FIELDS:
            _required(protocol, key)
    else:
        for key in ("start_date", "train_end_date", "validation_end_date", "end_date"):
            _required(protocol, key)

    start = _date(_required(protocol, "start_date"), "start_date")
    end = _date(_required(protocol, "end_date"), "end_date")
    if collection:
        if start > end:
            raise HistoricalInputError("Require start_date <= end_date")
    else:
        train_end = _date(_required(protocol, "train_end_date"), "train_end_date")
        validation_end = _date(_required(protocol, "validation_end_date"), "validation_end_date")
        if not start <= train_end < validation_end < end:
            raise HistoricalInputError("Require start <= train_end < validation_end < end")

    _positive_integer_list(_required(protocol, "horizon_hours"), "horizon_hours")

    if collection:
        _nonempty_string(protocol["series_ticker"], "series_ticker")
        _nonempty_string(protocol["station_id"], "station_id")
        latitude = _number(protocol["station_latitude"], "station_latitude")
        longitude = _number(protocol["station_longitude"], "station_longitude")
        if not -90 <= latitude <= 90:
            raise HistoricalInputError("station_latitude must be between -90 and 90")
        if not -180 <= longitude <= 180:
            raise HistoricalInputError("station_longitude must be between -180 and 180")
        _integer(
            protocol["outcome_window_start_utc_hour"],
            "outcome_window_start_utc_hour",
            minimum=0,
            maximum=23,
        )
        _integer(
            protocol["forecast_valid_start_utc_hour"],
            "forecast_valid_start_utc_hour",
            minimum=0,
            maximum=23,
        )
        for key in (
            "forecast_valid_hours",
            "forecast_max_age_hours",
            "market_candle_interval_minutes",
        ):
            _integer(protocol[key], key, minimum=1)
        for key in ("forecast_min_lead_minutes", "market_max_candle_age_minutes"):
            if _number(protocol[key], key) < 0:
                raise HistoricalInputError(f"{key} must be nonnegative")
        for key in ("market_price_policy", "availability_policy", "target_policy"):
            _nonempty_string(protocol[key], key)

    if not collection:
        if _number(protocol["residual_std_floor_f"], "residual_std_floor_f") <= 0:
            raise HistoricalInputError("residual_std_floor_f must be positive")
        if not 0 < _number(protocol["log_loss_epsilon"], "log_loss_epsilon") <= 1:
            raise HistoricalInputError("log_loss_epsilon must be in (0, 1]")
        for key in ("bootstrap_replicates", "bootstrap_block_dates"):
            _integer(protocol[key], key, minimum=1)
        if type(protocol["bootstrap_seed"]) is not int:
            raise HistoricalInputError("bootstrap_seed must be an integer")
        _positive_integer_list(
            protocol["bootstrap_sensitivity_block_dates"],
            "bootstrap_sensitivity_block_dates",
        )

    return protocol
