"""Frozen protocol and immutable-partition inputs for the historical replication."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from typing import Any

from .config import StudyConfig
from .historical import HistoricalError, SourceCache, _market_pages
from .historical_inputs import COLLECTION_PROTOCOL_FIELDS, HistoricalInputError, validate_protocol
from .sources import SourceError

FROZEN_REPLICATION_SHA256 = "f38377221e332dc78358e38d28dbc60e52225521ca4bac11b426d0a839a2cd33"
REPLICATION_SCHEMA = "historical_replication_v1"
COMBINED_SCHEMA = "historical_replication_panel_v1"
TARGET_FIELDS = tuple(
    key for key in COLLECTION_PROTOCOL_FIELDS if key not in {"start_date", "end_date"}
)


class ReplicationInputError(ValueError):
    """A frozen replication protocol or collected partition is inconsistent."""


def _sha(body: bytes) -> str:
    return hashlib.sha256(body).hexdigest()


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ReplicationInputError(f"{name}_must_be_sha256")
    try:
        int(value, 16)
    except ValueError:
        raise ReplicationInputError(f"{name}_must_be_sha256") from None
    return value


def _day(value: object, name: str) -> date:
    if not isinstance(value, str):
        raise ReplicationInputError(f"{name}_must_use_yyyy_mm_dd")
    try:
        result = date.fromisoformat(value)
    except ValueError:
        raise ReplicationInputError(f"{name}_must_use_yyyy_mm_dd") from None
    if result.isoformat() != value:
        raise ReplicationInputError(f"{name}_must_use_yyyy_mm_dd")
    return result


def _instant(value: object, name: str) -> datetime:
    if not isinstance(value, str):
        raise ReplicationInputError(f"{name}_must_be_timezone_aware")
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        raise ReplicationInputError(f"{name}_must_be_timezone_aware") from None
    if result.tzinfo is None:
        raise ReplicationInputError(f"{name}_must_be_timezone_aware")
    return result.astimezone(UTC)


def _mapping(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ReplicationInputError(f"{name}_must_be_an_object")
    return value


def _date_count(first: date, last: date) -> int:
    return (last - first).days + 1


def validate_replication_protocol(raw: object) -> dict[str, Any]:
    """Validate the project-specific v2 replication structure without claiming file freeze."""
    protocol = deepcopy(_mapping(raw, "replication_protocol"))
    if protocol.get("schema_version") != REPLICATION_SCHEMA:
        raise ReplicationInputError(f"expected_{REPLICATION_SCHEMA}")
    if protocol.get("analysis_class") != "exploratory_historical_replication":
        raise ReplicationInputError("invalid_replication_analysis_class")
    required_objects = (
        "collection",
        "legacy",
        "models",
        "oof",
        "bootstrap",
        "conditions",
        "metrics",
        "sensitivities",
        "pre_outcome_amendment",
    )
    for name in required_objects:
        _mapping(protocol.get(name), name)
    try:
        collection = validate_protocol(protocol["collection"], collection=True)
    except HistoricalInputError as exc:
        raise ReplicationInputError(f"invalid_collection_protocol: {exc}") from None
    first = _day(collection["start_date"], "collection.start_date")
    last = _day(collection["end_date"], "collection.end_date")
    legacy_end = _day(protocol["legacy"].get("end_date"), "legacy.end_date")
    new_start = _day(protocol.get("new_start_date"), "new_start_date")
    if legacy_end + timedelta(days=1) != new_start:
        raise ReplicationInputError("legacy_and_new_partitions_must_be_adjacent")
    if not first <= legacy_end < new_start <= last:
        raise ReplicationInputError("invalid_replication_collection_chronology")
    if (
        _date_count(first, last) != 395
        or _date_count(first, legacy_end) != 92
        or _date_count(new_start, last) != 303
    ):
        raise ReplicationInputError("replication_date_counts_must_be_395_92_303")
    train_end = _day(collection.get("train_end_date"), "collection.train_end_date")
    validation_end = _day(collection.get("validation_end_date"), "collection.validation_end_date")
    if not first <= train_end < validation_end < last:
        raise ReplicationInputError("invalid_replication_split_chronology")
    validation_fit = _instant(protocol.get("validation_fit_at"), "validation_fit_at")
    final_fit = _instant(protocol.get("final_fit_at"), "final_fit_at")
    if validation_fit.date() != train_end or final_fit.date() != validation_end:
        raise ReplicationInputError("fit_instants_must_match_split_boundaries")
    if protocol["pre_outcome_amendment"].get("new_holdout_outcomes_inspected") is not False:
        raise ReplicationInputError("pre_outcome_amendment_must_precede_holdout_inspection")
    _sha256(protocol["legacy"].get("panel_sha256"), "legacy_panel_sha256")
    _sha256(protocol["legacy"].get("results_sha256"), "legacy_results_sha256")
    run_dir = protocol["legacy"].get("run_dir")
    if (
        not isinstance(run_dir, str)
        or Path(run_dir).is_absolute()
        or ".." in Path(run_dir).parts
        or not Path(run_dir).parts
        or Path(run_dir).parts[0] != "outputs"
    ):
        raise ReplicationInputError("legacy_run_dir_must_be_project_outputs_relative")
    attempts = protocol["bootstrap"].get("attempts")
    if type(attempts) is not int or attempts <= 0:
        raise ReplicationInputError("bootstrap_attempts_must_be_positive_integer")
    if protocol["bootstrap"].get("strata") != ["split", "calendar_month"]:
        raise ReplicationInputError("bootstrap_strata_must_be_split_and_calendar_month")
    if protocol["models"].get("candidates") != [
        "p0",
        "p1_lambda_0.1",
        "p1_lambda_1",
        "p1_lambda_10",
    ]:
        raise ReplicationInputError("invalid_replication_model_candidates")
    return protocol


def load_frozen_replication_protocol(path: Path) -> dict[str, Any]:
    """Load only the exact reviewed replication protocol file."""
    try:
        body = path.read_bytes()
    except OSError as exc:
        raise ReplicationInputError(f"cannot_read_replication_protocol: {exc.strerror}") from None
    if _sha(body) != FROZEN_REPLICATION_SHA256:
        raise ReplicationInputError("frozen_replication_protocol_sha256_mismatch")
    try:
        raw = json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError):
        raise ReplicationInputError("invalid_replication_protocol_json") from None
    return validate_replication_protocol(raw)


def derive_new_collection_protocol(replication: dict[str, Any]) -> dict[str, Any]:
    """Derive the bounded 303-day collector input without altering the outer protocol."""
    frozen = validate_replication_protocol(replication)
    result = deepcopy(frozen["collection"])
    result["start_date"] = frozen["new_start_date"]
    result = validate_protocol(result, collection=True)
    count = _date_count(
        date.fromisoformat(result["start_date"]), date.fromisoformat(result["end_date"])
    )
    if count != 303 or count > 366:
        raise ReplicationInputError("new_collection_partition_must_have_303_dates")
    return result


def _validate_target_rules(markets: list[dict[str, Any]], event_id: str) -> None:
    """Check target identity without requiring a numeric settlement label."""
    if not markets:
        raise HistoricalError("missing_historical_markets")
    for market in markets:
        if not isinstance(market, dict):
            raise HistoricalError("invalid_historical_market_metadata")
        rules = market.get("rules_primary")
        if (
            market.get("event_ticker") != event_id
            or not isinstance(rules, str)
            or "Central Park, New York" not in rules
            or "National Weather Service's Climatological Report (Daily)" not in rules
        ):
            raise HistoricalError("unmatched_historical_rules")


def _settlement_label_availability(markets: list[dict[str, Any]]) -> tuple[str, str | None]:
    """Describe source-label availability without deriving or substituting a label."""
    if any(market.get("expiration_value") == "" for market in markets):
        return "unavailable", "empty_expiration_value"
    if any(market.get("expiration_value") is None for market in markets):
        return "unavailable", "missing_expiration_value"
    return "present", None


def inventory_partition_targets(
    config: StudyConfig, collection_protocol: dict[str, Any], raw_directory: Path
) -> list[dict[str, Any]]:
    """Reconstruct per-event target-rule checks from an immutable offline raw cache."""
    protocol = validate_protocol(collection_protocol, collection=True)
    raw_directory = (config.project_root / raw_directory).resolve()
    if not raw_directory.is_relative_to(config.project_root / "outputs"):
        raise ReplicationInputError("target_inventory_cache_must_be_inside_project_outputs")
    cache = SourceCache(
        config.project_root,
        config.project_root / "outputs",
        cache_dir=raw_directory,
        offline=True,
        evidence_reference=config.evidence_reference,
    )
    first, last = (
        date.fromisoformat(protocol["start_date"]),
        date.fromisoformat(protocol["end_date"]),
    )
    inventory = []
    incompatible_reasons = {
        "unmatched_historical_rules",
        "noninteger_bin_boundary",
        "unsupported_bin_predicate",
        "invalid_bin_bounds",
        "incomplete_bin_tails",
        "bin_gap_or_overlap",
        "settlement_result_mismatch",
        "conflicting_settlement_values",
    }
    for offset in range(_date_count(first, last)):
        day = first + timedelta(days=offset)
        event_id = protocol["series_ticker"] + "-" + day.strftime("%y%b%d").upper()
        evidence: list[dict[str, Any]] = []
        label_status, label_reason = "unverified", "target_rules_unverified"
        try:
            markets, evidence = _market_pages(cache, event_id)
            _validate_target_rules(markets, event_id)
        except (HistoricalError, SourceError, KeyError, ValueError) as exc:
            reason = str(exc)
            status = "incompatible" if reason in incompatible_reasons else "unverified"
        else:
            reason = None
            status = "compatible"
            label_status, label_reason = _settlement_label_availability(markets)
        inventory.append(
            {
                "outcome_date": day.isoformat(),
                "event_id": event_id,
                "status": status,
                "reason": reason,
                "settlement_label_status": label_status,
                "settlement_label_reason": label_reason,
                "rules_source_sha256": [
                    item["content_sha256"]
                    for item in evidence
                    if isinstance(item.get("content_sha256"), str)
                ],
            }
        )
    return inventory


def _decode_panel(body: bytes, name: str) -> dict[str, Any]:
    try:
        panel = json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError):
        raise ReplicationInputError(f"invalid_{name}_panel_json") from None
    return _mapping(panel, f"{name}_panel")


def _validate_partition(
    panel: dict[str, Any], expected_protocol: dict[str, Any], name: str
) -> None:
    if panel.get("schema_version") != "historical_panel_v1":
        raise ReplicationInputError(f"invalid_{name}_panel_schema")
    if panel.get("data_origin") != "empirical_historical":
        raise ReplicationInputError(f"invalid_{name}_panel_origin")
    if panel.get("complete_requested_dates") is not True:
        raise ReplicationInputError(f"incomplete_{name}_partition")
    if panel.get("protocol") != expected_protocol:
        raise ReplicationInputError(f"{name}_partition_protocol_mismatch")
    if panel.get("protocol_sha256") != _sha(_json_bytes(expected_protocol)):
        raise ReplicationInputError(f"{name}_partition_protocol_sha256_mismatch")
    if not isinstance(panel.get("cases"), list) or not isinstance(panel.get("exclusions"), list):
        raise ReplicationInputError(f"invalid_{name}_partition_rows")
    counts = _mapping(panel.get("counts"), f"{name}_counts")
    expected_dates = _date_count(
        date.fromisoformat(expected_protocol["start_date"]),
        date.fromisoformat(expected_protocol["end_date"]),
    )
    if (
        counts.get("requested_dates") != expected_dates
        or counts.get("processed_dates") != expected_dates
        or counts.get("cases") != len(panel["cases"])
        or counts.get("exclusions") != len(panel["exclusions"])
    ):
        raise ReplicationInputError(f"invalid_{name}_partition_counts")
    provenance = _mapping(panel.get("provenance"), f"{name}_provenance")
    if not isinstance(provenance.get("raw_directory"), str):
        raise ReplicationInputError(f"missing_{name}_raw_directory")


def _partition_slots(
    panel: dict[str, Any], protocol: dict[str, Any], name: str
) -> tuple[dict[tuple[str, int], dict], dict[tuple[str, int], dict]]:
    first, last = (
        date.fromisoformat(protocol["start_date"]),
        date.fromisoformat(protocol["end_date"]),
    )
    horizons = set(protocol["horizon_hours"])
    cases: dict[tuple[str, int], dict] = {}
    exclusions: dict[tuple[str, int], dict] = {}
    for row_kind, rows, destination in (
        ("case", panel["cases"], cases),
        ("exclusion", panel["exclusions"], exclusions),
    ):
        for source in rows:
            row = _mapping(source, f"{name}_{row_kind}")
            day = _day(row.get("outcome_date"), f"{name}_{row_kind}.outcome_date")
            horizon = row.get("horizon_hours")
            if day < first or day > last or horizon not in horizons or type(horizon) is not int:
                raise ReplicationInputError(f"{name}_{row_kind}_outside_partition")
            event_id = protocol["series_ticker"] + "-" + day.strftime("%y%b%d").upper()
            if row.get("event_id") != event_id:
                raise ReplicationInputError(f"{name}_{row_kind}_event_identity_mismatch")
            slot = (day.isoformat(), horizon)
            if slot in destination or slot in cases or slot in exclusions:
                raise ReplicationInputError(f"duplicate_{name}_partition_slot")
            if row_kind == "case":
                start = datetime.combine(day, time(protocol["outcome_window_start_utc_hour"]), UTC)
                if (
                    row.get("outcome_window_start_at") != start.isoformat()
                    or row.get("outcome_window_end_at") != (start + timedelta(days=1)).isoformat()
                    or row.get("historical_rules_assumption") is not True
                    or not isinstance(row.get("market_rules_source_sha256"), list)
                    or not row["market_rules_source_sha256"]
                ):
                    raise ReplicationInputError(f"{name}_case_target_evidence_mismatch")
                for item in row["market_rules_source_sha256"]:
                    _sha256(item, f"{name}_case_rule_source")
            destination[slot] = row
    return cases, exclusions


def _target_map(
    rows: list[dict[str, Any]], collection: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    first, last = (
        date.fromisoformat(collection["start_date"]),
        date.fromisoformat(collection["end_date"]),
    )
    expected = {
        (first + timedelta(days=offset)).isoformat() for offset in range(_date_count(first, last))
    }
    result: dict[str, dict[str, Any]] = {}
    for source in rows:
        row = _mapping(source, "target_inventory_row")
        day = row.get("outcome_date")
        if day not in expected or day in result:
            raise ReplicationInputError("target_inventory_date_mismatch")
        parsed = date.fromisoformat(day)
        expected_event = collection["series_ticker"] + "-" + parsed.strftime("%y%b%d").upper()
        if row.get("event_id") != expected_event:
            raise ReplicationInputError("target_inventory_event_identity_mismatch")
        if row.get("status") not in {"compatible", "incompatible", "unverified"}:
            raise ReplicationInputError("invalid_target_inventory_status")
        if row["status"] == "compatible" and not row.get("rules_source_sha256"):
            raise ReplicationInputError("compatible_target_requires_rule_evidence")
        for item in row.get("rules_source_sha256", []):
            _sha256(item, "target_rule_source")
        result[day] = deepcopy(row)
    if set(result) != expected:
        raise ReplicationInputError("target_inventory_must_cover_all_395_dates")
    return result


def combine_replication_panels(
    replication: dict[str, Any],
    legacy_panel_bytes: bytes,
    new_panel_bytes: bytes,
    *,
    replication_protocol_sha256: str,
    target_inventory: list[dict[str, Any]],
) -> dict[str, Any]:
    """Combine immutable panels only after full cohort and target-continuity inventory."""
    frozen = validate_replication_protocol(replication)
    replication_protocol_sha256 = _sha256(
        replication_protocol_sha256, "replication_protocol_sha256"
    )
    if _sha(legacy_panel_bytes) != frozen["legacy"]["panel_sha256"]:
        raise ReplicationInputError("legacy_panel_sha256_mismatch")
    legacy = _decode_panel(legacy_panel_bytes, "legacy")
    new = _decode_panel(new_panel_bytes, "new")
    try:
        legacy_protocol = validate_protocol(legacy.get("protocol"), collection=True)
    except HistoricalInputError as exc:
        raise ReplicationInputError(f"invalid_legacy_collection_protocol: {exc}") from None
    if (
        legacy_protocol["start_date"] != frozen["collection"]["start_date"]
        or legacy_protocol["end_date"] != frozen["legacy"]["end_date"]
    ):
        raise ReplicationInputError("legacy_partition_date_mismatch")
    derived = derive_new_collection_protocol(frozen)
    for key in TARGET_FIELDS:
        if legacy_protocol.get(key) != frozen["collection"].get(key):
            raise ReplicationInputError("legacy_partition_target_protocol_mismatch")
    _validate_partition(legacy, legacy_protocol, "legacy")
    _validate_partition(new, derived, "new")
    legacy_cases, legacy_exclusions = _partition_slots(legacy, legacy_protocol, "legacy")
    new_cases, new_exclusions = _partition_slots(new, derived, "new")
    cases = legacy_cases | new_cases
    exclusions = legacy_exclusions | new_exclusions
    targets = _target_map(target_inventory, frozen["collection"])
    first = date.fromisoformat(frozen["collection"]["start_date"])
    last = date.fromisoformat(frozen["collection"]["end_date"])
    inventory = []
    for offset in range(_date_count(first, last)):
        day = (first + timedelta(days=offset)).isoformat()
        for horizon in frozen["collection"]["horizon_hours"]:
            slot = (day, horizon)
            if slot in cases:
                status, reason = "case", None
            elif slot in exclusions:
                status, reason = "excluded", exclusions[slot].get("reason")
            else:
                status, reason = "missing", "partition_slot_not_recorded"
            inventory.append(
                {
                    "outcome_date": day,
                    "horizon_hours": horizon,
                    "status": status,
                    "reason": reason,
                    "target_status": targets[day]["status"],
                }
            )
    missing_slots = sum(row["status"] == "missing" for row in inventory)
    incompatible_dates = sorted(
        day for day, row in targets.items() if row["status"] == "incompatible"
    )
    unverified_dates = sorted(day for day, row in targets.items() if row["status"] == "unverified")
    analysis_ready = not missing_slots and not incompatible_dates and not unverified_dates
    combined_cases = [cases[key] for key in sorted(cases)]
    combined_exclusions = [exclusions[key] for key in sorted(exclusions)]
    return {
        "schema_version": "historical_panel_v1",
        "replication_schema_version": COMBINED_SCHEMA,
        "data_origin": "empirical_historical",
        "analysis_class": frozen["analysis_class"],
        "complete_requested_dates": missing_slots == 0,
        "protocol_sha256": _sha(_json_bytes(frozen["collection"])),
        "protocol": frozen["collection"],
        "cases": combined_cases,
        "exclusions": combined_exclusions,
        "replication_inventory": inventory,
        "target_inventory": [targets[day] for day in sorted(targets)],
        "target_continuity": {
            "analysis_ready": analysis_ready,
            "incompatible_dates": incompatible_dates,
            "unverified_dates": unverified_dates,
            "missing_slots": missing_slots,
        },
        "provenance": {
            "replication_protocol_sha256": replication_protocol_sha256,
            "partitions": [
                {
                    "role": "immutable_legacy",
                    "start_date": legacy_protocol["start_date"],
                    "end_date": legacy_protocol["end_date"],
                    "panel_sha256": _sha(legacy_panel_bytes),
                    "raw_directory": legacy["provenance"]["raw_directory"],
                },
                {
                    "role": "new_acquisition",
                    "start_date": derived["start_date"],
                    "end_date": derived["end_date"],
                    "panel_sha256": _sha(new_panel_bytes),
                    "raw_directory": new["provenance"]["raw_directory"],
                },
            ],
        },
        "counts": {
            "requested_dates": 395,
            "processed_dates": 395,
            "cases": len(combined_cases),
            "events": len({row["event_id"] for row in combined_cases}),
            "exclusions": len(combined_exclusions),
        },
    }
