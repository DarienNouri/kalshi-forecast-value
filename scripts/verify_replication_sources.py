#!/usr/bin/env python3
"""Rebuild replication data-quality evidence from local immutable inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from kalshi_information_quality_lab.storage import StorageError, publish_immutable_file

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PANEL = Path("outputs/historical-replication/panel.json")
DEFAULT_OUTPUT = Path("outputs/historical-replication/data-quality.json")


class VerificationError(ValueError):
    """Saved replication evidence is missing, inconsistent, or modified."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def _contained(root: Path, value: Path, name: str) -> Path:
    root = root.resolve()
    candidate = value if value.is_absolute() else root / value
    candidate = candidate.resolve()
    require(candidate.is_relative_to(root), f"{name} escapes the project root")
    return candidate


def _read_json(path: Path, name: str, *, limit: int = 32 * 1024 * 1024) -> dict[str, Any]:
    require(path.is_file(), f"Missing {name}: {path}")
    require(0 < path.stat().st_size <= limit, f"Invalid {name} size: {path}")

    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            require(key not in result, f"Duplicate JSON key in {name}: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(path.read_bytes(), object_pairs_hook=unique)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise VerificationError(f"Invalid {name}: {path}") from exc
    require(isinstance(value, dict), f"{name} must be a JSON object")
    return value


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            while block := stream.read(1024 * 1024):
                digest.update(block)
    except OSError as exc:
        raise VerificationError(f"Cannot read source body: {path}") from exc
    return digest.hexdigest()


def _sha256(value: object, name: str) -> str:
    require(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value),
        f"Invalid {name}",
    )
    assert isinstance(value, str)
    return value


def _date_bounds(first: object, last: object) -> tuple[date, date, int]:
    require(isinstance(first, str) and isinstance(last, str), "Invalid protocol date range")
    assert isinstance(first, str) and isinstance(last, str)
    try:
        start, end = date.fromisoformat(first), date.fromisoformat(last)
    except (TypeError, ValueError) as exc:
        raise VerificationError("Invalid protocol date range") from exc
    require(start <= end, "Invalid protocol date range")
    return start, end, (end - start).days + 1


def _dates(start: date, count: int) -> list[str]:
    return [(start + timedelta(days=offset)).isoformat() for offset in range(count)]


def _regular_contained(root: Path, path: Path, name: str) -> Path:
    require(not path.is_symlink(), f"{name} must not be a symlink")
    resolved = _contained(root, path, name)
    require(resolved.is_file(), f"Missing {name}: {path}")
    return resolved


def _audit_raw(root: Path, relative: object) -> dict[str, Any]:
    require(isinstance(relative, str) and bool(relative), "Missing raw directory")
    assert isinstance(relative, str)
    require(not Path(relative).is_absolute(), "Raw directory must be project-relative")
    require(not (root / relative).is_symlink(), "Raw directory must not be a symlink")
    raw = _contained(root, Path(relative), "raw directory")
    require(raw.is_dir(), f"Missing raw directory: {relative}")
    receipts = {path.stem: path for path in raw.glob("*.json")}
    bodies = {path.stem: path for path in raw.glob("*.body")}
    require(receipts.keys() == bodies.keys(), f"Raw receipt/body inventory mismatch: {relative}")
    require(bool(receipts), f"Empty raw directory: {relative}")
    total_bytes = 0
    for stem in sorted(receipts):
        receipt_path = _regular_contained(root, receipts[stem], "raw receipt")
        body = _regular_contained(root, bodies[stem], "raw source body")
        receipt = _read_json(receipt_path, "raw receipt", limit=128 * 1024)
        source_uri = receipt.get("source_uri")
        require(
            isinstance(source_uri, str) and bool(source_uri), "Raw receipt source URI is invalid"
        )
        assert isinstance(source_uri, str)
        require(
            stem == hashlib.sha256(source_uri.encode()).hexdigest(), "Raw receipt is not URL-keyed"
        )
        expected_hash = _sha256(receipt.get("content_sha256"), "raw content SHA-256")
        expected_bytes = receipt.get("bytes")
        require(type(expected_bytes) is int and expected_bytes >= 0, "Raw byte count is invalid")
        assert isinstance(expected_bytes, int)
        require(receipt.get("origin") == "empirical", "Raw source origin is not empirical")
        require(receipt.get("http_status") == 200, "Raw source HTTP status is not 200")
        require(body.stat().st_size == expected_bytes, "Raw source byte count changed")
        require(_file_hash(body) == expected_hash, "Raw source content hash changed")
        total_bytes += expected_bytes
    return {
        "raw_directory": relative,
        "objects": len(receipts),
        "bytes": total_bytes,
        "all_hashes_verified": True,
    }


def build_data_quality(root: Path, panel_path: Path) -> dict[str, Any]:
    """Validate a combined panel and all retained receipts, then derive coverage."""
    root = root.resolve()
    panel_path = _contained(root, panel_path, "panel path")
    panel = _read_json(panel_path, "replication panel")
    require(panel.get("schema_version") == "historical_panel_v1", "Invalid panel schema")
    require(
        panel.get("replication_schema_version") == "historical_replication_panel_v1",
        "Invalid replication panel schema",
    )
    require(panel.get("data_origin") == "empirical_historical", "Panel origin is not empirical")
    require(panel.get("complete_requested_dates") is True, "Panel is not complete")
    protocol = panel.get("protocol")
    require(isinstance(protocol, dict), "Missing panel protocol")
    assert isinstance(protocol, dict)
    horizons = protocol.get("horizon_hours")
    require(
        isinstance(horizons, list)
        and bool(horizons)
        and all(type(item) is int for item in horizons)
        and len(horizons) == len(set(horizons)),
        "Invalid protocol horizons",
    )
    assert isinstance(horizons, list)
    inventory = panel.get("replication_inventory")
    require(isinstance(inventory, list), "Missing replication slot inventory")
    assert isinstance(inventory, list)
    first, _, date_count = _date_bounds(protocol.get("start_date"), protocol.get("end_date"))
    require(
        date_count * len(horizons) == len(inventory),
        "Replication slot count conflicts with the protocol date span",
    )
    expected_dates = _dates(first, date_count)
    expected_slots = set()
    for day in expected_dates:
        for horizon in horizons:
            expected_slots.add((day, horizon))
    slots: dict[tuple[str, int], dict[str, Any]] = {}
    for source in inventory:
        require(isinstance(source, dict), "Invalid replication slot row")
        slot = (source.get("outcome_date"), source.get("horizon_hours"))
        require(
            slot in expected_slots and slot not in slots, "Invalid or duplicate replication slot"
        )
        require(source.get("status") in {"case", "excluded"}, "Incomplete replication slot")
        require(source.get("target_status") == "compatible", "Replication slot target mismatch")
        slots[slot] = source
    require(set(slots) == expected_slots, "Replication slot inventory is incomplete")

    counts = panel.get("counts")
    require(isinstance(counts, dict), "Missing panel counts")
    assert isinstance(counts, dict)
    case_rows = panel.get("cases")
    exclusion_rows = panel.get("exclusions")
    require(
        isinstance(case_rows, list) and isinstance(exclusion_rows, list),
        "Missing panel case or exclusion rows",
    )
    assert isinstance(case_rows, list) and isinstance(exclusion_rows, list)
    cases = sum(row["status"] == "case" for row in slots.values())
    excluded_cases = sum(row["status"] == "excluded" for row in slots.values())
    require(
        counts.get("requested_dates") == len(expected_dates)
        and counts.get("processed_dates") == len(expected_dates)
        and counts.get("cases") == cases == len(case_rows)
        and counts.get("exclusions") == excluded_cases == len(exclusion_rows),
        "Panel counts conflict with the slot inventory",
    )

    target = panel.get("target_continuity")
    require(
        isinstance(target, dict)
        and target.get("analysis_ready") is True
        and target.get("missing_slots") == 0
        and target.get("incompatible_dates") == []
        and target.get("unverified_dates") == [],
        "Target continuity gate is not complete",
    )
    target_inventory = panel.get("target_inventory")
    require(isinstance(target_inventory, list), "Missing target inventory")
    assert isinstance(target_inventory, list)
    compatible = {
        row.get("outcome_date")
        for row in target_inventory
        if isinstance(row, dict) and row.get("status") == "compatible"
    }
    require(
        len(target_inventory) == len(expected_dates) and compatible == set(expected_dates),
        "Target inventory does not cover every requested date",
    )

    status_by_date: dict[str, set[str]] = defaultdict(set)
    for (day, _), row in slots.items():
        status_by_date[day].add(row["status"])
    require(
        all(len(statuses) == 1 for statuses in status_by_date.values()),
        "A date mixes usable and excluded slots",
    )
    eligible_dates = {day for day, statuses in status_by_date.items() if statuses == {"case"}}
    excluded_dates = set(expected_dates) - eligible_dates
    monthly: dict[str, dict[str, int | str]] = {}
    for day in expected_dates:
        month = day[:7]
        row = monthly.setdefault(month, {"month": month, "requested": 0})
        row["requested"] = int(row["requested"]) + 1
        key = "case" if day in eligible_dates else "excluded"
        row[key] = int(row.get(key, 0)) + 1

    exclusions = Counter(row.get("reason") for row in slots.values() if row["status"] == "excluded")
    require(
        all(isinstance(reason, str) and reason for reason in exclusions), "Missing exclusion reason"
    )
    label_reasons = {"empty_expiration_value", "could not convert string to float: ''"}
    label_excluded_dates = {
        day
        for (day, _), row in slots.items()
        if row["status"] == "excluded" and row.get("reason") in label_reasons
    }
    limitations: list[str] = []
    if label_excluded_dates:
        limitations.append(
            "Missing numeric settlement labels condition analysis on source-complete dates."
        )
        cohort_months = {day[:7] for day in expected_dates}
        seasonal_count = sum(
            date.fromisoformat(day).month in {10, 11, 12, 1} for day in label_excluded_dates
        )
        if (
            len(cohort_months) >= 6
            and len(label_excluded_dates) >= 2
            and seasonal_count / len(label_excluded_dates) >= .75
        ):
            limitations.append(
                "Concentrated autumn/winter missingness reduces seasonal development coverage; "
                "resampling does not recover omitted information."
            )
        validation_end_value = protocol.get("validation_end_date")
        require(isinstance(validation_end_value, str), "Invalid validation end date")
        assert isinstance(validation_end_value, str)
        validation_end = date.fromisoformat(validation_end_value)
        holdout_excluded = sorted(
            day for day in label_excluded_dates if date.fromisoformat(day) > validation_end
        )
        if len(holdout_excluded) == 1:
            observed = date.fromisoformat(holdout_excluded[0])
            limitations.append(
                f"{observed:%B} {observed.day}, {observed.year} is excluded from holdout "
                "because its numeric expiration value is empty."
            )

    provenance = panel.get("provenance")
    require(isinstance(provenance, dict), "Missing panel provenance")
    assert isinstance(provenance, dict)
    partitions = provenance.get("partitions")
    require(isinstance(partitions, list) and bool(partitions), "Missing panel partitions")
    assert isinstance(partitions, list)
    raw_integrity = []
    for partition in partitions:
        require(isinstance(partition, dict), "Invalid panel partition")
        raw_relative = partition.get("raw_directory")
        require(isinstance(raw_relative, str), "Missing partition raw directory")
        require(not Path(raw_relative).is_absolute(), "Raw directory must be project-relative")
        raw = _contained(root, Path(raw_relative), "partition raw directory")
        partition_panel = _regular_contained(root, raw.parent / "panel.json", "partition panel")
        expected_hash = _sha256(partition.get("panel_sha256"), "partition panel SHA-256")
        require(_file_hash(partition_panel) == expected_hash, "Partition panel hash changed")
        raw_integrity.append(_audit_raw(root, raw_relative))

    return {
        "schema_version": "replication_data_quality_v1",
        "panel_file_sha256": _file_hash(panel_path),
        "requested_dates": len(expected_dates),
        "eligible_dates": len(eligible_dates),
        "excluded_dates": len(excluded_dates),
        "eligible_cases": cases,
        "target_rules_compatible_dates": len(compatible),
        "exclusions": dict(sorted(exclusions.items())),
        "monthly_coverage": list(monthly.values()),
        "raw_integrity": raw_integrity,
        "limitations": limitations,
    }


def _encoded(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def verify_and_publish(root: Path, panel_path: Path, output_path: Path) -> dict[str, Any]:
    """Build and immutably publish the verification record after every check passes."""
    root = root.resolve()
    result = build_data_quality(root, panel_path)
    output = _contained(root, output_path, "output path")
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        publish_immutable_file(output.parent, output.name, _encoded(result))
    except (OSError, StorageError) as exc:
        raise VerificationError(f"Cannot publish immutable data-quality record: {exc}") from None
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--panel", type=Path, default=DEFAULT_PANEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    try:
        result = verify_and_publish(args.root, args.panel, args.output)
    except (OSError, TypeError, ValueError) as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, sort_keys=True), file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "ok": True,
                "panel_file_sha256": result["panel_file_sha256"],
                "requested_dates": result["requested_dates"],
                "eligible_dates": result["eligible_dates"],
                "eligible_cases": result["eligible_cases"],
                "output": str(_contained(args.root, args.output, "output path")),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
