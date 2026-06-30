"""Small identity and publication guards for historical run directories."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

RUN_SCHEMA = "historical_run_v1"


class HistoricalRunError(ValueError):
    """A run directory conflicts with immutable scientific inputs or outputs."""


@dataclass(frozen=True)
class RunStatus:
    exists: bool
    complete: bool
    legacy: bool
    input_sha256: str | None
    output_sha256: str | None


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _sha(body: bytes) -> str:
    return hashlib.sha256(body).hexdigest()


def _require_sha(value: str, field: str) -> str:
    if len(value) != 64:
        raise HistoricalRunError(f"invalid_{field}")
    try:
        int(value, 16)
    except ValueError:
        raise HistoricalRunError(f"invalid_{field}") from None
    return value


def protocol_identity(protocol: dict[str, Any]) -> str:
    """Return the identity used by saved v1 protocol and panel artifacts."""
    return _sha(_json_bytes(protocol))


def collection_input_identity(protocol: dict[str, Any]) -> str:
    """Identify scientific collection inputs while excluding operational replay flags."""
    return _sha(
        _json_bytes(
            {
                "schema_version": "historical_collection_input_v1",
                "protocol_sha256": protocol_identity(protocol),
            }
        )
    )


def results_input_identity(*, protocol_sha256: str, panel_sha256: str) -> str:
    """Identify the exact protocol and panel used to calculate a result."""
    return _sha(
        _json_bytes(
            {
                "schema_version": "historical_results_input_v1",
                "protocol_sha256": _require_sha(protocol_sha256, "protocol_sha256"),
                "panel_sha256": _require_sha(panel_sha256, "panel_sha256"),
            }
        )
    )


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError, UnicodeDecodeError, TypeError):
        raise HistoricalRunError("invalid_historical_run_artifact") from None
    if not isinstance(value, dict):
        raise HistoricalRunError("invalid_historical_run_artifact")
    return value


def _load_manifest(output_dir: Path, kind: str) -> dict[str, Any] | None:
    path = output_dir / "run.json"
    if not path.exists():
        return None
    value = _load_json(path)
    required = {
        "schema_version",
        "run_kind",
        "protocol_sha256",
        "input_sha256",
        "output_sha256",
        "state",
    }
    if kind == "results":
        required.add("panel_sha256")
    if (
        value.get("schema_version") != RUN_SCHEMA
        or value.get("run_kind") != kind
        or not required <= set(value)
        or value.get("state") not in {"partial", "complete"}
    ):
        raise HistoricalRunError("invalid_historical_run_manifest")
    for field in ("protocol_sha256", "input_sha256", "output_sha256"):
        _require_sha(value[field], field)
    if kind == "results":
        _require_sha(value["panel_sha256"], "panel_sha256")
    return value


def guard_collection_run(
    output_dir: Path, *, input_sha256: str, requested_complete: bool
) -> RunStatus:
    """Read-only preflight before collection creates a directory or touches a cache."""
    _require_sha(input_sha256, "input_sha256")
    manifest = _load_manifest(output_dir, "collection")
    if manifest is not None:
        if manifest["input_sha256"] != input_sha256:
            raise HistoricalRunError("scientific_inputs_require_new_output_directory")
        complete = manifest["state"] == "complete"
        panel_path = output_dir / "panel.json"
        if not panel_path.is_file() or _sha(panel_path.read_bytes()) != manifest["output_sha256"]:
            raise HistoricalRunError("run_output_integrity_failure")
        if complete and not requested_complete:
            raise HistoricalRunError("complete_run_cannot_be_replaced_by_partial")
        return RunStatus(True, complete, False, input_sha256, manifest["output_sha256"])

    protocol_path = output_dir / "protocol.json"
    panel_path = output_dir / "panel.json"
    if not protocol_path.exists() and not panel_path.exists():
        return RunStatus(output_dir.exists(), False, False, None, None)
    if not protocol_path.exists():
        raise HistoricalRunError("invalid_historical_run_artifact")
    saved_input = collection_input_identity(_load_json(protocol_path))
    if saved_input != input_sha256:
        raise HistoricalRunError("scientific_inputs_require_new_output_directory")
    if not panel_path.exists():
        return RunStatus(True, False, True, saved_input, None)
    saved_panel = _load_json(panel_path)
    if not isinstance(saved_panel.get("complete_requested_dates"), bool):
        raise HistoricalRunError("invalid_historical_run_artifact")
    if saved_panel.get("protocol_sha256") != protocol_identity(_load_json(protocol_path)):
        raise HistoricalRunError("invalid_historical_run_artifact")
    complete = saved_panel["complete_requested_dates"]
    if complete and not requested_complete:
        raise HistoricalRunError("complete_run_cannot_be_replaced_by_partial")
    return RunStatus(True, complete, True, saved_input, _sha(panel_path.read_bytes()))


def guard_panel_publication(
    output_dir: Path, *, input_sha256: str, proposed_panel_sha256: str
) -> RunStatus:
    """Refuse changed bytes at a completed panel's final publication boundary."""
    _require_sha(proposed_panel_sha256, "proposed_panel_sha256")
    status = guard_collection_run(output_dir, input_sha256=input_sha256, requested_complete=True)
    if status.complete and status.output_sha256 != proposed_panel_sha256:
        raise HistoricalRunError("completed_panel_bytes_differ")
    return status


def guard_results_publication(
    output_dir: Path, *, input_sha256: str, proposed_result_sha256: str
) -> RunStatus:
    """Refuse a result publication that changes completed inputs or exact bytes."""
    _require_sha(input_sha256, "input_sha256")
    _require_sha(proposed_result_sha256, "proposed_result_sha256")
    manifest = _load_manifest(output_dir, "results")
    if manifest is not None:
        if manifest["input_sha256"] != input_sha256:
            raise HistoricalRunError("scientific_inputs_require_new_output_directory")
        complete = manifest["state"] == "complete"
        result_path = output_dir / "historical-results.json"
        if not result_path.is_file() or _sha(result_path.read_bytes()) != manifest["output_sha256"]:
            raise HistoricalRunError("run_output_integrity_failure")
        if complete and manifest["output_sha256"] != proposed_result_sha256:
            raise HistoricalRunError("completed_results_bytes_differ")
        return RunStatus(True, complete, False, input_sha256, manifest["output_sha256"])

    result_path = output_dir / "historical-results.json"
    if not result_path.exists():
        return RunStatus(output_dir.exists(), False, False, None, None)
    result_bytes = result_path.read_bytes()
    result = _load_json(result_path)
    protocol_hash = result.get("input_protocol_sha256")
    panel_hash = result.get("input_panel_sha256")
    if not isinstance(protocol_hash, str) or not isinstance(panel_hash, str):
        raise HistoricalRunError("legacy_results_missing_input_identity")
    saved_input = results_input_identity(protocol_sha256=protocol_hash, panel_sha256=panel_hash)
    if saved_input != input_sha256:
        raise HistoricalRunError("scientific_inputs_require_new_output_directory")
    saved_output = _sha(result_bytes)
    if saved_output != proposed_result_sha256:
        raise HistoricalRunError("completed_results_bytes_differ")
    return RunStatus(True, True, True, saved_input, saved_output)


def _write_manifest(output_dir: Path, value: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "run.json"
    temporary = output_dir / ".run.json.pending"
    body = _json_bytes(value)
    with temporary.open("wb") as stream:
        stream.write(body)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def write_collection_manifest(
    output_dir: Path,
    *,
    protocol_sha256: str,
    input_sha256: str,
    panel_sha256: str,
    complete: bool,
) -> None:
    _write_manifest(
        output_dir,
        {
            "schema_version": RUN_SCHEMA,
            "run_kind": "collection",
            "protocol_sha256": _require_sha(protocol_sha256, "protocol_sha256"),
            "input_sha256": _require_sha(input_sha256, "input_sha256"),
            "output_sha256": _require_sha(panel_sha256, "panel_sha256"),
            "state": "complete" if complete else "partial",
        },
    )


def write_results_manifest(
    output_dir: Path,
    *,
    protocol_sha256: str,
    panel_sha256: str,
    input_sha256: str,
    result_sha256: str,
    complete: bool = True,
) -> None:
    _write_manifest(
        output_dir,
        {
            "schema_version": RUN_SCHEMA,
            "run_kind": "results",
            "protocol_sha256": _require_sha(protocol_sha256, "protocol_sha256"),
            "panel_sha256": _require_sha(panel_sha256, "panel_sha256"),
            "input_sha256": _require_sha(input_sha256, "input_sha256"),
            "output_sha256": _require_sha(result_sha256, "result_sha256"),
            "state": "complete" if complete else "partial",
        },
    )
