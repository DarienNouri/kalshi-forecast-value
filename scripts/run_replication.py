#!/usr/bin/env python3
"""Collect, assemble, or analyze the frozen historical replication."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from kalshi_information_quality_lab.config import StudyConfig, load_config
from kalshi_information_quality_lab.historical import collect_historical_panel
from kalshi_information_quality_lab.historical_replication_inputs import (
    combine_replication_panels,
    derive_new_collection_protocol,
    inventory_partition_targets,
    load_frozen_replication_protocol,
)
from kalshi_information_quality_lab.historical_runs import (
    guard_results_publication,
    results_input_identity,
    write_results_manifest,
)
from kalshi_information_quality_lab.storage import publish_immutable_file

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = ROOT / "outputs/historical-replication"
DEFAULT_CONFIG = ROOT / "config/historical-study.toml"
DEFAULT_PROTOCOL = ROOT / "config/replication-protocol.json"


class ReplicationDriverError(ValueError):
    """The replication driver rejected an input or unsafe run boundary."""


class ReplicationGateError(ReplicationDriverError):
    """The assembled cohort is recorded but cannot be analyzed."""


def _sha(body: bytes) -> str:
    return hashlib.sha256(body).hexdigest()


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _semantic_hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _json_object(path: Path, name: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ReplicationDriverError(f"cannot_read_{name}: {exc}") from None
    if not isinstance(value, dict):
        raise ReplicationDriverError(f"{name}_must_be_json_object")
    return value


def _resolve(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _validate_run_dir(run_dir: Path, legacy_dir: Path) -> Path:
    outputs = (ROOT / "outputs").resolve()
    run_dir = _resolve(run_dir)
    legacy_dir = _resolve(legacy_dir)
    if run_dir == outputs or not run_dir.is_relative_to(outputs):
        raise ReplicationDriverError("run_dir_must_be_inside_project_outputs")
    if (
        run_dir == legacy_dir
        or run_dir.is_relative_to(legacy_dir)
        or legacy_dir.is_relative_to(run_dir)
    ):
        raise ReplicationDriverError("replication_run_must_be_disjoint_from_legacy_run")
    return run_dir


def _load_frozen_inputs(
    protocol_path: Path, config_path: Path
) -> tuple[dict[str, Any], bytes, StudyConfig]:
    protocol_path = _resolve(protocol_path)
    protocol = load_frozen_replication_protocol(protocol_path)
    protocol_bytes = protocol_path.read_bytes()
    config = load_config(_resolve(config_path), project_root=ROOT)
    return protocol, protocol_bytes, config


def _publish_protocol_snapshots(
    run_dir: Path, protocol_bytes: bytes, new_protocol: dict[str, Any]
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    publish_immutable_file(run_dir, "protocol.json", protocol_bytes)
    publish_immutable_file(run_dir, "new-acquisition-protocol.json", _json_bytes(new_protocol))


def _collect(
    protocol: dict[str, Any],
    config: StudyConfig,
    run_dir: Path,
    *,
    offline: bool,
) -> dict[str, Any]:
    new_protocol = derive_new_collection_protocol(protocol)
    panel = collect_historical_panel(
        config,
        new_protocol,
        run_dir / "new-acquisition",
        offline=offline,
        progress=lambda message: print(message, file=sys.stderr, flush=True),
    )
    return {
        "complete": panel.get("complete_requested_dates") is True,
        "counts": panel.get("counts"),
        "panel_path": str((run_dir / "new-acquisition/panel.json").resolve()),
    }


def _raw_directory(panel: dict[str, Any], name: str) -> Path:
    provenance = panel.get("provenance")
    if not isinstance(provenance, dict) or not isinstance(provenance.get("raw_directory"), str):
        raise ReplicationDriverError(f"{name}_panel_missing_raw_directory")
    raw = Path(provenance["raw_directory"])
    if raw.is_absolute() or ".." in raw.parts:
        raise ReplicationDriverError(f"{name}_raw_directory_must_be_project_relative")
    resolved = (ROOT / raw).resolve()
    if not resolved.is_relative_to(ROOT / "outputs"):
        raise ReplicationDriverError(f"{name}_raw_directory_outside_outputs")
    return raw


def _assemble(
    protocol: dict[str, Any],
    protocol_sha256: str,
    config: StudyConfig,
    run_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    legacy_dir = _resolve(Path(protocol["legacy"]["run_dir"]))
    legacy_panel_path = legacy_dir / "panel.json"
    legacy_results_path = legacy_dir / "analysis/historical-results.json"
    new_panel_path = run_dir / "new-acquisition/panel.json"
    try:
        legacy_panel_bytes = legacy_panel_path.read_bytes()
        legacy_results_bytes = legacy_results_path.read_bytes()
        new_panel_bytes = new_panel_path.read_bytes()
    except OSError as exc:
        raise ReplicationDriverError(f"replication_partition_missing: {exc.filename}") from None
    if _sha(legacy_panel_bytes) != protocol["legacy"]["panel_sha256"]:
        raise ReplicationDriverError("legacy_panel_file_sha256_mismatch")
    if _sha(legacy_results_bytes) != protocol["legacy"]["results_sha256"]:
        raise ReplicationDriverError("legacy_results_file_sha256_mismatch")
    legacy_panel = _json_object(legacy_panel_path, "legacy_panel")
    new_panel = _json_object(new_panel_path, "new_panel")
    legacy_protocol = dict(protocol["collection"])
    legacy_protocol["end_date"] = protocol["legacy"]["end_date"]
    legacy_protocol["train_end_date"] = legacy_panel["protocol"]["train_end_date"]
    legacy_protocol["validation_end_date"] = legacy_panel["protocol"]["validation_end_date"]
    new_protocol = derive_new_collection_protocol(protocol)
    print("Checking target rules from the earlier study", file=sys.stderr, flush=True)
    legacy_inventory = inventory_partition_targets(
        config, legacy_protocol, _raw_directory(legacy_panel, "legacy")
    )
    print("Inventorying new-partition target rules", file=sys.stderr, flush=True)
    new_inventory = inventory_partition_targets(
        config, new_protocol, _raw_directory(new_panel, "new")
    )
    combined = combine_replication_panels(
        protocol,
        legacy_panel_bytes,
        new_panel_bytes,
        replication_protocol_sha256=protocol_sha256,
        target_inventory=[*legacy_inventory, *new_inventory],
    )
    body = _json_bytes(combined)
    gate = combined["target_continuity"]
    analysis_ready = (
        combined.get("complete_requested_dates") is True and gate.get("analysis_ready") is True
    )
    if analysis_ready:
        panel_path = publish_immutable_file(run_dir, "panel.json", body)
        status = "ready"
    else:
        diagnostics_dir = run_dir / "diagnostics"
        diagnostics_dir.mkdir(parents=True, exist_ok=True)
        panel_path = publish_immutable_file(
            diagnostics_dir,
            f"assembly-{_sha(body)}.json",
            body,
        )
        status = "gate_failed"
    summary = {
        "status": status,
        "complete_requested_dates": combined["complete_requested_dates"],
        "analysis_ready": analysis_ready,
        "counts": combined["counts"],
        "target_continuity": gate,
        "panel_path": str(panel_path.resolve()),
        "panel_file_sha256": _sha(body),
    }
    return combined, summary


def _gate(panel: dict[str, Any]) -> None:
    target = panel.get("target_continuity")
    if panel.get("complete_requested_dates") is not True:
        raise ReplicationGateError("replication_collection_incomplete")
    if not isinstance(target, dict) or target.get("analysis_ready") is not True:
        raise ReplicationGateError("replication_target_continuity_not_ready")


def _preflight_existing_results(
    panel: dict[str, Any], protocol: dict[str, Any], result_dir: Path
) -> None:
    result_path = result_dir / "historical-results.json"
    protocol_sha256 = _semantic_hash(protocol)
    panel_sha256 = _semantic_hash(panel)
    input_sha256 = results_input_identity(
        protocol_sha256=protocol_sha256,
        panel_sha256=panel_sha256,
    )
    if not result_path.exists():
        if (result_dir / "run.json").exists():
            guard_results_publication(
                result_dir,
                input_sha256=input_sha256,
                proposed_result_sha256="0" * 64,
            )
        return
    saved = _json_object(result_path, "existing_replication_results")
    if saved.get("schema_version") != "historical_replication_evaluation_v1":
        raise ReplicationDriverError("existing_results_are_not_replication_v1")
    if (
        saved.get("input_protocol_sha256") != protocol_sha256
        or saved.get("input_panel_sha256") != panel_sha256
    ):
        raise ReplicationDriverError("existing_results_scientific_inputs_mismatch")
    body = result_path.read_bytes()
    guard_results_publication(
        result_dir,
        input_sha256=input_sha256,
        proposed_result_sha256=_sha(body),
    )


def _analyze(protocol: dict[str, Any], run_dir: Path, *, workers: int) -> dict[str, Any]:
    from kalshi_information_quality_lab.historical_replication import evaluate_replication

    panel = _json_object(run_dir / "panel.json", "combined_panel")
    _gate(panel)
    result_dir = run_dir / "analysis"
    _preflight_existing_results(panel, protocol, result_dir)
    result = evaluate_replication(
        panel,
        protocol,
        workers=workers,
        progress=lambda message: print(message, file=sys.stderr, flush=True),
    )
    body = _json_bytes(result)
    protocol_sha256 = result["input_protocol_sha256"]
    panel_sha256 = result["input_panel_sha256"]
    input_sha256 = results_input_identity(
        protocol_sha256=protocol_sha256,
        panel_sha256=panel_sha256,
    )
    result_sha256 = _sha(body)
    status = guard_results_publication(
        result_dir,
        input_sha256=input_sha256,
        proposed_result_sha256=result_sha256,
    )
    result_dir.mkdir(parents=True, exist_ok=True)
    publish_immutable_file(result_dir, "historical-results.json", body)
    if not (status.complete and status.legacy):
        write_results_manifest(
            result_dir,
            protocol_sha256=protocol_sha256,
            panel_sha256=panel_sha256,
            input_sha256=input_sha256,
            result_sha256=result_sha256,
        )
    bootstrap = result.get("bootstrap", {}).get("primary", {})
    return {
        "analysis_status": result.get("analysis_status"),
        "cohort": result.get("cohort"),
        "bootstrap_attempted": bootstrap.get("attempted"),
        "bootstrap_valid": bootstrap.get("valid"),
        "results_path": str((result_dir / "historical-results.json").resolve()),
        "results_file_sha256": result_sha256,
        "input_panel_sha256": panel_sha256,
        "input_protocol_sha256": protocol_sha256,
    }


def run(args: argparse.Namespace) -> tuple[int, dict[str, Any]]:
    if type(args.workers) is not int or not 1 <= args.workers <= 8:
        raise ReplicationDriverError("workers_must_be_integer_from_1_to_8")
    protocol, protocol_bytes, config = _load_frozen_inputs(args.protocol, args.config)
    legacy_dir = Path(protocol["legacy"]["run_dir"])
    run_dir = _validate_run_dir(args.run_dir, legacy_dir)
    new_protocol = derive_new_collection_protocol(protocol)
    _publish_protocol_snapshots(run_dir, protocol_bytes, new_protocol)
    stages: dict[str, Any] = {}
    if args.stage in {"collect", "all"}:
        stages["collect"] = _collect(protocol, config, run_dir, offline=args.offline)
        if not stages["collect"]["complete"]:
            return 2, {"ok": False, "status": "collection_incomplete", "stages": stages}
    panel: dict[str, Any] | None = None
    if args.stage in {"assemble", "all"}:
        panel, stages["assemble"] = _assemble(protocol, _sha(protocol_bytes), config, run_dir)
        if not stages["assemble"]["analysis_ready"]:
            return 2, {"ok": False, "status": "gate_failed", "stages": stages}
    if args.stage in {"analyze", "all"}:
        if panel is None:
            panel = _json_object(run_dir / "panel.json", "combined_panel")
        try:
            _gate(panel)
        except ReplicationGateError as exc:
            return 2, {
                "ok": False,
                "status": "gate_failed",
                "error": str(exc),
                "stages": stages,
            }
        stages["analyze"] = _analyze(protocol, run_dir, workers=args.workers)
    return 0, {"ok": True, "status": "complete", "stage": args.stage, "stages": stages}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("collect", "assemble", "analyze", "all"), default="all")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--workers", type=int, default=1)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    code, payload = run(_parse_args(argv))
    print(json.dumps(payload, indent=2, sort_keys=True))
    return code


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ReplicationDriverError, ValueError, OSError) as exc:
        print(
            json.dumps({"ok": False, "status": "error", "error": str(exc)}, indent=2),
            file=sys.stderr,
        )
        raise SystemExit(1) from None
