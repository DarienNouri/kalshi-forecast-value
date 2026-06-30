#!/usr/bin/env python3
"""Reproduce a historical run from its retained cache without network access.

The source run is immutable. All progress, panels, reports, logs, and evidence are
written to a separate destination.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import selectors
import subprocess
import sys
import time
from copy import deepcopy
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "outputs/historical-study"
DEFAULT_OUTPUT = ROOT / "outputs/reproduction"
BOOTSTRAP = """
import socket
def deny(*args, **kwargs):
    raise RuntimeError('Network disabled during historical reproduction')
socket.socket.connect = deny
socket.create_connection = deny
socket.getaddrinfo = deny
from kalshi_information_quality_lab.cli import main
raise SystemExit(main())
"""


class ReproductionError(ValueError):
    """The requested replay cannot establish the reproduction contract."""


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def scientific_hash(value: dict[str, Any], *, result: bool = False) -> str:
    """Compare all research fields after removing obsolete administrative metadata.

    Result-to-panel identity is checked separately against each exact panel. Removing
    an administrative panel field changes that identity without changing the study.
    """
    value = deepcopy(value)
    value.get("provenance", {}).pop("execution_basis", None)
    if result:
        value.pop("input_panel_sha256", None)
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def check_result_panel(panel: dict[str, Any], result: dict[str, Any]) -> None:
    identity = hashlib.sha256(
        json.dumps(panel, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()
    if result.get("input_panel_sha256") != identity:
        raise ReproductionError("result does not identify its input panel")


def _tree_identity(root: Path) -> str:
    """Hash names, sizes, and bytes so source mutation is detectable after replay."""
    accumulator = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix().encode()
        accumulator.update(len(relative).to_bytes(8, "big"))
        accumulator.update(relative)
        accumulator.update(path.stat().st_size.to_bytes(8, "big"))
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                accumulator.update(chunk)
    return accumulator.hexdigest()


def _json_object(path: Path, name: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ReproductionError(f"cannot read {name}: {exc}") from None
    if not isinstance(value, dict):
        raise ReproductionError(f"{name} must be a JSON object")
    return value


def _date_count(protocol: dict[str, Any]) -> int:
    try:
        start = date.fromisoformat(protocol["start_date"])
        end = date.fromisoformat(protocol["end_date"])
    except (KeyError, TypeError, ValueError):
        raise ReproductionError("protocol requires valid start_date and end_date") from None
    if end < start:
        raise ReproductionError("protocol end_date precedes start_date")
    return (end - start).days + 1


def _terminate(process: subprocess.Popen[bytes]) -> int:
    if process.poll() is None:
        process.terminate()
    try:
        return process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        return process.wait(timeout=5)


def invoke(
    args: list[str],
    name: str,
    *,
    log_dir: Path,
    timeout_seconds: float,
    interrupt_after: str | None = None,
) -> dict[str, Any]:
    """Run one offline child with a deadline that includes reading its output."""
    if timeout_seconds <= 0:
        raise ReproductionError("child timeout must be positive")
    log = log_dir / f"{name}.log"
    argv = [sys.executable, "-u", "-c", BOOTSTRAP, *args]
    process = subprocess.Popen(
        argv,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=os.environ.copy(),
    )
    assert process.stdout is not None
    selector = selectors.DefaultSelector()
    selector.register(process.stdout, selectors.EVENT_READ)
    deadline = time.monotonic() + timeout_seconds
    pending = b""
    terminated = False
    timed_out = False
    with log.open("wb") as stream:
        while selector.get_map():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                timed_out = True
                break
            events = selector.select(min(remaining, .25))
            if not events:
                if process.poll() is not None:
                    events = [(selector.get_key(process.stdout), selectors.EVENT_READ)]
                else:
                    continue
            for key, _ in events:
                chunk = os.read(key.fd, 65536)
                if not chunk:
                    selector.unregister(key.fileobj)
                    continue
                stream.write(chunk)
                stream.flush()
                pending += chunk
                lines = pending.split(b"\n")
                pending = lines.pop()
                for raw_line in lines:
                    line = raw_line.decode(errors="replace")
                    if line.startswith("20"):
                        print(line, flush=True)
                    if interrupt_after is not None and line.startswith(interrupt_after):
                        process.terminate()
                        terminated = True
                        interrupt_after = None
        if pending:
            stream.write(pending)
    selector.close()
    process.stdout.close()
    code = _terminate(process) if timed_out or terminated else process.wait(timeout=5)
    if timed_out:
        raise ReproductionError(f"{name} exceeded {timeout_seconds:g}s; see {log}")
    if terminated:
        if code >= 0:
            raise ReproductionError(f"{name} did not terminate by signal; exit {code}; see {log}")
    elif code != 0:
        raise ReproductionError(f"{name} exited {code}; see {log}")
    try:
        stdout_path = log.relative_to(ROOT).as_posix()
    except ValueError:
        stdout_path = str(log.resolve())
    return {
        "argv": argv,
        "exit_code": code,
        "stdout_path": stdout_path,
        "stdout_sha256": digest(log),
        "interrupted": terminated,
        "timed_out": False,
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--config", type=Path, default=ROOT / "config/historical-study.toml")
    parser.add_argument("--protocol", type=Path, default=ROOT / "config/historical-protocol.json")
    parser.add_argument("--child-timeout", type=float, default=900)
    return parser.parse_args(argv)


def _resolved(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    source = _resolved(args.run_dir).resolve()
    output = _resolved(args.output_dir).resolve()
    config = _resolved(args.config).resolve()
    protocol_path = _resolved(args.protocol).resolve()
    if source == output:
        raise ReproductionError("source run and reproduction output must differ")
    try:
        output.relative_to(source)
    except ValueError:
        pass
    else:
        raise ReproductionError("reproduction output must not be inside the source run")
    if args.child_timeout <= 0:
        raise ReproductionError("child timeout must be positive")

    source_panel = source / "panel.json"
    source_results = source / "analysis/historical-results.json"
    source_raw = source / "raw"
    for path, name in (
        (source_panel, "source panel"),
        (source_results, "source results"),
        (source_raw, "source raw cache"),
        (config, "study config"),
        (protocol_path, "protocol"),
    ):
        if not path.exists():
            raise ReproductionError(f"missing {name}: {path}")
    if Path(sys.prefix).resolve() == (ROOT / ".venv").resolve():
        raise ReproductionError("use uv run --isolated for clean-environment evidence")
    protocol = _json_object(protocol_path, "protocol")
    expected_dates = _date_count(protocol)
    first_date = str(protocol["start_date"])
    source_result_value = _json_object(source_results, "source results")
    expected_panel_hash = digest(source_panel)
    expected_result_hash = digest(source_results)
    panel_value = _json_object(source_panel, "source panel")
    check_result_panel(panel_value, source_result_value)
    protocol_identity = hashlib.sha256(
        json.dumps(protocol, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()
    if source_result_value.get("input_protocol_sha256") != protocol_identity:
        raise ReproductionError("requested protocol differs from the source result's protocol")
    expected_scientific_panel = scientific_hash(panel_value)
    expected_scientific_results = scientific_hash(source_result_value, result=True)

    protected = [source_panel, source_results, protocol_path, config]
    before_files = {str(path): digest(path) for path in protected}
    before_tree = _tree_identity(source_raw)
    existing_evidence_path = output / "reproduction.json"
    existing_panel = output / "panel.json"
    existing_results = output / "analysis/historical-results.json"
    if existing_evidence_path.exists() and existing_panel.exists() and existing_results.exists():
        evidence = _json_object(existing_evidence_path, "reproduction evidence")
        replayed_panel = _json_object(existing_panel, "replayed panel")
        replayed_results = _json_object(existing_results, "replayed results")
        check_result_panel(replayed_panel, replayed_results)
        if (
            evidence.get("schema_version") != 3
            or evidence.get("scientific_content_matches") is not True
            or evidence.get("panel_scientific_sha256") != expected_scientific_panel
            or evidence.get("results_scientific_sha256") != expected_scientific_results
            or evidence.get("protocol_sha256") != digest(protocol_path)
            or evidence.get("config_sha256") != digest(config)
            or evidence.get("environment", {}).get("uv_lock_sha256") != digest(ROOT / "uv.lock")
            or scientific_hash(replayed_panel) != expected_scientific_panel
            or scientific_hash(replayed_results, result=True) != expected_scientific_results
            or evidence.get("panel_sha256") != digest(existing_panel)
            or evidence.get("results_sha256") != digest(existing_results)
            or evidence.get("source_panel_sha256") != expected_panel_hash
            or evidence.get("source_results_sha256") != expected_result_hash
            or evidence.get("source_raw_tree_sha256") != before_tree
            or evidence.get("source_preserved") is not True
        ):
            raise ReproductionError("existing reproduction does not match the current source run")
        print(json.dumps(evidence, indent=2, sort_keys=True))
        return 0
    for stale in (
        existing_panel,
        output / "progress.json",
        existing_results,
    ):
        if stale.exists():
            raise ReproductionError(
                "reproduction destination already contains run state; "
                f"choose a new --output-dir: {stale}"
            )
    logs = output / "reproduction"
    logs.mkdir(parents=True, exist_ok=True)
    common = [
        "historical-study",
        "--config",
        str(config),
        "--protocol",
        str(protocol_path),
        "--output-dir",
        str(output),
        "--cache-dir",
        str(source_raw),
        "--offline",
        "--json",
    ]
    commands = [
        invoke(
            common,
            "interrupted",
            log_dir=logs,
            timeout_seconds=args.child_timeout,
            interrupt_after=f"{first_date}:",
        )
    ]
    if (output / "panel.json").exists():
        raise ReproductionError("interrupted replay published a completed panel")
    commands.append(invoke(common, "replayed", log_dir=logs, timeout_seconds=args.child_timeout))
    destination_panel = output / "panel.json"
    replayed_panel = _json_object(destination_panel, "replayed panel")
    if scientific_hash(replayed_panel) != expected_scientific_panel:
        raise ReproductionError("replayed research panel differs from the source panel")
    commands.append(
        invoke(
            [
                "historical-report",
                "--panel",
                str(destination_panel),
                "--protocol",
                str(protocol_path),
                "--output-dir",
                str(output / "analysis"),
                "--json",
            ],
            "evaluated",
            log_dir=logs,
            timeout_seconds=args.child_timeout,
        )
    )
    destination_results = output / "analysis/historical-results.json"
    replayed_results = _json_object(destination_results, "replayed results")
    check_result_panel(replayed_panel, replayed_results)
    if scientific_hash(replayed_results, result=True) != expected_scientific_results:
        raise ReproductionError("reproduced research results differ from the source results")
    progress = _json_object(output / "progress.json", "replay progress")
    if progress.get("fetched") != 0 or progress.get("days_completed") != expected_dates:
        raise ReproductionError("offline replay did not complete every date with zero downloads")

    after_files = {str(path): digest(path) for path in protected}
    after_tree = _tree_identity(source_raw)
    if after_files != before_files or after_tree != before_tree:
        raise ReproductionError("source run or reused raw cache changed during reproduction")
    isolated_prefix = Path(sys.prefix).resolve()
    evidence = {
        "schema_version": 3,
        "recorded_at": datetime.now(UTC).isoformat(),
        "source_run": str(source),
        "output_run": str(output),
        "panel_sha256": digest(destination_panel),
        "results_sha256": digest(destination_results),
        "source_panel_sha256": expected_panel_hash,
        "source_results_sha256": expected_result_hash,
        "panel_scientific_sha256": expected_scientific_panel,
        "results_scientific_sha256": expected_scientific_results,
        "scientific_content_matches": True,
        "comparison": "all research fields; obsolete administrative scope metadata excluded; "
        "each result's panel identity verified separately",
        "protocol_sha256": digest(protocol_path),
        "config_sha256": digest(config),
        "source_raw_tree_sha256": before_tree,
        "source_preserved": True,
        "commands": commands,
        "environment": {
            "python_version": platform.python_version(),
            "python_executable": sys.executable,
            "prefix": str(isolated_prefix),
            "project_venv": str((ROOT / ".venv").resolve()),
            "isolated_prefix_verified": isolated_prefix != (ROOT / ".venv").resolve(),
            "uv_lock_sha256": digest(ROOT / "uv.lock"),
            "network": "uv --offline plus socket connections denied in child processes",
        },
        "recovery": {
            "interrupted": commands[0]["interrupted"],
            "resumed": True,
            "interrupted_exit_code": commands[0]["exit_code"],
            "resumed_exit_code": commands[1]["exit_code"],
            "processed_dates_before_interruption": 1,
            "processed_dates_after_resume": progress["days_completed"],
            "source_downloads_on_replay": progress["fetched"],
            "source_replays": progress.get("replayed"),
            "interruption": "SIGTERM after the first completed date in a separate OS process",
            "scope": "Recovery from an interrupted replay of retained raw inputs",
        },
    }
    evidence_path = output / "reproduction.json"
    evidence_path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    print(json.dumps(evidence, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ReproductionError, OSError) as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, indent=2), file=sys.stderr)
        raise SystemExit(1) from None
