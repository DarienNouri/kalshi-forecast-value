"""Commands for collecting, evaluating and inspecting the historical study."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

from .config import ConfigError, load_config


def _json_object(path: Path, name: str) -> dict:
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    return value


def _error(error: ValueError | OSError) -> int:
    print(json.dumps({"ok": False, "error": str(error)}, indent=2))
    return 1


def _run_status(run_dir: Path) -> dict:
    """Inspect saved artifact identities without certifying statistical validity."""
    directory = run_dir.resolve()
    if not directory.is_dir():
        raise ValueError(f"Historical run directory does not exist: {run_dir}")
    panel_path = directory / "panel.json"
    if not panel_path.exists():
        collecting = (directory / "progress.json").exists() or (
            directory / "new-acquisition/progress.json"
        ).exists()
        return {
            "ok": True,
            "run_dir": str(directory),
            "state": "collecting" if collecting else "empty",
            "input_identity_verified": False,
        }
    panel = _json_object(panel_path, "Historical panel")
    if panel.get("schema_version") != "historical_panel_v1":
        raise ValueError("Expected historical_panel_v1")
    if panel.get("data_origin") != "empirical_historical":
        raise ValueError("Expected empirical_historical origin")
    complete = panel.get("complete_requested_dates") is True
    state = "collected" if complete else "incomplete"
    verified = False
    results_path = directory / "analysis/historical-results.json"
    if results_path.exists():
        results = _json_object(results_path, "Historical results")
        if not complete:
            raise ValueError("Historical evaluation cannot describe an incomplete panel")
        expected_schema = (
            "historical_replication_evaluation_v1"
            if panel.get("replication_schema_version") == "historical_replication_panel_v1"
            else "historical_evaluation_v1"
        )
        if results.get("schema_version") != expected_schema:
            raise ValueError(f"Expected {expected_schema}")
        panel_identity = hashlib.sha256(
            json.dumps(panel, sort_keys=True, allow_nan=False).encode()
        ).hexdigest()
        if results.get("input_panel_sha256") != panel_identity:
            raise ValueError("Historical result input panel identity mismatch")
        state, verified = "evaluated", True
    return {
        "ok": True,
        "run_dir": str(directory),
        "state": state,
        "data_origin": panel["data_origin"],
        "counts": panel.get("counts", {}),
        "collection_complete": complete,
        "evaluation_present": results_path.exists(),
        "input_identity_verified": verified,
        "note": "Artifact status and panel identity only; research review is separate.",
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kiq",
        description="Collect, evaluate and inspect the empirical historical weather study.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    status = commands.add_parser("status", help="Inspect one saved historical run.")
    status.add_argument(
        "--run-dir",
        type=Path,
        default=Path("outputs/historical-study"),
    )
    status.add_argument("--json", action="store_true")
    validate = commands.add_parser(
        "validate-config",
        help="Validate the empirical study configuration.",
    )
    validate.add_argument("--config", type=Path, default=Path("config/study.toml"))
    validate.add_argument("--json", action="store_true")

    history = commands.add_parser(
        "historical-study",
        help="Collect a bounded historical cohort or replay saved sources.",
    )
    history.add_argument("--config", type=Path, default=Path("config/historical-study.toml"))
    history.add_argument("--protocol", type=Path, default=Path("config/historical-protocol.json"))
    history.add_argument("--output-dir", type=Path, default=Path("outputs/historical-study"))
    history.add_argument("--cache-dir", type=Path)
    history.add_argument("--offline", action="store_true")
    history.add_argument("--max-days", type=int)
    history.add_argument("--json", action="store_true")

    report = commands.add_parser(
        "historical-report",
        help="Fit and evaluate a complete historical panel.",
    )
    report.add_argument("--panel", type=Path, default=Path("outputs/historical-study/panel.json"))
    report.add_argument("--protocol", type=Path, default=Path("config/historical-protocol.json"))
    report.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/historical-study/analysis"),
    )
    report.add_argument("--json", action="store_true")

    render = commands.add_parser(
        "historical-render",
        help="Render saved historical results without collection or fitting.",
    )
    render.add_argument("--results", type=Path, required=True)
    render.add_argument("--output-dir", type=Path, required=True)
    render.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    if args.command == "status":
        try:
            result = _run_status(args.run_dir)
        except (ValueError, OSError) as exc:
            return _error(exc)
        print(json.dumps(result, indent=2))
        return 0

    if args.command == "historical-study":
        from .historical import collect_historical_panel
        from .historical_inputs import validate_protocol

        try:
            config = load_config(args.config)
            protocol = validate_protocol(json.loads(args.protocol.read_bytes()), collection=True)
            options = {"cache_dir": args.cache_dir} if args.cache_dir is not None else {}
            panel = collect_historical_panel(
                config,
                protocol,
                args.output_dir,
                offline=args.offline,
                max_days=args.max_days,
                progress=lambda message: print(message, file=sys.stderr),
                **options,
            )
            complete = panel["complete_requested_dates"] is True
            eligible = bool(panel["cases"])
            result = {
                "ok": complete and eligible,
                "status": "incomplete" if not complete else "complete" if eligible else "empty",
                "data_origin": panel["data_origin"],
                "counts": panel["counts"],
                "complete_requested_dates": complete,
                "panel_path": str((args.output_dir / "panel.json").resolve()),
            }
        except (ValueError, OSError) as exc:
            return _error(exc)
        print(json.dumps(result, indent=2))
        return 2 if not complete else 0 if eligible else 1

    if args.command == "historical-report":
        from .historical_evaluation import evaluate_historical_panel, write_historical_report

        try:
            panel = _json_object(args.panel, "Historical panel")
            protocol = json.loads(args.protocol.read_bytes())
            if panel.get("complete_requested_dates") is not True:
                raise ValueError("historical_collection_incomplete")
            result = evaluate_historical_panel(panel, protocol)
            paths = write_historical_report(result, args.output_dir)
        except (ValueError, OSError) as exc:
            return _error(exc)
        print(json.dumps({"ok": True, "artifacts": paths}, indent=2))
        return 0

    if args.command == "historical-render":
        from .historical_reporting import render_saved_historical_results

        try:
            paths = render_saved_historical_results(args.results, args.output_dir)
        except (ValueError, OSError) as exc:
            return _error(exc)
        print(json.dumps({"ok": True, "artifacts": paths}, indent=2))
        return 0

    try:
        load_config(args.config)
    except ConfigError as exc:
        print(f"kiq: {exc}", file=sys.stderr)
        return 2

    result = {"config_valid": True, "config_path": str(args.config)}
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"Configuration valid: {args.config}")
    return 0
