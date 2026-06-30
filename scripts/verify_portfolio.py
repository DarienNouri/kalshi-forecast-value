#!/usr/bin/env python3
"""Independently verify a static portfolio build and its saved scientific identities."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT = ROOT / "reports/portfolio-evidence.json"


class PortfolioVerificationError(ValueError):
    """The portfolio evidence manifest conflicts with its inputs or outputs."""


def _digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _json_object(path: Path, name: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise PortfolioVerificationError(f"cannot read {name}: {exc}") from None
    if not isinstance(value, dict):
        raise PortfolioVerificationError(f"{name} must be a JSON object")
    return value


def _path(value: object, name: str) -> Path:
    if not isinstance(value, str) or not value:
        raise PortfolioVerificationError(f"{name} path is invalid")
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _semantic_hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        raise PortfolioVerificationError(f"replication {name} is invalid")
    return float(value)


def _count(value: object, name: str) -> int:
    if type(value) is not int or value < 0:
        raise PortfolioVerificationError(f"replication {name} is invalid")
    return value


def _replication_comparison(item: object, name: str) -> dict[str, Any]:
    if not isinstance(item, dict):
        raise PortfolioVerificationError(f"replication {name} comparison is invalid")
    reason = item.get("interval_unavailable_reason")
    intervals = {}
    for key in ("ci95", "ci975"):
        interval = item.get(key)
        if interval is None:
            if not isinstance(reason, str) or not reason:
                raise PortfolioVerificationError(f"replication {name} interval reason is missing")
            intervals[key] = None
        elif not isinstance(interval, list) or len(interval) != 2:
            raise PortfolioVerificationError(f"replication {name} interval is invalid")
        else:
            low = _finite(interval[0], f"{name} {key}")
            high = _finite(interval[1], f"{name} {key}")
            if low > high:
                raise PortfolioVerificationError(f"replication {name} interval is reversed")
            intervals[key] = [low, high]
    return {
        "horizon_hours": item.get("horizon_hours"),
        "method": item.get("method"),
        "reference": item.get("reference"),
        "metric": item.get("metric"),
        "mean_difference": _finite(item.get("mean_difference"), f"{name} difference"),
        **intervals,
        "valid_replicates": _count(item.get("valid_replicates"), f"{name} valid"),
        "undefined_replicates": _count(item.get("undefined_replicates"), f"{name} undefined"),
        "interval_unavailable_reason": reason,
    }


def _replication_bootstrap(results: dict[str, Any]) -> dict[str, Any]:
    bootstrap = results["bootstrap"]

    def block(name: str, value: dict[str, Any]) -> dict[str, Any]:
        comparisons = [
            _replication_comparison(item, name)
            for item in value["comparisons"]
            if item.get("method") == "trained_blend"
            and item.get("reference") == "public_only"
            and item.get("metric") == "brier"
        ]
        conditions: list[dict[str, Any]] = []
        for item in value["conditions"]:
            horizon = _count(item.get("horizon_hours"), f"{name} condition horizon")
            interaction = _replication_comparison(item["interactions"][0], f"{name} condition")
            interaction["horizon_hours"] = horizon
            conditions.append(
                {
                    "horizon_hours": horizon,
                    "supported": item["support"].get("supported") is True,
                    "support": item["support"],
                    "interaction": interaction,
                }
            )
        return {
            "mode": value["mode"],
            "block_days": value.get("block_days"),
            "attempted": value["attempted"],
            "valid": value["valid"],
            "failed": value["failed"],
            "selection_counts": value.get("selection_counts", {}),
            "primary_comparisons": sorted(
                comparisons, key=lambda item: item["horizon_hours"], reverse=True
            ),
            "conditions": sorted(conditions, key=lambda item: item["horizon_hours"], reverse=True),
        }

    sensitivity = bootstrap["sensitivity_by_block_days"]
    return {
        "full_refit": block("primary", bootstrap["primary"]),
        "sensitivity_by_block_days": {
            day: block(f"{day}-day sensitivity", sensitivity[day])
            for day in sorted(sensitivity, key=int)
        },
        "fixed_fit": block("fixed_fit", bootstrap["fixed_fit"]),
        "family_coverage": bootstrap.get("family_coverage"),
        "limitations": bootstrap.get("limitations"),
    }


def _replication_sensitivities(results: dict[str, Any]) -> dict[str, Any]:
    output = {}
    for name in ("label_substitution", "normalization_projection"):
        item = results["sensitivities"][name]
        record: dict[str, Any] = {"status": item["status"]}
        if item["status"] == "unavailable":
            record["reason"] = item["reason"]
        else:
            record["trained_blend_brier_deltas"] = sorted(
                [
                    {
                        "horizon_hours": row["horizon_hours"],
                        "difference_from_primary": float(row["brier_difference_from_primary"]),
                    }
                    for row in item["holdout_deltas"]
                    if row.get("method") == "trained_blend"
                    and "brier_difference_from_primary" in row
                ],
                key=lambda row: row["horizon_hours"],
                reverse=True,
            )
        output[name] = record
    return output


def _replication_holdout(results: dict[str, Any]) -> list[dict[str, Any]]:
    methods = ("public_only", "market_only", "equal_blend", "trained_blend", "p0_reference")
    rows = []
    for summary in results["summaries"]:
        if summary["split"] != "holdout":
            continue
        comparison = next(
            (
                item
                for item in summary.get("paired_comparisons", [])
                if item.get("method") == "trained_blend" and item.get("metric") == "brier"
            ),
            None,
        )
        rows.append(
            {
                "horizon_hours": summary["horizon_hours"],
                "cases": summary["cases"],
                "dates": summary["dates"],
                "scores": {
                    method: {
                        "brier": summary["scores"][method]["brier"],
                        "log_loss": summary["scores"][method]["log_loss"],
                    }
                    for method in methods
                },
                "trained_minus_public_brier": comparison,
            }
        )
    return sorted(rows, key=lambda item: item["horizon_hours"], reverse=True)


def _verify_replication(record: object, result_override: Path | None = None) -> str:
    if not isinstance(record, dict) or record.get("status") != "included":
        raise PortfolioVerificationError("replication result status is invalid")
    result_path = (
        result_override.resolve()
        if result_override is not None
        else _path(record.get("result_path"), "replication results")
    )
    if result_override is not None and not result_path.is_file():
        raise PortfolioVerificationError("explicit replication result file is missing")
    if not result_path.is_file():
        return "recorded_not_rechecked"
    if _digest(result_path) != record.get("result_file_sha256"):
        raise PortfolioVerificationError("replication result file hash conflict")
    results = _json_object(result_path, "replication results")
    if (
        results.get("schema_version") != "historical_replication_evaluation_v1"
        or results.get("data_origin") != "empirical_historical"
        or results.get("analysis_status") != "exploratory_historical_replication"
        or results.get("complete_requested_dates") is not True
    ):
        raise PortfolioVerificationError("replication result contract is invalid")
    if _semantic_hash(results["protocol"]) not in {
        results.get("input_protocol_sha256"),
        results.get("protocol_sha256"),
    } or results.get("input_protocol_sha256") != results.get("protocol_sha256"):
        raise PortfolioVerificationError("replication protocol semantic hash conflict")
    fit = results["fit"]
    oof = fit["oof"]
    blend_ids = set(fit["blend_training_case_ids"])
    blend_predictions = [
        prediction for prediction in oof["predictions"] if prediction.get("case_id") in blend_ids
    ]
    blend_dates = sorted({prediction["outcome_date"] for prediction in blend_predictions})
    if len(blend_predictions) != len(blend_ids) or not blend_dates:
        raise PortfolioVerificationError("replication blend prediction coverage is invalid")
    expected_fields = {
        "schema_version": results["schema_version"],
        "analysis_status": results["analysis_status"],
        "cohort": results["cohort"],
        "study_window": {
            key: results["protocol"]["collection"].get(key)
            for key in ("start_date", "train_end_date", "validation_end_date", "end_date")
        },
        "coverage": results["coverage"],
        "fit": {
            "selected_candidate": fit.get("selected_candidate"),
            "trained_market_weight": float(fit["trained_market_weight"]),
            "validation_fit_at": fit.get("validation_fit_at"),
            "final_fit_at": fit.get("final_fit_at"),
            "oof": {
                "spec": oof.get("spec"),
                "error_fold_start_month": oof.get("start_month"),
                "error_fold_end_month": oof.get("end_month"),
                "fit_count": len(oof.get("fits", [])),
                "prediction_count": len(oof.get("predictions", [])),
                "blend_prediction_start_date": blend_dates[0],
                "blend_prediction_end_date": blend_dates[-1],
                "blend_prediction_dates": len(blend_dates),
                "blend_prediction_cases": len(blend_predictions),
            },
        },
        "candidate_validation": [
            {
                "candidate": row.get("candidate"),
                "mean_brier": float(row["mean_brier"]),
                "cases": row["cases"],
                "dates": row["dates"],
            }
            for row in results["candidate_validation"]
        ],
        "holdout": _replication_holdout(results),
        "bootstrap": _replication_bootstrap(results),
        "sensitivities": _replication_sensitivities(results),
        "input_panel_sha256": results["input_panel_sha256"],
        "input_protocol_sha256": results["input_protocol_sha256"],
        "result_file_sha256": _digest(result_path),
    }
    for key, expected in expected_fields.items():
        if record.get(key) != expected:
            raise PortfolioVerificationError(f"replication aggregate {key} is stale or modified")
    panel_value = record.get("panel_path")
    if isinstance(panel_value, str):
        panel_path = _path(panel_value, "replication panel")
        if (
            panel_path.is_file()
            and _semantic_hash(_json_object(panel_path, "replication panel"))
            != results["input_panel_sha256"]
        ):
            raise PortfolioVerificationError("replication panel semantic hash conflict")
    quality = record.get("data_quality")
    if not isinstance(quality, dict):
        raise PortfolioVerificationError("replication data quality aggregate is missing")
    quality_path = _path(quality.get("record_path"), "replication data quality")
    if quality_path.is_file():
        if _digest(quality_path) != quality.get("record_sha256"):
            raise PortfolioVerificationError("replication data quality hash conflict")
        source_quality = _json_object(quality_path, "replication data quality")
        if (
            source_quality.get("eligible_cases") != results["cohort"].get("cases")
            or source_quality.get("eligible_dates") != results["cohort"].get("dates")
            or source_quality.get("requested_dates") != results["coverage"].get("requested_dates")
        ):
            raise PortfolioVerificationError("replication data quality conflicts with result")
        if isinstance(panel_value, str):
            panel_path = _path(panel_value, "replication panel")
            if panel_path.is_file() and source_quality.get("panel_file_sha256") != _digest(
                panel_path
            ):
                raise PortfolioVerificationError("replication data quality panel hash conflict")
        expected_monthly = [
            {
                "month": row["month"],
                "requested": row["requested"],
                "usable": row["case"],
                "excluded": row.get("excluded", 0),
            }
            for row in source_quality["monthly_coverage"]
        ]
        expected_quality = {
            "requested_dates": source_quality["requested_dates"],
            "eligible_dates": source_quality["eligible_dates"],
            "excluded_dates": source_quality["excluded_dates"],
            "eligible_cases": source_quality["eligible_cases"],
            "target_rules_compatible_dates": source_quality["target_rules_compatible_dates"],
            "monthly_coverage": expected_monthly,
            "limitations": source_quality.get("limitations", []),
            "panel_file_sha256": source_quality.get("panel_file_sha256"),
            "record_path": record["data_quality"]["record_path"],
            "record_sha256": _digest(quality_path),
        }
        if quality != expected_quality:
            raise PortfolioVerificationError(
                "replication data quality aggregate is stale or modified"
            )
    return "verified"


def _verify_file(record: object, name: str) -> Path:
    if not isinstance(record, dict):
        raise PortfolioVerificationError(f"{name} record is invalid")
    path = _path(record.get("path"), name)
    expected = record.get("sha256")
    if not isinstance(expected, str) or _digest(path) != expected:
        raise PortfolioVerificationError(f"{name} hash conflict")
    return path


def _verify_figures(output: dict[str, Any], replication: object, index_path: Path) -> int:
    figures = output.get("figures", {})
    if not isinstance(figures, dict):
        raise PortfolioVerificationError("figure records are invalid")
    if not isinstance(replication, dict) or replication.get("status") != "included":
        if figures:
            raise PortfolioVerificationError("figures require the replication result")
        return 0
    expected = {"monthly-coverage", "holdout-losses", "paired-differences"}
    if set(figures) != expected:
        raise PortfolioVerificationError("replication requires all three aggregate figures")
    scores = []
    dates = {}
    for row in replication["holdout"]:
        values = {}
        methods = ("public_only", "market_only", "equal_blend", "trained_blend", "p0_reference")
        for method in methods:
            values[method] = row["scores"][method]["brier"]
        scores.append(
            {"horizon_hours": row["horizon_hours"], "dates": row["dates"], "scores": values}
        )
        dates[row["horizon_hours"]] = row["dates"]
    contrasts = []
    for row in replication["bootstrap"]["full_refit"]["primary_comparisons"]:
        contrasts.append(
            {
                "horizon_hours": row["horizon_hours"],
                "dates": dates[row["horizon_hours"]],
                "mean_difference": row["mean_difference"],
                "ci95": row["ci95"],
                "ci975": row["ci975"],
            }
        )
    expected_data = {
        "monthly-coverage": replication["data_quality"]["monthly_coverage"],
        "holdout-losses": scores,
        "paired-differences": contrasts,
    }
    document = index_path.read_text()
    for name in sorted(expected):
        record = figures[name]
        path = _verify_file(record, name)
        if path != index_path.parent / "assets" / f"{name}.png":
            raise PortfolioVerificationError(f"{name} path does not match the page link")
        if f'src="assets/{name}.png"' not in document:
            raise PortfolioVerificationError(f"{name} is missing from the page")
        if not path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n"):
            raise PortfolioVerificationError(f"{name} is not a PNG figure")
        if record.get("data") != expected_data[name]:
            raise PortfolioVerificationError(f"{name} aggregate data conflict")
        if record.get("data_sha256") != _semantic_hash(expected_data[name]):
            raise PortfolioVerificationError(f"{name} data hash conflict")
        if record.get("source_result_sha256") != replication["result_file_sha256"]:
            raise PortfolioVerificationError(f"{name} source result hash conflict")
    return len(figures)


def verify(report_path: Path, *, results_override: Path | None = None) -> dict[str, Any]:
    report = _json_object(report_path.resolve(), "portfolio evidence")
    if report.get("schema_version") != "portfolio_evidence_v2":
        raise PortfolioVerificationError("unsupported portfolio evidence schema")
    if report.get("study") != "empirical_historical_weather_comparison":
        raise PortfolioVerificationError("portfolio study identity is invalid")
    inputs = report.get("inputs")
    if not isinstance(inputs, dict) or not isinstance(inputs.get("replication_results"), dict):
        raise PortfolioVerificationError("empirical result evidence is missing")
    replication_record = inputs["replication_results"]
    replication_status = _verify_replication(replication_record, results_override)
    replication_result_sha256 = (
        replication_record.get("result_file_sha256")
        if isinstance(replication_record, dict) and replication_record.get("status") == "included"
        else None
    )

    code = report.get("code")
    output = report.get("output")
    if not isinstance(code, dict) or not isinstance(output, dict):
        raise PortfolioVerificationError("code or output evidence is missing")
    for name in ("builder", "verifier", "style"):
        _verify_file(code.get(name), name)
    index_path = _verify_file(output.get("index"), "index")
    stylesheet_path = _verify_file(output.get("stylesheet"), "stylesheet")
    markdown_path = _verify_file(output.get("report"), "report")
    document = index_path.read_text()
    if '<link rel="stylesheet" href="assets/site.css">' not in document:
        raise PortfolioVerificationError("index does not use the relative stylesheet")
    if 'href="portfolio-evidence.json"' not in document:
        raise PortfolioVerificationError("index does not expose the relative evidence manifest")
    if re.search(r'(?:href|src)=["\'](?:/|https?://|file:)', document):
        raise PortfolioVerificationError("index contains a nonportable asset or page URL")
    if stylesheet_path.name != "site.css" or stylesheet_path.parent.name != "assets":
        raise PortfolioVerificationError("stylesheet path does not match the page link")
    if markdown_path != index_path.parent / "report.md":
        raise PortfolioVerificationError("report path does not match the rendered page")
    figure_count = _verify_figures(output, replication_record, index_path)
    return {
        "ok": True,
        "figures_verified": figure_count,
        "integrity_level": (
            "source_results_and_portfolio"
            if replication_status == "verified"
            else "portable_portfolio"
        ),
        "replication_result": replication_status,
        "replication_result_sha256": replication_result_sha256,
        "index_sha256": _digest(index_path),
        "report_sha256": _digest(markdown_path),
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--results", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    result = verify(args.report, results_override=args.results)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (PortfolioVerificationError, ValueError, OSError) as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, indent=2))
        raise SystemExit(1) from None
