#!/usr/bin/env python3
# ruff: noqa: E501, RUF001
"""Build a repository-prefix-safe static portfolio page from saved aggregate results."""

from __future__ import annotations

import argparse
import base64
import hashlib
import html
import json
import math
import os
import sys
from io import BytesIO
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPLICATION_RESULTS = ROOT / "outputs/historical-replication/analysis/historical-results.json"
DEFAULT_REPLICATION_PANEL = ROOT / "outputs/historical-replication/panel.json"
DEFAULT_REPLICATION_QUALITY = ROOT / "outputs/historical-replication/data-quality.json"
DEFAULT_OUTPUT = ROOT / "reports"
DEFAULT_REPORT = ROOT / "reports/portfolio-evidence.json"
STYLE = ROOT / "config/styles/custom_qb_dark_darien.mplstyle"
METHOD_LABELS = {
    "public_only": "public only",
    "market_only": "market only",
    "equal_blend": "equal blend",
    "trained_blend": "trained blend",
    "p0_reference": "pooled residual baseline",
}


class PortfolioBuildError(ValueError):
    """Saved evidence cannot support a truthful static portfolio build."""


def _digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _json_object(path: Path, name: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise PortfolioBuildError(f"cannot read {name}: {exc}") from None
    if not isinstance(value, dict):
        raise PortfolioBuildError(f"{name} must be a JSON object")
    return value


def _project_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return str(resolved)


def _atomic_write(path: Path, body: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(prefix=f".{path.name}.", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(body)
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.chmod(temporary, 0o644)
        os.replace(temporary, path)
        if path.read_bytes() != body:
            raise OSError(f"could not verify published file: {path}")
    finally:
        temporary.unlink(missing_ok=True)


def _holdout_rows(results: dict[str, Any], methods: tuple[str, ...]) -> list[dict[str, Any]]:
    rows = []
    for summary in results["summaries"]:
        if summary["split"] != "holdout":
            continue
        scores = summary["scores"]
        if any(method not in scores for method in methods):
            raise PortfolioBuildError("holdout summary is missing a required method")
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
                        "brier": scores[method]["brier"],
                        "log_loss": scores[method]["log_loss"],
                    }
                    for method in methods
                },
                "trained_minus_public_brier": comparison,
            }
        )
    if not rows:
        raise PortfolioBuildError("saved results contain no holdout summary")
    return sorted(rows, key=lambda item: item["horizon_hours"], reverse=True)


def _semantic_hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        raise PortfolioBuildError(f"replication {name} must be finite")
    return float(value)


def _count(value: object, name: str) -> int:
    if type(value) is not int or value < 0:
        raise PortfolioBuildError(f"replication {name} must be a nonnegative integer")
    return value


def _comparison(item: object, name: str) -> dict[str, Any]:
    if not isinstance(item, dict):
        raise PortfolioBuildError(f"replication {name} comparison is invalid")
    reason = item.get("interval_unavailable_reason")
    intervals = {}
    for key in ("ci95", "ci975"):
        interval = item.get(key)
        if interval is None:
            if not isinstance(reason, str) or not reason:
                raise PortfolioBuildError(f"replication {name} missing interval reason")
            intervals[key] = None
        elif (
            not isinstance(interval, list)
            or len(interval) != 2
            or _finite(interval[0], f"{name} {key}") > _finite(interval[1], f"{name} {key}")
        ):
            raise PortfolioBuildError(f"replication {name} {key} is invalid")
        else:
            intervals[key] = [float(interval[0]), float(interval[1])]
    return {
        "horizon_hours": item.get("horizon_hours"),
        "method": item.get("method"),
        "reference": item.get("reference"),
        "metric": item.get("metric"),
        "mean_difference": _finite(item.get("mean_difference"), f"{name} mean difference"),
        **intervals,
        "valid_replicates": _count(item.get("valid_replicates"), f"{name} valid replicates"),
        "undefined_replicates": _count(
            item.get("undefined_replicates"), f"{name} undefined replicates"
        ),
        "interval_unavailable_reason": reason,
    }


def _bootstrap_extract(value: dict[str, Any], protocol: dict[str, Any]) -> dict[str, Any]:
    bootstrap = value.get("bootstrap")
    if not isinstance(bootstrap, dict):
        raise PortfolioBuildError("replication bootstrap must be an object")
    settings = protocol.get("bootstrap")
    if not isinstance(settings, dict) or settings.get("attempts") != 5000:
        raise PortfolioBuildError("replication protocol must retain 5000 bootstrap attempts")
    levels = settings.get("interval_levels")
    if levels != [.95, .975] or settings.get("repeat_entire_fit") is not True:
        raise PortfolioBuildError("replication interval and refit protocol is invalid")

    def extract(name: str, block: object, expected_mode: str) -> dict[str, Any]:
        if not isinstance(block, dict) or block.get("mode") != expected_mode:
            raise PortfolioBuildError(f"replication {name} bootstrap mode is invalid")
        attempted = _count(block.get("attempted"), f"{name} attempted")
        valid = _count(block.get("valid"), f"{name} valid")
        failed = _count(block.get("failed"), f"{name} failed")
        if attempted != 5000 or valid + failed != attempted:
            raise PortfolioBuildError(f"replication {name} bootstrap counts are inconsistent")
        comparisons = block.get("comparisons")
        conditions = block.get("conditions")
        if not isinstance(comparisons, list) or not isinstance(conditions, list):
            raise PortfolioBuildError(f"replication {name} inference records are invalid")
        primary = [
            _comparison(item, name)
            for item in comparisons
            if isinstance(item, dict)
            and item.get("method") == "trained_blend"
            and item.get("reference") == "public_only"
            and item.get("metric") == "brier"
        ]
        if len(primary) != 2:
            raise PortfolioBuildError(f"replication {name} requires two primary contrasts")
        condition_records: list[dict[str, Any]] = []
        for condition in conditions:
            if not isinstance(condition, dict) or not isinstance(condition.get("support"), dict):
                raise PortfolioBuildError(f"replication {name} condition record is invalid")
            interactions = condition.get("interactions")
            if not isinstance(interactions, list) or len(interactions) != 1:
                raise PortfolioBuildError(f"replication {name} condition interaction is invalid")
            horizon = _count(condition.get("horizon_hours"), f"{name} condition horizon")
            interaction = _comparison(interactions[0], f"{name} condition")
            interaction["horizon_hours"] = horizon
            condition_records.append(
                {
                    "horizon_hours": horizon,
                    "supported": condition["support"].get("supported") is True,
                    "support": condition["support"],
                    "interaction": interaction,
                }
            )
        return {
            "mode": block["mode"],
            "block_days": block.get("block_days"),
            "attempted": attempted,
            "valid": valid,
            "failed": failed,
            "selection_counts": block.get("selection_counts", {}),
            "primary_comparisons": sorted(
                primary, key=lambda item: item["horizon_hours"], reverse=True
            ),
            "conditions": sorted(
                condition_records, key=lambda item: item["horizon_hours"], reverse=True
            ),
        }

    sensitivity = bootstrap.get("sensitivity_by_block_days")
    expected_sensitivity = {str(day) for day in settings.get("sensitivity_block_days", [])}
    if not isinstance(sensitivity, dict) or set(sensitivity) != expected_sensitivity:
        raise PortfolioBuildError("replication bootstrap block sensitivities are incomplete")
    return {
        "full_refit": extract(
            "primary",
            bootstrap.get("primary"),
            "full_refit_including_selection_oof_scale_blend_conditions",
        ),
        "sensitivity_by_block_days": {
            day: extract(
                f"{day}-day sensitivity",
                sensitivity[day],
                "full_refit_including_selection_oof_scale_blend_conditions",
            )
            for day in sorted(expected_sensitivity, key=int)
        },
        "fixed_fit": extract(
            "fixed_fit", bootstrap.get("fixed_fit"), "fixed_fit_holdout_resampling_only"
        ),
        "family_coverage": bootstrap.get("family_coverage"),
        "limitations": bootstrap.get("limitations"),
    }


def _sensitivity_extract(value: dict[str, Any]) -> dict[str, Any]:
    source = value.get("sensitivities")
    if not isinstance(source, dict):
        raise PortfolioBuildError("replication point sensitivities are missing")
    result = {}
    for name in ("label_substitution", "normalization_projection"):
        item = source.get(name)
        if not isinstance(item, dict) or item.get("status") not in {"evaluated", "unavailable"}:
            raise PortfolioBuildError(f"replication {name} sensitivity is invalid")
        record: dict[str, Any] = {"status": item["status"]}
        if item["status"] == "unavailable":
            if not isinstance(item.get("reason"), str) or not item["reason"]:
                raise PortfolioBuildError(f"replication {name} sensitivity reason is missing")
            record["reason"] = item["reason"]
        else:
            deltas = item.get("holdout_deltas")
            if not isinstance(deltas, list):
                raise PortfolioBuildError(f"replication {name} sensitivity deltas are missing")
            record["trained_blend_brier_deltas"] = sorted(
                [
                    {
                        "horizon_hours": _count(row.get("horizon_hours"), f"{name} horizon"),
                        "difference_from_primary": _finite(
                            row.get("brier_difference_from_primary"), f"{name} delta"
                        ),
                    }
                    for row in deltas
                    if isinstance(row, dict)
                    and row.get("method") == "trained_blend"
                    and "brier_difference_from_primary" in row
                ],
                key=lambda row: row["horizon_hours"],
                reverse=True,
            )
            if len(record["trained_blend_brier_deltas"]) != 2:
                raise PortfolioBuildError(f"replication {name} sensitivity is incomplete")
        result[name] = record
    return result


def _v2_aggregate(
    path: Path,
    *,
    panel_path: Path | None = None,
    data_quality_path: Path | None = None,
) -> dict[str, Any]:
    value = _json_object(path, "replication results")
    expected = {
        "schema_version": "historical_replication_evaluation_v1",
        "data_origin": "empirical_historical",
        "analysis_status": "exploratory_historical_replication",
        "complete_requested_dates": True,
    }
    for key, wanted in expected.items():
        if value.get(key) != wanted:
            raise PortfolioBuildError(f"replication {key} must equal {wanted}")
    for key in ("protocol", "fit", "cohort", "provenance"):
        if not isinstance(value.get(key), dict):
            raise PortfolioBuildError(f"replication {key} must be an object")
    for key in ("input_panel_sha256", "input_protocol_sha256", "protocol_sha256"):
        digest = value.get(key)
        if not isinstance(digest, str) or len(digest) != 64:
            raise PortfolioBuildError(f"replication {key} is invalid")
    if value["input_protocol_sha256"] != _semantic_hash(value["protocol"]) or value[
        "protocol_sha256"
    ] != _semantic_hash(value["protocol"]):
        raise PortfolioBuildError("replication protocol semantic hash conflict")
    cohort = value["cohort"]
    for key in ("cases", "events", "dates"):
        _count(cohort.get(key), f"cohort {key}")
    coverage = value.get("coverage")
    target = value.get("target_continuity")
    if not isinstance(coverage, dict) or not isinstance(target, dict):
        raise PortfolioBuildError("replication coverage and target continuity are required")
    if target.get("analysis_ready") is not True:
        raise PortfolioBuildError("replication target continuity is not ready")
    for key in ("requested_dates", "potential_cases", "eligible_cases", "excluded_cases"):
        _count(coverage.get(key), f"coverage {key}")
    if coverage["eligible_cases"] != cohort["cases"] or (
        coverage["eligible_cases"] + coverage["excluded_cases"] != coverage["potential_cases"]
    ):
        raise PortfolioBuildError("replication coverage counts are inconsistent")
    if panel_path is not None:
        panel = _json_object(panel_path, "replication panel")
        if _semantic_hash(panel) != value["input_panel_sha256"]:
            raise PortfolioBuildError("replication panel semantic hash conflict")
    if data_quality_path is None:
        raise PortfolioBuildError("replication data quality record is required")
    quality = _json_object(data_quality_path, "replication data quality")
    if quality.get("schema_version") != "replication_data_quality_v1":
        raise PortfolioBuildError("replication data quality schema is invalid")
    monthly = quality.get("monthly_coverage")
    if not isinstance(monthly, list) or not monthly:
        raise PortfolioBuildError("replication monthly coverage is missing")
    quality_counts = {
        key: _count(quality.get(key), f"data quality {key}")
        for key in ("requested_dates", "eligible_dates", "excluded_dates", "eligible_cases")
    }
    if (
        quality_counts["requested_dates"] != coverage["requested_dates"]
        or quality_counts["eligible_dates"] != cohort["dates"]
        or quality_counts["eligible_cases"] != cohort["cases"]
        or quality_counts["eligible_dates"] + quality_counts["excluded_dates"]
        != quality_counts["requested_dates"]
        or quality.get("target_rules_compatible_dates") != quality_counts["requested_dates"]
    ):
        raise PortfolioBuildError("replication data quality conflicts with evaluated cohort")
    if panel_path is not None and quality.get("panel_file_sha256") != _digest(panel_path):
        raise PortfolioBuildError("replication data quality panel file hash conflict")
    observed_monthly = []
    for row in monthly:
        if not isinstance(row, dict) or not isinstance(row.get("month"), str):
            raise PortfolioBuildError("replication monthly coverage row is invalid")
        requested = _count(row.get("requested"), "monthly requested")
        usable = _count(row.get("case"), "monthly usable")
        excluded = _count(row.get("excluded", 0), "monthly excluded")
        if usable + excluded != requested:
            raise PortfolioBuildError("replication monthly coverage counts are inconsistent")
        observed_monthly.append(
            {"month": row["month"], "requested": requested, "usable": usable, "excluded": excluded}
        )
    if (
        sum(row["requested"] for row in observed_monthly) != quality_counts["requested_dates"]
        or sum(row["usable"] for row in observed_monthly) != quality_counts["eligible_dates"]
    ):
        raise PortfolioBuildError("replication monthly totals are inconsistent")
    bootstrap = _bootstrap_extract(value, value["protocol"])
    candidate_rows = value.get("candidate_validation")
    if not isinstance(candidate_rows, list) or len(candidate_rows) != len(
        value["protocol"].get("models", {}).get("candidates", [])
    ):
        raise PortfolioBuildError("replication candidate validation is incomplete")
    oof = value["fit"].get("oof")
    if not isinstance(oof, dict) or not isinstance(oof.get("predictions"), list):
        raise PortfolioBuildError("replication OOF record is invalid")
    blend_case_ids = value["fit"].get("blend_training_case_ids")
    if not isinstance(blend_case_ids, list) or not all(
        isinstance(case_id, str) for case_id in blend_case_ids
    ):
        raise PortfolioBuildError("replication blend training IDs are invalid")
    blend_id_set = set(blend_case_ids)
    blend_predictions = [
        prediction
        for prediction in oof["predictions"]
        if isinstance(prediction, dict) and prediction.get("case_id") in blend_id_set
    ]
    blend_dates = sorted(
        {
            prediction["outcome_date"]
            for prediction in blend_predictions
            if isinstance(prediction.get("outcome_date"), str)
        }
    )
    if len(blend_predictions) != len(blend_id_set) or not blend_dates:
        raise PortfolioBuildError("replication blend prediction coverage is incomplete")
    aggregate = {
        "status": "included",
        "result_path": _project_path(path),
        "schema_version": value["schema_version"],
        "analysis_status": value["analysis_status"],
        "cohort": cohort,
        "study_window": {
            key: value["protocol"]["collection"].get(key)
            for key in ("start_date", "train_end_date", "validation_end_date", "end_date")
        },
        "coverage": coverage,
        "fit": {
            "selected_candidate": value["fit"].get("selected_candidate"),
            "trained_market_weight": _finite(
                value["fit"].get("trained_market_weight"), "trained market weight"
            ),
            "validation_fit_at": value["fit"].get("validation_fit_at"),
            "final_fit_at": value["fit"].get("final_fit_at"),
            "oof": {
                "spec": oof.get("spec"),
                "error_fold_start_month": oof.get("start_month"),
                "error_fold_end_month": oof.get("end_month"),
                "fit_count": len(oof.get("fits", [])),
                "prediction_count": len(oof["predictions"]),
                "blend_prediction_start_date": blend_dates[0],
                "blend_prediction_end_date": blend_dates[-1],
                "blend_prediction_dates": len(blend_dates),
                "blend_prediction_cases": len(blend_predictions),
            },
        },
        "candidate_validation": [
            {
                "candidate": row.get("candidate"),
                "mean_brier": _finite(row.get("mean_brier"), "candidate mean brier"),
                "cases": _count(row.get("cases"), "candidate cases"),
                "dates": _count(row.get("dates"), "candidate dates"),
            }
            for row in candidate_rows
            if isinstance(row, dict)
        ],
        "holdout": _holdout_rows(
            value,
            ("public_only", "market_only", "equal_blend", "trained_blend", "p0_reference"),
        ),
        "bootstrap": bootstrap,
        "sensitivities": _sensitivity_extract(value),
        "data_quality": {
            **quality_counts,
            "target_rules_compatible_dates": quality["target_rules_compatible_dates"],
            "monthly_coverage": observed_monthly,
            "limitations": quality.get("limitations", []),
            "panel_file_sha256": quality.get("panel_file_sha256"),
            "record_path": _project_path(data_quality_path),
            "record_sha256": _digest(data_quality_path),
        },
        "panel_path": _project_path(panel_path) if panel_path is not None else None,
        "input_panel_sha256": value.get("input_panel_sha256"),
        "input_protocol_sha256": value.get("input_protocol_sha256"),
        "result_file_sha256": _digest(path),
    }
    return aggregate


def _esc(value: object) -> str:
    """Render saved free text literally in inline Markdown and table cells."""
    text = " ".join(str(value).splitlines())
    text = html.escape(text)
    for character in "\\`*_[]()|$":
        text = text.replace(character, "\\" + character)
    return text


def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def _score_table_md(aggregate: dict[str, Any], methods: tuple[str, ...]) -> str:
    headers = ["checkpoint", *(METHOD_LABELS[m] for m in methods), "dates", "cases"]
    rows = []
    for item in aggregate["holdout"]:
        row = [f"{item['horizon_hours']}h"]
        row.extend(f"{item['scores'][m]['brier']:.4f}" for m in methods)
        row.extend((str(item["dates"]), str(item["cases"])))
        rows.append(row)
    return (
        "Mean multiclass Brier loss on the same held-out cases; lower is better.\n\n"
        + _md_table(headers, rows)
    )


def _interval_md(value: object) -> str:
    if not isinstance(value, list):
        return "unavailable"
    return f"[{value[0]:.4f}, {value[1]:.4f}]"


def _reason_label(reason: object) -> str:
    labels = {
        "insufficient_original_condition_group_support": (
            "too few dates in one disagreement group"
        ),
    }
    return labels.get(str(reason), str(reason).replace("_", " "))


def _replication_inference_md(replication: dict[str, Any]) -> str:
    rows = []
    blocks = [("full refit, 7-day blocks", replication["bootstrap"]["full_refit"])]
    blocks.extend(
        (
            f"full refit, {block_days}-day blocks",
            replication["bootstrap"]["sensitivity_by_block_days"][block_days],
        )
        for block_days in ("3", "14")
    )
    blocks.append(("fixed fit, 7-day blocks", replication["bootstrap"]["fixed_fit"]))
    for label, block in blocks:
        for item in block["primary_comparisons"]:
            reason = (
                _reason_label(item["interval_unavailable_reason"])
                if item["interval_unavailable_reason"]
                else "available"
            )
            rows.append(
                [
                    f"{item['horizon_hours']}h",
                    label,
                    f"{item['mean_difference']:.4f}",
                    _interval_md(item["ci95"]),
                    _interval_md(item["ci975"]),
                    f"{item['valid_replicates']} / {block['attempted']}",
                    _esc(reason),
                ]
            )
    headers = [
        "checkpoint", "fit treatment", "difference", "95% interval",
        "97.5% interval", "valid draws", "interval status",
    ]
    return (
        "Fitted blend minus public forecast Brier loss; negative values favor the blend. "
        "The nominal 97.5% intervals use a Bonferroni adjustment for the two horizon "
        "comparisons.\n\n"
        + _md_table(headers, rows)
    )


def _sensitivity_table_md(replication: dict[str, Any]) -> str:
    labels = {
        "label_substitution": "development-label substitution",
        "normalization_projection": "simplex normalization",
    }
    rows = []
    for name, label in labels.items():
        record = replication["sensitivities"][name]
        if record["status"] == "unavailable":
            rows.append([label, "unavailable", _esc(record["reason"])])
            continue
        for item in record["trained_blend_brier_deltas"]:
            rows.append([label, f"{item['horizon_hours']}h", f"{item['difference_from_primary']:.4f}"])
    headers = ["sensitivity", "checkpoint", "difference from primary"]
    return (
        "These point sensitivities have no separate bootstrap intervals.\n\n"
        + _md_table(headers, rows)
    )


def _monthly_coverage_md(replication: dict[str, Any]) -> str:
    rows = [
        [_esc(item["month"]), str(item["requested"]), str(item["usable"]), str(item["excluded"])]
        for item in replication["data_quality"]["monthly_coverage"]
    ]
    headers = ["month", "requested", "usable", "excluded"]
    return (
        "Excluded dates lacked usable numeric settlement labels.\n\n"
        + _md_table(headers, rows)
    )


def _condition_summary_md(replication: dict[str, Any]) -> str:
    parts = []
    for condition in replication["bootstrap"]["full_refit"]["conditions"]:
        interaction = condition["interaction"]
        groups = condition["support"]["groups"]
        counts = (
            f"high disagreement: {groups['high']['dates']} dates in "
            f"{groups['high']['calendar_blocks']} calendar blocks. low disagreement: "
            f"{groups['low']['dates']} dates in {groups['low']['calendar_blocks']} calendar blocks."
        )
        if (
            condition["supported"]
            and isinstance(interaction["ci95"], list)
            and isinstance(interaction["ci975"], list)
        ):
            ci95 = interaction["ci95"]
            ci975 = interaction["ci975"]
            status = (
                "The saved support thresholds were met. The direct high-minus-low "
                f"interaction is {interaction['mean_difference']:.4f}, 95% interval "
                f"{_interval_md(ci95)} ({'includes' if ci95[0] <= 0 <= ci95[1] else 'excludes'} "
                f"zero), 97.5% interval {_interval_md(ci975)} "
                f"({'includes' if ci975[0] <= 0 <= ci975[1] else 'excludes'} zero)."
            )
        else:
            status = "Interval unavailable: " + _esc(
                _reason_label(
                    interaction["interval_unavailable_reason"]
                    or "insufficient_original_condition_group_support"
                )
            )
        parts.append(f"**{condition['horizon_hours']}h disagreement subgroup.** {counts} {status}")
    parts.append(
        "These subgroup interactions are exploratory and do not establish a reliable "
        "pre-outcome selection rule."
    )
    return "\n\n".join(parts)


def _figure_md(name: str, caption: str) -> str:
    return f"![{caption}](assets/{name}.png)"


def _blend_description(weight: float) -> str:
    if weight == 1:
        return "trained blend equals market only; no mixing benefit shown"
    if weight == 0:
        return "trained blend equals public only; no mixing benefit shown"
    return "trained blend combines public and market forecasts"


def _model_label(model_id: object) -> str:
    labels = {
        "p0": "pooled residual baseline",
        "p0_reference": "pooled residual baseline",
        "p1_lambda_0.1": "seasonal ridge residual model, λ = 0.1",
        "p1_lambda_1": "seasonal ridge residual model, λ = 1",
        "p1_lambda_10": "seasonal ridge residual model, λ = 10",
    }
    return labels.get(str(model_id), str(model_id))


def _market_vs_public_clause(replication: dict[str, Any]) -> str:
    rows = replication["holdout"]
    lower = [
        row["horizon_hours"]
        for row in rows
        if row["scores"]["market_only"]["brier"] < row["scores"]["public_only"]["brier"]
    ]
    if rows and len(lower) == len(rows):
        return "the market had lower average Brier loss than the public forecast at each evaluated horizon"
    if not lower:
        return "the market did not have lower average Brier loss than the public forecast at any evaluated horizon"
    checkpoints = " and ".join(f"{h}h" for h in lower)
    return f"the market had lower average Brier loss than the public forecast at the {checkpoints} horizon"


def _weight_clause(weight: float) -> str:
    return f"the fitted blend weight was {weight:.4f} ({_blend_description(weight)})"


def _when_rule_clause(replication: dict[str, Any]) -> str:
    reliable = []
    for item in replication["bootstrap"]["full_refit"]["conditions"]:
        interaction = item["interaction"]
        ci975 = interaction.get("ci975")
        if item["supported"] and isinstance(ci975, list) and not (ci975[0] <= 0 <= ci975[1]):
            reliable.append(item["horizon_hours"])
    if reliable:
        checkpoints = " and ".join(f"{h}h" for h in reliable)
        return (
            f"the disagreement condition reliably widened this gap at {checkpoints} "
            "(97.5% interval excludes zero)"
        )
    return (
        "the disagreement analysis did not identify in advance when the market's "
        "advantage would be larger (the 97.5% interval included zero or group support "
        "was too small)"
    )


def _finding_summary(replication: dict[str, Any]) -> str:
    weight = replication["fit"]["trained_market_weight"]
    clauses = (
        _market_vs_public_clause(replication),
        _weight_clause(weight),
        _when_rule_clause(replication),
    )
    return (
        f"In this retained sample, {clauses[0]}. "
        f"{clauses[1].capitalize()}. "
        f"{clauses[2].capitalize()}."
    )


EQUATIONS_MD = r"""The multiclass Brier score for one case sums squared errors across settlement bins,
with no division by the bin count:

$$BS_i = \sum_{k=1}^{K} \left(p_{i,k} - \mathbb{1}[k = k_i^{*}]\right)^2$$

Here $p_{i,k}$ is the forecast probability assigned to bin $k$ for case $i$, $k_i^{*}$ is the
realized settlement bin, and $K$ is the number of bins in that contract. Reported scores average
$BS_i$ over the $N$ cases in a split:

$$\overline{BS} = \frac{1}{N}\sum_{i=1}^{N} BS_i$$

Log loss floors the winning-bin probability at $\epsilon = 10^{-6}$ and does not renormalize the
probability vector:

$$LL_i = -\log\left(\max(p_{i,k_i^{*}},\ \epsilon)\right), \qquad \overline{LL} = \frac{1}{N}\sum_{i=1}^{N} LL_i$$

The fitted blend is a fixed convex combination of the market and public probability vectors,
with the weight fit out-of-fold on pre-holdout dates and then held fixed for scoring:

$$p_{\text{blend}} = w \, p_{\text{market}} + (1 - w) \, p_{\text{public}}, \qquad w \in [0, 1]$$"""


def _report_markdown(replication: dict[str, Any]) -> str:
    methods = ("public_only", "market_only", "equal_blend", "trained_blend")
    window = replication["study_window"]
    quality = replication["data_quality"]
    oof = replication["fit"]["oof"]
    weight = replication["fit"]["trained_market_weight"]
    selected_model = _model_label(replication["fit"]["selected_candidate"])
    lines = [
        "# Kalshi information quality lab",
        "",
        "Do prediction-market prices add information beyond public weather forecasts? "
        "This report compares archived Kalshi New York daily-high markets with archived "
        "NOAA NDFD forecasts at 12 and 6 hours before the settlement day begins.",
        "",
        "## Data and methods",
        "",
        "The outcome is the Central Park settlement-day high, divided into market bins. "
        "Market probabilities use normalized hourly bid/ask candle closes, which are "
        "historical proxies rather than executable prices. The public predictor is the "
        "NOAA NDFD MaxT grid forecast, calibrated to settlement bins using earlier labels.",
        "",
        EQUATIONS_MD,
        "",
        f"The study covers {_esc(window['start_date'])} through {_esc(window['end_date'])}. "
        f"Of {quality['requested_dates']} requested dates, {quality['eligible_dates']} have "
        f"usable numeric settlement labels and enter the cohort; {quality['excluded_dates']} "
        f"are excluded. The retained cohort has {replication['cohort']['dates']} dates and "
        f"{replication['cohort']['cases']} scored checkpoint cases across the two horizons.",
        "",
        "## Main finding",
        "",
        _finding_summary(replication),
        "",
        "## Holdout scores",
        "",
        _score_table_md(replication, (*methods, "p0_reference")),
        "",
        _figure_md(
            "holdout-losses",
            "average multiclass brier loss by method and horizon on the same holdout dates; "
            "lower is better.",
        ),
        "",
        f"Model selection chose the {_esc(selected_model)}. Out-of-fold error folds run from "
        f"{_esc(oof['error_fold_start_month'])} through {_esc(oof['error_fold_end_month'])}. "
        f"The blend weight was fit on {oof['blend_prediction_cases']} eligible checkpoint "
        f"predictions across {oof['blend_prediction_dates']} dates, "
        f"{_esc(oof['blend_prediction_start_date'])} through "
        f"{_esc(oof['blend_prediction_end_date'])}. Its market weight is {weight:.4f}: "
        f"{_blend_description(weight)}.",
        "",
        "## Uncertainty",
        "",
        _replication_inference_md(replication),
        "",
        _figure_md(
            "paired-differences",
            "trained blend minus public-only brier loss, with 95% and nominal 97.5% "
            "full-refit intervals using seven-day blocks; negative values favor the blend.",
        ),
        "",
        "Each full-refit draw repeats model selection, calibration, blend fitting, and "
        "condition assignment. Fixed-fit draws hold the fitted pipeline constant. Blocks stay "
        "within calendar months; this underweights dates near month edges, especially with "
        "14-day blocks.",
        "",
        "## Sensitivities",
        "",
        _sensitivity_table_md(replication),
        "",
        "## Can disagreement identify the market's advantage?",
        "",
        _condition_summary_md(replication),
        "",
        "## Data coverage",
        "",
        "Missing numeric settlement labels are concentrated in autumn and winter. June 23, "
        "2026 is also absent from the holdout for this reason. This selection may bias seasonal "
        "and holdout comparisons.",
        "",
        _monthly_coverage_md(replication),
        "",
        _figure_md(
            "monthly-coverage",
            "usable and excluded dates by month for the historical study.",
        ),
        "",
        "## Limitations",
        "",
        "NDFD MaxT covers 12:00–00:00 UTC, while the Central Park settlement day runs "
        "from 05:00 UTC to 05:00 UTC. Calibration does not recover the missing overnight "
        "information.",
        "",
        "The bootstrap repeats model selection and fitting but cannot recover missing labels or "
        "unknown archive revisions. Month-stratified, nonwrapping blocks also underweight dates "
        "near month boundaries.",
        "",
        "Hourly bid/ask candle closes do not establish fills, depth, fees, or profitability.",
        "",
        "## Reproducibility",
        "",
        "This page is built from the retained empirical result. The [evidence manifest]"
        "(portfolio-evidence.json) hashes the source result, scientific inputs, report, page, "
        "and exact aggregate data behind each figure.",
        "",
        "The [replication notebook](../notebooks/06_historical_replication.ipynb), "
        "[historical study notebook](../notebooks/05_historical_study.ipynb), "
        "[NDFD extraction notebook](../notebooks/04_noaa_grid_validation.ipynb), source code, "
        "and replay scripts are tracked here. Raw inputs and fitted runs stay outside Git; see "
        "[sources and data boundaries](../docs/SOURCES.md).",
        "",
    ]
    return "\n".join(lines)


def _html_document(body_markdown: str) -> str:
    from matplotlib.mathtext import math_to_image
    from nbconvert.filters.markdown_mistune import IPythonRenderer, MarkdownWithMath

    class OfflineMathRenderer(IPythonRenderer):
        def _math_image(self, body: str, css_class: str) -> str:
            image = BytesIO()
            math_to_image(f"${body}$", image, dpi=144, format="png", color="#1a1a1a")
            source = base64.b64encode(image.getvalue()).decode("ascii")
            alt = html.escape(body, quote=True)
            return (
                f'<img class="{css_class}" src="data:image/png;base64,{source}" '
                f'alt="{alt}">'
            )

        def block_math(self, body: str) -> str:
            return f'<div class="math-display">{self._math_image(body, "math-block")}</div>'

        def inline_math(self, body: str) -> str:
            return self._math_image(body, "math-inline")

    renderer = OfflineMathRenderer(
        escape=False,
        allow_harmful_protocols=False,
        exclude_anchor_links=True,
    )
    body = MarkdownWithMath(renderer=renderer).render(body_markdown)
    body = body.replace("<table>", '<div class="table-wrap"><table>').replace(
        "</table>", "</table></div>"
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>kalshi information quality lab</title>
<link rel="icon" href="data:,">
<link rel="stylesheet" href="assets/site.css"></head>
<body>
{body}
<footer>local research artifact. no orders, recommendations, or deployment.</footer>
</body></html>"""


CSS = """body{font-family:-apple-system,system-ui,"Segoe UI",Helvetica,Arial,sans-serif;color:#1a1a1a;background:#fff;max-width:800px;margin:2.5rem auto;padding:0 1.25rem 4rem;line-height:1.55}
h1,h2,h3{font-weight:600;line-height:1.3}
h1{font-size:1.8rem;margin-bottom:.3em}
h2{font-size:1.3rem;margin-top:2.2em;border-top:1px solid #ddd;padding-top:1em}
h3{font-size:1.05rem}
p,ul,ol{margin:1em 0}
a{color:#0645ad}
a:visited{color:#0b0080}
code{background:#f2f2f2;padding:.1em .35em;border-radius:3px;font-size:.92em}
.table-wrap{overflow-x:auto;margin:1em 0}
table{border-collapse:collapse;width:100%}
th,td{border:1px solid #ccc;padding:6px 10px;text-align:right;font-size:.95rem}
th:first-child,td:first-child{text-align:left}
thead th{background:#f5f5f5}
img{max-width:100%;height:auto;border:1px solid #ccc;display:block}
.math-inline{display:inline;border:0;height:1.15em;width:auto;vertical-align:-.25em}
.math-display{overflow-x:auto;margin:1em 0;text-align:center}
.math-display img{display:inline-block;border:0;max-width:100%;height:auto}
em{color:#444;font-size:.95rem}
footer{margin-top:3em;padding-top:1em;border-top:1px solid #ddd;color:#555;font-size:.9rem}
:focus-visible{outline:2px solid #0645ad;outline-offset:2px}
@media(max-width:600px){body{margin:1rem auto;padding:0 1rem 3rem}}
"""


def _figure_data(replication: dict[str, Any]) -> dict[str, Any]:
    if replication["status"] != "included":
        return {}
    dates = {}
    for row in replication["holdout"]:
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
    scores = []
    for row in replication["holdout"]:
        values = {}
        for method in METHOD_LABELS:
            values[method] = row["scores"][method]["brier"]
        scores.append(
            {"horizon_hours": row["horizon_hours"], "dates": row["dates"], "scores": values}
        )
    return {
        "monthly-coverage": replication["data_quality"]["monthly_coverage"],
        "holdout-losses": scores,
        "paired-differences": contrasts,
    }


def _build_figures(replication: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    data = _figure_data(replication)
    import matplotlib as mpl

    mpl.use("Agg", force=True)
    import matplotlib.pyplot as plt

    records = {}
    with plt.style.context(STYLE):
        colors = mpl.rcParams["axes.prop_cycle"].by_key()["color"]
        for name, rows in data.items():
            fig, ax = plt.subplots()
            if name == "monthly-coverage":
                months = []
                usable = []
                excluded = []
                for row in rows:
                    months.append(row["month"])
                    usable.append(row["usable"])
                    excluded.append(row["excluded"])
                ax.bar(months, usable, label="usable", color=colors[0])
                ax.bar(months, excluded, bottom=usable, label="excluded", color=colors[1])
                ax.set_title("Monthly coverage")
                ax.set_ylabel("dates")
                ax.tick_params(axis="x", labelrotation=35)
                ax.set_ylim(0, max(row["requested"] for row in rows) + 6)
                ax.legend(loc="upper right", ncols=2)
                quality = replication["data_quality"]
                note = (
                    f"{quality['eligible_dates']} usable / {quality['requested_dates']} requested dates; "
                    f"{quality['excluded_dates']} excluded for numeric labels"
                )
            elif name == "holdout-losses":
                methods = list(METHOD_LABELS)
                labels = []
                for method in methods:
                    labels.append(METHOD_LABELS[method].replace(" ", "\n"))
                for idx, row in enumerate(rows):
                    positions = []
                    losses = []
                    for m_idx, method in enumerate(methods):
                        positions.append(m_idx + (idx - .5) * .36)
                        losses.append(row["scores"][method])
                    bars = ax.bar(
                        positions, losses, width=.34, color=colors[idx],
                        label=f"{row['horizon_hours']}h · {row['dates']} dates",
                    )
                    ax.bar_label(bars, fmt="%.3f", padding=3)
                ax.set_xticks(range(len(methods)), labels)
                ax.set_ylim(0, max(ax.get_ylim()[1], 1))
                ax.set_title("Holdout losses, April–June 2026")
                ax.set_ylabel("Mean multiclass Brier loss")
                ax.legend(loc="upper right")
                note = (
                    "same dates for every method; lower is better\n"
                    + _blend_description(replication["fit"]["trained_market_weight"])
                )
            else:
                fig.set_figheight(fig.get_figheight() * .75)
                labels = []
                seen = set()
                for idx, row in enumerate(rows):
                    labels.append(f"{row['horizon_hours']}h · {row['dates']} dates")
                    for key, color, width, label in (
                        ("ci975", colors[1], 2, "97.5% interval"),
                        ("ci95", colors[0], 6, "95% interval"),
                    ):
                        interval = row[key]
                        if interval is not None:
                            ax.hlines(
                                idx, interval[0], interval[1], color=color, linewidth=width,
                                label=label if key not in seen else "_nolegend_",
                            )
                            seen.add(key)
                    ax.scatter(
                        row["mean_difference"],
                        idx,
                        color=mpl.rcParams["text.color"],
                        zorder=3,
                    )
                ax.set_yticks(range(len(rows)), labels)
                ax.set_ylim(-.7, len(rows) - .3)
                ax.invert_yaxis()
                ax.grid(False)
                ax.grid(axis="x")
                ax.axvline(0, color=mpl.rcParams["axes.edgecolor"], linestyle="--")
                ax.set_title("Paired holdout differences, seven-day full refit")
                ax.set_xlabel("Fitted blend − public forecast Brier loss")
                if seen:
                    ax.legend(loc="upper left", ncols=2)
                    note = "97.5% intervals address two primary comparisons; approximate coverage on retained dates"
                else:
                    note = "intervals unavailable; points show paired mean differences on retained dates"
            fig.tight_layout(rect=(0, .16, 1, 1), pad=1.5)
            fig.text(
                .02,
                .03,
                note,
                ha="left",
                va="bottom",
                fontsize=9,
                color=mpl.rcParams["axes.labelcolor"],
            )
            buffer = BytesIO()
            fig.savefig(
                buffer,
                format="png",
                bbox_inches="tight",
                pad_inches=.25,
                dpi=300,
                metadata={"Software": "kalshi information quality lab"},
            )
            path = output_dir / "assets" / f"{name}.png"
            _atomic_write(path, buffer.getvalue())
            plt.close(fig)
            records[name] = {
                "path": _project_path(path),
                "sha256": _digest(path),
                "data": rows,
                "data_sha256": _semantic_hash(rows),
                "source_result_sha256": replication["result_file_sha256"],
                "renderer": "matplotlib",
                "renderer_version": mpl.__version__,
            }
    return records


def build(
    replication_path: Path,
    output_dir: Path,
    report_path: Path,
    *,
    replication_panel_path: Path = DEFAULT_REPLICATION_PANEL,
    replication_data_quality_path: Path = DEFAULT_REPLICATION_QUALITY,
) -> dict[str, Any]:
    replication = _v2_aggregate(
        replication_path.resolve(),
        panel_path=replication_panel_path.resolve(),
        data_quality_path=replication_data_quality_path.resolve(),
    )
    output_dir = output_dir.resolve()
    figures = _build_figures(replication, output_dir)
    css_path = output_dir / "assets/site.css"
    index_path = output_dir / "index.html"
    report_path_md = output_dir / "report.md"
    report_markdown = _report_markdown(replication)
    _atomic_write(css_path, CSS.encode())
    _atomic_write(report_path_md, report_markdown.encode())
    _atomic_write(index_path, _html_document(report_markdown).encode())
    code = {
        "builder": {
            "path": _project_path(Path(__file__)),
            "sha256": _digest(Path(__file__)),
        },
        "verifier": {
            "path": "scripts/verify_portfolio.py",
            "sha256": _digest(ROOT / "scripts/verify_portfolio.py"),
        },
        "style": {
            "path": _project_path(STYLE),
            "sha256": _digest(STYLE),
        },
    }
    evidence = {
        "schema_version": "portfolio_evidence_v2",
        "study": "empirical_historical_weather_comparison",
        "inputs": {"replication_results": replication},
        "code": code,
        "output": {
            "index": {"path": _project_path(index_path), "sha256": _digest(index_path)},
            "stylesheet": {"path": _project_path(css_path), "sha256": _digest(css_path)},
            "report": {"path": _project_path(report_path_md), "sha256": _digest(report_path_md)},
            "figures": figures,
        },
    }
    _atomic_write(
        report_path.resolve(), (json.dumps(evidence, indent=2, sort_keys=True) + "\n").encode()
    )
    return evidence


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replication-results", type=Path, default=DEFAULT_REPLICATION_RESULTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--replication-panel", type=Path, default=DEFAULT_REPLICATION_PANEL)
    parser.add_argument(
        "--replication-data-quality", type=Path, default=DEFAULT_REPLICATION_QUALITY
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    evidence = build(
        args.replication_results,
        args.output_dir,
        args.report,
        replication_panel_path=args.replication_panel,
        replication_data_quality_path=args.replication_data_quality,
    )
    print(json.dumps(evidence, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (PortfolioBuildError, ValueError, OSError) as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, indent=2), file=sys.stderr)
        raise SystemExit(1) from None
