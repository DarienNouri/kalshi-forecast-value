"""Render validated saved historical results without collecting data or fitting models."""

from __future__ import annotations

import hashlib
import html
import json
import math
import os
import re
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

from .historical_evaluation import METHODS


class HistoricalResultError(ValueError):
    """A saved historical result is unsafe or structurally invalid."""


def _require_mapping(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise HistoricalResultError(f"{name} must be an object")
    return value


def _require_list(value: object, name: str, *, nonempty: bool = False) -> list[Any]:
    if not isinstance(value, list) or (nonempty and not value):
        suffix = " a nonempty array" if nonempty else " an array"
        raise HistoricalResultError(f"{name} must be{suffix}")
    return value


def _finite(value: object, name: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise HistoricalResultError(f"{name} must be finite numeric data")
    result = float(value)
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        raise HistoricalResultError(f"{name} must be finite and at least {minimum}")
    return result


def _positive_count(value: object, name: str) -> int:
    if type(value) is not int or value < 1:
        raise HistoricalResultError(f"{name} must be a positive integer")
    return value


def _count(value: object, name: str) -> int:
    if type(value) is not int or value < 0:
        raise HistoricalResultError(f"{name} must be a nonnegative integer")
    return value


def _comparison(value: object, name: str, *, interaction: bool = False) -> None:
    comparison = _require_mapping(value, name)
    if comparison.get("method") not in METHODS[1:]:
        raise HistoricalResultError(f"{name}.method is invalid")
    if comparison.get("metric") not in {"brier", "log_loss"}:
        raise HistoricalResultError(f"{name}.metric is invalid")
    difference = comparison.get("mean_difference")
    if difference is not None:
        _finite(difference, f"{name}.mean_difference")
    interval = comparison.get("ci95")
    if interval is None:
        if not isinstance(comparison.get("interval_unavailable_reason"), str):
            raise HistoricalResultError(f"{name} requires an unavailable-interval reason")
    else:
        values = _require_list(interval, f"{name}.ci95")
        if len(values) != 2:
            raise HistoricalResultError(f"{name}.ci95 must have two endpoints")
        for index, endpoint in enumerate(values):
            _finite(endpoint, f"{name}.ci95[{index}]")
    if interaction:
        _count(comparison.get("valid_replicates"), f"{name}.valid_replicates")


def _sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[a-f0-9]{64}", value) is None:
        raise HistoricalResultError(f"{name} must be a lowercase SHA-256")
    return value


def _no_nonfinite_numbers(value: object, name: str = "results") -> None:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return
    if isinstance(value, int | float):
        _finite(value, name)
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _no_nonfinite_numbers(item, f"{name}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise HistoricalResultError(f"{name} contains a non-string field")
            _no_nonfinite_numbers(item, f"{name}.{key}")
        return
    raise HistoricalResultError(f"{name} contains unsupported data")


def validate_historical_results(value: object) -> dict[str, Any]:
    """Validate the saved v1 report contract without recalculating scientific results."""
    results = _require_mapping(value, "results")
    expected = {
        "schema_version": "historical_evaluation_v1",
        "data_origin": "empirical_historical",
        "analysis_status": "exploratory_historical_reconstruction",
    }
    for key, wanted in expected.items():
        if results.get(key) != wanted:
            raise HistoricalResultError(f"{key} must equal {wanted}")
    for key in ("input_panel_sha256", "input_protocol_sha256", "protocol_sha256"):
        _sha256(results.get(key), key)

    protocol = _require_mapping(results.get("protocol"), "protocol")
    horizons = _require_list(protocol.get("horizon_hours"), "protocol.horizon_hours", nonempty=True)
    if any(type(item) is not int or item <= 0 for item in horizons):
        raise HistoricalResultError("protocol.horizon_hours must contain positive integers")
    if len(set(horizons)) != len(horizons):
        raise HistoricalResultError("protocol.horizon_hours must not contain duplicates")

    fit = _require_mapping(results.get("fit"), "fit")
    weight = _finite(fit.get("trained_market_weight"), "fit.trained_market_weight")
    if not 0 <= weight <= 1:
        raise HistoricalResultError("fit.trained_market_weight must be in [0, 1]")
    for key in ("residual_mean_f", "residual_std_f", "disagreement_l1_median"):
        _finite(fit.get(key), f"fit.{key}")
    _positive_count(fit.get("training_cases"), "fit.training_cases")

    cohort = _require_mapping(results.get("cohort"), "cohort")
    cases = _require_list(results.get("case_results"), "case_results", nonempty=True)
    if _positive_count(cohort.get("cases"), "cohort.cases") != len(cases):
        raise HistoricalResultError("cohort.cases must equal the saved case count")
    for key in ("events", "dates"):
        _positive_count(cohort.get(key), f"cohort.{key}")

    summaries = _require_list(results.get("summaries"), "summaries", nonempty=True)
    for index, summary_value in enumerate(summaries):
        summary = _require_mapping(summary_value, f"summaries[{index}]")
        if summary.get("split") not in {"train", "validation", "holdout"}:
            raise HistoricalResultError(f"summaries[{index}].split is invalid")
        if summary.get("horizon_hours") not in horizons:
            raise HistoricalResultError(f"summaries[{index}].horizon_hours is undeclared")
        for key in ("cases", "events", "dates"):
            _positive_count(summary.get(key), f"summaries[{index}].{key}")
        scores = _require_mapping(summary.get("scores"), f"summaries[{index}].scores")
        for method in METHODS:
            method_scores = _require_mapping(
                scores.get(method), f"summaries[{index}].scores.{method}"
            )
            for metric in ("brier", "log_loss"):
                _finite(
                    method_scores.get(metric),
                    f"summaries[{index}].scores.{method}.{metric}",
                    minimum=0,
                )
        calibration = _require_mapping(
            summary.get("calibration"), f"summaries[{index}].calibration"
        )
        for method in METHODS:
            bins = _require_list(
                calibration.get(method), f"summaries[{index}].calibration.{method}"
            )
            for bin_index, bin_value in enumerate(bins):
                bin_ = _require_mapping(
                    bin_value, f"summaries[{index}].calibration.{method}[{bin_index}]"
                )
                count = _count(
                    bin_.get("count"),
                    f"summaries[{index}].calibration.{method}[{bin_index}].count",
                )
                if count:
                    for key in ("mean_probability", "observed_frequency"):
                        _finite(
                            bin_.get(key),
                            f"summaries[{index}].calibration.{method}[{bin_index}].{key}",
                        )
        comparisons = _require_list(
            summary.get("paired_comparisons"), f"summaries[{index}].paired_comparisons"
        )
        for comparison_index, comparison in enumerate(comparisons):
            _comparison(comparison, f"summaries[{index}].paired_comparisons[{comparison_index}]")
        conditions = _require_mapping(summary.get("conditions"), f"summaries[{index}].conditions")
        groups = _require_mapping(conditions.get("groups"), f"summaries[{index}].conditions.groups")
        if set(groups) != {"high", "low"}:
            raise HistoricalResultError(f"summaries[{index}].conditions.groups must be high/low")
        for group_name, group_value in groups.items():
            group = _require_mapping(
                group_value, f"summaries[{index}].conditions.groups.{group_name}"
            )
            for key in ("cases", "dates"):
                _count(group.get(key), f"summaries[{index}].conditions.groups.{group_name}.{key}")
            group_comparisons = _require_list(
                group.get("paired_comparisons"),
                f"summaries[{index}].conditions.groups.{group_name}.paired_comparisons",
            )
            for comparison_index, comparison in enumerate(group_comparisons):
                _comparison(
                    comparison,
                    "summaries"
                    f"[{index}].conditions.groups.{group_name}.paired_comparisons"
                    f"[{comparison_index}]",
                )
        interactions = _require_list(
            conditions.get("interactions"), f"summaries[{index}].conditions.interactions"
        )
        for comparison_index, comparison in enumerate(interactions):
            _comparison(
                comparison,
                f"summaries[{index}].conditions.interactions[{comparison_index}]",
                interaction=True,
            )

    for index, case_value in enumerate(cases):
        case = _require_mapping(case_value, f"case_results[{index}]")
        for key in ("case_id", "event_id", "outcome_date", "as_of_at"):
            if not isinstance(case.get(key), str) or not case[key]:
                raise HistoricalResultError(f"case_results[{index}].{key} must be nonempty")
        if case.get("split") not in {"train", "validation", "holdout"}:
            raise HistoricalResultError(f"case_results[{index}].split is invalid")
        if case.get("horizon_hours") not in horizons:
            raise HistoricalResultError(f"case_results[{index}].horizon_hours is undeclared")
        bins = _require_list(case.get("bins"), f"case_results[{index}].bins", nonempty=True)
        winner = case.get("winner_index")
        if type(winner) is not int or not 0 <= winner < len(bins):
            raise HistoricalResultError(f"case_results[{index}].winner_index is invalid")
        probabilities = _require_mapping(
            case.get("probabilities"), f"case_results[{index}].probabilities"
        )
        scores = _require_mapping(case.get("scores"), f"case_results[{index}].scores")
        for method in METHODS:
            vector = _require_list(
                probabilities.get(method),
                f"case_results[{index}].probabilities.{method}",
                nonempty=True,
            )
            if len(vector) != len(bins):
                raise HistoricalResultError(
                    f"case_results[{index}].probabilities.{method} has the wrong length"
                )
            total = sum(
                _finite(item, f"case_results[{index}].probabilities.{method}", minimum=0)
                for item in vector
            )
            if not math.isclose(total, 1, rel_tol=0, abs_tol=1e-8):
                raise HistoricalResultError(
                    f"case_results[{index}].probabilities.{method} must sum to one"
                )
            method_scores = _require_mapping(
                scores.get(method), f"case_results[{index}].scores.{method}"
            )
            for metric in ("brier", "log_loss"):
                _finite(
                    method_scores.get(metric),
                    f"case_results[{index}].scores.{method}.{metric}",
                    minimum=0,
                )

    for key, kind in (
        ("split_horizon_inventory", list),
        ("exclusions", list),
        ("exclusion_reason_counts", dict),
        ("provenance", dict),
        ("interpretation", dict),
    ):
        if not isinstance(results.get(key), kind):
            raise HistoricalResultError(f"{key} has the wrong type")
    _no_nonfinite_numbers(results)
    return results


def _decode_results(body: bytes) -> dict[str, Any]:
    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                raise HistoricalResultError(f"duplicate JSON field: {key}")
            result[key] = item
        return result

    try:
        decoded = json.loads(body, object_pairs_hook=unique_object)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise HistoricalResultError(f"invalid historical result JSON: {exc}") from None
    return validate_historical_results(decoded)


def _write_bytes_atomic(path: Path, body: bytes) -> None:
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
        os.replace(temporary, path)
        if path.read_bytes() != body:
            raise OSError(f"atomic publication verification failed: {path}")
    finally:
        temporary.unlink(missing_ok=True)


def _write_text_atomic(path: Path, body: str) -> None:
    _write_bytes_atomic(path, body.encode())


def _publish_path(path: Path, body: bytes) -> None:
    if path.exists():
        if path.read_bytes() != body:
            raise HistoricalResultError(f"refusing to replace different artifact: {path}")
        return
    _write_bytes_atomic(path, body)


def _table(headers: list[str], rows: list[list[Any]]) -> str:
    def cell(value: Any) -> str:
        if value is None:
            return "—"
        if isinstance(value, float):
            return f"{value:.4f}"
        return html.escape(str(value))

    body = []
    for row in rows:
        cells = []
        for value in row:
            cells.append(f"<td>{cell(value)}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    return (
        "<table><thead><tr>"
        + "".join(f"<th>{cell(x)}</th>" for x in headers)
        + "</tr></thead><tbody>"
        + "".join(body)
        + "</tbody></table>"
    )


def _render_historical_report(
    results: dict[str, Any], output_dir: Path, result_bytes: bytes
) -> dict[str, str]:
    """Write validated JSON plus a standalone HTML report with bundled Plotly assets."""
    import plotly.graph_objects as go

    output_dir = Path(output_dir)
    json_path = output_dir / "historical-results.json"
    html_path = output_dir / "historical-report.html"
    summaries = results["summaries"]
    score_rows = []
    for s in summaries:
        for method in METHODS:
            score_rows.append(
                [
                    s["split"],
                    s["horizon_hours"],
                    method,
                    s["scores"][method]["brier"],
                    s["scores"][method]["log_loss"],
                    s["events"],
                    s["dates"],
                ]
            )
    comparison_rows = []
    for s in summaries:
        for c in s["paired_comparisons"]:
            comparison_rows.append(
                [
                    s["split"],
                    s["horizon_hours"],
                    c["method"],
                    c["metric"],
                    c["mean_difference"],
                    " to ".join(f"{v:.4f}" for v in c["ci95"])
                    if c["ci95"]
                    else c["interval_unavailable_reason"],
                    s["dates"],
                ]
            )
    group_rows = []
    for s in summaries:
        for group, g in sorted(s["conditions"]["groups"].items()):
            for c in g["paired_comparisons"]:
                group_rows.append(
                    [
                        s["split"],
                        s["horizon_hours"],
                        group,
                        g["cases"],
                        g["dates"],
                        c["method"],
                        c["metric"],
                        c["mean_difference"],
                        " to ".join(f"{v:.4f}" for v in c["ci95"])
                        if c["ci95"]
                        else c["interval_unavailable_reason"].replace("_", " "),
                    ]
                )
    interaction_rows = []
    for s in summaries:
        for c in s["conditions"]["interactions"]:
            interaction_rows.append(
                [
                    s["split"],
                    s["horizon_hours"],
                    c["method"],
                    c["metric"],
                    c["mean_difference"],
                    " to ".join(f"{v:.4f}" for v in c["ci95"])
                    if c["ci95"]
                    else c["interval_unavailable_reason"].replace("_", " "),
                    c["valid_replicates"],
                ]
            )
    pieces = [
        "<h1>Historical reconstruction: market and public weather forecasts</h1>",
        "<p>Empirical, exploratory local study. Negative paired loss differences favor the method. "
        "Historical candles do not establish fills. Reconstructed forecast availability "
        "is not an observed publication timestamp.</p>",
        _table(
            [
                "Cases",
                "Distinct events",
                "Distinct dates",
                "Fitted market weight",
                "Residual mean °F",
                "Residual SD °F",
            ],
            [
                [
                    results["cohort"]["cases"],
                    results["cohort"]["events"],
                    results["cohort"]["dates"],
                    results["fit"]["trained_market_weight"],
                    results["fit"]["residual_mean_f"],
                    results["fit"]["residual_std_f"],
                ]
            ],
        ),
        '<section id="event-inspector" aria-labelledby="event-inspector-title">'
        '<h2 id="event-inspector-title">Inspect an event and checkpoint</h2>'
        "<p>Saved probabilities, outcomes and source records for the same scored case. "
        "All times below are UTC. Changing the selection does not fit or fetch anything.</p>"
        '<label for="case-select">Event and checkpoint</label> '
        '<select id="case-select"></select>'
        '<p id="case-summary" aria-live="polite"></p>'
        '<p id="case-notices" class="notice"></p>'
        '<div id="case-probability-chart"></div>'
        '<div id="case-probabilities"></div>'
        '<h3>Scores for this case</h3><div id="case-scores"></div>'
        '<h3>Forecast timing and target</h3><div id="case-timing"></div>'
        '<h3>Historical candle closes</h3><div id="case-candles"></div>'
        "<details><summary>Selected event source lineage</summary>"
        '<pre id="case-lineage"></pre></details></section>',
        "<h2>Scores on identical cases</h2>",
        _table(
            ["Split", "Horizon h", "Method", "Brier", "Log loss", "Events", "Dates"],
            score_rows,
        ),
    ]
    figure = go.Figure()
    heldout = [s for s in summaries if s["split"] == "holdout"]
    for method in METHODS:
        figure.add_bar(
            name=method,
            x=[f"{s['horizon_hours']}h" for s in heldout],
            y=[s["scores"][method]["brier"] for s in heldout],
        )
    figure.update_layout(
        title="Holdout multiclass Brier loss",
        barmode="group",
        yaxis_title="Mean Brier loss",
        template="plotly_white",
    )
    pieces.append(
        figure.to_html(full_html=False, include_plotlyjs=True, div_id="holdout-brier-chart")
    )
    pieces.extend(
        [
            "<h2>Paired differences and date-block uncertainty</h2>",
            "<p>95% marginal intervals use shared circular date blocks. Training is in-sample. "
            "Block-length sensitivity and replicate support are retained in the JSON.</p>",
            _table(
                [
                    "Split",
                    "Horizon h",
                    "Method - public",
                    "Metric",
                    "Difference",
                    "95% interval",
                    "Dates",
                ],
                comparison_rows,
            ),
            "<h2>Calibration</h2><p>Descriptive bin frequencies sharing event outcomes.</p>",
        ]
    )
    calibration = go.Figure()
    calibration.add_scatter(
        x=[0, 1],
        y=[0, 1],
        mode="lines",
        name="Perfect calibration",
        line={"dash": "dash", "color": "gray"},
    )
    for s in heldout:
        for method in METHODS:
            bins = [b for b in s["calibration"][method] if b["count"]]
            calibration.add_scatter(
                x=[b["mean_probability"] for b in bins],
                y=[b["observed_frequency"] for b in bins],
                text=[f"Bin observations: {b['count']}" for b in bins],
                mode="lines+markers",
                name=f"{method}, {s['horizon_hours']}h",
            )
    calibration.update_layout(
        title="Holdout calibration",
        xaxis_title="Mean forecast probability",
        yaxis_title="Observed frequency",
        template="plotly_white",
    )
    pieces.append(
        calibration.to_html(
            full_html=False, include_plotlyjs=False, div_id="holdout-calibration-chart"
        )
    )
    support = []
    for summary in heldout:
        groups = []
        for group, values in sorted(summary["conditions"]["groups"].items()):
            groups.append(f"{html.escape(group)} disagreement {values['dates']} dates")
        support.append(f"{summary['horizon_hours']}h: " + ", ".join(groups))
    pieces.extend(
        [
            "<h2>Disagreement conditions</h2>",
            "<p>High disagreement means L1(market, public) &gt; the training median "
            f"{results['fit']['disagreement_l1_median']:.4f}. "
            "Interactions compare high-minus-low paired loss differences with shared draws. "
            "Undefined groups receive no interval.</p>",
            '<p class="notice">Holdout support: '
            + "; ".join(support)
            + ". Sparse groups limit the condition comparison. Missing intervals below "
            "show their saved reason; they are not zero-width intervals.</p>",
            _table(
                [
                    "Split",
                    "Horizon h",
                    "Group",
                    "Cases",
                    "Dates",
                    "Method - public",
                    "Metric",
                    "Difference",
                    "95% interval",
                ],
                group_rows,
            ),
            _table(
                [
                    "Split",
                    "Horizon h",
                    "Method",
                    "Metric",
                    "High - low interaction",
                    "95% interval",
                    "Valid draws",
                ],
                interaction_rows,
            ),
            "<h2>Exclusions and lineage</h2>",
        ]
    )
    for label, data in (
        ("Exclusion counts", results["exclusion_reason_counts"]),
        ("Exclusion records", results["exclusions"]),
        ("Training fit and label availability", results["fit"]),
        ("Protocol", results["protocol"]),
        ("Source lineage", results["provenance"]),
        ("Interpretation and limits", results["interpretation"]),
    ):
        escaped = html.escape(json.dumps(data, indent=2, sort_keys=True))
        pieces.append(f"<details><summary>{label}</summary><pre>{escaped}</pre></details>")
    pieces.append(
        f"<p>Panel SHA-256: {html.escape(results['input_panel_sha256'])}<br>"
        f"Protocol SHA-256: {html.escape(results['protocol_sha256'])}</p>"
    )
    # escape the script boundary; insert source values as text in the DOM
    case_data = (
        json.dumps(results["case_results"], allow_nan=False, sort_keys=True)
        .replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
    )
    pieces.append('<script id="case-data" type="application/json">' + case_data + "</script>")
    pieces.append(
        r"""<script>
(() => {
  const cases = JSON.parse(document.getElementById('case-data').textContent);
  const methods = ['public_only', 'market_only', 'equal_blend', 'trained_blend'];
  const select = document.getElementById('case-select');
  const text = value => value === null || value === undefined ? 'Not recorded' : String(value);
  const number = value => typeof value === 'number' ? value.toFixed(4) : text(value);
  const range = bin => bin.lower === null ? '≤ ' + bin.upper + '°F'
    : bin.upper === null ? '≥ ' + bin.lower + '°F'
    : bin.lower + ' to ' + bin.upper + '°F';
  function table(id, headers, rows) {
    const node = document.createElement('table');
    const head = node.createTHead().insertRow();
    headers.forEach(value => {
      const cell = document.createElement('th');
      cell.scope = 'col';
      cell.textContent = text(value);
      head.append(cell);
    });
    const body = node.createTBody();
    rows.forEach(values => {
      const row = body.insertRow();
      values.forEach(value => { row.insertCell().textContent = text(value); });
    });
    document.getElementById(id).replaceChildren(node);
  }
  cases.forEach((row, index) => {
    const option = document.createElement('option');
    option.value = String(index);
    option.textContent = row.outcome_date + ' · ' + row.horizon_hours + 'h · '
      + row.split + ' · ' + row.case_id;
    select.append(option);
  });
  function renderCase() {
    const row = cases[Number(select.value)];
    if (!row) return;
    const forecast = row.forecast || {};
    const object = forecast.object || {};
    const candles = row.market_observations || [];
    const winningBin = row.bins[row.winner_index];
    const reconciliation = row.ncei_tmax_f === null || row.ncei_tmax_f === undefined
      ? 'NCEI TMAX is not recorded.'
      : row.ncei_matches_settlement
        ? 'NCEI TMAX agrees with settlement (' + row.ncei_tmax_f + '°F).'
        : 'NCEI mismatch: TMAX ' + row.ncei_tmax_f + '°F versus settlement '
          + row.outcome_f + '°F. The case is retained.';
    document.getElementById('case-summary').textContent = row.case_id
      + ' · ' + row.split + ' · settlement ' + row.outcome_f + '°F · winning bin '
      + range(winningBin) + '. ' + reconciliation;
    const missingTiming = ['issued_at', 'available_at', 'valid_start_at', 'valid_end_at']
      .some(key => !forecast[key]);
    document.getElementById('case-notices').textContent =
      'Availability is a historical archive bound, not an observed first-publication time. '
      + 'NDFD daytime grid MaxT and the full-day station settlement window differ. '
      + 'Hourly bid/ask closes do not establish fills or depth. '
      + (missingTiming || candles.length !== row.bins.length
        ? 'Some saved forecast timing or candle records are missing; see “Not recorded” below. '
        : '')
      + (row.split === 'train' && !row.used_for_fit
        ? 'This training-date case was excluded from fitting because its label was unavailable.'
        : '');
    table('case-probabilities', ['Bin', 'Market ticker', ...methods, 'Observed winner'],
      row.bins.map((bin, i) => [range(bin), bin.ticker,
        ...methods.map(method => number(row.probabilities[method][i])),
        i === row.winner_index ? 'Yes' : 'No']));
    table('case-scores', ['Method', 'Brier', 'Log loss'], methods.map(method =>
      [method, number(row.scores[method].brier), number(row.scores[method].log_loss)]));
    table('case-timing', ['Saved field', 'Value'], [
      ['Prediction checkpoint', row.as_of_at],
      ['Forecast reference / issue time', forecast.issued_at],
      ['Bulletin nominal time', object.nominal_at],
      ['Archive last modification', object.last_modified_at],
      ['Forecast availability bound', row.forecast_available_at || forecast.available_at],
      ['Forecast valid start', forecast.valid_start_at],
      ['Forecast valid end', forecast.valid_end_at],
      ['Settlement window start', row.outcome_window_start_at],
      ['Settlement window end', row.outcome_window_end_at],
      ['Outcome availability', row.outcome_available_at],
      ['NDFD MaxT °F', number(row.forecast_f)],
      ['Fitted public mean °F', number(row.public_mean_f)],
      ['Nearest grid-cell distance km', number(forecast.grid_distance_km)],
      ['L1 disagreement / group', number(row.disagreement_l1) + ' / ' + row.disagreement_group],
      ['Sum of raw market midpoints', number(row.market_probability_sum_before_normalization)]
    ]);
    table('case-candles', ['Market ticker', 'Candle end UTC', 'Age minutes', 'Bid $', 'Ask $',
      'Raw midpoint', 'Spread'], row.bins.map((bin, i) => {
      const candle = candles[i] || {};
      return [bin.ticker, candle.candle_end_at, number(candle.candle_age_minutes),
        candle.bid_dollars, candle.ask_dollars, number(candle.probability), number(candle.spread)];
    }));
    document.getElementById('case-lineage').textContent = JSON.stringify({
      case_id: row.case_id,
      market_rules_source_sha256: row.market_rules_source_sha256 || [],
      market_candle_source_sha256: row.market_source_sha256 || {},
      forecast_source_sha256: forecast.source_sha256 || null,
      forecast_listing_sha256: forecast.listing_sha256 || null,
      forecast_source_uri: forecast.source_uri || null,
      forecast_archive_object: forecast.object || null,
      forecast_message_number: forecast.message_number || null
    }, null, 2);
    Plotly.react('case-probability-chart', methods.map(method => ({
      type: 'bar', name: method, x: row.bins.map(range), y: row.probabilities[method]
    })), {
      title: {text: 'Saved probabilities at this checkpoint'}, barmode: 'group',
      yaxis: {title: {text: 'Probability'}, range: [0, 1]},
      xaxis: {title: {text: 'Settlement temperature bin'}},
      legend: {orientation: 'h', x: 0, y: 1.05, yanchor: 'bottom'},
      margin: {t: 100, b: 70},
      paper_bgcolor: '#ffffff', plot_bgcolor: '#ffffff'
    }, {responsive: true, displaylogo: false});
  }
  select.addEventListener('change', renderCase);
  if (cases.length) renderCase();
  else {
    select.disabled = true;
    document.getElementById('case-summary').textContent = 'No saved cases are available.';
  }
})();
</script>"""
    )
    document = (
        '<!doctype html><html lang="en"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        "<title>Historical weather forecast evaluation</title><style>"
        "body{font:16px system-ui;color:#183047;background:#f5f7f9;"
        "max-width:1180px;margin:36px auto;padding:0 20px}"
        "table{border-collapse:collapse;background:white;font-size:13px;"
        "margin:20px 0;width:100%;display:block;overflow:auto}"
        "th,td{text-align:left;padding:9px;border-bottom:1px solid #d8e0e7}th{background:#e8eef4}"
        "pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px}"
        "details{margin:16px 0}h2{margin-top:40px}"
        "select{font:inherit;max-width:100%;padding:10px;margin:8px 0;background:white}"
        "label{display:block;font-weight:600}.notice{border-left:3px solid #b17726;"
        "padding:12px 16px;background:#fff7e9}#case-probability-chart{min-height:420px}"
        "#case-summary{font-weight:600;line-height:1.6}"
        "</style><main>" + "\n".join(pieces) + "</main></html>"
    )
    _publish_path(json_path, result_bytes)
    _write_text_atomic(html_path, document)
    return {"json": str(json_path.resolve()), "html": str(html_path.resolve())}


def _publish_validated_results(
    results: dict[str, Any], output_dir: Path, body: bytes
) -> dict[str, str]:
    from .historical_runs import (
        guard_results_publication,
        results_input_identity,
        write_results_manifest,
    )

    protocol_sha256 = results["input_protocol_sha256"]
    panel_sha256 = results["input_panel_sha256"]
    input_sha256 = results_input_identity(
        protocol_sha256=protocol_sha256,
        panel_sha256=panel_sha256,
    )
    result_sha256 = hashlib.sha256(body).hexdigest()
    status = guard_results_publication(
        output_dir,
        input_sha256=input_sha256,
        proposed_result_sha256=result_sha256,
    )
    paths = _render_historical_report(results, output_dir, body)
    if not (status.complete and status.legacy):
        write_results_manifest(
            output_dir,
            protocol_sha256=protocol_sha256,
            panel_sha256=panel_sha256,
            input_sha256=input_sha256,
            result_sha256=result_sha256,
        )
    return paths


def write_historical_report(results: dict[str, Any], output_dir: Path) -> dict[str, str]:
    """Compatibility writer for already evaluated in-memory historical results."""
    validated = validate_historical_results(results)
    body = (json.dumps(validated, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    return _publish_validated_results(validated, Path(output_dir), body)


def render_saved_historical_results(results_path: Path, output_dir: Path) -> dict[str, str]:
    """Render a saved empirical result without refitting, collecting, or changing its bytes."""
    results_path = Path(results_path)
    output_dir = Path(output_dir)
    body = results_path.read_bytes()
    results = _decode_results(body)
    source = results_path.resolve()
    destination = (output_dir / "historical-results.json").resolve()
    if source == destination:
        raise HistoricalResultError("saved result source and render destination must differ")

    return _publish_validated_results(results, output_dir, body)
