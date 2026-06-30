"""Portable checks for the retained empirical study and its core calculations."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from kalshi_information_quality_lab import cli
from kalshi_information_quality_lab.config import load_config
from kalshi_information_quality_lab.evaluation import log_loss, multiclass_brier
from kalshi_information_quality_lab.historical_evaluation import _validated_cases
from kalshi_information_quality_lab.historical_inputs import (
    HistoricalInputError,
    validate_protocol,
)
from kalshi_information_quality_lab.historical_models import normal_bin_probabilities
from kalshi_information_quality_lab.historical_replication import _fit_pipeline

ROOT = Path(__file__).parents[1]
EVIDENCE = ROOT / "reports/portfolio-evidence.json"
RESULTS = ROOT / "outputs/historical-replication/analysis/historical-results.json"
PANEL = ROOT / "outputs/historical-replication/panel.json"


def test_portable_manifest_preserves_empirical_cohort_and_holdout_scores() -> None:
    evidence = json.loads(EVIDENCE.read_bytes())
    replication = evidence["inputs"]["replication_results"]
    assert evidence["schema_version"] == "portfolio_evidence_v2"
    assert replication["cohort"] == {"cases": 652, "dates": 326, "events": 326}
    by_horizon = {row["horizon_hours"]: row for row in replication["holdout"]}
    assert set(by_horizon) == {6, 12}
    assert by_horizon[12]["scores"]["market_only"]["brier"] == pytest.approx(
        .6939115500641683
    )
    assert by_horizon[6]["scores"]["public_only"]["brier"] == pytest.approx(
        .7616730880899202
    )
    assert replication["fit"]["trained_market_weight"] == pytest.approx(1)
    assert evidence["study"] == "empirical_historical_weather_comparison"


def test_saved_empirical_case_scores_recompute_from_probabilities() -> None:
    if not RESULTS.is_file():
        pytest.skip("private empirical result is unavailable in this checkout")
    result = json.loads(RESULTS.read_bytes())
    epsilon = result["protocol"]["metrics"]["log_loss_epsilon"]
    assert len(result["case_results"]) == 652
    for row in result["case_results"]:
        winner = row["bins"][row["winner_index"]]["ticker"]
        assert normal_bin_probabilities(
            row["public_mean_f"],
            row["public_std_f"],
            row["bins"],
        ) == pytest.approx(row["probabilities"]["public_only"])
        for method, values in row["probabilities"].items():
            probabilities = {
                bin_["ticker"]: value
                for bin_, value in zip(row["bins"], values, strict=True)
            }
            assert multiclass_brier(probabilities, winner) == pytest.approx(
                row["scores"][method]["brier"]
            )
            assert log_loss(probabilities, winner, epsilon=epsilon) == pytest.approx(
                row["scores"][method]["log_loss"]
            )


def test_actual_replication_cohort_refits_to_saved_model_selection() -> None:
    if not RESULTS.is_file() or not PANEL.is_file():
        pytest.skip("private empirical panel and result are unavailable in this checkout")
    result = json.loads(RESULTS.read_bytes())
    panel = json.loads(PANEL.read_bytes())
    protocol = result["protocol"]
    rows = _validated_cases(panel, protocol["collection"])
    fit = _fit_pipeline(rows, protocol)
    actual = fit.to_dict()
    expected = result["fit"]

    assert actual["selected_candidate"] == expected["selected_candidate"]
    assert actual["public_model"]["spec"] == expected["public_model"]["spec"]
    assert actual["public_model"]["ridge_lambda"] == expected["public_model"]["ridge_lambda"]
    assert actual["public_model"]["coefficients"] == pytest.approx(
        expected["public_model"]["coefficients"], rel=1e-12, abs=1e-12
    )
    assert actual["p0_reference_model"]["coefficients"] == pytest.approx(
        expected["p0_reference_model"]["coefficients"], rel=1e-12, abs=1e-12
    )
    assert actual["predictive_std_by_horizon"] == pytest.approx(
        expected["predictive_std_by_horizon"], rel=1e-12, abs=1e-12
    )
    assert actual["trained_market_weight"] == pytest.approx(
        expected["trained_market_weight"], rel=1e-12, abs=1e-12
    )
    assert [row["candidate"] for row in fit.candidate_validation] == [
        row["candidate"] for row in result["candidate_validation"]
    ]
    assert [row["mean_brier"] for row in fit.candidate_validation] == pytest.approx(
        [row["mean_brier"] for row in result["candidate_validation"]],
        rel=1e-12,
        abs=1e-12,
    )


def test_frozen_protocol_validates_and_corruption_is_rejected() -> None:
    protocol = json.loads((ROOT / "config/historical-protocol.json").read_bytes())
    assert validate_protocol(protocol, collection=True) == protocol
    corrupted = deepcopy(protocol)
    corrupted["market_max_candle_age_minutes"] = -1
    with pytest.raises(HistoricalInputError, match="nonnegative"):
        validate_protocol(corrupted, collection=True)


def test_empirical_config_and_cli_surface(capsys: pytest.CaptureFixture[str]) -> None:
    config = load_config(ROOT / "config/study.toml", project_root=ROOT)
    assert config.data_mode == "historical"
    assert config.protocol_status == "frozen"
    assert config.local_research_scope == "documented_local_research"
    assert config.evidence_reference == "docs/SOURCES.md"

    assert cli.main(["validate-config", "--config", str(ROOT / "config/study.toml"), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["config_valid"] is True
    help_text = cli._build_parser().format_help()
    for command in ("historical-study", "historical-report", "historical-render", "status"):
        assert command in help_text
