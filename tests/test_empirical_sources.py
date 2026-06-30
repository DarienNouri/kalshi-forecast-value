import hashlib
import json
import math
import runpy
import shutil
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import pytest

from kalshi_information_quality_lab.historical_replication_inputs import (
    combine_replication_panels,
)

ROOT = Path(__file__).resolve().parents[1]
PANEL = ROOT / "outputs/historical-replication/panel.json"
CHECKS = runpy.run_path(str(ROOT / "scripts/verify_replication_sources.py"))
REPRODUCTION = ROOT / "outputs/reproduction"


@pytest.fixture(scope="module")
def panel():
    if not PANEL.is_file():
        pytest.skip("retained empirical source cache is not installed")
    data = json.loads(PANEL.read_text())
    for part in data["provenance"]["partitions"]:
        if not (ROOT / part["raw_directory"]).is_dir():
            pytest.skip("retained empirical source cache is not installed")
    return data


def test_retained_source_bytes_and_cohort(panel):
    result = CHECKS["build_data_quality"](ROOT, PANEL.relative_to(ROOT))
    assert result["requested_dates"] == 395
    assert result["eligible_dates"] == 326
    assert result["eligible_cases"] == len(panel["cases"]) == 652
    assert result["excluded_dates"] == 69


def test_retained_inputs_precede_prediction(panel):
    for row in panel["cases"]:
        as_of = datetime.fromisoformat(row["as_of_at"])
        start = datetime.fromisoformat(row["outcome_window_start_at"])
        end = datetime.fromisoformat(row["outcome_window_end_at"])
        issued = datetime.fromisoformat(row["forecast"]["issued_at"])
        available = datetime.fromisoformat(row["forecast_available_at"])
        assert issued <= available <= as_of < start < end, row["case_id"]
        assert (start - as_of).total_seconds() == row["horizon_hours"] * 3600
        assert datetime.fromisoformat(row["outcome_available_at"]) >= end
        assert len(row["market_observations"]) == len(row["bins"])
        for quote in row["market_observations"]:
            assert datetime.fromisoformat(quote["candle_end_at"]) <= as_of
        assert math.isclose(sum(row["market_probabilities"]), 1, abs_tol=1e-12)


def test_real_partition_combine_omits_administrative_metadata(panel):
    protocol_path = ROOT / "config/replication-protocol.json"
    legacy_path = ROOT / "outputs/historical-study/panel.json"
    new_path = ROOT / "outputs/historical-replication/new-acquisition/panel.json"
    if not legacy_path.is_file() or not new_path.is_file():
        pytest.skip("retained empirical partitions are not installed")

    protocol_bytes = protocol_path.read_bytes()
    combined = combine_replication_panels(
        json.loads(protocol_bytes),
        legacy_path.read_bytes(),
        new_path.read_bytes(),
        replication_protocol_sha256=hashlib.sha256(protocol_bytes).hexdigest(),
        target_inventory=panel["target_inventory"],
    )
    assert "execution_basis" not in combined["provenance"]

    retained_research = deepcopy(panel)
    retained_research["provenance"].pop("execution_basis", None)
    assert combined == retained_research


def test_changed_source_bytes_are_rejected(panel, tmp_path):
    raw = ROOT / panel["provenance"]["partitions"][0]["raw_directory"]
    body = min(raw.glob("*.body"), key=lambda path: path.stat().st_size)
    receipt = body.with_suffix(".json")
    copied = tmp_path / "raw"
    copied.mkdir()
    shutil.copyfile(body, copied / body.name)
    shutil.copyfile(receipt, copied / receipt.name)

    CHECKS["_audit_raw"](tmp_path, "raw")
    (copied / body.name).write_bytes(body.read_bytes() + b"\n")
    with pytest.raises(CHECKS["VerificationError"]):
        CHECKS["_audit_raw"](tmp_path, "raw")


def test_replay_compares_research_content_and_panel_identity():
    panel_path = ROOT / "outputs/historical-study/panel.json"
    results_path = ROOT / "outputs/historical-study/analysis/historical-results.json"
    if not panel_path.is_file() or not results_path.is_file():
        pytest.skip("retained historical study is not installed")
    replay = runpy.run_path(str(ROOT / "scripts/reproduce_historical.py"))
    original = json.loads(panel_path.read_bytes())
    results = json.loads(results_path.read_bytes())
    replay["check_result_panel"](original, results)

    cleaned = deepcopy(original)
    cleaned["provenance"].pop("execution_basis", None)
    assert replay["scientific_hash"](cleaned) == replay["scientific_hash"](original)
    with pytest.raises(replay["ReproductionError"], match="input panel"):
        replay["check_result_panel"](cleaned, results)
    linked = deepcopy(results)
    linked["input_panel_sha256"] = hashlib.sha256(
        json.dumps(cleaned, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()
    replay["check_result_panel"](cleaned, linked)
    assert replay["scientific_hash"](linked, result=True) == replay["scientific_hash"](
        results, result=True
    )

    corrupted = deepcopy(cleaned)
    corrupted["cases"][0]["outcome_f"] += 1
    assert replay["scientific_hash"](corrupted) != replay["scientific_hash"](original)


def test_cached_reproduction_rechecks_real_artifacts_and_requested_inputs(
    tmp_path, monkeypatch, capsys
):
    retained = ROOT / "outputs/historical-study"
    receipt_path = REPRODUCTION / "reproduction.json"
    replayed_panel = REPRODUCTION / "panel.json"
    replayed_results = REPRODUCTION / "analysis/historical-results.json"
    source_panel = retained / "panel.json"
    source_results = retained / "analysis/historical-results.json"
    source_raw = retained / "raw"
    if not all(
        path.is_file()
        for path in (
            receipt_path,
            replayed_panel,
            replayed_results,
            source_panel,
            source_results,
        )
    ) or not source_raw.is_dir():
        pytest.skip("completed retained-cache reproduction is not installed")

    source = tmp_path / "source"
    (source / "analysis").mkdir(parents=True)
    shutil.copyfile(source_panel, source / "panel.json")
    shutil.copyfile(source_results, source / "analysis/historical-results.json")
    (source / "raw").symlink_to(source_raw, target_is_directory=True)
    output = tmp_path / "reproduction"
    (output / "analysis").mkdir(parents=True)
    for artifact, relative in (
        (receipt_path, "reproduction.json"),
        (replayed_panel, "panel.json"),
        (replayed_results, "analysis/historical-results.json"),
    ):
        shutil.copyfile(artifact, output / relative)
    config = tmp_path / "historical-study.toml"
    protocol = tmp_path / "historical-protocol.json"
    shutil.copyfile(ROOT / "config/historical-study.toml", config)
    shutil.copyfile(ROOT / "config/historical-protocol.json", protocol)

    replay = runpy.run_path(str(ROOT / "scripts/reproduce_historical.py"))
    receipt = json.loads(receipt_path.read_bytes())
    main = replay["main"]
    monkeypatch.setattr(replay["sys"], "prefix", str(tmp_path / "isolated-prefix"))
    monkeypatch.setitem(
        main.__globals__, "_tree_identity", lambda _: receipt["source_raw_tree_sha256"]
    )
    args = [
        "--run-dir",
        str(source),
        "--output-dir",
        str(output),
        "--config",
        str(config),
        "--protocol",
        str(protocol),
    ]
    assert main(args) == 0
    capsys.readouterr()

    for field, value in {
        "schema_version": 2,
        "scientific_content_matches": False,
        "panel_scientific_sha256": "0" * 64,
        "results_scientific_sha256": "0" * 64,
        "panel_sha256": "0" * 64,
        "results_sha256": "0" * 64,
        "source_panel_sha256": "0" * 64,
        "source_results_sha256": "0" * 64,
        "protocol_sha256": "0" * 64,
        "config_sha256": "0" * 64,
        "source_raw_tree_sha256": "0" * 64,
        "source_preserved": False,
    }.items():
        corrupted_receipt = deepcopy(receipt)
        corrupted_receipt[field] = value
        (output / "reproduction.json").write_text(json.dumps(corrupted_receipt))
        with pytest.raises(replay["ReproductionError"], match="existing reproduction"):
            main(args)

    corrupted_receipt = deepcopy(receipt)
    corrupted_receipt["environment"]["uv_lock_sha256"] = "0" * 64
    (output / "reproduction.json").write_text(json.dumps(corrupted_receipt))
    with pytest.raises(replay["ReproductionError"], match="existing reproduction"):
        main(args)

    shutil.copyfile(receipt_path, output / "reproduction.json")
    changed_protocol = json.loads(protocol.read_bytes())
    changed_protocol["end_date"] = changed_protocol["start_date"]
    protocol.write_text(json.dumps(changed_protocol))
    with pytest.raises(replay["ReproductionError"], match="requested protocol differs"):
        main(args)

    shutil.copyfile(ROOT / "config/historical-protocol.json", protocol)
    config.write_text(config.read_text() + "\n# changed requested configuration\n")
    with pytest.raises(replay["ReproductionError"], match="existing reproduction"):
        main(args)
