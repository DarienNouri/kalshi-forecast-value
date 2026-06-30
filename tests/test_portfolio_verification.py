"""Portable verification catches changed empirical evidence and report artifacts."""

import importlib.util
import json
from copy import deepcopy
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
MANIFEST = ROOT / "reports/portfolio-evidence.json"
RESULTS = ROOT / "outputs/historical-replication/analysis/historical-results.json"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


verifier = _load("verify_portfolio", ROOT / "scripts/verify_portfolio.py")


def _manifest_copy(tmp_path: Path, value: dict | None = None) -> Path:
    path = tmp_path / "portfolio-evidence.json"
    path.write_text(json.dumps(value or json.loads(MANIFEST.read_bytes())))
    return path


def test_checked_in_portfolio_verifies() -> None:
    result = verifier.verify(MANIFEST)

    assert result["ok"] is True
    assert result["figures_verified"] == 3
    assert result["replication_result"] in {"verified", "recorded_not_rechecked"}
    assert result["integrity_level"] in {
        "source_results_and_portfolio",
        "portable_portfolio",
    }


def test_portable_manifest_verifies_when_private_result_is_absent(tmp_path: Path) -> None:
    value = json.loads(MANIFEST.read_bytes())
    value["inputs"]["replication_results"]["result_path"] = str(
        tmp_path / "unavailable-private-result.json"
    )
    result = verifier.verify(_manifest_copy(tmp_path, value))

    assert result["ok"] is True
    assert result["replication_result"] == "recorded_not_rechecked"
    assert result["integrity_level"] == "portable_portfolio"


def test_changed_empirical_aggregate_is_rejected(tmp_path: Path) -> None:
    value = json.loads(MANIFEST.read_bytes())
    value["inputs"]["replication_results"]["holdout"][0]["scores"]["market_only"][
        "brier"
    ] = 0

    with pytest.raises(verifier.PortfolioVerificationError, match=r"stale|aggregate data"):
        verifier.verify(_manifest_copy(tmp_path, value))


def test_changed_output_hash_is_rejected(tmp_path: Path) -> None:
    value = json.loads(MANIFEST.read_bytes())
    value["output"]["index"]["sha256"] = "0" * 64

    with pytest.raises(verifier.PortfolioVerificationError, match="index hash conflict"):
        verifier.verify(_manifest_copy(tmp_path, value))


def test_changed_style_hash_is_rejected(tmp_path: Path) -> None:
    value = json.loads(MANIFEST.read_bytes())
    value["code"]["style"]["sha256"] = "0" * 64

    with pytest.raises(verifier.PortfolioVerificationError, match="style hash conflict"):
        verifier.verify(_manifest_copy(tmp_path, value))


def test_changed_result_copy_is_rejected_before_use(tmp_path: Path) -> None:
    if not RESULTS.is_file():
        pytest.skip("retained private result is not present")
    value = json.loads(RESULTS.read_bytes())
    changed = deepcopy(value)
    holdout = next(row for row in changed["summaries"] if row["split"] == "holdout")
    holdout["scores"]["market_only"]["brier"] += .01
    result_path = tmp_path / "historical-results.json"
    result_path.write_text(json.dumps(changed))

    with pytest.raises(verifier.PortfolioVerificationError, match="result file hash conflict"):
        verifier.verify(MANIFEST, results_override=result_path)


def test_figure_data_are_bound_to_the_empirical_result() -> None:
    evidence = json.loads(MANIFEST.read_bytes())
    replication = evidence["inputs"]["replication_results"]
    figures = evidence["output"]["figures"]

    assert set(figures) == {"holdout-losses", "paired-differences", "monthly-coverage"}
    assert all(
        record["source_result_sha256"] == replication["result_file_sha256"]
        for record in figures.values()
    )
    assert {row["horizon_hours"] for row in figures["holdout-losses"]["data"]} == {6, 12}
