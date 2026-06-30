"""The checked-in portfolio presents the saved empirical study faithfully."""

import importlib.util
import json
import struct
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).parents[1]
MANIFEST = ROOT / "reports/portfolio-evidence.json"
SCRIPT = ROOT / "scripts/build_portfolio.py"
SPEC = importlib.util.spec_from_file_location("build_portfolio", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
portfolio = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(portfolio)


def _evidence() -> dict:
    return json.loads(MANIFEST.read_bytes())


def _png_dimensions(path: Path) -> tuple[int, int]:
    with path.open("rb") as stream:
        assert stream.read(8) == b"\x89PNG\r\n\x1a\n"
        assert stream.read(4) == b"\x00\x00\x00\r"
        assert stream.read(4) == b"IHDR"
        return struct.unpack(">II", stream.read(8))


def test_report_is_generated_from_the_recorded_empirical_aggregate() -> None:
    replication = _evidence()["inputs"]["replication_results"]

    assert portfolio._report_markdown(replication) == (ROOT / "reports/report.md").read_text()
    assert replication["cohort"] == {"cases": 652, "dates": 326, "events": 326}
    assert replication["fit"]["selected_candidate"] == "p1_lambda_1"
    assert replication["fit"]["trained_market_weight"] == 1


def test_report_uses_explanatory_model_names_without_changing_machine_ids() -> None:
    markdown = (ROOT / "reports/report.md").read_text()
    document = (ROOT / "reports/index.html").read_text()

    for text in (markdown, document):
        assert "pooled residual baseline" in text
        assert "seasonal ridge residual model" in text
        assert "p0 reference" not in text
        assert "p1_lambda_1" not in text
    assert _evidence()["inputs"]["replication_results"]["fit"]["selected_candidate"] == (
        "p1_lambda_1"
    )


def test_site_has_portable_links_equations_and_empirical_figures() -> None:
    document = (ROOT / "reports/index.html").read_text()
    markdown = (ROOT / "reports/report.md").read_text()
    stylesheet = (ROOT / "reports/assets/site.css").read_text()

    class Links(HTMLParser):
        def __init__(self) -> None:
            super().__init__()
            self.values: list[str] = []

        def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
            del tag
            self.values.extend(
                value
                for key, value in attrs
                if key in {"href", "src"} and value is not None
            )

    links = Links()
    links.feed(document)
    assert "assets/site.css" in links.values
    assert "portfolio-evidence.json" in links.values
    assert "../notebooks/06_historical_replication.ipynb" in links.values
    assert "../notebooks/05_historical_study.ipynb" in links.values
    assert "../notebooks/04_noaa_grid_validation.ipynb" in links.values
    assert all(
        value.startswith("#") or not value.startswith(("/", "http://", "https://", "file:"))
        for value in links.values
        if not value.startswith("data:")
    )
    assert "$$BS_i =" in markdown and "$$LL_i =" in markdown
    assert 'class="math-block"' in document and 'class="math-inline"' in document
    assert "assets/holdout-losses.png" in markdown
    assert "assets/paired-differences.png" in markdown
    assert "assets/monthly-coverage.png" in markdown
    assert "@media(max-width:600px)" in stylesheet
    assert ":focus-visible" in stylesheet


def test_report_figures_are_high_resolution() -> None:
    figures = _evidence()["output"]["figures"]

    for record in figures.values():
        width, height = _png_dimensions(ROOT / record["path"])
        assert width >= 2000
        assert height >= 900


def test_manifest_contains_only_scientific_inputs_and_portable_artifacts() -> None:
    assert set(_evidence()) == {"schema_version", "study", "inputs", "code", "output"}
