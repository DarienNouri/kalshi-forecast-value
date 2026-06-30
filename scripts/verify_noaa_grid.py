#!/usr/bin/env python3
"""Validate the NDFD grid reader against the primary historical NOAA bulletin."""

from __future__ import annotations

import argparse
import json
from importlib.metadata import version
from pathlib import Path
from typing import Any

from eccodes import (
    codes_get_api_version,
    codes_get_values,
    codes_grib_new_from_file,
    codes_release,
)

from kalshi_information_quality_lab.ndfd import (
    GeographicCoordinate,
    GridCoordinate,
    extract_ndfd_max_temperature,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = (
    ROOT
    / "outputs/historical-study/raw"
    / "7b84db4a743fb287b2ad6cc31fa6e04523500a7a06024e196af94394835e1c1a.body"
)
DEFAULT_OUTPUT = ROOT / "outputs/noaa-grid-verification.json"
EXPECTED_SHA256 = "4f4887a39454f5bd09a21f41314dd802805cc25c2f563417aa6034b411738494"


def _relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return path.name


def _naive_odd_row_value(source: Path) -> float:
    gid = None
    try:
        with source.open("rb") as stream:
            for _ in range(2):
                if gid is not None:
                    codes_release(gid)
                gid = codes_grib_new_from_file(stream)
        if gid is None:
            raise AssertionError("source has no GRIB message 2")
        return float(codes_get_values(gid).reshape(1377, 2145)[859, 1811])
    finally:
        if gid is not None:
            codes_release(gid)


def verify(source: Path) -> dict[str, Any]:
    if not source.is_file():
        raise FileNotFoundError(f"required private NOAA study cache is missing: {source}")

    extraction = extract_ndfd_max_temperature(
        source,
        message_number=2,
        grid_coordinates=(
            GridCoordinate(row=859, column=1811),
            GridCoordinate(row=860, column=1811),
            GridCoordinate(row=861, column=1811),
            GridCoordinate(row=0, column=0),
        ),
        geographic_coordinates=(
            GeographicCoordinate(latitude=40.7789, longitude=-73.9692),
        ),
    )
    previous, central_park, following, missing, nearest = extraction.points
    naive_odd = _naive_odd_row_value(source)

    assert extraction.source_sha256 == EXPECTED_SHA256
    assert extraction.message_number == 2
    assert extraction.reference_at.isoformat() == "2025-05-31T16:30:00+00:00"
    assert extraction.valid_start_at.isoformat() == "2025-06-01T12:00:00+00:00"
    assert extraction.valid_end_at.isoformat() == "2025-06-02T00:00:00+00:00"
    assert extraction.grid.scanning_mode == 80
    assert extraction.grid.alternative_row_scanning is True
    assert extraction.missing_count == 1_479_351
    assert (previous.row, previous.column, previous.temperature_k) == (859, 1811, 293.7)
    assert (central_park.row, central_park.column, central_park.temperature_k) == (
        860,
        1811,
        293.7,
    )
    assert (following.row, following.column, following.temperature_k) == (861, 1811, 294.3)
    assert missing.missing and missing.temperature_k is None and missing.temperature_f is None
    assert naive_odd == 307.6 and naive_odd != previous.temperature_k
    assert (nearest.row, nearest.column, nearest.temperature_k) == (860, 1811, 293.7)
    assert nearest.distance_km is not None and nearest.distance_km < .4

    return {
        "schema_version": "noaa_grid_verification_v1",
        "passed": True,
        "data_origin": "empirical_historical",
        "research_scope": extraction.research_scope,
        "source_path": _relative(source),
        "source_sha256": extraction.source_sha256,
        "message_number": extraction.message_number,
        "decoder": {
            "python_package": f"eccodes=={version('eccodes')}",
            "native_api_version": codes_get_api_version(),
            "aligned_api": "codes_get_array(gid, 'latLonValues')",
        },
        "checks": [
            "exact_primary_source_identity",
            "frozen_message_interval",
            "mode80_coordinate_value_alignment",
            "missing_value_preservation",
            "central_park_nearest_cell",
            "saved_panel_temperature",
        ],
        "negative_naive_reshape": {
            "row": 859,
            "column": 1811,
            "aligned_temperature_k": previous.temperature_k,
            "naive_temperature_k": naive_odd,
        },
        "extraction": extraction.to_dict(),
        "limitations": [
            "This checks one retained NOAA bulletin and its frozen message 2.",
            "Archive timestamps bound availability but do not prove local receipt.",
            "The daytime MaxT interval does not equal the full settlement-day window.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = verify(args.source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"passed": True, "evidence": _relative(args.output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
