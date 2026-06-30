from __future__ import annotations

import math
from datetime import UTC, datetime
from pathlib import Path

import pytest
from eccodes import codes_get_values, codes_grib_new_from_file, codes_release

from kalshi_information_quality_lab.ndfd import (
    GeographicCoordinate,
    GridCoordinate,
    NdfdDecodeError,
    extract_ndfd_max_temperature,
    find_ndfd_max_temperature_message,
)

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = (
    ROOT
    / "outputs/historical-study/raw"
    / "7b84db4a743fb287b2ad6cc31fa6e04523500a7a06024e196af94394835e1c1a.body"
)
SOURCE_SHA256 = "4f4887a39454f5bd09a21f41314dd802805cc25c2f563417aa6034b411738494"


@pytest.fixture(scope="module")
def sample_path() -> Path:
    if not SAMPLE.is_file():
        pytest.skip("private NOAA study cache is not present in this checkout")
    return SAMPLE


@pytest.fixture(scope="module")
def extraction(sample_path: Path):
    return extract_ndfd_max_temperature(
        sample_path,
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


def test_primary_source_message_metadata_missingness_and_units(extraction) -> None:
    assert extraction.data_origin == "public_noaa_source_sample"
    assert extraction.source_sha256 == SOURCE_SHA256
    assert extraction.message_number == 2
    assert extraction.reference_at.isoformat() == "2025-05-31T16:30:00+00:00"
    assert extraction.valid_start_at.isoformat() == "2025-06-01T12:00:00+00:00"
    assert extraction.valid_end_at.isoformat() == "2025-06-02T00:00:00+00:00"
    assert extraction.forecast_start_hours == 19.5
    assert extraction.interval_hours == 12
    assert extraction.source_units == "K"
    assert extraction.height_above_ground_m == 2
    assert extraction.missing_value == 9999
    assert extraction.missing_count == 1_479_351
    assert extraction.grid.rows == 1377
    assert extraction.grid.columns == 2145
    assert extraction.grid.scanning_mode == 80
    assert extraction.grid.alternative_row_scanning is True

    previous, central_park, following, missing, nearest = extraction.points
    assert (previous.row, previous.column, previous.temperature_k) == (859, 1811, 293.7)
    assert (central_park.row, central_park.column, central_park.temperature_k) == (
        860,
        1811,
        293.7,
    )
    assert central_park.temperature_f == pytest.approx(68.99)
    assert (following.row, following.column, following.temperature_k) == (861, 1811, 294.3)
    assert missing.missing is True
    assert missing.temperature_k is None
    assert missing.temperature_f is None
    assert nearest.temperature_k == central_park.temperature_k


def test_mode_80_triplets_reject_naive_odd_row_reshape(sample_path: Path) -> None:
    aligned = extract_ndfd_max_temperature(
        sample_path,
        message_number=2,
        grid_coordinates=(GridCoordinate(row=859, column=1811),),
    ).points[0]
    gid = None
    try:
        with sample_path.open("rb") as stream:
            for _ in range(2):
                if gid is not None:
                    codes_release(gid)
                gid = codes_grib_new_from_file(stream)
        assert gid is not None
        naive = codes_get_values(gid).reshape(1377, 2145)[859, 1811]
    finally:
        if gid is not None:
            codes_release(gid)

    assert aligned.temperature_k == 293.7
    assert naive == 307.6
    assert aligned.temperature_k != naive


def test_geographic_nearest_matches_saved_central_park_cell(extraction) -> None:
    nearest = extraction.points[-1]
    assert nearest.extraction_method == "nearest_geographic"
    assert (nearest.row, nearest.column) == (860, 1811)
    assert nearest.requested_latitude == pytest.approx(40.7789)
    assert nearest.requested_longitude == pytest.approx(-73.9692)
    assert nearest.distance_km == pytest.approx(.3665548851681874)
    assert nearest.temperature_k == 293.7
    assert nearest.temperature_f == pytest.approx(68.99)

    serialized = extraction.to_dict()
    assert serialized["reference_at"].endswith("Z")
    assert serialized["points"][-1]["temperature_k"] == 293.7
    assert "availability" not in serialized


def test_real_message_selection_uses_target_interval_and_issue_boundary(sample_path: Path) -> None:
    selected = find_ndfd_max_temperature_message(
        sample_path,
        valid_start_at=datetime(2025, 6, 1, 12, tzinfo=UTC),
        valid_end_at=datetime(2025, 6, 2, tzinfo=UTC),
        issued_by=datetime(2025, 5, 31, 16, 30, tzinfo=UTC),
    )
    assert selected == 2


@pytest.mark.parametrize(
    ("valid_start_at", "valid_end_at", "issued_by"),
    [
        (
            datetime(2025, 6, 4, 12, tzinfo=UTC),
            datetime(2025, 6, 5, tzinfo=UTC),
            datetime(2025, 5, 31, 16, 30, tzinfo=UTC),
        ),
        (
            datetime(2025, 6, 1, 12, tzinfo=UTC),
            datetime(2025, 6, 2, tzinfo=UTC),
            datetime(2025, 5, 31, 16, 29, tzinfo=UTC),
        ),
    ],
)
def test_real_message_selection_rejects_absent_target_or_future_issue(
    sample_path: Path,
    valid_start_at: datetime,
    valid_end_at: datetime,
    issued_by: datetime,
) -> None:
    with pytest.raises(NdfdDecodeError, match="no NDFD MaxT message matches"):
        find_ndfd_max_temperature_message(
            sample_path,
            valid_start_at=valid_start_at,
            valid_end_at=valid_end_at,
            issued_by=issued_by,
        )


def test_bounds_and_request_types_fail_closed(sample_path: Path) -> None:
    with pytest.raises(NdfdDecodeError, match="outside"):
        extract_ndfd_max_temperature(
            sample_path,
            message_number=2,
            grid_coordinates=(GridCoordinate(row=1377, column=0),),
        )
    with pytest.raises(NdfdDecodeError, match="requested latitude"):
        extract_ndfd_max_temperature(
            sample_path,
            message_number=2,
            geographic_coordinates=(GeographicCoordinate(latitude=math.nan, longitude=0),),
        )


@pytest.mark.parametrize("message_number", [0, -1, True])
def test_message_number_is_explicitly_positive(message_number: int) -> None:
    with pytest.raises(NdfdDecodeError, match="message_number"):
        extract_ndfd_max_temperature(SAMPLE, message_number=message_number)
