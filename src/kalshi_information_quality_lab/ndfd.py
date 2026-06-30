"""Decode aligned point values from NOAA NDFD maximum-temperature GRIB2 bulletins.

This module validates a deliberately narrow production path: one explicitly selected
12-hour maximum-temperature message on the NDFD Lambert grid.  The returned source
metadata describes a NOAA source sample only.  It does not infer historical availability,
select a settlement station, or equate the NDFD daytime interval with a market target.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
from eccodes import (
    CodesInternalError,
    codes_get,
    codes_get_array,
    codes_get_long,
    codes_new_from_message,
    codes_release,
)

_MAX_BULLETIN_BYTES = 64 * 1024 * 1024
_EARTH_RADIUS_KM = 6_371.2


class NdfdDecodeError(ValueError):
    """The input cannot be used by the supported NDFD MaxT extraction path."""


@dataclass(frozen=True, kw_only=True)
class GridCoordinate:
    """A zero-based row and column in ecCodes' canonical geographic grid order."""

    row: int
    column: int


@dataclass(frozen=True, kw_only=True)
class GeographicCoordinate:
    """A geographic point used only to select the nearest NDFD grid cell."""

    latitude: float
    longitude: float


@dataclass(frozen=True, kw_only=True)
class NdfdGrid:
    rows: int
    columns: int
    point_count: int
    grid_type: str
    grid_template: int
    scanning_mode: int
    alternative_row_scanning: bool
    i_scans_negatively: bool
    j_scans_positively: bool


@dataclass(frozen=True, kw_only=True)
class NdfdPoint:
    extraction_method: str
    row: int
    column: int
    latitude: float
    longitude: float
    source_longitude_degrees_east: float
    missing: bool
    temperature_k: float | None
    temperature_f: float | None
    requested_latitude: float | None = None
    requested_longitude: float | None = None
    distance_km: float | None = None


@dataclass(frozen=True, kw_only=True)
class NdfdMaxTemperatureExtraction:
    """One decoded NOAA source message with explicitly requested point extractions."""

    schema_version: str
    data_origin: str
    research_scope: str
    source_name: str
    source_sha256: str
    message_number: int
    reference_at: datetime
    valid_start_at: datetime
    valid_end_at: datetime
    forecast_start_hours: float
    interval_hours: int
    variable: str
    source_units: str
    height_above_ground_m: float
    missing_value: float
    missing_count: int
    grid: NdfdGrid
    points: tuple[NdfdPoint, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a stable JSON-ready representation for notebooks and evidence scripts."""
        result = asdict(self)
        for key in ("reference_at", "valid_start_at", "valid_end_at"):
            result[key] = getattr(self, key).isoformat().replace("+00:00", "Z")
        return result


def _whole_number(value: object, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise NdfdDecodeError(f"{name} must be an integer greater than or equal to {minimum}")
    return value


def _geographic_coordinate(value: GeographicCoordinate) -> tuple[float, float]:
    if not isinstance(value, GeographicCoordinate):
        raise NdfdDecodeError("geographic requests must be GeographicCoordinate values")
    latitude = value.latitude
    longitude = value.longitude
    if (
        isinstance(latitude, bool)
        or not isinstance(latitude, int | float)
        or not math.isfinite(latitude)
        or not -90 <= latitude <= 90
    ):
        raise NdfdDecodeError("requested latitude must be finite and within [-90, 90]")
    if (
        isinstance(longitude, bool)
        or not isinstance(longitude, int | float)
        or not math.isfinite(longitude)
        or not -180 <= longitude <= 180
    ):
        raise NdfdDecodeError("requested longitude must be finite and within [-180, 180]")
    return float(latitude), float(longitude)


def _utc_datetime(date_value: int, time_value: int, label: str) -> datetime:
    text = f"{date_value:08d}{time_value:04d}"
    try:
        return datetime.strptime(text, "%Y%m%d%H%M").replace(tzinfo=UTC)
    except ValueError:
        raise NdfdDecodeError(f"{label} is not a valid UTC date and time") from None


def _end_of_interval(gid: int) -> datetime:
    try:
        return datetime(
            codes_get_long(gid, "yearOfEndOfOverallTimeInterval"),
            codes_get_long(gid, "monthOfEndOfOverallTimeInterval"),
            codes_get_long(gid, "dayOfEndOfOverallTimeInterval"),
            codes_get_long(gid, "hourOfEndOfOverallTimeInterval"),
            codes_get_long(gid, "minuteOfEndOfOverallTimeInterval"),
            codes_get_long(gid, "secondOfEndOfOverallTimeInterval"),
            tzinfo=UTC,
        )
    except ValueError:
        raise NdfdDecodeError("message has an invalid end-of-interval timestamp") from None


def _validate_message(gid: int) -> tuple[NdfdGrid, datetime, datetime, datetime, float, int]:
    expected_long = {
        "edition": 2,
        "parameterCategory": 0,
        "parameterNumber": 4,
        "productDefinitionTemplateNumber": 8,
        "typeOfStatisticalProcessing": 2,
        "lengthOfTimeRange": 12,
        "indicatorOfUnitForTimeRange": 1,
        "typeOfFirstFixedSurface": 103,
        "scaleFactorOfFirstFixedSurface": 0,
        "scaledValueOfFirstFixedSurface": 2,
        "gridDefinitionTemplateNumber": 30,
    }
    for key, expected in expected_long.items():
        actual = codes_get_long(gid, key)
        if actual != expected:
            raise NdfdDecodeError(
                f"unsupported NDFD MaxT message: {key} is {actual}, expected {expected}"
            )
    if codes_get(gid, "units") != "K":
        raise NdfdDecodeError("unsupported NDFD MaxT message: source units must be K")
    if codes_get(gid, "gridType") != "lambert":
        raise NdfdDecodeError("unsupported NDFD MaxT message: grid must be Lambert conformal")

    columns = codes_get_long(gid, "Ni")
    rows = codes_get_long(gid, "Nj")
    point_count = codes_get_long(gid, "numberOfPoints")
    if rows <= 0 or columns <= 0 or rows * columns != point_count:
        raise NdfdDecodeError("message does not contain a complete rectangular NDFD grid")
    scanning_mode = codes_get_long(gid, "scanningMode")
    if scanning_mode not in {64, 80}:
        raise NdfdDecodeError(
            f"unsupported NDFD MaxT message: scanning mode {scanning_mode} is not 64 or 80"
        )

    reference_at = _utc_datetime(
        codes_get_long(gid, "dataDate"), codes_get_long(gid, "dataTime"), "reference time"
    )
    forecast_unit = codes_get_long(gid, "indicatorOfUnitOfTimeRange")
    if forecast_unit not in {0, 1}:
        raise NdfdDecodeError(
            f"unsupported NDFD MaxT message: forecast time unit {forecast_unit} "
            "is not minutes (0) or hours (1)"
        )
    forecast_time = codes_get_long(gid, "forecastTime")
    if forecast_time < 0:
        raise NdfdDecodeError("forecast interval start cannot precede the reference time")
    # afternoon bulletins use minute offsets; integer arithmetic keeps mismatches visible
    forecast_minutes = forecast_time * (60 if forecast_unit == 1 else 1)
    forecast_start_hours = forecast_minutes / 60
    valid_start_at = reference_at + timedelta(minutes=forecast_minutes)
    valid_end_at = _end_of_interval(gid)
    interval_hours = codes_get_long(gid, "lengthOfTimeRange")
    if valid_end_at - valid_start_at != timedelta(hours=interval_hours):
        raise NdfdDecodeError("message timestamps do not describe the declared 12-hour interval")

    grid = NdfdGrid(
        rows=rows,
        columns=columns,
        point_count=point_count,
        grid_type="lambert",
        grid_template=30,
        scanning_mode=scanning_mode,
        alternative_row_scanning=bool(codes_get_long(gid, "alternativeRowScanning")),
        i_scans_negatively=bool(codes_get_long(gid, "iScansNegatively")),
        j_scans_positively=bool(codes_get_long(gid, "jScansPositively")),
    )
    return grid, reference_at, valid_start_at, valid_end_at, forecast_start_hours, interval_hours


def _signed_longitude(longitude_degrees_east: float) -> float:
    return (longitude_degrees_east + 180) % 360 - 180


def _point(
    triplets: np.ndarray,
    missing_value: float,
    row: int,
    column: int,
    *,
    method: str,
    requested_latitude: float | None = None,
    requested_longitude: float | None = None,
    distance_km: float | None = None,
) -> NdfdPoint:
    latitude, source_longitude, value = (float(item) for item in triplets[row, column])
    missing = value == missing_value
    if not missing and not math.isfinite(value):
        raise NdfdDecodeError("message contains a non-finite nonmissing temperature")
    value_k = None if missing else value
    value_f = None if missing else (value - 273.15) * 9 / 5 + 32
    return NdfdPoint(
        extraction_method=method,
        row=row,
        column=column,
        latitude=latitude,
        longitude=_signed_longitude(source_longitude),
        source_longitude_degrees_east=source_longitude,
        missing=missing,
        temperature_k=value_k,
        temperature_f=value_f,
        requested_latitude=requested_latitude,
        requested_longitude=requested_longitude,
        distance_km=distance_km,
    )


def _nearest_cell(
    triplets: np.ndarray, latitude: float, longitude: float
) -> tuple[int, int, float]:
    latitudes = triplets[:, :, 0]
    longitudes = triplets[:, :, 1]
    latitude_radians = np.radians(latitudes)
    delta_latitude = latitude_radians - math.radians(latitude)
    delta_longitude_degrees = (longitudes - longitude + 180) % 360 - 180
    delta_longitude = np.radians(delta_longitude_degrees)
    haversine = np.sin(delta_latitude / 2) ** 2 + (
        math.cos(math.radians(latitude))
        * np.cos(latitude_radians)
        * np.sin(delta_longitude / 2) ** 2
    )
    flat_index = int(np.argmin(haversine))
    row, column = (int(item) for item in np.unravel_index(flat_index, haversine.shape))
    selected_haversine = min(1, max(0, float(haversine[row, column])))
    distance = 2 * _EARTH_RADIUS_KM * math.asin(math.sqrt(selected_haversine))
    if distance < 1e-9:
        distance = 0.0
    return row, column, distance


def _source_content(path: Path) -> tuple[bytes, str]:
    try:
        stat = path.stat()
        if not path.is_file() or stat.st_size <= 0 or stat.st_size > _MAX_BULLETIN_BYTES:
            raise NdfdDecodeError("source must be a nonempty regular file no larger than 64 MiB")
        content = path.read_bytes()
        if len(content) != stat.st_size:
            raise NdfdDecodeError("source changed while it was being read")
        return content, hashlib.sha256(content).hexdigest()
    except NdfdDecodeError:
        raise
    except OSError:
        raise NdfdDecodeError("source is not a readable GRIB2 bulletin") from None


def _grib_messages(content: bytes) -> Iterator[bytes]:
    cursor = 0
    while cursor < len(content):
        start = content.find(b"GRIB", cursor)
        if start < 0:
            return
        if start + 16 > len(content) or content[start + 7] != 2:
            raise NdfdDecodeError("source contains a malformed or unsupported GRIB message header")
        length = int.from_bytes(content[start + 8 : start + 16], byteorder="big")
        end = start + length
        if length < 20 or end > len(content) or content[end - 4 : end] != b"7777":
            raise NdfdDecodeError("source contains a truncated or malformed GRIB message")
        yield content[start:end]
        cursor = end


def _selected_grib_message(content: bytes, message_number: int) -> bytes:
    for current, message in enumerate(_grib_messages(content), start=1):
        if current == message_number:
            return message
    raise NdfdDecodeError(f"source has no GRIB message {message_number}")


def find_ndfd_max_temperature_message(
    path: Path,
    *,
    valid_start_at: datetime,
    valid_end_at: datetime,
    issued_by: datetime,
) -> int:
    """Find a unique eligible target interval using headers without decoding grids.

    The issue cutoff tests the GRIB reference time only. The caller must separately
    establish publication availability and enforce its historical as-of policy.
    """
    for label, value in (
        ("valid_start_at", valid_start_at),
        ("valid_end_at", valid_end_at),
        ("issued_by", issued_by),
    ):
        if not isinstance(value, datetime) or value.utcoffset() is None:
            raise NdfdDecodeError(f"{label} must be a timezone-aware datetime")
    if valid_end_at - valid_start_at != timedelta(hours=12):
        raise NdfdDecodeError("requested target must be a 12-hour interval")

    content, _source_sha256 = _source_content(Path(path))
    matches: list[int] = []
    try:
        for number, message in enumerate(_grib_messages(content), start=1):
            gid = codes_new_from_message(message)
            if gid is None:
                raise NdfdDecodeError("source contains an unreadable GRIB message")
            try:
                # current-day fields can cover a period that has already started
                if _end_of_interval(gid) != valid_end_at:
                    continue
                _grid, reference, start, end, _forecast_hours, _interval_hours = _validate_message(
                    gid
                )
                if start == valid_start_at and end == valid_end_at and reference <= issued_by:
                    matches.append(number)
            finally:
                codes_release(gid)
    except NdfdDecodeError:
        raise
    except (CodesInternalError, OSError, OverflowError, TypeError, ValueError):
        raise NdfdDecodeError("source is malformed or not a supported GRIB2 bulletin") from None

    if not matches:
        raise NdfdDecodeError(
            "no NDFD MaxT message matches the target interval before the issue cutoff"
        )
    if len(matches) > 1:
        raise NdfdDecodeError(
            "multiple NDFD MaxT messages match the target interval and issue cutoff"
        )
    return matches[0]


def extract_ndfd_max_temperature(
    path: Path,
    *,
    message_number: int,
    grid_coordinates: tuple[GridCoordinate, ...] = (),
    geographic_coordinates: tuple[GeographicCoordinate, ...] = (),
) -> NdfdMaxTemperatureExtraction:
    """Decode one supported message and return aligned requested cell values.

    `message_number` is one-based. Rows and columns are zero-based in the canonical
    `latLonValues` ordering returned by ecCodes. This is critical for NDFD scan mode 80:
    reshaping `codes_get_values` directly reverses the association on alternating rows.
    """
    selected_message = _whole_number(message_number, "message_number", minimum=1)
    source = Path(path)
    content, source_sha256 = _source_content(source)
    coordinates = tuple(grid_coordinates)
    requested_geographic = tuple(geographic_coordinates)
    gid: int | None = None
    try:
        message = _selected_grib_message(content, selected_message)
        gid = codes_new_from_message(message)
        assert gid is not None
        grid, reference_at, valid_start_at, valid_end_at, forecast_hours, interval_hours = (
            _validate_message(gid)
        )

        for coordinate in coordinates:
            if not isinstance(coordinate, GridCoordinate):
                raise NdfdDecodeError("grid requests must be GridCoordinate values")
            row = _whole_number(coordinate.row, "grid row")
            column = _whole_number(coordinate.column, "grid column")
            if row >= grid.rows or column >= grid.columns:
                raise NdfdDecodeError(
                    f"grid coordinate ({row}, {column}) is outside {grid.rows}x{grid.columns}"
                )
        validated_geographic = tuple(
            _geographic_coordinate(coordinate) for coordinate in requested_geographic
        )

        raw_triplets = np.asarray(codes_get_array(gid, "latLonValues"), dtype=np.float64)
        if raw_triplets.size != grid.point_count * 3:
            raise NdfdDecodeError(
                "decoder did not return one coordinate/value triple per grid cell"
            )
        triplets = raw_triplets.reshape(grid.rows, grid.columns, 3)
        missing_value = float(codes_get(gid, "missingValue"))
        missing_count = int(np.count_nonzero(triplets[:, :, 2] == missing_value))
        declared_missing = codes_get_long(gid, "numberOfMissing")
        if missing_count != declared_missing:
            raise NdfdDecodeError(
                "decoded missing-value mask does not match the GRIB missing-value count"
            )

        points = [
            _point(
                triplets,
                missing_value,
                coordinate.row,
                coordinate.column,
                method="grid_coordinate",
            )
            for coordinate in coordinates
        ]
        for _coordinate, (latitude, longitude) in zip(
            requested_geographic, validated_geographic, strict=True
        ):
            row, column, distance = _nearest_cell(triplets, latitude, longitude)
            points.append(
                _point(
                    triplets,
                    missing_value,
                    row,
                    column,
                    method="nearest_geographic",
                    requested_latitude=latitude,
                    requested_longitude=longitude,
                    distance_km=distance,
                )
            )

        return NdfdMaxTemperatureExtraction(
            schema_version="ndfd_max_temperature_extraction_v1",
            data_origin="public_noaa_source_sample",
            research_scope=(
                "decoder validation only; no Kalshi station/window mapping, historical coverage, "
                "availability time, or empirical forecast comparison"
            ),
            source_name=source.name,
            source_sha256=source_sha256,
            message_number=selected_message,
            reference_at=reference_at,
            valid_start_at=valid_start_at,
            valid_end_at=valid_end_at,
            forecast_start_hours=forecast_hours,
            interval_hours=interval_hours,
            variable="maximum_temperature",
            source_units="K",
            height_above_ground_m=2.0,
            missing_value=missing_value,
            missing_count=missing_count,
            grid=grid,
            points=tuple(points),
        )
    except NdfdDecodeError:
        raise
    except (CodesInternalError, OSError, OverflowError, TypeError, ValueError):
        raise NdfdDecodeError(
            "source is malformed, truncated, or not a supported GRIB2 bulletin"
        ) from None
    finally:
        if gid is not None:
            codes_release(gid)
