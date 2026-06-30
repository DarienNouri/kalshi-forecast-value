"""A bounded, resumable historical weather study over exact saved source bodies."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import xml.etree.ElementTree as ET
from collections.abc import Callable
from datetime import UTC, date, datetime, time, timedelta
from itertools import pairwise
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlencode

from .config import StudyConfig
from .contracts import EvidenceReference, SourceUseScope
from .domain import DataOrigin
from .historical_inputs import validate_protocol
from .historical_runs import (
    HistoricalRunError,
    collection_input_identity,
    guard_collection_run,
    guard_panel_publication,
    protocol_identity,
    write_collection_manifest,
)
from .ndfd import (
    GeographicCoordinate,
    NdfdDecodeError,
    extract_ndfd_max_temperature,
    find_ndfd_max_temperature_message,
)
from .sources import DocumentRequest, FetchedDocument, SourceError, fetch_document
from .storage import StorageError, publish_immutable_file

API = "https://external-api.kalshi.com/trade-api/v2"
BUCKET = "https://noaa-ndfd-pds.s3.amazonaws.com"


class HistoricalError(ValueError):
    """Invalid study input or saved evidence."""


def _sha(body: bytes) -> str:
    return hashlib.sha256(body).hexdigest()


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".pending")
    with temporary.open("wb") as stream:
        stream.write(_json_bytes(value))
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _instant(value: str) -> datetime:
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is None:
        raise HistoricalError("timezone_required")
    return result.astimezone(UTC)


class SourceCache:
    """Successful GET snapshots keyed by URL, with verified bytes on every replay."""

    def __init__(
        self,
        root: Path,
        output: Path,
        *,
        cache_dir: Path | None = None,
        offline: bool = False,
        evidence_reference: str = "docs/SOURCES.md",
        fetcher: Callable[..., FetchedDocument] = fetch_document,
    ) -> None:
        self.root = root
        self.directory = cache_dir if cache_dir is not None else output / "raw"
        if cache_dir is None or not offline:
            self.directory.mkdir(parents=True, exist_ok=True)
        elif not self.directory.is_dir():
            raise HistoricalError("offline_cache_directory_missing")
        self.offline = offline
        self.fetcher = fetcher
        self.fetched = 0
        self.replayed = 0
        self.evidence_reference = evidence_reference
        self.scope_hash = _sha((root / evidence_reference).read_bytes())

    def get(
        self, source: str, url: str, *, format: Literal["json", "text", "grib2"] = "json"
    ) -> tuple[Path, dict]:
        key = _sha(url.encode())
        body_path = self.directory / f"{key}.body"
        metadata_path = self.directory / f"{key}.json"
        if metadata_path.exists():
            try:
                metadata = json.loads(metadata_path.read_bytes())
                body = body_path.read_bytes()
                valid = (
                    metadata["source_uri"] == url
                    and metadata["source_name"] == source
                    and metadata["content_sha256"] == _sha(body)
                    and metadata["bytes"] == len(body)
                )
            except (OSError, json.JSONDecodeError, UnicodeDecodeError, KeyError, TypeError):
                valid = False
            if not valid:
                raise HistoricalError("cached_source_integrity_failure")
            self.replayed += 1
            return body_path, metadata
        if self.offline:
            raise HistoricalError("offline_source_missing")
        scope = SourceUseScope(
            source_name=source,
            origin=DataOrigin.EMPIRICAL,
            allowed_uses=("collection", "retention", "modeling", "local_artifacts"),
            reviewed_at=datetime(2026, 9, 14, tzinfo=UTC),
            evidence=(
                EvidenceReference(path=self.evidence_reference, content_sha256=self.scope_hash),
            ),
        )
        if format not in {"json", "text", "grib2"}:
            raise HistoricalError("invalid_source_format")
        request = DocumentRequest(
            source_name=source,
            url=url,
            origin=DataOrigin.EMPIRICAL,
            format=format,
            max_bytes=16 * 1024 * 1024,
            timeout_seconds=30,
        )
        document = self.fetcher(request, project_root=self.root, scope=scope)
        try:
            publish_immutable_file(self.directory, body_path.name, document.body)
        except StorageError:
            raise HistoricalError("uncommitted_source_conflict") from None
        metadata = document.to_dict()
        _write_json(metadata_path, metadata)
        self.fetched += 1
        return body_path, metadata

    def json(self, source: str, url: str) -> tuple[Any, dict]:
        path, metadata = self.get(source, url)
        return json.loads(path.read_bytes()), metadata


def normalize_markets(markets: list[dict], event_id: str) -> tuple[list[dict], float, str]:
    """Reconcile integer outcome predicates and archive rules before reading prices."""
    if not markets or len({m["ticker"] for m in markets}) != len(markets):
        raise HistoricalError("missing_or_duplicate_markets")
    bins: list[dict] = []
    outcomes: set[float] = set()
    settlements: list[datetime] = []
    for market in markets:
        rules = market.get("rules_primary", "")
        if (
            market["event_ticker"] != event_id
            or "Central Park, New York" not in rules
            or "National Weather Service's Climatological Report (Daily)" not in rules
        ):
            raise HistoricalError("unmatched_historical_rules")
        lower, upper = market.get("floor_strike"), market.get("cap_strike")
        for boundary in (lower, upper):
            if boundary is not None and (type(boundary) is not int):
                raise HistoricalError("noninteger_bin_boundary")
        kind = market["strike_type"]
        if kind == "less" and lower is None and upper is not None:
            upper -= 1
        elif kind == "greater" and lower is not None and upper is None:
            lower += 1
        elif kind != "between" or lower is None or upper is None:
            raise HistoricalError("unsupported_bin_predicate")
        if lower is not None and upper is not None and lower > upper:
            raise HistoricalError("invalid_bin_bounds")
        bins.append({"ticker": market["ticker"], "lower": lower, "upper": upper})
        value = float(market["expiration_value"])
        if not math.isfinite(value) or not value.is_integer():
            raise HistoricalError("invalid_settlement_temperature")
        outcomes.add(value)
        settlements.append(_instant(market["settlement_ts"]))
    bins.sort(key=lambda b: -math.inf if b["lower"] is None else b["lower"])
    if bins[0]["lower"] is not None or bins[-1]["upper"] is not None:
        raise HistoricalError("incomplete_bin_tails")
    for left, right in pairwise(bins):
        if left["upper"] is None or right["lower"] != left["upper"] + 1:
            raise HistoricalError("bin_gap_or_overlap")
    if len(outcomes) != 1:
        raise HistoricalError("conflicting_settlement_values")
    outcome = next(iter(outcomes))
    winning = [
        b["ticker"]
        for b in bins
        if (b["lower"] is None or outcome >= b["lower"])
        and (b["upper"] is None or outcome <= b["upper"])
    ]
    reported = [m["ticker"] for m in markets if m.get("result") == "yes"]
    if winning != reported or any(m.get("result") not in {"yes", "no"} for m in markets):
        raise HistoricalError("settlement_result_mismatch")
    return bins, outcome, max(settlements).isoformat()


def market_at(candles: list[dict], as_of: datetime, max_age_minutes: int) -> dict:
    available = [c for c in candles if c["end_period_ts"] <= as_of.timestamp()]
    if not available:
        raise HistoricalError("missing_market_candle")
    candle = max(available, key=lambda c: c["end_period_ts"])
    age = as_of.timestamp() - candle["end_period_ts"]
    if age > max_age_minutes * 60:
        raise HistoricalError("stale_market_candle")
    bid, ask = candle["yes_bid"]["close"], candle["yes_ask"]["close"]
    if bid is None or ask is None:
        raise HistoricalError("missing_bid_ask_close")
    bid_f, ask_f = float(bid), float(ask)
    if not 0 <= bid_f <= ask_f <= 1:
        raise HistoricalError("invalid_bid_ask_close")
    return {
        "probability": (bid_f + ask_f) / 2,
        "bid_dollars": bid,
        "ask_dollars": ask,
        "candle_end_at": datetime.fromtimestamp(candle["end_period_ts"], UTC).isoformat(),
        "candle_age_minutes": age / 60,
        "spread": ask_f - bid_f,
    }


def eligible_noaa_objects(xml: bytes, as_of: datetime, protocol: dict) -> list[dict]:
    namespace = {"s": "http://s3.amazonaws.com/doc/2006-03-01/"}
    document = ET.fromstring(xml)
    if document.findtext("s:IsTruncated", namespaces=namespace) == "true":
        raise HistoricalError("truncated_noaa_listing")
    candidates = []
    for item in document.findall("s:Contents", namespace):
        key = item.findtext("s:Key", namespaces=namespace) or ""
        match = re.search(r"/YGUZ98_KWBN_(\d{12})$", key)
        if match is None:
            continue
        nominal = datetime.strptime(match[1], "%Y%m%d%H%M").replace(tzinfo=UTC)
        modified = _instant(item.findtext("s:LastModified", namespaces=namespace) or "")
        if (
            nominal + timedelta(minutes=protocol["forecast_min_lead_minutes"]) <= as_of
            and modified <= as_of
            and as_of - nominal <= timedelta(hours=protocol["forecast_max_age_hours"])
        ):
            candidates.append(
                {
                    "key": key,
                    "last_modified_at": modified.isoformat(),
                    "nominal_at": nominal.isoformat(),
                    "bytes": int(item.findtext("s:Size", namespaces=namespace) or 0),
                }
            )
    return sorted(candidates, key=lambda x: x["nominal_at"], reverse=True)


def _forecast(cache: SourceCache, day: date, as_of: datetime, protocol: dict) -> dict:
    prefix = f"wmo/maxt/{as_of:%Y/%m/%d}/"
    listing_url = BUCKET + "/?" + urlencode({"list-type": 2, "prefix": prefix, "max-keys": 1000})
    listing, listing_meta = cache.get("noaa", listing_url, format="text")
    candidates = eligible_noaa_objects(listing.read_bytes(), as_of, protocol)
    start = datetime.combine(day, time(protocol["forecast_valid_start_utc_hour"]), UTC)
    end = start + timedelta(hours=protocol["forecast_valid_hours"])
    rejected = []
    for candidate in candidates[:3]:
        body, metadata = cache.get("noaa", BUCKET + "/" + candidate["key"], format="grib2")
        try:
            message = find_ndfd_max_temperature_message(
                body, valid_start_at=start, valid_end_at=end, issued_by=as_of
            )
            extraction = extract_ndfd_max_temperature(
                body,
                message_number=message,
                geographic_coordinates=(
                    GeographicCoordinate(
                        latitude=protocol["station_latitude"],
                        longitude=protocol["station_longitude"],
                    ),
                ),
            )
            point = extraction.points[0]
            if point.missing or point.temperature_f is None:
                raise HistoricalError("missing_forecast_grid_point")
            return {
                "forecast_f": point.temperature_f,
                "issued_at": extraction.reference_at.isoformat(),
                "valid_start_at": extraction.valid_start_at.isoformat(),
                "valid_end_at": extraction.valid_end_at.isoformat(),
                "available_at": max(
                    _instant(candidate["last_modified_at"]),
                    _instant(candidate["nominal_at"])
                    + timedelta(minutes=protocol["forecast_min_lead_minutes"]),
                    extraction.reference_at,
                ).isoformat(),
                "availability_basis": "historical_archive_bound",
                "object": candidate,
                "message_number": message,
                "grid_row": point.row,
                "grid_column": point.column,
                "grid_distance_km": point.distance_km,
                "source_sha256": metadata["content_sha256"],
                "source_uri": metadata["source_uri"],
                "listing_sha256": listing_meta["content_sha256"],
                "rejected_vintages": rejected,
            }
        except NdfdDecodeError as exc:
            rejected.append({"key": candidate["key"], "reason": str(exc)})
    raise HistoricalError("missing_eligible_noaa_forecast")


def _market_pages(cache: SourceCache, event_id: str) -> tuple[list[dict], list[dict]]:
    markets, evidence = [], []
    for partition in ("/historical/markets", "/markets"):
        cursor = ""
        seen = set()
        for _ in range(10):
            query = {"event_ticker": event_id, "limit": 100}
            if cursor:
                query["cursor"] = cursor
            response, metadata = cache.json("kalshi", API + partition + "?" + urlencode(query))
            markets.extend(response["markets"])
            evidence.append(metadata)
            cursor = response.get("cursor", "")
            if not cursor:
                break
            if cursor in seen:
                raise HistoricalError("repeated_market_cursor")
            seen.add(cursor)
        else:
            raise HistoricalError("market_page_limit")
        if markets:
            break
    return markets, evidence


def collect_day(cache: SourceCache, day: date, protocol: dict, cutoff: datetime) -> dict:
    event_id = protocol["series_ticker"] + "-" + day.strftime("%y%b%d").upper()
    start = datetime.combine(day, time(protocol["outcome_window_start_utc_hour"]), UTC)
    cases, exclusions = [], []
    markets, evidence = _market_pages(cache, event_id)
    try:
        bins, outcome, settled = normalize_markets(markets, event_id)
    except (HistoricalError, KeyError, ValueError) as exc:
        return {
            "event_id": event_id,
            "cases": [],
            "exclusions": [
                {
                    "event_id": event_id,
                    "outcome_date": day.isoformat(),
                    "horizon_hours": h,
                    "reason": str(exc),
                }
                for h in protocol["horizon_hours"]
            ],
        }
    observations, candle_metadata = {}, {}
    opened = max(_instant(m["open_time"]) for m in markets)
    first = start - timedelta(hours=max(protocol["horizon_hours"]) + 3)
    last = start - timedelta(hours=min(protocol["horizon_hours"]))
    for market in markets:
        ticker = market["ticker"]
        route = (
            f"/historical/markets/{ticker}/candlesticks"
            if _instant(market["settlement_ts"]) < cutoff
            else f"/series/{protocol['series_ticker']}/markets/{ticker}/candlesticks"
        )
        query = urlencode(
            {
                "start_ts": int(first.timestamp()),
                "end_ts": int(last.timestamp()),
                "period_interval": protocol["market_candle_interval_minutes"],
            }
        )
        response, metadata = cache.json("kalshi", API + route + "?" + query)
        observations[ticker] = response["candlesticks"]
        candle_metadata[ticker] = metadata["content_sha256"]
    for horizon in protocol["horizon_hours"]:
        as_of = start - timedelta(hours=horizon)
        try:
            if opened > as_of:
                raise HistoricalError("market_not_open_at_checkpoint")
            vector = [
                market_at(
                    observations[b["ticker"]], as_of, protocol["market_max_candle_age_minutes"]
                )
                for b in bins
            ]
            total = sum(v["probability"] for v in vector)
            if total <= 0:
                raise HistoricalError("zero_market_probability_mass")
            forecast = _forecast(cache, day, as_of, protocol)
            cases.append(
                {
                    "case_id": f"{event_id}-{horizon}h",
                    "event_id": event_id,
                    "outcome_date": day.isoformat(),
                    "horizon_hours": horizon,
                    "outcome_window_start_at": start.isoformat(),
                    "outcome_window_end_at": (start + timedelta(days=1)).isoformat(),
                    "as_of_at": as_of.isoformat(),
                    "market_open_at": opened.isoformat(),
                    "forecast_f": forecast["forecast_f"],
                    "forecast": forecast,
                    "forecast_available_at": forecast["available_at"],
                    "outcome_f": outcome,
                    "outcome_available_at": settled,
                    "bins": bins,
                    "market_probabilities": [v["probability"] / total for v in vector],
                    "market_probability_sum_before_normalization": total,
                    "market_observations": vector,
                    "market_source_sha256": candle_metadata,
                    "market_rules_source_sha256": [e["content_sha256"] for e in evidence],
                    "historical_rules_assumption": True,
                }
            )
        except (HistoricalError, SourceError, NdfdDecodeError) as exc:
            exclusions.append(
                {
                    "event_id": event_id,
                    "outcome_date": day.isoformat(),
                    "horizon_hours": horizon,
                    "reason": str(exc),
                }
            )
    return {"event_id": event_id, "cases": cases, "exclusions": exclusions}


def collect_historical_panel(
    config: StudyConfig,
    protocol: dict,
    output_dir: Path,
    *,
    cache_dir: Path | None = None,
    offline: bool = False,
    max_days: int | None = None,
    progress: Callable[[str], None] = print,
) -> dict:
    if (
        config.data_mode != "historical"
        or config.local_research_scope != "documented_local_research"
    ):
        raise HistoricalError("historical_execution_config_required")
    protocol = validate_protocol(protocol, collection=True)
    output_dir = (config.project_root / output_dir).resolve()
    if not output_dir.is_relative_to(config.project_root / "outputs"):
        raise HistoricalError("historical_output_must_be_inside_project_outputs")
    first, last = (
        date.fromisoformat(protocol["start_date"]),
        date.fromisoformat(protocol["end_date"]),
    )
    if not 0 <= (last - first).days < 366 or (max_days is not None and max_days < 1):
        raise HistoricalError("invalid_historical_date_range")
    days = (last - first).days + 1
    limit = min(days, max_days or days)
    input_sha256 = collection_input_identity(protocol)
    try:
        initial_status = guard_collection_run(
            output_dir,
            input_sha256=input_sha256,
            requested_complete=limit == days,
        )
    except HistoricalRunError as exc:
        raise HistoricalError(str(exc)) from None
    if cache_dir is not None:
        cache_dir = (config.project_root / cache_dir).resolve()
        if not cache_dir.is_relative_to(config.project_root / "outputs"):
            raise HistoricalError("historical_cache_must_be_inside_project_outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    protocol_path = output_dir / "protocol.json"
    if protocol_path.exists() and json.loads(protocol_path.read_bytes()) != protocol:
        raise HistoricalError("saved_protocol_mismatch_use_new_output_directory")
    _write_json(protocol_path, protocol)
    cache = SourceCache(
        config.project_root,
        output_dir,
        cache_dir=cache_dir,
        offline=offline,
        evidence_reference=config.evidence_reference,
    )
    cutoff_data, cutoff_meta = cache.json("kalshi", API + "/historical/cutoff")
    cutoff = _instant(cutoff_data["market_settled_ts"])
    cases, exclusions = [], []
    for offset in range(limit):
        day = first + timedelta(days=offset)
        try:
            collected = collect_day(cache, day, protocol, cutoff)
        except (HistoricalError, SourceError, NdfdDecodeError) as exc:
            collected = {
                "cases": [],
                "exclusions": [
                    {
                        "event_id": protocol["series_ticker"]
                        + "-"
                        + day.strftime("%y%b%d").upper(),
                        "outcome_date": day.isoformat(),
                        "horizon_hours": h,
                        "reason": str(exc),
                    }
                    for h in protocol["horizon_hours"]
                ],
            }
        cases.extend(collected["cases"])
        exclusions.extend(collected["exclusions"])
        _write_json(output_dir / "days" / f"{day}.json", collected)
        progress(
            f"{day}: {len(collected['cases'])} cases, {len(collected['exclusions'])} exclusions"
        )
        _write_json(
            output_dir / "progress.json",
            {
                "days_completed": offset + 1,
                "days_requested": days,
                "cases": len(cases),
                "exclusions": len(exclusions),
                "last_date": day.isoformat(),
                "fetched": cache.fetched,
                "replayed": cache.replayed,
            },
        )
    ncei_url = "https://www.ncei.noaa.gov/access/services/data/v1?" + urlencode(
        {
            "dataset": "daily-summaries",
            "stations": protocol["station_id"],
            "startDate": first.isoformat(),
            "endDate": (first + timedelta(days=limit - 1)).isoformat(),
            "format": "json",
            "units": "standard",
            "includeStationLocation": "true",
        }
    )
    ncei_path, ncei_meta = cache.get("noaa", ncei_url, format="text")
    daily = {record["DATE"]: record for record in json.loads(ncei_path.read_bytes())}
    for case in cases:
        record = daily.get(case["outcome_date"], {})
        tmax = float(record["TMAX"]) if record.get("TMAX") is not None else None
        case["ncei_tmax_f"] = tmax
        case["ncei_matches_settlement"] = tmax == case["outcome_f"] if tmax is not None else None
    panel = {
        "schema_version": "historical_panel_v1",
        "data_origin": "empirical_historical",
        "analysis_class": "exploratory_historical",
        "complete_requested_dates": limit == days,
        "protocol_sha256": _sha(_json_bytes(protocol)),
        "protocol": protocol,
        "cases": cases,
        "exclusions": exclusions,
        "provenance": {
            "cutoff": cutoff_meta,
            "ncei": ncei_meta,
            "raw_directory": str(cache.directory.relative_to(config.project_root)),
        },
        "counts": {
            "requested_dates": days,
            "processed_dates": limit,
            "cases": len(cases),
            "events": len({c["event_id"] for c in cases}),
            "exclusions": len(exclusions),
        },
    }
    panel_bytes = _json_bytes(panel)
    panel_sha256 = _sha(panel_bytes)
    try:
        guard_panel_publication(
            output_dir,
            input_sha256=input_sha256,
            proposed_panel_sha256=panel_sha256,
        )
    except HistoricalRunError as exc:
        raise HistoricalError(str(exc)) from None
    _write_json(output_dir / "panel.json", panel)
    if not (initial_status.complete and initial_status.legacy):
        write_collection_manifest(
            output_dir,
            protocol_sha256=protocol_identity(protocol),
            input_sha256=input_sha256,
            panel_sha256=panel_sha256,
            complete=limit == days,
        )
    return panel
