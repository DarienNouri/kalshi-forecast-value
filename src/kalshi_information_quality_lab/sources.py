"""Bounded read-only source documents; no station inference or availability guess.

The HTTPS transport requires reviewed, hash-checked local collection and retention
scope. Scope validation records consistency, not independent legal permission.

Kalshi routes follow docs.kalshi.com/getting_started/historical_data and the market
candlestick references. NOAA's public bucket is registry.opendata.aws/noaa-ndfd/.
Raw JSON strings/nulls survive decoding; NDFD interpretation belongs to ndfd.py.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import UTC, date, datetime
from decimal import Decimal
from email.utils import parsedate_to_datetime
from http.client import HTTPException
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qsl, urlsplit
from urllib.request import Request, build_opener

from . import auth
from .domain import AvailabilityBasis, DataOrigin, DomainError, require_utc

if TYPE_CHECKING:
    from .contracts import SourceUseScope

_TICKER = r"[A-Z0-9][A-Z0-9_.-]{0,127}"
_MARKET_FILTERS = frozenset(
    {
        "limit",
        "cursor",
        "event_ticker",
        "series_ticker",
        "status",
        "tickers",
        "min_close_ts",
        "max_close_ts",
        "min_created_ts",
        "max_created_ts",
        "min_settled_ts",
        "max_settled_ts",
    }
)
_CANDLE_FILTERS = frozenset({"start_ts", "end_ts", "period_interval"})
_RETRYABLE = frozenset({429, 500, 502, 503, 504})
_MAX_BYTES = 64 * 1024 * 1024


class SourceError(ValueError):
    """Stable diagnostic and sanitized attempts, never headers or response bodies."""

    def __init__(self, code: str, attempts: tuple[FetchAttempt, ...] = ()) -> None:
        self.code = code
        self.attempts = attempts
        super().__init__(code)


def _utc(value: object) -> datetime:
    try:
        return require_utc(value, "source_time")
    except DomainError:
        raise SourceError("invalid_source_time") from None


def _query(query: str, allowed: frozenset[str]) -> dict[str, str]:
    try:
        pairs = parse_qsl(query, keep_blank_values=True, strict_parsing=True, max_num_fields=20)
    except ValueError:
        raise SourceError("unsupported_source_url") from None
    result: dict[str, str] = {}
    for key, value in pairs:
        if (
            key not in allowed
            or key in result
            or not value
            or len(value) > 1024
            or any(not 32 < ord(c) < 127 for c in value)
        ):
            raise SourceError("unsupported_source_url")
        result[key] = value
    return result


def _source_url(source: str, url: str) -> None:
    if (
        not isinstance(url, str)
        or not url
        or len(url) > 4096
        or any(not 32 < ord(c) < 127 for c in url)
        or "\\" in url
    ):
        raise SourceError("unsupported_source_url")
    try:
        parts = urlsplit(url)
        if (
            parts.scheme != "https"
            or parts.fragment
            or parts.username is not None
            or parts.password is not None
            or parts.port is not None
            or parts.netloc != parts.hostname
            or "%" in parts.path
            or any(segment in {".", ".."} for segment in parts.path.split("/"))
        ):
            raise ValueError
    except ValueError:
        raise SourceError("unsupported_source_url") from None
    path = parts.path
    if source == "kalshi" and parts.hostname == "external-api.kalshi.com":
        prefix = "/trade-api/v2"
        if not path.startswith(prefix + "/"):
            raise SourceError("unsupported_source_url")
        route = path.removeprefix(prefix)
        if route in {"/markets", "/historical/markets"}:
            _query(parts.query, _MARKET_FILTERS)
            return
        if route == "/historical/cutoff" or re.fullmatch(
            rf"/(?:markets|historical/markets|series)/{_TICKER}", route
        ):
            _query(parts.query, frozenset())
            return
        if re.fullmatch(rf"/events/{_TICKER}", route):
            _query(parts.query, frozenset({"with_nested_markets"}))
            return
        if re.fullmatch(
            rf"/(?:series/{_TICKER}/markets|historical/markets)/{_TICKER}/candlesticks", route
        ):
            allowed = _CANDLE_FILTERS | (
                {"include_latest_before_start"} if route.startswith("/series/") else set()
            )
            params = _query(parts.query, frozenset(allowed))
            if not params.keys() >= _CANDLE_FILTERS:
                raise SourceError("invalid_candle_range")
            try:
                start, end = (int(params[key]) for key in ("start_ts", "end_ts"))
                interval = int(params["period_interval"])
                if start < 0 or end < start or interval not in {1, 60, 1440}:
                    raise ValueError
            except ValueError:
                raise SourceError("invalid_candle_range") from None
            return
    elif source == "noaa":
        if parts.hostname == "noaa-ndfd-pds.s3.amazonaws.com" and path == "/":
            params = _query(
                parts.query, frozenset({"list-type", "prefix", "max-keys", "continuation-token"})
            )
            try:
                match = re.fullmatch(r"wmo/maxt/(\d{4}/\d{2}/\d{2})/", params["prefix"])
                if (
                    params["list-type"] != "2"
                    or match is None
                    or not re.fullmatch(r"[1-9]\d{0,3}", params["max-keys"])
                    or int(params["max-keys"]) > 1000
                ):
                    raise ValueError
                date.fromisoformat(match.group(1).replace("/", "-"))
            except (KeyError, ValueError):
                raise SourceError("unsupported_source_url") from None
            return
        if parts.hostname == "www.ncei.noaa.gov" and path == "/access/services/data/v1":
            required = frozenset(
                {
                    "dataset",
                    "stations",
                    "startDate",
                    "endDate",
                    "format",
                    "units",
                    "includeStationLocation",
                }
            )
            params = _query(parts.query, required)
            try:
                if (
                    params.keys() != required
                    or params["dataset"] != "daily-summaries"
                    or not re.fullmatch(r"USW\d{8}", params["stations"])
                    or params["format"] != "json"
                    or params["units"] != "standard"
                    or params["includeStationLocation"] != "true"
                    or any(
                        not re.fullmatch(r"\d{4}-\d{2}-\d{2}", params[key])
                        for key in ("startDate", "endDate")
                    )
                ):
                    raise ValueError
                start = date.fromisoformat(params["startDate"])
                end = date.fromisoformat(params["endDate"])
                if not 0 <= (end - start).days < 366:
                    raise ValueError
            except (KeyError, ValueError):
                raise SourceError("unsupported_source_url") from None
            return
        bucket = parts.hostname == "noaa-ndfd-pds.s3.amazonaws.com" and path.startswith(
            ("/wmo/", "/opnl/", "/expr/")
        )
        archive = parts.hostname == "www.ncei.noaa.gov" and path.startswith(
            ("/data/national-digital-forecast-database/", "/thredds/fileServer/ndfd/")
        )
        if (bucket or archive) and re.fullmatch(r"/[A-Za-z0-9_./-]+", path):
            _query(parts.query, frozenset())
            return
    elif source == "nws" and parts.hostname == "api.weather.gov":
        if (
            re.fullmatch(r"/points/-?\d+(?:\.\d+)?,-?\d+(?:\.\d+)?", path)
            or re.fullmatch(r"/gridpoints/[A-Z]{3}/\d+,\d+(?:/forecast(?:/hourly)?)?", path)
            or re.fullmatch(r"/products/[0-9a-fA-F-]{36}", path)
        ):
            _query(parts.query, frozenset())
            return
    raise SourceError("unsupported_source_url")


@dataclass(frozen=True, kw_only=True)
class DocumentRequest:
    source_name: str
    url: str
    origin: DataOrigin
    format: Literal["json", "text", "grib2"] = "json"
    max_bytes: int = 8 * 1024 * 1024
    timeout_seconds: float = 10
    max_attempts: int = 3
    max_retry_delay_seconds: float = 30

    def __post_init__(self) -> None:
        _source_url(self.source_name, self.url)
        try:
            object.__setattr__(self, "origin", DataOrigin(self.origin))
        except ValueError:
            raise SourceError("invalid_origin") from None
        if self.format not in {"json", "text", "grib2"}:
            raise SourceError("invalid_document_format")
        for value, maximum in ((self.max_bytes, _MAX_BYTES), (self.max_attempts, 10)):
            if type(value) is not int or not 1 <= value <= maximum:
                raise SourceError("invalid_request_limits")
        for value in (self.timeout_seconds, self.max_retry_delay_seconds):
            if type(value) not in (float, int) or not math.isfinite(value) or not 0 < value <= 60:
                raise SourceError("invalid_request_limits")

    @property
    def request_sha256(self) -> str:
        content = json.dumps(asdict(self), sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(content).hexdigest()


@dataclass(frozen=True, kw_only=True)
class HTTPReply:
    status: int
    body: bytes = field(repr=False)
    content_type: str | None = None
    retry_after: str | None = None
    content_encoding: str | None = None
    final_url: str | None = None

    def __post_init__(self) -> None:
        if type(self.status) is not int or not 100 <= self.status <= 599:
            raise SourceError("invalid_http_reply")
        if not isinstance(self.body, bytes):
            raise SourceError("invalid_http_reply")
        if any(
            value is not None and (not isinstance(value, str) or len(value) > 4096)
            for value in (
                self.content_type,
                self.retry_after,
                self.content_encoding,
                self.final_url,
            )
        ):
            raise SourceError("invalid_http_reply")


@dataclass(frozen=True)
class FetchAttempt:
    requested_at: datetime
    received_at: datetime
    outcome: str
    http_status: int | None
    retry_delay_seconds: float


@dataclass(frozen=True, kw_only=True)
class FetchedDocument:
    request: DocumentRequest
    requested_at: datetime
    received_at: datetime
    body: bytes = field(repr=False)
    content_type: str | None
    attempts: tuple[FetchAttempt, ...]
    scope_sha256: str | None = None
    content_sha256: str = field(init=False)
    issued_at: None = field(init=False, default=None)

    def __post_init__(self) -> None:
        _utc(self.requested_at)
        _utc(self.received_at)
        if self.received_at < self.requested_at:
            raise SourceError("invalid_receipt_time")
        if not isinstance(self.body, bytes) or len(self.body) > self.request.max_bytes:
            raise SourceError("invalid_document_body")
        object.__setattr__(self, "content_sha256", hashlib.sha256(self.body).hexdigest())

    @property
    def origin(self) -> DataOrigin:
        return self.request.origin

    @property
    def available_at(self) -> datetime | None:
        return self.received_at if self.origin is DataOrigin.PROSPECTIVE else None

    @property
    def availability_basis(self) -> AvailabilityBasis:
        return (
            AvailabilityBasis.OBSERVED_RECEIPT
            if self.origin is DataOrigin.PROSPECTIVE
            else AvailabilityBasis.UNKNOWN
        )

    def to_dict(self) -> dict[str, Any]:
        """Metadata only; exact body bytes are stored separately by the caller."""
        return {
            "schema_version": "source_document_v1",
            "source_name": self.request.source_name,
            "source_uri": self.request.url,
            "origin": self.origin.value,
            "request_sha256": self.request.request_sha256,
            "content_sha256": self.content_sha256,
            "scope_sha256": self.scope_sha256,
            "requested_at": self.requested_at.isoformat(),
            "received_at": self.received_at.isoformat(),
            "issued_at": None,
            "available_at": self.available_at.isoformat() if self.available_at else None,
            "availability_basis": self.availability_basis.value,
            "content_type": self.content_type,
            "bytes": len(self.body),
            "http_status": 200,
            "attempts": [
                asdict(a)
                | {
                    "requested_at": a.requested_at.isoformat(),
                    "received_at": a.received_at.isoformat(),
                }
                for a in self.attempts
            ],
        }


def _pairs(values: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in values:
        if key in result:
            raise ValueError
        result[key] = value
    return result


def _constant(value: str) -> Any:
    raise ValueError


def _json(body: bytes) -> dict[str, Any]:
    try:
        result = json.loads(
            body.decode("utf-8"),
            object_pairs_hook=_pairs,
            parse_constant=_constant,
            parse_float=Decimal,
        )
        if not isinstance(result, dict):
            raise ValueError
        return result
    except (ValueError, UnicodeError, RecursionError):
        raise SourceError("invalid_json") from None


def decode_json_document(document: FetchedDocument) -> dict[str, Any]:
    """Decode strict UTF-8 object JSON, preserving decimal strings, numbers and nulls."""
    if document.request.format != "json":
        raise SourceError("document_is_not_json")
    return _json(document.body)


def _open_document(request: Request, timeout: float, max_bytes: int) -> HTTPReply:
    """HTTPS GET with no redirects, transparent decompression or error-body retention."""
    if request.get_method() != "GET":
        raise SourceError("unsupported_http_method")
    try:
        deadline = time.monotonic() + timeout
        with build_opener(auth._RejectRedirect()).open(request, timeout=timeout) as response:
            encoding = response.headers.get("Content-Encoding")
            if encoding not in (None, "", "identity"):
                raise SourceError("unsupported_content_encoding")
            chunks = []
            size = 0
            while size <= max_bytes:
                if time.monotonic() >= deadline:
                    raise TimeoutError
                chunk = response.read1(min(65536, max_bytes + 1 - size))
                if not chunk:
                    break
                chunks.append(chunk)
                size += len(chunk)
            if size > max_bytes:
                raise SourceError("response_too_large")
            return HTTPReply(
                status=response.status,
                body=b"".join(chunks),
                content_type=response.headers.get("Content-Type"),
                content_encoding=encoding,
                final_url=response.geturl(),
                retry_after=response.headers.get("Retry-After"),
            )
    except HTTPError as exc:
        try:
            return HTTPReply(status=exc.code, body=b"", retry_after=exc.headers.get("Retry-After"))
        finally:
            exc.close()


def _delay(value: str | None, at: datetime, attempt: int, maximum: float) -> float:
    seconds = float(2 ** (attempt - 1))
    if value is not None and len(value) <= 128:
        try:
            if value.isascii() and value.isdigit():
                seconds = float(int(value))
            else:
                parsed = parsedate_to_datetime(value)
                if parsed.tzinfo is not None:
                    seconds = (parsed.astimezone(UTC) - at).total_seconds()
        except (ValueError, TypeError, OverflowError):
            pass
    return min(maximum, max(0.0, seconds))


def _scope(request: DocumentRequest, scope: SourceUseScope | None, at: datetime, root: Path) -> str:
    if scope is None:
        raise SourceError("source_use_required")
    from .contracts import require_source_use

    try:
        require_source_use(
            scope,
            source_name=request.source_name,
            origin=request.origin,
            uses=("collection", "retention"),
            at=at,
            project_root=root,
        )
        encoded = json.dumps(asdict(scope), default=str, sort_keys=True).encode()
        return hashlib.sha256(encoded).hexdigest()
    except (ValueError, TypeError, OSError):
        raise SourceError("source_use_refused") from None


def fetch_document(
    request: DocumentRequest,
    *,
    project_root: Path,
    scope: SourceUseScope | None = None,
    transport: Callable[[Request, float, int], HTTPReply] | None = None,
    clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    sleep: Callable[[float], None] = time.sleep,
) -> FetchedDocument:
    """Fetch one bounded document, recording attempts but never advancing collector cursors.

    Use max_attempts=1 if an outer collector owns retries. Historical documents retain
    unknown availability until an independently evidenced mapping supplies it; only
    prospective collection obtains observed-receipt availability automatically.
    """
    if not isinstance(request, DocumentRequest):
        raise SourceError("invalid_request")
    request.__post_init__()
    opener = transport or _open_document
    attempts: list[FetchAttempt] = []
    first_requested = None
    for attempt in range(1, request.max_attempts + 1):
        started = _utc(clock())
        if first_requested is None:
            first_requested = started
        if attempts and started < attempts[-1].received_at:
            raise SourceError("invalid_receipt_time", tuple(attempts))
        try:
            scope_hash = _scope(request, scope, started, project_root)
        except SourceError as exc:
            raise SourceError(exc.code, tuple(attempts)) from None
        headers = {
            "Accept": "application/geo+json, application/json"
            if request.format == "json"
            else "*/*",
            "Accept-Encoding": "identity",
            "User-Agent": "kalshi-information-quality-lab/0.1",
        }
        if request.source_name == "kalshi":
            try:
                credentials = auth.load_credentials(project_root)
                if credentials.environment != "production":
                    raise SourceError("credential_environment_mismatch")
                headers.update(auth.sign_request(credentials, "GET", urlsplit(request.url).path))
            except auth.CredentialError:
                raise SourceError("credential_failure", tuple(attempts)) from None
        http_request = Request(request.url, headers=headers, method="GET")
        reply = None
        failure = None
        try:
            reply = opener(http_request, float(request.timeout_seconds), request.max_bytes)
            if not isinstance(reply, HTTPReply):
                raise SourceError("invalid_http_reply")
        except HTTPError as exc:
            try:
                reply = HTTPReply(
                    status=exc.code, body=b"", retry_after=exc.headers.get("Retry-After")
                )
            finally:
                exc.close()
        except (URLError, OSError, HTTPException):
            failure = "network_failure"
        except SourceError:
            raise
        except Exception:
            raise SourceError("unexpected_transport", tuple(attempts)) from None
        received = _utc(clock())
        if received < started:
            raise SourceError("invalid_receipt_time", tuple(attempts))
        status = reply.status if reply is not None else None
        retryable = failure is not None or status in _RETRYABLE
        retry_delay = (
            _delay(
                reply.retry_after if reply else None,
                received,
                attempt,
                float(request.max_retry_delay_seconds),
            )
            if retryable and attempt < request.max_attempts
            else 0.0
        )
        attempts.append(
            FetchAttempt(started, received, failure or f"http_{status}", status, retry_delay)
        )
        if reply is not None and status == 200:
            if reply.final_url is not None and reply.final_url != request.url:
                raise SourceError("redirect_refused", tuple(attempts))
            if reply.content_encoding not in (None, "", "identity"):
                raise SourceError("unsupported_content_encoding", tuple(attempts))
            if len(reply.body) > request.max_bytes:
                raise SourceError("response_too_large", tuple(attempts))
            if request.format == "json":
                _json(reply.body)
            elif request.format == "text":
                try:
                    reply.body.decode("utf-8")
                except UnicodeError:
                    raise SourceError("invalid_utf8", tuple(attempts)) from None
            return FetchedDocument(
                request=request,
                requested_at=first_requested,
                received_at=received,
                body=reply.body,
                content_type=reply.content_type,
                attempts=tuple(attempts),
                scope_sha256=scope_hash,
            )
        if not retryable:
            code = (
                "redirect_refused" if status is not None and 300 <= status < 400 else "http_failure"
            )
            raise SourceError(code, tuple(attempts))
        if attempt == request.max_attempts:
            raise SourceError("retry_exhausted", tuple(attempts))
        sleep(retry_delay)
    raise AssertionError("bounded attempt loop must return or fail")
