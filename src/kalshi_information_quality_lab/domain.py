"""Shared source provenance and timestamp validation for the empirical study."""

from datetime import UTC, datetime
from enum import StrEnum


class DomainError(ValueError):
    """A source or timestamp violates an empirical pipeline invariant."""


class DataOrigin(StrEnum):
    """Whether evidence is historical or observed during prospective collection."""

    EMPIRICAL = "empirical"
    PROSPECTIVE = "prospective"


class AvailabilityBasis(StrEnum):
    """How a document's historical availability is established."""

    OBSERVED_RECEIPT = "observed_receipt"
    UNKNOWN = "unknown"


def require_utc(value: object, name: str) -> datetime:
    """Return a UTC instant or reject naive and non-datetime values."""
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise DomainError(f"{name} must be a timezone-aware datetime")
    try:
        result = value.astimezone(UTC)
    except (OverflowError, OSError, ValueError):
        raise DomainError(f"{name} must be a valid timezone-aware datetime") from None
    if result.utcoffset() is None:
        raise DomainError(f"{name} must be a timezone-aware datetime")
    return result
