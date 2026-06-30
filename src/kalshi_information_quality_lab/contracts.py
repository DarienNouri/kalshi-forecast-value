"""Explicit, hash-checked local scope for empirical source retrieval."""

import hashlib
import os
import re
import stat
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from .domain import DataOrigin, DomainError, require_utc


class ContractError(ValueError):
    """A source-use declaration or evidence reference is invalid."""


def _identifier(value: object) -> str:
    if (
        not isinstance(value, str)
        or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", value) is None
    ):
        raise ContractError("invalid_identifier")
    return value


def _origin(value: object) -> DataOrigin:
    try:
        return DataOrigin(value)
    except (TypeError, ValueError):
        raise ContractError("invalid_origin") from None


def _utc(value: object) -> datetime:
    try:
        return require_utc(value, "scope_time").astimezone(UTC)
    except DomainError:
        raise ContractError("invalid_scope_time") from None


def _uses(value: object) -> tuple[str, ...]:
    if not isinstance(value, tuple) or not value:
        raise ContractError("invalid_source_uses")
    uses = tuple(_identifier(item) for item in value)
    if len(set(uses)) != len(uses):
        raise ContractError("invalid_source_uses")
    return uses


@dataclass(frozen=True)
class EvidenceReference:
    path: str
    content_sha256: str

    def __post_init__(self) -> None:
        candidate = Path(self.path)
        if candidate.is_absolute() or ".." in candidate.parts or not candidate.parts:
            raise ContractError("invalid_evidence_reference")
        if re.fullmatch(r"[a-f0-9]{64}", self.content_sha256) is None:
            raise ContractError("invalid_evidence_reference")

    def verify(self, project_root: Path) -> None:
        root = project_root.resolve(strict=True)
        candidate = root.joinpath(self.path)
        parent = candidate.parent.resolve(strict=True)
        if parent != root and not parent.is_relative_to(root):
            raise ContractError("evidence_reference_escape")
        parent_fd = os.open(parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            fd = os.open(candidate.name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=parent_fd)
            try:
                if not stat.S_ISREG(os.fstat(fd).st_mode):
                    raise ContractError("invalid_evidence_reference")
                with os.fdopen(fd, "rb", closefd=False) as stream:
                    digest = hashlib.sha256(stream.read()).hexdigest()
            finally:
                os.close(fd)
        finally:
            os.close(parent_fd)
        if digest != self.content_sha256:
            raise ContractError("evidence_hash_mismatch")


@dataclass(frozen=True)
class SourceUseScope:
    source_name: str
    origin: DataOrigin
    allowed_uses: tuple[str, ...]
    reviewed_at: datetime
    evidence: tuple[EvidenceReference, ...]
    expires_at: datetime | None = None

    def __post_init__(self) -> None:
        _identifier(self.source_name)
        object.__setattr__(self, "origin", _origin(self.origin))
        object.__setattr__(self, "allowed_uses", _uses(self.allowed_uses))
        if not isinstance(self.evidence, tuple) or not self.evidence or any(
            not isinstance(ref, EvidenceReference) for ref in self.evidence
        ):
            raise ContractError("invalid_evidence_references")
        reviewed = _utc(self.reviewed_at)
        object.__setattr__(self, "reviewed_at", reviewed)
        if self.expires_at is not None:
            expires = _utc(self.expires_at)
            if expires <= reviewed:
                raise ContractError("invalid_scope_expiry")
            object.__setattr__(self, "expires_at", expires)


def require_source_use(
    scope: SourceUseScope,
    *,
    source_name: str,
    origin: DataOrigin,
    uses: tuple[str, ...],
    at: datetime,
    project_root: Path,
) -> None:
    """Validate recorded local scope without inferring third-party permission."""
    if not isinstance(scope, SourceUseScope):
        raise ContractError("invalid_source_scope")
    requested_origin = _origin(origin)
    requested_uses = _uses(uses)
    instant = _utc(at)
    if scope.source_name != _identifier(source_name) or scope.origin != requested_origin:
        raise ContractError("source_scope_mismatch")
    if instant < scope.reviewed_at or (
        scope.expires_at is not None and instant >= scope.expires_at
    ):
        raise ContractError("source_scope_not_current")
    if not set(requested_uses) <= set(scope.allowed_uses):
        raise ContractError("source_use_not_recorded")
    for reference in scope.evidence:
        reference.verify(project_root)
