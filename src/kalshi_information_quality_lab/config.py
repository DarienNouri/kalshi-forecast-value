"""Parse the empirical study configuration without inferring source permission."""

import tomllib
from dataclasses import dataclass
from pathlib import Path


class ConfigError(ValueError):
    """A missing, malformed, or unsupported study configuration."""


@dataclass(frozen=True)
class StudyConfig:
    project_root: Path
    project_name: str
    title: str
    hypothesis: str
    candidate_cities: tuple[str, ...]
    horizons_hours: tuple[int, ...]
    horizon_anchor: str
    protocol_status: str
    primary_score: str
    data_mode: str
    data_root: Path
    local_research_scope: str
    evidence_reference: str


def _table(value: object, name: str, keys: set[str]) -> dict[str, object]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ConfigError(f"{name} must be a table")
    actual = set(value)
    if missing := keys - actual:
        raise ConfigError(f"{name}: missing fields: {', '.join(sorted(missing))}")
    if unknown := actual - keys:
        raise ConfigError(f"{name}: unknown fields: {', '.join(sorted(unknown))}")
    return value


def _text(value: object, name: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        raise ConfigError(f"{name} must be {'a' if allow_empty else 'a nonempty'} string")
    return value.strip()


def _choice(value: object, name: str, choices: set[str]) -> str:
    selected = _text(value, name)
    if selected not in choices:
        raise ConfigError(f"{name} must be one of: {', '.join(sorted(choices))}")
    return selected


def load_config(path: Path, *, project_root: Path | None = None) -> StudyConfig:
    """Resolve config and data paths from the project root, defaulting to the current directory.

    Callers must supply the repository root or run from it. Absolute configuration paths
    do not change the data root. No directories are created and no services are contacted.
    """
    project_root = (project_root or Path.cwd()).resolve()
    if not path.is_absolute():
        path = project_root / path
    try:
        with path.open("rb") as handle:
            raw = tomllib.load(handle)
    except OSError as exc:
        raise ConfigError(f"cannot read config {path}: {exc.strerror}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"invalid TOML in {path}: {exc}") from exc

    root = _table(raw, "config", {"project", "study", "data", "access"})
    project = _table(root["project"], "project", {"name"})
    study = _table(
        root["study"],
        "study",
        {
            "title",
            "hypothesis",
            "candidate_cities",
            "horizons_hours",
            "horizon_anchor",
            "protocol_status",
            "primary_score",
        },
    )
    data = _table(root["data"], "data", {"mode", "root"})
    access = _table(root["access"], "access", {"local_research_scope", "evidence_reference"})

    raw_cities = study["candidate_cities"]
    if not isinstance(raw_cities, list) or not raw_cities:
        raise ConfigError("study.candidate_cities must be a nonempty array")
    cities = tuple(_text(city, "study.candidate_cities entry") for city in raw_cities)
    if len({city.casefold() for city in cities}) != len(cities):
        raise ConfigError("study.candidate_cities must contain unique cities")

    raw_horizons = study["horizons_hours"]
    if not isinstance(raw_horizons, list) or not raw_horizons:
        raise ConfigError("study.horizons_hours must be a nonempty array")
    horizons: list[int] = []
    for horizon in raw_horizons:
        if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon <= 0:
            raise ConfigError("study.horizons_hours entries must be positive integers")
        horizons.append(horizon)
    if len(set(horizons)) != len(horizons):
        raise ConfigError("study.horizons_hours must contain unique values")

    data_root = _text(data["root"], "data.root")
    if Path(data_root).is_absolute() or ".." in Path(data_root).parts:
        raise ConfigError("data.root must be a project-relative path without '..'")
    resolved_data_root = (project_root / data_root).resolve()
    if not resolved_data_root.is_relative_to(project_root):
        raise ConfigError("data.root resolves outside the project root")
    local_scope = _choice(
        access["local_research_scope"],
        "access.local_research_scope",
        {"documented_local_research"},
    )
    evidence_reference = _text(
        access["evidence_reference"], "access.evidence_reference", allow_empty=True
    )
    if not evidence_reference:
        raise ConfigError("documented local research scope requires access.evidence_reference")

    return StudyConfig(
        project_root=project_root,
        project_name=_text(project["name"], "project.name"),
        title=_text(study["title"], "study.title"),
        hypothesis=_choice(study["hypothesis"], "study.hypothesis", {"whether_and_when"}),
        candidate_cities=cities,
        horizons_hours=tuple(horizons),
        horizon_anchor=_choice(
            study["horizon_anchor"], "study.horizon_anchor", {"outcome_window_start"}
        ),
        protocol_status=_choice(study["protocol_status"], "study.protocol_status", {"frozen"}),
        primary_score=_choice(study["primary_score"], "study.primary_score", {"multiclass_brier"}),
        data_mode=_choice(data["mode"], "data.mode", {"historical"}),
        data_root=resolved_data_root,
        local_research_scope=local_scope,
        evidence_reference=evidence_reference,
    )
