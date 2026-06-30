"""Explicit local credentials for read-only Kalshi source requests."""

import base64
import time
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from urllib.request import HTTPRedirectHandler, Request

from cryptography.exceptions import UnsupportedAlgorithm
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa

_LOCAL_CONFIG = Path("config/credentials.local.toml")
_MAX_CONFIG_BYTES = 16_384
_MAX_ID_BYTES = 1_024
_MAX_KEY_BYTES = 65_536


class CredentialError(ValueError):
    """Credential failure safe to display without secret or response-body details."""


@dataclass(frozen=True)
class _CredentialConfig:
    environment: str
    api_key_id_file: Path = field(repr=False)
    private_key_file: Path = field(repr=False)


@dataclass(frozen=True)
class KalshiCredentials:
    """Loaded credentials; never serialize this object or log its sensitive fields."""

    environment: str
    api_key_id: str = field(repr=False)
    private_key: rsa.RSAPrivateKey = field(repr=False)


def _bounded_read(path: Path, limit: int, label: str) -> bytes:
    try:
        if not path.is_file():
            raise CredentialError(f"{label} must reference a readable regular file")
        with path.open("rb") as handle:
            content = handle.read(limit + 1)
    except OSError:
        raise CredentialError(f"cannot read {label}") from None
    if not content or len(content) > limit:
        raise CredentialError(f"{label} is empty or exceeds its size limit")
    return content


def _load_configuration(project_root: Path) -> _CredentialConfig:
    content = _bounded_read(project_root / _LOCAL_CONFIG, _MAX_CONFIG_BYTES, "credential config")
    try:
        raw = tomllib.loads(content.decode("utf-8"))
    except (UnicodeError, tomllib.TOMLDecodeError):
        raise CredentialError("credential config is not valid UTF-8 TOML") from None
    if set(raw) != {"environment", "api_key_id_file", "private_key_file"}:
        raise CredentialError("credential config has missing or unsupported fields")
    environment = raw["environment"]
    if environment != "production":
        raise CredentialError("credential environment must be production")

    paths: dict[str, Path] = {}
    for name in ("api_key_id_file", "private_key_file"):
        value = raw[name]
        if not isinstance(value, str) or not value.strip() or "\x00" in value:
            raise CredentialError(f"{name} must be a nonempty file path")
        try:
            path = Path(value).expanduser()
            paths[name] = (path if path.is_absolute() else project_root / path).resolve()
        except (OSError, RuntimeError, ValueError):
            raise CredentialError(f"cannot resolve {name}") from None
    return _CredentialConfig(environment, paths["api_key_id_file"], paths["private_key_file"])


def load_credentials(project_root: Path) -> KalshiCredentials:
    """Read only the two explicitly configured files, retaining no persistent copies."""
    config = _load_configuration(project_root)
    content = _bounded_read(config.api_key_id_file, _MAX_ID_BYTES, "API key ID file")
    try:
        api_key_id = content.decode("ascii").strip()
    except UnicodeError:
        raise CredentialError(
            "API key ID must contain printable ASCII without whitespace"
        ) from None
    if not api_key_id or any(not 33 <= ord(character) <= 126 for character in api_key_id):
        raise CredentialError("API key ID must contain printable ASCII without whitespace")
    key_bytes = _bounded_read(config.private_key_file, _MAX_KEY_BYTES, "private key file")
    try:
        private_key = serialization.load_pem_private_key(key_bytes, password=None)
    except (ValueError, TypeError, UnsupportedAlgorithm):
        raise CredentialError("private key must be a valid unencrypted RSA PEM") from None
    if not isinstance(private_key, rsa.RSAPrivateKey):
        raise CredentialError("private key must be a valid unencrypted RSA PEM")
    return KalshiCredentials(config.environment, api_key_id, private_key)


def sign_request(
    credentials: KalshiCredentials,
    method: str,
    path: str,
    *,
    timestamp_ms: int | None = None,
) -> dict[str, str]:
    """Sign the uppercase method and absolute API path, excluding any query string.

    The returned headers contain credentials/signatures and must never be logged.
    """
    if not method.isascii() or not method.isalpha() or not path.startswith("/"):
        raise CredentialError("signing requires an HTTP method and absolute API path")
    if timestamp_ms is None:
        timestamp_ms = time.time_ns() // 1_000_000
    if isinstance(timestamp_ms, bool) or not isinstance(timestamp_ms, int) or timestamp_ms < 0:
        raise CredentialError("signing timestamp must be nonnegative integer milliseconds")
    timestamp = str(timestamp_ms)
    message = f"{timestamp}{method.upper()}{path.split('?', 1)[0]}".encode()
    try:
        signature = credentials.private_key.sign(
            message,
            padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.DIGEST_LENGTH),
            hashes.SHA256(),
        )
    except (ValueError, TypeError, UnsupportedAlgorithm):
        raise CredentialError("could not sign the authentication request") from None
    return {
        "KALSHI-ACCESS-KEY": credentials.api_key_id,
        "KALSHI-ACCESS-TIMESTAMP": timestamp,
        "KALSHI-ACCESS-SIGNATURE": base64.b64encode(signature).decode("ascii"),
    }


class _RejectRedirect(HTTPRedirectHandler):
    def redirect_request(
        self, req: Request, fp: object, code: int, msg: str, headers: object, newurl: str
    ) -> None:
        """Never forward authentication headers to any redirected URL."""
        return None
