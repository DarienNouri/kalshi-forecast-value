"""Atomic, immutable publication for empirical run artifacts."""

import os
import stat
from pathlib import Path
from uuid import uuid4


class StorageError(ValueError):
    """An immutable artifact path or stored value is unsafe or inconsistent."""


def _read_bytes(directory: int, name: str) -> bytes:
    fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
    with os.fdopen(fd, "rb") as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise StorageError("artifact path must be a regular file")
        return stream.read()


def _publish(directory: int, name: str, body: bytes) -> None:
    """Link fully written bytes without replacing an existing final path."""
    try:
        previous = _read_bytes(directory, name)
    except FileNotFoundError:
        pass
    else:
        if previous != body:
            raise StorageError("immutable artifact conflicts with stored bytes")
        os.fsync(directory)
        return

    pending = f".{uuid4().hex}.pending"
    fd = os.open(
        pending,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
        0o600,
        dir_fd=directory,
    )
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(body)
            stream.flush()
            os.fsync(stream.fileno())
        if _read_bytes(directory, pending) != body:
            raise StorageError("staged artifact checksum mismatch")
        try:
            os.link(
                pending,
                name,
                src_dir_fd=directory,
                dst_dir_fd=directory,
                follow_symlinks=False,
            )
        except FileExistsError:
            if _read_bytes(directory, name) != body:
                raise StorageError("immutable artifact conflicts with stored bytes") from None
        os.fsync(directory)
        if _read_bytes(directory, name) != body:
            raise StorageError("published artifact checksum mismatch")
    finally:
        os.unlink(pending, dir_fd=directory)


def publish_immutable_file(directory: Path, name: str, body: bytes) -> Path:
    """Atomically publish exact bytes under one already-created directory."""
    if Path(name).name != name or name in {"", ".", ".."}:
        raise StorageError("immutable filename must be one path component")
    if not isinstance(body, bytes):
        raise StorageError("immutable file content must be bytes")
    try:
        fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    except OSError:
        raise StorageError("immutable artifact directory is missing or unsafe") from None
    try:
        _publish(fd, name, body)
    except OSError:
        raise StorageError("immutable artifact publication failed") from None
    finally:
        os.close(fd)
    return directory / name
