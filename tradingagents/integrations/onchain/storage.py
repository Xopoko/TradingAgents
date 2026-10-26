"""Small durable JSON boundary for the local-fork runner and worker."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

from pydantic import BaseModel


class StateError(ValueError):
    pass


def write_model(path: Path, model: BaseModel) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", prefix=f".{path.name}-", suffix=".tmp",
            dir=path.parent, delete=False,
        ) as stream:
            temporary = stream.name
            stream.write(model.model_dump_json(indent=2))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        temporary = None
        if os.name != "nt":
            descriptor = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
    finally:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)


def read_model(path: Path, model, default=None):
    try:
        if not path.exists():
            return default
        if path.stat().st_size > 8 * 1024 * 1024:
            raise StateError("state_too_large")
        return model.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise StateError("state_corrupt") from exc
