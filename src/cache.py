"""Small JSON cache utilities for weather data payloads."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

JsonPayload = Any


def cache_key(namespace: str, params: dict[str, object]) -> str:
    """Return a deterministic key for a namespace and JSON-like params."""
    normalized = json.dumps(
        {"namespace": namespace, "params": params},
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    safe_namespace = "".join(c if c.isalnum() or c in "-_" else "_" for c in namespace)
    return f"{safe_namespace}-{digest}"


class JsonCache:
    """Tiny file-backed JSON cache."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def path_for(self, namespace: str, params: dict[str, object]) -> Path:
        return self.root / namespace / f"{cache_key(namespace, params)}.json"

    def get(self, namespace: str, params: dict[str, object]) -> JsonPayload:
        path = self.path_for(namespace, params)
        if not path.exists():
            return None
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def set(self, namespace: str, params: dict[str, object], payload: JsonPayload) -> None:
        path = self.path_for(namespace, params)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, sort_keys=True)
