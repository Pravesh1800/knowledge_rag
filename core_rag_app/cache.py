from __future__ import annotations

import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


LEGACY_ROOT_ENV = "PDF" + "_VISION_RAG_ROOT"
PROJECT_ROOT = Path(
    os.getenv("EVIDENCE_MESH_ROOT")
    or os.getenv(LEGACY_ROOT_ENV)
    or Path(__file__).resolve().parent
).resolve()
CACHE_DIR = PROJECT_ROOT / "cache"


def cache_enabled() -> bool:
    return os.getenv("EVIDENCE_MESH_CACHE", "1").strip().lower() not in {"0", "false", "no", "off"}


def stable_hash(payload: Any) -> str:
    normalized = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def cache_path(namespace: str, key_payload: Any, suffix: str = ".json") -> Path:
    digest = stable_hash(key_payload)
    return CACHE_DIR / namespace / digest[:2] / f"{digest}{suffix}"


def read_cache(namespace: str, key_payload: Any) -> dict[str, Any] | None:
    if not cache_enabled():
        return None
    path = cache_path(namespace, key_payload)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return None


def write_cache(namespace: str, key_payload: Any, value: Any) -> None:
    if not cache_enabled():
        return
    path = cache_path(namespace, key_payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "cached_at": datetime.now(timezone.utc).isoformat(),
        "key": key_payload,
        "value": value,
    }
    temp_path = path.with_name(f"{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    temp_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    temp_path.replace(path)

