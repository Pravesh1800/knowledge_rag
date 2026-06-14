from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from openai import OpenAI


APP_ROOT = Path(__file__).resolve().parent
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENAI_BASE_URL = "https://api.openai.com/v1"
DEFAULT_PROVIDER = "openrouter"
DEFAULT_OPENROUTER_MODEL = "google/gemini-3.1-flash-lite"
DEFAULT_OPENROUTER_MAP_MODEL = "deepseek/deepseek-v4-flash"
DEFAULT_OPENROUTER_SEARCH_MODEL = "deepseek/deepseek-v4-flash"
DEFAULT_OPENAI_MODEL = "gpt-4.1-mini"
DEFAULT_OPENAI_MAP_MODEL = "gpt-4.1-mini"
DEFAULT_OPENAI_SEARCH_MODEL = "gpt-4.1-mini"
PROJECT_ENV_KEYS = {
    "EVIDENCE_MESH_ROOT",
    "PDF_VISION_RAG_ROOT",
    "EVIDENCE_MESH_PROJECT_ID",
}


def load_dotenv(env_path: Path | None = None, override: bool = True) -> None:
    path = env_path or APP_ROOT / ".env"
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("\"'")
        if key in PROJECT_ENV_KEYS and os.getenv(key):
            continue
        if override:
            os.environ[key] = value
        else:
            os.environ.setdefault(key, value)


def get_provider() -> str:
    provider = (
        os.getenv("LLM_PROVIDER")
        or os.getenv("AI_PROVIDER")
        or DEFAULT_PROVIDER
    ).strip().lower()
    return provider if provider in {"openrouter", "openai"} else DEFAULT_PROVIDER


def provider_prefix(provider: str | None = None) -> str:
    return get_provider().upper() if provider is None else provider.upper()


def get_api_key(provider: str | None = None) -> str:
    active = provider or get_provider()
    prefix = provider_prefix(active)
    return (
        os.getenv("LLM_API_KEY")
        or os.getenv(f"{prefix}_API_KEY")
        or ""
    ).strip()


def get_base_url(provider: str | None = None) -> str:
    active = provider or get_provider()
    prefix = provider_prefix(active)
    if active == "openrouter":
        return (
            os.getenv("LLM_BASE_URL")
            or os.getenv("OPENROUTER_BASE_URL")
            or OPENROUTER_BASE_URL
        ).strip()
    return (
        os.getenv("LLM_BASE_URL")
        or os.getenv("OPENAI_BASE_URL")
        or OPENAI_BASE_URL
    ).strip()


def default_model(provider: str, task: str = "") -> str:
    if provider == "openai":
        if task == "map":
            return DEFAULT_OPENAI_MAP_MODEL
        if task == "search":
            return DEFAULT_OPENAI_SEARCH_MODEL
        return DEFAULT_OPENAI_MODEL
    if task == "map":
        return DEFAULT_OPENROUTER_MAP_MODEL
    if task == "search":
        return DEFAULT_OPENROUTER_SEARCH_MODEL
    return DEFAULT_OPENROUTER_MODEL


def get_model(task: str = "") -> str:
    active = get_provider()
    prefix = provider_prefix(active)
    task_key = task.upper()
    candidates = []
    if task_key:
        candidates.extend([
            f"LLM_{task_key}_MODEL",
            f"{prefix}_{task_key}_MODEL",
        ])
        if active == "openrouter":
            candidates.append(f"OPENROUTER_{task_key}_MODEL")
    candidates.extend([
        "LLM_MODEL",
        f"{prefix}_MODEL",
    ])
    if active == "openrouter":
        candidates.append("OPENROUTER_MODEL")
    for key in candidates:
        value = os.getenv(key)
        if value:
            return value.strip()
    return default_model(active, task)


def get_client_timeout_seconds(provider: str | None = None) -> float:
    active = provider or get_provider()
    prefix = provider_prefix(active)
    for key in ("LLM_TIMEOUT_SECONDS", f"{prefix}_TIMEOUT_SECONDS"):
        value = os.getenv(key)
        if value:
            try:
                return max(1.0, float(value))
            except ValueError:
                continue
    return 120.0


def get_client_max_retries(provider: str | None = None) -> int:
    active = provider or get_provider()
    prefix = provider_prefix(active)
    for key in ("LLM_MAX_RETRIES", f"{prefix}_MAX_RETRIES"):
        value = os.getenv(key)
        if value:
            try:
                return max(0, int(value))
            except ValueError:
                continue
    return 1


def create_chat_client() -> tuple[OpenAI, str, str]:
    load_dotenv(override=False)
    active = get_provider()
    api_key = get_api_key(active)
    if not api_key:
        expected = "OPENAI_API_KEY" if active == "openai" else "OPENROUTER_API_KEY"
        raise RuntimeError(f"{expected} is missing in .env")

    kwargs: dict[str, Any] = {
        "api_key": api_key,
        "timeout": get_client_timeout_seconds(active),
        "max_retries": get_client_max_retries(active),
    }
    base_url = get_base_url(active)
    if base_url:
        kwargs["base_url"] = base_url
    if active == "openrouter":
        kwargs["default_headers"] = {
            "HTTP-Referer": os.getenv("OPENROUTER_SITE_URL", "http://localhost"),
            "X-Title": os.getenv("OPENROUTER_APP_NAME", "Evidence Mesh"),
        }
    return OpenAI(**kwargs), get_model(), active


def runtime_settings() -> dict[str, Any]:
    load_dotenv()
    active = get_provider()
    key = get_api_key(active)
    return {
        "provider": active,
        "api_key_present": bool(key),
        "api_key_hint": f"...{key[-4:]}" if len(key) >= 4 else "",
        "model": get_model(),
        "map_model": get_model("map"),
        "search_model": get_model("search"),
        "openrouter_model": os.getenv("OPENROUTER_MODEL", DEFAULT_OPENROUTER_MODEL),
        "openrouter_map_model": os.getenv("OPENROUTER_MAP_MODEL", DEFAULT_OPENROUTER_MAP_MODEL),
        "openrouter_search_model": os.getenv("OPENROUTER_SEARCH_MODEL", DEFAULT_OPENROUTER_SEARCH_MODEL),
        "openai_model": os.getenv("OPENAI_MODEL", DEFAULT_OPENAI_MODEL),
        "openai_map_model": os.getenv("OPENAI_MAP_MODEL", DEFAULT_OPENAI_MAP_MODEL),
        "openai_search_model": os.getenv("OPENAI_SEARCH_MODEL", DEFAULT_OPENAI_SEARCH_MODEL),
        "base_url": get_base_url(active),
        "openrouter_base_url": os.getenv("OPENROUTER_BASE_URL", OPENROUTER_BASE_URL),
        "openai_base_url": os.getenv("OPENAI_BASE_URL", OPENAI_BASE_URL),
    }


def update_dotenv(updates: dict[str, str], env_path: Path | None = None) -> None:
    path = env_path or APP_ROOT / ".env"
    path.parent.mkdir(parents=True, exist_ok=True)
    existing: dict[str, str] = {}
    order: list[str] = []
    if path.exists():
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            if not line or line.lstrip().startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            existing[key] = value.strip()
            order.append(key)
    for key, value in updates.items():
        if value == "":
            continue
        if key not in existing:
            order.append(key)
        existing[key] = value
    path.write_text("\n".join(f"{key}={existing[key]}" for key in order) + "\n", encoding="utf-8")
    load_dotenv(path)
