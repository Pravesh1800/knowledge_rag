from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from openai import OpenAI

from llm_config import OPENAI_BASE_URL, load_dotenv
from schema import card_id_from_record
from storage import read_cards


LEGACY_ROOT_ENV = "PDF" + "_VISION_RAG_ROOT"
PROJECT_ROOT = Path(
    os.getenv("EVIDENCE_MESH_ROOT")
    or os.getenv(LEGACY_ROOT_ENV)
    or Path(__file__).resolve().parent
).resolve()
INDEXES_DIR = PROJECT_ROOT / "indexes"
CARD_EMBEDDINGS_PATH = INDEXES_DIR / "card_embeddings.json"

CARD_EMBEDDINGS_SCHEMA_VERSION = "card_embeddings.v1.0"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_EMBEDDING_BATCH_SIZE = 64
DEFAULT_EMBEDDING_TEXT_CHARS = 3200


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def stable_text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:24]


def embedding_text(card: dict[str, Any], max_chars: int = DEFAULT_EMBEDDING_TEXT_CHARS) -> str:
    parts = [
        f"Card: {card.get('card_name', '')}",
        f"Description: {card.get('card_description', '')}",
        f"Document: {card.get('document_name', '')}",
        f"Page: {card.get('page_no', '')}",
        f"Source: {card.get('card_source', '')}",
        f"Tags: {', '.join(str(tag) for tag in card.get('tags', []) or [])}",
        f"Content: {card.get('content', '')}",
    ]
    text = re.sub(r"\s+", " ", "\n".join(str(part) for part in parts)).strip()
    return text[:max_chars]


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f"{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    temp_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    temp_path.replace(path)


def embedding_settings() -> dict[str, str]:
    load_dotenv()
    provider = os.getenv("EMBEDDING_PROVIDER", "openai").strip().lower() or "openai"
    if provider not in {"openai", "openrouter"}:
        provider = "openai"
    prefix = provider.upper()
    api_key = ""
    key_candidates = (
        (
            os.getenv("EMBEDDING_API_KEY"),
            os.getenv("OPENAI_API_KEY"),
        )
        if provider == "openai"
        else (
            os.getenv("EMBEDDING_API_KEY"),
            os.getenv("OPENROUTER_API_KEY"),
            os.getenv("LLM_API_KEY"),
        )
    )
    for candidate in key_candidates:
        candidate = (candidate or "").strip()
        lowered = candidate.lower()
        if not candidate or lowered.startswith("your_") or "your_" in lowered or lowered.endswith("_here"):
            continue
        if provider == "openai" and lowered.startswith("sk-or-"):
            continue
        api_key = candidate
        break
    base_url = (
        os.getenv("EMBEDDING_BASE_URL")
        or os.getenv(f"{prefix}_BASE_URL")
        or (OPENAI_BASE_URL if provider == "openai" else "https://openrouter.ai/api/v1")
    ).strip()
    model = (
        os.getenv("EMBEDDING_MODEL")
        or os.getenv(f"{prefix}_EMBEDDING_MODEL")
        or DEFAULT_EMBEDDING_MODEL
    ).strip()
    return {"provider": provider, "api_key": api_key, "base_url": base_url, "model": model}


def create_embedding_client() -> tuple[OpenAI, str, str]:
    settings = embedding_settings()
    if not settings["api_key"]:
        raise RuntimeError("No embedding API key configured. Set EMBEDDING_API_KEY or OPENAI_API_KEY.")
    kwargs: dict[str, Any] = {"api_key": settings["api_key"]}
    if settings["base_url"]:
        kwargs["base_url"] = settings["base_url"]
    if settings["provider"] == "openrouter":
        kwargs["default_headers"] = {
            "HTTP-Referer": os.getenv("OPENROUTER_SITE_URL", "http://localhost"),
            "X-Title": os.getenv("OPENROUTER_APP_NAME", "Evidence Mesh"),
        }
    return OpenAI(**kwargs), settings["model"], settings["provider"]


def normalize_vector(vector: list[float]) -> list[float]:
    magnitude = math.sqrt(sum(float(value) * float(value) for value in vector))
    if not magnitude:
        return []
    return [float(value) / magnitude for value in vector]


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    return max(0.0, sum(float(a) * float(b) for a, b in zip(left, right)))


def batch_items(items: list[Any], size: int) -> list[list[Any]]:
    return [items[index : index + size] for index in range(0, len(items), size)]


def embed_texts(client: OpenAI, model: str, texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    response = client.embeddings.create(model=model, input=texts)
    vectors = [normalize_vector(list(item.embedding)) for item in response.data]
    if len(vectors) != len(texts):
        raise RuntimeError(f"Embedding API returned {len(vectors)} vector(s) for {len(texts)} text(s).")
    return vectors


def read_card_embeddings(path: Path = CARD_EMBEDDINGS_PATH) -> dict[str, Any]:
    payload = read_json(path, {})
    if not isinstance(payload, dict):
        return {"schema_version": CARD_EMBEDDINGS_SCHEMA_VERSION, "cards": {}}
    cards = payload.get("cards")
    if not isinstance(cards, dict):
        payload["cards"] = {}
    return payload


def write_card_embeddings(payload: dict[str, Any], path: Path = CARD_EMBEDDINGS_PATH) -> None:
    payload["schema_version"] = CARD_EMBEDDINGS_SCHEMA_VERSION
    payload["updated_at"] = utc_now()
    write_json(path, payload)


def build_card_embeddings(
    cards: list[dict[str, Any]],
    *,
    path: Path = CARD_EMBEDDINGS_PATH,
    force: bool = False,
) -> dict[str, Any]:
    client, model, provider = create_embedding_client()
    payload = read_card_embeddings(path)
    existing_cards: dict[str, Any] = payload.get("cards", {})
    batch_size = max(1, int(os.getenv("EMBEDDING_BATCH_SIZE", str(DEFAULT_EMBEDDING_BATCH_SIZE))))

    work_items: list[tuple[str, str, str]] = []
    valid_ids = set()
    for card in cards:
        card_id = card_id_from_record(card)
        valid_ids.add(card_id)
        text = embedding_text(card)
        text_hash = stable_text_hash(text)
        existing = existing_cards.get(card_id)
        if not force and isinstance(existing, dict) and existing.get("text_hash") == text_hash and existing.get("model") == model:
            continue
        work_items.append((card_id, text_hash, text))

    for card_id in list(existing_cards):
        if card_id not in valid_ids:
            existing_cards.pop(card_id, None)

    for batch in batch_items(work_items, batch_size):
        vectors = embed_texts(client, model, [item[2] for item in batch])
        for (card_id, text_hash, _text), vector in zip(batch, vectors):
            existing_cards[card_id] = {
                "card_id": card_id,
                "text_hash": text_hash,
                "model": model,
                "provider": provider,
                "dimensions": len(vector),
                "embedding": vector,
                "updated_at": utc_now(),
            }
        write_card_embeddings(
            {
                **payload,
                "model": model,
                "provider": provider,
                "card_count": len(existing_cards),
                "cards": existing_cards,
            },
            path,
        )

    result = {
        **payload,
        "model": model,
        "provider": provider,
        "card_count": len(existing_cards),
        "cards": existing_cards,
    }
    write_card_embeddings(result, path)
    return result


def embed_query(query: str) -> tuple[list[float], str] | tuple[None, str]:
    try:
        client, model, _provider = create_embedding_client()
        vectors = embed_texts(client, model, [query])
        return (vectors[0] if vectors else None), model
    except Exception:
        return None, ""


def vector_scores(query: str, cards: list[dict[str, Any]], path: Path = CARD_EMBEDDINGS_PATH) -> dict[str, float]:
    payload = read_card_embeddings(path)
    embeddings = payload.get("cards", {})
    if not isinstance(embeddings, dict) or not embeddings:
        return {}
    query_vector, model = embed_query(query)
    if not query_vector:
        return {}
    scores: dict[str, float] = {}
    for card in cards:
        card_id = card_id_from_record(card)
        item = embeddings.get(card_id)
        if not isinstance(item, dict):
            continue
        if model and item.get("model") != model:
            continue
        vector = item.get("embedding")
        if not isinstance(vector, list):
            continue
        score = cosine_similarity(query_vector, [float(value) for value in vector])
        if score > 0:
            scores[card_id] = score
    return scores


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build or refresh card embeddings for hybrid retrieval.")
    subparsers = parser.add_subparsers(dest="command")
    build = subparsers.add_parser("build", help="Build missing or changed card embeddings.")
    build.add_argument("--force", action="store_true", help="Rebuild all embeddings even if hashes match.")
    build.add_argument("--project-id", help="Storage project id. Defaults to EVIDENCE_MESH_PROJECT_ID.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command != "build":
        raise SystemExit("Run: python embeddings.py build")
    cards = read_cards(args.project_id)
    if not cards:
        raise SystemExit("No cards found. Run indexer.py first.")
    result = build_card_embeddings(cards, force=bool(args.force))
    print(f"Built embedding index for {result.get('card_count', 0)} card(s): {CARD_EMBEDDINGS_PATH}")


if __name__ == "__main__":
    main()
