from __future__ import annotations

import argparse
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from openai import OpenAI

from cache import read_cache, stable_hash, write_cache
from llm_config import create_chat_client, get_model
from schema import card_id_from_record
from storage import read_cards


LEGACY_ROOT_ENV = "PDF" + "_VISION_RAG_ROOT"
PROJECT_ROOT = Path(
    os.getenv("EVIDENCE_MESH_ROOT")
    or os.getenv(LEGACY_ROOT_ENV)
    or Path(__file__).resolve().parent
).resolve()
INDEXES_DIR = PROJECT_ROOT / "indexes"
ENTITY_CLAIM_INDEX_PATH = INDEXES_DIR / "entity_claim_index.json"

ENTITY_CLAIM_SCHEMA_VERSION = "entity_claim_index.v1.0"
ENTITY_CLAIM_PROMPT_VERSION = "entity_claim_prompt.v1.0"
DEFAULT_ENTITY_CLAIM_CONTENT_CHARS = 2600
DEFAULT_ENTITY_CLAIM_BATCH_SIZE = 1
ALLOWED_ENTITY_TYPES = {
    "organization",
    "person",
    "place",
    "system",
    "document",
    "concept",
    "requirement",
    "date",
    "amount",
    "metric",
    "risk",
    "obligation",
    "exception",
    "other",
}
ALLOWED_CLAIM_TYPES = {
    "fact",
    "requirement",
    "obligation",
    "risk",
    "exception",
    "definition",
    "metric",
    "date",
    "decision",
    "dependency",
    "contradiction",
    "other",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f"{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    temp_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    temp_path.replace(path)


def normalize_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip())


def text_hash(card: dict[str, Any]) -> str:
    payload = {
        "card_name": card.get("card_name", ""),
        "card_description": card.get("card_description", ""),
        "document_name": card.get("document_name", ""),
        "page_no": card.get("page_no"),
        "content": str(card.get("content", ""))[:DEFAULT_ENTITY_CLAIM_CONTENT_CHARS],
        "tags": card.get("tags", []),
    }
    return stable_hash(payload)


def card_prompt_payload(card: dict[str, Any]) -> dict[str, Any]:
    return {
        "card_id": card_id_from_record(card),
        "card_name": card.get("card_name", ""),
        "card_description": card.get("card_description", ""),
        "document_name": card.get("document_name", ""),
        "page_no": card.get("page_no"),
        "tags": card.get("tags", []),
        "content": str(card.get("content", ""))[:DEFAULT_ENTITY_CLAIM_CONTENT_CHARS],
    }


def extraction_prompt(card: dict[str, Any]) -> str:
    return f"""
Extract typed retrieval anchors from this evidence card.

Evidence card:
{json.dumps(card_prompt_payload(card), ensure_ascii=False)}

Return compact, precise anchors only. Do not infer beyond the text.

Entity types:
organization, person, place, system, document, concept, requirement, date, amount, metric, risk, obligation, exception, other

Claim types:
fact, requirement, obligation, risk, exception, definition, metric, date, decision, dependency, contradiction, other

Return only valid JSON:
{{
  "entities": [
    {{
      "name": "canonical entity text",
      "type": "organization|person|place|system|document|concept|requirement|date|amount|metric|risk|obligation|exception|other",
      "aliases": ["short aliases from text"],
      "evidence": "short exact-ish phrase from the card"
    }}
  ],
  "claims": [
    {{
      "claim": "atomic factual statement, requirement, obligation, metric, date, risk, exception, or definition",
      "type": "fact|requirement|obligation|risk|exception|definition|metric|date|decision|dependency|contradiction|other",
      "subject": "main entity or topic",
      "object": "target, value, condition, date, amount, or related entity",
      "evidence": "short source phrase",
      "confidence": 0.0
    }}
  ]
}}
""".strip()


def parse_json_response(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def clamp_score(value: Any, default: float = 0.7) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        score = default
    return max(0.0, min(1.0, score))


def clean_entity(raw: dict[str, Any], card: dict[str, Any]) -> dict[str, Any] | None:
    name = normalize_text(raw.get("name"))
    if not name:
        return None
    entity_type = normalize_text(raw.get("type")).lower()
    if entity_type not in ALLOWED_ENTITY_TYPES:
        entity_type = "other"
    aliases = [
        normalize_text(alias)
        for alias in raw.get("aliases", []) or []
        if normalize_text(alias) and normalize_text(alias).lower() != name.lower()
    ][:6]
    return {
        "name": name[:180],
        "type": entity_type,
        "aliases": aliases,
        "evidence": normalize_text(raw.get("evidence"))[:280],
        "card_id": card_id_from_record(card),
        "card_name": card.get("card_name", ""),
        "document_id": card.get("document_id", ""),
        "document_name": card.get("document_name", ""),
        "page_no": int(card.get("page_no") or 0),
    }


def clean_claim(raw: dict[str, Any], card: dict[str, Any]) -> dict[str, Any] | None:
    claim = normalize_text(raw.get("claim"))
    if not claim:
        return None
    claim_type = normalize_text(raw.get("type")).lower()
    if claim_type not in ALLOWED_CLAIM_TYPES:
        claim_type = "other"
    return {
        "claim": claim[:420],
        "type": claim_type,
        "subject": normalize_text(raw.get("subject"))[:180],
        "object": normalize_text(raw.get("object"))[:220],
        "evidence": normalize_text(raw.get("evidence"))[:320],
        "confidence": clamp_score(raw.get("confidence"), 0.72),
        "card_id": card_id_from_record(card),
        "card_name": card.get("card_name", ""),
        "document_id": card.get("document_id", ""),
        "document_name": card.get("document_name", ""),
        "page_no": int(card.get("page_no") or 0),
    }


def heuristic_entities(card: dict[str, Any]) -> list[dict[str, Any]]:
    text = " ".join(
        [
            str(card.get("card_name", "")),
            str(card.get("card_description", "")),
            str(card.get("document_name", "")),
            str(card.get("content", ""))[:DEFAULT_ENTITY_CLAIM_CONTENT_CHARS],
        ]
    )
    candidates: list[dict[str, Any]] = []
    patterns = [
        (r"\b[A-Z][A-Za-z0-9&./()-]*(?:\s+[A-Z][A-Za-z0-9&./()-]*){1,5}\b", "concept"),
        (r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b\d{4}-\d{2}-\d{2}\b", "date"),
        (r"\b(?:Rs\.?|INR|USD|\$)\s?[\d,]+(?:\.\d+)?\b|\b\d+(?:\.\d+)?\s?(?:%|percent|days|months|years)\b", "metric"),
    ]
    seen = set()
    for pattern, entity_type in patterns:
        for match in re.finditer(pattern, text):
            value = normalize_text(match.group(0))
            key = value.lower()
            if len(value) < 3 or key in seen:
                continue
            seen.add(key)
            entity = clean_entity(
                {
                    "name": value,
                    "type": entity_type,
                    "aliases": [],
                    "evidence": value,
                },
                card,
            )
            if entity:
                candidates.append(entity)
            if len(candidates) >= 18:
                return candidates
    return candidates


def heuristic_claims(card: dict[str, Any]) -> list[dict[str, Any]]:
    content = str(card.get("content", ""))
    sentences = [
        normalize_text(sentence)
        for sentence in re.split(r"(?<=[.!?])\s+|\n+", content)
        if len(normalize_text(sentence)) >= 35
    ]
    claims = []
    for sentence in sentences[:8]:
        lowered = sentence.lower()
        claim_type = "other"
        if any(term in lowered for term in ("must", "shall", "required", "requirement")):
            claim_type = "requirement"
        elif any(term in lowered for term in ("risk", "delay", "penalty", "liable", "failure")):
            claim_type = "risk"
        elif any(term in lowered for term in ("except", "unless", "exemption")):
            claim_type = "exception"
        elif re.search(r"\b\d+(?:\.\d+)?\s?(?:%|percent|days|months|years)\b", lowered):
            claim_type = "metric"
        claim = clean_claim(
            {
                "claim": sentence,
                "type": claim_type,
                "subject": card.get("card_name", ""),
                "object": "",
                "evidence": sentence[:220],
                "confidence": 0.55,
            },
            card,
        )
        if claim:
            claims.append(claim)
    return claims


def extract_card_anchors(
    card: dict[str, Any],
    *,
    client: OpenAI | None,
    model: str,
    dry_run: bool = False,
) -> dict[str, Any]:
    card_hash = text_hash(card)
    prompt = extraction_prompt(card)
    cache_key = {
        "version": ENTITY_CLAIM_PROMPT_VERSION,
        "card_id": card_id_from_record(card),
        "text_hash": card_hash,
        "model": model,
        "prompt_hash": stable_hash(prompt),
    }
    raw: dict[str, Any] = {}
    if not dry_run and client is not None:
        cached = read_cache("entity_claim_extraction", cache_key)
        if cached is not None:
            raw = cached.get("value", {})
        else:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": "You extract typed retrieval anchors from document evidence. Return only valid JSON.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
                response_format={"type": "json_object"},
                max_tokens=int(os.getenv("EVIDENCE_MESH_ENTITY_CLAIM_MAX_TOKENS", "1800")),
            )
            raw = parse_json_response(response.choices[0].message.content or "{}")
            write_cache("entity_claim_extraction", cache_key, raw)

    entities = [
        entity
        for raw_entity in raw.get("entities", []) or []
        if isinstance(raw_entity, dict)
        for entity in [clean_entity(raw_entity, card)]
        if entity is not None
    ]
    claims = [
        claim
        for raw_claim in raw.get("claims", []) or []
        if isinstance(raw_claim, dict)
        for claim in [clean_claim(raw_claim, card)]
        if claim is not None
    ]
    if not entities:
        entities = heuristic_entities(card)
    if not claims:
        claims = heuristic_claims(card)
    return {
        "card_id": card_id_from_record(card),
        "text_hash": card_hash,
        "model": model if not dry_run else "heuristic",
        "updated_at": utc_now(),
        "entities": entities[:24],
        "claims": claims[:16],
    }


def read_entity_claim_index(path: Path = ENTITY_CLAIM_INDEX_PATH) -> dict[str, Any]:
    payload = read_json(path, {})
    if not isinstance(payload, dict):
        return {"schema_version": ENTITY_CLAIM_SCHEMA_VERSION, "cards": {}}
    if not isinstance(payload.get("cards"), dict):
        payload["cards"] = {}
    return payload


def write_entity_claim_index(payload: dict[str, Any], path: Path = ENTITY_CLAIM_INDEX_PATH) -> None:
    payload["schema_version"] = ENTITY_CLAIM_SCHEMA_VERSION
    payload["updated_at"] = utc_now()
    write_json(path, payload)


def build_entity_claim_index(
    cards: list[dict[str, Any]],
    *,
    path: Path = ENTITY_CLAIM_INDEX_PATH,
    force: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    client = None
    model = get_model("map")
    if not dry_run:
        client, _default_model, _provider = create_chat_client()
        model = get_model("map")
    payload = read_entity_claim_index(path)
    indexed_cards: dict[str, Any] = payload.get("cards", {})
    valid_ids = set()
    checkpoint_every = max(1, int(os.getenv("EVIDENCE_MESH_ENTITY_CLAIM_CHECKPOINT_EVERY", "100")))
    changed_since_checkpoint = 0
    for card in cards:
        card_id = card_id_from_record(card)
        valid_ids.add(card_id)
        current_hash = text_hash(card)
        existing = indexed_cards.get(card_id)
        if not force and isinstance(existing, dict) and existing.get("text_hash") == current_hash:
            continue
        indexed_cards[card_id] = extract_card_anchors(
            card,
            client=client,
            model=model,
            dry_run=dry_run,
        )
        changed_since_checkpoint += 1
        if changed_since_checkpoint >= checkpoint_every:
            write_entity_claim_index(
                {
                    **payload,
                    "model": model if not dry_run else "heuristic",
                    "card_count": len(indexed_cards),
                    "cards": indexed_cards,
                },
                path,
            )
            changed_since_checkpoint = 0
    for card_id in list(indexed_cards):
        if card_id not in valid_ids:
            indexed_cards.pop(card_id, None)
    result = {
        **payload,
        "model": model if not dry_run else "heuristic",
        "card_count": len(indexed_cards),
        "cards": indexed_cards,
    }
    write_entity_claim_index(result, path)
    return result


def keyword_score(query: str, text: str) -> float:
    query_terms = {term for term in re.findall(r"[a-z0-9]+", query.lower()) if len(term) > 2}
    if not query_terms:
        return 0.0
    text_terms = set(re.findall(r"[a-z0-9]+", text.lower()))
    return len(query_terms & text_terms) / len(query_terms)


def entity_claim_card_scores(
    query: str,
    cards: list[dict[str, Any]],
    path: Path = ENTITY_CLAIM_INDEX_PATH,
) -> dict[str, float]:
    payload = read_entity_claim_index(path)
    indexed_cards = payload.get("cards", {})
    if not isinstance(indexed_cards, dict) or not indexed_cards:
        return {}
    scores: dict[str, float] = {}
    for card in cards:
        card_id = card_id_from_record(card)
        anchors = indexed_cards.get(card_id)
        if not isinstance(anchors, dict):
            continue
        entity_text = " ".join(
            " ".join(
                [
                    str(entity.get("name", "")),
                    str(entity.get("type", "")),
                    " ".join(str(alias) for alias in entity.get("aliases", []) or []),
                    str(entity.get("evidence", "")),
                ]
            )
            for entity in anchors.get("entities", []) or []
            if isinstance(entity, dict)
        )
        claim_text = " ".join(
            " ".join(
                [
                    str(claim.get("claim", "")),
                    str(claim.get("type", "")),
                    str(claim.get("subject", "")),
                    str(claim.get("object", "")),
                    str(claim.get("evidence", "")),
                ]
            )
            for claim in anchors.get("claims", []) or []
            if isinstance(claim, dict)
        )
        entity_score = keyword_score(query, entity_text)
        claim_score = keyword_score(query, claim_text)
        score = (0.42 * entity_score) + (0.58 * claim_score)
        if score > 0:
            scores[card_id] = score
    return scores


def anchors_for_card(card_id: str, path: Path = ENTITY_CLAIM_INDEX_PATH) -> dict[str, Any]:
    payload = read_entity_claim_index(path)
    item = payload.get("cards", {}).get(card_id, {})
    return item if isinstance(item, dict) else {}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build typed entity and claim indexes for retrieval.")
    subparsers = parser.add_subparsers(dest="command")
    build = subparsers.add_parser("build", help="Build missing or changed entity/claim anchors.")
    build.add_argument("--force", action="store_true", help="Rebuild all anchors even if hashes match.")
    build.add_argument("--dry-run", action="store_true", help="Use deterministic heuristic anchors instead of LLM extraction.")
    build.add_argument("--project-id", help="Storage project id. Defaults to EVIDENCE_MESH_PROJECT_ID.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command != "build":
        raise SystemExit("Run: python entity_claims.py build")
    cards = read_cards(args.project_id)
    if not cards:
        raise SystemExit("No cards found. Run indexer.py first.")
    result = build_entity_claim_index(cards, force=bool(args.force), dry_run=bool(args.dry_run))
    print(f"Built entity/claim index for {result.get('card_count', 0)} card(s): {ENTITY_CLAIM_INDEX_PATH}")


if __name__ == "__main__":
    main()
