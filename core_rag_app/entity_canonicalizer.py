from __future__ import annotations

import argparse
import json
import os
import re
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from entity_claims import ENTITY_CLAIM_INDEX_PATH, read_entity_claim_index


LEGACY_ROOT_ENV = "PDF" + "_VISION_RAG_ROOT"
PROJECT_ROOT = Path(
    os.getenv("EVIDENCE_MESH_ROOT")
    or os.getenv(LEGACY_ROOT_ENV)
    or Path(__file__).resolve().parent
).resolve()
INDEXES_DIR = PROJECT_ROOT / "indexes"
CANONICAL_ENTITY_INDEX_PATH = INDEXES_DIR / "canonical_entities.json"

CANONICAL_ENTITY_SCHEMA_VERSION = "canonical_entities.v1.0"
STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "of",
    "for",
    "to",
    "in",
    "on",
    "at",
    "by",
    "with",
    "from",
    "section",
    "part",
    "page",
    "document",
    "pdf",
    "xls",
    "xlsx",
}
ORG_SUFFIXES = {
    "limited",
    "ltd",
    "inc",
    "corp",
    "corporation",
    "company",
    "co",
    "authority",
    "department",
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


def normalize_surface(value: Any) -> str:
    text = str(value or "").lower()
    text = re.sub(r"\.[a-z0-9]{2,5}\b", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    tokens = [token for token in text.split() if token and token not in STOPWORDS]
    return " ".join(tokens)


def token_key(value: Any) -> str:
    tokens = normalize_surface(value).split()
    tokens = [token for token in tokens if token not in ORG_SUFFIXES]
    return " ".join(sorted(dict.fromkeys(tokens)))


def acronym(value: Any) -> str:
    tokens = [
        token
        for token in normalize_surface(value).split()
        if token not in STOPWORDS and token not in ORG_SUFFIXES and not token.isdigit()
    ]
    if len(tokens) < 2:
        return ""
    return "".join(token[0] for token in tokens if token)


def explicit_acronym(value: Any) -> str:
    text = str(value or "").strip()
    compacted = re.sub(r"[^A-Za-z]", "", text)
    if 3 <= len(compacted) <= 10 and compacted.upper() == compacted and re.fullmatch(r"[A-Z. \-]+", text):
        return compacted.lower()
    return ""


def canonical_id_for(key: str) -> str:
    import hashlib

    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
    return f"entity_{digest}"


def choose_canonical_name(names: list[str]) -> str:
    names = sorted({name.strip() for name in names if name.strip()}, key=lambda item: (len(item), item.lower()))
    if not names:
        return "Unknown Entity"
    preferred = sorted(names, key=lambda item: (-len(item.split()), -len(item), item.lower()))
    return preferred[0]


def entity_records(entity_index_path: Path = ENTITY_CLAIM_INDEX_PATH) -> list[dict[str, Any]]:
    payload = read_entity_claim_index(entity_index_path)
    records = []
    for card_id, card_payload in (payload.get("cards", {}) or {}).items():
        if not isinstance(card_payload, dict):
            continue
        for entity in card_payload.get("entities", []) or []:
            if not isinstance(entity, dict):
                continue
            name = str(entity.get("name", "")).strip()
            if not name:
                continue
            aliases = [str(alias).strip() for alias in entity.get("aliases", []) or [] if str(alias).strip()]
            records.append(
                {
                    **entity,
                    "card_id": str(entity.get("card_id") or card_id),
                    "name": name,
                    "aliases": aliases,
                }
            )
    return records


def build_canonical_entity_index(
    *,
    entity_index_path: Path = ENTITY_CLAIM_INDEX_PATH,
    output_path: Path = CANONICAL_ENTITY_INDEX_PATH,
) -> dict[str, Any]:
    records = entity_records(entity_index_path)
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    alias_to_keys: dict[str, set[str]] = defaultdict(set)

    for record in records:
        names = [record.get("name", ""), *(record.get("aliases", []) or [])]
        keys = {token_key(name) for name in names if token_key(name)}
        acronyms = {explicit_acronym(name) for name in names if explicit_acronym(name)}
        for item in acronyms:
            if len(item) >= 3:
                alias_to_keys[item].update(keys)
        primary_key = sorted(keys, key=lambda item: (-len(item.split()), -len(item), item))[0] if keys else normalize_surface(record.get("name", ""))
        groups[primary_key].append(record)

    # Merge groups that share a meaningful acronym.
    parent = {key: key for key in groups}

    def find(key: str) -> str:
        parent.setdefault(key, key)
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    def union(left: str, right: str) -> None:
        root_left = find(left)
        root_right = find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    for acronym_value, keys in alias_to_keys.items():
        keys = {key for key in keys if key in groups}
        if len(acronym_value) < 3 or len(keys) < 2:
            continue
        ordered = sorted(keys)
        for key in ordered[1:]:
            union(ordered[0], key)

    merged: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for key, items in groups.items():
        merged[find(key)].extend(items)

    canonical_entities: dict[str, dict[str, Any]] = {}
    card_entities: dict[str, list[str]] = defaultdict(list)
    alias_lookup: dict[str, str] = {}
    for key, items in merged.items():
        names = []
        types = defaultdict(int)
        cards = set()
        documents = set()
        for item in items:
            names.append(str(item.get("name", "")))
            names.extend(str(alias) for alias in item.get("aliases", []) or [])
            types[str(item.get("type", "other"))] += 1
            if item.get("card_id"):
                cards.add(str(item.get("card_id")))
            if item.get("document_name"):
                documents.add(str(item.get("document_name")))
        canonical_name = choose_canonical_name(names)
        canonical_key = token_key(canonical_name) or key
        canonical_id = canonical_id_for(canonical_key)
        aliases = sorted({name for name in names if name and name.lower() != canonical_name.lower()})
        canonical_entities[canonical_id] = {
            "canonical_id": canonical_id,
            "canonical_name": canonical_name,
            "canonical_key": canonical_key,
            "type": max(types.items(), key=lambda item: item[1])[0] if types else "other",
            "aliases": aliases[:80],
            "mention_count": len(items),
            "card_ids": sorted(cards),
            "document_names": sorted(documents)[:40],
        }
        for card_id in cards:
            card_entities[card_id].append(canonical_id)
        for alias in [canonical_name, *aliases]:
            normalized = normalize_surface(alias)
            if normalized:
                alias_lookup[normalized] = canonical_id
            acronym_value = acronym(alias)
            if acronym_value:
                alias_lookup[acronym_value] = canonical_id

    payload = {
        "schema_version": CANONICAL_ENTITY_SCHEMA_VERSION,
        "created_at": utc_now(),
        "updated_at": utc_now(),
        "entity_count": len(canonical_entities),
        "source_entity_mentions": len(records),
        "entities": canonical_entities,
        "card_entities": {card_id: sorted(set(ids)) for card_id, ids in card_entities.items()},
        "alias_lookup": alias_lookup,
    }
    write_json(output_path, payload)
    return payload


def read_canonical_entity_index(path: Path = CANONICAL_ENTITY_INDEX_PATH) -> dict[str, Any]:
    payload = read_json(path, {})
    if not isinstance(payload, dict):
        return {"schema_version": CANONICAL_ENTITY_SCHEMA_VERSION, "entities": {}, "card_entities": {}, "alias_lookup": {}}
    payload.setdefault("entities", {})
    payload.setdefault("card_entities", {})
    payload.setdefault("alias_lookup", {})
    return payload


def canonical_entity_scores(query: str, path: Path = CANONICAL_ENTITY_INDEX_PATH) -> dict[str, float]:
    payload = read_canonical_entity_index(path)
    entities = payload.get("entities", {})
    if not isinstance(entities, dict) or not entities:
        return {}
    query_norm = normalize_surface(query)
    query_tokens = set(query_norm.split())
    query_acronyms = {token for token in re.findall(r"\b[A-Za-z]{2,}\b", query) if token.isupper()}
    scores: dict[str, float] = {}
    for entity_id, entity in entities.items():
        if not isinstance(entity, dict):
            continue
        names = [entity.get("canonical_name", ""), *(entity.get("aliases", []) or [])]
        best = 0.0
        for name in names:
            name_norm = normalize_surface(name)
            if not name_norm:
                continue
            name_tokens = set(name_norm.split())
            overlap = len(query_tokens & name_tokens) / max(1, len(name_tokens))
            if name_norm and name_norm in query_norm:
                overlap = max(overlap, 1.0)
            name_acronym = acronym(name)
            if name_acronym and name_acronym in query_acronyms:
                overlap = max(overlap, 1.0)
            best = max(best, overlap)
        if best > 0:
            scores[str(entity_id)] = best
    return scores


def canonical_card_scores(query: str, path: Path = CANONICAL_ENTITY_INDEX_PATH) -> dict[str, float]:
    payload = read_canonical_entity_index(path)
    card_entities = payload.get("card_entities", {})
    entity_scores = canonical_entity_scores(query, path)
    if not isinstance(card_entities, dict) or not entity_scores:
        return {}
    scores: dict[str, float] = {}
    for card_id, entity_ids in card_entities.items():
        values = [entity_scores.get(str(entity_id), 0.0) for entity_id in entity_ids or []]
        values = [value for value in values if value > 0]
        if values:
            scores[str(card_id)] = max(values)
    return scores


def canonical_entities_for_card(card_id: str, path: Path = CANONICAL_ENTITY_INDEX_PATH) -> list[dict[str, Any]]:
    payload = read_canonical_entity_index(path)
    entities = payload.get("entities", {})
    ids = payload.get("card_entities", {}).get(card_id, [])
    result = []
    for entity_id in ids[:12]:
        entity = entities.get(str(entity_id))
        if isinstance(entity, dict):
            result.append(
                {
                    "canonical_id": entity.get("canonical_id", ""),
                    "canonical_name": entity.get("canonical_name", ""),
                    "type": entity.get("type", ""),
                    "aliases": entity.get("aliases", [])[:8],
                }
            )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build canonical entity aliases for retrieval.")
    subparsers = parser.add_subparsers(dest="command")
    build = subparsers.add_parser("build", help="Build canonical entity index from entity_claim_index.json.")
    build.add_argument("--entity-index", type=Path, default=ENTITY_CLAIM_INDEX_PATH)
    build.add_argument("--output", type=Path, default=CANONICAL_ENTITY_INDEX_PATH)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command != "build":
        raise SystemExit("Run: python entity_canonicalizer.py build")
    result = build_canonical_entity_index(entity_index_path=args.entity_index, output_path=args.output)
    print(
        f"Built canonical entity index with {result.get('entity_count', 0)} canonical entities "
        f"from {result.get('source_entity_mentions', 0)} mentions: {args.output}"
    )


if __name__ == "__main__":
    main()
