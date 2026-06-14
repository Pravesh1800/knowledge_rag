from __future__ import annotations

import hashlib
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


CARD_INDEX_SCHEMA_VERSION = "card_index.v1.1"
KNOWLEDGE_GRAPH_SCHEMA_VERSION = "knowledge_graph.v1.1"
SEARCH_RESULT_SCHEMA_VERSION = "search_result.v1.1"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_id_part(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value).strip().lower())


def normalize_for_match(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def stable_id(prefix: str, *parts: Any) -> str:
    normalized = json.dumps(parts, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}_{digest}"


def card_stable_id(document_id: str, page_no: int, card_name: str) -> str:
    return stable_id("card", document_id, int(page_no or 0), normalize_for_match(card_name))


def card_id_from_record(card: dict[str, Any]) -> str:
    if card.get("card_id"):
        return str(card["card_id"])
    return card_stable_id(
        str(card.get("document_id", "")),
        int(card.get("page_no") or 0),
        str(card.get("card_name", "")),
    )


def cluster_stable_id(document_id: str, card_ids: list[str], cluster_name: str = "") -> str:
    return stable_id("cluster", document_id, sorted(card_ids), normalize_id_part(cluster_name))


def domain_stable_id(document_id: str, cluster_ids: list[str], domain_name: str = "") -> str:
    return stable_id("domain", document_id, sorted(cluster_ids), normalize_id_part(domain_name))


def relationship_stable_id(
    main_domain_id: str,
    related_domain_id: str,
    relationship_type: str,
    relationship_description: str,
) -> str:
    return stable_id(
        "relationship",
        main_domain_id,
        related_domain_id,
        normalize_id_part(relationship_type),
        normalize_id_part(relationship_description)[:240],
    )


def clamp_score(value: Any, default: float = 0.0) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        score = default
    return max(0.0, min(1.0, score))


def default_relationship_quality(relationship: dict[str, Any]) -> dict[str, Any]:
    document_scope = str(relationship.get("document_scope", "") or "")
    if not document_scope:
        document_scope = (
            "same_document"
            if relationship.get("main_document_id")
            and relationship.get("main_document_id") == relationship.get("related_document_id")
            else "unknown"
        )
    evidence = str(relationship.get("evidence", "")).strip()
    has_cluster_links = bool(relationship.get("cluster_links"))
    has_card_links = bool(relationship.get("card_links"))
    evidence_strength = 0.75 if has_card_links else 0.6 if has_cluster_links else 0.45 if evidence else 0.25
    source_coverage = 0.8 if has_cluster_links and has_card_links else 0.6 if has_card_links else 0.45 if has_cluster_links else 0.25
    generation_method = str(relationship.get("generation_method") or "model_generated")
    confidence_score = 0.72
    if document_scope == "same_document":
        confidence_score += 0.08
    if has_card_links:
        confidence_score += 0.08
    if has_cluster_links:
        confidence_score += 0.05
    if generation_method == "deterministic_fallback":
        confidence_score = min(confidence_score, 0.55)
    return {
        "confidence_score": clamp_score(relationship.get("confidence_score"), confidence_score),
        "evidence_strength": clamp_score(relationship.get("evidence_strength"), evidence_strength),
        "source_coverage": clamp_score(relationship.get("source_coverage"), source_coverage),
        "document_scope": document_scope,
        "generation_method": generation_method,
    }


def migrate_card_index_payload(raw: Any) -> tuple[list[dict[str, Any]], bool]:
    changed = False
    if isinstance(raw, dict):
        cards = raw.get("cards", [])
        changed = raw.get("schema_version") != CARD_INDEX_SCHEMA_VERSION
    elif isinstance(raw, list):
        cards = raw
        changed = True
    else:
        return [], True

    if not isinstance(cards, list):
        return [], True

    migrated: list[dict[str, Any]] = []
    for raw_card in cards:
        if not isinstance(raw_card, dict):
            changed = True
            continue
        card = dict(raw_card)
        if card.get("schema_version") != CARD_INDEX_SCHEMA_VERSION:
            card["schema_version"] = CARD_INDEX_SCHEMA_VERSION
            changed = True
        if not card.get("card_id"):
            card["card_id"] = card_id_from_record(card)
            changed = True
        migrated.append(card)
    return migrated, changed


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f"{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    temp_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    last_error: OSError | None = None
    for attempt in range(8):
        try:
            temp_path.replace(path)
            return
        except PermissionError as exc:
            last_error = exc
            time.sleep(0.15 * (attempt + 1))
    try:
        temp_path.unlink(missing_ok=True)
    finally:
        if last_error is not None:
            raise last_error


def read_card_index(path: Path, persist_migration: bool = False) -> list[dict[str, Any]]:
    raw = read_json(path, [])
    cards, changed = migrate_card_index_payload(raw)
    if persist_migration and changed:
        write_card_index(path, cards, raw)
    return cards


def write_card_index(path: Path, cards: list[dict[str, Any]], previous_payload: Any | None = None) -> None:
    previous_created_at = None
    if isinstance(previous_payload, dict):
        previous_created_at = previous_payload.get("created_at")
    payload = {
        "schema_version": CARD_INDEX_SCHEMA_VERSION,
        "created_at": previous_created_at or utc_now(),
        "updated_at": utc_now(),
        "cards": cards,
    }
    write_json(path, payload)


def migrate_knowledge_graph_payload(
    raw: Any,
    cards: list[dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], bool]:
    changed = False
    if not isinstance(raw, dict):
        raw = {}
        changed = True

    graph = dict(raw)
    if graph.get("schema_version") != KNOWLEDGE_GRAPH_SCHEMA_VERSION:
        graph["schema_version"] = KNOWLEDGE_GRAPH_SCHEMA_VERSION
        changed = True

    cards_by_name = {
        str(card.get("card_name", "")): card
        for card in (cards or [])
        if card.get("card_name")
    }

    clusters: list[dict[str, Any]] = []
    for raw_cluster in graph.get("clusters", []) or []:
        if not isinstance(raw_cluster, dict):
            changed = True
            continue
        cluster = dict(raw_cluster)
        card_names = list(cluster.get("card_names", []) or [])
        card_ids = list(cluster.get("card_ids", []) or [])
        if not card_ids:
            card_ids = [
                card_id_from_record(cards_by_name[name])
                for name in card_names
                if name in cards_by_name
            ]
            cluster["card_ids"] = card_ids
            changed = True
        if not cluster.get("cluster_id"):
            cluster["cluster_id"] = cluster_stable_id(
                str(cluster.get("document_id", "")),
                card_ids,
                str(cluster.get("cluster_name", "")),
            )
            changed = True
        clusters.append(cluster)
    graph["clusters"] = clusters

    clusters_by_name = {
        str(cluster.get("cluster_name", "")): cluster
        for cluster in clusters
        if cluster.get("cluster_name")
    }

    domains: list[dict[str, Any]] = []
    for raw_domain in graph.get("domains", []) or []:
        if not isinstance(raw_domain, dict):
            changed = True
            continue
        domain = dict(raw_domain)
        cluster_names = list(domain.get("cluster_names", []) or [])
        cluster_ids = list(domain.get("cluster_ids", []) or [])
        if not cluster_ids:
            cluster_ids = [
                str(clusters_by_name[name].get("cluster_id", ""))
                for name in cluster_names
                if name in clusters_by_name
            ]
            domain["cluster_ids"] = cluster_ids
            changed = True
        if not domain.get("domain_id"):
            domain["domain_id"] = domain_stable_id(
                str(domain.get("document_id", "")),
                cluster_ids,
                str(domain.get("domain_name", "")),
            )
            changed = True
        domains.append(domain)
    graph["domains"] = domains

    domains_by_name = {
        str(domain.get("domain_name", "")): domain
        for domain in domains
        if domain.get("domain_name")
    }

    relationships: list[dict[str, Any]] = []
    for raw_relationship in graph.get("domain_relationships", []) or []:
        if not isinstance(raw_relationship, dict):
            changed = True
            continue
        relationship = dict(raw_relationship)
        main_domain = domains_by_name.get(str(relationship.get("main_domain", "")))
        related_domain = domains_by_name.get(str(relationship.get("related_domain", "")))
        if main_domain and not relationship.get("main_domain_id"):
            relationship["main_domain_id"] = main_domain.get("domain_id", "")
            changed = True
        if related_domain and not relationship.get("related_domain_id"):
            relationship["related_domain_id"] = related_domain.get("domain_id", "")
            changed = True
        if not relationship.get("relationship_id"):
            relationship["relationship_id"] = relationship_stable_id(
                str(relationship.get("main_domain_id", "")),
                str(relationship.get("related_domain_id", "")),
                str(relationship.get("relationship_type", "other")),
                str(relationship.get("relationship_description", "")),
            )
            changed = True
        quality = default_relationship_quality(relationship)
        for key, value in quality.items():
            if relationship.get(key) != value:
                relationship[key] = value
                changed = True

        for link in relationship.get("cluster_links", []) or []:
            if not isinstance(link, dict):
                continue
            main_cluster = clusters_by_name.get(str(link.get("main_cluster", "")))
            related_cluster = clusters_by_name.get(str(link.get("related_cluster", "")))
            if main_cluster and not link.get("main_cluster_id"):
                link["main_cluster_id"] = main_cluster.get("cluster_id", "")
                changed = True
            if related_cluster and not link.get("related_cluster_id"):
                link["related_cluster_id"] = related_cluster.get("cluster_id", "")
                changed = True

        for link in relationship.get("card_links", []) or []:
            if not isinstance(link, dict):
                continue
            main_card = cards_by_name.get(str(link.get("main_card", "")))
            related_card = cards_by_name.get(str(link.get("related_card", "")))
            if main_card and not link.get("main_card_id"):
                link["main_card_id"] = card_id_from_record(main_card)
                changed = True
            if related_card and not link.get("related_card_id"):
                link["related_card_id"] = card_id_from_record(related_card)
                changed = True

        relationships.append(relationship)
    graph["domain_relationships"] = relationships

    if "relationship_pairs_done" in graph and "relationship_checks_done" not in graph:
        graph["relationship_checks_done"] = graph.get("relationship_pairs_done")
        changed = True
    if "relationship_pairs_total" in graph and "relationship_checks_total" not in graph:
        graph["relationship_checks_total"] = graph.get("relationship_pairs_total")
        changed = True
    if "processed_relationship_pairs" in graph and "processed_relationship_checks" not in graph:
        graph["processed_relationship_checks"] = graph.get("processed_relationship_pairs")
        changed = True
    if "failed_relationship_pairs" in graph and "failed_relationship_checks" not in graph:
        graph["failed_relationship_checks"] = graph.get("failed_relationship_pairs")
        changed = True

    return graph, changed


def read_knowledge_graph(
    path: Path,
    cards: list[dict[str, Any]] | None = None,
    persist_migration: bool = False,
) -> dict[str, Any]:
    raw = read_json(path, {})
    graph, changed = migrate_knowledge_graph_payload(raw, cards)
    if persist_migration and changed and path.exists():
        write_json(path, graph)
    return graph
