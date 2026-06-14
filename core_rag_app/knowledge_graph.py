from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from openai import OpenAI

from cache import read_cache, stable_hash, write_cache
from community_summaries import build_community_summaries
from llm_config import create_chat_client, get_model
from schema import KNOWLEDGE_GRAPH_SCHEMA_VERSION, clamp_score, read_knowledge_graph
from storage import (
    read_cards,
    read_knowledge_graph_state,
    read_relationship_pair_checks,
    read_relationship_payloads,
    record_relationship_pair_check,
    record_graph_build_run,
    sync_knowledge_graph,
    write_graph_audit_state,
    write_knowledge_graph_state,
    write_pipeline_progress_state,
)


LEGACY_ROOT_ENV = "PDF" + "_VISION_RAG_ROOT"
PROJECT_ROOT = Path(
    os.getenv("EVIDENCE_MESH_ROOT")
    or os.getenv(LEGACY_ROOT_ENV)
    or Path(__file__).resolve().parent
).resolve()
INDEXES_DIR = PROJECT_ROOT / "indexes"
CARD_INDEX_PATH = INDEXES_DIR / "card_index.json"
KNOWLEDGE_GRAPH_PATH = INDEXES_DIR / "knowledge_graph.json"
GRAPH_AUDIT_PATH = INDEXES_DIR / "graph_audit.json"
PIPELINE_PROGRESS_PATH = PROJECT_ROOT / "logs" / "pipeline_progress.json"
RELATIONSHIP_METRICS_PATH = PROJECT_ROOT / "logs" / "relationship_metrics.jsonl"

DEFAULT_RELATIONSHIP_WORKERS = 1
MAX_RELATIONSHIP_PAIR_ATTEMPTS = 3
LEGACY_PROCESSED_RELATIONSHIP_CHECKS = "processed_relationship" + "_pairs"
CLUSTER_PROMPT_VERSION = "cluster_prompt.v1.0"
DOMAIN_PROMPT_VERSION = "domain_prompt.v1.0"
RELATIONSHIP_PROMPT_VERSION = "relationship_prompt.v1.2"
METRICS_LOCK = threading.Lock()
DEFAULT_FULL_CHECKPOINT_EVERY_CHECKS = 25
DEFAULT_FULL_CHECKPOINT_EVERY_SECONDS = 60.0


@dataclass
class Cluster:
    cluster_id: str
    cluster_name: str
    cluster_description: str
    document_id: str
    document_name: str
    card_ids: list[str]
    card_names: list[str]


@dataclass
class Domain:
    domain_id: str
    domain_name: str
    domain_description: str
    document_id: str
    document_name: str
    cluster_ids: list[str]
    cluster_names: list[str]


@dataclass
class DomainRelationship:
    relationship_id: str
    main_domain_id: str
    related_domain_id: str
    main_domain: str
    related_domain: str
    relationship_type: str
    relationship_description: str
    evidence: str
    cluster_links: list[dict[str, str]]
    card_links: list[dict[str, str]]
    document_scope: str
    confidence_score: float
    evidence_strength: float
    source_coverage: float
    generation_method: str


def load_dotenv() -> None:
    protected_keys = {"EVIDENCE_MESH_ROOT", LEGACY_ROOT_ENV, "EVIDENCE_MESH_PROJECT_ID"}
    env_paths = [PROJECT_ROOT / ".env", Path(__file__).resolve().parent / ".env"]
    for env_path in env_paths:
        if not env_path.exists():
            continue
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            if key in protected_keys and os.getenv(key):
                continue
            os.environ[key] = value.strip().strip("\"'")


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f"{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    temp_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    last_error: OSError | None = None
    for attempt in range(30):
        try:
            temp_path.replace(path)
            return
        except PermissionError as exc:
            last_error = exc
            time.sleep(min(2.0, 0.15 * (attempt + 1)))
    try:
        temp_path.unlink(missing_ok=True)
    finally:
        if last_error is not None:
            raise last_error


def append_metric(event: dict[str, Any]) -> None:
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "pid": os.getpid(),
        **event,
    }
    RELATIONSHIP_METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(payload, ensure_ascii=False)
    with METRICS_LOCK:
        with RELATIONSHIP_METRICS_PATH.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")


def write_pipeline_progress(data: dict[str, Any]) -> None:
    payload = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        **data,
    }
    write_pipeline_progress_state(payload)
    try:
        write_json(PIPELINE_PROGRESS_PATH, payload)
    except OSError as exc:
        print(f"Warning: could not export pipeline_progress.json: {exc}")


def relationship_checkpoint_settings() -> tuple[int, float]:
    try:
        every_checks = int(os.getenv("RELATIONSHIP_FULL_CHECKPOINT_EVERY_CHECKS", str(DEFAULT_FULL_CHECKPOINT_EVERY_CHECKS)))
    except ValueError:
        every_checks = DEFAULT_FULL_CHECKPOINT_EVERY_CHECKS
    try:
        every_seconds = float(os.getenv("RELATIONSHIP_FULL_CHECKPOINT_EVERY_SECONDS", str(DEFAULT_FULL_CHECKPOINT_EVERY_SECONDS)))
    except ValueError:
        every_seconds = DEFAULT_FULL_CHECKPOINT_EVERY_SECONDS
    return max(1, every_checks), max(5.0, every_seconds)


def write_relationship_lightweight_progress(
    *,
    status: str,
    cluster_count: int,
    domain_count: int,
    relationship_count: int,
    relationship_checks_done: int,
    relationship_checks_total: int,
    failed_pair_errors: dict[str, str] | None = None,
) -> None:
    started = time.perf_counter()
    write_pipeline_progress(
        {
            "stage": "knowledge_graph" if status != "complete" else "complete",
            "message": f"Knowledge graph status: {status}.",
            "knowledge_graph_status": status,
            "cluster_count": cluster_count,
            "domain_count": domain_count,
            "relationship_count": relationship_count,
            "relationship_checks_done": relationship_checks_done,
            "relationship_checks_total": relationship_checks_total,
            "failed_relationship_check_count": len(failed_pair_errors or {}),
        }
    )
    append_metric(
        {
            "event": "relationship_lightweight_progress",
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "status": status,
            "relationship_checks_done": relationship_checks_done,
            "relationship_checks_total": relationship_checks_total,
            "relationship_count": relationship_count,
        }
    )


def clean_name(value: str, fallback: str) -> str:
    value = re.sub(r"\s+", " ", str(value).strip())
    value = re.sub(r"[^a-zA-Z0-9 _./&()-]+", "", value)
    return value[:120] or fallback


def normalize_id_part(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value).strip().lower())


def normalize_card_id_part(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value).lower()).strip()


def stable_id(prefix: str, *parts: Any) -> str:
    normalized = json.dumps(parts, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}_{digest}"


def card_stable_id(card: dict[str, Any]) -> str:
    if card.get("card_id"):
        return str(card["card_id"])
    return stable_id(
        "card",
        str(card.get("document_id", "")),
        int(card.get("page_no") or 0),
        normalize_card_id_part(card.get("card_name", "")),
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


def relationship_quality(
    *,
    document_scope: str,
    cluster_links: list[dict[str, str]],
    card_links: list[dict[str, str]],
    evidence: str,
    generation_method: str,
    confidence_score: Any = None,
    evidence_strength: Any = None,
    source_coverage: Any = None,
) -> dict[str, Any]:
    has_cluster_links = bool(cluster_links)
    has_card_links = bool(card_links)
    default_evidence_strength = (
        0.82
        if has_card_links and evidence
        else 0.68
        if has_card_links or has_cluster_links
        else 0.45
        if evidence
        else 0.25
    )
    default_source_coverage = (
        0.85
        if has_card_links and has_cluster_links
        else 0.65
        if has_card_links
        else 0.52
        if has_cluster_links
        else 0.3
    )
    default_confidence = 0.72
    if document_scope == "same_document":
        default_confidence += 0.08
    if has_card_links:
        default_confidence += 0.08
    if has_cluster_links:
        default_confidence += 0.05
    if generation_method == "deterministic_fallback":
        default_confidence = min(default_confidence, 0.55)
        default_evidence_strength = min(default_evidence_strength, 0.5)

    return {
        "confidence_score": clamp_score(confidence_score, default_confidence),
        "evidence_strength": clamp_score(evidence_strength, default_evidence_strength),
        "source_coverage": clamp_score(source_coverage, default_source_coverage),
        "document_scope": document_scope,
        "generation_method": generation_method,
    }


def normalized_name(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value).lower()).strip()


def is_suspiciously_generic_name(value: str, node_type: str) -> bool:
    normalized = normalized_name(value)
    if len(normalized) < 4:
        return True
    generic_names = {
        "untitled",
        "untitled card",
        "card",
        "cluster",
        "domain",
        "document",
        "document overview",
        "general",
        "general information",
        "misc",
        "miscellaneous",
        "other",
        "text based cards",
        "image based cards",
    }
    if normalized in generic_names:
        return True
    if node_type == "card" and re.fullmatch(r"(page|section|item|content)\s*\d*", normalized):
        return True
    if node_type in {"cluster", "domain"} and re.fullmatch(r"(cluster|domain|group|section)\s*\d*", normalized):
        return True
    return False


def audit_issue(
    category: str,
    severity: str,
    message: str,
    **details: Any,
) -> dict[str, Any]:
    return {
        "category": category,
        "severity": severity,
        "message": message,
        **details,
    }


def graph_audit_status(issue_count: int, high_count: int, medium_count: int) -> str:
    if high_count:
        return "fail"
    if medium_count:
        return "warn"
    if issue_count:
        return "notice"
    return "pass"


def hash_payload(payload: Any) -> str:
    normalized = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def source_manifest_from_cards(cards: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped = group_cards_by_document(cards)
    snapshot: list[dict[str, Any]] = []
    for document_id, document_cards in sorted(grouped.items()):
        card_signature_payload = [
            {
                "card_id": card_stable_id(card),
                "card_name": card.get("card_name", ""),
                "card_description": card.get("card_description", ""),
                "page_no": int(card.get("page_no") or 0),
                "content": card.get("content", ""),
                "tags": card.get("tags", []),
            }
            for card in sorted(
                document_cards,
                key=lambda item: (
                    int(item.get("page_no") or 0),
                    str(item.get("card_id") or card_stable_id(item)),
                ),
            )
        ]
        snapshot.append(
            {
                "document_id": document_id,
                "document_name": document_cards[0].get("document_name", "Unknown Document"),
                "page_count": len({int(card.get("page_no") or 0) for card in document_cards}),
                "card_count": len(document_cards),
                "card_signature": hash_payload(card_signature_payload),
            }
        )
    return snapshot


def source_manifest_hash(source_manifest: list[dict[str, Any]]) -> str:
    return hash_payload(source_manifest)


def card_set_hash(cards: list[dict[str, Any]]) -> str:
    payload = [
        {
            "card_id": card_stable_id(card),
            "card_name": card.get("card_name", ""),
            "card_description": card.get("card_description", ""),
            "document_id": card.get("document_id", ""),
            "document_name": card.get("document_name", ""),
            "page_no": card.get("page_no"),
            "content": card.get("content", ""),
            "tags": card.get("tags", []),
        }
        for card in sorted(cards, key=lambda item: str(item.get("card_id") or card_stable_id(item)))
    ]
    return stable_hash(payload)


def cluster_set_hash(clusters: list[Cluster]) -> str:
    payload = [
        {
            "cluster_id": cluster.cluster_id,
            "cluster_name": cluster.cluster_name,
            "cluster_description": cluster.cluster_description,
            "document_id": cluster.document_id,
            "card_ids": sorted(cluster.card_ids),
            "card_names": sorted(cluster.card_names),
        }
        for cluster in sorted(clusters, key=lambda item: item.cluster_id)
    ]
    return stable_hash(payload)


def domain_pair_hash(main: dict[str, Any], related: dict[str, Any]) -> str:
    payload = {
        "main": main,
        "related": related,
    }
    return stable_hash(payload)


def base_name(name: str) -> str:
    return re.sub(r"_v\d+$", "", name).strip()


def unique_versioned_name(name: str, used_names: set[str]) -> str:
    cleaned = clean_name(name, "Untitled")
    if cleaned not in used_names:
        used_names.add(cleaned)
        return cleaned

    root = base_name(cleaned)
    version = 2
    while f"{root}_v{version}" in used_names:
        version += 1
    versioned = f"{root}_v{version}"
    used_names.add(versioned)
    return versioned


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
        candidate = match.group(0)
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            repaired = re.sub(r",\s*([}\]])", r"\1", candidate)
            repaired = re.sub(r"(?<!\\)'", '"', repaired)
            return json.loads(repaired)


def normalize_links(raw_links: Any, required_keys: list[str]) -> list[dict[str, str]]:
    if not isinstance(raw_links, list):
        return []
    normalized: list[dict[str, str]] = []
    for raw in raw_links:
        if not isinstance(raw, dict):
            continue
        item = {key: str(raw.get(key, "")).strip() for key in required_keys}
        if any(item.values()):
            normalized.append(item)
    return normalized


def group_cards_by_document(cards: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for card in cards:
        grouped[card.get("document_id", "unknown")].append(card)
    return dict(grouped)


def card_payload(cards: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "card_id": card_stable_id(card),
            "card_name": card.get("card_name", ""),
            "card_description": card.get("card_description", ""),
            "page_no": card.get("page_no"),
            "card_source": card.get("card_source", ""),
            "tags": card.get("tags", []),
            "content_excerpt": str(card.get("content", ""))[:700],
        }
        for card in cards
    ]


def clusters_prompt(document_name: str, cards: list[dict[str, Any]]) -> str:
    return f"""
Create card clusters within one document.

Document: {document_name}

Cards:
{json.dumps(card_payload(cards), ensure_ascii=False)}

Rules:
1. Group similar or strongly related cards into clusters.
2. A cluster should represent a coherent section, theme, workstream, evidence group, or conceptual area.
3. Every card must appear in exactly one cluster.
4. Use the exact card names provided.
5. Cluster names must be short, stable, and descriptive.
6. Do not create duplicate cluster names in this response.

Return only valid JSON:
{{
  "clusters": [
    {{
      "cluster_name": "Cluster Name",
      "cluster_description": "What this cluster represents",
      "card_names": ["Exact Card Name"]
    }}
  ]
}}
""".strip()


def domains_prompt(document_name: str, clusters: list[dict[str, Any]]) -> str:
    payload = [
        {
            "cluster_name": cluster["cluster_name"],
            "cluster_description": cluster["cluster_description"],
            "card_names": cluster["card_names"],
        }
        for cluster in clusters
    ]
    return f"""
Group related clusters into larger domains within one document.

Document: {document_name}

Clusters:
{json.dumps(payload, ensure_ascii=False)}

Rules:
1. Group similar or connected clusters into domains.
2. A domain should represent a broad document area, major theme, business domain, or major analytical region.
3. Every cluster must appear in exactly one domain.
4. Use the exact cluster names provided.
5. Domain names must be short, stable, and descriptive.
6. Do not create duplicate domain names in this response.

Return only valid JSON:
{{
  "domains": [
    {{
      "domain_name": "Domain Name",
      "domain_description": "What this domain represents",
      "cluster_names": ["Exact Cluster Name"]
    }}
  ]
}}
""".strip()


def fallback_cluster_prompt(document_name: str, cards: list[dict[str, Any]]) -> str:
    return f"""
Name and describe a cluster for these leftover cards.

Document: {document_name}

Cards:
{json.dumps(card_payload(cards), ensure_ascii=False)}

Rules:
1. Do not use words like unassigned, miscellaneous, other, leftover, or uncategorized.
2. The name must help search find these cards.
3. Prefer domain terms visible in the cards.
4. Keep the name short and descriptive.

Return only valid JSON:
{{
  "cluster_name": "Specific Searchable Cluster Name",
  "cluster_description": "Concrete description of the cards and when this cluster is relevant"
}}
""".strip()


def fallback_domain_prompt(document_name: str, clusters: list[Cluster]) -> str:
    payload = [
        {
            "cluster_name": cluster.cluster_name,
            "cluster_description": cluster.cluster_description,
            "card_names": cluster.card_names,
        }
        for cluster in clusters
    ]
    return f"""
Name and describe a domain for these leftover clusters.

Document: {document_name}

Clusters:
{json.dumps(payload, ensure_ascii=False)}

Rules:
1. Do not use words like unassigned, miscellaneous, other, leftover, or uncategorized.
2. The name must help search find these clusters.
3. Prefer domain terms visible in the clusters.
4. Keep the name short and descriptive.

Return only valid JSON:
{{
  "domain_name": "Specific Searchable Domain Name",
  "domain_description": "Concrete description of the clusters and when this domain is relevant"
}}
""".strip()


def common_terms(text: str, limit: int = 5) -> list[str]:
    stopwords = {
        "the", "and", "for", "with", "from", "this", "that", "shall", "work",
        "document", "card", "cards", "page", "details", "detail", "system",
        "supply", "providing", "installation", "complete", "required", "under",
    }
    words = re.findall(r"[A-Za-z][A-Za-z0-9&/-]{2,}", text)
    counts: dict[str, int] = defaultdict(int)
    display: dict[str, str] = {}
    for word in words:
        key = word.lower().strip("-/")
        if key in stopwords or len(key) < 3:
            continue
        counts[key] += 1
        display.setdefault(key, word.strip("-/"))
    ranked = sorted(counts, key=lambda key: (-counts[key], key))
    return [display[key] for key in ranked[:limit]]


def local_fallback_cluster(cards: list[dict[str, Any]]) -> tuple[str, str]:
    text = " ".join(
        [
            str(card.get("card_name", "")) + " "
            + str(card.get("card_description", "")) + " "
            + str(card.get("content", ""))[:300]
            for card in cards
        ]
    )
    terms = common_terms(text)
    name = " ".join(terms[:4]) if terms else "Document Card Cluster"
    description = (
        f"Cards covering {', '.join(terms)}."
        if terms
        else "Cards grouped together because they were not captured by narrower generated clusters."
    )
    return name, description


def local_fallback_domain(clusters: list[Cluster]) -> tuple[str, str]:
    text = " ".join(
        [
            cluster.cluster_name + " "
            + cluster.cluster_description + " "
            + " ".join(cluster.card_names[:20])
            for cluster in clusters
        ]
    )
    terms = common_terms(text)
    name = " ".join(terms[:4]) if terms else "Document Knowledge Area"
    description = (
        f"Domain covering clusters about {', '.join(terms)}."
        if terms
        else "Domain grouping related document clusters for search and retrieval."
    )
    return name, description


def domain_pair_key(main_domain: str, related_domain: str) -> str:
    return f"{main_domain} -> {related_domain}"


def domain_names_from_pair_key(pair_key: str) -> tuple[str, str] | None:
    if " -> " not in pair_key:
        return None
    main, related = pair_key.split(" -> ", 1)
    main = main.strip()
    related = related.strip()
    if not main or not related:
        return None
    return main, related


def cluster_from_dict(raw: dict[str, Any], cards_by_name: dict[str, dict[str, Any]]) -> Cluster:
    card_names = list(raw.get("card_names", []))
    card_ids = list(raw.get("card_ids", [])) or [
        str(cards_by_name[name].get("card_id") or card_stable_id(cards_by_name[name]))
        for name in card_names
        if name in cards_by_name
    ]
    return Cluster(
        cluster_id=(
            str(raw.get("cluster_id", "")).strip()
            or cluster_stable_id(
                str(raw.get("document_id", "")).strip(),
                card_ids,
                str(raw.get("cluster_name", "")).strip(),
            )
        ),
        cluster_name=str(raw.get("cluster_name", "")).strip(),
        cluster_description=str(raw.get("cluster_description", "")).strip(),
        document_id=str(raw.get("document_id", "")).strip(),
        document_name=str(raw.get("document_name", "")).strip(),
        card_ids=card_ids,
        card_names=card_names,
    )


def domain_from_dict(raw: dict[str, Any], clusters_by_name: dict[str, Cluster]) -> Domain:
    cluster_names = list(raw.get("cluster_names", []))
    cluster_ids = list(raw.get("cluster_ids", [])) or [
        clusters_by_name[name].cluster_id
        for name in cluster_names
        if name in clusters_by_name
    ]
    return Domain(
        domain_id=(
            str(raw.get("domain_id", "")).strip()
            or domain_stable_id(
                str(raw.get("document_id", "")).strip(),
                cluster_ids,
                str(raw.get("domain_name", "")).strip(),
            )
        ),
        domain_name=str(raw.get("domain_name", "")).strip(),
        domain_description=str(raw.get("domain_description", "")).strip(),
        document_id=str(raw.get("document_id", "")).strip(),
        document_name=str(raw.get("document_name", "")).strip(),
        cluster_ids=cluster_ids,
        cluster_names=cluster_names,
    )


def relationship_from_dict(raw: dict[str, Any]) -> DomainRelationship:
    main_domain_id = str(raw.get("main_domain_id", "")).strip()
    related_domain_id = str(raw.get("related_domain_id", "")).strip()
    relationship_type = str(raw.get("relationship_type", "other")).strip() or "other"
    relationship_description = str(raw.get("relationship_description", "")).strip()
    evidence = str(raw.get("evidence", "")).strip()
    cluster_links = raw.get("cluster_links", []) if isinstance(raw.get("cluster_links"), list) else []
    card_links = raw.get("card_links", []) if isinstance(raw.get("card_links"), list) else []
    document_scope = str(raw.get("document_scope", "")).strip() or "unknown"
    quality = relationship_quality(
        document_scope=document_scope,
        cluster_links=cluster_links,
        card_links=card_links,
        evidence=evidence,
        generation_method=str(raw.get("generation_method") or "model_generated"),
        confidence_score=raw.get("confidence_score"),
        evidence_strength=raw.get("evidence_strength"),
        source_coverage=raw.get("source_coverage"),
    )
    return DomainRelationship(
        relationship_id=(
            str(raw.get("relationship_id", "")).strip()
            or relationship_stable_id(
                main_domain_id or str(raw.get("main_domain", "")).strip(),
                related_domain_id or str(raw.get("related_domain", "")).strip(),
                relationship_type,
                relationship_description,
            )
        ),
        main_domain_id=main_domain_id,
        related_domain_id=related_domain_id,
        main_domain=str(raw.get("main_domain", "")).strip(),
        related_domain=str(raw.get("related_domain", "")).strip(),
        relationship_type=relationship_type,
        relationship_description=relationship_description,
        evidence=evidence,
        cluster_links=cluster_links,
        card_links=card_links,
        document_scope=str(quality["document_scope"]),
        confidence_score=float(quality["confidence_score"]),
        evidence_strength=float(quality["evidence_strength"]),
        source_coverage=float(quality["source_coverage"]),
        generation_method=str(quality["generation_method"]),
    )


def semantic_fallback_cluster(
    client: OpenAI | None,
    model: str,
    document_name: str,
    cards: list[dict[str, Any]],
) -> tuple[str, str]:
    if client is not None:
        try:
            prompt = fallback_cluster_prompt(document_name, cards)
            cache_key = {
                "version": CLUSTER_PROMPT_VERSION,
                "schema_version": KNOWLEDGE_GRAPH_SCHEMA_VERSION,
                "model": model,
                "document_name": document_name,
                "card_set_hash": card_set_hash(cards),
                "prompt_hash": stable_hash(prompt),
                "fallback": True,
            }
            cached = read_cache("cluster_generation", cache_key)
            if cached is not None:
                result = cached.get("value", {})
            else:
                result = openrouter_json(client, model, prompt)
                write_cache("cluster_generation", cache_key, result)
            name = str(result.get("cluster_name", "")).strip()
            description = str(result.get("cluster_description", "")).strip()
            if name and description:
                return name, description
        except Exception as exc:
            print(f"Warning: fallback cluster naming failed; using local name. {exc}")
    return local_fallback_cluster(cards)


def semantic_fallback_domain(
    client: OpenAI | None,
    model: str,
    document_name: str,
    clusters: list[Cluster],
) -> tuple[str, str]:
    if client is not None:
        try:
            prompt = fallback_domain_prompt(document_name, clusters)
            cache_key = {
                "version": DOMAIN_PROMPT_VERSION,
                "schema_version": KNOWLEDGE_GRAPH_SCHEMA_VERSION,
                "model": model,
                "document_name": document_name,
                "cluster_set_hash": cluster_set_hash(clusters),
                "prompt_hash": stable_hash(prompt),
                "fallback": True,
            }
            cached = read_cache("domain_generation", cache_key)
            if cached is not None:
                result = cached.get("value", {})
            else:
                result = openrouter_json(client, model, prompt)
                write_cache("domain_generation", cache_key, result)
            name = str(result.get("domain_name", "")).strip()
            description = str(result.get("domain_description", "")).strip()
            if name and description:
                return name, description
        except Exception as exc:
            print(f"Warning: fallback domain naming failed; using local name. {exc}")
    return local_fallback_domain(clusters)


def domain_relationship_prompt(
    main_domain: dict[str, Any],
    related_domain: dict[str, Any],
    cluster_lookup: dict[str, dict[str, Any]],
    card_lookup: dict[str, dict[str, Any]],
) -> str:
    def expand_domain(domain: dict[str, Any]) -> dict[str, Any]:
        clusters = []
        for cluster_name in domain.get("cluster_names", []):
            cluster = cluster_lookup.get(cluster_name)
            if not cluster:
                continue
            cards = [
                {
                    "card_id": card_lookup.get(card_name, {}).get("card_id", ""),
                    "card_name": card_name,
                    "card_description": card_lookup.get(card_name, {}).get("card_description", ""),
                    "page_no": card_lookup.get(card_name, {}).get("page_no"),
                    "card_source": card_lookup.get(card_name, {}).get("card_source", ""),
                    "tags": card_lookup.get(card_name, {}).get("tags", []),
                    "content_excerpt": str(card_lookup.get(card_name, {}).get("content", ""))[:600],
                }
                for card_name in cluster.get("card_names", [])
            ]
            clusters.append(
                {
                    "cluster_id": cluster.get("cluster_id", ""),
                    "cluster_name": cluster_name,
                    "cluster_description": cluster.get("cluster_description", ""),
                    "cards": cards,
                }
            )
        return {
            "domain_id": domain.get("domain_id", ""),
            "domain_name": domain.get("domain_name", ""),
            "domain_description": domain.get("domain_description", ""),
            "document_name": domain.get("document_name", ""),
            "clusters": clusters,
        }

    return f"""
Find every meaningful relationship from the main domain to the related domain.

Important:
- The relationship is directional: main_domain -> related_domain.
- One domain may have multiple relationships with another domain.
- Return all meaningful relationships, not just the strongest one.
- If there is no meaningful relationship, return an empty relationships array.
- Add quality scores from 0.0 to 1.0.
- confidence_score: how likely this relationship is correct.
- evidence_strength: how directly the cited clusters/cards prove the relationship.
- source_coverage: how much of the relevant source material is represented by the links/evidence.

Relationship examples:
- continuation: related domain continues or extends information from main domain
- dependency: main domain depends on information from related domain
- prerequisite: related domain is needed before main domain can be understood
- evidence: related domain provides evidence/support for main domain
- contradiction: related domain conflicts with main domain
- comparison: related domain gives a comparable/contrasting view
- cause_effect: main domain causes, enables, blocks, or affects related domain
- shared_context: both domains describe the same larger context from different angles
- visual_support: related domain contains image/chart/map evidence for main domain

Main domain:
{json.dumps(expand_domain(main_domain), ensure_ascii=False)}

Related domain:
{json.dumps(expand_domain(related_domain), ensure_ascii=False)}

Return only valid JSON:
{{
  "relationships": [
    {{
      "relationship_type": "continuation | dependency | prerequisite | evidence | contradiction | comparison | cause_effect | shared_context | visual_support | other",
      "relationship_description": "Exactly how the related domain relates to the main domain",
      "evidence": "Short evidence using cluster/card names, page numbers, or content",
      "confidence_score": 0.0,
      "evidence_strength": 0.0,
      "source_coverage": 0.0,
      "cluster_links": [
        {{
          "main_cluster": "Exact cluster name from main domain",
          "related_cluster": "Exact cluster name from related domain",
          "relationship": "How these clusters connect for this relationship"
        }}
      ],
      "card_links": [
        {{
          "main_card": "Exact card name from main domain",
          "related_card": "Exact card name from related domain",
          "relationship": "How these cards connect for this relationship"
        }}
      ]
    }}
  ]
}}
""".strip()


def repair_json_prompt(original_prompt: str, malformed: str, error: str) -> str:
    return f"""
The previous response was supposed to be valid JSON, but parsing failed.

Parsing error:
{error}

Original task prompt:
{original_prompt[:12000]}

Malformed response:
{malformed[:12000]}

Return only one corrected valid JSON object. Preserve as much useful content as possible.
Do not include markdown fences, commentary, or explanatory text.
""".strip()


def openrouter_json(
    client: OpenAI,
    model: str,
    prompt: str,
    trace: dict[str, Any] | None = None,
) -> dict[str, Any]:
    last_content = ""
    last_error: Exception | None = None
    for attempt in range(1, 3):
        started = time.perf_counter()
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": "You organize document cards into knowledge graphs. Return only valid JSON.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0 if attempt > 1 else 0.1,
                response_format={"type": "json_object"},
            )
        except Exception as exc:
            append_metric(
                {
                    "event": "relationship_api_error",
                    "attempt": attempt,
                    "elapsed_seconds": round(time.perf_counter() - started, 3),
                    "model": model,
                    "error": str(exc)[:500],
                    **(trace or {}),
                }
            )
            raise
        api_elapsed = time.perf_counter() - started
        last_content = response.choices[0].message.content or "{}"
        usage = getattr(response, "usage", None)
        append_metric(
            {
                "event": "relationship_api_response",
                "attempt": attempt,
                "elapsed_seconds": round(api_elapsed, 3),
                "model": model,
                "response_model": getattr(response, "model", ""),
                "raw_chars": len(last_content),
                "prompt_tokens": getattr(usage, "prompt_tokens", None),
                "completion_tokens": getattr(usage, "completion_tokens", None),
                "total_tokens": getattr(usage, "total_tokens", None),
                **(trace or {}),
            }
        )
        try:
            return parse_json_response(last_content)
        except Exception as exc:
            last_error = exc
            print(f"Warning: model JSON parse failed on attempt {attempt}; retrying. {exc}")

    for repair_attempt in range(1, 3):
        started = time.perf_counter()
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You repair malformed JSON. Return only valid JSON.",
                },
                {
                    "role": "user",
                    "content": repair_json_prompt(prompt, last_content, str(last_error or "")),
                },
            ],
            temperature=0,
            response_format={"type": "json_object"},
        )
        api_elapsed = time.perf_counter() - started
        last_content = response.choices[0].message.content or "{}"
        usage = getattr(response, "usage", None)
        append_metric(
            {
                "event": "relationship_json_repair_response",
                "attempt": repair_attempt,
                "elapsed_seconds": round(api_elapsed, 3),
                "model": model,
                "response_model": getattr(response, "model", ""),
                "raw_chars": len(last_content),
                "prompt_tokens": getattr(usage, "prompt_tokens", None),
                "completion_tokens": getattr(usage, "completion_tokens", None),
                "total_tokens": getattr(usage, "total_tokens", None),
                **(trace or {}),
            }
        )
        try:
            return parse_json_response(last_content)
        except Exception as exc:
            last_error = exc
            print(f"Warning: JSON repair failed on attempt {repair_attempt}; retrying. {exc}")

    raise last_error or ValueError("OpenRouter response was not valid JSON.")


def build_lookups(
    cards: list[dict[str, Any]],
    clusters: list[Cluster],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    for card in cards:
        card.setdefault("card_id", card_stable_id(card))
    card_lookup = {card["card_name"]: card for card in cards}
    cluster_lookup = {
        cluster.cluster_name: asdict(cluster) for cluster in clusters
    }
    return card_lookup, cluster_lookup


def dry_clusters(document_id: str, document_name: str, cards: list[dict[str, Any]]) -> list[Cluster]:
    clusters_by_tag: dict[str, list[dict[str, str]]] = defaultdict(list)
    for card in cards:
        tags = card.get("tags") or ["text"]
        key = "Image-Based Cards" if "image" in tags else "Text-Based Cards"
        clusters_by_tag[key].append(
            {"card_id": card_stable_id(card), "card_name": card["card_name"]}
        )

    clusters: list[Cluster] = []
    for name, card_refs in clusters_by_tag.items():
        card_ids = [card["card_id"] for card in card_refs]
        card_names = [card["card_name"] for card in card_refs]
        clusters.append(
            Cluster(
                cluster_id=cluster_stable_id(document_id, card_ids, name),
                cluster_name=name,
                cluster_description=f"Cards grouped under {name.lower()} for {document_name}.",
                document_id=document_id,
                document_name=document_name,
                card_ids=card_ids,
                card_names=card_names,
            )
        )
    return clusters


def dry_domains(document_id: str, document_name: str, clusters: list[Cluster]) -> list[Domain]:
    cluster_ids = [cluster.cluster_id for cluster in clusters]
    cluster_names = [cluster.cluster_name for cluster in clusters]
    return [
        Domain(
            domain_id=domain_stable_id(document_id, cluster_ids, "Document Overview"),
            domain_name="Document Overview",
            domain_description=f"Top-level domain for the major clusters in {document_name}.",
            document_id=document_id,
            document_name=document_name,
            cluster_ids=cluster_ids,
            cluster_names=cluster_names,
        )
    ]


def dry_domain_relationships(domains: list[Domain]) -> list[DomainRelationship]:
    relationships: list[DomainRelationship] = []
    for main in domains:
        for related in domains:
            if main.domain_name == related.domain_name:
                continue
            same_document = main.document_id == related.document_id
            cluster_links = [
                {
                    "main_cluster": main.cluster_names[0] if main.cluster_names else "",
                    "related_cluster": related.cluster_names[0] if related.cluster_names else "",
                    "relationship": "Dry-run placeholder cluster connection.",
                }
            ]
            quality = relationship_quality(
                document_scope="same_document" if same_document else "cross_document",
                cluster_links=cluster_links,
                card_links=[],
                evidence="Dry-run placeholder relationship based on domain coexistence.",
                generation_method="deterministic_fallback",
            )
            relationships.append(
                DomainRelationship(
                    relationship_id=relationship_stable_id(
                        main.domain_id,
                        related.domain_id,
                        "shared_context" if same_document else "comparison",
                        f"{related.domain_name} may provide related context for {main.domain_name}.",
                    ),
                    main_domain_id=main.domain_id,
                    related_domain_id=related.domain_id,
                    main_domain=main.domain_name,
                    related_domain=related.domain_name,
                    relationship_type="shared_context" if same_document else "comparison",
                    relationship_description=(
                        f"{related.domain_name} may provide related context for "
                        f"{main.domain_name}."
                    ),
                    evidence="Dry-run placeholder relationship based on domain coexistence.",
                    cluster_links=cluster_links,
                    card_links=[],
                    document_scope=str(quality["document_scope"]),
                    confidence_score=float(quality["confidence_score"]),
                    evidence_strength=float(quality["evidence_strength"]),
                    source_coverage=float(quality["source_coverage"]),
                    generation_method=str(quality["generation_method"]),
                )
            )
    return relationships


def build_clusters(
    client: OpenAI | None,
    model: str,
    document_id: str,
    document_name: str,
    cards: list[dict[str, Any]],
    used_names: set[str],
) -> list[Cluster]:
    if client is None:
        raw_clusters = [asdict(item) for item in dry_clusters(document_id, document_name, cards)]
    else:
        try:
            prompt = clusters_prompt(document_name, cards)
            cache_key = {
                "version": CLUSTER_PROMPT_VERSION,
                "schema_version": KNOWLEDGE_GRAPH_SCHEMA_VERSION,
                "document_id": document_id,
                "model": model,
                "card_set_hash": card_set_hash(cards),
                "prompt_hash": stable_hash(prompt),
            }
            cached = read_cache("cluster_generation", cache_key)
            if cached is not None:
                result = cached.get("value", {})
            else:
                result = openrouter_json(client, model, prompt)
                write_cache("cluster_generation", cache_key, result)
            raw_clusters = result.get("clusters", [])
        except Exception as exc:
            print(f"Warning: cluster grouping failed for {document_name}; using deterministic grouping. {exc}")
            raw_clusters = [asdict(item) for item in dry_clusters(document_id, document_name, cards)]

    card_names = {card["card_name"] for card in cards}
    card_by_name = {card["card_name"]: card for card in cards}
    assigned: set[str] = set()
    clusters: list[Cluster] = []

    for raw in raw_clusters:
        names = [name for name in raw.get("card_names", []) if name in card_names]
        if not names:
            continue
        assigned.update(names)
        cluster_name = unique_versioned_name(raw.get("cluster_name", "Cluster"), used_names)
        card_ids = [card_stable_id(card_by_name[name]) for name in names]
        clusters.append(
            Cluster(
                cluster_id=cluster_stable_id(document_id, card_ids, cluster_name),
                cluster_name=cluster_name,
                cluster_description=str(raw.get("cluster_description", "")).strip(),
                document_id=document_id,
                document_name=document_name,
                card_ids=card_ids,
                card_names=names,
            )
        )

    missing = sorted(card_names - assigned)
    if missing:
        card_by_name = {card["card_name"]: card for card in cards}
        for card_name in missing:
            card = card_by_name[card_name]
            fallback_name, fallback_description = semantic_fallback_cluster(
                client,
                model,
                document_name,
                [card],
            )
            cluster_name = unique_versioned_name(fallback_name, used_names)
            card_ids = [card_stable_id(card)]
            clusters.append(
                Cluster(
                    cluster_id=cluster_stable_id(document_id, card_ids, cluster_name),
                    cluster_name=cluster_name,
                    cluster_description=fallback_description,
                    document_id=document_id,
                    document_name=document_name,
                    card_ids=card_ids,
                    card_names=[card_name],
                )
            )

    return clusters


def build_domains(
    client: OpenAI | None,
    model: str,
    document_id: str,
    document_name: str,
    clusters: list[Cluster],
    used_names: set[str],
) -> list[Domain]:
    if client is None:
        raw_domains = [asdict(item) for item in dry_domains(document_id, document_name, clusters)]
    else:
        try:
            prompt = domains_prompt(document_name, [asdict(cluster) for cluster in clusters])
            cache_key = {
                "version": DOMAIN_PROMPT_VERSION,
                "schema_version": KNOWLEDGE_GRAPH_SCHEMA_VERSION,
                "document_id": document_id,
                "model": model,
                "cluster_set_hash": cluster_set_hash(clusters),
                "prompt_hash": stable_hash(prompt),
            }
            cached = read_cache("domain_generation", cache_key)
            if cached is not None:
                result = cached.get("value", {})
            else:
                result = openrouter_json(client, model, prompt)
                write_cache("domain_generation", cache_key, result)
            raw_domains = result.get("domains", [])
        except Exception as exc:
            print(f"Warning: domain grouping failed for {document_name}; using deterministic grouping. {exc}")
            raw_domains = [asdict(item) for item in dry_domains(document_id, document_name, clusters)]

    cluster_names = {cluster.cluster_name for cluster in clusters}
    cluster_by_name = {cluster.cluster_name: cluster for cluster in clusters}
    assigned: set[str] = set()
    domains: list[Domain] = []

    for raw in raw_domains:
        names = [name for name in raw.get("cluster_names", []) if name in cluster_names]
        if not names:
            continue
        assigned.update(names)
        domain_name = unique_versioned_name(raw.get("domain_name", "Domain"), used_names)
        cluster_ids = [cluster_by_name[name].cluster_id for name in names]
        domains.append(
            Domain(
                domain_id=domain_stable_id(document_id, cluster_ids, domain_name),
                domain_name=domain_name,
                domain_description=str(raw.get("domain_description", "")).strip(),
                document_id=document_id,
                document_name=document_name,
                cluster_ids=cluster_ids,
                cluster_names=names,
            )
        )

    missing = sorted(cluster_names - assigned)
    if missing:
        for cluster_name in missing:
            cluster = cluster_by_name[cluster_name]
            fallback_name, fallback_description = semantic_fallback_domain(
                client,
                model,
                document_name,
                [cluster],
            )
            domain_name = unique_versioned_name(fallback_name, used_names)
            cluster_ids = [cluster.cluster_id]
            domains.append(
                Domain(
                    domain_id=domain_stable_id(document_id, cluster_ids, domain_name),
                    domain_name=domain_name,
                    domain_description=fallback_description,
                    document_id=document_id,
                    document_name=document_name,
                    cluster_ids=cluster_ids,
                    cluster_names=[cluster_name],
                )
            )

    return domains


def create_client() -> tuple[OpenAI, str]:
    try:
        client, _model, _provider = create_chat_client()
    except RuntimeError as exc:
        raise SystemExit(f"{exc}, or run with --dry-run.") from exc
    return client, get_model("map")


def relationship_worker_count(explicit_workers: int | None = None) -> int:
    if explicit_workers is not None:
        return max(1, explicit_workers)
    raw_value = os.getenv("OPENROUTER_RELATIONSHIP_WORKERS", str(DEFAULT_RELATIONSHIP_WORKERS))
    try:
        return max(1, int(raw_value))
    except ValueError:
        print(f"Warning: invalid OPENROUTER_RELATIONSHIP_WORKERS={raw_value!r}; using 1.")
        return 1


def extract_domain_relationship_pair(
    client: OpenAI,
    model: str,
    main: dict[str, Any],
    related: dict[str, Any],
    cluster_lookup: dict[str, dict[str, Any]],
    card_lookup: dict[str, dict[str, Any]],
) -> list[DomainRelationship]:
    prompt = domain_relationship_prompt(main, related, cluster_lookup, card_lookup)
    cache_key = {
        "version": RELATIONSHIP_PROMPT_VERSION,
        "schema_version": KNOWLEDGE_GRAPH_SCHEMA_VERSION,
        "model": model,
        "main_domain_id": main.get("domain_id", ""),
        "related_domain_id": related.get("domain_id", ""),
        "domain_pair_hash": domain_pair_hash(main, related),
        "prompt_hash": stable_hash(prompt),
    }
    cached = read_cache("relationship_generation", cache_key)
    if cached is not None:
        result = cached.get("value", {})
    else:
        trace = {
            "main_domain": str(main.get("domain_name", "")),
            "related_domain": str(related.get("domain_name", "")),
            "main_domain_id": str(main.get("domain_id", "")),
            "related_domain_id": str(related.get("domain_id", "")),
        }
        result = openrouter_json(
            client,
            model,
            prompt,
            trace,
        )
        write_cache("relationship_generation", cache_key, result)
    document_scope = (
        "same_document"
        if main.get("document_id") == related.get("document_id")
        else "cross_document"
    )
    extracted: list[DomainRelationship] = []
    raw_relationships = result.get("relationships", []) if isinstance(result, dict) else []
    if isinstance(raw_relationships, dict):
        raw_relationships = [raw_relationships]
    if not isinstance(raw_relationships, list):
        raw_relationships = []
    for raw in raw_relationships:
        if not isinstance(raw, dict):
            raw_text = str(raw).strip()
            if not raw_text or raw_text.lower() in {"none", "no relationship", "no relationships"}:
                continue
            raw = {
                "relationship_type": "other",
                "relationship_description": raw_text,
                "evidence": raw_text,
                "cluster_links": [],
                "card_links": [],
                "confidence_score": 0.35,
                "evidence_strength": 0.2,
                "source_coverage": 0.1,
            }
        relationship_type = clean_name(raw.get("relationship_type", "other"), "other")
        relationship_description = str(raw.get("relationship_description", "")).strip()
        cluster_links = normalize_links(
            raw.get("cluster_links", []),
            ["main_cluster", "related_cluster", "relationship"],
        )
        for link in cluster_links:
            link["main_cluster_id"] = str(cluster_lookup.get(link.get("main_cluster", ""), {}).get("cluster_id", ""))
            link["related_cluster_id"] = str(cluster_lookup.get(link.get("related_cluster", ""), {}).get("cluster_id", ""))
        card_links = normalize_links(
            raw.get("card_links", []),
            ["main_card", "related_card", "relationship"],
        )
        for link in card_links:
            link["main_card_id"] = str(card_lookup.get(link.get("main_card", ""), {}).get("card_id", ""))
            link["related_card_id"] = str(card_lookup.get(link.get("related_card", ""), {}).get("card_id", ""))
        evidence = str(raw.get("evidence", "")).strip()
        quality = relationship_quality(
            document_scope=document_scope,
            cluster_links=cluster_links,
            card_links=card_links,
            evidence=evidence,
            generation_method="model_generated",
            confidence_score=raw.get("confidence_score"),
            evidence_strength=raw.get("evidence_strength"),
            source_coverage=raw.get("source_coverage"),
        )
        extracted.append(
            DomainRelationship(
                relationship_id=relationship_stable_id(
                    str(main.get("domain_id", "")),
                    str(related.get("domain_id", "")),
                    relationship_type,
                    relationship_description,
                ),
                main_domain_id=str(main.get("domain_id", "")),
                related_domain_id=str(related.get("domain_id", "")),
                main_domain=main["domain_name"],
                related_domain=related["domain_name"],
                relationship_type=relationship_type,
                relationship_description=relationship_description,
                evidence=evidence,
                cluster_links=cluster_links,
                card_links=card_links,
                document_scope=str(quality["document_scope"]),
                confidence_score=float(quality["confidence_score"]),
                evidence_strength=float(quality["evidence_strength"]),
                source_coverage=float(quality["source_coverage"]),
                generation_method=str(quality["generation_method"]),
            )
        )
    return extracted


def extract_domain_relationship_pair_with_retries(
    client: OpenAI,
    model: str,
    main: dict[str, Any],
    related: dict[str, Any],
    cluster_lookup: dict[str, dict[str, Any]],
    card_lookup: dict[str, dict[str, Any]],
    max_attempts: int = MAX_RELATIONSHIP_PAIR_ATTEMPTS,
) -> list[DomainRelationship]:
    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            if attempt > 1:
                print(
                    f"Retrying domain relationship pair attempt {attempt}/{max_attempts}: "
                    f"{main['domain_name']} -> {related['domain_name']}..."
                )
                time.sleep(min(8.0, 1.5 * attempt))
            return extract_domain_relationship_pair(
                client,
                model,
                main,
                related,
                cluster_lookup,
                card_lookup,
            )
        except Exception as exc:
            last_error = exc
            print(
                f"Warning: relationship extraction attempt {attempt}/{max_attempts} failed for "
                f"{main['domain_name']} -> {related['domain_name']}. {exc}"
            )
    raise last_error or RuntimeError("Relationship extraction failed without an exception.")


def build_domain_relationships(
    client: OpenAI | None,
    model: str,
    cards: list[dict[str, Any]],
    clusters: list[Cluster],
    domains: list[Domain],
    on_progress: Any | None = None,
    existing_relationships: list[DomainRelationship] | None = None,
    processed_pair_keys: set[str] | None = None,
    max_workers: int | None = None,
) -> tuple[list[DomainRelationship], dict[str, str], set[str]]:
    processed_pair_keys = processed_pair_keys or set()
    domain_dicts = [asdict(domain) for domain in domains]
    relationships: list[DomainRelationship] = list(existing_relationships or [])
    failed_pair_errors: dict[str, str] = {}
    processed_pair_keys.update(
        domain_pair_key(relationship.main_domain, relationship.related_domain)
        for relationship in relationships
        if relationship.main_domain and relationship.related_domain
    )
    relationship_checks_total = max(0, len(domain_dicts) * (len(domain_dicts) - 1))
    relationship_checks_done = len(processed_pair_keys)
    pending_pairs: list[tuple[dict[str, Any], dict[str, Any], str]] = []
    for main in domain_dicts:
        for related in domain_dicts:
            if main["domain_name"] == related["domain_name"]:
                continue
            pair_key = domain_pair_key(main["domain_name"], related["domain_name"])
            if pair_key not in processed_pair_keys:
                pending_pairs.append((main, related, pair_key))

    if client is None:
        dry_relationships = {
            domain_pair_key(relationship.main_domain, relationship.related_domain): relationship
            for relationship in dry_domain_relationships(domains)
        }
        for _, _, pair_key in pending_pairs:
            relationship = dry_relationships.get(pair_key)
            if relationship is not None:
                relationships.append(relationship)
            processed_pair_keys.add(pair_key)
        if on_progress is not None:
            on_progress(relationships, relationship_checks_total, relationship_checks_total, processed_pair_keys)
        return relationships, {}, processed_pair_keys

    card_lookup, cluster_lookup = build_lookups(cards, clusters)
    workers = relationship_worker_count(max_workers)

    if workers <= 1 or len(pending_pairs) <= 1:
        for main, related, pair_key in pending_pairs:
            print(f"Extracting domain relationships: {main['domain_name']} -> {related['domain_name']}...")
            try:
                extracted = extract_domain_relationship_pair_with_retries(
                    client,
                    model,
                    main,
                    related,
                    cluster_lookup,
                    card_lookup,
                )
            except Exception as exc:
                message = (
                    f"Failed after {MAX_RELATIONSHIP_PAIR_ATTEMPTS} attempts for "
                    f"{main['domain_name']} -> {related['domain_name']}: {exc}"
                )
                print(f"ERROR: {message}")
                failed_pair_errors[pair_key] = message
                if on_progress is not None:
                    on_progress(relationships, relationship_checks_done, relationship_checks_total, processed_pair_keys)
                continue
            relationships.extend(extracted)
            record_relationship_pair_check(
                pair_key=pair_key,
                main_domain=main["domain_name"],
                related_domain=related["domain_name"],
                status="complete",
                relationships=[asdict(relationship) for relationship in extracted],
            )
            processed_pair_keys.add(pair_key)
            relationship_checks_done += 1
            if on_progress is not None:
                on_progress(relationships, relationship_checks_done, relationship_checks_total, processed_pair_keys)
        return relationships, failed_pair_errors, processed_pair_keys

    print(f"Extracting domain relationships with {workers} parallel workers...")
    worker_local = threading.local()

    def worker_client() -> OpenAI:
        existing_client = getattr(worker_local, "client", None)
        if existing_client is None:
            existing_client, _ = create_client()
            worker_local.client = existing_client
        return existing_client

    def process_pair(pair: tuple[dict[str, Any], dict[str, Any], str]) -> tuple[str, list[DomainRelationship], str | None]:
        main, related, pair_key = pair
        print(f"Extracting domain relationships: {main['domain_name']} -> {related['domain_name']}...")
        try:
            extracted = extract_domain_relationship_pair_with_retries(
                worker_client(),
                model,
                main,
                related,
                cluster_lookup,
                card_lookup,
            )
            return pair_key, extracted, None
        except Exception as exc:
            main, related, _ = pair
            return pair_key, [], (
                f"Failed after {MAX_RELATIONSHIP_PAIR_ATTEMPTS} attempts for "
                f"{main['domain_name']} -> {related['domain_name']}: {exc}"
            )

    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_pair = {executor.submit(process_pair, pair): pair for pair in pending_pairs}
        for future in as_completed(future_to_pair):
            main, related, _original_pair_key = future_to_pair[future]
            pair_key, extracted, error = future.result()
            if error:
                print(f"ERROR: {error}")
                failed_pair_errors[pair_key] = error
                record_relationship_pair_check(
                    pair_key=pair_key,
                    main_domain=main["domain_name"],
                    related_domain=related["domain_name"],
                    status="failed",
                    relationships=[],
                    error=error,
                )
                if on_progress is not None:
                    on_progress(relationships, relationship_checks_done, relationship_checks_total, processed_pair_keys)
                continue
            relationships.extend(extracted)
            record_relationship_pair_check(
                pair_key=pair_key,
                main_domain=main["domain_name"],
                related_domain=related["domain_name"],
                status="complete",
                relationships=[asdict(relationship) for relationship in extracted],
            )
            processed_pair_keys.add(pair_key)
            relationship_checks_done += 1
            if on_progress is not None:
                on_progress(relationships, relationship_checks_done, relationship_checks_total, processed_pair_keys)

    return relationships, failed_pair_errors, processed_pair_keys


def write_knowledge_graph_progress(
    clusters: list[Cluster],
    domains: list[Domain],
    domain_relationships: list[DomainRelationship],
    status: str,
    relationship_checks_done: int = 0,
    relationship_checks_total: int = 0,
    processed_pair_keys: set[str] | None = None,
    failed_pair_errors: dict[str, str] | None = None,
    source_manifest: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    checkpoint_started = time.perf_counter()
    knowledge_graph = {
        "schema_version": KNOWLEDGE_GRAPH_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_manifest": source_manifest or [],
        "source_manifest_hash": source_manifest_hash(source_manifest or []),
        "status": status,
        "clusters": [asdict(cluster) for cluster in clusters],
        "domains": [asdict(domain) for domain in domains],
        "domain_relationships": [asdict(relationship) for relationship in domain_relationships],
        "relationship_checks_done": relationship_checks_done,
        "relationship_checks_total": relationship_checks_total,
        "processed_relationship_checks": sorted(processed_pair_keys or []),
        "failed_relationship_checks": failed_pair_errors or {},
    }
    sync_started = time.perf_counter()
    sync_knowledge_graph(knowledge_graph)
    sync_elapsed = time.perf_counter() - sync_started
    state_started = time.perf_counter()
    write_knowledge_graph_state(knowledge_graph)
    state_elapsed = time.perf_counter() - state_started
    json_elapsed = 0.0
    try:
        json_started = time.perf_counter()
        write_json(KNOWLEDGE_GRAPH_PATH, knowledge_graph)
        json_elapsed = time.perf_counter() - json_started
    except OSError as exc:
        print(f"Warning: could not export knowledge_graph.json: {exc}")
    checkpoint_elapsed = time.perf_counter() - checkpoint_started
    append_metric(
        {
            "event": "relationship_checkpoint",
            "elapsed_seconds": round(checkpoint_elapsed, 3),
            "sync_seconds": round(sync_elapsed, 3),
            "state_seconds": round(state_elapsed, 3),
            "json_export_seconds": round(json_elapsed, 3),
            "status": status,
            "relationship_checks_done": relationship_checks_done,
            "relationship_checks_total": relationship_checks_total,
            "relationship_count": len(domain_relationships),
        }
    )
    write_pipeline_progress(
        {
            "stage": "knowledge_graph" if status != "complete" else "complete",
            "message": f"Knowledge graph status: {status}.",
            "knowledge_graph_status": status,
            "cluster_count": len(clusters),
            "domain_count": len(domains),
            "relationship_count": len(domain_relationships),
            "relationship_checks_done": relationship_checks_done,
            "relationship_checks_total": relationship_checks_total,
            "failed_relationship_check_count": len(failed_pair_errors or {}),
        }
    )
    return knowledge_graph


def build_graph_audit(
    cards: list[dict[str, Any]],
    clusters: list[Cluster],
    domains: list[Domain],
    relationships: list[DomainRelationship],
    source_manifest: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    issues: list[dict[str, Any]] = []
    cards_by_id = {str(card.get("card_id") or card_stable_id(card)): card for card in cards}
    card_ids = set(cards_by_id)
    clustered_card_ids: set[str] = set()
    for cluster in clusters:
        clustered_card_ids.update(str(card_id) for card_id in cluster.card_ids if card_id)

    for card_id in sorted(card_ids - clustered_card_ids):
        card = cards_by_id[card_id]
        issues.append(
            audit_issue(
                "cards_with_no_cluster",
                "high",
                "Card is indexed but not assigned to any cluster.",
                card_id=card_id,
                card_name=card.get("card_name", ""),
                document_id=card.get("document_id", ""),
                document_name=card.get("document_name", ""),
                page_no=card.get("page_no"),
            )
        )

    cluster_names = {cluster.cluster_name for cluster in clusters}
    card_names = {str(card.get("card_name", "")) for card in cards}
    domain_cluster_names: set[str] = set()
    for domain in domains:
        domain_cluster_names.update(domain.cluster_names)
    for cluster in clusters:
        if cluster.cluster_name not in domain_cluster_names:
            issues.append(
                audit_issue(
                    "clusters_with_no_domain",
                    "high",
                    "Cluster is not assigned to any domain.",
                    cluster_id=cluster.cluster_id,
                    cluster_name=cluster.cluster_name,
                    document_id=cluster.document_id,
                    document_name=cluster.document_name,
                )
            )
        for card_name in cluster.card_names:
            if card_name not in card_names:
                issues.append(
                    audit_issue(
                        "orphaned_cluster_card_reference",
                        "medium",
                        "Cluster references a card name that is not in the card index.",
                        cluster_id=cluster.cluster_id,
                        cluster_name=cluster.cluster_name,
                        card_name=card_name,
                    )
                )

    relationship_domain_names = {
        relationship.main_domain for relationship in relationships if relationship.main_domain
    } | {
        relationship.related_domain for relationship in relationships if relationship.related_domain
    }
    for domain in domains:
        if domain.domain_name not in relationship_domain_names and len(domains) > 1:
            issues.append(
                audit_issue(
                    "domains_with_no_relationships",
                    "medium",
                    "Domain has no incoming or outgoing relationships.",
                    domain_id=domain.domain_id,
                    domain_name=domain.domain_name,
                    document_id=domain.document_id,
                    document_name=domain.document_name,
                )
            )
        for cluster_name in domain.cluster_names:
            if cluster_name not in cluster_names:
                issues.append(
                    audit_issue(
                        "orphaned_domain_cluster_reference",
                        "medium",
                        "Domain references a cluster name that is not in the graph.",
                        domain_id=domain.domain_id,
                        domain_name=domain.domain_name,
                        cluster_name=cluster_name,
                    )
                )

    duplicate_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for card in cards:
        duplicate_key = "|".join(
            [
                normalized_name(card.get("card_name", "")),
                str(card.get("document_id", "")),
                str(card.get("page_no", "")),
            ]
        )
        duplicate_groups[duplicate_key].append(card)
    for group in duplicate_groups.values():
        if len(group) > 1:
            issues.append(
                audit_issue(
                    "duplicate_cards",
                    "medium",
                    "Multiple cards have the same normalized name on the same document/page.",
                    cards=[
                        {
                            "card_id": item.get("card_id", ""),
                            "card_name": item.get("card_name", ""),
                            "document_id": item.get("document_id", ""),
                            "document_name": item.get("document_name", ""),
                            "page_no": item.get("page_no"),
                        }
                        for item in group
                    ],
                )
            )

    for card in cards:
        if is_suspiciously_generic_name(str(card.get("card_name", "")), "card"):
            issues.append(
                audit_issue(
                    "suspiciously_generic_names",
                    "low",
                    "Card name is suspiciously generic.",
                    node_type="card",
                    card_id=card.get("card_id", ""),
                    name=card.get("card_name", ""),
                    document_name=card.get("document_name", ""),
                    page_no=card.get("page_no"),
                )
            )
    for cluster in clusters:
        if is_suspiciously_generic_name(cluster.cluster_name, "cluster"):
            issues.append(
                audit_issue(
                    "suspiciously_generic_names",
                    "low",
                    "Cluster name is suspiciously generic.",
                    node_type="cluster",
                    cluster_id=cluster.cluster_id,
                    name=cluster.cluster_name,
                    document_name=cluster.document_name,
                )
            )
    for domain in domains:
        if is_suspiciously_generic_name(domain.domain_name, "domain"):
            issues.append(
                audit_issue(
                    "suspiciously_generic_names",
                    "low",
                    "Domain name is suspiciously generic.",
                    node_type="domain",
                    domain_id=domain.domain_id,
                    name=domain.domain_name,
                    document_name=domain.document_name,
                )
            )

    for relationship in relationships:
        quality_score = (
            0.5 * relationship.confidence_score
            + 0.3 * relationship.evidence_strength
            + 0.2 * relationship.source_coverage
        )
        relationship_payload = {
            "relationship_id": relationship.relationship_id,
            "main_domain": relationship.main_domain,
            "related_domain": relationship.related_domain,
            "relationship_type": relationship.relationship_type,
            "confidence_score": relationship.confidence_score,
            "evidence_strength": relationship.evidence_strength,
            "source_coverage": relationship.source_coverage,
            "quality_score": round(quality_score, 4),
            "document_scope": relationship.document_scope,
            "generation_method": relationship.generation_method,
        }
        if quality_score < 0.55:
            issues.append(
                audit_issue(
                    "weak_relationships",
                    "medium",
                    "Relationship quality score is below threshold.",
                    **relationship_payload,
                )
            )
        if not relationship.evidence.strip():
            issues.append(
                audit_issue(
                    "empty_evidence",
                    "medium",
                    "Relationship has empty evidence.",
                    **relationship_payload,
                )
            )
        if not relationship.cluster_links and not relationship.card_links:
            issues.append(
                audit_issue(
                    "empty_evidence",
                    "low",
                    "Relationship has no cluster or card evidence links.",
                    **relationship_payload,
                )
            )

    card_document_ids = {str(card.get("document_id", "")) for card in cards if card.get("document_id")}
    cluster_document_ids = {cluster.document_id for cluster in clusters if cluster.document_id}
    domain_document_ids = {domain.document_id for domain in domains if domain.document_id}
    source_manifest = source_manifest or []
    for source in source_manifest:
        document_id = str(source.get("document_id", ""))
        if not document_id:
            continue
        if document_id not in card_document_ids:
            issues.append(
                audit_issue(
                    "orphaned_documents_pages",
                    "high",
                    "Source document has no cards in the card index.",
                    document_id=document_id,
                    document_name=source.get("document_name", ""),
                )
            )
        if document_id not in cluster_document_ids:
            issues.append(
                audit_issue(
                    "orphaned_documents_pages",
                    "high",
                    "Source document has no clusters in the graph.",
                    document_id=document_id,
                    document_name=source.get("document_name", ""),
                )
            )
        if document_id not in domain_document_ids:
            issues.append(
                audit_issue(
                    "orphaned_documents_pages",
                    "high",
                    "Source document has no domains in the graph.",
                    document_id=document_id,
                    document_name=source.get("document_name", ""),
                )
            )

    page_clusters: set[tuple[str, int]] = set()
    for cluster in clusters:
        for card_id in cluster.card_ids:
            card = cards_by_id.get(str(card_id))
            if card and card.get("document_id") and card.get("page_no"):
                page_clusters.add((str(card.get("document_id")), int(card.get("page_no") or 0)))
    indexed_pages = {
        (str(card.get("document_id")), int(card.get("page_no") or 0))
        for card in cards
        if card.get("document_id") and card.get("page_no")
    }
    for document_id, page_no in sorted(indexed_pages - page_clusters):
        page_cards = [
            card
            for card in cards
            if str(card.get("document_id")) == document_id and int(card.get("page_no") or 0) == page_no
        ]
        issues.append(
            audit_issue(
                "orphaned_documents_pages",
                "high",
                "Indexed page has cards, but none of those cards are assigned to clusters.",
                document_id=document_id,
                document_name=page_cards[0].get("document_name", "") if page_cards else "",
                page_no=page_no,
            )
        )

    summary_by_category: dict[str, int] = defaultdict(int)
    summary_by_severity: dict[str, int] = defaultdict(int)
    for issue in issues:
        summary_by_category[str(issue["category"])] += 1
        summary_by_severity[str(issue["severity"])] += 1
    return {
        "schema_version": "graph_audit.v1.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": graph_audit_status(
            len(issues),
            summary_by_severity.get("high", 0),
            summary_by_severity.get("medium", 0),
        ),
        "issue_count": len(issues),
        "summary_by_category": dict(sorted(summary_by_category.items())),
        "summary_by_severity": dict(sorted(summary_by_severity.items())),
        "checks": [
            "cards_with_no_cluster",
            "clusters_with_no_domain",
            "domains_with_no_relationships",
            "duplicate_cards",
            "weak_relationships",
            "suspiciously_generic_names",
            "empty_evidence",
            "orphaned_documents_pages",
        ],
        "issues": issues,
    }


def write_graph_audit(
    cards: list[dict[str, Any]],
    clusters: list[Cluster],
    domains: list[Domain],
    relationships: list[DomainRelationship],
    source_manifest: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    audit = build_graph_audit(cards, clusters, domains, relationships, source_manifest)
    write_graph_audit_state(audit)
    try:
        write_json(GRAPH_AUDIT_PATH, audit)
    except OSError as exc:
        print(f"Warning: could not export graph_audit.json: {exc}")
    return audit


def build_knowledge_graph(
    dry_run: bool,
    relationship_workers: int | None = None,
    build_summaries: bool = False,
) -> None:
    load_dotenv()
    cards = read_cards()
    if not cards:
        raise SystemExit("No cards found in PostgreSQL. Run indexer.py first.")

    client = None
    model = get_model("map")
    if not dry_run:
        client, model = create_client()

    grouped = group_cards_by_document(cards)
    current_source_manifest = source_manifest_from_cards(cards)
    current_source_by_id = {
        str(item.get("document_id", "")): item
        for item in current_source_manifest
        if item.get("document_id")
    }
    cards_by_name = {card["card_name"]: card for card in cards}
    used_cluster_names: set[str] = set()
    used_domain_names: set[str] = set()
    all_clusters: list[Cluster] = []
    all_domains: list[Domain] = []
    domain_relationships: list[DomainRelationship] = []
    processed_pair_keys: set[str] = set()
    existing_map = read_knowledge_graph_state()
    if not existing_map:
        existing_map = read_knowledge_graph(KNOWLEDGE_GRAPH_PATH, cards, persist_migration=True)
    db_pair_checks = read_relationship_pair_checks()
    db_relationship_payloads = read_relationship_payloads()
    if db_pair_checks or db_relationship_payloads:
        existing_relationships_by_id = {
            str(item.get("relationship_id", "")): item
            for item in existing_map.get("domain_relationships", []) or []
            if item.get("relationship_id")
        }
        for relationship in db_relationship_payloads:
            relationship_id = str(relationship.get("relationship_id", ""))
            if relationship_id:
                existing_relationships_by_id[relationship_id] = relationship
        existing_map["domain_relationships"] = list(existing_relationships_by_id.values())

        processed_from_pairs = {
            str(item.get("pair_key", ""))
            for item in db_pair_checks
            if item.get("status") == "complete" and item.get("pair_key")
        }
        failed_from_pairs = {
            str(item.get("pair_key", "")): str(item.get("error", "Pair failed."))
            for item in db_pair_checks
            if item.get("status") == "failed" and item.get("pair_key")
        }
        existing_processed = set(
            existing_map.get("processed_relationship_checks")
            or existing_map.get(LEGACY_PROCESSED_RELATIONSHIP_CHECKS)
            or []
        )
        existing_processed.update(processed_from_pairs)
        existing_map["processed_relationship_checks"] = sorted(existing_processed)
        existing_failed = dict(existing_map.get("failed_relationship_checks") or {})
        existing_failed.update(failed_from_pairs)
        existing_map["failed_relationship_checks"] = existing_failed
        if processed_from_pairs:
            existing_map["relationship_checks_done"] = max(
                int(existing_map.get("relationship_checks_done") or 0),
                len(existing_processed),
            )
    existing_document_ids = {
        item.get("document_id")
        for item in existing_map.get("clusters", [])
        if item.get("document_id")
    }
    expected_document_ids = set(grouped)
    can_resume_relationships = (
            existing_map.get("status") in {"building_relationships", "failed_relationships"}
        and existing_document_ids == expected_document_ids
        and existing_map.get("clusters")
        and existing_map.get("domains")
    )
    previous_source_manifest = existing_map.get("source_manifest", [])
    previous_source_by_id = {
        str(item.get("document_id", "")): item
        for item in previous_source_manifest
        if isinstance(item, dict) and item.get("document_id")
    }
    if previous_source_by_id:
        unchanged_document_ids = {
            document_id
            for document_id, current_source in current_source_by_id.items()
            if previous_source_by_id.get(document_id, {}).get("card_signature")
            == current_source.get("card_signature")
        }
    else:
        unchanged_document_ids = expected_document_ids & existing_document_ids
    changed_document_ids = expected_document_ids - unchanged_document_ids
    removed_document_ids = existing_document_ids - expected_document_ids
    has_existing_graph = bool(existing_map.get("clusters") and existing_map.get("domains"))

    if has_existing_graph and unchanged_document_ids:
        print(
            "Preserving knowledge graph sections for "
            f"{len(unchanged_document_ids)} unchanged document(s)."
        )
        if changed_document_ids:
            print(f"Regenerating graph sections for {len(changed_document_ids)} new or changed document(s).")
        if removed_document_ids:
            print(f"Pruning graph sections for {len(removed_document_ids)} removed document(s).")

        all_clusters = [
            cluster_from_dict(raw, cards_by_name)
            for raw in existing_map.get("clusters", [])
            if raw.get("document_id") in unchanged_document_ids
        ]
        for cluster in all_clusters:
            if cluster.cluster_name:
                used_cluster_names.add(cluster.cluster_name)

        clusters_by_name = {cluster.cluster_name: cluster for cluster in all_clusters}
        all_domains = [
            domain_from_dict(raw, clusters_by_name)
            for raw in existing_map.get("domains", [])
            if raw.get("document_id") in unchanged_document_ids
        ]
        for domain in all_domains:
            if domain.domain_name:
                used_domain_names.add(domain.domain_name)

        preserved_domain_ids = {domain.domain_id for domain in all_domains if domain.domain_id}
        preserved_domain_names = {domain.domain_name for domain in all_domains if domain.domain_name}
        domain_relationships = []
        for raw in existing_map.get("domain_relationships", []):
            relationship = relationship_from_dict(raw)
            main_preserved = (
                relationship.main_domain_id in preserved_domain_ids
                or relationship.main_domain in preserved_domain_names
            )
            related_preserved = (
                relationship.related_domain_id in preserved_domain_ids
                or relationship.related_domain in preserved_domain_names
            )
            if main_preserved and related_preserved:
                domain_relationships.append(relationship)

        old_processed_pair_keys = set(
            existing_map.get("processed_relationship_checks")
            or existing_map.get(LEGACY_PROCESSED_RELATIONSHIP_CHECKS)
            or []
        )
        processed_pair_keys = {
            pair_key
            for pair_key in old_processed_pair_keys
            if (
                (names := domain_names_from_pair_key(pair_key)) is not None
                and names[0] in preserved_domain_names
                and names[1] in preserved_domain_names
            )
        }
        processed_pair_keys.update(
            domain_pair_key(relationship.main_domain, relationship.related_domain)
            for relationship in domain_relationships
            if relationship.main_domain and relationship.related_domain
        )
        write_knowledge_graph_progress(
            all_clusters,
            all_domains,
            domain_relationships,
            "incremental_started" if changed_document_ids else "building_relationships",
            len(processed_pair_keys),
            max(0, len(all_domains) * (len(all_domains) - 1)),
            processed_pair_keys,
            source_manifest=current_source_manifest,
        )

    if can_resume_relationships:
        print("Resuming domain-to-domain relationships from the existing partial knowledge graph...")
        if not all_clusters:
            all_clusters = [
                cluster_from_dict(raw, cards_by_name)
                for raw in existing_map.get("clusters", [])
            ]
        clusters_by_name = {cluster.cluster_name: cluster for cluster in all_clusters}
        if not all_domains:
            all_domains = [
                domain_from_dict(raw, clusters_by_name)
                for raw in existing_map.get("domains", [])
            ]
        if not domain_relationships:
            domain_relationships = [
                relationship_from_dict(raw)
                for raw in existing_map.get("domain_relationships", [])
                if raw.get("main_domain") and raw.get("related_domain")
            ]
        domains_by_name = {domain.domain_name: domain for domain in all_domains}
        for relationship in domain_relationships:
            main_domain = domains_by_name.get(relationship.main_domain)
            related_domain = domains_by_name.get(relationship.related_domain)
            if main_domain and not relationship.main_domain_id:
                relationship.main_domain_id = main_domain.domain_id
            if related_domain and not relationship.related_domain_id:
                relationship.related_domain_id = related_domain.domain_id
            if relationship.main_domain_id and relationship.related_domain_id:
                relationship.relationship_id = relationship_stable_id(
                    relationship.main_domain_id,
                    relationship.related_domain_id,
                    relationship.relationship_type,
                    relationship.relationship_description,
                )
        processed_pair_keys = set(
            processed_pair_keys
            or
            existing_map.get("processed_relationship_checks")
            or existing_map.get(LEGACY_PROCESSED_RELATIONSHIP_CHECKS)
            or []
        )
        processed_pair_keys.update(
            domain_pair_key(relationship.main_domain, relationship.related_domain)
            for relationship in domain_relationships
            if relationship.main_domain and relationship.related_domain
        )
        write_knowledge_graph_progress(
            all_clusters,
            all_domains,
            domain_relationships,
            "building_relationships",
            len(processed_pair_keys),
            max(0, len(all_domains) * (len(all_domains) - 1)),
            processed_pair_keys,
            source_manifest=current_source_manifest,
        )
    else:
        if not has_existing_graph:
            write_knowledge_graph_progress(
                all_clusters,
                all_domains,
                domain_relationships,
                "started",
                source_manifest=current_source_manifest,
            )

        for document_id, document_cards in grouped.items():
            if has_existing_graph and document_id not in changed_document_ids:
                continue
            document_name = document_cards[0].get("document_name", "Unknown Document")
            print(f"Building clusters for {document_name}...")
            clusters = build_clusters(
                client=client,
                model=model,
                document_id=document_id,
                document_name=document_name,
                cards=document_cards,
                used_names=used_cluster_names,
            )
            all_clusters.extend(clusters)

            print(f"Building domains for {document_name}...")
            domains = build_domains(
                client=client,
                model=model,
                document_id=document_id,
                document_name=document_name,
                clusters=clusters,
                used_names=used_domain_names,
            )
            all_domains.extend(domains)
            write_knowledge_graph_progress(
                all_clusters,
                all_domains,
                domain_relationships,
                f"grouped {document_name}",
                source_manifest=current_source_manifest,
            )

    print("Extracting domain-to-domain relationships...")
    current_failed_pair_errors: dict[str, str] = {}
    checkpoint_every_checks, checkpoint_every_seconds = relationship_checkpoint_settings()
    last_full_checkpoint_done = len(processed_pair_keys)
    last_full_checkpoint_at = time.perf_counter()
    print(
        "Full graph checkpoints will run every "
        f"{checkpoint_every_checks} completed checks or {checkpoint_every_seconds:.0f}s."
    )

    def relationship_progress(
        relationships: list[DomainRelationship],
        pairs_done: int,
        pairs_total: int,
        pair_keys: set[str],
    ) -> None:
        nonlocal last_full_checkpoint_at, last_full_checkpoint_done
        now = time.perf_counter()
        should_full_checkpoint = (
            pairs_done >= pairs_total
            or pairs_done - last_full_checkpoint_done >= checkpoint_every_checks
            or now - last_full_checkpoint_at >= checkpoint_every_seconds
        )
        if should_full_checkpoint:
            write_knowledge_graph_progress(
                all_clusters,
                all_domains,
                relationships,
                "building_relationships",
                pairs_done,
                pairs_total,
                set(pair_keys),
                current_failed_pair_errors,
                current_source_manifest,
            )
            last_full_checkpoint_done = pairs_done
            last_full_checkpoint_at = time.perf_counter()
            return
        write_relationship_lightweight_progress(
            status="building_relationships",
            cluster_count=len(all_clusters),
            domain_count=len(all_domains),
            relationship_count=len(relationships),
            relationship_checks_done=pairs_done,
            relationship_checks_total=pairs_total,
            failed_pair_errors=current_failed_pair_errors,
        )

    domain_relationships, failed_pair_errors, processed_pair_keys = build_domain_relationships(
        client=client,
        model=model,
        cards=cards,
        clusters=all_clusters,
        domains=all_domains,
        on_progress=relationship_progress,
        existing_relationships=domain_relationships,
        processed_pair_keys=processed_pair_keys,
        max_workers=relationship_workers,
    )
    current_failed_pair_errors.update(failed_pair_errors)

    total_relationship_checks = max(0, len(all_domains) * (len(all_domains) - 1))
    all_pair_keys = {
        domain_pair_key(main.domain_name, related.domain_name)
        for main in all_domains
        for related in all_domains
        if main.domain_name != related.domain_name
    }
    successful_pair_keys = set(processed_pair_keys)
    final_status = "complete"
    final_pair_keys = all_pair_keys
    final_checks_done = total_relationship_checks
    if failed_pair_errors or len(successful_pair_keys) < total_relationship_checks:
        final_status = "failed_relationships"
        final_pair_keys = successful_pair_keys
        final_checks_done = len(successful_pair_keys)
        if not failed_pair_errors:
            missing_pair_keys = sorted(all_pair_keys - successful_pair_keys)
            failed_pair_errors = {
                key: "Pair did not complete and will be retried on the next run."
                for key in missing_pair_keys
            }
    final_graph = write_knowledge_graph_progress(
        all_clusters,
        all_domains,
        domain_relationships,
        final_status,
        final_checks_done,
        total_relationship_checks,
        final_pair_keys,
        failed_pair_errors,
        current_source_manifest,
    )
    audit = write_graph_audit(
        cards,
        all_clusters,
        all_domains,
        domain_relationships,
        current_source_manifest,
    )
    record_graph_build_run(final_graph, audit)
    write_pipeline_progress(
        {
            "stage": "complete" if final_status == "complete" else "knowledge_graph",
            "message": (
                f"Knowledge graph status: {final_status}. "
                f"Graph audit: {audit['status']} with {audit['issue_count']} issue(s)."
            ),
            "knowledge_graph_status": final_status,
            "cluster_count": len(all_clusters),
            "domain_count": len(all_domains),
            "relationship_count": len(domain_relationships),
            "relationship_checks_done": final_checks_done,
            "relationship_checks_total": total_relationship_checks,
            "failed_relationship_check_count": len(failed_pair_errors or {}),
            "graph_audit_status": audit["status"],
            "graph_audit_issue_count": audit["issue_count"],
            "graph_audit_summary_by_category": audit["summary_by_category"],
            "graph_audit_summary_by_severity": audit["summary_by_severity"],
        }
    )
    if build_summaries:
        print("Building community summaries...")
        build_community_summaries(cards, final_graph, dry_run=dry_run)
    print(
        f"Wrote {len(all_clusters)} clusters, {len(all_domains)} domains, "
        f"and {len(domain_relationships)} domain relationships "
        f"to {KNOWLEDGE_GRAPH_PATH}."
    )
    print(
        f"Graph audit {audit['status']}: {audit['issue_count']} issue(s) "
        f"written to {GRAPH_AUDIT_PATH}."
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build card clusters and domains.")
    parser.add_argument("--dry-run", action="store_true", help="Build without calling OpenRouter.")
    parser.add_argument(
        "--relationship-workers",
        type=int,
        default=None,
        help="Number of parallel workers for domain-to-domain relationship checks.",
    )
    parser.add_argument(
        "--build-summaries",
        action="store_true",
        help="Build or refresh precomputed domain/community summaries after the graph is complete.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    build_knowledge_graph(
        dry_run=args.dry_run,
        relationship_workers=args.relationship_workers,
        build_summaries=args.build_summaries,
    )


if __name__ == "__main__":
    main()




