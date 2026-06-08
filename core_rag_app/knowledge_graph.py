from __future__ import annotations

import argparse
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


LEGACY_ROOT_ENV = "PDF" + "_VISION_RAG_ROOT"
PROJECT_ROOT = Path(
    os.getenv("EVIDENCE_MESH_ROOT")
    or os.getenv(LEGACY_ROOT_ENV)
    or Path(__file__).resolve().parent
).resolve()
INDEXES_DIR = PROJECT_ROOT / "indexes"
CARD_INDEX_PATH = INDEXES_DIR / "card_index.json"
KNOWLEDGE_GRAPH_PATH = INDEXES_DIR / "knowledge_graph.json"
PIPELINE_PROGRESS_PATH = PROJECT_ROOT / "logs" / "pipeline_progress.json"

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "google/gemini-3.1-flash-lite"
DEFAULT_RELATIONSHIP_WORKERS = 1
MAX_RELATIONSHIP_PAIR_ATTEMPTS = 3
LEGACY_PROCESSED_RELATIONSHIP_CHECKS = "processed_relationship" + "_pairs"


@dataclass
class Cluster:
    cluster_name: str
    cluster_description: str
    document_id: str
    document_name: str
    card_names: list[str]


@dataclass
class Domain:
    domain_name: str
    domain_description: str
    document_id: str
    document_name: str
    cluster_names: list[str]


@dataclass
class DomainRelationship:
    main_domain: str
    related_domain: str
    relationship_type: str
    relationship_description: str
    evidence: str
    cluster_links: list[dict[str, str]]
    card_links: list[dict[str, str]]
    document_scope: str


def load_dotenv() -> None:
    env_paths = [PROJECT_ROOT / ".env", Path(__file__).resolve().parent / ".env"]
    for env_path in env_paths:
        if not env_path.exists():
            continue
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ[key.strip()] = value.strip().strip("\"'")


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


def write_pipeline_progress(data: dict[str, Any]) -> None:
    write_json(
        PIPELINE_PROGRESS_PATH,
        {
            "updated_at": datetime.now(timezone.utc).isoformat(),
            **data,
        },
    )


def clean_name(value: str, fallback: str) -> str:
    value = re.sub(r"\s+", " ", str(value).strip())
    value = re.sub(r"[^a-zA-Z0-9 _./&()-]+", "", value)
    return value[:120] or fallback


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


def relationship_from_dict(raw: dict[str, Any]) -> DomainRelationship:
    return DomainRelationship(
        main_domain=str(raw.get("main_domain", "")).strip(),
        related_domain=str(raw.get("related_domain", "")).strip(),
        relationship_type=str(raw.get("relationship_type", "other")).strip() or "other",
        relationship_description=str(raw.get("relationship_description", "")).strip(),
        evidence=str(raw.get("evidence", "")).strip(),
        cluster_links=raw.get("cluster_links", []) if isinstance(raw.get("cluster_links"), list) else [],
        card_links=raw.get("card_links", []) if isinstance(raw.get("card_links"), list) else [],
        document_scope=str(raw.get("document_scope", "")).strip(),
    )


def semantic_fallback_cluster(
    client: OpenAI | None,
    model: str,
    document_name: str,
    cards: list[dict[str, Any]],
) -> tuple[str, str]:
    if client is not None:
        try:
            result = openrouter_json(client, model, fallback_cluster_prompt(document_name, cards))
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
            result = openrouter_json(client, model, fallback_domain_prompt(document_name, clusters))
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
                    "cluster_name": cluster_name,
                    "cluster_description": cluster.get("cluster_description", ""),
                    "cards": cards,
                }
            )
        return {
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


def openrouter_json(client: OpenAI, model: str, prompt: str) -> dict[str, Any]:
    last_content = ""
    last_error: Exception | None = None
    for attempt in range(1, 3):
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
        last_content = response.choices[0].message.content or "{}"
        try:
            return parse_json_response(last_content)
        except Exception as exc:
            last_error = exc
            print(f"Warning: model JSON parse failed on attempt {attempt}; retrying. {exc}")

    for repair_attempt in range(1, 3):
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
        last_content = response.choices[0].message.content or "{}"
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
    card_lookup = {card["card_name"]: card for card in cards}
    cluster_lookup = {
        cluster.cluster_name: asdict(cluster) for cluster in clusters
    }
    return card_lookup, cluster_lookup


def dry_clusters(document_id: str, document_name: str, cards: list[dict[str, Any]]) -> list[Cluster]:
    clusters_by_tag: dict[str, list[str]] = defaultdict(list)
    for card in cards:
        tags = card.get("tags") or ["text"]
        key = "Image-Based Cards" if "image" in tags else "Text-Based Cards"
        clusters_by_tag[key].append(card["card_name"])

    return [
        Cluster(
            cluster_name=name,
            cluster_description=f"Cards grouped under {name.lower()} for {document_name}.",
            document_id=document_id,
            document_name=document_name,
            card_names=card_names,
        )
        for name, card_names in clusters_by_tag.items()
    ]


def dry_domains(document_id: str, document_name: str, clusters: list[Cluster]) -> list[Domain]:
    return [
        Domain(
            domain_name="Document Overview",
            domain_description=f"Top-level domain for the major clusters in {document_name}.",
            document_id=document_id,
            document_name=document_name,
            cluster_names=[cluster.cluster_name for cluster in clusters],
        )
    ]


def dry_domain_relationships(domains: list[Domain]) -> list[DomainRelationship]:
    relationships: list[DomainRelationship] = []
    for main in domains:
        for related in domains:
            if main.domain_name == related.domain_name:
                continue
            same_document = main.document_id == related.document_id
            relationships.append(
                DomainRelationship(
                    main_domain=main.domain_name,
                    related_domain=related.domain_name,
                    relationship_type="shared_context" if same_document else "comparison",
                    relationship_description=(
                        f"{related.domain_name} may provide related context for "
                        f"{main.domain_name}."
                    ),
                    evidence="Dry-run placeholder relationship based on domain coexistence.",
                    cluster_links=[
                        {
                            "main_cluster": main.cluster_names[0] if main.cluster_names else "",
                            "related_cluster": related.cluster_names[0] if related.cluster_names else "",
                            "relationship": "Dry-run placeholder cluster connection.",
                        }
                    ],
                    card_links=[],
                    document_scope="same_document" if same_document else "cross_document",
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
            result = openrouter_json(client, model, clusters_prompt(document_name, cards))
            raw_clusters = result.get("clusters", [])
        except Exception as exc:
            print(f"Warning: cluster grouping failed for {document_name}; using deterministic grouping. {exc}")
            raw_clusters = [asdict(item) for item in dry_clusters(document_id, document_name, cards)]

    card_names = {card["card_name"] for card in cards}
    assigned: set[str] = set()
    clusters: list[Cluster] = []

    for raw in raw_clusters:
        names = [name for name in raw.get("card_names", []) if name in card_names]
        if not names:
            continue
        assigned.update(names)
        clusters.append(
            Cluster(
                cluster_name=unique_versioned_name(raw.get("cluster_name", "Cluster"), used_names),
                cluster_description=str(raw.get("cluster_description", "")).strip(),
                document_id=document_id,
                document_name=document_name,
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
            clusters.append(
                Cluster(
                    cluster_name=unique_versioned_name(fallback_name, used_names),
                    cluster_description=fallback_description,
                    document_id=document_id,
                    document_name=document_name,
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
            result = openrouter_json(
                client,
                model,
                domains_prompt(document_name, [asdict(cluster) for cluster in clusters]),
            )
            raw_domains = result.get("domains", [])
        except Exception as exc:
            print(f"Warning: domain grouping failed for {document_name}; using deterministic grouping. {exc}")
            raw_domains = [asdict(item) for item in dry_domains(document_id, document_name, clusters)]

    cluster_names = {cluster.cluster_name for cluster in clusters}
    assigned: set[str] = set()
    domains: list[Domain] = []

    for raw in raw_domains:
        names = [name for name in raw.get("cluster_names", []) if name in cluster_names]
        if not names:
            continue
        assigned.update(names)
        domains.append(
            Domain(
                domain_name=unique_versioned_name(raw.get("domain_name", "Domain"), used_names),
                domain_description=str(raw.get("domain_description", "")).strip(),
                document_id=document_id,
                document_name=document_name,
                cluster_names=names,
            )
        )

    missing = sorted(cluster_names - assigned)
    if missing:
        cluster_by_name = {cluster.cluster_name: cluster for cluster in clusters}
        for cluster_name in missing:
            cluster = cluster_by_name[cluster_name]
            fallback_name, fallback_description = semantic_fallback_domain(
                client,
                model,
                document_name,
                [cluster],
            )
            domains.append(
                Domain(
                    domain_name=unique_versioned_name(fallback_name, used_names),
                    domain_description=fallback_description,
                    document_id=document_id,
                    document_name=document_name,
                    cluster_names=[cluster_name],
                )
            )

    return domains


def create_client() -> tuple[OpenAI, str]:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit("Set OPENROUTER_API_KEY in .env, or run with --dry-run.")
    client = OpenAI(
        base_url=os.getenv("OPENROUTER_BASE_URL", OPENROUTER_BASE_URL),
        api_key=api_key,
        default_headers={
            "HTTP-Referer": os.getenv("OPENROUTER_SITE_URL", "http://localhost"),
            "X-Title": os.getenv("OPENROUTER_APP_NAME", "Evidence Mesh"),
        },
    )
    return client, os.getenv("OPENROUTER_MAP_MODEL", os.getenv("OPENROUTER_MODEL", DEFAULT_MODEL))


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
    result = openrouter_json(
        client,
        model,
        domain_relationship_prompt(main, related, cluster_lookup, card_lookup),
    )
    document_scope = (
        "same_document"
        if main.get("document_id") == related.get("document_id")
        else "cross_document"
    )
    extracted: list[DomainRelationship] = []
    for raw in result.get("relationships", []):
        relationship_type = clean_name(raw.get("relationship_type", "other"), "other")
        extracted.append(
            DomainRelationship(
                main_domain=main["domain_name"],
                related_domain=related["domain_name"],
                relationship_type=relationship_type,
                relationship_description=str(raw.get("relationship_description", "")).strip(),
                evidence=str(raw.get("evidence", "")).strip(),
                cluster_links=normalize_links(
                    raw.get("cluster_links", []),
                    ["main_cluster", "related_cluster", "relationship"],
                ),
                card_links=normalize_links(
                    raw.get("card_links", []),
                    ["main_card", "related_card", "relationship"],
                ),
                document_scope=document_scope,
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
    if client is None:
        relationships = dry_domain_relationships(domains)
        if on_progress is not None:
            total_pairs = max(0, len(domains) * (len(domains) - 1))
            all_pair_keys = {
                domain_pair_key(main.domain_name, related.domain_name)
                for main in domains
                for related in domains
                if main.domain_name != related.domain_name
            }
            on_progress(relationships, total_pairs, total_pairs, all_pair_keys)
        return relationships, {}, all_pair_keys

    card_lookup, cluster_lookup = build_lookups(cards, clusters)
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
    workers = relationship_worker_count(max_workers)

    pending_pairs: list[tuple[dict[str, Any], dict[str, Any], str]] = []
    for main in domain_dicts:
        for related in domain_dicts:
            if main["domain_name"] == related["domain_name"]:
                continue
            pair_key = domain_pair_key(main["domain_name"], related["domain_name"])
            if pair_key not in processed_pair_keys:
                pending_pairs.append((main, related, pair_key))

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
            pair_key, extracted, error = future.result()
            if error:
                print(f"ERROR: {error}")
                failed_pair_errors[pair_key] = error
                if on_progress is not None:
                    on_progress(relationships, relationship_checks_done, relationship_checks_total, processed_pair_keys)
                continue
            relationships.extend(extracted)
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
) -> None:
    knowledge_graph = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "clusters": [asdict(cluster) for cluster in clusters],
        "domains": [asdict(domain) for domain in domains],
        "domain_relationships": [asdict(relationship) for relationship in domain_relationships],
        "relationship_checks_done": relationship_checks_done,
        "relationship_checks_total": relationship_checks_total,
        "processed_relationship_checks": sorted(processed_pair_keys or []),
        "failed_relationship_checks": failed_pair_errors or {},
    }
    write_json(KNOWLEDGE_GRAPH_PATH, knowledge_graph)
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


def build_knowledge_graph(dry_run: bool, relationship_workers: int | None = None) -> None:
    load_dotenv()
    cards = read_json(CARD_INDEX_PATH, [])
    if not cards:
        raise SystemExit("No card index found. Run indexer.py first.")

    client = None
    model = os.getenv("OPENROUTER_MAP_MODEL", os.getenv("OPENROUTER_MODEL", DEFAULT_MODEL))
    if not dry_run:
        client, model = create_client()

    grouped = group_cards_by_document(cards)
    used_cluster_names: set[str] = set()
    used_domain_names: set[str] = set()
    all_clusters: list[Cluster] = []
    all_domains: list[Domain] = []
    domain_relationships: list[DomainRelationship] = []
    processed_pair_keys: set[str] = set()
    existing_map = read_json(KNOWLEDGE_GRAPH_PATH, {})
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

    if can_resume_relationships:
        print("Resuming domain-to-domain relationships from the existing partial knowledge graph...")
        all_clusters = [
            Cluster(
                cluster_name=str(raw.get("cluster_name", "")).strip(),
                cluster_description=str(raw.get("cluster_description", "")).strip(),
                document_id=str(raw.get("document_id", "")).strip(),
                document_name=str(raw.get("document_name", "")).strip(),
                card_names=list(raw.get("card_names", [])),
            )
            for raw in existing_map.get("clusters", [])
        ]
        all_domains = [
            Domain(
                domain_name=str(raw.get("domain_name", "")).strip(),
                domain_description=str(raw.get("domain_description", "")).strip(),
                document_id=str(raw.get("document_id", "")).strip(),
                document_name=str(raw.get("document_name", "")).strip(),
                cluster_names=list(raw.get("cluster_names", [])),
            )
            for raw in existing_map.get("domains", [])
        ]
        domain_relationships = [
            relationship_from_dict(raw)
            for raw in existing_map.get("domain_relationships", [])
            if raw.get("main_domain") and raw.get("related_domain")
        ]
        processed_pair_keys = set(
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
        )
    else:
        write_knowledge_graph_progress(all_clusters, all_domains, domain_relationships, "started")

        for document_id, document_cards in grouped.items():
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
            )

    print("Extracting domain-to-domain relationships...")
    current_failed_pair_errors: dict[str, str] = {}

    def relationship_progress(
        relationships: list[DomainRelationship],
        pairs_done: int,
        pairs_total: int,
        pair_keys: set[str],
    ) -> None:
        write_knowledge_graph_progress(
            all_clusters,
            all_domains,
            relationships,
            "building_relationships",
            pairs_done,
            pairs_total,
            set(pair_keys),
            current_failed_pair_errors,
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
    write_knowledge_graph_progress(
        all_clusters,
        all_domains,
        domain_relationships,
        final_status,
        final_checks_done,
        total_relationship_checks,
        final_pair_keys,
        failed_pair_errors,
    )
    print(
        f"Wrote {len(all_clusters)} clusters, {len(all_domains)} domains, "
        f"and {len(domain_relationships)} domain relationships "
        f"to {KNOWLEDGE_GRAPH_PATH}."
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
    return parser


def main() -> None:
    args = build_parser().parse_args()
    build_knowledge_graph(dry_run=args.dry_run, relationship_workers=args.relationship_workers)


if __name__ == "__main__":
    main()




