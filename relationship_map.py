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


PROJECT_ROOT = Path(os.getenv("PDF_VISION_RAG_ROOT", Path(__file__).resolve().parent)).resolve()
INDEXES_DIR = PROJECT_ROOT / "indexes"
TOPIC_INDEX_PATH = INDEXES_DIR / "topic_index.json"
RELATIONSHIP_MAP_PATH = INDEXES_DIR / "relationship_map.json"
PIPELINE_PROGRESS_PATH = PROJECT_ROOT / "logs" / "pipeline_progress.json"

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "google/gemini-3.1-flash-lite"
DEFAULT_RELATION_WORKERS = 1
MAX_RELATION_PAIR_ATTEMPTS = 3


@dataclass
class Community:
    community_name: str
    community_description: str
    document_id: str
    document_name: str
    topic_names: list[str]


@dataclass
class Biome:
    biome_name: str
    biome_description: str
    document_id: str
    document_name: str
    community_names: list[str]


@dataclass
class BiomeRelation:
    main_biome: str
    related_biome: str
    relation_type: str
    relation_description: str
    evidence: str
    community_links: list[dict[str, str]]
    topic_links: list[dict[str, str]]
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


def group_topics_by_document(topics: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for topic in topics:
        grouped[topic.get("document_id", "unknown")].append(topic)
    return dict(grouped)


def topic_payload(topics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "topic_name": topic.get("topic_name", ""),
            "topic_description": topic.get("topic_description", ""),
            "page_no": topic.get("page_no"),
            "topic_source": topic.get("topic_source", ""),
            "tags": topic.get("tags", []),
            "content_excerpt": str(topic.get("content", ""))[:700],
        }
        for topic in topics
    ]


def communities_prompt(document_name: str, topics: list[dict[str, Any]]) -> str:
    return f"""
Create topic communities within one document.

Document: {document_name}

Topics:
{json.dumps(topic_payload(topics), ensure_ascii=False)}

Rules:
1. Group similar or strongly related topics into communities.
2. A community should represent a coherent section, theme, workstream, evidence group, or conceptual area.
3. Every topic must appear in exactly one community.
4. Use the exact topic names provided.
5. Community names must be short, stable, and descriptive.
6. Do not create duplicate community names in this response.

Return only valid JSON:
{{
  "communities": [
    {{
      "community_name": "Community Name",
      "community_description": "What this community represents",
      "topic_names": ["Exact Topic Name"]
    }}
  ]
}}
""".strip()


def biomes_prompt(document_name: str, communities: list[dict[str, Any]]) -> str:
    payload = [
        {
            "community_name": community["community_name"],
            "community_description": community["community_description"],
            "topic_names": community["topic_names"],
        }
        for community in communities
    ]
    return f"""
Group related communities into larger biomes within one document.

Document: {document_name}

Communities:
{json.dumps(payload, ensure_ascii=False)}

Rules:
1. Group similar or connected communities into biomes.
2. A biome should represent a broad document area, major theme, business domain, or major analytical region.
3. Every community must appear in exactly one biome.
4. Use the exact community names provided.
5. Biome names must be short, stable, and descriptive.
6. Do not create duplicate biome names in this response.

Return only valid JSON:
{{
  "biomes": [
    {{
      "biome_name": "Biome Name",
      "biome_description": "What this biome represents",
      "community_names": ["Exact Community Name"]
    }}
  ]
}}
""".strip()


def fallback_community_prompt(document_name: str, topics: list[dict[str, Any]]) -> str:
    return f"""
Name and describe a community for these leftover topics.

Document: {document_name}

Topics:
{json.dumps(topic_payload(topics), ensure_ascii=False)}

Rules:
1. Do not use words like unassigned, miscellaneous, other, leftover, or uncategorized.
2. The name must help search find these topics.
3. Prefer domain terms visible in the topics.
4. Keep the name short and descriptive.

Return only valid JSON:
{{
  "community_name": "Specific Searchable Community Name",
  "community_description": "Concrete description of the topics and when this community is relevant"
}}
""".strip()


def fallback_biome_prompt(document_name: str, communities: list[Community]) -> str:
    payload = [
        {
            "community_name": community.community_name,
            "community_description": community.community_description,
            "topic_names": community.topic_names,
        }
        for community in communities
    ]
    return f"""
Name and describe a biome for these leftover communities.

Document: {document_name}

Communities:
{json.dumps(payload, ensure_ascii=False)}

Rules:
1. Do not use words like unassigned, miscellaneous, other, leftover, or uncategorized.
2. The name must help search find these communities.
3. Prefer domain terms visible in the communities.
4. Keep the name short and descriptive.

Return only valid JSON:
{{
  "biome_name": "Specific Searchable Biome Name",
  "biome_description": "Concrete description of the communities and when this biome is relevant"
}}
""".strip()


def common_terms(text: str, limit: int = 5) -> list[str]:
    stopwords = {
        "the", "and", "for", "with", "from", "this", "that", "shall", "work",
        "document", "topic", "topics", "page", "details", "detail", "system",
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


def local_fallback_community(topics: list[dict[str, Any]]) -> tuple[str, str]:
    text = " ".join(
        [
            str(topic.get("topic_name", "")) + " "
            + str(topic.get("topic_description", "")) + " "
            + str(topic.get("content", ""))[:300]
            for topic in topics
        ]
    )
    terms = common_terms(text)
    name = " ".join(terms[:4]) if terms else "Document Topic Cluster"
    description = (
        f"Topics covering {', '.join(terms)}."
        if terms
        else "Topics grouped together because they were not captured by narrower generated communities."
    )
    return name, description


def local_fallback_biome(communities: list[Community]) -> tuple[str, str]:
    text = " ".join(
        [
            community.community_name + " "
            + community.community_description + " "
            + " ".join(community.topic_names[:20])
            for community in communities
        ]
    )
    terms = common_terms(text)
    name = " ".join(terms[:4]) if terms else "Document Knowledge Area"
    description = (
        f"Biome covering communities about {', '.join(terms)}."
        if terms
        else "Biome grouping related document communities for search and retrieval."
    )
    return name, description


def biome_pair_key(main_biome: str, related_biome: str) -> str:
    return f"{main_biome} -> {related_biome}"


def relation_from_dict(raw: dict[str, Any]) -> BiomeRelation:
    return BiomeRelation(
        main_biome=str(raw.get("main_biome", "")).strip(),
        related_biome=str(raw.get("related_biome", "")).strip(),
        relation_type=str(raw.get("relation_type", "other")).strip() or "other",
        relation_description=str(raw.get("relation_description", "")).strip(),
        evidence=str(raw.get("evidence", "")).strip(),
        community_links=raw.get("community_links", []) if isinstance(raw.get("community_links"), list) else [],
        topic_links=raw.get("topic_links", []) if isinstance(raw.get("topic_links"), list) else [],
        document_scope=str(raw.get("document_scope", "")).strip(),
    )


def semantic_fallback_community(
    client: OpenAI | None,
    model: str,
    document_name: str,
    topics: list[dict[str, Any]],
) -> tuple[str, str]:
    if client is not None:
        try:
            result = openrouter_json(client, model, fallback_community_prompt(document_name, topics))
            name = str(result.get("community_name", "")).strip()
            description = str(result.get("community_description", "")).strip()
            if name and description:
                return name, description
        except Exception as exc:
            print(f"Warning: fallback community naming failed; using local name. {exc}")
    return local_fallback_community(topics)


def semantic_fallback_biome(
    client: OpenAI | None,
    model: str,
    document_name: str,
    communities: list[Community],
) -> tuple[str, str]:
    if client is not None:
        try:
            result = openrouter_json(client, model, fallback_biome_prompt(document_name, communities))
            name = str(result.get("biome_name", "")).strip()
            description = str(result.get("biome_description", "")).strip()
            if name and description:
                return name, description
        except Exception as exc:
            print(f"Warning: fallback biome naming failed; using local name. {exc}")
    return local_fallback_biome(communities)


def biome_relation_prompt(
    main_biome: dict[str, Any],
    related_biome: dict[str, Any],
    community_lookup: dict[str, dict[str, Any]],
    topic_lookup: dict[str, dict[str, Any]],
) -> str:
    def expand_biome(biome: dict[str, Any]) -> dict[str, Any]:
        communities = []
        for community_name in biome.get("community_names", []):
            community = community_lookup.get(community_name)
            if not community:
                continue
            topics = [
                {
                    "topic_name": topic_name,
                    "topic_description": topic_lookup.get(topic_name, {}).get("topic_description", ""),
                    "page_no": topic_lookup.get(topic_name, {}).get("page_no"),
                    "topic_source": topic_lookup.get(topic_name, {}).get("topic_source", ""),
                    "tags": topic_lookup.get(topic_name, {}).get("tags", []),
                    "content_excerpt": str(topic_lookup.get(topic_name, {}).get("content", ""))[:600],
                }
                for topic_name in community.get("topic_names", [])
            ]
            communities.append(
                {
                    "community_name": community_name,
                    "community_description": community.get("community_description", ""),
                    "topics": topics,
                }
            )
        return {
            "biome_name": biome.get("biome_name", ""),
            "biome_description": biome.get("biome_description", ""),
            "document_name": biome.get("document_name", ""),
            "communities": communities,
        }

    return f"""
Find every meaningful relationship from the main biome to the related biome.

Important:
- The relationship is directional: main_biome -> related_biome.
- One biome may have multiple relations with another biome.
- Return all meaningful relations, not just the strongest one.
- If there is no meaningful relation, return an empty relations array.

Relation examples:
- continuation: related biome continues or extends information from main biome
- dependency: main biome depends on information from related biome
- prerequisite: related biome is needed before main biome can be understood
- evidence: related biome provides evidence/support for main biome
- contradiction: related biome conflicts with main biome
- comparison: related biome gives a comparable/contrasting view
- cause_effect: main biome causes, enables, blocks, or affects related biome
- shared_context: both biomes describe the same larger context from different angles
- visual_support: related biome contains image/chart/map evidence for main biome

Main biome:
{json.dumps(expand_biome(main_biome), ensure_ascii=False)}

Related biome:
{json.dumps(expand_biome(related_biome), ensure_ascii=False)}

Return only valid JSON:
{{
  "relations": [
    {{
      "relation_type": "continuation | dependency | prerequisite | evidence | contradiction | comparison | cause_effect | shared_context | visual_support | other",
      "relation_description": "Exactly how the related biome relates to the main biome",
      "evidence": "Short evidence using community/topic names, page numbers, or content",
      "community_links": [
        {{
          "main_community": "Exact community name from main biome",
          "related_community": "Exact community name from related biome",
          "relation": "How these communities connect for this relation"
        }}
      ],
      "topic_links": [
        {{
          "main_topic": "Exact topic name from main biome",
          "related_topic": "Exact topic name from related biome",
          "relation": "How these topics connect for this relation"
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
                    "content": "You organize document topics into relationship maps. Return only valid JSON.",
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
    topics: list[dict[str, Any]],
    communities: list[Community],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    topic_lookup = {topic["topic_name"]: topic for topic in topics}
    community_lookup = {
        community.community_name: asdict(community) for community in communities
    }
    return topic_lookup, community_lookup


def dry_communities(document_id: str, document_name: str, topics: list[dict[str, Any]]) -> list[Community]:
    communities_by_tag: dict[str, list[str]] = defaultdict(list)
    for topic in topics:
        tags = topic.get("tags") or ["text"]
        key = "Image-Based Topics" if "image" in tags else "Text-Based Topics"
        communities_by_tag[key].append(topic["topic_name"])

    return [
        Community(
            community_name=name,
            community_description=f"Topics grouped under {name.lower()} for {document_name}.",
            document_id=document_id,
            document_name=document_name,
            topic_names=topic_names,
        )
        for name, topic_names in communities_by_tag.items()
    ]


def dry_biomes(document_id: str, document_name: str, communities: list[Community]) -> list[Biome]:
    return [
        Biome(
            biome_name="Document Overview",
            biome_description=f"Top-level biome for the major communities in {document_name}.",
            document_id=document_id,
            document_name=document_name,
            community_names=[community.community_name for community in communities],
        )
    ]


def dry_biome_relations(biomes: list[Biome]) -> list[BiomeRelation]:
    relations: list[BiomeRelation] = []
    for main in biomes:
        for related in biomes:
            if main.biome_name == related.biome_name:
                continue
            same_document = main.document_id == related.document_id
            relations.append(
                BiomeRelation(
                    main_biome=main.biome_name,
                    related_biome=related.biome_name,
                    relation_type="shared_context" if same_document else "comparison",
                    relation_description=(
                        f"{related.biome_name} may provide related context for "
                        f"{main.biome_name}."
                    ),
                    evidence="Dry-run placeholder relation based on biome coexistence.",
                    community_links=[
                        {
                            "main_community": main.community_names[0] if main.community_names else "",
                            "related_community": related.community_names[0] if related.community_names else "",
                            "relation": "Dry-run placeholder community connection.",
                        }
                    ],
                    topic_links=[],
                    document_scope="same_document" if same_document else "cross_document",
                )
            )
    return relations


def build_communities(
    client: OpenAI | None,
    model: str,
    document_id: str,
    document_name: str,
    topics: list[dict[str, Any]],
    used_names: set[str],
) -> list[Community]:
    if client is None:
        raw_communities = [asdict(item) for item in dry_communities(document_id, document_name, topics)]
    else:
        try:
            result = openrouter_json(client, model, communities_prompt(document_name, topics))
            raw_communities = result.get("communities", [])
        except Exception as exc:
            print(f"Warning: community grouping failed for {document_name}; using deterministic grouping. {exc}")
            raw_communities = [asdict(item) for item in dry_communities(document_id, document_name, topics)]

    topic_names = {topic["topic_name"] for topic in topics}
    assigned: set[str] = set()
    communities: list[Community] = []

    for raw in raw_communities:
        names = [name for name in raw.get("topic_names", []) if name in topic_names]
        if not names:
            continue
        assigned.update(names)
        communities.append(
            Community(
                community_name=unique_versioned_name(raw.get("community_name", "Community"), used_names),
                community_description=str(raw.get("community_description", "")).strip(),
                document_id=document_id,
                document_name=document_name,
                topic_names=names,
            )
        )

    missing = sorted(topic_names - assigned)
    if missing:
        topic_by_name = {topic["topic_name"]: topic for topic in topics}
        for topic_name in missing:
            topic = topic_by_name[topic_name]
            fallback_name, fallback_description = semantic_fallback_community(
                client,
                model,
                document_name,
                [topic],
            )
            communities.append(
                Community(
                    community_name=unique_versioned_name(fallback_name, used_names),
                    community_description=fallback_description,
                    document_id=document_id,
                    document_name=document_name,
                    topic_names=[topic_name],
                )
            )

    return communities


def build_biomes(
    client: OpenAI | None,
    model: str,
    document_id: str,
    document_name: str,
    communities: list[Community],
    used_names: set[str],
) -> list[Biome]:
    if client is None:
        raw_biomes = [asdict(item) for item in dry_biomes(document_id, document_name, communities)]
    else:
        try:
            result = openrouter_json(
                client,
                model,
                biomes_prompt(document_name, [asdict(community) for community in communities]),
            )
            raw_biomes = result.get("biomes", [])
        except Exception as exc:
            print(f"Warning: biome grouping failed for {document_name}; using deterministic grouping. {exc}")
            raw_biomes = [asdict(item) for item in dry_biomes(document_id, document_name, communities)]

    community_names = {community.community_name for community in communities}
    assigned: set[str] = set()
    biomes: list[Biome] = []

    for raw in raw_biomes:
        names = [name for name in raw.get("community_names", []) if name in community_names]
        if not names:
            continue
        assigned.update(names)
        biomes.append(
            Biome(
                biome_name=unique_versioned_name(raw.get("biome_name", "Biome"), used_names),
                biome_description=str(raw.get("biome_description", "")).strip(),
                document_id=document_id,
                document_name=document_name,
                community_names=names,
            )
        )

    missing = sorted(community_names - assigned)
    if missing:
        community_by_name = {community.community_name: community for community in communities}
        for community_name in missing:
            community = community_by_name[community_name]
            fallback_name, fallback_description = semantic_fallback_biome(
                client,
                model,
                document_name,
                [community],
            )
            biomes.append(
                Biome(
                    biome_name=unique_versioned_name(fallback_name, used_names),
                    biome_description=fallback_description,
                    document_id=document_id,
                    document_name=document_name,
                    community_names=[community_name],
                )
            )

    return biomes


def create_client() -> tuple[OpenAI, str]:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit("Set OPENROUTER_API_KEY in .env, or run with --dry-run.")
    client = OpenAI(
        base_url=os.getenv("OPENROUTER_BASE_URL", OPENROUTER_BASE_URL),
        api_key=api_key,
        default_headers={
            "HTTP-Referer": os.getenv("OPENROUTER_SITE_URL", "http://localhost"),
            "X-Title": os.getenv("OPENROUTER_APP_NAME", "PDF Vision RAG"),
        },
    )
    return client, os.getenv("OPENROUTER_MAP_MODEL", os.getenv("OPENROUTER_MODEL", DEFAULT_MODEL))


def relation_worker_count(explicit_workers: int | None = None) -> int:
    if explicit_workers is not None:
        return max(1, explicit_workers)
    raw_value = os.getenv("OPENROUTER_RELATION_WORKERS", str(DEFAULT_RELATION_WORKERS))
    try:
        return max(1, int(raw_value))
    except ValueError:
        print(f"Warning: invalid OPENROUTER_RELATION_WORKERS={raw_value!r}; using 1.")
        return 1


def extract_biome_relation_pair(
    client: OpenAI,
    model: str,
    main: dict[str, Any],
    related: dict[str, Any],
    community_lookup: dict[str, dict[str, Any]],
    topic_lookup: dict[str, dict[str, Any]],
) -> list[BiomeRelation]:
    result = openrouter_json(
        client,
        model,
        biome_relation_prompt(main, related, community_lookup, topic_lookup),
    )
    document_scope = (
        "same_document"
        if main.get("document_id") == related.get("document_id")
        else "cross_document"
    )
    extracted: list[BiomeRelation] = []
    for raw in result.get("relations", []):
        relation_type = clean_name(raw.get("relation_type", "other"), "other")
        extracted.append(
            BiomeRelation(
                main_biome=main["biome_name"],
                related_biome=related["biome_name"],
                relation_type=relation_type,
                relation_description=str(raw.get("relation_description", "")).strip(),
                evidence=str(raw.get("evidence", "")).strip(),
                community_links=normalize_links(
                    raw.get("community_links", []),
                    ["main_community", "related_community", "relation"],
                ),
                topic_links=normalize_links(
                    raw.get("topic_links", []),
                    ["main_topic", "related_topic", "relation"],
                ),
                document_scope=document_scope,
            )
        )
    return extracted


def extract_biome_relation_pair_with_retries(
    client: OpenAI,
    model: str,
    main: dict[str, Any],
    related: dict[str, Any],
    community_lookup: dict[str, dict[str, Any]],
    topic_lookup: dict[str, dict[str, Any]],
    max_attempts: int = MAX_RELATION_PAIR_ATTEMPTS,
) -> list[BiomeRelation]:
    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            if attempt > 1:
                print(
                    f"Retrying biome relation pair attempt {attempt}/{max_attempts}: "
                    f"{main['biome_name']} -> {related['biome_name']}..."
                )
                time.sleep(min(8.0, 1.5 * attempt))
            return extract_biome_relation_pair(
                client,
                model,
                main,
                related,
                community_lookup,
                topic_lookup,
            )
        except Exception as exc:
            last_error = exc
            print(
                f"Warning: relation extraction attempt {attempt}/{max_attempts} failed for "
                f"{main['biome_name']} -> {related['biome_name']}. {exc}"
            )
    raise last_error or RuntimeError("Relation extraction failed without an exception.")


def build_biome_relations(
    client: OpenAI | None,
    model: str,
    topics: list[dict[str, Any]],
    communities: list[Community],
    biomes: list[Biome],
    on_progress: Any | None = None,
    existing_relations: list[BiomeRelation] | None = None,
    processed_pair_keys: set[str] | None = None,
    max_workers: int | None = None,
) -> tuple[list[BiomeRelation], dict[str, str], set[str]]:
    processed_pair_keys = processed_pair_keys or set()
    if client is None:
        relations = dry_biome_relations(biomes)
        if on_progress is not None:
            total_pairs = max(0, len(biomes) * (len(biomes) - 1))
            all_pair_keys = {
                biome_pair_key(main.biome_name, related.biome_name)
                for main in biomes
                for related in biomes
                if main.biome_name != related.biome_name
            }
            on_progress(relations, total_pairs, total_pairs, all_pair_keys)
        return relations, {}, all_pair_keys

    topic_lookup, community_lookup = build_lookups(topics, communities)
    biome_dicts = [asdict(biome) for biome in biomes]
    relations: list[BiomeRelation] = list(existing_relations or [])
    failed_pair_errors: dict[str, str] = {}
    processed_pair_keys.update(
        biome_pair_key(relation.main_biome, relation.related_biome)
        for relation in relations
        if relation.main_biome and relation.related_biome
    )
    relation_pairs_total = max(0, len(biome_dicts) * (len(biome_dicts) - 1))
    relation_pairs_done = len(processed_pair_keys)
    workers = relation_worker_count(max_workers)

    pending_pairs: list[tuple[dict[str, Any], dict[str, Any], str]] = []
    for main in biome_dicts:
        for related in biome_dicts:
            if main["biome_name"] == related["biome_name"]:
                continue
            pair_key = biome_pair_key(main["biome_name"], related["biome_name"])
            if pair_key not in processed_pair_keys:
                pending_pairs.append((main, related, pair_key))

    if workers <= 1 or len(pending_pairs) <= 1:
        for main, related, pair_key in pending_pairs:
            print(f"Extracting biome relations: {main['biome_name']} -> {related['biome_name']}...")
            try:
                extracted = extract_biome_relation_pair_with_retries(
                    client,
                    model,
                    main,
                    related,
                    community_lookup,
                    topic_lookup,
                )
            except Exception as exc:
                message = (
                    f"Failed after {MAX_RELATION_PAIR_ATTEMPTS} attempts for "
                    f"{main['biome_name']} -> {related['biome_name']}: {exc}"
                )
                print(f"ERROR: {message}")
                failed_pair_errors[pair_key] = message
                if on_progress is not None:
                    on_progress(relations, relation_pairs_done, relation_pairs_total, processed_pair_keys)
                continue
            relations.extend(extracted)
            processed_pair_keys.add(pair_key)
            relation_pairs_done += 1
            if on_progress is not None:
                on_progress(relations, relation_pairs_done, relation_pairs_total, processed_pair_keys)
        return relations, failed_pair_errors, processed_pair_keys

    print(f"Extracting biome relations with {workers} parallel workers...")
    worker_local = threading.local()

    def worker_client() -> OpenAI:
        existing_client = getattr(worker_local, "client", None)
        if existing_client is None:
            existing_client, _ = create_client()
            worker_local.client = existing_client
        return existing_client

    def process_pair(pair: tuple[dict[str, Any], dict[str, Any], str]) -> tuple[str, list[BiomeRelation], str | None]:
        main, related, pair_key = pair
        print(f"Extracting biome relations: {main['biome_name']} -> {related['biome_name']}...")
        try:
            extracted = extract_biome_relation_pair_with_retries(
                worker_client(),
                model,
                main,
                related,
                community_lookup,
                topic_lookup,
            )
            return pair_key, extracted, None
        except Exception as exc:
            main, related, _ = pair
            return pair_key, [], (
                f"Failed after {MAX_RELATION_PAIR_ATTEMPTS} attempts for "
                f"{main['biome_name']} -> {related['biome_name']}: {exc}"
            )

    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_pair = {executor.submit(process_pair, pair): pair for pair in pending_pairs}
        for future in as_completed(future_to_pair):
            pair_key, extracted, error = future.result()
            if error:
                print(f"ERROR: {error}")
                failed_pair_errors[pair_key] = error
                if on_progress is not None:
                    on_progress(relations, relation_pairs_done, relation_pairs_total, processed_pair_keys)
                continue
            relations.extend(extracted)
            processed_pair_keys.add(pair_key)
            relation_pairs_done += 1
            if on_progress is not None:
                on_progress(relations, relation_pairs_done, relation_pairs_total, processed_pair_keys)

    return relations, failed_pair_errors, processed_pair_keys


def write_relationship_map_progress(
    communities: list[Community],
    biomes: list[Biome],
    biome_relations: list[BiomeRelation],
    status: str,
    relation_pairs_done: int = 0,
    relation_pairs_total: int = 0,
    processed_pair_keys: set[str] | None = None,
    failed_pair_errors: dict[str, str] | None = None,
) -> None:
    relationship_map = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "communities": [asdict(community) for community in communities],
        "biomes": [asdict(biome) for biome in biomes],
        "biome_relations": [asdict(relation) for relation in biome_relations],
        "relation_pairs_done": relation_pairs_done,
        "relation_pairs_total": relation_pairs_total,
        "processed_relation_pairs": sorted(processed_pair_keys or []),
        "failed_relation_pairs": failed_pair_errors or {},
    }
    write_json(RELATIONSHIP_MAP_PATH, relationship_map)
    write_pipeline_progress(
        {
            "stage": "relationship_map" if status != "complete" else "complete",
            "message": f"Relationship map status: {status}.",
            "relationship_status": status,
            "community_count": len(communities),
            "biome_count": len(biomes),
            "relation_count": len(biome_relations),
            "relation_pairs_done": relation_pairs_done,
            "relation_pairs_total": relation_pairs_total,
            "failed_relation_pair_count": len(failed_pair_errors or {}),
        }
    )


def build_relationship_map(dry_run: bool, relation_workers: int | None = None) -> None:
    load_dotenv()
    topics = read_json(TOPIC_INDEX_PATH, [])
    if not topics:
        raise SystemExit("No topic index found. Run indexer.py first.")

    client = None
    model = os.getenv("OPENROUTER_MAP_MODEL", os.getenv("OPENROUTER_MODEL", DEFAULT_MODEL))
    if not dry_run:
        client, model = create_client()

    grouped = group_topics_by_document(topics)
    used_community_names: set[str] = set()
    used_biome_names: set[str] = set()
    all_communities: list[Community] = []
    all_biomes: list[Biome] = []
    biome_relations: list[BiomeRelation] = []
    processed_pair_keys: set[str] = set()
    existing_map = read_json(RELATIONSHIP_MAP_PATH, {})
    existing_document_ids = {
        item.get("document_id")
        for item in existing_map.get("communities", [])
        if item.get("document_id")
    }
    expected_document_ids = set(grouped)
    can_resume_relations = (
        existing_map.get("status") in {"building_relations", "failed_relations"}
        and existing_document_ids == expected_document_ids
        and existing_map.get("communities")
        and existing_map.get("biomes")
    )

    if can_resume_relations:
        print("Resuming biome-to-biome relations from the existing partial relationship map...")
        all_communities = [
            Community(
                community_name=str(raw.get("community_name", "")).strip(),
                community_description=str(raw.get("community_description", "")).strip(),
                document_id=str(raw.get("document_id", "")).strip(),
                document_name=str(raw.get("document_name", "")).strip(),
                topic_names=list(raw.get("topic_names", [])),
            )
            for raw in existing_map.get("communities", [])
        ]
        all_biomes = [
            Biome(
                biome_name=str(raw.get("biome_name", "")).strip(),
                biome_description=str(raw.get("biome_description", "")).strip(),
                document_id=str(raw.get("document_id", "")).strip(),
                document_name=str(raw.get("document_name", "")).strip(),
                community_names=list(raw.get("community_names", [])),
            )
            for raw in existing_map.get("biomes", [])
        ]
        biome_relations = [
            relation_from_dict(raw)
            for raw in existing_map.get("biome_relations", [])
            if raw.get("main_biome") and raw.get("related_biome")
        ]
        processed_pair_keys = set(existing_map.get("processed_relation_pairs", []))
        processed_pair_keys.update(
            biome_pair_key(relation.main_biome, relation.related_biome)
            for relation in biome_relations
            if relation.main_biome and relation.related_biome
        )
        write_relationship_map_progress(
            all_communities,
            all_biomes,
            biome_relations,
            "building_relations",
            len(processed_pair_keys),
            max(0, len(all_biomes) * (len(all_biomes) - 1)),
            processed_pair_keys,
        )
    else:
        write_relationship_map_progress(all_communities, all_biomes, biome_relations, "started")

        for document_id, document_topics in grouped.items():
            document_name = document_topics[0].get("document_name", "Unknown Document")
            print(f"Building communities for {document_name}...")
            communities = build_communities(
                client=client,
                model=model,
                document_id=document_id,
                document_name=document_name,
                topics=document_topics,
                used_names=used_community_names,
            )
            all_communities.extend(communities)

            print(f"Building biomes for {document_name}...")
            biomes = build_biomes(
                client=client,
                model=model,
                document_id=document_id,
                document_name=document_name,
                communities=communities,
                used_names=used_biome_names,
            )
            all_biomes.extend(biomes)
            write_relationship_map_progress(
                all_communities,
                all_biomes,
                biome_relations,
                f"grouped {document_name}",
            )

    print("Extracting biome-to-biome relations...")
    current_failed_pair_errors: dict[str, str] = {}

    def relation_progress(
        relations: list[BiomeRelation],
        pairs_done: int,
        pairs_total: int,
        pair_keys: set[str],
    ) -> None:
        write_relationship_map_progress(
            all_communities,
            all_biomes,
            relations,
            "building_relations",
            pairs_done,
            pairs_total,
            set(pair_keys),
            current_failed_pair_errors,
        )

    biome_relations, failed_pair_errors, processed_pair_keys = build_biome_relations(
        client=client,
        model=model,
        topics=topics,
        communities=all_communities,
        biomes=all_biomes,
        on_progress=relation_progress,
        existing_relations=biome_relations,
        processed_pair_keys=processed_pair_keys,
        max_workers=relation_workers,
    )
    current_failed_pair_errors.update(failed_pair_errors)

    total_relation_pairs = max(0, len(all_biomes) * (len(all_biomes) - 1))
    all_pair_keys = {
        biome_pair_key(main.biome_name, related.biome_name)
        for main in all_biomes
        for related in all_biomes
        if main.biome_name != related.biome_name
    }
    successful_pair_keys = set(processed_pair_keys)
    final_status = "complete"
    final_pair_keys = all_pair_keys
    final_pairs_done = total_relation_pairs
    if failed_pair_errors or len(successful_pair_keys) < total_relation_pairs:
        final_status = "failed_relations"
        final_pair_keys = successful_pair_keys
        final_pairs_done = len(successful_pair_keys)
        if not failed_pair_errors:
            missing_pair_keys = sorted(all_pair_keys - successful_pair_keys)
            failed_pair_errors = {
                key: "Pair did not complete and will be retried on the next run."
                for key in missing_pair_keys
            }
    write_relationship_map_progress(
        all_communities,
        all_biomes,
        biome_relations,
        final_status,
        final_pairs_done,
        total_relation_pairs,
        final_pair_keys,
        failed_pair_errors,
    )
    print(
        f"Wrote {len(all_communities)} communities, {len(all_biomes)} biomes, "
        f"and {len(biome_relations)} biome relations "
        f"to {RELATIONSHIP_MAP_PATH}."
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build topic communities and biomes.")
    parser.add_argument("--dry-run", action="store_true", help="Build without calling OpenRouter.")
    parser.add_argument(
        "--relation-workers",
        type=int,
        default=None,
        help="Number of parallel workers for biome-to-biome relation checks.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    build_relationship_map(dry_run=args.dry_run, relation_workers=args.relation_workers)


if __name__ == "__main__":
    main()
