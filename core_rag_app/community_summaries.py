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
from entity_claims import anchors_for_card
from llm_config import create_chat_client, get_model
from schema import card_id_from_record, read_knowledge_graph
from storage import read_cards


LEGACY_ROOT_ENV = "PDF" + "_VISION_RAG_ROOT"
PROJECT_ROOT = Path(
    os.getenv("EVIDENCE_MESH_ROOT")
    or os.getenv(LEGACY_ROOT_ENV)
    or Path(__file__).resolve().parent
).resolve()
INDEXES_DIR = PROJECT_ROOT / "indexes"
KNOWLEDGE_GRAPH_PATH = INDEXES_DIR / "knowledge_graph.json"
COMMUNITY_SUMMARIES_PATH = INDEXES_DIR / "community_summaries.json"

COMMUNITY_SUMMARY_SCHEMA_VERSION = "community_summaries.v1.0"
COMMUNITY_SUMMARY_PROMPT_VERSION = "community_summary_prompt.v1.0"
DEFAULT_CARD_CONTENT_CHARS = 900
DEFAULT_MAX_CARDS_PER_DOMAIN = 28


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


def compact_text(value: Any, limit: int) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip())[:limit]


def graph_signature(domain: dict[str, Any], clusters: list[dict[str, Any]], cards: list[dict[str, Any]], relationships: list[dict[str, Any]]) -> str:
    payload = {
        "domain": {
            "domain_id": domain.get("domain_id", ""),
            "domain_name": domain.get("domain_name", ""),
            "domain_description": domain.get("domain_description", ""),
            "cluster_ids": domain.get("cluster_ids", []),
            "cluster_names": domain.get("cluster_names", []),
        },
        "clusters": [
            {
                "cluster_id": cluster.get("cluster_id", ""),
                "cluster_name": cluster.get("cluster_name", ""),
                "cluster_description": cluster.get("cluster_description", ""),
                "card_ids": cluster.get("card_ids", []),
                "card_names": cluster.get("card_names", []),
            }
            for cluster in clusters
        ],
        "cards": [
            {
                "card_id": card_id_from_record(card),
                "card_name": card.get("card_name", ""),
                "page_no": card.get("page_no"),
                "content": compact_text(card.get("content", ""), 600),
            }
            for card in cards
        ],
        "relationships": [
            {
                "relationship_id": relationship.get("relationship_id", ""),
                "main_domain": relationship.get("main_domain", ""),
                "related_domain": relationship.get("related_domain", ""),
                "relationship_type": relationship.get("relationship_type", ""),
                "relationship_description": relationship.get("relationship_description", ""),
                "evidence": relationship.get("evidence", ""),
            }
            for relationship in relationships
        ],
    }
    return stable_hash(payload)


def domain_context(
    domain: dict[str, Any],
    graph: dict[str, Any],
    cards: list[dict[str, Any]],
) -> dict[str, Any]:
    clusters_by_name = {
        str(cluster.get("cluster_name", "")): cluster
        for cluster in graph.get("clusters", []) or []
    }
    cards_by_name = {
        str(card.get("card_name", "")): card
        for card in cards
    }
    domain_clusters = [
        clusters_by_name[name]
        for name in domain.get("cluster_names", []) or []
        if name in clusters_by_name
    ]
    domain_cards: list[dict[str, Any]] = []
    seen_cards = set()
    for cluster in domain_clusters:
        for card_name in cluster.get("card_names", []) or []:
            card = cards_by_name.get(str(card_name))
            if not card:
                continue
            card_id = card_id_from_record(card)
            if card_id in seen_cards:
                continue
            seen_cards.add(card_id)
            domain_cards.append(card)
    relationships = [
        relationship
        for relationship in graph.get("domain_relationships", []) or []
        if relationship.get("main_domain") == domain.get("domain_name")
        or relationship.get("related_domain") == domain.get("domain_name")
    ]
    return {"clusters": domain_clusters, "cards": domain_cards, "relationships": relationships}


def summary_input(domain: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    max_cards = max(4, int(os.getenv("EVIDENCE_MESH_SUMMARY_MAX_CARDS", str(DEFAULT_MAX_CARDS_PER_DOMAIN))))
    cards = []
    for card in context["cards"][:max_cards]:
        card_id = card_id_from_record(card)
        anchors = anchors_for_card(card_id)
        cards.append(
            {
                "card_id": card_id,
                "card_name": card.get("card_name", ""),
                "document_name": card.get("document_name", ""),
                "page_no": card.get("page_no"),
                "description": card.get("card_description", ""),
                "content": compact_text(card.get("content", ""), DEFAULT_CARD_CONTENT_CHARS),
                "entities": (anchors.get("entities") or [])[:8],
                "claims": (anchors.get("claims") or [])[:8],
            }
        )
    return {
        "domain": {
            "domain_id": domain.get("domain_id", ""),
            "domain_name": domain.get("domain_name", ""),
            "domain_description": domain.get("domain_description", ""),
            "document_name": domain.get("document_name", ""),
        },
        "clusters": [
            {
                "cluster_id": cluster.get("cluster_id", ""),
                "cluster_name": cluster.get("cluster_name", ""),
                "cluster_description": cluster.get("cluster_description", ""),
                "card_names": cluster.get("card_names", []),
            }
            for cluster in context["clusters"]
        ],
        "relationships": [
            {
                "main_domain": relationship.get("main_domain", ""),
                "related_domain": relationship.get("related_domain", ""),
                "relationship_type": relationship.get("relationship_type", ""),
                "relationship_description": relationship.get("relationship_description", ""),
                "evidence": relationship.get("evidence", ""),
                "confidence_score": relationship.get("confidence_score", 0.0),
            }
            for relationship in context["relationships"][:24]
        ],
        "cards": cards,
    }


def summary_prompt(domain: dict[str, Any], context: dict[str, Any]) -> str:
    return f"""
Create a cited community summary for this document evidence domain.

Input:
{json.dumps(summary_input(domain, context), ensure_ascii=False)}

Rules:
1. Summarize only what the cards support.
2. Keep citations as card/document/page references.
3. Preserve important requirements, obligations, dates, numbers, exceptions, risks, definitions, and contradictions.
4. Mention relationship context when it helps future retrieval.
5. Be compact but information dense.

Return only valid JSON:
{{
  "summary": "Dense domain summary with cited card/page references.",
  "key_points": ["Important supported point with citation"],
  "requirements": ["Requirement or obligation with citation"],
  "metrics": ["Date, amount, threshold, duration, or other measurable detail with citation"],
  "risks": ["Risk, exception, conflict, or uncertainty with citation"],
  "related_context": ["How this domain connects to other domains"],
  "representative_card_ids": ["card_id"]
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


def heuristic_summary(domain: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    cards = context["cards"]
    key_points = []
    requirements = []
    metrics = []
    risks = []
    for card in cards[:12]:
        citation = f"{card.get('document_name', '')}, page {card.get('page_no', '')}"
        text = compact_text(card.get("content", ""), 240)
        if text:
            key_points.append(f"{card.get('card_name', '')}: {text} ({citation})")
        lowered = text.lower()
        if any(term in lowered for term in ("must", "shall", "required", "obligation")):
            requirements.append(f"{text} ({citation})")
        if re.search(r"\b\d+(?:\.\d+)?\s?(?:%|percent|days|months|years)\b|(?:rs\.?|inr|usd|\$)\s?[\d,]+", lowered):
            metrics.append(f"{text} ({citation})")
        if any(term in lowered for term in ("risk", "delay", "penalty", "failure", "except", "unless")):
            risks.append(f"{text} ({citation})")
    related_context = [
        f"{relationship.get('main_domain', '')} -> {relationship.get('related_domain', '')}: "
        f"{relationship.get('relationship_type', '')} - {relationship.get('relationship_description', '')}"
        for relationship in context["relationships"][:8]
    ]
    return {
        "summary": (
            f"{domain.get('domain_name', 'Domain')} covers {domain.get('domain_description', '')}. "
            f"It includes {len(context['clusters'])} cluster(s) and {len(cards)} evidence card(s)."
        ),
        "key_points": key_points[:10],
        "requirements": requirements[:8],
        "metrics": metrics[:8],
        "risks": risks[:8],
        "related_context": related_context,
        "representative_card_ids": [card_id_from_record(card) for card in cards[:8]],
    }


def clean_summary(raw: dict[str, Any], domain: dict[str, Any], context: dict[str, Any], signature: str, model: str) -> dict[str, Any]:
    cards = context["cards"]
    card_ids = {card_id_from_record(card) for card in cards}
    representative_ids = [
        str(card_id)
        for card_id in raw.get("representative_card_ids", []) or []
        if str(card_id) in card_ids
    ][:12]
    if not representative_ids:
        representative_ids = [card_id_from_record(card) for card in cards[:8]]
    return {
        "domain_id": domain.get("domain_id", ""),
        "domain_name": domain.get("domain_name", ""),
        "document_id": domain.get("document_id", ""),
        "document_name": domain.get("document_name", ""),
        "signature": signature,
        "model": model,
        "updated_at": utc_now(),
        "cluster_count": len(context["clusters"]),
        "card_count": len(cards),
        "relationship_count": len(context["relationships"]),
        "summary": compact_text(raw.get("summary", ""), 2200),
        "key_points": [compact_text(item, 500) for item in raw.get("key_points", []) or []][:12],
        "requirements": [compact_text(item, 500) for item in raw.get("requirements", []) or []][:10],
        "metrics": [compact_text(item, 400) for item in raw.get("metrics", []) or []][:10],
        "risks": [compact_text(item, 500) for item in raw.get("risks", []) or []][:10],
        "related_context": [compact_text(item, 500) for item in raw.get("related_context", []) or []][:12],
        "representative_card_ids": representative_ids,
    }


def summarize_domain(
    domain: dict[str, Any],
    graph: dict[str, Any],
    cards: list[dict[str, Any]],
    *,
    client: OpenAI | None,
    model: str,
    dry_run: bool = False,
) -> dict[str, Any]:
    context = domain_context(domain, graph, cards)
    signature = graph_signature(domain, context["clusters"], context["cards"], context["relationships"])
    if dry_run or client is None:
        return clean_summary(heuristic_summary(domain, context), domain, context, signature, "heuristic")
    prompt = summary_prompt(domain, context)
    cache_key = {
        "version": COMMUNITY_SUMMARY_PROMPT_VERSION,
        "domain_id": domain.get("domain_id", ""),
        "signature": signature,
        "model": model,
        "prompt_hash": stable_hash(prompt),
    }
    cached = read_cache("community_summary", cache_key)
    if cached is not None:
        raw = cached.get("value", {})
    else:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You summarize evidence communities for GraphRAG retrieval. Return only valid JSON.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            response_format={"type": "json_object"},
            max_tokens=int(os.getenv("EVIDENCE_MESH_COMMUNITY_SUMMARY_MAX_TOKENS", "2200")),
        )
        raw = parse_json_response(response.choices[0].message.content or "{}")
        write_cache("community_summary", cache_key, raw)
    return clean_summary(raw, domain, context, signature, model)


def read_community_summaries(path: Path = COMMUNITY_SUMMARIES_PATH) -> dict[str, Any]:
    payload = read_json(path, {})
    if not isinstance(payload, dict):
        return {"schema_version": COMMUNITY_SUMMARY_SCHEMA_VERSION, "domains": {}}
    if not isinstance(payload.get("domains"), dict):
        payload["domains"] = {}
    return payload


def write_community_summaries(payload: dict[str, Any], path: Path = COMMUNITY_SUMMARIES_PATH) -> None:
    payload["schema_version"] = COMMUNITY_SUMMARY_SCHEMA_VERSION
    payload["updated_at"] = utc_now()
    write_json(path, payload)


def build_community_summaries(
    cards: list[dict[str, Any]],
    graph: dict[str, Any],
    *,
    path: Path = COMMUNITY_SUMMARIES_PATH,
    force: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    client = None
    model = get_model("map")
    if not dry_run:
        client, _default_model, _provider = create_chat_client()
        model = get_model("map")
    payload = read_community_summaries(path)
    summaries: dict[str, Any] = payload.get("domains", {})
    valid_ids = set()
    for domain in graph.get("domains", []) or []:
        domain_id = str(domain.get("domain_id") or domain.get("domain_name", ""))
        if not domain_id:
            continue
        valid_ids.add(domain_id)
        context = domain_context(domain, graph, cards)
        signature = graph_signature(domain, context["clusters"], context["cards"], context["relationships"])
        existing = summaries.get(domain_id)
        if not force and isinstance(existing, dict) and existing.get("signature") == signature:
            continue
        summaries[domain_id] = summarize_domain(
            domain,
            graph,
            cards,
            client=client,
            model=model,
            dry_run=dry_run,
        )
        write_community_summaries(
            {
                **payload,
                "model": model if not dry_run else "heuristic",
                "domain_count": len(summaries),
                "domains": summaries,
            },
            path,
        )
    for domain_id in list(summaries):
        if domain_id not in valid_ids:
            summaries.pop(domain_id, None)
    result = {
        **payload,
        "model": model if not dry_run else "heuristic",
        "domain_count": len(summaries),
        "domains": summaries,
    }
    write_community_summaries(result, path)
    return result


def keyword_score(query: str, text: str) -> float:
    query_terms = {term for term in re.findall(r"[a-z0-9]+", query.lower()) if len(term) > 2}
    if not query_terms:
        return 0.0
    text_terms = set(re.findall(r"[a-z0-9]+", text.lower()))
    return len(query_terms & text_terms) / len(query_terms)


def community_summary_scores(query: str, graph: dict[str, Any], path: Path = COMMUNITY_SUMMARIES_PATH) -> dict[str, float]:
    payload = read_community_summaries(path)
    summaries = payload.get("domains", {})
    if not isinstance(summaries, dict) or not summaries:
        return {}
    scores: dict[str, float] = {}
    for domain in graph.get("domains", []) or []:
        domain_id = str(domain.get("domain_id") or domain.get("domain_name", ""))
        summary = summaries.get(domain_id)
        if not isinstance(summary, dict):
            continue
        text = " ".join(
            [
                str(summary.get("domain_name", "")),
                str(summary.get("summary", "")),
                " ".join(str(item) for item in summary.get("key_points", []) or []),
                " ".join(str(item) for item in summary.get("requirements", []) or []),
                " ".join(str(item) for item in summary.get("metrics", []) or []),
                " ".join(str(item) for item in summary.get("risks", []) or []),
                " ".join(str(item) for item in summary.get("related_context", []) or []),
            ]
        )
        score = keyword_score(query, text)
        if score > 0:
            scores[domain_id] = score
            if summary.get("domain_name"):
                scores[str(summary.get("domain_name"))] = score
    return scores


def top_community_summaries(query: str, limit: int = 6, path: Path = COMMUNITY_SUMMARIES_PATH) -> list[dict[str, Any]]:
    payload = read_community_summaries(path)
    summaries = payload.get("domains", {})
    if not isinstance(summaries, dict):
        return []
    scored = []
    for summary in summaries.values():
        if not isinstance(summary, dict):
            continue
        text = " ".join(
            [
                str(summary.get("domain_name", "")),
                str(summary.get("summary", "")),
                " ".join(str(item) for item in summary.get("key_points", []) or []),
                " ".join(str(item) for item in summary.get("requirements", []) or []),
                " ".join(str(item) for item in summary.get("metrics", []) or []),
                " ".join(str(item) for item in summary.get("risks", []) or []),
            ]
        )
        score = keyword_score(query, text)
        if score > 0:
            item = dict(summary)
            item["score"] = round(score, 3)
            scored.append(item)
    scored.sort(key=lambda item: item["score"], reverse=True)
    return scored[:limit]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build precomputed community summaries for GraphRAG retrieval.")
    subparsers = parser.add_subparsers(dest="command")
    build = subparsers.add_parser("build", help="Build missing or changed domain/community summaries.")
    build.add_argument("--force", action="store_true", help="Rebuild all summaries even if signatures match.")
    build.add_argument("--dry-run", action="store_true", help="Use deterministic heuristic summaries instead of LLM summarization.")
    build.add_argument("--project-id", help="Storage project id. Defaults to EVIDENCE_MESH_PROJECT_ID.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command != "build":
        raise SystemExit("Run: python community_summaries.py build")
    cards = read_cards(args.project_id)
    if not cards:
        raise SystemExit("No cards found. Run indexer.py first.")
    graph = read_knowledge_graph(KNOWLEDGE_GRAPH_PATH, cards, persist_migration=True)
    if not graph.get("domains"):
        raise SystemExit("No graph domains found. Run knowledge_graph.py first.")
    result = build_community_summaries(cards, graph, force=bool(args.force), dry_run=bool(args.dry_run))
    print(f"Built community summaries for {result.get('domain_count', 0)} domain(s): {COMMUNITY_SUMMARIES_PATH}")


if __name__ == "__main__":
    main()
