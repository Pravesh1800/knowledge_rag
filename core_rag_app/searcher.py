from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from openai import OpenAI

from community_summaries import community_summary_scores
from embeddings import vector_scores
from entity_canonicalizer import canonical_card_scores, canonical_entities_for_card
from entity_claims import anchors_for_card, entity_claim_card_scores
from llm_config import create_chat_client, get_model
from reranker import candidate_limit, rerank_hits
from schema import SEARCH_RESULT_SCHEMA_VERSION, card_id_from_record, read_knowledge_graph
from storage import read_cards, record_search_run


LEGACY_ROOT_ENV = "PDF" + "_VISION_RAG_ROOT"
PROJECT_ROOT = Path(
    os.getenv("EVIDENCE_MESH_ROOT")
    or os.getenv(LEGACY_ROOT_ENV)
    or Path(__file__).resolve().parent
).resolve()
INDEXES_DIR = PROJECT_ROOT / "indexes"
CARD_INDEX_PATH = INDEXES_DIR / "card_index.json"
KNOWLEDGE_GRAPH_PATH = INDEXES_DIR / "knowledge_graph.json"
SEARCH_RESULTS_DIR = INDEXES_DIR / "search_results"

DEFAULT_SEARCH_MAX_TOKENS = 1024
RELATIONSHIP_TYPE_MODE_BOOSTS = {
    "contradiction_check": {"contradiction": 0.36, "comparison": 0.18, "exception": 0.14},
    "comparison": {"comparison": 0.30, "shared_context": 0.18, "contradiction": 0.16},
    "gap_analysis": {"prerequisite": 0.18, "dependency": 0.18, "continuation": 0.12, "shared_context": 0.10},
    "risk_analysis": {"cause_effect": 0.24, "dependency": 0.18, "prerequisite": 0.16, "contradiction": 0.14},
    "multi_hop": {"dependency": 0.22, "prerequisite": 0.20, "cause_effect": 0.20, "evidence": 0.16},
    "global_summary": {"shared_context": 0.18, "continuation": 0.16, "evidence": 0.12},
    "general": {"evidence": 0.12, "shared_context": 0.10, "continuation": 0.08},
}


@dataclass
class SearchHit:
    card_id: str
    card_name: str
    document_name: str
    page_no: int
    relevance_reason: str
    content: str
    card_source: str
    tags: list[str]
    related_cards: list[dict[str, Any]]
    typed_anchors: dict[str, Any]


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
            if os.getenv(key) is not None:
                continue
            os.environ[key] = value.strip().strip("\"'")


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def slugify(value: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip()).strip("_").lower()
    return value or "search"


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


def create_client() -> tuple[OpenAI, str]:
    try:
        client, _model, _provider = create_chat_client()
    except RuntimeError as exc:
        raise SystemExit(f"{exc}, or run with --dry-run.") from exc
    return client, get_model("search")


def keyword_score(query: str, text: str) -> float:
    query_terms = {term for term in re.findall(r"[a-z0-9]+", query.lower()) if len(term) > 2}
    if not query_terms:
        return 0
    text_terms = set(re.findall(r"[a-z0-9]+", text.lower()))
    return len(query_terms & text_terms) / len(query_terms)


def relationship_quality_score(relationship: dict[str, Any]) -> float:
    try:
        confidence = float(relationship.get("confidence_score", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    try:
        evidence_strength = float(relationship.get("evidence_strength", 0.0))
    except (TypeError, ValueError):
        evidence_strength = 0.0
    try:
        source_coverage = float(relationship.get("source_coverage", 0.0))
    except (TypeError, ValueError):
        source_coverage = 0.0
    return max(0.0, min(1.0, 0.5 * confidence + 0.3 * evidence_strength + 0.2 * source_coverage))


def dry_rank(query: str, candidates: list[dict[str, Any]], text_keys: list[str]) -> list[dict[str, Any]]:
    ranked = []
    for candidate in candidates:
        text = " ".join(str(candidate.get(key, "")) for key in text_keys)
        score = keyword_score(query, text)
        if "relationship_quality_score" in candidate:
            score = (0.72 * score) + (0.28 * float(candidate.get("relationship_quality_score") or 0.0))
        if score > 0:
            ranked.append(
                {
                    "name": candidate.get("name") or candidate.get("domain_name") or candidate.get("cluster_name") or candidate.get("card_name"),
                    "score": score,
                    "reason": "Keyword overlap with query.",
                }
            )
    return sorted(ranked, key=lambda item: item["score"], reverse=True)


def rank_prompt(query: str, level: str, candidates: list[dict[str, Any]]) -> str:
    return f"""
Rank which {level} nodes are relevant to the search query.

Query:
{query}

Candidates:
{json.dumps(candidates, ensure_ascii=False)}

Rules:
1. Return only candidates that are useful for answering the query.
2. Prefer branches that are likely to contain deep evidence, not just keyword overlap.
3. A broad query may require multiple branches.
4. Keep the returned names exactly as provided.

Return only valid JSON:
{{
  "ranked": [
    {{
      "name": "Exact candidate name",
      "score": 0.0,
      "reason": "Why this branch should be searched"
    }}
  ]
}}
""".strip()


def llm_rank(
    client: OpenAI,
    model: str,
    query: str,
    level: str,
    candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You guide recursive tree search over a document knowledge graph. Return only valid JSON.",
            },
            {"role": "user", "content": rank_prompt(query, level, candidates)},
        ],
        temperature=0,
        response_format={"type": "json_object"},
        max_tokens=int(os.getenv("OPENROUTER_SEARCH_MAX_TOKENS", str(DEFAULT_SEARCH_MAX_TOKENS))),
    )
    result = parse_json_response(response.choices[0].message.content or "{}")
    return result.get("ranked", [])


class TreeSearcher:
    def __init__(
        self,
        query: str,
        dry_run: bool,
        max_hits: int,
        storage_project_id: str | None = None,
        knowledge_graph_path: Path | None = None,
        query_mode: dict[str, Any] | None = None,
    ) -> None:
        self.query = query
        self.dry_run = dry_run
        self.max_hits = max_hits
        self.query_mode = query_mode or {"mode": "general"}
        mode_multiplier = float(self.query_mode.get("max_hits_multiplier", 1.0) or 1.0)
        self.candidate_target = max(max_hits, int(candidate_limit() * max(0.75, mode_multiplier)))
        self.use_community_summaries = bool(self.query_mode.get("use_community_summaries", True))
        self.relationship_expansion = bool(self.query_mode.get("relationship_expansion", True))
        self.storage_project_id = storage_project_id
        self.knowledge_graph_path = knowledge_graph_path or KNOWLEDGE_GRAPH_PATH
        self.cards = read_cards(storage_project_id)
        self.map = read_knowledge_graph(self.knowledge_graph_path, self.cards, persist_migration=True)
        if not self.cards:
            raise SystemExit("No cards found in PostgreSQL. Run indexer.py first.")
        if not self.map:
            raise SystemExit("No knowledge graph found. Run knowledge_graph.py first.")

        self.card_lookup = {card["card_name"]: card for card in self.cards}
        self.card_lookup_by_id = {card_id_from_record(card): card for card in self.cards}
        self.semantic_scores = {} if dry_run else vector_scores(query, self.cards)
        self.entity_claim_scores = entity_claim_card_scores(query, self.cards)
        self.canonical_entity_scores = canonical_card_scores(query)
        self.community_scores = community_summary_scores(query, self.map) if self.use_community_summaries else {}
        self.cluster_lookup = {
            cluster["cluster_name"]: cluster
            for cluster in self.map.get("clusters", [])
        }
        self.domain_lookup = {
            domain["domain_name"]: domain for domain in self.map.get("domains", [])
        }
        self.relationships_by_domain: dict[str, list[dict[str, Any]]] = {}
        self.card_relationship_links: dict[str, list[dict[str, Any]]] = {}
        self.relationship_domain_scores: dict[str, float] = {}
        self.relationship_card_scores: dict[str, float] = {}
        for relationship in self.map.get("domain_relationships", []):
            self.relationships_by_domain.setdefault(relationship["main_domain"], []).append(relationship)
            relationship_score = self.relationship_relevance_score(relationship)
            for domain_key in (
                relationship.get("main_domain", ""),
                relationship.get("related_domain", ""),
                relationship.get("main_domain_id", ""),
                relationship.get("related_domain_id", ""),
            ):
                if domain_key:
                    self.relationship_domain_scores[str(domain_key)] = max(
                        self.relationship_domain_scores.get(str(domain_key), 0.0),
                        relationship_score,
                    )
            for link in relationship.get("card_links", []):
                main_card = link.get("main_card", "")
                related_card = link.get("related_card", "")
                if main_card and related_card:
                    card_link_score = max(relationship_score, relationship_quality_score(relationship) * 0.5)
                    for card_key in (main_card, related_card, link.get("main_card_id", ""), link.get("related_card_id", "")):
                        if card_key:
                            self.relationship_card_scores[str(card_key)] = max(
                                self.relationship_card_scores.get(str(card_key), 0.0),
                                card_link_score,
                            )
                    self.card_relationship_links.setdefault(main_card, []).append(
                        {
                            "card_name": related_card,
                            "card_id": link.get("related_card_id", ""),
                            "direction": "outgoing",
                            "relationship": link.get("relationship", ""),
                            "relationship_type": relationship.get("relationship_type", ""),
                            "relationship_description": relationship.get("relationship_description", ""),
                            "confidence_score": relationship.get("confidence_score", 0.0),
                            "evidence_strength": relationship.get("evidence_strength", 0.0),
                            "source_coverage": relationship.get("source_coverage", 0.0),
                            "document_scope": relationship.get("document_scope", ""),
                            "generation_method": relationship.get("generation_method", ""),
                            "relationship_quality_score": relationship_quality_score(relationship),
                            "source_domain": relationship.get("main_domain", ""),
                            "source_domain_id": relationship.get("main_domain_id", ""),
                            "related_domain": relationship.get("related_domain", ""),
                            "related_domain_id": relationship.get("related_domain_id", ""),
                        }
                    )
                    self.card_relationship_links.setdefault(related_card, []).append(
                        {
                            "card_name": main_card,
                            "card_id": link.get("main_card_id", ""),
                            "direction": "incoming",
                            "relationship": link.get("relationship", ""),
                            "relationship_type": relationship.get("relationship_type", ""),
                            "relationship_description": relationship.get("relationship_description", ""),
                            "confidence_score": relationship.get("confidence_score", 0.0),
                            "evidence_strength": relationship.get("evidence_strength", 0.0),
                            "source_coverage": relationship.get("source_coverage", 0.0),
                            "document_scope": relationship.get("document_scope", ""),
                            "generation_method": relationship.get("generation_method", ""),
                            "relationship_quality_score": relationship_quality_score(relationship),
                            "source_domain": relationship.get("related_domain", ""),
                            "source_domain_id": relationship.get("related_domain_id", ""),
                            "related_domain": relationship.get("main_domain", ""),
                            "related_domain_id": relationship.get("main_domain_id", ""),
                        }
                    )

        self.visited: set[str] = set()
        self.trace: list[dict[str, Any]] = []
        self.hits: list[SearchHit] = []
        self.client: OpenAI | None = None
        self.model = get_model("search")
        try:
            self.llm_rank_budget = int(os.getenv("EVIDENCE_MESH_LLM_RANK_BUDGET", "1"))
        except ValueError:
            self.llm_rank_budget = -1
        self.llm_rank_calls = 0
        if not dry_run:
            self.client, self.model = create_client()

    def relationship_relevance_score(self, relationship: dict[str, Any]) -> float:
        quality = relationship_quality_score(relationship)
        text = " ".join(
            str(relationship.get(key, ""))
            for key in (
                "main_domain",
                "related_domain",
                "relationship_type",
                "relationship_description",
                "evidence",
                "document_scope",
            )
        )
        lexical = keyword_score(self.query, text)
        mode = str(self.query_mode.get("mode", "general"))
        relationship_type = str(relationship.get("relationship_type", "other")).strip().lower()
        type_boost = RELATIONSHIP_TYPE_MODE_BOOSTS.get(mode, RELATIONSHIP_TYPE_MODE_BOOSTS["general"]).get(
            relationship_type,
            0.0,
        )
        return max(0.0, min(1.0, (0.46 * quality) + (0.38 * lexical) + type_boost))

    def rank(self, level: str, candidates: list[dict[str, Any]], text_keys: list[str]) -> list[dict[str, Any]]:
        if not candidates:
            return []
        if self.dry_run or self.client is None:
            return dry_rank(self.query, candidates, text_keys)
        if self.llm_rank_budget >= 0 and self.llm_rank_calls >= self.llm_rank_budget:
            return dry_rank(self.query, candidates, text_keys)
        try:
            self.llm_rank_calls += 1
            return llm_rank(self.client, self.model, self.query, level, candidates)
        except Exception as exc:
            print(f"Warning: LLM ranking failed at {level}; falling back to keyword ranking. {exc}")
            return dry_rank(self.query, candidates, text_keys)

    def apply_community_boost(self, ranked: list[dict[str, Any]]) -> list[dict[str, Any]]:
        boosted = []
        for item in ranked:
            domain_name = str(item.get("name", ""))
            community_boost = self.community_scores.get(domain_name, 0.0)
            relationship_boost = self.relationship_domain_scores.get(domain_name, 0.0)
            total_boost = (0.35 * community_boost) + (0.28 * relationship_boost)
            if not total_boost:
                boosted.append(item)
                continue
            copy = dict(item)
            try:
                base = float(copy.get("score", 0.0))
            except (TypeError, ValueError):
                base = 0.0
            copy["score"] = base + total_boost
            if community_boost:
                copy["reason"] = f"{copy.get('reason', '')} Community summary match {community_boost:.2f}.".strip()
            if relationship_boost:
                copy["reason"] = f"{copy.get('reason', '')} Relationship signal {relationship_boost:.2f}.".strip()
            boosted.append(copy)
        return sorted(boosted, key=lambda entry: float(entry.get("score", 0.0)), reverse=True)

    def hybrid_card_score(self, card: dict[str, Any], graph_score: float = 0.0) -> float:
        text = " ".join(
            [
                str(card.get("card_name", "")),
                str(card.get("card_description", "")),
                str(card.get("content", "")),
                str(card.get("document_name", "")),
                " ".join(str(tag) for tag in card.get("tags", []) or []),
            ]
        )
        lexical = keyword_score(self.query, text)
        card_id = card_id_from_record(card)
        semantic = self.semantic_scores.get(card_id, 0.0)
        entity_claim = self.entity_claim_scores.get(card_id, 0.0)
        canonical_entity = self.canonical_entity_scores.get(card_id, 0.0)
        relationship = max(
            self.relationship_card_scores.get(card_id, 0.0),
            self.relationship_card_scores.get(str(card.get("card_name", "")), 0.0),
        )
        mode = str(self.query_mode.get("mode", "general"))
        if mode == "exact_lookup":
            return (
                (0.30 * semantic)
                + (0.30 * lexical)
                + (0.22 * entity_claim)
                + (0.08 * canonical_entity)
                + (0.05 * relationship)
                + (0.05 * graph_score)
            )
        if mode in {"global_summary", "comparison", "contradiction_check", "gap_analysis", "risk_analysis"}:
            return (
                (0.30 * semantic)
                + (0.16 * lexical)
                + (0.20 * entity_claim)
                + (0.10 * canonical_entity)
                + (0.16 * relationship)
                + (0.08 * graph_score)
            )
        return (
            (0.33 * semantic)
            + (0.20 * lexical)
            + (0.18 * entity_claim)
            + (0.09 * canonical_entity)
            + (0.11 * relationship)
            + (0.09 * graph_score)
        )

    def ranked_hybrid_cards(self, limit: int | None = None) -> list[dict[str, Any]]:
        candidates = []
        for card in self.cards:
            score = self.hybrid_card_score(card)
            if score <= 0:
                continue
            candidates.append(
                {
                    "card": card,
                    "score": score,
                    "semantic_score": self.semantic_scores.get(card_id_from_record(card), 0.0),
                    "entity_claim_score": self.entity_claim_scores.get(card_id_from_record(card), 0.0),
                    "canonical_entity_score": self.canonical_entity_scores.get(card_id_from_record(card), 0.0),
                    "relationship_score": max(
                        self.relationship_card_scores.get(card_id_from_record(card), 0.0),
                        self.relationship_card_scores.get(str(card.get("card_name", "")), 0.0),
                    ),
                    "keyword_score": keyword_score(
                        self.query,
                        " ".join(
                            [
                                str(card.get("card_name", "")),
                                str(card.get("card_description", "")),
                                str(card.get("content", "")),
                                str(card.get("document_name", "")),
                                " ".join(str(tag) for tag in card.get("tags", []) or []),
                            ]
                        ),
                    ),
                }
            )
        candidates.sort(key=lambda item: item["score"], reverse=True)
        return candidates[:limit] if limit is not None else candidates

    def add_hybrid_card_hits(self, limit: int) -> None:
        if limit <= 0:
            return
        added = 0
        for item in self.ranked_hybrid_cards(limit=max(limit * 3, limit)):
            if added >= limit or len(self.hits) >= self.candidate_target:
                return
            card = item["card"]
            card_name = str(card.get("card_name", ""))
            if not card_name or f"card:{card_name}" in self.visited:
                continue
            semantic = float(item.get("semantic_score") or 0.0)
            entity_claim = float(item.get("entity_claim_score") or 0.0)
            canonical_entity = float(item.get("canonical_entity_score") or 0.0)
            relationship = float(item.get("relationship_score") or 0.0)
            lexical = float(item.get("keyword_score") or 0.0)
            reason = (
                f"Hybrid retrieval score {item['score']:.2f} "
                f"(semantic {semantic:.2f}, entity/claim {entity_claim:.2f}, canonical entity {canonical_entity:.2f}, relationship {relationship:.2f}, keyword {lexical:.2f})."
            )
            self.mark_visit("hybrid_card", card_name, reason)
            self.visit_card(card_name, reason)
            added += 1

    def actively_expand_relationship_cards(self, limit: int) -> None:
        if limit <= 0 or not self.relationship_expansion:
            return
        seed_names = [hit.card_name for hit in self.hits]
        candidates: list[dict[str, Any]] = []
        for seed_name in seed_names:
            for link in self.card_relationship_links.get(seed_name, []):
                related_name = str(link.get("card_name", ""))
                if not related_name or f"card:{related_name}" in self.visited:
                    continue
                related_card = self.card_lookup.get(related_name)
                if not related_card:
                    continue
                link_quality = float(link.get("relationship_quality_score") or 0.0)
                relationship_type = str(link.get("relationship_type", "")).strip().lower()
                mode = str(self.query_mode.get("mode", "general"))
                type_boost = RELATIONSHIP_TYPE_MODE_BOOSTS.get(mode, RELATIONSHIP_TYPE_MODE_BOOSTS["general"]).get(
                    relationship_type,
                    0.0,
                )
                relation_text = " ".join(
                    str(link.get(key, ""))
                    for key in (
                        "relationship",
                        "relationship_type",
                        "relationship_description",
                        "document_scope",
                        "source_domain",
                        "related_domain",
                    )
                )
                score = (
                    0.44 * self.hybrid_card_score(related_card)
                    + 0.34 * link_quality
                    + 0.16 * keyword_score(self.query, relation_text)
                    + type_boost
                )
                candidates.append(
                    {
                        "card": related_card,
                        "seed_name": seed_name,
                        "link": link,
                        "score": score,
                    }
                )
        candidates.sort(key=lambda item: item["score"], reverse=True)
        added = 0
        for item in candidates:
            if added >= limit or len(self.hits) >= self.candidate_target:
                return
            card = item["card"]
            card_name = str(card.get("card_name", ""))
            if not card_name or f"card:{card_name}" in self.visited:
                continue
            link = item["link"]
            reason = (
                f"Relationship expansion from {item['seed_name']}: "
                f"{link.get('relationship_type', 'related')} "
                f"quality {float(link.get('relationship_quality_score') or 0.0):.2f}; "
                f"{link.get('relationship_description', '')}"
            )
            self.mark_visit("relationship_expansion", card_name, reason)
            self.visit_card(card_name, reason)
            added += 1

    def search(self) -> dict[str, Any]:
        mode = str(self.query_mode.get("mode", "general"))
        first_pass = max(2, min(10, self.candidate_target // 3))
        if mode == "exact_lookup":
            first_pass = max(4, min(16, self.candidate_target // 2))
        elif mode in {"global_summary", "comparison", "contradiction_check", "gap_analysis"}:
            first_pass = max(2, min(8, self.candidate_target // 4))
        self.add_hybrid_card_hits(first_pass)
        if self.relationship_expansion and len(self.hits) < self.candidate_target:
            self.actively_expand_relationship_cards(max(2, min(10, self.candidate_target - len(self.hits))))
        domain_candidates = [
            {
                "name": domain["domain_name"],
                "domain_id": domain.get("domain_id", ""),
                "domain_name": domain["domain_name"],
                "domain_description": domain.get("domain_description", ""),
                "document_name": domain.get("document_name", ""),
                "cluster_names": domain.get("cluster_names", []),
            }
            for domain in self.map.get("domains", [])
        ]
        ranked_domains = self.apply_community_boost(
            self.rank("domain", domain_candidates, ["domain_name", "domain_description", "document_name"])
        )
        for item in ranked_domains:
            self.visit_domain(item["name"], item.get("reason", ""))
            if len(self.hits) >= self.candidate_target:
                break

        if self.relationship_expansion and len(self.hits) < self.candidate_target:
            self.actively_expand_relationship_cards(max(2, min(12, self.candidate_target - len(self.hits))))

        if len(self.hits) < self.candidate_target:
            self.add_hybrid_card_hits(self.candidate_target - len(self.hits))

        reranked_hits = rerank_hits(
            self.query,
            [hit.__dict__ for hit in self.hits[: self.candidate_target]],
            self.max_hits,
            client=self.client,
            model=self.model,
            dry_run=self.dry_run,
        )

        return {
            "schema_version": SEARCH_RESULT_SCHEMA_VERSION,
            "query": self.query,
            "query_mode": self.query_mode,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "trace": self.trace,
            "reranked": True,
            "candidate_count": len(self.hits[: self.candidate_target]),
            "hits": reranked_hits,
        }

    def mark_visit(self, node_type: str, name: str, reason: str) -> bool:
        key = f"{node_type}:{name}"
        if key in self.visited:
            return False
        self.visited.add(key)
        self.trace.append({"node_type": node_type, "name": name, "reason": reason})
        return True

    def visit_domain(self, domain_name: str, reason: str) -> None:
        if not self.mark_visit("domain", domain_name, reason):
            return
        domain = self.domain_lookup.get(domain_name)
        if not domain:
            return

        cluster_candidates = []
        for cluster_name in domain.get("cluster_names", []):
            cluster = self.cluster_lookup.get(cluster_name)
            if cluster:
                cluster_candidates.append(
                    {
                        "name": cluster_name,
                        "cluster_id": cluster.get("cluster_id", ""),
                        "cluster_name": cluster_name,
                        "cluster_description": cluster.get("cluster_description", ""),
                        "card_names": cluster.get("card_names", []),
                    }
                )
        ranked_clusters = self.rank(
            "cluster",
            cluster_candidates,
            ["cluster_name", "cluster_description"],
        )
        found_before = len(self.hits)
        for item in ranked_clusters:
            self.visit_cluster(item["name"], item.get("reason", ""))
            if len(self.hits) >= self.candidate_target:
                return

        mode = str(self.query_mode.get("mode", "general"))
        active_modes = {"multi_hop", "comparison", "contradiction_check", "gap_analysis", "risk_analysis", "global_summary"}
        if self.relationship_expansion and (len(self.hits) == found_before or mode in active_modes):
            self.follow_related_domains(domain_name)

    def follow_related_domains(self, domain_name: str) -> None:
        relationships = self.relationships_by_domain.get(domain_name, [])
        relationship_candidates = [
            {
                "name": relationship["related_domain"],
                "related_domain": relationship["related_domain"],
                "relationship_type": relationship.get("relationship_type", ""),
                "relationship_description": relationship.get("relationship_description", ""),
                "evidence": relationship.get("evidence", ""),
                "confidence_score": relationship.get("confidence_score", 0.0),
                "evidence_strength": relationship.get("evidence_strength", 0.0),
                "source_coverage": relationship.get("source_coverage", 0.0),
                "document_scope": relationship.get("document_scope", ""),
                "generation_method": relationship.get("generation_method", ""),
                "relationship_quality_score": relationship_quality_score(relationship),
            }
            for relationship in relationships
            if f"domain:{relationship['related_domain']}" not in self.visited
        ]
        relationship_candidates = sorted(
            relationship_candidates,
            key=lambda item: float(item.get("relationship_quality_score") or 0.0),
            reverse=True,
        )
        ranked = self.rank(
            "related domain",
            relationship_candidates,
            [
                "related_domain",
                "relationship_type",
                "relationship_description",
                "evidence",
                "document_scope",
                "generation_method",
            ],
        )
        for item in ranked:
            self.visit_domain(item["name"], f"Related domain: {item.get('reason', '')}")
            if len(self.hits) >= self.candidate_target:
                return

    def visit_cluster(self, cluster_name: str, reason: str) -> None:
        if not self.mark_visit("cluster", cluster_name, reason):
            return
        cluster = self.cluster_lookup.get(cluster_name)
        if not cluster:
            return
        card_candidates = []
        for card_name in cluster.get("card_names", []):
            card = self.card_lookup.get(card_name)
            if card:
                card_candidates.append(
                    {
                        "name": card_name,
                        "card_id": card.get("card_id", ""),
                        "card_name": card_name,
                        "card_description": card.get("card_description", ""),
                        "content": str(card.get("content", ""))[:1600],
                        "tags": card.get("tags", []),
                        "card_source": card.get("card_source", ""),
                        "page_no": card.get("page_no"),
                        "document_name": card.get("document_name", ""),
                    }
                )
        ranked_cards = self.rank(
            "card",
            card_candidates,
            ["card_name", "card_description", "content", "tags", "document_name"],
        )
        for item in ranked_cards:
            self.visit_card(item["name"], item.get("reason", ""))
            if len(self.hits) >= self.candidate_target:
                return

    def visit_card(self, card_name: str, reason: str) -> None:
        if not self.mark_visit("card", card_name, reason):
            return
        card = self.card_lookup.get(card_name)
        if not card:
            return
        related_cards = self.find_helpful_related_cards(card_name)
        card_id = card_id_from_record(card)
        anchors = anchors_for_card(card_id)
        self.hits.append(
            SearchHit(
                card_id=str(card.get("card_id", card_id)),
                card_name=card_name,
                document_name=card.get("document_name", ""),
                page_no=int(card.get("page_no") or 0),
                relevance_reason=reason,
                content=str(card.get("content", "")),
                card_source=card.get("card_source", ""),
                tags=card.get("tags", []),
                related_cards=related_cards,
                typed_anchors={
                    "entities": (anchors.get("entities") or [])[:8],
                    "claims": (anchors.get("claims") or [])[:6],
                    "canonical_entities": canonical_entities_for_card(card_id)[:8],
                },
            )
        )

    def find_helpful_related_cards(self, card_name: str) -> list[dict[str, Any]]:
        relationship_links = self.card_relationship_links.get(card_name, [])
        candidates = []
        for link in relationship_links:
            related_name = link.get("card_name", "")
            if not related_name or f"relationship-card:{card_name}->{related_name}" in self.visited:
                continue
            related_card = self.card_lookup.get(related_name)
            if not related_card:
                continue
            candidates.append(
                {
                    "name": related_name,
                    "card_id": related_card.get("card_id", link.get("card_id", "")),
                    "card_name": related_name,
                    "card_description": related_card.get("card_description", ""),
                    "content": str(related_card.get("content", ""))[:1400],
                    "document_name": related_card.get("document_name", ""),
                    "page_no": related_card.get("page_no"),
                    "relationship": link.get("relationship", ""),
                    "relationship_type": link.get("relationship_type", ""),
                    "relationship_description": link.get("relationship_description", ""),
                    "confidence_score": link.get("confidence_score", 0.0),
                    "evidence_strength": link.get("evidence_strength", 0.0),
                    "source_coverage": link.get("source_coverage", 0.0),
                    "document_scope": link.get("document_scope", ""),
                    "generation_method": link.get("generation_method", ""),
                    "relationship_quality_score": link.get("relationship_quality_score", 0.0),
                    "source_domain_id": link.get("source_domain_id", ""),
                    "related_domain_id": link.get("related_domain_id", ""),
                    "direction": link.get("direction", ""),
                }
            )

        ranked = self.rank(
            "related card",
            candidates,
            [
                "card_name",
                "card_description",
                "content",
                "relationship",
                "relationship_type",
                "relationship_description",
                "document_scope",
                "generation_method",
                "document_name",
            ],
        )
        related_results: list[dict[str, Any]] = []
        ranked_names = [item["name"] for item in ranked[:5]]
        for related_name in ranked_names:
            self.visited.add(f"relationship-card:{card_name}->{related_name}")
            related_card = self.card_lookup.get(related_name)
            link = next(
                (item for item in relationship_links if item.get("card_name") == related_name),
                None,
            )
            if not related_card or not link:
                continue
            related_results.append(
                {
                    "card_name": related_name,
                    "card_id": related_card.get("card_id", link.get("card_id", "")),
                    "document_name": related_card.get("document_name", ""),
                    "page_no": int(related_card.get("page_no") or 0),
                    "relationship_type": link.get("relationship_type", ""),
                    "relationship": link.get("relationship", ""),
                    "relationship_description": link.get("relationship_description", ""),
                    "confidence_score": link.get("confidence_score", 0.0),
                    "evidence_strength": link.get("evidence_strength", 0.0),
                    "source_coverage": link.get("source_coverage", 0.0),
                    "document_scope": link.get("document_scope", ""),
                    "generation_method": link.get("generation_method", ""),
                    "relationship_quality_score": link.get("relationship_quality_score", 0.0),
                    "source_domain_id": link.get("source_domain_id", ""),
                    "related_domain_id": link.get("related_domain_id", ""),
                    "direction": link.get("direction", ""),
                    "content": str(related_card.get("content", "")),
                    "card_source": related_card.get("card_source", ""),
                    "tags": related_card.get("tags", []),
                }
            )
        return related_results


def run_search(
    query: str,
    dry_run: bool,
    max_hits: int,
    storage_project_id: str | None = None,
    knowledge_graph_path: Path | None = None,
    query_mode: dict[str, Any] | None = None,
) -> dict[str, Any]:
    load_dotenv()
    searcher = TreeSearcher(
        query=query,
        dry_run=dry_run,
        max_hits=max_hits,
        storage_project_id=storage_project_id,
        knowledge_graph_path=knowledge_graph_path,
        query_mode=query_mode,
    )
    result = searcher.search()
    result["search_run_id"] = f"search_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S%f')}"
    output_path = SEARCH_RESULTS_DIR / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{slugify(query)[:60]}.json"
    write_json(output_path, result)
    record_search_run(result, storage_project_id)
    print(f"Wrote search result to {output_path}")
    for hit in result["hits"]:
        print(f"- {hit['card_name']} ({hit['document_name']}, page {hit['page_no']})")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Recursive tree search over domains, clusters, and cards.")
    parser.add_argument("query", help="Broad search query.")
    parser.add_argument("--dry-run", action="store_true", help="Use keyword ranking instead of OpenRouter.")
    parser.add_argument("--max-hits", type=int, default=12, help="Maximum card hits to return.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_search(query=args.query, dry_run=args.dry_run, max_hits=args.max_hits)


if __name__ == "__main__":
    main()


