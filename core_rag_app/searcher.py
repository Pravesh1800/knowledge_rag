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

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "deepseek/deepseek-v4-flash"
DEFAULT_SEARCH_MAX_TOKENS = 1024


@dataclass
class SearchHit:
    card_name: str
    document_name: str
    page_no: int
    relevance_reason: str
    content: str
    card_source: str
    tags: list[str]
    related_cards: list[dict[str, Any]]


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
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit("Set OPENROUTER_API_KEY in .env, or run with --dry-run.")
    client = OpenAI(
        base_url=os.getenv("OPENROUTER_BASE_URL", OPENROUTER_BASE_URL),
        api_key=api_key,
        timeout=float(os.getenv("OPENROUTER_SEARCH_TIMEOUT_SECONDS", os.getenv("OPENROUTER_TIMEOUT_SECONDS", "120"))),
        max_retries=int(os.getenv("OPENROUTER_MAX_RETRIES", "1")),
        default_headers={
            "HTTP-Referer": os.getenv("OPENROUTER_SITE_URL", "http://localhost"),
            "X-Title": os.getenv("OPENROUTER_APP_NAME", "Evidence Mesh"),
        },
    )
    return client, os.getenv("OPENROUTER_SEARCH_MODEL", DEFAULT_MODEL)


def keyword_score(query: str, text: str) -> float:
    query_terms = {term for term in re.findall(r"[a-z0-9]+", query.lower()) if len(term) > 2}
    if not query_terms:
        return 0
    text_terms = set(re.findall(r"[a-z0-9]+", text.lower()))
    return len(query_terms & text_terms) / len(query_terms)


def dry_rank(query: str, candidates: list[dict[str, Any]], text_keys: list[str]) -> list[dict[str, Any]]:
    ranked = []
    for candidate in candidates:
        text = " ".join(str(candidate.get(key, "")) for key in text_keys)
        score = keyword_score(query, text)
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
    def __init__(self, query: str, dry_run: bool, max_hits: int) -> None:
        self.query = query
        self.dry_run = dry_run
        self.max_hits = max_hits
        self.cards = read_json(CARD_INDEX_PATH, [])
        self.map = read_json(KNOWLEDGE_GRAPH_PATH, {})
        if not self.cards:
            raise SystemExit("No card index found. Run indexer.py first.")
        if not self.map:
            raise SystemExit("No knowledge graph found. Run knowledge_graph.py first.")

        self.card_lookup = {card["card_name"]: card for card in self.cards}
        self.cluster_lookup = {
            cluster["cluster_name"]: cluster
            for cluster in self.map.get("clusters", [])
        }
        self.domain_lookup = {
            domain["domain_name"]: domain for domain in self.map.get("domains", [])
        }
        self.relationships_by_domain: dict[str, list[dict[str, Any]]] = {}
        self.card_relationship_links: dict[str, list[dict[str, Any]]] = {}
        for relationship in self.map.get("domain_relationships", []):
            self.relationships_by_domain.setdefault(relationship["main_domain"], []).append(relationship)
            for link in relationship.get("card_links", []):
                main_card = link.get("main_card", "")
                related_card = link.get("related_card", "")
                if main_card and related_card:
                    self.card_relationship_links.setdefault(main_card, []).append(
                        {
                            "card_name": related_card,
                            "direction": "outgoing",
                            "relationship": link.get("relationship", ""),
                            "relationship_type": relationship.get("relationship_type", ""),
                            "relationship_description": relationship.get("relationship_description", ""),
                            "source_domain": relationship.get("main_domain", ""),
                            "related_domain": relationship.get("related_domain", ""),
                        }
                    )
                    self.card_relationship_links.setdefault(related_card, []).append(
                        {
                            "card_name": main_card,
                            "direction": "incoming",
                            "relationship": link.get("relationship", ""),
                            "relationship_type": relationship.get("relationship_type", ""),
                            "relationship_description": relationship.get("relationship_description", ""),
                            "source_domain": relationship.get("related_domain", ""),
                            "related_domain": relationship.get("main_domain", ""),
                        }
                    )

        self.visited: set[str] = set()
        self.trace: list[dict[str, Any]] = []
        self.hits: list[SearchHit] = []
        self.client: OpenAI | None = None
        self.model = os.getenv("OPENROUTER_SEARCH_MODEL", DEFAULT_MODEL)
        if not dry_run:
            self.client, self.model = create_client()

    def rank(self, level: str, candidates: list[dict[str, Any]], text_keys: list[str]) -> list[dict[str, Any]]:
        if not candidates:
            return []
        if self.dry_run or self.client is None:
            return dry_rank(self.query, candidates, text_keys)
        try:
            return llm_rank(self.client, self.model, self.query, level, candidates)
        except Exception as exc:
            print(f"Warning: LLM ranking failed at {level}; falling back to keyword ranking. {exc}")
            return dry_rank(self.query, candidates, text_keys)

    def search(self) -> dict[str, Any]:
        domain_candidates = [
            {
                "name": domain["domain_name"],
                "domain_name": domain["domain_name"],
                "domain_description": domain.get("domain_description", ""),
                "document_name": domain.get("document_name", ""),
                "cluster_names": domain.get("cluster_names", []),
            }
            for domain in self.map.get("domains", [])
        ]
        ranked_domains = self.rank("domain", domain_candidates, ["domain_name", "domain_description", "document_name"])
        for item in ranked_domains:
            self.visit_domain(item["name"], item.get("reason", ""))
            if len(self.hits) >= self.max_hits:
                break

        return {
            "query": self.query,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "trace": self.trace,
            "hits": [hit.__dict__ for hit in self.hits[: self.max_hits]],
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
            if len(self.hits) >= self.max_hits:
                return

        if len(self.hits) == found_before:
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
            }
            for relationship in relationships
            if f"domain:{relationship['related_domain']}" not in self.visited
        ]
        ranked = self.rank(
            "related domain",
            relationship_candidates,
            ["related_domain", "relationship_type", "relationship_description", "evidence"],
        )
        for item in ranked:
            self.visit_domain(item["name"], f"Related domain: {item.get('reason', '')}")
            if len(self.hits) >= self.max_hits:
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
            if len(self.hits) >= self.max_hits:
                return

    def visit_card(self, card_name: str, reason: str) -> None:
        if not self.mark_visit("card", card_name, reason):
            return
        card = self.card_lookup.get(card_name)
        if not card:
            return
        related_cards = self.find_helpful_related_cards(card_name)
        self.hits.append(
            SearchHit(
                card_name=card_name,
                document_name=card.get("document_name", ""),
                page_no=int(card.get("page_no") or 0),
                relevance_reason=reason,
                content=str(card.get("content", "")),
                card_source=card.get("card_source", ""),
                tags=card.get("tags", []),
                related_cards=related_cards,
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
                    "card_name": related_name,
                    "card_description": related_card.get("card_description", ""),
                    "content": str(related_card.get("content", ""))[:1400],
                    "document_name": related_card.get("document_name", ""),
                    "page_no": related_card.get("page_no"),
                    "relationship": link.get("relationship", ""),
                    "relationship_type": link.get("relationship_type", ""),
                    "relationship_description": link.get("relationship_description", ""),
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
                    "document_name": related_card.get("document_name", ""),
                    "page_no": int(related_card.get("page_no") or 0),
                    "relationship_type": link.get("relationship_type", ""),
                    "relationship": link.get("relationship", ""),
                    "relationship_description": link.get("relationship_description", ""),
                    "direction": link.get("direction", ""),
                    "content": str(related_card.get("content", "")),
                    "card_source": related_card.get("card_source", ""),
                    "tags": related_card.get("tags", []),
                }
            )
        return related_results


def run_search(query: str, dry_run: bool, max_hits: int) -> dict[str, Any]:
    load_dotenv()
    searcher = TreeSearcher(query=query, dry_run=dry_run, max_hits=max_hits)
    result = searcher.search()
    output_path = SEARCH_RESULTS_DIR / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{slugify(query)[:60]}.json"
    write_json(output_path, result)
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


