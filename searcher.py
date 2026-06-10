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


PROJECT_ROOT = Path(os.getenv("PDF_VISION_RAG_ROOT", Path(__file__).resolve().parent)).resolve()
INDEXES_DIR = PROJECT_ROOT / "indexes"
TOPIC_INDEX_PATH = INDEXES_DIR / "topic_index.json"
RELATIONSHIP_MAP_PATH = INDEXES_DIR / "relationship_map.json"
SEARCH_RESULTS_DIR = INDEXES_DIR / "search_results"

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "deepseek/deepseek-v4-flash"
DEFAULT_SEARCH_MAX_TOKENS = 1024


@dataclass
class SearchHit:
    topic_name: str
    document_name: str
    page_no: int
    relevance_reason: str
    content: str
    topic_source: str
    tags: list[str]
    related_topics: list[dict[str, Any]]


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
        max_retries=int(os.getenv("OPENROUTER_SEARCH_MAX_RETRIES", os.getenv("OPENROUTER_MAX_RETRIES", "1"))),
        default_headers={
            "HTTP-Referer": os.getenv("OPENROUTER_SITE_URL", "http://localhost"),
            "X-Title": os.getenv("OPENROUTER_APP_NAME", "PDF Vision RAG"),
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
                    "name": candidate.get("name") or candidate.get("biome_name") or candidate.get("community_name") or candidate.get("topic_name"),
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
                "content": "You guide recursive tree search over a document relationship map. Return only valid JSON.",
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
        project_root: Path | None = None,
        indexes_dir: Path | None = None,
        topic_index_path: Path | None = None,
        relationship_map_path: Path | None = None,
        search_results_dir: Path | None = None,
    ) -> None:
        self.query = query
        self.dry_run = dry_run
        self.max_hits = max_hits
        self.project_root = (project_root or PROJECT_ROOT).resolve()
        self.indexes_dir = (indexes_dir or self.project_root / "indexes").resolve()
        self.topic_index_path = topic_index_path or self.indexes_dir / "topic_index.json"
        self.relationship_map_path = relationship_map_path or self.indexes_dir / "relationship_map.json"
        self.search_results_dir = search_results_dir or self.indexes_dir / "search_results"
        self.topics = read_json(self.topic_index_path, [])
        self.map = read_json(self.relationship_map_path, {})
        if not self.topics:
            raise SystemExit("No topic index found. Run indexer.py first.")
        if not self.map:
            raise SystemExit("No relationship map found. Run relationship_map.py first.")

        self.topic_lookup = {topic["topic_name"]: topic for topic in self.topics}
        self.community_lookup = {
            community["community_name"]: community
            for community in self.map.get("communities", [])
        }
        self.biome_lookup = {
            biome["biome_name"]: biome for biome in self.map.get("biomes", [])
        }
        self.relations_by_biome: dict[str, list[dict[str, Any]]] = {}
        self.topic_relation_links: dict[str, list[dict[str, Any]]] = {}
        for relation in self.map.get("biome_relations", []):
            self.relations_by_biome.setdefault(relation["main_biome"], []).append(relation)
            for link in relation.get("topic_links", []):
                main_topic = link.get("main_topic", "")
                related_topic = link.get("related_topic", "")
                if main_topic and related_topic:
                    self.topic_relation_links.setdefault(main_topic, []).append(
                        {
                            "topic_name": related_topic,
                            "direction": "outgoing",
                            "relation": link.get("relation", ""),
                            "relation_type": relation.get("relation_type", ""),
                            "relation_description": relation.get("relation_description", ""),
                            "source_biome": relation.get("main_biome", ""),
                            "related_biome": relation.get("related_biome", ""),
                        }
                    )
                    self.topic_relation_links.setdefault(related_topic, []).append(
                        {
                            "topic_name": main_topic,
                            "direction": "incoming",
                            "relation": link.get("relation", ""),
                            "relation_type": relation.get("relation_type", ""),
                            "relation_description": relation.get("relation_description", ""),
                            "source_biome": relation.get("related_biome", ""),
                            "related_biome": relation.get("main_biome", ""),
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
        biome_candidates = [
            {
                "name": biome["biome_name"],
                "biome_name": biome["biome_name"],
                "biome_description": biome.get("biome_description", ""),
                "document_name": biome.get("document_name", ""),
                "community_names": biome.get("community_names", []),
            }
            for biome in self.map.get("biomes", [])
        ]
        ranked_biomes = self.rank("biome", biome_candidates, ["biome_name", "biome_description", "document_name"])
        for item in ranked_biomes:
            self.visit_biome(item["name"], item.get("reason", ""))
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

    def visit_biome(self, biome_name: str, reason: str) -> None:
        if not self.mark_visit("biome", biome_name, reason):
            return
        biome = self.biome_lookup.get(biome_name)
        if not biome:
            return

        community_candidates = []
        for community_name in biome.get("community_names", []):
            community = self.community_lookup.get(community_name)
            if community:
                community_candidates.append(
                    {
                        "name": community_name,
                        "community_name": community_name,
                        "community_description": community.get("community_description", ""),
                        "topic_names": community.get("topic_names", []),
                    }
                )
        ranked_communities = self.rank(
            "community",
            community_candidates,
            ["community_name", "community_description"],
        )
        found_before = len(self.hits)
        for item in ranked_communities:
            self.visit_community(item["name"], item.get("reason", ""))
            if len(self.hits) >= self.max_hits:
                return

        if len(self.hits) == found_before:
            self.follow_related_biomes(biome_name)

    def follow_related_biomes(self, biome_name: str) -> None:
        relations = self.relations_by_biome.get(biome_name, [])
        relation_candidates = [
            {
                "name": relation["related_biome"],
                "related_biome": relation["related_biome"],
                "relation_type": relation.get("relation_type", ""),
                "relation_description": relation.get("relation_description", ""),
                "evidence": relation.get("evidence", ""),
            }
            for relation in relations
            if f"biome:{relation['related_biome']}" not in self.visited
        ]
        ranked = self.rank(
            "related biome",
            relation_candidates,
            ["related_biome", "relation_type", "relation_description", "evidence"],
        )
        for item in ranked:
            self.visit_biome(item["name"], f"Related biome: {item.get('reason', '')}")
            if len(self.hits) >= self.max_hits:
                return

    def visit_community(self, community_name: str, reason: str) -> None:
        if not self.mark_visit("community", community_name, reason):
            return
        community = self.community_lookup.get(community_name)
        if not community:
            return
        topic_candidates = []
        for topic_name in community.get("topic_names", []):
            topic = self.topic_lookup.get(topic_name)
            if topic:
                topic_candidates.append(
                    {
                        "name": topic_name,
                        "topic_name": topic_name,
                        "topic_description": topic.get("topic_description", ""),
                        "content": str(topic.get("content", ""))[:1600],
                        "tags": topic.get("tags", []),
                        "topic_source": topic.get("topic_source", ""),
                        "page_no": topic.get("page_no"),
                        "document_name": topic.get("document_name", ""),
                    }
                )
        ranked_topics = self.rank(
            "topic",
            topic_candidates,
            ["topic_name", "topic_description", "content", "tags", "document_name"],
        )
        for item in ranked_topics:
            self.visit_topic(item["name"], item.get("reason", ""))
            if len(self.hits) >= self.max_hits:
                return

    def visit_topic(self, topic_name: str, reason: str) -> None:
        if not self.mark_visit("topic", topic_name, reason):
            return
        topic = self.topic_lookup.get(topic_name)
        if not topic:
            return
        related_topics = self.find_helpful_related_topics(topic_name)
        self.hits.append(
            SearchHit(
                topic_name=topic_name,
                document_name=topic.get("document_name", ""),
                page_no=int(topic.get("page_no") or 0),
                relevance_reason=reason,
                content=str(topic.get("content", "")),
                topic_source=topic.get("topic_source", ""),
                tags=topic.get("tags", []),
                related_topics=related_topics,
            )
        )

    def find_helpful_related_topics(self, topic_name: str) -> list[dict[str, Any]]:
        relation_links = self.topic_relation_links.get(topic_name, [])
        candidates = []
        for link in relation_links:
            related_name = link.get("topic_name", "")
            if not related_name or f"relation-topic:{topic_name}->{related_name}" in self.visited:
                continue
            related_topic = self.topic_lookup.get(related_name)
            if not related_topic:
                continue
            candidates.append(
                {
                    "name": related_name,
                    "topic_name": related_name,
                    "topic_description": related_topic.get("topic_description", ""),
                    "content": str(related_topic.get("content", ""))[:1400],
                    "document_name": related_topic.get("document_name", ""),
                    "page_no": related_topic.get("page_no"),
                    "relation": link.get("relation", ""),
                    "relation_type": link.get("relation_type", ""),
                    "relation_description": link.get("relation_description", ""),
                    "direction": link.get("direction", ""),
                }
            )

        ranked = self.rank(
            "related topic",
            candidates,
            [
                "topic_name",
                "topic_description",
                "content",
                "relation",
                "relation_type",
                "relation_description",
                "document_name",
            ],
        )
        related_results: list[dict[str, Any]] = []
        ranked_names = [item["name"] for item in ranked[:5]]
        for related_name in ranked_names:
            self.visited.add(f"relation-topic:{topic_name}->{related_name}")
            related_topic = self.topic_lookup.get(related_name)
            link = next(
                (item for item in relation_links if item.get("topic_name") == related_name),
                None,
            )
            if not related_topic or not link:
                continue
            related_results.append(
                {
                    "topic_name": related_name,
                    "document_name": related_topic.get("document_name", ""),
                    "page_no": int(related_topic.get("page_no") or 0),
                    "relation_type": link.get("relation_type", ""),
                    "relation": link.get("relation", ""),
                    "relation_description": link.get("relation_description", ""),
                    "direction": link.get("direction", ""),
                    "content": str(related_topic.get("content", "")),
                    "topic_source": related_topic.get("topic_source", ""),
                    "tags": related_topic.get("tags", []),
                }
            )
        return related_results


def run_search(query: str, dry_run: bool, max_hits: int) -> dict[str, Any]:
    load_dotenv()
    searcher = TreeSearcher(query=query, dry_run=dry_run, max_hits=max_hits)
    result = searcher.search()
    output_path = searcher.search_results_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{slugify(query)[:60]}.json"
    write_json(output_path, result)
    print(f"Wrote search result to {output_path}")
    for hit in result["hits"]:
        print(f"- {hit['topic_name']} ({hit['document_name']}, page {hit['page_no']})")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Recursive tree search over biomes, communities, and topics.")
    parser.add_argument("query", help="Broad search query.")
    parser.add_argument("--dry-run", action="store_true", help="Use keyword ranking instead of OpenRouter.")
    parser.add_argument("--max-hits", type=int, default=12, help="Maximum topic hits to return.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_search(query=args.query, dry_run=args.dry_run, max_hits=args.max_hits)


if __name__ == "__main__":
    main()
