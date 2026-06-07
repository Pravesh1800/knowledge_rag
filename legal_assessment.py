from __future__ import annotations

import json
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from openai import OpenAI

from legal_rules import LEGAL_ROWS
from legal_rules import detect_legal_flags
import searcher as searcher_module


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "google/gemini-3.1-flash-lite"
DEFAULT_SEARCH_MODEL = "deepseek/deepseek-v4-flash"
DEFAULT_WEB_RESEARCH_MODEL = "perplexity/sonar-reasoning-pro"
DEFAULT_AGENT_MAX_TOKENS = 2048
DEFAULT_WEB_MAX_TOKENS = 1024
MAX_AGENT_STEPS_PER_ROW = 14
MAX_EVIDENCE_PER_ROW = 18
DEFAULT_LEGAL_WORKERS = 7
DEFAULT_LEGAL_PARALLEL_RETRIES = 3


def load_dotenv(project_root: Path) -> None:
    for env_path in [project_root / ".env", Path(__file__).resolve().parent / ".env"]:
        if not env_path.exists():
            continue
        for line in env_path.read_text(encoding="utf-8-sig").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ[key.strip()] = value.strip().strip("\"'")


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def parse_json_response(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    candidates = [stripped]
    match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
    if match:
        candidates.append(match.group(0))
    repaired_candidates = []
    for candidate in candidates:
        # Common model slip in long JSON arrays: adjacent objects with no comma.
        repaired_candidates.append(re.sub(r"(\})\s*(\{)", r"\1,\2", candidate))
        # Same issue when an object is followed by a quoted key in an array item list.
        repaired_candidates.append(re.sub(r"(\})\s*(\")", r"\1,\2", candidate))
    candidates.extend(repaired_candidates)
    seen = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
            return {"items": parsed}
        except json.JSONDecodeError as exc:
            last_error = exc
            continue
    if "last_error" in locals():
        raise last_error
    return {}


def create_client() -> tuple[OpenAI, str]:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is missing in .env")
    client = OpenAI(
        base_url=os.getenv("OPENROUTER_BASE_URL", OPENROUTER_BASE_URL),
        api_key=api_key,
        timeout=float(os.getenv("OPENROUTER_TIMEOUT_SECONDS", "120")),
        max_retries=int(os.getenv("OPENROUTER_MAX_RETRIES", "1")),
        default_headers={
            "HTTP-Referer": os.getenv("OPENROUTER_SITE_URL", "http://localhost"),
            "X-Title": os.getenv("OPENROUTER_APP_NAME", "PDF Vision RAG"),
        },
    )
    return client, os.getenv("OPENROUTER_LEGAL_AGENT_MODEL", os.getenv("OPENROUTER_MODEL", DEFAULT_MODEL))


def json_completion(
    client: OpenAI,
    model: str,
    prompt: str,
    system: str,
    max_tokens: int | None = None,
) -> dict[str, Any]:
    max_tokens = max_tokens or int(os.getenv("OPENROUTER_AGENT_MAX_TOKENS", str(DEFAULT_AGENT_MAX_TOKENS)))
    last_content = ""
    last_error: Exception | None = None
    for _ in range(2):
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            response_format={"type": "json_object"},
            max_tokens=max_tokens,
        )
        last_content = response.choices[0].message.content or "{}"
        try:
            return parse_json_response(last_content)
        except Exception as exc:
            last_error = exc

    repair_error = str(last_error or "Unknown JSON parse error")
    repair_content = last_content
    for repair_attempt in range(1, 4):
        repair_prompt = f"""
Repair this malformed model output into valid JSON only.

Exact parser error to fix:
{repair_error}

Required output:
- valid JSON object only
- no markdown fences
- no comments
- double-quote every object key and string
- close any unterminated string
- insert missing commas between adjacent properties, objects, and array items
- remove trailing commas
- preserve all fields and meanings that can be recovered

Malformed output:
{repair_content}
""".strip()
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a JSON repair engine. Return only one valid JSON object."},
                {"role": "user", "content": repair_prompt},
            ],
            temperature=0,
            response_format={"type": "json_object"},
            max_tokens=max_tokens,
        )
        repair_content = response.choices[0].message.content or "{}"
        try:
            return parse_json_response(repair_content)
        except Exception as exc:
            repair_error = str(exc)

    if last_error is not None:
        raise last_error
    raise ValueError("Model returned malformed JSON and repair failed.")


def evidence_key(item: dict[str, Any]) -> str:
    return "|".join(
        [
            str(item.get("document_id", "")),
            str(item.get("page_no", "")),
            str(item.get("topic_name", "")),
        ]
    )


def topic_to_evidence(topic: dict[str, Any], channel: str, reason: str = "") -> dict[str, Any]:
    return {
        "document_id": topic.get("document_id", ""),
        "document_name": topic.get("document_name", ""),
        "page_no": topic.get("page_no"),
        "topic_name": topic.get("topic_name", ""),
        "topic_description": topic.get("topic_description", ""),
        "content": str(topic.get("content", "")),
        "tags": topic.get("tags", []),
        "legal_flags": topic.get("legal_flags", []),
        "legal_flag_reasons": topic.get("legal_flag_reasons", {}),
        "legal_confidence": topic.get("legal_confidence", "low"),
        "source_channels": [channel],
        "match_reason": reason,
    }


def merge_evidence(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for item in items:
        key = evidence_key(item)
        if not key.strip("|"):
            continue
        if key not in merged:
            merged[key] = item
            continue
        channels = set(merged[key].get("source_channels", []))
        channels.update(item.get("source_channels", []))
        merged[key]["source_channels"] = sorted(channels)
        if item.get("match_reason"):
            existing = merged[key].get("match_reason", "")
            merged[key]["match_reason"] = "; ".join(part for part in [existing, item["match_reason"]] if part)
    return sorted(
        merged.values(),
        key=lambda item: (
            0 if set(item.get("source_channels", [])) == {"flagged", "searched"} else 1,
            0 if "flagged" in item.get("source_channels", []) else 1,
            str(item.get("document_name", "")),
            int(item.get("page_no") or 0),
        ),
    )


def compact_evidence(items: list[dict[str, Any]], limit: int = MAX_EVIDENCE_PER_ROW) -> list[dict[str, Any]]:
    compacted = []
    for item in items[:limit]:
        compacted.append(
            {
                "document_id": item.get("document_id", ""),
                "document_name": item.get("document_name", ""),
                "page_no": item.get("page_no"),
                "topic_name": item.get("topic_name", ""),
                "topic_description": item.get("topic_description", ""),
                "excerpt": str(item.get("content", ""))[:1800],
                "source_channels": item.get("source_channels", []),
                "match_reason": item.get("match_reason", ""),
            }
        )
    return compacted


class LegalAssessmentAgent:
    def __init__(self, project_root: Path) -> None:
        self.project_root = project_root.resolve()
        self.indexes_dir = self.project_root / "indexes"
        self.reports_dir = self.project_root / "reports"
        self.topics = read_json(self.indexes_dir / "topic_index.json", [])
        self.map = read_json(self.indexes_dir / "relationship_map.json", {})
        if not self.topics:
            raise RuntimeError("No topic index found. Upload and index documents first.")
        if not self.map:
            raise RuntimeError("No relationship map found. Upload and index documents first.")
        self.ensure_legal_flags()
        self.topic_lookup = {topic.get("topic_name", ""): topic for topic in self.topics}
        self.community_lookup = {
            community.get("community_name", ""): community
            for community in self.map.get("communities", [])
        }
        self.biome_lookup = {
            biome.get("biome_name", ""): biome
            for biome in self.map.get("biomes", [])
        }
        load_dotenv(self.project_root)
        self.client, self.model = create_client()
        self.web_research_model = os.getenv("OPENROUTER_WEB_RESEARCH_MODEL", DEFAULT_WEB_RESEARCH_MODEL)
        self.logs: list[dict[str, Any]] = []
        self.log_lock = threading.Lock()
        self.progress_path = self.reports_dir / "legal_assessment_7_deadly_sins.progress.json"

    def ensure_legal_flags(self) -> None:
        changed = False
        for topic in self.topics:
            if "legal_flags" in topic and "legal_flag_reasons" in topic and "legal_confidence" in topic:
                continue
            flags, reasons, confidence = detect_legal_flags(topic)
            topic["legal_flags"] = flags
            topic["legal_flag_reasons"] = reasons
            topic["legal_confidence"] = confidence
            changed = True
        if changed:
            write_json(self.indexes_dir / "topic_index.json", self.topics)

    def log(self, message: str, row_id: str | None = None, detail: dict[str, Any] | None = None) -> None:
        entry = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "row_id": row_id,
            "message": message,
            "detail": detail or {},
        }
        with self.log_lock:
            self.logs.append(entry)
            logs = list(self.logs)
            write_json(
                self.progress_path,
                {
                    "report_type": "legal_assessment_7_deadly_sins",
                    "status": "running",
                    "project_id": self.project_root.name,
                    "updated_at": entry["created_at"],
                    "logs": logs,
                },
            )

    def flagged_evidence(self, row_id: str) -> list[dict[str, Any]]:
        evidence = [
            topic_to_evidence(
                topic,
                "flagged",
                str((topic.get("legal_flag_reasons") or {}).get(row_id, "Marked during indexing as relevant.")),
            )
            for topic in self.topics
            if row_id in (topic.get("legal_flags") or [])
        ]
        return evidence

    def searched_evidence(self, row: dict[str, Any], query: str | None = None, max_hits: int = 12) -> list[dict[str, Any]]:
        os.environ["PDF_VISION_RAG_ROOT"] = str(self.project_root)
        os.environ.setdefault("OPENROUTER_SEARCH_MODEL", os.getenv("OPENROUTER_SEARCH_MODEL", DEFAULT_SEARCH_MODEL))
        searcher_module.PROJECT_ROOT = self.project_root
        searcher_module.INDEXES_DIR = self.indexes_dir
        searcher_module.TOPIC_INDEX_PATH = self.indexes_dir / "topic_index.json"
        searcher_module.RELATIONSHIP_MAP_PATH = self.indexes_dir / "relationship_map.json"
        searcher_module.SEARCH_RESULTS_DIR = self.indexes_dir / "search_results"
        search_query = query or f"{row['topic']}. {row['decision_rule']} Search terms: {', '.join(row['search_terms'])}"
        searcher = searcher_module.TreeSearcher(query=search_query, dry_run=False, max_hits=max_hits)
        result = searcher.search()
        evidence: list[dict[str, Any]] = []
        for hit in result.get("hits", []):
            evidence.append(
                {
                    "document_id": "",
                    "document_name": hit.get("document_name", ""),
                    "page_no": hit.get("page_no"),
                    "topic_name": hit.get("topic_name", ""),
                    "topic_description": "",
                    "content": hit.get("content", ""),
                    "source_channels": ["searched"],
                    "match_reason": hit.get("relevance_reason", "Found by targeted search."),
                }
            )
            for related in hit.get("related_topics", []):
                evidence.append(
                    {
                        "document_id": "",
                        "document_name": related.get("document_name", ""),
                        "page_no": related.get("page_no"),
                        "topic_name": related.get("topic_name", ""),
                        "topic_description": related.get("topic_description", ""),
                        "content": related.get("content", ""),
                        "source_channels": ["searched"],
                        "match_reason": related.get("relation", "Related topic found by search."),
                    }
                )
        # Fill missing document_id by matching topic name/doc/page back to topic index.
        by_triplet = {
            (topic.get("topic_name"), topic.get("document_name"), int(topic.get("page_no") or 0)): topic
            for topic in self.topics
        }
        enriched = []
        for item in evidence:
            topic = by_triplet.get((item.get("topic_name"), item.get("document_name"), int(item.get("page_no") or 0)))
            if topic:
                enriched.append(topic_to_evidence(topic, "searched", item.get("match_reason", "")))
            else:
                enriched.append(item)
        return enriched

    def project_inventory(self) -> dict[str, Any]:
        return {
            "documents": sorted({topic.get("document_name", "") for topic in self.topics if topic.get("document_name")}),
            "biomes": [
                {
                    "biome_name": biome.get("biome_name", ""),
                    "biome_description": biome.get("biome_description", ""),
                    "community_names": biome.get("community_names", []),
                }
                for biome in self.map.get("biomes", [])
            ],
            "communities": [
                {
                    "community_name": community.get("community_name", ""),
                    "community_description": community.get("community_description", ""),
                    "document_name": community.get("document_name", ""),
                    "topic_names": community.get("topic_names", [])[:30],
                }
                for community in self.map.get("communities", [])
            ],
            "topic_count": len(self.topics),
        }

    def list_biomes(self) -> list[dict[str, Any]]:
        return [
            {
                "biome_name": biome.get("biome_name", ""),
                "biome_description": biome.get("biome_description", ""),
                "document_name": biome.get("document_name", ""),
                "community_names": biome.get("community_names", []),
            }
            for biome in self.map.get("biomes", [])
        ]

    def get_biome(self, biome_name: str) -> dict[str, Any]:
        biome = self.biome_lookup.get(biome_name)
        if not biome:
            return {"error": f"Biome not found: {biome_name}"}
        communities = [
            self.community_lookup.get(name, {"community_name": name, "error": "missing"})
            for name in biome.get("community_names", [])
        ]
        return {"biome": biome, "communities": communities}

    def list_communities(self, document_name: str | None = None) -> list[dict[str, Any]]:
        communities = self.map.get("communities", [])
        if document_name:
            communities = [
                community for community in communities
                if document_name.lower() in str(community.get("document_name", "")).lower()
            ]
        return [
            {
                "community_name": community.get("community_name", ""),
                "community_description": community.get("community_description", ""),
                "document_name": community.get("document_name", ""),
                "topic_names": community.get("topic_names", []),
            }
            for community in communities
        ]

    def get_community(self, community_name: str) -> dict[str, Any]:
        community = self.community_lookup.get(community_name)
        if not community:
            return {"error": f"Community not found: {community_name}"}
        topics = [
            self.topic_lookup.get(name, {"topic_name": name, "error": "missing"})
            for name in community.get("topic_names", [])
        ]
        return {
            "community": community,
            "topics": [
                {
                    "topic_name": topic.get("topic_name", ""),
                    "topic_description": topic.get("topic_description", ""),
                    "document_name": topic.get("document_name", ""),
                    "page_no": topic.get("page_no"),
                    "legal_flags": topic.get("legal_flags", []),
                    "content_excerpt": str(topic.get("content", ""))[:1200],
                }
                for topic in topics
            ],
        }

    def list_topics(self, query: str | None = None, limit: int = 40) -> list[dict[str, Any]]:
        rows = self.topics
        if query:
            terms = [term for term in re.findall(r"[a-z0-9]+", query.lower()) if len(term) > 2]
            scored = []
            for topic in rows:
                text = " ".join(
                    str(topic.get(key, ""))
                    for key in ["topic_name", "topic_description", "content", "document_name"]
                ).lower()
                score = sum(1 for term in terms if term in text)
                if score:
                    scored.append((score, topic))
            scored.sort(key=lambda item: item[0], reverse=True)
            rows = [topic for _, topic in scored]
        return [
            {
                "topic_name": topic.get("topic_name", ""),
                "topic_description": topic.get("topic_description", ""),
                "document_name": topic.get("document_name", ""),
                "page_no": topic.get("page_no"),
                "legal_flags": topic.get("legal_flags", []),
            }
            for topic in rows[:limit]
        ]

    def get_topic(self, topic_name: str) -> dict[str, Any]:
        topic = self.topic_lookup.get(topic_name)
        if not topic:
            return {"error": f"Topic not found: {topic_name}"}
        return topic

    def web_research(self, query: str, purpose: str) -> dict[str, Any]:
        prompt = f"""
Research the public web for the following project/bid/company question.

Query:
{query}

Purpose:
{purpose}

Return concise, decision-useful findings. Prefer official client, government, tender, company, regulatory, or reputable news sources.

Return JSON if possible:
{{
  "summary": "short answer",
  "findings": ["finding with source context"],
  "citations": [
    {{"title": "source title", "url": "https://...", "note": "what it supports"}}
  ],
  "limits": "what could not be verified"
}}
""".strip()
        try:
            response = self.client.chat.completions.create(
                model=self.web_research_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an internet research assistant for bid strategy and project due diligence. Cite sources with URLs.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
                max_tokens=int(os.getenv("OPENROUTER_WEB_MAX_TOKENS", str(DEFAULT_WEB_MAX_TOKENS))),
            )
            choices = getattr(response, "choices", None) or []
            content = choices[0].message.content if choices else ""
            if not content:
                raise RuntimeError("Web research provider returned no content.")
        except Exception as exc:
            return {
                "model": self.web_research_model,
                "query": query,
                "purpose": purpose,
                "result": {
                    "summary": "",
                    "findings": [],
                    "citations": [],
                    "limits": f"Web research failed and was skipped: {exc}",
                },
            }
        try:
            parsed = parse_json_response(content)
        except Exception:
            parsed = {"summary": content, "findings": [], "citations": [], "limits": "Model did not return parseable JSON."}
        return {
            "model": self.web_research_model,
            "query": query,
            "purpose": purpose,
            "result": parsed,
        }

    def legal_research(self, query: str, purpose: str, jurisdiction: str = "India") -> dict[str, Any]:
        prompt = f"""
Research legal/statutory/public-law context for a tender legal assessment.

Jurisdiction:
{jurisdiction}

Legal research query:
{query}

Purpose:
{purpose}

Look for:
- statutes, rules, regulations, government guidance, tribunal/court context, standard legal meaning, or standard clause interpretation
- official sources first, then reputable legal/commentary sources if official sources are not enough
- clause concepts relevant to arbitration, force majeure, liquidated damages, liability caps, consequential damages, effective date/advance payment, currency/forex variation, public procurement, and tender risk allocation

Important limits:
- Do not provide legal advice.
- Do not decide the tender answer from public law alone.
- Tender/document wording controls the final Yes/No.
- Use legal research only to interpret terms, identify applicable law context, or decide what additional tender wording to search for.

Return JSON:
{{
  "summary": "short legal-context answer",
  "applicable_law_or_clause_context": ["statute/rule/case/standard-clause context with source"],
  "search_implications": ["exact tender terms/clauses the specialist should search for next"],
  "citations": [
    {{"title": "source title", "url": "https://...", "note": "what it supports"}}
  ],
  "limits": "what could not be verified or why tender text still controls"
}}
""".strip()
        try:
            response = self.client.chat.completions.create(
                model=self.web_research_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a legal research assistant for tender due diligence. Cite sources with URLs and clearly state limits.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
                max_tokens=int(os.getenv("OPENROUTER_WEB_MAX_TOKENS", str(DEFAULT_WEB_MAX_TOKENS))),
            )
            choices = getattr(response, "choices", None) or []
            content = choices[0].message.content if choices else ""
            if not content:
                raise RuntimeError("Legal research provider returned no content.")
        except Exception as exc:
            return {
                "model": self.web_research_model,
                "query": query,
                "purpose": purpose,
                "jurisdiction": jurisdiction,
                "result": {
                    "summary": "",
                    "applicable_law_or_clause_context": [],
                    "search_implications": [],
                    "citations": [],
                    "limits": f"Legal research failed and was skipped: {exc}",
                },
            }
        try:
            parsed = parse_json_response(content)
        except Exception:
            parsed = {
                "summary": content,
                "applicable_law_or_clause_context": [],
                "search_implications": [],
                "citations": [],
                "limits": "Model did not return parseable JSON.",
            }
        return {
            "model": self.web_research_model,
            "query": query,
            "purpose": purpose,
            "jurisdiction": jurisdiction,
            "result": parsed,
        }

    def search_topics_keyword(self, terms: list[str], limit: int = 12) -> list[dict[str, Any]]:
        terms = [term.lower() for term in terms if term]
        scored = []
        for topic in self.topics:
            text = " ".join(
                str(topic.get(key, ""))
                for key in ["topic_name", "topic_description", "content", "document_name"]
            ).lower()
            score = sum(1 for term in terms if term in text)
            if score:
                scored.append((score, topic))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [topic_to_evidence(topic, "agent_lookup", f"Agent keyword lookup matched {score} terms.") for score, topic in scored[:limit]]

    def run_agent_action(self, action: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
        action_type = str(action.get("action", "")).strip()
        if action_type == "search":
            query = str(action.get("query", "")).strip() or row["topic"]
            self.log(f"Running an extra project search for: {query}", row["id"])
            return {"evidence": compact_evidence(self.searched_evidence(row, query=query, max_hits=10))}
        if action_type == "keyword_lookup":
            terms = [str(term) for term in action.get("terms", []) if str(term).strip()]
            self.log(f"Scanning all indexed topics for exact terms: {', '.join(terms[:8])}", row["id"])
            return {"evidence": compact_evidence(self.search_topics_keyword(terms, limit=12))}
        if action_type == "list_inventory":
            self.log("Reviewing the project map: documents, biomes, communities, and topic coverage.", row["id"])
            return {"inventory": self.project_inventory()}
        if action_type == "list_biomes":
            self.log("Listing all biomes so the specialist can understand the project map.", row["id"])
            return {"biomes": self.list_biomes()}
        if action_type == "get_biome":
            biome_name = str(action.get("biome_name", "")).strip()
            self.log(f"Opening biome: {biome_name}", row["id"])
            return self.get_biome(biome_name)
        if action_type == "list_communities":
            document_name = str(action.get("document_name", "")).strip() or None
            self.log(
                f"Listing communities{f' for {document_name}' if document_name else ''}.",
                row["id"],
            )
            return {"communities": self.list_communities(document_name)}
        if action_type == "get_community":
            community_name = str(action.get("community_name", "")).strip()
            self.log(f"Opening community and its topics: {community_name}", row["id"])
            return self.get_community(community_name)
        if action_type == "list_topics":
            query = str(action.get("query", "")).strip() or None
            self.log(
                f"Listing indexed topics{f' matching: {query}' if query else ''}.",
                row["id"],
            )
            return {"topics": self.list_topics(query=query, limit=int(action.get("limit") or 40))}
        if action_type == "get_topic":
            topic_name = str(action.get("topic_name", "")).strip()
            self.log(f"Opening full topic content: {topic_name}", row["id"])
            topic = self.get_topic(topic_name)
            if "error" not in topic:
                return {"topic": topic, "evidence": compact_evidence([topic_to_evidence(topic, "agent_inspected", "Specialist opened full topic content.")], limit=1)}
            return topic
        if action_type == "web_research":
            query = str(action.get("query", "")).strip()
            purpose = str(action.get("purpose", "")).strip() or row["topic"]
            self.log(f"Researching the public web with Sonar Reasoning Pro: {query}", row["id"])
            return {"web_research": self.web_research(query, purpose)}
        if action_type == "legal_research":
            query = str(action.get("query", "")).strip()
            purpose = str(action.get("purpose", "")).strip() or row["topic"]
            jurisdiction = str(action.get("jurisdiction", "")).strip() or "India"
            self.log(f"Researching legal/statutory context with Sonar Reasoning Pro: {query}", row["id"], {"jurisdiction": jurisdiction})
            return {"legal_research": self.legal_research(query, purpose, jurisdiction)}
        if action_type == "answer":
            return {"answer": action}
        self.log("The specialist requested an unknown action, so I asked it to answer with available evidence.", row["id"], {"action": action})
        return {"error": "unknown action"}

    def specialist_prompt(
        self,
        row: dict[str, Any],
        evidence: list[dict[str, Any]],
        step_results: list[dict[str, Any]],
    ) -> str:
        return f"""
You are a specialist legal assessment agent. Your only job is to answer one row of SUEZ's "Legal Assessment of 7 Deadly Sins".

You must not miss relevant wording. You have starter evidence from legal flags and targeted search, and you can ask for more project context before answering.

Current row:
{json.dumps(row, ensure_ascii=False)}

How to interpret this row:
- description: explains the commercial/legal risk SUEZ is checking.
- what_to_find: exact type of tender wording that can support Yes.
- not_enough: wording that may look relevant but must still be No if it does not satisfy the legal test.
- examples: calibration examples for Yes, No, and edge cases. Do not require exact example wording, but apply the same logic.
- decision_rule: final strict Yes/No rule.

Starter and accumulated evidence:
{json.dumps(compact_evidence(evidence), ensure_ascii=False)}

Previous step results:
{json.dumps(step_results[-6:], ensure_ascii=False)}

Available actions. Return exactly one JSON object:
1. {{"action":"search","query":"specific project search query"}}
2. {{"action":"keyword_lookup","terms":["exact term", "another term"]}}
3. {{"action":"list_inventory"}}
4. {{"action":"list_biomes"}}
5. {{"action":"get_biome","biome_name":"exact biome name"}}
6. {{"action":"list_communities","document_name":"optional document name filter"}}
7. {{"action":"get_community","community_name":"exact community name"}}
8. {{"action":"list_topics","query":"optional keyword query","limit":40}}
9. {{"action":"get_topic","topic_name":"exact topic name"}}
10. {{"action":"web_research","query":"public web research query","purpose":"why external context is needed"}}
11. {{"action":"legal_research","query":"legal/statutory/clause research query","jurisdiction":"India or other relevant jurisdiction","purpose":"why legal context is needed"}}
12. {{"action":"answer","yes_no":"Yes|No","comment":"SUEZ-style table comment","citations":[{{"document_name":"","page_no":0,"topic_name":"","excerpt":""}}],"legal_context_citations":[{{"title":"","url":"","note":""}}],"absence_basis":"what was searched if No due to absence/silence","confidence":"low|medium|high","notes":"short audit note"}}

Rules:
- First understand the row's description, what_to_find, not_enough, and decision_rule. Use all four when deciding.
- Use row examples to calibrate the threshold. They are not exact search strings; they define what counts and what does not.
- Choose Yes only when the exact decision rule is satisfied.
- Choose No when the clause is absent, silent, incomplete, or mentions the topic without satisfying the row's legal test.
- Treat "not_enough" examples as explicit traps. If the evidence only matches not_enough, answer No.
- If you need broader context, ask for list_inventory once.
- If you need map context, inspect biomes and communities.
- If you need precise wording, use search, keyword_lookup, list_topics, get_community, or get_topic.
- If you need public external context about the bid, company, authority, funding program, market, competitor, or regulation, use web_research.
- If you need legal/statutory context, standard clause meaning, public procurement law context, arbitration tribunal context, or what clause wording to search for, use legal_research.
- Do not use web_research to override tender text. Tender/document evidence controls legal Yes/No answers.
- Do not use legal_research to override tender text. Legal research helps interpret terms and guide search, but tender/document evidence controls final Yes/No.
- Continue iterating until you are clear enough to answer the table row.
- Do not answer until you have checked both the provided starter evidence and any extra context needed to avoid missing relevant wording.
""".strip()

    def verifier_prompt(self, row: dict[str, Any], draft: dict[str, Any], evidence: list[dict[str, Any]]) -> str:
        return f"""
You verify one row of SUEZ's Legal Assessment of 7 Deadly Sins.

Row rule:
{json.dumps(row, ensure_ascii=False)}

Draft answer:
{json.dumps(draft, ensure_ascii=False)}

Evidence available:
{json.dumps(compact_evidence(evidence), ensure_ascii=False)}

Return only valid JSON:
{{
  "verified": true,
  "final_yes_no": "Yes|No",
  "final_comment": "final concise table comment",
  "legal_context_note": "how legal/statutory research was used, if used",
  "warnings": ["warning if any"],
  "verification_note": "why the final answer is supported"
}}

Verification rules:
- Verify against the row's description, what_to_find, not_enough, and decision_rule, not only the row title.
- Use row examples as calibration for what should be accepted or rejected.
- A citation must actually support the answer.
- Be strict: a clause mention is not enough if the exact SUEZ legal test is not met.
- If the draft relies only on wording described in not_enough, force final_yes_no to No and warn why.
- Public legal research can support interpretation/search strategy, but cannot replace missing tender wording for a Yes.
- If evidence is weak or absent, final_yes_no should usually be No with a clear warning.
""".strip()

    def call_json(self, prompt: str, system: str) -> dict[str, Any]:
        return json_completion(
            self.client,
            self.model,
            prompt,
            system,
            int(os.getenv("OPENROUTER_AGENT_MAX_TOKENS", str(DEFAULT_AGENT_MAX_TOKENS))),
        )

    def answer_row(self, row: dict[str, Any]) -> dict[str, Any]:
        self.log(f"Starting legal specialist review for row {row['s_no']}: {row['topic']}", row["id"])
        flagged = self.flagged_evidence(row["id"])
        self.log(f"Collected {len(flagged)} pre-flagged topic(s) from the index.", row["id"])
        searched = self.searched_evidence(row)
        self.log(f"Ran the required fresh targeted search and collected {len(searched)} searched topic(s).", row["id"])
        evidence = merge_evidence(flagged + searched)
        self.log(f"Merged and deduplicated evidence to {len(evidence)} unique topic/page candidate(s).", row["id"])

        step_results: list[dict[str, Any]] = []
        draft: dict[str, Any] | None = None
        for step in range(1, MAX_AGENT_STEPS_PER_ROW + 1):
            self.log(f"Specialist thinking step {step}: deciding whether more project context is needed.", row["id"])
            action = self.call_json(
                self.specialist_prompt(row, evidence, step_results),
                "Return only valid JSON. You are a legal clause assessment specialist.",
            )
            self.log(
                f"Specialist requested tool action: {action.get('action', 'unknown')}.",
                row["id"],
                {"action": action},
            )
            result = self.run_agent_action(action, row)
            self.log(
                f"Tool action completed: {action.get('action', 'unknown')}.",
                row["id"],
                {"result_keys": sorted(result.keys())},
            )
            step_results.append({"action": action, "result": result})
            for item in result.get("evidence", []):
                evidence.append(
                    {
                        "document_id": item.get("document_id", ""),
                        "document_name": item.get("document_name", ""),
                        "page_no": item.get("page_no"),
                        "topic_name": item.get("topic_name", ""),
                        "topic_description": item.get("topic_description", ""),
                        "content": item.get("excerpt", ""),
                        "source_channels": ["agent_lookup"],
                        "match_reason": item.get("match_reason", ""),
                    }
                )
            evidence = merge_evidence(evidence)
            if "answer" in result:
                draft = result["answer"]
                self.log(f"Specialist drafted row {row['s_no']} as {draft.get('yes_no', 'unknown')}.", row["id"])
                break

        if draft is None:
            self.log("Specialist reached the step limit; forcing an answer from accumulated evidence.", row["id"])
            draft = self.call_json(
                self.specialist_prompt(row, evidence, step_results)
                + "\n\nYou must now return action=answer using the accumulated evidence.",
                "Return only valid JSON.",
            )
            if draft.get("action") == "answer":
                draft = draft

        verifier = self.call_json(
            self.verifier_prompt(row, draft, evidence),
            "Return only valid JSON. You are a strict verifier of legal table answers.",
        )
        self.log(
            f"Verifier finalized row {row['s_no']} as {verifier.get('final_yes_no', draft.get('yes_no', 'No'))}.",
            row["id"],
            {"warnings": verifier.get("warnings", [])},
        )
        return {
            "s_no": row["s_no"],
            "row_id": row["id"],
            "topic": row["topic"],
            "yes_no": verifier.get("final_yes_no") or draft.get("yes_no") or "No",
            "comments": verifier.get("final_comment") or draft.get("comment") or "",
            "citations": draft.get("citations", []),
            "legal_context_citations": draft.get("legal_context_citations", []),
            "legal_context_note": verifier.get("legal_context_note", ""),
            "evidence": compact_evidence(evidence, limit=MAX_EVIDENCE_PER_ROW),
            "absence_basis": draft.get("absence_basis", ""),
            "confidence": draft.get("confidence", "medium"),
            "verifier": verifier,
            "agent_steps": step_results,
        }

    def answer_row_with_retries(self, row: dict[str, Any]) -> dict[str, Any]:
        attempts = max(1, int(os.getenv("OPENROUTER_LEGAL_ROW_RETRIES", str(DEFAULT_LEGAL_PARALLEL_RETRIES))))
        last_error: Exception | None = None
        for attempt in range(1, attempts + 1):
            try:
                if attempt > 1:
                    self.log(f"Retrying legal row attempt {attempt}/{attempts}.", row["id"])
                return self.answer_row(row)
            except Exception as exc:
                last_error = exc
                self.log(
                    f"Legal row attempt {attempt}/{attempts} failed: {exc}",
                    row["id"],
                    {"error": str(exc)},
                )
        assert last_error is not None
        raise last_error

    def generate(self) -> dict[str, Any]:
        self.log("Starting Legal Assessment of 7 Deadly Sins generation.")
        workers = max(1, int(os.getenv("OPENROUTER_LEGAL_WORKERS", str(DEFAULT_LEGAL_WORKERS))))
        workers = min(workers, len(LEGAL_ROWS))
        self.log(f"Running Legal Assessment rows with {workers} parallel worker(s).")
        rows_by_id: dict[str, dict[str, Any]] = {}
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(self.answer_row_with_retries, row): row
                for row in LEGAL_ROWS
            }
            for future in as_completed(futures):
                row = futures[future]
                rows_by_id[row["id"]] = future.result()
        rows = [
            rows_by_id[row["id"]]
            for row in LEGAL_ROWS
            if row["id"] in rows_by_id
        ]
        report = {
            "report_type": "legal_assessment_7_deadly_sins",
            "status": "verified",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "project_id": self.project_root.name,
            "generation_settings": {
                "row_workers": workers,
            },
            "rows": rows,
            "logs": self.logs,
        }
        output_path = self.reports_dir / "legal_assessment_7_deadly_sins.json"
        write_json(output_path, report)
        self.log(f"Saved Legal Assessment report to {output_path.name}.")
        report["logs"] = self.logs
        write_json(output_path, report)
        write_json(
            self.progress_path,
            {
                "report_type": "legal_assessment_7_deadly_sins",
                "status": "complete",
                "project_id": self.project_root.name,
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "logs": self.logs,
            },
        )
        return report


def generate_legal_assessment(project_root: Path) -> dict[str, Any]:
    return LegalAssessmentAgent(project_root).generate()


def load_legal_assessment(project_root: Path) -> dict[str, Any]:
    return read_json(project_root / "reports" / "legal_assessment_7_deadly_sins.json", {})


def load_legal_assessment_progress(project_root: Path) -> dict[str, Any]:
    return read_json(project_root / "reports" / "legal_assessment_7_deadly_sins.progress.json", {})
