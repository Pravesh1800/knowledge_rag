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

import searcher as searcher_module
from legal_assessment import (
    DEFAULT_AGENT_MAX_TOKENS,
    DEFAULT_MODEL,
    DEFAULT_SEARCH_MODEL,
    DEFAULT_WEB_MAX_TOKENS,
    DEFAULT_WEB_RESEARCH_MODEL,
    OPENROUTER_BASE_URL,
    compact_evidence,
    load_dotenv,
    merge_evidence,
    parse_json_response,
    read_json,
    topic_to_evidence,
    write_json,
)
from prebid_rules import (
    MANDATORY_AUDIT_CATEGORIES,
    PBQ_DUPLICATE_CONTROL_RULES,
    PBQ_GENERIC_EXAMPLES,
    PBQ_MANDATORY_QUERY_TEST,
    PBQ_PRIORITY_SCORING,
    PBQ_SEND_READY_RULES,
    PBQ_STRUCTURED_FIELDS,
    detect_prebid_flags,
)


MAX_AGENT_STEPS = 12
MAX_EVIDENCE = 24
MAX_QUERIES_PER_CATEGORY = int(os.getenv("OPENROUTER_PREBID_MAX_QUERIES_PER_CATEGORY", "10"))
MAX_MANUAL_COVERAGE_ADDITIONS = int(os.getenv("OPENROUTER_PREBID_MANUAL_COVERAGE_ADDITIONS", "6"))
DEFAULT_PREBID_WORKERS = 6
DEFAULT_PREBID_PARALLEL_RETRIES = 3
DEFAULT_PREBID_CONTEXT_CACHE = True
APP_ROOT = Path(__file__).resolve().parent
WATER_PBQ_PLAYBOOK_PATH = APP_ROOT / "playbooks" / "water_infrastructure_pbq_playbook.json"
ENGINEERING_PBQ_PLAYBOOK_PATH = APP_ROOT / "playbooks" / "water_infrastructure_engineering_pbq_playbook.json"


PBQ_STYLE_EXAMPLES = [
    {
        "clause_description": "Pre-qualification Criteria",
        "query": (
            "We request you to kindly allow Indian subsidiaries to use experience of Parent company for "
            "demonstrating the Technical and Financial capacity for claiming the experience as stated in "
            "Annexure C Pre-qualification. Please review and confirm."
        ),
    },
    {
        "clause_description": "Physical Requirement",
        "query": (
            "The two clauses are at variance. We understand that above shall be during last 7 financial "
            "years. Kindly review and confirm."
        ),
    },
    {
        "clause_description": "Drawings and DPR provided",
        "query": "Please provide AutoCAD versions file of all drawings provided along with DPR.",
    },
    {
        "clause_description": "Insurance for Works",
        "query": (
            "During the construction and Operation Phase, requirement of insurance coverages are not well "
            "defined in the tender document. Please elaborate and provide the requirement of Insurance "
            "cover for Design Build and Operation & Maintenance Period."
        ),
    },
]


def load_water_pbq_playbook() -> dict[str, Any]:
    playbook = read_json(WATER_PBQ_PLAYBOOK_PATH, {})
    engineering_playbook = read_json(ENGINEERING_PBQ_PLAYBOOK_PATH, {})
    if engineering_playbook:
        playbook["engineering_detail_playbook"] = engineering_playbook
    return playbook


def create_client() -> tuple[OpenAI, str, str]:
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
    agent_model = os.getenv(
        "OPENROUTER_PREBID_AGENT_MODEL",
        os.getenv("OPENROUTER_LEGAL_AGENT_MODEL", os.getenv("OPENROUTER_MODEL", DEFAULT_MODEL)),
    )
    web_model = os.getenv("OPENROUTER_WEB_RESEARCH_MODEL", DEFAULT_WEB_RESEARCH_MODEL)
    return client, agent_model, web_model


def normalize_text(value: str) -> str:
    replacements = {
        "â‚¬": "€",
        "â€“": "-",
        "â€”": "-",
        "â€™": "'",
        "â€˜": "'",
        "â€œ": '"',
        "â€": '"',
        "ï‚·": "-",
        "â€¢": "-",
    }
    for bad, good in replacements.items():
        value = value.replace(bad, good)
    return value


def normalize_payload(value: Any) -> Any:
    if isinstance(value, str):
        return normalize_text(value)
    if isinstance(value, list):
        return [normalize_payload(item) for item in value]
    if isinstance(value, dict):
        return {key: normalize_payload(item) for key, item in value.items()}
    return value


class PreBidQueryAgent:
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
        self.ensure_prebid_flags()
        self.playbook = load_water_pbq_playbook()
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
        self.client, self.model, self.web_research_model = create_client()
        self.logs: list[dict[str, Any]] = []
        self.log_lock = threading.Lock()
        self.search_lock = threading.Lock()
        self.progress_path = self.reports_dir / "prebid_queries.progress.json"

    def ensure_prebid_flags(self) -> None:
        changed = False
        for topic in self.topics:
            if "prebid_flags" in topic and "prebid_flag_reasons" in topic and "prebid_confidence" in topic:
                continue
            flags, reasons, confidence = detect_prebid_flags(topic)
            topic["prebid_flags"] = flags
            topic["prebid_flag_reasons"] = reasons
            topic["prebid_confidence"] = confidence
            changed = True
        if changed:
            write_json(self.indexes_dir / "topic_index.json", self.topics)

    def log(self, message: str, category_id: str | None = None, detail: dict[str, Any] | None = None) -> None:
        entry = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "category_id": category_id,
            "message": normalize_text(message),
            "detail": normalize_payload(detail or {}),
        }
        with self.log_lock:
            self.logs.append(entry)
            logs = list(self.logs)
            write_json(
                self.progress_path,
                {
                    "report_type": "prebid_queries",
                    "status": "running",
                    "project_id": self.project_root.name,
                    "updated_at": entry["created_at"],
                    "logs": logs,
                },
            )

    def flagged_evidence(self, flag_ids: list[str]) -> list[dict[str, Any]]:
        wanted = set(flag_ids)
        evidence = []
        for topic in self.topics:
            flags = set(topic.get("prebid_flags") or [])
            if not flags & wanted:
                continue
            reasons = topic.get("prebid_flag_reasons") or {}
            reason = "; ".join(reasons.get(flag, "") for flag in flags & wanted if reasons.get(flag))
            item = topic_to_evidence(topic, "flagged", reason or "Marked during indexing as possible PBQ issue evidence.")
            item["prebid_flags"] = topic.get("prebid_flags", [])
            item["prebid_confidence"] = topic.get("prebid_confidence", "low")
            evidence.append(item)
        return evidence

    def searched_evidence(self, query: str, max_hits: int = 14) -> list[dict[str, Any]]:
        # searcher.py uses module-level project paths, so protect this block while
        # category agents run in parallel.
        with self.search_lock:
            os.environ["PDF_VISION_RAG_ROOT"] = str(self.project_root)
            os.environ.setdefault("OPENROUTER_SEARCH_MODEL", os.getenv("OPENROUTER_SEARCH_MODEL", DEFAULT_SEARCH_MODEL))
            searcher_module.PROJECT_ROOT = self.project_root
            searcher_module.INDEXES_DIR = self.indexes_dir
            searcher_module.TOPIC_INDEX_PATH = self.indexes_dir / "topic_index.json"
            searcher_module.RELATIONSHIP_MAP_PATH = self.indexes_dir / "relationship_map.json"
            searcher_module.SEARCH_RESULTS_DIR = self.indexes_dir / "search_results"
            tree_searcher = searcher_module.TreeSearcher(query=query, dry_run=False, max_hits=max_hits)
            result = tree_searcher.search()
        by_triplet = {
            (topic.get("topic_name"), topic.get("document_name"), int(topic.get("page_no") or 0)): topic
            for topic in self.topics
        }
        evidence = []
        for hit in result.get("hits", []):
            topic = by_triplet.get((hit.get("topic_name"), hit.get("document_name"), int(hit.get("page_no") or 0)))
            if topic:
                evidence.append(topic_to_evidence(topic, "searched", hit.get("relevance_reason", "Found by targeted PBQ search.")))
            for related in hit.get("related_topics", []):
                related_topic = by_triplet.get((related.get("topic_name"), related.get("document_name"), int(related.get("page_no") or 0)))
                if related_topic:
                    evidence.append(topic_to_evidence(related_topic, "searched", related.get("relation", "Related topic found by PBQ search.")))
        return evidence

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
        return {
            "biome": biome,
            "communities": [
                self.community_lookup.get(name, {"community_name": name, "error": "missing"})
                for name in biome.get("community_names", [])
            ],
        }

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
                    "prebid_flags": topic.get("prebid_flags", []),
                    "content_excerpt": str(topic.get("content", ""))[:1400],
                }
                for topic in topics
            ],
        }

    def list_topics(self, query: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
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
                "prebid_flags": topic.get("prebid_flags", []),
            }
            for topic in rows[:limit]
        ]

    def get_topic(self, topic_name: str) -> dict[str, Any]:
        topic = self.topic_lookup.get(topic_name)
        if not topic:
            return {"error": f"Topic not found: {topic_name}"}
        return topic

    def keyword_lookup(self, terms: list[str], limit: int = 16) -> list[dict[str, Any]]:
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

    def web_research(self, query: str, purpose: str) -> dict[str, Any]:
        prompt = f"""
Research the public web only for external context. Do not use public web results as contractual proof.

Query:
{query}

Purpose:
{purpose}

Return JSON with summary, findings, citations, and limits.
""".strip()
        response = self.client.chat.completions.create(
            model=self.web_research_model,
            messages=[
                {"role": "system", "content": "You are an internet research assistant for bid clarification context. Cite URLs."},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            max_tokens=int(os.getenv("OPENROUTER_WEB_MAX_TOKENS", str(DEFAULT_WEB_MAX_TOKENS))),
        )
        content = response.choices[0].message.content or ""
        try:
            parsed = parse_json_response(content)
        except Exception:
            parsed = {"summary": content, "findings": [], "citations": [], "limits": "Model did not return parseable JSON."}
        return {"model": self.web_research_model, "query": query, "purpose": purpose, "result": parsed}

    def run_action(self, action: dict[str, Any], category: dict[str, Any]) -> dict[str, Any]:
        action_type = str(action.get("action", "")).strip()
        category_id = category["id"]
        if action_type == "search":
            query = str(action.get("query", "")).strip() or category["goal"]
            self.log(f"Running an extra PBQ project search for: {query}", category_id)
            return {"evidence": compact_evidence(self.searched_evidence(query), limit=MAX_EVIDENCE)}
        if action_type == "keyword_lookup":
            terms = [str(term) for term in action.get("terms", []) if str(term).strip()]
            self.log(f"Scanning all topics for PBQ gap terms: {', '.join(terms[:8])}", category_id)
            return {"evidence": compact_evidence(self.keyword_lookup(terms), limit=MAX_EVIDENCE)}
        if action_type == "list_biomes":
            self.log("Listing all biomes to make sure the PBQ audit covers the full project map.", category_id)
            return {"biomes": self.list_biomes()}
        if action_type == "get_biome":
            name = str(action.get("biome_name", "")).strip()
            self.log(f"Opening biome for PBQ audit: {name}", category_id)
            return self.get_biome(name)
        if action_type == "list_communities":
            document_name = str(action.get("document_name", "")).strip() or None
            self.log(f"Listing communities{f' for {document_name}' if document_name else ''}.", category_id)
            return {"communities": self.list_communities(document_name)}
        if action_type == "get_community":
            name = str(action.get("community_name", "")).strip()
            self.log(f"Opening community and topics for PBQ audit: {name}", category_id)
            return self.get_community(name)
        if action_type == "list_topics":
            query = str(action.get("query", "")).strip() or None
            self.log(f"Listing topics{f' matching: {query}' if query else ''}.", category_id)
            return {"topics": self.list_topics(query=query, limit=int(action.get("limit") or 50))}
        if action_type == "get_topic":
            name = str(action.get("topic_name", "")).strip()
            self.log(f"Opening full topic content for query drafting: {name}", category_id)
            topic = self.get_topic(name)
            if "error" not in topic:
                return {"topic": topic, "evidence": compact_evidence([topic_to_evidence(topic, "agent_inspected", "PBQ specialist opened full topic content.")], limit=1)}
            return topic
        if action_type == "web_research":
            query = str(action.get("query", "")).strip()
            purpose = str(action.get("purpose", "")).strip() or category["goal"]
            self.log(f"Researching public context with Sonar Reasoning Pro: {query}", category_id)
            return {"web_research": self.web_research(query, purpose)}
        if action_type == "answer":
            return {"answer": action}
        self.log("PBQ specialist requested an unknown action.", category_id, {"action": action})
        return {"error": "unknown action"}

    def starter_query(self, category: dict[str, Any]) -> str:
        return (
            f"{category['title']}. {category['goal']} "
            "Run both deep risk discovery and manual tender-engineer PBQ coverage. "
            "Find missing information, contradictions, unclear risk allocation, missing BOQ, undefined responsibility, "
            "pricing/schedule impact, standard bidder clarifications, practical design/construction/O&M gaps, "
            "and clauses requiring bidder query wording."
        )

    def playbook_context(self) -> dict[str, Any]:
        return self.playbook

    def context_cache_enabled(self) -> bool:
        value = os.getenv("OPENROUTER_PREBID_CONTEXT_CACHE", "1" if DEFAULT_PREBID_CONTEXT_CACHE else "0")
        return value.strip().lower() not in {"0", "false", "no", "off"}

    def cache_session_id(self) -> str:
        engineering = self.playbook.get("engineering_detail_playbook", {})
        parts = [
            "prebid",
            self.project_root.name,
            str(self.playbook.get("playbook_id", "water_infrastructure_pbq_playbook")),
            str(self.playbook.get("version", "unknown")),
            str(engineering.get("playbook_id", "engineering")),
            str(engineering.get("version", "unknown")),
        ]
        return "-".join(re.sub(r"[^a-zA-Z0-9_.:-]+", "-", part).strip("-") for part in parts if part)[:256]

    def cached_context_block(self) -> str:
        return f"""
Stable reusable PBQ context. This block is intentionally identical across PBQ category calls for prompt caching.

Full reusable generalized water-infrastructure PBQ playbook:
{json.dumps(self.playbook_context(), ensure_ascii=False)}

PBQ wording examples to imitate:
{json.dumps(PBQ_STYLE_EXAMPLES, ensure_ascii=False)}

Mandatory query test:
{json.dumps(PBQ_MANDATORY_QUERY_TEST, ensure_ascii=False)}

Priority scoring:
{json.dumps(PBQ_PRIORITY_SCORING, ensure_ascii=False)}

Generic good/bad/edge examples:
{json.dumps(PBQ_GENERIC_EXAMPLES, ensure_ascii=False)}

Send-ready wording rules:
{json.dumps(PBQ_SEND_READY_RULES, ensure_ascii=False)}

Structured fields:
{json.dumps(PBQ_STRUCTURED_FIELDS, ensure_ascii=False)}

Rules for using this cached context:
- Read the full reusable playbook before drafting, verifying, or adding manual coverage rows.
- Use the playbook as bidder intelligence and a coverage checklist, not as project fact.
- Do not copy or infer any project-specific fact from the playbook.
- Generate rows only when supported by project/tender evidence or a material absence after search.
- Prefer concrete document/page/clause references from evidence; use "-" when unknown instead of inventing.
""".strip()

    def specialist_prompt(self, category: dict[str, Any], evidence: list[dict[str, Any]], step_results: list[dict[str, Any]]) -> str:
        return f"""
You are a dedicated Pre-Bid Query Document Generator for the bidder.
Your job is to create PBQ-style bidder queries for one mandatory audit category.

Audit category:
{json.dumps(category, ensure_ascii=False)}

Use the cached reusable PBQ context supplied before this dynamic request.

Evidence gathered from flagged topics, fresh search, and inspected project map:
{json.dumps(compact_evidence(evidence, limit=MAX_EVIDENCE), ensure_ascii=False)}

Previous tool results:
{json.dumps(step_results[-8:], ensure_ascii=False)}

Available actions. Return exactly one JSON object:
1. {{"action":"search","query":"specific project search query"}}
2. {{"action":"keyword_lookup","terms":["exact term", "another term"]}}
3. {{"action":"list_biomes"}}
4. {{"action":"get_biome","biome_name":"exact biome name"}}
5. {{"action":"list_communities","document_name":"optional document name filter"}}
6. {{"action":"get_community","community_name":"exact community name"}}
7. {{"action":"list_topics","query":"optional keyword query","limit":50}}
8. {{"action":"get_topic","topic_name":"exact topic name"}}
9. {{"action":"web_research","query":"public web query","purpose":"why external context is needed"}}
10. {{
  "action":"answer",
  "queries":[
    {{
      "priority":"Critical|High|Medium|Low",
      "impact_area":"pricing|schedule|design|construction|compliance|O&M|risk|eligibility|payment|approvals|other",
      "action_requested":"provide|confirm|clarify|revise|relax|include in BOQ|define responsibility|amend clause|other",
      "evidence_strength":"direct|inferred|absence_after_search|weak",
      "tender_vol_section":"section/volume if known, else '-'",
      "page_no":"page number or '-'",
      "clause_no":"clause number if known, else '-'",
      "tender_reference":"combined tender reference if known, else '-'",
      "issue_summary":"one-line gap/conflict/ambiguity summary",
      "clause_description":"short clause/scope description",
      "bidder_query":"PBQ-style query text ready to send to the employer",
      "basis":"why this query is mandatory for pricing/risk/schedule/compliance",
      "duplicate_group":"stable short label for similar queries, else '-'",
      "citations":[{{"document_name":"","page_no":0,"topic_name":"","excerpt":""}}],
      "category":"{category['title']}"
    }}
  ],
  "coverage_note":"what was checked and what was not found"
}}

Rules:
- Apply the mandatory query test before including any row. If a candidate fails the test, omit it.
- Read the full reusable playbook before drafting. Use it as bidder intelligence and coverage checklist, not as project fact.
- Run two mental passes before answering: (1) deep commercial/legal/risk discovery and (2) manual tender-engineer PBQ coverage.
- Do not filter out practical Medium-value clarification rows when they affect pricing, quantity, drawings, scope split, BOQ inclusion, design basis, site access, O&M, approval responsibility, or cash flow.
- Use generic examples as quality calibration only; do not copy them as project facts.
- Go through this category deeply. Search more if evidence is thin.
- Generate queries only when there is a real tender/document basis or a material missing item.
- Do not summarize; create the actual bidder query document rows.
- Wording should sound like the provided PBQ examples: "Please provide...", "We understand that...", "Kindly review and confirm.", "We request..."
- Each query must close a gap, conflict, missing document/data, unclear responsibility, impossible condition, risk allocation issue, or pricing/schedule ambiguity.
- Each query must ask for one concrete employer action and state why it matters.
- Same broad topic with different employer actions can be separate rows; deduplicate only when the required clarification/action is materially the same.
- Assign priority from the priority scoring rules, not from wording intensity.
- Do not include weak "please clarify" rows unless the exact ambiguity and requested action are stated.
- Prefer concrete clause/page fields, but use "-" when not available.
- Do not use web research as contractual proof.
- Return no more than {MAX_QUERIES_PER_CATEGORY} queries for this category; choose the mandatory/high-impact rows plus practical manual-PBQ rows that the full playbook says a bidder normally needs before pricing/submission.
""".strip()

    def verifier_prompt(self, category: dict[str, Any], draft: dict[str, Any], evidence: list[dict[str, Any]], step_results: list[dict[str, Any]]) -> str:
        return f"""
Verify proposed PBQ rows for one category.

Category:
{json.dumps(category, ensure_ascii=False)}

Draft:
{json.dumps(draft, ensure_ascii=False)}

Evidence:
{json.dumps(compact_evidence(evidence, limit=MAX_EVIDENCE), ensure_ascii=False)}

Tool results:
{json.dumps(step_results[-8:], ensure_ascii=False)}

Use the cached reusable PBQ context supplied before this dynamic request.

Mandatory query test:
{json.dumps(PBQ_MANDATORY_QUERY_TEST, ensure_ascii=False)}

Duplicate control rules:
{json.dumps(PBQ_DUPLICATE_CONTROL_RULES, ensure_ascii=False)}

Send-ready wording rules:
{json.dumps(PBQ_SEND_READY_RULES, ensure_ascii=False)}

Return JSON:
{{
  "verified": true,
  "queries": [
    {{
      "priority":"Critical|High|Medium|Low",
      "impact_area":"pricing|schedule|design|construction|compliance|O&M|risk|eligibility|payment|approvals|other",
      "action_requested":"provide|confirm|clarify|revise|relax|include in BOQ|define responsibility|amend clause|other",
      "evidence_strength":"direct|inferred|absence_after_search|weak",
      "tender_vol_section":"section/volume if known, else '-'",
      "page_no":"page number or '-'",
      "clause_no":"clause number if known, else '-'",
      "tender_reference":"combined tender reference if known, else '-'",
      "issue_summary":"one-line gap/conflict/ambiguity summary",
      "clause_description":"short clause/scope description",
      "bidder_query":"final PBQ-style query text",
      "basis":"why this query is needed",
      "duplicate_group":"stable short label for similar queries, else '-'",
      "citations":[{{"document_name":"","page_no":0,"topic_name":"","excerpt":""}}],
      "category":"{category['title']}"
    }}
  ],
  "warnings":["warning if any"],
  "verification_note":"what you checked"
}}

Rules:
- Remove duplicate, vague, unsupported, or low-value queries.
- Apply the mandatory query test. Remove any row that does not pass.
- Apply duplicate control rules and consolidate similar rows.
- Keep rows that are mandatory to clarify before pricing/submission.
- Also keep practical Medium-value manual-PBQ rows when the playbook indicates the information is normally required to price, design, construct, operate, insure, schedule, or legally accept a water infrastructure tender.
- Do not downgrade a row only because it is a standard bidder clarification; standard missing drawings, BOQ inclusions, site constraints, O&M handover, permissions, payment mechanics, and industry-norm relaxations are valid when supported by project evidence or material absence.
- Same broad topic with different requested employer action should not be collapsed into one row.
- Make bidder_query send-ready and specific.
- Ensure action_requested matches the bidder_query.
- Ensure priority matches impact_area and basis.
- Keep basis concise and evidence-linked.
- Do not invent page or clause numbers.
""".strip()

    def manual_coverage_gap_prompt(
        self,
        category: dict[str, Any],
        verified_queries: list[dict[str, Any]],
        evidence: list[dict[str, Any]],
        step_results: list[dict[str, Any]],
    ) -> str:
        return f"""
You are performing a second-pass manual PBQ coverage review for a water-infrastructure tender.

Purpose:
- The first pass finds deep/high-impact risks.
- This pass catches practical tender-engineer clarifications similar to a manual PBQ register.
- Use only generalized playbook patterns. Do not copy or infer any project-specific fact from the playbook.

Category:
{json.dumps(category, ensure_ascii=False)}

Already verified query rows for this category:
{json.dumps(verified_queries, ensure_ascii=False)}

Use the cached reusable PBQ context supplied before this dynamic request.

Evidence:
{json.dumps(compact_evidence(evidence, limit=MAX_EVIDENCE), ensure_ascii=False)}

Tool results:
{json.dumps(step_results[-8:], ensure_ascii=False)}

Return JSON:
{{
  "additional_queries": [
    {{
      "priority":"Critical|High|Medium|Low",
      "impact_area":"pricing|schedule|design|construction|compliance|O&M|risk|eligibility|payment|approvals|other",
      "action_requested":"provide|confirm|clarify|revise|relax|include in BOQ|define responsibility|amend clause|other",
      "evidence_strength":"direct|inferred|absence_after_search|weak",
      "tender_vol_section":"section/volume if known, else '-'",
      "page_no":"page number or '-'",
      "clause_no":"clause number if known, else '-'",
      "tender_reference":"combined tender reference if known, else '-'",
      "issue_summary":"one-line gap/conflict/ambiguity summary",
      "clause_description":"short clause/scope description",
      "bidder_query":"final PBQ-style query text",
      "basis":"why this query is needed",
      "duplicate_group":"stable short label for similar queries, else '-'",
      "citations":[{{"document_name":"","page_no":0,"topic_name":"","excerpt":""}}],
      "category":"{category['title']}"
    }}
  ],
  "coverage_review_note":"which generalized checklist items were checked and why additional rows were or were not needed"
}}

Rules:
- Add only rows missing from the already verified queries.
- Do not add a row unless it is supported by tender/project evidence or a material absence after search.
- Keep practical Medium rows when a bidder would normally need the answer to price, design, construct, operate, insure, schedule, or accept the work.
- Examples of valid manual-coverage gaps: missing drawings, missing dimensions, BOQ inclusion, quantity/location clarification, site/access/utility responsibility, payment schedule mechanics, O&M handover, permissions, standard industry relaxation, typo/conflict, and scope confirmation.
- Do not repeat a row if the requested employer action is materially already present.
- Return no more than {MAX_MANUAL_COVERAGE_ADDITIONS} additional queries.
""".strip()

    def call_json(self, prompt: str, system: str) -> dict[str, Any]:
        max_tokens = int(os.getenv("OPENROUTER_PREBID_AGENT_MAX_TOKENS", os.getenv("OPENROUTER_AGENT_MAX_TOKENS", str(DEFAULT_AGENT_MAX_TOKENS))))
        response = self.cached_chat_completion(prompt, system, max_tokens)
        content = response.choices[0].message.content or "{}"
        try:
            return parse_json_response(content)
        except Exception as first_error:
            self.log(
                "PBQ JSON response was malformed; asking model to repair the same output.",
                detail={"error": str(first_error)},
            )
            return self.repair_json(content, str(first_error), max_tokens)

    def cached_chat_completion(self, prompt: str, system: str, max_tokens: int):
        cache_enabled = self.context_cache_enabled()
        user_content: Any
        extra_body: dict[str, Any] = {"session_id": self.cache_session_id()}
        if cache_enabled:
            cache_control: dict[str, str] = {"type": "ephemeral"}
            cache_ttl = os.getenv("OPENROUTER_PREBID_CACHE_TTL", "").strip()
            if cache_ttl:
                cache_control["ttl"] = cache_ttl
            user_content = [
                {
                    "type": "text",
                    "text": self.cached_context_block(),
                    "cache_control": cache_control,
                },
                {
                    "type": "text",
                    "text": prompt,
                },
            ]
        else:
            user_content = self.cached_context_block() + "\n\nDynamic request:\n" + prompt

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user_content},
            ],
            "temperature": 0,
            "response_format": {"type": "json_object"},
            "max_tokens": max_tokens,
            "extra_body": extra_body,
        }
        return self.client.chat.completions.create(**kwargs)

    def repair_json(self, malformed_content: str, parser_error: str, max_tokens: int) -> dict[str, Any]:
        repair_content = malformed_content
        repair_error = parser_error
        for _ in range(3):
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
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a JSON repair engine. Return only one valid JSON object."},
                    {"role": "user", "content": repair_prompt},
                ],
                temperature=0,
                response_format={"type": "json_object"},
                max_tokens=max_tokens,
                extra_body={"session_id": self.cache_session_id()},
            )
            repair_content = response.choices[0].message.content or "{}"
            try:
                return parse_json_response(repair_content)
            except Exception as exc:
                repair_error = str(exc)
        raise ValueError(f"Model returned malformed JSON and repair failed: {repair_error}")

    def audit_category(self, category: dict[str, Any]) -> dict[str, Any]:
        self.log(f"Starting mandatory PBQ audit category: {category['title']}", category["id"])
        flagged = self.flagged_evidence(category["flag_ids"])
        self.log(f"Collected {len(flagged)} pre-flagged PBQ candidate topic(s).", category["id"])
        searched = self.searched_evidence(self.starter_query(category))
        self.log(f"Ran required fresh PBQ search and collected {len(searched)} searched topic(s).", category["id"])
        evidence = merge_evidence(flagged + searched)
        self.log(f"Merged and deduplicated PBQ evidence to {len(evidence)} candidate topic/page item(s).", category["id"])

        step_results: list[dict[str, Any]] = []
        draft: dict[str, Any] | None = None
        for step in range(1, MAX_AGENT_STEPS + 1):
            self.log(f"PBQ specialist thinking step {step}: checking if more context is mandatory.", category["id"])
            action = self.call_json(
                self.specialist_prompt(category, evidence, step_results),
                "Return only valid JSON. You are a strict pre-bid query generator.",
            )
            self.log(f"PBQ specialist requested tool action: {action.get('action', 'unknown')}.", category["id"], {"action": action})
            result = self.run_action(action, category)
            self.log(f"Tool action completed: {action.get('action', 'unknown')}.", category["id"], {"result_keys": sorted(result.keys())})
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
                self.log(f"PBQ specialist drafted {len(draft.get('queries', []))} query row(s) for {category['title']}.", category["id"])
                break

        if draft is None:
            self.log("PBQ specialist reached step limit; forcing query rows from accumulated evidence.", category["id"])
            draft = self.call_json(
                self.specialist_prompt(category, evidence, step_results)
                + "\n\nYou must now return action=answer using the accumulated evidence.",
                "Return only valid JSON.",
            )

        verifier = self.call_json(
            self.verifier_prompt(category, draft, evidence, step_results),
            "Return only valid JSON. You verify pre-bid query rows.",
        )
        queries = verifier.get("queries")
        if not isinstance(queries, list):
            queries = draft.get("queries", [])
        self.log(
            f"Verifier finalized {len(queries)} PBQ row(s) for {category['title']}.",
            category["id"],
            {"warnings": verifier.get("warnings", [])},
        )
        coverage_review = self.call_json(
            self.manual_coverage_gap_prompt(category, queries, evidence, step_results),
            "Return only valid JSON. You find missing practical manual PBQ coverage rows.",
        )
        additions = coverage_review.get("additional_queries", [])
        if not isinstance(additions, list):
            additions = []
        self.log(
            f"Manual PBQ coverage review proposed {len(additions)} additional row(s) for {category['title']}.",
            category["id"],
            {"coverage_review_note": coverage_review.get("coverage_review_note", "")},
        )
        if additions:
            combined_draft = {
                "queries": queries + additions,
                "coverage_note": coverage_review.get("coverage_review_note", ""),
            }
            combined_verifier = self.call_json(
                self.verifier_prompt(category, combined_draft, evidence, step_results),
                "Return only valid JSON. You verify combined deep-risk and manual PBQ coverage rows.",
            )
            combined_queries = combined_verifier.get("queries")
            if isinstance(combined_queries, list):
                queries = combined_queries
                verifier = combined_verifier
            self.log(
                f"Combined verifier finalized {len(queries)} PBQ row(s) for {category['title']} after manual coverage pass.",
                category["id"],
                {"warnings": verifier.get("warnings", [])},
            )
        return normalize_payload(
            {
                "category_id": category["id"],
                "title": category["title"],
                "queries": queries,
                "evidence": compact_evidence(evidence, limit=MAX_EVIDENCE),
                "agent_steps": step_results,
                "verifier": verifier,
                "manual_coverage_review": coverage_review,
            }
        )

    def audit_category_with_retries(self, category: dict[str, Any]) -> dict[str, Any]:
        attempts = max(1, int(os.getenv("OPENROUTER_PREBID_CATEGORY_RETRIES", str(DEFAULT_PREBID_PARALLEL_RETRIES))))
        last_error: Exception | None = None
        for attempt in range(1, attempts + 1):
            try:
                if attempt > 1:
                    self.log(f"Retrying PBQ audit category attempt {attempt}/{attempts}.", category["id"])
                return self.audit_category(category)
            except Exception as exc:
                last_error = exc
                self.log(
                    f"PBQ audit category attempt {attempt}/{attempts} failed: {exc}",
                    category["id"],
                    {"error": str(exc)},
                )
        assert last_error is not None
        raise last_error

    @staticmethod
    def query_key(row: dict[str, Any]) -> str:
        text = " ".join(
            str(row.get(key, ""))
            for key in [
                "tender_vol_section",
                "page_no",
                "clause_no",
                "tender_reference",
                "issue_summary",
                "clause_description",
                "action_requested",
                "bidder_query",
            ]
        ).lower()
        return re.sub(r"[^a-z0-9]+", " ", text).strip()

    def final_rows(self, sections: list[dict[str, Any]]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        seen: set[str] = set()
        priority_rank = {"Critical": 0, "High": 1, "Medium": 2, "Low": 3}
        for section in sections:
            for query in section.get("queries", []):
                if not isinstance(query, dict):
                    continue
                key = self.query_key(query)
                if not key or key in seen:
                    continue
                seen.add(key)
                query.setdefault("category", section.get("title", ""))
                query.setdefault("priority", "Medium")
                query.setdefault("impact_area", "-")
                query.setdefault("action_requested", "-")
                query.setdefault("evidence_strength", "-")
                query.setdefault("tender_reference", "-")
                query.setdefault("issue_summary", query.get("clause_description", ""))
                query.setdefault("duplicate_group", "-")
                rows.append(query)
        rows.sort(key=lambda row: (priority_rank.get(str(row.get("priority")), 2), str(row.get("category", ""))))
        for index, row in enumerate(rows, start=1):
            row["s_no"] = index
        return rows

    def generate(self) -> dict[str, Any]:
        self.log("Starting Pre-Bid Query document generation.")
        workers = max(1, int(os.getenv("OPENROUTER_PREBID_CATEGORY_WORKERS", str(DEFAULT_PREBID_WORKERS))))
        workers = min(workers, len(MANDATORY_AUDIT_CATEGORIES))
        self.log(f"Mandatory audit will cover all configured gap categories using {workers} parallel category worker(s) before creating the final PBQ table.")
        sections_by_id: dict[str, dict[str, Any]] = {}
        failures: list[dict[str, str]] = []
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(self.audit_category_with_retries, category): category
                for category in MANDATORY_AUDIT_CATEGORIES
            }
            for future in as_completed(futures):
                category = futures[future]
                try:
                    sections_by_id[category["id"]] = future.result()
                except Exception as exc:
                    failures.append({"category_id": category["id"], "title": category["title"], "error": str(exc)})
                    self.log(f"PBQ audit category failed after retries: {category['title']}: {exc}", category["id"])
        if failures:
            write_json(
                self.progress_path,
                {
                    "report_type": "prebid_queries",
                    "status": "failed",
                    "project_id": self.project_root.name,
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                    "failures": failures,
                    "logs": self.logs,
                },
            )
            raise RuntimeError(f"Pre-Bid Query generation failed for {len(failures)} category/categories: {failures}")
        sections = [
            sections_by_id[category["id"]]
            for category in MANDATORY_AUDIT_CATEGORIES
            if category["id"] in sections_by_id
        ]
        rows = self.final_rows(sections)
        report = normalize_payload(
            {
                "report_type": "prebid_queries",
                "status": "verified",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "project_id": self.project_root.name,
                "title": "Pre-Bid Queries",
                "generation_settings": {
                    "playbook_id": self.playbook.get("playbook_id", "-"),
                    "playbook_version": self.playbook.get("version", "-"),
                    "playbook_scope": self.playbook.get("derived_from", "General reusable PBQ playbook."),
                    "engineering_playbook_id": self.playbook.get("engineering_detail_playbook", {}).get("playbook_id", "-"),
                    "engineering_playbook_version": self.playbook.get("engineering_detail_playbook", {}).get("version", "-"),
                    "full_playbook_ingestion": True,
                    "context_cache_enabled": self.context_cache_enabled(),
                    "context_cache_session_id": self.cache_session_id(),
                    "context_cache_ttl": os.getenv("OPENROUTER_PREBID_CACHE_TTL", "provider default"),
                    "max_queries_per_category": MAX_QUERIES_PER_CATEGORY,
                    "max_manual_coverage_additions_per_category": MAX_MANUAL_COVERAGE_ADDITIONS,
                    "category_workers": workers,
                    "manual_coverage_gap_pass": True,
                    "generation_modes": self.playbook.get("generation_modes", []),
                },
                "columns": [
                    "S. No.",
                    "Priority",
                    "Impact Area",
                    "Action Requested",
                    "Evidence Strength",
                    "Tender Vol. / Section",
                    "Page No.",
                    "Clause No.",
                    "Tender Reference",
                    "Issue Summary",
                    "Clause Description",
                    "Bidder's Query",
                    "Basis / Why This Is Needed",
                ],
                "rows": rows,
                "sections": sections,
                "logs": self.logs,
            }
        )
        self.logs = report["logs"]
        output_path = self.reports_dir / "prebid_queries.json"
        write_json(output_path, report)
        self.log(f"Saved Pre-Bid Query document to {output_path.name}.")
        report["logs"] = self.logs
        write_json(output_path, report)
        write_json(
            self.progress_path,
            {
                "report_type": "prebid_queries",
                "status": "complete",
                "project_id": self.project_root.name,
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "logs": self.logs,
            },
        )
        return report


def generate_prebid_queries(project_root: Path) -> dict[str, Any]:
    return PreBidQueryAgent(project_root).generate()


def load_prebid_queries(project_root: Path) -> dict[str, Any]:
    return read_json(project_root / "reports" / "prebid_queries.json", {})


def load_prebid_queries_progress(project_root: Path) -> dict[str, Any]:
    return read_json(project_root / "reports" / "prebid_queries.progress.json", {})


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate the Pre-Bid Query report for a project.")
    parser.add_argument("project_root", type=Path, help="Path to the indexed project directory.")
    args = parser.parse_args()
    result = generate_prebid_queries(args.project_root)
    print(json.dumps({"status": result.get("status", "complete"), "rows": len(result.get("rows", []))}, ensure_ascii=False))
