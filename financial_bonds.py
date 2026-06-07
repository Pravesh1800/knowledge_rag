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
from financial_rules import FINANCIAL_EXTRACTION_SCHEMA, FINANCIAL_ROWS, detect_financial_flags
from legal_assessment import (
    DEFAULT_AGENT_MAX_TOKENS,
    DEFAULT_MODEL,
    DEFAULT_SEARCH_MODEL,
    DEFAULT_WEB_RESEARCH_MODEL,
    DEFAULT_WEB_MAX_TOKENS,
    OPENROUTER_BASE_URL,
    compact_evidence,
    load_dotenv,
    json_completion,
    merge_evidence,
    parse_json_response,
    read_json,
    topic_to_evidence,
    write_json,
)


MAX_AGENT_STEPS = 14
MAX_EVIDENCE = 22
DEFAULT_FINANCIAL_WORKERS = 6
DEFAULT_FINANCIAL_PARALLEL_RETRIES = 3


TEXT_REPLACEMENTS = {
    "Mâ\x82¬": "M€",
    "â\x82¬": "€",
    "â\x80\x93": "-",
    "â\x80\x94": "-",
    "â\x80\x99": "'",
    "â\x80\x98": "'",
    "â\x80\x9c": '"',
    "â\x80\x9d": '"',
    "ï\x82\x97": "-",
    "â\x80¢": "-",
}


def normalize_text(value: str) -> str:
    for bad, good in TEXT_REPLACEMENTS.items():
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
        "OPENROUTER_FINANCIAL_AGENT_MODEL",
        os.getenv("OPENROUTER_LEGAL_AGENT_MODEL", os.getenv("OPENROUTER_MODEL", DEFAULT_MODEL)),
    )
    web_model = os.getenv("OPENROUTER_WEB_RESEARCH_MODEL", DEFAULT_WEB_RESEARCH_MODEL)
    return client, agent_model, web_model


class FinancialBondsAgent:
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
        self.ensure_financial_flags()
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
        self.progress_path = self.reports_dir / "financial_bonds.progress.json"

    def ensure_financial_flags(self) -> None:
        changed = False
        for topic in self.topics:
            if "financial_flags" in topic and "financial_flag_reasons" in topic and "financial_confidence" in topic:
                continue
            flags, reasons, confidence = detect_financial_flags(topic)
            topic["financial_flags"] = flags
            topic["financial_flag_reasons"] = reasons
            topic["financial_confidence"] = confidence
            changed = True
        if changed:
            write_json(self.indexes_dir / "topic_index.json", self.topics)

    def log(self, message: str, row_id: str | None = None, detail: dict[str, Any] | None = None) -> None:
        entry = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "row_id": row_id,
            "message": normalize_text(message),
            "detail": normalize_payload(detail or {}),
        }
        with self.log_lock:
            self.logs.append(entry)
            logs = list(self.logs)
            write_json(
                self.progress_path,
                {
                    "report_type": "financial_bonds",
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
            flags = set(topic.get("financial_flags") or [])
            if not flags & wanted:
                continue
            reasons = topic.get("financial_flag_reasons") or {}
            reason = "; ".join(reasons.get(flag, "") for flag in flags & wanted if reasons.get(flag))
            evidence.append(topic_to_evidence(topic, "flagged", reason or "Marked during indexing as financially relevant."))
        return evidence

    def searched_evidence(self, query: str, max_hits: int = 12) -> list[dict[str, Any]]:
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
                evidence.append(topic_to_evidence(topic, "searched", hit.get("relevance_reason", "Found by targeted search.")))
            for related in hit.get("related_topics", []):
                related_topic = by_triplet.get((related.get("topic_name"), related.get("document_name"), int(related.get("page_no") or 0)))
                if related_topic:
                    evidence.append(topic_to_evidence(related_topic, "searched", related.get("relation", "Related topic found by search.")))
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
                    "financial_flags": topic.get("financial_flags", []),
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
                "financial_flags": topic.get("financial_flags", []),
            }
            for topic in rows[:limit]
        ]

    def get_topic(self, topic_name: str) -> dict[str, Any]:
        topic = self.topic_lookup.get(topic_name)
        if not topic:
            return {"error": f"Topic not found: {topic_name}"}
        return topic

    def keyword_lookup(self, terms: list[str], limit: int = 12) -> list[dict[str, Any]]:
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
Research the public web only for contextual support. Do not use public web results as contractual proof.

Query:
{query}

Purpose:
{purpose}

Return JSON with summary, findings, citations, and limits.
""".strip()
        response = self.client.chat.completions.create(
            model=self.web_research_model,
            messages=[
                {"role": "system", "content": "You are an internet research assistant for bid finance context. Cite URLs."},
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

    def run_action(self, action: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
        action_type = str(action.get("action", "")).strip()
        if action_type == "search":
            query = str(action.get("query", "")).strip() or row["topic"]
            self.log(f"Running an extra project search for: {query}", row["id"])
            return {"evidence": compact_evidence(self.searched_evidence(query), limit=MAX_EVIDENCE)}
        if action_type == "keyword_lookup":
            terms = [str(term) for term in action.get("terms", []) if str(term).strip()]
            self.log(f"Scanning all indexed topics for exact financial terms: {', '.join(terms[:8])}", row["id"])
            return {"evidence": compact_evidence(self.keyword_lookup(terms), limit=MAX_EVIDENCE)}
        if action_type == "list_biomes":
            self.log("Listing all biomes for financial bond context.", row["id"])
            return {"biomes": self.list_biomes()}
        if action_type == "get_biome":
            name = str(action.get("biome_name", "")).strip()
            self.log(f"Opening biome: {name}", row["id"])
            return self.get_biome(name)
        if action_type == "list_communities":
            document_name = str(action.get("document_name", "")).strip() or None
            self.log(f"Listing communities{f' for {document_name}' if document_name else ''}.", row["id"])
            return {"communities": self.list_communities(document_name)}
        if action_type == "get_community":
            name = str(action.get("community_name", "")).strip()
            self.log(f"Opening community and its topics: {name}", row["id"])
            return self.get_community(name)
        if action_type == "list_topics":
            query = str(action.get("query", "")).strip() or None
            self.log(f"Listing topics{f' matching: {query}' if query else ''}.", row["id"])
            return {"topics": self.list_topics(query=query, limit=int(action.get("limit") or 40))}
        if action_type == "get_topic":
            name = str(action.get("topic_name", "")).strip()
            self.log(f"Opening full topic content: {name}", row["id"])
            topic = self.get_topic(name)
            if "error" not in topic:
                return {"topic": topic, "evidence": compact_evidence([topic_to_evidence(topic, "agent_inspected", "Financial specialist opened full topic content.")], limit=1)}
            return topic
        if action_type == "web_research":
            query = str(action.get("query", "")).strip()
            purpose = str(action.get("purpose", "")).strip() or row["goal"]
            self.log(f"Researching the public web with Sonar Reasoning Pro: {query}", row["id"])
            return {"web_research": self.web_research(query, purpose)}
        if action_type == "answer":
            return {"answer": action}
        self.log("Financial specialist requested an unknown action.", row["id"], {"action": action})
        return {"error": "unknown action"}

    def specialist_prompt(self, row: dict[str, Any], evidence: list[dict[str, Any]], step_results: list[dict[str, Any]]) -> str:
        return f"""
You are a specialist financial-bonds agent for SUEZ bid review.
Your only job is to answer one row of the Financial Bonds table.

Row:
{json.dumps(row, ensure_ascii=False)}

How to interpret this row:
- description: what this financial item means.
- extract_fields: fields that matter most for this row.
- not_enough: common false-positive evidence that must not be used for this row.
- examples: generic calibration examples; do not require exact example wording.
- goal: final extraction objective.

Structured extraction schema:
{json.dumps(FINANCIAL_EXTRACTION_SCHEMA, ensure_ascii=False)}

Evidence:
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
7. {{"action":"list_topics","query":"optional keyword query","limit":40}}
8. {{"action":"get_topic","topic_name":"exact topic name"}}
9. {{"action":"web_research","query":"public web query","purpose":"why external context is needed"}}
10. {{"action":"answer","extraction":{{"required_status":"required|not_required|not_found|unclear","amount":"","percentage":"","basis":"","instrument":"","cash_or_bg":"","validity":"","recovery":"","release_condition":"","conditions":"","exact_clause_excerpt":"","not_found_basis":""}},"comment":"concise table comment with exact amount/percentage/validity/recovery/none status","citations":[{{"document_name":"","page_no":0,"topic_name":"","excerpt":""}}],"confidence":"low|medium|high","notes":"short audit note"}}

Rules:
- First understand description, extract_fields, not_enough, examples, and goal.
- Use examples only as generic calibration, not project facts.
- Use tender/project evidence as contractual truth.
- Do not use web research as proof of a bond/security requirement.
- Exact values matter: INR amounts, percentages, CV basis, validity, recovery, cash vs BG, None/Not Required.
- Do not mix rows. Bid security, advance payment BG, DB performance security, O&M performance security, retention, security deposit, and parent company guarantee are different instruments unless the tender expressly combines them.
- Fill the structured extraction before writing the comment.
- Use "not_required" only when the tender expressly says none/not required. Use "not_found" when you searched but found no clause.
- If evidence matches only the row's not_enough warning, do not use it as the answer.
- If the row is None or Not Required, cite the evidence or searched basis.
- Continue iterating until the row is clear enough to answer.
""".strip()

    def verifier_prompt(self, row: dict[str, Any], draft: dict[str, Any], evidence: list[dict[str, Any]], step_results: list[dict[str, Any]]) -> str:
        return f"""
Verify one Financial Bonds table row.

Row:
{json.dumps(row, ensure_ascii=False)}

Structured extraction schema:
{json.dumps(FINANCIAL_EXTRACTION_SCHEMA, ensure_ascii=False)}

Draft:
{json.dumps(draft, ensure_ascii=False)}

Evidence:
{json.dumps(compact_evidence(evidence, limit=MAX_EVIDENCE), ensure_ascii=False)}

Tool results:
{json.dumps(step_results[-8:], ensure_ascii=False)}

Return JSON:
{{
  "verified": true,
  "final_extraction": {{"required_status":"required|not_required|not_found|unclear","amount":"","percentage":"","basis":"","instrument":"","cash_or_bg":"","validity":"","recovery":"","release_condition":"","conditions":"","exact_clause_excerpt":"","not_found_basis":""}},
  "final_comment": "final concise table comment",
  "warnings": ["warning if any"],
  "verification_note": "why the row is supported"
}}

Rules:
- Verify the draft against description, extract_fields, not_enough, examples, and goal.
- Check every amount, percentage, validity period, recovery condition, and None/Not Required claim.
- Check that the value belongs to the correct financial row and was not borrowed from a similar instrument.
- Use not_required only when expressly stated; use not_found or unclear when absence is inferred from search.
- Confirm the final_comment faithfully summarizes final_extraction.
- Remove unsupported values.
- Keep the final comment compact but complete enough for the Financial Bonds table.
""".strip()

    def call_json(self, prompt: str, system: str) -> dict[str, Any]:
        return json_completion(
            self.client,
            self.model,
            prompt,
            system,
            int(os.getenv("OPENROUTER_AGENT_MAX_TOKENS", str(DEFAULT_AGENT_MAX_TOKENS))),
        )

    def starter_query(self, row: dict[str, Any]) -> str:
        return f"{row['topic']}. {row['goal']} Find exact amount, percentage, validity, recovery, BG/cash condition, and whether none/not required applies."

    def answer_row(self, row: dict[str, Any]) -> dict[str, Any]:
        self.log(f"Starting financial specialist review for row {row['s_no']}: {row['topic']}", row["id"])
        flagged = self.flagged_evidence(row["flag_ids"])
        self.log(f"Collected {len(flagged)} pre-flagged financial topic(s) from the index.", row["id"])
        searched = self.searched_evidence(self.starter_query(row))
        self.log(f"Ran the required fresh targeted project search and collected {len(searched)} searched topic(s).", row["id"])
        evidence = merge_evidence(flagged + searched)
        self.log(f"Merged and deduplicated financial evidence to {len(evidence)} unique topic/page candidate(s).", row["id"])

        step_results: list[dict[str, Any]] = []
        draft: dict[str, Any] | None = None
        for step in range(1, MAX_AGENT_STEPS + 1):
            self.log(f"Financial specialist thinking step {step}: deciding whether more project context is needed.", row["id"])
            action = self.call_json(
                self.specialist_prompt(row, evidence, step_results),
                "Return only valid JSON. You are a financial bonds specialist.",
            )
            self.log(f"Financial specialist requested tool action: {action.get('action', 'unknown')}.", row["id"], {"action": action})
            result = self.run_action(action, row)
            self.log(f"Tool action completed: {action.get('action', 'unknown')}.", row["id"], {"result_keys": sorted(result.keys())})
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
                self.log(f"Financial specialist drafted row {row['s_no']}.", row["id"])
                break

        if draft is None:
            self.log("Financial specialist reached the step limit; forcing an answer from accumulated evidence.", row["id"])
            draft = self.call_json(
                self.specialist_prompt(row, evidence, step_results)
                + "\n\nYou must now return action=answer using the accumulated evidence.",
                "Return only valid JSON.",
            )

        verifier = self.call_json(
            self.verifier_prompt(row, draft, evidence, step_results),
            "Return only valid JSON. You are a strict verifier of financial bond table rows.",
        )
        self.log(
            f"Verifier finalized financial row {row['s_no']}.",
            row["id"],
            {"warnings": verifier.get("warnings", [])},
        )
        return normalize_payload({
            "s_no": row["s_no"],
            "row_id": row["id"],
            "topic": row["topic"],
            "extraction": verifier.get("final_extraction") or draft.get("extraction", {}),
            "comments": verifier.get("final_comment") or draft.get("comment") or "",
            "citations": draft.get("citations", []),
            "evidence": compact_evidence(evidence, limit=MAX_EVIDENCE),
            "agent_steps": step_results,
            "verifier": verifier,
            "confidence": draft.get("confidence", "medium"),
        })

    def answer_row_with_retries(self, row: dict[str, Any]) -> dict[str, Any]:
        attempts = max(1, int(os.getenv("OPENROUTER_FINANCIAL_ROW_RETRIES", str(DEFAULT_FINANCIAL_PARALLEL_RETRIES))))
        last_error: Exception | None = None
        for attempt in range(1, attempts + 1):
            try:
                if attempt > 1:
                    self.log(f"Retrying financial row attempt {attempt}/{attempts}.", row["id"])
                return self.answer_row(row)
            except Exception as exc:
                last_error = exc
                self.log(
                    f"Financial row attempt {attempt}/{attempts} failed: {exc}",
                    row["id"],
                    {"error": str(exc)},
                )
        assert last_error is not None
        raise last_error

    def generate(self) -> dict[str, Any]:
        self.log("Starting Financial Bonds generation.")
        workers = max(1, int(os.getenv("OPENROUTER_FINANCIAL_WORKERS", str(DEFAULT_FINANCIAL_WORKERS))))
        workers = min(workers, len(FINANCIAL_ROWS))
        self.log(f"Running Financial Bonds rows with {workers} parallel worker(s).")
        rows_by_id: dict[str, dict[str, Any]] = {}
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(self.answer_row_with_retries, row): row
                for row in FINANCIAL_ROWS
            }
            for future in as_completed(futures):
                row = futures[future]
                rows_by_id[row["id"]] = future.result()
        rows = [
            rows_by_id[row["id"]]
            for row in FINANCIAL_ROWS
            if row["id"] in rows_by_id
        ]
        report = normalize_payload({
            "report_type": "financial_bonds",
            "status": "verified",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "project_id": self.project_root.name,
            "generation_settings": {
                "row_workers": workers,
            },
            "rows": rows,
            "logs": self.logs,
        })
        self.logs = report["logs"]
        output_path = self.reports_dir / "financial_bonds.json"
        write_json(output_path, report)
        self.log(f"Saved Financial Bonds report to {output_path.name}.")
        report["logs"] = self.logs
        write_json(output_path, report)
        write_json(
            self.progress_path,
            {
                "report_type": "financial_bonds",
                "status": "complete",
                "project_id": self.project_root.name,
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "logs": self.logs,
            },
        )
        return report


def generate_financial_bonds(project_root: Path) -> dict[str, Any]:
    return FinancialBondsAgent(project_root).generate()


def load_financial_bonds(project_root: Path) -> dict[str, Any]:
    return read_json(project_root / "reports" / "financial_bonds.json", {})


def load_financial_bonds_progress(project_root: Path) -> dict[str, Any]:
    return read_json(project_root / "reports" / "financial_bonds.progress.json", {})
