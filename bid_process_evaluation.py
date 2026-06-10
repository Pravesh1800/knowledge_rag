from __future__ import annotations

import json
import os
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
    OPENROUTER_BASE_URL,
    compact_evidence,
    load_dotenv,
    merge_evidence,
    parse_json_response,
    read_json,
    topic_to_evidence,
    write_json,
)


MAX_EVIDENCE = 24
DEFAULT_BID_PROCESS_WORKERS = 2
DEFAULT_BID_PROCESS_RETRIES = 3


BID_PROCESS_GROUPS = [
    {
        "id": "bid_submission",
        "title": "Bid Submission",
        "fields": [
            "Submission Date and Time",
            "Bid Security",
            "Bid Validity",
            "Bidding Stage / Envelope Structure",
            "Mode of Submission",
            "Portal / Submission Website",
            "Hard Copy Requirements",
        ],
        "query": "bid submission date time bid security EMD bid validity bidding stage envelope e-procurement online portal hard copy submission",
        "terms": [
            "bid submission",
            "bid security",
            "EMD",
            "bid validity",
            "envelope",
            "e-procurement",
            "portal",
            "hard copy",
        ],
    },
    {
        "id": "bid_evaluation",
        "title": "Bid Evaluation",
        "fields": [
            "Prequalification Evaluation",
            "Technical Evaluation",
            "Financial Evaluation",
            "Evaluation Method",
            "Lowest Bid Basis",
            "Whole Life Cost / NPV Treatment",
            "Technical Submission Requirement",
            "Pass / Fail or Scoring Rules",
        ],
        "query": "bid evaluation prequalification pass fail technical evaluation financial evaluation lowest bid whole life cost NPV technical score financial score qualification document",
        "terms": [
            "bid evaluation",
            "prequalification",
            "pass",
            "fail",
            "technical evaluation",
            "financial evaluation",
            "lowest",
            "whole life cost",
            "NPV",
            "technical score",
            "financial score",
        ],
    },
]


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
    model = os.getenv(
        "OPENROUTER_BID_PROCESS_MODEL",
        os.getenv("OPENROUTER_LEGAL_AGENT_MODEL", os.getenv("OPENROUTER_MODEL", DEFAULT_MODEL)),
    )
    return client, model


class BidProcessEvaluationAgent:
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
        load_dotenv(self.project_root)
        self.client, self.model = create_client()
        self.logs: list[dict[str, Any]] = []
        self.log_lock = threading.Lock()
        self.progress_path = self.reports_dir / "bid_process_evaluation.progress.json"

    def log(self, message: str, group_id: str | None = None, detail: dict[str, Any] | None = None) -> None:
        entry = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "group_id": group_id,
            "message": message,
            "detail": detail or {},
        }
        with self.log_lock:
            self.logs.append(entry)
            write_json(
                self.progress_path,
                {
                    "report_type": "bid_process_evaluation",
                    "status": "running",
                    "project_id": self.project_root.name,
                    "updated_at": entry["created_at"],
                    "logs": list(self.logs),
                },
            )

    def searched_evidence(self, query: str, max_hits: int = 14) -> list[dict[str, Any]]:
        os.environ["PDF_VISION_RAG_ROOT"] = str(self.project_root)
        os.environ.setdefault("OPENROUTER_SEARCH_MODEL", os.getenv("OPENROUTER_SEARCH_MODEL", DEFAULT_SEARCH_MODEL))
        tree_searcher = searcher_module.TreeSearcher(
            query=query,
            dry_run=False,
            max_hits=max_hits,
            project_root=self.project_root,
            indexes_dir=self.indexes_dir,
            topic_index_path=self.indexes_dir / "topic_index.json",
            relationship_map_path=self.indexes_dir / "relationship_map.json",
            search_results_dir=self.indexes_dir / "search_results",
        )
        result = tree_searcher.search()
        by_triplet = {
            (topic.get("topic_name"), topic.get("document_name"), int(topic.get("page_no") or 0)): topic
            for topic in self.topics
        }
        evidence = []
        for hit in result.get("hits", []):
            topic = by_triplet.get((hit.get("topic_name"), hit.get("document_name"), int(hit.get("page_no") or 0)))
            if topic:
                evidence.append(topic_to_evidence(topic, "searched", hit.get("relevance_reason", "Found by bid process search.")))
            for related in hit.get("related_topics", []):
                related_topic = by_triplet.get((related.get("topic_name"), related.get("document_name"), int(related.get("page_no") or 0)))
                if related_topic:
                    evidence.append(topic_to_evidence(related_topic, "searched", related.get("relation", "Related bid process topic.")))
        return evidence

    def keyword_evidence(self, terms: list[str], limit: int = 16) -> list[dict[str, Any]]:
        lowered = [term.lower() for term in terms if term]
        scored: list[tuple[int, dict[str, Any]]] = []
        for topic in self.topics:
            haystack = " ".join(
                [
                    str(topic.get("topic_name", "")),
                    str(topic.get("topic_description", "")),
                    str(topic.get("content", ""))[:5000],
                ]
            ).lower()
            score = sum(1 for term in lowered if term in haystack)
            if score:
                scored.append((score, topic))
        scored.sort(key=lambda item: (-item[0], str(item[1].get("document_name", "")), int(item[1].get("page_no") or 0)))
        return [
            topic_to_evidence(topic, "keyword", f"Keyword lookup matched {score} bid-process term(s).")
            for score, topic in scored[:limit]
        ]

    def extraction_prompt(self, group: dict[str, Any], evidence: list[dict[str, Any]]) -> str:
        return f"""
You are extracting a Bid Process and Evaluation fact sheet from tender documents.

The output should be slide-ready: short label/value lines under "Bid Submission" and "Bid Evaluation".
Do not invent facts from the example screenshot. Use only evidence.

Group:
{json.dumps(group, ensure_ascii=False)}

Evidence:
{json.dumps(compact_evidence(evidence, limit=MAX_EVIDENCE), ensure_ascii=False)}

Return valid JSON only:
{{
  "group_id": "{group["id"]}",
  "group_title": "{group["title"]}",
  "items": [
    {{
      "label": "field label",
      "value": "short exact value, or '-' if not found",
      "status": "found|not_found|ambiguous",
      "notes": "short note only if needed",
      "citations": [{{"document_name": "", "page_no": 0, "topic_name": "", "excerpt": ""}}],
      "confidence": "low|medium|high"
    }}
  ],
  "coverage_note": "what was searched and any gaps"
}}

Rules:
- Extract all requested fields where possible.
- Keep values short: dates, amounts, days, pass/fail, lowest bid basis, online/e-procurement mode.
- If a field is absent, include it as not_found with value "-".
- Every found or ambiguous item needs at least one citation.
- For financial evaluation, distinguish lowest price, WLC/NPV, technical score, and pass/fail if the documents do.
""".strip()

    def verifier_prompt(self, group: dict[str, Any], draft: dict[str, Any], evidence: list[dict[str, Any]]) -> str:
        return f"""
Verify this Bid Process and Evaluation extraction.

Group:
{json.dumps(group, ensure_ascii=False)}

Draft:
{json.dumps(draft, ensure_ascii=False)}

Evidence:
{json.dumps(compact_evidence(evidence, limit=MAX_EVIDENCE), ensure_ascii=False)}

Return valid JSON only:
{{
  "items": [
    {{
      "label": "",
      "value": "",
      "status": "found|not_found|ambiguous",
      "notes": "",
      "citations": [{{"document_name": "", "page_no": 0, "topic_name": "", "excerpt": ""}}],
      "confidence": "low|medium|high"
    }}
  ],
  "warnings": [],
  "coverage_note": ""
}}

Verification rules:
- Remove unsupported claims.
- Keep exact dates, times, amounts, validity periods, submission modes, and evaluation rules.
- Do not use example screenshot values unless they are present in this project's evidence.
- Preserve all requested fields, marking absent fields as not_found.
""".strip()

    def repair_json(self, malformed_content: str, parser_error: str, max_tokens: int, group_id: str, stage: str) -> dict[str, Any]:
        repair_content = malformed_content
        repair_error = parser_error
        for attempt in range(1, 4):
            repair_prompt = f"""
Repair this malformed model output into valid JSON only.

Exact parser error to fix:
{repair_error}

Preserve all recoverable labels, values, statuses, notes, citations, and warnings.
Do not summarize, shorten, or regenerate.

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
            )
            repair_content = response.choices[0].message.content or "{}"
            try:
                fixed = parse_json_response(repair_content)
                self.log(f"Bid Process {stage} JSON repair succeeded on attempt {attempt}/3.", group_id)
                return fixed
            except Exception as exc:
                repair_error = str(exc)
                self.log(f"Bid Process {stage} JSON repair attempt {attempt}/3 failed.", group_id, {"error": repair_error})
        raise ValueError(f"Bid Process {stage} returned malformed JSON and repair failed: {repair_error}")

    def call_json(self, prompt: str, system: str, max_tokens: int, group_id: str, stage: str) -> dict[str, Any]:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            response_format={"type": "json_object"},
            max_tokens=max_tokens,
        )
        content = response.choices[0].message.content or "{}"
        try:
            return parse_json_response(content)
        except Exception as exc:
            self.log(f"Bid Process {stage} returned malformed JSON; repairing without restarting.", group_id, {"error": str(exc)})
            return self.repair_json(content, str(exc), max_tokens, group_id, stage)

    def answer_group(self, group: dict[str, Any]) -> dict[str, Any]:
        self.log(f"Searching bid-process evidence for {group['title']}.", group["id"])
        evidence = merge_evidence(self.searched_evidence(group["query"]) + self.keyword_evidence(group["terms"]))
        self.log(f"Collected {len(evidence)} evidence topic(s) for {group['title']}.", group["id"])
        max_tokens = int(os.getenv("OPENROUTER_BID_PROCESS_MAX_TOKENS", str(DEFAULT_AGENT_MAX_TOKENS + 1024)))
        draft = self.call_json(
            self.extraction_prompt(group, evidence),
            "Return only valid JSON. You extract concise bid-process facts from tender evidence.",
            max_tokens,
            group["id"],
            "extraction",
        )
        verified = self.call_json(
            self.verifier_prompt(group, draft, evidence),
            "Return only valid JSON. You verify bid-process facts against evidence.",
            max_tokens,
            group["id"],
            "verifier",
        )
        items = verified.get("items", []) if isinstance(verified.get("items", []), list) else []
        self.log(f"Verifier finalized {len(items)} Bid Process item(s) for {group['title']}.", group["id"], {"warnings": verified.get("warnings", [])})
        return {
            "group_id": group["id"],
            "title": group["title"],
            "items": items,
            "coverage_note": verified.get("coverage_note") or draft.get("coverage_note", ""),
            "evidence": compact_evidence(evidence, limit=MAX_EVIDENCE),
            "verifier": verified,
        }

    def answer_group_with_retries(self, group: dict[str, Any]) -> dict[str, Any]:
        attempts = max(1, int(os.getenv("OPENROUTER_BID_PROCESS_SECTION_RETRIES", str(DEFAULT_BID_PROCESS_RETRIES))))
        last_error: Exception | None = None
        for attempt in range(1, attempts + 1):
            try:
                if attempt > 1:
                    self.log(f"Retrying Bid Process group attempt {attempt}/{attempts}.", group["id"])
                return self.answer_group(group)
            except Exception as exc:
                last_error = exc
                self.log(f"Bid Process group attempt {attempt}/{attempts} failed: {exc}", group["id"], {"error": str(exc)})
                if any(term in str(exc).lower() for term in ["key limit exceeded", "daily limit", "error code: 403"]):
                    break
        assert last_error is not None
        raise last_error

    def final_rows(self, groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for group in groups:
            for item in group.get("items", []):
                rows.append(
                    {
                        "s_no": len(rows) + 1,
                        "group_id": group.get("group_id", ""),
                        "group_title": group.get("title", ""),
                        "label": item.get("label", ""),
                        "value": item.get("value", "-"),
                        "status": item.get("status", "not_found"),
                        "notes": item.get("notes", ""),
                        "citations": item.get("citations", []),
                        "confidence": item.get("confidence", "medium"),
                    }
                )
        return rows

    def generate(self) -> dict[str, Any]:
        self.log("Starting Bid Process and Evaluation generation.")
        workers = max(1, int(os.getenv("OPENROUTER_BID_PROCESS_WORKERS", str(DEFAULT_BID_PROCESS_WORKERS))))
        workers = min(workers, len(BID_PROCESS_GROUPS))
        self.log(f"Running Bid Process groups with {workers} parallel worker(s).")
        groups_by_id: dict[str, dict[str, Any]] = {}
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(self.answer_group_with_retries, group): group for group in BID_PROCESS_GROUPS}
            for future in as_completed(futures):
                group = futures[future]
                groups_by_id[group["id"]] = future.result()
        groups = [groups_by_id[group["id"]] for group in BID_PROCESS_GROUPS if group["id"] in groups_by_id]
        rows = self.final_rows(groups)
        report = {
            "report_type": "bid_process_evaluation",
            "status": "verified",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "project_id": self.project_root.name,
            "title": "Bid Process and Evaluation",
            "generation_settings": {
                "group_workers": workers,
                "groups": [group["id"] for group in BID_PROCESS_GROUPS],
            },
            "columns": ["S. No.", "Group", "Label", "Value", "Status", "Notes", "Confidence"],
            "rows": rows,
            "groups": groups,
            "logs": self.logs,
        }
        output_path = self.reports_dir / "bid_process_evaluation.json"
        write_json(output_path, report)
        self.log(f"Saved Bid Process and Evaluation report to {output_path.name}.")
        report["logs"] = self.logs
        write_json(output_path, report)
        write_json(
            self.progress_path,
            {
                "report_type": "bid_process_evaluation",
                "status": "complete",
                "project_id": self.project_root.name,
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "logs": self.logs,
            },
        )
        return report


def generate_bid_process_evaluation(project_root: Path) -> dict[str, Any]:
    return BidProcessEvaluationAgent(project_root).generate()


def load_bid_process_evaluation(project_root: Path) -> dict[str, Any]:
    return read_json(project_root / "reports" / "bid_process_evaluation.json", {})


def load_bid_process_evaluation_progress(project_root: Path) -> dict[str, Any]:
    return read_json(project_root / "reports" / "bid_process_evaluation.progress.json", {})


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate the Bid Process and Evaluation report for a project.")
    parser.add_argument("project_root", type=Path, help="Path to the indexed project directory.")
    args = parser.parse_args()
    result = generate_bid_process_evaluation(args.project_root)
    print(json.dumps({"status": result.get("status", "complete"), "rows": len(result.get("rows", []))}, ensure_ascii=False))
