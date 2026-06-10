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
    json_completion,
    load_dotenv,
    merge_evidence,
    read_json,
    topic_to_evidence,
    write_json,
)


MAX_EVIDENCE = 30
DEFAULT_LIABILITY_WORKERS = 5
DEFAULT_LIABILITY_RETRIES = 3


LIABILITY_SECTIONS = [
    {
        "id": "db_delay_lds",
        "title": "LDs for delay during DB Period",
        "liability_category": "Financial Liabilities & Penalties",
        "period": "DB Period",
        "liability_type": "Delay LD",
        "query": "liquidated damages delay delay damages completion milestone DB period design build period cap percentage contract price balance work",
        "terms": ["liquidated damages", "delay", "milestone", "completion", "DB period", "design build", "cap", "balance work"],
    },
    {
        "id": "db_performance_penalties",
        "title": "LDs / penalties for performance during DB Period",
        "liability_category": "Financial Liabilities & Penalties",
        "period": "DB Period",
        "liability_type": "Performance Penalty",
        "query": "penalty DB period technical personnel field laboratory labour machinery trench shoring boards restoration pipeline road restoration engineer in charge",
        "terms": ["penalty", "technical personnel", "field laboratory", "labour", "machinery", "trench", "shoring", "road restoration"],
    },
    {
        "id": "om_performance_lds",
        "title": "LDs for performance during O&M Period",
        "liability_category": "Financial Liabilities & Penalties",
        "period": "O&M Period",
        "liability_type": "Performance LD",
        "query": "O&M performance liquidated damages water quality CPHEEO water loss unaccounted flow UF breach penalty safe potable water",
        "terms": ["O&M", "performance", "water quality", "CPHEEO", "water loss", "unaccounted flow", "UF", "safe potable"],
    },
    {
        "id": "om_operational_penalties",
        "title": "Penalty for performance during O&M Period",
        "liability_category": "Financial Liabilities & Penalties",
        "period": "O&M Period",
        "liability_type": "O&M Penalty",
        "query": "O&M penalty guaranteed electricity consumption energy meter downtime report CMS service manpower monthly personnel repair notice instrument safety equipment staff",
        "terms": ["O&M penalty", "guaranteed electricity", "energy meter", "downtime", "CMS", "manpower", "repair", "safety equipment"],
    },
    {
        "id": "caps_and_limits",
        "title": "Caps, Ceilings, Refunds, and Recovery Limits",
        "liability_category": "Financial Liabilities & Penalties",
        "period": "Contract Wide",
        "liability_type": "Liability Cap",
        "query": "cap ceiling maximum penalty limit liability cap refund retained amount recovery contract value O&M contract value DB contract value",
        "terms": ["cap", "ceiling", "maximum", "refund", "retained", "recovery", "contract value", "liability"],
    },
    {
        "id": "service_level_benchmarks",
        "title": "Service Level Benchmarks",
        "liability_category": "Other Liabilities & Penalties",
        "period": "O&M Period",
        "liability_type": "Service Level",
        "query": "service level benchmark potable drinking water IS 10500 24x7 water supply complaint redressal leakage restoration bulk flow meter outlet inlet pump house",
        "terms": ["service level", "IS 10500", "24x7", "complaint", "leakage", "bulk flow meter", "safe potable"],
    },
    {
        "id": "old_structure_liabilities",
        "title": "Maintenance of Old Structures",
        "liability_category": "Other Liabilities & Penalties",
        "period": "DB / O&M Interface",
        "liability_type": "Existing Asset Liability",
        "query": "maintenance old structures existing WTP intake pump house pipeline rehabilitation replacement major minor replacement old assets constructed year",
        "terms": ["old structures", "existing WTP", "intake", "pump house", "pipeline", "rehabilitation", "replacement", "major replacement"],
    },
]


def create_client() -> tuple[OpenAI, str]:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is missing in .env")
    client = OpenAI(
        base_url=os.getenv("OPENROUTER_BASE_URL", OPENROUTER_BASE_URL),
        api_key=api_key,
        timeout=float(os.getenv("OPENROUTER_TIMEOUT_SECONDS", "180")),
        max_retries=int(os.getenv("OPENROUTER_MAX_RETRIES", "2")),
        default_headers={
            "HTTP-Referer": os.getenv("OPENROUTER_SITE_URL", "http://localhost"),
            "X-Title": os.getenv("OPENROUTER_APP_NAME", "PDF Vision RAG"),
        },
    )
    model = os.getenv(
        "OPENROUTER_LIABILITY_AGENT_MODEL",
        os.getenv("OPENROUTER_FINANCIAL_AGENT_MODEL", os.getenv("OPENROUTER_MODEL", DEFAULT_MODEL)),
    )
    return client, model


class FinancialLiabilitiesPenaltiesAgent:
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
        self.partial_lock = threading.Lock()
        self.progress_path = self.reports_dir / "financial_liabilities_penalties.progress.json"
        self.partial_path = self.reports_dir / "financial_liabilities_penalties.partial.json"

    def log(self, message: str, section_id: str | None = None, detail: dict[str, Any] | None = None) -> None:
        entry = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "section_id": section_id,
            "message": message,
            "detail": detail or {},
        }
        with self.log_lock:
            self.logs.append(entry)
            write_json(
                self.progress_path,
                {
                    "report_type": "financial_liabilities_penalties",
                    "status": "running",
                    "project_id": self.project_root.name,
                    "updated_at": entry["created_at"],
                    "logs": list(self.logs),
                },
            )

    def keyword_evidence(self, terms: list[str], limit: int = 28) -> list[dict[str, Any]]:
        lowered = [term.lower() for term in terms if term]
        scored: list[tuple[int, dict[str, Any]]] = []
        for topic in self.topics:
            haystack = " ".join(
                [
                    str(topic.get("topic_name", "")),
                    str(topic.get("topic_description", "")),
                    str(topic.get("content", ""))[:6000],
                    " ".join(str(tag) for tag in topic.get("tags", [])),
                ]
            ).lower()
            score = sum(1 for term in lowered if term in haystack)
            if score:
                scored.append((score, topic))
        scored.sort(key=lambda item: (-item[0], str(item[1].get("document_name", "")), int(item[1].get("page_no") or 0)))
        return [
            topic_to_evidence(topic, "keyword_liability_scan", f"Matched {score} liability keyword(s).")
            for score, topic in scored[:limit]
        ]

    def searched_evidence(self, query: str, max_hits: int = 14) -> list[dict[str, Any]]:
        os.environ["PDF_VISION_RAG_ROOT"] = str(self.project_root)
        os.environ.setdefault("OPENROUTER_SEARCH_MODEL", os.getenv("OPENROUTER_SEARCH_MODEL", DEFAULT_SEARCH_MODEL))
        dry_run = os.getenv("OPENROUTER_LIABILITY_SEARCH_DRY_RUN", "1").strip().lower() in {"1", "true", "yes", "on"}
        tree_searcher = searcher_module.TreeSearcher(
            query=query,
            dry_run=dry_run,
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
                evidence.append(topic_to_evidence(topic, "searched_liability", hit.get("relevance_reason", "Found by liabilities search.")))
            for related in hit.get("related_topics", []):
                related_topic = by_triplet.get((related.get("topic_name"), related.get("document_name"), int(related.get("page_no") or 0)))
                if related_topic:
                    evidence.append(topic_to_evidence(related_topic, "searched_related_liability", related.get("relation", "Related liability topic.")))
        return evidence

    def prompt(self, section: dict[str, Any], evidence: list[dict[str, Any]]) -> str:
        return f"""
Create the Financial Liabilities & Penalties table entries for this section.

Section:
{json.dumps(section, ensure_ascii=False)}

Evidence:
{json.dumps(evidence[:MAX_EVIDENCE], ensure_ascii=False)}

What to extract:
- liquidated damages, delay damages, penalties, service-level penalties, O&M penalties, non-performance charges
- caps, ceilings, maximum exposure, refund/release terms, recovery basis
- exact amount/rate/percentage, per-day/per-month/per-event basis, triggering condition, applicable period
- old asset maintenance and replacement obligations where they create liability or cost exposure
- service benchmarks such as drinking water standards, water loss, guaranteed power/energy, report downtime, complaint redressal

Rules:
1. Return only entries that are supported by the evidence.
2. Prefer exact clause wording and exact figures over generic summaries.
3. Split distinct penalties into separate rows even if they are under one clause.
4. Do not invent missing caps. If a cap is not found, set cap to "-".
5. Use concise but complete comments. Include the trigger, rate/amount, and basis in the comment.
6. Add document_name and page_no from the strongest citation.

Return only valid JSON:
{{
  "rows": [
    {{
      "topic": "Short penalty/liability topic",
      "liability_category": "{section['liability_category']}",
      "comments": "Exact operational/legal meaning with amount/rate and trigger.",
      "cap": "Cap or ceiling, or -",
      "period": "{section['period']}",
      "liability_type": "{section['liability_type']}",
      "amount_or_rate": "Amount / rate / percentage",
      "basis": "per day / per month / per event / contract value / other basis",
      "document_name": "Source document name",
      "page_no": "Source page",
      "confidence": "high|medium|low",
      "evidence_citations": [
        {{
          "document_name": "Source document name",
          "page_no": 1,
          "topic_name": "Evidence topic",
          "excerpt": "Short supporting excerpt"
        }}
      ]
    }}
  ],
  "section_note": "Coverage note for this section"
}}
""".strip()

    def normalize_row(self, section: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
        citations = row.get("evidence_citations") if isinstance(row.get("evidence_citations"), list) else []
        first_citation = next((item for item in citations if isinstance(item, dict)), {})
        return {
            "topic": row.get("topic") or section["title"],
            "liability_category": row.get("liability_category") or section["liability_category"],
            "comments": row.get("comments") or "-",
            "cap": row.get("cap") or "-",
            "period": row.get("period") or section["period"],
            "liability_type": row.get("liability_type") or section["liability_type"],
            "amount_or_rate": row.get("amount_or_rate") or "-",
            "basis": row.get("basis") or "-",
            "document_name": row.get("document_name") or first_citation.get("document_name", "-"),
            "page_no": row.get("page_no") or first_citation.get("page_no", "-"),
            "confidence": row.get("confidence") or "medium",
            "evidence_citations": citations,
        }

    def save_partial_section(self, section: dict[str, Any], result: dict[str, Any], evidence: list[dict[str, Any]]) -> None:
        entry = {
            "section_id": section["id"],
            "title": section["title"],
            "rows": result.get("rows", []),
            "section_note": result.get("section_note", ""),
            "evidence": evidence[:MAX_EVIDENCE],
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        with self.partial_lock:
            partial = read_json(
                self.partial_path,
                {
                    "report_type": "financial_liabilities_penalties",
                    "status": "partial",
                    "project_id": self.project_root.name,
                    "sections": {},
                },
            )
            if not isinstance(partial, dict):
                partial = {"report_type": "financial_liabilities_penalties", "status": "partial", "project_id": self.project_root.name, "sections": {}}
            partial.setdefault("sections", {})
            partial["sections"][section["id"]] = entry
            partial["updated_at"] = entry["updated_at"]
            write_json(self.partial_path, partial)

    def answer_section(self, section: dict[str, Any]) -> dict[str, Any]:
        self.log(f"Starting liability section: {section['title']}", section["id"])
        evidence = merge_evidence(self.keyword_evidence(section["terms"]) + self.searched_evidence(section["query"]))
        self.log(f"Collected {len(evidence)} evidence item(s) for {section['title']}.", section["id"])
        max_tokens = int(os.getenv("OPENROUTER_LIABILITY_MAX_TOKENS", str(DEFAULT_AGENT_MAX_TOKENS + 2048)))
        result = json_completion(
            self.client,
            self.model,
            self.prompt(section, evidence),
            "Return only valid JSON. You are a senior contracts specialist extracting financial liabilities and penalties from tender documents.",
            max_tokens=max_tokens,
        )
        rows = result.get("rows", [])
        if not isinstance(rows, list):
            rows = []
        normalized = [self.normalize_row(section, row) for row in rows if isinstance(row, dict)]
        result["rows"] = normalized
        self.save_partial_section(section, result, evidence)
        self.log(f"Finalized {len(normalized)} liability row(s) for {section['title']}.", section["id"])
        return {
            "section_id": section["id"],
            "title": section["title"],
            "liability_category": section["liability_category"],
            "period": section["period"],
            "liability_type": section["liability_type"],
            "rows": normalized,
            "section_note": result.get("section_note", ""),
        }

    def answer_section_with_retries(self, section: dict[str, Any]) -> dict[str, Any]:
        attempts = max(1, int(os.getenv("OPENROUTER_LIABILITY_SECTION_RETRIES", str(DEFAULT_LIABILITY_RETRIES))))
        last_error: Exception | None = None
        for attempt in range(1, attempts + 1):
            try:
                if attempt > 1:
                    self.log(f"Retrying liability section attempt {attempt}/{attempts}.", section["id"], {"previous_error": str(last_error)})
                return self.answer_section(section)
            except Exception as exc:
                last_error = exc
                self.log(f"Liability section attempt {attempt}/{attempts} failed: {exc}", section["id"], {"error_type": type(exc).__name__})
        raise RuntimeError(str(last_error or "Liability section failed."))

    def generate(self) -> dict[str, Any]:
        self.log("Starting Financial Liabilities & Penalties generation.")
        workers = min(max(1, int(os.getenv("OPENROUTER_LIABILITY_SECTION_WORKERS", str(DEFAULT_LIABILITY_WORKERS)))), len(LIABILITY_SECTIONS))
        sections_by_id: dict[str, dict[str, Any]] = {}
        failures = []
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(self.answer_section_with_retries, section): section for section in LIABILITY_SECTIONS}
            for future in as_completed(futures):
                section = futures[future]
                try:
                    sections_by_id[section["id"]] = future.result()
                except Exception as exc:
                    failures.append({"section_id": section["id"], "title": section["title"], "error": str(exc)})
                    self.log(f"Liability section failed after retries: {section['title']}: {exc}", section["id"])
        if failures:
            write_json(
                self.progress_path,
                {
                    "report_type": "financial_liabilities_penalties",
                    "status": "failed",
                    "project_id": self.project_root.name,
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                    "failures": failures,
                    "logs": self.logs,
                },
            )
            raise RuntimeError(f"Financial Liabilities & Penalties generation failed for {len(failures)} section(s): {failures}")

        sections = [sections_by_id[section["id"]] for section in LIABILITY_SECTIONS if section["id"] in sections_by_id]
        rows = []
        for section in sections:
            for row in section.get("rows", []):
                rows.append(row)
        for index, row in enumerate(rows, start=1):
            row["s_no"] = index
        report = {
            "report_type": "financial_liabilities_penalties",
            "status": "verified",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "project_id": self.project_root.name,
            "title": "Financial Liabilities & Penalties",
            "columns": [
                "S. No.",
                "Topic",
                "Liability Category",
                "Comments",
                "CAP",
                "Period",
                "Liability Type",
                "Amount / Rate",
                "Basis",
                "Document Name",
                "Page No.",
                "Confidence",
                "Evidence",
            ],
            "rows": rows,
            "sections": sections,
            "logs": self.logs,
        }
        output_path = self.reports_dir / "financial_liabilities_penalties.json"
        write_json(output_path, report)
        self.log(f"Saved Financial Liabilities & Penalties report to {output_path.name}.")
        report["logs"] = self.logs
        write_json(output_path, report)
        write_json(
            self.progress_path,
            {
                "report_type": "financial_liabilities_penalties",
                "status": "complete",
                "project_id": self.project_root.name,
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "logs": self.logs,
            },
        )
        return report


def generate_financial_liabilities_penalties(project_root: Path) -> dict[str, Any]:
    return FinancialLiabilitiesPenaltiesAgent(project_root).generate()


def load_financial_liabilities_penalties(project_root: Path) -> dict[str, Any]:
    return read_json(project_root / "reports" / "financial_liabilities_penalties.json", {})


def load_financial_liabilities_penalties_progress(project_root: Path) -> dict[str, Any]:
    return read_json(project_root / "reports" / "financial_liabilities_penalties.progress.json", {})


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate Financial Liabilities & Penalties for a project.")
    parser.add_argument("project_root", type=Path, help="Path to the indexed project directory.")
    args = parser.parse_args()
    result = generate_financial_liabilities_penalties(args.project_root)
    print(json.dumps({"status": result.get("status", "complete"), "rows": len(result.get("rows", []))}, ensure_ascii=False))
