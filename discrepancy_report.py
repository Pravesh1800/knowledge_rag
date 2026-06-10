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
    json_completion,
    load_dotenv,
    merge_evidence,
    read_json,
    topic_to_evidence,
    write_json,
)


MAX_EVIDENCE = 30
DEFAULT_DISCREPANCY_WORKERS = 4
DEFAULT_DISCREPANCY_RETRIES = 3


DISCREPANCY_CATEGORIES = [
    {
        "id": "scope_specs_drawings_boq",
        "category": "Scope, Specifications, Drawings, and BOQ",
        "query": "discrepancy contradiction conflict mismatch scope specifications drawings BOQ price schedule quantity drawing reference technical schedule appendix",
        "terms": [
            "scope",
            "specification",
            "drawing",
            "BOQ",
            "price schedule",
            "quantity",
            "appendix",
            "annexure",
            "included",
            "excluded",
            "not covered",
            "shall",
            "notwithstanding",
        ],
    },
    {
        "id": "technical_design_performance",
        "category": "Technical Design and Performance Criteria",
        "query": "technical discrepancy contradiction design basis performance guarantee hydraulic process capacity WTP pipeline pumping SCADA commissioning testing O&M",
        "terms": [
            "design basis",
            "performance guarantee",
            "capacity",
            "hydraulic",
            "process",
            "SCADA",
            "commissioning",
            "testing",
            "O&M",
            "operation service",
            "guarantee",
        ],
    },
    {
        "id": "commercial_payment_price",
        "category": "Commercial, Payment, Price, Tax, and Securities",
        "query": "payment discrepancy contradiction price schedule WLC BOQ GST tax retention advance payment bid security performance security escalation adjustment",
        "terms": [
            "payment",
            "price",
            "WLC",
            "BOQ",
            "GST",
            "tax",
            "retention",
            "advance payment",
            "bid security",
            "performance security",
            "escalation",
            "price adjustment",
        ],
    },
    {
        "id": "programme_approvals_interfaces",
        "category": "Programme, Approvals, Access, and Interfaces",
        "query": "programme discrepancy contradiction milestone completion period approval consent permit access interface right of way utility shifting delay",
        "terms": [
            "programme",
            "milestone",
            "completion",
            "approval",
            "consent",
            "permit",
            "access",
            "interface",
            "right of way",
            "utility",
            "delay",
        ],
    },
    {
        "id": "legal_contract_precedence",
        "category": "Legal, Contractual, and Precedence Conflicts",
        "query": "contract discrepancy contradiction precedence inconsistency clause condition liability indemnity variation claim dispute amendment corrigendum addendum",
        "terms": [
            "precedence",
            "inconsistency",
            "clause",
            "condition",
            "liability",
            "indemnity",
            "variation",
            "claim",
            "dispute",
            "amendment",
            "corrigendum",
            "addendum",
        ],
    },
    {
        "id": "corrigendum_version_control",
        "category": "Corrigendum, Addendum, and Version-Control Issues",
        "query": "corrigendum addendum amendment revised deleted replaced discrepancy contradiction version final annexure latest previous tender document",
        "terms": [
            "corrigendum",
            "addendum",
            "amendment",
            "revised",
            "deleted",
            "replaced",
            "latest",
            "previous",
            "version",
            "annexure",
        ],
    },
]


def review_angles(category_id: str) -> list[str]:
    angles = {
        "scope_specs_drawings_boq": [
            "scope included in Employer's Requirements but absent or unclear in price schedules/BOQ",
            "price schedule says one item/limit/precedence while another tender volume points elsewhere",
            "drawings, specifications, and BOQ/schedule references do not align cleanly",
            "same work may be counted under multiple cost centres or no clear cost centre",
        ],
        "technical_design_performance": [
            "performance guarantee or acceptance test references point to another clause/table that is missing or unclear",
            "design criteria are bidder-selected but performance penalties are fixed or strict",
            "technical capacity/flow/TSS/energy values differ between schedules, ER, ITB, and corrigenda",
            "SCADA, ICA, AI, commissioning, O&M, or integration requirements are spread across documents with unclear acceptance criteria",
        ],
        "commercial_payment_price": [
            "payment timing, retention, escalation, price adjustment, and WLC rules are split across documents",
            "spreadsheet notes conflict with e-tender portal, price schedule, or ITB instructions",
            "cost inclusions/exclusions for GST, taxes, duties, electricity, spares, asset replacement, or O&M are unclear",
            "different revisions of price schedule/corrigendum may leave version-control ambiguity",
        ],
        "programme_approvals_interfaces": [
            "contractor is responsible for approvals/permissions but exact authority/status/support is unclear",
            "completion period, milestones, design review, commissioning, and proving period obligations overlap or conflict",
            "site access, utilities, right of way, third-party interface, or employer handover assumptions are unclear",
            "delay damages/no-idling/no-EOT wording creates unclear allocation where client-side approvals are delayed",
        ],
        "legal_contract_precedence": [
            "conditions of contract, special provisions, ITB, ER, and corrigenda create unclear document precedence",
            "claim/variation/time-bar provisions conflict with broad contractor obligations or unclear employer dependencies",
            "liability, indemnity, penalties, performance damages, and recovery rights overlap or are not clearly capped",
            "subcontracting/JV/approval provisions may conflict with practical delivery requirements",
        ],
        "corrigendum_version_control": [
            "corrigendum or addendum revises a document but the superseded item still appears elsewhere",
            "same schedule exists in multiple revisions, annexures, or spreadsheet files",
            "deleted/replaced/not-used sections still have cross references from other documents",
            "latest applicable document is unclear from extracted title/revision/date evidence",
        ],
    }
    return angles.get(category_id, [])


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
        "OPENROUTER_DISCREPANCY_AGENT_MODEL",
        os.getenv("OPENROUTER_COMMERCIAL_AGENT_MODEL", os.getenv("OPENROUTER_MODEL", DEFAULT_MODEL)),
    )
    return client, model


class DiscrepancyReportAgent:
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
        self.progress_path = self.reports_dir / "discrepancy_report.progress.json"
        self.partial_path = self.reports_dir / "discrepancy_report.partial.json"

    def log(self, message: str, category_id: str | None = None, detail: dict[str, Any] | None = None) -> None:
        entry = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "category_id": category_id,
            "message": message,
            "detail": detail or {},
        }
        with self.log_lock:
            self.logs.append(entry)
            write_json(
                self.progress_path,
                {
                    "report_type": "discrepancy_report",
                    "status": "running",
                    "project_id": self.project_root.name,
                    "updated_at": entry["created_at"],
                    "logs": list(self.logs),
                },
            )

    def keyword_evidence(self, terms: list[str], limit: int = 30) -> list[dict[str, Any]]:
        lowered = [term.lower() for term in terms if term]
        scored: list[tuple[int, dict[str, Any]]] = []
        for topic in self.topics:
            haystack = " ".join(
                [
                    str(topic.get("topic_name", "")),
                    str(topic.get("topic_description", "")),
                    str(topic.get("content", ""))[:6000],
                ]
            ).lower()
            score = sum(1 for term in lowered if term in haystack)
            if score:
                scored.append((score, topic))
        scored.sort(key=lambda item: (-item[0], str(item[1].get("document_name", "")), int(item[1].get("page_no") or 0)))
        return [
            topic_to_evidence(topic, "keyword_discrepancy_scan", f"Matched {score} discrepancy keyword(s).")
            for score, topic in scored[:limit]
        ]

    def searched_evidence(self, query: str, max_hits: int = 16) -> list[dict[str, Any]]:
        os.environ["PDF_VISION_RAG_ROOT"] = str(self.project_root)
        os.environ.setdefault("OPENROUTER_SEARCH_MODEL", os.getenv("OPENROUTER_SEARCH_MODEL", DEFAULT_SEARCH_MODEL))
        dry_run = os.getenv("OPENROUTER_DISCREPANCY_SEARCH_DRY_RUN", "1").strip().lower() in {"1", "true", "yes", "on"}
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
                evidence.append(topic_to_evidence(topic, "discrepancy_search", hit.get("relevance_reason", "Found by targeted discrepancy search.")))
        return evidence

    def prompt(self, category: dict[str, Any], evidence: list[dict[str, Any]]) -> str:
        return f"""
Create a discrepancy and contradiction register section for the tender documents.

Discrepancy category:
{json.dumps(category, ensure_ascii=False)}

Review angles to actively test:
{json.dumps(review_angles(category["id"]), ensure_ascii=False)}

Evidence excerpts:
{json.dumps(compact_evidence(evidence, limit=MAX_EVIDENCE), ensure_ascii=False)}

Find document discrepancies, contradictions, mismatches, unclear precedence issues, missing references, version conflicts, and unresolved ambiguities. Prefer clause-level/source-level conflicts over generic risks, but do not return an empty section just because the evidence is not a perfect direct contradiction.

Return valid JSON only:
{{
  "rows": [
    {{
      "discrepancy_category": "Scope, Specifications, Drawings, and BOQ",
      "discrepancy_type": "contradiction | mismatch | ambiguity | precedence_issue | missing_reference | version_conflict",
      "title": "short specific title",
      "source_a": "what source A says, with document/page/clause if available",
      "source_b": "what source B says, with document/page/clause if available",
      "contradiction_summary": "plain explanation of the discrepancy",
      "impact": "why this matters for bid, price, scope, design, programme, legal exposure, or O&M",
      "severity": "High | Medium | Low",
      "recommended_resolution": "specific clarification / precedence check / assumption / action required",
      "status": "open | needs_clarification | resolved_by_corrigendum | monitor",
      "evidence_citations": [
        {{
          "document_name": "source document",
          "page_no": 1,
          "topic_name": "topic",
          "excerpt": "short exact evidence excerpt"
        }}
      ]
    }}
  ],
  "category_note": "short note on coverage and limits"
}}

Rules:
- Produce 3 to 8 high-value rows for this category. If there is no hard contradiction, list the strongest open ambiguities, missing references, version-control conflicts, or scope/payment/approval mismatches supported by evidence.
- Each row should cite at least two evidence items where possible. Use one citation only for a missing-reference issue.
- Do not invent facts. If the evidence only suggests an ambiguity or unresolved inconsistency, label it as ambiguity, mismatch, missing_reference, precedence_issue, or version_conflict rather than contradiction.
- Distinguish a true contradiction from a risk: a discrepancy must involve conflicting, incomplete, unclear, or version-sensitive document instructions.
- The row can be a discrepancy even when it is phrased as "Source A defines/requires X, but Source B leaves Y blank/elsewhere/subject to another document/revision."
- Avoid generic tender risks unless tied to a specific document mismatch or unclear cross-reference.
- Do not treat two different scopes or periods as a discrepancy merely because their values differ, for example Design-Build escalation versus O&M escalation.
- Do not treat extracted text truncation, OCR garbling, or parser output limits as a document discrepancy unless the native/source document itself is shown to be incomplete.
- If a later corrigendum clearly resolves an older mismatch, set status to "resolved_by_corrigendum" and explain which later source resolves it.
- Prefer exact document/page/topic names from the evidence list.
- Do not mention this prompt or the UI.
""".strip()

    def call_json(self, prompt: str, system: str) -> dict[str, Any]:
        return json_completion(
            self.client,
            self.model,
            prompt,
            system,
            int(os.getenv("OPENROUTER_DISCREPANCY_AGENT_MAX_TOKENS", os.getenv("OPENROUTER_AGENT_MAX_TOKENS", str(DEFAULT_AGENT_MAX_TOKENS)))),
        )

    def normalize_rows(self, category: dict[str, Any], result: dict[str, Any]) -> list[dict[str, Any]]:
        rows = result.get("rows", []) if isinstance(result, dict) else []
        if not isinstance(rows, list):
            rows = []
        normalized = []
        allowed_types = {"contradiction", "mismatch", "ambiguity", "precedence_issue", "missing_reference", "version_conflict"}
        allowed_severity = {"High", "Medium", "Low"}
        allowed_status = {"open", "needs_clarification", "resolved_by_corrigendum", "monitor"}
        for row in rows:
            if not isinstance(row, dict):
                continue
            discrepancy_type = str(row.get("discrepancy_type") or "ambiguity").strip()
            severity = str(row.get("severity") or "Medium").strip().title()
            status = str(row.get("status") or "needs_clarification").strip().lower()
            clean = {
                "discrepancy_category": row.get("discrepancy_category") or category["category"],
                "discrepancy_type": discrepancy_type if discrepancy_type in allowed_types else "ambiguity",
                "title": row.get("title") or "-",
                "source_a": row.get("source_a") or "-",
                "source_b": row.get("source_b") or "-",
                "contradiction_summary": row.get("contradiction_summary") or "-",
                "impact": row.get("impact") or "-",
                "severity": severity if severity in allowed_severity else "Medium",
                "recommended_resolution": row.get("recommended_resolution") or "-",
                "status": status if status in allowed_status else "needs_clarification",
                "evidence_citations": row.get("evidence_citations") if isinstance(row.get("evidence_citations"), list) else [],
            }
            normalized.append(clean)
        return normalized

    def save_partial_category(self, category: dict[str, Any], rows: list[dict[str, Any]], result: dict[str, Any], evidence: list[dict[str, Any]]) -> None:
        entry = {
            "category_id": category["id"],
            "title": category["category"],
            "status": "finalized",
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "row_count": len(rows),
            "rows": rows,
            "category_note": result.get("category_note", ""),
            "evidence": compact_evidence(evidence, limit=MAX_EVIDENCE),
        }
        with self.partial_lock:
            partial = read_json(
                self.partial_path,
                {
                    "report_type": "discrepancy_report",
                    "status": "partial",
                    "project_id": self.project_root.name,
                    "categories": {},
                },
            )
            if not isinstance(partial, dict):
                partial = {"report_type": "discrepancy_report", "status": "partial", "project_id": self.project_root.name, "categories": {}}
            partial.setdefault("categories", {})
            partial["categories"][category["id"]] = entry
            partial["updated_at"] = entry["updated_at"]
            write_json(self.partial_path, partial)

    def answer_category(self, category: dict[str, Any]) -> dict[str, Any]:
        self.log(f"Starting discrepancy category: {category['category']}", category["id"])
        evidence = merge_evidence(self.keyword_evidence(category["terms"]) + self.searched_evidence(category["query"]))
        self.log(f"Collected {len(evidence)} evidence item(s) for discrepancy category.", category["id"])
        result = self.call_json(
            self.prompt(category, evidence),
            "Return only valid JSON. You are a senior tender reviewer identifying discrepancies and contradictions between contract documents.",
        )
        rows = self.normalize_rows(category, result)
        if not rows:
            self.log("Initial discrepancy pass returned 0 rows; retrying as open-issue discrepancy register.", category["id"])
            retry_prompt = self.prompt(category, evidence) + """

The previous pass returned no rows. That is too conservative for this workflow.
Return 3 to 6 open discrepancy rows from the supplied evidence. They may be hard contradictions, ambiguities, missing references, version conflicts, or unresolved scope/payment/approval mismatches. Keep the label honest; do not call an ambiguity a contradiction.
""".strip()
            result = self.call_json(
                retry_prompt,
                "Return only valid JSON. You are a senior tender reviewer creating an open discrepancy register for bid clarification.",
            )
            rows = self.normalize_rows(category, result)
        self.save_partial_category(category, rows, result, evidence)
        self.log(f"Finalized {len(rows)} discrepancy row(s) for {category['category']}.", category["id"])
        return {"category_id": category["id"], "title": category["category"], "rows": rows, "category_note": result.get("category_note", "")}

    def answer_category_with_retries(self, category: dict[str, Any]) -> dict[str, Any]:
        attempts = max(1, int(os.getenv("OPENROUTER_DISCREPANCY_CATEGORY_RETRIES", str(DEFAULT_DISCREPANCY_RETRIES))))
        last_error: Exception | None = None
        for attempt in range(1, attempts + 1):
            try:
                if attempt > 1:
                    self.log(f"Retrying discrepancy category attempt {attempt}/{attempts}.", category["id"], {"previous_error": str(last_error)})
                return self.answer_category(category)
            except Exception as exc:
                last_error = exc
                self.log(f"Discrepancy category attempt {attempt}/{attempts} failed: {exc}", category["id"], {"error_type": type(exc).__name__})
        raise RuntimeError(str(last_error or "Discrepancy category failed."))

    def generate(self) -> dict[str, Any]:
        self.log("Starting Discrepancy Register generation.")
        workers = min(max(1, int(os.getenv("OPENROUTER_DISCREPANCY_CATEGORY_WORKERS", str(DEFAULT_DISCREPANCY_WORKERS)))), len(DISCREPANCY_CATEGORIES))
        sections_by_id: dict[str, dict[str, Any]] = {}
        failures = []
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(self.answer_category_with_retries, category): category for category in DISCREPANCY_CATEGORIES}
            for future in as_completed(futures):
                category = futures[future]
                try:
                    sections_by_id[category["id"]] = future.result()
                except Exception as exc:
                    failures.append({"category_id": category["id"], "title": category["category"], "error": str(exc)})
                    self.log(f"Discrepancy category failed after retries: {category['category']}: {exc}", category["id"])
        if failures:
            write_json(
                self.progress_path,
                {
                    "report_type": "discrepancy_report",
                    "status": "failed",
                    "project_id": self.project_root.name,
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                    "failures": failures,
                    "logs": self.logs,
                },
            )
            raise RuntimeError(f"Discrepancy Register generation failed for {len(failures)} category/categories: {failures}")

        sections = [sections_by_id[category["id"]] for category in DISCREPANCY_CATEGORIES if category["id"] in sections_by_id]
        rows = []
        for section in sections:
            rows.extend(section.get("rows", []))
        for index, row in enumerate(rows, start=1):
            row["s_no"] = index

        report = {
            "report_type": "discrepancy_report",
            "status": "verified",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "project_id": self.project_root.name,
            "title": "Discrepancy and Contradiction Register",
            "columns": [
                "S. No.",
                "Discrepancy Category",
                "Type",
                "Title",
                "Source A",
                "Source B",
                "Discrepancy / Contradiction Summary",
                "Impact",
                "Severity",
                "Recommended Resolution",
                "Status",
                "Evidence",
            ],
            "rows": rows,
            "sections": sections,
            "logs": self.logs,
        }
        output_path = self.reports_dir / "discrepancy_report.json"
        write_json(output_path, report)
        self.log(f"Saved Discrepancy Register report to {output_path.name}.")
        report["logs"] = self.logs
        write_json(output_path, report)
        write_json(
            self.progress_path,
            {
                "report_type": "discrepancy_report",
                "status": "complete",
                "project_id": self.project_root.name,
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "logs": self.logs,
            },
        )
        return report


def generate_discrepancy_report(project_root: Path) -> dict[str, Any]:
    return DiscrepancyReportAgent(project_root).generate()


def load_discrepancy_report(project_root: Path) -> dict[str, Any]:
    return read_json(project_root / "reports" / "discrepancy_report.json", {})


def load_discrepancy_report_progress(project_root: Path) -> dict[str, Any]:
    return read_json(project_root / "reports" / "discrepancy_report.progress.json", {})


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate the tender discrepancy register for a project.")
    parser.add_argument("project_root", type=Path, help="Path to the indexed project directory.")
    args = parser.parse_args()
    result = generate_discrepancy_report(args.project_root)
    print(json.dumps({"status": result.get("status", "complete"), "rows": len(result.get("rows", []))}, ensure_ascii=False))
