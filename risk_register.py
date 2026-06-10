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


MAX_EVIDENCE = 26
DEFAULT_RISK_WORKERS = 4
DEFAULT_RISK_RETRIES = 3


RISK_CATEGORIES = [
    {
        "id": "technical_operational",
        "risk_category": "TECHNICAL AND OPERATIONAL RISKS",
        "risk_type": "Technical Risk",
        "title": "Technical, Design, Performance, and O&M Risks",
        "query": "technical risk design basis drawings specifications WTP pumping pipeline hydraulic surge SCADA O&M performance testing commissioning existing assets",
        "terms": [
            "design basis",
            "drawings",
            "specification",
            "hydraulic",
            "surge",
            "water hammer",
            "SCADA",
            "commissioning",
            "performance test",
            "O&M",
            "existing assets",
            "clarifier",
            "filter",
            "pumping",
            "pipeline",
        ],
    },
    {
        "id": "planning_schedule",
        "risk_category": "PLANNING AND SCHEDULE RISKS",
        "risk_type": "Planning Risk",
        "title": "Programme, Milestone, Interface, and Delay Risks",
        "query": "planning risk completion period milestone programme schedule delay EOT interface access handover commissioning approvals",
        "terms": [
            "completion period",
            "milestone",
            "programme",
            "delay",
            "EOT",
            "site access",
            "handover",
            "interface",
            "commissioning",
        ],
    },
    {
        "id": "financial_commercial",
        "risk_category": "FINANCIAL AND COMMERCIAL RISKS",
        "risk_type": "Financial and Fiscal Risk",
        "title": "Cash Flow, Payment, Price, Tax, and Security Risks",
        "query": "financial risk payment schedule cash flow price escalation GST tax bid security performance security retention advance payment currency BOQ",
        "terms": [
            "payment",
            "cash flow",
            "price escalation",
            "GST",
            "tax",
            "bid security",
            "performance security",
            "retention",
            "advance payment",
            "currency",
            "BOQ",
        ],
    },
    {
        "id": "legal_contractual",
        "risk_category": "LEGAL AND CONTRACTUAL RISKS",
        "risk_type": "Contractual Risk",
        "title": "Liability, Indemnity, Change, Claims, and Dispute Risks",
        "query": "contractual risk liability indemnity consequential damages liquidated damages termination change in law variation claim dispute arbitration subcontracting",
        "terms": [
            "liability",
            "indemnity",
            "consequential",
            "liquidated damages",
            "termination",
            "change in law",
            "variation",
            "claim",
            "dispute",
            "arbitration",
            "subcontracting",
        ],
    },
    {
        "id": "permits_land_utilities",
        "risk_category": "PERMITS, LAND, AND UTILITY RISKS",
        "risk_type": "Client / Interface Risk",
        "title": "Right of Way, Approvals, Utilities, and Site Constraint Risks",
        "query": "right of way land acquisition permissions approvals permits utility shifting HT line electrical substation existing pipes encumbrance site constraints",
        "terms": [
            "right of way",
            "land acquisition",
            "permission",
            "approval",
            "permit",
            "utility shifting",
            "HT line",
            "substation",
            "existing pipe",
            "encumbrance",
        ],
    },
    {
        "id": "environment_hse_social",
        "risk_category": "ENVIRONMENTAL, HSE, AND SOCIAL RISKS",
        "risk_type": "Environmental / HSE Risk",
        "title": "Sludge, Disposal, Safety, Community, and Environmental Risks",
        "query": "environmental risk HSE safety sludge disposal spoil disposal pollution consent community traffic labour camp hazardous waste",
        "terms": [
            "sludge",
            "disposal",
            "environment",
            "pollution",
            "consent",
            "safety",
            "HSE",
            "community",
            "traffic",
            "hazardous",
        ],
    },
    {
        "id": "procurement_supply_chain",
        "risk_category": "PROCUREMENT AND SUPPLY CHAIN RISKS",
        "risk_type": "Procurement Risk",
        "title": "Vendor, Equipment, Import, Spare, and Long-Lead Risks",
        "query": "procurement risk vendor approval equipment supply long lead imported items spares make list manufacturer inspection delivery",
        "terms": [
            "vendor",
            "manufacturer",
            "make list",
            "approval",
            "equipment",
            "long lead",
            "import",
            "spares",
            "inspection",
            "delivery",
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
        timeout=float(os.getenv("OPENROUTER_TIMEOUT_SECONDS", "180")),
        max_retries=int(os.getenv("OPENROUTER_MAX_RETRIES", "2")),
        default_headers={
            "HTTP-Referer": os.getenv("OPENROUTER_SITE_URL", "http://localhost"),
            "X-Title": os.getenv("OPENROUTER_APP_NAME", "PDF Vision RAG"),
        },
    )
    model = os.getenv(
        "OPENROUTER_RISK_AGENT_MODEL",
        os.getenv("OPENROUTER_COMMERCIAL_AGENT_MODEL", os.getenv("OPENROUTER_MODEL", DEFAULT_MODEL)),
    )
    return client, model


class RiskRegisterAgent:
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
        self.progress_path = self.reports_dir / "risk_register.progress.json"
        self.partial_path = self.reports_dir / "risk_register.partial.json"

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
                    "report_type": "risk_register",
                    "status": "running",
                    "project_id": self.project_root.name,
                    "updated_at": entry["created_at"],
                    "logs": list(self.logs),
                },
            )

    def keyword_evidence(self, terms: list[str], limit: int = 24) -> list[dict[str, Any]]:
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
            topic_to_evidence(topic, "keyword_risk_scan", f"Matched {score} risk keyword(s).")
            for score, topic in scored[:limit]
        ]

    def searched_evidence(self, query: str, max_hits: int = 14) -> list[dict[str, Any]]:
        os.environ["PDF_VISION_RAG_ROOT"] = str(self.project_root)
        os.environ.setdefault("OPENROUTER_SEARCH_MODEL", os.getenv("OPENROUTER_SEARCH_MODEL", DEFAULT_SEARCH_MODEL))
        dry_run = os.getenv("OPENROUTER_RISK_SEARCH_DRY_RUN", "1").strip().lower() in {"1", "true", "yes", "on"}
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
                evidence.append(topic_to_evidence(topic, "risk_search", hit.get("relevance_reason", "Found by targeted risk search.")))
        return evidence

    def prompt(self, category: dict[str, Any], evidence: list[dict[str, Any]]) -> str:
        return f"""
Create a project risk register section for this risk category.

Risk category:
{json.dumps(category, ensure_ascii=False)}

Evidence excerpts:
{json.dumps(compact_evidence(evidence, limit=MAX_EVIDENCE), ensure_ascii=False)}

Output rows must be relevant to this project's tender evidence, but do not copy any example risk list.

Scoring:
- occurrence: 1 low / 2 medium / 3 high before mitigation
- impact: 1 low / 2 medium / 3 high before mitigation
- residual_occurrence: expected occurrence after action plan
- residual_impact: expected impact after action plan

Return valid JSON only:
{{
  "rows": [
    {{
      "risk_category": "TECHNICAL AND OPERATIONAL RISKS",
      "risk_type": "Technical Risk",
      "risk_title": "short title / sub-category",
      "risk_description": "specific risk and why it exists in this tender",
      "occurrence": 1,
      "impact": 3,
      "achieved_actions": "what is already known, assumed, clarified, priced, excluded, or included",
      "action_plan": "clear mitigation action + responsible party + timing/deadline",
      "residual_occurrence": 1,
      "residual_impact": 2,
      "basis": "documented | inferred_from_absence | commercial_assumption",
      "evidence_citations": [
        {{
          "document_name": "source document",
          "page_no": 1,
          "topic_name": "topic",
          "excerpt": "short evidence excerpt"
        }}
      ]
    }}
  ],
  "category_note": "short note on coverage and limits"
}}

Rules:
- Produce 4 to 8 high-value rows for this category if evidence supports them.
- Prefer risks that affect cost, programme, technical compliance, legal exposure, bid assumptions, O&M, approvals, utilities, interfaces, or claim exposure.
- Include absence risks when normal tender inputs are missing or unclear.
- Do not invent exact amounts unless evidence supports them.
- Keep actions realistic for bid stage and post-award stage.
- Do not mention the screenshot or template.
""".strip()

    def call_json(self, prompt: str, system: str) -> dict[str, Any]:
        return json_completion(
            self.client,
            self.model,
            prompt,
            system,
            int(os.getenv("OPENROUTER_RISK_AGENT_MAX_TOKENS", os.getenv("OPENROUTER_AGENT_MAX_TOKENS", str(DEFAULT_AGENT_MAX_TOKENS)))),
        )

    def save_partial_category(self, category: dict[str, Any], result: dict[str, Any], evidence: list[dict[str, Any]]) -> None:
        rows = result.get("rows", []) if isinstance(result, dict) else []
        if not isinstance(rows, list):
            rows = []
        entry = {
            "category_id": category["id"],
            "title": category["title"],
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
                    "report_type": "risk_register",
                    "status": "partial",
                    "project_id": self.project_root.name,
                    "categories": {},
                },
            )
            if not isinstance(partial, dict):
                partial = {"report_type": "risk_register", "status": "partial", "project_id": self.project_root.name, "categories": {}}
            partial.setdefault("categories", {})
            partial["categories"][category["id"]] = entry
            partial["updated_at"] = entry["updated_at"]
            write_json(self.partial_path, partial)

    def answer_category(self, category: dict[str, Any]) -> dict[str, Any]:
        self.log(f"Starting risk category: {category['title']}", category["id"])
        evidence = merge_evidence(self.keyword_evidence(category["terms"]) + self.searched_evidence(category["query"]))
        self.log(f"Collected {len(evidence)} evidence item(s) for risk category.", category["id"])
        result = self.call_json(
            self.prompt(category, evidence),
            "Return only valid JSON. You are a senior tender risk manager creating a bid risk register.",
        )
        rows = result.get("rows", [])
        if not isinstance(rows, list):
            rows = []
        normalized_rows = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            clean = {
                "risk_category": row.get("risk_category") or category["risk_category"],
                "risk_type": row.get("risk_type") or category["risk_type"],
                "risk_title": row.get("risk_title") or row.get("title") or "-",
                "risk_description": row.get("risk_description") or "-",
                "occurrence": int(row.get("occurrence") or 2),
                "impact": int(row.get("impact") or 2),
                "achieved_actions": row.get("achieved_actions") or "-",
                "action_plan": row.get("action_plan") or "-",
                "residual_occurrence": int(row.get("residual_occurrence") or 1),
                "residual_impact": int(row.get("residual_impact") or 1),
                "basis": row.get("basis") or "documented",
                "evidence_citations": row.get("evidence_citations") if isinstance(row.get("evidence_citations"), list) else [],
            }
            for key in ["occurrence", "impact", "residual_occurrence", "residual_impact"]:
                clean[key] = min(3, max(1, int(clean[key])))
            normalized_rows.append(clean)
        result["rows"] = normalized_rows
        self.save_partial_category(category, result, evidence)
        self.log(f"Finalized {len(normalized_rows)} risk row(s) for {category['title']}.", category["id"])
        return {"category_id": category["id"], "title": category["title"], "rows": normalized_rows, "category_note": result.get("category_note", "")}

    def answer_category_with_retries(self, category: dict[str, Any]) -> dict[str, Any]:
        attempts = max(1, int(os.getenv("OPENROUTER_RISK_CATEGORY_RETRIES", str(DEFAULT_RISK_RETRIES))))
        last_error: Exception | None = None
        for attempt in range(1, attempts + 1):
            try:
                if attempt > 1:
                    self.log(f"Retrying risk category attempt {attempt}/{attempts}.", category["id"], {"previous_error": str(last_error)})
                return self.answer_category(category)
            except Exception as exc:
                last_error = exc
                self.log(f"Risk category attempt {attempt}/{attempts} failed: {exc}", category["id"], {"error_type": type(exc).__name__})
        raise RuntimeError(str(last_error or "Risk category failed."))

    def generate(self) -> dict[str, Any]:
        self.log("Starting Risk Register generation.")
        workers = min(max(1, int(os.getenv("OPENROUTER_RISK_CATEGORY_WORKERS", str(DEFAULT_RISK_WORKERS)))), len(RISK_CATEGORIES))
        sections_by_id: dict[str, dict[str, Any]] = {}
        failures = []
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(self.answer_category_with_retries, category): category for category in RISK_CATEGORIES}
            for future in as_completed(futures):
                category = futures[future]
                try:
                    sections_by_id[category["id"]] = future.result()
                except Exception as exc:
                    failures.append({"category_id": category["id"], "title": category["title"], "error": str(exc)})
                    self.log(f"Risk category failed after retries: {category['title']}: {exc}", category["id"])
        if failures:
            write_json(
                self.progress_path,
                {
                    "report_type": "risk_register",
                    "status": "failed",
                    "project_id": self.project_root.name,
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                    "failures": failures,
                    "logs": self.logs,
                },
            )
            raise RuntimeError(f"Risk Register generation failed for {len(failures)} category/categories: {failures}")

        sections = [sections_by_id[category["id"]] for category in RISK_CATEGORIES if category["id"] in sections_by_id]
        rows = []
        for section in sections:
            for row in section.get("rows", []):
                rows.append(row)
        for index, row in enumerate(rows, start=1):
            row["s_no"] = index
        report = {
            "report_type": "risk_register",
            "status": "verified",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "project_id": self.project_root.name,
            "title": "Risk Register",
            "columns": [
                "S. No.",
                "Risk Category",
                "Risks",
                "Risk Title / sub-category",
                "Risk Description",
                "Occurrence",
                "Impact",
                "Achieved Actions",
                "Actions plan (action + resp. + deadline)",
                "Residual Occurrence",
                "Residual Impact",
                "Basis",
                "Evidence",
            ],
            "rows": rows,
            "sections": sections,
            "logs": self.logs,
        }
        output_path = self.reports_dir / "risk_register.json"
        write_json(output_path, report)
        self.log(f"Saved Risk Register report to {output_path.name}.")
        report["logs"] = self.logs
        write_json(output_path, report)
        write_json(
            self.progress_path,
            {
                "report_type": "risk_register",
                "status": "complete",
                "project_id": self.project_root.name,
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "logs": self.logs,
            },
        )
        return report


def generate_risk_register(project_root: Path) -> dict[str, Any]:
    return RiskRegisterAgent(project_root).generate()


def load_risk_register(project_root: Path) -> dict[str, Any]:
    return read_json(project_root / "reports" / "risk_register.json", {})


def load_risk_register_progress(project_root: Path) -> dict[str, Any]:
    return read_json(project_root / "reports" / "risk_register.progress.json", {})


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate the tender risk register for a project.")
    parser.add_argument("project_root", type=Path, help="Path to the indexed project directory.")
    args = parser.parse_args()
    result = generate_risk_register(args.project_root)
    print(json.dumps({"status": result.get("status", "complete"), "rows": len(result.get("rows", []))}, ensure_ascii=False))
