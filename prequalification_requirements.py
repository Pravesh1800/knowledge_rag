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


MAX_EVIDENCE = 28
DEFAULT_PREQUAL_WORKERS = 7
DEFAULT_PREQUAL_PARALLEL_RETRIES = 3


PREQUALIFICATION_SECTIONS = [
    {
        "id": "bidder_identity_eligibility",
        "title": "Bidder Identity and Eligibility",
        "query": "pre-qualification eligibility bidder legal entity joint venture consortium lead member parent company affiliate blacklisting conflict of interest registration",
        "terms": [
            "pre-qualification",
            "eligibility",
            "bidder",
            "joint venture",
            "consortium",
            "lead member",
            "parent company",
            "affiliate",
            "blacklisting",
            "conflict of interest",
        ],
    },
    {
        "id": "technical_experience",
        "title": "Technical Experience",
        "query": "qualification criteria technical experience similar work construction commissioning completed works water supply treatment pipeline pumping reservoirs experience certificate",
        "terms": [
            "qualification criteria",
            "technical experience",
            "similar work",
            "commissioning",
            "completed",
            "work order",
            "completion certificate",
            "water supply",
            "pipeline",
            "pumping",
            "reservoir",
        ],
    },
    {
        "id": "financial_capacity",
        "title": "Financial Capacity",
        "query": "prequalification financial capacity annual turnover net worth profitability solvency working capital audited balance sheet CA certificate",
        "terms": [
            "turnover",
            "net worth",
            "profitability",
            "solvency",
            "working capital",
            "audited",
            "balance sheet",
            "chartered accountant",
            "CA certificate",
            "financial capacity",
        ],
    },
    {
        "id": "operation_maintenance_experience",
        "title": "Operation and Maintenance Experience",
        "query": "prequalification operation maintenance O&M experience water treatment plant pumping station distribution network service level operation period",
        "terms": [
            "operation",
            "maintenance",
            "O&M",
            "operate and maintain",
            "service level",
            "water treatment plant",
            "pumping station",
            "distribution network",
        ],
    },
    {
        "id": "key_personnel_equipment",
        "title": "Key Personnel, Plant, and Equipment",
        "query": "qualification key personnel project manager engineer manpower plant machinery equipment contractor resources CV deployment schedule",
        "terms": [
            "key personnel",
            "personnel",
            "project manager",
            "engineer",
            "manpower",
            "plant",
            "machinery",
            "equipment",
            "resources",
            "CV",
        ],
    },
    {
        "id": "submission_proofs_formats",
        "title": "Submission Proofs and Formats",
        "query": "prequalification documents forms annexures schedules certificates power of attorney undertaking affidavit notarized proof required bid submission format",
        "terms": [
            "annexure",
            "form",
            "schedule",
            "certificate",
            "power of attorney",
            "undertaking",
            "affidavit",
            "notarized",
            "proof",
            "bid submission",
        ],
    },
    {
        "id": "mandatory_compliance",
        "title": "Mandatory Compliance and Bid Security Preconditions",
        "query": "mandatory qualification compliance bid security EMD bid validity declarations tax GST PAN litigation debarment rejection non responsive",
        "terms": [
            "mandatory",
            "bid security",
            "EMD",
            "bid validity",
            "declaration",
            "GST",
            "PAN",
            "litigation",
            "debarment",
            "non-responsive",
            "rejection",
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
        "OPENROUTER_PREQUAL_AGENT_MODEL",
        os.getenv("OPENROUTER_LEGAL_AGENT_MODEL", os.getenv("OPENROUTER_MODEL", DEFAULT_MODEL)),
    )
    return client, model


class PrequalificationRequirementsAgent:
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
        self.topic_lookup = {topic.get("topic_name", ""): topic for topic in self.topics}
        load_dotenv(self.project_root)
        self.client, self.model = create_client()
        self.logs: list[dict[str, Any]] = []
        self.log_lock = threading.Lock()
        self.search_lock = threading.Lock()
        self.progress_path = self.reports_dir / "prequalification_requirements.progress.json"

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
                    "report_type": "prequalification_requirements",
                    "status": "running",
                    "project_id": self.project_root.name,
                    "updated_at": entry["created_at"],
                    "logs": list(self.logs),
                },
            )

    def searched_evidence(self, query: str, max_hits: int = 14) -> list[dict[str, Any]]:
        # searcher.py uses module-level project paths, so protect this setup while
        # all seven pre-qualification sections run in parallel.
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
                evidence.append(topic_to_evidence(topic, "searched", hit.get("relevance_reason", "Found by targeted pre-qualification search.")))
            for related in hit.get("related_topics", []):
                related_topic = by_triplet.get((related.get("topic_name"), related.get("document_name"), int(related.get("page_no") or 0)))
                if related_topic:
                    evidence.append(topic_to_evidence(related_topic, "searched", related.get("relation", "Related topic found by pre-qualification search.")))
        return evidence

    def keyword_evidence(self, terms: list[str], limit: int = 18) -> list[dict[str, Any]]:
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
            topic_to_evidence(topic, "keyword", f"Keyword lookup matched {score} pre-qualification term(s).")
            for score, topic in scored[:limit]
        ]

    def extraction_prompt(self, section: dict[str, Any], evidence: list[dict[str, Any]]) -> str:
        return f"""
You are extracting pre-qualification requirements from an indexed tender corpus.

Section to extract:
{json.dumps(section, ensure_ascii=False)}

Evidence:
{json.dumps(compact_evidence(evidence, limit=MAX_EVIDENCE), ensure_ascii=False)}

Return valid JSON only with:
{{
  "section_id": "{section["id"]}",
  "section_title": "{section["title"]}",
  "rows": [
    {{
      "requirement_type": "Eligibility|Technical|Financial|O&M|Personnel/Equipment|Submission Proof|Mandatory Compliance|Bid Security|Other",
      "requirement_area": "short label",
      "tender_vol_section": "exact section/volume if available, else -",
      "document_name": "source document name if clear, else -",
      "page_no": "source page number if clear, else -",
      "clause_no": "exact clause/reference if available, else -",
      "requirement_text": "the actual pre-qualification requirement in clear plain English",
      "threshold_or_value": "exact amount/percentage/years/capacity/count if stated, else -",
      "applicable_to": "single bidder/JV/lead member/member/parent/subcontractor/all bidders/unknown",
      "proof_required": "forms, certificates, statements, authorizations, or evidence required; - if not found",
      "compliance_note": "short note on how bidder should verify compliance or ambiguity",
      "citations": [
        {{"document_name": "", "page_no": 0, "topic_name": "", "excerpt": ""}}
      ],
      "confidence": "low|medium|high"
    }}
  ],
  "coverage_note": "what was searched and any gaps"
}}

Rules:
- Extract requirements only; do not write bidder questions.
- Do not invent thresholds, values, periods, or eligibility rules. Use "-" where the documents do not state the value.
- Every row needs at least one tender citation when possible.
- If evidence only hints at a requirement but lacks exact wording, keep confidence low and explain the missing wording in compliance_note.
- Merge duplicates inside this section.
- Preserve clause numbers, page numbers, document names, and requirement wording accurately.
""".strip()

    def verifier_prompt(self, section: dict[str, Any], draft: dict[str, Any], evidence: list[dict[str, Any]]) -> str:
        return f"""
Verify and clean this pre-qualification extraction.

Section:
{json.dumps(section, ensure_ascii=False)}

Draft:
{json.dumps(draft, ensure_ascii=False)}

Evidence:
{json.dumps(compact_evidence(evidence, limit=MAX_EVIDENCE), ensure_ascii=False)}

Return valid JSON only with:
{{
  "rows": [
    {{
      "requirement_type": "",
      "requirement_area": "",
      "tender_vol_section": "",
      "document_name": "",
      "page_no": "",
      "clause_no": "",
      "requirement_text": "",
      "threshold_or_value": "",
      "applicable_to": "",
      "proof_required": "",
      "compliance_note": "",
      "citations": [{{"document_name": "", "page_no": 0, "topic_name": "", "excerpt": ""}}],
      "confidence": "low|medium|high"
    }}
  ],
  "warnings": [],
  "coverage_note": ""
}}

Verification rules:
- Remove rows that are not actually pre-qualification, eligibility, qualification proof, or mandatory bid precondition.
- Fix unsupported values to "-" instead of guessing.
- Keep the final answer complete; do not shorten just because JSON repair or cleanup is needed.
- Prefer exact tender/source wording over polished paraphrase where thresholds are involved.
- Keep only distinct rows.
""".strip()

    def answer_section(self, section: dict[str, Any]) -> dict[str, Any]:
        self.log(f"Searching pre-qualification evidence for {section['title']}.", section["id"])
        searched = self.searched_evidence(section["query"])
        keyword = self.keyword_evidence(section["terms"])
        evidence = merge_evidence(searched + keyword)
        self.log(f"Collected {len(evidence)} evidence topic(s) for {section['title']}.", section["id"])
        draft = json_completion(
            self.client,
            self.model,
            self.extraction_prompt(section, evidence),
            "Return only valid JSON. You are a strict tender pre-qualification extraction specialist.",
            max_tokens=int(os.getenv("OPENROUTER_PREQUAL_MAX_TOKENS", str(DEFAULT_AGENT_MAX_TOKENS + 1024))),
        )
        verifier = json_completion(
            self.client,
            self.model,
            self.verifier_prompt(section, draft, evidence),
            "Return only valid JSON. You verify tender pre-qualification extractions against evidence.",
            max_tokens=int(os.getenv("OPENROUTER_PREQUAL_MAX_TOKENS", str(DEFAULT_AGENT_MAX_TOKENS + 1024))),
        )
        rows = verifier.get("rows", []) if isinstance(verifier.get("rows", []), list) else []
        self.log(f"Verifier finalized {len(rows)} pre-qualification row(s) for {section['title']}.", section["id"], {"warnings": verifier.get("warnings", [])})
        return {
            "section_id": section["id"],
            "title": section["title"],
            "rows": rows,
            "coverage_note": verifier.get("coverage_note") or draft.get("coverage_note", ""),
            "evidence": compact_evidence(evidence, limit=MAX_EVIDENCE),
            "verifier": verifier,
        }

    def answer_section_with_retries(self, section: dict[str, Any]) -> dict[str, Any]:
        attempts = max(1, int(os.getenv("OPENROUTER_PREQUAL_SECTION_RETRIES", str(DEFAULT_PREQUAL_PARALLEL_RETRIES))))
        last_error: Exception | None = None
        for attempt in range(1, attempts + 1):
            try:
                if attempt > 1:
                    self.log(f"Retrying pre-qualification section attempt {attempt}/{attempts}.", section["id"])
                return self.answer_section(section)
            except Exception as exc:
                last_error = exc
                error_text = str(exc).lower()
                self.log(f"Pre-qualification section attempt {attempt}/{attempts} failed: {exc}", section["id"], {"error": str(exc)})
                if "key limit exceeded" in error_text or "daily limit" in error_text or "error code: 403" in error_text:
                    self.log(
                        "Stopping retries for this section because OpenRouter returned a daily/key limit error.",
                        section["id"],
                        {"error": str(exc)},
                    )
                    break
        assert last_error is not None
        raise last_error

    def final_rows(self, sections: list[dict[str, Any]]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        seen: set[str] = set()
        for section in sections:
            for row in section.get("rows", []):
                key = "|".join(
                    str(row.get(field, "")).strip().lower()
                    for field in ["requirement_type", "requirement_area", "clause_no", "requirement_text", "threshold_or_value"]
                )
                if key in seen:
                    continue
                seen.add(key)
                rows.append(
                    {
                        "s_no": len(rows) + 1,
                        "section_id": section.get("section_id", ""),
                        "section_title": section.get("title", ""),
                        "requirement_type": row.get("requirement_type", "Other"),
                        "requirement_area": row.get("requirement_area", "-"),
                        "tender_vol_section": row.get("tender_vol_section", "-"),
                        "document_name": row.get("document_name", "-"),
                        "page_no": row.get("page_no", "-"),
                        "clause_no": row.get("clause_no", "-"),
                        "requirement_text": row.get("requirement_text", ""),
                        "threshold_or_value": row.get("threshold_or_value", "-"),
                        "applicable_to": row.get("applicable_to", "-"),
                        "proof_required": row.get("proof_required", "-"),
                        "compliance_note": row.get("compliance_note", ""),
                        "citations": row.get("citations", []),
                        "confidence": row.get("confidence", "medium"),
                    }
                )
        return rows

    def generate(self) -> dict[str, Any]:
        self.log("Starting Pre-Qualification Requirements generation.")
        workers = max(1, int(os.getenv("OPENROUTER_PREQUAL_WORKERS", str(DEFAULT_PREQUAL_WORKERS))))
        workers = min(workers, len(PREQUALIFICATION_SECTIONS))
        self.log(f"Running pre-qualification sections with {workers} parallel worker(s).")
        sections_by_id: dict[str, dict[str, Any]] = {}
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(self.answer_section_with_retries, section): section for section in PREQUALIFICATION_SECTIONS}
            for future in as_completed(futures):
                section = futures[future]
                sections_by_id[section["id"]] = future.result()
        sections = [
            sections_by_id[section["id"]]
            for section in PREQUALIFICATION_SECTIONS
            if section["id"] in sections_by_id
        ]
        rows = self.final_rows(sections)
        report = {
            "report_type": "prequalification_requirements",
            "status": "verified",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "project_id": self.project_root.name,
            "title": "Pre-Qualification Requirements",
            "generation_settings": {
                "section_workers": workers,
                "sections": [section["id"] for section in PREQUALIFICATION_SECTIONS],
            },
            "columns": [
                "S. No.",
                "Requirement Type",
                "Requirement Area",
                "Tender Vol. / Section",
                "Document Name",
                "Page No.",
                "Clause No.",
                "Requirement Text",
                "Threshold / Value",
                "Applicable To",
                "Proof Required",
                "Compliance Note",
                "Confidence",
            ],
            "rows": rows,
            "sections": sections,
            "logs": self.logs,
        }
        output_path = self.reports_dir / "prequalification_requirements.json"
        write_json(output_path, report)
        self.log(f"Saved Pre-Qualification Requirements report to {output_path.name}.")
        report["logs"] = self.logs
        write_json(output_path, report)
        write_json(
            self.progress_path,
            {
                "report_type": "prequalification_requirements",
                "status": "complete",
                "project_id": self.project_root.name,
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "logs": self.logs,
            },
        )
        return report


def generate_prequalification_requirements(project_root: Path) -> dict[str, Any]:
    return PrequalificationRequirementsAgent(project_root).generate()


def load_prequalification_requirements(project_root: Path) -> dict[str, Any]:
    return read_json(project_root / "reports" / "prequalification_requirements.json", {})


def load_prequalification_requirements_progress(project_root: Path) -> dict[str, Any]:
    return read_json(project_root / "reports" / "prequalification_requirements.progress.json", {})


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate the Pre-Qualification Requirements report for a project.")
    parser.add_argument("project_root", type=Path, help="Path to the indexed project directory.")
    args = parser.parse_args()
    result = generate_prequalification_requirements(args.project_root)
    print(json.dumps({"status": result.get("status", "complete"), "rows": len(result.get("rows", []))}, ensure_ascii=False))
