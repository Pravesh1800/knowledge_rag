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
DEFAULT_KEY_INFO_WORKERS = 8
DEFAULT_KEY_INFO_RETRIES = 3


KEY_INFO_GROUPS = [
    {
        "id": "client_consultants_entities",
        "title": "Client, Consultants, and Bidding Entity",
        "fields": ["Client", "Consultant", "Project Management Consultant", "Bidding Entity", "JV / Consortium"],
        "query": "client employer consultant project management consultant bidder bidding entity JV consortium tender inviting authority",
        "terms": ["client", "employer", "consultant", "project management", "bidder", "JV", "consortium", "tender inviting authority"],
    },
    {
        "id": "contract_duration_model",
        "title": "Contractual Duration and Contract Model",
        "fields": ["Design-Build Duration", "O&M Duration", "Contract Model", "Conditions of Contract"],
        "query": "contract duration design build period operation maintenance O&M years DBO conditions of contract GCC PCC",
        "terms": ["duration", "design-build", "O&M", "operation", "maintenance", "DBO", "GCC", "PCC", "conditions of contract"],
    },
    {
        "id": "estimate_and_payment",
        "title": "Client Estimate and Payment",
        "fields": ["Client's Estimate", "DB Estimate", "O&M Estimate", "O&M Payment Frequency", "Payment Basis"],
        "query": "client estimate contract value DB cost O&M cost payment frequency quarterly monthly payment basis price schedule",
        "terms": ["estimate", "contract value", "DB cost", "O&M cost", "payment", "quarterly", "monthly", "price schedule"],
    },
    {
        "id": "financing_and_currency",
        "title": "Financing and Currency",
        "fields": ["Financing", "Funding Source", "Currency", "Foreign Currency", "Exchange Rate"],
        "query": "financing funding source government AMRUT municipal currency INR foreign currency exchange rate tender",
        "terms": ["financing", "funding", "AMRUT", "currency", "INR", "foreign currency", "exchange rate"],
    },
    {
        "id": "price_adjustment_escalation",
        "title": "Price Adjustment / Escalation",
        "fields": ["DB Price Adjustment", "O&M Price Adjustment", "Escalation Rate", "Price Adjustment Formula", "Indices"],
        "query": "price adjustment escalation fixed escalation O&M price adjustment formula indices DB price adjustment change of cost",
        "terms": ["price adjustment", "escalation", "indices", "formula", "change of cost", "fixed escalation"],
    },
    {
        "id": "key_dates",
        "title": "Key Dates",
        "fields": ["Tender Issue Date", "Pre-Bid Meeting", "Bid Submission Date", "Bid Validity", "Clarification Deadline", "Anticipated CIF"],
        "query": "tender issue date pre-bid meeting bid submission date bid validity clarification deadline CIF anticipated key dates",
        "terms": ["tender issue", "pre-bid", "bid submission", "bid validity", "clarification", "date", "deadline", "CIF"],
    },
    {
        "id": "anticipated_competition",
        "title": "Anticipated Competition",
        "fields": ["Anticipated Competition", "Competitors", "Potential Bidders"],
        "query": "anticipated competition competitors potential bidders L&T Triveni NCC Jindal Patel engineering tender",
        "terms": ["competition", "competitor", "potential bidder", "L&T", "Triveni", "NCC", "Jindal", "Patel"],
    },
    {
        "id": "scope_snapshot",
        "title": "Scope Snapshot",
        "fields": ["Major DB Scope", "Major O&M Scope", "Rehabilitation Scope", "Major Capacities"],
        "query": "major scope design build O&M rehabilitation WTP intake pump house pipeline capacity MLD scope snapshot",
        "terms": ["scope", "design", "build", "O&M", "rehabilitation", "WTP", "intake", "pipeline", "MLD"],
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
        "OPENROUTER_KEY_INFORMATION_MODEL",
        os.getenv("OPENROUTER_LEGAL_AGENT_MODEL", os.getenv("OPENROUTER_MODEL", DEFAULT_MODEL)),
    )
    return client, model


class KeyInformationAgent:
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
        self.progress_path = self.reports_dir / "key_information.progress.json"

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
                    "report_type": "key_information",
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
                evidence.append(topic_to_evidence(topic, "searched", hit.get("relevance_reason", "Found by key information search.")))
            for related in hit.get("related_topics", []):
                related_topic = by_triplet.get((related.get("topic_name"), related.get("document_name"), int(related.get("page_no") or 0)))
                if related_topic:
                    evidence.append(topic_to_evidence(related_topic, "searched", related.get("relation", "Related key information topic.")))
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
            topic_to_evidence(topic, "keyword", f"Keyword lookup matched {score} key-information term(s).")
            for score, topic in scored[:limit]
        ]

    def extraction_prompt(self, group: dict[str, Any], evidence: list[dict[str, Any]]) -> str:
        return f"""
You are extracting a Key Information fact sheet from an indexed tender corpus.

The output should look like a management slide with short labels and exact values.
Do not write paragraphs. Do not infer from industry norms unless the field is clearly marked "not found".

Group to extract:
{json.dumps(group, ensure_ascii=False)}

Evidence:
{json.dumps(compact_evidence(evidence, limit=MAX_EVIDENCE), ensure_ascii=False)}

Return valid JSON only:
{{
  "group_id": "{group["id"]}",
  "group_title": "{group["title"]}",
  "items": [
    {{
      "label": "one of the requested fields or a closely related field",
      "values": ["short value", "second value if needed"],
      "status": "found|not_found|ambiguous",
      "notes": "short note only if needed",
      "citations": [{{"document_name": "", "page_no": 0, "topic_name": "", "excerpt": ""}}],
      "confidence": "low|medium|high"
    }}
  ],
  "coverage_note": "what was searched and any gaps"
}}

Rules:
- Extract the requested fields when present.
- A field may have multiple values, for example DB and O&M durations, multiple consultants, or several key dates.
- If not found, include the field with status "not_found", values [], and notes explaining what was searched.
- For anticipated competition, only list names if explicitly found in documents. Do not invent competitors.
- Every found or ambiguous item needs at least one citation.
- Keep values slide-short, for example "INR", "Quarterly", "DB: 2.5 years", "O&M: 10 years".
""".strip()

    def verifier_prompt(self, group: dict[str, Any], draft: dict[str, Any], evidence: list[dict[str, Any]]) -> str:
        return f"""
Verify this Key Information extraction against the evidence.

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
      "values": [],
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
- Remove unsupported values.
- Convert unsupported values to status "not_found" instead of guessing.
- Keep exact dates, durations, amounts, currencies, percentages, and entity names.
- Do not make project-specific assumptions from the example screenshot.
- Keep the result complete and fact-sheet friendly.
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
                self.log(f"Key Information {stage} JSON repair succeeded on attempt {attempt}/3.", group_id)
                return fixed
            except Exception as exc:
                repair_error = str(exc)
                self.log(f"Key Information {stage} JSON repair attempt {attempt}/3 failed.", group_id, {"error": repair_error})
        raise ValueError(f"Key Information {stage} returned malformed JSON and repair failed: {repair_error}")

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
            self.log(f"Key Information {stage} returned malformed JSON; repairing without restarting.", group_id, {"error": str(exc)})
            return self.repair_json(content, str(exc), max_tokens, group_id, stage)

    def answer_group(self, group: dict[str, Any]) -> dict[str, Any]:
        self.log(f"Searching key-information evidence for {group['title']}.", group["id"])
        evidence = merge_evidence(self.searched_evidence(group["query"]) + self.keyword_evidence(group["terms"]))
        self.log(f"Collected {len(evidence)} evidence topic(s) for {group['title']}.", group["id"])
        max_tokens = int(os.getenv("OPENROUTER_KEY_INFORMATION_MAX_TOKENS", str(DEFAULT_AGENT_MAX_TOKENS + 1024)))
        draft = self.call_json(
            self.extraction_prompt(group, evidence),
            "Return only valid JSON. You extract concise key-information facts from tender evidence.",
            max_tokens,
            group["id"],
            "extraction",
        )
        verified = self.call_json(
            self.verifier_prompt(group, draft, evidence),
            "Return only valid JSON. You verify key-information facts against evidence.",
            max_tokens,
            group["id"],
            "verifier",
        )
        items = verified.get("items", []) if isinstance(verified.get("items", []), list) else []
        self.log(f"Verifier finalized {len(items)} Key Information item(s) for {group['title']}.", group["id"], {"warnings": verified.get("warnings", [])})
        return {
            "group_id": group["id"],
            "title": group["title"],
            "items": items,
            "coverage_note": verified.get("coverage_note") or draft.get("coverage_note", ""),
            "evidence": compact_evidence(evidence, limit=MAX_EVIDENCE),
            "verifier": verified,
        }

    def answer_group_with_retries(self, group: dict[str, Any]) -> dict[str, Any]:
        attempts = max(1, int(os.getenv("OPENROUTER_KEY_INFORMATION_SECTION_RETRIES", str(DEFAULT_KEY_INFO_RETRIES))))
        last_error: Exception | None = None
        for attempt in range(1, attempts + 1):
            try:
                if attempt > 1:
                    self.log(f"Retrying Key Information group attempt {attempt}/{attempts}.", group["id"])
                return self.answer_group(group)
            except Exception as exc:
                last_error = exc
                self.log(f"Key Information group attempt {attempt}/{attempts} failed: {exc}", group["id"], {"error": str(exc)})
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
                        "values": item.get("values", []),
                        "status": item.get("status", "not_found"),
                        "notes": item.get("notes", ""),
                        "citations": item.get("citations", []),
                        "confidence": item.get("confidence", "medium"),
                    }
                )
        return rows

    def generate(self) -> dict[str, Any]:
        self.log("Starting Key Information generation.")
        workers = max(1, int(os.getenv("OPENROUTER_KEY_INFORMATION_WORKERS", str(DEFAULT_KEY_INFO_WORKERS))))
        workers = min(workers, len(KEY_INFO_GROUPS))
        self.log(f"Running Key Information groups with {workers} parallel worker(s).")
        groups_by_id: dict[str, dict[str, Any]] = {}
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(self.answer_group_with_retries, group): group for group in KEY_INFO_GROUPS}
            for future in as_completed(futures):
                group = futures[future]
                groups_by_id[group["id"]] = future.result()
        groups = [groups_by_id[group["id"]] for group in KEY_INFO_GROUPS if group["id"] in groups_by_id]
        rows = self.final_rows(groups)
        report = {
            "report_type": "key_information",
            "status": "verified",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "project_id": self.project_root.name,
            "title": "Key Information",
            "generation_settings": {
                "group_workers": workers,
                "groups": [group["id"] for group in KEY_INFO_GROUPS],
            },
            "columns": ["S. No.", "Group", "Label", "Values", "Status", "Notes", "Confidence"],
            "rows": rows,
            "groups": groups,
            "logs": self.logs,
        }
        output_path = self.reports_dir / "key_information.json"
        write_json(output_path, report)
        self.log(f"Saved Key Information report to {output_path.name}.")
        report["logs"] = self.logs
        write_json(output_path, report)
        write_json(
            self.progress_path,
            {
                "report_type": "key_information",
                "status": "complete",
                "project_id": self.project_root.name,
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "logs": self.logs,
            },
        )
        return report


def generate_key_information(project_root: Path) -> dict[str, Any]:
    return KeyInformationAgent(project_root).generate()


def load_key_information(project_root: Path) -> dict[str, Any]:
    return read_json(project_root / "reports" / "key_information.json", {})


def load_key_information_progress(project_root: Path) -> dict[str, Any]:
    return read_json(project_root / "reports" / "key_information.progress.json", {})


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate the Key Information report for a project.")
    parser.add_argument("project_root", type=Path, help="Path to the indexed project directory.")
    args = parser.parse_args()
    result = generate_key_information(args.project_root)
    print(json.dumps({"status": result.get("status", "complete"), "rows": len(result.get("rows", []))}, ensure_ascii=False))
