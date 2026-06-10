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


MAX_EVIDENCE = 28
DEFAULT_PROJECT_BACKGROUND_WORKERS = 7
DEFAULT_PROJECT_BACKGROUND_RETRIES = 3


PROJECT_BACKGROUND_SECTIONS = [
    {
        "id": "authority_and_project_context",
        "title": "Authority and Project Context",
        "query": "project background employer authority client municipal corporation project name water supply scheme city location objective overview",
        "terms": ["employer", "authority", "municipal", "corporation", "project", "water supply", "objective", "overview"],
    },
    {
        "id": "demand_and_service_need",
        "title": "Demand and Service Need",
        "query": "project background water demand projected population peak demand MLD future demand service level augmentation",
        "terms": ["demand", "population", "projected", "MLD", "service level", "augmentation", "water requirement"],
    },
    {
        "id": "water_sources_and_conveyance",
        "title": "Water Sources and Conveyance",
        "query": "water source river dam intake raw water pipeline clear water pump house conveyance pumping main distance",
        "terms": ["river", "dam", "source", "intake", "raw water", "clear water", "pipeline", "pump house", "conveyance"],
    },
    {
        "id": "existing_assets_and_prior_phases",
        "title": "Existing Assets and Prior Phases",
        "query": "existing water supply project phases old WTP capacity rehabilitation previous phase constructed year existing pump house reservoir",
        "terms": ["existing", "phase", "old", "rehabilitation", "WTP", "capacity", "constructed", "commissioned", "pump house"],
    },
    {
        "id": "proposed_works",
        "title": "Proposed Works",
        "query": "proposed works new WTP intake pump house booster pumping station break pressure tank clear water pumping SCADA GIS substation",
        "terms": ["proposed", "new", "WTP", "intake", "pump", "booster", "break pressure", "SCADA", "substation"],
    },
    {
        "id": "scope_and_contract_model",
        "title": "Scope and Contract Model",
        "query": "scope of work design build operate DBO O&M operation maintenance rehabilitation pipeline laying contract period",
        "terms": ["scope", "design", "build", "operate", "DBO", "O&M", "operation", "maintenance", "contract period"],
    },
    {
        "id": "key_figures_and_dates",
        "title": "Key Figures and Dates",
        "query": "project background key figures dates bid submission contract duration capacity MLD years milestone cost NPV WLC",
        "terms": ["date", "duration", "years", "MLD", "capacity", "cost", "NPV", "WLC", "milestone"],
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
        "OPENROUTER_PROJECT_BACKGROUND_MODEL",
        os.getenv("OPENROUTER_LEGAL_AGENT_MODEL", os.getenv("OPENROUTER_MODEL", DEFAULT_MODEL)),
    )
    return client, model


class ProjectBackgroundAgent:
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
        self.progress_path = self.reports_dir / "project_background.progress.json"

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
                    "report_type": "project_background",
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
                evidence.append(topic_to_evidence(topic, "searched", hit.get("relevance_reason", "Found by project background search.")))
            for related in hit.get("related_topics", []):
                related_topic = by_triplet.get((related.get("topic_name"), related.get("document_name"), int(related.get("page_no") or 0)))
                if related_topic:
                    evidence.append(topic_to_evidence(related_topic, "searched", related.get("relation", "Related background topic.")))
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
            topic_to_evidence(topic, "keyword", f"Keyword lookup matched {score} project-background term(s).")
            for score, topic in scored[:limit]
        ]

    def extraction_prompt(self, section: dict[str, Any], evidence: list[dict[str, Any]]) -> str:
        return f"""
You are generating a factual Project Background document from an indexed tender corpus.

Target style:
- Board/management slide bullets like "Project Background".
- Short, factual, highly specific bullets.
- Include nested sub-bullets for prior phases, source systems, or component lists when useful.
- Do not write analysis, recommendations, risks, strategy, or bidder queries.

Section:
{json.dumps(section, ensure_ascii=False)}

Evidence:
{json.dumps(compact_evidence(evidence, limit=MAX_EVIDENCE), ensure_ascii=False)}

Return valid JSON only:
{{
  "section_id": "{section["id"]}",
  "section_title": "{section["title"]}",
  "bullets": [
    {{
      "bullet": "one concise factual bullet suitable for a slide",
      "sub_bullets": ["optional nested factual point", "optional nested factual point"],
      "key_figures": ["capacity/distance/year/date/value if supported"],
      "document_name": "primary source document name",
      "page_no": "primary source page",
      "citations": [{{"document_name": "", "page_no": 0, "topic_name": "", "excerpt": ""}}],
      "confidence": "low|medium|high"
    }}
  ],
  "coverage_note": "what was searched and any gaps"
}}

Rules:
- Only make factual claims supported by evidence.
- Prefer concrete numbers: capacities, years, dates, locations, source names, contract duration, project components.
- If evidence is not enough for this section, return fewer bullets and explain the gap in coverage_note.
- Every bullet needs at least one citation.
- Keep bullet text concise, but do not omit important supported facts.
- No project-specific visual styling instructions; content only.
""".strip()

    def verifier_prompt(self, section: dict[str, Any], draft: dict[str, Any], evidence: list[dict[str, Any]]) -> str:
        return f"""
Verify this Project Background section against the evidence.

Section:
{json.dumps(section, ensure_ascii=False)}

Draft:
{json.dumps(draft, ensure_ascii=False)}

Evidence:
{json.dumps(compact_evidence(evidence, limit=MAX_EVIDENCE), ensure_ascii=False)}

Return valid JSON only:
{{
  "bullets": [
    {{
      "bullet": "",
      "sub_bullets": [],
      "key_figures": [],
      "document_name": "",
      "page_no": "",
      "citations": [{{"document_name": "", "page_no": 0, "topic_name": "", "excerpt": ""}}],
      "confidence": "low|medium|high"
    }}
  ],
  "warnings": [],
  "coverage_note": ""
}}

Verification rules:
- Remove unsupported, generic, or strategy-like statements.
- Fix overclaims to exactly what the evidence supports.
- Keep factual background details; do not collapse everything into one broad summary.
- Preserve nested sub-bullets for asset phases, source systems, or scope component lists.
- Do not invent client names, project locations, source rivers, capacities, years, distances, or dates.
""".strip()

    def repair_json(self, malformed_content: str, parser_error: str, max_tokens: int, section_id: str, stage: str) -> dict[str, Any]:
        repair_content = malformed_content
        repair_error = parser_error
        for attempt in range(1, 4):
            repair_prompt = f"""
Repair this malformed model output into valid JSON only.

Exact parser error to fix:
{repair_error}

Requirements:
- valid JSON object only
- no markdown fences
- preserve all recoverable Project Background bullets, sub_bullets, figures, citations, and warnings
- do not summarize, shorten, or regenerate the answer

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
                self.log(f"Project Background {stage} JSON repair succeeded on attempt {attempt}/3.", section_id)
                return fixed
            except Exception as exc:
                repair_error = str(exc)
                self.log(f"Project Background {stage} JSON repair attempt {attempt}/3 failed.", section_id, {"error": repair_error})
        raise ValueError(f"Project Background {stage} returned malformed JSON and repair failed: {repair_error}")

    def call_json(self, prompt: str, system: str, max_tokens: int, section_id: str, stage: str) -> dict[str, Any]:
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
            self.log(f"Project Background {stage} returned malformed JSON; repairing without restarting.", section_id, {"error": str(exc)})
            return self.repair_json(content, str(exc), max_tokens, section_id, stage)

    def answer_section(self, section: dict[str, Any]) -> dict[str, Any]:
        self.log(f"Searching project-background evidence for {section['title']}.", section["id"])
        evidence = merge_evidence(self.searched_evidence(section["query"]) + self.keyword_evidence(section["terms"]))
        self.log(f"Collected {len(evidence)} evidence topic(s) for {section['title']}.", section["id"])
        max_tokens = int(os.getenv("OPENROUTER_PROJECT_BACKGROUND_MAX_TOKENS", str(DEFAULT_AGENT_MAX_TOKENS + 1024)))
        draft = self.call_json(
            self.extraction_prompt(section, evidence),
            "Return only valid JSON. You extract factual project-background slide bullets from tender evidence.",
            max_tokens,
            section["id"],
            "extraction",
        )
        verified = self.call_json(
            self.verifier_prompt(section, draft, evidence),
            "Return only valid JSON. You verify project-background bullets against evidence.",
            max_tokens,
            section["id"],
            "verifier",
        )
        bullets = verified.get("bullets", []) if isinstance(verified.get("bullets", []), list) else []
        self.log(f"Verifier finalized {len(bullets)} Project Background bullet(s) for {section['title']}.", section["id"], {"warnings": verified.get("warnings", [])})
        return {
            "section_id": section["id"],
            "title": section["title"],
            "bullets": bullets,
            "coverage_note": verified.get("coverage_note") or draft.get("coverage_note", ""),
            "evidence": compact_evidence(evidence, limit=MAX_EVIDENCE),
            "verifier": verified,
        }

    def answer_section_with_retries(self, section: dict[str, Any]) -> dict[str, Any]:
        attempts = max(1, int(os.getenv("OPENROUTER_PROJECT_BACKGROUND_SECTION_RETRIES", str(DEFAULT_PROJECT_BACKGROUND_RETRIES))))
        last_error: Exception | None = None
        for attempt in range(1, attempts + 1):
            try:
                if attempt > 1:
                    self.log(f"Retrying Project Background section attempt {attempt}/{attempts}.", section["id"])
                return self.answer_section(section)
            except Exception as exc:
                last_error = exc
                self.log(f"Project Background section attempt {attempt}/{attempts} failed: {exc}", section["id"], {"error": str(exc)})
                if any(term in str(exc).lower() for term in ["key limit exceeded", "daily limit", "error code: 403"]):
                    break
        assert last_error is not None
        raise last_error

    def final_rows(self, sections: list[dict[str, Any]]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for section in sections:
            for bullet in section.get("bullets", []):
                rows.append(
                    {
                        "s_no": len(rows) + 1,
                        "section_id": section.get("section_id", ""),
                        "section_title": section.get("title", ""),
                        "bullet": bullet.get("bullet", ""),
                        "sub_bullets": bullet.get("sub_bullets", []),
                        "key_figures": bullet.get("key_figures", []),
                        "document_name": bullet.get("document_name", "-"),
                        "page_no": bullet.get("page_no", "-"),
                        "citations": bullet.get("citations", []),
                        "confidence": bullet.get("confidence", "medium"),
                    }
                )
        return rows

    def generate(self) -> dict[str, Any]:
        self.log("Starting Project Background generation.")
        workers = max(1, int(os.getenv("OPENROUTER_PROJECT_BACKGROUND_WORKERS", str(DEFAULT_PROJECT_BACKGROUND_WORKERS))))
        workers = min(workers, len(PROJECT_BACKGROUND_SECTIONS))
        self.log(f"Running Project Background sections with {workers} parallel worker(s).")
        sections_by_id: dict[str, dict[str, Any]] = {}
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(self.answer_section_with_retries, section): section for section in PROJECT_BACKGROUND_SECTIONS}
            for future in as_completed(futures):
                section = futures[future]
                sections_by_id[section["id"]] = future.result()
        sections = [sections_by_id[section["id"]] for section in PROJECT_BACKGROUND_SECTIONS if section["id"] in sections_by_id]
        rows = self.final_rows(sections)
        report = {
            "report_type": "project_background",
            "status": "verified",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "project_id": self.project_root.name,
            "title": "Project Background",
            "generation_settings": {
                "section_workers": workers,
                "sections": [section["id"] for section in PROJECT_BACKGROUND_SECTIONS],
            },
            "columns": [
                "S. No.",
                "Section",
                "Bullet",
                "Sub Bullets",
                "Key Figures",
                "Document Name",
                "Page No.",
                "Confidence",
            ],
            "rows": rows,
            "sections": sections,
            "logs": self.logs,
        }
        output_path = self.reports_dir / "project_background.json"
        write_json(output_path, report)
        self.log(f"Saved Project Background report to {output_path.name}.")
        report["logs"] = self.logs
        write_json(output_path, report)
        write_json(
            self.progress_path,
            {
                "report_type": "project_background",
                "status": "complete",
                "project_id": self.project_root.name,
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "logs": self.logs,
            },
        )
        return report


def generate_project_background(project_root: Path) -> dict[str, Any]:
    return ProjectBackgroundAgent(project_root).generate()


def load_project_background(project_root: Path) -> dict[str, Any]:
    return read_json(project_root / "reports" / "project_background.json", {})


def load_project_background_progress(project_root: Path) -> dict[str, Any]:
    return read_json(project_root / "reports" / "project_background.progress.json", {})


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate the Project Background report for a project.")
    parser.add_argument("project_root", type=Path, help="Path to the indexed project directory.")
    args = parser.parse_args()
    result = generate_project_background(args.project_root)
    print(json.dumps({"status": result.get("status", "complete"), "rows": len(result.get("rows", []))}, ensure_ascii=False))
