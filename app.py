from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import fitz
from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from openai import OpenAI
from pydantic import BaseModel

from bid_process_evaluation import (
    generate_bid_process_evaluation,
    load_bid_process_evaluation,
    load_bid_process_evaluation_progress,
)
from commercial_strategy import (
    generate_commercial_strategy,
    load_commercial_strategy,
    load_commercial_strategy_progress,
)
from discrepancy_report import (
    generate_discrepancy_report,
    load_discrepancy_report,
    load_discrepancy_report_progress,
)
from financial_bonds import (
    generate_financial_bonds,
    load_financial_bonds,
    load_financial_bonds_progress,
)
from financial_liabilities_penalties import (
    generate_financial_liabilities_penalties,
    load_financial_liabilities_penalties,
    load_financial_liabilities_penalties_progress,
)
from key_information import (
    generate_key_information,
    load_key_information,
    load_key_information_progress,
)
from legal_assessment import (
    generate_legal_assessment,
    load_legal_assessment,
    load_legal_assessment_progress,
)
from prebid_queries import (
    generate_prebid_queries,
    load_prebid_queries,
    load_prebid_queries_progress,
)
from prequalification_requirements import (
    generate_prequalification_requirements,
    load_prequalification_requirements,
    load_prequalification_requirements_progress,
)
from project_background import (
    generate_project_background,
    load_project_background,
    load_project_background_progress,
)
from risk_register import (
    generate_risk_register,
    load_risk_register,
    load_risk_register_progress,
)


APP_ROOT = Path(__file__).resolve().parent
PROJECTS_ROOT = APP_ROOT / "projects"
STATIC_ROOT = APP_ROOT / "frontend"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "google/gemini-3.1-flash-lite"
PROJECT_LOCKS: dict[str, threading.Lock] = {}
PROJECT_LOCKS_GUARD = threading.Lock()


app = FastAPI(title="PDF Vision RAG")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory=STATIC_ROOT), name="static")


class ProjectCreate(BaseModel):
    name: str


class ChatRequest(BaseModel):
    question: str
    max_hits: int = 10
    history: list[dict[str, str]] = []


def project_lock(project_id: str) -> threading.Lock:
    with PROJECT_LOCKS_GUARD:
        if project_id not in PROJECT_LOCKS:
            PROJECT_LOCKS[project_id] = threading.Lock()
        return PROJECT_LOCKS[project_id]


def load_dotenv() -> None:
    env_path = APP_ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ[key.strip()] = value.strip().strip("\"'")


def slugify(value: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip()).strip("-").lower()
    return value or "project"


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    for attempt in range(4):
        try:
            return json.loads(path.read_text(encoding="utf-8-sig"))
        except json.JSONDecodeError:
            if attempt == 3:
                return default
            time.sleep(0.05)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f"{path.name}.tmp")
    temp_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    temp_path.replace(path)


def write_pipeline_progress(root: Path, data: dict[str, Any]) -> None:
    payload = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        **data,
    }
    write_json(root / "logs" / "pipeline_progress.json", payload)


def clear_generated_outputs(root: Path) -> None:
    for name in ("indexes", "reports"):
        path = root / name
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)


def project_path(project_id: str) -> Path:
    path = (PROJECTS_ROOT / project_id).resolve()
    if not str(path).startswith(str(PROJECTS_ROOT.resolve())):
        raise HTTPException(status_code=400, detail="Invalid project id")
    if not path.exists():
        raise HTTPException(status_code=404, detail="Project not found")
    return path


def project_meta_path(project_id: str) -> Path:
    return project_path(project_id) / "project.json"


def list_project_meta() -> list[dict[str, Any]]:
    PROJECTS_ROOT.mkdir(parents=True, exist_ok=True)
    projects = []
    for item in PROJECTS_ROOT.iterdir():
        meta_path = item / "project.json"
        if meta_path.exists():
            projects.append(update_project_stats(item.name))
    return sorted(projects, key=lambda item: item.get("updated_at", ""), reverse=True)


def run_pipeline_command(project_root: Path, args: list[str]) -> str:
    load_dotenv()
    env = os.environ.copy()
    env["PDF_VISION_RAG_ROOT"] = str(project_root)
    env["PYTHONIOENCODING"] = "utf-8"
    result = subprocess.run(
        [sys.executable, *args],
        cwd=APP_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=3600,
    )
    output = (result.stdout or "") + (result.stderr or "")
    log_path = project_root / "logs" / "pipeline.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n[{datetime.now(timezone.utc).isoformat()}] $ {' '.join(args)}\n")
        log.write(output)
        if result.returncode != 0:
            log.write(f"\nCommand failed with exit code {result.returncode}\n")
    if result.returncode != 0:
        raise RuntimeError(output.strip() or f"Command failed: {' '.join(args)}")
    return output


def project_total_pages(root: Path, manifest: list[dict[str, Any]]) -> tuple[int, list[dict[str, Any]]]:
    docs = []
    total = 0
    for record in manifest:
        pages = 1
        stored_path = root / str(record.get("stored_path", ""))
        if stored_path.suffix.lower() == ".pdf" and stored_path.exists():
            try:
                with fitz.open(stored_path) as document:
                    pages = document.page_count
            except Exception:
                pages = 1
        total += pages
        docs.append(
            {
                "document_id": record.get("document_id", ""),
                "document_name": record.get("original_name", ""),
                "pages": pages,
            }
        )
    return total, docs


def build_pipeline_progress(project_id: str) -> dict[str, Any]:
    root = project_path(project_id)
    manifest = read_json(root / "documents" / "manifest.json", [])
    topics = read_json(root / "indexes" / "topic_index.json", [])
    relationship_map = read_json(root / "indexes" / "relationship_map.json", {})
    saved = read_json(root / "logs" / "pipeline_progress.json", {})
    failed_pages = read_json(root / "indexes" / "failed_pages.json", [])
    placeholder_pages = read_json(root / "indexes" / "placeholder_pages.json", [])
    total_pages, docs = project_total_pages(root, manifest)
    indexed_pages = {
        (entry.get("document_id"), int(entry.get("page_no") or 0))
        for entry in topics
        if entry.get("document_id") and entry.get("page_no")
    }
    indexed_count = len(indexed_pages)
    last_topic = topics[-1] if topics else {}
    relation_status = relationship_map.get("status", "")
    mapped_document_ids = {
        item.get("document_id")
        for collection_name in ("communities", "biomes")
        for item in relationship_map.get(collection_name, [])
        if item.get("document_id")
    }
    relationship_document_count = len(mapped_document_ids)
    relationship_total_documents = len(manifest)
    relation_pair_done = int(
        relationship_map.get("relation_pairs_done")
        or saved.get("relation_pairs_done")
        or 0
    )
    relation_pair_total = int(
        relationship_map.get("relation_pairs_total")
        or saved.get("relation_pairs_total")
        or 0
    )
    if not manifest:
        stage = "idle"
        message = "No documents uploaded yet."
        stage_done = 0
        stage_total = 0
        stage_unit = "documents"
        stage_percent = 0
        progress_label = "No documents uploaded"
    elif saved.get("stage") == "uploaded" and not topics and not relationship_map:
        stage = "uploaded"
        message = saved.get("message") or f"Document inventory updated with {len(manifest)} file(s). Start indexing when the full set is ready."
        stage_done = len(manifest)
        stage_total = len(manifest)
        stage_unit = "documents uploaded"
        stage_percent = 100
        progress_label = f"{len(manifest)} document(s) uploaded; indexing not started"
    elif saved.get("stage") == "failed":
        stage = "failed"
        message = saved.get("message") or "Pipeline failed."
        stage_done = indexed_count
        stage_total = total_pages
        stage_unit = "pages indexed"
        stage_percent = round((indexed_count / total_pages) * 100, 1) if total_pages else 0
        progress_label = (
            f"{indexed_count} / {total_pages} pages indexed; "
            f"{len(failed_pages)} failed; {len(placeholder_pages)} placeholder"
        )
    elif indexed_count < total_pages:
        stage = "indexing"
        message = saved.get("message") or f"Indexing pages: {indexed_count} of {total_pages} complete."
        stage_done = indexed_count
        stage_total = total_pages
        stage_unit = "pages indexed"
        stage_percent = round((indexed_count / total_pages) * 100, 1) if total_pages else 0
        progress_label = f"{indexed_count} / {total_pages} pages indexed"
    elif relation_status and relation_status != "complete":
        stage = "relationship_map"
        message = f"Building relationship map: {relation_status}."
        if relation_status == "building_relations" and relation_pair_total:
            stage_done = min(relation_pair_done, relation_pair_total)
            stage_total = relation_pair_total
            stage_unit = "biome relation checks"
            stage_percent = round((stage_done / stage_total) * 100, 1) if stage_total else 0
            progress_label = f"{stage_done} / {stage_total} biome relation checks"
        elif relation_status == "building_relations":
            stage_done = relationship_document_count
            stage_total = relationship_total_documents
            stage_unit = "documents mapped"
            stage_percent = 100 if stage_total and stage_done >= stage_total else (
                round((stage_done / stage_total) * 100, 1) if stage_total else 0
            )
            progress_label = (
                f"{stage_done} / {stage_total} documents mapped. "
                f"Building biome relations; {len(relationship_map.get('biome_relations', []))} found so far."
            )
        else:
            stage_done = relationship_document_count
            stage_total = relationship_total_documents
            stage_unit = "documents mapped"
            stage_percent = round((stage_done / stage_total) * 100, 1) if stage_total else 0
            progress_label = f"{stage_done} / {stage_total} documents mapped"
    elif relation_status == "complete":
        stage = "complete"
        message = "Index and relationship map are complete."
        stage_done = relationship_total_documents or len(docs)
        stage_total = relationship_total_documents or len(docs)
        stage_unit = "documents mapped"
        stage_percent = 100
        progress_label = "Index and relationship map complete"
    else:
        stage = saved.get("stage") or "relationship_map"
        message = saved.get("message") or "Indexing complete. Relationship map is starting or waiting."
        stage_done = relationship_document_count
        stage_total = relationship_total_documents
        stage_unit = "documents mapped"
        stage_percent = round((stage_done / stage_total) * 100, 1) if stage_total else 0
        progress_label = f"{stage_done} / {stage_total} documents mapped"
    percent = round((indexed_count / total_pages) * 100, 1) if total_pages else 0
    return {
        "project_id": project_id,
        "stage": stage,
        "message": message,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "document_count": len(manifest),
        "total_pages": total_pages,
        "indexed_pages": indexed_count,
        "percent": min(100, percent),
        "stage_done": stage_done,
        "stage_total": stage_total,
        "stage_unit": stage_unit,
        "stage_percent": min(100, stage_percent),
        "progress_label": progress_label,
        "topic_count": len(topics),
        "community_count": len(relationship_map.get("communities", [])),
        "biome_count": len(relationship_map.get("biomes", [])),
        "relation_count": len(relationship_map.get("biome_relations", [])),
        "relationship_status": relation_status,
        "relationship_document_count": relationship_document_count,
        "relationship_total_documents": relationship_total_documents,
        "relation_pairs_done": relation_pair_done,
        "relation_pairs_total": relation_pair_total,
        "failed_relation_pair_count": len(relationship_map.get("failed_relation_pairs", {}) or {}),
        "failed_pages": len(failed_pages),
        "placeholder_pages": len(placeholder_pages),
        "current_document": saved.get("current_document") or last_topic.get("document_name", ""),
        "current_page": saved.get("current_page") or last_topic.get("page_no", ""),
        "last_topic": last_topic.get("topic_name", ""),
        "documents": docs,
    }


def update_project_stats(project_id: str) -> dict[str, Any]:
    root = project_path(project_id)
    meta = read_json(root / "project.json", {})
    manifest = read_json(root / "documents" / "manifest.json", [])
    topics = read_json(root / "indexes" / "topic_index.json", [])
    relationship_map = read_json(root / "indexes" / "relationship_map.json", {})
    meta.update(
        {
            "document_count": len(manifest),
            "topic_count": len(topics),
            "community_count": len(relationship_map.get("communities", [])),
            "biome_count": len(relationship_map.get("biomes", [])),
            "relation_count": len(relationship_map.get("biome_relations", [])),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    write_json(root / "project.json", meta)
    return meta


REPORT_PROGRESS_FILES = {
    "legal_assessment_7_deadly_sins": "legal_assessment_7_deadly_sins.progress.json",
    "commercial_drivers_strategy_to_win": "commercial_strategy.progress.json",
    "financial_bonds": "financial_bonds.progress.json",
    "financial_liabilities_penalties": "financial_liabilities_penalties.progress.json",
    "prebid_queries": "prebid_queries.progress.json",
    "prequalification_requirements": "prequalification_requirements.progress.json",
    "project_background": "project_background.progress.json",
    "key_information": "key_information.progress.json",
    "bid_process_evaluation": "bid_process_evaluation.progress.json",
    "risk_register": "risk_register.progress.json",
    "discrepancy_report": "discrepancy_report.progress.json",
}


def mark_report_failed(root: Path, report_type: str, exc: Exception) -> None:
    progress_path = root / "reports" / REPORT_PROGRESS_FILES[report_type]
    progress = read_json(progress_path, {})
    logs = progress.get("logs", [])
    logs.append(
        {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "message": f"Agent failed: {exc}",
            "detail": {"error_type": type(exc).__name__},
        }
    )
    write_json(
        progress_path,
        {
            **progress,
            "report_type": report_type,
            "status": "failed",
            "project_id": root.name,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "logs": logs,
        },
    )


def create_openrouter_client() -> tuple[OpenAI, str]:
    load_dotenv()
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise HTTPException(status_code=400, detail="OPENROUTER_API_KEY is missing in .env")
    client = OpenAI(
        base_url=os.getenv("OPENROUTER_BASE_URL", OPENROUTER_BASE_URL),
        api_key=api_key,
        default_headers={
            "HTTP-Referer": os.getenv("OPENROUTER_SITE_URL", "http://localhost"),
            "X-Title": os.getenv("OPENROUTER_APP_NAME", "PDF Vision RAG"),
        },
    )
    return client, os.getenv("OPENROUTER_MODEL", DEFAULT_MODEL)


def compact_history(history: list[dict[str, str]], limit: int = 8) -> list[dict[str, str]]:
    compacted = []
    for item in history[-limit:]:
        role = item.get("role", "").strip().lower()
        content = item.get("content", "").strip()
        if role in {"user", "assistant"} and content:
            compacted.append({"role": role, "content": content[:2000]})
    return compacted


def planner_prompt(question: str, history: list[dict[str, str]]) -> str:
    return f"""
You are the retrieval planner for a project-document chatbot.

Your job is to turn the user's latest message into a high-quality standalone search query for the document parser.
Use the conversation history to resolve follow-ups like "same", "more info", "what about this", or pronouns.

If the question asks for risk, constraints, blockers, critical items, project-killers, "what can cost us the project",
or high-priority issues, expand the parser query into concrete retrieval categories such as:
- bid disqualification, pass/fail prequalification, bid security, submission date, bid validity
- termination, rejection, non-compliance, statutory or standard compliance
- performance security, bonds, retention, guarantees, payment terms
- liquidated damages, penalties, caps, service level benchmarks
- water quality, UFW/water loss, complaint response, O&M obligations
- technical execution dependencies such as SCADA, flow meters, pumping, surge protection, lab, manpower, machinery, safety

Return only valid JSON with:
- parser_query: a standalone query preserving the user's intent and topic
- needs_retrieval: true unless the user is only greeting, thanking, or asking about the app itself
- intent: one of greeting, follow_up, factual_lookup, risk_analysis, comparison, summary, unknown
- answer_focus: short instruction for the final chatbot about how to structure the answer
- must_include_terms: short list of important retrieval terms/categories that should be represented

Conversation history:
{json.dumps(history, ensure_ascii=False)}

Latest user message:
{question}
""".strip()


def plan_parser_query(client: OpenAI, model: str, question: str, history: list[dict[str, str]]) -> dict[str, Any]:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "Return only valid JSON."},
            {"role": "user", "content": planner_prompt(question, history)},
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )
    raw = response.choices[0].message.content or "{}"
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = {}
    parser_query = str(parsed.get("parser_query") or question).strip()
    must_include_terms = parsed.get("must_include_terms", [])
    if not isinstance(must_include_terms, list):
        must_include_terms = []
    terms = [str(term).strip() for term in must_include_terms[:12] if str(term).strip()]
    if terms and any(
        word in str(parsed.get("intent", "")).lower() or word in question.lower()
        for word in ["risk", "constraint", "killer", "cost", "failure", "priority"]
    ):
        parser_query = f"{parser_query}. Include: {', '.join(terms)}."
    return {
        "parser_query": parser_query or question,
        "needs_retrieval": bool(parsed.get("needs_retrieval", True)),
        "intent": str(parsed.get("intent") or "unknown"),
        "answer_focus": str(parsed.get("answer_focus") or ""),
        "must_include_terms": terms,
    }


def answer_prompt(
    question: str,
    parser_query: str,
    plan: dict[str, Any],
    history: list[dict[str, str]],
    search_result: dict[str, Any],
) -> str:
    evidence = []
    for hit in search_result.get("hits", []):
        evidence.append(
            {
                "topic_name": hit.get("topic_name"),
                "document_name": hit.get("document_name"),
                "page_no": hit.get("page_no"),
                "content": hit.get("content"),
                "related_topics": hit.get("related_topics", []),
            }
        )
    return f"""
You are the customer-facing chatbot for this project.
The parser/search tool has already been called with a standalone query.

Use the conversation history to preserve context, especially for follow-up questions.
Use only the evidence below for factual claims about project documents.

Conversation history:
{json.dumps(history, ensure_ascii=False)}

Latest user question:
{question}

Parser query used:
{parser_query}

Planner intent:
{plan.get("intent", "unknown")}

Answer focus:
{plan.get("answer_focus", "")}

Important retrieval categories:
{json.dumps(plan.get("must_include_terms", []), ensure_ascii=False)}

Evidence:
{json.dumps(evidence, ensure_ascii=False)}

Requirements:
- Give a direct answer first.
- Then include concise supporting evidence.
- Cite document name and page number for important claims.
- If evidence is insufficient for the contextual follow-up, say exactly what is missing instead of switching topics.
- Do not answer from unrelated evidence just because it was retrieved.
- For risk/constraint questions, group the answer by severity: bid/project-killer, major commercial exposure, operational/technical performance risk.
""".strip()


@app.get("/")
def home() -> FileResponse:
    return FileResponse(STATIC_ROOT / "index.html")


@app.get("/api/projects")
def list_projects() -> list[dict[str, Any]]:
    return list_project_meta()


@app.post("/api/projects")
def create_project(payload: ProjectCreate) -> dict[str, Any]:
    PROJECTS_ROOT.mkdir(parents=True, exist_ok=True)
    base_id = slugify(payload.name)
    project_id = base_id
    version = 2
    while (PROJECTS_ROOT / project_id).exists():
        project_id = f"{base_id}-{version}"
        version += 1
    root = PROJECTS_ROOT / project_id
    root.mkdir(parents=True)
    meta = {
        "project_id": project_id,
        "name": payload.name.strip() or project_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "document_count": 0,
        "topic_count": 0,
        "community_count": 0,
        "biome_count": 0,
        "relation_count": 0,
    }
    write_json(root / "project.json", meta)
    return meta


@app.get("/api/projects/{project_id}")
def get_project(project_id: str) -> dict[str, Any]:
    return update_project_stats(project_id)


@app.get("/api/projects/{project_id}/documents")
def get_documents(project_id: str) -> list[dict[str, Any]]:
    root = project_path(project_id)
    return read_json(root / "documents" / "manifest.json", [])


@app.delete("/api/projects/{project_id}/documents/{document_id}")
def delete_document(project_id: str, document_id: str) -> dict[str, Any]:
    root = project_path(project_id)
    with project_lock(project_id):
        manifest_path = root / "documents" / "manifest.json"
        manifest = read_json(manifest_path, [])
        target = next((record for record in manifest if record.get("document_id") == document_id), None)
        if target is None:
            raise HTTPException(status_code=404, detail="Document not found")

        remaining = [record for record in manifest if record.get("document_id") != document_id]
        for key in ("stored_path", "source_path"):
            raw_path = target.get(key)
            if not raw_path:
                continue
            path = (root / raw_path).resolve() if key == "stored_path" else Path(raw_path).resolve()
            allowed_roots = [root.resolve()]
            if any(str(path).startswith(str(allowed)) for allowed in allowed_roots) and path.exists() and path.is_file():
                path.unlink()

        write_json(manifest_path, remaining)
        clear_generated_outputs(root)
        project = update_project_stats(project_id)
        write_pipeline_progress(
            root,
            {
                "stage": "uploaded" if remaining else "idle",
                "message": (
                    f"Removed {target.get('original_name', 'document')}. "
                    f"{len(remaining)} file(s) remain. Start indexing when the full set is ready."
                    if remaining
                    else "All documents removed. Upload the full set before indexing."
                ),
            },
        )
    return {"project": project, "documents": remaining}


@app.get("/api/projects/{project_id}/pipeline-progress")
def get_pipeline_progress(project_id: str) -> dict[str, Any]:
    return build_pipeline_progress(project_id)


@app.websocket("/ws/projects/{project_id}/pipeline-progress")
async def websocket_pipeline_progress(websocket: WebSocket, project_id: str) -> None:
    await websocket.accept()
    last_payload = ""
    last_progress: dict[str, Any] | None = None
    try:
        while True:
            try:
                progress = build_pipeline_progress(project_id)
                last_progress = progress
            except Exception as exc:
                progress = last_progress or {
                    "project_id": project_id,
                    "stage": "syncing",
                    "message": f"Waiting for progress files: {exc}",
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                }
            payload = json.dumps(progress, ensure_ascii=False, sort_keys=True)
            if payload != last_payload:
                await websocket.send_json(progress)
                last_payload = payload
            if progress.get("stage") in {"uploaded", "complete", "failed"}:
                await asyncio.sleep(0.5)
                await websocket.close()
                return
            await asyncio.sleep(1)
    except WebSocketDisconnect:
        return
    except Exception:
        return


@app.get("/api/projects/{project_id}/reports/legal-assessment")
def get_legal_assessment(project_id: str) -> dict[str, Any]:
    root = project_path(project_id)
    return load_legal_assessment(root)


@app.get("/api/projects/{project_id}/reports/legal-assessment/progress")
def get_legal_assessment_progress(project_id: str) -> dict[str, Any]:
    root = project_path(project_id)
    return load_legal_assessment_progress(root)


@app.post("/api/projects/{project_id}/reports/legal-assessment")
def create_legal_assessment(project_id: str) -> dict[str, Any]:
    root = project_path(project_id)
    try:
        return generate_legal_assessment(root)
    except RuntimeError as exc:
        mark_report_failed(root, "legal_assessment_7_deadly_sins", exc)
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        mark_report_failed(root, "legal_assessment_7_deadly_sins", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/projects/{project_id}/reports/commercial-strategy")
def get_commercial_strategy(project_id: str) -> dict[str, Any]:
    root = project_path(project_id)
    return load_commercial_strategy(root)


@app.get("/api/projects/{project_id}/reports/commercial-strategy/progress")
def get_commercial_strategy_progress(project_id: str) -> dict[str, Any]:
    root = project_path(project_id)
    return load_commercial_strategy_progress(root)


@app.post("/api/projects/{project_id}/reports/commercial-strategy")
def create_commercial_strategy(project_id: str) -> dict[str, Any]:
    root = project_path(project_id)
    try:
        return generate_commercial_strategy(root)
    except RuntimeError as exc:
        mark_report_failed(root, "commercial_drivers_strategy_to_win", exc)
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        mark_report_failed(root, "commercial_drivers_strategy_to_win", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/projects/{project_id}/reports/financial-bonds")
def get_financial_bonds(project_id: str) -> dict[str, Any]:
    root = project_path(project_id)
    return load_financial_bonds(root)


@app.get("/api/projects/{project_id}/reports/financial-bonds/progress")
def get_financial_bonds_progress(project_id: str) -> dict[str, Any]:
    root = project_path(project_id)
    return load_financial_bonds_progress(root)


@app.post("/api/projects/{project_id}/reports/financial-bonds")
def create_financial_bonds(project_id: str) -> dict[str, Any]:
    root = project_path(project_id)
    try:
        return generate_financial_bonds(root)
    except RuntimeError as exc:
        mark_report_failed(root, "financial_bonds", exc)
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        mark_report_failed(root, "financial_bonds", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/projects/{project_id}/reports/financial-liabilities-penalties")
def get_financial_liabilities_penalties(project_id: str) -> dict[str, Any]:
    root = project_path(project_id)
    return load_financial_liabilities_penalties(root)


@app.get("/api/projects/{project_id}/reports/financial-liabilities-penalties/progress")
def get_financial_liabilities_penalties_progress(project_id: str) -> dict[str, Any]:
    root = project_path(project_id)
    return load_financial_liabilities_penalties_progress(root)


@app.post("/api/projects/{project_id}/reports/financial-liabilities-penalties")
def create_financial_liabilities_penalties(project_id: str) -> dict[str, Any]:
    root = project_path(project_id)
    try:
        return generate_financial_liabilities_penalties(root)
    except RuntimeError as exc:
        mark_report_failed(root, "financial_liabilities_penalties", exc)
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        mark_report_failed(root, "financial_liabilities_penalties", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/projects/{project_id}/reports/prebid-queries")
def get_prebid_queries(project_id: str) -> dict[str, Any]:
    root = project_path(project_id)
    return load_prebid_queries(root)


@app.get("/api/projects/{project_id}/reports/prebid-queries/progress")
def get_prebid_queries_progress(project_id: str) -> dict[str, Any]:
    root = project_path(project_id)
    return load_prebid_queries_progress(root)


@app.post("/api/projects/{project_id}/reports/prebid-queries")
def create_prebid_queries(project_id: str) -> dict[str, Any]:
    root = project_path(project_id)
    try:
        return generate_prebid_queries(root)
    except RuntimeError as exc:
        mark_report_failed(root, "prebid_queries", exc)
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        mark_report_failed(root, "prebid_queries", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/projects/{project_id}/reports/prequalification-requirements")
def get_prequalification_requirements(project_id: str) -> dict[str, Any]:
    root = project_path(project_id)
    return load_prequalification_requirements(root)


@app.get("/api/projects/{project_id}/reports/prequalification-requirements/progress")
def get_prequalification_requirements_progress(project_id: str) -> dict[str, Any]:
    root = project_path(project_id)
    return load_prequalification_requirements_progress(root)


@app.post("/api/projects/{project_id}/reports/prequalification-requirements")
def create_prequalification_requirements(project_id: str) -> dict[str, Any]:
    root = project_path(project_id)
    try:
        return generate_prequalification_requirements(root)
    except RuntimeError as exc:
        mark_report_failed(root, "prequalification_requirements", exc)
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        mark_report_failed(root, "prequalification_requirements", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/projects/{project_id}/reports/project-background")
def get_project_background(project_id: str) -> dict[str, Any]:
    root = project_path(project_id)
    return load_project_background(root)


@app.get("/api/projects/{project_id}/reports/project-background/progress")
def get_project_background_progress(project_id: str) -> dict[str, Any]:
    root = project_path(project_id)
    return load_project_background_progress(root)


@app.post("/api/projects/{project_id}/reports/project-background")
def create_project_background(project_id: str) -> dict[str, Any]:
    root = project_path(project_id)
    try:
        return generate_project_background(root)
    except RuntimeError as exc:
        mark_report_failed(root, "project_background", exc)
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        mark_report_failed(root, "project_background", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/projects/{project_id}/reports/key-information")
def get_key_information(project_id: str) -> dict[str, Any]:
    root = project_path(project_id)
    return load_key_information(root)


@app.get("/api/projects/{project_id}/reports/key-information/progress")
def get_key_information_progress(project_id: str) -> dict[str, Any]:
    root = project_path(project_id)
    return load_key_information_progress(root)


@app.post("/api/projects/{project_id}/reports/key-information")
def create_key_information(project_id: str) -> dict[str, Any]:
    root = project_path(project_id)
    try:
        return generate_key_information(root)
    except RuntimeError as exc:
        mark_report_failed(root, "key_information", exc)
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        mark_report_failed(root, "key_information", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/projects/{project_id}/reports/bid-process-evaluation")
def get_bid_process_evaluation(project_id: str) -> dict[str, Any]:
    root = project_path(project_id)
    return load_bid_process_evaluation(root)


@app.get("/api/projects/{project_id}/reports/bid-process-evaluation/progress")
def get_bid_process_evaluation_progress(project_id: str) -> dict[str, Any]:
    root = project_path(project_id)
    return load_bid_process_evaluation_progress(root)


@app.post("/api/projects/{project_id}/reports/bid-process-evaluation")
def create_bid_process_evaluation(project_id: str) -> dict[str, Any]:
    root = project_path(project_id)
    try:
        return generate_bid_process_evaluation(root)
    except RuntimeError as exc:
        mark_report_failed(root, "bid_process_evaluation", exc)
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        mark_report_failed(root, "bid_process_evaluation", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/projects/{project_id}/reports/risk-register")
def get_risk_register(project_id: str) -> dict[str, Any]:
    root = project_path(project_id)
    return load_risk_register(root)


@app.get("/api/projects/{project_id}/reports/risk-register/progress")
def get_risk_register_progress(project_id: str) -> dict[str, Any]:
    root = project_path(project_id)
    return load_risk_register_progress(root)


@app.post("/api/projects/{project_id}/reports/risk-register")
def create_risk_register(project_id: str) -> dict[str, Any]:
    root = project_path(project_id)
    try:
        return generate_risk_register(root)
    except RuntimeError as exc:
        mark_report_failed(root, "risk_register", exc)
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        mark_report_failed(root, "risk_register", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/projects/{project_id}/reports/discrepancy-report")
def get_discrepancy_report(project_id: str) -> dict[str, Any]:
    root = project_path(project_id)
    return load_discrepancy_report(root)


@app.get("/api/projects/{project_id}/reports/discrepancy-report/progress")
def get_discrepancy_report_progress(project_id: str) -> dict[str, Any]:
    root = project_path(project_id)
    return load_discrepancy_report_progress(root)


@app.post("/api/projects/{project_id}/reports/discrepancy-report")
def create_discrepancy_report(project_id: str) -> dict[str, Any]:
    root = project_path(project_id)
    try:
        return generate_discrepancy_report(root)
    except RuntimeError as exc:
        mark_report_failed(root, "discrepancy_report", exc)
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        mark_report_failed(root, "discrepancy_report", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/projects/{project_id}/upload")
def upload_documents(project_id: str, files: list[UploadFile] = File(...)) -> dict[str, Any]:
    root = project_path(project_id)
    uploads_dir = root / "uploads"
    uploads_dir.mkdir(parents=True, exist_ok=True)
    logs = []
    with project_lock(project_id):
        saved_files = []
        write_pipeline_progress(root, {"stage": "uploading", "message": "Saving uploaded files."})
        for file in files:
            safe_name = Path(file.filename or "document").name
            target = uploads_dir / safe_name
            with target.open("wb") as handle:
                shutil.copyfileobj(file.file, handle)
            saved_files.append(target)
        write_pipeline_progress(root, {"stage": "ingesting", "message": f"Saved {len(saved_files)} file(s). Reconciling full document inventory."})

        try:
            write_pipeline_progress(root, {"stage": "ingesting", "message": "Adding any uploaded files missing from the document inventory."})
            logs.append(run_pipeline_command(root, ["ingest.py", "add", str(uploads_dir)]))
            project = update_project_stats(project_id)
            write_pipeline_progress(
                root,
                {
                    "stage": "uploaded",
                    "message": f"Document inventory updated with {project.get('document_count', 0)} file(s). Start indexing when the full set is ready.",
                },
            )
        except RuntimeError as exc:
            write_pipeline_progress(root, {"stage": "failed", "message": str(exc)})
            update_project_stats(project_id)
            raise HTTPException(status_code=500, detail=str(exc))

    return {"project": update_project_stats(project_id), "logs": "\n".join(logs)}


@app.post("/api/projects/{project_id}/build-index")
def build_project_index(project_id: str) -> dict[str, Any]:
    root = project_path(project_id)
    manifest = read_json(root / "documents" / "manifest.json", [])
    if not manifest:
        raise HTTPException(status_code=400, detail="Upload documents before starting index generation.")
    logs = []
    with project_lock(project_id):
        try:
            write_pipeline_progress(root, {"stage": "indexing", "message": "Indexing all uploaded documents page by page."})
            logs.append(run_pipeline_command(root, ["indexer.py"]))
            write_pipeline_progress(root, {"stage": "relationship_map", "message": "Building communities, biomes, and relationships for the full document set."})
            logs.append(run_pipeline_command(root, ["relationship_map.py"]))
            write_pipeline_progress(root, {"stage": "complete", "message": "Index and relationship map are complete."})
        except RuntimeError as exc:
            write_pipeline_progress(root, {"stage": "failed", "message": str(exc)})
            update_project_stats(project_id)
            raise HTTPException(status_code=500, detail=str(exc))
    return {"project": update_project_stats(project_id), "logs": "\n".join(logs)}


@app.post("/api/projects/{project_id}/chat")
def chat(project_id: str, payload: ChatRequest) -> dict[str, Any]:
    root = project_path(project_id)
    client, model = create_openrouter_client()
    history = compact_history(payload.history)
    plan = plan_parser_query(client, model, payload.question, history)
    if not plan["needs_retrieval"]:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a concise project chatbot. Do not invent document facts without retrieval."},
                *history,
                {"role": "user", "content": payload.question},
            ],
            temperature=0.2,
        )
        return {
            "answer": response.choices[0].message.content or "",
            "parser_query": plan["parser_query"],
            "plan": plan,
            "search": {"hits": [], "trace": []},
        }

    try:
        run_pipeline_command(
            root,
            ["searcher.py", plan["parser_query"], "--max-hits", str(payload.max_hits)],
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    result_files = sorted(
        (root / "indexes" / "search_results").glob("*.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not result_files:
        raise HTTPException(status_code=500, detail="Search completed but no result file was produced")
    search_result = read_json(result_files[0], {})
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You answer questions over project documents with citations."},
            {"role": "user", "content": answer_prompt(payload.question, plan["parser_query"], plan, history, search_result)},
        ],
        temperature=0.2,
    )
    answer = response.choices[0].message.content or ""
    return {"answer": answer, "parser_query": plan["parser_query"], "plan": plan, "search": search_result}


@app.get("/{app_route:path}")
def frontend_route(app_route: str):
    if app_route == "projects" or app_route.startswith("projects/"):
        return FileResponse(STATIC_ROOT / "index.html")
    if app_route in {"documents", "chat", "map"}:
        return RedirectResponse(url="/projects")
    if not app_route.startswith(("api/", "static/")):
        return FileResponse(STATIC_ROOT / "index.html")
    raise HTTPException(status_code=404, detail="Not found")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
