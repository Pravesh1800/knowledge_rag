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

from cache import read_cache, stable_hash, write_cache
from llm_config import create_chat_client, runtime_settings, update_dotenv
from schema import read_knowledge_graph
from storage import (
    clear_cards,
    read_cards,
    read_graph_audit_state,
    read_knowledge_graph_state,
    read_pipeline_progress_state,
    sync_documents,
    sync_knowledge_graph,
    write_pipeline_progress_state,
)


APP_ROOT = Path(__file__).resolve().parent
PROJECTS_ROOT = APP_ROOT / "projects"
STATIC_ROOT = APP_ROOT / "frontend"
STATIC_V2_ROOT = APP_ROOT / "frontend_v2"
SEARCH_PLANNER_PROMPT_VERSION = "search_planner_prompt.v1.0"
PROJECT_LOCKS: dict[str, threading.Lock] = {}
PROJECT_LOCKS_GUARD = threading.Lock()
LEGACY_RELATIONSHIP_CHECKS_DONE = "relationship" + "_pairs_done"
LEGACY_RELATIONSHIP_CHECKS_TOTAL = "relationship" + "_pairs_total"
LEGACY_FAILED_RELATIONSHIP_CHECKS = "failed_relationship" + "_pairs"


app = FastAPI(title="Evidence Mesh")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory=STATIC_ROOT), name="static")
app.mount("/v2/static", StaticFiles(directory=STATIC_V2_ROOT), name="static_v2")


class ProjectCreate(BaseModel):
    name: str


class ChatRequest(BaseModel):
    question: str
    max_hits: int = 10
    history: list[dict[str, str]] = []
    query_mode: str = "auto"


class RuntimeSettingsUpdate(BaseModel):
    provider: str
    api_key: str = ""
    model: str = ""
    map_model: str = ""
    search_model: str = ""
    openrouter_model: str = ""
    openrouter_map_model: str = ""
    openrouter_search_model: str = ""
    openai_model: str = ""
    openai_map_model: str = ""
    openai_search_model: str = ""


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


def append_jsonl(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(data, ensure_ascii=False) + "\n")


def write_pipeline_progress(root: Path, data: dict[str, Any]) -> None:
    payload = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        **data,
    }
    write_pipeline_progress_state(payload, root.name)
    try:
        write_json(root / "logs" / "pipeline_progress.json", payload)
    except OSError as exc:
        print(f"Warning: could not export pipeline_progress.json: {exc}")


def clear_generated_outputs(root: Path) -> None:
    for name in ("indexes",):
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


def read_project_meta_fast(project_id: str) -> dict[str, Any]:
    meta = read_json(project_meta_path(project_id), {})
    if not meta:
        meta = {}
    meta.setdefault("project_id", project_id)
    meta.setdefault("name", project_id)
    meta.setdefault("created_at", "")
    meta.setdefault("updated_at", meta.get("created_at", ""))
    meta.setdefault("document_count", 0)
    meta.setdefault("card_count", 0)
    meta.setdefault("cluster_count", 0)
    meta.setdefault("domain_count", 0)
    meta.setdefault("relationship_count", 0)
    return meta


def list_project_meta() -> list[dict[str, Any]]:
    PROJECTS_ROOT.mkdir(parents=True, exist_ok=True)
    projects = []
    for item in PROJECTS_ROOT.iterdir():
        meta_path = item / "project.json"
        if meta_path.exists():
            projects.append(read_project_meta_fast(item.name))
    return sorted(projects, key=lambda item: item.get("updated_at", ""), reverse=True)


def run_pipeline_command(project_root: Path, args: list[str]) -> str:
    load_dotenv()
    env = os.environ.copy()
    env["EVIDENCE_MESH_ROOT"] = str(project_root)
    env["EVIDENCE_MESH_PROJECT_ID"] = project_root.name
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
    cards = read_cards(project_id)
    knowledge_graph = read_knowledge_graph_state(project_id)
    if not knowledge_graph:
        knowledge_graph = read_knowledge_graph(
            root / "indexes" / "knowledge_graph.json",
            cards,
            persist_migration=True,
        )
    graph_audit = read_graph_audit_state(project_id) or read_json(root / "indexes" / "graph_audit.json", {})
    saved = read_pipeline_progress_state(project_id) or read_json(root / "logs" / "pipeline_progress.json", {})
    failed_pages = read_json(root / "indexes" / "failed_pages.json", [])
    placeholder_pages = read_json(root / "indexes" / "placeholder_pages.json", [])
    total_pages, docs = project_total_pages(root, manifest)
    indexed_pages = {
        (entry.get("document_id"), int(entry.get("page_no") or 0))
        for entry in cards
        if entry.get("document_id") and entry.get("page_no")
    }
    indexed_count = len(indexed_pages)
    last_card = cards[-1] if cards else {}
    knowledge_graph_status = knowledge_graph.get("status", "")
    mapped_document_ids = {
        item.get("document_id")
        for collection_name in ("clusters", "domains")
        for item in knowledge_graph.get(collection_name, [])
        if item.get("document_id")
    }
    knowledge_graph_document_count = len(mapped_document_ids)
    knowledge_graph_total_documents = len(manifest)
    relationship_checks_done = int(
        knowledge_graph.get("relationship_checks_done")
        or saved.get("relationship_checks_done")
        or knowledge_graph.get(LEGACY_RELATIONSHIP_CHECKS_DONE)
        or saved.get(LEGACY_RELATIONSHIP_CHECKS_DONE)
        or 0
    )
    relationship_checks_total = int(
        knowledge_graph.get("relationship_checks_total")
        or saved.get("relationship_checks_total")
        or knowledge_graph.get(LEGACY_RELATIONSHIP_CHECKS_TOTAL)
        or saved.get(LEGACY_RELATIONSHIP_CHECKS_TOTAL)
        or 0
    )
    saved_relationship_checks_done = int(
        saved.get("relationship_checks_done")
        or saved.get(LEGACY_RELATIONSHIP_CHECKS_DONE)
        or 0
    )
    saved_relationship_checks_total = int(
        saved.get("relationship_checks_total")
        or saved.get(LEGACY_RELATIONSHIP_CHECKS_TOTAL)
        or 0
    )
    if saved.get("knowledge_graph_status") == "building_relationships" and saved_relationship_checks_done >= relationship_checks_done:
        relationship_checks_done = saved_relationship_checks_done
        relationship_checks_total = saved_relationship_checks_total or relationship_checks_total
    failed_relationship_checks = (
        knowledge_graph.get("failed_relationship_checks")
        or knowledge_graph.get(LEGACY_FAILED_RELATIONSHIP_CHECKS)
        or {}
    )
    if not manifest:
        stage = "idle"
        message = "No documents uploaded yet."
        stage_done = 0
        stage_total = 0
        stage_unit = "documents"
        stage_percent = 0
        progress_label = "No documents uploaded"
    elif saved.get("stage") == "uploaded" and not cards and not knowledge_graph:
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
    elif knowledge_graph_status and knowledge_graph_status != "complete":
        stage = "knowledge_graph"
        message = f"Building knowledge graph: {knowledge_graph_status}."
        if knowledge_graph_status == "building_relationships" and relationship_checks_total:
            stage_done = min(relationship_checks_done, relationship_checks_total)
            stage_total = relationship_checks_total
            stage_unit = "domain relationship checks"
            stage_percent = round((stage_done / stage_total) * 100, 1) if stage_total else 0
            progress_label = f"{stage_done} / {stage_total} domain relationship checks"
        elif knowledge_graph_status == "building_relationships":
            stage_done = knowledge_graph_document_count
            stage_total = knowledge_graph_total_documents
            stage_unit = "documents mapped"
            stage_percent = 100 if stage_total and stage_done >= stage_total else (
                round((stage_done / stage_total) * 100, 1) if stage_total else 0
            )
            progress_label = (
                f"{stage_done} / {stage_total} documents mapped. "
                f"Building domain relationships; {len(knowledge_graph.get('domain_relationships', []))} found so far."
            )
        else:
            stage_done = knowledge_graph_document_count
            stage_total = knowledge_graph_total_documents
            stage_unit = "documents mapped"
            stage_percent = round((stage_done / stage_total) * 100, 1) if stage_total else 0
            progress_label = f"{stage_done} / {stage_total} documents mapped"
    elif knowledge_graph_status == "complete":
        stage = "complete"
        message = "Index and knowledge graph are complete."
        stage_done = knowledge_graph_total_documents or len(docs)
        stage_total = knowledge_graph_total_documents or len(docs)
        stage_unit = "documents mapped"
        stage_percent = 100
        progress_label = "Index and knowledge graph complete"
    else:
        stage = saved.get("stage") or "knowledge_graph"
        message = saved.get("message") or "Indexing complete. Knowledge graph is starting or waiting."
        stage_done = knowledge_graph_document_count
        stage_total = knowledge_graph_total_documents
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
        "card_count": len(cards),
        "cluster_count": len(knowledge_graph.get("clusters", [])),
        "domain_count": len(knowledge_graph.get("domains", [])),
        "relationship_count": (
            int(saved.get("relationship_count") or 0)
            if saved.get("knowledge_graph_status") == "building_relationships"
            and int(saved.get("relationship_count") or 0) >= len(knowledge_graph.get("domain_relationships", []))
            else len(knowledge_graph.get("domain_relationships", []))
        ),
        "knowledge_graph_status": knowledge_graph_status,
        "knowledge_graph_document_count": knowledge_graph_document_count,
        "knowledge_graph_total_documents": knowledge_graph_total_documents,
        "relationship_checks_done": relationship_checks_done,
        "relationship_checks_total": relationship_checks_total,
        "failed_relationship_check_count": len(failed_relationship_checks),
        "graph_audit_status": graph_audit.get("status") or saved.get("graph_audit_status", ""),
        "graph_audit_issue_count": (
            graph_audit["issue_count"]
            if "issue_count" in graph_audit
            else saved.get("graph_audit_issue_count", 0)
        ),
        "graph_audit_summary_by_category": (
            graph_audit.get("summary_by_category")
            or saved.get("graph_audit_summary_by_category")
            or {}
        ),
        "graph_audit_summary_by_severity": (
            graph_audit.get("summary_by_severity")
            or saved.get("graph_audit_summary_by_severity")
            or {}
        ),
        "failed_pages": len(failed_pages),
        "placeholder_pages": len(placeholder_pages),
        "current_document": saved.get("current_document") or last_card.get("document_name", ""),
        "current_page": saved.get("current_page") or last_card.get("page_no", ""),
        "last_card": last_card.get("card_name", ""),
        "documents": docs,
    }


def update_project_stats(project_id: str) -> dict[str, Any]:
    root = project_path(project_id)
    meta = read_json(root / "project.json", {})
    manifest = read_json(root / "documents" / "manifest.json", [])
    cards = read_cards(project_id)
    knowledge_graph = read_knowledge_graph_state(project_id)
    if not knowledge_graph:
        knowledge_graph = read_knowledge_graph(
            root / "indexes" / "knowledge_graph.json",
            cards,
            persist_migration=True,
        )
    sync_documents(manifest, project_id)
    sync_knowledge_graph(knowledge_graph, project_id)
    meta.update(
        {
            "document_count": len(manifest),
            "card_count": len(cards),
            "cluster_count": len(knowledge_graph.get("clusters", [])),
            "domain_count": len(knowledge_graph.get("domains", [])),
            "relationship_count": len(knowledge_graph.get("domain_relationships", [])),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    write_json(root / "project.json", meta)
    return meta


def create_llm_client() -> tuple[OpenAI, str]:
    try:
        client, model, _provider = create_chat_client()
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return client, model


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

If the question asks for risks, constraints, blockers, contradictions, missing information,
dependencies, or high-priority issues, expand the parser query into concrete retrieval
categories such as requirements, responsibilities, deadlines, assumptions, definitions,
exceptions, dependencies, standards, evidence, and cross-document conflicts.

Return only valid JSON with:
- parser_query: a standalone query preserving the user's intent and card
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
    prompt = planner_prompt(question, history)
    cache_key = {
        "version": SEARCH_PLANNER_PROMPT_VERSION,
        "model": model,
        "prompt_hash": stable_hash(prompt),
    }
    cached = read_cache("search_planning", cache_key)
    if cached is not None and isinstance(cached.get("value"), dict):
        return cached["value"]
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "Return only valid JSON."},
            {"role": "user", "content": prompt},
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
    plan = {
        "parser_query": parser_query or question,
        "needs_retrieval": bool(parsed.get("needs_retrieval", True)),
        "intent": str(parsed.get("intent") or "unknown"),
        "answer_focus": str(parsed.get("answer_focus") or ""),
        "must_include_terms": terms,
    }
    write_cache("search_planning", cache_key, plan)
    return plan


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
                "card_id": hit.get("card_id"),
                "card_name": hit.get("card_name"),
                "document_name": hit.get("document_name"),
                "page_no": hit.get("page_no"),
                "content": hit.get("content"),
                "related_cards": hit.get("related_cards", []),
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
- If evidence is insufficient for the contextual follow-up, say exactly what is missing instead of switching cards.
- Do not answer from unrelated evidence just because it was retrieved.
- For risk/constraint questions, group the answer by severity and explain the evidence behind each item.
""".strip()


@app.get("/")
def home() -> FileResponse:
    return FileResponse(STATIC_ROOT / "index.html")


@app.get("/v2")
def home_v2() -> FileResponse:
    return FileResponse(STATIC_V2_ROOT / "index.html")


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
        "card_count": 0,
        "cluster_count": 0,
        "domain_count": 0,
        "relationship_count": 0,
    }
    write_json(root / "project.json", meta)
    return meta


@app.get("/api/projects/{project_id}")
def get_project(project_id: str) -> dict[str, Any]:
    return read_project_meta_fast(project_id)


@app.get("/api/runtime-settings")
def get_runtime_settings() -> dict[str, Any]:
    return runtime_settings()


@app.post("/api/runtime-settings")
def save_runtime_settings(payload: RuntimeSettingsUpdate) -> dict[str, Any]:
    provider = payload.provider.strip().lower()
    if provider not in {"openai", "openrouter"}:
        raise HTTPException(status_code=400, detail="Provider must be openai or openrouter")

    updates = {
        "LLM_PROVIDER": provider,
        "LLM_MODEL": payload.model.strip(),
        "LLM_MAP_MODEL": payload.map_model.strip(),
        "LLM_SEARCH_MODEL": payload.search_model.strip(),
        "OPENROUTER_MODEL": payload.openrouter_model.strip(),
        "OPENROUTER_MAP_MODEL": payload.openrouter_map_model.strip(),
        "OPENROUTER_SEARCH_MODEL": payload.openrouter_search_model.strip(),
        "OPENAI_MODEL": payload.openai_model.strip(),
        "OPENAI_MAP_MODEL": payload.openai_map_model.strip(),
        "OPENAI_SEARCH_MODEL": payload.openai_search_model.strip(),
    }
    if payload.api_key.strip():
        updates["LLM_API_KEY"] = payload.api_key.strip()
        updates["OPENAI_API_KEY" if provider == "openai" else "OPENROUTER_API_KEY"] = payload.api_key.strip()

    update_dotenv(updates)
    return runtime_settings()


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
        sync_documents(remaining, project_id)
        clear_generated_outputs(root)
        clear_cards(project_id)
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


@app.post("/api/projects/{project_id}/upload")
def upload_documents(project_id: str, files: list[UploadFile] = File(...)) -> dict[str, Any]:
    root = project_path(project_id)
    uploads_dir = root / "uploads"
    uploads_dir.mkdir(parents=True, exist_ok=True)
    logs = []
    with project_lock(project_id):
        manifest_before = read_json(root / "documents" / "manifest.json", [])
        saved_files = []
        batch_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        batch_dir = uploads_dir / batch_id
        batch_dir.mkdir(parents=True, exist_ok=True)
        write_pipeline_progress(root, {"stage": "uploading", "message": "Saving uploaded files."})
        for index, file in enumerate(files, start=1):
            safe_name = Path(file.filename or "document").name
            target = batch_dir / safe_name
            if target.exists():
                target = batch_dir / f"{target.stem}_{index}{target.suffix}"
            with target.open("wb") as handle:
                shutil.copyfileobj(file.file, handle)
            saved_files.append(target)
        append_jsonl(
            root / "logs" / "upload_events.jsonl",
            {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "event": "upload_saved",
                "project_id": project_id,
                "batch_id": batch_id,
                "manifest_count_before": len(manifest_before),
                "uploaded_file_count": len(saved_files),
                "uploaded_files": [path.name for path in saved_files],
            },
        )
        write_pipeline_progress(root, {"stage": "ingesting", "message": f"Saved {len(saved_files)} file(s). Reconciling full document inventory."})

        try:
            write_pipeline_progress(root, {"stage": "ingesting", "message": "Adding any uploaded files missing from the document inventory."})
            logs.append(run_pipeline_command(root, ["ingest.py", "add", str(batch_dir)]))
            project = update_project_stats(project_id)
            manifest_after = read_json(root / "documents" / "manifest.json", [])
            if len(manifest_after) < len(manifest_before):
                raise RuntimeError(
                    "Upload safety check failed: document count decreased from "
                    f"{len(manifest_before)} to {len(manifest_after)}."
                )
            append_jsonl(
                root / "logs" / "upload_events.jsonl",
                {
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "event": "upload_ingested",
                    "project_id": project_id,
                    "batch_id": batch_id,
                    "manifest_count_before": len(manifest_before),
                    "manifest_count_after": len(manifest_after),
                    "uploaded_file_count": len(saved_files),
                },
            )
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
            write_pipeline_progress(root, {"stage": "knowledge_graph", "message": "Building clusters, domains, and relationships for the full document set."})
            logs.append(run_pipeline_command(root, ["knowledge_graph.py"]))
            write_pipeline_progress(root, {"stage": "complete", "message": "Index and knowledge graph are complete."})
        except RuntimeError as exc:
            write_pipeline_progress(root, {"stage": "failed", "message": str(exc)})
            update_project_stats(project_id)
            raise HTTPException(status_code=500, detail=str(exc))
    return {"project": update_project_stats(project_id), "logs": "\n".join(logs)}


@app.post("/api/projects/{project_id}/chat")
def chat(project_id: str, payload: ChatRequest) -> dict[str, Any]:
    root = project_path(project_id)
    client, model = create_llm_client()
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
    if app_route == "v2" or app_route.startswith("v2/"):
        return FileResponse(STATIC_V2_ROOT / "index.html")
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




