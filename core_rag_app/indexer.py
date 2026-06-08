from __future__ import annotations

import argparse
import base64
import json
import os
import re
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from collections.abc import Iterable
from typing import Any

import fitz
from openai import OpenAI

fitz.TOOLS.mupdf_display_errors(False)
fitz.TOOLS.mupdf_display_warnings(False)

LEGACY_ROOT_ENV = "PDF" + "_VISION_RAG_ROOT"
PROJECT_ROOT = Path(
    os.getenv("EVIDENCE_MESH_ROOT")
    or os.getenv(LEGACY_ROOT_ENV)
    or Path(__file__).resolve().parent
).resolve()
MANIFEST_PATH = PROJECT_ROOT / "documents" / "manifest.json"
INDEXES_DIR = PROJECT_ROOT / "indexes"
PAGES_DIR = INDEXES_DIR / "pages"
CARD_INDEX_PATH = INDEXES_DIR / "card_index.json"
BAD_RESPONSES_DIR = INDEXES_DIR / "bad_responses"
FAILED_PAGES_PATH = INDEXES_DIR / "failed_pages.json"
PLACEHOLDER_PAGES_PATH = INDEXES_DIR / "placeholder_pages.json"
PIPELINE_PROGRESS_PATH = PROJECT_ROOT / "logs" / "pipeline_progress.json"
MODEL_HITS_PATH = PROJECT_ROOT / "logs" / "model_hits.jsonl"

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "google/gemini-3.1-flash-lite"
DEFAULT_PAGE_RETRIES = 3
DEFAULT_PAGE_TEXT_CHARS = 6000
DEFAULT_CARD_CONTENT_CHARS = 1800
DEFAULT_INDEX_MAX_TOKENS = 2500
DEFAULT_CARD_MATCH_MAX_TOKENS = 500
SPREADSHEET_EXTENSIONS = {".xls", ".xlsx", ".xlsm"}


@dataclass(frozen=True)
class PageUnit:
    document_id: str
    document_name: str
    page_no: int
    text: str
    image_path: str | None = None


@dataclass
class CardEntry:
    card_name: str
    card_description: str
    document_id: str
    document_name: str
    page_no: int
    content: str
    source_type: str
    card_source: str
    tags: list[str]
    image_descriptions: list[dict[str, str]]
    created_at: str


def load_dotenv() -> None:
    env_paths = [PROJECT_ROOT / ".env", Path(__file__).resolve().parent / ".env"]
    for env_path in env_paths:
        if not env_path.exists():
            continue
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ[key.strip()] = value.strip().strip("\"'")


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f"{path.name}.tmp")
    temp_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    for attempt in range(1, 21):
        try:
            temp_path.replace(path)
            return
        except PermissionError:
            if attempt == 20:
                raise
            time.sleep(0.25)


def write_pipeline_progress(data: dict[str, Any]) -> None:
    write_json(
        PIPELINE_PROGRESS_PATH,
        {
            "updated_at": datetime.now(timezone.utc).isoformat(),
            **data,
        },
    )


def write_model_hit(context: str, requested_model: str, response: Any, extra: dict[str, Any] | None = None) -> None:
    usage = getattr(response, "usage", None)
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "context": context,
        "requested_model": requested_model,
        "response_model": getattr(response, "model", ""),
        "response_id": getattr(response, "id", ""),
        "usage": {
            "prompt_tokens": getattr(usage, "prompt_tokens", None),
            "completion_tokens": getattr(usage, "completion_tokens", None),
            "total_tokens": getattr(usage, "total_tokens", None),
        },
    }
    if extra:
        payload.update(extra)
    MODEL_HITS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with MODEL_HITS_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def page_issue(page: PageUnit, issue_type: str, message: str, attempts: int) -> dict[str, Any]:
    return {
        "document_id": page.document_id,
        "document_name": page.document_name,
        "page_no": page.page_no,
        "issue_type": issue_type,
        "message": message,
        "attempts": attempts,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }


def issue_key(issue: dict[str, Any]) -> tuple[str, int]:
    return (str(issue.get("document_id", "")), int(issue.get("page_no") or 0))


def page_key(page: PageUnit) -> tuple[str, int]:
    return (page.document_id, page.page_no)


def upsert_issue(path: Path, issue: dict[str, Any]) -> list[dict[str, Any]]:
    issues = read_json(path, [])
    key = issue_key(issue)
    issues = [item for item in issues if issue_key(item) != key]
    issues.append(issue)
    write_json(path, issues)
    return issues


def clear_issue(path: Path, page: PageUnit) -> list[dict[str, Any]]:
    issues = read_json(path, [])
    key = page_key(page)
    remaining = [item for item in issues if issue_key(item) != key]
    if len(remaining) != len(issues):
        write_json(path, remaining)
    return remaining


def count_record_pages(record: dict[str, Any]) -> int:
    path = PROJECT_ROOT / record["stored_path"]
    if record.get("extension") == ".pdf" and path.exists():
        try:
            document = fitz.open(path)
            count = document.page_count
            document.close()
            return count
        except Exception:
            return 1
    if record.get("extension") in {".txt", ".md", ".markdown", ".csv", ".tsv", ".json", ".jsonl", ".xml", ".html", ".htm"}:
        text = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
        chunks = [chunk.strip() for chunk in re.split(r"\f|\n-{3,}\n", text) if chunk.strip()]
        return max(1, len(chunks))
    if record.get("extension") in SPREADSHEET_EXTENSIONS and path.exists():
        try:
            return max(1, len(spreadsheet_text_chunks(path)))
        except Exception:
            return 1
    return 1


def slugify(value: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip()).strip("_").lower()
    return value or "untitled"


def clean_card_name(value: str) -> str:
    value = re.sub(r"\s+", " ", value.strip())
    value = re.sub(r"[^a-zA-Z0-9 _./&()-]+", "", value)
    return value[:120] or "Untitled Card"


def existing_card_names(index: list[dict[str, Any]]) -> list[str]:
    return [entry["card_name"] for entry in index]


def next_versioned_name(base_name: str, index: list[dict[str, Any]]) -> str:
    names = set(existing_card_names(index))
    if base_name not in names:
        return base_name

    version = 2
    while f"{base_name}_v{version}" in names:
        version += 1
    return f"{base_name}_v{version}"


def base_card_name(card_name: str) -> str:
    return re.sub(r"_v\d+$", "", card_name).strip()


def normalize_for_match(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def likely_related_by_name(candidate_name: str, existing_name: str) -> bool:
    candidate = normalize_for_match(base_card_name(candidate_name))
    existing = normalize_for_match(base_card_name(existing_name))
    if not candidate or not existing:
        return False
    if candidate == existing or candidate in existing or existing in candidate:
        return True
    candidate_words = set(candidate.split())
    existing_words = set(existing.split())
    if not candidate_words or not existing_words:
        return False
    overlap = candidate_words & existing_words
    return len(overlap) >= min(2, len(candidate_words), len(existing_words))


def candidate_existing_cards(
    raw_card: dict[str, Any],
    index: list[dict[str, Any]],
    limit: int = 12,
) -> list[dict[str, Any]]:
    card_name = str(raw_card.get("card_name", ""))
    matches = [
        entry for entry in index if likely_related_by_name(card_name, entry["card_name"])
    ]
    if not matches:
        matches = index[-limit:]
    return matches[-limit:]


def render_pdf_page(page: fitz.Page, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pixmap = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
    pixmap.save(output_path)


def build_pdf_pages(record: dict[str, Any]) -> Iterable[PageUnit]:
    pdf_path = PROJECT_ROOT / record["stored_path"]
    document = fitz.open(pdf_path)
    for page_index, page in enumerate(document, start=1):
        image_path = (
            PAGES_DIR
            / record["document_id"][:12]
            / f"page_{page_index:04d}.png"
        )
        render_pdf_page(page, image_path)
        yield PageUnit(
            document_id=record["document_id"],
            document_name=record["original_name"],
            page_no=page_index,
            text=page.get_text("text").strip(),
            image_path=str(image_path.relative_to(PROJECT_ROOT)),
        )


def build_text_pages(record: dict[str, Any]) -> Iterable[PageUnit]:
    path = PROJECT_ROOT / record["stored_path"]
    text = path.read_text(encoding="utf-8", errors="replace")
    chunks = [chunk.strip() for chunk in re.split(r"\f|\n-{3,}\n", text) if chunk.strip()]
    if not chunks:
        chunks = [text.strip()]

    for index, chunk in enumerate(chunks, start=1):
        yield PageUnit(
            document_id=record["document_id"],
            document_name=record["original_name"],
            page_no=index,
            text=chunk,
        )


def build_image_page(record: dict[str, Any]) -> Iterable[PageUnit]:
    yield PageUnit(
        document_id=record["document_id"],
        document_name=record["original_name"],
        page_no=1,
        text="",
        image_path=record["stored_path"],
    )


def spreadsheet_text_chunks(path: Path, max_chars: int = 9000) -> list[str]:
    import pandas as pd

    sheets = pd.read_excel(path, sheet_name=None, header=None, dtype=str)
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for sheet_name, frame in sheets.items():
        current.append(f"Sheet: {sheet_name}")
        current_len += len(current[-1])
        for row_index, row in frame.iterrows():
            cells = []
            for col_index, value in row.items():
                if pd.isna(value):
                    continue
                text = str(value).strip()
                if text:
                    cells.append(f"C{int(col_index) + 1}: {text}")
            if not cells:
                continue
            line = f"Row {int(row_index) + 1}: " + " | ".join(cells)
            if current and current_len + len(line) > max_chars:
                chunks.append("\n".join(current))
                current = [f"Sheet: {sheet_name} continued", line]
                current_len = sum(len(item) for item in current)
            else:
                current.append(line)
                current_len += len(line)

    if current:
        chunks.append("\n".join(current))
    return chunks or ["Spreadsheet contained no readable non-empty cells."]


def build_spreadsheet_pages(record: dict[str, Any]) -> Iterable[PageUnit]:
    path = PROJECT_ROOT / record["stored_path"]
    try:
        chunks = spreadsheet_text_chunks(path)
    except Exception as exc:
        chunks = [f"Spreadsheet extraction failed for {record['original_name']}: {exc}"]

    for index, chunk in enumerate(chunks, start=1):
        yield PageUnit(
            document_id=record["document_id"],
            document_name=record["original_name"],
            page_no=index,
            text=chunk,
        )


def build_page_units(record: dict[str, Any]) -> Iterable[PageUnit]:
    extension = record["extension"]
    if extension == ".pdf":
        yield from build_pdf_pages(record)
        return
    if extension in {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".bmp"}:
        yield from build_image_page(record)
        return
    if extension in {".txt", ".md", ".markdown", ".csv", ".tsv", ".json", ".jsonl", ".xml", ".html", ".htm"}:
        yield from build_text_pages(record)
        return
    if extension in SPREADSHEET_EXTENSIONS:
        yield from build_spreadsheet_pages(record)
        return

    yield PageUnit(
        document_id=record["document_id"],
        document_name=record["original_name"],
        page_no=1,
        text=(
            "This file type is registered for Gemini file ingestion. "
            "Page extraction for this extension will be handled by Gemini in a later connector step."
        ),
    )


def page_prompt(page: PageUnit, index: list[dict[str, Any]]) -> str:
    known = existing_card_names(index)
    known_text = "\n".join(f"- {name}" for name in known[-80:]) or "- none"
    page_text = compact_page_text(page.text, max_chars=int(os.getenv("INDEX_PAGE_TEXT_CHARS", str(DEFAULT_PAGE_TEXT_CHARS))))
    if not page_text:
        page_text = "[No embedded text extracted. Use the page image.]"

    return f"""
You are creating a page-level card index for a mixed document corpus.

Document: {page.document_name}
Page number: {page.page_no}

Existing card names:
{known_text}

Page embedded text:
{page_text}

Task:
1. Read this page as a full document page, using the image when provided.
2. Identify every distinct new card found on this page.
3. For each card, write a concise but useful card description.
4. Store only concise actual page content relevant to that card. Do not copy long repeated drawing labels, table noise, or repeated words.
5. If the page contains diagrams, photos, screenshots, drawings, charts, tables-as-images, or other meaningful images, create detailed image descriptions.
6. Image descriptions must include nearby page context and what the image contributes to the document.
7. For each card, set card_source to:
   - "text" if it mainly comes from normal document text.
   - "image" if it mainly comes from a chart, map, photo, drawing, screenshot, diagram, or other visual.
   - "mixed" if it needs both text and visual evidence.
8. Add the tag "image" to tags whenever card_source is "image" or "mixed", or whenever image_descriptions is not empty.
9. Do not invent content that is not visible or supported.
10. Keep each card content under {int(os.getenv("INDEX_CARD_CONTENT_CHARS", str(DEFAULT_CARD_CONTENT_CHARS)))} characters.
11. If a drawing repeats labels like MATCH LINE many times, mention the repetition once instead of copying it.
12. If no meaningful card exists, return an empty cards array.

Return only valid JSON with this shape:
{{
  "cards": [
    {{
      "card_name": "short stable card name",
      "card_description": "brief definition of what the content is about",
      "card_source": "text | image | mixed",
      "tags": ["image"],
      "content": "actual content from this page related to the card",
      "image_descriptions": [
        {{
          "image_label": "Figure or visual identifier if present",
          "description": "detailed visual description with surrounding document context"
        }}
      ]
    }}
  ]
}}
""".strip()


def compact_page_text(text: str, max_chars: int) -> str:
    if not text:
        return ""

    lines = [line.strip() for line in text.splitlines()]
    compacted: list[str] = []
    counts: dict[str, int] = {}
    last_line = ""
    repeated_run = 0

    for line in lines:
        if not line:
            continue
        normalized = re.sub(r"\s+", " ", line).strip()
        key = normalized.lower()
        counts[key] = counts.get(key, 0) + 1

        if normalized == last_line:
            repeated_run += 1
            if repeated_run == 2:
                compacted.append(f"{normalized} [repeated]")
            continue

        repeated_run = 1
        last_line = normalized

        if counts[key] > 8 and len(normalized) < 80:
            continue
        compacted.append(normalized)

    result = "\n".join(compacted)
    if len(result) > max_chars:
        return result[:max_chars].rstrip() + "\n[truncated for indexing]"
    return result


def image_data_url(path: Path) -> str:
    mime = "image/png"
    if path.suffix.lower() in {".jpg", ".jpeg"}:
        mime = "image/jpeg"
    elif path.suffix.lower() == ".webp":
        mime = "image/webp"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def parse_json_response(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        json_object = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
        if json_object:
            candidate = json_object.group(0)
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                repaired = re.sub(r",\s*([}\]])", r"\1", candidate)
                repaired = re.sub(r"(?<!\\)'", '"', repaired)
                return json.loads(repaired)
        raise


def save_bad_response(context: str, text: str) -> None:
    BAD_RESPONSES_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    path = BAD_RESPONSES_DIR / f"{timestamp}_{slugify(context)}.txt"
    path.write_text(text, encoding="utf-8")


def card_match_prompt(
    raw_card: dict[str, Any],
    candidates: list[dict[str, Any]],
) -> str:
    candidate_payload = [
        {
            "card_name": entry["card_name"],
            "card_description": entry.get("card_description", ""),
            "content_excerpt": str(entry.get("content", ""))[:1200],
            "page_no": entry.get("page_no"),
            "document_name": entry.get("document_name"),
        }
        for entry in candidates
    ]
    return f"""
Decide whether the candidate card should become a version of an existing card.

Versioning rule:
- If an existing card has a similar name AND its content is similar to, continuing, or extending the candidate card, return same_card=true.
- If only the name is vaguely similar but the content is different, return same_card=false.
- If same_card=true, choose the best existing card and return its base name without any _vN suffix.
- Do not merge content. This decision is only for naming the new card.

Candidate card:
{json.dumps(raw_card, ensure_ascii=False)}

Existing candidate cards:
{json.dumps(candidate_payload, ensure_ascii=False)}

Return only valid JSON:
{{
  "same_card": true,
  "matched_base_name": "Card Name",
  "reason": "short reason"
}}
""".strip()


def decide_card_name_dry(raw_card: dict[str, Any], index: list[dict[str, Any]]) -> str:
    base_name = clean_card_name(str(raw_card.get("card_name", "Untitled Card")))
    for entry in candidate_existing_cards(raw_card, index):
        if likely_related_by_name(base_name, entry["card_name"]):
            return next_versioned_name(base_card_name(entry["card_name"]), index)
    return next_versioned_name(base_name, index)


def decide_card_name_openrouter(
    client: OpenAI,
    model: str,
    raw_card: dict[str, Any],
    index: list[dict[str, Any]],
) -> str:
    base_name = clean_card_name(str(raw_card.get("card_name", "Untitled Card")))
    candidates = candidate_existing_cards(raw_card, index)
    if not candidates:
        return next_versioned_name(base_name, index)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You compare document-index cards. Return only valid JSON.",
            },
            {"role": "user", "content": card_match_prompt(raw_card, candidates)},
        ],
        temperature=0,
        response_format={"type": "json_object"},
        max_tokens=int(os.getenv("OPENROUTER_CARD_MATCH_MAX_TOKENS", str(DEFAULT_CARD_MATCH_MAX_TOKENS))),
    )
    write_model_hit(
        "card_match",
        model,
        response,
        {"candidate_card": base_name},
    )
    message = response.choices[0].message.content or "{}"
    try:
        decision = parse_json_response(message)
    except json.JSONDecodeError:
        save_bad_response("card_match", message)
        print(
            "Warning: card-match model response was not valid JSON; "
            "using deterministic card version fallback."
        )
        return decide_card_name_dry(raw_card, index)
    if bool(decision.get("same_card")):
        matched_base = clean_card_name(str(decision.get("matched_base_name") or base_name))
        return next_versioned_name(base_card_name(matched_base), index)
    return next_versioned_name(base_name, index)


def openrouter_extract_page(
    client: OpenAI,
    model: str,
    page: PageUnit,
    index: list[dict[str, Any]],
) -> dict[str, Any]:
    content: list[dict[str, Any]] = [
        {"type": "text", "text": page_prompt(page, index)}
    ]
    if page.image_path:
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": image_data_url(PROJECT_ROOT / page.image_path)},
            }
        )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You are a careful document indexing assistant. Return only valid JSON.",
            },
            {"role": "user", "content": content},
        ],
        temperature=0.1,
        response_format={"type": "json_object"},
        max_tokens=int(os.getenv("OPENROUTER_INDEX_MAX_TOKENS", str(DEFAULT_INDEX_MAX_TOKENS))),
    )
    write_model_hit(
        "page_extract",
        model,
        response,
        {
            "document_id": page.document_id,
            "document_name": page.document_name,
            "page_no": page.page_no,
        },
    )
    message = response.choices[0].message.content or "{}"
    try:
        return parse_json_response(message)
    except json.JSONDecodeError:
        save_bad_response(f"page_{page.document_id[:12]}_{page.page_no}", message)
        raise


def dry_extract_page(page: PageUnit) -> dict[str, Any]:
    text = page.text.strip()
    if not text:
        text = f"Image-only page from {page.document_name}."
    heading = next((line.strip("# ").strip() for line in text.splitlines() if line.strip()), page.document_name)
    card_name = clean_card_name(heading[:80])
    image_descriptions = []
    if page.image_path:
        card_source = "mixed" if text else "image"
        tags = ["image"]
        image_descriptions.append(
            {
                "image_label": f"Page {page.page_no} render",
                "description": (
                    "Dry-run placeholder: this page has a rendered visual available. "
                    "Set OPENROUTER_API_KEY to generate a detailed model image description."
                ),
            }
        )
    else:
        card_source = "text"
        tags = []
    return {
        "cards": [
            {
                "card_name": card_name,
                "card_description": f"Content found on page {page.page_no} of {page.document_name}.",
                "card_source": card_source,
                "tags": tags,
                "content": text[:4000],
                "image_descriptions": image_descriptions,
            }
        ]
    }


def extract_page_with_retries(
    page: PageUnit,
    index: list[dict[str, Any]],
    client: OpenAI | None,
    model: str,
    dry_run: bool,
    max_attempts: int,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    if dry_run or client is None:
        return dry_extract_page(page), None

    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            return openrouter_extract_page(client, model, page, index), None
        except json.JSONDecodeError as exc:
            last_error = exc
            print(f"Warning: page {page.page_no} returned malformed JSON on attempt {attempt}/{max_attempts}.")
        except Exception as exc:
            last_error = exc
            print(f"Warning: page {page.page_no} failed on attempt {attempt}/{max_attempts}: {exc}")

    message = str(last_error or "Unknown page extraction error")
    if isinstance(last_error, json.JSONDecodeError):
        placeholder = page_issue(page, "placeholder_after_malformed_json", message, max_attempts)
        upsert_issue(PLACEHOLDER_PAGES_PATH, placeholder)
        print(
            f"Warning: page {page.page_no} still returned malformed JSON after "
            f"{max_attempts} attempt(s); using a placeholder card."
        )
        return dry_extract_page(page), placeholder

    failure = page_issue(page, "failed_after_retries", message, max_attempts)
    upsert_issue(FAILED_PAGES_PATH, failure)
    return None, failure


def normalize_card_source(raw_card: dict[str, Any]) -> str:
    value = str(raw_card.get("card_source", "text")).strip().lower()
    if value not in {"text", "image", "mixed"}:
        value = "image" if raw_card.get("image_descriptions") else "text"
    return value


def normalize_tags(raw_card: dict[str, Any], card_source: str) -> list[str]:
    raw_tags = raw_card.get("tags") or []
    if isinstance(raw_tags, str):
        raw_tags = [raw_tags]
    tags = {
        slugify(str(tag)).replace("_", "-")
        for tag in raw_tags
        if str(tag).strip()
    }
    if card_source in {"image", "mixed"} or raw_card.get("image_descriptions"):
        tags.add("image")
    return sorted(tags)


def append_cards(
    extracted: dict[str, Any],
    page: PageUnit,
    index: list[dict[str, Any]],
    client: OpenAI | None,
    model: str,
    dry_run: bool,
) -> int:
    added = 0
    for raw_card in extracted.get("cards", []):
        card_name = (
            decide_card_name_dry(raw_card, index)
            if dry_run or client is None
            else decide_card_name_openrouter(client, model, raw_card, index)
        )
        card_source = normalize_card_source(raw_card)
        entry = CardEntry(
            card_name=card_name,
            card_description=str(raw_card.get("card_description", "")).strip(),
            document_id=page.document_id,
            document_name=page.document_name,
            page_no=page.page_no,
            content=str(raw_card.get("content", "")).strip(),
            source_type="page",
            card_source=card_source,
            tags=normalize_tags(raw_card, card_source),
            image_descriptions=raw_card.get("image_descriptions") or [],
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        entry_dict = asdict(entry)
        index.append(entry_dict)
        added += 1
    return added


def index_documents(dry_run: bool, limit_pages: int | None, reset: bool) -> None:
    load_dotenv()
    INDEXES_DIR.mkdir(parents=True, exist_ok=True)
    manifest = read_json(MANIFEST_PATH, [])
    index = [] if reset else read_json(CARD_INDEX_PATH, [])

    if not manifest:
        raise SystemExit("No ingested documents found. Run: python ingest.py add <path>")

    client = None
    model = os.getenv("OPENROUTER_MODEL", DEFAULT_MODEL)
    if not dry_run:
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise SystemExit("Set OPENROUTER_API_KEY in .env, or run with --dry-run.")
        client = OpenAI(
            base_url=os.getenv("OPENROUTER_BASE_URL", OPENROUTER_BASE_URL),
            api_key=api_key,
            timeout=float(os.getenv("OPENROUTER_TIMEOUT_SECONDS", "120")),
            max_retries=int(os.getenv("OPENROUTER_MAX_RETRIES", "1")),
            default_headers={
                "HTTP-Referer": os.getenv("OPENROUTER_SITE_URL", "http://localhost"),
                "X-Title": os.getenv("OPENROUTER_APP_NAME", "Evidence Mesh"),
            },
        )

    processed_pages = 0
    total_pages = sum(count_record_pages(record) for record in manifest)
    max_attempts = max(1, int(os.getenv("OPENROUTER_PAGE_RETRIES", str(DEFAULT_PAGE_RETRIES))))
    indexed_pages = {
        (entry.get("document_id"), int(entry.get("page_no") or 0))
        for entry in index
    }
    failed_pages = read_json(FAILED_PAGES_PATH, [])
    placeholder_pages = read_json(PLACEHOLDER_PAGES_PATH, [])
    placeholder_keys = {issue_key(issue) for issue in placeholder_pages}
    write_pipeline_progress(
        {
            "stage": "indexing",
            "message": f"Starting page-by-page indexing for {total_pages} page(s).",
            "total_pages": total_pages,
            "indexed_pages": len(indexed_pages),
            "card_count": len(index),
            "failed_pages": len(failed_pages),
            "placeholder_pages": len(placeholder_pages),
        }
    )
    for record in manifest:
        for page in build_page_units(record):
            current_key = page_key(page)
            if not reset and current_key in indexed_pages and current_key not in placeholder_keys:
                continue
            if current_key in placeholder_keys:
                index = [
                    entry
                    for entry in index
                    if (str(entry.get("document_id", "")), int(entry.get("page_no") or 0)) != current_key
                ]
                indexed_pages.discard(current_key)
            if limit_pages is not None and processed_pages >= limit_pages:
                write_json(CARD_INDEX_PATH, index)
                print(f"Page limit reached. Wrote {len(index)} card entries.")
                return

            print(f"Indexing {page.document_name}, page {page.page_no}...")
            write_pipeline_progress(
                {
                    "stage": "indexing",
                    "message": f"Indexing {page.document_name}, page {page.page_no}.",
                    "total_pages": total_pages,
                    "indexed_pages": len(indexed_pages),
                    "current_document": page.document_name,
                    "current_page": page.page_no,
                    "card_count": len(index),
                }
            )
            extracted, issue = extract_page_with_retries(
                page=page,
                index=index,
                client=client,
                model=model,
                dry_run=dry_run,
                max_attempts=max_attempts,
            )
            if extracted is None:
                failed_pages = read_json(FAILED_PAGES_PATH, [])
                placeholder_pages = read_json(PLACEHOLDER_PAGES_PATH, [])
                write_pipeline_progress(
                    {
                        "stage": "indexing",
                        "message": (
                            f"Failed to index {page.document_name}, page {page.page_no} "
                            f"after {max_attempts} attempt(s). Continuing remaining pages."
                        ),
                        "total_pages": total_pages,
                        "indexed_pages": len(indexed_pages),
                        "current_document": page.document_name,
                        "current_page": page.page_no,
                        "card_count": len(index),
                        "failed_pages": len(failed_pages),
                        "placeholder_pages": len(placeholder_pages),
                    }
                )
                continue

            added = append_cards(
                extracted=extracted,
                page=page,
                index=index,
                client=client,
                model=model,
                dry_run=dry_run,
            )
            print(f"Added {added} card(s).")
            processed_pages += 1
            indexed_pages.add((page.document_id, page.page_no))
            failed_pages = clear_issue(FAILED_PAGES_PATH, page)
            if issue is None:
                placeholder_pages = clear_issue(PLACEHOLDER_PAGES_PATH, page)
            else:
                placeholder_pages = read_json(PLACEHOLDER_PAGES_PATH, [])
            write_json(CARD_INDEX_PATH, index)
            write_pipeline_progress(
                {
                    "stage": "indexing",
                    "message": f"Indexed {len(indexed_pages)} of {total_pages} page(s).",
                    "total_pages": total_pages,
                    "indexed_pages": len(indexed_pages),
                    "current_document": page.document_name,
                    "current_page": page.page_no,
                    "card_count": len(index),
                    "failed_pages": len(failed_pages),
                    "placeholder_pages": len(placeholder_pages),
                }
            )

    failed_pages = read_json(FAILED_PAGES_PATH, [])
    placeholder_pages = read_json(PLACEHOLDER_PAGES_PATH, [])
    if failed_pages or placeholder_pages:
        write_pipeline_progress(
            {
                "stage": "failed",
                "message": (
                    f"Indexing needs retry: {len(failed_pages)} failed page(s), "
                    f"{len(placeholder_pages)} placeholder page(s). "
                    "Rerun indexing before building relationships."
                ),
                "total_pages": total_pages,
                "indexed_pages": len(indexed_pages),
                "card_count": len(index),
                "failed_pages": len(failed_pages),
                "placeholder_pages": len(placeholder_pages),
            }
        )
        raise SystemExit(
            f"Indexing needs retry for {len(failed_pages)} failed page(s) and "
            f"{len(placeholder_pages)} placeholder page(s). See {FAILED_PAGES_PATH} and {PLACEHOLDER_PAGES_PATH}."
        )

    write_pipeline_progress(
        {
            "stage": "indexing_complete",
            "message": f"Index complete. Wrote {len(index)} card entries.",
            "total_pages": total_pages,
            "indexed_pages": len(indexed_pages),
            "card_count": len(index),
            "failed_pages": len(failed_pages),
            "placeholder_pages": len(placeholder_pages),
        }
    )
    print(f"Index complete. Wrote {len(index)} card entries to {CARD_INDEX_PATH}.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create a page-by-page card index.")
    parser.add_argument("--dry-run", action="store_true", help="Index without calling Gemini.")
    parser.add_argument("--limit-pages", type=int, help="Stop after N pages.")
    parser.add_argument("--reset", action="store_true", help="Start a fresh card index.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    index_documents(dry_run=args.dry_run, limit_pages=args.limit_pages, reset=args.reset)


if __name__ == "__main__":
    main()


