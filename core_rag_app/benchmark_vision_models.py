from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from openai import OpenAI


os.environ.setdefault(
    "EVIDENCE_MESH_ROOT",
    str(Path(__file__).resolve().parent / "projects" / "test"),
)
os.environ.setdefault("EVIDENCE_MESH_PROJECT_ID", "test")

from indexer import (  # noqa: E402
    DEFAULT_INDEX_MAX_TOKENS,
    PageUnit,
    build_pdf_pages,
    image_data_url,
    load_dotenv,
    page_prompt,
    parse_json_response,
    read_json,
)
from llm_config import get_base_url, get_model  # noqa: E402


APP_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = Path(os.environ["EVIDENCE_MESH_ROOT"])
OUT_PATH = Path(os.getenv("BENCHMARK_OUT_PATH", str(PROJECT_ROOT / "logs" / "vision_model_benchmark.json")))


def create_client() -> OpenAI:
    load_dotenv()
    api_key = os.getenv("LLM_API_KEY") or os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("No API key configured.")
    return OpenAI(
        base_url=get_base_url("openrouter"),
        api_key=api_key,
        timeout=float(os.getenv("OPENROUTER_TIMEOUT_SECONDS", "120")),
        max_retries=0,
        default_headers={
            "HTTP-Referer": os.getenv("OPENROUTER_SITE_URL", "http://localhost"),
            "X-Title": os.getenv("OPENROUTER_APP_NAME", "Evidence Mesh"),
        },
    )


def sample_pages() -> list[PageUnit]:
    manifest = read_json(PROJECT_ROOT / "documents" / "manifest.json", [])
    fixed_document = os.getenv("BENCHMARK_DOCUMENT_NAME", "").strip()
    fixed_pages = [
        int(item.strip())
        for item in os.getenv("BENCHMARK_PAGE_NUMBERS", "").split(",")
        if item.strip().isdigit()
    ]
    if fixed_document and fixed_pages:
        target_record = next(
            record
            for record in manifest
            if str(record.get("original_name", "")).lower() == fixed_document.lower()
        )
        wanted = set(fixed_pages)
        pages = []
        for page in build_pdf_pages(target_record):
            if page.page_no in wanted:
                pages.append(page)
            if len(pages) >= len(wanted):
                break
        return pages
    failed = read_json(PROJECT_ROOT / "indexes" / "failed_pages.json", [])
    target_names = {
        str(item.get("document_name", ""))
        for item in failed
        if str(item.get("document_name", "")).lower().endswith(".pdf")
    }
    target_pages = [
        int(item.get("page_no") or 0)
        for item in failed
        if str(item.get("document_name", "")) in target_names and int(item.get("page_no") or 0) > 0
    ][:5]
    target_record = next(
        (
            record
            for record in manifest
            if record.get("original_name") in target_names and record.get("extension") == ".pdf"
        ),
        None,
    )
    if not target_record:
        target_record = next(record for record in manifest if record.get("extension") == ".pdf")
        target_pages = [1, 2, 3, 4, 5]
    wanted = set(target_pages[:5])
    pages = []
    for page in build_pdf_pages(target_record):
        if page.page_no in wanted:
            pages.append(page)
        if len(pages) >= 5:
            break
    return pages


def extract_with_model(client: OpenAI, model: str, page: PageUnit) -> dict[str, Any]:
    prompt = page_prompt(page, [])
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    if page.image_path:
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": image_data_url(PROJECT_ROOT / page.image_path)},
            }
        )
    started = time.perf_counter()
    request: dict[str, Any] = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You are a careful document indexing assistant. Return only valid JSON.",
            },
            {"role": "user", "content": content},
        ],
        "temperature": 0.1,
        "response_format": {"type": "json_object"},
        "max_tokens": int(os.getenv("OPENROUTER_INDEX_MAX_TOKENS", str(DEFAULT_INDEX_MAX_TOKENS))),
    }
    if "gpt-5" in model.lower():
        request["max_tokens"] = int(os.getenv("GPT5_NANO_INDEX_MAX_TOKENS", str(request["max_tokens"])))
        if os.getenv("GPT5_NANO_REASONING_EFFORT"):
            request["reasoning_effort"] = os.getenv("GPT5_NANO_REASONING_EFFORT")
    response = client.chat.completions.create(**request)
    elapsed = round(time.perf_counter() - started, 2)
    raw = response.choices[0].message.content or "{}"
    parsed = parse_json_response(raw)
    usage = getattr(response, "usage", None)
    return {
        "ok": True,
        "elapsed_seconds": elapsed,
        "response_model": getattr(response, "model", ""),
        "usage": {
            "prompt_tokens": getattr(usage, "prompt_tokens", None),
            "completion_tokens": getattr(usage, "completion_tokens", None),
            "total_tokens": getattr(usage, "total_tokens", None),
        },
        "raw_chars": len(raw),
        "parsed": parsed,
    }


def score_result(result: dict[str, Any]) -> dict[str, Any]:
    if not result.get("ok"):
        return {
            "valid_json": False,
            "card_count": 0,
            "content_chars": 0,
            "image_description_count": 0,
            "specificity_score": 0,
        }
    cards = result.get("parsed", {}).get("cards", [])
    if not isinstance(cards, list):
        cards = []
    content = "\n".join(str(card.get("content", "")) for card in cards if isinstance(card, dict))
    image_descriptions = [
        desc
        for card in cards
        if isinstance(card, dict)
        for desc in (card.get("image_descriptions") or [])
        if isinstance(desc, dict)
    ]
    clause_terms = len(set(word.lower() for word in content.split() if any(char.isdigit() for char in word)))
    useful_fields = sum(
        1
        for card in cards
        if isinstance(card, dict)
        and card.get("card_name")
        and card.get("card_description")
        and card.get("content")
        and card.get("card_source")
    )
    return {
        "valid_json": True,
        "card_count": len(cards),
        "content_chars": len(content),
        "image_description_count": len(image_descriptions),
        "specificity_score": useful_fields + min(6, clause_terms),
    }


def main() -> None:
    client = create_client()
    models = os.getenv("BENCHMARK_MODELS")
    if models:
        model_list = [model.strip() for model in models.split(",") if model.strip()]
    else:
        model_list = [get_model(), os.getenv("GPT5_NANO_OPENROUTER_MODEL", "openai/gpt-5-nano")]
    page_limit = int(os.getenv("BENCHMARK_PAGE_LIMIT", "5"))
    pages = sample_pages()[:page_limit]
    output: dict[str, Any] = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "models": model_list,
        "pages": [
            {
                "document_id": page.document_id,
                "document_name": page.document_name,
                "page_no": page.page_no,
                "image_path": page.image_path,
                "text_chars": len(page.text or ""),
            }
            for page in pages
        ],
        "results": {},
    }
    for model in model_list:
        model_results = []
        for page in pages:
            try:
                result = extract_with_model(client, model, page)
            except Exception as exc:
                result = {"ok": False, "error": str(exc)}
            result["score"] = score_result(result)
            result["document_name"] = page.document_name
            result["page_no"] = page.page_no
            model_results.append(result)
        output["results"][model] = model_results
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
