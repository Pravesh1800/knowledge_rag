from __future__ import annotations

import json
import os
import re
from typing import Any

from openai import OpenAI

from llm_config import create_chat_client, get_model


DEFAULT_RERANK_CANDIDATES = 40
DEFAULT_RERANK_CONTENT_CHARS = 1200
DEFAULT_RERANK_MAX_TOKENS = 1800


def parse_json_response(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def keyword_score(query: str, text: str) -> float:
    query_terms = {term for term in re.findall(r"[a-z0-9]+", query.lower()) if len(term) > 2}
    if not query_terms:
        return 0
    text_terms = set(re.findall(r"[a-z0-9]+", text.lower()))
    return len(query_terms & text_terms) / len(query_terms)


def rerank_enabled() -> bool:
    return os.getenv("EVIDENCE_MESH_RERANK", "1").strip().lower() not in {"0", "false", "no", "off"}


def candidate_limit() -> int:
    try:
        return max(1, int(os.getenv("EVIDENCE_MESH_RERANK_CANDIDATES", str(DEFAULT_RERANK_CANDIDATES))))
    except ValueError:
        return DEFAULT_RERANK_CANDIDATES


def candidate_view(hit: dict[str, Any], index: int) -> dict[str, Any]:
    related = []
    for item in (hit.get("related_cards") or [])[:2]:
        related.append(
            {
                "card_name": item.get("card_name", ""),
                "document_name": item.get("document_name", ""),
                "page_no": item.get("page_no", ""),
                "relationship_type": item.get("relationship_type", ""),
                "relationship_description": item.get("relationship_description", ""),
            }
        )
    return {
        "id": str(hit.get("card_id") or f"candidate_{index}"),
        "position": index,
        "card_name": hit.get("card_name", ""),
        "document_name": hit.get("document_name", ""),
        "page_no": hit.get("page_no", ""),
        "relevance_reason": hit.get("relevance_reason", ""),
        "card_source": hit.get("card_source", ""),
        "tags": hit.get("tags", []),
        "content": str(hit.get("content", ""))[:DEFAULT_RERANK_CONTENT_CHARS],
        "related_cards": related,
        "typed_anchors": {
            "entities": (hit.get("typed_anchors", {}) or {}).get("entities", [])[:6],
            "claims": (hit.get("typed_anchors", {}) or {}).get("claims", [])[:5],
            "canonical_entities": (hit.get("typed_anchors", {}) or {}).get("canonical_entities", [])[:6],
        },
    }


def rerank_prompt(query: str, candidates: list[dict[str, Any]], max_hits: int) -> str:
    return f"""
Rerank retrieved evidence cards for the user's question.

Question:
{query}

Candidates:
{json.dumps(candidates, ensure_ascii=False)}

Rules:
1. Prefer cards that directly answer the question with specific evidence.
2. Keep supporting cards when they provide necessary context, definitions, numbers, dates, exceptions, or contradictions.
3. Penalize generic, duplicate, tangential, or merely keyword-matching cards.
4. Return at most {max_hits} candidates.
5. Use each candidate id at most once.

Return only valid JSON:
{{
  "ranked": [
    {{
      "id": "candidate id",
      "score": 0.0,
      "reason": "Short evidence-specific reason"
    }}
  ]
}}
""".strip()


def fallback_rerank(query: str, hits: list[dict[str, Any]], max_hits: int) -> list[dict[str, Any]]:
    scored = []
    for index, hit in enumerate(hits):
        text = " ".join(
            [
                str(hit.get("card_name", "")),
                str(hit.get("document_name", "")),
                str(hit.get("relevance_reason", "")),
                str(hit.get("content", "")),
                " ".join(str(tag) for tag in hit.get("tags", []) or []),
            ]
        )
        score = keyword_score(query, text)
        score += max(0.0, 0.03 * (len(hits) - index) / max(1, len(hits)))
        scored.append((score, index, hit))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [dict(hit) for _score, _index, hit in scored[:max_hits]]


def rerank_hits(
    query: str,
    hits: list[dict[str, Any]],
    max_hits: int,
    *,
    client: OpenAI | None = None,
    model: str | None = None,
    dry_run: bool = False,
) -> list[dict[str, Any]]:
    if not hits:
        return []
    if not rerank_enabled() or dry_run:
        return fallback_rerank(query, hits, max_hits)

    limited_hits = hits[: candidate_limit()]
    views = [candidate_view(hit, index) for index, hit in enumerate(limited_hits)]
    hit_by_id = {view["id"]: limited_hits[index] for index, view in enumerate(views)}
    try:
        active_client = client
        active_model = model or get_model("search")
        if active_client is None:
            active_client, _default_model, _provider = create_chat_client()
            active_model = model or get_model("search")
        response = active_client.chat.completions.create(
            model=active_model,
            messages=[
                {
                    "role": "system",
                    "content": "You rerank evidence for retrieval augmented generation. Return only valid JSON.",
                },
                {"role": "user", "content": rerank_prompt(query, views, max_hits)},
            ],
            temperature=0,
            response_format={"type": "json_object"},
            max_tokens=int(os.getenv("EVIDENCE_MESH_RERANK_MAX_TOKENS", str(DEFAULT_RERANK_MAX_TOKENS))),
        )
        payload = parse_json_response(response.choices[0].message.content or "{}")
    except Exception:
        return fallback_rerank(query, hits, max_hits)

    ranked_hits: list[dict[str, Any]] = []
    used: set[str] = set()
    for item in payload.get("ranked", []) or []:
        candidate_id = str(item.get("id", ""))
        if not candidate_id or candidate_id in used or candidate_id not in hit_by_id:
            continue
        hit = dict(hit_by_id[candidate_id])
        reason = str(item.get("reason", "")).strip()
        if reason:
            hit["rerank_reason"] = reason
            hit["relevance_reason"] = f"{hit.get('relevance_reason', '')} Reranked: {reason}".strip()
        try:
            hit["rerank_score"] = float(item.get("score", 0.0))
        except (TypeError, ValueError):
            hit["rerank_score"] = 0.0
        ranked_hits.append(hit)
        used.add(candidate_id)
        if len(ranked_hits) >= max_hits:
            break

    if not ranked_hits:
        return fallback_rerank(query, hits, max_hits)

    for hit in limited_hits:
        if len(ranked_hits) >= max_hits:
            break
        candidate_id = str(hit.get("card_id") or f"candidate_{limited_hits.index(hit)}")
        if candidate_id not in used:
            ranked_hits.append(dict(hit))
            used.add(candidate_id)
    return ranked_hits[:max_hits]
