from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from openai import OpenAI

from community_summaries import top_community_summaries
from embeddings import vector_scores
from entity_canonicalizer import canonical_card_scores, canonical_entities_for_card
from entity_claims import anchors_for_card, entity_claim_card_scores
from llm_config import create_chat_client, get_model
from query_modes import QUERY_MODES, infer_query_mode, merge_query_variants, mode_override
from reranker import candidate_limit, rerank_hits
from schema import card_id_from_record, read_knowledge_graph
from searcher import TreeSearcher, keyword_score, parse_json_response
from storage import read_cards, record_search_run


MAX_CHAT_AGENT_STEPS = int(os.getenv("EVIDENCE_MESH_CHAT_AGENT_STEPS", "1"))
MAX_CHAT_EVIDENCE = int(os.getenv("EVIDENCE_MESH_CHAT_MAX_EVIDENCE", "18"))
MAX_CONTENT_CHARS = int(os.getenv("EVIDENCE_MESH_CHAT_CONTENT_CHARS", "1400"))
CHAT_AGENT_SCHEMA_VERSION = "agentic_chat.v1.0"


def env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def compact_history(history: list[dict[str, str]], limit: int = 8) -> list[dict[str, str]]:
    compacted = []
    for item in history[-limit:]:
        role = str(item.get("role", "")).strip().lower()
        content = str(item.get("content", "")).strip()
        if role in {"user", "assistant"} and content:
            compacted.append({"role": role, "content": content[:2000]})
    return compacted


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def stable_hit_key(hit: dict[str, Any]) -> str:
    return str(hit.get("card_id") or f"{hit.get('document_name')}::{hit.get('page_no')}::{hit.get('card_name')}")


def evidence_view(hits: list[dict[str, Any]], limit: int = MAX_CHAT_EVIDENCE) -> list[dict[str, Any]]:
    view = []
    for hit in hits[:limit]:
        related_cards = []
        for related in (hit.get("related_cards") or [])[:3]:
            related_cards.append(
                {
                    "card_name": related.get("card_name", ""),
                    "document_name": related.get("document_name", ""),
                    "page_no": related.get("page_no", ""),
                    "relationship_type": related.get("relationship_type", ""),
                    "relationship_description": related.get("relationship_description", ""),
                    "content": str(related.get("content", ""))[:900],
                }
            )
        view.append(
            {
                "card_id": hit.get("card_id", ""),
                "card_name": hit.get("card_name", ""),
                "document_name": hit.get("document_name", ""),
                "page_no": hit.get("page_no", ""),
                "relevance_reason": hit.get("relevance_reason", ""),
                "card_source": hit.get("card_source", ""),
                "tags": hit.get("tags", []),
                "content": str(hit.get("content", ""))[:MAX_CONTENT_CHARS],
                "related_cards": related_cards,
                "typed_anchors": {
                    "entities": (hit.get("typed_anchors", {}) or {}).get("entities", [])[:6],
                    "claims": (hit.get("typed_anchors", {}) or {}).get("claims", [])[:5],
                    "canonical_entities": (hit.get("typed_anchors", {}) or {}).get("canonical_entities", [])[:6],
                },
            }
        )
    return view


class AgenticChat:
    def __init__(
        self,
        project_root: Path,
        project_id: str,
        question: str,
        history: list[dict[str, str]],
        max_hits: int = 10,
        query_mode_override: str = "auto",
    ) -> None:
        self.project_root = project_root.resolve()
        self.project_id = project_id
        self.question = question.strip()
        self.history = compact_history(history)
        self.max_hits = max(4, min(30, int(max_hits or 10)))
        self.knowledge_graph_path = self.project_root / "indexes" / "knowledge_graph.json"
        self.cards = read_cards(project_id)
        self.graph = read_knowledge_graph(self.knowledge_graph_path, self.cards, persist_migration=True)
        self.graph_ready = bool(self.graph.get("domains") and self.graph.get("clusters"))
        self.relationship_ready = bool(self.graph.get("domain_relationships"))
        self.client, self.model, _provider = create_chat_client()
        self.search_model = get_model("search")
        self.search_dry_run = env_flag("EVIDENCE_MESH_CHAT_SEARCH_DRY_RUN", True)
        self.trace: list[dict[str, Any]] = []
        self.search_results: list[dict[str, Any]] = []
        self.evidence_by_key: dict[str, dict[str, Any]] = {}
        self.queries_run: set[str] = set()
        self.community_context: list[dict[str, Any]] = []
        inferred = infer_query_mode(self.question, self.history)
        if query_mode_override and query_mode_override != "auto" and query_mode_override in QUERY_MODES:
            self.query_mode = mode_override(query_mode_override, self.question)
        else:
            self.query_mode = inferred

    def log(self, event: str, detail: dict[str, Any] | None = None) -> None:
        self.trace.append({"created_at": now_iso(), "event": event, "detail": detail or {}})

    def call_json(self, prompt: str, fallback: dict[str, Any]) -> dict[str, Any]:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "Return only valid JSON."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
                response_format={"type": "json_object"},
            )
            parsed = parse_json_response(response.choices[0].message.content or "{}")
            return parsed if isinstance(parsed, dict) else fallback
        except Exception as exc:
            self.log("json_call_failed", {"error": str(exc)})
            return fallback

    def plan_prompt(self) -> str:
        graph_status = {
            "cards": len(self.cards),
            "domains": len(self.graph.get("domains", []) or []),
            "clusters": len(self.graph.get("clusters", []) or []),
            "relationships": len(self.graph.get("domain_relationships", []) or []),
            "graph_ready": self.graph_ready,
            "relationship_ready": self.relationship_ready,
            "query_mode": self.query_mode.to_dict(),
        }
        return f"""
You are an agentic Evidence Mesh chat planner.

The user may ask factual, analytical, comparative, risk, missing-information, or follow-up questions.
Decide whether retrieval is needed. If retrieval is needed, create multiple search probes that explore
the evidence mesh. The mesh layers are:
- domains, also called old biomes in older app language
- clusters, also called old communities in older app language
- cards/pages
- relationships between domains and cards

Conversation history:
{json.dumps(self.history, ensure_ascii=False)}

Question:
{self.question}

Graph status:
{json.dumps(graph_status, ensure_ascii=False)}

Return JSON:
{{
  "needs_retrieval": true,
  "intent": "factual_lookup|risk_analysis|comparison|summary|missing_information|follow_up|greeting|unknown",
  "answer_focus": "How the final answer should be structured",
  "success_criteria": ["Specific evidence the agent must find before answering"],
  "search_queries": ["3 to 6 targeted search queries"],
  "graph_exploration": true
}}
""".strip()

    def plan(self) -> dict[str, Any]:
        fallback = {
            "needs_retrieval": self.query_mode.needs_retrieval,
            "intent": "unknown",
            "answer_focus": "Answer directly with citations from retrieved evidence.",
            "success_criteria": ["Find directly relevant document evidence with page citations."],
            "search_queries": [self.question],
            "graph_exploration": self.query_mode.graph_exploration,
        }
        plan = self.call_json(self.plan_prompt(), fallback)
        plan["query_mode"] = self.query_mode.to_dict()
        plan["needs_retrieval"] = bool(plan.get("needs_retrieval", self.query_mode.needs_retrieval))
        plan["graph_exploration"] = bool(plan.get("graph_exploration", self.query_mode.graph_exploration))
        queries = plan.get("search_queries")
        if not isinstance(queries, list) or not queries:
            plan["search_queries"] = [self.question]
        plan["search_queries"] = merge_query_variants(
            self.question,
            [str(query).strip() for query in plan["search_queries"] if str(query).strip()],
            self.query_mode,
            limit=8,
        )
        self.log("plan", plan)
        return plan

    def lexical_card_search(self, query: str, max_hits: int) -> dict[str, Any]:
        semantic_scores = vector_scores(query, self.cards)
        anchor_scores = entity_claim_card_scores(query, self.cards)
        canonical_scores = canonical_card_scores(query)
        ranked = []
        for card in self.cards:
            text = " ".join(
                [
                    str(card.get("card_name", "")),
                    str(card.get("card_description", "")),
                    str(card.get("content", "")),
                    str(card.get("document_name", "")),
                    " ".join(str(tag) for tag in card.get("tags", []) or []),
                ]
            )
            lexical = keyword_score(query, text)
            card_id = card_id_from_record(card)
            semantic = semantic_scores.get(card_id, 0.0)
            anchor = anchor_scores.get(card_id, 0.0)
            canonical = canonical_scores.get(card_id, 0.0)
            score = (0.44 * semantic) + (0.28 * lexical) + (0.19 * anchor) + (0.09 * canonical)
            if score > 0:
                ranked.append((score, lexical, semantic, anchor, canonical, card))
        ranked.sort(key=lambda item: item[0], reverse=True)
        hits = []
        candidate_hits = []
        for score, lexical, semantic, anchor, canonical, card in ranked[: candidate_limit()]:
            card_id = card_id_from_record(card)
            anchors = anchors_for_card(card_id)
            hits.append(
                {
                    "card_id": str(card.get("card_id", "")),
                    "card_name": card.get("card_name", ""),
                    "document_name": card.get("document_name", ""),
                    "page_no": int(card.get("page_no") or 0),
                    "relevance_reason": (
                        f"Hybrid fallback score {score:.2f} "
                        f"(semantic {semantic:.2f}, entity/claim {anchor:.2f}, canonical entity {canonical:.2f}, keyword {lexical:.2f})."
                    ),
                    "content": str(card.get("content", "")),
                    "card_source": card.get("card_source", ""),
                    "tags": card.get("tags", []),
                    "related_cards": [],
                    "typed_anchors": {
                        "entities": (anchors.get("entities") or [])[:8],
                        "claims": (anchors.get("claims") or [])[:6],
                        "canonical_entities": canonical_entities_for_card(card_id)[:8],
                    },
                }
            )
            candidate_hits.append(hits[-1])
        hits = rerank_hits(
            query,
            candidate_hits,
            max_hits,
            client=self.client,
            model=self.search_model,
            dry_run=self.search_dry_run,
        )
        return {
            "schema_version": "lexical_search.v1.0",
            "query": query,
            "created_at": now_iso(),
            "trace": [{"node_type": "cards", "name": "PostgreSQL card index", "reason": "Graph unavailable or incomplete."}],
            "hits": hits,
        }

    def search_evidence(self, query: str, max_hits: int | None = None) -> dict[str, Any]:
        query = query.strip()
        if not query or query.lower() in self.queries_run:
            return {"query": query, "hits": [], "trace": []}
        self.queries_run.add(query.lower())
        max_hits = max_hits or self.max_hits
        self.log("tool_call", {"tool": "search_evidence", "query": query, "graph_ready": self.graph_ready})
        if self.graph_ready:
            try:
                result = TreeSearcher(
                    query=query,
                    dry_run=self.search_dry_run,
                    max_hits=max_hits,
                    storage_project_id=self.project_id,
                    knowledge_graph_path=self.knowledge_graph_path,
                    query_mode=self.query_mode.to_dict(),
                ).search()
            except Exception as exc:
                self.log("tool_error", {"tool": "search_evidence", "query": query, "error": str(exc)})
                result = self.lexical_card_search(query, max_hits)
        else:
            result = self.lexical_card_search(query, max_hits)
        result["search_run_id"] = f"chat_search_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S%f')}"
        record_search_run(result, self.project_id)
        self.search_results.append(result)
        for hit in result.get("hits", []):
            key = stable_hit_key(hit)
            if key not in self.evidence_by_key:
                self.evidence_by_key[key] = hit
        self.log(
            "tool_result",
            {
                "tool": "search_evidence",
                "query": query,
                "hits": len(result.get("hits", [])),
                "trace_steps": len(result.get("trace", [])),
            },
        )
        return result

    def inspect_graph(self, query: str) -> dict[str, Any]:
        self.log("tool_call", {"tool": "inspect_graph", "query": query})
        if not self.graph_ready:
            result = {
                "ready": False,
                "message": "Domains/clusters are not built yet; chat is using card-level search only.",
                "domains": [],
                "clusters": [],
                "relationships": [],
            }
            self.log("tool_result", {"tool": "inspect_graph", **result})
            return result

        def ranked_items(items: list[dict[str, Any]], name_key: str, desc_key: str, limit: int = 8) -> list[dict[str, Any]]:
            ranked = []
            for item in items:
                text = f"{item.get(name_key, '')} {item.get(desc_key, '')} {item.get('document_name', '')}"
                ranked.append((keyword_score(query, text), item))
            ranked.sort(key=lambda pair: pair[0], reverse=True)
            return [
                {
                    "score": round(score, 3),
                    "name": item.get(name_key, ""),
                    "description": item.get(desc_key, ""),
                    "document_name": item.get("document_name", ""),
                }
                for score, item in ranked[:limit]
                if score > 0
            ]

        relationships = []
        for rel in self.graph.get("domain_relationships", []) or []:
            text = " ".join(
                str(rel.get(key, ""))
                for key in ("main_domain", "related_domain", "relationship_type", "relationship_description", "evidence")
            )
            score = keyword_score(query, text)
            if score > 0:
                relationships.append(
                    {
                        "score": round(score, 3),
                        "main_domain": rel.get("main_domain", ""),
                        "related_domain": rel.get("related_domain", ""),
                        "relationship_type": rel.get("relationship_type", ""),
                        "evidence": rel.get("evidence", ""),
                    }
                )
        relationships.sort(key=lambda item: item["score"], reverse=True)
        communities = top_community_summaries(query, limit=6)
        self.community_context = communities
        result = {
            "ready": True,
            "domains": ranked_items(self.graph.get("domains", []) or [], "domain_name", "domain_description"),
            "clusters": ranked_items(self.graph.get("clusters", []) or [], "cluster_name", "cluster_description"),
            "relationships": relationships[:8],
            "community_summaries": [
                {
                    "score": item.get("score", 0),
                    "domain_name": item.get("domain_name", ""),
                    "summary": item.get("summary", ""),
                    "key_points": item.get("key_points", [])[:5],
                    "requirements": item.get("requirements", [])[:4],
                    "metrics": item.get("metrics", [])[:4],
                    "risks": item.get("risks", [])[:4],
                }
                for item in communities
            ],
        }
        self.log(
            "tool_result",
            {
                "tool": "inspect_graph",
                "domains": len(result["domains"]),
                "clusters": len(result["clusters"]),
                "relationships": len(result["relationships"]),
                "community_summaries": len(result["community_summaries"]),
            },
        )
        return result

    def judge_prompt(self, plan: dict[str, Any]) -> str:
        return f"""
You are the verifier for an agentic document chat run.
Decide if the gathered evidence is enough to answer the user's question.

Question:
{self.question}

Plan:
{json.dumps(plan, ensure_ascii=False)}

Adaptive query mode:
{json.dumps(self.query_mode.to_dict(), ensure_ascii=False)}

Evidence:
{json.dumps(evidence_view(list(self.evidence_by_key.values())), ensure_ascii=False)}

Community summaries:
{json.dumps(self.community_context[:6], ensure_ascii=False)}

Search trace:
{json.dumps(self.trace[-20:], ensure_ascii=False)}

Return JSON:
{{
  "satisfactory": true,
  "confidence": "high|medium|low",
  "reason": "Why evidence is enough or not enough",
  "missing": ["missing evidence or angles"],
  "next_queries": ["0 to 4 better searches if not satisfactory"]
}}
""".strip()

    def judge(self, plan: dict[str, Any]) -> dict[str, Any]:
        fallback = {
            "satisfactory": len(self.evidence_by_key) >= min(4, self.max_hits),
            "confidence": "medium" if self.evidence_by_key else "low",
            "reason": "Fallback judge based on evidence count.",
            "missing": [] if self.evidence_by_key else ["No evidence retrieved."],
            "next_queries": [],
        }
        result = self.call_json(self.judge_prompt(plan), fallback)
        if not isinstance(result.get("next_queries"), list):
            result["next_queries"] = []
        result["next_queries"] = [str(query).strip() for query in result["next_queries"][:4] if str(query).strip()]
        self.log("coverage_judge", result)
        return result

    def final_prompt(self, plan: dict[str, Any], judge: dict[str, Any]) -> str:
        return f"""
You are the final answer writer for Evidence Mesh.
The agent has already performed recursive retrieval and coverage checks.

Question:
{self.question}

Conversation history:
{json.dumps(self.history, ensure_ascii=False)}

Answer focus:
{plan.get("answer_focus", "")}

Adaptive query mode:
{json.dumps(self.query_mode.to_dict(), ensure_ascii=False)}

Verifier:
{json.dumps(judge, ensure_ascii=False)}

Evidence:
{json.dumps(evidence_view(list(self.evidence_by_key.values())), ensure_ascii=False)}

Community summaries:
{json.dumps(self.community_context[:6], ensure_ascii=False)}

Rules:
- Answer directly first.
- Use only the provided evidence for document facts.
- Cite document name and page number for important claims.
- If evidence is incomplete, say what is missing clearly.
- Match the answer style to the adaptive query mode: exact lookups should be direct; comparisons should separate each side; contradiction checks should name conflicts; gap analysis should distinguish found evidence from missing evidence; global summaries should synthesize across community summaries and cite supporting cards.
- Do not mention internal tool names unless the user asks how the agent worked.
- Keep the answer concise but complete.
""".strip()

    def final_answer(self, plan: dict[str, Any], judge: dict[str, Any]) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You answer from retrieved project evidence with citations."},
                    {"role": "user", "content": self.final_prompt(plan, judge)},
                ],
                temperature=0.2,
            )
            answer = response.choices[0].message.content or ""
            if answer.strip():
                return answer
            self.log("final_answer_empty", {"model": self.model})
        except Exception as exc:
            self.log("final_answer_failed", {"error": str(exc)})
        return self.fallback_answer(judge)

    def fallback_answer(self, judge: dict[str, Any]) -> str:
        hits = list(self.evidence_by_key.values())[:6]
        if not hits:
            missing = "; ".join(str(item) for item in judge.get("missing", []) or [])
            return f"I could not retrieve enough cited evidence to answer this. Missing: {missing or 'direct supporting evidence'}."
        lines = ["I found these cited evidence sources, but the answer model did not return a usable response:"]
        for hit in hits:
            lines.append(
                f"- {hit.get('card_name', 'Evidence')} ({hit.get('document_name', 'document')}, "
                f"page {hit.get('page_no', '?')}): {str(hit.get('content', ''))[:280]}"
            )
        return "\n".join(lines)

    def run_fast_lookup(self) -> dict[str, Any]:
        plan = {
            "needs_retrieval": True,
            "intent": "factual_lookup",
            "answer_focus": "Answer directly with the most relevant cited evidence.",
            "success_criteria": ["Find direct evidence with document and page citations."],
            "search_queries": [self.question],
            "graph_exploration": self.query_mode.graph_exploration,
            "query_mode": self.query_mode.to_dict(),
            "fast_path": "exact_lookup",
        }
        self.log("fast_path", {"mode": self.query_mode.mode, "query": self.question})
        result = self.search_evidence(self.question, max_hits=self.max_hits)
        judge = {
            "satisfactory": bool(result.get("hits")),
            "confidence": "medium" if result.get("hits") else "low",
            "reason": "Fast exact lookup uses the top retrieved evidence without iterative replanning.",
            "missing": [] if result.get("hits") else ["No relevant evidence retrieved."],
            "next_queries": [],
        }
        answer = self.final_answer(plan, judge)
        return {
            "schema_version": CHAT_AGENT_SCHEMA_VERSION,
            "answer": answer,
            "parser_query": self.question,
            "plan": plan,
            "coverage": judge,
            "search": {
                "schema_version": "agentic_combined_search.v1.0",
                "query": self.question,
                "created_at": now_iso(),
                "trace": result.get("trace", []),
                "hits": list(self.evidence_by_key.values())[:MAX_CHAT_EVIDENCE],
                "search_runs": [result],
            },
            "agent_trace": self.trace,
        }

    def run(self) -> dict[str, Any]:
        if not self.question:
            return {
                "schema_version": CHAT_AGENT_SCHEMA_VERSION,
                "answer": "Ask me a question about the project documents.",
                "plan": {},
                "search": {"hits": [], "trace": []},
                "agent_trace": [],
            }
        if not self.cards:
            return {
                "schema_version": CHAT_AGENT_SCHEMA_VERSION,
                "answer": "No indexed evidence cards are available yet. Upload documents and build the mesh first.",
                "plan": {},
                "search": {"hits": [], "trace": []},
                "agent_trace": [],
            }

        if env_flag("EVIDENCE_MESH_CHAT_FAST_EXACT", True) and self.query_mode.mode == "exact_lookup":
            return self.run_fast_lookup()

        plan = self.plan()
        if not bool(plan.get("needs_retrieval", True)):
            answer = self.final_answer(plan, {"satisfactory": True, "confidence": "high", "reason": "No retrieval needed.", "missing": []})
            return {
                "schema_version": CHAT_AGENT_SCHEMA_VERSION,
                "answer": answer,
                "plan": plan,
                "search": {"hits": [], "trace": []},
                "agent_trace": self.trace,
            }

        if bool(plan.get("graph_exploration", True)) or self.query_mode.use_community_summaries:
            self.inspect_graph(self.question)
        pending = list(plan.get("search_queries") or [self.question])
        judge = {"satisfactory": False, "confidence": "low", "reason": "Not checked yet.", "missing": [], "next_queries": []}
        max_steps = min(MAX_CHAT_AGENT_STEPS, max(1, int(self.query_mode.max_agent_steps)))
        mode_max_hits = max(4, min(30, int(round(self.max_hits * self.query_mode.max_hits_multiplier))))
        for step in range(1, max_steps + 1):
            if not pending:
                pending = [self.question]
            self.log("agent_step", {"step": step, "queries": pending[:3]})
            for query in pending[:3]:
                self.search_evidence(query, max_hits=mode_max_hits)
            judge = self.judge(plan)
            if bool(judge.get("satisfactory")):
                break
            pending = [
                query
                for query in judge.get("next_queries", [])
                if query.lower() not in self.queries_run
            ]
            if not pending:
                break

        answer = self.final_answer(plan, judge)
        hits = list(self.evidence_by_key.values())
        combined_trace = []
        for result in self.search_results:
            combined_trace.extend(result.get("trace", []))
        return {
            "schema_version": CHAT_AGENT_SCHEMA_VERSION,
            "answer": answer,
            "parser_query": "; ".join(plan.get("search_queries", []) or [self.question]),
            "plan": plan,
            "coverage": judge,
            "search": {
                "schema_version": "agentic_combined_search.v1.0",
                "query": self.question,
                "created_at": now_iso(),
                "trace": combined_trace,
                "hits": hits[:MAX_CHAT_EVIDENCE],
                "search_runs": self.search_results,
            },
            "agent_trace": self.trace,
        }


def run_agentic_chat(
    project_root: Path,
    project_id: str,
    question: str,
    history: list[dict[str, str]],
    max_hits: int,
    query_mode: str = "auto",
) -> dict[str, Any]:
    return AgenticChat(
        project_root=project_root,
        project_id=project_id,
        question=question,
        history=history,
        max_hits=max_hits,
        query_mode_override=query_mode,
    ).run()
