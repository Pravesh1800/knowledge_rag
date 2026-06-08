from __future__ import annotations

import json
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from openai import OpenAI

import searcher as searcher_module
from commercial_rules import (
    COMMERCIAL_DRIVER_GUIDANCE,
    COMMERCIAL_FLAGS,
    COMMERCIAL_SCORING_CRITERIA,
    COMMERCIAL_SECTIONS,
    STRATEGY_TO_WIN_GUIDANCE,
    detect_commercial_flags,
)
from legal_assessment import (
    DEFAULT_AGENT_MAX_TOKENS,
    DEFAULT_MODEL,
    DEFAULT_SEARCH_MODEL,
    DEFAULT_WEB_RESEARCH_MODEL,
    DEFAULT_WEB_MAX_TOKENS,
    OPENROUTER_BASE_URL,
    compact_evidence,
    load_dotenv,
    json_completion,
    merge_evidence,
    parse_json_response,
    read_json,
    topic_to_evidence,
    write_json,
)


MAX_AGENT_STEPS = 16
MAX_EVIDENCE = 24

COMMERCIAL_SECTION_REQUIREMENTS = {
    "commercial_drivers": {
        "minimum_bullets": 8,
        "target_bullets": "8-10",
        "coverage_lanes": [
            "funding and payment confidence",
            "long-term O&M and recurring revenue",
            "reference value and strategic credentials",
            "scope scale and technical complexity",
            "high-value E&M or procurement packages",
            "pipeline, WTP, pumping, or civil execution scale",
            "client, consultant, authority, or stakeholder positioning",
            "future pipeline or follow-on opportunities",
            "pricing, cash-flow, or margin attractiveness",
        ],
    },
    "strategy_to_win": {
        "minimum_bullets": 5,
        "target_bullets": "5-7",
        "coverage_lanes": [
            "technical differentiation",
            "cost competitiveness and price strategy",
            "E&M procurement, vendor, and package strategy",
            "schedule, execution, and interface strategy",
            "qualification, JV, subcontractor, or partner strategy",
            "risk allocation, clarifications, and bid exclusions",
            "client, consultant, authority, and stakeholder positioning",
            "O&M, lifecycle cost, energy, chemical, or digital optimization",
        ],
    },
}


def create_client() -> tuple[OpenAI, str, str]:
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
    agent_model = os.getenv(
        "OPENROUTER_COMMERCIAL_AGENT_MODEL",
        os.getenv("OPENROUTER_LEGAL_AGENT_MODEL", os.getenv("OPENROUTER_MODEL", DEFAULT_MODEL)),
    )
    web_model = os.getenv("OPENROUTER_WEB_RESEARCH_MODEL", DEFAULT_WEB_RESEARCH_MODEL)
    return client, agent_model, web_model


class CommercialStrategyAgent:
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
        self.ensure_commercial_flags()
        self.topic_lookup = {topic.get("topic_name", ""): topic for topic in self.topics}
        self.community_lookup = {
            community.get("community_name", ""): community
            for community in self.map.get("communities", [])
        }
        self.biome_lookup = {
            biome.get("biome_name", ""): biome
            for biome in self.map.get("biomes", [])
        }
        load_dotenv(self.project_root)
        self.client, self.model, self.web_research_model = create_client()
        self.logs: list[dict[str, Any]] = []
        self.log_lock = threading.Lock()
        self.progress_path = self.reports_dir / "commercial_strategy.progress.json"

    def ensure_commercial_flags(self) -> None:
        changed = False
        for topic in self.topics:
            if "commercial_flags" in topic and "commercial_flag_reasons" in topic and "commercial_confidence" in topic:
                continue
            flags, reasons, confidence = detect_commercial_flags(topic)
            topic["commercial_flags"] = flags
            topic["commercial_flag_reasons"] = reasons
            topic["commercial_confidence"] = confidence
            changed = True
        if changed:
            write_json(self.indexes_dir / "topic_index.json", self.topics)

    def log(self, message: str, section_id: str | None = None, detail: dict[str, Any] | None = None) -> None:
        entry = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "section_id": section_id,
            "message": message,
            "detail": detail or {},
        }
        with self.log_lock:
            self.logs.append(entry)
            logs = list(self.logs)
            write_json(
                self.progress_path,
                {
                    "report_type": "commercial_drivers_strategy_to_win",
                    "status": "running",
                    "project_id": self.project_root.name,
                    "updated_at": entry["created_at"],
                    "logs": logs,
                },
            )

    def flagged_evidence(self, flag_ids: list[str]) -> list[dict[str, Any]]:
        evidence = []
        wanted = set(flag_ids)
        for topic in self.topics:
            flags = set(topic.get("commercial_flags") or [])
            if not flags & wanted:
                continue
            reasons = topic.get("commercial_flag_reasons") or {}
            reason = "; ".join(reasons.get(flag, "") for flag in flags & wanted if reasons.get(flag))
            evidence.append(topic_to_evidence(topic, "flagged", reason or "Marked during indexing as commercially relevant."))
        return evidence

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
                evidence.append(topic_to_evidence(topic, "searched", hit.get("relevance_reason", "Found by targeted search.")))
            for related in hit.get("related_topics", []):
                related_topic = by_triplet.get((related.get("topic_name"), related.get("document_name"), int(related.get("page_no") or 0)))
                if related_topic:
                    evidence.append(topic_to_evidence(related_topic, "searched", related.get("relation", "Related topic found by search.")))
        return evidence

    def project_inventory(self) -> dict[str, Any]:
        return {
            "documents": sorted({topic.get("document_name", "") for topic in self.topics if topic.get("document_name")}),
            "biomes": self.list_biomes(),
            "community_count": len(self.map.get("communities", [])),
            "topic_count": len(self.topics),
            "commercial_flags": [
                {"flag": flag["id"], "label": flag["label"], "count": sum(1 for topic in self.topics if flag["id"] in (topic.get("commercial_flags") or []))}
                for flag in COMMERCIAL_FLAGS
            ],
        }

    def list_biomes(self) -> list[dict[str, Any]]:
        return [
            {
                "biome_name": biome.get("biome_name", ""),
                "biome_description": biome.get("biome_description", ""),
                "document_name": biome.get("document_name", ""),
                "community_names": biome.get("community_names", []),
            }
            for biome in self.map.get("biomes", [])
        ]

    def get_biome(self, biome_name: str) -> dict[str, Any]:
        biome = self.biome_lookup.get(biome_name)
        if not biome:
            return {"error": f"Biome not found: {biome_name}"}
        return {
            "biome": biome,
            "communities": [
                self.community_lookup.get(name, {"community_name": name, "error": "missing"})
                for name in biome.get("community_names", [])
            ],
        }

    def list_communities(self, document_name: str | None = None) -> list[dict[str, Any]]:
        communities = self.map.get("communities", [])
        if document_name:
            communities = [
                community for community in communities
                if document_name.lower() in str(community.get("document_name", "")).lower()
            ]
        return [
            {
                "community_name": community.get("community_name", ""),
                "community_description": community.get("community_description", ""),
                "document_name": community.get("document_name", ""),
                "topic_names": community.get("topic_names", []),
            }
            for community in communities
        ]

    def get_community(self, community_name: str) -> dict[str, Any]:
        community = self.community_lookup.get(community_name)
        if not community:
            return {"error": f"Community not found: {community_name}"}
        topics = [
            self.topic_lookup.get(name, {"topic_name": name, "error": "missing"})
            for name in community.get("topic_names", [])
        ]
        return {
            "community": community,
            "topics": [
                {
                    "topic_name": topic.get("topic_name", ""),
                    "topic_description": topic.get("topic_description", ""),
                    "document_name": topic.get("document_name", ""),
                    "page_no": topic.get("page_no"),
                    "commercial_flags": topic.get("commercial_flags", []),
                    "content_excerpt": str(topic.get("content", ""))[:1200],
                }
                for topic in topics
            ],
        }

    def list_topics(self, query: str | None = None, limit: int = 40) -> list[dict[str, Any]]:
        rows = self.topics
        if query:
            terms = [term for term in re.findall(r"[a-z0-9]+", query.lower()) if len(term) > 2]
            scored = []
            for topic in rows:
                text = " ".join(
                    str(topic.get(key, ""))
                    for key in ["topic_name", "topic_description", "content", "document_name"]
                ).lower()
                score = sum(1 for term in terms if term in text)
                if score:
                    scored.append((score, topic))
            scored.sort(key=lambda item: item[0], reverse=True)
            rows = [topic for _, topic in scored]
        return [
            {
                "topic_name": topic.get("topic_name", ""),
                "topic_description": topic.get("topic_description", ""),
                "document_name": topic.get("document_name", ""),
                "page_no": topic.get("page_no"),
                "commercial_flags": topic.get("commercial_flags", []),
            }
            for topic in rows[:limit]
        ]

    def get_topic(self, topic_name: str) -> dict[str, Any]:
        topic = self.topic_lookup.get(topic_name)
        if not topic:
            return {"error": f"Topic not found: {topic_name}"}
        return topic

    def keyword_lookup(self, terms: list[str], limit: int = 12) -> list[dict[str, Any]]:
        terms = [term.lower() for term in terms if term]
        scored = []
        for topic in self.topics:
            text = " ".join(
                str(topic.get(key, ""))
                for key in ["topic_name", "topic_description", "content", "document_name"]
            ).lower()
            score = sum(1 for term in terms if term in text)
            if score:
                scored.append((score, topic))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [topic_to_evidence(topic, "agent_lookup", f"Agent keyword lookup matched {score} terms.") for score, topic in scored[:limit]]

    def web_research(self, query: str, purpose: str) -> dict[str, Any]:
        prompt = f"""
Research the public web for commercial bid strategy context.

Query:
{query}

Purpose:
{purpose}

Return JSON:
{{
  "summary": "short answer",
  "findings": ["finding with source context"],
  "citations": [
    {{"title": "source title", "url": "https://...", "note": "what it supports"}}
  ],
  "limits": "what could not be verified"
}}
""".strip()
        try:
            response = self.client.chat.completions.create(
                model=self.web_research_model,
                messages=[
                    {"role": "system", "content": "You are an internet research assistant for bid strategy. Cite URLs."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
                max_tokens=int(os.getenv("OPENROUTER_WEB_MAX_TOKENS", str(DEFAULT_WEB_MAX_TOKENS))),
            )
            choices = response.choices or []
            content = choices[0].message.content if choices else ""
        except Exception as exc:
            content = ""
            parsed = {
                "summary": "",
                "findings": [],
                "citations": [],
                "limits": f"Web research provider failed non-fatally: {type(exc).__name__}: {exc}",
            }
            return {"model": self.web_research_model, "query": query, "purpose": purpose, "result": parsed}
        if not content:
            parsed = {"summary": "", "findings": [], "citations": [], "limits": "Web research provider returned no content."}
            return {"model": self.web_research_model, "query": query, "purpose": purpose, "result": parsed}
        try:
            parsed = parse_json_response(content)
        except Exception:
            parsed = {"summary": content, "findings": [], "citations": [], "limits": "Model did not return parseable JSON."}
        return {"model": self.web_research_model, "query": query, "purpose": purpose, "result": parsed}

    def run_action(self, action: dict[str, Any], section: dict[str, Any]) -> dict[str, Any]:
        action_type = str(action.get("action", "")).strip()
        if action_type == "search":
            query = str(action.get("query", "")).strip() or section["title"]
            self.log(f"Running an extra project search for: {query}", section["id"])
            return {"evidence": compact_evidence(self.searched_evidence(query), limit=MAX_EVIDENCE)}
        if action_type == "keyword_lookup":
            terms = [str(term) for term in action.get("terms", []) if str(term).strip()]
            self.log(f"Scanning all indexed topics for exact commercial terms: {', '.join(terms[:8])}", section["id"])
            return {"evidence": compact_evidence(self.keyword_lookup(terms), limit=MAX_EVIDENCE)}
        if action_type == "list_inventory":
            self.log("Reviewing the project inventory, commercial flags, biomes, documents, and topic coverage.", section["id"])
            return {"inventory": self.project_inventory()}
        if action_type == "list_biomes":
            self.log("Listing all biomes for commercial context.", section["id"])
            return {"biomes": self.list_biomes()}
        if action_type == "get_biome":
            name = str(action.get("biome_name", "")).strip()
            self.log(f"Opening biome: {name}", section["id"])
            return self.get_biome(name)
        if action_type == "list_communities":
            document_name = str(action.get("document_name", "")).strip() or None
            self.log(f"Listing communities{f' for {document_name}' if document_name else ''}.", section["id"])
            return {"communities": self.list_communities(document_name)}
        if action_type == "get_community":
            name = str(action.get("community_name", "")).strip()
            self.log(f"Opening community and its topics: {name}", section["id"])
            return self.get_community(name)
        if action_type == "list_topics":
            query = str(action.get("query", "")).strip() or None
            self.log(f"Listing topics{f' matching: {query}' if query else ''}.", section["id"])
            return {"topics": self.list_topics(query=query, limit=int(action.get("limit") or 40))}
        if action_type == "get_topic":
            name = str(action.get("topic_name", "")).strip()
            self.log(f"Opening full topic content: {name}", section["id"])
            topic = self.get_topic(name)
            if "error" not in topic:
                return {"topic": topic, "evidence": compact_evidence([topic_to_evidence(topic, "agent_inspected", "Commercial specialist opened full topic content.")], limit=1)}
            return topic
        if action_type == "web_research":
            query = str(action.get("query", "")).strip()
            purpose = str(action.get("purpose", "")).strip() or section["goal"]
            self.log(f"Researching the public web with Sonar Reasoning Pro: {query}", section["id"])
            return {"web_research": self.web_research(query, purpose)}
        if action_type == "answer":
            return {"answer": action}
        self.log("Commercial specialist requested an unknown action.", section["id"], {"action": action})
        return {"error": "unknown action"}

    def specialist_prompt(self, section: dict[str, Any], evidence: list[dict[str, Any]], step_results: list[dict[str, Any]]) -> str:
        guidance = (
            COMMERCIAL_DRIVER_GUIDANCE
            if section.get("id") == "commercial_drivers"
            else STRATEGY_TO_WIN_GUIDANCE
        )
        requirements = self.section_requirements(section)
        return f"""
You are a specialist commercial bid-strategy agent for SUEZ.
Your job is to generate one section of the "Commercial Drivers and Strategy to WIN" document.
This is not a fixed checklist. The starter angles are hints, not mandatory headings. Discover the strongest project-specific bullets from the evidence.

Section:
{json.dumps(section, ensure_ascii=False)}

Generic guidance for this section:
{json.dumps(guidance, ensure_ascii=False)}

Completion requirements for this section:
{json.dumps(requirements, ensure_ascii=False)}

Bullet scoring criteria:
{json.dumps(COMMERCIAL_SCORING_CRITERIA, ensure_ascii=False)}

Available evidence from project flags and fresh searches:
{json.dumps(compact_evidence(evidence, limit=MAX_EVIDENCE), ensure_ascii=False)}

Previous tool results:
{json.dumps(step_results[-8:], ensure_ascii=False)}

Available actions. Return exactly one JSON object:
1. {{"action":"search","query":"specific project search query"}}
2. {{"action":"keyword_lookup","terms":["exact term", "another term"]}}
3. {{"action":"list_inventory"}}
4. {{"action":"list_biomes"}}
5. {{"action":"get_biome","biome_name":"exact biome name"}}
6. {{"action":"list_communities","document_name":"optional document name filter"}}
7. {{"action":"get_community","community_name":"exact community name"}}
8. {{"action":"list_topics","query":"optional keyword query","limit":40}}
9. {{"action":"get_topic","topic_name":"exact topic name"}}
10. {{"action":"web_research","query":"public web research query","purpose":"why external context is needed"}}
11. {{"action":"answer","bullets":[{{"text":"slide-style bullet","why_it_matters":"commercial or win-strategy implication","score":{{"commercial_value":0,"strategic_value":0,"evidence_strength":0,"win_relevance":0,"specificity":0,"risk_caveat":"none|low|medium|high"}},"evidence_citations":[{{"document_name":"","page_no":0,"topic_name":"","excerpt":""}}],"web_citations":[{{"title":"","url":"","note":""}}],"basis":"document|web|inference","caveat":"short caveat if any"}}],"rejected_candidates":[{{"text":"candidate considered but rejected","reason":"why weak or unsupported"}}],"notes":"short audit note","confidence":"low|medium|high"}}

Rules:
- Use the generic guidance as calibration only. Do not copy the examples as project facts.
- Use project/tender evidence for internal facts.
- Use web_research for public funding, market, competitor, award, authority, or company context.
- Do not claim SUEZ participation, relationships, or win themes unless supported by project evidence or clearly labelled as inference.
- Keep bullets concise and slide-ready.
- Every bullet must include why_it_matters and explain why it matters commercially or strategically to SUEZ.
- Score each bullet using the scoring criteria. Use 0-5 for numeric score fields. Higher is better.
- Prefer bullets with strong evidence_strength, high commercial_value or strategic_value, and clear win_relevance.
- Reject generic bullets such as "be competitive", "use technology", "manage risk", or "project is important" unless they are tied to specific evidence and a specific implication.
- Separate section logic: Commercial Drivers means why the project is attractive; Strategy to WIN means how SUEZ should position, price, qualify, partner, or execute to win.
- Add drivers/themes not listed in starter angles when evidence shows they are important.
- Omit starter angles when evidence is weak or they are not among the strongest points.
- Continue iterating until the section has the strongest evidence-backed bullets with citations.
- When returning action=answer, produce the full required answer: at least {requirements["minimum_bullets"]} bullets and target {requirements["target_bullets"]} bullets.
- Cover distinct lanes where evidence supports them: {", ".join(requirements["coverage_lanes"])}.
- Do not compress the section because JSON repair or fallback is needed. Keep each bullet concise, but preserve the full number of distinct points.
""".strip()

    def verifier_prompt(self, section: dict[str, Any], draft: dict[str, Any], evidence: list[dict[str, Any]], step_results: list[dict[str, Any]]) -> str:
        guidance = (
            COMMERCIAL_DRIVER_GUIDANCE
            if section.get("id") == "commercial_drivers"
            else STRATEGY_TO_WIN_GUIDANCE
        )
        requirements = self.section_requirements(section)
        return f"""
Verify one section of a SUEZ Commercial Drivers and Strategy to WIN document.
The generated section is intentionally flexible. Do not require every starter angle to appear.

Section:
{json.dumps(section, ensure_ascii=False)}

Generic guidance for this section:
{json.dumps(guidance, ensure_ascii=False)}

Completion requirements for this section:
{json.dumps(requirements, ensure_ascii=False)}

Scoring criteria:
{json.dumps(COMMERCIAL_SCORING_CRITERIA, ensure_ascii=False)}

Draft:
{json.dumps(draft, ensure_ascii=False)}

Project evidence:
{json.dumps(compact_evidence(evidence, limit=MAX_EVIDENCE), ensure_ascii=False)}

Tool results including web research:
{json.dumps(step_results[-8:], ensure_ascii=False)}

Return only JSON:
{{
  "verified": true,
  "bullets": [
    {{"text":"verified slide bullet","why_it_matters":"commercial or win-strategy implication","score":{{"commercial_value":0,"strategic_value":0,"evidence_strength":0,"win_relevance":0,"specificity":0,"risk_caveat":"none|low|medium|high"}},"evidence_citations":[],"web_citations":[],"basis":"document|web|inference","caveat":"short caveat if any"}}
  ],
  "rejected_candidates": [
    {{"text":"removed or downgraded candidate","reason":"unsupported, generic, low score, wrong section, or weak evidence"}}
  ],
  "warnings": ["unsupported or caveated items"],
  "verification_note": "short note"
}}

Rules:
- Remove unsupported claims.
- Remove generic bullets that do not say why the point matters to SUEZ.
- Confirm each bullet belongs in this section: drivers explain attractiveness; strategy explains how to win.
- Prefer the highest-value bullets by commercial_value, strategic_value, evidence_strength, win_relevance, and specificity.
- Preserve caveats for inferred or web-supported points.
- Keep public web facts separate from project document facts.
- Numbers, funding, scope, competitors, and future opportunities need citations or explicit inference labels.
- Preserve project-specific bullets that are commercially meaningful even if they do not match the starter angles.
- Remove weak starter-angle bullets if the evidence does not make them strategically important.
- Do not verify an under-filled section. The final bullets must contain at least {requirements["minimum_bullets"]} non-duplicate bullets and should target {requirements["target_bullets"]}.
- If the draft has fewer than {requirements["minimum_bullets"]} valid bullets, add missing evidence-backed bullets instead of accepting a short answer.
""".strip()

    def call_json(self, prompt: str, system: str) -> dict[str, Any]:
        return json_completion(
            self.client,
            self.model,
            prompt,
            system,
            int(os.getenv("OPENROUTER_AGENT_MAX_TOKENS", str(DEFAULT_AGENT_MAX_TOKENS))),
        )

    def section_requirements(self, section: dict[str, Any]) -> dict[str, Any]:
        return COMMERCIAL_SECTION_REQUIREMENTS.get(section.get("id", ""), COMMERCIAL_SECTION_REQUIREMENTS["commercial_drivers"])

    def full_answer_instruction(self, section: dict[str, Any], reason: str) -> str:
        requirements = self.section_requirements(section)
        return f"""
You must now return action=answer for the full section.

Reason:
{reason}

Hard requirements:
- Return valid JSON only.
- Return action=answer.
- Produce at least {requirements["minimum_bullets"]} non-duplicate bullets.
- Target {requirements["target_bullets"]} bullets if evidence supports them.
- Cover distinct lanes where evidence supports them: {", ".join(requirements["coverage_lanes"])}.
- Keep each bullet concise, but do not reduce the number of bullets.
- Every bullet must include text, why_it_matters, score, evidence_citations, web_citations, basis, and caveat.
- Do not return a partial or compressed answer because of JSON repair.
""".strip()

    def expand_underfilled_draft(
        self,
        section: dict[str, Any],
        draft: dict[str, Any],
        evidence: list[dict[str, Any]],
        step_results: list[dict[str, Any]],
        reason: str,
    ) -> dict[str, Any]:
        requirements = self.section_requirements(section)
        bullet_count = len(draft.get("bullets", []) if isinstance(draft, dict) else [])
        if bullet_count >= requirements["minimum_bullets"]:
            return draft
        self.log(
            f"Commercial section is under-filled ({bullet_count}/{requirements['minimum_bullets']} bullets); expanding before verification.",
            section["id"],
            {"reason": reason, "target_bullets": requirements["target_bullets"]},
        )
        expanded = self.call_json(
            self.specialist_prompt(section, evidence, step_results)
            + "\n\nExisting under-filled draft:\n"
            + json.dumps(draft, ensure_ascii=False)
            + "\n\n"
            + self.full_answer_instruction(section, reason),
            "Return only valid JSON with action=answer and a full bullets array. Do not request tools.",
        )
        expanded = self.normalize_action(expanded)
        if expanded.get("action") == "answer":
            return expanded
        return draft

    def normalize_action(self, action: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(action, dict):
            return {"action": "unknown"}
        if not action.get("action") and isinstance(action.get("bullets"), list):
            action = dict(action)
            action["action"] = "answer"
        if action.get("action") == "answer" and "answer" in action and isinstance(action["answer"], dict):
            nested = dict(action["answer"])
            nested["action"] = "answer"
            return nested
        return action

    def starter_query(self, section: dict[str, Any]) -> str:
        return f"{section['title']}. {section['goal']} Consider possible angles: {', '.join(section['starter_angles'])}. Also find any other project-specific commercial drivers or win themes supported by evidence."

    def section_flag_ids(self, section: dict[str, Any]) -> list[str]:
        if section["id"] == "commercial_drivers":
            return ["funding_source", "long_term_om", "reference_project", "future_opportunity", "client_relationship", "competition"]
        return ["technical_differentiation", "cost_strategy", "optimization_lever", "client_relationship", "competition"]

    def answer_section(self, section: dict[str, Any]) -> dict[str, Any]:
        self.log(f"Starting commercial specialist review for section: {section['title']}", section["id"])
        flagged = self.flagged_evidence(self.section_flag_ids(section))
        self.log(f"Collected {len(flagged)} pre-flagged commercial topic(s) from the index.", section["id"])
        searched = self.searched_evidence(self.starter_query(section))
        self.log(f"Ran the required fresh targeted project search and collected {len(searched)} searched topic(s).", section["id"])
        evidence = merge_evidence(flagged + searched)
        self.log(f"Merged and deduplicated commercial evidence to {len(evidence)} unique topic/page candidate(s).", section["id"])

        step_results: list[dict[str, Any]] = []
        draft: dict[str, Any] | None = None
        for step in range(1, MAX_AGENT_STEPS + 1):
            self.log(f"Commercial specialist thinking step {step}: deciding whether more project or web context is needed.", section["id"])
            try:
                action = self.call_json(
                    self.specialist_prompt(section, evidence, step_results),
                    "Return only valid JSON. You are a commercial bid-strategy specialist.",
                )
            except Exception as exc:
                self.log(
                    "Commercial specialist returned malformed JSON after repair attempts; forcing a complete full-section answer from accumulated evidence.",
                    section["id"],
                    {"error_type": type(exc).__name__, "error": str(exc)},
                )
                draft = self.call_json(
                    self.specialist_prompt(section, evidence, step_results)
                    + "\n\n"
                    + self.full_answer_instruction(section, "JSON repair failed, but the section must remain complete."),
                    "Return only valid JSON with action=answer and a complete bullets array. Do not request tools.",
                )
                draft = self.normalize_action(draft)
                if draft.get("action") == "answer":
                    draft = self.expand_underfilled_draft(
                        section,
                        draft,
                        evidence,
                        step_results,
                        "Recovery answer was below the required minimum.",
                    )
                    self.log(f"Commercial specialist drafted {len(draft.get('bullets', []))} bullet(s) for {section['title']}.", section["id"])
                    break
                raise
            action = self.normalize_action(action)
            self.log(f"Commercial specialist requested tool action: {action.get('action', 'unknown')}.", section["id"], {"action": action})
            result = self.run_action(action, section)
            self.log(f"Tool action completed: {action.get('action', 'unknown')}.", section["id"], {"result_keys": sorted(result.keys())})
            step_results.append({"action": action, "result": result})
            for item in result.get("evidence", []):
                evidence.append(
                    {
                        "document_id": item.get("document_id", ""),
                        "document_name": item.get("document_name", ""),
                        "page_no": item.get("page_no"),
                        "topic_name": item.get("topic_name", ""),
                        "topic_description": item.get("topic_description", ""),
                        "content": item.get("excerpt", ""),
                        "source_channels": ["agent_lookup"],
                        "match_reason": item.get("match_reason", ""),
                    }
                )
            evidence = merge_evidence(evidence)
            if "answer" in result:
                draft = result["answer"]
                self.log(f"Commercial specialist drafted {len(draft.get('bullets', []))} bullet(s) for {section['title']}.", section["id"])
                break

        if draft is None:
            self.log("Commercial specialist reached the step limit; forcing an answer from accumulated evidence.", section["id"])
            draft = self.call_json(
                self.specialist_prompt(section, evidence, step_results)
                + "\n\n"
                + self.full_answer_instruction(section, "The specialist reached the step limit."),
                "Return only valid JSON with action=answer and a complete bullets array.",
            )
            draft = self.normalize_action(draft)

        draft = self.expand_underfilled_draft(
            section,
            draft,
            evidence,
            step_results,
            "Draft was below the required minimum before verification.",
        )

        verifier = self.call_json(
            self.verifier_prompt(section, draft, evidence, step_results),
            "Return only valid JSON. You are a strict verifier of commercial bid-strategy bullets.",
        )
        verifier_bullet_count = len(verifier.get("bullets", []) if isinstance(verifier, dict) else [])
        requirements = self.section_requirements(section)
        if verifier_bullet_count < requirements["minimum_bullets"]:
            self.log(
                f"Verifier output is under-filled ({verifier_bullet_count}/{requirements['minimum_bullets']} bullets); expanding final answer.",
                section["id"],
                {"target_bullets": requirements["target_bullets"]},
            )
            expanded = self.expand_underfilled_draft(
                section,
                {"action": "answer", "bullets": verifier.get("bullets", []) or draft.get("bullets", []), "confidence": draft.get("confidence", "medium")},
                evidence,
                step_results,
                "Verifier output was below the required minimum.",
            )
            verifier["bullets"] = expanded.get("bullets", verifier.get("bullets", []))
        self.log(
            f"Verifier finalized {len(verifier.get('bullets', draft.get('bullets', [])))} bullet(s) for {section['title']}.",
            section["id"],
            {"warnings": verifier.get("warnings", [])},
        )
        return {
            "section_id": section["id"],
            "title": section["title"],
            "bullets": verifier.get("bullets") or draft.get("bullets", []),
            "evidence": compact_evidence(evidence, limit=MAX_EVIDENCE),
            "agent_steps": step_results,
            "verifier": verifier,
            "confidence": draft.get("confidence", "medium"),
        }

    def generate(self) -> dict[str, Any]:
        self.log("Starting Commercial Drivers and Strategy to WIN generation.")
        self.log("Running Commercial Drivers and Strategy to WIN sections in parallel.")
        sections_by_id: dict[str, dict[str, Any]] = {}
        max_workers = min(2, len(COMMERCIAL_SECTIONS))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self.answer_section, section): section
                for section in COMMERCIAL_SECTIONS
            }
            for future in as_completed(futures):
                section = futures[future]
                sections_by_id[section["id"]] = future.result()
        sections = [
            sections_by_id[section["id"]]
            for section in COMMERCIAL_SECTIONS
            if section["id"] in sections_by_id
        ]
        report = {
            "report_type": "commercial_drivers_strategy_to_win",
            "status": "verified",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "project_id": self.project_root.name,
            "sections": sections,
            "logs": self.logs,
        }
        output_path = self.reports_dir / "commercial_strategy.json"
        write_json(output_path, report)
        self.log(f"Saved Commercial Strategy report to {output_path.name}.")
        report["logs"] = self.logs
        write_json(output_path, report)
        write_json(
            self.progress_path,
            {
                "report_type": "commercial_drivers_strategy_to_win",
                "status": "complete",
                "project_id": self.project_root.name,
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "logs": self.logs,
            },
        )
        return report


def generate_commercial_strategy(project_root: Path) -> dict[str, Any]:
    return CommercialStrategyAgent(project_root).generate()


def load_commercial_strategy(project_root: Path) -> dict[str, Any]:
    return read_json(project_root / "reports" / "commercial_strategy.json", {})


def load_commercial_strategy_progress(project_root: Path) -> dict[str, Any]:
    return read_json(project_root / "reports" / "commercial_strategy.progress.json", {})


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate the Commercial Drivers and Strategy to WIN report.")
    parser.add_argument("project_root", type=Path, help="Path to the indexed project directory.")
    args = parser.parse_args()
    try:
        result = generate_commercial_strategy(args.project_root)
        print(json.dumps({"status": result.get("status", "complete"), "sections": len(result.get("sections", []))}, ensure_ascii=False))
    except Exception as exc:
        progress_path = args.project_root / "reports" / "commercial_strategy.progress.json"
        existing = read_json(progress_path, {})
        logs = existing.get("logs", []) if isinstance(existing, dict) else []
        logs.append(
            {
                "time": datetime.now(timezone.utc).isoformat(),
                "message": f"Agent failed: {exc}",
                "detail": {"error_type": type(exc).__name__},
            }
        )
        write_json(
            progress_path,
            {
                "report_type": "commercial_drivers_strategy_to_win",
                "status": "failed",
                "project_id": args.project_root.name,
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "logs": logs,
            },
        )
        raise
