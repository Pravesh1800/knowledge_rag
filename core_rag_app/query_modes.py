from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any


QUERY_MODES = {
    "exact_lookup",
    "multi_hop",
    "comparison",
    "global_summary",
    "contradiction_check",
    "gap_analysis",
    "risk_analysis",
    "follow_up",
    "general",
}


@dataclass(frozen=True)
class QueryMode:
    mode: str
    confidence: float
    reason: str
    needs_retrieval: bool
    graph_exploration: bool
    use_community_summaries: bool
    relationship_expansion: bool
    max_hits_multiplier: float
    max_agent_steps: int
    query_variants: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compact(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip())


def has_any(text: str, terms: tuple[str, ...]) -> bool:
    return any(term in text for term in terms)


def infer_query_mode(question: str, history: list[dict[str, str]] | None = None) -> QueryMode:
    q = compact(question)
    lower = q.lower()
    history = history or []
    if not q:
        return QueryMode("general", 0.5, "Empty question.", False, False, False, False, 1.0, 1, [])

    follow_up_terms = ("it", "that", "this", "they", "those", "above", "previous", "same")
    if history and len(lower.split()) <= 10 and has_any(lower, follow_up_terms):
        return QueryMode(
            "follow_up",
            0.72,
            "Short context-dependent follow-up.",
            True,
            True,
            True,
            True,
            1.2,
            4,
            [q],
        )

    contradiction_terms = (
        "contradict",
        "conflict",
        "inconsistent",
        "mismatch",
        "discrepancy",
        "disagree",
        "difference between",
        "does this conflict",
    )
    if has_any(lower, contradiction_terms):
        return QueryMode(
            "contradiction_check",
            0.86,
            "Question asks for conflicts, contradictions, or discrepancies.",
            True,
            True,
            True,
            True,
            1.8,
            5,
            [
                q,
                f"contradictions conflicts discrepancies {q}",
                f"inconsistent requirements exceptions {q}",
            ],
        )

    gap_terms = (
        "missing",
        "not mentioned",
        "not specified",
        "gap",
        "incomplete",
        "what is absent",
        "unanswered",
        "missing information",
    )
    if has_any(lower, gap_terms):
        return QueryMode(
            "gap_analysis",
            0.84,
            "Question asks for missing or incomplete evidence.",
            True,
            True,
            True,
            True,
            1.8,
            5,
            [
                q,
                f"specified requirements evidence {q}",
                f"missing unspecified gaps {q}",
            ],
        )

    comparison_terms = ("compare", "versus", " vs ", "difference", "differences", "similar", "better", "which one")
    if has_any(f" {lower} ", comparison_terms) or re.search(r"\bbetween\b.+\band\b", lower):
        return QueryMode(
            "comparison",
            0.82,
            "Question asks to compare two or more things.",
            True,
            True,
            True,
            True,
            1.7,
            5,
            [
                q,
                f"comparison similarities differences {q}",
                f"requirements metrics risks for each option {q}",
            ],
        )

    global_terms = (
        "summarize",
        "summary",
        "overview",
        "main themes",
        "key points",
        "overall",
        "across all",
        "all documents",
        "entire",
        "high level",
    )
    if has_any(lower, global_terms):
        return QueryMode(
            "global_summary",
            0.82,
            "Question asks for broad synthesis across domains or documents.",
            True,
            True,
            True,
            True,
            1.8,
            4,
            [
                q,
                f"overview key points requirements risks {q}",
                f"cross document themes {q}",
            ],
        )

    risk_terms = ("risk", "risks", "penalty", "liability", "delay", "failure", "issue", "concern", "exposure")
    if has_any(lower, risk_terms):
        return QueryMode(
            "risk_analysis",
            0.78,
            "Question asks for risk or exposure analysis.",
            True,
            True,
            True,
            True,
            1.6,
            5,
            [
                q,
                f"risks penalties liabilities exceptions {q}",
                f"mitigation obligations evidence {q}",
            ],
        )

    exact_terms = (
        "what is",
        "when",
        "where",
        "who",
        "how much",
        "how many",
        "date",
        "amount",
        "deadline",
        "page",
        "clause",
        "definition",
    )
    if lower.endswith("?") and has_any(lower, exact_terms):
        return QueryMode(
            "exact_lookup",
            0.74,
            "Question asks for a specific fact, value, date, definition, or citation.",
            True,
            True,
            False,
            False,
            1.0,
            3,
            [q],
        )

    multi_hop_terms = (
        "why",
        "how does",
        "impact",
        "depends",
        "dependency",
        "relationship",
        "link",
        "affect",
        "cause",
    )
    if has_any(lower, multi_hop_terms):
        return QueryMode(
            "multi_hop",
            0.72,
            "Question likely needs relationship traversal or multi-step evidence.",
            True,
            True,
            True,
            True,
            1.5,
            5,
            [
                q,
                f"relationships dependencies evidence {q}",
            ],
        )

    return QueryMode(
        "general",
        0.62,
        "Default evidence lookup mode.",
        True,
        True,
        True,
        True,
        1.2,
        4,
        [q],
    )


def mode_override(mode: str, question: str) -> QueryMode:
    base = infer_query_mode(question, [])
    presets = {
        "exact_lookup": (False, False, 1.0, 3),
        "global_summary": (True, True, 1.8, 4),
        "comparison": (True, True, 1.7, 5),
        "contradiction_check": (True, True, 1.8, 5),
        "gap_analysis": (True, True, 1.8, 5),
        "risk_analysis": (True, True, 1.6, 5),
        "multi_hop": (True, True, 1.5, 5),
        "follow_up": (True, True, 1.2, 4),
        "general": (True, True, 1.2, 4),
    }
    use_summaries, relationships, multiplier, steps = presets.get(mode, presets["general"])
    return QueryMode(
        mode=mode if mode in QUERY_MODES else "general",
        confidence=1.0,
        reason="Explicit query mode override.",
        needs_retrieval=True,
        graph_exploration=True,
        use_community_summaries=use_summaries,
        relationship_expansion=relationships,
        max_hits_multiplier=multiplier,
        max_agent_steps=steps,
        query_variants=base.query_variants or [compact(question)],
    )


def merge_query_variants(primary: str, planned: list[str], mode: QueryMode, limit: int = 8) -> list[str]:
    merged: list[str] = []
    for query in [primary, *mode.query_variants, *planned]:
        query = compact(str(query))
        if query and query.lower() not in {item.lower() for item in merged}:
            merged.append(query)
        if len(merged) >= limit:
            break
    return merged
