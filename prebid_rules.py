from __future__ import annotations

import re
from typing import Any


PREBID_FLAGS: list[dict[str, Any]] = [
    {
        "id": "qualification_eligibility",
        "label": "Qualification, eligibility, JV, parent-company experience",
        "search_terms": [
            "pre-qualification",
            "qualification criteria",
            "similar work",
            "parent company",
            "subsidiary",
            "joint venture",
            "JV",
            "financial capacity",
            "technical capacity",
        ],
    },
    {
        "id": "scope_gap_boq",
        "label": "Scope gaps, BOQ omissions, quantity responsibility",
        "search_terms": [
            "scope of works",
            "BOQ",
            "bill of quantities",
            "include the same in the BOQ",
            "quantity",
            "not included",
            "please clarify",
            "please confirm",
        ],
    },
    {
        "id": "drawings_surveys_data",
        "label": "Drawings, DPR, surveys, GAD, AutoCAD, geotech, topography",
        "search_terms": [
            "drawing",
            "drawings",
            "DPR",
            "AutoCAD",
            "GAD",
            "survey",
            "topographical",
            "geotechnical",
            "soil investigation",
            "L Survey",
        ],
    },
    {
        "id": "site_constraints_access",
        "label": "Site access, land, approach road, constraints",
        "search_terms": [
            "site visit",
            "approach road",
            "land",
            "right of way",
            "ROW",
            "barricading",
            "sheet piling",
            "existing structure",
            "alternate site",
        ],
    },
    {
        "id": "technical_conflict_design",
        "label": "Technical design/specification conflict",
        "search_terms": [
            "specification",
            "type of",
            "not coherent",
            "typo error",
            "design basis",
            "process",
            "MOC",
            "manual",
            "mechanical",
            "standby",
        ],
    },
    {
        "id": "existing_assets_om",
        "label": "Existing assets, O&M scope, major maintenance",
        "search_terms": [
            "O&M",
            "operation and maintenance",
            "major maintenance",
            "major repair",
            "routine maintenance",
            "existing facilities",
            "existing WTP",
            "old pumping main",
            "replacement",
        ],
    },
    {
        "id": "schedule_milestones_completion",
        "label": "Completion period, milestones, rainy season, handover, DLP",
        "search_terms": [
            "completion period",
            "30 months",
            "36 months",
            "rainy season",
            "milestone",
            "handover",
            "DLP",
            "defect liability",
            "trial run",
        ],
    },
    {
        "id": "payment_finance_security",
        "label": "Payment terms, advance, retention, securities, cash flow",
        "search_terms": [
            "payment",
            "advance",
            "retention",
            "performance bond",
            "bank guarantee",
            "bid security",
            "EMD",
            "running bill",
            "recovery",
        ],
    },
    {
        "id": "insurance_taxes_permits",
        "label": "Insurance, taxes, permits, approvals",
        "search_terms": [
            "insurance",
            "taxes",
            "GST",
            "permits",
            "permission",
            "approval",
            "statutory",
            "clearance",
            "license",
        ],
    },
    {
        "id": "legal_risk_allocation",
        "label": "Legal risk allocation, liability, LD, indemnity, arbitration",
        "search_terms": [
            "liability",
            "liquidated damages",
            "LD",
            "indemnity",
            "arbitration",
            "force majeure",
            "consequential",
            "damages",
            "risk",
        ],
    },
    {
        "id": "price_escalation_currency",
        "label": "Price escalation, currency, forex, indices",
        "search_terms": [
            "price adjustment",
            "escalation",
            "currency",
            "Indian Rupees",
            "INR",
            "foreign exchange",
            "indices",
            "formula",
        ],
    },
    {
        "id": "contradiction_silence",
        "label": "Contradictory, silent, unclear, missing or ambiguous tender provisions",
        "search_terms": [
            "at variance",
            "contradict",
            "silent",
            "not defined",
            "not specified",
            "unclear",
            "ambiguity",
            "missing",
            "please review",
            "kindly review",
        ],
    },
]


MANDATORY_AUDIT_CATEGORIES: list[dict[str, Any]] = [
    {
        "id": "qualification_eligibility",
        "title": "Qualification / Eligibility / JV / Parent Company",
        "goal": "Find qualification or eligibility clauses that may need relaxation, confirmation, parent-company reliance, JV/subsidiary clarification, or O&M experience clarification.",
        "flag_ids": ["qualification_eligibility", "contradiction_silence"],
    },
    {
        "id": "scope_boq_missing_items",
        "title": "Scope of Works / BOQ Missing Items",
        "goal": "Find scope items, equipment, quantities, interfaces, temporary works, or BOQ omissions that require clarification or inclusion.",
        "flag_ids": ["scope_gap_boq", "technical_conflict_design", "contradiction_silence"],
    },
    {
        "id": "drawings_surveys_and_design_data",
        "title": "Drawings / DPR / Surveys / Design Data",
        "goal": "Find missing drawings, DPR, GAD, AutoCAD, survey, geotechnical, topographical, route, level, or design basis data that bidders need before pricing.",
        "flag_ids": ["drawings_surveys_data", "site_constraints_access", "contradiction_silence"],
    },
    {
        "id": "site_access_constraints",
        "title": "Site Access / Land / Existing Constraints",
        "goal": "Find site constraints, land availability, approach access, existing structures, ROW, constructability, or alternate-site issues needing client confirmation.",
        "flag_ids": ["site_constraints_access", "drawings_surveys_data"],
    },
    {
        "id": "technical_specification_conflicts",
        "title": "Technical Specification Conflicts",
        "goal": "Find conflicting, incomplete, or impossible technical specifications across civil, mechanical, electrical, SCADA, WTP, pumping, pipeline, valves, instrumentation, and surge systems.",
        "flag_ids": ["technical_conflict_design", "scope_gap_boq", "contradiction_silence"],
    },
    {
        "id": "existing_assets_om_major_maintenance",
        "title": "Existing Assets / O&M / Major Maintenance",
        "goal": "Find unclear O&M obligations, existing asset condition risks, major repair thresholds, replacement limits, routine-vs-major maintenance gaps, and handover condition issues.",
        "flag_ids": ["existing_assets_om", "scope_gap_boq", "payment_finance_security"],
    },
    {
        "id": "schedule_milestones_completion",
        "title": "Schedule / Milestones / Completion Period",
        "goal": "Find unrealistic timelines, unclear milestones, rainy-season assumptions, commissioning/trial-run ambiguity, DLP/handover ambiguity, and extension needs.",
        "flag_ids": ["schedule_milestones_completion", "contradiction_silence"],
    },
    {
        "id": "payment_security_cashflow",
        "title": "Payment / Securities / Cash Flow",
        "goal": "Find payment, advance, recovery, retention, BG, bid security, performance security, billing, and cash-flow points that require clarification or relaxation.",
        "flag_ids": ["payment_finance_security", "price_escalation_currency", "contradiction_silence"],
    },
    {
        "id": "insurance_taxes_approvals",
        "title": "Insurance / Taxes / Permits / Approvals",
        "goal": "Find undefined insurance requirements, tax treatment, statutory approvals, permissions, clearances, and responsibility splits.",
        "flag_ids": ["insurance_taxes_permits", "legal_risk_allocation", "contradiction_silence"],
    },
    {
        "id": "legal_and_risk_allocation",
        "title": "Legal / Risk Allocation / Liability",
        "goal": "Find legal or contractual risks requiring pre-bid query: liability cap, LD, indemnity, consequential damages, arbitration, force majeure, risk transfer, and silent clauses.",
        "flag_ids": ["legal_risk_allocation", "contradiction_silence"],
    },
    {
        "id": "price_escalation_currency",
        "title": "Price Escalation / Currency / Indices",
        "goal": "Find price adjustment, O&M escalation, currency, forex, index formula, base date, and reimbursement gaps that require clarification.",
        "flag_ids": ["price_escalation_currency", "payment_finance_security", "contradiction_silence"],
    },
]


PBQ_MANDATORY_QUERY_TEST: list[str] = [
    "There is a real tender/document gap, conflict, silence, ambiguity, missing document/data, unclear responsibility, or impractical condition.",
    "The issue materially affects pricing, schedule, design, construction method, compliance, O&M obligation, risk allocation, or bid qualification.",
    "For manual PBQ coverage, standard practical clarifications are valid when they affect drawings, quantities, BOQ inclusion, site access, utilities, approvals, O&M handover, payment mechanics, or industry-norm relaxation even if the priority is only Medium.",
    "The employer can take a concrete action: provide, confirm, clarify, revise, relax, include in BOQ, issue drawing/data, define responsibility, or amend the clause.",
    "The query is specific enough to send and is not a generic request for clarification.",
    "The query is supported by project/tender evidence, or it clearly identifies absence after searching.",
]


PBQ_PRIORITY_SCORING: list[dict[str, Any]] = [
    {
        "priority": "Critical",
        "meaning": "Could materially affect bid eligibility, major price, risk acceptance, design basis, completion feasibility, or go/no-go decision.",
    },
    {
        "priority": "High",
        "meaning": "Important to price, schedule, scope, risk allocation, or technical compliance, but not necessarily bid-killing.",
    },
    {
        "priority": "Medium",
        "meaning": "Useful clarification that improves accuracy or reduces uncertainty but is not expected to change bid fundamentals.",
    },
    {
        "priority": "Low",
        "meaning": "Minor wording/data cleanup. Include only if clearly useful and not crowding out stronger queries.",
    },
]


PBQ_GENERIC_EXAMPLES: dict[str, list[str]] = {
    "good": [
        "Please provide the missing technical data/report/drawing required for design and pricing of the relevant works.",
        "We understand that the stated requirement applies only to the specified scope. Kindly review and confirm.",
        "The clauses appear to be inconsistent regarding responsibility/scope. Please clarify the applicable requirement for bidding.",
        "Please confirm whether the item is included in the bidder's scope and, if yes, provide the corresponding BOQ item/payment mechanism.",
    ],
    "bad": [
        "Please clarify the design.",
        "Please confirm all tender conditions.",
        "Please provide all missing information.",
        "Please reduce risk to bidder.",
    ],
    "edge": [
        "If the tender already answers the point clearly, do not raise a query.",
        "If the issue is only a bidder preference with no tender ambiguity, do not raise a query.",
        "If one query covers the same issue across multiple pages, consolidate instead of repeating.",
    ],
}


PBQ_DUPLICATE_CONTROL_RULES: list[str] = [
    "Remove duplicate rows that ask the same employer action for the same underlying issue.",
    "Merge similar queries across categories only when one consolidated query is clearer and no distinct employer action is lost.",
    "Do not merge rows merely because they share the same broad topic. Keep separate rows when the requested employer action, clause basis, pricing impact, or practical information need is different.",
    "Do not repeat a query only because the same topic appears in multiple documents or pages.",
    "Remove vague rows that do not identify a concrete missing item, conflict, risk, responsibility, or action requested.",
    "Remove rows already answered clearly by the tender evidence.",
]


PBQ_SEND_READY_RULES: list[str] = [
    "Identify clause, page, drawing, schedule, volume, or BOQ reference when available.",
    "State the bidder's understanding or the exact issue in one sentence.",
    "Ask one clear employer action: provide, confirm, clarify, revise, relax, include, define, or amend.",
    "Explain why the information is needed for pricing, design, schedule, compliance, risk, or execution.",
    "Use professional bidder wording and avoid argumentative language.",
    "Keep the query concise enough to paste into a pre-bid query register.",
]


PBQ_STRUCTURED_FIELDS: dict[str, str] = {
    "priority": "Critical | High | Medium | Low",
    "impact_area": "pricing | schedule | design | construction | compliance | O&M | risk | eligibility | payment | approvals | other",
    "action_requested": "provide | confirm | clarify | revise | relax | include in BOQ | define responsibility | amend clause | other",
    "evidence_strength": "direct | inferred | absence_after_search | weak",
    "tender_reference": "Volume/section/page/clause/drawing/BOQ reference if available, else '-'.",
    "issue_summary": "One-line description of the gap/conflict/ambiguity.",
    "bidder_query": "Send-ready query text.",
    "basis": "Why the query is needed for pricing/risk/schedule/design/compliance.",
    "duplicate_group": "Short stable label for similar queries; use '-' if unique.",
}


def _contains_term(text: str, term: str) -> bool:
    if len(term) <= 5 and term.upper() == term:
        return re.search(rf"\b{re.escape(term)}\b", text) is not None
    return term.lower() in text.lower()


def detect_prebid_flags(topic: dict[str, Any]) -> tuple[list[str], dict[str, str], str]:
    text = " ".join(
        str(topic.get(key, ""))
        for key in ["topic_name", "topic_description", "content", "document_name"]
    )
    flags: list[str] = []
    reasons: dict[str, str] = {}
    strong = 0
    for flag in PREBID_FLAGS:
        matched = [term for term in flag["search_terms"] if _contains_term(text, term)]
        if matched:
            flags.append(flag["id"])
            reasons[flag["id"]] = "Matched pre-bid query risk terms: " + ", ".join(matched[:6])
            if len(matched) >= 2:
                strong += 1
    if strong:
        confidence = "high"
    elif flags:
        confidence = "medium"
    else:
        confidence = "low"
    return flags, reasons, confidence
