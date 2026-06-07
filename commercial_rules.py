from __future__ import annotations

import re
from typing import Any


COMMERCIAL_FLAGS: list[dict[str, Any]] = [
    {
        "id": "funding_source",
        "label": "Funding source",
        "search_terms": ["AMRUT", "funding", "funded", "grant", "government", "Govt", "DB funding", "budget"],
    },
    {
        "id": "long_term_om",
        "label": "Long-term O&M",
        "search_terms": ["O&M", "operation and maintenance", "10 years", "ten years", "comprehensive O&M"],
    },
    {
        "id": "reference_project",
        "label": "Reference project",
        "search_terms": ["reference", "composite", "intake", "piping", "WTP", "booster", "substation", "SCADA"],
    },
    {
        "id": "future_opportunity",
        "label": "Future opportunity",
        "search_terms": ["future opportunity", "pipeline", "24x7", "TTP", "distribution", "upcoming", "DPR"],
    },
    {
        "id": "client_relationship",
        "label": "Client relationship",
        "search_terms": ["IMC", "UADD", "consultant", "connect", "relationship", "client", "authority"],
    },
    {
        "id": "technical_differentiation",
        "label": "Technical differentiation",
        "search_terms": ["technical differentiation", "SUEZ technology", "technology", "optimize", "advanced", "process"],
    },
    {
        "id": "cost_strategy",
        "label": "Cost strategy",
        "search_terms": ["cost oriented", "local market", "competitive", "competitiveness", "cost", "price", "commercial"],
    },
    {
        "id": "optimization_lever",
        "label": "Optimization lever",
        "search_terms": ["chemical consumption", "power consumption", "optimize", "energy", "lifecycle", "OPEX"],
    },
    {
        "id": "competition",
        "label": "Competition",
        "search_terms": ["competition", "competitor", "bidder", "L&T", "SPML", "NCC", "VA Tech", "Tata"],
    },
]


COMMERCIAL_DRIVER_GUIDANCE: list[dict[str, Any]] = [
    {
        "id": "funding_confidence",
        "label": "Funding source and payment confidence",
        "definition": (
            "A driver is strong when the documents show confirmed funding, budget approval, government allocation, "
            "multilateral support, or another credible payment source that reduces collection/payment risk."
        ),
        "strong_evidence": [
            "Named funding program or government/multilateral allocation.",
            "Approved budget, sanctioned amount, or financing structure.",
            "Clear payment mechanism or source of funds.",
        ],
        "weak_evidence": [
            "Generic mention that the employer is a public authority.",
            "Unfunded project description with no payment source.",
        ],
        "generic_examples": {
            "strong": "Confirmed government or institutional funding improves payment confidence and reduces receivable risk.",
            "weak": "The project is public sector. This is too generic unless funding/payment source is evidenced.",
        },
    },
    {
        "id": "long_term_om",
        "label": "Long-term O&M opportunity",
        "definition": (
            "A driver is strong when the project includes multi-year operation and maintenance, recurring service "
            "revenue, performance-linked operation, or an opportunity to operate strategic infrastructure after construction."
        ),
        "strong_evidence": [
            "O&M duration, scope, and operating assets are stated.",
            "Recurring service obligations or performance responsibility are described.",
            "Operational scale is large enough to matter commercially.",
        ],
        "weak_evidence": [
            "One-time construction scope only.",
            "O&M mentioned without duration, scope, or value implication.",
        ],
        "generic_examples": {
            "strong": "A multi-year O&M scope creates recurring revenue and strengthens lifecycle accountability.",
            "weak": "O&M is mentioned. This is weak unless duration and scope are clear.",
        },
    },
    {
        "id": "reference_value",
        "label": "Reference value and strategic credentials",
        "definition": (
            "A driver is strong when the project can become a valuable reference because of size, complexity, geography, "
            "technology, client profile, integrated scope, or relevance to future bids."
        ),
        "strong_evidence": [
            "Large capacity, major city/region, or strategic client.",
            "Integrated scope across multiple work packages or technologies.",
            "Eligibility/reference relevance for future bids.",
        ],
        "weak_evidence": [
            "Routine small scope with no strategic differentiation.",
            "Reference value asserted without scale or complexity evidence.",
        ],
        "generic_examples": {
            "strong": "An integrated, high-capacity scheme can strengthen references for similar future tenders.",
            "weak": "The project can be a reference. This is weak unless the scope/scale explains why.",
        },
    },
    {
        "id": "scope_scale_complexity",
        "label": "Scope scale and technical complexity",
        "definition": (
            "A driver is strong when the documents show broad or complex scope that allows SUEZ to differentiate through "
            "engineering, process knowledge, execution planning, or integrated delivery."
        ),
        "strong_evidence": [
            "Multiple technical systems or workstreams are included.",
            "Complex interfaces, performance requirements, or technology choices are present.",
            "Scope breadth creates room for value engineering or integrated delivery advantage.",
        ],
        "weak_evidence": [
            "A list of assets without commercial implication.",
            "Generic technical scope that any bidder can execute similarly.",
        ],
        "generic_examples": {
            "strong": "Complex integrated scope gives room to differentiate through design integration and execution control.",
            "weak": "The tender includes many assets. This needs a reason why the assets create commercial advantage.",
        },
    },
    {
        "id": "client_stakeholder_access",
        "label": "Client, consultant, and stakeholder access",
        "definition": (
            "A driver is strong when documents or reliable context show existing access, working relationship, named "
            "stakeholders, consultant familiarity, or institutional proximity that can improve bid intelligence or execution."
        ),
        "strong_evidence": [
            "Named client/authority/consultant relationships or prior interaction.",
            "Evidence of access to decision-makers, local bodies, or implementation agencies.",
            "Clear stakeholder map relevant to bid positioning.",
        ],
        "weak_evidence": [
            "Client name is known but no relationship or access is evidenced.",
            "Assumed relationship without source support.",
        ],
        "generic_examples": {
            "strong": "Documented stakeholder access can improve bid alignment and reduce execution uncertainty.",
            "weak": "The authority is named. This is not enough to claim relationship or access.",
        },
    },
    {
        "id": "future_pipeline",
        "label": "Future pipeline and follow-on opportunities",
        "definition": (
            "A driver is strong when the project can unlock future tenders, extensions, adjacent packages, O&M expansion, "
            "distribution upgrades, reuse, digital, energy, or other follow-on opportunities."
        ),
        "strong_evidence": [
            "Future projects, DPRs, pipeline works, approved upcoming schemes, or planned expansions are mentioned.",
            "Current scope creates a gateway to adjacent packages or repeat client work.",
            "Public context supports credible follow-on opportunities.",
        ],
        "weak_evidence": [
            "Generic hope for future business.",
            "Uncited market speculation.",
        ],
        "generic_examples": {
            "strong": "A documented future program can make the bid strategically valuable beyond the initial contract.",
            "weak": "This may lead to future work. This is weak unless a future pipeline is evidenced.",
        },
    },
]


STRATEGY_TO_WIN_GUIDANCE: list[dict[str, Any]] = [
    {
        "id": "technical_differentiation",
        "label": "Technical differentiation",
        "definition": (
            "A win theme is strong when it links SUEZ capability to a specific tender need, performance requirement, "
            "technology risk, water quality requirement, operating constraint, or interface complexity."
        ),
        "strong_evidence": [
            "Tender requirement where better technology/process knowledge matters.",
            "A performance or reliability requirement SUEZ can credibly address.",
            "A technical pain point that can become a differentiator.",
        ],
        "weak_evidence": [
            "Generic statement to be technically strong.",
            "Capability claim without tender-specific relevance.",
        ],
        "generic_examples": {
            "strong": "Position proven process control against specific performance and reliability requirements.",
            "weak": "Offer superior technology. This is too generic without a linked tender need.",
        },
    },
    {
        "id": "cost_competitiveness",
        "label": "Cost competitiveness and price strategy",
        "definition": (
            "A win theme is strong when it identifies how to be competitive on capex, opex, lifecycle cost, local sourcing, "
            "risk pricing, packaging, or value engineering without weakening compliance."
        ),
        "strong_evidence": [
            "Tender quantities, payment terms, O&M obligations, or cost drivers that affect bid price.",
            "Local market or sourcing angle with credible basis.",
            "Value engineering opportunity tied to scope or performance.",
        ],
        "weak_evidence": [
            "Simply saying be low cost.",
            "Pricing advice with no link to tender cost drivers.",
        ],
        "generic_examples": {
            "strong": "Use lifecycle-cost optimization to offset higher technical quality while staying price competitive.",
            "weak": "Quote aggressively. This is unsafe unless cost drivers and risks are understood.",
        },
    },
    {
        "id": "opex_optimization",
        "label": "Chemical, energy, and lifecycle optimization",
        "definition": (
            "A win theme is strong when it identifies operating-cost levers such as chemical consumption, power, manpower, "
            "maintenance, automation, asset life, water losses, or reliability that can improve lifecycle economics."
        ),
        "strong_evidence": [
            "O&M scope or performance requirements make lifecycle cost important.",
            "Chemical, energy, pumping, treatment, automation, or maintenance levers are visible.",
            "Optimization can improve both price and long-term delivery confidence.",
        ],
        "weak_evidence": [
            "Optimization mentioned without a specific lever.",
            "One-time capex saving that increases O&M risk.",
        ],
        "generic_examples": {
            "strong": "Target energy and chemical optimization where O&M scope rewards lower lifecycle cost.",
            "weak": "Optimize operations. This needs a specific lever and commercial effect.",
        },
    },
    {
        "id": "execution_localization",
        "label": "Execution, local delivery, and partner strategy",
        "definition": (
            "A win theme is strong when local delivery, subcontracting, logistics, approvals, site sequencing, or resource "
            "planning can materially improve cost, speed, compliance, or client confidence."
        ),
        "strong_evidence": [
            "Local approvals, site constraints, logistics, or construction sequencing matter.",
            "Scope can benefit from local contractors, suppliers, or execution footprint.",
            "Delivery model can reduce risk or improve competitiveness.",
        ],
        "weak_evidence": [
            "Generic local partner statement.",
            "Assumed localization without scope or market basis.",
        ],
        "generic_examples": {
            "strong": "Use local execution partners for civil/package delivery while retaining process and quality control.",
            "weak": "Use local partners. This needs a reason tied to scope, risk, or cost.",
        },
    },
    {
        "id": "risk_allocation",
        "label": "Risk allocation and qualification strategy",
        "definition": (
            "A win theme is strong when it identifies commercial/legal/technical risks that should be priced, clarified, "
            "qualified, transferred, or actively managed to protect margin and bid viability."
        ),
        "strong_evidence": [
            "Tender terms create liability, payment, design, interface, inflation, FX, land, approval, or performance risk.",
            "Clarification or qualification can materially improve bid risk profile.",
            "Risk pricing approach is tied to actual tender conditions.",
        ],
        "weak_evidence": [
            "Generic risk management statement.",
            "Ignoring project-killer risks to make the bid look attractive.",
        ],
        "generic_examples": {
            "strong": "Price and clarify high-impact risk clauses before final commercial submission.",
            "weak": "Manage risks carefully. This is not a strategy unless the risk and action are named.",
        },
    },
]


COMMERCIAL_SCORING_CRITERIA: list[dict[str, str]] = [
    {
        "id": "commercial_value",
        "question": "Does this point improve revenue, margin, payment confidence, recurring income, or cost position?",
    },
    {
        "id": "strategic_value",
        "question": "Does this point strengthen references, market position, client access, future pipeline, or differentiation?",
    },
    {
        "id": "evidence_strength",
        "question": "Is the point directly supported by project documents or reliable public context?",
    },
    {
        "id": "win_relevance",
        "question": "Does this point help SUEZ decide how to position, price, qualify, partner, or execute the bid?",
    },
    {
        "id": "specificity",
        "question": "Is the point specific enough to be useful on a bid-strategy slide, rather than generic consulting language?",
    },
    {
        "id": "risk_caveat",
        "question": "Does the point include any caveat or risk that could weaken the commercial conclusion?",
    },
]


COMMERCIAL_SECTIONS: list[dict[str, Any]] = [
    {
        "id": "commercial_drivers",
        "title": "SUEZ COMMERCIAL DRIVERS",
        "goal": "Identify why this project is commercially attractive or strategically important for SUEZ.",
        "starter_angles": [
            "funding source and payment confidence",
            "long-term O&M opportunity",
            "project scale and reference value",
            "client, authority, and consultant context",
            "future opportunities unlocked by the project",
            "competitor/award/public-market context when available",
        ],
        "selection_rule": "Use these as discovery prompts only. Generate the strongest project-specific commercial drivers, even if they differ from these angles.",
    },
    {
        "id": "strategy_to_win",
        "title": "STRATEGY TO WIN",
        "goal": "Recommend how SUEZ should position and price the bid to win.",
        "starter_angles": [
            "technical differentiation",
            "cost competitiveness with local market",
            "chemical, power, and lifecycle optimization",
            "local execution/subcontracting strategy where supported",
            "competitive positioning against market context",
        ],
        "selection_rule": "Use these as discovery prompts only. Generate the strongest project-specific win themes, even if they differ from these angles.",
    },
]


def _contains_term(text: str, term: str) -> bool:
    if len(term) <= 5 and term.upper() == term:
        return re.search(rf"\b{re.escape(term)}\b", text) is not None
    return term.lower() in text.lower()


def detect_commercial_flags(topic: dict[str, Any]) -> tuple[list[str], dict[str, str], str]:
    text = " ".join(
        str(topic.get(key, ""))
        for key in ["topic_name", "topic_description", "content", "document_name"]
    )
    flags: list[str] = []
    reasons: dict[str, str] = {}
    strong = 0
    for flag in COMMERCIAL_FLAGS:
        matched = [term for term in flag["search_terms"] if _contains_term(text, term)]
        if matched:
            flags.append(flag["id"])
            reasons[flag["id"]] = "Matched commercial terms: " + ", ".join(matched[:5])
            if len(matched) >= 2:
                strong += 1
    if strong:
        confidence = "high"
    elif flags:
        confidence = "medium"
    else:
        confidence = "low"
    return flags, reasons, confidence
