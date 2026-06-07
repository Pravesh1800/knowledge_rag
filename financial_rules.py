from __future__ import annotations

import re
from typing import Any


FINANCIAL_FLAGS: list[dict[str, Any]] = [
    {
        "id": "bid_security",
        "label": "Bid Security / Bid Bond / EMD",
        "search_terms": ["bid security", "bid bond", "EMD", "earnest money", "cash only", "tender security"],
    },
    {
        "id": "advance_payment",
        "label": "DB - Advance Payment",
        "search_terms": ["advance payment", "mobilization advance", "10%", "bank guarantee", "recovery", "recover"],
    },
    {
        "id": "db_performance_bond",
        "label": "DB - Performance Bond",
        "search_terms": ["DB performance bond", "performance security", "performance bond", "3%", "DB CV", "DBCV"],
    },
    {
        "id": "om_performance_bond",
        "label": "O&M - Performance Bond",
        "search_terms": ["O&M performance bond", "O&M performance security", "10% of O&M", "O&M CV", "OMCV"],
    },
    {
        "id": "db_retention_money",
        "label": "DB Retention Money",
        "search_terms": ["retention money", "cash retention", "7%", "running bill", "DLP", "defect liability"],
    },
    {
        "id": "om_retention_security",
        "label": "O&M Retention Money / Security Deposit",
        "search_terms": ["O&M retention", "security deposit", "O&M security", "retention/security", "none"],
    },
    {
        "id": "parent_company_guarantee",
        "label": "Parent Company Guarantee",
        "search_terms": ["parent company guarantee", "parent guarantee", "corporate guarantee", "not required"],
    },
    {
        "id": "validity",
        "label": "Validity",
        "search_terms": ["validity", "valid till", "valid up to", "completion certificate", "3 months"],
    },
    {
        "id": "recovery",
        "label": "Recovery",
        "search_terms": ["recovery", "recover", "running account", "running bill", "RA bill", "80% DB billing"],
    },
]


FINANCIAL_ROWS: list[dict[str, Any]] = [
    {
        "id": "bid_security",
        "s_no": 1,
        "topic": "Bid Security / Bid Bond (EMD - Earnest Money Deposit)",
        "description": "Security submitted with the bid to keep the bidder bound during tender validity. It is pre-award security and is different from post-award performance security.",
        "extract_fields": ["amount", "instrument", "cash_or_bg", "validity", "refund_or_forfeiture_condition"],
        "not_enough": "Do not use performance security, performance bond, retention money, or advance payment BG as bid security.",
        "examples": {
            "strong": "Bid security/EMD is stated as a fixed INR amount, with whether it is cash, BG, online payment, or another instrument.",
            "weak": "A performance security percentage is mentioned. That does not answer bid security.",
            "edge": "If both EMD and performance security appear nearby, use only the pre-bid EMD/bid security wording for this row.",
        },
        "goal": "Find bid security, bid bond, or EMD amount and whether it is cash-only or can be BG.",
        "flag_ids": ["bid_security"],
    },
    {
        "id": "advance_payment",
        "s_no": 2,
        "topic": "DB- Advance Payment",
        "description": "Advance paid to the contractor for design-build/mobilization before equivalent work is completed. Usually secured by a bank guarantee and recovered from running bills.",
        "extract_fields": ["percentage", "amount", "basis", "instrument", "recovery", "conditions", "validity"],
        "not_enough": "Do not confuse advance payment BG with performance bond/security. A BG securing advance payment is not a performance bond.",
        "examples": {
            "strong": "Advance payment is stated as X% of contract value against BG, with recovery from running bills or before a stated billing threshold.",
            "weak": "A performance security percentage is mentioned but no advance payment terms are stated.",
            "edge": "If advance payment exists but no recovery clause is found, state the advance and say recovery not found.",
        },
        "goal": "Find DB advance payment percentage, basis, bank guarantee condition, and recovery condition.",
        "flag_ids": ["advance_payment", "recovery"],
    },
    {
        "id": "db_performance_bond",
        "s_no": 3,
        "topic": "DB - Performance Bond",
        "description": "Post-award security for design-build obligations. It may be called performance security, performance guarantee, performance bond, or contract performance security.",
        "extract_fields": ["percentage", "amount", "basis", "instrument", "validity", "submission_timing", "release_condition"],
        "not_enough": "Do not use bid security, advance payment BG, retention money, or O&M performance bond as DB performance bond.",
        "examples": {
            "strong": "DB performance security is X% of DB contract value, submitted as BG, valid until completion certificate plus a stated period.",
            "weak": "Bid security/EMD amount is stated. That is pre-award and not DB performance bond.",
            "edge": "If one performance security covers both DB and O&M, clearly state the combined basis and do not split unless the tender splits it.",
        },
        "goal": "Find DB performance bond/security percentage, basis, and validity period.",
        "flag_ids": ["db_performance_bond", "validity"],
    },
    {
        "id": "om_performance_bond",
        "s_no": 4,
        "topic": "O&M - Performance Bond",
        "description": "Security for operation and maintenance period obligations. It is separate from design-build performance security unless the tender expressly combines them.",
        "extract_fields": ["percentage", "amount", "basis", "instrument", "validity", "submission_timing", "release_condition"],
        "not_enough": "Do not use DB performance security unless the clause expressly applies to O&M or states an O&M value/basis.",
        "examples": {
            "strong": "O&M performance security is X% of O&M contract value, valid until O&M completion certificate plus a stated period.",
            "weak": "DB performance security is mentioned but no O&M performance security is stated.",
            "edge": "If O&M security is stated as part of a combined performance guarantee, cite the exact combined wording.",
        },
        "goal": "Find O&M performance bond/security percentage, basis, and validity period.",
        "flag_ids": ["om_performance_bond", "validity"],
    },
    {
        "id": "db_retention_money",
        "s_no": 5,
        "topic": "DB Retention Money",
        "description": "Money retained from design-build running bills as security for performance, defects, or completion obligations. It is usually deducted over time and released/replaced later.",
        "extract_fields": ["percentage", "basis", "deduction_method", "recovery", "release_condition", "replacement_security", "dlp_linkage"],
        "not_enough": "Do not use performance security or bid security as retention money. Retention must involve deduction/withholding from bills or cash retention.",
        "examples": {
            "strong": "Retention is X% from each running bill up to Y%, with release or replacement after completion/DLP milestones.",
            "weak": "Performance security is X% of contract value. This is not retention unless deductions from bills are stated.",
            "edge": "If security deposit replaces retention after a milestone, include both retention and replacement conditions.",
        },
        "goal": "Find DB retention percentage, deduction method, release/replacement conditions, and DLP linkage.",
        "flag_ids": ["db_retention_money", "recovery", "validity"],
    },
    {
        "id": "om_retention_security",
        "s_no": 6,
        "topic": "O&M Retention Money/Security deposit",
        "description": "Retention or security deposit specifically for the O&M phase. It may be a deduction from O&M bills, a separate security deposit, or expressly not required.",
        "extract_fields": ["required_status", "percentage", "amount", "basis", "deduction_method", "validity", "release_condition"],
        "not_enough": "Do not infer O&M retention from DB retention unless the clause expressly applies to O&M.",
        "examples": {
            "strong": "O&M retention/security is expressly stated, or the tender expressly says none/not required.",
            "weak": "DB retention exists but no O&M retention/security clause is found.",
            "edge": "If no O&M retention is found after search, use not_found; use not_required only when the tender says none/not required.",
        },
        "goal": "Find whether O&M retention money or O&M security deposit is required, or confirm none.",
        "flag_ids": ["om_retention_security"],
    },
    {
        "id": "parent_company_guarantee",
        "s_no": 7,
        "topic": "Parent Company Guarantee",
        "description": "Guarantee issued by a parent/holding/group company to support the bidder's obligations. It is not the same as a bank guarantee or performance security.",
        "extract_fields": ["required_status", "guarantor_type", "scope", "amount_or_cap", "validity", "conditions"],
        "not_enough": "Do not treat bank guarantees, performance guarantees, bid bonds, or advance payment guarantees as parent company guarantee.",
        "examples": {
            "strong": "Tender expressly requires or expressly waives parent/corporate/company guarantee.",
            "weak": "Bank guarantee is required. That is not a parent company guarantee.",
            "edge": "If consortium/JV guarantees are mentioned, inspect whether they are parent-company support or merely member liability.",
        },
        "goal": "Find whether a parent company guarantee/corporate guarantee is required, or confirm not required.",
        "flag_ids": ["parent_company_guarantee"],
    },
]


FINANCIAL_EXTRACTION_SCHEMA: dict[str, Any] = {
    "required_status": "required | not_required | not_found | unclear",
    "amount": "Exact currency amount if stated, otherwise blank.",
    "percentage": "Exact percentage if stated, otherwise blank.",
    "basis": "The base value for percentage or amount, e.g. contract value, DB CV, O&M CV, bid value.",
    "instrument": "cash | bank guarantee | demand draft | online payment | security deposit | retention | other | not stated",
    "cash_or_bg": "Whether cash only, BG allowed, BG required, or not stated.",
    "validity": "Validity period or expiry trigger, including any plus-months period.",
    "recovery": "Recovery/deduction method and timing if applicable.",
    "release_condition": "Release, replacement, completion, certificate, or DLP condition if applicable.",
    "conditions": "Submission timing, prerequisites, forfeiture, refund, or other special conditions.",
    "exact_clause_excerpt": "Shortest exact evidence excerpt supporting the extracted values.",
    "not_found_basis": "What was searched and why value is absent if not_found/unclear.",
}


def _contains_term(text: str, term: str) -> bool:
    if len(term) <= 5 and term.upper() == term:
        return re.search(rf"\b{re.escape(term)}\b", text) is not None
    return term.lower() in text.lower()


def detect_financial_flags(topic: dict[str, Any]) -> tuple[list[str], dict[str, str], str]:
    text = " ".join(
        str(topic.get(key, ""))
        for key in ["topic_name", "topic_description", "content", "document_name"]
    )
    flags: list[str] = []
    reasons: dict[str, str] = {}
    strong = 0
    for flag in FINANCIAL_FLAGS:
        matched = [term for term in flag["search_terms"] if _contains_term(text, term)]
        if matched:
            flags.append(flag["id"])
            reasons[flag["id"]] = "Matched financial terms: " + ", ".join(matched[:5])
            if len(matched) >= 2:
                strong += 1
    if strong:
        confidence = "high"
    elif flags:
        confidence = "medium"
    else:
        confidence = "low"
    return flags, reasons, confidence
