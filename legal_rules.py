from __future__ import annotations

import re
from typing import Any


LEGAL_ROWS: list[dict[str, Any]] = [
    {
        "id": "consequential_damages_exclusion",
        "s_no": 1,
        "topic": "Exclusion of financial loss or consequential damages",
        "description": (
            "Check whether the tender protects SUEZ/contractor from remote, indirect, special, consequential, "
            "or business-loss claims. This includes loss of profit, loss of revenue, loss of production, loss of "
            "business opportunity, financing costs, or other financial losses that go beyond direct proven damages."
        ),
        "what_to_find": (
            "Find wording that expressly excludes or disclaims liability for financial loss, indirect loss, "
            "consequential damages, special damages, loss of profit, or similar categories."
        ),
        "not_enough": (
            "A general damages clause, LD clause, penalty clause, indemnity clause, or silence on consequential "
            "damages is not enough. If the tender does not expressly exclude these losses, answer No."
        ),
        "examples": {
            "yes": [
                "Neither party shall be liable for indirect, consequential, special, or loss of profit damages.",
                "The contractor shall not be liable for financial loss, loss of revenue, or consequential damages.",
            ],
            "no": [
                "The contractor shall pay damages for delay at the rates stated in the contract.",
                "The tender is silent about indirect, financial, or consequential loss.",
            ],
            "edge": [
                "An indemnity clause may mention losses, but it is No unless it expressly excludes consequential or financial loss.",
            ],
        },
        "decision_rule": "Yes only if the tender excludes financial loss, consequential damages, indirect damages, loss of profit, or similar remote damages.",
        "search_terms": [
            "financial loss",
            "consequential damage",
            "consequential damages",
            "indirect loss",
            "indirect damages",
            "loss of profit",
            "special damages",
            "remote damages",
            "exclusion of liability",
        ],
    },
    {
        "id": "liability_cap",
        "s_no": 2,
        "topic": "Limited liability with a cap not exceeding the contractual value",
        "description": (
            "Check whether SUEZ/contractor's total exposure under the contract is capped. The desired protection "
            "is an aggregate liability ceiling that does not exceed the contract value, contract price, or contractual value."
        ),
        "what_to_find": (
            "Find wording such as total liability, aggregate liability, maximum liability, contractor's liability "
            "shall not exceed, liability cap, capped at contract value, or capped at contract price."
        ),
        "not_enough": (
            "Caps only on liquidated damages, performance security, retention, insurance, or a single claim category "
            "are not enough unless the clause clearly caps total aggregate liability. Silence is No."
        ),
        "examples": {
            "yes": [
                "The contractor's aggregate liability under the contract shall not exceed the contract price.",
                "Total liability of the contractor shall be capped at 100% of the contract value.",
            ],
            "no": [
                "Liquidated damages shall not exceed 10% of the contract value.",
                "Performance security shall be 10% of the contract price.",
            ],
            "edge": [
                "A cap on LD or security is not a total liability cap unless the clause says aggregate/all liability is capped.",
            ],
        },
        "decision_rule": "Yes only if total or aggregate liability is capped at or below the contract value, contract price, or contractual value.",
        "search_terms": [
            "limited liability",
            "limitation of liability",
            "total liability",
            "aggregate liability",
            "liability cap",
            "contractual value",
            "contract value",
            "contract price",
            "shall not exceed",
            "maximum liability",
        ],
    },
    {
        "id": "arbitration",
        "s_no": 3,
        "topic": "Arbitration clause",
        "description": (
            "Check whether disputes are referred to arbitration, an arbitral tribunal, statutory arbitration board, "
            "or equivalent formal dispute resolution forum instead of only ordinary departmental decision-making."
        ),
        "what_to_find": (
            "Find an arbitration clause, arbitrator appointment process, arbitral tribunal reference, dispute "
            "resolution tribunal, Madhyastham Adhikaran, conciliation/arbitration sequence, or similar binding forum."
        ),
        "not_enough": (
            "A clause only saying the Engineer, employer, authority, or department will decide disputes is not enough "
            "unless it also provides arbitration or a tribunal/forum equivalent."
        ),
        "examples": {
            "yes": [
                "Disputes shall be referred to arbitration by a sole arbitrator appointed under the Arbitration Act.",
                "Disputes shall be referred to the Madhya Pradesh Arbitration Tribunal/Madhyastham Adhikaran.",
            ],
            "no": [
                "The Engineer-in-Charge's decision shall be final.",
                "Disputes shall be decided by the department without arbitration or tribunal reference.",
            ],
            "edge": [
                "A multi-step dispute process is Yes only if one step includes arbitration or an equivalent tribunal.",
            ],
        },
        "decision_rule": "Yes if the tender contains an arbitration or dispute resolution tribunal clause.",
        "search_terms": [
            "arbitration",
            "arbitral",
            "arbitrator",
            "dispute resolution",
            "tribunal",
            "madhyastham",
            "adhikaran",
            "conciliation",
        ],
    },
    {
        "id": "force_majeure",
        "s_no": 4,
        "topic": "Force Majeure Clause including war, terrorism, rebellion, strike",
        "description": (
            "Check whether the contract recognizes force majeure or exceptional events beyond the parties' control, "
            "especially serious political/social events such as war, terrorism, rebellion, riot, strike, or similar disruption."
        ),
        "what_to_find": (
            "Find a force majeure clause or equivalent clause listing events beyond control, including war, terrorism, "
            "rebellion, strike, riot, civil commotion, acts of God, government restraint, epidemic, or comparable events."
        ),
        "not_enough": (
            "A general extension-of-time clause, delay clause, rain/weather note, or employer discretion clause is not "
            "enough unless it clearly functions as force majeure and covers beyond-control events."
        ),
        "examples": {
            "yes": [
                "Force majeure includes war, terrorism, rebellion, riot, strike, and acts of God.",
                "Neither party is liable for delay caused by events beyond control including war, civil unrest, strike, or terrorism.",
            ],
            "no": [
                "The contractor may seek extension of time for delay, subject to employer approval.",
                "Rainy season and normal site conditions shall be considered in the schedule.",
            ],
            "edge": [
                "An extension-of-time clause can support Yes only if it expressly covers force majeure/beyond-control events.",
            ],
        },
        "decision_rule": "Yes only if a force majeure clause exists and covers events such as war, terrorism, rebellion, strike, or equivalent events beyond control.",
        "search_terms": [
            "force majeure",
            "war",
            "terrorism",
            "rebellion",
            "strike",
            "riot",
            "act of god",
            "beyond control",
        ],
    },
    {
        "id": "ld_full_discharge",
        "s_no": 5,
        "topic": "Liquidated damages fully discharge SUEZ of all its liability",
        "description": (
            "Check whether payment of liquidated damages is the exclusive remedy and fully discharges SUEZ/contractor "
            "from further liability for the relevant breach, usually delay. This is stronger than simply having LD."
        ),
        "what_to_find": (
            "Find wording that LD is the sole remedy, exclusive remedy, full and final compensation, full discharge, "
            "or that no further damages/claims are recoverable after LD."
        ),
        "not_enough": (
            "A clause that only imposes LD, delay damages, compensation, penalty, or a percentage cap is not enough. "
            "It must say LD fully discharges liability or is the exclusive/sole remedy. Otherwise answer No."
        ),
        "examples": {
            "yes": [
                "Payment of liquidated damages shall be the sole and exclusive remedy for delay.",
                "LD shall be full and final compensation and shall discharge the contractor from further liability for delay.",
            ],
            "no": [
                "Liquidated damages shall be 0.5% per week subject to a maximum of 10% of contract value.",
                "The employer may recover LD without prejudice to other rights and remedies.",
            ],
            "edge": [
                "An LD cap is not enough; the clause must also bar further claims or state sole/exclusive/full discharge.",
            ],
        },
        "decision_rule": "Yes only if the liquidated damages clause states that LD is the sole remedy or fully discharges contractor/SUEZ liability for delay or breach.",
        "search_terms": [
            "liquidated damages",
            "liquidated damage",
            "LD",
            "sole remedy",
            "full discharge",
            "fully discharge",
            "discharge of liability",
            "all liability",
            "delay damages",
            "penalty",
        ],
    },
    {
        "id": "effective_date_advance_payment",
        "s_no": 6,
        "topic": "Effective Date clause which includes advance Payment",
        "description": (
            "Check whether the contract's effective date, commencement, or start of obligations is conditional on "
            "advance payment or key conditions precedent such as CIF/conditions for effectiveness. The concern is "
            "whether SUEZ must start before receiving required advance payment."
        ),
        "what_to_find": (
            "Find wording linking effective date, commencement date, start date, notice to proceed, or mobilization "
            "to receipt/payment of advance payment, mobilization advance, CIF, or other conditions precedent."
        ),
        "not_enough": (
            "A standalone advance payment clause or standalone commencement date is not enough if they are not linked. "
            "If advance payment exists but is not a condition for effectiveness/commencement, answer No."
        ),
        "examples": {
            "yes": [
                "The effective date shall occur upon receipt of advance payment and fulfilment of conditions precedent.",
                "Commencement shall be from the date of advance payment release/CIF fulfilment.",
            ],
            "no": [
                "Advance payment shall be 10% against bank guarantee, but commencement starts from letter of acceptance.",
                "The work shall commence within 15 days of signing the agreement; advance payment is described separately.",
            ],
            "edge": [
                "If advance payment and commencement appear in different clauses, inspect whether one is expressly conditional on the other.",
            ],
        },
        "decision_rule": "Yes only if the contract effective date or commencement is linked to advance payment, CIF, or equivalent conditions precedent.",
        "search_terms": [
            "effective date",
            "commencement date",
            "advance payment",
            "conditions precedent",
            "CIF",
            "mobilization advance",
            "date of commencement",
            "contract agreement",
        ],
    },
    {
        "id": "currency_exchange_protection",
        "s_no": 7,
        "topic": "Protection in the event of currency exchange",
        "description": (
            "Check whether the tender protects SUEZ/contractor from foreign exchange or currency fluctuation risk. "
            "This matters where costs, equipment, loans, imports, or bid assumptions may be exposed to currency movement."
        ),
        "what_to_find": (
            "Find a currency variation, foreign exchange adjustment, exchange-rate compensation, payment adjustment, "
            "multi-currency payment, or explicit protection against currency fluctuation."
        ),
        "not_enough": (
            "A clause merely saying payments are in INR/Indian Rupees is not protection. Silence on currency variation "
            "or fixed INR payment normally means No."
        ),
        "examples": {
            "yes": [
                "Payments shall be adjusted for foreign exchange variation based on exchange-rate movement.",
                "The contractor shall be compensated for currency fluctuation on imported components.",
            ],
            "no": [
                "All payments shall be made in Indian Rupees.",
                "The quoted price shall be firm and inclusive of all currency fluctuation risk.",
            ],
            "edge": [
                "A price adjustment clause is Yes only if it covers currency/foreign exchange movement, not merely taxes or inflation.",
            ],
        },
        "decision_rule": "Yes only if the tender provides currency exchange variation, foreign exchange protection, or payment adjustment for exchange rate changes.",
        "search_terms": [
            "currency exchange",
            "foreign exchange",
            "exchange rate",
            "forex",
            "currency variation",
            "exchange variation",
            "Indian rupees",
            "INR",
            "payment currency",
        ],
    },
]


LEGAL_ROW_BY_ID = {row["id"]: row for row in LEGAL_ROWS}


def _contains_term(text: str, term: str) -> bool:
    if len(term) <= 4 and term.isupper():
        return re.search(rf"\b{re.escape(term)}\b", text) is not None
    return term.lower() in text.lower()


def detect_legal_flags(topic: dict[str, Any]) -> tuple[list[str], dict[str, str], str]:
    text = " ".join(
        str(topic.get(key, ""))
        for key in ["topic_name", "topic_description", "content", "document_name"]
    )
    flags: list[str] = []
    reasons: dict[str, str] = {}
    high_hits = 0
    for row in LEGAL_ROWS:
        matched = [term for term in row["search_terms"] if _contains_term(text, term)]
        if matched:
            flags.append(row["id"])
            reasons[row["id"]] = "Matched legal terms: " + ", ".join(matched[:5])
            if len(matched) >= 2:
                high_hits += 1
    if high_hits:
        confidence = "high"
    elif flags:
        confidence = "medium"
    else:
        confidence = "low"
    return flags, reasons, confidence
