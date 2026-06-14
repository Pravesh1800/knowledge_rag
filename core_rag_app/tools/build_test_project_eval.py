from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "evaluation_set.v1.0"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def normalize(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip())


def terms_from_text(text: str, limit: int = 2) -> list[str]:
    candidates = []
    for token in re.findall(r"[A-Za-z][A-Za-z0-9&/-]{3,}", text):
        low = token.lower().strip("-/")
        if low in {
            "with",
            "this",
            "that",
            "from",
            "shall",
            "will",
            "page",
            "volume",
            "section",
            "requirements",
            "specifications",
            "contract",
            "project",
            "document",
        }:
            continue
        if token not in candidates:
            candidates.append(token)
        if len(candidates) >= limit:
            break
    return candidates


def expected_from_card(card: dict[str, Any], with_terms: bool = True) -> dict[str, Any]:
    expected = {
        "card_id": card.get("card_id", ""),
        "document_name": card.get("document_name", ""),
        "page_no": card.get("page_no"),
    }
    if with_terms:
        terms = terms_from_text(f"{card.get('card_name', '')} {card.get('content', '')}")
        if terms:
            expected["must_contain_terms"] = terms
    return expected


def add_case(
    cases: list[dict[str, Any]],
    case_id: str,
    question: str,
    evidence_cards: list[dict[str, Any]],
    expected_answer_points: list[str],
    expected_cross_document: bool | None = None,
) -> None:
    seen = set()
    expected_evidence = []
    documents = set()
    for card in evidence_cards:
        card_id = str(card.get("card_id", ""))
        if not card_id or card_id in seen:
            continue
        seen.add(card_id)
        documents.add(str(card.get("document_name", "")))
        expected_evidence.append(expected_from_card(card))
    if not expected_evidence:
        return
    if expected_cross_document is None:
        expected_cross_document = len({doc for doc in documents if doc}) >= 2
    cases.append(
        {
            "id": case_id,
            "question": question,
            "expected_answer_points": expected_answer_points,
            "expected_evidence": expected_evidence,
            "expected_cross_document": expected_cross_document,
            "forbidden_answer_terms": [],
        }
    )


def build_manual_cases(cards_by_id: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []

    manual_specs = [
        (
            "power_supply_responsibility",
            "Where does the project say who pays for construction power and the new power connection?",
            ["card_d386f55df4319a7b", "card_0f6fb88b0746d5a0"],
            ["construction power paid by contractor", "new power connection paid by employer"],
        ),
        (
            "employer_identity",
            "Who is identified as the Employer for the 910 MLD WTP Panjrapur tender?",
            ["card_28a0dcd097b3e443"],
            ["Brihanmumbai Municipal Corporation", "Employer"],
        ),
        (
            "operating_license_parties",
            "Which source identifies the parties to the Operating Licence between the Employer and Contractor?",
            ["card_5416272bded6e222"],
            ["Brihanmumbai Municipal Corporation", "Contractor", "Operating Licence"],
        ),
        (
            "itb_defined_terms",
            "Where are the ITB defined terms such as Employer, Contract Forms, GST, JV, and Base Date listed?",
            ["card_730a2739692b9182"],
            ["Employer", "Contract Forms", "GST", "JV"],
        ),
        (
            "pre_bid_meeting_process",
            "What source explains the pre-bid meeting process and written question submission timing?",
            ["card_9d2f468da24e5db3"],
            ["pre-bid meeting", "questions", "3 working days"],
        ),
        (
            "price_schedule_notes",
            "Which evidence explains the whole life cost spreadsheet notes, discount rate, and cost limits?",
            ["card_2b436cb6386b27ef"],
            ["Whole Life Cost", "discount rate", "cost centre limits"],
        ),
    ]

    for case_id, question, ids, points in manual_specs:
        evidence = [cards_by_id[card_id] for card_id in ids if card_id in cards_by_id]
        add_case(cases, case_id, question, evidence, points)
    return cases


def relationship_quality(rel: dict[str, Any]) -> float:
    def as_float(value: Any) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    return (
        0.5 * as_float(rel.get("confidence_score"))
        + 0.3 * as_float(rel.get("evidence_strength"))
        + 0.2 * as_float(rel.get("source_coverage"))
    )


def build_relationship_cases(
    graph: dict[str, Any],
    cards_by_id: dict[str, dict[str, Any]],
    max_cases: int,
) -> list[dict[str, Any]]:
    relationships = [
        rel
        for rel in graph.get("domain_relationships", []) or []
        if isinstance(rel, dict) and rel.get("card_links")
    ]
    relationships.sort(
        key=lambda rel: (
            str(rel.get("document_scope")) == "cross_document",
            relationship_quality(rel),
            len(rel.get("card_links") or []),
        ),
        reverse=True,
    )

    cases: list[dict[str, Any]] = []
    used_pairs: set[tuple[str, str]] = set()
    used_types: dict[str, int] = {}
    used_docs: dict[str, int] = {}

    for rel in relationships:
        rel_type = normalize(rel.get("relationship_type")).lower() or "relationship"
        if used_types.get(rel_type, 0) >= 3:
            continue

        chosen_link = None
        for link in rel.get("card_links", []) or []:
            if not isinstance(link, dict):
                continue
            main = cards_by_id.get(str(link.get("main_card_id", "")))
            related = cards_by_id.get(str(link.get("related_card_id", "")))
            if not main or not related:
                continue
            pair_key = tuple(sorted([str(main["card_id"]), str(related["card_id"])]))
            if pair_key in used_pairs:
                continue
            doc_pair = " | ".join(sorted([str(main.get("document_name", "")), str(related.get("document_name", ""))]))
            if used_docs.get(doc_pair, 0) >= 2:
                continue
            chosen_link = (link, main, related, pair_key, doc_pair)
            break
        if not chosen_link:
            continue

        link, main, related, pair_key, doc_pair = chosen_link
        main_name = normalize(main.get("card_name"))
        related_name = normalize(related.get("card_name"))
        description = normalize(rel.get("relationship_description"))
        short_description = description
        if len(short_description) > 180:
            short_description = short_description[:177].rstrip() + "..."
        case_id = f"relationship_{len(cases) + 1:02d}_{re.sub(r'[^a-z0-9]+', '_', rel_type)[:24].strip('_')}"
        question = (
            f"Which cited evidence connects {main_name} with {related_name}, "
            f"and what is the {rel_type} between them?"
        )
        if short_description:
            question += f" Context: {short_description}"

        add_case(
            cases,
            case_id,
            question,
            [main, related],
            [main_name, related_name, rel_type],
            expected_cross_document=str(rel.get("document_scope")) == "cross_document",
        )
        used_pairs.add(pair_key)
        used_docs[doc_pair] = used_docs.get(doc_pair, 0) + 1
        used_types[rel_type] = used_types.get(rel_type, 0) + 1
        if len(cases) >= max_cases:
            break

    return cases


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a real eval set from the test project's indexed graph.")
    parser.add_argument("--project-root", default="projects/test", help="Path to the project root.")
    parser.add_argument("--out", default="eval_sets/test_project_graph_retrieval_eval.json", help="Output eval JSON path.")
    parser.add_argument("--relationship-cases", type=int, default=18, help="Number of graph relationship cases.")
    args = parser.parse_args()

    project_root = Path(args.project_root)
    indexes = project_root / "indexes"
    cards_payload = read_json(indexes / "card_index.json")
    graph = read_json(indexes / "knowledge_graph.json")
    cards = cards_payload.get("cards", []) if isinstance(cards_payload, dict) else cards_payload
    cards_by_id = {str(card.get("card_id")): card for card in cards if isinstance(card, dict) and card.get("card_id")}

    cases = build_manual_cases(cards_by_id)
    cases.extend(build_relationship_cases(graph, cards_by_id, max_cases=args.relationship_cases))

    eval_set = {
        "schema_version": SCHEMA_VERSION,
        "name": "test_project_graph_retrieval_eval",
        "description": (
            "Auto-built benchmark from the real test project. Includes practical project questions "
            "and high-quality card-linked graph relationships."
        ),
        "project_id": project_root.name,
        "case_count": len(cases),
        "cases": cases,
    }
    out_path = Path(args.out)
    write_json(out_path, eval_set)
    print(f"Wrote {len(cases)} cases to {out_path}")


if __name__ == "__main__":
    main()
