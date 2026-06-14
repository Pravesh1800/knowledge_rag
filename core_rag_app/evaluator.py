from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from searcher import PROJECT_ROOT, TreeSearcher, load_dotenv, slugify, write_json


EVALUATION_SET_SCHEMA_VERSION = "evaluation_set.v1.0"
EVALUATION_REPORT_SCHEMA_VERSION = "evaluation_report.v1.0"
EVAL_REPORTS_DIR = PROJECT_ROOT / "indexes" / "eval_reports"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def normalize(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


def text_contains_terms(text: str, terms: list[str]) -> bool:
    normalized = normalize(text)
    return all(normalize(term) in normalized for term in terms if str(term).strip())


def evidence_records(search_result: dict[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for hit in search_result.get("hits", []) or []:
        records.append(
            {
                "source": "hit",
                "card_id": hit.get("card_id", ""),
                "card_name": hit.get("card_name", ""),
                "document_name": hit.get("document_name", ""),
                "page_no": hit.get("page_no"),
                "content": hit.get("content", ""),
            }
        )
        for related in hit.get("related_cards", []) or []:
            records.append(
                {
                    "source": "related_card",
                    "card_id": related.get("card_id", ""),
                    "card_name": related.get("card_name", ""),
                    "document_name": related.get("document_name", ""),
                    "page_no": related.get("page_no"),
                    "content": related.get("content", ""),
                }
            )
    return records


def evidence_matches(expected: dict[str, Any], record: dict[str, Any]) -> bool:
    for key in ("card_id", "card_name", "document_name"):
        if expected.get(key) and normalize(expected.get(key)) != normalize(record.get(key)):
            return False
    if expected.get("page_no") is not None:
        try:
            if int(expected.get("page_no")) != int(record.get("page_no") or 0):
                return False
        except (TypeError, ValueError):
            return False
    terms = [str(term) for term in expected.get("must_contain_terms", []) or []]
    if terms:
        text = " ".join(
            [
                str(record.get("card_name", "")),
                str(record.get("document_name", "")),
                str(record.get("content", "")),
            ]
        )
        if not text_contains_terms(text, terms):
            return False
    return True


def check_expected_evidence(
    expected_evidence: list[dict[str, Any]],
    records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    matched = []
    missed = []
    for expected in expected_evidence:
        match = next((record for record in records if evidence_matches(expected, record)), None)
        if match:
            matched.append({"expected": expected, "matched": match})
        else:
            missed.append(expected)
    return matched, missed


def trace_names_by_type(trace: list[dict[str, Any]]) -> dict[str, set[str]]:
    names: dict[str, set[str]] = {"domain": set(), "cluster": set(), "card": set()}
    for item in trace:
        node_type = str(item.get("node_type", ""))
        name = str(item.get("name", ""))
        if node_type in names and name:
            names[node_type].add(name)
    return names


def candidate_cards_for_expected(
    expected: dict[str, Any],
    cards: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    candidates = []
    for card in cards:
        if expected.get("card_id") and normalize(expected.get("card_id")) != normalize(card.get("card_id")):
            continue
        if expected.get("card_name") and normalize(expected.get("card_name")) != normalize(card.get("card_name")):
            continue
        if expected.get("document_name") and normalize(expected.get("document_name")) != normalize(card.get("document_name")):
            continue
        if expected.get("page_no") is not None:
            try:
                if int(expected.get("page_no")) != int(card.get("page_no") or 0):
                    continue
            except (TypeError, ValueError):
                continue
        terms = [str(term) for term in expected.get("must_contain_terms", []) or []]
        if terms:
            text = " ".join(
                [
                    str(card.get("card_name", "")),
                    str(card.get("card_description", "")),
                    str(card.get("document_name", "")),
                    str(card.get("content", "")),
                ]
            )
            if not text_contains_terms(text, terms):
                continue
        candidates.append(card)
    return candidates


def build_card_graph_lookup(graph: dict[str, Any]) -> dict[str, dict[str, list[dict[str, Any]]]]:
    clusters_for_card: dict[str, list[dict[str, Any]]] = {}
    domains_for_cluster: dict[str, list[dict[str, Any]]] = {}

    for cluster in graph.get("clusters", []) or []:
        for card_name in cluster.get("card_names", []) or []:
            clusters_for_card.setdefault(str(card_name), []).append(cluster)
        for card_id in cluster.get("card_ids", []) or []:
            clusters_for_card.setdefault(str(card_id), []).append(cluster)

    for domain in graph.get("domains", []) or []:
        for cluster_name in domain.get("cluster_names", []) or []:
            domains_for_cluster.setdefault(str(cluster_name), []).append(domain)
        for cluster_id in domain.get("cluster_ids", []) or []:
            domains_for_cluster.setdefault(str(cluster_id), []).append(domain)

    return {
        "clusters_for_card": clusters_for_card,
        "domains_for_cluster": domains_for_cluster,
    }


def document_entered_search_path(
    expected_document_name: str,
    retrieved_documents: list[str],
    target_cards: list[dict[str, Any]],
    visited: dict[str, set[str]],
    lookup: dict[str, dict[str, list[dict[str, Any]]]],
) -> bool:
    if not expected_document_name:
        return True
    if normalize(expected_document_name) in {normalize(name) for name in retrieved_documents}:
        return True
    for card in target_cards:
        if str(card.get("card_name", "")) in visited["card"]:
            return True
        clusters = lookup["clusters_for_card"].get(str(card.get("card_name", "")), [])
        clusters += lookup["clusters_for_card"].get(str(card.get("card_id", "")), [])
        for cluster in clusters:
            if str(cluster.get("cluster_name", "")) in visited["cluster"]:
                return True
            domains = lookup["domains_for_cluster"].get(str(cluster.get("cluster_name", "")), [])
            domains += lookup["domains_for_cluster"].get(str(cluster.get("cluster_id", "")), [])
            if any(str(domain.get("domain_name", "")) in visited["domain"] for domain in domains):
                return True
    return False


def diagnose_missed_evidence(
    expected: dict[str, Any],
    search_result: dict[str, Any],
    records: list[dict[str, Any]],
    searcher: TreeSearcher,
) -> dict[str, Any]:
    graph = searcher.map
    lookup = build_card_graph_lookup(graph)
    visited = trace_names_by_type(search_result.get("trace", []) or [])
    retrieved_documents = sorted(
        {
            str(record.get("document_name", ""))
            for record in records
            if str(record.get("document_name", "")).strip()
        }
    )
    expected_document_name = str(expected.get("document_name", ""))
    target_cards = candidate_cards_for_expected(expected, searcher.cards)

    page_mismatch_records = []
    if expected.get("page_no") is not None:
        for record in records:
            same_card = False
            if expected.get("card_id") and normalize(expected.get("card_id")) == normalize(record.get("card_id")):
                same_card = True
            if expected.get("card_name") and normalize(expected.get("card_name")) == normalize(record.get("card_name")):
                same_card = True
            if not same_card and expected_document_name:
                same_card = normalize(expected_document_name) == normalize(record.get("document_name"))
            if same_card:
                try:
                    if int(expected.get("page_no")) != int(record.get("page_no") or 0):
                        page_mismatch_records.append(record)
                except (TypeError, ValueError):
                    page_mismatch_records.append(record)
    if page_mismatch_records:
        return {
            "primary_reason": "card_retrieved_but_page_mismatch",
            "explanation": "A likely evidence card/document was retrieved, but not from the expected page.",
            "expected": expected,
            "target_cards_found_in_index": len(target_cards),
            "page_mismatch_records": page_mismatch_records[:5],
            "retrieved_documents": retrieved_documents,
        }

    document_entered = document_entered_search_path(
        expected_document_name,
        retrieved_documents,
        target_cards,
        visited,
        lookup,
    )
    if expected_document_name and not document_entered:
        return {
            "primary_reason": "expected_document_never_entered_search_path",
            "explanation": "The expected document did not appear in retrieved hits and none of its mapped graph nodes were visited.",
            "expected": expected,
            "target_cards_found_in_index": len(target_cards),
            "retrieved_documents": retrieved_documents,
            "visited_domains": sorted(visited["domain"]),
            "visited_clusters": sorted(visited["cluster"]),
            "visited_cards": sorted(visited["card"]),
        }

    target_clusters_by_name: dict[str, dict[str, Any]] = {}
    target_domains_by_name: dict[str, dict[str, Any]] = {}
    for card in target_cards:
        clusters = lookup["clusters_for_card"].get(str(card.get("card_name", "")), [])
        clusters += lookup["clusters_for_card"].get(str(card.get("card_id", "")), [])
        for cluster in clusters:
            if cluster.get("cluster_name"):
                target_clusters_by_name[str(cluster["cluster_name"])] = cluster
            domains = lookup["domains_for_cluster"].get(str(cluster.get("cluster_name", "")), [])
            domains += lookup["domains_for_cluster"].get(str(cluster.get("cluster_id", "")), [])
            for domain in domains:
                if domain.get("domain_name"):
                    target_domains_by_name[str(domain["domain_name"])] = domain

    target_domain_names = set(target_domains_by_name)
    target_cluster_names = set(target_clusters_by_name)
    target_card_names = {str(card.get("card_name", "")) for card in target_cards if card.get("card_name")}
    target_card_ids = {str(card.get("card_id", "")) for card in target_cards if card.get("card_id")}

    visited_target_domains = sorted(target_domain_names & visited["domain"])
    visited_target_clusters = sorted(target_cluster_names & visited["cluster"])
    retrieved_target_cards = [
        record
        for record in records
        if str(record.get("card_name", "")) in target_card_names
        or str(record.get("card_id", "")) in target_card_ids
    ]

    if target_domain_names and not visited_target_domains:
        primary_reason = "no_matching_domain_visited"
        explanation = "The expected evidence is indexed under one or more domains, but none of those domains were visited."
    elif visited_target_domains and target_cluster_names and not visited_target_clusters:
        primary_reason = "domain_visited_but_cluster_missed"
        explanation = "The search entered a correct domain, but did not visit the cluster that contains the expected card."
    elif visited_target_clusters and not retrieved_target_cards:
        primary_reason = "cluster_visited_but_card_missed"
        explanation = "The search reached the expected cluster, but did not retrieve the expected card."
    elif not target_cards:
        primary_reason = "expected_evidence_not_found_in_index"
        explanation = "No indexed card matched the expected document/page/card/terms, so this may be an indexing or eval-set mismatch."
    else:
        primary_reason = "missed_expected_evidence_unknown"
        explanation = "The expected evidence was not retrieved, but the trace does not cleanly identify a single failure level."

    return {
        "primary_reason": primary_reason,
        "explanation": explanation,
        "expected": expected,
        "target_cards_found_in_index": len(target_cards),
        "target_cards": [
            {
                "card_id": card.get("card_id", ""),
                "card_name": card.get("card_name", ""),
                "document_name": card.get("document_name", ""),
                "page_no": card.get("page_no"),
            }
            for card in target_cards[:10]
        ],
        "target_domains": sorted(target_domain_names),
        "target_clusters": sorted(target_cluster_names),
        "visited_target_domains": visited_target_domains,
        "visited_target_clusters": visited_target_clusters,
        "retrieved_documents": retrieved_documents,
        "visited_domains": sorted(visited["domain"]),
        "visited_clusters": sorted(visited["cluster"]),
        "visited_cards": sorted(visited["card"]),
    }


def check_answer(case: dict[str, Any], records: list[dict[str, Any]]) -> dict[str, Any]:
    answer = str(case.get("candidate_answer", "")).strip()
    if not answer:
        return {"status": "not_run", "reason": "No candidate_answer provided in eval case."}

    forbidden_terms = [str(term) for term in case.get("forbidden_answer_terms", []) or []]
    forbidden_hits = [term for term in forbidden_terms if normalize(term) in normalize(answer)]
    evidence_text = " ".join(
        f"{record.get('document_name', '')} page {record.get('page_no', '')} {record.get('content', '')}"
        for record in records
    )
    expected_terms = [str(term) for term in case.get("expected_answer_points", []) or []]
    missing_expected_terms = [
        term for term in expected_terms if normalize(term) and normalize(term) not in normalize(answer)
    ]
    unsupported_expected_terms = [
        term for term in expected_terms if normalize(term) and normalize(term) not in normalize(evidence_text)
    ]
    status = "pass"
    if forbidden_hits or missing_expected_terms or unsupported_expected_terms:
        status = "fail"
    return {
        "status": status,
        "forbidden_terms_found": forbidden_hits,
        "missing_expected_answer_points": missing_expected_terms,
        "expected_points_not_supported_by_retrieved_evidence": unsupported_expected_terms,
    }


def evaluate_case(case: dict[str, Any], dry_run: bool, max_hits: int) -> dict[str, Any]:
    query = str(case.get("question", "")).strip()
    if not query:
        raise ValueError(f"Evaluation case {case.get('id', '<missing id>')} has no question.")

    searcher = TreeSearcher(query=query, dry_run=dry_run, max_hits=max_hits)
    search_result = searcher.search()
    records = evidence_records(search_result)
    expected_evidence = list(case.get("expected_evidence", []) or [])
    matched, missed = check_expected_evidence(expected_evidence, records)
    failure_diagnosis = [
        diagnose_missed_evidence(
            expected=expected,
            search_result=search_result,
            records=records,
            searcher=searcher,
        )
        for expected in missed
    ]
    expected_cross_document = bool(case.get("expected_cross_document", False))
    retrieved_documents = sorted(
        {
            str(record.get("document_name", ""))
            for record in records
            if str(record.get("document_name", "")).strip()
        }
    )
    cross_document_pass = True
    if expected_cross_document:
        cross_document_pass = len(retrieved_documents) >= 2

    right_card = any(
        item.get("card_id") or item.get("card_name")
        for item in (match["expected"] for match in matched)
    )
    right_page = any(
        item.get("document_name") and item.get("page_no") is not None
        for item in (match["expected"] for match in matched)
    )
    answer_check = check_answer(case, records)
    passed = not missed and cross_document_pass and answer_check.get("status") != "fail"
    return {
        "id": case.get("id", ""),
        "question": query,
        "passed": passed,
        "score": round(
            (
                (len(matched) / max(1, len(expected_evidence))) * 0.7
                + (0.15 if cross_document_pass else 0)
                + (0.15 if answer_check.get("status") in {"pass", "not_run"} else 0)
            ),
            4,
        ),
        "checks": {
            "expected_evidence_count": len(expected_evidence),
            "matched_evidence_count": len(matched),
            "missed_evidence_count": len(missed),
            "right_card": right_card,
            "right_page": right_page,
            "expected_cross_document": expected_cross_document,
            "cross_document_pass": cross_document_pass,
            "hallucination_check": answer_check,
        },
        "matched_evidence": matched,
        "missed_evidence": missed,
        "failure_diagnosis": failure_diagnosis,
        "retrieved_documents": retrieved_documents,
        "retrieved_hits": [
            {
                "card_id": hit.get("card_id", ""),
                "card_name": hit.get("card_name", ""),
                "document_name": hit.get("document_name", ""),
                "page_no": hit.get("page_no"),
                "related_card_count": len(hit.get("related_cards", []) or []),
            }
            for hit in search_result.get("hits", []) or []
        ],
        "trace": search_result.get("trace", []),
    }


def run_evaluation(eval_set_path: Path, dry_run: bool, max_hits: int) -> dict[str, Any]:
    load_dotenv()
    eval_set = read_json(eval_set_path)
    if eval_set.get("schema_version") != EVALUATION_SET_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported evaluation set schema: {eval_set.get('schema_version')}. "
            f"Expected {EVALUATION_SET_SCHEMA_VERSION}."
        )
    cases = list(eval_set.get("cases", []) or [])
    results = [evaluate_case(case, dry_run=dry_run, max_hits=max_hits) for case in cases]
    passed_count = sum(1 for result in results if result["passed"])
    failure_reason_counts: dict[str, int] = {}
    for result in results:
        for diagnosis in result.get("failure_diagnosis", []) or []:
            reason = str(diagnosis.get("primary_reason", "unknown"))
            failure_reason_counts[reason] = failure_reason_counts.get(reason, 0) + 1
    report = {
        "schema_version": EVALUATION_REPORT_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "eval_set": {
            "path": str(eval_set_path),
            "name": eval_set.get("name", eval_set_path.stem),
            "case_count": len(cases),
        },
        "mode": "dry_run" if dry_run else "model_ranked",
        "max_hits": max_hits,
        "summary": {
            "passed": passed_count,
            "failed": len(results) - passed_count,
            "pass_rate": round(passed_count / max(1, len(results)), 4),
            "average_score": round(
                sum(float(result["score"]) for result in results) / max(1, len(results)),
                4,
            ),
            "failure_reason_counts": dict(sorted(failure_reason_counts.items())),
        },
        "results": results,
    }
    output_path = (
        EVAL_REPORTS_DIR
        / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{slugify(str(eval_set.get('name') or eval_set_path.stem))}.json"
    )
    write_json(output_path, report)
    return {"report": report, "output_path": str(output_path)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Evidence Mesh retrieval evaluation sets.")
    parser.add_argument("eval_set", help="Path to an evaluation set JSON file.")
    parser.add_argument("--dry-run", action="store_true", help="Use deterministic keyword ranking.")
    parser.add_argument("--max-hits", type=int, default=12, help="Maximum hits per evaluation question.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = run_evaluation(Path(args.eval_set), dry_run=args.dry_run, max_hits=args.max_hits)
    report = result["report"]
    summary = report["summary"]
    print(f"Wrote evaluation report to {result['output_path']}")
    print(
        f"Passed {summary['passed']} / {summary['passed'] + summary['failed']} "
        f"cases, pass rate {summary['pass_rate']:.2%}, average score {summary['average_score']:.2%}."
    )
    for item in report["results"]:
        status = "PASS" if item["passed"] else "FAIL"
        print(
            f"- {status} {item['id']}: "
            f"{item['checks']['matched_evidence_count']} matched, "
            f"{item['checks']['missed_evidence_count']} missed"
        )
        for diagnosis in item.get("failure_diagnosis", []) or []:
            print(f"  reason: {diagnosis.get('primary_reason')} - {diagnosis.get('explanation')}")


if __name__ == "__main__":
    main()
