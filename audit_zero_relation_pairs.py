from __future__ import annotations

import argparse
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from relationship_map import (
    Biome,
    Community,
    MAX_RELATION_PAIR_ATTEMPTS,
    PROJECT_ROOT,
    RELATIONSHIP_MAP_PATH,
    TOPIC_INDEX_PATH,
    biome_pair_key,
    build_lookups,
    create_client,
    extract_biome_relation_pair_with_retries,
    load_dotenv,
    read_json,
    write_json,
)


AUDIT_PATH = PROJECT_ROOT / "indexes" / "relation_audit.json"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def relation_pair_keys(relations: list[dict[str, Any]]) -> set[str]:
    return {
        biome_pair_key(
            str(relation.get("main_biome", "")).strip(),
            str(relation.get("related_biome", "")).strip(),
        )
        for relation in relations
        if relation.get("main_biome") and relation.get("related_biome")
    }


def load_communities(raw_communities: list[dict[str, Any]]) -> list[Community]:
    return [
        Community(
            community_name=str(raw.get("community_name", "")).strip(),
            community_description=str(raw.get("community_description", "")).strip(),
            document_id=str(raw.get("document_id", "")).strip(),
            document_name=str(raw.get("document_name", "")).strip(),
            topic_names=list(raw.get("topic_names", [])),
        )
        for raw in raw_communities
    ]


def load_biomes(raw_biomes: list[dict[str, Any]]) -> list[Biome]:
    return [
        Biome(
            biome_name=str(raw.get("biome_name", "")).strip(),
            biome_description=str(raw.get("biome_description", "")).strip(),
            document_id=str(raw.get("document_id", "")).strip(),
            document_name=str(raw.get("document_name", "")).strip(),
            community_names=list(raw.get("community_names", [])),
        )
        for raw in raw_biomes
    ]


def write_audit(data: dict[str, Any]) -> None:
    write_json(AUDIT_PATH, {"updated_at": now_iso(), **data})


def audit_candidates(relationship_map: dict[str, Any]) -> list[str]:
    processed = set(relationship_map.get("processed_relation_pairs", []))
    relation_keys = relation_pair_keys(relationship_map.get("biome_relations", []))
    failed = set((relationship_map.get("failed_relation_pairs", {}) or {}).keys())
    return sorted(processed - relation_keys - failed)


def existing_audit() -> dict[str, Any]:
    return read_json(
        AUDIT_PATH,
        {
            "status": "not_started",
            "audit_source": str(RELATIONSHIP_MAP_PATH),
            "candidate_pairs": [],
            "audited_pairs": [],
            "verified_empty_pairs": [],
            "recovered_relations": [],
            "failed_relation_pairs": {},
        },
    )


def build_pair_lookup(biomes: list[Biome]) -> dict[str, tuple[dict[str, Any], dict[str, Any]]]:
    biome_dicts = [asdict(biome) for biome in biomes]
    return {
        biome_pair_key(main["biome_name"], related["biome_name"]): (main, related)
        for main in biome_dicts
        for related in biome_dicts
        if main["biome_name"] != related["biome_name"]
    }


def run_audit(workers: int, limit: int | None = None) -> None:
    load_dotenv()
    relationship_map = read_json(RELATIONSHIP_MAP_PATH, {})
    topics = read_json(TOPIC_INDEX_PATH, [])
    communities = load_communities(relationship_map.get("communities", []))
    biomes = load_biomes(relationship_map.get("biomes", []))
    topic_lookup, community_lookup = build_lookups(topics, communities)
    pair_lookup = build_pair_lookup(biomes)
    candidates = [key for key in audit_candidates(relationship_map) if key in pair_lookup]

    audit = existing_audit()
    audited_pairs = set(audit.get("audited_pairs", []))
    verified_empty_pairs = set(audit.get("verified_empty_pairs", []))
    recovered_relations = list(audit.get("recovered_relations", []))
    failed_relation_pairs = dict(audit.get("failed_relation_pairs", {}) or {})
    remaining = [
        key
        for key in candidates
        if key not in audited_pairs and key not in failed_relation_pairs
    ]
    if limit is not None:
        remaining = remaining[:limit]

    write_audit(
        {
            "status": "running",
            "audit_source": str(RELATIONSHIP_MAP_PATH),
            "started_at": audit.get("started_at") or now_iso(),
            "candidate_pairs": candidates,
            "total_candidates": len(candidates),
            "remaining_at_start": len(remaining),
            "audited_pairs": sorted(audited_pairs),
            "verified_empty_pairs": sorted(verified_empty_pairs),
            "recovered_relations": recovered_relations,
            "failed_relation_pairs": failed_relation_pairs,
        }
    )

    if not remaining:
        write_audit(
            {
                "status": "complete",
                "audit_source": str(RELATIONSHIP_MAP_PATH),
                "started_at": audit.get("started_at") or now_iso(),
                "completed_at": now_iso(),
                "candidate_pairs": candidates,
                "total_candidates": len(candidates),
                "audited_pairs": sorted(audited_pairs),
                "verified_empty_pairs": sorted(verified_empty_pairs),
                "recovered_relations": recovered_relations,
                "failed_relation_pairs": failed_relation_pairs,
            }
        )
        print("No zero-relation pairs remain to audit.")
        return

    client, model = create_client()
    worker_local = threading.local()

    def worker_client():
        existing_client = getattr(worker_local, "client", None)
        if existing_client is None:
            existing_client, _ = create_client()
            worker_local.client = existing_client
        return existing_client

    def audit_pair(pair_key: str) -> tuple[str, list[dict[str, Any]], str | None]:
        main, related = pair_lookup[pair_key]
        print(f"Auditing zero-relation pair: {pair_key}")
        try:
            extracted = extract_biome_relation_pair_with_retries(
                worker_client() if workers > 1 else client,
                model,
                main,
                related,
                community_lookup,
                topic_lookup,
            )
            return pair_key, [asdict(relation) for relation in extracted], None
        except Exception as exc:
            return pair_key, [], (
                f"Failed after {MAX_RELATION_PAIR_ATTEMPTS} attempts: {exc}"
            )

    completed_this_run = 0
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = {executor.submit(audit_pair, key): key for key in remaining}
        for future in as_completed(futures):
            pair_key, extracted, error = future.result()
            completed_this_run += 1
            audited_pairs.add(pair_key)
            if error:
                failed_relation_pairs[pair_key] = error
                print(f"ERROR: audit failed for {pair_key}. {error}")
            elif extracted:
                recovered_relations.extend(extracted)
                print(f"Recovered {len(extracted)} relation(s) for {pair_key}.")
            else:
                verified_empty_pairs.add(pair_key)
                print(f"Verified empty relation pair: {pair_key}.")

            write_audit(
                {
                    "status": "running",
                    "audit_source": str(RELATIONSHIP_MAP_PATH),
                    "started_at": audit.get("started_at") or now_iso(),
                    "candidate_pairs": candidates,
                    "total_candidates": len(candidates),
                    "audited_count": len(audited_pairs),
                    "completed_this_run": completed_this_run,
                    "audited_pairs": sorted(audited_pairs),
                    "verified_empty_pairs": sorted(verified_empty_pairs),
                    "recovered_relations": recovered_relations,
                    "failed_relation_pairs": failed_relation_pairs,
                }
            )
            time.sleep(0.1)

    final_status = "failed" if failed_relation_pairs else "complete"
    write_audit(
        {
            "status": final_status,
            "audit_source": str(RELATIONSHIP_MAP_PATH),
            "started_at": audit.get("started_at") or now_iso(),
            "completed_at": now_iso(),
            "candidate_pairs": candidates,
            "total_candidates": len(candidates),
            "audited_count": len(audited_pairs),
            "audited_pairs": sorted(audited_pairs),
            "verified_empty_pairs": sorted(verified_empty_pairs),
            "recovered_relations": recovered_relations,
            "failed_relation_pairs": failed_relation_pairs,
        }
    )
    print(
        f"Audit {final_status}: {len(audited_pairs)} audited, "
        f"{len(recovered_relations)} recovered relation records, "
        f"{len(verified_empty_pairs)} verified empty pairs, "
        f"{len(failed_relation_pairs)} failed pairs."
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit processed biome pairs that have zero relation records."
    )
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--limit", type=int, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_audit(workers=args.workers, limit=args.limit)


if __name__ == "__main__":
    main()
