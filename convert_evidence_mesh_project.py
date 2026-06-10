from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from commercial_rules import detect_commercial_flags
from financial_rules import detect_financial_flags
from legal_rules import detect_legal_flags
from prebid_rules import detect_prebid_flags


APP_ROOT = Path(__file__).resolve().parent


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def topic_from_card(card: dict[str, Any]) -> dict[str, Any]:
    topic = {
        "topic_name": str(card.get("card_name", "")).strip() or str(card.get("card_id", "Untitled")),
        "topic_description": str(card.get("card_description", "")).strip(),
        "document_id": str(card.get("document_id", "")).strip(),
        "document_name": str(card.get("document_name", "")).strip(),
        "page_no": int(card.get("page_no") or 0),
        "content": str(card.get("content", "")).strip(),
        "source_type": str(card.get("source_type", "page")).strip() or "page",
        "topic_source": str(card.get("card_source", "text")).strip() or "text",
        "tags": list(card.get("tags", [])) if isinstance(card.get("tags"), list) else [],
        "image_descriptions": list(card.get("image_descriptions", []))
        if isinstance(card.get("image_descriptions"), list)
        else [],
        "created_at": str(card.get("created_at") or datetime.now(timezone.utc).isoformat()),
    }
    legal_flags, legal_reasons, legal_confidence = detect_legal_flags(topic)
    commercial_flags, commercial_reasons, commercial_confidence = detect_commercial_flags(topic)
    financial_flags, financial_reasons, financial_confidence = detect_financial_flags(topic)
    prebid_flags, prebid_reasons, prebid_confidence = detect_prebid_flags(topic)
    topic.update(
        {
            "legal_flags": legal_flags,
            "legal_flag_reasons": legal_reasons,
            "legal_confidence": legal_confidence,
            "commercial_flags": commercial_flags,
            "commercial_flag_reasons": commercial_reasons,
            "commercial_confidence": commercial_confidence,
            "financial_flags": financial_flags,
            "financial_flag_reasons": financial_reasons,
            "financial_confidence": financial_confidence,
            "prebid_flags": prebid_flags,
            "prebid_flag_reasons": prebid_reasons,
            "prebid_confidence": prebid_confidence,
        }
    )
    return topic


def relation_from_domain_relationship(raw: dict[str, Any]) -> dict[str, Any]:
    return {
        "main_biome": str(raw.get("main_domain", "")).strip(),
        "related_biome": str(raw.get("related_domain", "")).strip(),
        "relation_type": str(raw.get("relationship_type", "other")).strip() or "other",
        "relation_description": str(raw.get("relationship_description", "")).strip(),
        "evidence": str(raw.get("evidence", "")).strip(),
        "community_links": [
            {
                "main_community": str(link.get("main_cluster", "")).strip(),
                "related_community": str(link.get("related_cluster", "")).strip(),
                "relation": str(link.get("relationship", "")).strip(),
            }
            for link in raw.get("cluster_links", [])
            if isinstance(link, dict)
        ],
        "topic_links": [
            {
                "main_topic": str(link.get("main_card", "")).strip(),
                "related_topic": str(link.get("related_card", "")).strip(),
                "relation": str(link.get("relationship", "")).strip(),
            }
            for link in raw.get("card_links", [])
            if isinstance(link, dict)
        ],
        "document_scope": str(raw.get("document_scope", "unknown")).strip() or "unknown",
        "confidence_score": raw.get("confidence_score"),
        "evidence_strength": raw.get("evidence_strength"),
        "source_coverage": raw.get("source_coverage"),
        "generation_method": raw.get("generation_method"),
    }


def convert(source_root: Path, target_root: Path, project_name: str) -> dict[str, Any]:
    card_index = read_json(source_root / "indexes" / "card_index.json", {})
    graph = read_json(source_root / "indexes" / "knowledge_graph.json", {})
    cards = card_index.get("cards", [])
    clusters = graph.get("clusters", [])
    domains = graph.get("domains", [])
    relationships = graph.get("domain_relationships", [])
    source_manifest = graph.get("source_manifest", [])

    topics = [topic_from_card(card) for card in cards if isinstance(card, dict)]
    communities = [
        {
            "community_name": str(cluster.get("cluster_name", "")).strip(),
            "community_description": str(cluster.get("cluster_description", "")).strip(),
            "document_id": str(cluster.get("document_id", "")).strip(),
            "document_name": str(cluster.get("document_name", "")).strip(),
            "topic_names": list(cluster.get("card_names", []))
            if isinstance(cluster.get("card_names"), list)
            else [],
        }
        for cluster in clusters
        if isinstance(cluster, dict)
    ]
    biomes = [
        {
            "biome_name": str(domain.get("domain_name", "")).strip(),
            "biome_description": str(domain.get("domain_description", "")).strip(),
            "document_id": str(domain.get("document_id", "")).strip(),
            "document_name": str(domain.get("document_name", "")).strip(),
            "community_names": list(domain.get("cluster_names", []))
            if isinstance(domain.get("cluster_names"), list)
            else [],
        }
        for domain in domains
        if isinstance(domain, dict)
    ]
    biome_relations = [
        relation_from_domain_relationship(rel)
        for rel in relationships
        if isinstance(rel, dict)
    ]

    now = datetime.now(timezone.utc).isoformat()
    manifest = [
        {
            "document_id": str(item.get("document_id", "")).strip(),
            "original_name": str(item.get("document_name", "")).strip(),
            "stored_name": str(item.get("document_name", "")).strip(),
            "stored_path": "",
            "source_path": "",
            "pages": int(item.get("page_count") or 0),
            "uploaded_at": now,
        }
        for item in source_manifest
        if isinstance(item, dict)
    ]

    relation_pairs_total = max(0, len(biomes) * (len(biomes) - 1))
    relationship_map = {
        "created_at": graph.get("created_at") or now,
        "updated_at": now,
        "status": "complete",
        "communities": communities,
        "biomes": biomes,
        "biome_relations": biome_relations,
        "relation_pairs_done": relation_pairs_total,
        "relation_pairs_total": relation_pairs_total,
        "processed_relation_pairs": [],
        "failed_relation_pairs": {},
        "relation_audit": {},
        "verified_empty_relation_pairs": [],
        "source": {
            "type": "evidence_mesh_adapter",
            "source_project": str(source_root),
            "schema_version": graph.get("schema_version", ""),
        },
    }

    project = read_json(target_root / "project.json", {})
    project.update(
        {
            "name": project_name,
            "updated_at": now,
            "document_count": len(manifest),
            "topic_count": len(topics),
            "community_count": len(communities),
            "biome_count": len(biomes),
            "relation_count": len(biome_relations),
        }
    )

    write_json(target_root / "documents" / "manifest.json", manifest)
    write_json(target_root / "indexes" / "topic_index.json", topics)
    write_json(target_root / "indexes" / "relationship_map.json", relationship_map)
    write_json(target_root / "project.json", project)
    write_json(
        target_root / "logs" / "pipeline_progress.json",
        {
            "updated_at": now,
            "stage": "complete",
            "message": "Index and relationship map imported from Evidence Mesh.",
            "document_count": len(manifest),
            "topic_count": len(topics),
            "community_count": len(communities),
            "biome_count": len(biomes),
            "relation_count": len(biome_relations),
            "indexed_pages": sum(int(item.get("pages") or 0) for item in manifest),
            "total_pages": sum(int(item.get("pages") or 0) for item in manifest),
        },
    )
    return project


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert core Evidence Mesh data into the tender generator project format.")
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--target-root", required=True)
    parser.add_argument("--project-name", required=True)
    args = parser.parse_args()
    project = convert(Path(args.source_root).resolve(), Path(args.target_root).resolve(), args.project_name)
    print(json.dumps(project, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
