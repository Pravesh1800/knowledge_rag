from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


LEGACY_ROOT_ENV = "PDF" + "_VISION_RAG_ROOT"
PROJECT_ROOT = Path(
    os.getenv("EVIDENCE_MESH_ROOT")
    or os.getenv(LEGACY_ROOT_ENV)
    or Path(__file__).resolve().parent
).resolve()
SCHEMA_PATH = Path(__file__).resolve().parent / "postgres_schema.sql"

_WARNED_AUTOMATIC_SYNC = False
_WARNED_IMPORT = False


def load_dotenv() -> None:
    env_paths = [PROJECT_ROOT / ".env", Path(__file__).resolve().parent / ".env"]
    for env_path in env_paths:
        if not env_path.exists():
            continue
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def database_url() -> str:
    load_dotenv()
    return os.getenv("DATABASE_URL", "").strip()


def storage_mode() -> str:
    load_dotenv()
    return os.getenv("EVIDENCE_MESH_STORAGE", "postgres").strip().lower() or "postgres"


def project_id() -> str:
    return os.getenv("EVIDENCE_MESH_PROJECT_ID") or PROJECT_ROOT.name


def postgres_enabled() -> bool:
    return bool(database_url())


def postgres_required() -> bool:
    return storage_mode() != "json"


def _stable_hash(payload: Any) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _jsonb(value: Any) -> Any:
    from psycopg.types.json import Jsonb

    return Jsonb(value)


def _get_psycopg() -> Any | None:
    global _WARNED_IMPORT
    try:
        import psycopg
    except ImportError:
        if not _WARNED_IMPORT:
            print(
                "PostgreSQL driver missing: install psycopg or run "
                "python -m pip install -r requirements.txt."
            )
            _WARNED_IMPORT = True
        return None
    return psycopg


def _connect() -> Any:
    psycopg = _get_psycopg()
    if psycopg is None:
        raise RuntimeError("psycopg is not installed.")
    url = database_url()
    if not url:
        raise RuntimeError("DATABASE_URL is not configured.")
    return psycopg.connect(url, connect_timeout=int(os.getenv("POSTGRES_CONNECT_TIMEOUT", "5")))


def _warn_sync_error(exc: Exception) -> None:
    global _WARNED_AUTOMATIC_SYNC
    if _WARNED_AUTOMATIC_SYNC:
        return
    print(f"PostgreSQL sync unavailable after error: {exc}")
    _WARNED_AUTOMATIC_SYNC = True


def _automatic_sync(
    operation: Callable[[Any, str], None],
    storage_project_id: str | None = None,
) -> None:
    if not postgres_enabled():
        if postgres_required():
            raise RuntimeError("DATABASE_URL is required because EVIDENCE_MESH_STORAGE defaults to postgres.")
        return
    try:
        with _connect() as conn:
            operation(conn, storage_project_id or project_id())
    except Exception as exc:
        if postgres_required():
            raise
        _warn_sync_error(exc)


def ensure_postgres_ready() -> None:
    if not postgres_enabled():
        raise RuntimeError("DATABASE_URL is required because PostgreSQL is the primary storage.")
    init_postgres_schema()


def read_cards(storage_project_id: str | None = None) -> list[dict[str, Any]]:
    if not postgres_required() and not postgres_enabled():
        return []
    with _connect() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT payload
                FROM cards
                WHERE project_id = %s
                ORDER BY document_name, page_no NULLS LAST, card_name, card_id
                """,
                (storage_project_id or project_id(),),
            )
            return [payload for (payload,) in cursor.fetchall() if isinstance(payload, dict)]


def storage_counts(storage_project_id: str | None = None) -> dict[str, int]:
    pid = storage_project_id or project_id()
    with _connect() as conn:
        with conn.cursor() as cursor:
            counts: dict[str, int] = {}
            for table in (
                "documents",
                "pages",
                "cards",
                "clusters",
                "domains",
                "relationships",
                "relationship_pair_checks",
                "search_runs",
                "graph_build_runs",
                "project_progress",
                "knowledge_graph_state",
                "graph_audits",
            ):
                cursor.execute(f"SELECT count(*) FROM {table} WHERE project_id = %s", (pid,))
                counts[table] = int(cursor.fetchone()[0])
            return counts


def indexed_page_keys(storage_project_id: str | None = None) -> set[tuple[str, int]]:
    if not postgres_required() and not postgres_enabled():
        return set()
    with _connect() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                "SELECT document_id, page_no FROM pages WHERE project_id = %s",
                (storage_project_id or project_id(),),
            )
            return {(str(document_id), int(page_no)) for document_id, page_no in cursor.fetchall()}


def clear_cards(storage_project_id: str | None = None) -> None:
    def operation(conn: Any, pid: str) -> None:
        with conn.cursor() as cursor:
            cursor.execute("DELETE FROM cards WHERE project_id = %s", (pid,))
            cursor.execute("DELETE FROM pages WHERE project_id = %s", (pid,))

    _automatic_sync(operation, storage_project_id)


def prune_cards_to_documents(valid_document_ids: set[str], storage_project_id: str | None = None) -> int:
    with _connect() as conn:
        pid = storage_project_id or project_id()
        with conn.cursor() as cursor:
            if valid_document_ids:
                cursor.execute(
                    "DELETE FROM cards WHERE project_id = %s AND NOT (document_id = ANY(%s))",
                    (pid, list(valid_document_ids)),
                )
                removed_cards = cursor.rowcount
                cursor.execute(
                    "DELETE FROM pages WHERE project_id = %s AND NOT (document_id = ANY(%s))",
                    (pid, list(valid_document_ids)),
                )
            else:
                cursor.execute("DELETE FROM cards WHERE project_id = %s", (pid,))
                removed_cards = cursor.rowcount
                cursor.execute("DELETE FROM pages WHERE project_id = %s", (pid,))
        return int(removed_cards or 0)


def replace_page_cards(
    cards: list[dict[str, Any]],
    document_id: str,
    page_no: int,
    document_name: str,
    storage_project_id: str | None = None,
) -> None:
    def operation(conn: Any, pid: str) -> None:
        with conn.cursor() as cursor:
            cursor.execute(
                "DELETE FROM cards WHERE project_id = %s AND document_id = %s AND page_no = %s",
                (pid, document_id, page_no),
            )
            cursor.execute(
                """
                INSERT INTO pages (
                  project_id, document_id, page_no, document_name,
                  card_count, payload, updated_at
                )
                VALUES (%s, %s, %s, %s, %s, %s, now())
                ON CONFLICT (project_id, document_id, page_no) DO UPDATE SET
                  document_name = EXCLUDED.document_name,
                  card_count = EXCLUDED.card_count,
                  payload = EXCLUDED.payload,
                  updated_at = now()
                """,
                (
                    pid,
                    document_id,
                    page_no,
                    document_name,
                    len(cards),
                    _jsonb(
                        {
                            "document_id": document_id,
                            "document_name": document_name,
                            "page_no": page_no,
                            "card_ids": [str(card.get("card_id", "")) for card in cards],
                            "card_names": [str(card.get("card_name", "")) for card in cards],
                        }
                    ),
                ),
            )
            for card in cards:
                card_id = str(card.get("card_id", ""))
                if not card_id:
                    continue
                cursor.execute(
                    """
                    INSERT INTO cards (
                      project_id, card_id, document_id, document_name, page_no,
                      card_name, card_description, card_source, tags, schema_version,
                      payload, updated_at
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, now())
                    ON CONFLICT (project_id, card_id) DO UPDATE SET
                      document_id = EXCLUDED.document_id,
                      document_name = EXCLUDED.document_name,
                      page_no = EXCLUDED.page_no,
                      card_name = EXCLUDED.card_name,
                      card_description = EXCLUDED.card_description,
                      card_source = EXCLUDED.card_source,
                      tags = EXCLUDED.tags,
                      schema_version = EXCLUDED.schema_version,
                      payload = EXCLUDED.payload,
                      updated_at = now()
                    """,
                    (
                        pid,
                        card_id,
                        document_id,
                        document_name,
                        page_no,
                        str(card.get("card_name", "")),
                        card.get("card_description", ""),
                        card.get("card_source", ""),
                        [str(tag) for tag in card.get("tags", []) or []],
                        card.get("schema_version", ""),
                        _jsonb(card),
                    ),
                )

    _automatic_sync(operation, storage_project_id)


def init_postgres_schema() -> None:
    if not SCHEMA_PATH.exists():
        raise FileNotFoundError(f"Missing PostgreSQL schema file: {SCHEMA_PATH}")
    with _connect() as conn:
        with conn.cursor() as cursor:
            cursor.execute(SCHEMA_PATH.read_text(encoding="utf-8"))


def _parse_datetime(value: Any) -> Any:
    if not value:
        return None
    if isinstance(value, datetime):
        return value
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _delete_missing(conn: Any, pid: str, table: str, id_column: str, keep_ids: list[str]) -> None:
    with conn.cursor() as cursor:
        if keep_ids:
            cursor.execute(
                f"DELETE FROM {table} WHERE project_id = %s AND NOT ({id_column} = ANY(%s))",
                (pid, keep_ids),
            )
        else:
            cursor.execute(f"DELETE FROM {table} WHERE project_id = %s", (pid,))


def sync_documents(records: list[dict[str, Any]], storage_project_id: str | None = None) -> None:
    def operation(conn: Any, pid: str) -> None:
        keep_ids = [str(record.get("document_id", "")) for record in records if record.get("document_id")]
        with conn.cursor() as cursor:
            for record in records:
                document_id = str(record.get("document_id", ""))
                if not document_id:
                    continue
                cursor.execute(
                    """
                    INSERT INTO documents (
                      project_id, document_id, original_name, stored_path, source_path,
                      extension, mime_type, size_bytes, sha256, ingested_at,
                      ingest_strategy, payload, updated_at
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, now())
                    ON CONFLICT (project_id, document_id) DO UPDATE SET
                      original_name = EXCLUDED.original_name,
                      stored_path = EXCLUDED.stored_path,
                      source_path = EXCLUDED.source_path,
                      extension = EXCLUDED.extension,
                      mime_type = EXCLUDED.mime_type,
                      size_bytes = EXCLUDED.size_bytes,
                      sha256 = EXCLUDED.sha256,
                      ingested_at = EXCLUDED.ingested_at,
                      ingest_strategy = EXCLUDED.ingest_strategy,
                      payload = EXCLUDED.payload,
                      updated_at = now()
                    """,
                    (
                        pid,
                        document_id,
                        str(record.get("original_name") or record.get("document_name") or ""),
                        record.get("stored_path"),
                        record.get("source_path"),
                        record.get("extension"),
                        record.get("mime_type"),
                        _as_int(record.get("size_bytes"), 0),
                        record.get("sha256"),
                        _parse_datetime(record.get("ingested_at")),
                        record.get("ingest_strategy"),
                        _jsonb(record),
                    ),
                )
        _delete_missing(conn, pid, "documents", "document_id", keep_ids)

    _automatic_sync(operation, storage_project_id)


def sync_cards(cards: list[dict[str, Any]], storage_project_id: str | None = None) -> None:
    def operation(conn: Any, pid: str) -> None:
        keep_card_ids = [str(card.get("card_id", "")) for card in cards if card.get("card_id")]
        pages: dict[tuple[str, int], dict[str, Any]] = {}
        with conn.cursor() as cursor:
            for card in cards:
                card_id = str(card.get("card_id", ""))
                document_id = str(card.get("document_id", ""))
                if not card_id or not document_id:
                    continue
                page_no = _as_int(card.get("page_no"), 0)
                page_key = (document_id, page_no)
                page_payload = pages.setdefault(
                    page_key,
                    {
                        "document_id": document_id,
                        "document_name": card.get("document_name", ""),
                        "page_no": page_no,
                        "card_ids": [],
                        "card_names": [],
                    },
                )
                page_payload["card_ids"].append(card_id)
                page_payload["card_names"].append(card.get("card_name", ""))
                cursor.execute(
                    """
                    INSERT INTO cards (
                      project_id, card_id, document_id, document_name, page_no,
                      card_name, card_description, card_source, tags, schema_version,
                      payload, updated_at
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, now())
                    ON CONFLICT (project_id, card_id) DO UPDATE SET
                      document_id = EXCLUDED.document_id,
                      document_name = EXCLUDED.document_name,
                      page_no = EXCLUDED.page_no,
                      card_name = EXCLUDED.card_name,
                      card_description = EXCLUDED.card_description,
                      card_source = EXCLUDED.card_source,
                      tags = EXCLUDED.tags,
                      schema_version = EXCLUDED.schema_version,
                      payload = EXCLUDED.payload,
                      updated_at = now()
                    """,
                    (
                        pid,
                        card_id,
                        document_id,
                        card.get("document_name", ""),
                        page_no,
                        str(card.get("card_name", "")),
                        card.get("card_description", ""),
                        card.get("card_source", ""),
                        [str(tag) for tag in card.get("tags", []) or []],
                        card.get("schema_version", ""),
                        _jsonb(card),
                    ),
                )

            keep_pages = []
            for (document_id, page_no), page_payload in pages.items():
                keep_pages.append((document_id, page_no))
                cursor.execute(
                    """
                    INSERT INTO pages (
                      project_id, document_id, page_no, document_name,
                      card_count, payload, updated_at
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, now())
                    ON CONFLICT (project_id, document_id, page_no) DO UPDATE SET
                      document_name = EXCLUDED.document_name,
                      card_count = EXCLUDED.card_count,
                      payload = EXCLUDED.payload,
                      updated_at = now()
                    """,
                    (
                        pid,
                        document_id,
                        page_no,
                        page_payload.get("document_name", ""),
                        len(page_payload.get("card_ids", [])),
                        _jsonb(page_payload),
                    ),
                )

            if keep_card_ids:
                cursor.execute("DELETE FROM cards WHERE project_id = %s AND NOT (card_id = ANY(%s))", (pid, keep_card_ids))
            else:
                cursor.execute("DELETE FROM cards WHERE project_id = %s", (pid,))
            cursor.execute("SELECT document_id, page_no FROM pages WHERE project_id = %s", (pid,))
            for document_id, page_no in cursor.fetchall():
                if (str(document_id), int(page_no)) not in keep_pages:
                    cursor.execute(
                        "DELETE FROM pages WHERE project_id = %s AND document_id = %s AND page_no = %s",
                        (pid, document_id, page_no),
                    )

    _automatic_sync(operation, storage_project_id)


def sync_knowledge_graph(graph: dict[str, Any], storage_project_id: str | None = None) -> None:
    def operation(conn: Any, pid: str) -> None:
        schema_version = str(graph.get("schema_version", ""))
        clusters = graph.get("clusters", []) or []
        domains = graph.get("domains", []) or []
        relationships = graph.get("domain_relationships", []) or []
        keep_cluster_ids = [str(item.get("cluster_id", "")) for item in clusters if item.get("cluster_id")]
        keep_domain_ids = [str(item.get("domain_id", "")) for item in domains if item.get("domain_id")]
        keep_relationship_ids = [
            str(item.get("relationship_id", ""))
            for item in relationships
            if item.get("relationship_id")
        ]
        with conn.cursor() as cursor:
            for cluster in clusters:
                cluster_id = str(cluster.get("cluster_id", ""))
                if not cluster_id:
                    continue
                cursor.execute(
                    """
                    INSERT INTO clusters (
                      project_id, cluster_id, document_id, document_name,
                      cluster_name, cluster_description, card_ids,
                      schema_version, payload, updated_at
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, now())
                    ON CONFLICT (project_id, cluster_id) DO UPDATE SET
                      document_id = EXCLUDED.document_id,
                      document_name = EXCLUDED.document_name,
                      cluster_name = EXCLUDED.cluster_name,
                      cluster_description = EXCLUDED.cluster_description,
                      card_ids = EXCLUDED.card_ids,
                      schema_version = EXCLUDED.schema_version,
                      payload = EXCLUDED.payload,
                      updated_at = now()
                    """,
                    (
                        pid,
                        cluster_id,
                        str(cluster.get("document_id", "")),
                        cluster.get("document_name", ""),
                        str(cluster.get("cluster_name", "")),
                        cluster.get("cluster_description", ""),
                        [str(card_id) for card_id in cluster.get("card_ids", []) or []],
                        schema_version,
                        _jsonb(cluster),
                    ),
                )

            for domain in domains:
                domain_id = str(domain.get("domain_id", ""))
                if not domain_id:
                    continue
                cursor.execute(
                    """
                    INSERT INTO domains (
                      project_id, domain_id, document_id, document_name,
                      domain_name, domain_description, cluster_ids,
                      schema_version, payload, updated_at
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, now())
                    ON CONFLICT (project_id, domain_id) DO UPDATE SET
                      document_id = EXCLUDED.document_id,
                      document_name = EXCLUDED.document_name,
                      domain_name = EXCLUDED.domain_name,
                      domain_description = EXCLUDED.domain_description,
                      cluster_ids = EXCLUDED.cluster_ids,
                      schema_version = EXCLUDED.schema_version,
                      payload = EXCLUDED.payload,
                      updated_at = now()
                    """,
                    (
                        pid,
                        domain_id,
                        str(domain.get("document_id", "")),
                        domain.get("document_name", ""),
                        str(domain.get("domain_name", "")),
                        domain.get("domain_description", ""),
                        [str(cluster_id) for cluster_id in domain.get("cluster_ids", []) or []],
                        schema_version,
                        _jsonb(domain),
                    ),
                )

            for relationship in relationships:
                relationship_id = str(relationship.get("relationship_id", ""))
                if not relationship_id:
                    continue
                cursor.execute(
                    """
                    INSERT INTO relationships (
                      project_id, relationship_id, main_domain_id, related_domain_id,
                      relationship_type, document_scope, confidence_score,
                      evidence_strength, source_coverage, generation_method,
                      schema_version, payload, updated_at
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, now())
                    ON CONFLICT (project_id, relationship_id) DO UPDATE SET
                      main_domain_id = EXCLUDED.main_domain_id,
                      related_domain_id = EXCLUDED.related_domain_id,
                      relationship_type = EXCLUDED.relationship_type,
                      document_scope = EXCLUDED.document_scope,
                      confidence_score = EXCLUDED.confidence_score,
                      evidence_strength = EXCLUDED.evidence_strength,
                      source_coverage = EXCLUDED.source_coverage,
                      generation_method = EXCLUDED.generation_method,
                      schema_version = EXCLUDED.schema_version,
                      payload = EXCLUDED.payload,
                      updated_at = now()
                    """,
                    (
                        pid,
                        relationship_id,
                        str(relationship.get("main_domain_id", "")),
                        str(relationship.get("related_domain_id", "")),
                        relationship.get("relationship_type", ""),
                        relationship.get("document_scope", ""),
                        _as_float(relationship.get("confidence_score")),
                        _as_float(relationship.get("evidence_strength")),
                        _as_float(relationship.get("source_coverage")),
                        relationship.get("generation_method", ""),
                        schema_version,
                        _jsonb(relationship),
                    ),
                )

        _delete_missing(conn, pid, "clusters", "cluster_id", keep_cluster_ids)
        _delete_missing(conn, pid, "domains", "domain_id", keep_domain_ids)
        _delete_missing(conn, pid, "relationships", "relationship_id", keep_relationship_ids)

    _automatic_sync(operation, storage_project_id)


def upsert_relationship_payloads(relationships: list[dict[str, Any]], storage_project_id: str | None = None) -> None:
    def operation(conn: Any, pid: str) -> None:
        schema_version = ""
        with conn.cursor() as cursor:
            for relationship in relationships:
                relationship_id = str(relationship.get("relationship_id", ""))
                if not relationship_id:
                    continue
                schema_version = str(relationship.get("schema_version") or schema_version)
                cursor.execute(
                    """
                    INSERT INTO relationships (
                      project_id, relationship_id, main_domain_id, related_domain_id,
                      relationship_type, document_scope, confidence_score,
                      evidence_strength, source_coverage, generation_method,
                      schema_version, payload, updated_at
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, now())
                    ON CONFLICT (project_id, relationship_id) DO UPDATE SET
                      main_domain_id = EXCLUDED.main_domain_id,
                      related_domain_id = EXCLUDED.related_domain_id,
                      relationship_type = EXCLUDED.relationship_type,
                      document_scope = EXCLUDED.document_scope,
                      confidence_score = EXCLUDED.confidence_score,
                      evidence_strength = EXCLUDED.evidence_strength,
                      source_coverage = EXCLUDED.source_coverage,
                      generation_method = EXCLUDED.generation_method,
                      schema_version = EXCLUDED.schema_version,
                      payload = EXCLUDED.payload,
                      updated_at = now()
                    """,
                    (
                        pid,
                        relationship_id,
                        str(relationship.get("main_domain_id", "")),
                        str(relationship.get("related_domain_id", "")),
                        relationship.get("relationship_type", ""),
                        relationship.get("document_scope", ""),
                        _as_float(relationship.get("confidence_score")),
                        _as_float(relationship.get("evidence_strength")),
                        _as_float(relationship.get("source_coverage")),
                        relationship.get("generation_method", ""),
                        schema_version,
                        _jsonb(relationship),
                    ),
                )

    _automatic_sync(operation, storage_project_id)


def record_relationship_pair_check(
    *,
    pair_key: str,
    main_domain: str,
    related_domain: str,
    status: str,
    relationships: list[dict[str, Any]] | None = None,
    error: str = "",
    storage_project_id: str | None = None,
) -> None:
    relationships = relationships or []
    relationship_ids = [str(item.get("relationship_id", "")) for item in relationships if item.get("relationship_id")]

    def operation(conn: Any, pid: str) -> None:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO relationship_pair_checks (
                  project_id, pair_key, main_domain, related_domain,
                  status, relationship_ids, error, payload, updated_at
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, now())
                ON CONFLICT (project_id, pair_key) DO UPDATE SET
                  main_domain = EXCLUDED.main_domain,
                  related_domain = EXCLUDED.related_domain,
                  status = EXCLUDED.status,
                  relationship_ids = EXCLUDED.relationship_ids,
                  error = EXCLUDED.error,
                  payload = EXCLUDED.payload,
                  updated_at = now()
                """,
                (
                    pid,
                    pair_key,
                    main_domain,
                    related_domain,
                    status,
                    relationship_ids,
                    error,
                    _jsonb(
                        {
                            "pair_key": pair_key,
                            "main_domain": main_domain,
                            "related_domain": related_domain,
                            "status": status,
                            "relationship_ids": relationship_ids,
                            "error": error,
                            "relationships": relationships,
                        }
                    ),
                ),
            )

    if relationships:
        upsert_relationship_payloads(relationships, storage_project_id)
    _automatic_sync(operation, storage_project_id)


def read_relationship_pair_checks(storage_project_id: str | None = None) -> list[dict[str, Any]]:
    with _connect() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT payload
                FROM relationship_pair_checks
                WHERE project_id = %s
                ORDER BY updated_at, pair_key
                """,
                (storage_project_id or project_id(),),
            )
            return [payload for (payload,) in cursor.fetchall() if isinstance(payload, dict)]


def read_relationship_payloads(storage_project_id: str | None = None) -> list[dict[str, Any]]:
    with _connect() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT payload
                FROM relationships
                WHERE project_id = %s
                ORDER BY relationship_id
                """,
                (storage_project_id or project_id(),),
            )
            return [payload for (payload,) in cursor.fetchall() if isinstance(payload, dict)]


def write_pipeline_progress_state(progress: dict[str, Any], storage_project_id: str | None = None) -> None:
    def operation(conn: Any, pid: str) -> None:
        payload = dict(progress)
        with conn.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO project_progress (
                  project_id, stage, message, payload, updated_at
                )
                VALUES (%s, %s, %s, %s, now())
                ON CONFLICT (project_id) DO UPDATE SET
                  stage = EXCLUDED.stage,
                  message = EXCLUDED.message,
                  payload = EXCLUDED.payload,
                  updated_at = now()
                """,
                (
                    pid,
                    str(payload.get("stage") or "unknown"),
                    str(payload.get("message") or ""),
                    _jsonb(payload),
                ),
            )

    _automatic_sync(operation, storage_project_id)


def read_pipeline_progress_state(storage_project_id: str | None = None) -> dict[str, Any]:
    with _connect() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                "SELECT payload, updated_at FROM project_progress WHERE project_id = %s",
                (storage_project_id or project_id(),),
            )
            row = cursor.fetchone()
            if not row:
                return {}
            payload, updated_at = row
            if not isinstance(payload, dict):
                return {}
            result = dict(payload)
            result.setdefault("updated_at", updated_at.isoformat() if updated_at else _utc_now())
            return result


def write_knowledge_graph_state(graph: dict[str, Any], storage_project_id: str | None = None) -> None:
    def operation(conn: Any, pid: str) -> None:
        payload = dict(graph)
        failed_relationship_checks = payload.get("failed_relationship_checks") or {}
        with conn.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO knowledge_graph_state (
                  project_id, status, cluster_count, domain_count,
                  relationship_count, relationship_checks_done,
                  relationship_checks_total, failed_relationship_check_count,
                  schema_version, payload, updated_at
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, now())
                ON CONFLICT (project_id) DO UPDATE SET
                  status = EXCLUDED.status,
                  cluster_count = EXCLUDED.cluster_count,
                  domain_count = EXCLUDED.domain_count,
                  relationship_count = EXCLUDED.relationship_count,
                  relationship_checks_done = EXCLUDED.relationship_checks_done,
                  relationship_checks_total = EXCLUDED.relationship_checks_total,
                  failed_relationship_check_count = EXCLUDED.failed_relationship_check_count,
                  schema_version = EXCLUDED.schema_version,
                  payload = EXCLUDED.payload,
                  updated_at = now()
                """,
                (
                    pid,
                    str(payload.get("status") or "unknown"),
                    len(payload.get("clusters", []) or []),
                    len(payload.get("domains", []) or []),
                    len(payload.get("domain_relationships", []) or []),
                    _as_int(payload.get("relationship_checks_done"), 0),
                    _as_int(payload.get("relationship_checks_total"), 0),
                    len(failed_relationship_checks) if isinstance(failed_relationship_checks, dict) else 0,
                    str(payload.get("schema_version", "")),
                    _jsonb(payload),
                ),
            )

    _automatic_sync(operation, storage_project_id)


def read_knowledge_graph_state(storage_project_id: str | None = None) -> dict[str, Any]:
    with _connect() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                "SELECT payload, updated_at FROM knowledge_graph_state WHERE project_id = %s",
                (storage_project_id or project_id(),),
            )
            row = cursor.fetchone()
            if not row:
                return {}
            payload, updated_at = row
            if not isinstance(payload, dict):
                return {}
            result = dict(payload)
            result.setdefault("updated_at", updated_at.isoformat() if updated_at else _utc_now())
            return result


def write_graph_audit_state(audit: dict[str, Any], storage_project_id: str | None = None) -> None:
    def operation(conn: Any, pid: str) -> None:
        payload = dict(audit)
        with conn.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO graph_audits (
                  project_id, status, issue_count, schema_version, payload, updated_at
                )
                VALUES (%s, %s, %s, %s, %s, now())
                ON CONFLICT (project_id) DO UPDATE SET
                  status = EXCLUDED.status,
                  issue_count = EXCLUDED.issue_count,
                  schema_version = EXCLUDED.schema_version,
                  payload = EXCLUDED.payload,
                  updated_at = now()
                """,
                (
                    pid,
                    str(payload.get("status") or "unknown"),
                    _as_int(payload.get("issue_count"), 0),
                    str(payload.get("schema_version", "")),
                    _jsonb(payload),
                ),
            )

    _automatic_sync(operation, storage_project_id)


def read_graph_audit_state(storage_project_id: str | None = None) -> dict[str, Any]:
    with _connect() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                "SELECT payload, updated_at FROM graph_audits WHERE project_id = %s",
                (storage_project_id or project_id(),),
            )
            row = cursor.fetchone()
            if not row:
                return {}
            payload, updated_at = row
            if not isinstance(payload, dict):
                return {}
            result = dict(payload)
            result.setdefault("updated_at", updated_at.isoformat() if updated_at else _utc_now())
            return result


def record_search_run(result: dict[str, Any], storage_project_id: str | None = None) -> None:
    def operation(conn: Any, pid: str) -> None:
        run_id = result.get("search_run_id") or f"search_{_stable_hash([result.get('created_at'), result.get('query'), result.get('hits')])}"
        with conn.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO search_runs (
                  project_id, search_run_id, query, hit_count, schema_version, payload
                )
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (project_id, search_run_id) DO UPDATE SET
                  query = EXCLUDED.query,
                  hit_count = EXCLUDED.hit_count,
                  schema_version = EXCLUDED.schema_version,
                  payload = EXCLUDED.payload
                """,
                (
                    pid,
                    str(run_id),
                    str(result.get("query", "")),
                    len(result.get("hits", []) or []),
                    result.get("schema_version", ""),
                    _jsonb(result),
                ),
            )

    _automatic_sync(operation, storage_project_id)


def record_graph_build_run(
    graph: dict[str, Any],
    audit: dict[str, Any] | None = None,
    storage_project_id: str | None = None,
) -> None:
    audit = audit or {}

    def operation(conn: Any, pid: str) -> None:
        run_id = f"graph_{_stable_hash([graph.get('created_at'), graph.get('updated_at'), graph.get('status'), graph.get('relationship_checks_done')])}"
        payload = {"graph": graph, "audit": audit}
        with conn.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO graph_build_runs (
                  project_id, graph_build_run_id, status, cluster_count,
                  domain_count, relationship_count, audit_status,
                  audit_issue_count, schema_version, payload
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (project_id, graph_build_run_id) DO UPDATE SET
                  status = EXCLUDED.status,
                  cluster_count = EXCLUDED.cluster_count,
                  domain_count = EXCLUDED.domain_count,
                  relationship_count = EXCLUDED.relationship_count,
                  audit_status = EXCLUDED.audit_status,
                  audit_issue_count = EXCLUDED.audit_issue_count,
                  schema_version = EXCLUDED.schema_version,
                  payload = EXCLUDED.payload
                """,
                (
                    pid,
                    run_id,
                    str(graph.get("status", "")),
                    len(graph.get("clusters", []) or []),
                    len(graph.get("domains", []) or []),
                    len(graph.get("domain_relationships", []) or []),
                    audit.get("status", ""),
                    _as_int(audit.get("issue_count"), 0),
                    graph.get("schema_version", ""),
                    _jsonb(payload),
                ),
            )

    _automatic_sync(operation, storage_project_id)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Evidence Mesh storage utilities.")
    parser.add_argument(
        "command",
        choices=["init-postgres", "status", "migrate-card-index", "migrate-runtime-state"],
        help="Initialize PostgreSQL tables or print storage status.",
    )
    parser.add_argument("--project-id", help="Project id to inspect or migrate.")
    parser.add_argument("--card-index", type=Path, help="Path to an existing card_index.json export.")
    args = parser.parse_args()
    if args.command == "init-postgres":
        init_postgres_schema()
        print("PostgreSQL schema is ready.")
    elif args.command == "migrate-card-index":
        from schema import read_card_index

        card_index_path = args.card_index or PROJECT_ROOT / "indexes" / "card_index.json"
        cards = read_card_index(card_index_path, persist_migration=True)
        sync_cards(cards, args.project_id)
        print(f"Migrated {len(cards)} card(s) from {card_index_path} into PostgreSQL.")
    elif args.command == "migrate-runtime-state":
        graph_path = PROJECT_ROOT / "indexes" / "knowledge_graph.json"
        progress_path = PROJECT_ROOT / "logs" / "pipeline_progress.json"
        audit_path = PROJECT_ROOT / "indexes" / "graph_audit.json"
        migrated = []
        if graph_path.exists():
            graph = json.loads(graph_path.read_text(encoding="utf-8-sig"))
            sync_knowledge_graph(graph, args.project_id)
            write_knowledge_graph_state(graph, args.project_id)
            migrated.append("knowledge_graph_state")
        if progress_path.exists():
            progress = json.loads(progress_path.read_text(encoding="utf-8-sig"))
            write_pipeline_progress_state(progress, args.project_id)
            migrated.append("project_progress")
        if audit_path.exists():
            audit = json.loads(audit_path.read_text(encoding="utf-8-sig"))
            write_graph_audit_state(audit, args.project_id)
            migrated.append("graph_audits")
        print(f"Migrated runtime state: {', '.join(migrated) if migrated else 'nothing found'}.")
    elif args.command == "status":
        print(f"PROJECT_ROOT={PROJECT_ROOT}")
        pid = args.project_id or project_id()
        print(f"EVIDENCE_MESH_PROJECT_ID={pid}")
        print(f"EVIDENCE_MESH_STORAGE={storage_mode()}")
        print(f"DATABASE_URL configured={postgres_enabled()}")
        try:
            counts = storage_counts(pid)
        except Exception as exc:
            print(f"PostgreSQL reachable=False")
            print(f"PostgreSQL error={exc}")
        else:
            print("PostgreSQL reachable=True")
            for table, count in counts.items():
                print(f"{table}={count}")


if __name__ == "__main__":
    main()
