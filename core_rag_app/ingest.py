from __future__ import annotations

import argparse
import hashlib
import json
import mimetypes
import os
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


LEGACY_ROOT_ENV = "PDF" + "_VISION_RAG_ROOT"
PROJECT_ROOT = Path(
    os.getenv("EVIDENCE_MESH_ROOT")
    or os.getenv(LEGACY_ROOT_ENV)
    or Path(__file__).resolve().parent
).resolve()
DOCUMENTS_DIR = PROJECT_ROOT / "documents"
ORIGINALS_DIR = DOCUMENTS_DIR / "originals"
MANIFEST_PATH = DOCUMENTS_DIR / "manifest.json"


SUPPORTED_EXTENSIONS = {
    ".txt",
    ".md",
    ".markdown",
    ".pdf",
    ".doc",
    ".docx",
    ".rtf",
    ".odt",
    ".ppt",
    ".pptx",
    ".xls",
    ".xlsx",
    ".xlsm",
    ".csv",
    ".tsv",
    ".html",
    ".htm",
    ".json",
    ".jsonl",
    ".xml",
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".tif",
    ".tiff",
    ".bmp",
}


@dataclass(frozen=True)
class DocumentRecord:
    document_id: str
    original_name: str
    stored_path: str
    source_path: str
    extension: str
    mime_type: str
    size_bytes: int
    sha256: str
    ingested_at: str
    ingest_strategy: str


def ensure_dirs() -> None:
    ORIGINALS_DIR.mkdir(parents=True, exist_ok=True)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest() -> list[dict]:
    if not MANIFEST_PATH.exists():
        return []
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8-sig"))


def save_manifest(records: list[dict]) -> None:
    MANIFEST_PATH.write_text(
        json.dumps(records, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def iter_files(path: Path) -> Iterable[Path]:
    if path.is_file():
        yield path
        return

    if path.is_dir():
        for child in path.rglob("*"):
            if child.is_file():
                yield child
        return

    raise FileNotFoundError(f"Path does not exist: {path}")


def guess_mime_type(path: Path) -> str:
    mime_type, _ = mimetypes.guess_type(path.name)
    return mime_type or "application/octet-stream"


def choose_ingest_strategy(extension: str, mime_type: str) -> str:
    if extension in {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".bmp"}:
        return "gemini_vision_image"
    if extension == ".pdf":
        return "gemini_file_pdf"
    if extension in {".doc", ".docx", ".rtf", ".odt"}:
        return "gemini_file_document"
    if extension in {".ppt", ".pptx", ".xls", ".xlsx"}:
        return "gemini_file_office"
    if mime_type.startswith("text/") or extension in {
        ".txt",
        ".md",
        ".markdown",
        ".csv",
        ".tsv",
        ".json",
        ".jsonl",
        ".xml",
    }:
        return "direct_text_or_gemini_file"
    return "gemini_file_generic"


def safe_stored_name(document_id: str, path: Path) -> str:
    cleaned_stem = "".join(
        char if char.isalnum() or char in {"-", "_"} else "_"
        for char in path.stem
    ).strip("_")
    if not cleaned_stem:
        cleaned_stem = "document"
    return f"{cleaned_stem}_{document_id[:12]}{path.suffix.lower()}"


def ingest_file(path: Path, existing_hashes: set[str]) -> DocumentRecord | None:
    resolved = path.resolve()
    if resolved.name.startswith("~$"):
        print(f"Skipped temporary Office file: {resolved}")
        return None
    extension = resolved.suffix.lower()
    sha256 = file_sha256(resolved)

    if sha256 in existing_hashes:
        print(f"Skipped duplicate: {resolved}")
        return None

    mime_type = guess_mime_type(resolved)
    document_id = sha256
    stored_name = safe_stored_name(document_id, resolved)
    stored_path = ORIGINALS_DIR / stored_name

    shutil.copy2(resolved, stored_path)

    return DocumentRecord(
        document_id=document_id,
        original_name=resolved.name,
        stored_path=str(stored_path.relative_to(PROJECT_ROOT)),
        source_path=str(resolved),
        extension=extension,
        mime_type=mime_type,
        size_bytes=stored_path.stat().st_size,
        sha256=sha256,
        ingested_at=datetime.now(timezone.utc).isoformat(),
        ingest_strategy=choose_ingest_strategy(extension, mime_type),
    )


def add_path(raw_path: str) -> None:
    ensure_dirs()
    manifest = load_manifest()
    existing_hashes = {record["sha256"] for record in manifest}
    new_records: list[DocumentRecord] = []

    for file_path in iter_files(Path(raw_path)):
        if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            print(f"Accepted generic file via Gemini fallback: {file_path}")
        record = ingest_file(file_path, existing_hashes)
        if record is None:
            continue
        existing_hashes.add(record.sha256)
        new_records.append(record)
        print(f"Ingested: {file_path} -> {record.stored_path}")

    if not new_records:
        print("No new documents ingested.")
        return

    manifest.extend(asdict(record) for record in new_records)
    save_manifest(manifest)
    print(f"Updated manifest: {MANIFEST_PATH}")


def list_documents() -> None:
    manifest = load_manifest()
    if not manifest:
        print("No documents ingested yet.")
        return

    for index, record in enumerate(manifest, start=1):
        print(
            f"{index}. {record['original_name']} "
            f"({record['mime_type']}, {record['ingest_strategy']})"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ingest documents for indexing.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    add_parser = subparsers.add_parser("add", help="Ingest a file or folder.")
    add_parser.add_argument("path", help="Path to a document or folder.")

    subparsers.add_parser("list", help="List ingested documents.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "add":
        add_path(args.path)
    elif args.command == "list":
        list_documents()


if __name__ == "__main__":
    main()


