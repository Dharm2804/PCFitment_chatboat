"""
PDF knowledge base: same flow as the article database in rag.py.

  data/QandAPdf/*.pdf        -> source PDFs
  data/pdfs.json             -> one entry per PDF (flags: flagupdate / delete)
  data/PDF_VectorDB/<name>/  -> one Chroma DB per PDF
"""
import json
import re
import shutil
from pathlib import Path

import chromadb
from chromadb.config import Settings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from pypdf import PdfReader

from embeddings import BASE_DIR, get_embeddings

PDF_DIR = (BASE_DIR / "data" / "QandAPdf").resolve()
PDF_DB_ROOT = (BASE_DIR / "data" / "PDF_VectorDB").resolve()
PDFS_JSON = (BASE_DIR / "data" / "pdfs.json").resolve()
DELETED_DIR = PDF_DIR / "_deleted"
COLLECTION_NAME = "pdf"

_stores = {}
_active_files = []


# ---------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------

def _clean(value) -> str:
    return (value or "").strip()


def save_pdfs_json(entries: list):
    PDFS_JSON.parent.mkdir(parents=True, exist_ok=True)
    tmp = PDFS_JSON.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2, ensure_ascii=False)
    tmp.replace(PDFS_JSON)


def load_pdfs_json() -> list:
    """Load pdfs.json; create it if missing. New PDFs dropped in the folder are added automatically."""
    PDF_DIR.mkdir(parents=True, exist_ok=True)

    if PDFS_JSON.exists():
        with open(PDFS_JSON, "r", encoding="utf-8") as f:
            entries = json.load(f)
    else:
        print(f"[CREATE] {PDFS_JSON.name} not found, creating it from {PDF_DIR.name}/")
        entries = []

    known = {_clean(e.get("fileName")).lower() for e in entries}
    next_id = max([e.get("id", 0) for e in entries], default=0) + 1
    changed = not PDFS_JSON.exists()

    for pdf in sorted(PDF_DIR.glob("*.pdf")):
        if pdf.name.lower() not in known:
            print(f"[FOUND] new PDF {pdf.name}")
            entries.append({
                "id": next_id,
                "fileName": pdf.name,
                "previousFileName": "",
                "flagupdate": False,
                "delete": False,
            })
            next_id += 1
            changed = True

    if changed:
        save_pdfs_json(entries)
    return entries


# ---------------------------------------------------------
# Folder naming / stores
# ---------------------------------------------------------

def folder_name(file_name: str) -> str:
    stem = Path(file_name).stem
    return re.sub(r"[^A-Za-z0-9_-]+", "-", stem).strip("-") or "pdf"


def folder_path(file_name: str) -> Path:
    return PDF_DB_ROOT / folder_name(file_name)


def get_store(file_name: str) -> Chroma:
    if file_name not in _stores:
        path = folder_path(file_name)
        path.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(
            path=str(path),
            settings=Settings(anonymized_telemetry=False),
        )
        _stores[file_name] = Chroma(
            client=client,
            embedding_function=get_embeddings(),
            collection_name=COLLECTION_NAME,
            collection_metadata={"hnsw:space": "cosine"},
        )
    return _stores[file_name]


def pdf_is_indexed(file_name: str) -> bool:
    path = folder_path(file_name)
    if not (path / "chroma.sqlite3").exists():
        return False
    return get_store(file_name)._collection.count() > 0


def remove_pdf_from_database(file_name: str) -> bool:
    _stores.pop(file_name, None)
    try:
        from chromadb.api.shared_system_client import SharedSystemClient
        SharedSystemClient.clear_system_cache()
    except Exception:
        pass

    path = folder_path(file_name)
    existed = path.exists()
    shutil.rmtree(path, ignore_errors=True)
    print(f"       Deleted folder: {path}" if existed else f"       No folder to delete: {path}")
    return existed


# ---------------------------------------------------------
# Loading + indexing
# ---------------------------------------------------------

def load_pdf(file_name: str) -> list:
    """One Document per page that contains text."""
    pdf_path = PDF_DIR / file_name
    if not pdf_path.exists():
        raise FileNotFoundError(f"{pdf_path} not found")

    title = re.sub(r"[-_]+", " ", pdf_path.stem).strip()
    reader = PdfReader(str(pdf_path))
    docs = []
    for number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if not text:
            continue
        docs.append(Document(
            page_content=text,
            metadata={
                "title": title,
                "source": file_name,
                "category": "PDF",
                "type": "pdf",
                "page": number,
            },
        ))
    return docs


def index_pdf(file_name: str, splitter) -> bool:
    try:
        documents = load_pdf(file_name)
    except Exception as error:
        print(f"       Failed to load: {error}")
        remove_pdf_from_database(file_name)
        return False

    if not documents:
        print("       No text found (scanned PDF? needs OCR), skipping.")
        remove_pdf_from_database(file_name)
        return False

    chunks = splitter.split_documents(documents)
    get_store(file_name).add_documents(chunks)
    print(f"       Added {len(chunks)} chunks -> {folder_path(file_name)}")
    return True


def ensure_pdf_indexed(file_name: str, splitter) -> bool:
    label = folder_name(file_name)
    if pdf_is_indexed(file_name):
        print(f"[SKIP] PDF {label}")
        return True
    if folder_path(file_name).exists():
        print(f"[REPAIR] PDF {label} (folder exists but has no data)")
        remove_pdf_from_database(file_name)
    else:
        print(f"[NEW]  PDF {label} (folder not found, creating)")
    return index_pdf(file_name, splitter)


# ---------------------------------------------------------
# Sync driven by pdfs.json
# ---------------------------------------------------------

def sync_pdf_database():
    global _active_files

    PDF_DB_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"PDF Vector DB root: {PDF_DB_ROOT}")

    entries = load_pdfs_json()
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)

    remaining, json_changed = [], False

    for entry in entries:
        current = _clean(entry.get("fileName"))
        previous = _clean(entry.get("previousFileName"))
        flag_update = bool(entry.get("flagupdate", False))
        flag_delete = bool(entry.get("delete", False))

        if not current:
            print(f"[WARN] id={entry.get('id')} has no fileName, skipping.")
            remaining.append(entry)
            continue

        # 1) DELETE: drop DB + JSON entry; move the PDF aside so it is not re-added
        if flag_delete:
            print(f"[DELETE] PDF {folder_name(current)}")
            remove_pdf_from_database(current)
            src = PDF_DIR / current
            if src.exists():
                DELETED_DIR.mkdir(parents=True, exist_ok=True)
                shutil.move(str(src), str(DELETED_DIR / current))
            json_changed = True
            continue

        # 2) UPDATE: rebuild the DB (previousFileName empty -> same file, re-read it)
        if flag_update:
            old = previous or current
            print(f"[UPDATE] PDF {folder_name(old)} -> {folder_name(current)}")
            remove_pdf_from_database(old)
            if old != current:
                remove_pdf_from_database(current)

            if index_pdf(current, splitter):
                entry["flagupdate"] = False
                entry["previousFileName"] = ""
                json_changed = True
            remaining.append(entry)
            continue

        # 3) NORMAL
        ensure_pdf_indexed(current, splitter)
        remaining.append(entry)

    if json_changed:
        save_pdfs_json(remaining)
        print(f"\nUpdated {PDFS_JSON.name}")

    _active_files = [_clean(e.get("fileName")) for e in remaining if _clean(e.get("fileName"))]
    indexed = [f for f in _active_files if pdf_is_indexed(f)]
    total = sum(get_store(f)._collection.count() for f in indexed)
    print(f"\nPDF DBs ready. Databases: {len(indexed)}, total chunks: {total}")


# ---------------------------------------------------------
# Search
# ---------------------------------------------------------

def search_pdfs_scored(question: str, k: int = 4, query_vector=None):
    if query_vector is None:
        query_vector = get_embeddings().embed_query(question)

    scored = []
    for file_name in _active_files:
        if not (folder_path(file_name) / "chroma.sqlite3").exists():
            continue
        try:
            collection = get_store(file_name)._collection
            count = collection.count()
            if count == 0:
                continue
            result = collection.query(
                query_embeddings=[query_vector],
                n_results=min(k, count),
                include=["documents", "metadatas", "distances"],
            )
            for text, meta, dist in zip(
                result["documents"][0], result["metadatas"][0], result["distances"][0]
            ):
                scored.append((Document(page_content=text, metadata=meta or {}), dist))
        except Exception as error:
            print(f"PDF search failed in {folder_name(file_name)}: {error}")

    scored.sort(key=lambda pair: pair[1])
    return scored[:k]