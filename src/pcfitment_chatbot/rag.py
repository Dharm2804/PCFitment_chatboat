import json
import re
import shutil
from pathlib import Path

import chromadb
from chromadb.config import Settings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from loader import CATEGORIES, load_article


BASE_DIR = Path(__file__).resolve().parents[2]
DB_ROOT = (BASE_DIR / "data" / "chroma_dbs").resolve()      # one sub-folder per article
ARTICLES_JSON = (BASE_DIR / "data" / "articles.json").resolve()
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME = "article"
DEFAULT_CATEGORY = "General"

DEBUG_SEARCH = True   # prints which articles were matched; set False when done

_embeddings = None
_stores = {}
_active_urls = []          # URLs that currently exist (filled by sync_vector_database)


def get_embeddings():
    """Load the embedding model only once."""
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embeddings


# ---------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------

def save_articles_json(entries: list):
    """Atomic write so a crash can't corrupt the file."""
    ARTICLES_JSON.parent.mkdir(parents=True, exist_ok=True)
    tmp = ARTICLES_JSON.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2, ensure_ascii=False)
    tmp.replace(ARTICLES_JSON)


def ensure_articles_json():
    """Create articles.json (seeded from CATEGORIES) if it does not exist."""
    if ARTICLES_JSON.exists():
        return

    print(f"[CREATE] {ARTICLES_JSON.name} not found, creating it from CATEGORIES")

    entries, next_id = [], 1
    for urls in CATEGORIES.values():
        for url in urls:
            entries.append({
                "id": next_id,
                "currentUrl": url,
                "previousUrl": "",
                "flagupdate": False,
                "delete": False,
            })
            next_id += 1
    save_articles_json(entries)


def load_articles_json() -> list:
    ensure_articles_json()
    with open(ARTICLES_JSON, "r", encoding="utf-8") as f:
        return json.load(f)


def _clean(value) -> str:
    return (value or "").strip()


# ---------------------------------------------------------
# URL -> folder name  (115001118751-Signing-Up  ->  Signing-Up)
# ---------------------------------------------------------

_folder_cache = {}


def folder_name(url: str) -> str:
    if url in _folder_cache:
        return _folder_cache[url]

    slug = url.rstrip("/").split("/")[-1]
    name = re.sub(r"^\d+-", "", slug)
    name = re.sub(r"[^A-Za-z0-9_-]+", "-", name).strip("-") or slug

    # collision with a different URL -> fall back to the full slug (includes the ID)
    if name in _folder_cache.values():
        name = re.sub(r"[^A-Za-z0-9_-]+", "-", slug).strip("-")

    _folder_cache[url] = name
    return name


def folder_path(url: str) -> Path:
    return DB_ROOT / folder_name(url)


def category_for(url: str) -> str:
    for category, urls in CATEGORIES.items():
        if url in urls:
            return category
    return DEFAULT_CATEGORY


# ---------------------------------------------------------
# Stores: each URL gets its OWN Chroma client in its OWN folder
# ---------------------------------------------------------

def get_store(url: str) -> Chroma:
    if url not in _stores:
        path = folder_path(url)
        path.mkdir(parents=True, exist_ok=True)

        client = chromadb.PersistentClient(
            path=str(path),
            settings=Settings(anonymized_telemetry=False),
        )
        _stores[url] = Chroma(
            client=client,
            embedding_function=get_embeddings(),
            collection_name=COLLECTION_NAME,
            collection_metadata={"hnsw:space": "cosine"},
        )
    return _stores[url]


def url_is_indexed(url: str) -> bool:
    """True only if the folder already has a database with chunks in it."""
    path = folder_path(url)
    if not (path / "chroma.sqlite3").exists():
        return False
    return get_store(url)._collection.count() > 0


def remove_url_from_database(url: str) -> bool:
    """Close the store and delete the article's folder. Returns True if a folder was removed."""
    _stores.pop(url, None)

    # release open sqlite handles (important on Windows)
    try:
        from chromadb.api.shared_system_client import SharedSystemClient
        SharedSystemClient.clear_system_cache()
    except Exception:
        pass

    path = folder_path(url)
    existed = path.exists()
    shutil.rmtree(path, ignore_errors=True)
    print(f"       Deleted folder: {path}" if existed else f"       No folder to delete: {path}")
    return existed


# ---------------------------------------------------------
# Indexing
# ---------------------------------------------------------

def index_url(url: str, splitter) -> bool:
    """Create the folder + DB for one article. Returns True on success."""
    try:
        documents = load_article(url, category_for(url))
    except Exception as error:
        print(f"       Failed to load: {error}")
        remove_url_from_database(url)          # don't leave an empty folder behind
        return False

    if not documents:
        print("       No content found, skipping.")
        remove_url_from_database(url)
        return False

    chunks = splitter.split_documents(documents)
    get_store(url).add_documents(chunks)       # get_store creates the folder if missing
    print(f"       Added {len(chunks)} chunks -> {folder_path(url)}")
    return True


def ensure_indexed(url: str, splitter) -> bool:
    """If the DB folder is missing (or empty/broken), create it. Otherwise leave it alone."""
    label = folder_name(url)

    if url_is_indexed(url):
        print(f"[SKIP] {label}")
        return True

    if folder_path(url).exists():
        print(f"[REPAIR] {label} (folder exists but has no data)")
        remove_url_from_database(url)
    else:
        print(f"[NEW]  {label} (folder not found, creating)")

    return index_url(url, splitter)


# ---------------------------------------------------------
# Sync driven by the JSON file
# ---------------------------------------------------------

def sync_vector_database():
    global _active_urls

    DB_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"Vector DB root: {DB_ROOT}")

    entries = load_articles_json()
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)

    remaining = []          # entries to write back to the JSON
    json_changed = False

    for entry in entries:
        current = _clean(entry.get("currentUrl"))
        previous = _clean(entry.get("previousUrl"))
        flag_update = bool(entry.get("flagupdate", False))
        flag_delete = bool(entry.get("delete", False))

        if not current:
            print(f"[WARN] id={entry.get('id')} has no currentUrl, skipping.")
            remaining.append(entry)
            continue

        label = folder_name(current)

        # ---- 1) DELETE: remove current URL's folder, drop entry from JSON ----
        if flag_delete:
            print(f"[DELETE] {label}")
            remove_url_from_database(current)
            json_changed = True
            continue                      # entry is not kept

        # ---- 2) UPDATE ----
        if flag_update:
            old_url = previous or current          # previous empty -> same as current
            print(f"[UPDATE] {folder_name(old_url)} -> {label}")

            remove_url_from_database(old_url)      # fine if it doesn't exist
            if old_url != current:
                remove_url_from_database(current)  # make sure the new one starts clean

            if index_url(current, splitter):       # creates the new folder + DB
                entry["flagupdate"] = False
                entry["previousUrl"] = ""
                json_changed = True
            # on failure the flags stay set, so it retries on next start

            remaining.append(entry)
            continue

        # ---- 3) NORMAL: create if it does not exist ----
        ensure_indexed(current, splitter)
        remaining.append(entry)

    if json_changed:
        save_articles_json(remaining)
        print(f"\nUpdated {ARTICLES_JSON.name}")

    _active_urls = [_clean(e.get("currentUrl")) for e in remaining if _clean(e.get("currentUrl"))]

    indexed = [u for u in _active_urls if url_is_indexed(u)]
    total = sum(get_store(u)._collection.count() for u in indexed)
    print(f"\nVector DBs ready. Databases: {len(indexed)}, total chunks: {total}")

    if total == 0:
        raise RuntimeError("Vector databases are empty. No articles could be loaded.")


# ---------------------------------------------------------
# Search across all article databases
# ---------------------------------------------------------

def search_all(question: str, k: int = 4):
    """Search every active article DB, merge by cosine distance, return the global top k."""
    urls = _active_urls or [
        _clean(e.get("currentUrl"))
        for e in load_articles_json()
        if _clean(e.get("currentUrl")) and not e.get("delete")
    ]

    query_vector = get_embeddings().embed_query(question)   # embed only once
    scored = []

    for url in urls:
        if not (folder_path(url) / "chroma.sqlite3").exists():
            continue
        try:
            collection = get_store(url)._collection
            count = collection.count()
            if count == 0:
                continue

            result = collection.query(
                query_embeddings=[query_vector],
                n_results=min(k, count),
                include=["documents", "metadatas", "distances"],
            )

            for text, meta, dist in zip(
                result["documents"][0],
                result["metadatas"][0],
                result["distances"][0],
            ):
                scored.append((Document(page_content=text, metadata=meta or {}), dist))

        except Exception as error:
            print(f"Search failed in {folder_name(url)}: {error}")

    scored.sort(key=lambda pair: pair[1])    # lower distance = more similar
    top = scored[:k]

    if DEBUG_SEARCH:
        print("Top matches:")
        for doc, dist in top:
            print(f"  {dist:.3f}  {doc.metadata.get('title', '?')}")

    return [doc for doc, _ in top]