"""Combines the two independent knowledge bases (articles + PDFs)."""
from embeddings import get_embeddings
from rag import search_all_scored, sync_vector_database
from pdf_rag import search_pdfs_scored, sync_pdf_database

DEBUG_SEARCH = True


def sync_all():
    """Build/update both vector databases. PDF problems never block the article DB."""
    sync_vector_database()
    try:
        sync_pdf_database()
    except Exception as error:
        print(f"[WARN] PDF database sync failed: {error}")


def search_everything(question: str, k: int = 6):
    """Search article DBs and PDF DBs, merge by distance, return the global top k documents."""
    query_vector = get_embeddings().embed_query(question)   # embed once

    scored = search_all_scored(question, k, query_vector)
    scored += search_pdfs_scored(question, k, query_vector)
    scored.sort(key=lambda pair: pair[1])
    top = scored[:k]

    if DEBUG_SEARCH:
        print("Top matches (articles + PDFs):")
        for doc, dist in top:
            kind = "PDF" if doc.metadata.get("type") == "pdf" else "WEB"
            print(f"  {dist:.3f}  [{kind}] {doc.metadata.get('title', '?')}")

    return [doc for doc, _ in top]