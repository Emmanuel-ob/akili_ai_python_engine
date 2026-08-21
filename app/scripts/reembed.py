"""
Re-chunk and re-embed stored knowledge-base content.

Engine Revamp Phase 4. The chunker changed from fixed-width character slicing
to paragraph-first splitting, so every existing chunk boundary is stale: rows
embedded under the old scheme begin and end mid-word and retrieve badly.

    python -m app.scripts.reembed --dry-run
    python -m app.scripts.reembed --business-id <id> --commit
    python -m app.scripts.reembed --commit

Safety properties, in order of how much they matter:

  DRY RUN BY DEFAULT      writing requires an explicit --commit
  RESUMABLE               rows already at the current CHUNKER_VERSION are
                          skipped, so an interrupted run continues rather
                          than starting over
  ORPHAN-FREE             a source document's old chunks are deleted before
                          new ones are written. Chunk ids are positional, so
                          re-chunking into FEWER pieces would otherwise leave
                          the surplus behind, still searchable
  PER-DOCUMENT ISOLATION  one bad document is logged and skipped; it does not
                          stop the corpus

Reconstruction note: this rebuilds from the stored `text` column, not from the
originally uploaded file. Any truncation the old pipeline already applied is
inherited. Full fidelity would mean re-processing source files, which is a
different piece of work.
"""

import argparse
import sys
from collections import defaultdict
from typing import Dict, List, Tuple

import psycopg2
from psycopg2.extras import Json, execute_values

from app.core.config import settings
from app.core.logging_config import logger
from app.services.chunk_ids import CHUNK_MARKER, group_by_source, source_id_of
from app.services.chunking import CHUNKER_VERSION, ChunkingService
from app.services.embeddings import EmbeddingService
from app.services.vectorstore import VectorStoreService

def load_rows(cur, business_id: str = None) -> List[Tuple]:
    """Fetch every stored row, optionally scoped to one business."""
    sql = (
        "SELECT doc_id, business_id, chatbot_id, text, metadata"
        " FROM embeddings"
    )
    params = ()
    if business_id:
        sql += " WHERE business_id = %s"
        params = (business_id,)
    sql += " ORDER BY doc_id"
    cur.execute(sql, params)
    return cur.fetchall()


def already_current(chunks: List[Tuple]) -> bool:
    """True when every row for this source is at the current chunker version."""
    for row in chunks:
        metadata = row[4] or {}
        if metadata.get("chunker_version") != CHUNKER_VERSION:
            return False
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--commit", action="store_true", help="write changes")
    ap.add_argument("--business-id", help="limit to one business")
    ap.add_argument(
        "--force",
        action="store_true",
        help="re-embed even rows already at the current chunker version",
    )
    args = ap.parse_args()
    writing = args.commit

    vectorstore = VectorStoreService()
    embedder = EmbeddingService()
    provider = embedder.get_provider_info()

    conn = psycopg2.connect(**vectorstore.connection_params)
    cur = conn.cursor()

    rows = load_rows(cur, args.business_id)
    grouped = group_by_source(rows)

    print(f"database:  {vectorstore.connection_params['database']}")
    print(f"scope:     {args.business_id or 'ALL businesses'}")
    print(f"embedder:  {provider['provider']} / {provider['model']}")
    print(f"chunker:   version {CHUNKER_VERSION}")
    print("MODE:      " + ("COMMIT (writing)" if writing else "DRY RUN (no writes)"))
    print(f"\n{len(rows)} rows across {len(grouped)} source documents\n")

    stats = {"skipped": 0, "rechunked": 0, "failed": 0, "before": 0, "after": 0}

    for source_id, chunks in sorted(grouped.items()):
        if already_current(chunks) and not args.force:
            stats["skipped"] += 1
            continue

        try:
            business_id = chunks[0][1]
            chatbot_id = chunks[0][2]
            metadata = dict(chunks[0][4] or {})
            full_text = "\n\n".join(c[3] for c in chunks if c[3])

            new_chunks = ChunkingService.chunk_text(full_text)
            if not new_chunks:
                logger.warning(f"{source_id}: produced no chunks, skipping")
                stats["failed"] += 1
                continue

            stats["before"] += len(chunks)
            stats["after"] += len(new_chunks)
            delta = len(new_chunks) - len(chunks)
            print(
                f"  {source_id[:70]:<70} {len(chunks):>3} -> {len(new_chunks):>3}"
                f" ({delta:+d})"
            )

            if not writing:
                continue

            # Delete the whole old set first. Without this, re-chunking into
            # fewer pieces leaves the surplus behind as searchable orphans.
            vectorstore.delete_by_pattern(
                VectorStoreService.chunk_pattern_for(source_id)
            )
            cur.execute("DELETE FROM embeddings WHERE doc_id = %s", (source_id,))
            conn.commit()

            payload = []
            for i, chunk in enumerate(new_chunks):
                vector = embedder.generate_single_embedding(chunk)
                doc_id = (
                    source_id
                    if len(new_chunks) == 1
                    else f"{source_id}{CHUNK_MARKER}{i}"
                )
                payload.append(
                    (
                        doc_id,
                        business_id,
                        chatbot_id,
                        vector,
                        chunk,
                        Json(
                            {
                                **metadata,
                                "embedding_provider": provider["provider"],
                                "embedding_model": provider["model"],
                                "chunker_version": CHUNKER_VERSION,
                            }
                        ),
                    )
                )

            execute_values(
                cur,
                "INSERT INTO embeddings (doc_id, business_id, chatbot_id,"
                " embedding, text, metadata) VALUES %s",
                payload,
            )
            conn.commit()
            stats["rechunked"] += 1

        except Exception as exc:
            # One bad document must not stop the corpus.
            conn.rollback()
            logger.error(f"{source_id}: re-embed failed: {exc}")
            stats["failed"] += 1

    print(
        f"\nsources: {stats['rechunked']} rechunked, {stats['skipped']} already "
        f"current, {stats['failed']} failed"
    )
    if stats["before"]:
        print(f"chunks:  {stats['before']} -> {stats['after']}")

    if not writing:
        print("\nDRY RUN: nothing written. Re-run with --commit to apply.")

    cur.close()
    conn.close()
    return 1 if stats["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
