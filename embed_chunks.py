"""
Embed Docling chunk files with a local sentence-transformers model and store
the vectors in a local Chroma database, so you have a working RAG index over
the RFPs. Each chunking strategy gets its own Chroma collection, so different
strategies can be compared without overwriting each other.

Usage:
    python embed_chunks.py                                  # build the "small" (default) strategy
    python embed_chunks.py --strategy large                 # build the "large" strategy
    python embed_chunks.py --strategy large --query "submission deadline"
"""

import argparse
import json
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

# strategy name -> (chunks directory, Chroma collection name)
STRATEGIES = {
    "small": (Path(__file__).parent / "docling-output", "rfps_small"),
    "large": (Path(__file__).parent / "docling-output-large", "rfps_large"),
}
DB_DIR = Path(__file__).parent / "chroma-db"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # small, fast, runs fully on CPU


def load_chunks(chunks_dir: Path):
    for path in sorted(chunks_dir.glob("*.chunks.jsonl")):
        source_stem = path.name.removesuffix(".chunks.jsonl")
        with path.open(encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                record.setdefault("source_file", source_stem)  # older files lack this field
                yield record


def build_db(strategy: str) -> None:
    chunks_dir, collection_name = STRATEGIES[strategy]

    model = SentenceTransformer(EMBEDDING_MODEL)
    client = chromadb.PersistentClient(path=str(DB_DIR))
    collection = client.get_or_create_collection(collection_name)

    ids, texts, metadatas = [], [], []
    for record in load_chunks(chunks_dir):
        ids.append(f"{record['source_file']}::{record['chunk_index']}")
        texts.append(record["text"])
        metadatas.append(
            {
                "source_file": record["source_file"],
                "headings": " > ".join(record.get("headings") or []),
                "page_numbers": json.dumps(record.get("page_numbers") or []),
            }
        )

    print(f"[{strategy}] Embedding {len(texts)} chunks with {EMBEDDING_MODEL} ...")
    embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True).tolist()

    collection.upsert(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)
    print(f"[{strategy}] Stored {len(ids)} chunks in Chroma collection '{collection_name}' at {DB_DIR}")


def run_query(strategy: str, question: str, n_results: int = 5) -> None:
    _, collection_name = STRATEGIES[strategy]

    model = SentenceTransformer(EMBEDDING_MODEL)
    client = chromadb.PersistentClient(path=str(DB_DIR))
    collection = client.get_or_create_collection(collection_name)

    query_embedding = model.encode([question], normalize_embeddings=True).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=n_results)

    for doc, meta, dist in zip(
        results["documents"][0], results["metadatas"][0], results["distances"][0]
    ):
        print(f"\n[{meta['source_file']}] (distance={dist:.3f}) headings={meta['headings']}")
        print(doc[:300])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", choices=STRATEGIES.keys(), default="small")
    parser.add_argument("--query", help="Run a similarity search against the existing DB instead of rebuilding it")
    args = parser.parse_args()

    if args.query:
        run_query(args.strategy, args.query)
    else:
        build_db(args.strategy)


if __name__ == "__main__":
    main()
