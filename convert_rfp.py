"""
Run Docling on one or more RFP PDFs and print/save what it extracts, so you
can eyeball the output before building a full vectorization pipeline.

For each input PDF, writes two files into ./docling-output/:
  - <name>.md            human-readable Markdown (for you to read)
  - <name>.chunks.jsonl   one JSON object per chunk (what you'd embed for a vector DB)

Source can be a local file, a local directory of PDFs, or an S3 location.
S3 credentials are picked up from the standard boto3 chain (~/.aws/credentials,
an AWS_PROFILE env var, or AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY env vars) -
nothing is read from this script.

Usage:
    python convert_rfp.py "MarCom-RFP-Webinar_3.14.24-001.pdf"
    python convert_rfp.py ./some-folder-of-pdfs
    python convert_rfp.py s3://my-rfp-bucket/rfps/

    # Compare a different chunking strategy without touching the default output:
    python convert_rfp.py ./s3-downloads --out-dir docling-output-large --chunk-max-tokens 1024
"""

import argparse
import json
import sys
from pathlib import Path

from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker
from docling_core.transforms.chunker.tokenizer.openai import OpenAITokenizer
import tiktoken

DOWNLOAD_DIR = Path(__file__).parent / "s3-downloads"


def iter_local_pdfs(source: str):
    path = Path(source)
    if not path.exists():
        print(f"Path not found: {path}")
        sys.exit(1)
    if path.is_dir():
        yield from sorted(path.glob("*.pdf"))
    else:
        yield path


def iter_s3_pdfs(s3_uri: str):
    """List every .pdf under an s3://bucket/prefix URI, download each to
    DOWNLOAD_DIR, and yield the local paths."""
    import boto3

    bucket, _, prefix = s3_uri.removeprefix("s3://").partition("/")
    DOWNLOAD_DIR.mkdir(exist_ok=True)

    s3 = boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")
    keys = [
        obj["Key"]
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix)
        for obj in page.get("Contents", [])
        if obj["Key"].lower().endswith(".pdf")
    ]

    if not keys:
        print(f"No .pdf objects found under {s3_uri}")
        sys.exit(1)

    for key in keys:
        local_path = DOWNLOAD_DIR / Path(key).name
        print(f"Downloading s3://{bucket}/{key} -> {local_path}")
        s3.download_file(bucket, key, str(local_path))
        yield local_path


def process_pdf(pdf_path: Path, converter: DocumentConverter, chunker: HybridChunker, out_dir: Path) -> None:
    # 1. Convert the PDF into Docling's structured document representation
    #    (layout analysis, OCR for scanned/image content, table structure recognition).
    print(f"Converting {pdf_path.name} ...")
    result = converter.convert(str(pdf_path))
    doc = result.document

    # 2. Write Markdown so a human can read exactly what was extracted.
    md_path = out_dir / f"{pdf_path.stem}.md"
    md_path.write_text(doc.export_to_markdown(), encoding="utf-8")
    print(f"Wrote Markdown -> {md_path}")

    # 3. Chunk the document with Docling's hybrid chunker. This is what you'd
    #    actually embed and store in a vector DB: each chunk keeps its section
    #    heading(s) and source page number(s) as metadata.
    chunks_path = out_dir / f"{pdf_path.stem}.chunks.jsonl"
    with chunks_path.open("w", encoding="utf-8") as f:
        for i, chunk in enumerate(chunker.chunk(doc)):
            record = {
                "source_file": pdf_path.name,
                "chunk_index": i,
                "text": chunker.contextualize(chunk),  # heading context + chunk text
                "raw_text": chunk.text,
                "headings": chunk.meta.headings,
                "page_numbers": sorted(
                    {
                        prov.page_no
                        for item in chunk.meta.doc_items
                        for prov in item.prov
                    }
                ),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(f"  chunk {i}: {record['headings']} (page {record['page_numbers']})")

    print(f"Wrote {i + 1} chunks -> {chunks_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", help="Local PDF file, local folder of PDFs, or s3://bucket/prefix")
    parser.add_argument("--out-dir", default="docling-output", help="Output directory (default: docling-output)")
    parser.add_argument(
        "--chunk-max-tokens",
        type=int,
        default=None,
        help="Max tokens per chunk before HybridChunker merges/splits (default: the tokenizer's own limit, ~256)",
    )
    args = parser.parse_args()

    out_dir = Path(__file__).parent / args.out_dir
    out_dir.mkdir(exist_ok=True)

    pdf_paths = iter_s3_pdfs(args.source) if args.source.startswith("s3://") else iter_local_pdfs(args.source)

    # Reuse one converter/chunker across files - loading the underlying models
    # is the expensive part, so this avoids redoing it per PDF.
    converter = DocumentConverter()
    if args.chunk_max_tokens:
        # Token counting decoupled from the (short-context) embedding model's
        # tokenizer, so a larger chunk-size strategy isn't capped at ~256 tokens.
        tokenizer = OpenAITokenizer(tokenizer=tiktoken.get_encoding("cl100k_base"), max_tokens=args.chunk_max_tokens)
        chunker = HybridChunker(tokenizer=tokenizer)
    else:
        chunker = HybridChunker()

    count = 0
    for pdf_path in pdf_paths:
        process_pdf(pdf_path, converter, chunker, out_dir)
        count += 1

    print(f"\nDone. Processed {count} PDF(s). Output in {out_dir}")


if __name__ == "__main__":
    main()
