"""Token-based chunking pipeline for novel texts.

Splits cleaned novel texts into overlapping passages of 256 tokens with
64-token overlap, using BGE-large-en-v1.5's tokenizer for accurate token
counting. Chunks must not cross book boundaries.

Usage:
    python 02_chunk_and_embed/chunk_texts.py
"""

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import os
import json
from tqdm import tqdm
from transformers import AutoTokenizer

from utils.config import Config


def chunk_text_by_tokens(text: str, tokenizer, chunk_size: int = 256,
                         overlap: int = 64) -> list[dict]:
    """Split text into overlapping chunks by token count.

    Uses the BGE tokenizer for accurate token counting. Tries to break at
    sentence/paragraph boundaries within the last 20% of each chunk.

    Returns list of dicts with 'text', 'char_start', 'char_end' keys.
    """
    encoding = tokenizer(text, return_offsets_mapping=True,
                         add_special_tokens=False)
    tokens = encoding["input_ids"]
    offsets = encoding["offset_mapping"]

    if not tokens:
        return []

    if len(tokens) <= chunk_size:
        return [{"text": text.strip(), "char_start": 0, "char_end": len(text)}] \
            if text.strip() else []

    chunks = []
    stride = chunk_size - overlap
    start_tok = 0

    while start_tok < len(tokens):
        end_tok = min(start_tok + chunk_size, len(tokens))

        # Try to break at a sentence/paragraph boundary in the last 20%
        if end_tok < len(tokens):
            search_start_tok = start_tok + int(chunk_size * 0.8)
            best_break = None
            for t in range(end_tok - 1, search_start_tok - 1, -1):
                char_end = offsets[t][1]
                # Check for paragraph break
                if text[char_end:char_end + 2] == "\n\n":
                    best_break = t + 1
                    break
                # Check for sentence break
                if text[char_end:char_end + 2] == ". " or \
                   text[char_end:char_end + 2] == ".\n":
                    best_break = t + 1
                    break
            if best_break is not None:
                end_tok = best_break

        char_start = offsets[start_tok][0]
        char_end = offsets[end_tok - 1][1]
        chunk_text = text[char_start:char_end].strip()

        if chunk_text:
            chunks.append({
                "text": chunk_text,
                "char_start": char_start,
                "char_end": char_end,
            })

        if end_tok >= len(tokens):
            break
        start_tok += stride

    return chunks


def run(config: Config | None = None):
    """Chunk all cleaned book texts using token-based chunking."""
    if config is None:
        config = Config()
    config.ensure_dirs()

    print(f"Loading tokenizer for {config.embedding_model}...")
    tokenizer = AutoTokenizer.from_pretrained(config.embedding_model)

    metadata_path = os.path.join(config.raw_dir, "book_metadata.json")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(
            f"No book metadata at {metadata_path}. Run download_gutenberg first."
        )
    with open(metadata_path, "r") as f:
        books = json.load(f)

    all_chunks = []
    for book in tqdm(books, desc="Chunking books"):
        book_path = os.path.join(config.raw_dir, f"book_{book['id']}.txt")
        if not os.path.exists(book_path):
            print(f"  [SKIP] Missing text file for {book['title']}")
            continue

        with open(book_path, "r", encoding="utf-8") as f:
            text = f.read()

        raw_chunks = chunk_text_by_tokens(
            text, tokenizer, config.chunk_size, config.chunk_overlap
        )
        for i, chunk_data in enumerate(raw_chunks):
            all_chunks.append({
                "chunk_id": f"book_{book['id']}_chunk_{i:05d}",
                "book_id": book["id"],
                "book_title": book["title"],
                "author": book["author"],
                "position": i,
                "total_chunks": len(raw_chunks),
                "text": chunk_data["text"],
                "char_offset_start": chunk_data["char_start"],
                "char_offset_end": chunk_data["char_end"],
            })

    # Save chunks
    chunks_path = os.path.join(config.chunks_dir, "chunks.json")
    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)

    books_chunked = len(set(c["book_id"] for c in all_chunks))
    print(f"\nChunked {books_chunked} books into {len(all_chunks):,} passages")
    print(f"Chunk size: {config.chunk_size} tokens, overlap: {config.chunk_overlap} tokens")
    print(f"Saved to {chunks_path}")

    return all_chunks


if __name__ == "__main__":
    run()
