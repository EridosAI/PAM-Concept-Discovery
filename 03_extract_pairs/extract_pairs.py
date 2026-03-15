"""Temporal co-occurrence pair generation.

For each chunk at position t in a book, pairs it with chunks at
positions t-w through t+w (excluding itself). This captures narrative
association: passages near each other in the same story are associated.

Usage:
    python 03_extract_pairs/extract_pairs.py
    python 03_extract_pairs/extract_pairs.py --window 5
"""

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import os
import json
import argparse
import random
from collections import defaultdict
from tqdm import tqdm

from utils.config import Config


def generate_temporal_pairs(chunks: list[dict], window: int,
                            max_pairs_per_book: int) -> list[tuple[int, int]]:
    """Generate (anchor_idx, positive_idx) pairs from temporal co-occurrence.

    Groups chunks by book, then for each chunk creates pairs with chunks
    within +/- window positions in the same book.
    """
    # Group chunk indices by book
    book_chunks = defaultdict(list)
    for i, chunk in enumerate(chunks):
        book_chunks[chunk["book_id"]].append((i, chunk["position"]))

    all_pairs = []
    for book_id, chunk_list in tqdm(book_chunks.items(), desc="Generating pairs"):
        # Sort by position within book
        chunk_list.sort(key=lambda x: x[1])

        book_pairs = []
        for idx_a, (global_idx_a, pos_a) in enumerate(chunk_list):
            for idx_b, (global_idx_b, pos_b) in enumerate(chunk_list):
                if idx_a == idx_b:
                    continue
                if abs(pos_a - pos_b) <= window:
                    book_pairs.append((global_idx_a, global_idx_b))

        # Cap per book to avoid single-book dominance
        if len(book_pairs) > max_pairs_per_book:
            book_pairs = random.sample(book_pairs, max_pairs_per_book)

        all_pairs.extend(book_pairs)

    return all_pairs


def run(config: Config | None = None, window: int | None = None):
    """Generate and save temporal co-occurrence pairs."""
    if config is None:
        config = Config()
    config.ensure_dirs()

    w = window or config.temporal_window

    # Load chunks
    chunks_path = os.path.join(config.chunks_dir, "chunks.json")
    if not os.path.exists(chunks_path):
        raise FileNotFoundError(
            f"No chunks at {chunks_path}. Run chunk.py first."
        )
    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    print(f"Generating temporal pairs (window={w}) from {len(chunks):,} chunks...")
    pairs = generate_temporal_pairs(chunks, w, config.max_pairs_per_book)

    # Deduplicate
    pairs = list(set(pairs))
    random.shuffle(pairs)

    # Save
    pairs_path = os.path.join(config.pairs_dir, f"temporal_pairs_w{w}.json")
    with open(pairs_path, "w") as f:
        json.dump(pairs, f)

    # Stats
    book_ids = set(chunks[a]["book_id"] for a, _ in pairs[:10000])
    print(f"\nGenerated {len(pairs):,} unique pairs (window={w})")
    print(f"Books represented (sample): {len(book_ids)}")
    print(f"Saved to {pairs_path}")

    return pairs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate temporal pairs")
    parser.add_argument("--window", type=int, default=None,
                        help="Temporal window size")
    args = parser.parse_args()
    run(window=args.window)
