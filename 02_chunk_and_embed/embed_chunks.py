"""BGE-large-en-v1.5 embedding pipeline.

Embeds chunked passages and builds a FAISS index for cosine similarity
retrieval. Embeddings are L2-normalised so inner product = cosine.

Usage:
    python 02_chunk_and_embed/embed_chunks.py
"""

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import os
import json
import argparse
import numpy as np
import faiss
from tqdm import tqdm

from utils.config import Config


def embed_chunks(chunks: list[dict], model_name: str, batch_size: int) -> np.ndarray:
    """Embed chunk texts with a SentenceTransformer model.

    Returns (N, D) float32 array of L2-normalised embeddings.
    """
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    texts = [c["text"] for c in chunks]

    print(f"Embedding {len(texts):,} passages with {model_name}...")
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    return np.array(embeddings, dtype=np.float32)


def run(config: Config | None = None):
    """Embed all chunks and build FAISS index."""
    if config is None:
        config = Config()
    config.ensure_dirs()

    # Load chunks
    chunks_path = os.path.join(config.chunks_dir, "chunks.json")
    if not os.path.exists(chunks_path):
        raise FileNotFoundError(
            f"No chunks at {chunks_path}. Run chunk.py first."
        )
    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    # Embed
    embeddings = embed_chunks(chunks, config.embedding_model, config.embed_batch_size)
    print(f"Embeddings shape: {embeddings.shape}")

    # Save embeddings
    emb_path = os.path.join(config.embeddings_dir, "embeddings.npy")
    np.save(emb_path, embeddings)

    # Save chunk IDs (parallel array)
    chunk_ids = [c["chunk_id"] for c in chunks]
    ids_path = os.path.join(config.embeddings_dir, "chunk_ids.json")
    with open(ids_path, "w") as f:
        json.dump(chunk_ids, f)

    # Build and save FAISS index
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # inner product on normalised = cosine
    index.add(embeddings)
    index_path = os.path.join(config.indices_dir, "full_corpus.index")
    faiss.write_index(index, index_path)

    print(f"Saved embeddings to {emb_path}")
    print(f"Saved chunk IDs to {ids_path}")
    print(f"Saved FAISS index ({index.ntotal} vectors, {dim}d) to {index_path}")

    return embeddings, chunk_ids


if __name__ == "__main__":
    run()
