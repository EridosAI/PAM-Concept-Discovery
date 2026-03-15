"""Centralized configuration for concept discovery experiments.

All hyperparameters in one place. The PAM predictor architecture is
intentionally identical to AAR v1 -- the variable under test is data scale.
"""

import os
from dataclasses import dataclass, field
from typing import List

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@dataclass
class Config:
    # -- Paths --
    base_dir: str = BASE_DIR
    raw_dir: str = os.path.join(BASE_DIR, "data", "raw")
    chunks_dir: str = os.path.join(BASE_DIR, "data", "chunks")
    embeddings_dir: str = os.path.join(BASE_DIR, "data", "embeddings")
    pairs_dir: str = os.path.join(BASE_DIR, "data", "pairs")
    queries_dir: str = os.path.join(BASE_DIR, "data", "queries")
    indices_dir: str = os.path.join(BASE_DIR, "data", "indices")
    models_dir: str = os.path.join(BASE_DIR, "models")
    results_dir: str = os.path.join(BASE_DIR, "results")
    figures_dir: str = os.path.join(BASE_DIR, "results", "figures")

    # -- Corpus --
    max_books: int = 250       # download target (~200 after length filter)
    min_book_words: int = 20000  # skip texts < 20K words

    # -- Chunking (token-based, using BGE tokenizer) --
    chunk_size: int = 256      # tokens
    chunk_overlap: int = 64    # tokens

    # -- Embedding --
    embedding_model: str = "BAAI/bge-large-en-v1.5"
    embedding_dim: int = 1024
    embed_batch_size: int = 64

    # -- PAM Predictor (DO NOT MODIFY -- identical to AAR v1) --
    hidden_dim: int = 1024
    num_layers: int = 4

    # -- Training --
    batch_size: int = 512
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    temperature: float = 0.05
    epochs: int = 100
    grad_clip: float = 1.0

    # -- Temporal Pair Generation --
    temporal_window: int = 3   # chunks in each direction
    max_pairs_per_book: int = 5000  # cap to avoid single-book dominance

    # -- Retrieval / Scoring --
    initial_k: int = 100       # FAISS candidates before reranking
    alpha: float = 0.50        # fixed, no tuning
    bidirectional: bool = True

    # -- Evaluation --
    k_values: List[int] = field(default_factory=lambda: [5, 10, 20])
    scale_points: List[int] = field(
        default_factory=lambda: [10, 25, 50, 100, 200]  # novels
    )
    num_eval_queries: int = 200

    # -- LLM (query generation) --
    anthropic_model: str = "claude-sonnet-4-20250514"

    # -- Reproducibility --
    random_seed: int = 42

    def ensure_dirs(self):
        """Create all output directories."""
        for d in [
            self.raw_dir, self.chunks_dir, self.embeddings_dir,
            self.pairs_dir, self.queries_dir, self.indices_dir,
            self.models_dir, self.results_dir, self.figures_dir,
            os.path.join(self.results_dir, "scale_curves"),
            os.path.join(self.results_dir, "ablations"),
        ]:
            os.makedirs(d, exist_ok=True)
