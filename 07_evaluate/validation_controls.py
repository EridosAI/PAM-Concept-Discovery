"""Ablation controls for AAR v2.

Implements all ablations from Section 6 of the experiment design:
  6.1 Temporal shuffle control
  6.2 Scale-matched cosine baseline (handled in scale_evaluation.py)
  6.3 Random retrieval baseline
  6.4 BM25 reranking baseline
  6.5 Similar-positives control
  6.6 Chunk size sensitivity (requires re-running corpus pipeline)

Usage:
    python 07_evaluate/validation_controls.py
    python 07_evaluate/validation_controls.py --ablation shuffle
"""

import os
import json
import random
import argparse
import time
import numpy as np
import torch
import pandas as pd
from tqdm import tqdm

from utils.config import Config
from src.evaluation.metrics import evaluate_queries, compute_median_similarity
from src.evaluation.scale_evaluation import (
    load_model, make_retrieval_fns, remap_queries_to_subset, build_novel_subset
)
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "04_train"))
from train import AssociationMLP, train, PairDataset
from src.training.generate_pairs import generate_temporal_pairs


# ---------------------------------------------------------------------------
# 6.1 Temporal shuffle control
# ---------------------------------------------------------------------------

def ablation_shuffle(config: Config, chunks: list[dict],
                     embeddings: np.ndarray, queries: list[dict],
                     device: str = "cuda") -> dict:
    """Train predictor on shuffled temporal ordering, evaluate.

    Shuffles chunk positions within each book while preserving all
    embeddings. If PAM's signal is genuine temporal structure, the
    shuffled model should collapse to cosine-level performance.
    """
    print("\n--- Ablation 6.1: Temporal shuffle ---")

    # Shuffle positions within each book
    shuffled_chunks = []
    book_groups = {}
    for c in chunks:
        bid = c["book_id"]
        if bid not in book_groups:
            book_groups[bid] = []
        book_groups[bid].append(c.copy())

    rng = random.Random(42)
    for bid, group in book_groups.items():
        positions = [c["position"] for c in group]
        rng.shuffle(positions)
        for c, new_pos in zip(group, positions):
            c["position"] = new_pos
        shuffled_chunks.extend(group)

    # Restore original global ordering (by chunk_id) so indices match embeddings
    chunk_id_to_idx = {c["chunk_id"]: i for i, c in enumerate(chunks)}
    shuffled_chunks.sort(key=lambda c: chunk_id_to_idx[c["chunk_id"]])
    shuffled_pairs = generate_temporal_pairs(
        shuffled_chunks, config.temporal_window, config.max_pairs_per_book
    )
    shuffled_pairs = list(set(shuffled_pairs))
    random.shuffle(shuffled_pairs)
    print(f"  Shuffled pairs: {len(shuffled_pairs):,}")

    # Train on shuffled pairs
    model = train(config, shuffled_pairs, embeddings)

    # Evaluate at full corpus scale
    median_sim = compute_median_similarity(embeddings)
    retrieval_fns = make_retrieval_fns(embeddings, model, config, device)

    results = {}
    for method_name in ["cosine", "aar", "pam_only"]:
        method_results = evaluate_queries(
            queries, retrieval_fns[method_name], embeddings,
            config.k_values, median_sim
        )
        for k, v in method_results.items():
            results[f"shuffle_{method_name}_{k}"] = v

    print(f"  Results: {results}")
    return results


# ---------------------------------------------------------------------------
# 6.3 Random retrieval baseline
# ---------------------------------------------------------------------------

def ablation_random(config: Config, embeddings: np.ndarray,
                    queries: list[dict]) -> dict:
    """Random passage retrieval -- establishes chance floor."""
    print("\n--- Ablation 6.3: Random retrieval ---")
    n = embeddings.shape[0]
    max_k = max(config.k_values)

    def random_retrieve(query_emb):
        indices = np.random.choice(n, size=max_k, replace=False)
        scores = np.zeros(max_k)
        return indices, scores

    median_sim = compute_median_similarity(embeddings)
    results = evaluate_queries(
        queries, random_retrieve, embeddings, config.k_values, median_sim
    )
    print(f"  Results: {results}")
    return {"random_" + k: v for k, v in results.items()}


# ---------------------------------------------------------------------------
# 6.4 BM25 reranking baseline
# ---------------------------------------------------------------------------

def ablation_bm25(config: Config, chunks: list[dict],
                  embeddings: np.ndarray,
                  queries: list[dict]) -> dict:
    """BM25 lexical similarity reranking baseline."""
    print("\n--- Ablation 6.4: BM25 reranking ---")

    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        print("  rank_bm25 not installed. pip install rank_bm25")
        return {}

    # Build BM25 index
    texts = [c["text"] for c in chunks]
    tokenized = [t.lower().split() for t in texts]
    bm25 = BM25Okapi(tokenized)
    max_k = max(config.k_values)

    def bm25_retrieve(query_emb):
        # BM25 needs text -- use the question text attached to query
        # This is a limitation: we need to pass question text through
        # For now, return BM25 scores for all docs and take top-k
        # The query_emb is unused; we need the text query
        return np.arange(max_k), np.zeros(max_k)

    # BM25 requires text queries, not embeddings
    # We need to handle this differently
    median_sim = compute_median_similarity(embeddings)
    results = {}

    for q in tqdm(queries, desc="BM25 eval"):
        question = q.get("question", "")
        if not question:
            continue
        tokens = question.lower().split()
        scores = bm25.get_scores(tokens)
        top_indices = np.argsort(-scores)[:max_k]

        for k in config.k_values:
            gold = q["gold_indices"]
            retrieved = set(int(i) for i in top_indices[:k])
            if gold:
                recall = len(retrieved & gold) / len(gold)
            else:
                recall = 0.0
            results.setdefault(f"bm25_TAR@{k}", []).append(recall)

    results = {k: float(np.mean(v)) for k, v in results.items()}
    print(f"  Results: {results}")
    return results


# ---------------------------------------------------------------------------
# 6.5 Similar-positives control
# ---------------------------------------------------------------------------

def ablation_similar_positives(config: Config, embeddings: np.ndarray,
                               queries: list[dict],
                               device: str = "cuda") -> dict:
    """Train on FAISS nearest-neighbor pairs instead of temporal pairs.

    If the predictor learns from similarity structure rather than
    temporal structure, this should perform well. AAR showed it
    actually degrades retrieval.
    """
    print("\n--- Ablation 6.5: Similar-positives ---")
    import faiss as faiss_lib

    n = embeddings.shape[0]
    dim = embeddings.shape[1]

    # Build index and find nearest neighbors
    index = faiss_lib.IndexFlatIP(dim)
    index.add(embeddings.astype(np.float32))

    k_neighbors = 7  # match temporal window's typical yield
    print(f"  Finding {k_neighbors} nearest neighbors for {n:,} passages...")
    _, nn_indices = index.search(embeddings.astype(np.float32), k_neighbors + 1)

    # Create pairs from nearest neighbors (excluding self)
    sim_pairs = []
    for i in range(n):
        for j in nn_indices[i]:
            j = int(j)
            if j != i:
                sim_pairs.append((i, j))
    sim_pairs = list(set(sim_pairs))
    random.shuffle(sim_pairs)

    # Cap to match temporal pair count
    temporal_pairs_path = os.path.join(
        config.pairs_dir, f"temporal_pairs_w{config.temporal_window}.json"
    )
    if os.path.exists(temporal_pairs_path):
        with open(temporal_pairs_path) as f:
            temporal_count = len(json.load(f))
        sim_pairs = sim_pairs[:temporal_count]

    print(f"  Similar-positive pairs: {len(sim_pairs):,}")

    # Train
    model = train(config, sim_pairs, embeddings)

    # Evaluate
    median_sim = compute_median_similarity(embeddings)
    retrieval_fns = make_retrieval_fns(embeddings, model, config, device)

    results = {}
    for method_name in ["aar", "pam_only"]:
        method_results = evaluate_queries(
            queries, retrieval_fns[method_name], embeddings,
            config.k_values, median_sim
        )
        for k, v in method_results.items():
            results[f"sim_pos_{method_name}_{k}"] = v

    print(f"  Results: {results}")
    return results


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

ABLATIONS = {
    "shuffle": ablation_shuffle,
    "random": ablation_random,
    "bm25": ablation_bm25,
    "similar_positives": ablation_similar_positives,
}


def run(config: Config | None = None, ablation: str | None = None):
    """Run ablation experiments."""
    if config is None:
        config = Config()
    config.ensure_dirs()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load data
    print("Loading data...")
    embeddings = np.load(os.path.join(config.embeddings_dir, "embeddings.npy"))
    with open(os.path.join(config.chunks_dir, "chunks.json"), "r", encoding="utf-8") as f:
        chunks = json.load(f)
    with open(os.path.join(config.queries_dir, "eval_queries.json"), "r", encoding="utf-8") as f:
        raw_queries = json.load(f)

    # Embed queries
    from sentence_transformers import SentenceTransformer
    embed_model = SentenceTransformer(config.embedding_model)
    for q in raw_queries:
        q["query_embedding"] = embed_model.encode(
            q["question"], normalize_embeddings=True
        ).astype(np.float32)
        q["gold_indices"] = set(q["gold_indices"])

    # Run selected or all ablations
    all_results = {}
    ablations_to_run = [ablation] if ablation else list(ABLATIONS.keys())

    for abl_name in ablations_to_run:
        if abl_name not in ABLATIONS:
            print(f"Unknown ablation: {abl_name}")
            continue

        fn = ABLATIONS[abl_name]
        if abl_name == "shuffle":
            results = fn(config, chunks, embeddings, raw_queries, device)
        elif abl_name == "similar_positives":
            results = fn(config, embeddings, raw_queries, device)
        elif abl_name == "random":
            results = fn(config, embeddings, raw_queries)
        elif abl_name == "bm25":
            results = fn(config, chunks, embeddings, raw_queries)
        else:
            results = fn(config, embeddings, raw_queries)

        all_results.update(results)

    # Save (merge with existing results to avoid overwriting)
    out_path = os.path.join(config.results_dir, "ablations", "ablation_results.json")
    if os.path.exists(out_path):
        with open(out_path, "r") as f:
            existing = json.load(f)
        existing.update(all_results)
        all_results = existing
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved ablation results to {out_path}")

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ablation controls")
    parser.add_argument("--ablation", type=str, default=None,
                        choices=list(ABLATIONS.keys()),
                        help="Run a specific ablation (default: all)")
    args = parser.parse_args()
    run(ablation=args.ablation)
