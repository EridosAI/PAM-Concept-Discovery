"""Main scale degradation curves.

Measures TAR@k and CDR@k at increasing corpus sizes for cosine,
AAR-reranked, and PAM-only retrieval. This is the primary experiment:
does PAM degrade less than cosine as the corpus grows?

Scale points are in NOVEL counts (10, 25, 50, 100, 200). Novel selection
for subsets: sorted by Gutenberg ID, take first N (deterministic, seed 42).

Usage:
    python 07_evaluate/scale_evaluation.py
"""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), '04_train'))

import os
import json
import time
import random
import numpy as np
import torch
import faiss
import pandas as pd
from tqdm import tqdm

from utils.config import Config
from utils.faiss_utils import build_index, search
from metrics import (
    evaluate_queries, compute_median_similarity
)
from train import AssociationMLP, bidirectional_score


def load_model(config: Config, device: str = "cuda",
               model_path: str | None = None) -> AssociationMLP:
    """Load trained PAM predictor."""
    model = AssociationMLP(
        embedding_dim=config.embedding_dim,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
    )
    if model_path is None:
        model_path = os.path.join(config.models_dir, "pam_predictor.pt")
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    return model


def build_novel_subset(chunks: list[dict], embeddings: np.ndarray,
                       num_novels: int, seed: int = 42) -> tuple:
    """Select first N novels (sorted by Gutenberg ID) and return their chunks.

    Novel selection: sorted by Gutenberg ID, take first N.
    This is deterministic and reproducible.

    Returns (subset_indices, subset_embeddings, selected_book_ids).
    """
    # Get unique book IDs sorted by Gutenberg ID
    book_ids = sorted(set(c["book_id"] for c in chunks))
    selected_books = set(book_ids[:num_novels])

    # Gather indices for selected books
    selected_indices = [
        i for i, c in enumerate(chunks) if c["book_id"] in selected_books
    ]
    selected_indices = sorted(selected_indices)
    subset_embeddings = embeddings[selected_indices].astype(np.float32)

    return selected_indices, subset_embeddings, selected_books


def make_retrieval_fns(subset_embeddings: np.ndarray,
                       model: AssociationMLP,
                       config: Config,
                       device: str = "cuda"):
    """Create cosine, AAR-reranked, and PAM-only retrieval functions.

    Each returns (indices_in_subset, scores).
    """
    index = build_index(subset_embeddings, use_gpu=False)
    max_k = max(config.k_values)

    def cosine_retrieve(query_emb):
        q = query_emb.astype(np.float32).reshape(1, -1)
        distances, indices = index.search(q, max_k)
        return indices[0], distances[0]

    def aar_retrieve(query_emb):
        """Cosine + PAM blend (alpha weighting)."""
        q = query_emb.astype(np.float32).reshape(1, -1)
        distances, indices = index.search(q, config.initial_k)
        cos_scores = distances[0]
        candidate_indices = indices[0]

        q_tensor = torch.tensor(q, dtype=torch.float32).to(device)
        cand_embs = torch.tensor(
            subset_embeddings[candidate_indices], dtype=torch.float32
        ).to(device)

        if config.bidirectional:
            pam_scores = bidirectional_score(model, q_tensor, cand_embs)
        else:
            with torch.no_grad():
                assoc_q = model(q_tensor)
                pam_scores = torch.mm(assoc_q, cand_embs.t()).squeeze(0)
        pam_scores = pam_scores.cpu().numpy()

        combined = (1 - config.alpha) * cos_scores + config.alpha * pam_scores
        reranked = np.argsort(-combined)
        return candidate_indices[reranked][:max_k], combined[reranked][:max_k]

    def pam_only_retrieve(query_emb):
        """Pure PAM retrieval: query through predictor, rank by association."""
        q = query_emb.astype(np.float32).reshape(1, -1)
        q_tensor = torch.tensor(q, dtype=torch.float32).to(device)

        with torch.no_grad():
            assoc_q = model(q_tensor)
        assoc_np = assoc_q.cpu().numpy().astype(np.float32)

        distances, indices = index.search(assoc_np, max_k)
        return indices[0], distances[0]

    return {
        "cosine": cosine_retrieve,
        "aar": aar_retrieve,
        "pam_only": pam_only_retrieve,
    }


def remap_queries_to_subset(queries: list[dict], subset_indices: list[int],
                            embeddings: np.ndarray) -> list[dict]:
    """Remap query gold indices to the subset index space.

    Only keeps queries whose source book is in the subset.
    """
    global_to_local = {g: l for l, g in enumerate(subset_indices)}

    remapped = []
    for q in queries:
        local_gold = set()
        for gidx in q["gold_indices"]:
            if gidx in global_to_local:
                local_gold.add(global_to_local[gidx])

        if not local_gold:
            continue

        source_global = q.get("source_global_idx")
        if source_global is not None and source_global in global_to_local:
            query_emb = embeddings[source_global]
        else:
            query_emb = q["query_embedding"]

        remapped.append({
            "query_embedding": query_emb,
            "gold_indices": local_gold,
            "question": q.get("question", ""),
            "source_book_id": q.get("source_book_id"),
        })

    return remapped


def run(config: Config | None = None):
    """Run the full scale degradation evaluation."""
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

    chunk_id_to_idx = {}
    with open(os.path.join(config.embeddings_dir, "chunk_ids.json"), "r", encoding="utf-8") as f:
        chunk_ids = json.load(f)
    for i, cid in enumerate(chunk_ids):
        chunk_id_to_idx[cid] = i

    # Embed queries
    from sentence_transformers import SentenceTransformer
    embed_model = SentenceTransformer(config.embedding_model)

    for q in raw_queries:
        q["query_embedding"] = embed_model.encode(
            q["question"], normalize_embeddings=True
        ).astype(np.float32)
        q["gold_indices"] = set(q["gold_indices"])
        source_cid = q.get("source_chunk_id")
        if source_cid and source_cid in chunk_id_to_idx:
            q["source_global_idx"] = chunk_id_to_idx[source_cid]

    model = load_model(config, device)

    total_novels = len(set(c["book_id"] for c in chunks))
    print(f"Corpus: {len(chunks):,} chunks from {total_novels} novels")

    # Run at each scale point (novel counts)
    all_results = []
    per_query_results = []

    for num_novels in config.scale_points:
        if num_novels > total_novels:
            print(f"\n[SKIP] {num_novels} novels > available {total_novels}")
            continue

        print(f"\n{'='*60}")
        print(f"Scale: {num_novels} novels")
        print(f"{'='*60}")

        subset_indices, subset_embs, selected_books = build_novel_subset(
            chunks, embeddings, num_novels, config.random_seed
        )
        print(f"  Subset: {len(subset_indices):,} chunks from {len(selected_books)} novels")

        queries = remap_queries_to_subset(
            raw_queries, subset_indices, embeddings
        )
        print(f"  Queries with gold in subset: {len(queries)}")
        if len(queries) < 10:
            print("  [WARN] Too few queries, skipping scale point")
            continue

        median_sim = compute_median_similarity(subset_embs)
        print(f"  Median cosine similarity: {median_sim:.4f}")

        retrieval_fns = make_retrieval_fns(
            subset_embs, model, config, device
        )

        for method_name, fn in retrieval_fns.items():
            t0 = time.time()
            results = evaluate_queries(
                queries, fn, subset_embs, config.k_values, median_sim
            )
            elapsed = time.time() - t0

            row = {
                "scale_novels": num_novels,
                "scale_chunks": len(subset_indices),
                "method": method_name,
                "num_queries": len(queries),
                "median_sim": median_sim,
                "eval_time_s": elapsed,
                **results,
            }
            all_results.append(row)
            print(f"  {method_name}: {results}")

    # Save results
    df = pd.DataFrame(all_results)
    out_csv = os.path.join(config.results_dir, "scale_curves", "scale_results.csv")
    df.to_csv(out_csv, index=False)

    out_json = os.path.join(config.results_dir, "scale_curves", "scale_results.json")
    with open(out_json, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nSaved results to {out_csv}")
    return df


if __name__ == "__main__":
    run()
