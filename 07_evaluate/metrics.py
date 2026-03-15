"""Evaluation metrics: TAR@k, CDR@k, and multi-hop reachability.

TAR@k  -- Temporal Association Recall at k: fraction of temporally
           associated passages in the top-k results.
CDR@k  -- Cross-Distance Recall at k: TAR@k restricted to associated
           passages that are distant in embedding space (cosine below
           corpus median). This is where PAM's value lives.
Reachability -- passages reachable via multi-hop PAM traversal that
               are not reachable via cosine at any depth in top-100.
"""

import numpy as np
from collections import defaultdict


def tar_at_k(retrieved_indices: np.ndarray, gold_indices: set[int],
             k: int) -> float:
    """Temporal Association Recall at k.

    What fraction of gold (temporally associated) passages appear in top-k?
    """
    if not gold_indices:
        return 0.0
    retrieved_set = set(int(i) for i in retrieved_indices[:k])
    return len(retrieved_set & gold_indices) / len(gold_indices)


def cdr_at_k(retrieved_indices: np.ndarray, gold_indices: set[int],
             query_embedding: np.ndarray, all_embeddings: np.ndarray,
             k: int, median_sim: float) -> float:
    """Cross-Distance Recall at k.

    Like TAR@k but only counts gold passages whose cosine similarity
    to the query is below the corpus median. These are the "hard"
    associations that cosine similarity struggles with.
    """
    if not gold_indices:
        return 0.0

    # Filter gold to only distant passages
    distant_gold = set()
    for gidx in gold_indices:
        sim = float(np.dot(query_embedding, all_embeddings[gidx]))
        if sim < median_sim:
            distant_gold.add(gidx)

    if not distant_gold:
        return 0.0

    retrieved_set = set(int(i) for i in retrieved_indices[:k])
    return len(retrieved_set & distant_gold) / len(distant_gold)


def compute_median_similarity(embeddings: np.ndarray,
                              sample_size: int = 10000) -> float:
    """Estimate corpus median pairwise cosine similarity by sampling."""
    n = embeddings.shape[0]
    if n < 2:
        return 0.0
    rng = np.random.default_rng(42)
    idx_a = rng.integers(0, n, size=sample_size)
    idx_b = rng.integers(0, n, size=sample_size)
    sims = np.sum(embeddings[idx_a] * embeddings[idx_b], axis=1)
    return float(np.median(sims))


def multi_hop_reachability(start_idx: int, model, embeddings: np.ndarray,
                           faiss_index, max_hops: int = 3,
                           k_per_hop: int = 20,
                           cosine_top_k: int = 100,
                           device: str = "cuda") -> dict:
    """Measure passages reachable via PAM hops but not via cosine.

    Returns dict with:
      - cosine_reachable: set of indices reachable by cosine top-100
      - pam_reachable: set of indices reachable by multi-hop PAM
      - pam_only: passages reachable only via PAM
    """
    import torch

    # Cosine reachable (flat top-K)
    query = embeddings[start_idx:start_idx + 1].astype(np.float32)
    _, cos_indices = faiss_index.search(query, cosine_top_k)
    cosine_reachable = set(int(i) for i in cos_indices[0])

    # PAM multi-hop
    pam_reachable = set()
    frontier = {start_idx}

    for hop in range(max_hops):
        next_frontier = set()
        for idx in frontier:
            emb = torch.tensor(
                embeddings[idx:idx + 1], dtype=torch.float32
            ).to(device)
            with torch.no_grad():
                assoc_query = model(emb)
            assoc_np = assoc_query.cpu().numpy().astype(np.float32)
            _, hop_indices = faiss_index.search(assoc_np, k_per_hop)
            for hi in hop_indices[0]:
                hi = int(hi)
                if hi != start_idx and hi not in pam_reachable:
                    next_frontier.add(hi)
        pam_reachable.update(next_frontier)
        frontier = next_frontier

    pam_only = pam_reachable - cosine_reachable

    return {
        "cosine_reachable": cosine_reachable,
        "pam_reachable": pam_reachable,
        "pam_only": pam_only,
        "num_cosine": len(cosine_reachable),
        "num_pam": len(pam_reachable),
        "num_pam_only": len(pam_only),
    }


def evaluate_queries(queries: list[dict], retrieval_fn,
                     embeddings: np.ndarray, k_values: list[int],
                     median_sim: float | None = None) -> dict:
    """Evaluate a retrieval function on a set of queries.

    Each query dict should have:
      - query_embedding: np.ndarray (1024d)
      - gold_indices: set of int (temporally associated passage indices)

    Returns dict with mean TAR@k and CDR@k for each k.
    """
    if median_sim is None:
        median_sim = compute_median_similarity(embeddings)

    results = defaultdict(list)

    for q in queries:
        query_emb = q["query_embedding"]
        gold = q["gold_indices"]

        retrieved_indices, _ = retrieval_fn(query_emb)

        for k in k_values:
            results[f"TAR@{k}"].append(
                tar_at_k(retrieved_indices, gold, k)
            )
            results[f"CDR@{k}"].append(
                cdr_at_k(retrieved_indices, gold, query_emb, embeddings,
                         k, median_sim)
            )

    return {key: float(np.mean(vals)) for key, vals in results.items()}
