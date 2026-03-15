"""Concept discovery: cluster passages in PAM association space vs. original space.

Tests whether PAM-transformed embeddings group passages by narrative/structural
concepts rather than topical similarity. Uses existing 250-novel model and data.

Diagnostic: If top "interesting" clusters have mean original cosine < 0.3 and
50+ books, the association space is grouping dissimilar passages from across the
corpus — strong evidence for emergent narrative concepts.

Usage:
    python 05_cluster/cluster.py
"""

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "04_train"))

import os
import gc
import json
import time
import numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans

from utils.config import Config
from model import AssociationMLP


def load_v23_data(config):
    """Load embeddings and chunk metadata. Adapt paths for your setup."""
    import numpy as np, json
    emb = np.load(os.path.join(config.embeddings_dir, "embeddings.npy"))
    chunks_path = os.path.join(config.chunks_dir, "chunks.json")
    with open(chunks_path) as f:
        chunks = json.load(f)
    chunk_ids = [c["chunk_id"] for c in chunks]
    return chunks, emb, chunk_ids


def load_book_ids_for_scale(config, n, total):
    """Load book ID list for a given scale point."""
    path = os.path.join(config.base_dir, "data", "corpus", f"book_ids_{n}.txt")
    if os.path.exists(path):
        with open(path) as f:
            return [int(line.strip()) for line in f if line.strip()]
    return []


def build_novel_subset_v23(chunks, embeddings, chunk_ids, book_ids):
    """Filter embeddings to a subset of books."""
    import numpy as np
    book_set = set(book_ids)
    indices = [i for i, c in enumerate(chunks) if c.get("book_id") in book_set]
    return np.array(indices), embeddings[indices]


K = 50          # number of clusters
TOP_N = 10      # passages closest to each centroid
TOP_INTERESTING = 10  # clusters to highlight


def transform_embeddings_cpu(model, embeddings, batch_size=10_000):
    """Transform embeddings through PAM model on CPU in batches."""
    model.eval()
    n = len(embeddings)
    z_list = []
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch = torch.tensor(embeddings[start:end], dtype=torch.float32)
        with torch.no_grad():
            z = model(batch)
        z_list.append(z.numpy())
        if (start // batch_size) % 10 == 0:
            print(f"  Transformed {end:,}/{n:,}")
    return np.concatenate(z_list, axis=0).astype(np.float32)


def cluster_and_analyze(embeddings, original_embeddings, chunks, space_name):
    """Run K-means and compute per-cluster metadata.

    Args:
        embeddings: vectors to cluster on (may be PAM-transformed or original)
        original_embeddings: always the untransformed BGE embeddings (for
            computing intra-cluster cosine similarity diagnostic)
        chunks: list of chunk dicts with text/metadata
    Returns:
        list of cluster dicts with metadata + top passages
    """
    n = len(embeddings)
    print(f"\n  K-means (k={K}) on {n:,} vectors ({space_name}) ...")
    t0 = time.time()
    km = MiniBatchKMeans(
        n_clusters=K,
        batch_size=4096,
        n_init=3,
        random_state=42,
    )
    labels = km.fit_predict(embeddings)
    centroids = km.cluster_centers_
    elapsed = time.time() - t0
    print(f"  Clustering complete in {elapsed:.1f}s")

    # Per-cluster analysis
    cluster_results = []
    for c in range(K):
        mask = labels == c
        indices = np.where(mask)[0]
        size = len(indices)
        if size == 0:
            continue

        # Book diversity
        book_ids = set()
        for idx in indices:
            book_ids.add(chunks[idx]["book_id"])
        n_books = len(book_ids)

        # Mean intra-cluster cosine similarity in ORIGINAL embedding space
        # Sample if cluster is large to keep runtime reasonable
        orig_vecs = original_embeddings[indices]
        if size > 500:
            sample_idx = np.random.RandomState(c).choice(size, 500, replace=False)
            sample_vecs = orig_vecs[sample_idx]
        else:
            sample_vecs = orig_vecs

        # Cosine similarity (vectors are L2-normalized, so dot product = cosine)
        sim_matrix = sample_vecs @ sample_vecs.T
        # Mean of upper triangle (excluding diagonal)
        n_s = len(sample_vecs)
        triu_idx = np.triu_indices(n_s, k=1)
        mean_orig_cos = float(np.mean(sim_matrix[triu_idx]))

        # Top N passages closest to centroid
        centroid = centroids[c].reshape(1, -1)
        cluster_vecs = embeddings[indices]
        dists = cluster_vecs @ centroid.T  # inner product (higher = closer)
        top_local = np.argsort(-dists.ravel())[:TOP_N]
        top_global = indices[top_local]

        top_passages = []
        for gidx in top_global:
            ch = chunks[gidx]
            top_passages.append({
                "text": ch["text"][:500],  # truncate for readability
                "book_title": ch["book_title"],
                "author": ch.get("author", ""),
                "book_id": ch["book_id"],
                "position": ch["position"],
                "total_chunks": ch["total_chunks"],
                "position_pct": round(ch["position"] / max(ch["total_chunks"], 1), 3),
            })

        cluster_results.append({
            "cluster_id": c,
            "size": size,
            "n_books": n_books,
            "mean_original_cosine": round(mean_orig_cos, 4),
            "top_passages": top_passages,
        })

    # Sort by cluster_id for consistent output
    cluster_results.sort(key=lambda x: x["cluster_id"])
    return cluster_results


def rank_by_interestingness(clusters):
    """Rank clusters: low original cosine + high book diversity = interesting.

    Score = book_diversity_ratio * (1 - mean_original_cosine)
    where book_diversity_ratio = n_books / max_possible_books (250)
    """
    max_books = 250
    scored = []
    for cl in clusters:
        diversity = cl["n_books"] / max_books
        dissimilarity = 1.0 - cl["mean_original_cosine"]
        score = diversity * dissimilarity
        scored.append({**cl, "interestingness_score": round(score, 4)})
    scored.sort(key=lambda x: -x["interestingness_score"])
    return scored


def main():
    config = Config()
    config.ensure_dirs()
    np.random.seed(42)

    results_dir = os.path.join(config.base_dir, "results", "concept_discovery")
    os.makedirs(results_dir, exist_ok=True)

    # --- Load 250-novel subset ---
    print("Loading v2.3 corpus ...")
    all_chunks, all_embeddings, all_chunk_ids = load_v23_data(config)
    total_novels = len(set(c["book_id"] for c in all_chunks))
    print(f"Total corpus: {total_novels} novels, {len(all_chunks):,} chunks")

    print("\nBuilding 250-novel subset ...")
    book_ids = load_book_ids_for_scale(config, 250, total_novels)
    global_indices, subset_emb = build_novel_subset_v23(
        all_chunks, all_embeddings, book_ids
    )
    subset_chunks = [all_chunks[g] for g in global_indices]
    n_novels = len(set(c["book_id"] for c in subset_chunks))
    print(f"Subset: {len(subset_chunks):,} chunks from {n_novels} novels")

    # Free full corpus
    del all_embeddings, all_chunk_ids
    gc.collect()

    # --- Load PAM model ---
    model_path = os.path.join(config.models_dir, "pam_predictor.pt")
    print(f"\nLoading PAM model: {model_path}")
    model = AssociationMLP(
        embedding_dim=config.embedding_dim,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
    )
    model.load_state_dict(
        torch.load(model_path, map_location="cpu", weights_only=True)
    )
    model.eval()
    print(f"  {sum(p.numel() for p in model.parameters()):,} params")

    # --- Transform embeddings through PAM ---
    print("\nTransforming embeddings through PAM model (CPU) ...")
    t0 = time.time()
    z_all = transform_embeddings_cpu(model, subset_emb, batch_size=10_000)
    print(f"  Done: {z_all.shape} in {time.time()-t0:.0f}s")

    del model
    gc.collect()

    # =====================================================================
    # Cluster in PAM association space
    # =====================================================================
    print(f"\n{'='*70}")
    print("CLUSTERING IN PAM ASSOCIATION SPACE")
    print(f"{'='*70}")
    pam_clusters = cluster_and_analyze(
        z_all, subset_emb, subset_chunks, "PAM space"
    )

    # =====================================================================
    # Cluster in original embedding space (baseline)
    # =====================================================================
    print(f"\n{'='*70}")
    print("CLUSTERING IN ORIGINAL EMBEDDING SPACE (BASELINE)")
    print(f"{'='*70}")
    cosine_clusters = cluster_and_analyze(
        subset_emb, subset_emb, subset_chunks, "cosine space"
    )

    # =====================================================================
    # Rank PAM clusters by interestingness
    # =====================================================================
    print(f"\n{'='*70}")
    print("RANKING PAM CLUSTERS BY INTERESTINGNESS")
    print(f"{'='*70}")
    interesting = rank_by_interestingness(pam_clusters)
    top_interesting = interesting[:TOP_INTERESTING]

    # =====================================================================
    # Save results
    # =====================================================================
    pam_path = os.path.join(results_dir, "pam_clusters.json")
    cos_path = os.path.join(results_dir, "cosine_clusters.json")
    int_path = os.path.join(results_dir, "interesting_clusters.json")

    with open(pam_path, "w") as f:
        json.dump(pam_clusters, f, indent=2)
    print(f"\nPAM clusters    -> {pam_path}")

    with open(cos_path, "w") as f:
        json.dump(cosine_clusters, f, indent=2)
    print(f"Cosine clusters -> {cos_path}")

    with open(int_path, "w") as f:
        json.dump(top_interesting, f, indent=2)
    print(f"Interesting     -> {int_path}")

    # =====================================================================
    # Print summary
    # =====================================================================
    print(f"\n{'='*70}")
    print("AGGREGATE COMPARISON")
    print(f"{'='*70}")

    pam_cosines = [c["mean_original_cosine"] for c in pam_clusters]
    cos_cosines = [c["mean_original_cosine"] for c in cosine_clusters]
    pam_books = [c["n_books"] for c in pam_clusters]
    cos_books = [c["n_books"] for c in cosine_clusters]

    print(f"  {'Metric':<35} {'PAM space':>12} {'Cosine space':>14}")
    print(f"  {'-'*63}")
    print(f"  {'Mean intra-cluster orig cosine':<35} {np.mean(pam_cosines):>12.4f} {np.mean(cos_cosines):>14.4f}")
    print(f"  {'Median intra-cluster orig cosine':<35} {np.median(pam_cosines):>12.4f} {np.median(cos_cosines):>14.4f}")
    print(f"  {'Mean book diversity (n_books)':<35} {np.mean(pam_books):>12.1f} {np.mean(cos_books):>14.1f}")
    print(f"  {'Clusters with orig cosine < 0.3':<35} {sum(1 for c in pam_cosines if c < 0.3):>12d} {sum(1 for c in cos_cosines if c < 0.3):>14d}")
    print(f"  {'Clusters with 50+ books':<35} {sum(1 for b in pam_books if b >= 50):>12d} {sum(1 for b in cos_books if b >= 50):>14d}")

    # --- Print top 10 interesting clusters ---
    print(f"\n{'='*70}")
    print(f"TOP {TOP_INTERESTING} INTERESTING PAM CLUSTERS")
    print(f"(low original cosine + high book diversity)")
    print(f"{'='*70}")

    for rank, cl in enumerate(top_interesting, 1):
        print(f"\n  Cluster {cl['cluster_id']}: "
              f"{cl['size']:,} passages, {cl['n_books']} books, "
              f"mean original cosine: {cl['mean_original_cosine']:.3f}, "
              f"score: {cl['interestingness_score']:.4f}")
        print(f"  ---")
        for i, p in enumerate(cl["top_passages"][:5], 1):
            text_preview = p["text"][:200].replace("\n", " ")
            pos_pct = f"{p['position_pct']*100:.0f}%"
            print(f"  [{i}] \"{text_preview}...\"")
            print(f"      — {p['book_title']} ({p['author']}), "
                  f"pos {p['position']}/{p['total_chunks']} ({pos_pct})")


if __name__ == "__main__":
    main()
