"""
Context-Enriched BGE Baseline — window-averaged embeddings + k-means
Addresses the critique: "maybe PAM just learns to average nearby embeddings"
If context-averaging produces less specialised clusters than PAM, the learned
transformation is doing something beyond simple temporal smoothing.
"""

import os
import numpy as np
import faiss
import json
import time
from pathlib import Path

BASE = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EMB_PATH = BASE / "data" / "embeddings.npy"
CHUNKS_PATH = BASE / "data" / "chunks" / "chunks.jsonl"
RESULTS_DIR = BASE / "results"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

WINDOW = 15  # same temporal window as PAM pair generation
K_VALUES = [100, 250]
N_TRAIN_SAMPLE = 1_000_000
COSINE_SAMPLE = 1000
SEED = 42


def extract_book_ids(chunks_path, n_total):
    """Extract book_id for each chunk to detect book boundaries."""
    print(f"Extracting book_ids from {n_total:,} chunks...")
    book_ids = np.empty(n_total, dtype=np.int32)
    t0 = time.time()
    with open(chunks_path, 'r') as f:
        for i, line in enumerate(f):
            if i % 2_000_000 == 0 and i > 0:
                print(f"  {i:,} / {n_total:,} ({i/n_total*100:.0f}%)")
            obj = json.loads(line)
            book_ids[i] = obj["book_id"]
    print(f"  Done in {time.time()-t0:.0f}s")
    return book_ids


def compute_context_averaged_embeddings(embeddings, book_ids, window):
    """
    For each chunk, average it with its w neighbors on each side,
    respecting book boundaries. Returns new L2-normalized embedding matrix.
    """
    n, d = embeddings.shape
    print(f"Computing context-averaged embeddings (window={window})...")
    t0 = time.time()

    # Find book ranges
    book_ranges = {}
    current_start = 0
    current_book = int(book_ids[0])
    for i in range(1, n):
        bid = int(book_ids[i])
        if bid != current_book:
            book_ranges[current_book] = (current_start, i)
            current_start = i
            current_book = bid
    book_ranges[current_book] = (current_start, n)
    print(f"  Found {len(book_ranges)} books")

    avg_emb = np.zeros((n, d), dtype=np.float32)
    books_done = 0

    for bid, (start, end) in book_ranges.items():
        book_len = end - start
        book_emb = np.array(embeddings[start:end], dtype=np.float64)

        # Cumulative sum for O(1) window averaging
        cumsum = np.zeros((book_len + 1, d), dtype=np.float64)
        for i in range(book_len):
            cumsum[i+1] = cumsum[i] + book_emb[i]

        for i in range(book_len):
            lo = max(0, i - window)
            hi = min(book_len - 1, i + window)
            count = hi - lo + 1
            avg = (cumsum[hi+1] - cumsum[lo]) / count
            avg_emb[start + i] = avg.astype(np.float32)

        books_done += 1
        if books_done % 500 == 0:
            print(f"  {books_done}/{len(book_ranges)} books processed")

    # L2 normalize
    norms = np.linalg.norm(avg_emb, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    avg_emb = avg_emb / norms

    print(f"  Context averaging done in {time.time()-t0:.0f}s")
    return avg_emb


def run_kmeans(embeddings, k, n_train=N_TRAIN_SAMPLE):
    """FAISS k-means on a subsample, then full assignment."""
    n, d = embeddings.shape
    rng = np.random.RandomState(SEED)
    train_idx = rng.choice(n, size=min(n_train, n), replace=False)
    train_data = np.ascontiguousarray(embeddings[train_idx].astype(np.float32))

    print(f"  Training k-means (k={k}) on {len(train_idx):,} samples...")
    t0 = time.time()
    kmeans = faiss.Kmeans(d, k, niter=20, verbose=True, seed=SEED)
    kmeans.train(train_data)
    print(f"  Training done in {time.time()-t0:.0f}s")

    print(f"  Assigning all {n:,} vectors...")
    t0 = time.time()
    batch_size = 500_000
    labels = np.empty(n, dtype=np.int32)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch = np.ascontiguousarray(embeddings[start:end].astype(np.float32))
        D, I = kmeans.index.search(batch, 1)
        labels[start:end] = I[:, 0]
    print(f"  Assignment done in {time.time()-t0:.0f}s")

    return labels, kmeans.centroids


def compute_cluster_stats(embeddings, labels, book_ids, k):
    """Compute per-cluster stats matching BGE raw baseline format."""
    print(f"  Computing cluster stats for k={k}...")
    rng = np.random.RandomState(SEED)
    stats = {}

    for c in range(k):
        mask = labels == c
        cluster_size = int(mask.sum())
        cluster_books = book_ids[mask]
        unique_books = len(np.unique(cluster_books))

        book_vals, book_counts = np.unique(cluster_books, return_counts=True)
        top_idx = np.argmax(book_counts)
        top_book = int(book_vals[top_idx])
        top_book_frac = float(book_counts[top_idx]) / cluster_size

        # Intra-cluster cosine (sample-based)
        cluster_indices = np.where(mask)[0]
        if len(cluster_indices) > COSINE_SAMPLE:
            sample_idx = rng.choice(cluster_indices, COSINE_SAMPLE, replace=False)
        else:
            sample_idx = cluster_indices

        sample_emb = np.ascontiguousarray(embeddings[sample_idx].astype(np.float32))
        nrm = np.linalg.norm(sample_emb, axis=1, keepdims=True)
        nrm = np.maximum(nrm, 1e-8)
        sample_normed = sample_emb / nrm
        sim_matrix = sample_normed @ sample_normed.T
        n_s = len(sample_idx)
        if n_s > 1:
            cosine_mean = float((sim_matrix.sum() - np.trace(sim_matrix)) / (n_s * (n_s - 1)))
        else:
            cosine_mean = 1.0

        stats[c] = {
            "cluster": c,
            "size": cluster_size,
            "books": unique_books,
            "cosine": round(cosine_mean, 4),
            "top_book": top_book,
            "top_book_frac": round(top_book_frac, 4),
        }

        if (c + 1) % 50 == 0:
            print(f"    {c+1}/{k} clusters done")

    return stats


def main():
    print("=" * 60)
    print("Context-Enriched BGE Baseline (window-averaged)")
    print(f"Window size: {WINDOW} (matches PAM training)")
    print("=" * 60)

    print(f"\nLoading embeddings from {EMB_PATH}...")
    embeddings = np.load(str(EMB_PATH), mmap_mode='r')
    n, d = embeddings.shape
    print(f"  Shape: {n:,} x {d}")

    book_ids = extract_book_ids(str(CHUNKS_PATH), n)
    avg_emb = compute_context_averaged_embeddings(embeddings, book_ids, WINDOW)

    for k in K_VALUES:
        print(f"\n{'='*60}")
        print(f"K-MEANS k={k} (context-enriched)")
        print(f"{'='*60}")

        threshold = 100 if k <= 250 else k // 2
        labels, centroids = run_kmeans(avg_emb, k)
        stats = compute_cluster_stats(avg_emb, labels, book_ids, k)

        passing = sum(1 for s in stats.values() if s["books"] >= threshold)
        mean_cosine = round(np.mean([s["cosine"] for s in stats.values()]), 4)
        mean_books = round(np.mean([s["books"] for s in stats.values()]), 1)
        mean_dominance = round(np.mean([s["top_book_frac"] for s in stats.values()]), 4)

        stats_json = {
            "k": k,
            "window": WINDOW,
            "threshold": threshold,
            "passing": passing,
            "total": k,
            "mean_cosine": mean_cosine,
            "mean_books": mean_books,
            "mean_dominance": mean_dominance,
            "clusters": {str(c): stats[c] for c in range(k)},
        }
        stats_path = RESULTS_DIR / f"context_enriched_baseline_k{k}.json"
        with open(stats_path, 'w') as f:
            json.dump(stats_json, f, indent=2)
        print(f"  Saved: {stats_path}")

        print(f"\n  === k={k} Summary ===")
        print(f"  Passing: {passing}/{k} (threshold={threshold} books)")
        print(f"  Mean cosine: {mean_cosine:.4f}")
        print(f"  Mean books/cluster: {mean_books:.0f}")
        print(f"  Mean dominance: {mean_dominance:.4f}")

    print(f"\n{'='*60}")
    print("Done!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
