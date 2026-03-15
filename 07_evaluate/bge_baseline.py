"""
BGE Raw Baseline Clustering — k=100 and k=250
Critical baseline: does raw BGE (no PAM) produce functional clusters?
"""

import numpy as np
import faiss
import json
import os
import time
from pathlib import Path

BASE = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EMB_PATH = BASE / "data" / "embeddings.npy"
CHUNKS_PATH = BASE / "data" / "chunks" / "chunks.jsonl"
RESULTS_DIR = BASE / "results"
DATA_DIR = BASE / "data"

RESULTS_DIR.mkdir(exist_ok=True)
K_VALUES = [100, 250]
N_TRAIN_SAMPLE = 1_000_000
COSINE_SAMPLE = 1000  # samples per cluster for intra-cluster cosine
N_READOUT_SAMPLES = 10
SEED = 42


def extract_metadata(chunks_path, n_total):
    """Extract book_id, position, total_chunks from JSONL without loading text."""
    print(f"Extracting metadata from {n_total:,} chunks...")
    book_ids = np.empty(n_total, dtype=np.int32)
    positions = np.empty(n_total, dtype=np.int32)
    total_chunks = np.empty(n_total, dtype=np.int32)
    t0 = time.time()
    with open(chunks_path, 'r') as f:
        for i, line in enumerate(f):
            if i % 2_000_000 == 0 and i > 0:
                print(f"  {i:,} / {n_total:,} ({i/n_total*100:.0f}%)")
            obj = json.loads(line)
            book_ids[i] = obj["book_id"]
            positions[i] = obj["position"]
            total_chunks[i] = obj["total_chunks"]
    print(f"  Done in {time.time()-t0:.0f}s")
    return book_ids, positions, total_chunks


def get_text_for_indices(chunks_path, indices):
    """Load text for specific line indices from JSONL."""
    indices_set = set(indices)
    texts = {}
    with open(chunks_path, 'r') as f:
        for i, line in enumerate(f):
            if i in indices_set:
                obj = json.loads(line)
                texts[i] = obj["text"]
                if len(texts) == len(indices_set):
                    break
    return texts


def run_kmeans(embeddings, k, n_train=N_TRAIN_SAMPLE):
    """FAISS k-means on a subsample, then full assignment."""
    n = embeddings.shape[0]
    d = embeddings.shape[1]

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
    distances = np.empty(n, dtype=np.float32)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch = np.ascontiguousarray(embeddings[start:end].astype(np.float32))
        D, I = kmeans.index.search(batch, 1)
        labels[start:end] = I[:, 0]
        distances[start:end] = D[:, 0]
        if start % 2_000_000 == 0 and start > 0:
            print(f"    {start:,} / {n:,}")
    print(f"  Assignment done in {time.time()-t0:.0f}s")

    return labels, distances, kmeans.centroids


def compute_cluster_stats(embeddings, labels, distances, centroids, book_ids, k):
    """Compute per-cluster stats: size, book diversity, cosine, dominance."""
    print(f"  Computing cluster stats for k={k}...")
    rng = np.random.RandomState(SEED)
    stats = {}

    for c in range(k):
        mask = labels == c
        cluster_size = int(mask.sum())
        cluster_books = book_ids[mask]
        unique_books = len(np.unique(cluster_books))

        # Book dominance
        book_vals, book_counts = np.unique(cluster_books, return_counts=True)
        top_idx = np.argmax(book_counts)
        top_book = int(book_vals[top_idx])
        top_book_frac = float(book_counts[top_idx]) / cluster_size

        # Top 5 books
        sorted_idx = np.argsort(-book_counts)[:5]
        top_5 = [{"book": int(book_vals[j]), "count": int(book_counts[j])}
                 for j in sorted_idx]

        # Intra-cluster cosine similarity (sample-based)
        cluster_indices = np.where(mask)[0]
        if len(cluster_indices) > COSINE_SAMPLE:
            sample_idx = rng.choice(cluster_indices, COSINE_SAMPLE, replace=False)
        else:
            sample_idx = cluster_indices

        sample_emb = np.ascontiguousarray(embeddings[sample_idx].astype(np.float32))
        # L2 normalize for cosine
        norms = np.linalg.norm(sample_emb, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        sample_normed = sample_emb / norms
        # Mean pairwise cosine = mean of (X @ X.T) off-diagonal
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
            "top_5_books": top_5,
        }

        if (c + 1) % 50 == 0:
            print(f"    {c+1}/{k} clusters done")

    return stats


def find_nearest_to_centroid(embeddings, labels, centroids, k, n_samples=N_READOUT_SAMPLES):
    """Find n closest samples to each centroid."""
    nearest = {}
    for c in range(k):
        mask = np.where(labels == c)[0]
        cluster_emb = np.ascontiguousarray(embeddings[mask].astype(np.float32))
        centroid = centroids[c:c+1].astype(np.float32)

        # Cosine similarity: normalize both
        c_norm = centroid / np.maximum(np.linalg.norm(centroid, axis=1, keepdims=True), 1e-8)
        e_norms = np.linalg.norm(cluster_emb, axis=1, keepdims=True)
        e_normed = cluster_emb / np.maximum(e_norms, 1e-8)
        sims = (e_normed @ c_norm.T).ravel()

        top_k = min(n_samples, len(sims))
        top_indices = np.argsort(-sims)[:top_k]

        nearest[c] = [(int(mask[j]), float(sims[j])) for j in top_indices]

    return nearest


def generate_readout(stats, nearest, book_ids, positions, total_chunks_arr, texts, k, threshold):
    """Generate markdown readout matching PAM format."""
    passing = sum(1 for s in stats.values() if s["books"] >= threshold)

    lines = [f"# BGE Raw Cluster Readout — k={k}\n"]
    lines.append(f"Threshold: {threshold}+ unique books | Passing: {passing}/{k} clusters\n")

    for c in range(k):
        s = stats[c]
        lines.append(f"## Cluster {c}\n")
        lines.append(
            f"**Size:** {s['size']:,} passages | "
            f"**Books:** {s['books']:,} | "
            f"**Cosine:** {s['cosine']:.4f} | "
            f"**Top book:** {s['top_book']} ({s['top_book_frac']*100:.1f}%)\n"
        )

        for rank, (idx, sim) in enumerate(nearest[c], 1):
            bid = int(book_ids[idx])
            pos = int(positions[idx])
            tc = int(total_chunks_arr[idx])
            pos_pct = pos / max(tc - 1, 1) * 100
            text = texts.get(idx, "[text unavailable]")
            # Truncate to ~200 chars
            if len(text) > 200:
                text = text[:200]

            lines.append(f"### Sample {rank} (sim={sim:.4f})")
            lines.append(f"Book {bid} | Position: {pos_pct:.1f}%")
            lines.append(f"> {text}\n")

        lines.append("---\n")

    return "\n".join(lines)


def main():
    print("=" * 60)
    print("BGE Raw Baseline Clustering")
    print("=" * 60)

    # Load embeddings
    print(f"\nLoading embeddings from {EMB_PATH}...")
    embeddings = np.load(str(EMB_PATH), mmap_mode='r')
    n, d = embeddings.shape
    print(f"  Shape: {n:,} × {d}")

    # Extract metadata
    book_ids, positions, total_chunks_arr = extract_metadata(str(CHUNKS_PATH), n)

    for k in K_VALUES:
        print(f"\n{'='*60}")
        print(f"K-MEANS k={k}")
        print(f"{'='*60}")

        # Threshold: match PAM convention
        threshold = 100 if k <= 250 else k // 2

        # K-means
        labels, distances, centroids = run_kmeans(embeddings, k)

        # Save labels and centroids
        np.save(str(DATA_DIR / f"bge_raw_labels_k{k}.npy"), labels)
        np.save(str(DATA_DIR / f"bge_raw_centroids_k{k}.npy"), centroids)
        print(f"  Saved labels and centroids")

        # Cluster stats
        stats = compute_cluster_stats(embeddings, labels, distances, centroids, book_ids, k)

        # Find nearest to centroid
        print(f"  Finding {N_READOUT_SAMPLES} nearest-to-centroid per cluster...")
        nearest = find_nearest_to_centroid(embeddings, labels, centroids, k)

        # Collect all indices needing text
        all_text_indices = []
        for c in range(k):
            all_text_indices.extend([idx for idx, _ in nearest[c]])
        print(f"  Loading text for {len(all_text_indices):,} samples...")
        texts = get_text_for_indices(str(CHUNKS_PATH), all_text_indices)

        # Generate readout
        readout = generate_readout(stats, nearest, book_ids, positions, total_chunks_arr, texts, k, threshold)
        readout_path = RESULTS_DIR / f"bge_raw_cluster_readout_k{k}.md"
        with open(readout_path, 'w') as f:
            f.write(readout)
        print(f"  Saved readout: {readout_path}")

        # Save stats JSON
        passing = sum(1 for s in stats.values() if s["books"] >= threshold)
        stats_json = {
            "k": k,
            "threshold": threshold,
            "passing": passing,
            "total": k,
            "mean_cosine": round(np.mean([s["cosine"] for s in stats.values()]), 4),
            "mean_books": round(np.mean([s["books"] for s in stats.values()]), 1),
            "mean_dominance": round(np.mean([s["top_book_frac"] for s in stats.values()]), 4),
            "clusters": {str(c): stats[c] for c in range(k)},
        }
        stats_path = RESULTS_DIR / f"bge_raw_cluster_stats_k{k}.json"
        with open(stats_path, 'w') as f:
            json.dump(stats_json, f, indent=2)
        print(f"  Saved stats: {stats_path}")

        # Print summary
        print(f"\n  === k={k} Summary ===")
        print(f"  Passing: {passing}/{k} (threshold={threshold} books)")
        print(f"  Mean cosine: {stats_json['mean_cosine']:.4f}")
        print(f"  Mean books/cluster: {stats_json['mean_books']:.0f}")
        print(f"  Mean dominance: {stats_json['mean_dominance']:.4f}")

    print(f"\n{'='*60}")
    print("Done!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
