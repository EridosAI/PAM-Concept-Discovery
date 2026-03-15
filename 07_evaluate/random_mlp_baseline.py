#!/usr/bin/env python3
"""Generate cluster readout for Random MLP baseline.

Replicates Phase 0 (chunk + embed) then Phase 2b (random MLP transform + k-means)
from the validation pipeline, then generates a full 100-cluster readout.

Run: python -u random_mlp_readout.py 2>&1 | tee /dev/shm/workspace/readout_log.txt
"""

import os
import re
import json
import time
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict, Counter

# ============================================================================
# Configuration (MUST match validation_pipeline.py exactly)
# ============================================================================

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(_REPO_ROOT, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
RESULTS_DIR = os.path.join(_REPO_ROOT, "results")

MANIFEST_PATH = os.path.join(DATA_DIR, "corpus_manifest.jsonl")
BOOK_IDS_PATH = os.path.join(DATA_DIR, "book_ids_all.txt")
CHUNKS_PATH = os.path.join(DATA_DIR, "chunks", "chunks.jsonl")
EMBEDDINGS_PATH = os.path.join(DATA_DIR, "embeddings", "embeddings.npy")
SHARD_DIR = os.path.join(DATA_DIR, "embedding_shards")
METADATA_PATH = os.path.join(DATA_DIR, "passage_metadata.npz")

EMBEDDING_MODEL_NAME = "BAAI/bge-large-en-v1.5"
EMBEDDING_DIM = 1024

CHUNK_SIZE = 50
CHUNK_OVERLAP = 15

NUM_LAYERS = 8
HIDDEN_DIM = 2048

COSINE_SAMPLE_SIZE = 2000


def timestamp():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ============================================================================
# Model definition (identical to validation_pipeline.py)
# ============================================================================

class AssociationMLP(nn.Module):
    def __init__(self, embedding_dim=EMBEDDING_DIM, hidden_dim=HIDDEN_DIM,
                 num_layers=NUM_LAYERS):
        super().__init__()
        layers = []
        layers.append(nn.Linear(embedding_dim, hidden_dim))
        layers.append(nn.LayerNorm(hidden_dim))
        layers.append(nn.GELU())
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.GELU())
        layers.append(nn.Linear(hidden_dim, embedding_dim))
        self.net = nn.Sequential(*layers)
        self.residual_weight = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        transformed = self.net(x)
        alpha = torch.sigmoid(self.residual_weight)
        out = alpha * x + (1 - alpha) * transformed
        return F.normalize(out, dim=-1)


# ============================================================================
# Gutenberg text cleaning (identical to validation_pipeline.py)
# ============================================================================

def clean_gutenberg_text(text):
    start_markers = [
        "*** START OF THE PROJECT GUTENBERG",
        "*** START OF THIS PROJECT GUTENBERG",
        "*END*THE SMALL PRINT",
    ]
    start_idx = 0
    for marker in start_markers:
        idx = text.find(marker)
        if idx != -1:
            nl = text.find("\n", idx)
            if nl != -1:
                start_idx = nl + 1
            break

    end_markers = [
        "*** END OF THE PROJECT GUTENBERG",
        "*** END OF THIS PROJECT GUTENBERG",
        "End of the Project Gutenberg",
        "End of Project Gutenberg",
    ]
    end_idx = len(text)
    for marker in end_markers:
        idx = text.find(marker)
        if idx != -1:
            end_idx = idx
            break

    cleaned = text[start_idx:end_idx].strip()
    cleaned = re.sub(r"\r\n", "\n", cleaned)
    cleaned = re.sub(r"[^\x09\x0a\x20-\x7e\x80-\ufffd]", "", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned


def chunk_text_by_tokens(text, tokenizer, chunk_size=50, overlap=15):
    encoding = tokenizer(text, return_offsets_mapping=True,
                         add_special_tokens=False)
    tokens = encoding["input_ids"]
    offsets = encoding["offset_mapping"]

    if not tokens:
        return []

    if len(tokens) <= chunk_size:
        return [{"text": text.strip(), "char_start": 0, "char_end": len(text),
                 "token_count": len(tokens)}] if text.strip() else []

    chunks = []
    stride = chunk_size - overlap
    start_tok = 0

    while start_tok < len(tokens):
        end_tok = min(start_tok + chunk_size, len(tokens))

        if end_tok < len(tokens):
            search_start_tok = start_tok + int(chunk_size * 0.8)
            best_break = None
            for t in range(end_tok - 1, search_start_tok - 1, -1):
                char_end = offsets[t][1]
                if char_end + 2 <= len(text):
                    if text[char_end:char_end + 2] == "\n\n":
                        best_break = t + 1
                        break
                    if text[char_end:char_end + 2] in (". ", ".\n"):
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
                "token_count": end_tok - start_tok,
            })

        if end_tok >= len(tokens):
            break
        start_tok += stride

    return chunks


# ============================================================================
# Phase 0a: Chunk 2000 novels
# ============================================================================

def phase0_rechunk():
    print(f"\n{'='*70}")
    print(f"PHASE 0a: Re-chunking 2000 novels (50 tokens, 15 overlap)")
    print(f"{'='*70}")

    if os.path.exists(CHUNKS_PATH):
        n_lines = sum(1 for _ in open(CHUNKS_PATH))
        print(f"[CACHED] Chunks exist: {CHUNKS_PATH} ({n_lines:,} chunks)")
        return n_lines

    manifest = {}
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line.strip())
            manifest[entry["book_id"]] = entry

    with open(BOOK_IDS_PATH, "r") as f:
        book_ids = [int(line.strip()) for line in f if line.strip()]

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME)
    print(f"Tokenizer loaded. Processing {len(book_ids)} books...")

    total_chunks = 0
    books_processed = 0
    books_skipped = 0

    t0 = time.time()
    with open(CHUNKS_PATH, "w", encoding="utf-8") as out_f:
        for i, bid in enumerate(book_ids):
            book_path = os.path.join(RAW_DIR, f"book_{bid}.txt")
            raw_path = os.path.join(RAW_DIR, f"pg{bid}.txt")

            text = None
            if os.path.exists(book_path):
                with open(book_path, "r", encoding="utf-8", errors="replace") as f:
                    text = f.read()
            elif os.path.exists(raw_path):
                with open(raw_path, "r", encoding="utf-8", errors="replace") as f:
                    raw_text = f.read()
                text = clean_gutenberg_text(raw_text)
            else:
                books_skipped += 1
                if books_skipped <= 10:
                    print(f"  [SKIP] No text for book {bid}")
                continue

            if not text or len(text.split()) < 1000:
                books_skipped += 1
                continue

            meta = manifest.get(bid, {})
            title = meta.get("title", f"Book {bid}")
            author = meta.get("author", "Unknown")

            raw_chunks = chunk_text_by_tokens(text, tokenizer, CHUNK_SIZE, CHUNK_OVERLAP)

            for j, chunk_data in enumerate(raw_chunks):
                chunk_entry = {
                    "chunk_id": f"book_{bid}_chunk_{j:06d}",
                    "book_id": bid,
                    "book_title": title,
                    "author": author,
                    "position": j,
                    "total_chunks": len(raw_chunks),
                    "text": chunk_data["text"],
                    "token_count": chunk_data["token_count"],
                }
                out_f.write(json.dumps(chunk_entry, ensure_ascii=False) + "\n")

            total_chunks += len(raw_chunks)
            books_processed += 1

            if (i + 1) % 100 == 0:
                print(f"  [{timestamp()}] {i+1}/{len(book_ids)} books | "
                      f"{total_chunks:,} chunks")

    elapsed = time.time() - t0
    print(f"Phase 0a complete in {elapsed:.0f}s | {books_processed} books | "
          f"{total_chunks:,} chunks | {books_skipped} skipped")
    return total_chunks


# ============================================================================
# Phase 0b: Embed all chunks
# ============================================================================

def phase0_embed():
    print(f"\n{'='*70}")
    print(f"PHASE 0b: Embedding chunks with BGE-large-en-v1.5")
    print(f"{'='*70}")

    if os.path.exists(EMBEDDINGS_PATH):
        emb = np.load(EMBEDDINGS_PATH, mmap_mode='r')
        print(f"[CACHED] Embeddings exist: {emb.shape}")
        return emb.shape[0]

    os.makedirs(SHARD_DIR, exist_ok=True)

    total_chunks = sum(1 for _ in open(CHUNKS_PATH))
    print(f"Total chunks: {total_chunks:,}")

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    print(f"Embedding model loaded")

    SHARD_SIZE = 100_000
    BATCH_SIZE_EMBED = 512
    n_shards = (total_chunks + SHARD_SIZE - 1) // SHARD_SIZE

    existing_shards = set()
    if os.path.exists(SHARD_DIR):
        for fname in os.listdir(SHARD_DIR):
            if fname.startswith("shard_") and fname.endswith(".npy"):
                shard_num = int(fname.split("_")[1].split(".")[0])
                existing_shards.add(shard_num)

    t0 = time.time()
    current_shard = 0
    shard_texts = []

    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f):
            chunk = json.loads(line)
            shard_texts.append(chunk["text"])

            if len(shard_texts) == SHARD_SIZE or line_num == total_chunks - 1:
                shard_path = os.path.join(SHARD_DIR, f"shard_{current_shard:03d}.npy")

                if current_shard in existing_shards:
                    print(f"  [{timestamp()}] Shard {current_shard}/{n_shards-1} cached")
                else:
                    print(f"  [{timestamp()}] Embedding shard {current_shard}/{n_shards-1} "
                          f"({len(shard_texts)} chunks)...")
                    shard_t0 = time.time()
                    embeddings = model.encode(
                        shard_texts,
                        batch_size=BATCH_SIZE_EMBED,
                        normalize_embeddings=True,
                        show_progress_bar=True,
                    ).astype(np.float32)
                    np.save(shard_path, embeddings)
                    print(f"    Saved {embeddings.shape} in {time.time()-shard_t0:.0f}s")

                shard_texts = []
                current_shard += 1

    print(f"  [{timestamp()}] Concatenating {n_shards} shards...")
    all_embeddings = []
    for s in range(n_shards):
        shard_path = os.path.join(SHARD_DIR, f"shard_{s:03d}.npy")
        shard_data = np.load(shard_path)
        all_embeddings.append(shard_data)

    final = np.concatenate(all_embeddings, axis=0).astype(np.float32)
    np.save(EMBEDDINGS_PATH, final)
    elapsed = time.time() - t0
    print(f"Phase 0b complete in {elapsed:.0f}s | Embeddings: {final.shape}")

    # Clean up shards to save space
    import shutil
    shutil.rmtree(SHARD_DIR)
    print(f"  Cleaned up embedding shards")

    return final.shape[0]


# ============================================================================
# Phase 0m: Generate passage metadata
# ============================================================================

def phase0_metadata():
    print(f"\n{'='*70}")
    print(f"PHASE 0m: Generating passage metadata")
    print(f"{'='*70}")

    if os.path.exists(METADATA_PATH):
        meta = np.load(METADATA_PATH)
        print(f"[CACHED] Metadata exists: {len(meta['book_ids'])} passages")
        return

    book_ids_list = []
    position_in_book = []
    token_counts = []

    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            chunk = json.loads(line)
            book_ids_list.append(chunk["book_id"])
            pos = chunk["position"] / max(chunk["total_chunks"] - 1, 1)
            position_in_book.append(pos)
            token_counts.append(chunk["token_count"])

    np.savez(METADATA_PATH,
             book_ids=np.array(book_ids_list, dtype=np.int32),
             position_in_book=np.array(position_in_book, dtype=np.float32),
             token_counts=np.array(token_counts, dtype=np.int16))
    print(f"  Saved metadata for {len(book_ids_list):,} passages")


# ============================================================================
# Random MLP transform + K-means (streaming, no 33GB save)
# ============================================================================

def random_mlp_kmeans():
    print(f"\n{'='*70}")
    print(f"RANDOM MLP: Transform + K-means k=100")
    print(f"{'='*70}")

    labels_path = os.path.join(DATA_DIR, "random_mlp_labels_k100.npy")
    centroids_path = os.path.join(DATA_DIR, "random_mlp_centroids_k100.npy")

    if os.path.exists(labels_path):
        labels = np.load(labels_path)
        print(f"[CACHED] Random MLP k-means exists: {len(labels):,} labels")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Random MLP with seed=999 (identical to validation run)
    torch.manual_seed(999)
    model = AssociationMLP(EMBEDDING_DIM, HIDDEN_DIM, NUM_LAYERS)
    model.to(device)
    model.eval()
    print(f"  Random MLP: {sum(p.numel() for p in model.parameters()):,} params (seed=999)")

    embeddings = np.load(EMBEDDINGS_PATH, mmap_mode='r')
    n_chunks = embeddings.shape[0]

    # Transform in memory (streaming)
    print(f"  Transforming {n_chunks:,} embeddings through random MLP...")
    t0 = time.time()
    BATCH = 50_000
    z_list = []
    for start in range(0, n_chunks, BATCH):
        end = min(start + BATCH, n_chunks)
        batch = torch.tensor(np.array(embeddings[start:end]),
                           dtype=torch.float32).to(device)
        with torch.no_grad():
            z_batch = model(batch)
        z_list.append(z_batch.cpu().numpy())
        if (start // BATCH) % 20 == 0:
            print(f"    [{timestamp()}] {end:,}/{n_chunks:,}")

    z_all = np.concatenate(z_list, axis=0).astype(np.float32)
    del z_list
    print(f"  Transformed in {time.time()-t0:.0f}s: {z_all.shape}")

    # K-means k=100
    import faiss
    print(f"  K-means k=100 (niter=50, gpu, seed=42)...")
    t0 = time.time()
    kmeans = faiss.Kmeans(EMBEDDING_DIM, 100, niter=50, verbose=True,
                         gpu=True, seed=42)
    kmeans.train(z_all)
    _, labels = kmeans.index.search(z_all, 1)
    labels = labels.flatten()

    np.save(labels_path, labels)
    np.save(centroids_path, kmeans.centroids)

    print(f"  K-means done in {time.time()-t0:.0f}s")
    print(f"  Labels: {len(labels):,}, Centroids: {kmeans.centroids.shape}")

    # Keep z_all in memory for readout (need it for nearest-to-centroid)
    return z_all


# ============================================================================
# Generate cluster readout
# ============================================================================

def generate_readout(z_all=None):
    print(f"\n{'='*70}")
    print(f"GENERATING CLUSTER READOUT")
    print(f"{'='*70}")

    labels = np.load(os.path.join(DATA_DIR, "random_mlp_labels_k100.npy"))
    centroids = np.load(os.path.join(DATA_DIR, "random_mlp_centroids_k100.npy"))
    meta = np.load(METADATA_PATH)
    book_ids = meta["book_ids"]
    position_in_book = meta["position_in_book"]
    token_counts = meta["token_counts"]
    embeddings = np.load(EMBEDDINGS_PATH, mmap_mode='r')

    k = 100
    N_PASSAGES = 10

    # If z_all not passed, regenerate (needed for nearest-to-centroid)
    if z_all is None:
        print(f"  Regenerating random MLP transforms for readout...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(999)
        model = AssociationMLP(EMBEDDING_DIM, HIDDEN_DIM, NUM_LAYERS)
        model.to(device).eval()
        n_chunks = embeddings.shape[0]
        z_list = []
        for start in range(0, n_chunks, 50_000):
            end = min(start + 50_000, n_chunks)
            batch = torch.tensor(np.array(embeddings[start:end]),
                               dtype=torch.float32).to(device)
            with torch.no_grad():
                z_batch = model(batch)
            z_list.append(z_batch.cpu().numpy())
        z_all = np.concatenate(z_list, axis=0).astype(np.float32)
        del z_list, model

    # Load chunk texts for passage extraction
    print(f"  Loading chunk texts...")
    t0 = time.time()
    chunk_texts = []
    chunk_metadata = []
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            chunk = json.loads(line)
            chunk_texts.append(chunk["text"])
            chunk_metadata.append({
                "book_id": chunk["book_id"],
                "book_title": chunk["book_title"],
                "author": chunk["author"],
                "position": chunk["position"],
                "total_chunks": chunk["total_chunks"],
            })
    print(f"  Loaded {len(chunk_texts):,} chunk texts in {time.time()-t0:.0f}s")

    # Build readout
    readout_lines = []
    readout_lines.append("# Random MLP Cluster Readout (k=100)")
    readout_lines.append(f"\nGenerated: {timestamp()}")
    readout_lines.append(f"Architecture: 8L/2048h, 29.4M params, torch seed=999, NO TRAINING")
    readout_lines.append(f"K-means: k=100, niter=50, faiss-gpu, seed=42")
    readout_lines.append(f"Total passages: {len(labels):,}")
    readout_lines.append("")

    # Also build summary JSON
    summary_data = []

    rng_np = np.random.RandomState(42)

    for c in range(k):
        mask = labels == c
        indices = np.where(mask)[0]
        if len(indices) == 0:
            continue

        cluster_size = len(indices)
        cluster_books = book_ids[indices]
        unique_books = len(set(cluster_books.tolist()))
        book_counts = Counter(cluster_books.tolist())
        dom_ratio = max(book_counts.values()) / cluster_size
        top_book_id = max(book_counts, key=book_counts.get)

        # Mean cosine in ORIGINAL embedding space (sampled)
        if cluster_size > COSINE_SAMPLE_SIZE:
            sample_idx = rng_np.choice(cluster_size, COSINE_SAMPLE_SIZE, replace=False)
            sample_embs = np.array(embeddings[indices[sample_idx]])
        else:
            sample_embs = np.array(embeddings[indices])
        norms = np.linalg.norm(sample_embs, axis=1, keepdims=True)
        normed = sample_embs / np.maximum(norms, 1e-8)
        sim = normed @ normed.T
        triu = np.triu_indices(len(normed), k=1)
        mean_cos = float(np.mean(sim[triu])) if len(normed) > 1 else 0.0

        mean_pos = float(np.mean(position_in_book[indices]))
        std_pos = float(np.std(position_in_book[indices]))
        mean_tok = float(np.mean(token_counts[indices]))

        # Filter check
        passes_filter = unique_books >= 50 and mean_cos < 0.50 and dom_ratio < 0.15

        # Nearest to centroid in transformed space
        centroid = centroids[c].reshape(1, -1)
        cluster_z = z_all[indices]
        c_norm = centroid / np.maximum(np.linalg.norm(centroid, axis=1, keepdims=True), 1e-8)
        z_norm = cluster_z / np.maximum(np.linalg.norm(cluster_z, axis=1, keepdims=True), 1e-8)
        sims = (z_norm @ c_norm.T).flatten()
        top_indices = np.argsort(-sims)[:N_PASSAGES]
        nearest_global_indices = indices[top_indices]

        # Top 5 books by count
        top_books = sorted(book_counts.items(), key=lambda x: -x[1])[:5]

        # Write readout
        filter_tag = "PASS" if passes_filter else "FAIL"
        readout_lines.append(f"{'='*70}")
        readout_lines.append(f"## Cluster {c} [{filter_tag}]")
        readout_lines.append(f"Passages: {cluster_size:,} | Books: {unique_books} | "
                           f"Mean Cosine: {mean_cos:.4f} | Dom Ratio: {dom_ratio:.4f}")
        readout_lines.append(f"Mean Position: {mean_pos:.3f} | Std Position: {std_pos:.3f} | "
                           f"Mean Tokens: {mean_tok:.1f}")
        readout_lines.append(f"Top books: {', '.join(f'{bid}({cnt})' for bid, cnt in top_books)}")
        readout_lines.append("")

        for rank, gi in enumerate(nearest_global_indices):
            cm = chunk_metadata[gi]
            pos_frac = cm["position"] / max(cm["total_chunks"] - 1, 1)
            readout_lines.append(f"  [{rank+1}] Book {cm['book_id']} \"{cm['book_title']}\" "
                               f"by {cm['author']} "
                               f"(pos {cm['position']}/{cm['total_chunks']}, {pos_frac:.2f})")
            # Show passage text (trim to ~300 chars for readability)
            text = chunk_texts[gi]
            if len(text) > 300:
                text = text[:300] + "..."
            readout_lines.append(f"      {text}")
            readout_lines.append("")

        readout_lines.append("")

        summary_data.append({
            "cluster_id": c,
            "n_passages": cluster_size,
            "n_unique_books": unique_books,
            "mean_original_cosine": round(mean_cos, 4),
            "dominance_ratio": round(dom_ratio, 4),
            "mean_position": round(mean_pos, 3),
            "std_position": round(std_pos, 3),
            "mean_tokens": round(mean_tok, 1),
            "passes_filter": passes_filter,
            "top_books": [{"book_id": bid, "count": cnt} for bid, cnt in top_books],
        })

    # Write files
    readout_path = os.path.join(RESULTS_DIR, "random_mlp_cluster_readout.md")
    with open(readout_path, "w", encoding="utf-8") as f:
        f.write("\n".join(readout_lines))
    print(f"  Saved: {readout_path}")

    summary_path = os.path.join(RESULTS_DIR, "random_mlp_cluster_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary_data, f, indent=2)
    print(f"  Saved: {summary_path}")

    # Print overview
    n_pass = sum(1 for s in summary_data if s["passes_filter"])
    cosines = [s["mean_original_cosine"] for s in summary_data]
    books = [s["n_unique_books"] for s in summary_data]
    print(f"\n  SUMMARY:")
    print(f"  Filter pass: {n_pass}/{k}")
    print(f"  Mean cosine: {np.mean(cosines):.4f}")
    print(f"  Mean book diversity: {np.mean(books):.1f}")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print(f"{'='*70}")
    print(f"Random MLP Cluster Readout Pipeline")
    print(f"Started: {timestamp()}")
    print(f"{'='*70}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)

    t0_total = time.time()

    # Phase 0: Chunk + Embed
    phase0_rechunk()
    phase0_embed()
    phase0_metadata()

    # Random MLP transform + K-means (returns z_all for readout)
    z_all = random_mlp_kmeans()

    # Generate readout
    generate_readout(z_all)

    elapsed = time.time() - t0_total
    print(f"\n{'='*70}")
    print(f"COMPLETE in {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"{'='*70}")
