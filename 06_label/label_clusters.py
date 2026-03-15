#!/usr/bin/env python3
"""
Label PAM 10K clusters using Anthropic Batch API.

Phases:
  1. Build byte-offset index for chunks_clean.jsonl (random access)
  2. Sample chunks per cluster
  3. Create batch API requests (10 clusters per request)
  4. Submit batch, poll, collect results
  5. Parse and save per-k JSON
  6. Validation summary
"""

import json
import os
import sys
import time
import numpy as np
import anthropic

# ── Config ──────────────────────────────────────────────────────────────────
API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
MODEL = "claude-sonnet-4-5-20250929"

# Paths relative to repo root
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(_REPO_ROOT, "data")
RESULTS_DIR = os.path.join(_REPO_ROOT, "data", "cluster_labels")

K_LEVELS = [50, 100, 250, 500, 1000, 2000]
SAMPLES_PER_CLUSTER = 5
CLUSTERS_PER_REQUEST = 10
MAX_CHUNK_CHARS = 200
SEED = 42
POLL_INTERVAL = 30  # seconds

SYSTEM_PROMPT = """You are labeling narrative concept clusters discovered by an unsupervised model trained on 10,000 Project Gutenberg novels. Each cluster contains text chunks grouped by temporal co-occurrence patterns (NOT semantic similarity).

For each cluster, provide:
1. A short label (2-6 words) describing the narrative function
2. A one-sentence description

Labels should describe NARRATIVE FUNCTIONS — what role these chunks play in storytelling structure. Good examples: "Romantic tension builds", "Arrival at new setting", "Internal moral debate", "Comic relief / wit". Bad examples: "About love", "Travel", "Thinking".

Respond with ONLY a JSON array, no markdown, no backticks:
[{"id": 0, "label": "...", "description": "..."}, ...]"""


# ── Phase 1: Build chunk offset index ───────────────────────────────────────
def build_offset_index():
    """Scan chunks_clean.jsonl once, record byte offset of each line."""
    offset_path = os.path.join(DATA_DIR, "chunk_offsets.npy")
    if os.path.exists(offset_path):
        print(f"[Phase 1] Offset index already exists: {offset_path}")
        return np.load(offset_path)

    print("[Phase 1] Building byte offset index for chunks_clean.jsonl...")
    jsonl_path = os.path.join(DATA_DIR, "chunks_clean.jsonl")
    offsets = []
    with open(jsonl_path, "rb") as f:
        while True:
            pos = f.tell()
            line = f.readline()
            if not line:
                break
            offsets.append(pos)
            if len(offsets) % 5_000_000 == 0:
                print(f"  ...{len(offsets):,} lines indexed")

    offsets = np.array(offsets, dtype=np.int64)
    np.save(offset_path, offsets)
    print(f"  Done: {len(offsets):,} lines → {offset_path}")
    return offsets


def read_chunks_by_indices(offsets, indices):
    """Read specific chunk lines from JSONL using byte offsets."""
    jsonl_path = os.path.join(DATA_DIR, "chunks_clean.jsonl")
    chunks = []
    with open(jsonl_path, "rb") as f:
        for idx in indices:
            f.seek(int(offsets[idx]))
            line = f.readline()
            chunk = json.loads(line)
            chunks.append(chunk)
    return chunks


# ── Phase 2: Sample chunks per cluster ──────────────────────────────────────
def sample_clusters(k, offsets, book_ids):
    """For each cluster at level k, sample chunk indices and read text."""
    print(f"\n[Phase 2] Sampling chunks for k={k}...")
    labels = np.load(os.path.join(DATA_DIR, f"pam_labels_k{k}.npy"))
    rng = np.random.RandomState(SEED)

    cluster_ids = np.unique(labels)
    print(f"  {len(cluster_ids)} clusters, {len(labels):,} chunks total")

    cluster_data = []
    for cid in cluster_ids:
        member_indices = np.where(labels == cid)[0]
        n_members = len(member_indices)

        # Count unique books in this cluster
        cluster_book_ids = book_ids[member_indices]
        n_books = len(np.unique(cluster_book_ids))

        # Sample
        n_sample = min(SAMPLES_PER_CLUSTER, n_members)
        sampled = rng.choice(member_indices, size=n_sample, replace=False)

        # Read chunk text
        chunks = read_chunks_by_indices(offsets, sampled)

        samples = []
        for chunk in chunks:
            text = chunk["text"][:MAX_CHUNK_CHARS]
            bid = chunk["book_id"]
            samples.append({"text": text, "book_id": bid})

        cluster_data.append({
            "id": int(cid),
            "size": n_members,
            "n_books": n_books,
            "samples": samples,
        })

    print(f"  Sampled {len(cluster_data)} clusters")
    return cluster_data


# ── Phase 3: Create batch requests ─────────────────────────────────────────
def build_batch_requests(k, cluster_data):
    """Group clusters into batches of CLUSTERS_PER_REQUEST and build API requests."""
    requests = []
    for batch_idx in range(0, len(cluster_data), CLUSTERS_PER_REQUEST):
        batch_clusters = cluster_data[batch_idx:batch_idx + CLUSTERS_PER_REQUEST]
        batch_num = batch_idx // CLUSTERS_PER_REQUEST

        # Build user message
        parts = ["Label these clusters. Each shows sample text chunks from different books.\n"]
        for cl in batch_clusters:
            parts.append(f"\n=== Cluster {cl['id']} (k={k}, {cl['size']:,} chunks, {cl['n_books']} books) ===")
            for s in cl["samples"]:
                parts.append(f"[Book #{s['book_id']}] {s['text']}")
                parts.append("---")

        user_msg = "\n".join(parts)

        requests.append({
            "custom_id": f"k{k}_batch{batch_num}",
            "params": {
                "model": MODEL,
                "max_tokens": 1024,
                "system": SYSTEM_PROMPT,
                "messages": [{"role": "user", "content": user_msg}],
            },
        })

    return requests


# ── Phase 4: Submit batch, poll, collect ────────────────────────────────────
def submit_and_collect(client, requests, k):
    """Submit a batch, poll until done, return results keyed by custom_id."""
    print(f"\n[Phase 4] Submitting batch for k={k}: {len(requests)} requests...")

    # Submit
    batch = client.messages.batches.create(requests=requests)
    batch_id = batch.id
    print(f"  Batch ID: {batch_id}")

    # Poll
    while True:
        batch = client.messages.batches.retrieve(batch_id)
        counts = batch.request_counts
        print(f"  Status: {batch.processing_status} | "
              f"processing={counts.processing} succeeded={counts.succeeded} "
              f"errored={counts.errored}")
        if batch.processing_status == "ended":
            break
        time.sleep(POLL_INTERVAL)

    # Collect results
    results = {}
    failed_ids = []
    for result in client.messages.batches.results(batch_id):
        cid = result.custom_id
        if result.result.type == "succeeded":
            text = result.result.message.content[0].text
            results[cid] = text
        else:
            print(f"  FAILED: {cid} ({result.result.type})")
            failed_ids.append(cid)

    print(f"  Collected {len(results)} results, {len(failed_ids)} failures")
    return results, failed_ids


def retry_failed(client, all_requests, failed_ids, k):
    """Retry just the failed request IDs."""
    if not failed_ids:
        return {}

    retry_requests = [r for r in all_requests if r["custom_id"] in failed_ids]
    print(f"\n[Retry] Retrying {len(retry_requests)} failed requests for k={k}...")
    results, still_failed = submit_and_collect(client, retry_requests, k)
    if still_failed:
        print(f"  WARNING: {len(still_failed)} requests still failed after retry: {still_failed}")
    return results


# ── Phase 5: Parse and save ─────────────────────────────────────────────────
def parse_and_save(k, cluster_data, raw_results):
    """Parse JSON from responses, save cluster_labels_k{K}.json."""
    labels_dict = {}
    parse_errors = []

    # Map cluster IDs to their batch custom_id
    cluster_to_batch = {}
    for batch_idx in range(0, len(cluster_data), CLUSTERS_PER_REQUEST):
        batch_num = batch_idx // CLUSTERS_PER_REQUEST
        custom_id = f"k{k}_batch{batch_num}"
        for cl in cluster_data[batch_idx:batch_idx + CLUSTERS_PER_REQUEST]:
            cluster_to_batch[cl["id"]] = custom_id

    for custom_id, text in raw_results.items():
        try:
            # Strip any accidental markdown fences
            text = text.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1]
                if text.endswith("```"):
                    text = text[:-3]
                text = text.strip()

            parsed = json.loads(text)
            for item in parsed:
                cid = str(item["id"])
                labels_dict[cid] = {
                    "label": item["label"],
                    "description": item["description"],
                }
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            parse_errors.append((custom_id, str(e)))
            print(f"  Parse error for {custom_id}: {e}")

    output = {
        "k": k,
        "num_clusters": len(labels_dict),
        "labels": labels_dict,
    }

    out_path = os.path.join(RESULTS_DIR, f"cluster_labels_k{k}.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"  Saved {len(labels_dict)} labels → {out_path}")

    return labels_dict, parse_errors


# ── Phase 6: Validation ────────────────────────────────────────────────────
def validate_all():
    """Load all saved label files, print summary, save combined."""
    print("\n" + "=" * 60)
    print("[Phase 6] Validation Summary")
    print("=" * 60)

    combined = {}
    for k in K_LEVELS:
        path = os.path.join(RESULTS_DIR, f"cluster_labels_k{k}.json")
        if not os.path.exists(path):
            print(f"\n  k={k}: FILE MISSING")
            continue

        with open(path) as f:
            data = json.load(f)

        labels_dict = data["labels"]
        expected = len(np.unique(np.load(os.path.join(DATA_DIR, f"pam_labels_k{k}.npy"))))
        labeled = len(labels_dict)

        print(f"\n  k={k}: {labeled}/{expected} clusters labeled", end="")
        if labeled < expected:
            missing = expected - labeled
            print(f" ({missing} MISSING)", end="")
        print()

        # Flag small clusters
        labels_arr = np.load(os.path.join(DATA_DIR, f"pam_labels_k{k}.npy"))
        small_clusters = []
        for cid_str, info in labels_dict.items():
            cid = int(cid_str)
            size = np.sum(labels_arr == cid)
            if size < 50:
                small_clusters.append((cid, size, info["label"]))
        if small_clusters:
            print(f"    Small clusters (<50 chunks): {len(small_clusters)}")
            for cid, size, label in small_clusters[:5]:
                print(f"      Cluster {cid}: {size} chunks — \"{label}\"")
            if len(small_clusters) > 5:
                print(f"      ...and {len(small_clusters) - 5} more")

        # Flag duplicate labels
        all_labels = [info["label"] for info in labels_dict.values()]
        seen = {}
        dupes = []
        for lbl in all_labels:
            lbl_lower = lbl.lower().strip()
            if lbl_lower in seen:
                dupes.append(lbl)
            seen[lbl_lower] = seen.get(lbl_lower, 0) + 1
        dupes_unique = {lbl for lbl, cnt in seen.items() if cnt > 1}
        if dupes_unique:
            print(f"    Duplicate labels: {len(dupes_unique)}")
            for d in list(dupes_unique)[:5]:
                print(f"      \"{d}\" (×{seen[d]})")

        combined[str(k)] = data

    # Save combined
    combined_path = os.path.join(RESULTS_DIR, "cluster_labels_all.json")
    with open(combined_path, "w") as f:
        json.dump(combined, f, indent=2)
    print(f"\n  Combined → {combined_path}")


# ── Main ────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("PAM 10K Cluster Labeling via Anthropic Batch API")
    print("=" * 60)

    client = anthropic.Anthropic(api_key=API_KEY)

    # Phase 1
    offsets = build_offset_index()
    print(f"  Offset index: {len(offsets):,} entries")

    # Load metadata
    meta = np.load(os.path.join(DATA_DIR, "chunk_metadata_clean.npz"))
    book_ids = meta["book_ids"]

    # Process each k level
    for k in K_LEVELS:
        # Check if already done
        out_path = os.path.join(RESULTS_DIR, f"cluster_labels_k{k}.json")
        if os.path.exists(out_path):
            print(f"\n[Skip] k={k} already labeled: {out_path}")
            continue

        # Phase 2: Sample
        cluster_data = sample_clusters(k, offsets, book_ids)

        # Phase 3: Build requests
        requests = build_batch_requests(k, cluster_data)
        print(f"  Built {len(requests)} batch requests for k={k}")

        # Phase 4: Submit and collect
        raw_results, failed_ids = submit_and_collect(client, requests, k)

        # Retry failures
        if failed_ids:
            retry_results = retry_failed(client, requests, failed_ids, k)
            raw_results.update(retry_results)

        # Phase 5: Parse and save (checkpoint)
        labels_dict, parse_errors = parse_and_save(k, cluster_data, raw_results)
        if parse_errors:
            print(f"  WARNING: {len(parse_errors)} parse errors for k={k}")

    # Phase 6: Validation
    validate_all()

    print("\nDone!")


if __name__ == "__main__":
    main()
