"""
Inductive Concept Test — 3 unseen novels through PAM + raw BGE baseline.

Downloads 3 well-known Gutenberg novels NOT in training, processes them through
the full pipeline, and assigns each chunk to nearest centroid at k=100 and k=1000
for both PAM-transformed and raw BGE embeddings.

Novels:
  345  — Dracula (Bram Stoker) — Gothic Horror
  1342 — Pride and Prejudice (Jane Austen) — Social/Romance
  2701 — Moby Dick (Herman Melville) — Adventure/Literary
"""

import sys
import os
import re
import json
import time
import urllib.request
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

BASE = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS_DIR = BASE / "results"
MODEL_PATH = BASE / "data" / "model_checkpoint" / "pam_10k_warm_epoch_150.pt"
PAM_CENTROIDS = {
    100: BASE / "data" / "centroids" / "pam_centroids_k100.npy",
    1000: BASE / "data" / "centroids" / "pam_centroids_k1000.npy",
}
BGE_CENTROIDS = {
    100: BASE / "data" / "centroids" / "bge_raw_centroids_k100.npy",
    250: BASE / "data" / "centroids" / "bge_raw_centroids_k250.npy",
}

NOVELS = [
    (345, "Dracula", "Bram Stoker", "Gothic Horror"),
    (1342, "Pride and Prejudice", "Jane Austen", "Social/Romance"),
    (2701, "Moby Dick", "Herman Melville", "Adventure/Literary"),
]

CHUNK_SIZE = 50
CHUNK_OVERLAP = 15
EMBEDDING_DIM = 1024
HIDDEN_DIM = 2048
NUM_LAYERS = 8


def p(msg):
    print(msg, flush=True)


# ── Model definition (must match training exactly) ──────────────────────────

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


# ── Gutenberg download + cleaning ───────────────────────────────────────────

def download_novel(book_id, retries=3):
    """Download plain text from Project Gutenberg with retries."""
    url = f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt"
    for attempt in range(retries):
        try:
            p(f"  Downloading {url} (attempt {attempt+1})...")
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                return resp.read().decode("utf-8", errors="replace")
        except Exception as e:
            p(f"  Download failed: {e}")
            if attempt < retries - 1:
                time.sleep(2)
    raise RuntimeError(f"Failed to download book {book_id} after {retries} attempts")


def clean_gutenberg_text(text):
    """Strip Gutenberg header/footer."""
    start_markers = [
        "*** START OF THE PROJECT GUTENBERG",
        "*** START OF THIS PROJECT GUTENBERG",
        "*END*THE SMALL PRINT",
    ]
    start_idx = 0
    for marker in start_markers:
        pos = text.find(marker)
        if pos != -1:
            newline = text.find("\n", pos)
            if newline != -1:
                start_idx = newline + 1
            break

    end_markers = [
        "*** END OF THE PROJECT GUTENBERG",
        "*** END OF THIS PROJECT GUTENBERG",
        "End of the Project Gutenberg",
        "End of Project Gutenberg",
    ]
    end_idx = len(text)
    for marker in end_markers:
        pos = text.find(marker)
        if pos != -1:
            end_idx = pos
            break

    cleaned = text[start_idx:end_idx].strip()
    cleaned = re.sub(r'\r\n', '\n', cleaned)
    cleaned = re.sub(r'\r', '\n', cleaned)
    return cleaned


# ── Chunking (reuses logic from src/corpus/chunk.py) ────────────────────────

def chunk_text_by_tokens(text, tokenizer, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split text into overlapping chunks by token count."""
    encoding = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    tokens = encoding["input_ids"]
    offsets = encoding["offset_mapping"]

    if not tokens:
        return []
    if len(tokens) <= chunk_size:
        return [{"text": text.strip()}] if text.strip() else []

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
            chunks.append({"text": chunk_text})

        if end_tok >= len(tokens):
            break
        start_tok += stride

    return chunks


# ── Embedding ────────────────────────────────────────────────────────────────

def embed_chunks(chunks, model):
    """Embed chunk texts with SentenceTransformer, returns L2-normalized."""
    texts = [c["text"] for c in chunks]
    p(f"  Embedding {len(texts)} chunks...")
    embeddings = model.encode(
        texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True
    )
    return np.array(embeddings, dtype=np.float32)


# ── Cluster assignment ───────────────────────────────────────────────────────

def assign_clusters(embeddings, centroids):
    """Assign each embedding to nearest centroid via cosine (IP on L2-normed)."""
    # Normalize centroids
    norms = np.linalg.norm(centroids, axis=1, keepdims=True)
    centroids_normed = centroids / np.maximum(norms, 1e-8)
    # Cosine similarity
    sims = embeddings @ centroids_normed.T  # (N, K)
    labels = sims.argmax(axis=1)
    scores = sims[np.arange(len(labels)), labels]
    return labels, scores


# ── Readout generation ───────────────────────────────────────────────────────

def generate_readout(book_id, title, author, genre, chunks,
                     pam_labels_100, pam_labels_1000,
                     bge_labels_100, bge_labels_250):
    """Generate sequential readout markdown."""
    n = len(chunks)
    pam_k100_used = len(set(pam_labels_100))
    pam_k1000_used = len(set(pam_labels_1000))
    bge_k100_used = len(set(bge_labels_100))
    bge_k250_used = len(set(bge_labels_250))

    lines = [
        f"# Inductive Test — {title} (Book {book_id})",
        f"**Author:** {author} | **Genre:** {genre}",
        f"**Chunks:** {n}",
        f"",
        f"| System | k | Clusters used |",
        f"|--------|---|---------------|",
        f"| PAM | 100 | {pam_k100_used}/100 |",
        f"| PAM | 1000 | {pam_k1000_used}/1000 |",
        f"| BGE raw | 100 | {bge_k100_used}/100 |",
        f"| BGE raw | 250 | {bge_k250_used}/250 |",
        f"",
    ]

    for i in range(n):
        text = chunks[i]["text"][:250]
        pos_pct = i / max(n - 1, 1) * 100
        lines.append(
            f"## Chunk {i} ({pos_pct:.1f}%) | "
            f"PAM-k100={pam_labels_100[i]} | PAM-k1000={pam_labels_1000[i]} | "
            f"BGE-k100={bge_labels_100[i]} | BGE-k250={bge_labels_250[i]}"
        )
        lines.append(f"> {text}")
        lines.append("")

    return "\n".join(lines)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    p("=" * 60)
    p("Inductive Concept Test — 3 Unseen Novels")
    p("=" * 60)

    # Load tokenizer
    p("\nLoading BGE tokenizer...")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-large-en-v1.5")

    # Load embedding model
    p("Loading BGE-large-en-v1.5 embedding model...")
    from sentence_transformers import SentenceTransformer
    embed_model = SentenceTransformer("BAAI/bge-large-en-v1.5")

    # Load PAM model
    p("Loading PAM model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pam_model = AssociationMLP(EMBEDDING_DIM, HIDDEN_DIM, NUM_LAYERS).to(device)
    checkpoint = torch.load(str(MODEL_PATH), map_location=device, weights_only=True)
    if "model_state_dict" in checkpoint:
        pam_model.load_state_dict(checkpoint["model_state_dict"])
    else:
        pam_model.load_state_dict(checkpoint)
    pam_model.eval()
    alpha = torch.sigmoid(pam_model.residual_weight).item()
    p(f"  PAM loaded on {device}, alpha={alpha:.3f}")

    # Load centroids
    p("Loading centroids...")
    pam_cent = {k: np.load(str(v)) for k, v in PAM_CENTROIDS.items()}
    bge_cent = {k: np.load(str(v)) for k, v in BGE_CENTROIDS.items()}
    for name, d in [("PAM", pam_cent), ("BGE", bge_cent)]:
        for k, c in d.items():
            p(f"  {name} k={k}: {c.shape}")

    # Process each novel
    for book_id, title, author, genre in NOVELS:
        p(f"\n{'='*60}")
        p(f"Processing: {title} (Book {book_id})")
        p(f"{'='*60}")

        # Download
        raw_text = download_novel(book_id)
        p(f"  Downloaded: {len(raw_text):,} chars")

        # Clean
        text = clean_gutenberg_text(raw_text)
        p(f"  Cleaned: {len(text):,} chars")

        # Chunk
        chunks = chunk_text_by_tokens(text, tokenizer)
        p(f"  Chunked: {len(chunks)} passages")

        # Embed (raw BGE)
        bge_embeddings = embed_chunks(chunks, embed_model)
        p(f"  Embedded: {bge_embeddings.shape}")

        # PAM transform
        p("  PAM transform...")
        with torch.no_grad():
            bge_tensor = torch.from_numpy(bge_embeddings).to(device)
            pam_embeddings = pam_model(bge_tensor).cpu().numpy()
        p(f"  PAM output: {pam_embeddings.shape}")

        # Assign clusters — PAM embeddings to PAM centroids
        pam_labels_100, _ = assign_clusters(pam_embeddings, pam_cent[100])
        pam_labels_1000, _ = assign_clusters(pam_embeddings, pam_cent[1000])
        p(f"  PAM clusters: k100 uses {len(set(pam_labels_100))}/100, "
          f"k1000 uses {len(set(pam_labels_1000))}/1000")

        # Assign clusters — raw BGE embeddings to BGE centroids
        bge_labels_100, _ = assign_clusters(bge_embeddings, bge_cent[100])
        bge_labels_250, _ = assign_clusters(bge_embeddings, bge_cent[250])
        p(f"  BGE clusters: k100 uses {len(set(bge_labels_100))}/100, "
          f"k250 uses {len(set(bge_labels_250))}/250")

        # Generate readout
        readout = generate_readout(
            book_id, title, author, genre, chunks,
            pam_labels_100, pam_labels_1000,
            bge_labels_100, bge_labels_250,
        )
        out_path = RESULTS_DIR / f"inductive_test_{book_id}.md"
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(readout)
        p(f"  Saved: {out_path}")

    p(f"\n{'='*60}")
    p("Done! All 3 inductive test readouts generated.")
    p(f"{'='*60}")


if __name__ == "__main__":
    main()
