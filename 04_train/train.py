"""PAM predictor training -- adapted from AAR v1.

Same 4-layer MLP with learned residual, symmetric InfoNCE (CLIP-style
in-batch negatives), batch size 512. Architecture is intentionally
identical to AAR -- the variable under test is data scale.

Usage:
    python 04_train/train.py
    python 04_train/train.py --epochs 50 --pairs temporal_pairs_w3.json
"""

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import os
import json
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from utils.config import Config


# ---------------------------------------------------------------------------
# Model -- identical to AAR v1 (4-layer MLP, 1024 hidden, learned residual)
# ---------------------------------------------------------------------------

class AssociationMLP(nn.Module):
    """PAM predictor: transforms embeddings into association-query space.

    DO NOT MODIFY architecture -- same 4-layer MLP, hidden 1024, ~4.2M params.
    """

    def __init__(self, embedding_dim: int = 1024, hidden_dim: int = 1024,
                 num_layers: int = 4):
        super().__init__()
        layers = []
        # Input projection
        layers.append(nn.Linear(embedding_dim, hidden_dim))
        layers.append(nn.LayerNorm(hidden_dim))
        layers.append(nn.GELU())
        # Hidden layers
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.GELU())
        # Output projection back to embedding dim
        layers.append(nn.Linear(hidden_dim, embedding_dim))
        self.net = nn.Sequential(*layers)
        # Learned residual -- association query is a modification, not replacement
        self.residual_weight = nn.Parameter(torch.tensor(0.5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        transformed = self.net(x)
        alpha = torch.sigmoid(self.residual_weight)
        out = alpha * x + (1 - alpha) * transformed
        return F.normalize(out, dim=-1)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class PairDataset(Dataset):
    """Dataset of (anchor_embedding, positive_embedding) pairs."""

    def __init__(self, pairs: list[tuple[int, int]],
                 embeddings: torch.Tensor):
        self.pairs = pairs
        self.embeddings = embeddings

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        anchor_idx, positive_idx = self.pairs[idx]
        return self.embeddings[anchor_idx], self.embeddings[positive_idx]


# ---------------------------------------------------------------------------
# Loss -- symmetric InfoNCE (CLIP-style)
# ---------------------------------------------------------------------------

def clip_loss(anchor_transformed: torch.Tensor, positives: torch.Tensor,
              temperature: float = 0.05) -> torch.Tensor:
    """CLIP-style in-batch contrastive loss.

    Builds B x B similarity matrix. Diagonal = positives.
    With B=512, each anchor gets 511 in-batch negatives.
    """
    logits = torch.mm(anchor_transformed, positives.t()) / temperature
    labels = torch.arange(logits.size(0), device=logits.device)
    loss_a = F.cross_entropy(logits, labels)
    loss_b = F.cross_entropy(logits.t(), labels)
    return (loss_a + loss_b) / 2


# ---------------------------------------------------------------------------
# Bi-directional scoring (from AAR)
# ---------------------------------------------------------------------------

def bidirectional_score(model: AssociationMLP, query_emb: torch.Tensor,
                        candidate_embs: torch.Tensor) -> torch.Tensor:
    """Score candidates using both forward and reverse association.

    forward:  sim(MLP(query), candidate)
    reverse:  sim(query, MLP(candidate))
    combined: (forward + reverse) / 2
    """
    with torch.no_grad():
        fwd_query = model(query_emb)  # [1, D]
        fwd_scores = torch.mm(fwd_query, candidate_embs.t()).squeeze(0)

        rev_candidates = model(candidate_embs)  # [K, D]
        rev_scores = torch.mm(query_emb, rev_candidates.t()).squeeze(0)

    return (fwd_scores + rev_scores) / 2


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(config: Config, pairs: list[tuple[int, int]],
          embeddings: np.ndarray) -> AssociationMLP:
    """Train the PAM predictor with CLIP-style contrastive learning."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32)
    dataset = PairDataset(pairs, embeddings_tensor)
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=True,  # consistent batch size for CLIP loss
    )

    model = AssociationMLP(
        embedding_dim=config.embedding_dim,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model: {config.embedding_dim}d -> {config.hidden_dim}h x "
          f"{config.num_layers}L, {param_count:,} params")
    print(f"Training on {device}, batch_size={config.batch_size} "
          f"({config.batch_size - 1} in-batch negatives)")
    print(f"Pairs: {len(pairs):,}, epochs: {config.epochs}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs
    )

    best_acc = 0.0
    t0 = time.time()

    for epoch in range(config.epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for anchors, positives in loader:
            anchors = anchors.to(device)
            positives = positives.to(device)

            anchor_transformed = model(anchors)
            loss = clip_loss(anchor_transformed, positives, config.temperature)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()

            B = anchors.size(0)
            total_loss += loss.item() * B

            with torch.no_grad():
                sim = torch.mm(anchor_transformed, positives.t())
                preds = sim.argmax(dim=1)
                labels = torch.arange(B, device=device)
                correct += (preds == labels).sum().item()
                total += B

        scheduler.step()
        avg_loss = total_loss / len(dataset)
        accuracy = correct / total if total > 0 else 0.0

        marker = ""
        if accuracy > best_acc:
            best_acc = accuracy
            marker = " *"

        print(f"Epoch {epoch + 1:3d}/{config.epochs} -- "
              f"Loss: {avg_loss:.4f} -- Accuracy: {accuracy:.4f}{marker}")

    elapsed = time.time() - t0
    print(f"\nTraining complete in {elapsed:.0f}s. Best accuracy: {best_acc:.4f}")
    return model


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run(config: Config | None = None, pairs_file: str | None = None,
        epochs: int | None = None):
    """Load data, train, and save model."""
    if config is None:
        config = Config()
    if epochs is not None:
        config.epochs = epochs
    config.ensure_dirs()

    # Load embeddings
    emb_path = os.path.join(config.embeddings_dir, "embeddings.npy")
    if not os.path.exists(emb_path):
        raise FileNotFoundError(f"No embeddings at {emb_path}. Run embed.py first.")
    embeddings = np.load(emb_path)
    print(f"Loaded embeddings: {embeddings.shape}")

    # Load pairs
    if pairs_file is None:
        pairs_file = f"temporal_pairs_w{config.temporal_window}.json"
    pairs_path = os.path.join(config.pairs_dir, pairs_file)
    if not os.path.exists(pairs_path):
        raise FileNotFoundError(
            f"No pairs at {pairs_path}. Run generate_pairs.py first."
        )
    with open(pairs_path, "r") as f:
        pairs = json.load(f)
    # Convert to list of tuples
    pairs = [(a, b) for a, b in pairs]
    print(f"Loaded {len(pairs):,} pairs from {pairs_file}")

    # Train
    model = train(config, pairs, embeddings)

    # Save
    model_path = os.path.join(config.models_dir, "pam_predictor.pt")
    torch.save(model.state_dict(), model_path)
    print(f"Saved model to {model_path}")

    # Save training metadata
    meta = {
        "embedding_dim": config.embedding_dim,
        "hidden_dim": config.hidden_dim,
        "num_layers": config.num_layers,
        "pairs_file": pairs_file,
        "num_pairs": len(pairs),
        "epochs": config.epochs,
        "batch_size": config.batch_size,
        "param_count": sum(p.numel() for p in model.parameters()),
    }
    meta_path = os.path.join(config.models_dir, "training_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PAM predictor")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--pairs", type=str, default=None,
                        help="Pairs filename in data/pairs/")
    args = parser.parse_args()
    run(epochs=args.epochs, pairs_file=args.pairs)
