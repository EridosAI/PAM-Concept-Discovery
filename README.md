# Concept Discovery Through Predictive Associative Memory

**Unsupervised concept discovery from temporal co-occurrence at corpus scale.**

Code, data, and interactive demo for the paper *"From Topic to Transition Structure: Unsupervised Concept Discovery at Corpus Scale via Predictive Associative Memory"* by Jason Dury (Eridos AI).

**[Interactive Demo](https://eridos.ai/demo)** · **[Demo Guide](https://eridos.ai/concept-discovery)** · **[Paper (arXiv)](TODO)** · **[Eridos AI](https://eridos.ai)**

---

## The Idea

A chase scene in a Victorian adventure novel and a chase scene in a Russian psychological novel look nothing alike — different words, different settings, different century. But they perform the same work in a story. They sit in the same structural position: tension building before them, resolution or escape after. The words are different; the *shape* is the same.

Instead of asking "which passages use similar words?" we asked "which passages tend to appear near the same kinds of neighbours?" We trained a small neural network on 373 million of these neighbourhood relationships, extracted from 9,766 Project Gutenberg texts spanning four centuries. The model couldn't memorise them all — it had to compress, and in compressing, it found the patterns that recur across thousands of books.

What emerged were hundreds of structural patterns — from broad modes like "direct confrontation" and "lyrical landscape meditation" to precise registers like "sailor dialect," "courtroom cross-examination," and "Darwin-Huxley scientific correspondence." These aren't topics. They're recurring shapes in how text works — and they appear across authors, genres, and centuries.

## Key Results

| | |
|---|---|
| **Corpus** | 9,766 Project Gutenberg texts, 24.96M passages, 373M co-occurrence pairs |
| **Model** | 4-layer contrastive MLP, 29.4M parameters, 42.75% training accuracy |
| **Clustering** | 6 granularities: k=50, 100, 250, 500, 1,000, 2,000 |
| **Book diversity** | k=100 clusters average 4,508 books each (46% of corpus) |
| **Inductive transfer** | Unseen novels assigned to existing clusters without retraining |
| **Shuffle control** | Temporal shuffle collapses cross-boundary recall by 95.2% (2K pilot) |

Association-space clusters group by transition structure (narrative function, discourse register, literary tradition), while embedding-similarity clusters group by topic. Unseen novels concentrate into a selective subset of PAM clusters; raw BGE assignment saturates nearly all clusters.

## Repository Contents

```
PAM-Concept-Discovery/
├── README.md
├── LICENSE
├── requirements.txt
│
├── utils/                    # Shared configuration and utilities
│   ├── config.py
│   └── faiss_utils.py
│
├── 01_download_corpus/       # Gutenberg corpus download via Gutendex API
│   └── download_gutenberg.py
│
├── 02_chunk_and_embed/       # Token-based chunking and BGE embedding
│   ├── chunk_texts.py
│   └── embed_chunks.py
│
├── 03_extract_pairs/         # Temporal co-occurrence pair extraction
│   └── extract_pairs.py
│
├── 04_train/                 # Contrastive MLP training (model + training loop)
│   └── train.py
│
├── 05_cluster/               # k-means clustering in PAM association space
│   └── cluster.py
│
├── 06_label/                 # LLM cluster labelling via Anthropic Batch API
│   └── label_clusters.py
│
├── 07_evaluate/              # Baselines, controls, and unseen-novel evaluation
│   ├── bge_baseline.py
│   ├── context_enriched_baseline.py
│   ├── random_mlp_baseline.py
│   ├── validation_controls.py
│   ├── unseen_novel_eval.py
│   └── metrics.py
│
├── demo/                     # Interactive demo (static HTML + pre-loaded data)
│   ├── index.html
│   ├── worker.js
│   └── demo_data/
│
├── data/                     # Pre-computed outputs (see Data section)
│   ├── cluster_assignments/
│   ├── cluster_labels/       # LLM-generated labels (included)
│   ├── centroids/
│   └── model_checkpoint/
│
└── figures/                  # Paper figures
    ├── fig1_pipeline.png
    ├── fig2_bge_vs_pam.png
    ├── fig3_multiresolution.png
    ├── fig4_selectivity.png
    └── fig5_authorial_signatures.png
```

## Reproduction

### Requirements

- Python 3.10+
- PyTorch 2.0+
- sentence-transformers (for BGE-large-en-v1.5)
- faiss-cpu or faiss-gpu
- scikit-learn
- numpy, pandas

```bash
pip install -r requirements.txt
```

### Hardware

- **Training:** ~2 hours on an RTX 5090 or A100 (150 epochs over 373M pairs)
- **Embedding:** ~24M passages × BGE-large-en-v1.5 — significant compute; expect several hours on GPU
- **Clustering:** k-means on 24.96M × 1024 vectors — requires ~100GB RAM for the largest k values
- **Labelling:** Anthropic Batch API (~$2.75 for all 3,900 clusters across 6 k values)

### Quick verification (pre-computed outputs)

To verify claims without rerunning the full pipeline, download the pre-computed data (see Data section) and run the evaluation scripts directly:

```bash
# Unseen novel evaluation (no CLI args — paths configured via data/ layout)
python 07_evaluate/unseen_novel_eval.py

# Validation controls
python 07_evaluate/validation_controls.py --ablation shuffle
```

### Full pipeline

```bash
# 1. Download corpus (--max sets book count; default 250)
python 01_download_corpus/download_gutenberg.py --max 10000

# 2. Chunk and embed (no CLI args — uses Config defaults: 256 tokens, 64 overlap, BGE-large-en-v1.5)
python 02_chunk_and_embed/chunk_texts.py
python 02_chunk_and_embed/embed_chunks.py

# 3. Extract co-occurrence pairs (--window sets temporal window; default 3)
python 03_extract_pairs/extract_pairs.py --window 15

# 4. Train association model (--epochs and --pairs are configurable; other hyperparams in utils/config.py)
python 04_train/train.py --epochs 150

# 5. Cluster in PAM association space (no CLI args — edit K and parameters at top of script)
python 05_cluster/cluster.py

# 6. Label clusters (requires ANTHROPIC_API_KEY environment variable)
ANTHROPIC_API_KEY=your-key python 06_label/label_clusters.py

# 7. Evaluate
python 07_evaluate/unseen_novel_eval.py
python 07_evaluate/validation_controls.py
```

> **Note:** Scripts without CLI arguments use defaults from `utils/config.py`. Edit that file or the constants at the top of each script to adjust parameters.

## Data

Pre-computed outputs are available for download:

| File | Size | Contents |
|---|---|---|
| `cluster_assignments.jsonl` | ~2 GB | Cluster ID for each passage at all 6 k values |
| `cluster_labels.json` | ~1 MB | LLM-generated labels for all 3,900 clusters |
| `centroids.npz` | ~150 MB | k-means centroids at all 6 k values |
| `model_checkpoint.pt` | ~120 MB | Trained association model (29.4M params) |
| `chunk_text.jsonl` | ~6.9 GB | Raw passage text with byte-offset index |

> **Download link:** TODO (will be hosted on Zenodo or GitHub Releases)

The raw corpus is not redistributed. Texts are downloaded directly from Project Gutenberg via the Gutendex API using the download script.

## Interactive Demo

The [live demo](https://eridos.ai/demo) provides four exploration modes:

- **Narrative Timeline** — a novel's structure as a colour-coded sequence of concepts, with a resolution slider (k=50 to k=2,000)
- **Hierarchical View** — stacked concept bars at all 6 granularities for any novel
- **Cluster Explorer** — browse all clusters with labels, sample passages, and AI-powered structural explanations
- **Book List** — search and select from the full corpus plus 10 unseen evaluation novels

The demo runs as a static HTML page with pre-loaded data for 15 novels. An AI explainer feature uses the Anthropic API to provide on-demand analysis of cluster contents.

To run locally with full corpus access, see `demo/README.md`.

## Citation

```bibtex
@article{dury2026concept,
  title={From Topic to Transition Structure: Unsupervised Concept Discovery at Corpus Scale via Predictive Associative Memory},
  author={Dury, Jason},
  journal={arXiv preprint arXiv:TODO},
  year={2026}
}
```

## Related Work

This paper is the third in the PAM research programme:

1. **[Predictive Associative Memory](https://github.com/EridosAI/PAM-Benchmark)** — the foundational framework establishing that temporal co-occurrence trains predictors capable of faithful associative recall across representational boundaries. ([arXiv](https://arxiv.org/abs/2602.11322))

2. **[Association-Augmented Retrieval](https://github.com/EridosAI/AAR)** — the applied paper showing +8.6 Recall@5 on HotpotQA through corpus-specific association learning. Inductive transfer fails in text, partially works in biology.

3. **Concept Discovery** (this repo) — the emergence paper showing that the same framework, under compression, discovers structural patterns that transfer to unseen texts.

The three papers share a training signal (temporal co-occurrence) and architecture (contrastive MLP) but operate in different compression regimes — from full memorisation (PAM) through corpus-specific learning (AAR) to pattern extraction (this work).

## License

MIT License. See [LICENSE](LICENSE).
