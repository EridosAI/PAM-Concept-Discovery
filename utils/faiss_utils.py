"""FAISS index management utilities."""

import numpy as np
import faiss


def build_index(embeddings: np.ndarray, use_gpu: bool = True) -> faiss.Index:
    """Build FAISS inner-product index (cosine similarity on normalized vectors)."""
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    if use_gpu and faiss.get_num_gpus() > 0:
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)
    index.add(embeddings.astype(np.float32))
    return index


def save_index(index: faiss.Index, path: str):
    """Save FAISS index to disk (converts GPU index to CPU first if needed)."""
    if hasattr(index, "index"):  # GPU index wrapper
        index = faiss.index_gpu_to_cpu(index)
    faiss.write_index(index, path)


def load_index(path: str, use_gpu: bool = False) -> faiss.Index:
    """Load FAISS index from disk."""
    index = faiss.read_index(path)
    if use_gpu and faiss.get_num_gpus() > 0:
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)
    return index


def search(index: faiss.Index, query: np.ndarray, k: int = 100):
    """Search index. Returns (distances, indices) arrays."""
    query = query.astype(np.float32)
    if query.ndim == 1:
        query = query.reshape(1, -1)
    return index.search(query, k)


def build_subset_index(embeddings: np.ndarray, indices: np.ndarray,
                       use_gpu: bool = True) -> faiss.Index:
    """Build a FAISS index from a subset of embeddings (for scale experiments)."""
    subset = embeddings[indices].astype(np.float32)
    return build_index(subset, use_gpu=use_gpu)
