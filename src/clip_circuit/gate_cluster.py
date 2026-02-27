from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans


@dataclass
class GateClusterResult:
    labels: torch.Tensor  # [B, N]
    centroids: torch.Tensor  # [K, D]
    cluster_sizes: torch.Tensor  # [K]
    cluster_activation_rates: torch.Tensor  # [K]


def cluster_ffn_gates(
    ffn_gate_l: torch.Tensor,
    num_clusters: int,
    sample_tokens: int = 20000,
    random_state: int = 0,
) -> GateClusterResult:
    if ffn_gate_l.ndim != 3:
        raise ValueError(f'Expected ffn_gate_l shape [B, N, D], got {ffn_gate_l.shape}.')
    if num_clusters <= 1:
        raise ValueError('num_clusters must be > 1.')

    batch_size, token_count, hidden_dim = ffn_gate_l.shape
    flat = ffn_gate_l.reshape(batch_size * token_count, hidden_dim).float().cpu().numpy()

    if num_clusters > flat.shape[0]:
        raise ValueError('num_clusters cannot exceed number of samples.')

    if sample_tokens > 0 and flat.shape[0] > sample_tokens:
        rng = np.random.default_rng(seed=random_state)
        indices = rng.choice(flat.shape[0], size=sample_tokens, replace=False)
        train_data = flat[indices]
    else:
        train_data = flat

    kmeans = MiniBatchKMeans(
        n_clusters=num_clusters,
        random_state=random_state,
        batch_size=min(2048, max(128, train_data.shape[0])),
        n_init='auto',
    )
    kmeans.fit(train_data)
    labels = kmeans.predict(flat)

    labels_t = torch.from_numpy(labels).reshape(batch_size, token_count).long()
    centroids_t = torch.from_numpy(kmeans.cluster_centers_).float()

    cluster_sizes = torch.bincount(labels_t.reshape(-1), minlength=num_clusters)
    cluster_activation_rates = centroids_t.mean(dim=-1)

    return GateClusterResult(
        labels=labels_t,
        centroids=centroids_t,
        cluster_sizes=cluster_sizes,
        cluster_activation_rates=cluster_activation_rates,
    )
