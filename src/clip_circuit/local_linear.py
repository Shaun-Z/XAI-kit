from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class ClusterEdgeFitResult:
    prototypes: torch.Tensor  # [K, H, N]
    counts: torch.Tensor  # [K]


def fit_cluster_edge_prototypes(labels: torch.Tensor, attn_logits_l1: torch.Tensor) -> ClusterEdgeFitResult:
    if labels.ndim != 2:
        raise ValueError(f'Expected labels shape [B, N], got {labels.shape}.')
    if attn_logits_l1.ndim != 4:
        raise ValueError(f'Expected attn_logits_l1 shape [B, H, N, N], got {attn_logits_l1.shape}.')

    batch_size, token_count = labels.shape
    b2, num_heads, src_tokens, dst_tokens = attn_logits_l1.shape
    if batch_size != b2 or token_count != src_tokens:
        raise ValueError('labels and attn_logits_l1 shapes are incompatible.')
    if src_tokens != dst_tokens:
        raise ValueError('Expected square attention logits on token dimensions.')

    num_clusters = int(labels.max().item()) + 1
    prototypes = torch.zeros(num_clusters, num_heads, dst_tokens, dtype=attn_logits_l1.dtype)
    counts = torch.zeros(num_clusters, dtype=torch.long)

    # Aggregate logits by source-token cluster label.
    for b in range(batch_size):
        for i in range(src_tokens):
            k = int(labels[b, i].item())
            prototypes[k] += attn_logits_l1[b, :, i, :]
            counts[k] += 1

    safe_counts = counts.clamp_min(1).to(attn_logits_l1.dtype).view(-1, 1, 1)
    prototypes = prototypes / safe_counts

    return ClusterEdgeFitResult(prototypes=prototypes, counts=counts)


def reconstruct_logits_from_prototypes(labels: torch.Tensor, prototypes: torch.Tensor) -> torch.Tensor:
    if labels.ndim != 2:
        raise ValueError(f'Expected labels shape [B, N], got {labels.shape}.')
    if prototypes.ndim != 3:
        raise ValueError(f'Expected prototypes shape [K, H, N], got {prototypes.shape}.')

    batch_size, token_count = labels.shape
    num_clusters, num_heads, dst_tokens = prototypes.shape
    if token_count != dst_tokens:
        raise ValueError('labels token count and prototype dst token count must match.')

    if labels.max().item() >= num_clusters:
        raise ValueError('labels contains cluster id outside prototype range.')

    pred = torch.zeros(batch_size, num_heads, token_count, dst_tokens, dtype=prototypes.dtype)
    for b in range(batch_size):
        for i in range(token_count):
            pred[b, :, i, :] = prototypes[int(labels[b, i].item())]
    return pred


def mse_and_r2(y_true: torch.Tensor, y_pred: torch.Tensor) -> tuple[float, float]:
    if y_true.shape != y_pred.shape:
        raise ValueError('y_true and y_pred must have the same shape.')

    y_true_f = y_true.float()
    y_pred_f = y_pred.float()
    mse = torch.mean((y_true_f - y_pred_f) ** 2).item()

    var = torch.var(y_true_f, unbiased=False).item()
    if var == 0:
        r2 = 1.0 if mse == 0 else 0.0
    else:
        r2 = 1.0 - mse / var
    return float(mse), float(r2)
