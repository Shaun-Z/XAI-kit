from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from clip_circuit.local_linear import reconstruct_logits_from_prototypes


@dataclass
class TargetEdgeEffect:
    cluster_id: int
    head: int
    dst_token: int
    support: int
    edge_effect: float
    edge_pvalue: float
    alignment_gain: float
    alignment_pvalue: float


def _permutation_pvalue(
    observed_values: torch.Tensor,
    reference_values: torch.Tensor,
    num_permutations: int,
    random_state: int,
) -> float:
    if observed_values.numel() == 0 or reference_values.numel() == 0:
        return 1.0
    if num_permutations <= 0:
        return 1.0

    rng = torch.Generator(device=reference_values.device)
    rng.manual_seed(random_state)
    observed_mean = float(observed_values.mean().item())
    n = observed_values.numel()
    m = reference_values.numel()
    replace = n > m

    ge_count = 0
    for _ in range(num_permutations):
        if replace:
            idx = torch.randint(0, m, (n,), generator=rng)
        else:
            idx = torch.randperm(m, generator=rng)[:n]
        sample_mean = float(reference_values[idx].mean().item())
        if sample_mean >= observed_mean:
            ge_count += 1

    return float((ge_count + 1) / (num_permutations + 1))


def summarize_target_edge_effects(
    labels: torch.Tensor,
    prototypes: torch.Tensor,
    attn_logits_l1: torch.Tensor,
    paths: list[dict[str, Any]],
    top_n: int = 20,
    num_permutations: int = 200,
    random_state: int = 0,
) -> list[TargetEdgeEffect]:
    if labels.ndim != 2:
        raise ValueError(f'Expected labels shape [B, N], got {labels.shape}.')
    if prototypes.ndim != 3:
        raise ValueError(f'Expected prototypes shape [K, H, N], got {prototypes.shape}.')
    if attn_logits_l1.ndim != 4:
        raise ValueError(f'Expected attn_logits_l1 shape [B, H, N, N], got {attn_logits_l1.shape}.')

    baseline = reconstruct_logits_from_prototypes(labels, prototypes)
    b, n = labels.shape
    if baseline.shape[:3] != (b, prototypes.shape[1], n):
        raise ValueError('Shape mismatch between labels and prototypes.')

    selected = paths[:top_n]
    effects: list[TargetEdgeEffect] = []

    for idx, path in enumerate(selected):
        k = int(path['cluster_id'])
        h = int(path['head'])
        j = int(path['dst_token'])

        if not (0 <= k < prototypes.shape[0] and 0 <= h < prototypes.shape[1] and 0 <= j < prototypes.shape[2]):
            continue

        mask_k = labels == k
        non_k = ~mask_k
        support = int(mask_k.sum().item())

        base_edge = baseline[:, h, :, j]

        ablated = prototypes.clone()
        ablated[k, h, j] = 0.0
        ablated_pred = reconstruct_logits_from_prototypes(labels, ablated)
        ablated_edge = ablated_pred[:, h, :, j]

        edge_delta = (base_edge - ablated_edge).abs()
        edge_effect = float(edge_delta[mask_k].mean().item()) if support > 0 else 0.0
        edge_pvalue = _permutation_pvalue(
            observed_values=edge_delta[mask_k],
            reference_values=edge_delta[non_k],
            num_permutations=num_permutations,
            random_state=random_state + idx * 2,
        )

        teacher_edge = attn_logits_l1[:, h, :, j]
        base_err = (teacher_edge - base_edge).abs()
        ablated_err = (teacher_edge - ablated_edge).abs()
        err_increase = ablated_err - base_err
        alignment_gain = float(err_increase[mask_k].mean().item()) if support > 0 else 0.0
        alignment_pvalue = _permutation_pvalue(
            observed_values=err_increase[mask_k],
            reference_values=err_increase[non_k],
            num_permutations=num_permutations,
            random_state=random_state + idx * 2 + 1,
        )

        effects.append(
            TargetEdgeEffect(
                cluster_id=k,
                head=h,
                dst_token=j,
                support=support,
                edge_effect=edge_effect,
                edge_pvalue=edge_pvalue,
                alignment_gain=alignment_gain,
                alignment_pvalue=alignment_pvalue,
            )
        )

    effects.sort(key=lambda x: x.edge_effect, reverse=True)
    return effects
