from __future__ import annotations

from dataclasses import dataclass

import torch

from clip_circuit.local_linear import reconstruct_logits_from_prototypes


@dataclass
class ClusterInterventionEffect:
    cluster_id: int
    support: int
    ablation_effect: float
    injection_effect: float
    ablation_pvalue: float
    injection_pvalue: float


def _mean_abs_delta_on_rows(
    baseline: torch.Tensor,
    changed: torch.Tensor,
    row_mask: torch.Tensor,
) -> float:
    # baseline/changed: [B, H, N, N], row_mask: [B, N]
    row_mask4 = row_mask[:, None, :, None].expand_as(baseline)
    if row_mask4.sum().item() == 0:
        return 0.0
    delta = (baseline - changed).abs()
    return float(delta[row_mask4].mean().item())


def _row_effects(
    baseline: torch.Tensor,
    changed: torch.Tensor,
) -> torch.Tensor:
    # Return per-row effect with shape [B, N].
    delta = (baseline - changed).abs()
    return delta.mean(dim=(1, 3))


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


def summarize_cluster_interventions(
    labels: torch.Tensor,
    prototypes: torch.Tensor,
    max_clusters: int | None = None,
    num_permutations: int = 200,
    random_state: int = 0,
) -> list[ClusterInterventionEffect]:
    if labels.ndim != 2:
        raise ValueError(f'Expected labels shape [B, N], got {labels.shape}.')
    if prototypes.ndim != 3:
        raise ValueError(f'Expected prototypes shape [K, H, N], got {prototypes.shape}.')

    baseline = reconstruct_logits_from_prototypes(labels, prototypes)
    num_clusters = prototypes.shape[0]
    limit = num_clusters if max_clusters is None else min(max_clusters, num_clusters)

    effects: list[ClusterInterventionEffect] = []
    for k in range(limit):
        mask_k = labels == k
        support = int(mask_k.sum().item())

        ablated = prototypes.clone()
        ablated[k] = 0.0
        pred_ablated = reconstruct_logits_from_prototypes(labels, ablated)
        ablation_effect = _mean_abs_delta_on_rows(baseline, pred_ablated, mask_k)
        ablation_row_effects = _row_effects(baseline, pred_ablated)

        non_k_mask = ~mask_k
        injected_pred = baseline.clone()
        if non_k_mask.any():
            for b in range(labels.shape[0]):
                for i in range(labels.shape[1]):
                    if bool(non_k_mask[b, i].item()):
                        injected_pred[b, :, i, :] = prototypes[k]
        injection_effect = _mean_abs_delta_on_rows(baseline, injected_pred, non_k_mask)
        injection_row_effects = _row_effects(baseline, injected_pred)

        ablation_pvalue = _permutation_pvalue(
            observed_values=ablation_row_effects[mask_k],
            reference_values=ablation_row_effects[non_k_mask],
            num_permutations=num_permutations,
            random_state=random_state + k * 2,
        )
        injection_pvalue = _permutation_pvalue(
            observed_values=injection_row_effects[non_k_mask],
            reference_values=injection_row_effects[mask_k],
            num_permutations=num_permutations,
            random_state=random_state + k * 2 + 1,
        )

        effects.append(
            ClusterInterventionEffect(
                cluster_id=k,
                support=support,
                ablation_effect=ablation_effect,
                injection_effect=injection_effect,
                ablation_pvalue=ablation_pvalue,
                injection_pvalue=injection_pvalue,
            )
        )

    effects.sort(key=lambda x: x.ablation_effect, reverse=True)
    return effects
