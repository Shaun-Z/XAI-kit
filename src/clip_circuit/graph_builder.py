from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class CircuitPath:
    cluster_id: int
    head: int
    dst_token: int
    weight: float
    support: int

    def to_dict(self) -> dict[str, int | float]:
        return {
            "cluster_id": self.cluster_id,
            "head": self.head,
            "dst_token": self.dst_token,
            "weight": self.weight,
            "support": self.support,
        }


def build_topk_circuit_paths(
    prototypes: torch.Tensor,
    counts: torch.Tensor,
    top_k: int = 50,
) -> list[CircuitPath]:
    if prototypes.ndim != 3:
        raise ValueError(f'Expected prototypes shape [K, H, N], got {prototypes.shape}.')
    if counts.ndim != 1:
        raise ValueError(f'Expected counts shape [K], got {counts.shape}.')
    if prototypes.shape[0] != counts.shape[0]:
        raise ValueError('prototypes and counts must share the same cluster dimension.')
    if top_k <= 0:
        raise ValueError('top_k must be positive.')

    k_count, num_heads, num_tokens = prototypes.shape
    flat = prototypes.reshape(k_count, num_heads * num_tokens)

    scored: list[tuple[float, int, int, int]] = []
    for k in range(k_count):
        for flat_idx in range(num_heads * num_tokens):
            head = flat_idx // num_tokens
            dst_token = flat_idx % num_tokens
            weight = float(flat[k, flat_idx].item())
            scored.append((abs(weight), k, head, dst_token))

    scored.sort(key=lambda x: x[0], reverse=True)
    selected = scored[:top_k]

    paths: list[CircuitPath] = []
    for _, k, head, dst_token in selected:
        paths.append(
            CircuitPath(
                cluster_id=k,
                head=head,
                dst_token=dst_token,
                weight=float(prototypes[k, head, dst_token].item()),
                support=int(counts[k].item()),
            )
        )
    return paths


def build_path_groups(paths: list[CircuitPath]) -> dict[str, dict[str, list[dict[str, int | float]]]]:
    by_head: dict[str, list[dict[str, int | float]]] = {}
    by_cluster: dict[str, list[dict[str, int | float]]] = {}

    for path in paths:
        item = path.to_dict()
        head_key = f"head_{path.head}"
        cluster_key = f"cluster_{path.cluster_id}"
        by_head.setdefault(head_key, []).append(item)
        by_cluster.setdefault(cluster_key, []).append(item)

    return {"by_head": by_head, "by_cluster": by_cluster}
