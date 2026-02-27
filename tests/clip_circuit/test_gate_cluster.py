import torch

from clip_circuit.gate_cluster import cluster_ffn_gates


def test_cluster_ffn_gates_shapes_and_signal():
    # Two clear groups in 4-dim gate space.
    group_a = torch.tensor([[1, 1, 0, 0], [1, 1, 0, 0]], dtype=torch.uint8)
    group_b = torch.tensor([[0, 0, 1, 1], [0, 0, 1, 1]], dtype=torch.uint8)
    ffn_gate = torch.stack([torch.cat([group_a, group_b], dim=0)], dim=0)  # [1, 4, 4]

    result = cluster_ffn_gates(ffn_gate, num_clusters=2, sample_tokens=0, random_state=0)

    assert result.labels.shape == (1, 4)
    assert result.centroids.shape == (2, 4)
    assert result.cluster_sizes.sum().item() == 4

    # Centroids should match the two prototype gate patterns up to permutation.
    prototypes = torch.tensor([[1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0]])
    dists = torch.cdist(result.centroids, prototypes)
    assert torch.min(dists[0]).item() < 0.2
    assert torch.min(dists[1]).item() < 0.2


def test_cluster_ffn_gates_invalid_rank_raises():
    bad = torch.zeros(4, 4, dtype=torch.uint8)
    try:
        _ = cluster_ffn_gates(bad, num_clusters=2)
    except ValueError as exc:
        assert 'Expected ffn_gate_l shape [B, N, D]' in str(exc)
    else:
        raise AssertionError('Expected ValueError for invalid rank input.')
