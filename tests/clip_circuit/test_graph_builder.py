import torch

from clip_circuit.graph_builder import build_path_groups, build_topk_circuit_paths


def test_build_topk_circuit_paths_returns_sorted_entries():
    prototypes = torch.tensor(
        [
            [[0.1, -2.0, 0.3]],
            [[1.5, 0.2, -0.4]],
        ]
    )  # [K=2, H=1, N=3]
    counts = torch.tensor([5, 7])

    paths = build_topk_circuit_paths(prototypes=prototypes, counts=counts, top_k=2)

    assert len(paths) == 2
    assert paths[0].cluster_id == 0
    assert paths[0].dst_token == 1
    assert abs(paths[0].weight + 2.0) < 1e-6
    assert paths[0].support == 5

    assert paths[1].cluster_id == 1
    assert paths[1].dst_token == 0
    assert abs(paths[1].weight - 1.5) < 1e-6


def test_build_path_groups_by_head_and_cluster():
    prototypes = torch.tensor([[[2.0, -1.0]], [[0.5, 0.1]]])  # [K=2,H=1,N=2]
    counts = torch.tensor([3, 9])
    paths = build_topk_circuit_paths(prototypes=prototypes, counts=counts, top_k=3)

    groups = build_path_groups(paths)

    assert 'head_0' in groups['by_head']
    assert 'cluster_0' in groups['by_cluster']
    assert len(groups['by_head']['head_0']) == 3
    assert groups['by_cluster']['cluster_0'][0]['support'] == 3
