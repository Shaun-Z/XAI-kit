import torch

from clip_circuit.target_causal import summarize_target_edge_effects


def test_summarize_target_edge_effects_basic():
    labels = torch.tensor([[0, 1]], dtype=torch.long)
    prototypes = torch.tensor(
        [
            [[2.0, 0.0]],
            [[0.0, 0.5]],
        ]
    )  # [K=2,H=1,N=2]

    # Build teacher logits with strong dependence on cluster 0 at dst_token 0.
    attn_logits = torch.tensor([[[[2.2, 0.0], [0.1, 0.5]]]])
    paths = [
        {'cluster_id': 0, 'head': 0, 'dst_token': 0},
        {'cluster_id': 1, 'head': 0, 'dst_token': 1},
    ]

    effects = summarize_target_edge_effects(
        labels=labels,
        prototypes=prototypes,
        attn_logits_l1=attn_logits,
        paths=paths,
        top_n=2,
        num_permutations=100,
        random_state=0,
    )

    assert len(effects) == 2
    best = effects[0]
    assert best.cluster_id == 0
    assert best.head == 0
    assert best.dst_token == 0
    assert best.support == 1
    assert best.edge_effect > 0.0
    assert 0.0 <= best.edge_pvalue <= 1.0
    assert 0.0 <= best.alignment_pvalue <= 1.0
