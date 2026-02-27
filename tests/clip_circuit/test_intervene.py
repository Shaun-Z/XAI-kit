import torch

from clip_circuit.intervene import summarize_cluster_interventions


def test_summarize_cluster_interventions_outputs_effects_with_pvalues():
    labels = torch.tensor([[0, 0, 1, 1]], dtype=torch.long)
    prototypes = torch.tensor(
        [
            [[2.0, 2.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.2, 0.2]],
        ]
    )  # [K=2, H=1, N=4]

    effects = summarize_cluster_interventions(
        labels=labels,
        prototypes=prototypes,
        max_clusters=2,
        num_permutations=200,
        random_state=0,
    )

    assert len(effects) == 2
    ids = {e.cluster_id for e in effects}
    assert ids == {0, 1}

    for e in effects:
        assert e.support == 2
        assert e.ablation_effect >= 0.0
        assert e.injection_effect >= 0.0
        assert 0.0 <= e.ablation_pvalue <= 1.0
        assert 0.0 <= e.injection_pvalue <= 1.0

    # cluster 0 has stronger prototype, should have stronger ablation effect.
    effect0 = [e for e in effects if e.cluster_id == 0][0]
    effect1 = [e for e in effects if e.cluster_id == 1][0]
    assert effect0.ablation_effect > effect1.ablation_effect
