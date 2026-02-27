import torch

from clip_circuit.local_linear import fit_cluster_edge_prototypes, mse_and_r2, reconstruct_logits_from_prototypes


def test_fit_cluster_edge_prototypes_and_reconstruct_exact():
    labels = torch.tensor([[0, 1]], dtype=torch.long)  # B=1, N=2

    # H=1, N=2: source token 0 always [1,2], source token 1 always [3,4]
    attn_logits = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])

    fit = fit_cluster_edge_prototypes(labels, attn_logits)
    pred = reconstruct_logits_from_prototypes(labels, fit.prototypes)
    mse, r2 = mse_and_r2(attn_logits, pred)

    assert fit.prototypes.shape == (2, 1, 2)
    assert torch.equal(fit.counts, torch.tensor([1, 1]))
    assert torch.allclose(pred, attn_logits)
    assert mse == 0.0
    assert r2 == 1.0


def test_reconstruct_rejects_out_of_range_label():
    labels = torch.tensor([[0, 2]], dtype=torch.long)
    prototypes = torch.zeros(2, 1, 2)
    try:
        _ = reconstruct_logits_from_prototypes(labels, prototypes)
    except ValueError as exc:
        assert 'outside prototype range' in str(exc)
    else:
        raise AssertionError('Expected ValueError for out-of-range label.')
