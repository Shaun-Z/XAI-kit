import torch

from clip_circuit.hooks import build_attention_logits


def test_build_attention_logits_shape_and_values():
    # B=1, heads=2, tokens=3, dim=4
    q = torch.arange(1, 1 + 1 * 2 * 3 * 4, dtype=torch.float32).reshape(1, 2, 3, 4)
    k = q.clone()

    logits = build_attention_logits(q, k)

    assert logits.shape == (1, 2, 3, 3)
    manual = torch.matmul(q[0, 0], k[0, 0].transpose(0, 1)) / (4 ** 0.5)
    assert torch.allclose(logits[0, 0], manual)
