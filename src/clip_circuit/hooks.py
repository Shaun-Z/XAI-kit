from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F


@dataclass
class TraceStore:
    layer_index: int
    next_layer_index: int
    h_l: torch.Tensor | None = None
    ffn_gate_l: torch.Tensor | None = None
    q_l1: torch.Tensor | None = None
    k_l1: torch.Tensor | None = None
    attn_logits_l1: torch.Tensor | None = None
    _handles: list[Any] = field(default_factory=list)

    def clear(self) -> None:
        self.h_l = None
        self.ffn_gate_l = None
        self.q_l1 = None
        self.k_l1 = None
        self.attn_logits_l1 = None

    def remove(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()


def build_attention_logits(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    if q.ndim != 4 or k.ndim != 4:
        raise ValueError('Expected q and k to have shape [B, H, N, D].')
    if q.shape != k.shape:
        raise ValueError(f'q and k must have the same shape, got {q.shape} vs {k.shape}.')

    head_dim = q.shape[-1]
    return torch.matmul(q, k.transpose(-2, -1)) / (head_dim ** 0.5)


def _extract_qk_from_vision_attention(attn_module: torch.nn.Module, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if hasattr(attn_module, "qkv"):
        qkv = attn_module.qkv(hidden_states)
        batch_size, token_count, _ = qkv.shape
        num_heads = attn_module.num_heads
        head_dim = qkv.shape[-1] // (3 * num_heads)
        qkv = qkv.reshape(batch_size, token_count, 3, num_heads, head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        return qkv[0], qkv[1]

    if not hasattr(attn_module, "in_proj_weight"):
        raise ValueError("Unsupported attention module: expected qkv or in_proj_weight.")

    if hidden_states.ndim != 3:
        raise ValueError(f"Expected 3D hidden_states, got shape {hidden_states.shape}.")

    # open_clip MultiheadAttention takes [N, B, C] unless batch_first=True.
    if getattr(attn_module, "batch_first", False):
        x = hidden_states
    else:
        x = hidden_states.transpose(0, 1)

    qkv = F.linear(x, attn_module.in_proj_weight, attn_module.in_proj_bias)
    q, k, _ = qkv.chunk(3, dim=-1)

    batch_size, token_count, _ = q.shape
    num_heads = attn_module.num_heads
    head_dim = q.shape[-1] // num_heads

    q = q.reshape(batch_size, token_count, num_heads, head_dim).permute(0, 2, 1, 3)
    k = k.reshape(batch_size, token_count, num_heads, head_dim).permute(0, 2, 1, 3)
    return q, k


def register_clip_vit_hooks(model: torch.nn.Module, layer_index: int, next_layer_index: int) -> TraceStore:
    vision_layers = model.visual.transformer.resblocks
    if layer_index < 0 or next_layer_index < 0:
        raise ValueError('Layer indices must be non-negative.')
    if layer_index >= len(vision_layers) or next_layer_index >= len(vision_layers):
        raise ValueError('Layer index out of range for model.visual.transformer.resblocks.')

    layer_l = vision_layers[layer_index]
    layer_l1 = vision_layers[next_layer_index]
    store = TraceStore(layer_index=layer_index, next_layer_index=next_layer_index)

    def _ffn_in_hook(_module: torch.nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
        store.h_l = inputs[0].detach().cpu()

    def _ffn_out_hook(_module: torch.nn.Module, _inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        store.ffn_gate_l = (output.detach() > 0).to(torch.uint8).cpu()

    def _next_attn_in_hook(module: torch.nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
        hidden_states = inputs[0]
        q, k = _extract_qk_from_vision_attention(module, hidden_states)
        logits = build_attention_logits(q, k)
        store.q_l1 = q.detach().cpu()
        store.k_l1 = k.detach().cpu()
        store.attn_logits_l1 = logits.detach().cpu()

    store._handles.extend(
        [
            layer_l.mlp[0].register_forward_pre_hook(_ffn_in_hook),
            layer_l.mlp[1].register_forward_hook(_ffn_out_hook),
            layer_l1.attn.register_forward_pre_hook(_next_attn_in_hook),
        ]
    )

    return store
