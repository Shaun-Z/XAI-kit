# XAI-kit

A lightweight workspace for interpretability experiments, currently focused on
CLIP ViT circuit-style analysis:

1. Trace hidden states / FFN gates / next-layer attention logits
2. Cluster FFN gate patterns into token-level feature groups
3. Fit cluster-conditioned attention edge prototypes
4. Export top-k circuit paths
5. Run prototype-space ablation/injection analysis
6. Run target-edge causal tests on real trace logits

## Agent Handoff

- Detailed transfer document for a new coding agent:
  - `docs/handoffs/2026-02-27-clip-circuit-agent-handoff.md`

## Setup

```bash
cd /Users/xiangyuzhou/Documents/GitRepo/XAI-kit
uv sync
```

## Circuit PoC Pipeline (CLIP ViT)

### Step 1: Collect trace

```bash
uv run python scripts/run_trace.py \
  --model ViT-B-16 \
  --pretrained openai \
  --image-dir artifacts/sample_images \
  --target-layer 6 \
  --next-attn-layer 7 \
  --max-images 32 \
  --out artifacts/traces/day1_l6_l7.pt
```

Output keys:
- `images`
- `layer`
- `next_layer`
- `h_l` shape `[B, N, D_model]`
- `ffn_gate_l` shape `[B, N, D_ffn]`
- `attn_logits_l1` shape `[B, H, N, N]`

### Step 2: Cluster FFN gates

```bash
uv run python scripts/run_cluster.py \
  --trace artifacts/traces/day1_l6_l7.pt \
  --num-clusters 16 \
  --sample-tokens 20000 \
  --out artifacts/features/day2_l6_gates_k16.pt
```

Output keys:
- `trace_file`
- `layer`
- `next_layer`
- `num_clusters`
- `labels` shape `[B, N]`
- `centroids` shape `[K, D_ffn]`
- `cluster_sizes` shape `[K]`
- `cluster_activation_rates` shape `[K]`

### Step 3: Fit edge prototypes

```bash
uv run python scripts/run_fit.py \
  --trace artifacts/traces/day1_l6_l7.pt \
  --features artifacts/features/day2_l6_gates_k16.pt \
  --out artifacts/circuits/day3_l6_edge_fit_k16.pt
```

Output keys:
- `trace_file`
- `features_file`
- `layer`
- `next_layer`
- `prototypes` shape `[K, H, N]`
- `counts` shape `[K]`
- `mse`
- `r2`

### Step 4: Export top-k circuit paths

```bash
uv run python scripts/run_circuit.py \
  --fit artifacts/circuits/day3_l6_edge_fit_k16.pt \
  --top-k 50 \
  --heatmap-dir artifacts/circuits/day4_heatmaps_k16 \
  --out artifacts/circuits/day4_l6_topk_paths_k16.json
```

Output keys:
- `fit_file`
- `layer`
- `next_layer`
- `top_k`
- `paths` with records:
  - `cluster_id`
  - `head`
  - `dst_token`
  - `weight`
  - `support`
- `groups`:
  - `by_head`
  - `by_cluster`

Heatmaps:
- `artifacts/circuits/day4_heatmaps_k16/head_<head_id>.png`

### Step 5: Prototype-space causal summary

```bash
uv run python scripts/run_causal.py \
  --features artifacts/features/day2_l6_gates_k16.pt \
  --fit artifacts/circuits/day3_l6_edge_fit_k16.pt \
  --max-clusters 16 \
  --num-permutations 200 \
  --seed 0 \
  --out artifacts/reports/day5_l6_causal_k16.json
```

Output keys:
- `features_file`
- `fit_file`
- `max_clusters`
- `num_permutations`
- `seed`
- `effects` with records:
  - `cluster_id`
  - `support`
  - `ablation_effect`
  - `injection_effect`
  - `ablation_pvalue`
  - `injection_pvalue`

### Step 6: Target-edge causal test (real trace)

```bash
uv run python scripts/run_target_causal.py \
  --trace artifacts/traces/day1_l6_l7.pt \
  --features artifacts/features/day2_l6_gates_k16.pt \
  --fit artifacts/circuits/day3_l6_edge_fit_k16.pt \
  --circuit artifacts/circuits/day4_l6_topk_paths_k16.json \
  --top-n 20 \
  --num-permutations 200 \
  --seed 0 \
  --out artifacts/reports/day6_l6_target_causal_k16.json
```

Output keys:
- `trace_file`
- `features_file`
- `fit_file`
- `circuit_file`
- `top_n`
- `num_permutations`
- `seed`
- `effects` with records:
  - `cluster_id`
  - `head`
  - `dst_token`
  - `support`
  - `edge_effect`
  - `edge_pvalue`
  - `alignment_gain`
  - `alignment_pvalue`

## Tests

```bash
uv run pytest tests/clip_circuit -q
```

Current coverage includes:
- attention logit reconstruction utility
- FFN gate clustering behavior and input validation
- cluster-conditioned edge prototype fitting and reconstruction
- top-k circuit path extraction
- prototype-space intervention summary
- target-edge causal effect testing
