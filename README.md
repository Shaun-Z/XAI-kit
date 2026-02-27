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
  --imagenet-root /data/imagenet \
  --split val \
  --target-layer 6 \
  --next-attn-layer 7 \
  --max-images 512 \
  --out artifacts/imagenet_val/traces/l6_l7_512.pt
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
  --trace artifacts/imagenet_val/traces/l6_l7_512.pt \
  --num-clusters 16 \
  --sample-tokens 20000 \
  --seed 0 \
  --out artifacts/imagenet_val/features/l6_gates_k16_512.pt
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
  --trace artifacts/imagenet_val/traces/l6_l7_512.pt \
  --features artifacts/imagenet_val/features/l6_gates_k16_512.pt \
  --out artifacts/imagenet_val/circuits/l6_edge_fit_k16_512.pt
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
  --fit artifacts/imagenet_val/circuits/l6_edge_fit_k16_512.pt \
  --top-k 50 \
  --heatmap-dir artifacts/imagenet_val/circuits/heatmaps_k16_512 \
  --out artifacts/imagenet_val/circuits/l6_topk_paths_k16_512.json
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
  --features artifacts/imagenet_val/features/l6_gates_k16_512.pt \
  --fit artifacts/imagenet_val/circuits/l6_edge_fit_k16_512.pt \
  --max-clusters 16 \
  --num-permutations 200 \
  --seed 0 \
  --out artifacts/imagenet_val/reports/l6_causal_k16_512.json
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
  --trace artifacts/imagenet_val/traces/l6_l7_512.pt \
  --features artifacts/imagenet_val/features/l6_gates_k16_512.pt \
  --fit artifacts/imagenet_val/circuits/l6_edge_fit_k16_512.pt \
  --circuit artifacts/imagenet_val/circuits/l6_topk_paths_k16_512.json \
  --top-n 20 \
  --num-permutations 200 \
  --seed 0 \
  --out artifacts/imagenet_val/reports/l6_target_causal_k16_512.json
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

## Script Roles (Detailed)

This section explains what each runner script does in the pipeline and how its
output is consumed by later stages.

### `scripts/run_trace.py`

Purpose:
- Collect first-order CLIP ViT traces from real images as the raw data source
  for all downstream analysis.

Main inputs:
- Model config: `--model`, `--pretrained`
- Data source: `--imagenet-root`, `--split {train,val}` (default `val`)
- Analysis target: `--target-layer`, `--next-attn-layer`
- Sampling budget: `--max-images`

What it does:
- Loads CLIP with `open_clip`.
- Loads ImageNet using `torchvision.datasets.ImageNet`.
- Registers hooks to capture:
  - target layer hidden states (`h_l`)
  - target layer FFN gate activations (`ffn_gate_l`)
  - next attention layer logits (`attn_logits_l1`)
- Iterates dataset samples, preprocesses each image, runs `model.encode_image`,
  and stores captured tensors.

Output and downstream use:
- Writes a `.pt` trace file.
- This file is the source of:
  - gate clustering (`run_cluster.py`)
  - edge prototype fitting (`run_fit.py`)
  - real-logit causal testing (`run_target_causal.py`)

### `scripts/run_cluster.py`

Purpose:
- Convert dense FFN gate activations into discrete token-level feature groups
  (clusters) that become circuit "feature nodes."

Main inputs:
- `--trace`: output from `run_trace.py`
- `--num-clusters`: number of cluster centroids `K`
- `--sample-tokens`: optional token subsampling for faster clustering
- `--seed`: clustering randomness control

What it does:
- Loads `ffn_gate_l` from trace.
- Runs `cluster_ffn_gates(...)` to assign each token to a cluster.
- Computes cluster summary statistics (sizes and activation rates).

Output and downstream use:
- Writes a `.pt` feature file containing `labels`, `centroids`,
  `cluster_sizes`, `cluster_activation_rates`.
- `labels` is consumed by:
  - `run_fit.py` (prototype fitting)
  - `run_causal.py` (prototype-space interventions)
  - `run_target_causal.py` (target-edge causal effects)

### `scripts/run_fit.py`

Purpose:
- Learn cluster-conditioned attention edge prototypes, i.e. average edge
  patterns per cluster.

Main inputs:
- `--trace`: provides real `attn_logits_l1`
- `--features`: provides token cluster `labels`

What it does:
- Fits `prototypes[K, H, N]` via `fit_cluster_edge_prototypes(...)`.
- Reconstructs attention logits from prototypes and labels.
- Computes reconstruction quality (`mse`, `r2`).

Output and downstream use:
- Writes a `.pt` fit file with `prototypes`, `counts`, `mse`, `r2`.
- This file is consumed by:
  - `run_circuit.py` to extract top-k paths
  - `run_causal.py` for intervention analysis
  - `run_target_causal.py` for edge-level causal effects

### `scripts/run_circuit.py`

Purpose:
- Export interpretable circuit edges (top-k strongest prototype edges) and
  optional visualization assets.

Main inputs:
- `--fit`: output from `run_fit.py`
- `--top-k`: number of path entries to keep globally
- `--heatmap-dir`: optional path to save per-head prototype heatmaps

What it does:
- Calls `build_topk_circuit_paths(...)` to rank edges by weight/support-derived
  priority.
- Builds grouped summaries (`by_head`, `by_cluster`) for inspection.
- Optionally renders per-head heatmaps from `prototypes`.

Output and downstream use:
- Writes a JSON circuit file with `paths` and grouped views.
- Used by `run_target_causal.py` to select candidate edges for causal testing.

### `scripts/run_causal.py`

Purpose:
- Run fast prototype-space causal diagnostics at the cluster level.

Main inputs:
- `--features`: cluster labels
- `--fit`: prototypes
- `--max-clusters`: limit number of clusters to evaluate
- `--num-permutations`, `--seed`: permutation-test settings

What it does:
- Applies cluster ablation/injection in prototype space via
  `summarize_cluster_interventions(...)`.
- Estimates effect sizes and permutation p-values for each tested cluster.

Output and downstream use:
- Writes a JSON report of cluster-level causal effects.
- Primarily used for ranking and diagnostic interpretation rather than being
  required by subsequent scripts.

### `scripts/run_target_causal.py`

Purpose:
- Evaluate causal impact of specific circuit edges against real trace logits
  (more grounded than prototype-only summaries).

Main inputs:
- `--trace`: real `attn_logits_l1`
- `--features`: cluster labels
- `--fit`: prototypes
- `--circuit`: top path candidates from `run_circuit.py`
- `--top-n`: number of candidate edges to evaluate
- `--num-permutations`, `--seed`: permutation-test settings

What it does:
- Selects top candidate edges from circuit paths.
- Computes:
  - edge effect (`edge_effect`, `edge_pvalue`)
  - prototype-real alignment gain (`alignment_gain`, `alignment_pvalue`)
- Uses permutation testing to estimate significance.

Output and downstream use:
- Writes a JSON report for target-edge causal evidence.
- This is currently the final quantitative report in the PoC pipeline.

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
