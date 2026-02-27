from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from clip_circuit.local_linear import fit_cluster_edge_prototypes, mse_and_r2, reconstruct_logits_from_prototypes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Fit cluster-conditioned attention edge prototypes.')
    parser.add_argument('--trace', type=Path, required=True)
    parser.add_argument('--features', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    trace = torch.load(args.trace, map_location='cpu')
    features = torch.load(args.features, map_location='cpu')

    if 'attn_logits_l1' not in trace:
        raise SystemExit(f'Missing key attn_logits_l1 in trace file: {args.trace}')
    if 'labels' not in features:
        raise SystemExit(f'Missing key labels in features file: {args.features}')

    labels = features['labels'].long()
    attn_logits = trace['attn_logits_l1'].float()

    fit = fit_cluster_edge_prototypes(labels=labels, attn_logits_l1=attn_logits)
    pred = reconstruct_logits_from_prototypes(labels=labels, prototypes=fit.prototypes)
    mse, r2 = mse_and_r2(attn_logits, pred)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            'trace_file': str(args.trace),
            'features_file': str(args.features),
            'layer': trace.get('layer'),
            'next_layer': trace.get('next_layer'),
            'prototypes': fit.prototypes,
            'counts': fit.counts,
            'mse': mse,
            'r2': r2,
        },
        args.out,
    )

    print(f'Saved fit to {args.out}; mse={mse:.6f}, r2={r2:.6f}')


if __name__ == '__main__':
    main()
