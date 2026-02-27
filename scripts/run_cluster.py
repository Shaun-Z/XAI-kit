from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from clip_circuit.gate_cluster import cluster_ffn_gates


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Cluster CLIP FFN gate features from trace file.')
    parser.add_argument('--trace', type=Path, required=True)
    parser.add_argument('--num-clusters', type=int, default=16)
    parser.add_argument('--sample-tokens', type=int, default=20000)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out', type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    trace = torch.load(args.trace, map_location='cpu')
    if 'ffn_gate_l' not in trace:
        raise SystemExit(f'Missing key ffn_gate_l in trace file: {args.trace}')

    result = cluster_ffn_gates(
        ffn_gate_l=trace['ffn_gate_l'],
        num_clusters=args.num_clusters,
        sample_tokens=args.sample_tokens,
        random_state=args.seed,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            'trace_file': str(args.trace),
            'layer': trace.get('layer'),
            'next_layer': trace.get('next_layer'),
            'num_clusters': args.num_clusters,
            'labels': result.labels,
            'centroids': result.centroids,
            'cluster_sizes': result.cluster_sizes,
            'cluster_activation_rates': result.cluster_activation_rates,
        },
        args.out,
    )

    print(
        f'Saved clusters to {args.out} with {args.num_clusters} clusters; '
        f'tokens={result.labels.numel()}.'
    )


if __name__ == '__main__':
    main()
