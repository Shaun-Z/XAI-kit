from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from clip_circuit.intervene import summarize_cluster_interventions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run prototype-space ablation/injection analysis by cluster.')
    parser.add_argument('--features', type=Path, required=True)
    parser.add_argument('--fit', type=Path, required=True)
    parser.add_argument('--max-clusters', type=int, default=16)
    parser.add_argument('--num-permutations', type=int, default=200)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out', type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    features = torch.load(args.features, map_location='cpu')
    fit = torch.load(args.fit, map_location='cpu')

    if 'labels' not in features:
        raise SystemExit(f'Missing labels in features file: {args.features}')
    if 'prototypes' not in fit:
        raise SystemExit(f'Missing prototypes in fit file: {args.fit}')

    effects = summarize_cluster_interventions(
        labels=features['labels'].long(),
        prototypes=fit['prototypes'].float(),
        max_clusters=args.max_clusters,
        num_permutations=args.num_permutations,
        random_state=args.seed,
    )

    payload = {
        'features_file': str(args.features),
        'fit_file': str(args.fit),
        'max_clusters': args.max_clusters,
        'num_permutations': args.num_permutations,
        'seed': args.seed,
        'effects': [asdict(e) for e in effects],
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    print(f'Saved causal summary to {args.out}; effects={len(effects)}')


if __name__ == '__main__':
    main()
