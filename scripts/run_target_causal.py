from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / 'src'
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from clip_circuit.target_causal import summarize_target_edge_effects


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Evaluate target-edge causal effects on real trace logits.')
    parser.add_argument('--trace', type=Path, required=True)
    parser.add_argument('--features', type=Path, required=True)
    parser.add_argument('--fit', type=Path, required=True)
    parser.add_argument('--circuit', type=Path, required=True)
    parser.add_argument('--top-n', type=int, default=20)
    parser.add_argument('--num-permutations', type=int, default=200)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out', type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    trace = torch.load(args.trace, map_location='cpu')
    features = torch.load(args.features, map_location='cpu')
    fit = torch.load(args.fit, map_location='cpu')
    circuit = json.loads(args.circuit.read_text())

    for key in ['attn_logits_l1']:
        if key not in trace:
            raise SystemExit(f'Missing {key} in trace file: {args.trace}')
    if 'labels' not in features:
        raise SystemExit(f'Missing labels in features file: {args.features}')
    if 'prototypes' not in fit:
        raise SystemExit(f'Missing prototypes in fit file: {args.fit}')
    if 'paths' not in circuit:
        raise SystemExit(f'Missing paths in circuit file: {args.circuit}')

    effects = summarize_target_edge_effects(
        labels=features['labels'].long(),
        prototypes=fit['prototypes'].float(),
        attn_logits_l1=trace['attn_logits_l1'].float(),
        paths=circuit['paths'],
        top_n=args.top_n,
        num_permutations=args.num_permutations,
        random_state=args.seed,
    )

    payload = {
        'trace_file': str(args.trace),
        'features_file': str(args.features),
        'fit_file': str(args.fit),
        'circuit_file': str(args.circuit),
        'top_n': args.top_n,
        'num_permutations': args.num_permutations,
        'seed': args.seed,
        'effects': [asdict(e) for e in effects],
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    print(f'Saved target-edge causal summary to {args.out}; effects={len(effects)}')


if __name__ == '__main__':
    main()
