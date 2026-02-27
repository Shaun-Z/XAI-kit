from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from clip_circuit.graph_builder import build_path_groups, build_topk_circuit_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Export top-k circuit paths from fitted prototypes.')
    parser.add_argument('--fit', type=Path, required=True)
    parser.add_argument('--top-k', type=int, default=50)
    parser.add_argument('--heatmap-dir', type=Path, default=None)
    parser.add_argument('--out', type=Path, required=True)
    return parser.parse_args()


def save_head_heatmaps(prototypes: torch.Tensor, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    k_count, num_heads, _ = prototypes.shape
    for head in range(num_heads):
        data = prototypes[:, head, :].cpu().numpy()  # [K, N]
        fig, ax = plt.subplots(figsize=(10, max(2, k_count * 0.25)))
        im = ax.imshow(data, aspect='auto', cmap='coolwarm')
        ax.set_title(f'Prototype Weights by Cluster (head={head})')
        ax.set_xlabel('dst_token')
        ax.set_ylabel('cluster_id')
        fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
        fig.tight_layout()
        fig.savefig(out_dir / f'head_{head}.png', dpi=150)
        plt.close(fig)


def main() -> None:
    args = parse_args()
    fit = torch.load(args.fit, map_location='cpu')

    if 'prototypes' not in fit or 'counts' not in fit:
        raise SystemExit(f'Missing prototypes/counts in fit file: {args.fit}')

    paths = build_topk_circuit_paths(
        prototypes=fit['prototypes'].float(),
        counts=fit['counts'].long(),
        top_k=args.top_k,
    )

    payload = {
        'fit_file': str(args.fit),
        'layer': fit.get('layer'),
        'next_layer': fit.get('next_layer'),
        'top_k': args.top_k,
        'paths': [p.to_dict() for p in paths],
        'groups': build_path_groups(paths),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    if args.heatmap_dir is not None:
        save_head_heatmaps(fit['prototypes'].float(), args.heatmap_dir)
    print(f'Saved circuit paths to {args.out}; paths={len(paths)}')


if __name__ == '__main__':
    main()
