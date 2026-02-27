from __future__ import annotations

import argparse
from pathlib import Path
import sys

import open_clip
import torch
from torchvision.datasets import ImageNet

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from clip_circuit.hooks import register_clip_vit_hooks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Collect CLIP ViT trace for circuit analysis.')
    parser.add_argument('--model', default='ViT-B-16')
    parser.add_argument('--pretrained', default='openai')
    parser.add_argument('--imagenet-root', type=Path, required=True)
    parser.add_argument('--split', choices=['train', 'val'], default='val')
    parser.add_argument('--target-layer', type=int, required=True)
    parser.add_argument('--next-attn-layer', type=int, required=True)
    parser.add_argument('--max-images', type=int, default=32)
    parser.add_argument('--out', type=Path, required=True)
    return parser.parse_args()


def get_imagenet_id(dataset: ImageNet, index: int) -> str:
    if hasattr(dataset, 'samples') and index < len(dataset.samples):
        sample_path = Path(dataset.samples[index][0])
        try:
            return str(sample_path.relative_to(dataset.root))
        except ValueError:
            return str(sample_path)
    return f'index:{index}'


def main() -> None:
    args = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model, _, preprocess = open_clip.create_model_and_transforms(args.model, pretrained=args.pretrained)
    model = model.to(device).eval()

    if not args.imagenet_root.exists():
        raise SystemExit(f'ImageNet root does not exist: {args.imagenet_root}')

    dataset = ImageNet(root=str(args.imagenet_root), split=args.split)
    if len(dataset) == 0:
        raise SystemExit(f'ImageNet split is empty: root={args.imagenet_root}, split={args.split}')
    limit = min(args.max_images, len(dataset))

    store = register_clip_vit_hooks(model, args.target_layer, args.next_attn_layer)

    images: list[str] = []
    h_l: list[torch.Tensor] = []
    ffn_gate_l: list[torch.Tensor] = []
    attn_logits_l1: list[torch.Tensor] = []

    with torch.no_grad():
        for idx in range(limit):
            image, _ = dataset[idx]
            image = image.convert('RGB')
            pixel_values = preprocess(image).unsqueeze(0).to(device)
            store.clear()
            _ = model.encode_image(pixel_values)

            if store.h_l is None or store.ffn_gate_l is None or store.attn_logits_l1 is None:
                continue

            images.append(get_imagenet_id(dataset, idx))
            h_l.append(store.h_l)
            ffn_gate_l.append(store.ffn_gate_l)
            attn_logits_l1.append(store.attn_logits_l1)

    store.remove()

    if not images:
        raise SystemExit('Tracing produced no valid records.')

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            'images': images,
            'layer': args.target_layer,
            'next_layer': args.next_attn_layer,
            'h_l': torch.cat(h_l, dim=0),
            'ffn_gate_l': torch.cat(ffn_gate_l, dim=0),
            'attn_logits_l1': torch.cat(attn_logits_l1, dim=0),
        },
        args.out,
    )
    print(f'Saved trace to {args.out} with {len(images)} images.')


if __name__ == '__main__':
    main()
