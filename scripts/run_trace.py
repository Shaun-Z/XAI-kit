from __future__ import annotations

import argparse
from pathlib import Path
import sys

import open_clip
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from clip_circuit.hooks import register_clip_vit_hooks


IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Collect CLIP ViT trace for circuit analysis.')
    parser.add_argument('--model', default='ViT-B-16')
    parser.add_argument('--pretrained', default='openai')
    parser.add_argument('--image-dir', type=Path, default=Path('examples'))
    parser.add_argument('--target-layer', type=int, required=True)
    parser.add_argument('--next-attn-layer', type=int, required=True)
    parser.add_argument('--max-images', type=int, default=32)
    parser.add_argument('--out', type=Path, required=True)
    return parser.parse_args()


def list_images(image_dir: Path, max_images: int) -> list[Path]:
    if not image_dir.exists():
        return []
    files = [p for p in sorted(image_dir.rglob('*')) if p.suffix.lower() in IMAGE_EXTS]
    return files[:max_images]


def main() -> None:
    args = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model, _, preprocess = open_clip.create_model_and_transforms(args.model, pretrained=args.pretrained)
    model = model.to(device).eval()

    image_paths = list_images(args.image_dir, args.max_images)
    if not image_paths:
        raise SystemExit(f'No images found in {args.image_dir}.')

    store = register_clip_vit_hooks(model, args.target_layer, args.next_attn_layer)

    images: list[str] = []
    h_l: list[torch.Tensor] = []
    ffn_gate_l: list[torch.Tensor] = []
    attn_logits_l1: list[torch.Tensor] = []

    with torch.no_grad():
        for path in image_paths:
            image = Image.open(path).convert('RGB')
            pixel_values = preprocess(image).unsqueeze(0).to(device)
            store.clear()
            _ = model.encode_image(pixel_values)

            if store.h_l is None or store.ffn_gate_l is None or store.attn_logits_l1 is None:
                continue

            images.append(str(path))
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
