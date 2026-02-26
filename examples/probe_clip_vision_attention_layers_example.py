#!/usr/bin/env python3
"""
Probe CLIP vision attention matrices for multiple images.

What this script does:
1) Feeds 20 different images into CLIP vision encoder (default).
2) Probes all attention heads in layer 0, layer 1, and the last layer.
3) Saves visualizations into per-layer directories.
4) Saves original images alongside attention visualizations in each layer dir.

Usage:
    python examples/probe_clip_vision_attention_layers_example.py
    python examples/probe_clip_vision_attention_layers_example.py \
        --image-dir /path/to/images --num-images 20 --output-dir outputs

python3 examples/probe_clip_vision_attention_layers_example.py \
  --imagenet-root /data/imagenet \
  --split val \
  --num-images 20 \
  --output-dir clip_vision_attention_outputs
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch

# Allow running from repository root without editable install.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from captum._utils.transformer.accessor import ActivationAccessor
from captum._utils.transformer.visualization import visualize_attention_heads


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Probe CLIP vision attention matrices (all heads) for layer 0, "
            "layer 1, and last layer over multiple images."
        )
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="openai/clip-vit-base-patch32",
        help="HuggingFace model name for CLIP vision encoder.",
    )
    parser.add_argument(
        "--imagenet-root",
        type=str,
        default="/data/imagenet",
        help=(
            "ImageNet root directory. The script will try "
            "<imagenet-root>/<split> first."
        ),
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["train", "val"],
        help="ImageNet split to sample from.",
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=20,
        help="Number of images to process.",
    )
    parser.add_argument(
        "--image-dir",
        type=str,
        default=None,
        help=(
            "Optional directory with input images. If omitted, synthetic "
            "images are generated."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="clip_vision_attention_outputs",
        help="Output directory.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for synthetic image generation.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="DPI for saved attention figures.",
    )
    return parser.parse_args()


def _load_images_from_dir(
    image_dir: Path,
    num_images: int,
    image_size: int,
) -> Tuple[List[Image.Image], List[str]]:
    patterns = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp", "*.JPEG")
    files: List[Path] = []
    for pattern in patterns:
        files.extend(sorted(image_dir.rglob(pattern)))

    if len(files) < num_images:
        raise ValueError(
            f"Need at least {num_images} images in {image_dir}, "
            f"but found {len(files)}."
        )

    images: List[Image.Image] = []
    names: List[str] = []
    for path in files[:num_images]:
        img = Image.open(path).convert("RGB").resize((image_size, image_size))
        images.append(img)
        names.append(path.stem)
    return images, names


def _load_imagenet_split_images(
    imagenet_root: Path,
    split: str,
    num_images: int,
    image_size: int,
) -> Tuple[List[Image.Image], List[str]]:
    split_dir = imagenet_root / split
    if not split_dir.is_dir():
        raise ValueError(f"ImageNet split directory not found: {split_dir}")

    class_dirs = sorted(
        d for d in split_dir.iterdir() if d.is_dir()
    )
    if not class_dirs:
        raise ValueError(
            f"No class subdirectories found under {split_dir}. "
            "Expected ImageNet layout: <split>/<wnid>/*.JPEG"
        )

    per_class_files: List[Tuple[str, List[Path]]] = []
    patterns = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp", "*.JPEG")
    for class_dir in class_dirs:
        files: List[Path] = []
        for pattern in patterns:
            files.extend(sorted(class_dir.glob(pattern)))
        if files:
            per_class_files.append((class_dir.name, files))

    if not per_class_files:
        raise ValueError(f"No image files found under {split_dir}.")

    # Round-robin across classes to maximize category diversity.
    chosen: List[Tuple[str, Path]] = []
    cursor = 0
    while len(chosen) < num_images:
        class_name, files = per_class_files[cursor % len(per_class_files)]
        pick_idx = len(chosen) // len(per_class_files)
        if pick_idx < len(files):
            chosen.append((class_name, files[pick_idx]))
        cursor += 1

        if cursor > num_images * max(2, len(per_class_files)) and len(chosen) < num_images:
            break

    if len(chosen) < num_images:
        raise ValueError(
            f"Could only sample {len(chosen)} images from {split_dir}, "
            f"but requested {num_images}."
        )

    images: List[Image.Image] = []
    names: List[str] = []
    for idx, (class_name, path) in enumerate(chosen[:num_images]):
        img = Image.open(path).convert("RGB").resize((image_size, image_size))
        images.append(img)
        names.append(f"{class_name}_{idx:03d}_{path.stem}")
    return images, names


def _make_synthetic_images(
    num_images: int,
    image_size: int,
    seed: int,
) -> Tuple[List[Image.Image], List[str]]:
    rng = np.random.default_rng(seed)
    images: List[Image.Image] = []
    names: List[str] = []

    for idx in range(num_images):
        base = rng.integers(0, 256, size=(image_size, image_size, 3), dtype=np.uint8)

        # Add deterministic stripe patterns so images are visibly distinct.
        stripe_step = max(4, (idx % 10) + 4)
        base[:, ::stripe_step, 0] = (base[:, ::stripe_step, 0] // 2 + 100).astype(
            np.uint8
        )
        base[::stripe_step, :, 1] = (base[::stripe_step, :, 1] // 2 + 80).astype(
            np.uint8
        )

        images.append(Image.fromarray(base, mode="RGB"))
        names.append(f"synthetic_{idx:02d}")

    return images, names


def _prepare_images(
    imagenet_root: str | None,
    split: str,
    image_dir: str | None,
    num_images: int,
    image_size: int,
    seed: int,
) -> Tuple[List[Image.Image], List[str]]:
    if imagenet_root is not None:
        root = Path(imagenet_root)
        split_dir = root / split
        if split_dir.is_dir():
            return _load_imagenet_split_images(
                root, split, num_images, image_size
            )

    if image_dir is not None:
        return _load_images_from_dir(Path(image_dir), num_images, image_size)

    return _make_synthetic_images(num_images, image_size, seed)


def _layer_output_dir(base_out: Path, layer_idx: int, is_last: bool) -> Path:
    if is_last:
        dirname = f"layer_{layer_idx}_last"
    else:
        dirname = f"layer_{layer_idx}"
    out = base_out / dirname
    out.mkdir(parents=True, exist_ok=True)
    return out


def main() -> None:
    args = parse_args()

    if args.num_images <= 0:
        raise ValueError("--num-images must be > 0")

    try:
        from transformers import CLIPImageProcessor, CLIPVisionModel
    except ImportError as exc:
        raise ImportError(
            "transformers is required. Install with: pip install transformers"
        ) from exc

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"Loading model: {args.model_name}")
    processor = CLIPImageProcessor.from_pretrained(args.model_name)
    model = CLIPVisionModel.from_pretrained(
        args.model_name,
        attn_implementation="eager",
    ).to(device)
    model.eval()

    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads
    image_size = model.config.image_size
    patch_size = model.config.patch_size
    num_patches = (image_size // patch_size) ** 2
    tokens: Sequence[str] = ["[CLS]"] + [f"p{i}" for i in range(num_patches)]

    target_layers = [0, 1, num_layers - 1]
    print(
        f"Vision encoder: layers={num_layers}, heads={num_heads}, "
        f"image_size={image_size}, patch_size={patch_size}"
    )
    print(f"Target layers: {target_layers}")

    images, names = _prepare_images(
        imagenet_root=args.imagenet_root,
        split=args.split,
        image_dir=args.image_dir,
        num_images=args.num_images,
        image_size=image_size,
        seed=args.seed,
    )
    print(f"Prepared {len(images)} images.")

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    layer_dirs = {
        layer_idx: _layer_output_dir(
            output_root,
            layer_idx=layer_idx,
            is_last=(layer_idx == num_layers - 1),
        )
        for layer_idx in target_layers
    }

    accessor = ActivationAccessor(model, "clip_vision")

    for idx, (image, image_name) in enumerate(zip(images, names)):
        print(f"[{idx + 1}/{len(images)}] Processing {image_name}")

        encoded = processor(images=image, return_tensors="pt")
        pixel_values = encoded["pixel_values"].to(device)

        for layer_idx in target_layers:
            out_dir = layer_dirs[layer_idx]

            original_path = out_dir / f"img_{idx:02d}_{image_name}_input.png"
            image.save(original_path)

            attn = accessor.get_attention_weights(
                f"L{layer_idx}",
                pixel_values=pixel_values,
                output_attentions=True,
            )
            fig = visualize_attention_heads(
                attn,
                tokens=tokens,
                title=(
                    f"CLIP Vision Attention | Layer {layer_idx} | "
                    f"Image {idx:02d} ({image_name})"
                ),
                use_pyplot=False,
            )
            attn_path = out_dir / f"img_{idx:02d}_{image_name}_all_heads.png"
            fig.savefig(attn_path, dpi=args.dpi, bbox_inches="tight")
            plt.close(fig)

    print("Done.")
    print(f"Outputs saved to: {output_root.resolve()}")
    for layer_idx in target_layers:
        print(f"  Layer {layer_idx}: {layer_dirs[layer_idx].resolve()}")


if __name__ == "__main__":
    main()
