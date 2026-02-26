#!/usr/bin/env python3
"""
Probe CLIP text attention matrices for multiple text examples.

What this script does:
1) Feeds multiple text inputs into CLIP text encoder.
2) Probes all attention heads in layer 0, layer 1, and the last layer.
3) Saves visualizations into per-layer directories.
4) Saves the original text alongside attention visualizations in each layer dir.

Usage:
    python3 examples/probe_clip_text_attention_layers_example.py
    python3 examples/probe_clip_text_attention_layers_example.py \
      --text-file /path/to/texts.txt --num-texts 20 \
      --output-dir clip_text_attention_outputs
"""

from __future__ import annotations

import argparse
from pathlib import Path
import random
import sys
from typing import List, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch

# Allow running from repository root without editable install.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from captum._utils.transformer.accessor import ActivationAccessor
from captum._utils.transformer.visualization import visualize_attention_heads


DEFAULT_TEXTS: List[str] = [
    "a photo of a golden retriever running on grass",
    "a red sports car parked on a city street",
    "a bowl of fresh fruit on a wooden table",
    "a snowy mountain under a clear blue sky",
    "a person riding a bicycle near the beach",
    "a close-up of a sunflower in daylight",
    "a cat sleeping on a gray sofa",
    "a plate of sushi with chopsticks",
    "a busy night market with neon lights",
    "an old castle on top of a hill",
    "a black-and-white portrait of an elderly man",
    "a child flying a kite in an open field",
    "a laptop and notebook on an office desk",
    "a wooden cabin by a frozen lake",
    "a crowded subway station during rush hour",
    "a slice of chocolate cake with strawberries",
    "a surfer riding a wave at sunset",
    "a vintage motorcycle parked beside a cafe",
    "a stack of books near a window",
    "a bird perched on a tree branch",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Probe CLIP text attention matrices (all heads) for layer 0, "
            "layer 1, and last layer over multiple texts."
        )
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="openai/clip-vit-base-patch32",
        help="HuggingFace model name for CLIP text encoder.",
    )
    parser.add_argument(
        "--text-file",
        type=str,
        default=None,
        help=(
            "Optional path to a text file with one prompt per line. "
            "If omitted, built-in examples are used."
        ),
    )
    parser.add_argument(
        "--num-texts",
        type=int,
        default=10,
        help="Number of texts to process.",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle texts before selecting the first --num-texts.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used when --shuffle is enabled.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="clip_text_attention_outputs",
        help="Output directory.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="DPI for saved attention figures.",
    )
    return parser.parse_args()


def _load_texts_from_file(path: Path) -> List[str]:
    if not path.is_file():
        raise ValueError(f"Text file not found: {path}")
    texts = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    texts = [line for line in texts if line]
    if not texts:
        raise ValueError(f"No non-empty lines found in: {path}")
    return texts


def _prepare_texts(
    text_file: str | None,
    num_texts: int,
    shuffle: bool,
    seed: int,
) -> List[str]:
    if num_texts <= 0:
        raise ValueError("--num-texts must be > 0")

    if text_file is not None:
        texts = _load_texts_from_file(Path(text_file))
    else:
        texts = list(DEFAULT_TEXTS)

    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(texts)

    if len(texts) < num_texts:
        raise ValueError(
            f"Need at least {num_texts} texts, but only {len(texts)} are available."
        )
    return texts[:num_texts]


def _layer_output_dir(base_out: Path, layer_idx: int, is_last: bool) -> Path:
    dirname = f"layer_{layer_idx}_last" if is_last else f"layer_{layer_idx}"
    out = base_out / dirname
    out.mkdir(parents=True, exist_ok=True)
    return out


def main() -> None:
    args = parse_args()

    try:
        from transformers import AutoTokenizer, CLIPTextModel
    except ImportError as exc:
        raise ImportError(
            "transformers is required. Install with: pip install transformers"
        ) from exc

    texts = _prepare_texts(
        text_file=args.text_file,
        num_texts=args.num_texts,
        shuffle=args.shuffle,
        seed=args.seed,
    )
    print(f"Prepared {len(texts)} text examples.")

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Loading model: {args.model_name}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = CLIPTextModel.from_pretrained(
        args.model_name,
        attn_implementation="eager",
    ).to(device)
    model.eval()

    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads
    target_layers = [0, 1, num_layers - 1]
    print(f"Text encoder: layers={num_layers}, heads={num_heads}")
    print(f"Target layers: {target_layers}")

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

    accessor = ActivationAccessor(model, "clip_text")

    for idx, text in enumerate(texts):
        text_name = f"text_{idx:02d}"
        print(f"[{idx + 1}/{len(texts)}] Processing: {text}")

        encoded = tokenizer(text, return_tensors="pt", truncation=True)
        input_ids = encoded["input_ids"].to(device)
        tokens: Sequence[str] = tokenizer.convert_ids_to_tokens(input_ids[0])

        for layer_idx in target_layers:
            out_dir = layer_dirs[layer_idx]

            # Save the original text next to the visualization.
            text_path = out_dir / f"{text_name}_input.txt"
            text_path.write_text(text + "\n", encoding="utf-8")

            attn = accessor.get_attention_weights(
                f"L{layer_idx}",
                input_ids=input_ids,
                output_attentions=True,
            )
            fig = visualize_attention_heads(
                attn,
                tokens=tokens,
                title=f"CLIP Text Attention | Layer {layer_idx} | {text_name}",
                use_pyplot=False,
            )
            attn_path = out_dir / f"{text_name}_all_heads.png"
            fig.savefig(attn_path, dpi=args.dpi, bbox_inches="tight")
            plt.close(fig)

    print("Done.")
    print(f"Outputs saved to: {output_root.resolve()}")
    for layer_idx in target_layers:
        print(f"  Layer {layer_idx}: {layer_dirs[layer_idx].resolve()}")


if __name__ == "__main__":
    main()
