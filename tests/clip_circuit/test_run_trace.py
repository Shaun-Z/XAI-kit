import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_trace


def test_parse_args_supports_imagenet_root_and_default_val_split(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_trace.py",
            "--imagenet-root",
            "/data/imagenet",
            "--target-layer",
            "6",
            "--next-attn-layer",
            "7",
            "--out",
            "artifacts/tmp_trace.pt",
        ],
    )

    args = run_trace.parse_args()

    assert args.imagenet_root == Path("/data/imagenet")
    assert args.split == "val"
