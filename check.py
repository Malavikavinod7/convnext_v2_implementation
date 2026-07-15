from __future__ import annotations

import argparse
from pathlib import Path

import torch

from model import get_model


def main():
    parser = argparse.ArgumentParser(description="Load and inspect a trained checkpoint")
    parser.add_argument("--checkpoint", type=str, default=str(Path(__file__).resolve().parent / "outputs" / "best_model.pth"), help="Path to the checkpoint")
    parser.add_argument("--num-classes", type=int, default=2, help="Number of classes in the trained model")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model = get_model(num_classes=args.num_classes, pretrained=False)
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model.eval()

    print(f"Model loaded successfully from: {checkpoint_path}")
    print(model)


if __name__ == "__main__":
    main()


