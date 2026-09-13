"""Train + evaluate a reproducible synthetic experiment and write JSON results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from data.dataset import Config, MockWorldData
from evaluate import evaluate
from train import train_loop


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", type=Path, default=Path("artifacts/results.json"))
    args = p.parse_args()
    cfg = Config()._replace(epochs=args.epochs or Config().epochs, seed=args.seed)
    world = MockWorldData(cfg)
    params, history = train_loop(world, cfg)
    metrics = evaluate(params, world, cfg)
    payload = {"config": cfg._asdict(), "training": history, "evaluation": metrics}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
    print(json.dumps(metrics, indent=2))
