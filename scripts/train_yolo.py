from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/public_dataset/yolo_seed/seed_germination.yaml")
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--imgsz", type=int, default=416)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--name", default="seed_yolov8n")
    parser.add_argument("--project", default="runs")
    args = parser.parse_args()

    model = YOLO(args.model)
    results = model.train(
        data=str(Path(args.data)),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=0,
        project=args.project,
        name=args.name,
        exist_ok=True,
        patience=3,
        verbose=True,
    )
    print(results)

    best = Path(getattr(model.trainer, "best", ""))
    if best.exists():
        Path("models").mkdir(exist_ok=True)
        target = Path("models") / "seed_detector.pt"
        target.write_bytes(best.read_bytes())
        print(f"Copied best model to {target}")


if __name__ == "__main__":
    main()
