"""YOLO-backed seed detector."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from ultralytics import YOLO

from src.detector import Detection, ImageAnalysis


CLASS_NAMES = {0: "non_germinated", 1: "germinated"}


class YOLOSeedDetector:
    mode_name = "YOLO trained detector"

    def __init__(
        self,
        weights: str | Path,
        conf: float = 0.25,
        iou: float = 0.5,
        device: str | int | None = None,
    ) -> None:
        self.weights = Path(weights)
        self.conf = conf
        self.iou = iou
        self.device = device
        self.model = YOLO(str(self.weights))

    def analyze(self, image: Image.Image) -> ImageAnalysis:
        arr = np.array(image.convert("RGB"))
        kwargs = {"source": arr, "conf": self.conf, "iou": self.iou, "verbose": False}
        if self.device is not None:
            kwargs["device"] = self.device

        result = self.model.predict(**kwargs)[0]
        detections: list[Detection] = []
        if result.boxes is None:
            return ImageAnalysis(detections=detections)

        boxes = result.boxes.xyxy.detach().cpu().numpy()
        classes = result.boxes.cls.detach().cpu().numpy().astype(int)
        confs = result.boxes.conf.detach().cpu().numpy()
        for (x1, y1, x2, y2), cls, conf in zip(boxes, classes, confs):
            x = int(round(float(x1)))
            y = int(round(float(y1)))
            w = int(round(float(x2 - x1)))
            h = int(round(float(y2 - y1)))
            detections.append(
                Detection(
                    x=x,
                    y=y,
                    w=w,
                    h=h,
                    area=float(max(0, w) * max(0, h)),
                    aspect_ratio=float(max(w / max(1, h), h / max(1, w))),
                    label=CLASS_NAMES.get(int(cls), str(int(cls))),
                    confidence=float(conf),
                )
            )

        detections.sort(key=lambda item: (item.y, item.x))
        return ImageAnalysis(detections=detections)
