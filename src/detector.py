from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import cv2
import numpy as np
import pandas as pd
from PIL import Image


@dataclass
class Detection:
    x: int
    y: int
    w: int
    h: int
    area: float
    aspect_ratio: float
    label: str
    confidence: float


@dataclass
class ImageAnalysis:
    detections: list[Detection]

    @property
    def total(self) -> int:
        return len(self.detections)

    @property
    def germinated(self) -> int:
        return sum(1 for item in self.detections if item.label == "germinated")

    @property
    def non_germinated(self) -> int:
        return self.total - self.germinated

    @property
    def germination_rate(self) -> float:
        if self.total == 0:
            return 0.0
        return self.germinated / self.total * 100.0

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "x": item.x,
                    "y": item.y,
                    "w": item.w,
                    "h": item.h,
                    "area": round(item.area, 1),
                    "aspect_ratio": round(item.aspect_ratio, 2),
                    "label": "已发芽" if item.label == "germinated" else "未发芽",
                    "confidence": round(item.confidence, 2),
                }
                for item in self.detections
            ]
        )


def _pil_to_bgr(image: Image.Image) -> np.ndarray:
    rgb = np.array(image.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def analyze_image(image: Image.Image) -> ImageAnalysis:
    bgr = _pil_to_bgr(image)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Petri-dish backgrounds are usually bright; seeds and sprouts are darker.
    _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    roi = np.zeros_like(mask)
    center = (image.width // 2, image.height // 2)
    radius = int(min(image.width, image.height) * 0.43)
    cv2.circle(roi, center, radius, 255, -1)
    mask = cv2.bitwise_and(mask, roi)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    close_kernel = np.ones((9, 9), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    image_area = image.width * image.height
    min_area = max(80, image_area * 0.00045)
    max_area = image_area * 0.08

    detections: list[Detection] = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < min_area or area > max_area:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        if w < 8 or h < 8:
            continue

        aspect_ratio = max(w / h, h / w)
        hull_area = float(cv2.contourArea(cv2.convexHull(contour))) or area
        solidity = area / hull_area
        extent = area / float(w * h)

        germinated_score = 0.0
        if aspect_ratio > 1.7:
            germinated_score += 0.45
        if solidity < 0.82:
            germinated_score += 0.25
        if extent < 0.55:
            germinated_score += 0.2
        if area > min_area * 1.9:
            germinated_score += 0.1

        label = "germinated" if germinated_score >= 0.45 else "non_germinated"
        confidence = min(0.95, max(0.55, 0.55 + abs(germinated_score - 0.45)))
        detections.append(Detection(x, y, w, h, area, aspect_ratio, label, confidence))

    detections.sort(key=lambda item: (item.y, item.x))
    return ImageAnalysis(detections)


def annotate_detections(image: Image.Image, detections: Iterable[Detection]) -> Image.Image:
    bgr = _pil_to_bgr(image)
    for detection in detections:
        color = (60, 180, 75) if detection.label == "germinated" else (80, 80, 230)
        cv2.rectangle(
            bgr,
            (detection.x, detection.y),
            (detection.x + detection.w, detection.y + detection.h),
            color,
            2,
        )
        label = "G" if detection.label == "germinated" else "N"
        cv2.putText(
            bgr,
            label,
            (detection.x, max(12, detection.y - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)
