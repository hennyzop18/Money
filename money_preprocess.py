from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np
from PIL import Image


@dataclass
class CropResult:
    image: Image.Image
    bbox: Tuple[int, int, int, int]
    used_crop: bool


def _clamp_bbox(x1: int, y1: int, x2: int, y2: int, w: int, h: int) -> Tuple[int, int, int, int]:
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(1, min(x2, w))
    y2 = max(1, min(y2, h))
    if x2 <= x1 + 1:
        x2 = min(w, x1 + 2)
    if y2 <= y1 + 1:
        y2 = min(h, y1 + 2)
    return x1, y1, x2, y2


def crop_largest_object(pil_img: Image.Image, min_area_ratio: float = 0.08) -> CropResult:
    rgb = np.array(pil_img.convert("RGB"))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(gray, 50, 150)
    kernel = np.ones((5, 5), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=2)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    h, w = gray.shape[:2]
    img_area = float(h * w)

    best_bbox: Optional[Tuple[int, int, int, int]] = None
    best_area = 0.0

    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        area = float(cw * ch)
        if area > best_area:
            best_area = area
            best_bbox = (x, y, x + cw, y + ch)

    if best_bbox is None:
        return CropResult(image=pil_img, bbox=(0, 0, w, h), used_crop=False)

    if best_area < img_area * float(min_area_ratio):
        return CropResult(image=pil_img, bbox=(0, 0, w, h), used_crop=False)

    x1, y1, x2, y2 = best_bbox

    pad = int(0.04 * max(w, h))
    x1 -= pad
    y1 -= pad
    x2 += pad
    y2 += pad
    x1, y1, x2, y2 = _clamp_bbox(x1, y1, x2, y2, w, h)

    cropped = pil_img.crop((x1, y1, x2, y2))
    return CropResult(image=cropped, bbox=(x1, y1, x2, y2), used_crop=True)
