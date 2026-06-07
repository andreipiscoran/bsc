from __future__ import annotations

import cv2
import numpy as np


def load_image_from_upload(uploaded_file) -> np.ndarray:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Failed to decode uploaded image.")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def ensure_rgb(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    return image


def to_gray(image: np.ndarray) -> np.ndarray:
    image = ensure_rgb(image)
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def normalize_map(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float32)
    min_val = float(values.min())
    max_val = float(values.max())
    if max_val - min_val < 1e-8:
        return np.zeros_like(values, dtype=np.float32)
    return (values - min_val) / (max_val - min_val)


def to_uint8(values: np.ndarray) -> np.ndarray:
    return (255.0 * normalize_map(values)).astype(np.uint8)


def fit_image_to_box(image: np.ndarray, max_width: int = 640, max_height: int = 420) -> np.ndarray:
    height, width = image.shape[:2]
    if height <= 0 or width <= 0:
        return image

    scale_w = max_width / width
    scale_h = max_height / height
    scale = min(scale_w, scale_h, 1.0)

    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))

    if new_width == width and new_height == height:
        return image

    return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
