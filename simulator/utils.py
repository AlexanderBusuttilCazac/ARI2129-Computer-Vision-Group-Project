"""Image processing utilities for gamma correction and tone mapping."""

from __future__ import annotations

import cv2
import numpy as np


def to_float01(image_bgr: np.ndarray) -> np.ndarray:
    """Convert uint8 BGR image to float32 in [0, 1]."""
    return image_bgr.astype(np.float32) / 255.0


def to_uint8(image01: np.ndarray) -> np.ndarray:
    """Convert float image in [0, 1] to uint8."""
    clipped = np.clip(image01, 0.0, 1.0)
    return (clipped * 255.0 + 0.5).astype(np.uint8)


def apply_power_law(image01: np.ndarray, gamma: float, c: float = 1.0) -> np.ndarray:
    """Apply power law transform s = c * r^gamma."""
    image01 = np.clip(image01, 0.0, 1.0)
    return np.clip(c * np.power(image01, gamma), 0.0, 1.0)


def srgb_encode(linear: np.ndarray) -> np.ndarray:
    """Encode linear RGB to sRGB transfer function."""
    linear = np.clip(linear, 0.0, 1.0)
    threshold = 0.0031308
    low = 12.92 * linear
    high = 1.055 * np.power(linear, 1.0 / 2.4) - 0.055
    return np.where(linear <= threshold, low, high)


def srgb_decode(encoded: np.ndarray) -> np.ndarray:
    """Decode sRGB encoded values to linear RGB."""
    encoded = np.clip(encoded, 0.0, 1.0)
    threshold = 0.04045
    low = encoded / 12.92
    high = np.power((encoded + 0.055) / 1.055, 2.4)
    return np.where(encoded <= threshold, low, high)


def log_like_encode(linear: np.ndarray, gain: float = 9.0) -> np.ndarray:
    """Simple log-like encoding approximation for educational use."""
    linear = np.clip(linear, 0.0, 1.0)
    return np.log1p(gain * linear) / np.log1p(gain)


def rec2020_like_oetf(linear: np.ndarray) -> np.ndarray:
    """Simplified Rec.2020-style OETF approximation."""
    linear = np.clip(linear, 0.0, 1.0)
    alpha = 1.0993
    beta = 0.0181
    low = 4.5 * linear
    high = alpha * np.power(linear, 0.45) - (alpha - 1.0)
    return np.where(linear < beta, low, high)


def reinhard_tone_map(image01: np.ndarray, exposure: float = 1.0, white_point: float = 4.0) -> np.ndarray:
    """Global Reinhard tone mapping on luminance preserving chroma."""
    img = np.clip(image01 * exposure, 0.0, None)
    b, g, r = cv2.split(img)
    luminance = 0.0722 * b + 0.7152 * g + 0.2126 * r
    l_white2 = max(white_point * white_point, 1e-6)
    mapped_l = (luminance * (1.0 + luminance / l_white2)) / (1.0 + luminance)
    scale = mapped_l / np.maximum(luminance, 1e-6)
    mapped = cv2.merge((b * scale, g * scale, r * scale))
    return np.clip(mapped, 0.0, 1.0)


def sigmoid_tone_curve(image01: np.ndarray, strength: float = 8.0) -> np.ndarray:
    """Apply sigmoid S-curve for contrast shaping."""
    x = np.clip(image01, 0.0, 1.0)
    y = 1.0 / (1.0 + np.exp(-strength * (x - 0.5)))
    y0 = 1.0 / (1.0 + np.exp(strength * 0.5))
    y1 = 1.0 / (1.0 + np.exp(-strength * 0.5))
    return np.clip((y - y0) / (y1 - y0), 0.0, 1.0)
