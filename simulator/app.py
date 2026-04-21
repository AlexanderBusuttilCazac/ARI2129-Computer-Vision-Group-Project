"""Streamlit simulator for gamma correction and tone processing."""

from __future__ import annotations

import cv2
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from utils import (
    apply_power_law,
    log_like_encode,
    rec2020_like_oetf,
    reinhard_tone_map,
    sigmoid_tone_curve,
    srgb_decode,
    srgb_encode,
    to_float01,
    to_uint8,
)


st.set_page_config(page_title="Gamma & Tone Processing Simulator", layout="wide")
st.title("Point Processing Simulator: Gamma Correction and Tone Processing")

st.sidebar.header("Controls")
operator = st.sidebar.selectbox(
    "Transformation",
    [
        "Power Law",
        "sRGB Encode",
        "sRGB Decode",
        "Log-like Encode",
        "Rec.2020-like OETF",
        "Sigmoid S-curve",
        "Reinhard HDR Tone Map",
    ],
)

gamma = st.sidebar.slider("Gamma", min_value=0.2, max_value=3.0, value=1.0, step=0.05)
exposure = st.sidebar.slider("Exposure", min_value=0.1, max_value=4.0, value=1.0, step=0.1)
white_point = st.sidebar.slider("White point", min_value=1.0, max_value=8.0, value=4.0, step=0.5)
strength = st.sidebar.slider("S-curve strength", min_value=2.0, max_value=20.0, value=8.0, step=0.5)

uploaded = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg", "bmp"])

if uploaded is not None:
    file_bytes = np.frombuffer(uploaded.read(), np.uint8)
    input_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
else:
    # Educational synthetic gradient if no image is uploaded.
    x = np.linspace(0, 255, 512, dtype=np.uint8)
    gradient = np.tile(x, (256, 1))
    input_bgr = cv2.merge((gradient, gradient, gradient))

input01 = to_float01(input_bgr)

if operator == "Power Law":
    output01 = apply_power_law(input01, gamma=gamma)
elif operator == "sRGB Encode":
    output01 = srgb_encode(input01)
elif operator == "sRGB Decode":
    output01 = srgb_decode(input01)
elif operator == "Log-like Encode":
    output01 = log_like_encode(input01)
elif operator == "Rec.2020-like OETF":
    output01 = rec2020_like_oetf(input01)
elif operator == "Sigmoid S-curve":
    output01 = sigmoid_tone_curve(input01, strength=strength)
else:
    output01 = reinhard_tone_map(input01, exposure=exposure, white_point=white_point)

col1, col2 = st.columns(2)
with col1:
    st.subheader("Input")
    st.image(cv2.cvtColor(to_uint8(input01), cv2.COLOR_BGR2RGB), width="stretch")
with col2:
    st.subheader("Output")
    st.image(cv2.cvtColor(to_uint8(output01), cv2.COLOR_BGR2RGB), width="stretch")

st.subheader("Histogram Comparison")
fig, ax = plt.subplots(figsize=(8, 3))
ax.hist(input01.ravel(), bins=64, alpha=0.5, label="Input")
ax.hist(output01.ravel(), bins=64, alpha=0.5, label="Output")
ax.set_xlabel("Intensity (0-1)")
ax.set_ylabel("Frequency")
ax.legend()
st.pyplot(fig)

st.markdown(
    """
**Tip:** Use a gamma less than 1 to brighten dark details, and greater than 1 to darken.
For HDR compression, increase exposure to reveal shadow detail, then adjust white point to control highlight roll-off.
"""
)
