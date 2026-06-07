import cv2
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from utils.feature_extraction import (
    compute_color_statistics,
    compute_fft_features,
    compute_gabor_response,
    compute_glcm_features,
    compute_intensity_statistics,
    compute_noise_residual,
    compute_texture_features,
)
from utils.image_processing import fit_image_to_box, load_image_from_upload
from utils.ui import render_professional_sidebar
from utils.visualization import plot_color_statistics, plot_correlation_heatmap, plot_intensity_histogram


def make_demo_image(height: int = 320, width: int = 480) -> np.ndarray:
    base = np.zeros((height, width, 3), dtype=np.uint8)
    xx, yy = np.meshgrid(np.linspace(0, 255, width), np.linspace(0, 255, height))
    base[..., 0] = xx.astype(np.uint8)
    base[..., 1] = yy.astype(np.uint8)
    base[..., 2] = ((0.7 * xx + 0.3 * yy) % 255).astype(np.uint8)
    noise = np.random.normal(0, 12, size=base.shape).astype(np.int16)
    return np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)


st.set_page_config(page_title="Feature Exploration", layout="wide")
render_professional_sidebar("Feature Exploration")
st.title("Feature Exploration")

DISPLAY_WIDTH = 640
DISPLAY_HEIGHT = 360

uploaded_file = st.file_uploader("Upload image for feature exploration (optional)", type=["png", "jpg", "jpeg"])
image = load_image_from_upload(uploaded_file) if uploaded_file else make_demo_image()

st.image(
    fit_image_to_box(image, max_width=DISPLAY_WIDTH, max_height=DISPLAY_HEIGHT),
    caption="Image used for feature exploration",
    width=DISPLAY_WIDTH,
)

tabs = st.tabs(["Texture", "Color", "Frequency", "Noise", "Intensity"])

with tabs[0]:
    st.subheader("Texture Features")
    st.markdown("**Methods:** Local Binary Patterns (LBP), GLCM, and Gabor filtering")

    texture = compute_texture_features(image)
    glcm = compute_glcm_features(image)
    gabor = compute_gabor_response(image)

    c1, c2 = st.columns(2)
    c1.image(
        fit_image_to_box(texture["lbp"], max_width=DISPLAY_WIDTH, max_height=DISPLAY_HEIGHT),
        caption="LBP map",
        clamp=True,
        width=DISPLAY_WIDTH,
    )
    c2.image(
        fit_image_to_box(texture["gradient_magnitude"], max_width=DISPLAY_WIDTH, max_height=DISPLAY_HEIGHT),
        caption="Gradient magnitude",
        clamp=True,
        width=DISPLAY_WIDTH,
    )

    st.image(
        fit_image_to_box(gabor, max_width=DISPLAY_WIDTH, max_height=DISPLAY_HEIGHT),
        caption="Gabor response",
        clamp=True,
        width=DISPLAY_WIDTH,
    )
    st.json(glcm)

with tabs[1]:
    st.subheader("Color Features")
    color_stats = compute_color_statistics(image)
    st.plotly_chart(plot_color_statistics(color_stats["means"], color_stats["stds"]), use_container_width=True)
    st.plotly_chart(plot_correlation_heatmap(color_stats["correlation"]), use_container_width=True)

with tabs[2]:
    st.subheader("Frequency Features")
    st.markdown("**Methods:** FFT spectrum and spectral profile")
    fft_data = compute_fft_features(image)

    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    axs[0].imshow(np.log1p(fft_data["magnitude"]), cmap="magma")
    axs[0].set_title("FFT magnitude")
    axs[0].axis("off")

    axs[1].plot(fft_data["radial_profile"][:200])
    axs[1].set_title("Radial spectral profile")
    axs[1].set_xlabel("Radius")
    axs[1].set_ylabel("Energy")
    st.pyplot(fig)

with tabs[3]:
    st.subheader("Noise Features")
    noise_data = compute_noise_residual(image)
    st.image(
        fit_image_to_box(noise_data["residual_map"], max_width=DISPLAY_WIDTH, max_height=DISPLAY_HEIGHT),
        caption="Noise residual map",
        clamp=True,
        width=DISPLAY_WIDTH,
    )
    st.write(
        {
            "residual_std": round(noise_data["std"], 4),
            "residual_var": round(noise_data["var"], 4),
        }
    )

with tabs[4]:
    st.subheader("Intensity Statistics")
    intensity = compute_intensity_statistics(image)

    c1, c2 = st.columns(2)
    with c1:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        st.image(
            fit_image_to_box(gray, max_width=DISPLAY_WIDTH, max_height=DISPLAY_HEIGHT),
            caption="Grayscale",
            clamp=True,
            width=DISPLAY_WIDTH,
        )
    with c2:
        st.metric("Mean", f"{intensity['mean']:.2f}")
        st.metric("Variance", f"{intensity['variance']:.2f}")
        st.metric("Skewness", f"{intensity['skewness']:.3f}")
        st.metric("Kurtosis", f"{intensity['kurtosis']:.3f}")
        st.metric("Entropy", f"{intensity['entropy']:.3f}")

    st.plotly_chart(plot_intensity_histogram(intensity["histogram"], intensity["bins"]), use_container_width=True)
