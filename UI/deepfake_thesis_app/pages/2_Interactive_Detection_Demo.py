import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from utils.feature_extraction import (
    compute_fft_features,
    compute_noise_residual,
    compute_patch_anomaly_map,
    compute_texture_features,
)
from utils.image_processing import fit_image_to_box, load_image_from_upload
from utils.ui import render_professional_sidebar
from utils.visualization import overlay_heatmap


st.set_page_config(page_title="Deepfake Detection Demo", layout="wide")
render_professional_sidebar("Detection Demo")

st.title("Deepfake Detection Demo")
st.caption("Interactive statistical analysis of potential deepfake artifacts")

DISPLAY_WIDTH = 560
DISPLAY_HEIGHT = 360

with st.sidebar:
    st.subheader("Analysis Controls")
    st.caption("Tune patch-level anomaly aggregation and overlay visibility.")
    patch_size = st.select_slider("Patch size", options=[16, 32, 48, 64, 96], value=32)
    overlay_alpha = st.slider("Heatmap overlay alpha", min_value=0.1, max_value=0.9, value=0.45, step=0.05)
    suspicious_threshold = st.slider("Suspicious score threshold", min_value=0.40, max_value=0.95, value=0.70, step=0.05)

uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg", "bmp"])

if uploaded_file is None:
    st.info("Upload an image to start the deepfake statistical analysis.")
    st.stop()

image = load_image_from_upload(uploaded_file)

anomaly_data = compute_patch_anomaly_map(image, patch_size=patch_size)
fft_data = compute_fft_features(image)
noise_data = compute_noise_residual(image)
texture_data = compute_texture_features(image)

tabs = st.tabs(
    [
        "Suspicious Regions",
        "Frequency Artifacts",
        "Noise Residual",
        "Texture Analysis",
        "Interpretation",
    ]
)

with tabs[0]:
    st.subheader("Deepfake Suspicious Regions")
    overlay = overlay_heatmap(image, anomaly_data["anomaly_map"], alpha=overlay_alpha)
    original_view = fit_image_to_box(image, max_width=DISPLAY_WIDTH, max_height=DISPLAY_HEIGHT)
    overlay_view = fit_image_to_box(overlay, max_width=DISPLAY_WIDTH, max_height=DISPLAY_HEIGHT)
    anomaly_view = fit_image_to_box(anomaly_data["anomaly_map"], max_width=DISPLAY_WIDTH, max_height=DISPLAY_HEIGHT)

    col1, col2 = st.columns(2)
    with col1:
        st.image(original_view, caption="Original Image", width=DISPLAY_WIDTH)
    with col2:
        st.image(overlay_view, caption="Anomaly Heatmap Overlay", width=DISPLAY_WIDTH)

    st.image(anomaly_view, caption="Anomaly Score Map", clamp=True, width=DISPLAY_WIDTH)
    suspicious_ratio = float(np.mean(anomaly_data["anomaly_map"] > suspicious_threshold))
    st.progress(suspicious_ratio, text=f"Area above suspicious threshold: {100 * suspicious_ratio:.1f}%")
    st.caption(
        "Higher values indicate patch-level statistical inconsistencies in noise, frequency, and texture patterns."
    )

with tabs[1]:
    st.subheader("Frequency Artifact Visualization")

    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    axs[0].imshow(np.log1p(fft_data["magnitude"]), cmap="magma")
    axs[0].set_title("FFT Magnitude Spectrum")
    axs[0].axis("off")

    axs[1].imshow(fft_data["log_spectrum"], cmap="inferno")
    axs[1].set_title("Log Spectrum")
    axs[1].axis("off")
    st.pyplot(fig)

    st.line_chart(fft_data["radial_profile"][: min(200, len(fft_data["radial_profile"]))])
    st.caption("Synthetic content may produce abnormal concentration patterns in frequency bands.")

with tabs[2]:
    st.subheader("Noise Residual Visualization")
    residual_map = noise_data["residual_map"]
    residual_view = fit_image_to_box(residual_map, max_width=DISPLAY_WIDTH, max_height=DISPLAY_HEIGHT)

    col1, col2 = st.columns(2)
    with col1:
        st.image(residual_view, caption="Normalized Residual Map", clamp=True, width=DISPLAY_WIDTH)
    with col2:
        st.metric("Residual Standard Deviation", f"{noise_data['std']:.4f}")
        st.metric("Residual Variance", f"{noise_data['var']:.4f}")

    st.caption("Residual is computed as image - denoised(image), emphasizing high-frequency noise artifacts.")

with tabs[3]:
    st.subheader("Texture Analysis")
    lbp_view = fit_image_to_box(texture_data["lbp"], max_width=DISPLAY_WIDTH, max_height=DISPLAY_HEIGHT)
    grad_view = fit_image_to_box(texture_data["gradient_magnitude"], max_width=DISPLAY_WIDTH, max_height=DISPLAY_HEIGHT)

    col1, col2 = st.columns(2)
    with col1:
        st.image(lbp_view, caption="Local Binary Pattern Map", clamp=True, width=DISPLAY_WIDTH)
    with col2:
        st.image(
            grad_view,
            caption="Gradient Magnitude",
            clamp=True,
            width=DISPLAY_WIDTH,
        )

    st.bar_chart(texture_data["lbp_hist"])
    st.caption("Texture inconsistencies can indicate synthesis or post-processing artifacts.")

with tabs[4]:
    st.subheader("Interpretation Summary")
    mean_anomaly = float(np.mean(anomaly_data["anomaly_map"]))
    high_anomaly_ratio = float(np.mean(anomaly_data["anomaly_map"] > 0.7))
    threshold_ratio = float(np.mean(anomaly_data["anomaly_map"] > suspicious_threshold))

    c1, c2, c3 = st.columns(3)
    c1.metric("Mean anomaly score", f"{mean_anomaly:.3f}")
    c2.metric("High-anomaly area ratio", f"{100 * high_anomaly_ratio:.1f}%")
    c3.metric("Gradient mean", f"{texture_data['gradient_mean']:.3f}")

    summary_df = pd.DataFrame(
        {
            "Metric": [
                "Patch size",
                "Overlay alpha",
                "Suspicious threshold",
                "Mean anomaly score",
                "Area above threshold (%)",
                "Residual std",
                "Residual var",
                "Gradient mean",
                "FFT log-spectrum mean",
                "FFT log-spectrum std",
            ],
            "Value": [
                patch_size,
                overlay_alpha,
                suspicious_threshold,
                round(mean_anomaly, 4),
                round(100 * threshold_ratio, 2),
                round(noise_data["std"], 4),
                round(noise_data["var"], 4),
                round(texture_data["gradient_mean"], 4),
                round(fft_data["mean_log_spectrum"], 4),
                round(fft_data["std_log_spectrum"], 4),
            ],
        }
    )
    st.dataframe(summary_df, use_container_width=True, hide_index=True)
    st.download_button(
        "Download analysis summary (CSV)",
        data=summary_df.to_csv(index=False).encode("utf-8"),
        file_name="deepfake_analysis_summary.csv",
        mime="text/csv",
    )

    st.markdown(
        """
- This demo is an interpretability tool, not a final forensic verdict.
- Strong suspicious regions should be cross-validated with metadata, source context, and model-based detectors.
"""
    )
