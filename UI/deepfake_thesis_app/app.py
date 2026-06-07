import streamlit as st

from utils.ui import render_professional_sidebar


st.set_page_config(
    page_title="Statistical Deepfake Detection Thesis",
    layout="wide",
)

render_professional_sidebar("Home")

st.title("Statistical Deepfake Detection — Bachelor Thesis Platform")
st.caption("Interactive research dashboard for interpretable, statistical deepfake detection")

overview_col1, overview_col2, overview_col3 = st.columns(3)
overview_col1.metric("Core feature families", "5", "Texture, Color, Frequency, Noise, Intensity")
overview_col2.metric("Main analysis mode", "Patch Anomaly Mapping", "Local suspicious region localization")
overview_col3.metric("Primary thesis finding", "Noise features strongest", "Most robust standalone indicator")

st.markdown("### Platform sections")
st.markdown(
    """
- **Introduction:** problem context, motivation, and generation architectures
- **Deepfake Detection Demo:** upload image, inspect anomaly overlays, and review evidence maps
- **Feature Exploration:** interpret descriptor families with visual and statistical outputs
- **Experiments and Results:** compare feature-family discrimination performance
- **Conclusions:** summarize contributions, limitations, and future directions
"""
)

with st.expander("Thesis focus"):
    st.write(
        "The project emphasizes interpretable statistical signals (noise, frequency, texture, "
        "color, and intensity features) to identify potential synthetic imagery."
    )

st.info(
    "Best experience: upload RGB facial or scene images on the Demo page and compare "
    "suspicious-region heatmaps with frequency/noise maps."
)
