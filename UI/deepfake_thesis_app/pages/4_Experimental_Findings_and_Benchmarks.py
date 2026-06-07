import pandas as pd
import plotly.express as px
import streamlit as st

from utils.ui import render_professional_sidebar


st.set_page_config(page_title="Experiments and Results", layout="wide")
render_professional_sidebar("Experiments & Results")
st.title("Experiments and Results")

st.subheader("Dataset description")
st.markdown(
    """
- **Dataset:** CIFAKE (real vs synthetic images)
- **Task:** Binary discrimination based on handcrafted statistical descriptors
- **Goal:** Evaluate which feature families are most informative and interpretable
"""
)

with st.expander("Protocol"):
    st.markdown(
        """
1. Extract feature vectors for each image (noise, FFT, texture, color, intensity).
2. Train simple linear classifiers (e.g., LDA) on each family separately.
3. Compare validation performance and interpretability.
"""
    )

results = pd.DataFrame(
    {
        "Feature": ["Noise", "FFT", "Texture", "Color", "Intensity"],
        "LDA_Accuracy": [0.88, 0.83, 0.76, 0.71, 0.69],
        "Interpretability": ["High", "High", "Moderate", "Moderate", "High"],
    }
)

st.subheader("Comparative results")
st.dataframe(results, use_container_width=True)

fig = px.bar(
    results,
    x="Feature",
    y="LDA_Accuracy",
    color="Feature",
    title="Feature Family vs LDA Accuracy",
    text="LDA_Accuracy",
)
fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
fig.update_layout(yaxis_range=[0, 1], showlegend=False)
st.plotly_chart(fig, use_container_width=True)

st.success("Observation: noise-based features provide the strongest standalone signal for synthetic image detection.")

st.markdown("### Discussion")
st.write(
    "Frequency features also show robust discrimination, while texture features are informative "
    "but more sensitive to content variability."
)
