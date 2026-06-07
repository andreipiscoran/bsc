import streamlit as st

from utils.ui import render_professional_sidebar


st.set_page_config(page_title="Conclusions", layout="wide")
render_professional_sidebar("Conclusions")
st.title("Conclusions")

st.markdown(
    """
### Key findings
- Statistical methods provide interpretable forensic cues for synthetic-image analysis.
- Noise residual and frequency-domain features are the most reliable indicators in this study.
- Patch-level anomaly mapping helps localize suspicious regions for human review.
"""
)

st.markdown(
    """
### Research implications
- Statistical descriptors are computationally efficient and transparent.
- They are valuable as standalone signals and as support evidence for deep detectors.
"""
)

st.markdown(
    """
### Future work
- Integrate statistical descriptors with CNN or Vision Transformer embeddings.
- Evaluate robustness under compression, resizing, and post-processing pipelines.
- Extend analysis to video deepfakes with temporal consistency checks.
"""
)

st.info("A hybrid pipeline (interpretable statistics + deep learning) is a promising direction for practical deployment.")
