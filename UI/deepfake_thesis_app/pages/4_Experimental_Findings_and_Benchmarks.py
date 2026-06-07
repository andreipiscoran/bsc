import pandas as pd
import plotly.express as px
import streamlit as st

from utils.ui import render_professional_sidebar


st.set_page_config(page_title="Experiments and Results", layout="wide")
render_professional_sidebar("Experiments & Results")
st.title("Experiments and Results")

st.caption(
    "All figures below are taken from the reproducible benchmark artifacts in "
    "feature_benchmark/cifake_analysis/ and summarized in RESULTS.md."
)

st.subheader("Datasets and tasks")
st.markdown(
    """
- **CIFAKE** — real photographs vs. fully AI-generated images (120,000 samples).
- **FaceForensics++ (FF++)** — authentic vs. manipulated face video frames (2,000 samples).
- **Task:** binary discrimination from handcrafted statistical descriptors.
- **Goal:** quantify which feature families are most informative in-domain, and
  whether a detector transfers across datasets.
"""
)

# ---------------------------------------------------------------------------
# Experiment 1: CIFAKE linear classification (full scale, 120k samples)
# Source: noise_linear_benchmark_*.md / PERFORMANCE_COMPARISON.md
# ---------------------------------------------------------------------------
st.subheader("1. CIFAKE — linear classifier at full scale")
st.markdown(
    """
Logistic regression (`StandardScaler` + `LogisticRegression`, liblinear) trained
on 120,000 samples with an 84,000 / 36,000 train/test split. Fusing texture with
the noise baseline and tuning the regularization (C = 8, 5-fold CV) gives the
best model. Test ROC-AUC (0.943) matches the cross-validated estimate (0.943),
indicating no overfitting.
"""
)

cifake = pd.DataFrame(
    {
        "Configuration": ["Noise (baseline)", "Noise + Texture (tuned)"],
        "Dims": [48, 63],
        "Accuracy": [0.8554, 0.8714],
        "F1": [0.8539, 0.8702],
        "ROC_AUC": [0.9292, 0.9432],
    }
)
st.dataframe(cifake, use_container_width=True, hide_index=True)

cifake_fig = px.bar(
    cifake,
    x="Configuration",
    y=["Accuracy", "ROC_AUC"],
    barmode="group",
    title="CIFAKE Linear Classifier — Accuracy vs ROC-AUC",
    text_auto=".3f",
)
cifake_fig.update_layout(yaxis_range=[0, 1], legend_title_text="Metric")
st.plotly_chart(cifake_fig, use_container_width=True)

st.success(
    "Noise features alone reach 85.5% accuracy / 0.929 ROC-AUC; adding texture and "
    "tuning lifts this to 87.1% / 0.943 at negligible compute cost."
)

# ---------------------------------------------------------------------------
# Experiment 2: CIFAKE per-family LDA & covariance (200k samples)
# Source: cifake_analysis_report_20260308_114523.md
# ---------------------------------------------------------------------------
st.subheader("2. CIFAKE — per-family LDA & covariance")
st.markdown(
    """
Per-family analysis over 200,000 CIFAKE samples: supervised **LDA accuracy** and
the unsupervised **covariance-difference (Frobenius norm)** between the real and
fake feature distributions. On fully synthetic imagery, noise is the strongest
standalone family, followed by the frequency (FFT) and texture descriptors.
"""
)

cifake_fam = pd.DataFrame(
    {
        "Feature": ["Noise", "FFT", "Texture", "Color", "DCT", "Intensity", "Wavelet"],
        "LDA_Accuracy": [0.8482, 0.7987, 0.7624, 0.7299, 0.6823, 0.6432, 0.6308],
        "Cov_Diff_Frobenius": [
            2.0487e03,
            6.2333e11,
            1.0470e05,
            5.2340e03,
            4.6505e03,
            2.6082e03,
            8.9893e03,
        ],
    }
)
st.dataframe(
    cifake_fam,
    use_container_width=True,
    hide_index=True,
    column_config={
        "LDA_Accuracy": st.column_config.NumberColumn(format="%.4f"),
        "Cov_Diff_Frobenius": st.column_config.NumberColumn(format="%.3e"),
    },
)

cf_lda, cf_cov = st.columns(2)

with cf_lda:
    cifake_lda_fig = px.bar(
        cifake_fam,
        x="Feature",
        y="LDA_Accuracy",
        color="Feature",
        title="CIFAKE — LDA Accuracy by Family",
        text="LDA_Accuracy",
    )
    cifake_lda_fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    cifake_lda_fig.update_layout(yaxis_range=[0, 1], showlegend=False)
    st.plotly_chart(cifake_lda_fig, use_container_width=True)

with cf_cov:
    cifake_cov_fig = px.bar(
        cifake_fam,
        x="Feature",
        y="Cov_Diff_Frobenius",
        color="Feature",
        title="CIFAKE — Covariance Difference (Frobenius, log scale)",
        log_y=True,
    )
    cifake_cov_fig.update_layout(showlegend=False, yaxis_title="Cov diff (Frobenius)")
    st.plotly_chart(cifake_cov_fig, use_container_width=True)

st.caption(
    "As with FF++, covariance-difference magnitudes span many orders across families "
    "(FFT dwarfs the rest), so a log axis is used. LDA accuracy is the scale-invariant "
    "separability measure and the basis for the ranking."
)

# ---------------------------------------------------------------------------
# Experiment 3: FF++ per-family LDA (2000 samples)
# Source: faceforensicsplusplus_analysis_report_20260415_191246.md
# ---------------------------------------------------------------------------
st.subheader("3. FaceForensics++ — per-family LDA & covariance")
st.markdown(
    """
Two complementary views per feature family over 2,000 FF++ frames: supervised
**LDA accuracy** (how separable the classes are to a linear classifier) and the
unsupervised **covariance-difference (Frobenius norm)** between the real and fake
feature distributions (how much the second-order structure shifts). Manipulated
face video is markedly harder than fully synthetic imagery: the best single
family (colour) reaches only 62%.
"""
)

ffpp = pd.DataFrame(
    {
        "Feature": ["Color", "FFT", "Noise", "Wavelet", "Intensity", "DCT", "Texture"],
        "LDA_Accuracy": [0.6217, 0.6083, 0.6067, 0.5850, 0.5642, 0.5583, 0.5525],
        "Cov_Diff_Frobenius": [
            1.6184e03,
            6.1438e14,
            4.3192e05,
            1.4913e03,
            6.5081e02,
            5.9431e06,
            3.2537e03,
        ],
    }
)
st.dataframe(
    ffpp,
    use_container_width=True,
    hide_index=True,
    column_config={
        "LDA_Accuracy": st.column_config.NumberColumn(format="%.4f"),
        "Cov_Diff_Frobenius": st.column_config.NumberColumn(format="%.3e"),
    },
)

col_lda, col_cov = st.columns(2)

with col_lda:
    ffpp_fig = px.bar(
        ffpp,
        x="Feature",
        y="LDA_Accuracy",
        color="Feature",
        title="FF++ — LDA Accuracy by Family",
        text="LDA_Accuracy",
    )
    ffpp_fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    ffpp_fig.update_layout(yaxis_range=[0, 1], showlegend=False)
    st.plotly_chart(ffpp_fig, use_container_width=True)

with col_cov:
    cov_fig = px.bar(
        ffpp,
        x="Feature",
        y="Cov_Diff_Frobenius",
        color="Feature",
        title="FF++ — Covariance Difference (Frobenius, log scale)",
        log_y=True,
    )
    cov_fig.update_layout(showlegend=False, yaxis_title="Cov diff (Frobenius)")
    st.plotly_chart(cov_fig, use_container_width=True)

st.caption(
    "Note: covariance-difference magnitudes are not directly comparable across "
    "families because feature scales differ (e.g. FFT magnitudes are orders larger "
    "than colour statistics), so the chart uses a log axis. LDA accuracy is the "
    "scale-invariant measure of separability."
)

# ---------------------------------------------------------------------------
# Cross-dataset comparison + feature leaderboard
# ---------------------------------------------------------------------------
st.subheader("4. CIFAKE vs FF++ — per-family comparison")
st.markdown(
    """
The same seven feature families, scored by LDA accuracy on each dataset. Every
family is far more discriminative on fully synthetic CIFAKE images than on
manipulated FF++ video frames, and the ordering of families differs between the
two — direct evidence that "what makes an image fake" is dataset-specific.
"""
)

comparison = pd.merge(
    cifake_fam[["Feature", "LDA_Accuracy"]].rename(columns={"LDA_Accuracy": "CIFAKE"}),
    ffpp[["Feature", "LDA_Accuracy"]].rename(columns={"LDA_Accuracy": "FF++"}),
    on="Feature",
)

comp_long = comparison.melt(id_vars="Feature", var_name="Dataset", value_name="LDA_Accuracy")
comp_fig = px.bar(
    comp_long,
    x="Feature",
    y="LDA_Accuracy",
    color="Dataset",
    barmode="group",
    title="LDA Accuracy by Feature Family — CIFAKE vs FF++",
    text_auto=".3f",
)
comp_fig.update_layout(yaxis_range=[0, 1])
st.plotly_chart(comp_fig, use_container_width=True)

st.subheader("Feature family leaderboard")
st.caption(
    "Families ranked by mean LDA accuracy across both datasets. The gap column "
    "(CIFAKE − FF++) shows how much each family's signal degrades on manipulated "
    "video relative to fully synthetic imagery."
)

leaderboard = comparison.copy()
leaderboard["Average"] = leaderboard[["CIFAKE", "FF++"]].mean(axis=1)
leaderboard["Gap (CIFAKE - FF++)"] = leaderboard["CIFAKE"] - leaderboard["FF++"]
leaderboard = leaderboard.sort_values("Average", ascending=False).reset_index(drop=True)
leaderboard.insert(0, "Rank", leaderboard.index + 1)

st.dataframe(
    leaderboard,
    use_container_width=True,
    hide_index=True,
    column_config={
        "CIFAKE": st.column_config.NumberColumn(format="%.4f"),
        "FF++": st.column_config.NumberColumn(format="%.4f"),
        "Average": st.column_config.ProgressColumn(
            format="%.3f", min_value=0.0, max_value=1.0
        ),
        "Gap (CIFAKE - FF++)": st.column_config.NumberColumn(format="%.4f"),
    },
)

# ---------------------------------------------------------------------------
# Experiment 5: Cross-dataset transfer
# Source: cross_dataset_cifake_to_ffpp_*.md
# ---------------------------------------------------------------------------
st.subheader("5. Cross-dataset generalisation")
st.markdown(
    """
The noise-feature model trained on CIFAKE (120,000 samples) applied directly to
FF++ (400 samples) with no adaptation. Transfer collapses to chance.
"""
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Accuracy", "0.500")
c2.metric("Balanced accuracy", "0.500")
c3.metric("ROC-AUC", "0.524")
c4.metric("MCC", "0.000")

st.warning(
    "A CIFAKE-trained detector does not transfer to FF++ (ROC-AUC 0.52, MCC 0.00). "
    "Handcrafted-feature detectors learn dataset-specific generative signatures "
    "rather than a universal notion of 'fakeness'."
)

st.markdown("### Discussion")
st.write(
    "In-domain, noise and texture features give the strongest, most efficient "
    "signal on fully synthetic images, while colour and frequency features lead on "
    "manipulated face video. The central limitation is generalisation: performance "
    "does not carry across datasets, motivating domain-adaptation or hybrid "
    "statistical + deep-learning approaches."
)
