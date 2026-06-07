# Statistical Deepfake Detection Thesis App

Interactive Streamlit platform for a Bachelor Thesis on **interpretable statistical deepfake detection**.

This app is designed to:
- explain the deepfake problem context,
- demonstrate image-level and patch-level statistical analysis,
- visualize handcrafted feature families,
- present experimental benchmark results,
- summarize conclusions and future directions.

---

## 1) Project Purpose

This project implements a lightweight, explainable forensic workflow for synthetic image analysis.
Instead of relying only on black-box deep models, it investigates whether **statistical descriptors** can expose generation artifacts.

### Core thesis idea
Deepfake generators may leave measurable inconsistencies in:
- **Noise patterns**
- **Frequency spectrum**
- **Texture structure**
- **Color relationships**
- **Intensity distributions**

The app provides interactive visual evidence of these signals so they can support forensic reasoning.

---

## 2) What the App Should Do (Functional Expectations)

When running correctly, the app should:

1. Open as a multi-page Streamlit dashboard.
2. Provide a guided flow from research context to results and conclusions.
3. Accept image uploads on analysis pages and process them with OpenCV/NumPy pipelines.
4. Compute interpretable statistics and maps (FFT, residual noise, LBP, etc.).
5. Display plots, maps, metrics, and tables for human inspection.
6. Allow users to export a CSV summary from the detection demo page.

### Important scope note
This app is an **interpretability and research support tool**, not a legal-grade forensic verdict system.
It helps flag suspicious evidence; final decisions should combine additional metadata, provenance, and model-based methods.

---

## 3) Repository Structure

```
deepfake_thesis_app/
├── app.py
├── requirements.txt
├── assets/                      # placeholder for images/resources
├── data/                        # placeholder for datasets/intermediate files
├── pages/
│   ├── 1_Research_Problem_and_Context.py
│   ├── 2_Interactive_Detection_Demo.py
│   ├── 3_Statistical_Feature_Explorer.py
│   ├── 4_Experimental_Findings_and_Benchmarks.py
│   └── 5_Conclusions_and_Future_Work.py
└── utils/
    ├── feature_extraction.py
    ├── image_processing.py
    ├── ui.py
    └── visualization.py
```

Additional workspace file:
- `docs/Andrei_Piscoran___Licenta.pdf` (reference thesis document)

---

## 4) Technology Stack

- **Python**
- **Streamlit** (dashboard UI)
- **OpenCV** (image decoding/processing)
- **NumPy / SciPy** (numerical/statistical computation)
- **scikit-image** (LBP + GLCM texture features)
- **Matplotlib / Plotly** (visualization)
- **Pandas** (tabular summaries)

Dependencies are declared in `requirements.txt`.

---

## 5) Installation and Run

## Prerequisites
- Python 3.9+ recommended
- Linux/Windows/macOS

## Setup
```bash
cd deepfake_thesis_app
python -m venv .venv
source .venv/bin/activate     # Linux/macOS
# .venv\Scripts\activate      # Windows
pip install -r requirements.txt
```

## Launch
```bash
streamlit run app.py
```

Then open the local URL printed by Streamlit (typically `http://localhost:8501`).

---

## 6) Full Page-by-Page Behavior

## Home (`app.py`)
Purpose:
- entry page for the thesis platform,
- displays key metrics and navigation context.

What appears:
- high-level metrics (feature families, anomaly mapping mode, key finding),
- section summary list,
- thesis focus statement,
- usage hint for best experience.

## Page 1 — Research Problem and Context
File: `pages/1_Research_Problem_and_Context.py`

Purpose:
- motivate deepfake detection,
- explain threat relevance and why statistical methods matter.

What appears:
- definitions and risks,
- argument for interpretable statistical cues,
- graphviz diagram of generation architecture families,
- compact methodology overview.

## Page 2 — Interactive Detection Demo
File: `pages/2_Interactive_Detection_Demo.py`

Purpose:
- core interactive forensic demo for one uploaded image.

Inputs:
- uploaded image (`png`, `jpg`, `jpeg`, `bmp`)
- sidebar controls:
  - `patch_size` (16, 32, 48, 64, 96)
  - `overlay_alpha` (0.1–0.9)
  - `suspicious_threshold` (0.40–0.95)

Computed modules:
- patch anomaly map,
- FFT features,
- noise residual,
- texture descriptors.

Tabs and outputs:
1. **Suspicious Regions**
   - original image,
   - heatmap overlay,
   - anomaly score map,
   - area-above-threshold progress metric.

2. **Frequency Artifacts**
   - FFT magnitude + log-spectrum visualizations,
   - radial spectral profile line chart.

3. **Noise Residual**
   - normalized residual map,
   - residual standard deviation and variance.

4. **Texture Analysis**
   - LBP map,
   - gradient magnitude map,
   - LBP histogram.

5. **Interpretation**
   - summary metrics table,
   - downloadable CSV (`deepfake_analysis_summary.csv`),
   - caution that results are supportive evidence, not final verdict.

## Page 3 — Statistical Feature Explorer
File: `pages/3_Statistical_Feature_Explorer.py`

Purpose:
- isolate and explain each feature family independently.

Input behavior:
- optional upload,
- if no upload, generates an internal demo image.

Tabs:
1. **Texture**: LBP map, gradient map, Gabor response, GLCM stats JSON.
2. **Color**: RGB channel means/std and correlation heatmap.
3. **Frequency**: FFT magnitude and radial spectral profile.
4. **Noise**: residual map and residual statistics.
5. **Intensity**: grayscale view, mean/variance/skewness/kurtosis/entropy, histogram.

## Page 4 — Experimental Findings and Benchmarks
File: `pages/4_Experimental_Findings_and_Benchmarks.py`

Purpose:
- communicate comparative results from thesis experiments.

What appears:
- dataset and protocol summary,
- benchmark table,
- bar chart of LDA accuracy by feature family,
- interpretation notes.

Current benchmark values in app:
- Noise: `0.88`
- FFT: `0.83`
- Texture: `0.76`
- Color: `0.71`
- Intensity: `0.69`

## Page 5 — Conclusions and Future Work
File: `pages/5_Conclusions_and_Future_Work.py`

Purpose:
- thesis-style closing section.

What appears:
- key findings,
- practical implications,
- future research directions (hybrid models, robustness tests, video extension).

---

## 7) Core Processing Pipeline (Implementation-Level)

For an uploaded RGB image, the app executes the following logic:

1. Decode upload and enforce RGB format.
2. Convert to grayscale where needed.
3. Compute handcrafted descriptors in parallel domains.
4. Aggregate patch-level anomaly signals.
5. Normalize maps to `[0, 1]` for comparability.
6. Visualize and summarize results.

### Patch anomaly scoring
In `compute_patch_anomaly_map`, each patch score is a weighted combination of:
- residual noise variability,
- frequency-domain energy,
- gradient variability.

Current weighting:
- Residual component: `0.45`
- Frequency component: `0.35`
- Gradient component: `0.20`

The anomaly grid is normalized and resized to image resolution to produce the final heatmap.

---

## 8) Utility Modules Explained

## `utils/image_processing.py`
- image decoding from Streamlit uploader,
- RGB/gray conversion utilities,
- min-max normalization,
- image fit-to-box resizing for UI consistency.

## `utils/feature_extraction.py`
- `compute_fft_features`: FFT magnitude, log spectrum, radial profile, summary stats.
- `compute_noise_residual`: Gaussian denoise residual maps and moments.
- `compute_texture_features`: LBP + Sobel gradient features.
- `compute_glcm_features`: contrast/correlation/energy/homogeneity.
- `compute_gabor_response`: directional texture filtering.
- `compute_patch_anomaly_map`: patch-level fusion scoring.
- `compute_color_statistics`: channel means/std/correlation.
- `compute_intensity_statistics`: histogram moments + entropy.

## `utils/visualization.py`
- anomaly heatmap overlay (OpenCV colormap blend),
- Plotly figures for color and intensity analytics.

## `utils/ui.py`
- shared sidebar/theme CSS for consistent thesis app appearance.

---

## 9) Mapping to a BSc Thesis (Overleaf-Ready Guidance)

You can directly map the app into thesis chapters:

1. **Introduction**
   - motivation, risks, and problem definition from Page 1.

2. **Related Methods / Background**
   - generation families (GAN, diffusion, etc.) and statistical rationale.

3. **Methodology**
   - detailed feature extraction and patch anomaly pipeline from `utils/feature_extraction.py`.

4. **System Implementation**
   - architecture and Streamlit page design (`app.py` + `pages/`).

5. **Experiments and Results**
   - Page 4 protocol and benchmark values.

6. **Discussion**
   - interpretability strengths and operational constraints.

7. **Conclusions and Future Work**
   - directly aligned with Page 5.

### Suggested figure sources
- Anomaly overlay + suspicious map from Page 2,
- FFT and radial profile plots from Pages 2/3,
- GLCM/Gabor/LBP outputs from Page 3,
- benchmark bar chart from Page 4.

---

## 10) Limitations

- No end-to-end model training pipeline in this app version.
- Benchmark values are presented as thesis results, not recomputed in real-time.
- Single-image analysis focus (no video temporal analysis yet).
- Performance depends on image quality, compression, resizing, and post-processing.

---

## 11) Ethical and Practical Use

This software should be used responsibly:
- for research, education, and forensic support,
- not as sole evidence for accusations,
- with awareness of false positives/false negatives.

Always pair outputs with source validation, context analysis, and additional forensic methods.

---

## 12) Quick Troubleshooting

- If upload fails:
  - verify image format (`png/jpg/jpeg/bmp`) and file integrity.
- If Streamlit does not start:
  - check active virtual environment and reinstall dependencies.
- If plots are missing:
  - ensure `matplotlib` and `plotly` are installed from `requirements.txt`.
- If OpenCV errors occur:
  - reinstall `opencv-python` in the same active environment.

---

## 13) Minimal Reproducible Workflow (for Demo/Defense)

1. Run `streamlit run app.py`.
2. Open **Deepfake Detection Demo**.
3. Upload one real image and one suspected synthetic image.
4. Compare:
   - suspicious-region ratio,
   - FFT/radial profile behavior,
   - noise residual statistics,
   - texture maps.
5. Export CSV summaries and include tables/figures in Overleaf.

---

## 14) Authoring Note

This README is intentionally explicit so it can be used as a structured source for your Bachelor Thesis write-up and Overleaf chapter drafting.
