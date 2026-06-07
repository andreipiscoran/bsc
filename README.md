# Deepfake and Synthetic-Image Detection via Handcrafted Features

**B.Sc. Thesis — Babeș-Bolyai University**
**Author:** Andrei Piscoran

A reproducible machine-learning framework for detecting deepfakes and
AI-generated synthetic images from handcrafted computer-vision features. The
work investigates whether traditional, interpretable feature descriptors —
rather than end-to-end deep networks — can reliably distinguish authentic images
from manipulated or fully synthetic content, and how well such detectors
generalise across datasets.

---

## Abstract

Deep generative models have made synthetic and manipulated imagery increasingly
difficult to identify by eye, motivating automated forensic detection. This
thesis studies the discriminative power of seven handcrafted feature families —
DCT, intensity, colour, texture, wavelet, noise-residue, and FFT descriptors —
across two complementary datasets: CIFAKE (fully synthetic vs. natural images)
and FaceForensics++ (manipulated vs. authentic face video frames). The framework
provides feature extraction, statistical separability analysis, linear and LDA
classification, and cross-dataset transfer evaluation. A lightweight linear
model over noise and texture features attains 87.1% accuracy and 0.943 ROC-AUC
on CIFAKE at negligible computational cost, but transfer to FaceForensics++
collapses to near-chance — quantifying both the promise and the principal
limitation of handcrafted-feature detection. Empirical results are summarised in
[`RESULTS.md`](RESULTS.md).

---

## Research Questions

The thesis addresses six questions:

1. How effectively do handcrafted features separate real from synthetic or
   manipulated images?
2. What inherent distributional differences exist between the CIFAKE and
   FaceForensics++ datasets?
3. Do models trained on one dataset generalise to another?
4. Which feature families are most discriminative, and for which manipulation
   type?
5. How do the extraction methods scale with dataset size?
6. How robust are the features to compression and quality variation?

---

## Feature Families

The framework extracts seven categories of descriptors from each image:

| Family | Description |
|--------|-------------|
| **DCT** | Discrete Cosine Transform coefficients capturing frequency-domain and compression artifacts. |
| **Intensity** | Histogram statistics, entropy, and luminance distribution measures. |
| **Colour** | RGB channel statistics and correlations, HSV/YCbCr conversions, chroma-subsampling cues. |
| **Texture** | Local Binary Patterns (LBP), Gray-Level Co-occurrence Matrix (GLCM), Gabor responses, edge/gradient statistics. |
| **Wavelet** | Multi-scale wavelet decomposition and per-band energy distribution. |
| **Noise** | Laplacian variance, noise-floor estimation, and high-frequency residue profiles. |
| **FFT** | Radial frequency spectrum with magnitude and phase distribution for anomaly detection. |

---

## Datasets

**CIFAKE.** Authentic photographs paired with images produced by generative
models, organised as `data/{train,test}/{REAL,FAKE}`. Used at full scale
(120,000 samples) for the principal classification experiments.

**FaceForensics++ (FF++).** Authentic face recordings alongside deepfakes
produced via Face2Face, FaceSwap, NeuralTextures, and related techniques. Videos
are converted to frames and split automatically; the dataset can be fetched
through KaggleHub.

---

## Project Structure

```
bsc/
├── feature_benchmark/              # Feature extraction and analysis pipeline
│   ├── main.py                     # CLI entrypoint for analysis
│   ├── analyzer.py                 # Core analyzer: extraction, statistics, reporting
│   ├── feature_extractor.py        # Handcrafted feature implementations
│   ├── dataset.py                  # Dataset loading (CIFAKE, FaceForensics++)
│   ├── train.py                    # Training and evaluation pipeline
│   ├── extract_ffpp_frames.py      # FaceForensics++ video → frame conversion
│   ├── test_integration.py         # Dataset-support integration tests
│   ├── test_faceforensicspp.py     # FaceForensics++ benchmark
│   ├── FACEFORENSICS_INTEGRATION.md
│   └── cifake_analysis/            # Generated reports, metrics, and feature archives
│
├── UI/deepfake_thesis_app/         # Streamlit analysis dashboard
├── RESULTS.md                      # Summary of experimental findings
└── README.md
```

---

## Installation

**Prerequisites:** Python 3.8+, `pip`, and (optionally) a Kaggle API key for
FaceForensics++ auto-download.

```bash
git clone <repo-url>
cd bsc
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

For FaceForensics++ auto-download, place `kaggle.json` in `~/.kaggle/` and run
`chmod 600 ~/.kaggle/kaggle.json`.

---

## Usage

**Interactive CLI analysis**

```bash
cd feature_benchmark
python main.py
```

**Direct CIFAKE analysis**

```bash
python main.py --dataset cifake --data-dir ./cifake_data --samples 5000
```

**FaceForensics++ benchmark**

```bash
python test_faceforensicspp.py --samples 2000 --split test --auto-download
```

**Training and evaluation**

```bash
python train.py --dataset cifake --data-dir ./cifake_data --train-split 0.8 --val-split 0.1
```

**Interactive dashboard**

```bash
cd UI/deepfake_thesis_app
streamlit run app.py            # http://localhost:8501
```

---

## Methodology and Pipeline

Feature extraction proceeds from raw images through preprocessing to vectorised
feature archives:

```
Raw images → preprocessing → feature extraction (7 families) → feature vectors (.npz)
```

Analysis and classification then apply statistical separability testing,
dimensionality reduction, model training, and evaluation:

```
Feature vectors → statistical analysis → model training → evaluation
   • KS / Hotelling T² tests   • LDA, logistic regression   • Accuracy, ROC-AUC
   • covariance / KL divergence • 5-fold cross-validation    • confusion matrices
   • feature ranking            • cross-dataset transfer      • transfer metrics
```

Every run produces a timestamped Markdown report, compressed feature archives
(`*.npz`), and visualisations (distributions, ROC curves, confusion matrices).

---

## Core Modules

**`analyzer.py`** — the analysis engine. Loads either dataset, extracts all
seven feature families, computes feature statistics, runs PCA, executes the full
pipeline, and emits a Markdown report. Key parameters: `dataset_name`
(`cifake` / `faceforensics++`), `auto_download_kaggle`, `num_samples`,
`batch_size`.

**`feature_extractor.py`** — implements all feature methods with parallel
processing support across the seven families described above.

**`dataset.py`** — dataset handling with automatic label inference from
directory structure, a PyTorch dataset class with train/test splitting, and
KaggleHub-based FaceForensics++ download.

**`train.py`** — end-to-end training and evaluation, including multi-worker
feature extraction, cross-validation, hyperparameter search, and cross-dataset
evaluation.

---

## Command-Line Reference

`main.py`

```
--dataset        Dataset name (cifake / faceforensics++)
--data-dir       Path to dataset directory
--output-dir     Output directory for results
--samples        Maximum number of samples to process
--batch-size     Processing batch size
--auto-download  Enable FaceForensics++ auto-download
```

`train.py`

```
--dataset --data-dir --train-split --val-split --num-samples
--batch-size --epochs --learning-rate
```

`test_faceforensicspp.py`

```
--data-dir --dataset-ref --samples --split --output-dir
```

---

## Testing

```bash
cd feature_benchmark
python test_integration.py                      # module, dataset, extraction, reporting checks
python test_faceforensicspp.py --samples 100 --split test
```

The integration suite validates module imports, dataset loading for both
sources, feature extraction, analyzer initialisation, and report generation.

---

## Results

See [`RESULTS.md`](RESULTS.md) for the full report. In summary:

- **In-domain (CIFAKE):** noise + texture logistic regression reaches **87.1%
  accuracy** and **0.943 ROC-AUC** over 120,000 samples, with no measurable
  overfitting and sub-second training.
- **In-domain (FaceForensics++):** the best single feature family (colour)
  reaches **62.2%** LDA accuracy — manipulated face video is markedly harder than
  fully synthetic imagery.
- **Cross-dataset transfer:** a CIFAKE-trained detector applied to
  FaceForensics++ performs at **chance** (ROC-AUC 0.52, MCC 0.00), revealing
  dataset-specific rather than universal generative signatures.

---

## System Requirements

- **Memory:** 4 GB minimum, 16 GB recommended for large datasets.
- **Storage:** ≥ 50 GB for CIFAKE and FaceForensics++ combined.
- **GPU (optional):** CUDA 11.0+ for accelerated processing.

---

## Dependencies

PyTorch, TorchVision, scikit-learn, scikit-image, NumPy, SciPy, OpenCV,
Matplotlib, Streamlit, and KaggleHub. See the `requirements.txt` files in
`feature_benchmark/` and `UI/deepfake_thesis_app/`.

---

## References

- Rössler et al., *FaceForensics++: Learning to Detect Manipulated Facial Images*, 2019.
- Bird and Lotfi, *CIFAKE: Image Classification and Explainable Identification of AI-Generated Synthetic Images*, 2023.
- Surveys on media forensics and deepfake detection.

> Dataset citations above are indicative; verify exact authorship and years
> against the original publications before submission.

---

## Licence and Attribution

This project uses publicly available datasets and open-source libraries. Ensure
compliance with the respective dataset licences when publishing results.

---

## Notes

- All analysis artifacts are timestamped and reproducible.
- Fixed random seeds are used for repeatability.
- Feature extraction supports parallelisation for large datasets.
- FaceForensics++ auto-download requires Kaggle API authentication.
