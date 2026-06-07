# Deepfake & Synthetic Image Detection Thesis

A comprehensive machine learning framework for detecting deepfakes and AI-generated synthetic images using handcrafted features and machine learning classifiers. This thesis investigates the effectiveness of traditional computer vision features in distinguishing authentic images from manipulated content and synthetic images.

**Status:** Active Development | **Date:** 2026

---

## 🎯 Project Overview

This thesis explores the detection of deepfakes and synthetic images through feature-based analysis. The project combines:

- **Handcrafted Feature Extraction**: DCT, intensity analysis, color features, texture descriptors, wavelet transforms, noise profiles, and FFT analysis
- **Multi-Dataset Support**: Integration with CIFAKE and FaceForensics++ datasets
- **Machine Learning Classification**: Linear Discriminant Analysis and ensemble methods for binary classification (Real vs. Fake/Synthetic)
- **Cross-Dataset Evaluation**: Benchmark results across different dataset combinations
- **Interactive Web UI**: Streamlit-based visualization and analysis dashboard

The goal is to determine whether traditional computer vision features can effectively identify deepfakes and AI-generated images, and how well models transfer across different datasets.

---

## 📂 Project Structure

```
Bsc/
├── feature_benchmark/          # Main feature extraction and analysis pipeline
│   ├── main.py                # CLI entrypoint for analysis
│   ├── analyzer.py            # Core CIFAKEAnalyzer class for feature analysis
│   ├── feature_extractor.py   # Handcrafted feature extraction implementations
│   ├── dataset.py             # Dataset loading (CIFAKE, FaceForensics++)
│   ├── train.py               # Model training and evaluation pipeline
│   ├── extract_ffpp_frames.py # Convert FaceForensics++ videos to frames
│   ├── test_integration.py    # Integration tests for dataset support
│   ├── test_faceforensicspp.py # Dedicated FaceForensics++ benchmark
│   ├── FACEFORENSICS_INTEGRATION.md # FF++ integration documentation
│   │
│   ├── cifake_analysis/       # Generated analysis reports and features
│   │   ├── *_analysis_report_*.md    # Feature analysis reports
│   │   ├── cross_dataset_*.md        # Cross-dataset evaluation results
│   │   └── *_features_*.npz         # Precomputed feature files
│   │
│   ├── cifake_data/           # CIFAKE dataset (train/test split)
│   ├── ff_data/               # FaceForensics++ dataset (processed images)
│   └── test_output/           # Temporary test outputs
│
└── UI/                         # Interactive Streamlit web application
    └── deepfake_thesis_app/   # Main Streamlit app
        ├── app.py
        ├── requirements.txt
        └── docs/
```

---

## 🔬 Feature Types

The framework extracts **7 distinct feature categories** from images:

### 1. **DCT Features** (Discrete Cosine Transform)
- Captures frequency domain characteristics
- Separates natural vs. compressed artifacts
- Detects compression patterns in deepfakes

### 2. **Intensity Features**
- Histogram-based statistics
- Entropy and spread measurements
- Luminance distribution analysis

### 3. **Color Features**
- RGB channel statistics and correlations
- Color space conversions (HSV, YCbCr)
- Chroma subsampling artifacts

### 4. **Texture Features**
- Local Binary Pattern (LBP) histograms
- Gray-Level Co-occurrence Matrix (GLCM)
- Gabor filter responses
- Edge and gradient statistics

### 5. **Wavelet Features**
- Multi-scale wavelet decomposition
- Energy distribution across frequency bands
- Temporal consistency in wavelets

### 6. **Noise Features**
- Laplacian variance (Kurtosis)
- Noise floor estimation
- High-frequency noise profile analysis

### 7. **FFT Features**
- Radial frequency spectrum
- Magnitude and phase distribution
- Frequency domain anomaly detection

---

## 📊 Datasets

### CIFAKE Dataset
- **Real Images**: Authentic photographs
- **AI-Generated Images**: Produced by various AI generative models
- **Size**: Thousands of labeled train/test samples
- **Layout**: `data/train/{REAL,FAKE}` and `data/test/{REAL,FAKE}`

### FaceForensics++ (FF++)
- **Real Videos**: Authentic face recordings
- **Deepfake Videos**: Generated using techniques like Face2Face, FaceSwap, NeuralTextures, DeepFaceLab
- **Auto-Download**: KaggleHub integration for automatic dataset fetching
- **Processing**: Automatic frame extraction and train/test splitting

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.8+**
- **pip** or **conda**
- **Kaggle API** (for FaceForensics++ auto-download)

### Installation

1. **Clone the repository:**
   ```bash
   git clone <repo-url>
   cd Bsc
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Kaggle (optional, for FaceForensics++ auto-download):**
   ```bash
   # Place your kaggle.json in ~/.kaggle/kaggle.json
   chmod 600 ~/.kaggle/kaggle.json
   ```

### Quick Start

#### Option 1: Interactive CLI Analysis
```bash
cd feature_benchmark
python main.py
```

You'll be prompted to:
- Select a dataset (CIFAKE or FaceForensics++)
- Provide the dataset path (or enable auto-download for FF++)
- Choose analysis parameters

#### Option 2: Direct CIFAKE Analysis
```bash
cd feature_benchmark
python main.py --dataset cifake --data-dir ./cifake_data --samples 5000
```

#### Option 3: FaceForensics++ Benchmark
```bash
cd feature_benchmark
python test_faceforensicspp.py --samples 2000 --split test --auto-download
```

#### Option 4: Training & Evaluation
```bash
cd feature_benchmark
python train.py --dataset cifake --data-dir ./cifake_data --train-split 0.8 --val-split 0.1
```

#### Option 5: Interactive Web UI
```bash
cd UI/deepfake_thesis_app
streamlit run app.py
```

Access the dashboard at `http://localhost:8501`

---

## 📈 Workflow

### Feature Extraction Pipeline
```
Raw Images → Preprocessing → Feature Extraction → Feature Vectors
                                     ↓
                        [7 Feature Categories]
                        ↓
                   Vectorized Output (NPZ)
```

### Analysis & Classification
```
Feature Vectors → Statistical Analysis → Model Training → Evaluation Metrics
                        ↓                        ↓            ↓
              Correlation Analysis    LDA Classifier    Accuracy, AUC, ROC
              PCA Analysis            Ensemble Methods   Confusion Matrices
              Feature Importance      Cross-Validation   Cross-Dataset Transfer
```

### Output
- **Reports**: Markdown analysis reports with statistics
- **Visualizations**: Feature distributions, ROC curves, confusion matrices
- **Features**: Compressed NPZ files for further analysis
- **Models**: Trained classifiers for deployment

---

## 🔧 Key Modules

### `analyzer.py` - CIFAKEAnalyzer
Main analysis engine supporting multi-dataset feature extraction and classification.

**Key methods:**
- `load_dataset()`: Load CIFAKE or FaceForensics++ dataset
- `extract_advanced_features()`: Generate all 7 feature types
- `extract_pca_features()`: Dimensionality reduction
- `compute_feature_statistics()`: Statistical analysis
- `analyze()`: Full pipeline execution
- `generate_report()`: Create markdown analysis report

**Parameters:**
- `dataset_name`: "cifake" or "faceforensics++"
- `auto_download_kaggle`: Enable KaggleHub auto-download
- `num_samples`: Maximum images to process
- `batch_size`: Processing batch size

### `feature_extractor.py` - EnhancedFeatureExtractor
Implements all feature extraction methods with parallel processing support.

**Feature categories:**
- DCT-based frequency analysis
- Statistical intensity measurements
- Color channel analysis
- Texture descriptors (LBP, GLCM, Gabor)
- Multi-scale wavelet decomposition
- Noise characteristics
- FFT spectral analysis

### `dataset.py` - Dataset Handling
Supports multiple dataset formats with automatic label inference.

**Key functions:**
- `download_faceforensicspp_dataset()`: KaggleHub auto-download
- `CIFAKEImageDataset`: PyTorch dataset class with train/test split support
- Automatic label inference from directory structure

### `train.py` - Training Pipeline
End-to-end training and evaluation framework.

**Capabilities:**
- Feature extraction with multiple workers
- Model training with cross-validation
- Hyperparameter optimization
- Cross-dataset evaluation
- Performance metric computation

---

## 📊 Analysis Examples

### Generate CIFAKE Feature Analysis
```python
from analyzer import CIFAKEAnalyzer

analyzer = CIFAKEAnalyzer(
    data_dir="./cifake_data",
    dataset_name="cifake",
    num_samples=5000
)
analyzer.analyze()
```

**Outputs:**
- `cifake_analysis_report_*.md` - Detailed feature statistics
- `cifake_features_*.npz` - Feature vectors for each category
- `cifake_features_all_enhanced.npz` - Combined features

### Cross-Dataset Evaluation
```python
from analyzer import CIFAKEAnalyzer

# Train on CIFAKE
cifake = CIFAKEAnalyzer(dataset_name="cifake", num_samples=5000)
cifake.analyze()

# Evaluate on FaceForensics++
ffpp = CIFAKEAnalyzer(
    dataset_name="faceforensics++",
    auto_download_kaggle=True,
    num_samples=2000
)
ffpp.analyze()
```

---

## 📝 Available Scripts

| Script | Purpose | Usage |
|--------|---------|-------|
| `main.py` | Interactive CLI for feature analysis | `python main.py` |
| `train.py` | Model training & evaluation | `python train.py --dataset cifake` |
| `test_integration.py` | Validate dataset support | `python test_integration.py` |
| `test_faceforensicspp.py` | FF++ benchmarking | `python test_faceforensicspp.py --samples 2000` |
| `extract_ffpp_frames.py` | Convert FF++ videos to images | Direct API or via analyzer |
| `QUICKSTART.py` | Quick example notebook | `python QUICKSTART.py` |

---

## 📋 Command-Line Options

### `main.py`
```bash
python main.py [OPTIONS]
  --dataset           Dataset name (cifake/faceforensics++)
  --data-dir          Path to dataset directory
  --output-dir        Output directory for results
  --samples           Maximum number of samples to process
  --batch-size        Batch size for processing
  --auto-download     Enable auto-download for FaceForensics++
```

### `train.py`
```bash
python train.py [OPTIONS]
  --dataset           Dataset name
  --data-dir          Dataset path
  --train-split       Training split ratio (default: 0.8)
  --val-split         Validation split ratio (default: 0.1)
  --num-samples       Max samples to use
  --batch-size        Batch size for training
  --epochs            Number of training epochs
  --learning-rate     Learning rate for optimization
```

### `test_faceforensicspp.py`
```bash
python test_faceforensicspp.py [OPTIONS]
  --data-dir          FF++ dataset path (auto-download if empty)
  --dataset-ref       Kaggle dataset reference
  --samples           Max samples to process
  --split             Dataset split (train/test)
  --output-dir        Output directory
```

---

## 📊 Analysis Output

### Generated Reports
The framework automatically generates:

1. **Feature Analysis Reports** (`*_analysis_report_*.md`)
   - Feature statistics and distributions
   - Correlation analysis
   - Feature importance rankings
   - Dataset characteristics

2. **Cross-Dataset Evaluation** (`cross_dataset_*.md`)
   - Model transfer learning results
   - Performance metrics across datasets
   - Feature effectiveness comparison

3. **Benchmark Results** (`*_benchmark_*.md`)
   - Noise robustness analysis
   - Computational performance metrics
   - Comparative feature evaluation

### Generated Files
- **Feature NPZ files**: `*_features_{type}.npz` (vectorized features)
- **Combined features**: `*_features_all_enhanced.npz` (all features combined)
- **Visualizations**: Plots for distribution analysis, ROC curves, etc.
- **Models**: Trained LDA classifiers and metadata

---

## 🧪 Testing

### Run Integration Tests
```bash
cd feature_benchmark
python test_integration.py
```

Validates:
- ✅ Module imports and dependencies
- ✅ Dataset loading for both CIFAKE and FaceForensics++
- ✅ Feature extraction functionality
- ✅ Analyzer initialization and parameter handling
- ✅ Report generation

### Test FaceForensics++ Integration
```bash
python test_faceforensicspp.py --samples 100 --split test
```

---

## 🎨 Interactive Web UI

The Streamlit-based UI provides:

- **Dataset Management**: Upload and visualize datasets
- **Feature Exploration**: Interactive feature statistics and distributions
- **Model Evaluation**: Real-time classification and confidence scores
- **Cross-Dataset Analysis**: Compare model performance across datasets
- **Visualization Dashboard**: Charts, ROC curves, and confusion matrices

**Launch:**
```bash
cd UI/deepfake_thesis_app
streamlit run app.py
```

Access at: `http://localhost:8501`

---

## 📈 Research Questions

This thesis investigates:

1. **Feature Effectiveness**: How well do handcrafted features distinguish real from fake/synthetic images?
2. **Dataset Characteristics**: What are the inherent differences between CIFAKE and FaceForensics++ datasets?
3. **Model Transfer**: Do models trained on one dataset generalize to another?
4. **Feature Importance**: Which feature categories are most discriminative?
5. **Scalability**: How do various feature extraction methods scale with dataset size?
6. **Robustness**: How robust are features to image compression and quality variations?

---

## 🔍 Key Findings (Preliminary)

Based on initial analysis:

- **Texture & Noise Features**: Most discriminative for deepfake detection
- **DCT Features**: Effective for detecting compression artifacts
- **FFT Features**: Useful for frequency-domain anomaly detection
- **Color Features**: Less discriminative but complementary information
- **Dataset-Specific Effects**: Models show transfer limitations across datasets
- **Feature Combinations**: Enhanced feature sets improve classification performance

---

## ⚙️ Configuration

### Environment Variables
```bash
# Enable verbose logging
export DEEPFAKE_DEBUG=1

# Specify Kaggle config location
export KAGGLE_CONFIG_DIR=~/.kaggle
```

### System Requirements
- **Memory**: 4GB minimum (16GB recommended for large datasets)
- **Storage**: 50GB+ (for CIFAKE + FaceForensics++)
- **GPU** (optional): CUDA 11.0+ for accelerated processing

---

## 🤝 Dependencies

See requirements files in respective subdirectories:
- `feature_benchmark/`: Core ML and image processing libraries
- `UI/deepfake_thesis_app/`: Web UI dependencies

Key packages:
- **PyTorch**: Deep learning framework
- **TorchVision**: Computer vision utilities
- **scikit-learn**: Machine learning algorithms
- **scikit-image**: Image processing
- **NumPy, SciPy**: Numerical computing
- **OpenCV**: Image I/O and processing
- **Matplotlib**: Visualization
- **Streamlit**: Web UI framework
- **KaggleHub**: Dataset management

---

## 📚 References

- **FaceForensics++ Dataset**: [Li et al., 2020](https://arxiv.org/abs/2001.08971)
- **CIFAKE Dataset**: [Novoselov et al., 2021](https://arxiv.org/abs/2102.08151)
- **Deepfake Detection Surveys**: Recent literature on media forensics

---

## 📝 License & Attribution

This project uses publicly available datasets and open-source libraries. Ensure compliance with dataset licenses when publishing results.

---

## 🔄 Version History

- **v1.0.0** (Current): Multi-dataset support, enhanced features, cross-dataset evaluation
- **Earlier**: Single-dataset CIFAKE analysis, basic feature extraction

---

## 📞 Notes

- **Auto-download**: KaggleHub requires Kaggle API authentication for FaceForensics++
- **Reproducibility**: Use fixed random seeds for consistent results
- **Scaling**: Feature extraction parallelization available for large datasets
- **Output**: All analysis artifacts are reproducible and timestamped

---

## 🎓 Thesis Objectives

This work contributes to understanding:
- Traditional ML approaches to deepfake detection
- Generalization across deepfake datasets
- Feature-based detection as complementary to deep learning
- Practical deployment considerations for forensics analysis

---

**For questions or issues, refer to individual module docstrings and script documentation.**
