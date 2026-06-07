#!/usr/bin/env python3
"""
QUICK START: FaceForensics++ Integration
=========================================

This file demonstrates the key ways to use the new FF++ support.
Run any of the code blocks below in your environment.
"""

# ============================================================================
# Example 1: Interactive CLI (easiest for first-time use)
# ============================================================================
"""
$ python main.py
Dataset [cifake/faceforensics++]: faceforensics++
Enter dataset path (default: auto-download via Kaggle): 
Auto-download FF++ from Kaggle if path missing? [Y/n]: y
Number of samples (default: 5000): 1000
"""

# ============================================================================
# Example 2: Direct FF++ benchmark script (fastest for automation)
# ============================================================================
"""
# Download FF++ (first time) and analyze 500 test samples
$ python test_faceforensicspp.py --samples 500 --split test --output-dir ./ff_results

# Or use existing dataset directory
$ python test_faceforensicspp.py --data-dir /path/to/ff/data --samples 1000
"""

# ============================================================================
# Example 3: Programmatic usage in Python
# ============================================================================
"""
from analyzer import CIFAKEAnalyzer

# Setup analyzer for FaceForensics++
analyzer = CIFAKEAnalyzer(
    dataset_name="faceforensics++",    # Switches to FF++ mode
    data_dir="",                        # Empty = auto-download from Kaggle
    num_samples=2000,                   # Limit samples for faster testing
    auto_download_kaggle=True,          # Enable Kaggle auto-download
    output_dir="./results",
)

# Run full analysis: feature extraction + LDA benchmark + report generation
results = analyzer.run_analysis()

# Access results dictionary
for feature_type, stats in results.items():
    print(f"{feature_type}: LDA accuracy = {stats['lda_accuracy']:.4f}")

# Output files created:
# - results/faceforensicsplus_plus_features_noise.npz
# - results/faceforensicsplus_plus_features_texture.npz
# - results/faceforensicsplus_plus_features_all_enhanced.npz
# - results/faceforensicsplus_plus_analysis_report_<timestamp>.md
# - results/faceforensicsplus_plus_analysis_enhanced_visualization.png
"""

# ============================================================================
# Example 4: Hybrid CIFAKE + FaceForensics++ evaluation
# ============================================================================
"""
from analyzer import CIFAKEAnalyzer

datasets = [
    ("cifake", "./cifake_data"),
    ("faceforensics++", ""),  # Empty = auto-download
]

for ds_name, ds_path in datasets:
    analyzer = CIFAKEAnalyzer(
        dataset_name=ds_name,
        data_dir=ds_path,
        num_samples=1000,
        auto_download_kaggle=(ds_path == ""),
    )
    results = analyzer.run_analysis()
    print(f"\\n{ds_name.upper()} Results:")
    for feat, stats in sorted(results.items(), 
                              key=lambda x: x[1]['lda_accuracy'], 
                              reverse=True):
        print(f"  {feat:10s}: {stats['lda_accuracy']:.4f}")
"""

# ============================================================================
# Example 5: Using exported features with train.py
# ============================================================================
"""
# After running FF++ analysis, features are exported as NPZ files.
# Use them with your training pipeline:

$ python train.py \\
  --npz ./results/faceforensicsplus_plus_features_all_enhanced.npz \\
  --test-size 0.3 \\
  --tune

# This trains a logistic regression classifier on FF++ features
# and outputs performance metrics.
"""

# ============================================================================
# Requirements & Setup
# ============================================================================
"""
1. Install dependencies (if not already done):
   pip install kagglehub[pandas-datasets] opencv-python-headless \\
       matplotlib PyWavelets Pillow torch torchvision scikit-image

2. Setup Kaggle API (for first-time FF++ download):
   - Get your API key from: https://www.kaggle.com/account/api
   - Place in ~/.kaggle/kaggle.json (chmod 600)
   
   Or set environment variable:
   export KAGGLE_USERNAME=<your-username>
   export KAGGLE_KEY=<your-api-key>

3. Run the integration tests to confirm everything works:
   python test_integration.py
"""

# ============================================================================
# Available Datasets
# ============================================================================
"""
CIFAKE (default):
  - Path: ./cifake_data
  - Size: ~5000 train/test images per class
  - Usage: CIFAKEAnalyzer(dataset_name="cifake", ...)

FaceForensics++ (new):
  - Source: Kaggle (xdxd003/ff-c23)
  - Auto-download: <0.5 GB with auto-download flag
  - Usage: CIFAKEAnalyzer(dataset_name="faceforensics++", ...)
  - Aliases: "ff++", "ff-c23", "faceforensicspp"
"""

# ============================================================================
# Output File Naming
# ============================================================================
"""
Files are prefixed with a safe dataset name:
  
  CIFAKE:           cifake_features_*.npz
                    cifake_analysis_report_*.md
                    
  FaceForensics++:  faceforensicsplus_plus_features_*.npz
                    faceforensicsplus_plus_analysis_report_*.md
                    
This avoids filesystem issues with special characters (+ and -).
"""

if __name__ == "__main__":
    print(__doc__)
