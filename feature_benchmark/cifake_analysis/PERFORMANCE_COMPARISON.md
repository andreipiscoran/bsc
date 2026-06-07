# CIFAKE Linear Model Performance Comparison

Generated: 2026-03-08

## Baseline vs ImprovedNoise+Texture

### Baseline (Noise Only, No Tuning)
- Features: Noise (48 dims)
- Model: LogisticRegression (liblinear, C=1.0)
- **Accuracy: 0.855444**
- **F1-score: 0.853902**
- **ROC-AUC: 0.929247**
- Precision: 0.863110
- Recall: 0.844889

### Improved (Noise + Texture, Tuned)
- Features: Noise + Texture (63 dims)
- Model: LogisticRegression (liblinear, C=8.0) - **Auto-tuned via 5-fold CV**
- **Accuracy: 0.871361** ⬆️ +1.6%
- **F1-score: 0.870226** ⬆️ +1.6%
- **ROC-AUC: 0.943243** ⬆️ +1.4%
- Precision: 0.877976 ⬆️ +1.5%
- Recall: 0.862611 ⬆️ +1.8%
- Best CV ROC-AUC (train only): 0.942948

## Key Improvements

1. **Feature Fusion**: Combined noise residue patterns with texture (LBP, GLCM, Gabor) captures complementary discriminative signals
2. **Hyperparameter Tuning**: CV grid search found C=8.0 (stronger regularization) works better for these 63-dim features
3. **Validation**: 5-fold CV on train split prevents leakage; test AUC matches CV AUC (0.943 vs 0.943)

## How to Reproduce

```bash
# Baseline
python benchmark_noise_linear.py

# Improved (Noise + Texture with tuning)
python benchmark_noise_linear.py --noise-texture --tune

# Custom combination
python benchmark_noise_linear.py \\
  --extra-npz cifake_analysis/cifake_features_texture.npz \\
  --extra-npz cifake_analysis/cifake_features_fft.npz \\
  --tune
```

## Next Steps for Further Improvement

1. **Add FFT features**: `--extra-npz cifake_analysis/cifake_features_fft.npz`
2. **Ensemble**: Stack multiple linear models with different feature subsets
3. **Non-linear**: Try RandomForest / XGBoost on these 63 features
4. **Threshold tuning**: Optimize decision threshold for balanced accuracy
