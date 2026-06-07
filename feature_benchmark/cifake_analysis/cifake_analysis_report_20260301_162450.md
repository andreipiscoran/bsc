# CIFAKE Dataset Analysis Report

Generated: 2026-03-01 16:24:50
Dataset location: ./cifake_data
Samples analyzed: 20

## Summary Table

| Feature Type | Cov Diff (Frob) | Cov Diff (Spec) | KL Div | Mean KS | Sig@0.05 | Hotelling T2 |
|-------------|-----------------|------------------|--------|---------|----------|--------------|
| Dct | 9248.9002 | 9007.0241 | 4.8558 | 0.2150 | 1 | 31.8863 |
| Intensity | 2842.6453 | 2742.5546 | 10.8798 | 0.2038 | 0 | 32.9719 |
| Color | 6617.0781 | 5334.6923 | 33.4678 | 0.2167 | 0 | 86.6436 |
| Texture | 127255.9443 | 127254.3735 | 22.7325 | 0.2083 | 1 | 57.7569 |
| Wavelet | 6580.3831 | 6538.7815 | 4.2753 | 0.2146 | 0 | 21.0013 |

## Detailed Results

### Dct Features

```
covariance_difference_frobenius: 9248.900233
covariance_difference_spectral: 9007.024115
kl_divergence                 : 4.855796
ks_stat_mean                  : 0.215000
ks_stat_std                   : 0.073485
ks_pvalue_mean                : 0.422635
features_significant_005      : 1
features_significant_001      : 0
hotelling_t2                  : 31.886344
hotelling_f_stat              : 2.820715
```

### Intensity Features

```
covariance_difference_frobenius: 2842.645303
covariance_difference_spectral: 2742.554579
kl_divergence                 : 10.879824
ks_stat_mean                  : 0.203846
ks_stat_std                   : 0.045832
ks_pvalue_mean                : 0.422341
features_significant_005      : 0
features_significant_001      : 0
hotelling_t2                  : 32.971888
hotelling_f_stat              : 2.146099
```

### Color Features

```
covariance_difference_frobenius: 6617.078088
covariance_difference_spectral: 5334.692304
kl_divergence                 : 33.467782
ks_stat_mean                  : 0.216667
ks_stat_std                   : 0.046585
ks_pvalue_mean                : 0.362402
features_significant_005      : 0
features_significant_001      : 0
hotelling_t2                  : 86.643572
hotelling_f_stat              : 2.545618
```

### Texture Features

```
covariance_difference_frobenius: 127255.944286
covariance_difference_spectral: 127254.373519
kl_divergence                 : 22.732538
ks_stat_mean                  : 0.208333
ks_stat_std                   : 0.078351
ks_pvalue_mean                : 0.454654
features_significant_005      : 1
features_significant_001      : 0
hotelling_t2                  : 57.756896
hotelling_f_stat              : 3.159352
```

### Wavelet Features

```
covariance_difference_frobenius: 6580.383050
covariance_difference_spectral: 6538.781524
kl_divergence                 : 4.275317
ks_stat_mean                  : 0.214583
ks_stat_std                   : 0.059036
ks_pvalue_mean                : 0.397470
features_significant_005      : 0
features_significant_001      : 0
hotelling_t2                  : 21.001327
hotelling_f_stat              : 1.503300
```

## Interpretation

**Most discriminative feature type:** Dct (1 significant features at α=0.05)

### Feature Ranking by Discriminative Power

1. **Dct**: 1 significant features
2. **Texture**: 1 significant features
3. **Intensity**: 0 significant features
4. **Color**: 0 significant features
5. **Wavelet**: 0 significant features
