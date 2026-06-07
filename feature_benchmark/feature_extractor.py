from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import pywt
from scipy import stats
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from skimage.filters import gabor
from skimage.measure import shannon_entropy


FEATURE_TYPES = ['dct', 'intensity', 'color', 'texture', 'wavelet']
ENHANCED_FEATURE_TYPES = FEATURE_TYPES + ['noise', 'fft']


class FeatureExtractor:
    """Extracts handcrafted feature groups from RGB images."""

    def __init__(self, num_workers=None):
        self.num_workers = num_workers

    def extract_features_from_batch(self, images_np):
        """
        Extract all features from a batch of numpy images.

        Args:
            images_np: numpy array of shape (batch_size, H, W, C) with values in [0,1]
        """
        if self.num_workers and self.num_workers > 1:
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                per_image = list(executor.map(self._extract_features_single, images_np))
        else:
            per_image = [self._extract_features_single(img) for img in images_np]

        feature_keys = list(per_image[0].keys()) if per_image else FEATURE_TYPES
        batch_features = {key: [] for key in feature_keys}

        for feats in per_image:
            for key in feature_keys:
                batch_features[key].append(feats[key])

        for key in batch_features:
            batch_features[key] = np.array(batch_features[key])
            # Sanitize NaN values
            batch_features[key] = self._sanitize_features(batch_features[key])

        return batch_features

    def _sanitize_features(self, features_array):
        """
        Replace NaN and inf values with 0 or safe defaults.
        Ensures compatibility with sklearn classifiers.
        """
        features_array = np.asarray(features_array)
        # Replace NaN with 0
        features_array = np.nan_to_num(features_array, nan=0.0, posinf=0.0, neginf=0.0)
        return features_array

    def _extract_features_single(self, img):
        img_uint8 = (img * 255).astype(np.uint8)
        return {
            'dct': self._extract_dct_features_single(img_uint8),
            'intensity': self._extract_intensity_features_single(img_uint8),
            'color': self._extract_color_features_single(img_uint8),
            'texture': self._extract_texture_features_single(img_uint8),
            'wavelet': self._extract_wavelet_features_single(img_uint8),
        }

    def _extract_dct_features_single(self, img_uint8):
        gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)
        dct = cv2.dct(np.float32(gray) / 255.0)
        dct_flat = dct.flatten()

        features = [
            np.mean(dct_flat),
            np.std(dct_flat),
            np.median(dct_flat),
            stats.skew(dct_flat),
            stats.kurtosis(dct_flat),
            np.sum(np.abs(dct_flat) > 0.1),
            np.sum(dct_flat[:10]),
            np.sum(dct_flat[-100:]),
        ]

        h, w = dct.shape
        block_size = 8
        energies = []
        for i in range(0, h, block_size):
            for j in range(0, w, block_size):
                block = dct[i:i + block_size, j:j + block_size]
                energies.append(np.sum(block ** 2))
        features.extend([np.mean(energies), np.std(energies)])

        return np.array(features)

    def _extract_intensity_features_single(self, img_uint8):
        gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)

        features = [
            np.mean(gray),
            np.std(gray),
            np.median(gray),
            stats.skew(gray.flatten()),
            stats.kurtosis(gray.flatten()),
            shannon_entropy(gray),
            np.percentile(gray, 25),
            np.percentile(gray, 75),
            np.percentile(gray, 95) - np.percentile(gray, 5),
        ]

        hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
        hist = hist / hist.sum()
        hist_diff = np.diff(hist)

        features.extend([
            np.sum(hist[:128]),
            np.sum(hist[128:]),
            len(np.where(hist_diff > 0)[0]),
            len(np.where(hist_diff < 0)[0]),
        ])

        return np.array(features)

    def _extract_color_features_single(self, img_uint8):
        r, g, b = img_uint8[:, :, 0], img_uint8[:, :, 1], img_uint8[:, :, 2]

        features = [
            np.mean(r), np.std(r),
            np.mean(g), np.std(g),
            np.mean(b), np.std(b),
            np.mean(r - g),
            np.mean(r - b),
            np.mean(g - b),
            np.corrcoef(r.flatten(), g.flatten())[0, 1],
            np.corrcoef(r.flatten(), b.flatten())[0, 1],
            np.corrcoef(g.flatten(), b.flatten())[0, 1],
        ]

        hsv = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2HSV)
        h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
        features.extend([np.mean(h), np.std(h), np.mean(s), np.std(s), np.mean(v), np.std(v)])

        lab = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2LAB)
        l, a, b_lab = lab[:, :, 0], lab[:, :, 1], lab[:, :, 2]
        features.extend([np.mean(l), np.std(l), np.mean(a), np.std(a), np.mean(b_lab), np.std(b_lab)])

        return np.array(features)

    def _extract_texture_features_single(self, img_uint8):
        gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)
        features = []

        radius = 1
        n_points = 8 * radius
        lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
        lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3))
        lbp_hist = lbp_hist / lbp_hist.sum()

        features.extend([
            np.mean(lbp_hist[:5]),
            np.mean(lbp_hist[5:]),
            np.std(lbp_hist),
            lbp_hist[0],
        ])

        try:
            glcm = graycomatrix(gray, distances=[1], angles=[0], levels=256, symmetric=True)
            features.extend([
                graycoprops(glcm, 'contrast')[0, 0],
                graycoprops(glcm, 'dissimilarity')[0, 0],
                graycoprops(glcm, 'homogeneity')[0, 0],
                graycoprops(glcm, 'energy')[0, 0],
                graycoprops(glcm, 'correlation')[0, 0],
            ])
        except Exception:
            features.extend([0, 0, 0, 0, 0])

        for theta in [0, np.pi / 4, np.pi / 2]:
            try:
                real, _imag = gabor(gray, frequency=0.1, theta=theta)
                features.extend([np.mean(real), np.std(real)])
            except Exception:
                features.extend([0, 0])

        return np.array(features)

    def _extract_wavelet_features_single(self, img_uint8):
        gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)

        coeffs = pywt.wavedec2(gray, 'db1', level=2)
        cA, (cH, cV, cD) = coeffs[0], coeffs[1]

        features = [
            np.mean(cA), np.std(cA),
            np.mean(cH), np.std(cH),
            np.mean(cV), np.std(cV),
            np.mean(cD), np.std(cD),
        ]

        energy_total = np.sum(cA ** 2) + np.sum(cH ** 2) + np.sum(cV ** 2) + np.sum(cD ** 2)
        if energy_total > 0:
            features.extend([
                np.sum(cA ** 2) / energy_total,
                np.sum(cH ** 2) / energy_total,
                np.sum(cV ** 2) / energy_total,
                np.sum(cD ** 2) / energy_total,
            ])
        else:
            features.extend([0, 0, 0, 0])

        return np.array(features)


class EnhancedFeatureExtractor(FeatureExtractor):
    """Extended feature extractor with Noise and FFT analysis."""

    def _extract_features_single(self, img):
        img_uint8 = (img * 255).astype(np.uint8)

        base_features = {
            'dct': self._extract_dct_features_single(img_uint8),
            'intensity': self._extract_intensity_features_single(img_uint8),
            'color': self._extract_color_features_single(img_uint8),
            'texture': self._extract_texture_features_single(img_uint8),
            'wavelet': self._extract_wavelet_features_single(img_uint8),
        }

        base_features['noise'] = self._extract_noise_features_single(img_uint8)
        base_features['fft'] = self._extract_fft_features_single(img_uint8)

        return base_features

    def _extract_noise_features_single(self, img_uint8):
        gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY).astype(np.float32)
        features = []

        coeffs = pywt.wavedec2(gray, 'db2', level=3)

        for level in range(1, 4):
            cH, cV, cD = coeffs[level]
            features.extend([
                np.mean(np.abs(cH)), np.std(cH),
                np.mean(np.abs(cV)), np.std(cV),
                np.mean(np.abs(cD)), np.std(cD),
                stats.skew(cH.flatten()), stats.kurtosis(cH.flatten()),
                stats.skew(cV.flatten()), stats.kurtosis(cV.flatten()),
                stats.skew(cD.flatten()), stats.kurtosis(cD.flatten()),
            ])

        try:
            laplacian = cv2.Laplacian(gray, cv2.CV_32F)
        except cv2.error:
            laplacian = cv2.Laplacian(gray.astype(np.uint8), cv2.CV_16S).astype(np.float32)
        lap_abs = np.abs(laplacian)
        features.extend([
            np.mean(lap_abs),
            np.std(laplacian),
            np.percentile(lap_abs, 90),
            np.sum(lap_abs > np.mean(lap_abs) * 2),
        ])

        kernel = np.array([
            [-1, -1, -1],
            [-1, 8, -1],
            [-1, -1, -1],
        ]) / 8
        high_freq = cv2.filter2D(gray, -1, kernel)

        features.extend([
            np.mean(np.abs(high_freq)),
            np.std(high_freq),
            stats.skew(high_freq.flatten()),
            stats.kurtosis(high_freq.flatten()),
            np.percentile(high_freq, 95) - np.percentile(high_freq, 5),
        ])

        h, w = gray.shape
        block_correlations = []
        for i in range(0, h - 16, 16):
            for j in range(0, w - 16, 16):
                block = gray[i:i + 16, j:j + 16]
                pattern = block[::2, ::2]
                if pattern.size > 1:
                    corr = np.corrcoef(pattern.flatten(), block[1::2, 1::2].flatten())[0, 1]
                    if not np.isnan(corr):
                        block_correlations.append(corr)

        features.extend([
            np.mean(block_correlations) if block_correlations else 0,
            np.std(block_correlations) if block_correlations else 0,
        ])

        noise_residue = gray - cv2.GaussianBlur(gray, (5, 5), 1.0)
        noise_residue = noise_residue - np.mean(noise_residue)
        noise_residue = noise_residue / (np.std(noise_residue) + 1e-8)

        noise_quantized = np.digitize(noise_residue, bins=np.linspace(-3, 3, 20))
        hist, _ = np.histogram(noise_quantized, bins=20)
        hist = hist / (hist.sum() + 1e-12)
        hist = hist[hist > 0]
        noise_entropy = -np.sum(hist * np.log2(hist))
        features.append(noise_entropy)

        return np.array(features)

    def _extract_fft_features_single(self, img_uint8):
        gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)
        features = []

        f = np.fft.fft2(gray.astype(np.float32))
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
        phase_spectrum = np.angle(fshift)

        features.extend([
            np.mean(magnitude_spectrum),
            np.std(magnitude_spectrum),
            np.median(magnitude_spectrum),
            np.percentile(magnitude_spectrum, 90),
            np.percentile(magnitude_spectrum, 10),
            stats.skew(magnitude_spectrum.flatten()),
            stats.kurtosis(magnitude_spectrum.flatten()),
        ])

        features.extend([
            np.mean(phase_spectrum),
            np.std(phase_spectrum),
            np.mean(np.abs(phase_spectrum)),
        ])

        h, w = magnitude_spectrum.shape
        y, x = np.indices((h, w))
        center = np.array([(h - 1) / 2, (w - 1) / 2])
        r = np.hypot(x - center[1], y - center[0])

        total_energy = np.sum(magnitude_spectrum)
        if total_energy > 0:
            spectral_centroid = np.sum(r * magnitude_spectrum) / total_energy
            features.append(spectral_centroid)
        else:
            features.append(0)

        cumsum_energy = np.cumsum(np.sort(magnitude_spectrum.flatten()))
        rolloff_threshold = 0.85 * cumsum_energy[-1]
        spectral_rolloff = np.searchsorted(cumsum_energy, rolloff_threshold) / len(cumsum_energy)
        features.append(spectral_rolloff)

        radial_bands = [5, 15, 30, 50, 80, 120, 200]
        for i in range(len(radial_bands) - 1):
            mask = (r > radial_bands[i]) & (r < radial_bands[i + 1])
            if np.any(mask):
                features.append(np.mean(magnitude_spectrum[mask]))
            else:
                features.append(0)

        angles = np.arctan2(y - center[0], x - center[1])
        for angle_start in np.linspace(-np.pi, np.pi, 8)[:-1]:
            angle_end = angle_start + np.pi / 4
            mask = (angles > angle_start) & (angles < angle_end) & (r > 10)
            if np.any(mask):
                features.append(np.mean(magnitude_spectrum[mask]))
            else:
                features.append(0)

        geometric_mean = np.exp(np.mean(np.log(magnitude_spectrum + 1)))
        arithmetic_mean = np.mean(magnitude_spectrum)
        spectral_flatness = geometric_mean / (arithmetic_mean + 1e-8)
        features.append(spectral_flatness)

        low_freq_mask = r < min(h, w) * 0.1
        high_freq_mask = r > min(h, w) * 0.3
        low_freq_energy = np.sum(magnitude_spectrum[low_freq_mask])
        high_freq_energy = np.sum(magnitude_spectrum[high_freq_mask])
        high_freq_ratio = high_freq_energy / (low_freq_energy + 1e-8)
        features.append(high_freq_ratio)

        magnitude_flat = magnitude_spectrum.flatten()
        auto_corr = np.correlate(magnitude_flat, magnitude_flat, mode='same')
        auto_corr = auto_corr[len(auto_corr) // 2:]
        peaks = np.where((auto_corr[1:-1] > auto_corr[:-2]) & (auto_corr[1:-1] > auto_corr[2:]))[0] + 1
        features.extend([
            len(peaks) / len(auto_corr),
            np.mean(auto_corr[peaks]) if len(peaks) > 0 else 0,
            np.std(auto_corr[peaks]) if len(peaks) > 0 else 0,
        ])

        return np.array(features)
