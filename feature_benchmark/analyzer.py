import logging
import os
from datetime import datetime

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pywt
import torchvision.transforms as transforms
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from skimage.feature import local_binary_pattern
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import CIFAKEImageDataset, download_faceforensicspp_dataset
from feature_extractor import ENHANCED_FEATURE_TYPES, EnhancedFeatureExtractor

logger = logging.getLogger(__name__)


class CIFAKEAnalyzer:
    def __init__(
        self,
        data_dir="./cifake_data",
        output_dir="./cifake_analysis",
        dataset_name="cifake",
        image_size=256,
        num_samples=5000,
        batch_size=64,
        data_loader_workers=None,
        feature_workers=None,
        dataset_split=None,
        auto_download_kaggle=False,
        kaggle_dataset_ref="xdxd003/ff-c23",
    ):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.dataset_name = dataset_name.lower()
        self.image_size = int(image_size)
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.dataset_split = dataset_split
        self.auto_download_kaggle = auto_download_kaggle
        self.kaggle_dataset_ref = kaggle_dataset_ref

        cpu_count = os.cpu_count() or 1
        if data_loader_workers is None:
            self.data_loader_workers = max(1, min(8, cpu_count - 1))
        else:
            self.data_loader_workers = max(0, data_loader_workers)

        if feature_workers is None:
            self.feature_workers = max(2, min(8, cpu_count))
        else:
            self.feature_workers = max(1, feature_workers)

        os.makedirs(output_dir, exist_ok=True)

        self.transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3),
        ])

        self.feature_extractor = EnhancedFeatureExtractor(num_workers=self.feature_workers)
        self.dataset = None
        self.feature_results = {}
        self.covariance_matrices = {}

    @property
    def dataset_prefix(self):
        return self.dataset_name.replace('+', 'plus').replace('-', '_')

    # -------------------------------------------------------------

    def _normalize_to_uint8(self, arr):
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        min_v = float(np.min(arr))
        max_v = float(np.max(arr))
        if max_v - min_v < 1e-8:
            return np.zeros_like(arr, dtype=np.uint8)
        return ((arr - min_v) / (max_v - min_v) * 255).astype(np.uint8)

    def _build_feature_stage_image(self, feature_type, img_rgb):
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

        if feature_type == 'dct':
            dct = cv2.dct(np.float32(gray) / 255.0)
            return self._normalize_to_uint8(np.log1p(np.abs(dct)))

        if feature_type == 'intensity':
            equalized = cv2.equalizeHist(gray)
            return np.hstack([gray, equalized])

        if feature_type == 'color':
            r = img_rgb[:, :, 0]
            g = img_rgb[:, :, 1]
            b = img_rgb[:, :, 2]
            top = np.hstack([img_rgb, np.dstack([r, np.zeros_like(r), np.zeros_like(r)])])
            bottom = np.hstack([np.dstack([np.zeros_like(g), g, np.zeros_like(g)]), np.dstack([np.zeros_like(b), np.zeros_like(b), b])])
            return np.vstack([top, bottom])

        if feature_type == 'texture':
            radius = 1
            n_points = 8 * radius
            lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
            return self._normalize_to_uint8(lbp)

        if feature_type == 'wavelet':
            cA, (cH, cV, cD) = pywt.dwt2(gray.astype(np.float32), 'db1')
            top = np.hstack([self._normalize_to_uint8(cA), self._normalize_to_uint8(np.abs(cH))])
            bottom = np.hstack([self._normalize_to_uint8(np.abs(cV)), self._normalize_to_uint8(np.abs(cD))])
            return np.vstack([top, bottom])

        if feature_type == 'noise':
            gray_f = gray.astype(np.float32)
            residue = gray_f - cv2.GaussianBlur(gray_f, (5, 5), 1.0)
            lap = cv2.Laplacian(gray_f, cv2.CV_32F)
            return np.hstack([
                self._normalize_to_uint8(np.abs(residue)),
                self._normalize_to_uint8(np.abs(lap)),
            ])

        if feature_type == 'fft':
            fft = np.fft.fft2(gray.astype(np.float32))
            fft_shift = np.fft.fftshift(fft)
            mag = np.log1p(np.abs(fft_shift))
            return self._normalize_to_uint8(mag)

        return img_rgb

    def _select_showcase_samples(self, samples_per_class):
        selected = {0: [], 1: []}

        for img_path, label in zip(self.dataset.image_paths, self.dataset.labels):
            if label not in selected or len(selected[label]) >= samples_per_class:
                continue

            bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if bgr is None:
                continue

            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            selected[label].append((img_path, rgb))

            if all(len(selected[k]) >= samples_per_class for k in selected):
                break

        return selected

    def export_feature_stage_images(self, samples_per_class=3):
        stage_dir = os.path.join(self.output_dir, 'feature_stage_images')
        os.makedirs(stage_dir, exist_ok=True)

        samples = self._select_showcase_samples(samples_per_class=samples_per_class)
        if not samples[0] and not samples[1]:
            logger.warning("Could not collect sample images for feature-stage export.")
            return

        pair_count = min(len(samples[0]), len(samples[1]), samples_per_class)
        if pair_count == 0:
            logger.warning("Need at least one REAL and one FAKE sample for side-by-side stage comparison.")
            return

        for feature_type in ENHANCED_FEATURE_TYPES:
            fig, axes = plt.subplots(pair_count, 2, figsize=(10, 3.8 * pair_count), squeeze=False)

            for pair_idx in range(pair_count):
                real_path, real_rgb = samples[0][pair_idx]
                fake_path, fake_rgb = samples[1][pair_idx]

                real_stage = self._build_feature_stage_image(feature_type, real_rgb)
                fake_stage = self._build_feature_stage_image(feature_type, fake_rgb)

                real_ax = axes[pair_idx][0]
                real_ax.axis('off')
                if real_stage.ndim == 2:
                    real_ax.imshow(real_stage, cmap='gray')
                else:
                    real_ax.imshow(real_stage)
                real_ax.set_title(
                    f"Pair {pair_idx + 1} | REAL | {os.path.basename(real_path)[:24]}",
                    fontsize=9,
                )

                fake_ax = axes[pair_idx][1]
                fake_ax.axis('off')
                if fake_stage.ndim == 2:
                    fake_ax.imshow(fake_stage, cmap='gray')
                else:
                    fake_ax.imshow(fake_stage)
                fake_ax.set_title(
                    f"Pair {pair_idx + 1} | FAKE | {os.path.basename(fake_path)[:24]}",
                    fontsize=9,
                )

            fig.suptitle(
                f"Feature Stage Showcase (Side-by-Side REAL vs FAKE): {feature_type.upper()}",
                fontsize=14,
            )
            fig.tight_layout(rect=[0, 0, 1, 0.96])

            out_path = os.path.join(stage_dir, f"{feature_type}_stage_showcase.png")
            fig.savefig(out_path, dpi=180)
            plt.close(fig)

        logger.info("Feature stage showcase images saved to %s", stage_dir)

    # -------------------------------------------------------------

    def load_dataset(self):
        ffpp_aliases = {"faceforensics++", "faceforensicspp", "ff++", "ff-c23"}
        if self.dataset_name in ffpp_aliases and self.auto_download_kaggle:
            if not self.data_dir or not os.path.exists(self.data_dir):
                logger.info("Dataset path not found (%s). Triggering Kaggle download...", self.data_dir)
                self.data_dir = download_faceforensicspp_dataset(self.kaggle_dataset_ref)

        self.dataset = CIFAKEImageDataset(
            root_dir=self.data_dir,
            transform=self.transform,
            max_samples=self.num_samples,
            split=self.dataset_split,
        )
        return len(self.dataset) > 0

    # -------------------------------------------------------------

    def extract_all_features(self):
        real = {k: [] for k in ENHANCED_FEATURE_TYPES}
        fake = {k: [] for k in ENHANCED_FEATURE_TYPES}

        loader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.data_loader_workers,
            persistent_workers=self.data_loader_workers > 0,
        )

        logger.info(
            "Parallel settings: batch_size=%s, dataloader_workers=%s, feature_workers=%s",
            self.batch_size,
            self.data_loader_workers,
            self.feature_workers,
        )

        for images, labels, _ in tqdm(loader, desc="Extracting"):
            images_np = images.numpy().transpose(0, 2, 3, 1)
            images_np = images_np * 0.5 + 0.5

            feats = self.feature_extractor.extract_features_from_batch(images_np)

            for i, label in enumerate(labels):
                for key in feats:
                    if label == 0:
                        real[key].append(feats[key][i])
                    else:
                        fake[key].append(feats[key][i])

        for k in real:
            real[k] = np.array(real[k])
            fake[k] = np.array(fake[k])

        return real, fake

    # -------------------------------------------------------------

    def compute_covariance_difference(self, real_features, fake_features):
        """Compute Frobenius norm difference between covariance matrices"""
        cov_real = np.cov(real_features.T)
        cov_fake = np.cov(fake_features.T)
        
        cov_diff = np.linalg.norm(cov_real - cov_fake, 'fro')
        return cov_diff

    # -------------------------------------------------------------

    def evaluate_lda_accuracy(self, real_feats, fake_feats, test_size=0.3, random_state=42):
        """Train LDA classifier on features and return test accuracy."""
        X = np.vstack([real_feats, fake_feats])
        y = np.hstack([np.zeros(len(real_feats)), np.ones(len(fake_feats))])

        # Sanitize features: replace NaN/inf with 0
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        lda = LinearDiscriminantAnalysis()
        lda.fit(X_train, y_train)
        y_pred = lda.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        return accuracy

    # -------------------------------------------------------------

    def run_analysis(self):
        logger.info(
            "Starting %s analysis with enhanced features (Noise + FFT)...",
            self.dataset_name,
        )

        if not self.load_dataset():
            logger.error("Dataset loading failed.")
            return None

        self.export_feature_stage_images(samples_per_class=3)

        real_features, fake_features = self.extract_all_features()
        all_results = {}

        for feature_type in ENHANCED_FEATURE_TYPES:
            logger.info(f"Analyzing {feature_type} features...")

            # Compute covariance difference
            cov_diff = self.compute_covariance_difference(
                real_features[feature_type],
                fake_features[feature_type],
            )

            # Calculate LDA accuracy
            lda_acc = self.evaluate_lda_accuracy(
                real_features[feature_type], 
                fake_features[feature_type]
            )

            all_results[feature_type] = {
                'covariance_difference_frobenius': cov_diff,
                'lda_accuracy': lda_acc,
            }
            
            self.covariance_matrices[feature_type] = {
                'real': np.cov(real_features[feature_type].T),
                'fake': np.cov(fake_features[feature_type].T),
            }
            
            self.feature_results[feature_type] = {
                'real': real_features[feature_type],
                'fake': fake_features[feature_type],
                'stats': all_results[feature_type],
            }

        self.generate_report(all_results)
        self.visualize_results(all_results)
        self.export_features()

        return all_results

    # -------------------------------------------------------------

    def generate_report(self, all_results):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = os.path.join(self.output_dir, f"{self.dataset_prefix}_analysis_report_{timestamp}.md")

        with open(report_file, 'w') as f:
            f.write(f"# {self.dataset_name.upper()} Dataset Analysis Report (with Noise & FFT)\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Dataset location: {self.data_dir}\n")
            f.write(f"Samples analyzed: {self.num_samples}\n\n")

            f.write("## Summary Table\n\n")
            f.write("| Feature Type | Cov Diff (Frob) | LDA Accuracy |\n")
            f.write("|--------------|-----------------|--------------|\n")

            # Sort by LDA accuracy for better readability
            sorted_results = sorted(all_results.items(), key=lambda x: x[1]['lda_accuracy'], reverse=True)
            
            for feat_name, results in sorted_results:
                f.write(f"| {feat_name.capitalize()} | ")
                f.write(f"{results['covariance_difference_frobenius']:.4e} | ")
                f.write(f"{results['lda_accuracy']:.4f} |\n")

        logger.info(f"Report generated: {report_file}")

        print("\n" + "=" * 60)
        print("ENHANCED RESULTS - WITH NOISE & FFT")
        print("=" * 60)
        print(f"{'Feature Type':<15} {'Cov Diff':<12} {'LDA Acc':<10}")
        print("-" * 60)
        
        # Sort by LDA accuracy for display
        sorted_items = sorted(all_results.items(), key=lambda x: x[1]['lda_accuracy'], reverse=True)
        for feat_name, results in sorted_items:
            print(
                f"{feat_name.capitalize():<15} "
                f"{results['covariance_difference_frobenius']:<12.4e} "
                f"{results['lda_accuracy']:<10.4f}"
            )
        print("=" * 60)

    # -------------------------------------------------------------

    def visualize_results(self, all_results):
        # Sort by LDA accuracy
        sorted_items = sorted(all_results.items(), key=lambda x: x[1]['lda_accuracy'], reverse=True)
        feat_names = [item[0] for item in sorted_items]

        cov_diffs = np.array([all_results[f]['covariance_difference_frobenius'] for f in feat_names], dtype=float)
        cov_plot_values = np.maximum(cov_diffs, 1e-12)
        lda_accs = [all_results[f]['lda_accuracy'] for f in feat_names]

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Covariance difference plot
        bars1 = axes[0].bar(feat_names, cov_plot_values, color='skyblue')
        axes[0].set_title("Covariance Difference (Frobenius Norm, log scale)")
        axes[0].set_ylabel("Frobenius Norm (log)")
        axes[0].set_yscale('log')
        axes[0].tick_params(axis='x', rotation=45)
        
        # Highlight texture (your previous best)
        if 'texture' in feat_names:
            texture_idx = feat_names.index('texture')
            bars1[texture_idx].set_color('orange')
        
        # Highlight new features
        if 'noise' in feat_names:
            noise_idx = feat_names.index('noise')
            bars1[noise_idx].set_color('red')
        if 'fft' in feat_names:
            fft_idx = feat_names.index('fft')
            bars1[fft_idx].set_color('purple')

        # LDA accuracy plot
        bars2 = axes[1].bar(feat_names, lda_accs, color='lightgreen')
        axes[1].set_title("LDA Classification Accuracy")
        axes[1].set_ylim(0, 1)
        axes[1].set_ylabel("Accuracy")
        axes[1].tick_params(axis='x', rotation=45)
        axes[1].axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)  # Random guess line
        
        # Highlight texture (your previous best)
        if 'texture' in feat_names:
            texture_idx = feat_names.index('texture')
            bars2[texture_idx].set_color('orange')
        
        # Highlight new features
        if 'noise' in feat_names:
            noise_idx = feat_names.index('noise')
            bars2[noise_idx].set_color('red')
        if 'fft' in feat_names:
            fft_idx = feat_names.index('fft')
            bars2[fft_idx].set_color('purple')

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f'{self.dataset_prefix}_analysis_enhanced_visualization.png'))
        plt.close(fig)

        logger.info(f"Enhanced visualization saved to {self.output_dir}")

    # -------------------------------------------------------------

    def export_features(self):
        logger.info("Exporting enhanced features for ML...")

        X_list = []
        feature_names = []
        feature_info = {}

        for feat_name, data in self.feature_results.items():
            real_feats = data['real']
            fake_feats = data['fake']

            X = np.vstack([real_feats, fake_feats])
            y = np.hstack([np.zeros(len(real_feats)), np.ones(len(fake_feats))])

            # Save individual feature type
            filename = os.path.join(self.output_dir, f'{self.dataset_prefix}_features_{feat_name}.npz')
            np.savez(
                filename,
                X=X,
                y=y,
                feature_type=feat_name,
                feature_dim=X.shape[1],
                lda_accuracy=data['stats']['lda_accuracy'],
                label_mapping=np.array(['0=REAL', '1=FAKE']),
            )

            X_list.append(X)
            feature_names.append(feat_name)
            feature_info[feat_name] = {
                'dim': X.shape[1],
                'accuracy': data['stats']['lda_accuracy']
            }

        # Save combined features
        if len(X_list) > 1:
            X_combined = np.hstack(X_list)
            y_combined = np.hstack([
                np.zeros(len(self.feature_results['dct']['real'])),
                np.ones(len(self.feature_results['dct']['fake'])),
            ])

            np.savez(
                os.path.join(self.output_dir, f'{self.dataset_prefix}_features_all_enhanced.npz'),
                X=X_combined,
                y=y_combined,
                feature_names=feature_names,
                feature_info=feature_info,
                total_dim=X_combined.shape[1],
                label_mapping=np.array(['0=REAL', '1=FAKE']),
            )

            logger.info(f"Combined feature matrix shape: {X_combined.shape}")

        logger.info(f"Enhanced features exported to {self.output_dir}")