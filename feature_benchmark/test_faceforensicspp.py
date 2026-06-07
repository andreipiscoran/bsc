import argparse
import logging
from pathlib import Path

from analyzer import CIFAKEAnalyzer


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def parse_args():
    parser = argparse.ArgumentParser(description="Run FaceForensics++ test benchmark with handcrafted features.")
    parser.add_argument(
        "--data-dir",
        default="",
        help="Path to FaceForensics++ dataset. Leave empty to auto-download from Kaggle.",
    )
    parser.add_argument(
        "--dataset-ref",
        default="xdxd003/ff-c23",
        help="Kaggle dataset reference for FaceForensics++.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=2000,
        help="Maximum samples to process.",
    )
    parser.add_argument(
        "--split",
        default="test",
        choices=["train", "test"],
        help="Dataset split to evaluate.",
    )
    parser.add_argument(
        "--output-dir",
        default="./cifake_analysis",
        help="Directory for reports, visualizations and exported NPZ files.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    analyzer = CIFAKEAnalyzer(
        dataset_name="faceforensics++",
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        num_samples=args.samples,
        dataset_split=args.split,
        auto_download_kaggle=True,
        kaggle_dataset_ref=args.dataset_ref,
    )

    results = analyzer.run_analysis()
    if results is None:
        raise RuntimeError("FaceForensics++ analysis failed.")

    output_dir = Path(args.output_dir)
    print("\nFaceForensics++ test benchmark completed.")
    print(f"Output directory: {output_dir.resolve()}")
    print(
        "Main feature file:",
        output_dir / f"{analyzer.dataset_prefix}_features_all_enhanced.npz",
    )


if __name__ == "__main__":
    main()
