"""CLI entrypoint for CIFAKE simple feature comparison."""

import logging
import warnings
from pathlib import Path

from analyzer import CIFAKEAnalyzer
from dataset import download_faceforensicspp_dataset
from extract_ffpp_frames import extract_ffpp_dataset

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


FFPP_ALIASES = {"faceforensics++", "faceforensicspp", "ff++", "ff-c23"}


def is_image_dataset_layout(path_obj: Path):
    return (
        (path_obj / "train" / "REAL").exists()
        and (path_obj / "train" / "FAKE").exists()
        and (path_obj / "test" / "REAL").exists()
        and (path_obj / "test" / "FAKE").exists()
    )


def looks_like_ffpp_video_dataset(path_obj: Path):
    if (path_obj / "csv" / "FF++_Metadata.csv").exists():
        return True
    return any(path_obj.glob("**/*.mp4"))


def prepare_faceforensics_dataset(dataset_path: str, auto_download: bool, frames_per_video: int):
    path_obj = Path(dataset_path)

    if path_obj.exists() and is_image_dataset_layout(path_obj):
        return str(path_obj)

    raw_dir = path_obj
    if not raw_dir.exists():
        if not auto_download:
            raise FileNotFoundError(
                f"FaceForensics++ path not found: {raw_dir}. "
                "Enable auto-download or provide a valid path."
            )
        print("🔽 Downloading FaceForensics++ raw videos from Kaggle...")
        raw_dir = Path(download_faceforensicspp_dataset())

    if not looks_like_ffpp_video_dataset(raw_dir):
        if not auto_download:
            raise RuntimeError(
                f"Path does not look like a FaceForensics++ raw video dataset: {raw_dir}"
            )
        print("ℹ️ Provided path is not FF++ raw videos; downloading dataset from Kaggle instead...")
        raw_dir = Path(download_faceforensicspp_dataset())

        if not looks_like_ffpp_video_dataset(raw_dir):
            raise RuntimeError(
                f"Downloaded path is not recognized as FaceForensics++ raw video dataset: {raw_dir}"
            )

    processed_dir = Path("./ff_data")
    if is_image_dataset_layout(processed_dir):
        print(f"📁 Reusing existing extracted FF++ image dataset at: {processed_dir}")
        return str(processed_dir)

    print("🎞️ Converting FF++ videos to train/test REAL/FAKE image folders...")
    print(f"💾 Extracted frames will be saved to: {processed_dir.resolve()}")
    extract_ffpp_dataset(
        input_dir=str(raw_dir),
        output_dir=str(processed_dir),
        frames_per_video=frames_per_video,
        train_ratio=0.8,
        max_videos_per_class=0,
        seed=42,
        jpg_quality=95,
    )
    return str(processed_dir)


def main():
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║     CIFAKE Feature Selection (Simple Version)               ║
    ║     Real vs AI-Generated Images                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """)

    DATASET_NAME = input("Dataset [cifake/faceforensics++] (default: faceforensics++): ").strip().lower()
    if not DATASET_NAME:
        DATASET_NAME = "faceforensics++"

    default_data_dir = "./cifake_data" if DATASET_NAME == "cifake" else "./ff_data"
    DATA_DIR = input(f"Enter dataset path (default: {default_data_dir or 'auto-download via Kaggle'}): ").strip()
    if not DATA_DIR:
        DATA_DIR = default_data_dir

    AUTO_DOWNLOAD = False
    FRAMES_PER_VIDEO = 12
    if DATASET_NAME in FFPP_ALIASES:
        auto_input = input("Auto-download FF++ from Kaggle if path missing? [Y/n]: ").strip().lower()
        AUTO_DOWNLOAD = auto_input in {"", "y", "yes"}

        fpv_input = input("Frames per video [10-20] (default: 12): ").strip()
        if fpv_input:
            try:
                FRAMES_PER_VIDEO = int(fpv_input)
            except ValueError:
                print("⚠️ Invalid number entered, using default 12 frames/video.")
                FRAMES_PER_VIDEO = 12
        FRAMES_PER_VIDEO = max(10, min(20, FRAMES_PER_VIDEO))

        DATA_DIR = prepare_faceforensics_dataset(DATA_DIR, AUTO_DOWNLOAD, FRAMES_PER_VIDEO)
        print(f"✅ Using prepared FF++ image dataset at: {DATA_DIR}")

    NUM_SAMPLES = input("Number of samples (default: 5000): ").strip()
    NUM_SAMPLES = int(NUM_SAMPLES) if NUM_SAMPLES else 5000

    analyzer = CIFAKEAnalyzer(
        dataset_name=DATASET_NAME,
        data_dir=DATA_DIR,
        output_dir="./cifake_analysis",
        num_samples=NUM_SAMPLES,
        auto_download_kaggle=False,
    )

    analyzer.run_analysis()

    print("\n✅ DONE.")
    print("📊 Comparison plots + per-feature stage showcase images generated.")
    print("🖼️ Stage images saved in: ./cifake_analysis/feature_stage_images")
    print("🎯 Use highest accuracy feature for classification.")


if __name__ == "__main__":
    main()