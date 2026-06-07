import argparse
import csv
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
from pathlib import Path

import cv2


FFPP_FAKE_FOLDERS = {
    "deepfakes",
    "face2face",
    "faceswap",
    "neuraltextures",
    "faceshifter",
    "deepfakedetection",
}


def _is_ffpp_root(path_obj: Path):
    if (path_obj / "csv" / "FF++_Metadata.csv").exists():
        return True

    required = {
        "original",
        "deepfakes",
        "face2face",
        "faceswap",
        "neuraltextures",
        "faceshifter",
        "deepfakedetection",
    }
    child_dirs = {p.name.lower() for p in path_obj.iterdir() if p.is_dir()} if path_obj.exists() else set()
    return required.issubset(child_dirs)


def resolve_ffpp_root(input_dir: Path):
    if _is_ffpp_root(input_dir):
        return input_dir

    for child in input_dir.iterdir() if input_dir.exists() else []:
        if child.is_dir() and _is_ffpp_root(child):
            return child

    for subdir in input_dir.rglob("*") if input_dir.exists() else []:
        if not subdir.is_dir():
            continue
        if _is_ffpp_root(subdir):
            return subdir

    return input_dir


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract FaceForensics++ video frames into train/test REAL/FAKE image folders."
    )
    parser.add_argument(
        "--input-dir",
        default="/home/andrei/.cache/kagglehub/datasets/xdxd003/ff-c23/versions/1/FaceForensics++_C23",
        help="Path to FaceForensics++ folder containing class subfolders and csv metadata.",
    )
    parser.add_argument(
        "--output-dir",
        default="./ff_data",
        help="Output root folder. Script creates train/test/REAL/FAKE below this path.",
    )
    parser.add_argument(
        "--frames-per-video",
        type=int,
        default=12,
        help="Number of uniformly sampled frames to extract per video.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Train split ratio (remaining goes to test).",
    )
    parser.add_argument(
        "--max-videos-per-class",
        type=int,
        default=0,
        help="Optional cap per class label (0 = no cap).",
    )
    parser.add_argument(
        "--resize-width",
        type=int,
        default=0,
        help="Resize output frames to this width (0 = keep original).",
    )
    parser.add_argument(
        "--resize-height",
        type=int,
        default=0,
        help="Resize output frames to this height (0 = keep original).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for split reproducibility.")
    parser.add_argument(
        "--jpg-quality",
        type=int,
        default=95,
        help="JPEG quality for output images (1-100).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, min(8, (os.cpu_count() or 1))),
        help="Number of parallel worker threads for per-video extraction.",
    )
    return parser.parse_args()


def read_videos_from_metadata(input_dir: Path):
    metadata_path = input_dir / "csv" / "FF++_Metadata.csv"
    if not metadata_path.exists():
        return []

    videos = []
    with metadata_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rel_path = (row.get("File Path") or "").strip()
            label = (row.get("Label") or "").strip().upper()
            if not rel_path or label not in {"REAL", "FAKE"}:
                continue
            video_path = input_dir / rel_path
            if video_path.exists() and video_path.suffix.lower() == ".mp4":
                videos.append((video_path, label))
    return videos


def read_videos_from_folders(input_dir: Path):
    videos = []
    for mp4_path in sorted(input_dir.rglob("*.mp4")):
        top_folder = mp4_path.relative_to(input_dir).parts[0].lower()
        if top_folder == "original":
            label = "REAL"
        elif top_folder in FFPP_FAKE_FOLDERS:
            label = "FAKE"
        else:
            continue
        videos.append((mp4_path, label))
    return videos


def sample_uniform_indices(total_frames: int, n_samples: int):
    if total_frames <= 0:
        return []
    if total_frames <= n_samples:
        return list(range(total_frames))

    step = (total_frames - 1) / float(n_samples)
    indices = [int(round(i * step)) for i in range(n_samples)]
    return sorted(set(min(max(idx, 0), total_frames - 1) for idx in indices))


def extract_frames(video_path: Path, out_dir: Path, n_frames: int, resize_wh, jpg_quality: int):
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = sample_uniform_indices(total_frames, n_frames)
    if not indices:
        cap.release()
        return 0

    index_set = set(indices)
    frame_idx = 0
    saved = 0
    sample_idx = 0

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        if frame_idx in index_set:
            if resize_wh is not None:
                frame_bgr = cv2.resize(frame_bgr, resize_wh, interpolation=cv2.INTER_AREA)

            out_name = f"{video_path.stem}_f{frame_idx:05d}_{sample_idx:02d}.jpg"
            out_path = out_dir / out_name
            cv2.imwrite(str(out_path), frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), jpg_quality])
            saved += 1
            sample_idx += 1

            if saved >= len(indices):
                break

        frame_idx += 1

    cap.release()
    return saved


def apply_class_cap(items, max_per_class: int):
    if max_per_class <= 0:
        return items

    grouped = defaultdict(list)
    for item in items:
        grouped[item[1]].append(item)

    capped = []
    for label, entries in grouped.items():
        capped.extend(entries[:max_per_class])
    return capped


def split_dataset(items, train_ratio: float, seed: int):
    grouped = defaultdict(list)
    for item in items:
        grouped[item[1]].append(item)

    train_items = []
    test_items = []
    rng = random.Random(seed)

    for label, entries in grouped.items():
        entries = entries[:]
        rng.shuffle(entries)
        split_index = int(len(entries) * train_ratio)
        split_index = max(1, min(split_index, len(entries) - 1)) if len(entries) > 1 else len(entries)
        train_items.extend(entries[:split_index])
        test_items.extend(entries[split_index:])

    return train_items, test_items


def _extract_single_video_task(video_path, split_name, label, output_dir, frames_per_video, resize_wh, jpg_quality):
    target_dir = output_dir / split_name / label
    written = extract_frames(
        video_path=video_path,
        out_dir=target_dir,
        n_frames=frames_per_video,
        resize_wh=resize_wh,
        jpg_quality=jpg_quality,
    )
    return split_name, label, written


def extract_ffpp_dataset(
    input_dir,
    output_dir,
    frames_per_video=12,
    train_ratio=0.8,
    max_videos_per_class=0,
    resize_width=0,
    resize_height=0,
    seed=42,
    jpg_quality=95,
    workers=None,
):
    input_dir = Path(input_dir).resolve()
    output_dir = Path(output_dir).resolve()

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    input_dir = resolve_ffpp_root(input_dir)

    if frames_per_video <= 0:
        raise ValueError("frames_per_video must be > 0")

    if not (0.0 < train_ratio < 1.0):
        raise ValueError("train_ratio must be between 0 and 1")

    if workers is None:
        workers = max(1, min(8, (os.cpu_count() or 1)))
    if workers <= 0:
        raise ValueError("workers must be > 0")

    resize_wh = None
    if resize_width > 0 and resize_height > 0:
        resize_wh = (resize_width, resize_height)

    videos = read_videos_from_metadata(input_dir)
    source = "csv metadata"
    if not videos:
        videos = read_videos_from_folders(input_dir)
        source = "folder scan"

    if not videos:
        raise RuntimeError(
            f"No labeled MP4 videos found in input directory: {input_dir}. "
            "If using Kaggle cache, pass either the FF++ folder or its parent version directory."
        )

    videos = apply_class_cap(videos, max_videos_per_class)
    train_videos, test_videos = split_dataset(videos, train_ratio, seed)

    split_map = {
        "train": train_videos,
        "test": test_videos,
    }

    total_videos = len(train_videos) + len(test_videos)
    print(f"Preparing to process {total_videos} videos...")
    print(f"Frames will be saved under: {output_dir}")
    print(f"Parallel workers: {workers}")

    stats = {
        "train": {"REAL": 0, "FAKE": 0, "videos": 0},
        "test": {"REAL": 0, "FAKE": 0, "videos": 0},
    }

    progress_count = 0
    progress_step = 1
    start_time = time.perf_counter()

    tasks = []
    for split_name, entries in split_map.items():
        for video_path, label in entries:
            tasks.append((video_path, split_name, label))

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(
                _extract_single_video_task,
                video_path,
                split_name,
                label,
                output_dir,
                frames_per_video,
                resize_wh,
                jpg_quality,
            )
            for (video_path, split_name, label) in tasks
        ]

        for future in as_completed(futures):
            split_name, label, written = future.result()
            stats[split_name][label] += written
            stats[split_name]["videos"] += 1

            progress_count += 1
            if progress_count % progress_step == 0 or progress_count == total_videos:
                elapsed = time.perf_counter() - start_time
                avg_per_video = elapsed / max(progress_count, 1)
                remaining_videos = max(total_videos - progress_count, 0)
                eta_sec = avg_per_video * remaining_videos
                percent = 100.0 * progress_count / max(total_videos, 1)
                print(
                    f"Progress: {progress_count}/{total_videos} videos "
                    f"({percent:.1f}%) | elapsed {elapsed/60:.1f} min | ETA {eta_sec/60:.1f} min"
                )

    total_frames = sum(stats[s][c] for s in stats for c in ("REAL", "FAKE"))

    print("=== FaceForensics++ Frame Extraction Complete ===")
    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Source of labels: {source}")
    print(f"Videos processed: train={stats['train']['videos']}, test={stats['test']['videos']}")
    print(
        "Frames written: "
        f"train REAL={stats['train']['REAL']}, train FAKE={stats['train']['FAKE']}, "
        f"test REAL={stats['test']['REAL']}, test FAKE={stats['test']['FAKE']}"
    )
    print(f"Total frames written: {total_frames}")

    return {
        "input": str(input_dir),
        "output": str(output_dir),
        "source": source,
        "stats": stats,
        "total_frames": total_frames,
    }


def run():
    args = parse_args()
    extract_ffpp_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        frames_per_video=args.frames_per_video,
        train_ratio=args.train_ratio,
        max_videos_per_class=args.max_videos_per_class,
        resize_width=args.resize_width,
        resize_height=args.resize_height,
        seed=args.seed,
        jpg_quality=args.jpg_quality,
        workers=args.workers,
    )


if __name__ == "__main__":
    run()
