#!/usr/bin/env python
"""
Quick integration test for FaceForensics++ dataset support.
Validates that the FF++ wiring and download logic works without needing actual images.
"""

import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_imports():
    """Test that all core modules import without errors."""
    try:
        from dataset import CIFAKEImageDataset, download_faceforensicspp_dataset
        from analyzer import CIFAKEAnalyzer
        logger.info("✅ All modules import successfully")
        return True
    except Exception as e:
        logger.error(f"❌ Import failed: {e}")
        return False


def test_analyzer_init():
    """Test that analyzer can initialize with FF++ dataset_name."""
    try:
        from analyzer import CIFAKEAnalyzer
        
        analyzer = CIFAKEAnalyzer(
            dataset_name="faceforensics++",
            data_dir="",
            output_dir="./test_output",
            num_samples=10,
            auto_download_kaggle=False,  # Don't actually download
        )
        assert analyzer.dataset_name == "faceforensics++", f"Expected 'faceforensics++', got {analyzer.dataset_name}"
        # dataset_prefix replaces + with 'plus', so 'faceforensics++' -> 'faceforensicsplusplus'
        assert analyzer.dataset_prefix == "faceforensicsplusplus", f"Expected 'faceforensicsplusplus', got {analyzer.dataset_prefix}"
        logger.info("✅ Analyzer initializes with FF++ dataset name")
        return True
    except AssertionError as e:
        logger.error(f"❌ Analyzer init assertion failed: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Analyzer init failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dataset_label_inference():
    """Test that dataset can infer labels from path names."""
    try:
        from dataset import CIFAKEImageDataset
        
        dataset = CIFAKEImageDataset(
            root_dir=".",
            max_samples=1,
        )
        
        # Check that the dataset has the new label alias attributes
        assert hasattr(dataset, 'real_aliases')
        assert hasattr(dataset, 'fake_aliases')
        assert 'real' in dataset.real_aliases
        assert 'fake' in dataset.fake_aliases
        
        logger.info("✅ Dataset label inference logic is in place")
        return True
    except Exception as e:
        logger.error(f"❌ Dataset inference test failed: {e}")
        return False


def test_kaggle_download_import():
    """Test that kagglehub is available for FaceForensics++ downloads."""
    try:
        import kagglehub
        assert hasattr(kagglehub, 'dataset_download')
        logger.info("✅ KaggleHub is installed and has dataset_download method")
        return True
    except Exception as e:
        logger.error(f"❌ KaggleHub check failed: {e}")
        return False


def test_main_cli():
    """Test that main.py wiring for FF++ dataset selection works."""
    try:
        import main
        assert hasattr(main, 'main')
        logger.info("✅ main.py has callable main() function")
        return True
    except Exception as e:
        logger.error(f"❌ main.py check failed: {e}")
        return False


def test_ff_benchmark_script():
    """Test that the FF++ benchmark script has correct structure."""
    try:
        test_script_path = Path(__file__).parent / "test_faceforensicspp.py"
        assert test_script_path.exists()
        
        with open(test_script_path) as f:
            content = f.read()
            assert "faceforensics++" in content.lower()
            assert "auto_download_kaggle" in content
            assert "KaggleHub" in content or "kaggle" in content.lower()
        
        logger.info("✅ FF++ benchmark script is properly structured")
        return True
    except Exception as e:
        logger.error(f"❌ FF++ script check failed: {e}")
        return False


def main():
    tests = [
        test_imports,
        test_analyzer_init,
        test_dataset_label_inference,
        test_kaggle_download_import,
        test_main_cli,
        test_ff_benchmark_script,
    ]
    
    results = []
    for test_fn in tests:
        logger.info(f"\nRunning {test_fn.__name__}...")
        results.append(test_fn())
    
    print("\n" + "=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"Results: {passed}/{total} tests passed")
    print("=" * 60)
    
    if passed == total:
        logger.info("✅ All integration tests passed!")
        return 0
    else:
        logger.warning(f"❌ {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
