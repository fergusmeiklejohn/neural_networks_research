#!/usr/bin/env python3
"""
Robust script to save ALL training results before Paperspace shutdown
Handles multiple output directories and saves to persistent storage
"""

import glob
import json
import logging
import os
import shutil
from datetime import datetime
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)


def find_all_outputs(base_dir="outputs"):
    """Find all output directories that might contain results"""
    output_dirs = []

    # Common output directory patterns
    patterns = [
        "outputs/*/training_results.json",
        "outputs/*/vocabulary.json",
        "outputs/*/*.h5",
        "outputs/*/*.keras",
        "outputs/*/stage_*.h5",
        "outputs/*/checkpoint*",
    ]

    found_dirs = set()
    for pattern in patterns:
        for filepath in glob.glob(pattern):
            dir_path = os.path.dirname(filepath)
            if dir_path not in found_dirs:
                found_dirs.add(dir_path)
                output_dirs.append(dir_path)
                logger.info(f"Found output directory: {dir_path}")

    return sorted(output_dirs)


def find_model_files(directory):
    """Find all model-related files in a directory"""
    model_files = []
    extensions = [".h5", ".keras", ".json", ".pkl", ".txt", ".log"]

    for ext in extensions:
        pattern = os.path.join(directory, f"*{ext}")
        files = glob.glob(pattern)
        model_files.extend(files)

    # Also check for subdirectories
    for subdir in ["checkpoints", "logs", "models"]:
        subdir_path = os.path.join(directory, subdir)
        if os.path.exists(subdir_path):
            for ext in extensions:
                pattern = os.path.join(subdir_path, f"*{ext}")
                files = glob.glob(pattern)
                model_files.extend(files)

    return model_files


def save_critical_results(save_to_storage=True):
    """Save all critical results to a single directory and optionally to Paperspace storage"""

    # Create results directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(f"compositional_language_results_{timestamp}")
    results_dir.mkdir(exist_ok=True)

    logger.info(f"Saving results to: {results_dir}")

    # Create manifest to track what we save
    manifest = {
        "experiment": "Compositional Language - Distribution Invention",
        "timestamp": timestamp,
        "saved_files": [],
        "output_directories": [],
    }

    # 1. Find all output directories
    output_dirs = find_all_outputs()
    manifest["output_directories"] = output_dirs

    # 2. Copy files from each output directory
    for output_dir in output_dirs:
        dir_name = os.path.basename(output_dir)
        dest_dir = results_dir / f"outputs_{dir_name}"
        dest_dir.mkdir(exist_ok=True)

        # Find and copy all relevant files
        files = find_model_files(output_dir)
        for file_path in files:
            try:
                rel_path = os.path.relpath(file_path, output_dir)
                dest_path = dest_dir / rel_path
                dest_path.parent.mkdir(parents=True, exist_ok=True)

                shutil.copy2(file_path, dest_path)
                manifest["saved_files"].append(str(dest_path.relative_to(results_dir)))
                logger.info(f"✓ Saved: {rel_path}")
            except Exception as e:
                logger.error(f"✗ Failed to save {file_path}: {e}")

    # 3. Look for wandb runs
    wandb_dir = Path("wandb")
    if wandb_dir.exists():
        wandb_dest = results_dir / "wandb_runs"
        try:
            shutil.copytree(
                wandb_dir, wandb_dest, ignore=shutil.ignore_patterns("*.tmp", "*.log")
            )
            logger.info("✓ Saved wandb runs")
            manifest["wandb_saved"] = True
        except Exception as e:
            logger.error(f"✗ Failed to save wandb: {e}")
            manifest["wandb_saved"] = False

    # 4. Save data if not too large
    data_dir = Path("data/processed")
    if data_dir.exists():
        data_dest = results_dir / "data_processed"
        data_dest.mkdir(exist_ok=True)

        # Only copy essential processed files
        essential_files = ["vocabulary.json", "modification_pairs.pkl", "*.json"]
        for pattern in essential_files:
            for file_path in data_dir.glob(pattern):
                try:
                    shutil.copy2(file_path, data_dest / file_path.name)
                    logger.info(f"✓ Saved data file: {file_path.name}")
                except Exception as e:
                    logger.error(f"✗ Failed to save {file_path}: {e}")

    # 5. Extract and summarize results
    summary = {
        "experiment": "Compositional Language - Distribution Invention",
        "timestamp": timestamp,
        "training_complete": False,
        "results_found": {},
    }

    # Look for training results in all output directories
    for output_dir in output_dirs:
        results_file = os.path.join(output_dir, "training_results.json")
        if os.path.exists(results_file):
            try:
                with open(results_file, "r") as f:
                    results = json.load(f)

                dir_name = os.path.basename(output_dir)
                summary["results_found"][dir_name] = {
                    "test_results": results.get("test_results", {}),
                    "config": results.get("config", {}),
                }
                summary["training_complete"] = True

                logger.info(f"✓ Found results in {dir_name}")
            except Exception as e:
                logger.error(f"✗ Failed to load results from {results_file}: {e}")

    # 6. Save manifest and summary
    with open(results_dir / "MANIFEST.json", "w") as f:
        json.dump(manifest, f, indent=2)

    with open(results_dir / "SUMMARY.json", "w") as f:
        json.dump(summary, f, indent=2)

    # 7. Save to Paperspace storage if requested and available
    if save_to_storage and os.path.exists("/storage"):
        storage_dir = Path(f"/storage/compositional_language_results_{timestamp}")
        try:
            shutil.copytree(results_dir, storage_dir)
            logger.info(f"✓ Saved to persistent storage: {storage_dir}")
        except Exception as e:
            logger.error(f"✗ Failed to save to storage: {e}")

    # 8. Create a quick reference file
    with open(results_dir / "README.txt", "w") as f:
        f.write(f"Compositional Language Training Results\n")
        f.write(f"{'='*40}\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Output directories found: {len(output_dirs)}\n")
        f.write(f"Total files saved: {len(manifest['saved_files'])}\n")
        f.write(f"\nResults Summary:\n")

        if summary["results_found"]:
            for dir_name, results in summary["results_found"].items():
                f.write(f"\n{dir_name}:\n")
                for test_name, accuracy in results.get("test_results", {}).items():
                    f.write(f"  {test_name}: {accuracy:.2%}\n")
        else:
            f.write("\nNo training results found - training may not have completed.\n")

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("SAVE SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Results saved to: {results_dir}")
    logger.info(f"Output directories processed: {len(output_dirs)}")
    logger.info(f"Files saved: {len(manifest['saved_files'])}")

    if save_to_storage and os.path.exists("/storage"):
        logger.info(f"\nAlso saved to persistent storage!")

    print(f"\nTo download:")
    print(f"1. Zip the directory: zip -r {results_dir}.zip {results_dir}/")
    print(f"2. Download via Jupyter or use persistent storage")

    return results_dir


def check_latest_training():
    """Quick check to find the most recent training results"""
    logger.info("Checking for latest training results...")

    # Look for stage model files which indicate training progress
    stage_files = glob.glob("outputs/*/stage_*.h5")
    if stage_files:
        logger.info(f"Found {len(stage_files)} stage checkpoint files")

        # Group by directory
        by_dir = {}
        for f in stage_files:
            dir_name = os.path.dirname(f)
            if dir_name not in by_dir:
                by_dir[dir_name] = []
            by_dir[dir_name].append(f)

        # Show what we found
        for dir_name, files in by_dir.items():
            stages = sorted([int(os.path.basename(f).split("_")[1]) for f in files])
            logger.info(f"  {dir_name}: Completed stages {stages}")


if __name__ == "__main__":
    # First check what we have
    check_latest_training()

    # Then save everything
    save_critical_results()
