#!/usr/bin/env python3
"""Download the real ARC-AGI dataset from GitHub.

This downloads the actual 800+ task dataset from the official repository.
"""

from utils.imports import setup_project_paths

setup_project_paths()

import json
import subprocess
from pathlib import Path
from typing import Dict

import requests


class RealARCDownloader:
    """Downloads the official ARC-AGI dataset."""

    def __init__(self, data_dir: str = "data/arc_agi_official"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # GitHub API endpoints
        self.repo_owner = "fchollet"
        self.repo_name = "ARC-AGI"
        self.api_base = (
            f"https://api.github.com/repos/{self.repo_owner}/{self.repo_name}"
        )
        self.raw_base = f"https://raw.githubusercontent.com/{self.repo_owner}/{self.repo_name}/master"

    def download_via_git(self) -> bool:
        """Try to clone the repository using git."""
        repo_path = self.data_dir / "ARC-AGI"

        if repo_path.exists():
            print(f"Repository already exists at {repo_path}")
            # Try to pull latest
            try:
                subprocess.run(
                    ["git", "pull"], cwd=repo_path, check=True, capture_output=True
                )
                print("Updated to latest version")
                return True
            except subprocess.CalledProcessError:
                print("Could not update repository")
                return True  # Still exists

        print("Cloning ARC-AGI repository...")
        try:
            subprocess.run(
                [
                    "git",
                    "clone",
                    f"https://github.com/{self.repo_owner}/{self.repo_name}.git",
                ],
                cwd=self.data_dir,
                check=True,
                capture_output=True,
            )
            print("Successfully cloned repository")
            return True
        except subprocess.CalledProcessError as e:
            print(f"Git clone failed: {e}")
            return False

    def download_via_api(self) -> bool:
        """Download files using GitHub API."""
        print("Downloading via GitHub API...")

        datasets = ["training", "evaluation"]

        for dataset in datasets:
            dataset_path = self.data_dir / dataset
            dataset_path.mkdir(exist_ok=True)

            # Get file list from GitHub API
            api_url = f"{self.api_base}/contents/data/{dataset}"

            try:
                response = requests.get(api_url)
                response.raise_for_status()
                files = response.json()

                print(f"\nDownloading {dataset} dataset ({len(files)} files)...")

                # Download each file
                for i, file_info in enumerate(files):
                    if file_info["name"].endswith(".json"):
                        file_path = dataset_path / file_info["name"]

                        # Skip if already exists
                        if file_path.exists() and file_path.stat().st_size > 0:
                            continue

                        # Download file
                        file_url = file_info["download_url"]
                        file_response = requests.get(file_url)
                        file_response.raise_for_status()

                        with open(file_path, "w") as f:
                            f.write(file_response.text)

                        if (i + 1) % 50 == 0:
                            print(f"  Downloaded {i + 1}/{len(files)} files...")

                print(f"  Completed {dataset} dataset")

            except requests.exceptions.RequestException as e:
                print(f"  Error downloading {dataset}: {e}")
                return False

        return True

    def download_compact_version(self) -> bool:
        """Download the compact JSON versions (all tasks in one file)."""
        print("Downloading compact dataset files...")

        files_to_download = [
            (
                "data/training/arc-agi_training_challenges.json",
                "training_challenges.json",
            ),
            (
                "data/training/arc-agi_training_solutions.json",
                "training_solutions.json",
            ),
            (
                "data/evaluation/arc-agi_evaluation_challenges.json",
                "evaluation_challenges.json",
            ),
            (
                "data/evaluation/arc-agi_evaluation_solutions.json",
                "evaluation_solutions.json",
            ),
        ]

        for remote_path, local_name in files_to_download:
            local_path = self.data_dir / local_name

            if local_path.exists() and local_path.stat().st_size > 1000:
                print(f"  {local_name} already exists")
                continue

            url = f"{self.raw_base}/{remote_path}"
            print(f"  Downloading {local_name}...")

            try:
                response = requests.get(url)
                response.raise_for_status()

                with open(local_path, "w") as f:
                    f.write(response.text)

                # Verify it's valid JSON
                with open(local_path) as f:
                    data = json.load(f)
                    print(f"    Successfully downloaded {len(data)} tasks")

            except Exception as e:
                print(f"    Error: {e}")
                return False

        return True

    def process_compact_to_individual(self) -> None:
        """Convert compact JSON files to individual task files."""
        print("\nProcessing compact files to individual tasks...")

        # Process training
        challenges_file = self.data_dir / "training_challenges.json"
        solutions_file = self.data_dir / "training_solutions.json"

        if challenges_file.exists() and solutions_file.exists():
            output_dir = self.data_dir / "training"
            output_dir.mkdir(exist_ok=True)

            with open(challenges_file) as f:
                challenges = json.load(f)
            with open(solutions_file) as f:
                solutions = json.load(f)

            for task_id, challenge in challenges.items():
                task_data = challenge.copy()

                # Add solutions to test cases
                if task_id in solutions:
                    for i, test_case in enumerate(task_data.get("test", [])):
                        if i < len(solutions[task_id]):
                            test_case["output"] = solutions[task_id][i]

                task_file = output_dir / f"{task_id}.json"
                with open(task_file, "w") as f:
                    json.dump(task_data, f, indent=2)

            print(f"  Created {len(challenges)} training tasks")

        # Process evaluation similarly
        challenges_file = self.data_dir / "evaluation_challenges.json"
        solutions_file = self.data_dir / "evaluation_solutions.json"

        if challenges_file.exists() and solutions_file.exists():
            output_dir = self.data_dir / "evaluation"
            output_dir.mkdir(exist_ok=True)

            with open(challenges_file) as f:
                challenges = json.load(f)
            with open(solutions_file) as f:
                solutions = json.load(f)

            for task_id, challenge in challenges.items():
                task_data = challenge.copy()

                if task_id in solutions:
                    for i, test_case in enumerate(task_data.get("test", [])):
                        if i < len(solutions[task_id]):
                            test_case["output"] = solutions[task_id][i]

                task_file = output_dir / f"{task_id}.json"
                with open(task_file, "w") as f:
                    json.dump(task_data, f, indent=2)

            print(f"  Created {len(challenges)} evaluation tasks")

    def verify_dataset(self) -> Dict:
        """Verify downloaded dataset and return statistics."""
        stats = {}

        for dataset in ["training", "evaluation"]:
            dataset_path = self.data_dir / dataset
            if dataset_path.exists():
                task_files = list(dataset_path.glob("*.json"))
                stats[dataset] = len(task_files)
            else:
                stats[dataset] = 0

        # Check if we have the git repo
        repo_path = self.data_dir / "ARC-AGI"
        if repo_path.exists():
            git_training = list((repo_path / "data" / "training").glob("*.json"))
            git_evaluation = list((repo_path / "data" / "evaluation").glob("*.json"))
            stats["git_training"] = len(git_training)
            stats["git_evaluation"] = len(git_evaluation)

        return stats


def main():
    """Download and setup the real ARC-AGI dataset."""
    print("=" * 70)
    print("REAL ARC-AGI DATASET DOWNLOADER")
    print("=" * 70)

    downloader = RealARCDownloader()

    # Try different download methods
    success = False

    # Method 1: Try git clone
    print("\nMethod 1: Git clone...")
    if downloader.download_via_git():
        success = True

    # Method 2: Try compact download
    if not success:
        print("\nMethod 2: Compact JSON files...")
        if downloader.download_compact_version():
            downloader.process_compact_to_individual()
            success = True

    # Method 3: Try API download (slower)
    if not success:
        print("\nMethod 3: GitHub API...")
        success = downloader.download_via_api()

    # Verify what we have
    print("\n" + "=" * 70)
    print("VERIFICATION")
    print("=" * 70)

    stats = downloader.verify_dataset()

    print("\nDataset statistics:")
    for key, count in stats.items():
        print(f"  {key}: {count} tasks")

    if stats.get("training", 0) > 0 or stats.get("git_training", 0) > 0:
        print("\n✅ Successfully downloaded ARC-AGI dataset!")
        print("\nNext steps:")
        print("1. Run evaluation on real ARC tasks")
        print("2. Compare explicit vs neural approaches")
        print("3. Test TTA effectiveness on complex tasks")
    else:
        print("\n⚠️ Dataset download incomplete")
        print("Please check your internet connection and try again")
        print("Or manually download from: https://github.com/fchollet/ARC-AGI")

    print("=" * 70)


if __name__ == "__main__":
    main()
