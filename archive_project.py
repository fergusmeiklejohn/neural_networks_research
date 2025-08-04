#!/usr/bin/env python3
"""
Safe project archiving script for neural_networks_research

This script safely archives old experiments and failed approaches while
preserving git history and ensuring no data loss.

Usage:
    python archive_project.py --dry-run  # See what will be archived
    python archive_project.py            # Actually perform archiving
"""

import argparse
import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List


class ProjectArchiver:
    def __init__(self, project_root: str, dry_run: bool = True):
        self.project_root = Path(project_root)
        self.dry_run = dry_run
        self.archive_root = self.project_root / "archive"
        self.archive_log = []
        self.errors = []

    def get_files_to_archive(self) -> Dict[str, List[str]]:
        """Define files to archive organized by category"""

        physics_exp = "experiments/01_physics_worlds"
        lang_exp = "experiments/02_compositional_language"

        return {
            "pre_ood_illusion_discovery/pinn_experiments": [
                f"{physics_exp}/train_pinn_extractor.py",
                f"{physics_exp}/train_pinn_keras.py",
                f"{physics_exp}/train_pinn_keras3.py",
                f"{physics_exp}/train_pinn_paperspace.py",
                f"{physics_exp}/train_pinn_scaled.py",
                f"{physics_exp}/train_pinn_simple.py",
                f"{physics_exp}/train_pinn_simple_fit.py",
                f"{physics_exp}/train_pinn_tensorflow.py",
                f"{physics_exp}/test_pinn_simple.py",
                f"{physics_exp}/test_pinn_training.py",
                f"{physics_exp}/test_pinn_results.py",
                f"{physics_exp}/analyze_pinn_failure.py",
                f"{physics_exp}/train_minimal_pinn.py",
                f"{physics_exp}/test_minimal_pinn_debug.py",
                f"{physics_exp}/PINN_FAILURE_ANALYSIS.md",
                f"{physics_exp}/PINN_LESSONS_LEARNED.md",
                f"{physics_exp}/MINIMAL_PINN_ANALYSIS.md",
                f"{physics_exp}/MINIMAL_PINN_RESULTS.md",
            ],
            "failed_approaches/tta_experiments": [
                f"{physics_exp}/implement_jax_tta.py",
                f"{physics_exp}/test_jax_tta.py",
                f"{physics_exp}/test_jax_tta_simple.py",
                f"{physics_exp}/test_jax_tta_v2.py",
                f"{physics_exp}/test_tta_baselines.py",
                f"{physics_exp}/test_tta_demo.py",
                f"{physics_exp}/test_tta_fixed.py",
                f"{physics_exp}/test_tta_prediction_issue.py",
                f"{physics_exp}/test_tta_simple.py",
                f"{physics_exp}/test_tta_simple_debug.py",
                f"{physics_exp}/test_tta_simple_evaluation.py",
                f"{physics_exp}/test_tta_weight_fix.py",
                f"{physics_exp}/evaluate_tta.py",
                f"{physics_exp}/evaluate_tta_comprehensive.py",
                f"{physics_exp}/evaluate_tta_multistep.py",
                f"{physics_exp}/evaluate_tta_on_true_ood.py",
                f"{physics_exp}/evaluate_tta_simple.py",
                f"{physics_exp}/debug_tta_adaptation.py",
                f"{physics_exp}/debug_tta_convergence.py",
                f"{physics_exp}/diagnose_tta_zeros.py",
                f"{physics_exp}/analyze_tta_degradation.py",
                f"{physics_exp}/tune_tta_hyperparameters.py",
                f"{physics_exp}/tune_tta_hyperparameters_v2.py",
                f"{physics_exp}/tta_hyperparameter_search.py",
                f"{physics_exp}/minimal_tta_v2_test.py",
                f"{physics_exp}/quick_tta_v2_tuning.py",
                f"{physics_exp}/JAX_TTA_FIX_SUMMARY.md",
                f"{physics_exp}/TTA_*.md",
            ],
            "failed_approaches/distribution_modifiers": [
                f"{physics_exp}/train_modifier.py",
                f"{physics_exp}/train_modifier_final.py",
                f"{physics_exp}/train_modifier_keras3.py",
                f"{physics_exp}/train_modifier_minimal.py",
                f"{physics_exp}/train_modifier_numpy.py",
                f"{physics_exp}/train_modifier_simple.py",
                f"{physics_exp}/train_modifier_v2.py",
                f"{physics_exp}/train_distribution_modifier.py",
                f"{physics_exp}/distribution_modifier.py",
                f"{physics_exp}/evaluate_modifier.py",
            ],
            "failed_approaches/training_variants": [
                f"{physics_exp}/train_trajectory_generator.py",
                f"{physics_exp}/train_rule_extractor.py",
                f"{physics_exp}/train_progressive_curriculum.py",
                f"{physics_exp}/train_progressive_paperspace.py",
                f"{physics_exp}/simple_rule_trainer.py",
                f"{physics_exp}/improved_rule_trainer.py",
                f"{physics_exp}/normalized_rule_trainer.py",
            ],
            "old_documentation/status_files": [
                f"{physics_exp}/READY_TO_TRAIN.md",
                f"{physics_exp}/NEXT_STEPS_TENSORFLOW_TRAINING.md",
                f"{physics_exp}/baseline_training_progress.md",
                f"{physics_exp}/minimal_pinn_training_status.md",
                f"{physics_exp}/jax_tta_implementation_summary.md",
                f"{physics_exp}/IMPLEMENTATION_SUMMARY.md",
                f"{physics_exp}/DATA_ISOLATION_FIX_PLAN.md",
                "Temp-test-results/TTA Test Results (19th July 2025).md",
            ],
            "old_documentation/experiment_summaries": [
                f"{physics_exp}/results/progressive_curriculum_results_20250628.md",
                f"{physics_exp}/outputs/pinn_implementation_summary.md",
                f"{physics_exp}/outputs/distribution_modifier_summary.md",
            ],
            "relocated_documents": [
                # Note: SCIENTIFIC_WRITING_NOTES.md already moved to root as SCIENTIFIC_WRITING_GUIDE.md
                "papers/ood_illusion/SCIENTIFIC_WRITING_NOTES.md",  # Remove old location
            ],
            "pre_ood_illusion_discovery/complex_architectures": [
                f"{lang_exp}/models_v2_broken.py",
                f"{lang_exp}/paperspace_comprehensive_experiments_broken.py",
                f"{lang_exp}/train_progressive_fixed.py",
                f"{lang_exp}/train_progressive_minimal.py",
                f"{lang_exp}/train_progressive_nomixedprecision.py",
                f"{lang_exp}/train_progressive_optimized.py",
                f"{lang_exp}/train_progressive_simple.py",
            ],
        }

    def calculate_checksum(self, filepath: Path) -> str:
        """Calculate MD5 checksum of file"""
        if not filepath.exists():
            return ""

        hash_md5 = hashlib.md5()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def create_archive_structure(self):
        """Create archive directory structure"""
        if self.dry_run:
            print("[DRY RUN] Would create archive structure")
            return

        directories = [
            "archive",
            "archive/pre_ood_illusion_discovery/pinn_experiments",
            "archive/pre_ood_illusion_discovery/progressive_curriculum",
            "archive/pre_ood_illusion_discovery/complex_architectures",
            "archive/pre_ood_illusion_discovery/early_experiments",
            "archive/failed_approaches/tta_experiments",
            "archive/failed_approaches/distribution_modifiers",
            "archive/failed_approaches/training_variants",
            "archive/old_documentation/status_files",
            "archive/old_documentation/implementation_plans",
            "archive/old_documentation/experiment_summaries",
            "archive/relocated_documents",
            "archive/analysis_outputs/failed_model_outputs",
            "archive/analysis_outputs/intermediate_results",
        ]

        for dir_path in directories:
            full_path = self.project_root / dir_path
            full_path.mkdir(parents=True, exist_ok=True)

        # Create README in archive
        readme_content = f"""# Archive Directory

Created: {datetime.now().strftime('%Y-%m-%d')}

This directory contains archived experiments and documentation from before our OOD Illusion discovery.
These files are preserved for historical reference but are no longer part of active development.

## Structure

- `pre_ood_illusion_discovery/`: Early experiments before key insights
- `failed_approaches/`: Approaches that didn't work (TTA, complex architectures)
- `old_documentation/`: Superseded documentation and plans
- `analysis_outputs/`: Results from failed experiments

## Note

All files here have been verified with checksums before archiving. See ARCHIVE_LOG.json for details.
"""

        if not self.dry_run:
            readme_path = self.archive_root / "README.md"
            with open(readme_path, "w") as f:
                f.write(readme_content)

    def archive_file(self, source: str, dest_category: str) -> bool:
        """Archive a single file with verification"""
        source_path = self.project_root / source

        # Handle wildcards
        if "*" in source:
            # Simple wildcard handling for patterns like TTA_*.md
            pattern = source.split("/")[-1]
            parent_dir = source_path.parent
            if parent_dir.exists():
                for file_path in parent_dir.glob(pattern):
                    relative_path = file_path.relative_to(self.project_root)
                    self.archive_file(str(relative_path), dest_category)
            return True

        if not source_path.exists():
            self.errors.append(f"Source not found: {source}")
            return False

        dest_path = self.archive_root / dest_category / source_path.name

        if self.dry_run:
            print(
                f"[DRY RUN] Would archive: {source} -> {dest_category}/{source_path.name}"
            )
            return True

        try:
            # Calculate source checksum
            source_checksum = self.calculate_checksum(source_path)

            # Copy file
            shutil.copy2(source_path, dest_path)

            # Verify copy
            dest_checksum = self.calculate_checksum(dest_path)

            if source_checksum == dest_checksum:
                self.archive_log.append(
                    {
                        "source": str(source),
                        "destination": str(dest_path.relative_to(self.project_root)),
                        "checksum": source_checksum,
                        "timestamp": datetime.now().isoformat(),
                        "size": source_path.stat().st_size,
                    }
                )
                print(f"✓ Archived: {source}")
                return True
            else:
                self.errors.append(f"Checksum mismatch: {source}")
                return False

        except Exception as e:
            self.errors.append(f"Error archiving {source}: {str(e)}")
            return False

    def save_log(self):
        """Save archive log for recovery"""
        if self.dry_run:
            print("[DRY RUN] Would save archive log")
            return

        log_path = self.archive_root / "ARCHIVE_LOG.json"

        summary = {
            "archive_date": datetime.now().isoformat(),
            "total_files": len(self.archive_log),
            "total_size": sum(entry["size"] for entry in self.archive_log),
            "errors": self.errors,
            "files": self.archive_log,
        }

        with open(log_path, "w") as f:
            json.dump(summary, f, indent=2)

        print(f"\n✓ Archive log saved to {log_path}")
        print(f"  Total files archived: {len(self.archive_log)}")
        print(f"  Total size: {summary['total_size'] / 1024 / 1024:.2f} MB")
        if self.errors:
            print(f"  Errors encountered: {len(self.errors)}")

    def remove_archived_files(self):
        """Remove successfully archived files from original location"""
        if self.dry_run:
            print("\n[DRY RUN] Would remove original files after verification")
            return

        print("\n⚠️  Ready to remove original files.")
        response = input("Are you sure? (yes/no): ")

        if response.lower() != "yes":
            print("Skipping file removal. Original files preserved.")
            return

        removed_count = 0
        for entry in self.archive_log:
            source_path = self.project_root / entry["source"]
            if source_path.exists():
                source_path.unlink()
                removed_count += 1

        print(f"✓ Removed {removed_count} archived files from original locations")

    def generate_summary_report(self):
        """Generate a summary of what will be/was archived"""
        files_to_archive = self.get_files_to_archive()

        print("\n" + "=" * 60)
        print("ARCHIVE SUMMARY")
        print("=" * 60)

        total_files = 0
        for category, files in files_to_archive.items():
            # Count actual files (handle wildcards)
            file_count = 0
            for file_pattern in files:
                if "*" in file_pattern:
                    # Estimate wildcard matches
                    file_count += 5  # Conservative estimate
                else:
                    file_count += 1

            print(f"\n{category}:")
            print(f"  Files to archive: ~{file_count}")
            total_files += file_count

        print(f"\nTotal files to archive: ~{total_files}")
        print(f"Dry run mode: {self.dry_run}")

    def run(self):
        """Execute the archiving process"""
        print(f"Project Archiver - {'DRY RUN' if self.dry_run else 'LIVE'}")
        print(f"Project root: {self.project_root}")

        # Show summary
        self.generate_summary_report()

        if not self.dry_run:
            response = input("\nProceed with archiving? (yes/no): ")
            if response.lower() != "yes":
                print("Archiving cancelled.")
                return

        # Create structure
        self.create_archive_structure()

        # Archive files
        files_to_archive = self.get_files_to_archive()

        print("\nArchiving files...")
        for category, files in files_to_archive.items():
            print(f"\n{category}:")
            for file_path in files:
                self.archive_file(file_path, category)

        # Save log
        self.save_log()

        # Report errors
        if self.errors:
            print("\n⚠️  Errors encountered:")
            for error in self.errors:
                print(f"  - {error}")

        # Offer to remove originals
        if not self.dry_run and self.archive_log:
            self.remove_archived_files()

        print("\n✓ Archiving complete!")


def main():
    parser = argparse.ArgumentParser(description="Archive old project files safely")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be archived without actually doing it",
    )
    parser.add_argument(
        "--project-root",
        type=str,
        default="/Users/fergusmeiklejohn/conductor/repo/neural_networks_research/sydney",
        help="Project root directory",
    )

    args = parser.parse_args()

    archiver = ProjectArchiver(args.project_root, dry_run=args.dry_run)
    archiver.run()


if __name__ == "__main__":
    main()
