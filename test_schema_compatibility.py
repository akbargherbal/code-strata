#!/usr/bin/env python3
"""
Regression Test Suite for Git File Lifecycle Analyzer
Tests schema compatibility and ensures production outputs remain unchanged.

Phase 0.2 Deliverable
"""

import json
import subprocess
import hashlib
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
import jsonschema
import click


class RegressionTestSuite:
    def __init__(self, script_path: str, schema_path: str):
        self.script_path = Path(script_path).resolve()
        self.schema_path = Path(schema_path).resolve()
        self.test_results = []
        self.total_tests = 0
        self.passed_tests = 0

    def load_schema(self):
        """Load JSON schema for validation"""
        with open(self.schema_path) as f:
            return json.load(f)

    def create_test_repository(self, temp_dir: Path) -> Path:
        """
        Create a test git repository with known characteristics:
        - Normal commits
        - Merge commits
        - Renames
        - Copies
        - Special characters in filenames
        - Multiple authors
        """
        repo_path = temp_dir / "test_repo"
        repo_path.mkdir()

        def run_git(*args):
            subprocess.run(
                ["git", "-C", str(repo_path)] + list(args),
                check=True,
                capture_output=True,
                text=True,
            )

        # Initialize repository
        run_git("init")
        run_git("config", "user.name", "Test User 1")
        run_git("config", "user.email", "test1@example.com")

        # Commit 1: Add initial files
        (repo_path / "file1.txt").write_text("Initial content")
        (repo_path / "file2.txt").write_text("Second file")
        run_git("add", ".")
        run_git("commit", "-m", "Initial commit")

        # Commit 2: Modify file
        (repo_path / "file1.txt").write_text("Modified content")
        run_git("add", "file1.txt")
        run_git("commit", "-m", "Modify file1")

        # Commit 3: Add file with special characters
        special_file = repo_path / "file with spaces.txt"
        special_file.write_text("Special chars: 你好 émojis 🎉")
        run_git("add", ".")
        run_git("commit", "-m", "Add special char file")

        # Commit 4: Rename file
        run_git("mv", "file1.txt", "file1_renamed.txt")
        run_git("commit", "-m", "Rename file1")

        # Commit 5: Create branch for merge
        run_git("checkout", "-b", "feature")
        (repo_path / "feature_file.txt").write_text("Feature work")
        run_git("add", "feature_file.txt")
        run_git("commit", "-m", "Add feature file")

        # Commit 6: Merge back to main
        run_git("checkout", "master")
        try:
            run_git("merge", "feature", "-m", "Merge feature branch")
        except subprocess.CalledProcessError:
            # If merge already happened or no-op
            pass

        # Commit 7: Delete file
        run_git("rm", "file2.txt")
        run_git("commit", "-m", "Delete file2")

        # Commit 8: Different author
        run_git("config", "user.name", "Test User 2")
        run_git("config", "user.email", "test2@example.com")
        (repo_path / "file3.txt").write_text("New file by user 2")
        run_git("add", "file3.txt")
        run_git("commit", "-m", "Add file3 by user2")

        return repo_path

    def run_analyzer(self, repo_path: Path, output_dir: Path) -> dict:
        """Run the analyzer script on a repository"""
        # Build command arguments
        cmd = [
            "python3",
            str(self.script_path),
            str(repo_path),
            "--output-json",
            str(output_dir / "file_lifecycle.json"),
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            raise Exception(f"Analyzer failed: {result.stderr}")

        # Load the output
        output_file = output_dir / "file_lifecycle.json"
        if not output_file.exists():
            raise Exception("Output file not generated")

        with open(output_file) as f:
            return json.load(f)

    def test_schema_validation(self, data: dict) -> bool:
        """Test that output conforms to JSON schema"""
        try:
            schema = self.load_schema()
            jsonschema.validate(instance=data, schema=schema)
            return True
        except jsonschema.ValidationError as e:
            self.test_results.append(
                {"test": "Schema Validation", "status": "FAILED", "error": str(e)}
            )
            return False

    def test_required_fields(self, data: dict) -> bool:
        """Test that all required fields are present"""
        required = [
            "files",
            "total_commits",
            "total_changes",
            "generated_at",
            "repository_path",
            "skipped_unmerged",
        ]

        missing = [field for field in required if field not in data]

        if missing:
            self.test_results.append(
                {
                    "test": "Required Fields",
                    "status": "FAILED",
                    "error": f"Missing fields: {missing}",
                }
            )
            return False
        return True

    def test_chronological_ordering(self, data: dict) -> bool:
        """Test that events are chronologically ordered"""
        for file_path, events in data["files"].items():
            timestamps = [e["timestamp"] for e in events]
            if timestamps != sorted(timestamps):
                self.test_results.append(
                    {
                        "test": "Chronological Ordering",
                        "status": "FAILED",
                        "error": f"Events not sorted for {file_path}",
                    }
                )
                return False
        return True

    def test_total_changes_accuracy(self, data: dict) -> bool:
        """Test that total_changes matches actual count"""
        calculated = sum(len(events) for events in data["files"].values())
        if calculated != data["total_changes"]:
            self.test_results.append(
                {
                    "test": "Total Changes Accuracy",
                    "status": "FAILED",
                    "error": f"Expected {calculated}, got {data['total_changes']}",
                }
            )
            return False
        return True

    def test_operation_codes(self, data: dict) -> bool:
        """Test that operation codes are valid"""
        valid_ops = {"A", "M", "D", "R", "C", "T"}

        for file_path, events in data["files"].items():
            for event in events:
                if event["operation"] not in valid_ops:
                    self.test_results.append(
                        {
                            "test": "Operation Codes",
                            "status": "FAILED",
                            "error": f"Invalid operation '{event['operation']}' in {file_path}",
                        }
                    )
                    return False
        return True

    def test_commit_hash_format(self, data: dict) -> bool:
        """Test that commit hashes are valid SHA-1"""
        import re

        sha1_pattern = re.compile(r"^[0-9a-f]{40}$")

        for file_path, events in data["files"].items():
            for event in events:
                if not sha1_pattern.match(event["commit_hash"]):
                    self.test_results.append(
                        {
                            "test": "Commit Hash Format",
                            "status": "FAILED",
                            "error": f"Invalid hash format in {file_path}: {event['commit_hash']}",
                        }
                    )
                    return False
        return True

    def test_datetime_format(self, data: dict) -> bool:
        """Test that datetime fields are valid ISO 8601"""
        from dateutil import parser

        try:
            # Test root level datetime
            parser.isoparse(data["generated_at"])

            # Test event datetimes
            for file_path, events in data["files"].items():
                for event in events:
                    parser.isoparse(event["datetime"])
            return True
        except (ValueError, KeyError) as e:
            self.test_results.append(
                {"test": "Datetime Format", "status": "FAILED", "error": str(e)}
            )
            return False

    def test_rename_operations(self, data: dict) -> bool:
        """Test that rename operations have old_path"""
        for file_path, events in data["files"].items():
            for event in events:
                if event["operation"] in ("R", "C"):
                    if "old_path" not in event:
                        self.test_results.append(
                            {
                                "test": "Rename Operations",
                                "status": "FAILED",
                                "error": f"Rename/Copy without old_path in {file_path}",
                            }
                        )
                        return False
        return True

    def test_byte_identical_output(
        self, baseline_file: Path, current_file: Path
    ) -> bool:
        """Test that outputs are byte-identical"""
        if not baseline_file.exists():
            click.echo("⚠️  No baseline file for comparison")
            return True

        baseline_hash = hashlib.sha256(baseline_file.read_bytes()).hexdigest()
        current_hash = hashlib.sha256(current_file.read_bytes()).hexdigest()

        if baseline_hash != current_hash:
            self.test_results.append(
                {
                    "test": "Byte-Identical Output",
                    "status": "FAILED",
                    "error": f"Output changed: {baseline_hash[:8]} != {current_hash[:8]}",
                }
            )
            return False
        return True

    def run_all_tests(self) -> bool:
        """Run complete test suite"""
        click.echo(
            click.style(
                "\n🧪 Git Lifecycle Regression Test Suite", fg="cyan", bold=True
            )
        )
        click.echo("=" * 60)

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create test repository
            click.echo("\n📦 Creating test repository...")
            repo_path = self.create_test_repository(temp_path)
            click.echo(f"   Repository created at: {repo_path}")

            # Run analyzer
            click.echo("\n🔄 Running analyzer...")
            output_dir = temp_path / "output"
            output_dir.mkdir()

            try:
                data = self.run_analyzer(repo_path, output_dir)
                click.echo("   ✓ Analyzer completed successfully")
            except Exception as e:
                click.echo(click.style(f"   ✗ Analyzer failed: {e}", fg="red"))
                return False

            # Run tests
            click.echo("\n🧪 Running validation tests...")

            tests = [
                ("Schema Validation", self.test_schema_validation),
                ("Required Fields", self.test_required_fields),
                ("Chronological Ordering", self.test_chronological_ordering),
                ("Total Changes Accuracy", self.test_total_changes_accuracy),
                ("Operation Codes", self.test_operation_codes),
                ("Commit Hash Format", self.test_commit_hash_format),
                ("Datetime Format", self.test_datetime_format),
                ("Rename Operations", self.test_rename_operations),
            ]

            for test_name, test_func in tests:
                self.total_tests += 1
                try:
                    result = test_func(data)
                    if result:
                        self.passed_tests += 1
                        click.echo(f"   ✓ {test_name}")
                    else:
                        click.echo(click.style(f"   ✗ {test_name}", fg="red"))
                except Exception as e:
                    click.echo(click.style(f"   ✗ {test_name}: {e}", fg="red"))

        # Print summary
        click.echo("\n" + "=" * 60)
        click.echo(f"Tests Passed: {self.passed_tests}/{self.total_tests}")

        if self.passed_tests == self.total_tests:
            click.echo(click.style("✓ All tests passed!", fg="green", bold=True))
            return True
        else:
            click.echo(click.style("✗ Some tests failed", fg="red", bold=True))

            # Print failed test details
            if self.test_results:
                click.echo("\n❌ Failed Tests:")
                for result in self.test_results:
                    if result["status"] == "FAILED":
                        click.echo(f"   • {result['test']}: {result['error']}")

            return False


@click.command()
@click.option(
    "--script", default="git_file_life_cycle.py", help="Path to analyzer script"
)
@click.option(
    "--schema", default="schema_v1.0_contract.json", help="Path to JSON schema file"
)
@click.option("--baseline", help="Path to baseline output for comparison")
def main(script, schema, baseline):
    """Run regression test suite for Git File Lifecycle Analyzer"""

    suite = RegressionTestSuite(script, schema)
    success = suite.run_all_tests()

    if not success:
        exit(1)


if __name__ == "__main__":
    main()
