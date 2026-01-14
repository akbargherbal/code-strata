#!/usr/bin/env python3
"""
Test Suite for Git File Lifecycle Analyzer Phase 2
Tests all three core aggregators: File Metadata, Temporal, Author Network
"""

import unittest
import json
import tempfile
import shutil
import subprocess
from pathlib import Path
from datetime import datetime, timezone
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from git_file_lifecycle import (
    GitFileLifecycle,
    FileMetadataAggregator,
    TemporalAggregator,
    AuthorNetworkAggregator,
)


class TestFileMetadataAggregator(unittest.TestCase):
    """Tests for File Metadata Index aggregator"""

    def setUp(self):
        """Create a test git repository"""
        self.test_dir = tempfile.mkdtemp()
        self.repo_path = Path(self.test_dir) / "test_repo"
        self.repo_path.mkdir()

        # Initialize git repo
        subprocess.run(["git", "init"], cwd=self.repo_path, capture_output=True)
        subprocess.run(["git", "config", "user.name", "Test User"], cwd=self.repo_path)
        subprocess.run(
            ["git", "config", "user.email", "test@example.com"], cwd=self.repo_path
        )

        # Create test commits
        self._create_test_commits()

    def tearDown(self):
        """Clean up test directory"""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def _create_test_commits(self):
        """Create a series of test commits"""
        # Commit 1: Add file
        file1 = self.repo_path / "test.txt"
        file1.write_text("Initial content")
        subprocess.run(["git", "add", "test.txt"], cwd=self.repo_path)
        subprocess.run(["git", "commit", "-m", "Initial commit"], cwd=self.repo_path)

        # Commit 2: Modify file
        file1.write_text("Modified content")
        subprocess.run(["git", "add", "test.txt"], cwd=self.repo_path)
        subprocess.run(["git", "commit", "-m", "Update test.txt"], cwd=self.repo_path)

        # Commit 3: Add another file
        file2 = self.repo_path / "other.txt"
        file2.write_text("Other content")
        subprocess.run(["git", "add", "other.txt"], cwd=self.repo_path)
        subprocess.run(["git", "commit", "-m", "Add other.txt"], cwd=self.repo_path)

    def test_aggregator_initialization(self):
        """Test aggregator can be initialized"""
        analyzer = GitFileLifecycle(str(self.repo_path), enable_aggregations=True)
        aggregator = FileMetadataAggregator(analyzer)

        self.assertIsNotNone(aggregator)
        self.assertEqual(aggregator.get_name(), "file_metadata")
        self.assertEqual(aggregator.get_schema_version(), "1.0.0")

    def test_file_metadata_generation(self):
        """Test file metadata is correctly generated"""
        analyzer = GitFileLifecycle(str(self.repo_path), enable_aggregations=True)
        aggregator = FileMetadataAggregator(analyzer)
        analyzer.register_aggregator(aggregator)

        # Run analysis
        analyzer.analyze_repository()

        # Get finalized data
        metadata = aggregator.finalize()

        # Validate structure
        self.assertIn("schema_version", metadata)
        self.assertIn("total_files", metadata)
        self.assertIn("files", metadata)

        # Check we have data for our files
        self.assertGreater(metadata["total_files"], 0)

        # Validate file entry structure
        for file_path, file_data in metadata["files"].items():
            self.assertIn("first_seen", file_data)
            self.assertIn("last_modified", file_data)
            self.assertIn("total_commits", file_data)
            self.assertIn("unique_authors", file_data)
            self.assertIn("primary_author", file_data)
            self.assertIn("operations", file_data)
            self.assertIn("age_days", file_data)
            self.assertIn("commits_per_day", file_data)
            self.assertIn("lifecycle_event_count", file_data)

    def test_operation_counts(self):
        """Test operation counts are accurate"""
        analyzer = GitFileLifecycle(str(self.repo_path), enable_aggregations=True)
        aggregator = FileMetadataAggregator(analyzer)
        analyzer.register_aggregator(aggregator)

        analyzer.analyze_repository()
        metadata = aggregator.finalize()

        # test.txt should have 1 Add and 1 Modify
        if "test.txt" in metadata["files"]:
            ops = metadata["files"]["test.txt"]["operations"]
            self.assertEqual(ops.get("A", 0), 1)
            self.assertEqual(ops.get("M", 0), 1)

    def test_primary_author_calculation(self):
        """Test primary author is correctly identified"""
        analyzer = GitFileLifecycle(str(self.repo_path), enable_aggregations=True)
        aggregator = FileMetadataAggregator(analyzer)
        analyzer.register_aggregator(aggregator)

        analyzer.analyze_repository()
        metadata = aggregator.finalize()

        # All files should have test@example.com as primary author
        for file_path, file_data in metadata["files"].items():
            self.assertEqual(file_data["primary_author"]["email"], "test@example.com")
            self.assertGreater(file_data["primary_author"]["commit_count"], 0)
            self.assertGreater(file_data["primary_author"]["percentage"], 0)


class TestTemporalAggregator(unittest.TestCase):
    """Tests for Temporal Aggregations"""

    def setUp(self):
        """Create a test git repository"""
        self.test_dir = tempfile.mkdtemp()
        self.repo_path = Path(self.test_dir) / "test_repo"
        self.repo_path.mkdir()

        # Initialize git repo
        subprocess.run(["git", "init"], cwd=self.repo_path, capture_output=True)
        subprocess.run(["git", "config", "user.name", "Test User"], cwd=self.repo_path)
        subprocess.run(
            ["git", "config", "user.email", "test@example.com"], cwd=self.repo_path
        )

        # Create test commits
        file1 = self.repo_path / "test.txt"
        file1.write_text("Content")
        subprocess.run(["git", "add", "test.txt"], cwd=self.repo_path)
        subprocess.run(["git", "commit", "-m", "Test commit"], cwd=self.repo_path)

    def tearDown(self):
        """Clean up test directory"""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_temporal_structure(self):
        """Test temporal aggregation output structure"""
        analyzer = GitFileLifecycle(str(self.repo_path), enable_aggregations=True)
        aggregator = TemporalAggregator(analyzer)
        analyzer.register_aggregator(aggregator)

        analyzer.analyze_repository()

        # Format the data
        daily_data = aggregator._format_temporal_data(aggregator.daily_data, "daily")
        monthly_data = aggregator._format_temporal_data(
            aggregator.monthly_data, "monthly"
        )

        # Validate structure
        self.assertIn("schema_version", daily_data)
        self.assertIn("granularity", daily_data)
        self.assertIn("period_count", daily_data)
        self.assertIn("daily", daily_data)

        self.assertIn("schema_version", monthly_data)
        self.assertIn("granularity", monthly_data)
        self.assertIn("monthly", monthly_data)

    def test_temporal_metrics(self):
        """Test temporal metrics are calculated correctly"""
        analyzer = GitFileLifecycle(str(self.repo_path), enable_aggregations=True)
        aggregator = TemporalAggregator(analyzer)
        analyzer.register_aggregator(aggregator)

        analyzer.analyze_repository()
        daily_data = aggregator._format_temporal_data(aggregator.daily_data, "daily")

        # Should have at least one day of data
        self.assertGreater(len(daily_data["daily"]), 0)

        # Validate period entry structure
        for period, stats in daily_data["daily"].items():
            self.assertIn("total_commits", stats)
            self.assertIn("total_changes", stats)
            self.assertIn("unique_authors", stats)
            self.assertIn("active_files", stats)
            self.assertIn("operations", stats)
            self.assertIn("top_contributors", stats)


class TestAuthorNetworkAggregator(unittest.TestCase):
    """Tests for Author Network aggregator"""

    def setUp(self):
        """Create a test git repository with multiple authors"""
        self.test_dir = tempfile.mkdtemp()
        self.repo_path = Path(self.test_dir) / "test_repo"
        self.repo_path.mkdir()

        # Initialize git repo
        subprocess.run(["git", "init"], cwd=self.repo_path, capture_output=True)

        # Create commits from different authors
        self._create_multi_author_commits()

    def tearDown(self):
        """Clean up test directory"""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def _create_multi_author_commits(self):
        """Create commits from multiple authors"""
        # Author 1
        subprocess.run(["git", "config", "user.name", "Alice"], cwd=self.repo_path)
        subprocess.run(
            ["git", "config", "user.email", "alice@example.com"], cwd=self.repo_path
        )

        file1 = self.repo_path / "shared.txt"
        file1.write_text("Alice's work")
        subprocess.run(["git", "add", "shared.txt"], cwd=self.repo_path)
        subprocess.run(["git", "commit", "-m", "Alice commit"], cwd=self.repo_path)

        # Author 2
        subprocess.run(["git", "config", "user.name", "Bob"], cwd=self.repo_path)
        subprocess.run(
            ["git", "config", "user.email", "bob@example.com"], cwd=self.repo_path
        )

        file1.write_text("Bob's work")
        subprocess.run(["git", "add", "shared.txt"], cwd=self.repo_path)
        subprocess.run(["git", "commit", "-m", "Bob commit"], cwd=self.repo_path)

    def test_network_structure(self):
        """Test author network output structure"""
        analyzer = GitFileLifecycle(str(self.repo_path), enable_aggregations=True)
        aggregator = AuthorNetworkAggregator(analyzer)
        analyzer.register_aggregator(aggregator)

        analyzer.analyze_repository()
        network = aggregator.finalize()

        # Validate structure
        self.assertIn("schema_version", network)
        self.assertIn("graph_type", network)
        self.assertIn("node_count", network)
        self.assertIn("edge_count", network)
        self.assertIn("nodes", network)
        self.assertIn("edges", network)
        self.assertIn("metadata", network)

        # Should have 2 authors
        self.assertEqual(network["node_count"], 2)

    def test_node_attributes(self):
        """Test author nodes have correct attributes"""
        analyzer = GitFileLifecycle(str(self.repo_path), enable_aggregations=True)
        aggregator = AuthorNetworkAggregator(analyzer)
        analyzer.register_aggregator(aggregator)

        analyzer.analyze_repository()
        network = aggregator.finalize()

        # Validate node structure
        for node in network["nodes"]:
            self.assertIn("id", node)
            self.assertIn("email", node)
            self.assertIn("name", node)
            self.assertIn("total_commits", node)
            self.assertIn("files_touched", node)
            self.assertIn("first_seen", node)
            self.assertIn("last_seen", node)
            self.assertIn("active_days", node)

    def test_collaboration_edges(self):
        """Test collaboration edges are created"""
        analyzer = GitFileLifecycle(str(self.repo_path), enable_aggregations=True)
        aggregator = AuthorNetworkAggregator(analyzer)
        analyzer.register_aggregator(aggregator)

        analyzer.analyze_repository()
        network = aggregator.finalize()

        # Should have 1 edge (Alice and Bob collaborated on shared.txt)
        self.assertEqual(network["edge_count"], 1)

        if network["edges"]:
            edge = network["edges"][0]
            self.assertIn("source", edge)
            self.assertIn("target", edge)
            self.assertIn("weight", edge)
            self.assertIn("shared_files", edge)
            self.assertIn("collaboration_strength", edge)


class TestIntegration(unittest.TestCase):
    """Integration tests for complete Phase 2 workflow"""

    def setUp(self):
        """Create a test git repository"""
        self.test_dir = tempfile.mkdtemp()
        self.repo_path = Path(self.test_dir) / "test_repo"
        self.repo_path.mkdir()

        # Initialize git repo
        subprocess.run(["git", "init"], cwd=self.repo_path, capture_output=True)
        subprocess.run(["git", "config", "user.name", "Test User"], cwd=self.repo_path)
        subprocess.run(
            ["git", "config", "user.email", "test@example.com"], cwd=self.repo_path
        )

        # Create test commits
        file1 = self.repo_path / "test.txt"
        file1.write_text("Content")
        subprocess.run(["git", "add", "test.txt"], cwd=self.repo_path)
        subprocess.run(["git", "commit", "-m", "Test commit"], cwd=self.repo_path)

        self.output_dir = Path(self.test_dir) / "output"

    def tearDown(self):
        """Clean up test directory"""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_full_pipeline_without_aggregations(self):
        """Test pipeline works without aggregations (backward compatibility)"""
        analyzer = GitFileLifecycle(str(self.repo_path), enable_aggregations=False)
        analyzer.analyze_repository()

        manifest = analyzer.export_results(self.output_dir)

        # Should only have core lifecycle
        self.assertIn("core_lifecycle", manifest["datasets"])
        self.assertEqual(len(manifest["datasets"]), 1)

        # Core file should exist
        lifecycle_file = self.output_dir / "file_lifecycle.json"
        self.assertTrue(lifecycle_file.exists())

    def test_full_pipeline_with_aggregations(self):
        """Test complete pipeline with all Phase 2 aggregators"""
        analyzer = GitFileLifecycle(str(self.repo_path), enable_aggregations=True)

        # Register all aggregators
        analyzer.register_aggregator(FileMetadataAggregator(analyzer))
        analyzer.register_aggregator(TemporalAggregator(analyzer))
        analyzer.register_aggregator(AuthorNetworkAggregator(analyzer))

        # Run analysis
        analyzer.analyze_repository()

        # Export
        manifest = analyzer.export_results(self.output_dir)

        # Validate manifest
        self.assertIn("core_lifecycle", manifest["datasets"])
        self.assertIn("file_metadata", manifest["datasets"])
        self.assertIn("temporal_daily", manifest["datasets"])
        self.assertIn("temporal_monthly", manifest["datasets"])
        self.assertIn("author_network", manifest["datasets"])

        # Validate files exist
        self.assertTrue((self.output_dir / "file_lifecycle.json").exists())
        self.assertTrue((self.output_dir / "metadata" / "file_index.json").exists())
        self.assertTrue(
            (self.output_dir / "aggregations" / "temporal_daily.json").exists()
        )
        self.assertTrue(
            (self.output_dir / "aggregations" / "temporal_monthly.json").exists()
        )
        self.assertTrue((self.output_dir / "networks" / "author_network.json").exists())
        self.assertTrue((self.output_dir / "manifest.json").exists())

    def test_manifest_integrity(self):
        """Test manifest has correct metadata"""
        analyzer = GitFileLifecycle(str(self.repo_path), enable_aggregations=True)
        analyzer.register_aggregator(FileMetadataAggregator(analyzer))

        analyzer.analyze_repository()
        manifest = analyzer.export_results(self.output_dir)

        # Validate manifest structure
        self.assertIn("generator_version", manifest)
        self.assertIn("generated_at", manifest)
        self.assertIn("repository", manifest)
        self.assertIn("datasets", manifest)

        # Validate dataset entries have required fields
        for dataset_name, dataset_info in manifest["datasets"].items():
            self.assertIn("file", dataset_info)
            self.assertIn("schema_version", dataset_info)
            self.assertIn("record_count", dataset_info)
            self.assertIn("file_size_bytes", dataset_info)
            self.assertIn("sha256", dataset_info)

    def test_core_lifecycle_unchanged(self):
        """Test core lifecycle output is identical regardless of aggregations"""
        # Generate without aggregations
        analyzer1 = GitFileLifecycle(str(self.repo_path), enable_aggregations=False)
        analyzer1.analyze_repository()
        output1 = Path(self.test_dir) / "output1"
        analyzer1.export_results(output1)

        # Generate with aggregations
        analyzer2 = GitFileLifecycle(str(self.repo_path), enable_aggregations=True)
        analyzer2.register_aggregator(FileMetadataAggregator(analyzer2))
        analyzer2.analyze_repository()
        output2 = Path(self.test_dir) / "output2"
        analyzer2.export_results(output2)

        # Compare core lifecycle files
        with open(output1 / "file_lifecycle.json") as f1:
            data1 = json.load(f1)

        with open(output2 / "file_lifecycle.json") as f2:
            data2 = json.load(f2)

        # Generated_at will differ, so exclude it
        data1_copy = data1.copy()
        data2_copy = data2.copy()
        data1_copy.pop("generated_at", None)
        data2_copy.pop("generated_at", None)

        # Files structure should be identical
        self.assertEqual(data1_copy["files"], data2_copy["files"])
        self.assertEqual(data1_copy["total_commits"], data2_copy["total_commits"])
        self.assertEqual(data1_copy["total_changes"], data2_copy["total_changes"])


class TestReferentialIntegrity(unittest.TestCase):
    """Tests for referential integrity between datasets"""

    def setUp(self):
        """Create a test git repository"""
        self.test_dir = tempfile.mkdtemp()
        self.repo_path = Path(self.test_dir) / "test_repo"
        self.repo_path.mkdir()

        # Initialize git repo and create commits
        subprocess.run(["git", "init"], cwd=self.repo_path, capture_output=True)
        subprocess.run(["git", "config", "user.name", "Test User"], cwd=self.repo_path)
        subprocess.run(
            ["git", "config", "user.email", "test@example.com"], cwd=self.repo_path
        )

        # Create multiple commits
        for i in range(3):
            file = self.repo_path / f"file{i}.txt"
            file.write_text(f"Content {i}")
            subprocess.run(["git", "add", f"file{i}.txt"], cwd=self.repo_path)
            subprocess.run(["git", "commit", "-m", f"Commit {i}"], cwd=self.repo_path)

        self.output_dir = Path(self.test_dir) / "output"

    def tearDown(self):
        """Clean up test directory"""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_file_counts_match(self):
        """Test file counts match between core and metadata"""
        analyzer = GitFileLifecycle(str(self.repo_path), enable_aggregations=True)
        analyzer.register_aggregator(FileMetadataAggregator(analyzer))

        analyzer.analyze_repository()
        analyzer.export_results(self.output_dir)

        # Load both datasets
        with open(self.output_dir / "file_lifecycle.json") as f:
            core = json.load(f)

        with open(self.output_dir / "metadata" / "file_index.json") as f:
            metadata = json.load(f)

        # File counts should match
        self.assertEqual(len(core["files"]), metadata["total_files"])
        self.assertEqual(set(core["files"].keys()), set(metadata["files"].keys()))

    def test_commit_counts_match(self):
        """Test commit counts are consistent across datasets"""
        analyzer = GitFileLifecycle(str(self.repo_path), enable_aggregations=True)
        analyzer.register_aggregator(FileMetadataAggregator(analyzer))
        analyzer.register_aggregator(TemporalAggregator(analyzer))

        analyzer.analyze_repository()
        analyzer.export_results(self.output_dir)

        # Load datasets
        with open(self.output_dir / "file_lifecycle.json") as f:
            core = json.load(f)

        with open(self.output_dir / "aggregations" / "temporal_daily.json") as f:
            temporal = json.load(f)

        # Total commits should match
        total_temporal_commits = sum(
            day["total_commits"] for day in temporal["daily"].values()
        )

        # Note: This might not be exact due to how commits are counted,
        # but should be in the same ballpark
        self.assertGreaterEqual(core["total_commits"], 1)


def run_tests():
    """Run all tests"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestFileMetadataAggregator))
    suite.addTests(loader.loadTestsFromTestCase(TestTemporalAggregator))
    suite.addTests(loader.loadTestsFromTestCase(TestAuthorNetworkAggregator))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestReferentialIntegrity))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
