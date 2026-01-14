#!/usr/bin/env python3
"""
Test Suite for Phase 3: Advanced Datasets
Tests co-change network, file hierarchy, and milestone aggregators
"""

import unittest
import json
import tempfile
import shutil
from pathlib import Path
from datetime import datetime, timezone
import subprocess
import sys

# Import the Phase 3 analyzer
try:
    from git_file_lifecycle import (
        GitFileLifecycle,
        CochangeNetworkAggregator,
        FileHierarchyAggregator,
        MilestoneAggregator,
        FileMetadataAggregator,
        TemporalAggregator,
        AuthorNetworkAggregator
    )
except ImportError as e:
    print(f"Error: {e}")
    print("Make sure git_file_lifecycle_v2_phase3.py is in the same directory")
    sys.exit(1)


class TestCochangeNetworkAggregator(unittest.TestCase):
    """Test Co-change Network Aggregator"""

    def setUp(self):
        """Create a test repository"""
        self.test_dir = tempfile.mkdtemp()
        self.repo_path = Path(self.test_dir) / 'test_repo'
        self.repo_path.mkdir()
        
        # Initialize git repo
        subprocess.run(['git', 'init'], cwd=self.repo_path, check=True, capture_output=True)
        subprocess.run(['git', 'config', 'user.name', 'Test User'], cwd=self.repo_path, check=True, capture_output=True)
        subprocess.run(['git', 'config', 'user.email', 'test@example.com'], cwd=self.repo_path, check=True, capture_output=True)

    def tearDown(self):
        """Clean up test repository"""
        shutil.rmtree(self.test_dir)

    def test_cochange_network_initialization(self):
        """Test aggregator initialization"""
        analyzer = GitFileLifecycle(str(self.repo_path))
        aggregator = CochangeNetworkAggregator(analyzer)
        
        self.assertEqual(aggregator.get_name(), "cochange_network")
        self.assertEqual(aggregator.get_schema_version(), "1.0.0")

    def test_cochange_detection(self):
        """Test detection of files that change together"""
        # Create test files
        (self.repo_path / 'file1.txt').write_text('content1')
        (self.repo_path / 'file2.txt').write_text('content2')
        (self.repo_path / 'file3.txt').write_text('content3')
        
        # Commit file1 and file2 together
        subprocess.run(['git', 'add', 'file1.txt', 'file2.txt'], cwd=self.repo_path, check=True, capture_output=True)
        subprocess.run(['git', 'commit', '-m', 'Add files together'], cwd=self.repo_path, check=True, capture_output=True)
        
        # Commit file1 and file2 together again
        (self.repo_path / 'file1.txt').write_text('updated1')
        (self.repo_path / 'file2.txt').write_text('updated2')
        subprocess.run(['git', 'add', 'file1.txt', 'file2.txt'], cwd=self.repo_path, check=True, capture_output=True)
        subprocess.run(['git', 'commit', '-m', 'Update files together'], cwd=self.repo_path, check=True, capture_output=True)
        
        # Commit file3 separately
        subprocess.run(['git', 'add', 'file3.txt'], cwd=self.repo_path, check=True, capture_output=True)
        subprocess.run(['git', 'commit', '-m', 'Add file3'], cwd=self.repo_path, check=True, capture_output=True)
        
        # Analyze
        analyzer = GitFileLifecycle(str(self.repo_path), enable_aggregations=True)
        aggregator = CochangeNetworkAggregator(analyzer, min_cochanges=2)
        analyzer.register_aggregator(aggregator)
        analyzer.analyze_repository()
        
        data = aggregator.finalize()
        
        # Verify structure
        self.assertIn('nodes', data)
        self.assertIn('edges', data)
        self.assertIn('network_type', data)
        self.assertEqual(data['network_type'], 'file_coupling')
        
        # Verify file1 and file2 have a co-change edge
        edges = data['edges']
        file1_file2_edge = None
        for edge in edges:
            source_node = data['nodes'][edge['source']]
            target_node = data['nodes'][edge['target']]
            if (source_node['file_path'] == 'file1.txt' and target_node['file_path'] == 'file2.txt') or \
               (source_node['file_path'] == 'file2.txt' and target_node['file_path'] == 'file1.txt'):
                file1_file2_edge = edge
                break
        
        self.assertIsNotNone(file1_file2_edge, "file1.txt and file2.txt should have a co-change edge")
        self.assertGreaterEqual(file1_file2_edge['co_change_count'], 2)

    def test_coupling_strength_calculation(self):
        """Test coupling strength metric"""
        # Create files with different co-change patterns
        (self.repo_path / 'main.py').write_text('main')
        (self.repo_path / 'helper.py').write_text('helper')
        
        # Commit together multiple times
        for i in range(5):
            (self.repo_path / 'main.py').write_text(f'main{i}')
            (self.repo_path / 'helper.py').write_text(f'helper{i}')
            subprocess.run(['git', 'add', '.'], cwd=self.repo_path, check=True, capture_output=True)
            subprocess.run(['git', 'commit', '-m', f'Update {i}'], cwd=self.repo_path, check=True, capture_output=True)
        
        # Analyze
        analyzer = GitFileLifecycle(str(self.repo_path), enable_aggregations=True)
        aggregator = CochangeNetworkAggregator(analyzer)
        analyzer.register_aggregator(aggregator)
        analyzer.analyze_repository()
        
        data = aggregator.finalize()
        
        # Check that coupling strength is calculated
        self.assertTrue(len(data['edges']) > 0)
        for edge in data['edges']:
            self.assertIn('coupling_strength', edge)
            self.assertGreater(edge['coupling_strength'], 0)
            self.assertLessEqual(edge['coupling_strength'], 1.0)


class TestFileHierarchyAggregator(unittest.TestCase):
    """Test File Hierarchy Aggregator"""

    def setUp(self):
        """Create a test repository with directory structure"""
        self.test_dir = tempfile.mkdtemp()
        self.repo_path = Path(self.test_dir) / 'test_repo'
        self.repo_path.mkdir()
        
        # Initialize git repo
        subprocess.run(['git', 'init'], cwd=self.repo_path, check=True, capture_output=True)
        subprocess.run(['git', 'config', 'user.name', 'Test User'], cwd=self.repo_path, check=True, capture_output=True)
        subprocess.run(['git', 'config', 'user.email', 'test@example.com'], cwd=self.repo_path, check=True, capture_output=True)

    def tearDown(self):
        """Clean up test repository"""
        shutil.rmtree(self.test_dir)

    def test_hierarchy_initialization(self):
        """Test aggregator initialization"""
        analyzer = GitFileLifecycle(str(self.repo_path))
        aggregator = FileHierarchyAggregator(analyzer)
        
        self.assertEqual(aggregator.get_name(), "file_hierarchy")
        self.assertEqual(aggregator.get_schema_version(), "1.0.0")

    def test_directory_statistics(self):
        """Test directory-level statistics"""
        # Create directory structure
        (self.repo_path / 'src').mkdir()
        (self.repo_path / 'src' / 'main.py').write_text('main')
        (self.repo_path / 'src' / 'utils.py').write_text('utils')
        (self.repo_path / 'tests').mkdir()
        (self.repo_path / 'tests' / 'test_main.py').write_text('test')
        
        subprocess.run(['git', 'add', '.'], cwd=self.repo_path, check=True, capture_output=True)
        subprocess.run(['git', 'commit', '-m', 'Initial structure'], cwd=self.repo_path, check=True, capture_output=True)
        
        # Modify files
        (self.repo_path / 'src' / 'main.py').write_text('updated main')
        subprocess.run(['git', 'add', '.'], cwd=self.repo_path, check=True, capture_output=True)
        subprocess.run(['git', 'commit', '-m', 'Update main'], cwd=self.repo_path, check=True, capture_output=True)
        
        # Analyze
        analyzer = GitFileLifecycle(str(self.repo_path), enable_aggregations=True)
        aggregator = FileHierarchyAggregator(analyzer)
        analyzer.register_aggregator(aggregator)
        analyzer.analyze_repository()
        
        data = aggregator.finalize()
        
        # Verify structure
        self.assertIn('directories', data)
        self.assertIn('total_directories', data)
        
        # Verify src directory exists
        self.assertIn('src', data['directories'])
        src_stats = data['directories']['src']
        self.assertEqual(src_stats['total_files'], 2)
        self.assertGreaterEqual(src_stats['total_commits'], 1)

    def test_nested_directory_aggregation(self):
        """Test that parent directories aggregate child statistics"""
        # Create nested structure
        (self.repo_path / 'src').mkdir()
        (self.repo_path / 'src' / 'core').mkdir()
        (self.repo_path / 'src' / 'core' / 'engine.py').write_text('engine')
        
        subprocess.run(['git', 'add', '.'], cwd=self.repo_path, check=True, capture_output=True)
        subprocess.run(['git', 'commit', '-m', 'Add nested file'], cwd=self.repo_path, check=True, capture_output=True)
        
        # Analyze
        analyzer = GitFileLifecycle(str(self.repo_path), enable_aggregations=True)
        aggregator = FileHierarchyAggregator(analyzer)
        analyzer.register_aggregator(aggregator)
        analyzer.analyze_repository()
        
        data = aggregator.finalize()
        
        # Verify parent directories include child files
        self.assertIn('src', data['directories'])
        self.assertIn('src/core', data['directories'])
        
        # src should include engine.py in its total
        src_stats = data['directories']['src']
        self.assertGreaterEqual(src_stats['total_files'], 1)


class TestMilestoneAggregator(unittest.TestCase):
    """Test Milestone Aggregator"""

    def setUp(self):
        """Create a test repository"""
        self.test_dir = tempfile.mkdtemp()
        self.repo_path = Path(self.test_dir) / 'test_repo'
        self.repo_path.mkdir()
        
        # Initialize git repo
        subprocess.run(['git', 'init'], cwd=self.repo_path, check=True, capture_output=True)
        subprocess.run(['git', 'config', 'user.name', 'Test User'], cwd=self.repo_path, check=True, capture_output=True)
        subprocess.run(['git', 'config', 'user.email', 'test@example.com'], cwd=self.repo_path, check=True, capture_output=True)

    def tearDown(self):
        """Clean up test repository"""
        shutil.rmtree(self.test_dir)

    def test_milestone_initialization(self):
        """Test aggregator initialization"""
        analyzer = GitFileLifecycle(str(self.repo_path))
        aggregator = MilestoneAggregator(analyzer)
        
        self.assertEqual(aggregator.get_name(), "milestones")
        self.assertEqual(aggregator.get_schema_version(), "1.0.0")

    def test_file_milestone_detection(self):
        """Test detection of file lifecycle milestones"""
        # Create and modify a file
        (self.repo_path / 'app.py').write_text('initial')
        subprocess.run(['git', 'add', 'app.py'], cwd=self.repo_path, check=True, capture_output=True)
        subprocess.run(['git', 'commit', '-m', 'Create app'], cwd=self.repo_path, check=True, capture_output=True)
        
        # Multiple modifications
        for i in range(3):
            (self.repo_path / 'app.py').write_text(f'version {i}')
            subprocess.run(['git', 'add', 'app.py'], cwd=self.repo_path, check=True, capture_output=True)
            subprocess.run(['git', 'commit', '-m', f'Update {i}'], cwd=self.repo_path, check=True, capture_output=True)
        
        # Analyze
        analyzer = GitFileLifecycle(str(self.repo_path), enable_aggregations=True)
        aggregator = MilestoneAggregator(analyzer)
        analyzer.register_aggregator(aggregator)
        analyzer.analyze_repository()
        
        data = aggregator.finalize()
        
        # Verify structure
        self.assertIn('file_milestones', data)
        self.assertIn('tags', data)
        
        # Verify milestones for app.py
        self.assertIn('app.py', data['file_milestones'])
        milestones = data['file_milestones']['app.py']['milestones']
        
        # Should have at least creation and last_modified milestones
        milestone_types = [m['type'] for m in milestones]
        self.assertIn('created', milestone_types)

    def test_git_tag_detection(self):
        """Test detection of git tags"""
        # Create a commit and tag it
        (self.repo_path / 'README.md').write_text('readme')
        subprocess.run(['git', 'add', 'README.md'], cwd=self.repo_path, check=True, capture_output=True)
        subprocess.run(['git', 'commit', '-m', 'Initial commit'], cwd=self.repo_path, check=True, capture_output=True)
        subprocess.run(['git', 'tag', '-a', 'v1.0', '-m', 'Release 1.0'], cwd=self.repo_path, check=True, capture_output=True)
        
        # Analyze
        analyzer = GitFileLifecycle(str(self.repo_path), enable_aggregations=True)
        aggregator = MilestoneAggregator(analyzer)
        analyzer.register_aggregator(aggregator)
        analyzer.analyze_repository()
        
        data = aggregator.finalize()
        
        # Verify tag was detected
        self.assertGreater(len(data['tags']), 0)
        tag_names = [tag['tag_name'] for tag in data['tags']]
        self.assertIn('v1.0', tag_names)


class TestPhase3Integration(unittest.TestCase):
    """Integration tests for Phase 3"""

    def setUp(self):
        """Create a test repository"""
        self.test_dir = tempfile.mkdtemp()
        self.repo_path = Path(self.test_dir) / 'test_repo'
        self.repo_path.mkdir()
        
        # Initialize git repo
        subprocess.run(['git', 'init'], cwd=self.repo_path, check=True, capture_output=True)
        subprocess.run(['git', 'config', 'user.name', 'Test User'], cwd=self.repo_path, check=True, capture_output=True)
        subprocess.run(['git', 'config', 'user.email', 'test@example.com'], cwd=self.repo_path, check=True, capture_output=True)

    def tearDown(self):
        """Clean up test repository"""
        shutil.rmtree(self.test_dir)

    def test_full_pipeline_with_phase3(self):
        """Test complete analysis with all Phase 3 aggregators"""
        # Create test repository
        (self.repo_path / 'main.py').write_text('main')
        (self.repo_path / 'helper.py').write_text('helper')
        subprocess.run(['git', 'add', '.'], cwd=self.repo_path, check=True, capture_output=True)
        subprocess.run(['git', 'commit', '-m', 'Initial commit'], cwd=self.repo_path, check=True, capture_output=True)
        
        # Analyze with all Phase 3 aggregators
        analyzer = GitFileLifecycle(str(self.repo_path), enable_aggregations=True)
        analyzer.register_aggregator(CochangeNetworkAggregator(analyzer))
        analyzer.register_aggregator(FileHierarchyAggregator(analyzer))
        analyzer.register_aggregator(MilestoneAggregator(analyzer))
        
        data = analyzer.analyze_repository()
        
        # Verify core data
        self.assertIn('files', data)
        self.assertIn('total_commits', data)
        self.assertEqual(data['total_commits'], 1)
        
        # Export and verify all files are created
        output_dir = Path(self.test_dir) / 'output'
        manifest = analyzer.export_results(output_dir)
        
        # Verify manifest
        self.assertIn('datasets', manifest)
        self.assertIn('cochange_network', manifest['datasets'])
        self.assertIn('file_hierarchy', manifest['datasets'])
        self.assertIn('milestones', manifest['datasets'])
        
        # Verify files exist
        self.assertTrue((output_dir / 'file_lifecycle.json').exists())
        self.assertTrue((output_dir / 'networks' / 'cochange_network.json').exists())
        self.assertTrue((output_dir / 'aggregations' / 'directory_stats.json').exists())
        self.assertTrue((output_dir / 'milestones' / 'release_snapshots.json').exists())
        self.assertTrue((output_dir / 'manifest.json').exists())

    def test_core_lifecycle_unchanged(self):
        """Verify Phase 3 doesn't change core lifecycle output"""
        # Create simple repo
        (self.repo_path / 'test.txt').write_text('test')
        subprocess.run(['git', 'add', 'test.txt'], cwd=self.repo_path, check=True, capture_output=True)
        subprocess.run(['git', 'commit', '-m', 'Add test'], cwd=self.repo_path, check=True, capture_output=True)
        
        # Analyze without aggregations
        analyzer1 = GitFileLifecycle(str(self.repo_path), enable_aggregations=False)
        data1 = analyzer1.analyze_repository()
        
        # Analyze with Phase 3 aggregations
        analyzer2 = GitFileLifecycle(str(self.repo_path), enable_aggregations=True)
        analyzer2.register_aggregator(CochangeNetworkAggregator(analyzer2))
        analyzer2.register_aggregator(FileHierarchyAggregator(analyzer2))
        analyzer2.register_aggregator(MilestoneAggregator(analyzer2))
        data2 = analyzer2.analyze_repository()
        
        # Core lifecycle data should be identical
        self.assertEqual(data1['total_commits'], data2['total_commits'])
        self.assertEqual(data1['total_changes'], data2['total_changes'])
        self.assertEqual(len(data1['files']), len(data2['files']))

    def test_referential_integrity(self):
        """Test referential integrity across Phase 3 datasets"""
        # Create repository with structure
        (self.repo_path / 'src').mkdir()
        (self.repo_path / 'src' / 'main.py').write_text('main')
        (self.repo_path / 'src' / 'utils.py').write_text('utils')
        subprocess.run(['git', 'add', '.'], cwd=self.repo_path, check=True, capture_output=True)
        subprocess.run(['git', 'commit', '-m', 'Initial'], cwd=self.repo_path, check=True, capture_output=True)
        
        # Analyze
        analyzer = GitFileLifecycle(str(self.repo_path), enable_aggregations=True)
        cochange_agg = CochangeNetworkAggregator(analyzer)
        hierarchy_agg = FileHierarchyAggregator(analyzer)
        milestone_agg = MilestoneAggregator(analyzer)
        
        analyzer.register_aggregator(cochange_agg)
        analyzer.register_aggregator(hierarchy_agg)
        analyzer.register_aggregator(milestone_agg)
        
        data = analyzer.analyze_repository()
        
        # Get aggregation data
        cochange_data = cochange_agg.finalize()
        hierarchy_data = hierarchy_agg.finalize()
        milestone_data = milestone_agg.finalize()
        
        # Verify file counts match
        core_files = set(data['files'].keys())
        cochange_files = set(node['file_path'] for node in cochange_data['nodes'])
        hierarchy_all_files = set()
        for dir_stats in hierarchy_data['directories'].values():
            hierarchy_all_files.update(dir_stats.get('files', []))
        
        # Co-change network should have subset of core files
        self.assertTrue(cochange_files.issubset(core_files))


class TestBackwardCompatibility(unittest.TestCase):
    """Test backward compatibility with Phase 2"""

    def setUp(self):
        """Create a test repository"""
        self.test_dir = tempfile.mkdtemp()
        self.repo_path = Path(self.test_dir) / 'test_repo'
        self.repo_path.mkdir()
        
        # Initialize git repo
        subprocess.run(['git', 'init'], cwd=self.repo_path, check=True, capture_output=True)
        subprocess.run(['git', 'config', 'user.name', 'Test User'], cwd=self.repo_path, check=True, capture_output=True)
        subprocess.run(['git', 'config', 'user.email', 'test@example.com'], cwd=self.repo_path, check=True, capture_output=True)

    def tearDown(self):
        """Clean up test repository"""
        shutil.rmtree(self.test_dir)

    def test_phase2_aggregators_still_work(self):
        """Test that Phase 2 aggregators still work in Phase 3"""
        # Create test data
        (self.repo_path / 'file.txt').write_text('content')
        subprocess.run(['git', 'add', 'file.txt'], cwd=self.repo_path, check=True, capture_output=True)
        subprocess.run(['git', 'commit', '-m', 'Add file'], cwd=self.repo_path, check=True, capture_output=True)
        
        # Run with Phase 2 aggregators
        analyzer = GitFileLifecycle(str(self.repo_path), enable_aggregations=True)
        analyzer.register_aggregator(FileMetadataAggregator(analyzer))
        analyzer.register_aggregator(TemporalAggregator(analyzer))
        analyzer.register_aggregator(AuthorNetworkAggregator(analyzer))
        
        data = analyzer.analyze_repository()
        output_dir = Path(self.test_dir) / 'output'
        manifest = analyzer.export_results(output_dir)
        
        # Verify Phase 2 datasets exist
        self.assertIn('file_metadata', manifest['datasets'])
        self.assertIn('temporal_daily', manifest['datasets'])
        self.assertIn('author_network', manifest['datasets'])

    def test_version_number(self):
        """Test version number is updated to Phase 3"""
        analyzer = GitFileLifecycle(str(self.repo_path))
        self.assertEqual(analyzer.VERSION, "2.0.0-phase3")


def run_tests():
    """Run all tests"""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestCochangeNetworkAggregator))
    suite.addTests(loader.loadTestsFromTestCase(TestFileHierarchyAggregator))
    suite.addTests(loader.loadTestsFromTestCase(TestMilestoneAggregator))
    suite.addTests(loader.loadTestsFromTestCase(TestPhase3Integration))
    suite.addTests(loader.loadTestsFromTestCase(TestBackwardCompatibility))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print("="*70)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)