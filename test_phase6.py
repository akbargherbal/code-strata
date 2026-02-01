#!/usr/bin/env python3
"""
Quick test to verify the proposal-based aggregators work correctly.
"""

import sys
import json
from datetime import datetime, timezone

# Import from strata_updated
from strata import (
    HealthScorer,
    ProjectHierarchyAggregator,
    FileMetricsIndexAggregator,
    TemporalActivityMapAggregator,
    ProgressReporter,
)


class MockAnalyzer:
    """Mock analyzer for testing"""
    def __init__(self):
        self.repo_path = "/test/repo"


def test_health_scorer():
    """Test the health scoring logic"""
    print("Testing HealthScorer...")
    
    # Test healthy file
    result = HealthScorer.calculate_health_score(
        total_commits=50,
        unique_authors=5,
        churn_rate=0.1,
        age_days=365,
        file_path="src/app.py"
    )
    assert "health_score" in result
    assert "health_category" in result
    assert "bus_factor_status" in result
    assert 0 <= result["health_score"] <= 100
    print(f"  Healthy file: score={result['health_score']}, category={result['health_category']}")
    
    # Test critical file
    result = HealthScorer.calculate_health_score(
        total_commits=100,
        unique_authors=1,
        churn_rate=0.8,
        age_days=30,
        file_path="src/critical.py"
    )
    assert result["health_category"] in ["critical", "medium"]
    print(f"  Critical file: score={result['health_score']}, category={result['health_category']}")
    
    print("✓ HealthScorer tests passed")


def test_project_hierarchy_aggregator():
    """Test ProjectHierarchyAggregator"""
    print("\nTesting ProjectHierarchyAggregator...")
    
    analyzer = MockAnalyzer()
    reporter = ProgressReporter(quiet=True)
    agg = ProjectHierarchyAggregator(analyzer, reporter=reporter)
    
    # Simulate some commits
    commit1 = {
        "commit_hash": "abc123",
        "timestamp": 1000000,
        "author_email": "alice@example.com",
        "author_id": "alice",
    }
    
    changes1 = [
        {
            "file_path": "src/app.py",
            "operation": "A",
            "lines_added": 100,
            "lines_deleted": 0,
        },
        {
            "file_path": "src/utils/helpers.py",
            "operation": "A",
            "lines_added": 50,
            "lines_deleted": 0,
        },
    ]
    
    agg.process_commit(commit1, changes1)
    
    # Add more commits
    commit2 = {
        "commit_hash": "def456",
        "timestamp": 2000000,
        "author_email": "bob@example.com",
        "author_id": "bob",
    }
    
    changes2 = [
        {
            "file_path": "src/app.py",
            "operation": "M",
            "lines_added": 20,
            "lines_deleted": 10,
        },
    ]
    
    agg.process_commit(commit2, changes2)
    
    # Finalize and check structure
    result = agg.finalize()
    
    assert "meta" in result
    assert "tree" in result
    assert result["tree"]["type"] == "directory"
    assert "children" in result["tree"]
    assert len(result["tree"]["children"]) > 0
    
    # Check that files have attributes
    def find_files(node):
        files = []
        if node["type"] == "file":
            files.append(node)
        elif "children" in node:
            for child in node["children"]:
                files.extend(find_files(child))
        return files
    
    files = find_files(result["tree"])
    assert len(files) == 2
    
    for file in files:
        assert "attributes" in file
        assert "health_score" in file["attributes"]
        assert "health_category" in file["attributes"]
        assert "bus_factor_status" in file["attributes"]
        assert "churn_rate" in file["attributes"]
    
    print(f"  Generated tree with {len(files)} files")
    print("✓ ProjectHierarchyAggregator tests passed")


def test_file_metrics_index_aggregator():
    """Test FileMetricsIndexAggregator"""
    print("\nTesting FileMetricsIndexAggregator...")
    
    analyzer = MockAnalyzer()
    reporter = ProgressReporter(quiet=True)
    agg = FileMetricsIndexAggregator(analyzer, reporter=reporter)
    
    # Simulate commits
    commit1 = {
        "commit_hash": "abc123",
        "timestamp": 1000000,
        "author_email": "alice@example.com",
        "author_id": "alice",
    }
    
    changes1 = [
        {
            "file_path": "src/app.py",
            "operation": "A",
            "lines_added": 100,
            "lines_deleted": 0,
        },
        {
            "file_path": "src/utils.py",
            "operation": "A",
            "lines_added": 50,
            "lines_deleted": 0,
        },
    ]
    
    agg.process_commit(commit1, changes1)
    
    commit2 = {
        "commit_hash": "def456",
        "timestamp": 2000000,
        "author_email": "bob@example.com",
        "author_id": "bob",
    }
    
    changes2 = [
        {
            "file_path": "src/app.py",
            "operation": "M",
            "lines_added": 20,
            "lines_deleted": 10,
        },
        {
            "file_path": "src/utils.py",
            "operation": "M",
            "lines_added": 5,
            "lines_deleted": 2,
        },
    ]
    
    agg.process_commit(commit2, changes2)
    
    # Finalize and check structure
    result = agg.finalize()
    
    assert "src/app.py" in result
    assert "src/utils.py" in result
    
    app_metrics = result["src/app.py"]
    assert "identifiers" in app_metrics
    assert "volume" in app_metrics
    assert "coupling" in app_metrics
    assert "lifecycle" in app_metrics
    
    assert app_metrics["identifiers"]["primary_author_id"] == "alice"
    assert app_metrics["volume"]["lines_added"] == 120
    assert app_metrics["volume"]["lines_deleted"] == 10
    assert app_metrics["volume"]["total_commits"] == 2
    
    # Check coupling
    assert len(app_metrics["coupling"]["top_partners"]) > 0
    assert app_metrics["coupling"]["top_partners"][0]["path"] == "src/utils.py"
    
    print(f"  Generated index for {len(result)} files")
    print("✓ FileMetricsIndexAggregator tests passed")


def test_temporal_activity_map_aggregator():
    """Test TemporalActivityMapAggregator"""
    print("\nTesting TemporalActivityMapAggregator...")
    
    analyzer = MockAnalyzer()
    reporter = ProgressReporter(quiet=True)
    agg = TemporalActivityMapAggregator(analyzer, reporter=reporter)
    
    # Simulate commits in different weeks
    commit1 = {
        "commit_hash": "abc123",
        "timestamp": 1704067200,  # 2024-01-01
        "author_email": "alice@example.com",
        "author_id": "alice",
    }
    
    changes1 = [
        {
            "file_path": "src/app.py",
            "operation": "A",
            "lines_added": 100,
            "lines_deleted": 0,
        },
    ]
    
    agg.process_commit(commit1, changes1)
    
    commit2 = {
        "commit_hash": "def456",
        "timestamp": 1704672000,  # 2024-01-08 (next week)
        "author_email": "bob@example.com",
        "author_id": "bob",
    }
    
    changes2 = [
        {
            "file_path": "src/app.py",
            "operation": "M",
            "lines_added": 20,
            "lines_deleted": 10,
        },
    ]
    
    agg.process_commit(commit2, changes2)
    
    # Finalize and check structure
    result = agg.finalize()
    
    assert "meta" in result
    assert "data" in result
    assert result["meta"]["granularity"] == "week"
    assert result["meta"]["data_schema"] == ["commits", "lines_changed", "unique_authors"]
    
    # Check that we have data for directories
    assert len(result["data"]) > 0
    
    # Check data format
    for dir_path, weeks in result["data"].items():
        for week_key, values in weeks.items():
            assert len(values) == 3  # commits, lines_changed, unique_authors
            assert isinstance(values[0], int)  # commits
            assert isinstance(values[1], int)  # lines_changed
            assert isinstance(values[2], int)  # unique_authors
    
    print(f"  Generated activity map for {len(result['data'])} directories")
    print("✓ TemporalActivityMapAggregator tests passed")


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Proposal-Based Aggregators")
    print("=" * 60)
    
    try:
        test_health_scorer()
        test_project_hierarchy_aggregator()
        test_file_metrics_index_aggregator()
        test_temporal_activity_map_aggregator()
        
        print("\n" + "=" * 60)
        print("All tests passed! ✓")
        print("=" * 60)
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)