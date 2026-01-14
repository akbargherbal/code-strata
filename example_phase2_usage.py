#!/usr/bin/env python3
"""
Example: Using Phase 2 Aggregators
Demonstrates how to use File Metadata, Temporal, and Author Network aggregators
"""

import json
from pathlib import Path
from datetime import datetime
import sys

# Import the analyzer
try:
    from git_file_lifecycle_v2_phase2 import (
        GitFileLifecycle,
        FileMetadataAggregator,
        TemporalAggregator,
        AuthorNetworkAggregator
    )
except ImportError:
    print("Error: Make sure git_file_lifecycle_v2_phase2.py is in the same directory")
    sys.exit(1)


def example_basic_analysis(repo_path: str):
    """
    Example 1: Basic analysis with all Phase 2 aggregators
    """
    print("=" * 60)
    print("Example 1: Basic Analysis with Phase 2 Aggregators")
    print("=" * 60)
    
    # Create analyzer with aggregations enabled
    analyzer = GitFileLifecycle(repo_path, enable_aggregations=True)
    
    # Register all Phase 2 aggregators
    analyzer.register_aggregator(FileMetadataAggregator(analyzer))
    analyzer.register_aggregator(TemporalAggregator(analyzer))
    analyzer.register_aggregator(AuthorNetworkAggregator(analyzer))
    
    print(f"\nAnalyzing: {repo_path}")
    
    # Run analysis
    data = analyzer.analyze_repository()
    
    # Export results
    output_dir = Path("example_output_basic")
    manifest = analyzer.export_results(output_dir)
    
    print(f"\n✅ Analysis complete!")
    print(f"   Files analyzed: {len(data['files'])}")
    print(f"   Total commits: {data['total_commits']}")
    print(f"   Datasets generated: {len(manifest['datasets'])}")
    print(f"\nOutput directory: {output_dir}")


def example_analyze_file_metadata(repo_path: str):
    """
    Example 2: Analyze file metadata and find hotspots
    """
    print("\n" + "=" * 60)
    print("Example 2: File Metadata Analysis - Finding Hotspots")
    print("=" * 60)
    
    # Run analysis
    analyzer = GitFileLifecycle(repo_path, enable_aggregations=True)
    analyzer.register_aggregator(FileMetadataAggregator(analyzer))
    
    analyzer.analyze_repository()
    
    output_dir = Path("example_output_metadata")
    analyzer.export_results(output_dir)
    
    # Load and analyze metadata
    metadata_file = output_dir / "metadata" / "file_index.json"
    with open(metadata_file) as f:
        metadata = json.load(f)
    
    print(f"\n📊 File Metadata Summary")
    print(f"   Total files: {metadata['total_files']}")
    
    # Find hotspots (high activity files)
    files_by_activity = sorted(
        metadata['files'].items(),
        key=lambda x: x[1]['commits_per_day'],
        reverse=True
    )
    
    print(f"\n🔥 Top 5 Hotspot Files (by commits per day):")
    for i, (path, data) in enumerate(files_by_activity[:5], 1):
        print(f"   {i}. {path}")
        print(f"      Commits/day: {data['commits_per_day']:.3f}")
        print(f"      Total commits: {data['total_commits']}")
        print(f"      Primary author: {data['primary_author']['email']}")
        print()
    
    # Find files by primary author
    author_files = {}
    for path, data in metadata['files'].items():
        author = data['primary_author']['email']
        if author not in author_files:
            author_files[author] = []
        author_files[author].append(path)
    
    print(f"👥 Files by Primary Author:")
    for author, files in sorted(author_files.items(), key=lambda x: len(x[1]), reverse=True)[:5]:
        print(f"   {author}: {len(files)} files")


def example_temporal_analysis(repo_path: str):
    """
    Example 3: Temporal analysis - find activity patterns
    """
    print("\n" + "=" * 60)
    print("Example 3: Temporal Analysis - Activity Patterns")
    print("=" * 60)
    
    # Run analysis
    analyzer = GitFileLifecycle(repo_path, enable_aggregations=True)
    analyzer.register_aggregator(TemporalAggregator(analyzer))
    
    analyzer.analyze_repository()
    
    output_dir = Path("example_output_temporal")
    analyzer.export_results(output_dir)
    
    # Load temporal data
    daily_file = output_dir / "aggregations" / "temporal_daily.json"
    monthly_file = output_dir / "aggregations" / "temporal_monthly.json"
    
    with open(daily_file) as f:
        daily = json.load(f)
    
    with open(monthly_file) as f:
        monthly = json.load(f)
    
    print(f"\n📅 Temporal Summary")
    print(f"   Days tracked: {daily['period_count']}")
    print(f"   Months tracked: {monthly['period_count']}")
    
    # Find busiest day
    if daily['daily']:
        busiest_day = max(
            daily['daily'].items(),
            key=lambda x: x[1]['total_commits']
        )
        print(f"\n🎯 Busiest Day: {busiest_day[0]}")
        print(f"   Commits: {busiest_day[1]['total_commits']}")
        print(f"   Changes: {busiest_day[1]['total_changes']}")
        print(f"   Active files: {busiest_day[1]['active_files']}")
        print(f"   Contributors: {busiest_day[1]['unique_authors']}")
    
    # Monthly trends
    print(f"\n📈 Monthly Activity Trends:")
    for month, stats in sorted(monthly['monthly'].items())[-6:]:  # Last 6 months
        print(f"   {month}: {stats['total_commits']} commits, "
              f"{stats['unique_authors']} authors, "
              f"{stats['active_files']} files")
    
    # Calculate average daily activity
    if daily['daily']:
        total_commits = sum(d['total_commits'] for d in daily['daily'].values())
        avg_commits_per_day = total_commits / len(daily['daily'])
        print(f"\n📊 Average daily commits: {avg_commits_per_day:.2f}")


def example_author_network(repo_path: str):
    """
    Example 4: Author network analysis
    """
    print("\n" + "=" * 60)
    print("Example 4: Author Network Analysis")
    print("=" * 60)
    
    # Run analysis
    analyzer = GitFileLifecycle(repo_path, enable_aggregations=True)
    analyzer.register_aggregator(AuthorNetworkAggregator(analyzer))
    
    analyzer.analyze_repository()
    
    output_dir = Path("example_output_network")
    analyzer.export_results(output_dir)
    
    # Load network data
    network_file = output_dir / "networks" / "author_network.json"
    with open(network_file) as f:
        network = json.load(f)
    
    print(f"\n🕸️  Network Summary")
    print(f"   Authors (nodes): {network['node_count']}")
    print(f"   Collaborations (edges): {network['edge_count']}")
    print(f"   Network density: {network['edge_count'] / (network['node_count'] * (network['node_count'] - 1) / 2) if network['node_count'] > 1 else 0:.2%}")
    
    # Most prolific authors
    authors_by_commits = sorted(
        network['nodes'],
        key=lambda x: x['total_commits'],
        reverse=True
    )
    
    print(f"\n🏆 Top Contributors:")
    for i, author in enumerate(authors_by_commits[:5], 1):
        print(f"   {i}. {author['email']}")
        print(f"      Commits: {author['total_commits']}")
        print(f"      Files touched: {author['files_touched']}")
        print(f"      Active for: {author['active_days']} days")
        print()
    
    # Strongest collaborations
    if network['edges']:
        strongest_collabs = sorted(
            network['edges'],
            key=lambda x: x['weight'],
            reverse=True
        )
        
        print(f"🤝 Strongest Collaborations:")
        for i, edge in enumerate(strongest_collabs[:5], 1):
            source = network['nodes'][edge['source']]
            target = network['nodes'][edge['target']]
            print(f"   {i}. {source['email']} <-> {target['email']}")
            print(f"      Shared files: {edge['shared_files']}")
            print()


def example_custom_analysis(repo_path: str):
    """
    Example 5: Custom analysis combining multiple datasets
    """
    print("\n" + "=" * 60)
    print("Example 5: Custom Analysis - Combining Datasets")
    print("=" * 60)
    
    # Run full analysis
    analyzer = GitFileLifecycle(repo_path, enable_aggregations=True)
    analyzer.register_aggregator(FileMetadataAggregator(analyzer))
    analyzer.register_aggregator(TemporalAggregator(analyzer))
    analyzer.register_aggregator(AuthorNetworkAggregator(analyzer))
    
    analyzer.analyze_repository()
    
    output_dir = Path("example_output_custom")
    analyzer.export_results(output_dir)
    
    # Load all datasets
    with open(output_dir / "file_lifecycle.json") as f:
        lifecycle = json.load(f)
    
    with open(output_dir / "metadata" / "file_index.json") as f:
        metadata = json.load(f)
    
    with open(output_dir / "networks" / "author_network.json") as f:
        network = json.load(f)
    
    print(f"\n🔍 Custom Insights")
    
    # Insight 1: File ownership concentration
    print(f"\n1️⃣  File Ownership Analysis:")
    high_ownership = sum(
        1 for data in metadata['files'].values()
        if data['primary_author']['percentage'] > 80
    )
    low_ownership = sum(
        1 for data in metadata['files'].values()
        if data['primary_author']['percentage'] < 30
    )
    
    print(f"   Files with single owner (>80%): {high_ownership} ({high_ownership/metadata['total_files']*100:.1f}%)")
    print(f"   Files with shared ownership (<30%): {low_ownership} ({low_ownership/metadata['total_files']*100:.1f}%)")
    
    # Insight 2: Contributor diversity
    print(f"\n2️⃣  Contributor Diversity:")
    print(f"   Total contributors: {network['node_count']}")
    
    active_contributors = sum(
        1 for node in network['nodes']
        if node['total_commits'] >= 5
    )
    print(f"   Active contributors (5+ commits): {active_contributors}")
    
    # Insight 3: Code churn hotspots
    print(f"\n3️⃣  Code Churn Hotspots:")
    churn_files = [
        (path, data)
        for path, data in metadata['files'].items()
        if data['commits_per_day'] > 0.1  # More than 1 commit per 10 days
    ]
    
    print(f"   High-churn files (>0.1 commits/day): {len(churn_files)}")
    if churn_files:
        print(f"   Top churners:")
        for path, data in sorted(churn_files, key=lambda x: x[1]['commits_per_day'], reverse=True)[:3]:
            print(f"      {path}: {data['commits_per_day']:.3f} commits/day")
    
    # Insight 4: Collaboration health
    print(f"\n4️⃣  Collaboration Health:")
    isolated_authors = sum(
        1 for node in network['nodes']
        if not any(edge['source'] == node['id'] or edge['target'] == node['id'] for edge in network['edges'])
    )
    
    if network['node_count'] > 0:
        print(f"   Isolated contributors: {isolated_authors} ({isolated_authors/network['node_count']*100:.1f}%)")
        print(f"   Average collaborations per person: {2 * network['edge_count'] / network['node_count']:.2f}")


def main():
    """Main execution"""
    if len(sys.argv) < 2:
        print("Usage: python example_phase2_usage.py <repo_path>")
        print("\nExamples:")
        print("  python example_phase2_usage.py /path/to/your/repo")
        print("  python example_phase2_usage.py .")
        sys.exit(1)
    
    repo_path = sys.argv[1]
    
    # Validate repository
    if not Path(repo_path).is_dir():
        print(f"Error: Directory not found: {repo_path}")
        sys.exit(1)
    
    if not (Path(repo_path) / '.git').exists():
        print(f"Error: Not a git repository: {repo_path}")
        sys.exit(1)
    
    print("=" * 60)
    print("Git File Lifecycle Analyzer - Phase 2 Examples")
    print("=" * 60)
    print(f"\nRepository: {repo_path}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Run all examples
        example_basic_analysis(repo_path)
        example_analyze_file_metadata(repo_path)
        example_temporal_analysis(repo_path)
        example_author_network(repo_path)
        example_custom_analysis(repo_path)
        
        print("\n" + "=" * 60)
        print("✅ All examples completed successfully!")
        print("=" * 60)
        print("\nGenerated directories:")
        print("  - example_output_basic/")
        print("  - example_output_metadata/")
        print("  - example_output_temporal/")
        print("  - example_output_network/")
        print("  - example_output_custom/")
        print("\nExplore these directories to see the generated datasets!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
