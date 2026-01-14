#!/usr/bin/env python3
"""
Example: Using Phase 3 Aggregators
Demonstrates co-change network, file hierarchy, and milestone aggregators
"""

import json
from pathlib import Path
import sys

# Import the analyzer
try:
    from git_file_lifecycle import (
        GitFileLifecycle,
        CochangeNetworkAggregator,
        FileHierarchyAggregator,
        MilestoneAggregator,
        # Phase 2 aggregators also available
        FileMetadataAggregator,
        TemporalAggregator,
        AuthorNetworkAggregator,
    )
except ImportError:
    print("Error: Make sure git_file_lifecycle_v2_phase3.py is in the same directory")
    sys.exit(1)


def example_cochange_network(repo_path: str):
    """
    Example 1: Analyze co-change patterns (files that change together)
    """
    print("=" * 70)
    print("Example 1: Co-change Network Analysis")
    print("=" * 70)

    analyzer = GitFileLifecycle(repo_path, enable_aggregations=True)

    # Register co-change network aggregator
    # min_cochanges=3 means only show pairs that changed together 3+ times
    cochange_agg = CochangeNetworkAggregator(analyzer, min_cochanges=3)
    analyzer.register_aggregator(cochange_agg)

    print(f"\nAnalyzing: {repo_path}")
    data = analyzer.analyze_repository()

    # Get co-change data
    cochange_data = cochange_agg.finalize()

    print(f"\n✅ Co-change Network Results:")
    print(f"   Total files: {cochange_data['node_count']}")
    print(f"   Coupling relationships: {cochange_data['edge_count']}")

    # Show top coupled files
    if cochange_data["edges"]:
        print(f"\n📊 Top 10 Most Coupled File Pairs:")
        sorted_edges = sorted(
            cochange_data["edges"], key=lambda e: e["coupling_strength"], reverse=True
        )[:10]

        for i, edge in enumerate(sorted_edges, 1):
            source = cochange_data["nodes"][edge["source"]]["file_path"]
            target = cochange_data["nodes"][edge["target"]]["file_path"]
            strength = edge["coupling_strength"]
            count = edge["co_change_count"]
            print(f"   {i}. {source} <-> {target}")
            print(f"      Strength: {strength:.2f}, Co-changes: {count}")

    # Export
    output_dir = Path("example_output_cochange")
    analyzer.export_results(output_dir)
    print(f"\n📁 Output: {output_dir}/networks/cochange_network.json")


def example_file_hierarchy(repo_path: str):
    """
    Example 2: Analyze directory structure and statistics
    """
    print("\n" + "=" * 70)
    print("Example 2: File Hierarchy Analysis")
    print("=" * 70)

    analyzer = GitFileLifecycle(repo_path, enable_aggregations=True)
    hierarchy_agg = FileHierarchyAggregator(analyzer)
    analyzer.register_aggregator(hierarchy_agg)

    print(f"\nAnalyzing: {repo_path}")
    data = analyzer.analyze_repository()

    # Get hierarchy data
    hierarchy_data = hierarchy_agg.finalize()

    print(f"\n✅ File Hierarchy Results:")
    print(f"   Total directories: {hierarchy_data['total_directories']}")

    # Show directory statistics
    print(f"\n📊 Top 10 Most Active Directories:")
    sorted_dirs = sorted(
        hierarchy_data["directories"].items(),
        key=lambda x: x[1]["total_changes"],
        reverse=True,
    )[:10]

    for i, (dir_path, stats) in enumerate(sorted_dirs, 1):
        print(f"   {i}. {dir_path or '/'}")
        print(f"      Files: {stats['total_files']}, Changes: {stats['total_changes']}")
        print(
            f"      Authors: {stats['unique_authors']}, Activity: {stats['activity_score']:.2f}"
        )

    # Export
    output_dir = Path("example_output_hierarchy")
    analyzer.export_results(output_dir)
    print(f"\n📁 Output: {output_dir}/aggregations/directory_stats.json")


def example_milestones(repo_path: str):
    """
    Example 3: Track milestones and git tags
    """
    print("\n" + "=" * 70)
    print("Example 3: Milestone and Release Analysis")
    print("=" * 70)

    analyzer = GitFileLifecycle(repo_path, enable_aggregations=True)
    milestone_agg = MilestoneAggregator(analyzer)
    analyzer.register_aggregator(milestone_agg)

    print(f"\nAnalyzing: {repo_path}")
    data = analyzer.analyze_repository()

    # Get milestone data
    milestone_data = milestone_agg.finalize()

    print(f"\n✅ Milestone Results:")
    print(f"   Git tags found: {milestone_data['total_tags']}")
    print(f"   Files with milestones: {milestone_data['total_files_with_milestones']}")

    # Show tags
    if milestone_data["tags"]:
        print(f"\n📊 Git Tags/Releases:")
        for tag in milestone_data["tags"][:10]:
            print(f"   - {tag['tag_name']} ({tag['date']})")
            print(f"     Files: {tag['files_in_commit']}")

    # Show file milestones
    if milestone_data["file_milestones"]:
        print(f"\n📊 Sample File Milestones:")
        for file_path, info in list(milestone_data["file_milestones"].items())[:5]:
            print(f"   File: {file_path}")
            print(f"   Total events: {info['total_events']}")
            for milestone in info["milestones"][:3]:
                print(
                    f"      - {milestone['type']}: {milestone.get('timestamp', 'N/A')}"
                )

    # Export
    output_dir = Path("example_output_milestones")
    analyzer.export_results(output_dir)
    print(f"\n📁 Output: {output_dir}/milestones/release_snapshots.json")


def example_combined_analysis(repo_path: str):
    """
    Example 4: Run all Phase 3 aggregators together
    """
    print("\n" + "=" * 70)
    print("Example 4: Complete Phase 3 Analysis")
    print("=" * 70)

    analyzer = GitFileLifecycle(repo_path, enable_aggregations=True)

    # Register all Phase 3 aggregators
    analyzer.register_aggregator(CochangeNetworkAggregator(analyzer))
    analyzer.register_aggregator(FileHierarchyAggregator(analyzer))
    analyzer.register_aggregator(MilestoneAggregator(analyzer))

    print(f"\nAnalyzing: {repo_path}")
    data = analyzer.analyze_repository()

    # Export all results
    output_dir = Path("example_output_complete")
    manifest = analyzer.export_results(output_dir)

    print(f"\n✅ Complete Analysis Done!")
    print(f"   Files analyzed: {len(data['files'])}")
    print(f"   Total commits: {data['total_commits']}")
    print(f"   Datasets generated: {len(manifest['datasets'])}")

    print(f"\n📁 Output Directory: {output_dir}/")
    print(f"   Generated files:")
    for name, info in manifest["datasets"].items():
        print(f"      - {info['file']} ({info['record_count']} records)")

    return manifest


def example_phase2_and_phase3(repo_path: str):
    """
    Example 5: Use both Phase 2 and Phase 3 aggregators
    """
    print("\n" + "=" * 70)
    print("Example 5: Phase 2 + Phase 3 Complete Analysis")
    print("=" * 70)

    analyzer = GitFileLifecycle(repo_path, enable_aggregations=True)

    # Register Phase 2 aggregators
    analyzer.register_aggregator(FileMetadataAggregator(analyzer))
    analyzer.register_aggregator(TemporalAggregator(analyzer))
    analyzer.register_aggregator(AuthorNetworkAggregator(analyzer))

    # Register Phase 3 aggregators
    analyzer.register_aggregator(CochangeNetworkAggregator(analyzer))
    analyzer.register_aggregator(FileHierarchyAggregator(analyzer))
    analyzer.register_aggregator(MilestoneAggregator(analyzer))

    print(f"\nAnalyzing with all aggregators: {repo_path}")
    data = analyzer.analyze_repository()

    # Export
    output_dir = Path("example_output_full")
    manifest = analyzer.export_results(output_dir)

    print(f"\n✅ Full Analysis Complete!")
    print(f"\n📊 Generated Datasets:")
    print(f"   Phase 1 (Core):")
    print(f"      - file_lifecycle.json")
    print(f"   Phase 2 (Enrichment):")
    print(f"      - metadata/file_index.json")
    print(f"      - aggregations/temporal_daily.json")
    print(f"      - aggregations/temporal_monthly.json")
    print(f"      - networks/author_network.json")
    print(f"   Phase 3 (Advanced):")
    print(f"      - networks/cochange_network.json")
    print(f"      - aggregations/directory_stats.json")
    print(f"      - milestones/release_snapshots.json")

    print(f"\n📁 All outputs in: {output_dir}/")


def example_custom_analysis(repo_path: str):
    """
    Example 6: Custom analysis - find architectural hotspots
    """
    print("\n" + "=" * 70)
    print("Example 6: Custom Analysis - Architectural Hotspots")
    print("=" * 70)

    analyzer = GitFileLifecycle(repo_path, enable_aggregations=True)

    # Use co-change to find tightly coupled modules
    cochange_agg = CochangeNetworkAggregator(analyzer, min_cochanges=5)
    # Use hierarchy to understand directory organization
    hierarchy_agg = FileHierarchyAggregator(analyzer)

    analyzer.register_aggregator(cochange_agg)
    analyzer.register_aggregator(hierarchy_agg)

    print(f"\nAnalyzing: {repo_path}")
    data = analyzer.analyze_repository()

    # Get data
    cochange_data = cochange_agg.finalize()
    hierarchy_data = hierarchy_agg.finalize()

    # Find hotspot directories (high coupling + high activity)
    print(f"\n🔥 Architectural Hotspots:")
    print(f"   (High activity + tight coupling = needs attention)")

    hotspots = []
    for dir_path, stats in hierarchy_data["directories"].items():
        if stats["total_changes"] > 50 and stats["unique_authors"] > 2:
            hotspots.append((dir_path, stats))

    hotspots.sort(key=lambda x: x[1]["activity_score"], reverse=True)

    for dir_path, stats in hotspots[:5]:
        print(f"\n   📂 {dir_path or '/'}")
        print(f"      Activity Score: {stats['activity_score']:.2f}")
        print(f"      Total Changes: {stats['total_changes']}")
        print(f"      Authors: {stats['unique_authors']}")
        print(f"      Files: {stats['total_files']}")
        print(f"      Recommendation: Consider refactoring or better documentation")


def main():
    """Run all examples"""
    if len(sys.argv) < 2:
        print("Usage: python example_phase3_usage.py <repo_path>")
        print("\nExamples:")
        print("  python example_phase3_usage.py .")
        print("  python example_phase3_usage.py /path/to/repository")
        sys.exit(1)

    repo_path = sys.argv[1]

    print("🚀 Phase 3 Aggregator Examples")
    print("=" * 70)
    print(f"Repository: {repo_path}")
    print("=" * 70)

    try:
        # Run examples
        example_cochange_network(repo_path)
        example_file_hierarchy(repo_path)
        example_milestones(repo_path)
        example_combined_analysis(repo_path)
        example_phase2_and_phase3(repo_path)
        example_custom_analysis(repo_path)

        print("\n" + "=" * 70)
        print("✅ All examples completed successfully!")
        print("=" * 70)

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
