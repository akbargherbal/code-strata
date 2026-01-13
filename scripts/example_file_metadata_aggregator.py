#!/usr/bin/env python3
"""
Example Aggregator: File Metadata Index
Demonstrates how to create custom aggregators using Phase 1 framework.

This aggregator tracks per-file statistics including:
- First and last commit timestamps
- Total commits per file
- Unique authors
- Operation distribution
- Activity metrics
"""

from pathlib import Path
from collections import defaultdict
from datetime import datetime
from typing import Dict, Set


class FileMetadataAggregator:
    """
    Example aggregator that collects metadata about each file.
    Ready for Phase 2 integration.
    """
    
    def __init__(self, analyzer):
        """
        Initialize the aggregator.
        
        Args:
            analyzer: GitFileLifecycle instance
        """
        self.analyzer = analyzer
        self.file_stats = defaultdict(lambda: {
            "first_seen": None,
            "first_seen_timestamp": None,
            "last_modified": None,
            "last_modified_timestamp": None,
            "total_commits": 0,
            "authors": set(),
            "operations": {"A": 0, "M": 0, "D": 0, "R": 0, "C": 0, "T": 0}
        })
        self.errors = []
    
    def get_name(self) -> str:
        """Return the name of this aggregator"""
        return "file_metadata"
    
    def get_output_path(self, output_dir: Path) -> Path:
        """Return the output file path for this aggregator"""
        return output_dir / "metadata" / "file_index.json"
    
    def process_commit(self, commit: dict, changes: list):
        """
        Process each commit during streaming.
        
        Args:
            commit: Commit metadata dict with keys: hash, timestamp, datetime, 
                    author_name, author_email, subject
            changes: List of file changes, each with keys: path, operation, 
                     old_path (optional), similarity (optional)
        """
        try:
            for change in changes:
                file_path = change["path"]
                stats = self.file_stats[file_path]
                
                # Track first seen (file creation)
                if stats["first_seen"] is None:
                    stats["first_seen"] = commit["datetime"]
                    stats["first_seen_timestamp"] = commit["timestamp"]
                
                # Always update last modified
                stats["last_modified"] = commit["datetime"]
                stats["last_modified_timestamp"] = commit["timestamp"]
                
                # Count commits
                stats["total_commits"] += 1
                
                # Track unique authors
                stats["authors"].add(commit["author_email"])
                
                # Count operations
                op = change["operation"]
                if op in stats["operations"]:
                    stats["operations"][op] += 1
                    
        except Exception as e:
            self.errors.append(f"Error processing commit {commit.get('hash', 'unknown')}: {e}")
    
    def finalize(self) -> dict:
        """
        Compute final statistics after all commits processed.
        
        Returns:
            Dictionary containing the aggregated file metadata
        """
        output = {
            "schema_version": "1.0.0",
            "total_files": len(self.file_stats),
            "generation_method": "streaming",
            "files": {}
        }
        
        for file_path, stats in self.file_stats.items():
            # Calculate derived metrics
            age_seconds = (
                stats["last_modified_timestamp"] - stats["first_seen_timestamp"]
                if stats["last_modified_timestamp"] and stats["first_seen_timestamp"]
                else 0
            )
            age_days = age_seconds / 86400 if age_seconds > 0 else 0
            
            # Calculate activity score (commits per day)
            commits_per_day = (
                stats["total_commits"] / age_days 
                if age_days > 0 
                else stats["total_commits"]
            )
            
            # Determine primary author (most commits)
            # In real implementation, would track commits per author
            primary_author = {
                "email": list(stats["authors"])[0] if stats["authors"] else "unknown",
                "commit_count": stats["total_commits"],
                "percentage": 100.0  # Simplified
            }
            
            # Build output record
            output["files"][file_path] = {
                "first_seen": stats["first_seen"],
                "last_modified": stats["last_modified"],
                "total_commits": stats["total_commits"],
                "unique_authors": len(stats["authors"]),
                "primary_author": primary_author,
                "operations": stats["operations"],
                "age_days": round(age_days, 2),
                "commits_per_day": round(commits_per_day, 3),
                "lifecycle_event_count": stats["total_commits"]
            }
        
        return output
    
    def export(self, output_path: Path):
        """
        Export aggregated data to JSON file.
        
        Args:
            output_path: Path to output JSON file
        """
        import json
        
        # Ensure parent directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Get finalized data
        data = self.finalize()
        
        # Write to file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)


# Example usage in standalone script
if __name__ == "__main__":
    import sys
    import click
    
    # This would normally be imported from git_file_lifecycle_v2
    # Shown here for demonstration purposes
    
    @click.command()
    @click.argument("repo_path", type=click.Path(exists=True))
    @click.option("--output-dir", default="output", help="Output directory")
    def main(repo_path, output_dir):
        """
        Demonstrate FileMetadataAggregator usage.
        
        In production, this aggregator would be registered with
        the main GitFileLifecycle analyzer.
        """
        click.echo("📊 File Metadata Aggregator Example")
        click.echo("=" * 60)
        click.echo()
        
        # In production usage:
        # from git_file_lifecycle_v2 import GitFileLifecycle
        # 
        # analyzer = GitFileLifecycle(repo_path, enable_aggregations=True)
        # aggregator = FileMetadataAggregator(analyzer)
        # analyzer.register_aggregator(aggregator)
        # analyzer.analyze_repository()
        # analyzer.export_aggregators(output_dir)
        
        click.echo("✓ This is a demonstration aggregator")
        click.echo("✓ In Phase 2, this will generate metadata/file_index.json")
        click.echo()
        click.echo("Expected output structure:")
        click.echo("""
        {
          "schema_version": "1.0.0",
          "total_files": 1250,
          "files": {
            "src/main.py": {
              "first_seen": "2023-01-15T10:30:00Z",
              "last_modified": "2024-01-10T15:45:00Z",
              "total_commits": 45,
              "unique_authors": 7,
              "primary_author": {
                "email": "dev@example.com",
                "commit_count": 28,
                "percentage": 62.2
              },
              "operations": {"A": 1, "M": 43, "D": 0, "R": 1, "C": 0},
              "age_days": 360.0,
              "commits_per_day": 0.125,
              "lifecycle_event_count": 45
            }
          }
        }
        """)
    
    main()
