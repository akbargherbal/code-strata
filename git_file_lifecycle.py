#!/usr/bin/env python3
"""
Git File Lifecycle Analyzer - Version 2.0.0-phase2

Phase 2 Implementation: Core Enrichment Datasets
- File Metadata Index
- Temporal Aggregations (Daily/Monthly)
- Author Network

Maintains 100% backward compatibility with v1.0 schema.
"""

import subprocess
import json
import os
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Set
from collections import defaultdict
from abc import ABC, abstractmethod
import hashlib
import argparse


class DatasetAggregator(ABC):
    """Base class for generating additional datasets"""

    def __init__(self, analyzer: 'GitFileLifecycle'):
        self.analyzer = analyzer
        self.data = {}
        self.errors = []

    @abstractmethod
    def get_name(self) -> str:
        """Return aggregator name for manifest"""
        pass

    @abstractmethod
    def get_output_path(self, output_dir: Path) -> Path:
        """Return output file path"""
        pass

    @abstractmethod
    def get_schema_version(self) -> str:
        """Return schema version"""
        pass

    def process_commit(self, commit: dict, changes: list):
        """Called for each commit during streaming (override if needed)"""
        pass

    def finalize(self) -> dict:
        """Called after all commits processed (override if needed)"""
        return self.data

    def export(self, output_dir: Path) -> dict:
        """Export to JSON file and return manifest entry"""
        output_path = self.get_output_path(output_dir)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        data = self.finalize()
        
        # Write to file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        # Calculate file stats
        file_size = output_path.stat().st_size
        with open(output_path, 'rb') as f:
            sha256 = hashlib.sha256(f.read()).hexdigest()
        
        # Return manifest entry
        return {
            'file': str(output_path.relative_to(output_dir)),
            'schema_version': self.get_schema_version(),
            'record_count': self._get_record_count(data),
            'file_size_bytes': file_size,
            'sha256': sha256,
            'production_ready': self._is_production_ready(),
            'preview': not self._is_production_ready()
        }

    def _get_record_count(self, data: dict) -> int:
        """Calculate record count from data"""
        if 'files' in data:
            return len(data['files'])
        elif 'daily' in data or 'monthly' in data:
            return sum(len(v) for v in data.values() if isinstance(v, dict))
        elif 'nodes' in data:
            return len(data['nodes'])
        return 0

    def _is_production_ready(self) -> bool:
        """Override to mark dataset as production ready"""
        return False


class FileMetadataAggregator(DatasetAggregator):
    """
    Phase 2.1: File Metadata Index
    Tracks per-file statistics and metrics
    """

    def __init__(self, analyzer: 'GitFileLifecycle'):
        super().__init__(analyzer)
        self.file_stats = defaultdict(lambda: {
            'first_seen': None,
            'last_modified': None,
            'timestamps': [],
            'authors': defaultdict(int),
            'operations': defaultdict(int),
            'commits': []
        })

    def get_name(self) -> str:
        return "file_metadata"

    def get_output_path(self, output_dir: Path) -> Path:
        return output_dir / "metadata" / "file_index.json"

    def get_schema_version(self) -> str:
        return "1.0.0"

    def process_commit(self, commit: dict, changes: list):
        """Process each commit to collect file metadata"""
        for change in changes:
            file_path = change['file_path']
            operation = change['operation']
            timestamp = commit['timestamp']
            author_email = commit['author_email']

            stats = self.file_stats[file_path]
            
            # Track first and last seen
            if stats['first_seen'] is None:
                stats['first_seen'] = timestamp
            stats['last_modified'] = timestamp
            
            # Collect timestamps
            stats['timestamps'].append(timestamp)
            
            # Track authors
            stats['authors'][author_email] += 1
            
            # Track operations
            stats['operations'][operation] += 1
            
            # Track commits
            stats['commits'].append(commit['commit_hash'])

    def finalize(self) -> dict:
        """Compute final aggregations"""
        files_data = {}
        
        for file_path, stats in self.file_stats.items():
            # Calculate primary author
            if stats['authors']:
                primary_author_email = max(stats['authors'].items(), key=lambda x: x[1])[0]
                primary_author_count = stats['authors'][primary_author_email]
                total_commits = sum(stats['authors'].values())
                primary_percentage = (primary_author_count / total_commits * 100) if total_commits > 0 else 0
            else:
                primary_author_email = ""
                primary_author_count = 0
                primary_percentage = 0

            # Calculate age in days
            if stats['first_seen'] and stats['last_modified']:
                age_seconds = stats['last_modified'] - stats['first_seen']
                age_days = age_seconds / 86400  # seconds in a day
            else:
                age_days = 0

            # Calculate commits per day
            commits_per_day = len(set(stats['commits'])) / age_days if age_days > 0 else 0

            # Build file entry
            files_data[file_path] = {
                'first_seen': datetime.fromtimestamp(stats['first_seen'], tz=timezone.utc).isoformat() if stats['first_seen'] else None,
                'last_modified': datetime.fromtimestamp(stats['last_modified'], tz=timezone.utc).isoformat() if stats['last_modified'] else None,
                'total_commits': len(set(stats['commits'])),
                'unique_authors': len(stats['authors']),
                'primary_author': {
                    'email': primary_author_email,
                    'commit_count': primary_author_count,
                    'percentage': round(primary_percentage, 1)
                },
                'operations': dict(stats['operations']),
                'age_days': round(age_days, 1),
                'commits_per_day': round(commits_per_day, 3),
                'lifecycle_event_count': len(stats['commits'])
            }

        return {
            'schema_version': self.get_schema_version(),
            'total_files': len(files_data),
            'generation_method': 'streaming',
            'files': files_data
        }

    def _is_production_ready(self) -> bool:
        return False  # Preview feature in Phase 2


class TemporalAggregator(DatasetAggregator):
    """
    Phase 2.2: Temporal Aggregations
    Generates daily and monthly rollups
    """

    def __init__(self, analyzer: 'GitFileLifecycle'):
        super().__init__(analyzer)
        self.daily_data = defaultdict(lambda: {
            'commits': set(),
            'changes': 0,
            'authors': set(),
            'files': set(),
            'operations': defaultdict(int)
        })
        self.monthly_data = defaultdict(lambda: {
            'commits': set(),
            'changes': 0,
            'authors': set(),
            'files': set(),
            'operations': defaultdict(int)
        })

    def get_name(self) -> str:
        return "temporal_aggregations"

    def get_output_path(self, output_dir: Path) -> Path:
        # Returns daily path, monthly will be sibling
        return output_dir / "aggregations" / "temporal_daily.json"

    def get_schema_version(self) -> str:
        return "1.0.0"

    def process_commit(self, commit: dict, changes: list):
        """Aggregate data by day and month"""
        timestamp = commit['timestamp']
        dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)
        
        # Get day and month keys
        day_key = dt.strftime('%Y-%m-%d')
        month_key = dt.strftime('%Y-%m')
        
        # Update daily stats
        day_stats = self.daily_data[day_key]
        day_stats['commits'].add(commit['commit_hash'])
        day_stats['changes'] += len(changes)
        day_stats['authors'].add(commit['author_email'])
        
        # Update monthly stats
        month_stats = self.monthly_data[month_key]
        month_stats['commits'].add(commit['commit_hash'])
        month_stats['changes'] += len(changes)
        month_stats['authors'].add(commit['author_email'])
        
        # Track files and operations
        for change in changes:
            day_stats['files'].add(change['file_path'])
            day_stats['operations'][change['operation']] += 1
            
            month_stats['files'].add(change['file_path'])
            month_stats['operations'][change['operation']] += 1

    def finalize(self) -> dict:
        """Not used - we export separately"""
        return {}

    def export(self, output_dir: Path) -> dict:
        """Export both daily and monthly files"""
        # Export daily
        daily_path = output_dir / "aggregations" / "temporal_daily.json"
        daily_path.parent.mkdir(parents=True, exist_ok=True)
        
        daily_output = self._format_temporal_data(self.daily_data, 'daily')
        with open(daily_path, 'w', encoding='utf-8') as f:
            json.dump(daily_output, f, indent=2, ensure_ascii=False)
        
        # Export monthly
        monthly_path = output_dir / "aggregations" / "temporal_monthly.json"
        monthly_output = self._format_temporal_data(self.monthly_data, 'monthly')
        with open(monthly_path, 'w', encoding='utf-8') as f:
            json.dump(monthly_output, f, indent=2, ensure_ascii=False)
        
        # Calculate manifest entries
        daily_size = daily_path.stat().st_size
        with open(daily_path, 'rb') as f:
            daily_sha256 = hashlib.sha256(f.read()).hexdigest()
        
        monthly_size = monthly_path.stat().st_size
        with open(monthly_path, 'rb') as f:
            monthly_sha256 = hashlib.sha256(f.read()).hexdigest()
        
        # Return manifest entry for both
        return {
            'temporal_daily': {
                'file': str(daily_path.relative_to(output_dir)),
                'schema_version': self.get_schema_version(),
                'record_count': len(daily_output['daily']),
                'file_size_bytes': daily_size,
                'sha256': daily_sha256,
                'production_ready': False,
                'preview': True
            },
            'temporal_monthly': {
                'file': str(monthly_path.relative_to(output_dir)),
                'schema_version': self.get_schema_version(),
                'record_count': len(monthly_output['monthly']),
                'file_size_bytes': monthly_size,
                'sha256': monthly_sha256,
                'production_ready': False,
                'preview': True
            }
        }

    def _format_temporal_data(self, data: dict, granularity: str) -> dict:
        """Format temporal data for export"""
        formatted = {}
        
        for period, stats in sorted(data.items()):
            # Get top contributors
            # We don't have per-author counts in this aggregation
            # For simplicity, we'll list all authors
            authors = list(stats['authors'])
            
            formatted[period] = {
                'total_commits': len(stats['commits']),
                'total_changes': stats['changes'],
                'unique_authors': len(stats['authors']),
                'active_files': len(stats['files']),
                'operations': dict(stats['operations']),
                'top_contributors': [{'email': email, 'commits': 'N/A'} for email in authors[:5]]
            }
        
        return {
            'schema_version': self.get_schema_version(),
            'granularity': granularity,
            'period_count': len(formatted),
            granularity: formatted
        }


class AuthorNetworkAggregator(DatasetAggregator):
    """
    Phase 2.3: Author Network
    Tracks author collaboration through co-commits
    """

    def __init__(self, analyzer: 'GitFileLifecycle'):
        super().__init__(analyzer)
        self.authors = set()
        self.file_authors = defaultdict(set)  # file -> set of authors who touched it
        self.author_stats = defaultdict(lambda: {
            'commits': set(),
            'files': set(),
            'first_commit': None,
            'last_commit': None
        })

    def get_name(self) -> str:
        return "author_network"

    def get_output_path(self, output_dir: Path) -> Path:
        return output_dir / "networks" / "author_network.json"

    def get_schema_version(self) -> str:
        return "1.0.0"

    def process_commit(self, commit: dict, changes: list):
        """Track author activities and file collaborations"""
        author_email = commit['author_email']
        commit_hash = commit['commit_hash']
        timestamp = commit['timestamp']
        
        self.authors.add(author_email)
        
        # Update author stats
        stats = self.author_stats[author_email]
        stats['commits'].add(commit_hash)
        
        if stats['first_commit'] is None:
            stats['first_commit'] = timestamp
        stats['last_commit'] = timestamp
        
        # Track which files this author touched
        for change in changes:
            file_path = change['file_path']
            stats['files'].add(file_path)
            self.file_authors[file_path].add(author_email)

    def finalize(self) -> dict:
        """Compute collaboration graph"""
        # Build nodes
        nodes = []
        author_to_id = {}
        
        for idx, author_email in enumerate(sorted(self.authors)):
            stats = self.author_stats[author_email]
            author_to_id[author_email] = idx
            
            nodes.append({
                'id': idx,
                'email': author_email,
                'name': author_email.split('@')[0],  # Simple name extraction
                'total_commits': len(stats['commits']),
                'files_touched': len(stats['files']),
                'first_seen': datetime.fromtimestamp(stats['first_commit'], tz=timezone.utc).isoformat() if stats['first_commit'] else None,
                'last_seen': datetime.fromtimestamp(stats['last_commit'], tz=timezone.utc).isoformat() if stats['last_commit'] else None,
                'active_days': round((stats['last_commit'] - stats['first_commit']) / 86400, 1) if stats['first_commit'] and stats['last_commit'] else 0
            })
        
        # Build edges (collaboration through shared files)
        edges = []
        edge_id = 0
        collaboration_counts = defaultdict(int)
        
        # Count collaborations (authors who worked on same files)
        for file_path, authors in self.file_authors.items():
            authors_list = sorted(authors)
            # Create edges between all pairs of authors who touched this file
            for i in range(len(authors_list)):
                for j in range(i + 1, len(authors_list)):
                    author1 = authors_list[i]
                    author2 = authors_list[j]
                    pair = tuple(sorted([author1, author2]))
                    collaboration_counts[pair] += 1
        
        # Create edges
        for (author1, author2), count in collaboration_counts.items():
            edges.append({
                'id': edge_id,
                'source': author_to_id[author1],
                'target': author_to_id[author2],
                'weight': count,
                'shared_files': count,
                'collaboration_strength': min(count / 10, 1.0)  # Normalized 0-1
            })
            edge_id += 1
        
        return {
            'schema_version': self.get_schema_version(),
            'graph_type': 'undirected',
            'node_count': len(nodes),
            'edge_count': len(edges),
            'nodes': nodes,
            'edges': edges,
            'metadata': {
                'generated_at': datetime.now(timezone.utc).isoformat(),
                'description': 'Author collaboration network based on shared file modifications'
            }
        }

    def _is_production_ready(self) -> bool:
        return False  # Preview feature in Phase 2


class GitFileLifecycle:
    """Main analyzer class with Phase 2 aggregator support"""

    VERSION = "2.0.0-phase2"

    def __init__(self, repo_path: str, enable_aggregations: bool = False):
        self.repo_path = Path(repo_path).resolve()
        self.enable_aggregations = enable_aggregations
        self.aggregators: List[DatasetAggregator] = []
        self.lifecycle_data = {}
        self.errors = []
        self.stats = {
            'total_commits': 0,
            'total_changes': 0,
            'skipped_unmerged': 0
        }

    def register_aggregator(self, aggregator: DatasetAggregator):
        """Register a dataset aggregator"""
        self.aggregators.append(aggregator)

    def analyze_repository(self) -> Dict:
        """Main analysis method"""
        print(f"Analyzing repository: {self.repo_path}")
        print(f"Aggregations enabled: {self.enable_aggregations}")
        
        # Initialize data structure
        files = defaultdict(list)
        
        # Get git log with file changes
        try:
            result = self._run_git_log()
            commits = self._parse_git_log(result)
            
            print(f"Processing {len(commits)} commits...")
            
            # Process each commit
            for commit_data in commits:
                self.stats['total_commits'] += 1
                
                # Get file changes for this commit
                changes = self._get_commit_changes(commit_data['commit_hash'])
                
                # Update lifecycle data
                for change in changes:
                    file_path = change['file_path']
                    event = {
                        'commit_hash': commit_data['commit_hash'],
                        'timestamp': commit_data['timestamp'],
                        'datetime': commit_data['datetime'],
                        'operation': change['operation'],
                        'author_name': commit_data['author_name'],
                        'author_email': commit_data['author_email'],
                        'commit_subject': commit_data['commit_subject']
                    }
                    
                    # Add optional fields for renames/copies
                    if 'old_path' in change:
                        event['old_path'] = change['old_path']
                    if 'similarity' in change:
                        event['similarity'] = change['similarity']
                    
                    files[file_path].append(event)
                    self.stats['total_changes'] += 1
                
                # Process aggregators if enabled
                if self.enable_aggregations:
                    for aggregator in self.aggregators:
                        try:
                            aggregator.process_commit(commit_data, changes)
                        except Exception as e:
                            self.errors.append(f"Aggregator {aggregator.get_name()} error: {e}")
            
            # Build final lifecycle data (v1.0 compatible)
            self.lifecycle_data = {
                'files': dict(files),
                'total_commits': self.stats['total_commits'],
                'total_changes': self.stats['total_changes'],
                'generated_at': datetime.now(timezone.utc).isoformat(),
                'repository_path': str(self.repo_path),
                'skipped_unmerged': self.stats['skipped_unmerged']
            }
            
            return self.lifecycle_data
            
        except Exception as e:
            self.errors.append(f"Analysis failed: {e}")
            raise

    def _run_git_log(self) -> str:
        """Run git log command"""
        cmd = [
            'git', 'log',
            '--all',
            '--pretty=format:%H|%at|%aI|%an|%ae|%s',
            '--date=iso-strict'
        ]
        
        result = subprocess.run(
            cmd,
            cwd=self.repo_path,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace'
        )
        
        if result.returncode != 0:
            raise RuntimeError(f"Git log failed: {result.stderr}")
        
        return result.stdout

    def _parse_git_log(self, log_output: str) -> List[Dict]:
        """Parse git log output into commit data"""
        commits = []
        
        for line in log_output.strip().split('\n'):
            if not line:
                continue
            
            parts = line.split('|', 5)
            if len(parts) < 6:
                continue
            
            commits.append({
                'commit_hash': parts[0],
                'timestamp': int(parts[1]),
                'datetime': parts[2],
                'author_name': parts[3],
                'author_email': parts[4],
                'commit_subject': parts[5]
            })
        
        return commits

    def _get_commit_changes(self, commit_hash: str) -> List[Dict]:
        """Get file changes for a specific commit"""
        cmd = [
            'git', 'show',
            '--pretty=',
            '--name-status',
            '--find-renames',
            '--find-copies',
            commit_hash
        ]
        
        result = subprocess.run(
            cmd,
            cwd=self.repo_path,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace'
        )
        
        if result.returncode != 0:
            return []
        
        changes = []
        for line in result.stdout.strip().split('\n'):
            if not line:
                continue
            
            parts = line.split('\t')
            if len(parts) < 2:
                continue
            
            status = parts[0]
            
            # Handle different operation types
            if status.startswith('R'):
                # Rename: R100 old_path new_path
                if len(parts) >= 3:
                    similarity = int(status[1:]) if len(status) > 1 else 100
                    changes.append({
                        'file_path': parts[2],
                        'operation': 'R',
                        'old_path': parts[1],
                        'similarity': similarity
                    })
            elif status.startswith('C'):
                # Copy: C100 old_path new_path
                if len(parts) >= 3:
                    similarity = int(status[1:]) if len(status) > 1 else 100
                    changes.append({
                        'file_path': parts[2],
                        'operation': 'C',
                        'old_path': parts[1],
                        'similarity': similarity
                    })
            else:
                # Normal operations: A, M, D, T
                operation = status[0]
                if operation in ['A', 'M', 'D', 'T']:
                    changes.append({
                        'file_path': parts[1],
                        'operation': operation
                    })
                elif operation == 'U':
                    # Unmerged - skip
                    self.stats['skipped_unmerged'] += 1
        
        return changes

    def export_results(self, output_dir: Path) -> Dict[str, Any]:
        """Export all results including aggregations"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Export core lifecycle data (v1.0 compatible)
        lifecycle_file = output_dir / "file_lifecycle.json"
        with open(lifecycle_file, 'w', encoding='utf-8') as f:
            json.dump(self.lifecycle_data, f, indent=2, ensure_ascii=False)
        
        # Calculate lifecycle file stats
        lifecycle_size = lifecycle_file.stat().st_size
        with open(lifecycle_file, 'rb') as f:
            lifecycle_sha256 = hashlib.sha256(f.read()).hexdigest()
        
        # Export errors if any
        if self.errors:
            error_file = output_dir / "file_lifecycle_errors.txt"
            with open(error_file, 'w', encoding='utf-8') as f:
                for error in self.errors:
                    f.write(f"{error}\n")
        
        # Initialize manifest
        manifest = {
            'generator_version': self.VERSION,
            'generated_at': datetime.now(timezone.utc).isoformat(),
            'repository': str(self.repo_path),
            'datasets': {
                'core_lifecycle': {
                    'file': 'file_lifecycle.json',
                    'schema_version': '1.0.0',
                    'format': 'nested',
                    'record_count': len(self.lifecycle_data['files']),
                    'file_size_bytes': lifecycle_size,
                    'sha256': lifecycle_sha256,
                    'production_ready': True
                }
            }
        }
        
        # Export aggregations if enabled
        if self.enable_aggregations:
            print("\nExporting aggregations...")
            for aggregator in self.aggregators:
                try:
                    print(f"  - {aggregator.get_name()}")
                    result = aggregator.export(output_dir)
                    
                    # Handle both single and multiple manifest entries
                    if isinstance(result, dict):
                        if 'file' in result:
                            # Single entry
                            manifest['datasets'][aggregator.get_name()] = result
                        else:
                            # Multiple entries (e.g., temporal has daily and monthly)
                            manifest['datasets'].update(result)
                            
                except Exception as e:
                    self.errors.append(f"Failed to export {aggregator.get_name()}: {e}")
                    print(f"    ERROR: {e}")
        
        # Write manifest
        manifest_file = output_dir / "manifest.json"
        with open(manifest_file, 'w', encoding='utf-8') as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
        
        return manifest


def create_output_directory(base_name: str = "git_analysis_output") -> Path:
    """Create timestamped output directory"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"{base_name}_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def main():
    parser = argparse.ArgumentParser(
        description='Git File Lifecycle Analyzer v2.0.0-phase2',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        'repo_path',
        help='Path to git repository'
    )
    
    parser.add_argument(
        '--enable-aggregations',
        action='store_true',
        help='Enable Phase 2 aggregations (file metadata, temporal, author network)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        help='Custom output directory (default: git_analysis_output_TIMESTAMP)'
    )
    
    parser.add_argument(
        '--version',
        action='version',
        version=f'Git File Lifecycle Analyzer v{GitFileLifecycle.VERSION}'
    )
    
    args = parser.parse_args()
    
    # Validate repository
    if not os.path.isdir(args.repo_path):
        print(f"Error: Repository path not found: {args.repo_path}")
        sys.exit(1)
    
    git_dir = Path(args.repo_path) / '.git'
    if not git_dir.exists():
        print(f"Error: Not a git repository: {args.repo_path}")
        sys.exit(1)
    
    # Create analyzer
    analyzer = GitFileLifecycle(args.repo_path, enable_aggregations=args.enable_aggregations)
    
    # Register Phase 2 aggregators if enabled
    if args.enable_aggregations:
        print("\n=== Phase 2 Aggregators Enabled ===")
        analyzer.register_aggregator(FileMetadataAggregator(analyzer))
        analyzer.register_aggregator(TemporalAggregator(analyzer))
        analyzer.register_aggregator(AuthorNetworkAggregator(analyzer))
        print("  - File Metadata Index")
        print("  - Temporal Aggregations (Daily/Monthly)")
        print("  - Author Network")
        print("=" * 35)
    
    # Run analysis
    try:
        print("\nStarting analysis...")
        start_time = datetime.now()
        
        analyzer.analyze_repository()
        
        elapsed = datetime.now() - start_time
        print(f"\nAnalysis complete in {elapsed.total_seconds():.2f}s")
        print(f"  Commits processed: {analyzer.stats['total_commits']}")
        print(f"  Total changes: {analyzer.stats['total_changes']}")
        print(f"  Files tracked: {len(analyzer.lifecycle_data['files'])}")
        
        # Export results
        if args.output_dir:
            output_dir = Path(args.output_dir)
        else:
            output_dir = create_output_directory()
        
        print(f"\nExporting to: {output_dir}")
        manifest = analyzer.export_results(output_dir)
        
        print("\n=== Generated Files ===")
        print(f"  ✓ file_lifecycle.json ({manifest['datasets']['core_lifecycle']['record_count']} files)")
        
        if args.enable_aggregations:
            if 'file_metadata' in manifest['datasets']:
                print(f"  ✓ metadata/file_index.json ({manifest['datasets']['file_metadata']['record_count']} files)")
            if 'temporal_daily' in manifest['datasets']:
                print(f"  ✓ aggregations/temporal_daily.json ({manifest['datasets']['temporal_daily']['record_count']} days)")
            if 'temporal_monthly' in manifest['datasets']:
                print(f"  ✓ aggregations/temporal_monthly.json ({manifest['datasets']['temporal_monthly']['record_count']} months)")
            if 'author_network' in manifest['datasets']:
                print(f"  ✓ networks/author_network.json ({manifest['datasets']['author_network']['record_count']} authors)")
        
        print(f"  ✓ manifest.json")
        
        if analyzer.errors:
            print(f"  ⚠ file_lifecycle_errors.txt ({len(analyzer.errors)} errors)")
        
        print("\n✅ Analysis complete!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
