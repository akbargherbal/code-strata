#!/usr/bin/env python3
"""
Git File Lifecycle Analyzer - Version 2.0.0-phase3

Phase 3 Implementation: Advanced Datasets
- Co-change Network (files that change together)
- File Hierarchy Aggregation (directory-level statistics)
- Milestone/Tag Events (git tags and lifecycle milestones)

Maintains 100% backward compatibility with v1.0 and v2.0 schemas.
"""

import subprocess
import json
import os
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Set, Tuple
from collections import defaultdict, Counter
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
        elif 'directories' in data:
            return len(data['directories'])
        elif 'tags' in data:
            return len(data['tags'])
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


class CochangeNetworkAggregator(DatasetAggregator):
    """
    Phase 3.1: Co-change Network
    Tracks files that change together in the same commits
    """

    def __init__(self, analyzer: 'GitFileLifecycle', min_cochanges: int = 2):
        super().__init__(analyzer)
        self.min_cochanges = min_cochanges
        self.file_changes = defaultdict(int)  # file -> total changes
        self.cochange_counts = defaultdict(int)  # (file1, file2) -> count
        self.cochange_commits = defaultdict(list)  # (file1, file2) -> [commit_hashes]
        self.cochange_timestamps = defaultdict(lambda: {'first': None, 'last': None})

    def get_name(self) -> str:
        return "cochange_network"

    def get_output_path(self, output_dir: Path) -> Path:
        return output_dir / "networks" / "cochange_network.json"

    def get_schema_version(self) -> str:
        return "1.0.0"

    def process_commit(self, commit: dict, changes: list):
        """Track files that change together"""
        commit_hash = commit['commit_hash']
        timestamp = commit['timestamp']
        
        # Get all files in this commit
        files_in_commit = [change['file_path'] for change in changes]
        
        # Update file change counts
        for file_path in files_in_commit:
            self.file_changes[file_path] += 1
        
        # Track co-changes (pairs of files changed together)
        if len(files_in_commit) > 1:
            for i in range(len(files_in_commit)):
                for j in range(i + 1, len(files_in_commit)):
                    file1, file2 = sorted([files_in_commit[i], files_in_commit[j]])
                    pair = (file1, file2)
                    
                    self.cochange_counts[pair] += 1
                    self.cochange_commits[pair].append(commit_hash)
                    
                    # Track timestamps
                    ts_data = self.cochange_timestamps[pair]
                    if ts_data['first'] is None:
                        ts_data['first'] = timestamp
                    ts_data['last'] = timestamp

    def finalize(self) -> dict:
        """Build co-change network graph"""
        # Build nodes (files with their metrics)
        nodes = []
        file_to_id = {}
        
        # Sort files by change count for consistent IDs
        sorted_files = sorted(self.file_changes.keys())
        for idx, file_path in enumerate(sorted_files):
            file_to_id[file_path] = idx
            
            # Calculate coupling partners
            coupling_partners = sum(1 for pair in self.cochange_counts.keys() 
                                   if file_path in pair and self.cochange_counts[pair] >= self.min_cochanges)
            
            # Calculate average coupling strength
            coupling_strengths = []
            for pair, count in self.cochange_counts.items():
                if file_path in pair and count >= self.min_cochanges:
                    file1, file2 = pair
                    other_file = file2 if file1 == file_path else file1
                    strength = count / min(self.file_changes[file1], self.file_changes[file2])
                    coupling_strengths.append(strength)
            
            avg_coupling = sum(coupling_strengths) / len(coupling_strengths) if coupling_strengths else 0
            
            nodes.append({
                'id': idx,
                'file_path': file_path,
                'total_changes': self.file_changes[file_path],
                'coupling_partners': coupling_partners,
                'avg_coupling_strength': round(avg_coupling, 3)
            })
        
        # Build edges (co-changes meeting threshold)
        edges = []
        edge_id = 0
        
        for (file1, file2), count in self.cochange_counts.items():
            if count >= self.min_cochanges:
                # Calculate coupling strength (Jaccard-like metric)
                coupling_strength = count / min(self.file_changes[file1], self.file_changes[file2])
                
                ts_data = self.cochange_timestamps[(file1, file2)]
                
                edges.append({
                    'id': edge_id,
                    'source': file_to_id[file1],
                    'target': file_to_id[file2],
                    'weight': count,
                    'coupling_strength': round(coupling_strength, 3),
                    'co_change_count': count,
                    'source_total_changes': self.file_changes[file1],
                    'target_total_changes': self.file_changes[file2],
                    'first_co_change': datetime.fromtimestamp(ts_data['first'], tz=timezone.utc).isoformat() if ts_data['first'] else None,
                    'last_co_change': datetime.fromtimestamp(ts_data['last'], tz=timezone.utc).isoformat() if ts_data['last'] else None,
                    'co_change_commits': self.cochange_commits[(file1, file2)][:10]  # Limit to first 10
                })
                edge_id += 1
        
        return {
            'schema_version': self.get_schema_version(),
            'network_type': 'file_coupling',
            'analysis_parameters': {
                'min_co_changes': self.min_cochanges
            },
            'node_count': len(nodes),
            'edge_count': len(edges),
            'nodes': nodes,
            'edges': edges,
            'metadata': {
                'generated_at': datetime.now(timezone.utc).isoformat(),
                'description': 'File co-change network showing files that frequently change together'
            }
        }

    def _is_production_ready(self) -> bool:
        return False  # Preview feature in Phase 3


class FileHierarchyAggregator(DatasetAggregator):
    """
    Phase 3.2: File Hierarchy Aggregation
    Generates directory-level statistics and metrics
    """

    def __init__(self, analyzer: 'GitFileLifecycle'):
        super().__init__(analyzer)
        self.directory_stats = defaultdict(lambda: {
            'files': set(),
            'commits': set(),
            'authors': set(),
            'changes': 0,
            'operations': defaultdict(int),
            'first_seen': None,
            'last_modified': None
        })

    def get_name(self) -> str:
        return "file_hierarchy"

    def get_output_path(self, output_dir: Path) -> Path:
        return output_dir / "aggregations" / "directory_stats.json"

    def get_schema_version(self) -> str:
        return "1.0.0"

    def process_commit(self, commit: dict, changes: list):
        """Aggregate data by directory"""
        for change in changes:
            file_path = change['file_path']
            operation = change['operation']
            timestamp = commit['timestamp']
            
            # Get directory path (empty string for root-level files)
            dir_path = str(Path(file_path).parent) if '/' in file_path else ''
            if dir_path == '.':
                dir_path = ''
            
            # Also track all parent directories
            dirs_to_update = [dir_path]
            if dir_path:
                parts = dir_path.split('/')
                for i in range(len(parts)):
                    parent = '/'.join(parts[:i+1])
                    if parent and parent not in dirs_to_update:
                        dirs_to_update.append(parent)
            
            # Update statistics for directory and all parents
            for directory in dirs_to_update:
                stats = self.directory_stats[directory]
                stats['files'].add(file_path)
                stats['commits'].add(commit['commit_hash'])
                stats['authors'].add(commit['author_email'])
                stats['changes'] += 1
                stats['operations'][operation] += 1
                
                if stats['first_seen'] is None:
                    stats['first_seen'] = timestamp
                stats['last_modified'] = timestamp

    def finalize(self) -> dict:
        """Compute directory hierarchy"""
        directories_data = {}
        
        for dir_path, stats in self.directory_stats.items():
            # Calculate metrics
            age_seconds = (stats['last_modified'] - stats['first_seen']) if stats['first_seen'] and stats['last_modified'] else 0
            age_days = age_seconds / 86400
            
            # Determine depth
            depth = len(dir_path.split('/')) if dir_path else 0
            
            # Get immediate children (files or subdirectories)
            children_files = []
            children_dirs = set()
            
            for file_path in stats['files']:
                file_dir = str(Path(file_path).parent)
                if file_dir == '.':
                    file_dir = ''
                
                if file_dir == dir_path:
                    # This file is directly in this directory
                    children_files.append(file_path)
                elif file_dir.startswith(dir_path + '/') if dir_path else True:
                    # This file is in a subdirectory
                    if dir_path:
                        subdir = file_dir[len(dir_path)+1:].split('/')[0]
                        children_dirs.add(f"{dir_path}/{subdir}")
                    else:
                        subdir = file_dir.split('/')[0] if '/' in file_dir else file_dir
                        if subdir:
                            children_dirs.add(subdir)
            
            directories_data[dir_path if dir_path else '/'] = {
                'path': dir_path if dir_path else '/',
                'depth': depth,
                'total_files': len(stats['files']),
                'direct_files': len(children_files),
                'subdirectories': sorted(list(children_dirs)),
                'total_commits': len(stats['commits']),
                'total_changes': stats['changes'],
                'unique_authors': len(stats['authors']),
                'operations': dict(stats['operations']),
                'first_seen': datetime.fromtimestamp(stats['first_seen'], tz=timezone.utc).isoformat() if stats['first_seen'] else None,
                'last_modified': datetime.fromtimestamp(stats['last_modified'], tz=timezone.utc).isoformat() if stats['last_modified'] else None,
                'age_days': round(age_days, 1),
                'activity_score': round(stats['changes'] / age_days, 3) if age_days > 0 else 0
            }
        
        return {
            'schema_version': self.get_schema_version(),
            'total_directories': len(directories_data),
            'generation_method': 'streaming',
            'directories': directories_data,
            'metadata': {
                'generated_at': datetime.now(timezone.utc).isoformat(),
                'description': 'Directory-level aggregations and hierarchy statistics'
            }
        }

    def _is_production_ready(self) -> bool:
        return False  # Preview feature in Phase 3


class MilestoneAggregator(DatasetAggregator):
    """
    Phase 3.3: Milestone/Tag Events
    Tracks git tags and significant lifecycle milestones
    """

    def __init__(self, analyzer: 'GitFileLifecycle'):
        super().__init__(analyzer)
        self.tags = []  # Will be populated from git tags
        self.file_milestones = defaultdict(list)
        self.commit_to_files = defaultdict(set)  # commit_hash -> set of files
        self.file_timeline = defaultdict(list)  # file -> [(timestamp, commit_hash, operation)]

    def get_name(self) -> str:
        return "milestones"

    def get_output_path(self, output_dir: Path) -> Path:
        return output_dir / "milestones" / "release_snapshots.json"

    def get_schema_version(self) -> str:
        return "1.0.0"

    def process_commit(self, commit: dict, changes: list):
        """Track file events for milestone detection"""
        commit_hash = commit['commit_hash']
        timestamp = commit['timestamp']
        
        for change in changes:
            file_path = change['file_path']
            operation = change['operation']
            
            self.commit_to_files[commit_hash].add(file_path)
            self.file_timeline[file_path].append((timestamp, commit_hash, operation))

    def _fetch_git_tags(self):
        """Fetch git tags from repository"""
        try:
            result = subprocess.run(
                ['git', '-C', str(self.analyzer.repo_path), 'tag', '-l', '--format=%(refname:short)|%(creatordate:iso)|%(*objectname)|%(objectname)'],
                capture_output=True,
                text=True,
                check=True
            )
            
            for line in result.stdout.strip().split('\n'):
                if not line:
                    continue
                
                parts = line.split('|')
                if len(parts) >= 2:
                    tag_name = parts[0]
                    tag_date = parts[1]
                    # Get commit hash (annotated tags have *objectname, lightweight tags use objectname)
                    commit_hash = parts[2] if len(parts) > 2 and parts[2] else parts[3] if len(parts) > 3 else None
                    
                    if commit_hash:
                        self.tags.append({
                            'name': tag_name,
                            'date': tag_date,
                            'commit_hash': commit_hash
                        })
        except subprocess.CalledProcessError as e:
            self.errors.append(f"Error fetching git tags: {e}")

    def finalize(self) -> dict:
        """Generate milestone snapshots"""
        # Fetch git tags
        self._fetch_git_tags()
        
        # Build tag snapshots
        tag_snapshots = []
        for tag in self.tags:
            commit_hash = tag['commit_hash']
            files_at_tag = self.commit_to_files.get(commit_hash, set())
            
            # Count files by operation at this tag
            file_states = {'active': 0, 'added': 0, 'modified': 0, 'deleted': 0}
            
            for file_path in files_at_tag:
                # Look for operation in timeline
                for ts, ch, op in self.file_timeline[file_path]:
                    if ch == commit_hash:
                        if op == 'A':
                            file_states['added'] += 1
                        elif op == 'M':
                            file_states['modified'] += 1
                        elif op == 'D':
                            file_states['deleted'] += 1
                        file_states['active'] += 1
                        break
            
            tag_snapshots.append({
                'tag_name': tag['name'],
                'date': tag['date'],
                'commit_hash': commit_hash,
                'files_in_commit': len(files_at_tag),
                'file_states': file_states,
                'sample_files': sorted(list(files_at_tag))[:20]  # First 20 files
            })
        
        # Detect file lifecycle milestones
        file_milestones_data = {}
        for file_path, timeline in self.file_timeline.items():
            milestones = []
            
            # Sort timeline by timestamp
            sorted_timeline = sorted(timeline, key=lambda x: x[0])
            
            if sorted_timeline:
                # Creation milestone
                first_event = sorted_timeline[0]
                milestones.append({
                    'type': 'created',
                    'timestamp': datetime.fromtimestamp(first_event[0], tz=timezone.utc).isoformat(),
                    'commit_hash': first_event[1],
                    'operation': first_event[2]
                })
                
                # Detect activity bursts (periods of high activity)
                if len(sorted_timeline) > 10:
                    # Simple burst detection: 5+ events within 7 days
                    for i in range(len(sorted_timeline) - 4):
                        window_events = sorted_timeline[i:i+5]
                        time_span = window_events[-1][0] - window_events[0][0]
                        if time_span <= 7 * 86400:  # 7 days in seconds
                            milestones.append({
                                'type': 'activity_burst',
                                'timestamp': datetime.fromtimestamp(window_events[0][0], tz=timezone.utc).isoformat(),
                                'duration_days': round(time_span / 86400, 1),
                                'event_count': 5
                            })
                            break  # Only report first burst
                
                # Last modification
                last_event = sorted_timeline[-1]
                if len(sorted_timeline) > 1:  # Only if not same as creation
                    milestones.append({
                        'type': 'last_modified',
                        'timestamp': datetime.fromtimestamp(last_event[0], tz=timezone.utc).isoformat(),
                        'commit_hash': last_event[1],
                        'operation': last_event[2]
                    })
            
            if milestones:
                file_milestones_data[file_path] = {
                    'total_events': len(sorted_timeline),
                    'milestones': milestones
                }
        
        return {
            'schema_version': self.get_schema_version(),
            'total_tags': len(tag_snapshots),
            'total_files_with_milestones': len(file_milestones_data),
            'tags': tag_snapshots,
            'file_milestones': file_milestones_data,
            'metadata': {
                'generated_at': datetime.now(timezone.utc).isoformat(),
                'description': 'Git tags/releases and file lifecycle milestones'
            }
        }

    def _is_production_ready(self) -> bool:
        return False  # Preview feature in Phase 3


class GitFileLifecycle:
    """Main analyzer class with Phase 3 aggregator support"""

    VERSION = "2.0.0-phase3"

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
                    event = {
                        'commit_hash': change['commit_hash'],
                        'timestamp': commit_data['timestamp'],
                        'datetime': commit_data['datetime'],
                        'operation': change['operation'],
                        'author_name': commit_data['author_name'],
                        'author_email': commit_data['author_email'],
                        'commit_subject': commit_data['subject']
                    }
                    
                    # Add optional fields for renames/copies
                    if change.get('old_path'):
                        event['old_path'] = change['old_path']
                    if change.get('similarity'):
                        event['similarity'] = change['similarity']
                    
                    files[change['file_path']].append(event)
                    self.stats['total_changes'] += 1
                
                # Call aggregators if enabled
                if self.enable_aggregations and self.aggregators:
                    for aggregator in self.aggregators:
                        aggregator.process_commit(commit_data, changes)
            
            # Build final data structure
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
            self.errors.append(f"Analysis failed: {str(e)}")
            raise

    def _run_git_log(self) -> str:
        """Run git log command"""
        cmd = [
            'git', '-C', str(self.repo_path), 'log',
            '--all',
            '--format=%H|%at|%aI|%an|%ae|%s',
            '--reverse'
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return result.stdout

    def _parse_git_log(self, log_output: str) -> List[Dict]:
        """Parse git log output"""
        commits = []
        for line in log_output.strip().split('\n'):
            if not line:
                continue
            
            parts = line.split('|', 5)
            if len(parts) >= 6:
                commits.append({
                    'commit_hash': parts[0],
                    'timestamp': int(parts[1]),
                    'datetime': parts[2],
                    'author_name': parts[3],
                    'author_email': parts[4],
                    'subject': parts[5]
                })
        
        return commits

    def _get_commit_changes(self, commit_hash: str) -> List[Dict]:
        """Get file changes for a specific commit"""
        cmd = [
            'git', '-C', str(self.repo_path), 'diff-tree',
            '--no-commit-id', '--name-status', '-r', '-M', '-C',
            '--root',  # Handle initial commit
            commit_hash
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            changes = []
            
            for line in result.stdout.strip().split('\n'):
                if not line:
                    continue
                
                parts = line.split('\t')
                if len(parts) < 2:
                    continue
                
                status = parts[0]
                
                # Handle different change types
                if status.startswith('R'):
                    # Rename
                    similarity = status[1:] if len(status) > 1 else '100'
                    changes.append({
                        'commit_hash': commit_hash,
                        'operation': 'R',
                        'old_path': parts[1],
                        'file_path': parts[2],
                        'similarity': int(similarity)
                    })
                elif status.startswith('C'):
                    # Copy
                    similarity = status[1:] if len(status) > 1 else '100'
                    changes.append({
                        'commit_hash': commit_hash,
                        'operation': 'C',
                        'old_path': parts[1],
                        'file_path': parts[2],
                        'similarity': int(similarity)
                    })
                else:
                    # A, M, D, T
                    changes.append({
                        'commit_hash': commit_hash,
                        'operation': status[0],
                        'file_path': parts[1]
                    })
            
            return changes
            
        except subprocess.CalledProcessError as e:
            # Handle unmerged files
            if 'unmerged' in e.stderr.lower():
                self.stats['skipped_unmerged'] += 1
                return []
            raise

    def export_results(self, output_dir: Path) -> Dict:
        """Export all results including aggregations"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Export core lifecycle data (unchanged)
        lifecycle_path = output_dir / 'file_lifecycle.json'
        with open(lifecycle_path, 'w', encoding='utf-8') as f:
            json.dump(self.lifecycle_data, f, indent=2, ensure_ascii=False)
        
        # Calculate core lifecycle checksum
        with open(lifecycle_path, 'rb') as f:
            lifecycle_sha256 = hashlib.sha256(f.read()).hexdigest()
        
        # Export errors if any
        if self.errors:
            errors_path = output_dir / 'file_lifecycle_errors.txt'
            with open(errors_path, 'w', encoding='utf-8') as f:
                for error in self.errors:
                    f.write(f"{error}\n")
        
        # Build manifest
        manifest = {
            'generator_version': self.VERSION,
            'generated_at': datetime.now(timezone.utc).isoformat(),
            'repository_path': str(self.repo_path),
            'datasets': {
                'core_lifecycle': {
                    'file': 'file_lifecycle.json',
                    'schema_version': '1.0.0',
                    'format': 'nested',
                    'record_count': len(self.lifecycle_data['files']),
                    'file_size_bytes': lifecycle_path.stat().st_size,
                    'sha256': lifecycle_sha256,
                    'production_ready': True
                }
            }
        }
        
        # Export aggregations if enabled
        if self.enable_aggregations and self.aggregators:
            print("\nExporting aggregations...")
            for aggregator in self.aggregators:
                print(f"  - {aggregator.get_name()}...")
                try:
                    aggregator_manifest = aggregator.export(output_dir)
                    # Handle aggregators that return multiple datasets (like TemporalAggregator)
                    if isinstance(aggregator_manifest, dict):
                        for key, value in aggregator_manifest.items():
                            if key not in ['file', 'schema_version']:
                                manifest['datasets'][key] = value
                            else:
                                manifest['datasets'][aggregator.get_name()] = aggregator_manifest
                                break
                except Exception as e:
                    error_msg = f"Failed to export {aggregator.get_name()}: {str(e)}"
                    self.errors.append(error_msg)
                    print(f"  ⚠ {error_msg}")
        
        # Write manifest
        manifest_path = output_dir / 'manifest.json'
        with open(manifest_path, 'w', encoding='utf-8') as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
        
        return manifest


def main():
    parser = argparse.ArgumentParser(
        description='Git File Lifecycle Analyzer v2.0.0-phase3',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic analysis (v1.0 compatible)
  %(prog)s /path/to/repo

  # Enable all Phase 2 features
  %(prog)s /path/to/repo --enable-aggregations --phase2

  # Enable all Phase 3 features
  %(prog)s /path/to/repo --enable-aggregations --phase3

  # Enable all features (Phase 2 + Phase 3)
  %(prog)s /path/to/repo --enable-aggregations --phase2 --phase3
        """
    )
    
    parser.add_argument('repo_path', help='Path to git repository')
    parser.add_argument('--enable-aggregations', action='store_true',
                       help='Enable dataset aggregations')
    parser.add_argument('--phase2', action='store_true',
                       help='Enable Phase 2 aggregators (file metadata, temporal, author network)')
    parser.add_argument('--phase3', action='store_true',
                       help='Enable Phase 3 aggregators (co-change network, hierarchy, milestones)')
    parser.add_argument('--output', type=str,
                       help='Custom output directory (default: git_analysis_output_TIMESTAMP)')
    parser.add_argument('--version', action='version',
                       version=f'Git File Lifecycle Analyzer {GitFileLifecycle.VERSION}')
    
    args = parser.parse_args()
    
    try:
        # Create analyzer
        analyzer = GitFileLifecycle(args.repo_path, enable_aggregations=args.enable_aggregations)
        
        # Register Phase 2 aggregators
        if args.enable_aggregations and args.phase2:
            print("Registering Phase 2 aggregators...")
            analyzer.register_aggregator(FileMetadataAggregator(analyzer))
            analyzer.register_aggregator(TemporalAggregator(analyzer))
            analyzer.register_aggregator(AuthorNetworkAggregator(analyzer))
        
        # Register Phase 3 aggregators
        if args.enable_aggregations and args.phase3:
            print("Registering Phase 3 aggregators...")
            analyzer.register_aggregator(CochangeNetworkAggregator(analyzer))
            analyzer.register_aggregator(FileHierarchyAggregator(analyzer))
            analyzer.register_aggregator(MilestoneAggregator(analyzer))
        
        # Run analysis
        data = analyzer.analyze_repository()
        
        # Determine output directory
        if args.output:
            output_dir = Path(args.output)
        else:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_dir = Path(f'git_analysis_output_{timestamp}')
        
        # Export results
        manifest = analyzer.export_results(output_dir)
        
        # Print summary
        print(f"\n📊 Analysis Summary")
        print(f"{'='*60}")
        print(f"Repository: {args.repo_path}")
        print(f"Total commits: {data['total_commits']}")
        print(f"Total file changes: {data['total_changes']}")
        print(f"Unique files: {len(data['files'])}")
        print(f"\n📁 Output: {output_dir}/")
        print(f"  ✓ file_lifecycle.json ({len(data['files'])} files)")
        
        if args.enable_aggregations:
            if 'file_metadata' in manifest['datasets']:
                print(f"  ✓ metadata/file_index.json ({manifest['datasets']['file_metadata']['record_count']} files)")
            if 'temporal_daily' in manifest['datasets']:
                print(f"  ✓ aggregations/temporal_daily.json ({manifest['datasets']['temporal_daily']['record_count']} days)")
            if 'temporal_monthly' in manifest['datasets']:
                print(f"  ✓ aggregations/temporal_monthly.json ({manifest['datasets']['temporal_monthly']['record_count']} months)")
            if 'author_network' in manifest['datasets']:
                print(f"  ✓ networks/author_network.json ({manifest['datasets']['author_network']['record_count']} authors)")
            if 'cochange_network' in manifest['datasets']:
                print(f"  ✓ networks/cochange_network.json ({manifest['datasets']['cochange_network']['record_count']} files)")
            if 'file_hierarchy' in manifest['datasets']:
                print(f"  ✓ aggregations/directory_stats.json ({manifest['datasets']['file_hierarchy']['record_count']} directories)")
            if 'milestones' in manifest['datasets']:
                print(f"  ✓ milestones/release_snapshots.json ({manifest['datasets']['milestones']['record_count']} tags)")
        
        print(f"  ✓ manifest.json")
        
        if analyzer.errors:
            print(f"  ⚠ file_lifecycle_errors.txt ({len(analyzer.errors)} errors)")
        
        print("\n✅ Analysis complete!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()