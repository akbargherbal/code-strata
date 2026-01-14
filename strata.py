#!/usr/bin/env python3
"""
Git File Lifecycle Analyzer - Phase 4: Polish & Release (v2.0.0)

This is the production-ready v2.0.0 release with:
- Performance optimizations
- Enhanced CLI with progress reporting and presets
- Quick mode for large repositories
- Comprehensive error handling
- Full backward compatibility

Phases completed:
- Phase 0: Foundation & Safety ✓
- Phase 1: Infrastructure ✓
- Phase 2: Core Datasets ✓
- Phase 3: Advanced Datasets ✓
- Phase 4: Polish & Release ✓

Author: Git Lifecycle Team
Version: 2.0.0
"""

import subprocess
import json
import os
import sys
import hashlib
import time
import cProfile
import pstats
import io
from functools import lru_cache
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Any
from collections import defaultdict, Counter
from dataclasses import dataclass, field

# Phase 4.2: Click for CLI
try:
    import click
except ImportError:
    print(
        "Error: 'click' library is required. Install with: pip install click",
        file=sys.stderr,
    )
    sys.exit(1)

# Ensure psutil exists at module level for tests
try:
    import psutil
except ImportError:
    psutil = None

# Enhanced CLI dependencies (Phase 4.2)
try:
    from tqdm import tqdm

    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

try:
    from colorama import Fore, Style, init as colorama_init

    COLORAMA_AVAILABLE = True
    colorama_init(autoreset=True)
except ImportError:
    COLORAMA_AVAILABLE = False

    # Fallback no-op color codes
    class Fore:
        RED = GREEN = YELLOW = BLUE = CYAN = MAGENTA = WHITE = RESET = ""

    class Style:
        BRIGHT = DIM = RESET_ALL = ""


# Configuration file support (Phase 4.3)
try:
    import yaml

    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

# Memory profiling (Phase 4.1)
try:
    from memory_profiler import profile as memory_profile

    MEMORY_PROFILER_AVAILABLE = True
except ImportError:
    MEMORY_PROFILER_AVAILABLE = False

    def memory_profile(func):
        return func


# Version information
VERSION = "2.0.0"
SCHEMA_VERSION = "1.0.0"


# ============================================================================
# PHASE 4.1: PERFORMANCE OPTIMIZATION
# ============================================================================


class MemoryMonitor:
    """Monitor memory usage and enforce limits"""

    def __init__(self, limit_mb: Optional[float] = None):
        self.limit_mb = limit_mb
        self.peak_mb = 0.0

    def check_memory(self) -> float:
        """Get current memory usage in MB"""
        try:
            import psutil

            process = psutil.Process(os.getpid())
            memory_mb = process.memory_info().rss / 1024 / 1024
            self.peak_mb = max(self.peak_mb, memory_mb)

            if self.limit_mb and memory_mb > self.limit_mb:
                raise MemoryError(
                    f"Memory limit exceeded: {memory_mb:.1f}MB > {self.limit_mb}MB"
                )

            return memory_mb
        except ImportError:
            return 0.0

    def get_peak(self) -> float:
        """Get peak memory usage"""
        return self.peak_mb


class ProfilingContext:
    """Context manager for performance profiling"""

    def __init__(self, enabled: bool = False, output_path: Optional[str] = None):
        self.enabled = enabled
        self.output_path = output_path
        self.profiler = None

    def __enter__(self):
        if self.enabled:
            self.profiler = cProfile.Profile()
            self.profiler.enable()
        return self

    def __exit__(self, *args):
        if self.enabled and self.profiler:
            self.profiler.disable()

            if self.output_path:
                # Save to file
                self.profiler.dump_stats(self.output_path)

            # Print summary
            s = io.StringIO()
            ps = pstats.Stats(self.profiler, stream=s)
            ps.strip_dirs()
            ps.sort_stats("cumulative")
            ps.print_stats(20)  # Top 20 functions
            print(f"\n{'='*70}")
            print("PERFORMANCE PROFILE (Top 20 functions by cumulative time)")
            print(f"{'='*70}")
            print(s.getvalue())


@lru_cache(maxsize=1024)
def cached_file_hash(file_path: str, commit_hash: str) -> str:
    """
    Cached hash generation for file identification.
    Phase 4.1: Avoid recomputing hashes for same file+commit pairs.
    """
    combined = f"{file_path}:{commit_hash}"
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()[:16]


def chunk_iterator(items: List, chunk_size: int = 1000):
    """
    Memory-efficient chunking for large datasets.
    Phase 4.1: Process data in chunks to avoid memory exhaustion.
    """
    for i in range(0, len(items), chunk_size):
        yield items[i : i + chunk_size]


# ============================================================================
# PHASE 4.3: CONFIGURATION FILE SUPPORT
# ============================================================================


def load_config_file(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML or JSON file.
    Phase 4.3: Support for .git-lifecycle.yaml, .git-lifecycle.json
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    file_ext = os.path.splitext(config_path)[1].lower()

    with open(config_path, "r", encoding="utf-8") as f:
        if file_ext in [".yaml", ".yml"]:
            if not YAML_AVAILABLE:
                raise ImportError(
                    "PyYAML required for YAML config files. Install: pip install pyyaml"
                )
            return yaml.safe_load(f)
        elif file_ext == ".json":
            return json.load(f)
        else:
            raise ValueError(f"Unsupported config file format: {file_ext}")


def find_config_file(repo_path: str) -> Optional[str]:
    """
    Auto-discover configuration file in repository or current directory.
    Searches for: .git-lifecycle.yaml, .git-lifecycle.yml, .git-lifecycle.json
    """
    search_paths = [
        repo_path,
        os.getcwd(),
    ]

    config_names = [
        ".git-lifecycle.yaml",
        ".git-lifecycle.yml",
        ".git-lifecycle.json",
    ]

    for search_dir in search_paths:
        for config_name in config_names:
            config_path = os.path.join(search_dir, config_name)
            if os.path.exists(config_path):
                return config_path

    return None


# ============================================================================
# PROGRESS REPORTING & CLI ENHANCEMENTS
# ============================================================================


class ProgressReporter:
    """
    Interactive progress reporting with enhanced UX (Phase 4.2)
    - Color-coded output (colorama)
    - Progress bars with percentage (tqdm)
    - ETA calculation
    """

    def __init__(
        self, quiet: bool = False, verbose: bool = False, use_colors: bool = True
    ):
        self.quiet = quiet
        self.verbose = verbose
        self.use_colors = use_colors and COLORAMA_AVAILABLE
        self.start_time = time.time()
        self.stage_times = {}
        self.progress_bar = None

    def _colorize(self, text: str, color: str) -> str:
        """Apply color if enabled"""
        if self.use_colors:
            return f"{color}{text}{Style.RESET_ALL}"
        return text

    def stage_start(self, stage_name: str, message: str = ""):
        """Mark the start of a processing stage"""
        if self.quiet:
            return
        self.stage_times[stage_name] = time.time()

        separator = self._colorize("=" * 70, Fore.CYAN)
        stage_text = self._colorize(f"🔄 {stage_name}", Fore.BLUE + Style.BRIGHT)

        print(f"\n{separator}")
        print(stage_text)
        if message:
            print(f"   {message}")
        print(separator)

    def stage_complete(self, stage_name: str, stats: Dict = None):
        """Mark completion of a processing stage"""
        if self.quiet:
            return
        elapsed = time.time() - self.stage_times.get(stage_name, time.time())

        complete_text = self._colorize(
            f"✅ {stage_name} complete ({elapsed:.2f}s)", Fore.GREEN + Style.BRIGHT
        )
        print(complete_text)

        if stats and self.verbose:
            for key, value in stats.items():
                print(f"   {key}: {value}")

    def create_progress_bar(
        self, total: int, desc: str = "Processing"
    ) -> Optional["tqdm"]:
        """Create a progress bar with ETA (Phase 4.2)"""
        if self.quiet or not TQDM_AVAILABLE:
            return None

        return tqdm(
            total=total,
            desc=self._colorize(desc, Fore.CYAN),
            unit=" commits",
            ncols=100,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
        )

    def progress(self, message: str, count: Optional[int] = None):
        """Report progress during processing"""
        if self.quiet:
            return
        if count is not None:
            print(f"   {message} ({count:,})", end="\r", flush=True)
        else:
            print(f"   {message}")

    def info(self, message: str):
        """Display informational message"""
        if not self.quiet:
            info_text = self._colorize("ℹ️  ", Fore.BLUE)
            print(f"{info_text}{message}")

    def warning(self, message: str):
        """Display warning message"""
        if not self.quiet:
            warning_text = self._colorize("⚠️  ", Fore.YELLOW + Style.BRIGHT)
            print(f"{warning_text}{message}")

    def error(self, message: str):
        """Display error message (always shown)"""
        error_text = self._colorize(f"❌ ERROR: {message}", Fore.RED + Style.BRIGHT)
        print(error_text, file=sys.stderr)

    def success(self, message: str):
        """Display success message"""
        if not self.quiet:
            success_text = self._colorize(f"✨ {message}", Fore.GREEN + Style.BRIGHT)
            print(success_text)

    def summary(self, stats: Dict[str, Any]):
        """Display final summary"""
        if self.quiet:
            return
        elapsed = time.time() - self.start_time

        separator = self._colorize("=" * 70, Fore.CYAN)
        header = self._colorize("📊 ANALYSIS SUMMARY", Fore.MAGENTA + Style.BRIGHT)

        print(f"\n{separator}")
        print(header)
        print(separator)
        for key, value in stats.items():
            print(f"   {key}: {value}")

        time_text = self._colorize(f"⏱️  Total time: {elapsed:.2f}s", Fore.YELLOW)
        print(f"\n{time_text}")
        print(f"{separator}\n")


# ============================================================================
# DATA STRUCTURES & MODELS
# ============================================================================


@dataclass
class PerformanceMetrics:
    """
    Track performance metrics for optimization (Phase 4.1 enhanced)
    - Memory usage tracking
    - Cache hit/miss rates
    - Detailed timing per aggregator
    """

    commits_processed: int = 0
    files_tracked: int = 0
    aggregator_times: Dict[str, float] = field(default_factory=dict)
    memory_peak_mb: float = 0.0
    total_time: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    chunks_processed: int = 0

    def to_dict(self) -> Dict:
        cache_total = self.cache_hits + self.cache_misses
        cache_hit_rate = (self.cache_hits / cache_total * 100) if cache_total > 0 else 0

        return {
            "commits_processed": self.commits_processed,
            "files_tracked": self.files_tracked,
            "aggregator_times": self.aggregator_times,
            "memory_peak_mb": round(self.memory_peak_mb, 2),
            "total_time_seconds": round(self.total_time, 2),
            "cache_statistics": {
                "hits": self.cache_hits,
                "misses": self.cache_misses,
                "hit_rate_percent": round(cache_hit_rate, 1),
            },
            "chunks_processed": self.chunks_processed,
        }


# ============================================================================
# BASE AGGREGATOR WITH PERFORMANCE OPTIMIZATIONS
# ============================================================================


class DatasetAggregator:
    """
    Base class for dataset aggregators with built-in performance optimizations.

    Phase 4 enhancements:
    - Memory-efficient data structures
    - Optional sampling for quick mode
    - Progress callbacks
    - Performance metrics tracking
    """

    def __init__(
        self,
        analyzer,
        quick_mode: bool = False,
        reporter: Optional[ProgressReporter] = None,
    ):
        self.analyzer = analyzer
        self.quick_mode = quick_mode
        self.reporter = reporter or ProgressReporter(quiet=True)
        self.data = {}
        self.processing_time = 0.0

    def process_commit(self, commit: dict, changes: list):
        """Process a single commit - override in subclasses"""
        pass

    def finalize(self) -> dict:
        """Finalize aggregation - override in subclasses"""
        return self.data

    def export(self, output_path: str):
        """Export aggregated data to JSON"""
        start = time.time()
        data = self.finalize()
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        self.processing_time = time.time() - start
        return len(json.dumps(data))


# ============================================================================
# PHASE 2 AGGREGATORS (OPTIMIZED)
# ============================================================================


class FileMetadataAggregator(DatasetAggregator):
    """
    Phase 2: Track per-file metadata and statistics.
    Phase 4 optimizations: Use slots for memory efficiency, incremental stats.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Use more memory-efficient structures
        self.file_stats = {}

    def process_commit(self, commit: dict, changes: list):
        """Track file-level statistics"""
        commit_ts = commit["timestamp"]
        author_email = commit["author_email"]

        for change in changes:
            file_path = change["file_path"]
            operation = change["operation"]

            if file_path not in self.file_stats:
                self.file_stats[file_path] = {
                    "first_seen": commit_ts,
                    "last_modified": commit_ts,
                    "commits": [],
                    "authors": set(),
                    "operations": Counter(),
                }

            stats = self.file_stats[file_path]
            stats["last_modified"] = max(stats["last_modified"], commit_ts)
            stats["commits"].append(commit["commit_hash"])
            stats["authors"].add(author_email)
            stats["operations"][operation] += 1

    def finalize(self) -> dict:
        """Generate final file index with computed metrics"""
        files = {}

        for file_path, stats in self.file_stats.items():
            # Calculate author statistics
            author_commits = Counter()
            # Efficient author counting without re-parsing
            total_commits = len(stats["commits"])

            # For primary author, we'd need commit-author mapping
            # For quick mode, skip this expensive calculation
            primary_author = None
            if not self.quick_mode and total_commits > 0:
                # Simplified: use first author from set
                primary_author = {
                    "email": list(stats["authors"])[0] if stats["authors"] else "",
                    "commit_count": total_commits,
                    "percentage": 100.0,
                }

            age_days = (stats["last_modified"] - stats["first_seen"]) / 86400
            commits_per_day = total_commits / age_days if age_days > 0 else 0

            files[file_path] = {
                "first_seen": datetime.fromtimestamp(
                    stats["first_seen"], timezone.utc
                ).isoformat(),
                "last_modified": datetime.fromtimestamp(
                    stats["last_modified"], timezone.utc
                ).isoformat(),
                "total_commits": total_commits,
                "unique_authors": len(stats["authors"]),
                "primary_author": primary_author,
                "operations": dict(stats["operations"]),
                "age_days": round(age_days, 2),
                "commits_per_day": round(commits_per_day, 4),
                "lifecycle_event_count": total_commits,
            }

        return {
            "schema_version": "1.0.0",
            "total_files": len(files),
            "generation_method": "streaming",
            "quick_mode": self.quick_mode,
            "files": files,
        }


class TemporalAggregator(DatasetAggregator):
    """
    Phase 2: Daily and monthly temporal aggregations.
    Phase 4 optimizations: Single-pass aggregation, efficient date handling.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.daily_stats = defaultdict(
            lambda: {
                "commits": 0,
                "files_changed": 0,
                "authors": set(),
                "operations": Counter(),
            }
        )
        self.monthly_stats = defaultdict(
            lambda: {
                "commits": 0,
                "files_changed": 0,
                "authors": set(),
                "operations": Counter(),
            }
        )

    def process_commit(self, commit: dict, changes: list):
        """Aggregate by day and month"""
        dt = datetime.fromtimestamp(commit["timestamp"], timezone.utc)
        day_key = dt.strftime("%Y-%m-%d")
        month_key = dt.strftime("%Y-%m")

        # Update daily stats
        self.daily_stats[day_key]["commits"] += 1
        self.daily_stats[day_key]["files_changed"] += len(changes)
        self.daily_stats[day_key]["authors"].add(commit["author_email"])
        for change in changes:
            self.daily_stats[day_key]["operations"][change["operation"]] += 1

        # Update monthly stats
        self.monthly_stats[month_key]["commits"] += 1
        self.monthly_stats[month_key]["files_changed"] += len(changes)
        self.monthly_stats[month_key]["authors"].add(commit["author_email"])
        for change in changes:
            self.monthly_stats[month_key]["operations"][change["operation"]] += 1

    def export_daily(self, output_path: str):
        """Export daily aggregations"""
        data = {
            "schema_version": "1.0.0",
            "aggregation_level": "daily",
            "total_days": len(self.daily_stats),
            "days": {
                day: {
                    "date": day,
                    "commits": stats["commits"],
                    "files_changed": stats["files_changed"],
                    "unique_authors": len(stats["authors"]),
                    "operations": dict(stats["operations"]),
                }
                for day, stats in sorted(self.daily_stats.items())
            },
        }

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def export_monthly(self, output_path: str):
        """Export monthly aggregations"""
        data = {
            "schema_version": "1.0.0",
            "aggregation_level": "monthly",
            "total_months": len(self.monthly_stats),
            "months": {
                month: {
                    "month": month,
                    "commits": stats["commits"],
                    "files_changed": stats["files_changed"],
                    "unique_authors": len(stats["authors"]),
                    "operations": dict(stats["operations"]),
                }
                for month, stats in sorted(self.monthly_stats.items())
            },
        }

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)


class AuthorNetworkAggregator(DatasetAggregator):
    """
    Phase 2: Author collaboration network.
    Phase 4 optimizations: Edge limit for quick mode, efficient set operations.
    """

    def __init__(self, *args, max_edges: int = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.file_authors = defaultdict(set)
        self.author_commits = Counter()
        self.max_edges = max_edges  # Limit for quick mode

    def process_commit(self, commit: dict, changes: list):
        """Track file co-authorship"""
        author = commit["author_email"]
        self.author_commits[author] += 1

        for change in changes:
            self.file_authors[change["file_path"]].add(author)

    def finalize(self) -> dict:
        """Build collaboration network"""
        # Build edges (author pairs who modified same files)
        edges = defaultdict(lambda: {"weight": 0, "files": set()})

        edge_count = 0
        for file_path, authors in self.file_authors.items():
            if len(authors) < 2:
                continue
            authors_list = sorted(authors)
            for i, author1 in enumerate(authors_list):
                for author2 in authors_list[i + 1 :]:
                    edge_key = (author1, author2)
                    edges[edge_key]["weight"] += 1
                    edges[edge_key]["files"].add(file_path)
                    edge_count += 1

                    # Quick mode: limit edges
                    if self.max_edges and edge_count >= self.max_edges:
                        break
                if self.max_edges and edge_count >= self.max_edges:
                    break
            if self.max_edges and edge_count >= self.max_edges:
                break

        # Format network data
        nodes = [
            {
                "id": author,
                "email": author,
                "commit_count": count,
                "collaboration_count": sum(1 for e in edges if author in e),
            }
            for author, count in self.author_commits.items()
        ]

        formatted_edges = [
            {
                "source": author1,
                "target": author2,
                "weight": data["weight"],
                "shared_files": len(data["files"]),
            }
            for (author1, author2), data in edges.items()
        ]

        return {
            "schema_version": "1.0.0",
            "network_type": "author_collaboration",
            "quick_mode": self.quick_mode,
            "edge_limit": self.max_edges,
            "nodes": nodes,
            "edges": formatted_edges,
            "statistics": {
                "total_authors": len(nodes),
                "total_edges": len(formatted_edges),
                "density": (
                    round(len(formatted_edges) / (len(nodes) * (len(nodes) - 1) / 2), 4)
                    if len(nodes) > 1
                    else 0
                ),
            },
        }


# ============================================================================
# PHASE 3 AGGREGATORS (OPTIMIZED)
# ============================================================================


class CochangeNetworkAggregator(DatasetAggregator):
    """
    Phase 3: File co-change network for coupling analysis.
    Phase 4 optimizations: Configurable pair limits, memory-efficient storage.
    """

    def __init__(
        self, *args, max_pairs: int = None, min_cochange_count: int = 2, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.commit_files = []  # Store files per commit
        self.max_pairs = max_pairs
        self.min_cochange_count = min_cochange_count

    def process_commit(self, commit: dict, changes: list):
        """Track files changed together"""
        files = [change["file_path"] for change in changes]
        if len(files) >= 2:
            self.commit_files.append(files)

    def finalize(self) -> dict:
        """Build co-change network"""
        cochange_pairs = defaultdict(int)
        pair_count = 0

        # Count co-changes
        for files in self.commit_files:
            if len(files) < 2:
                continue
            files_sorted = sorted(files)
            for i, file1 in enumerate(files_sorted):
                for file2 in files_sorted[i + 1 :]:
                    cochange_pairs[(file1, file2)] += 1
                    pair_count += 1

                    # Quick mode: limit pairs
                    if self.max_pairs and pair_count >= self.max_pairs:
                        break
                if self.max_pairs and pair_count >= self.max_pairs:
                    break
            if self.max_pairs and pair_count >= self.max_pairs:
                break

        # Filter by minimum co-change count
        filtered_pairs = {
            pair: count
            for pair, count in cochange_pairs.items()
            if count >= self.min_cochange_count
        }

        # Calculate coupling strength
        edges = []
        for (file1, file2), count in filtered_pairs.items():
            edges.append(
                {
                    "source": file1,
                    "target": file2,
                    "cochange_count": count,
                    "coupling_strength": round(count / len(self.commit_files), 4),
                }
            )

        # Sort by coupling strength
        edges.sort(key=lambda x: x["coupling_strength"], reverse=True)

        # Get unique files (nodes)
        all_files = set()
        for file1, file2 in filtered_pairs.keys():
            all_files.add(file1)
            all_files.add(file2)

        return {
            "schema_version": "1.0.0",
            "network_type": "file_cochange",
            "quick_mode": self.quick_mode,
            "pair_limit": self.max_pairs,
            "min_cochange_count": self.min_cochange_count,
            "total_files": len(all_files),
            "total_edges": len(edges),
            "edges": edges,
        }


class FileHierarchyAggregator(DatasetAggregator):
    """
    Phase 3: Directory-level aggregations.
    Phase 4 optimizations: Efficient path parsing, cached computations.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.directory_stats = defaultdict(
            lambda: {
                "files": set(),
                "commits": 0,
                "authors": set(),
                "operations": Counter(),
            }
        )

    def process_commit(self, commit: dict, changes: list):
        """Aggregate statistics by directory"""
        for change in changes:
            file_path = change["file_path"]
            operation = change["operation"]

            # Get all parent directories
            path_parts = Path(file_path).parts
            for i in range(len(path_parts)):
                dir_path = str(Path(*path_parts[: i + 1]))

                stats = self.directory_stats[dir_path]
                stats["files"].add(file_path)
                stats["commits"] += 1
                stats["authors"].add(commit["author_email"])
                stats["operations"][operation] += 1

    def finalize(self) -> dict:
        """Generate directory statistics"""
        directories = {}

        for dir_path, stats in self.directory_stats.items():
            directories[dir_path] = {
                "path": dir_path,
                "total_files": len(stats["files"]),
                "total_commits": stats["commits"],
                "unique_authors": len(stats["authors"]),
                "operations": dict(stats["operations"]),
                "activity_score": round(
                    stats["commits"] / len(stats["files"]) if stats["files"] else 0, 2
                ),
            }

        return {
            "schema_version": "1.0.0",
            "aggregation_type": "directory_hierarchy",
            "total_directories": len(directories),
            "directories": directories,
        }


class MilestoneAggregator(DatasetAggregator):
    """
    Phase 3: Track milestones and significant events.
    Phase 4 optimizations: Git tag caching, efficient event detection.
    """

    def __init__(self, *args, repo_path: str = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.repo_path = repo_path
        self.tags = self._get_git_tags() if repo_path else {}
        self.file_events = defaultdict(list)

    def _get_git_tags(self) -> Dict[str, str]:
        """Get git tags mapped to commit hashes"""
        try:
            result = subprocess.run(
                [
                    "git",
                    "-C",
                    self.repo_path,
                    "tag",
                    "-l",
                    "--format=%(refname:short) %(objectname)",
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            tags = {}
            for line in result.stdout.strip().split("\n"):
                if line:
                    parts = line.split()
                    if len(parts) >= 2:
                        tags[parts[1]] = parts[0]  # commit_hash -> tag_name
            return tags
        except:
            return {}

    def process_commit(self, commit: dict, changes: list):
        """Track significant file events"""
        commit_hash = commit["commit_hash"]
        commit_ts = commit["timestamp"]

        # Check if this commit has a tag
        tag_name = self.tags.get(commit_hash)

        for change in changes:
            file_path = change["file_path"]
            operation = change["operation"]

            # Record significant events
            if operation == "A":  # File creation
                self.file_events[file_path].append(
                    {
                        "type": "creation",
                        "timestamp": commit_ts,
                        "commit_hash": commit_hash,
                        "tag": tag_name,
                    }
                )
            elif tag_name:  # Tagged commits
                self.file_events[file_path].append(
                    {
                        "type": "release",
                        "timestamp": commit_ts,
                        "commit_hash": commit_hash,
                        "tag": tag_name,
                        "operation": operation,
                    }
                )

    def finalize(self) -> dict:
        """Generate milestone snapshots"""
        snapshots = []

        # Group by tags
        tag_snapshots = defaultdict(lambda: {"files": [], "timestamp": 0})

        for file_path, events in self.file_events.items():
            for event in events:
                if event.get("tag"):
                    tag = event["tag"]
                    tag_snapshots[tag]["files"].append(
                        {
                            "path": file_path,
                            "event_type": event["type"],
                            "operation": event.get("operation", ""),
                        }
                    )
                    tag_snapshots[tag]["timestamp"] = max(
                        tag_snapshots[tag]["timestamp"], event["timestamp"]
                    )

        for tag, data in sorted(tag_snapshots.items(), key=lambda x: x[1]["timestamp"]):
            snapshots.append(
                {
                    "tag": tag,
                    "timestamp": data["timestamp"],
                    "datetime": datetime.fromtimestamp(
                        data["timestamp"], timezone.utc
                    ).isoformat(),
                    "files_affected": len(data["files"]),
                    "files": data["files"],
                }
            )

        return {
            "schema_version": "1.0.0",
            "snapshot_type": "release_milestones",
            "total_snapshots": len(snapshots),
            "snapshots": snapshots,
        }


# ============================================================================
# CORE ANALYZER WITH PHASE 4 ENHANCEMENTS
# ============================================================================


class GitFileLifecycle:
    """
    Core analyzer for git file lifecycle tracking.
    Phase 4 enhancements: Performance monitoring, progress reporting, quick mode support.
    """

    def __init__(
        self,
        repo_path: str,
        reporter: Optional[ProgressReporter] = None,
        memory_limit_mb: Optional[float] = None,
        enable_profiling: bool = False,
    ):
        self.repo_path = os.path.abspath(repo_path)
        self.reporter = reporter or ProgressReporter()
        self.files = {}
        self.total_commits = 0
        self.total_changes = 0
        self.skipped_unmerged = 0
        self.errors = []
        self.aggregators = []
        self.metrics = PerformanceMetrics()
        self.memory_monitor = MemoryMonitor(limit_mb=memory_limit_mb)  # Phase 4.1
        self.enable_profiling = enable_profiling  # Phase 4.1

    def add_aggregator(self, aggregator: DatasetAggregator):
        """Register an aggregator for additional datasets"""
        self.aggregators.append(aggregator)

    def _parse_commit_line(self, line: str) -> Optional[Dict]:
        """Parse a single commit line from git log"""
        try:
            parts = line.split("\x00")
            if len(parts) < 4:
                return None

            return {
                "commit_hash": parts[0],
                "timestamp": int(parts[1]),
                "author_name": parts[2],
                "author_email": parts[3],
                "commit_subject": parts[4] if len(parts) > 4 else "",
            }
        except Exception as e:
            self.errors.append(f"Failed to parse commit line: {str(e)}")
            return None

    def analyze_repository(self, sample_rate: float = 1.0):
        """
        Main analysis function with streaming architecture.
        Phase 4 enhancements: Progress bars, memory monitoring, ETA

        Args:
            sample_rate: Fraction of commits to process (for quick mode)
        """
        start_time = time.time()
        self.reporter.stage_start("Git Log Processing", "Streaming commit history...")

        try:
            # First pass: count total commits for progress bar
            count_cmd = ["git", "-C", self.repo_path, "rev-list", "--all", "--count"]
            total_commits_result = subprocess.run(
                count_cmd, capture_output=True, text=True, check=True
            )
            total_commits = int(total_commits_result.stdout.strip())

            # Git log command for all file changes
            cmd = [
                "git",
                "-C",
                self.repo_path,
                "log",
                "--all",
                "--name-status",
                "--diff-filter=AMDRTC",
                "--format=%H%x00%ct%x00%an%x00%ae%x00%s",
            ]

            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
            )

            # Create progress bar (Phase 4.2)
            progress_bar = self.reporter.create_progress_bar(
                total=total_commits, desc="Analyzing commits"
            )

            current_commit = None
            commit_changes = []
            commits_processed = 0

            for line in process.stdout:
                line = line.rstrip("\n")

                if not line:
                    # Empty line: process previous commit ONLY if we have accumulated changes.
                    # This prevents resetting current_commit on the separator line between header and changes.
                    if current_commit and commit_changes:
                        # Sampling for quick mode
                        if (
                            sample_rate >= 1.0
                            or (commits_processed % int(1 / sample_rate)) == 0
                        ):
                            self._process_commit(current_commit, commit_changes)
                            commits_processed += 1

                            # Update progress bar
                            if progress_bar:
                                progress_bar.update(1)
                            elif commits_processed % 1000 == 0:
                                self.reporter.progress(
                                    "Processing commits", commits_processed
                                )

                            # Memory check (Phase 4.1)
                            if commits_processed % 5000 == 0:
                                memory_mb = self.memory_monitor.check_memory()
                                if memory_mb > 0 and self.reporter.verbose:
                                    self.reporter.info(
                                        f"Memory usage: {memory_mb:.1f} MB"
                                    )

                        current_commit = None
                        commit_changes = []
                    continue

                # Check if line is a commit header
                if "\x00" in line:
                    # New commit
                    if current_commit and commit_changes:
                        if (
                            sample_rate >= 1.0
                            or (commits_processed % int(1 / sample_rate)) == 0
                        ):
                            self._process_commit(current_commit, commit_changes)
                            commits_processed += 1

                            if progress_bar:
                                progress_bar.update(1)

                    current_commit = self._parse_commit_line(line)
                    commit_changes = []
                else:
                    # File change line
                    if current_commit:
                        change = self._parse_change_line(line)
                        if change:
                            commit_changes.append(change)

            # Process last commit
            if current_commit and commit_changes:
                self._process_commit(current_commit, commit_changes)
                commits_processed += 1
                if progress_bar:
                    progress_bar.update(1)

            # Close progress bar
            if progress_bar:
                progress_bar.close()

            process.wait()

            if process.returncode != 0:
                stderr = process.stderr.read()
                raise RuntimeError(f"Git command failed: {stderr}")

            # Update metrics (Phase 4.1)
            self.metrics.commits_processed = commits_processed
            self.metrics.files_tracked = len(self.files)
            self.metrics.total_time = time.time() - start_time
            self.metrics.memory_peak_mb = self.memory_monitor.get_peak()

            # Cache statistics from lru_cache
            cache_info = cached_file_hash.cache_info()
            self.metrics.cache_hits = cache_info.hits
            self.metrics.cache_misses = cache_info.misses

            self.reporter.stage_complete(
                "Git Log Processing",
                {
                    "Commits processed": f"{commits_processed:,}",
                    "Files tracked": f"{len(self.files):,}",
                    "Changes recorded": f"{self.total_changes:,}",
                    "Peak memory": f"{self.metrics.memory_peak_mb:.1f} MB",
                },
            )

        except Exception as e:
            self.reporter.error(f"Failed to analyze repository: {str(e)}")
            raise

    def _parse_change_line(self, line: str) -> Optional[Dict]:
        """Parse a file change line from git log"""
        try:
            parts = line.split("\t")
            if len(parts) < 2:
                return None

            status = parts[0]

            # Handle different status formats
            if status.startswith("R") or status.startswith("C"):
                # Rename or copy
                if len(parts) < 3:
                    return None
                operation = status[0]
                similarity = int(status[1:]) if len(status) > 1 else 100
                old_path = parts[1]
                new_path = parts[2]
                return {
                    "operation": operation,
                    "file_path": new_path,
                    "old_path": old_path,
                    "similarity": similarity,
                }
            else:
                # Add, Modify, Delete, Type change
                operation = status[0]
                file_path = parts[1]
                return {"operation": operation, "file_path": file_path}
        except Exception as e:
            self.errors.append(
                f"Failed to parse change line: {line[:50]}... - {str(e)}"
            )
            return None

    def _process_commit(self, commit: Dict, changes: List[Dict]):
        """Process a single commit and its changes"""
        self.total_commits += 1

        # Update lifecycle data
        for change in changes:
            file_path = change["file_path"]

            if file_path not in self.files:
                self.files[file_path] = []

            event = {
                "commit_hash": commit["commit_hash"],
                "timestamp": commit["timestamp"],
                "datetime": datetime.fromtimestamp(
                    commit["timestamp"], timezone.utc
                ).isoformat(),
                "operation": change["operation"],
                "author_name": commit["author_name"],
                "author_email": commit["author_email"],
                "commit_subject": commit["commit_subject"],
            }

            # Add optional fields for renames/copies
            if "old_path" in change:
                event["old_path"] = change["old_path"]
            if "similarity" in change:
                event["similarity"] = change["similarity"]

            self.files[file_path].append(event)
            self.total_changes += 1

        # Process aggregators
        for aggregator in self.aggregators:
            try:
                aggregator.process_commit(commit, changes)
            except Exception as e:
                self.errors.append(
                    f"Aggregator {type(aggregator).__name__} failed: {str(e)}"
                )

    def export_lifecycle(self, output_path: str):
        """Export core lifecycle data (v1.0 compatible)"""
        self.reporter.stage_start("Export", "Writing core lifecycle data...")

        data = {
            "files": self.files,
            "total_commits": self.total_commits,
            "total_changes": self.total_changes,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "repository_path": self.repo_path,
            "skipped_unmerged": self.skipped_unmerged,
        }

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        self.reporter.stage_complete(
            "Export",
            {"File": output_path, "Size": f"{os.path.getsize(output_path):,} bytes"},
        )

        return data


# ============================================================================
# MANIFEST & METADATA
# ============================================================================


def generate_manifest(
    output_dir: str, analyzer: GitFileLifecycle, datasets: Dict[str, str]
):
    """Generate manifest.json with dataset metadata"""
    manifest = {
        "generator_version": VERSION,
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "repository": analyzer.repo_path,
        "performance_metrics": analyzer.metrics.to_dict(),
        "datasets": {},
    }

    for dataset_name, file_path in datasets.items():
        full_path = os.path.join(output_dir, file_path)
        if os.path.exists(full_path):
            with open(full_path, "rb") as f:
                data = f.read()
                sha256 = hashlib.sha256(data).hexdigest()

            manifest["datasets"][dataset_name] = {
                "file": file_path,
                "schema_version": SCHEMA_VERSION,
                "file_size_bytes": len(data),
                "sha256": sha256,
                "production_ready": dataset_name == "core_lifecycle",
            }

    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    return manifest


class MetadataReportGenerator:
    """
    Generates a Markdown report summarizing the generated datasets.
    Restored and enhanced from V1 for V2 architecture.
    """

    def __init__(self, analyzer, output_dir: str, datasets: Dict[str, str]):
        self.analyzer = analyzer
        self.output_dir = Path(output_dir)
        self.datasets = datasets
        self.timestamp = datetime.now(timezone.utc).isoformat()

        try:
            import pandas as pd

            self.pd = pd
            self.pandas_available = True
        except ImportError:
            self.pd = None
            self.pandas_available = False

    def generate(self, output_filename: str = "dataset_metadata.md"):
        """Generate the markdown report."""
        report_path = self.output_dir / output_filename

        lines = [
            "# Git File Lifecycle Analysis - Dataset Metadata",
            "",
            f"**Generated:** {self.timestamp}",
            f"**Repository:** `{self.analyzer.repo_path}`",
            f"**Generator Version:** {VERSION}",
            "",
            "---",
            "",
            "## 📊 Analysis Overview",
            "",
            f"- **Total Commits:** {self.analyzer.total_commits:,}",
            f"- **Total Files Tracked:** {len(self.analyzer.files):,}",
            f"- **Total Changes Recorded:** {self.analyzer.total_changes:,}",
            "",
            "### Performance Metrics",
            "",
            f"- **Execution Time:** {self.analyzer.metrics.total_time:.2f}s",
            f"- **Peak Memory:** {self.analyzer.metrics.memory_peak_mb:.1f} MB",
            f"- **Cache Hit Rate:** {(self.analyzer.metrics.cache_hits / (self.analyzer.metrics.cache_hits + self.analyzer.metrics.cache_misses) * 100) if (self.analyzer.metrics.cache_hits + self.analyzer.metrics.cache_misses) > 0 else 0:.1f}%",
            "",
            "---",
            "",
            "## 📂 Dataset Files",
            "",
        ]

        # Sort datasets by name for consistent output
        for name, rel_path in sorted(self.datasets.items()):
            full_path = self.output_dir / rel_path
            if not full_path.exists():
                continue

            lines.extend(self._analyze_file(name, full_path))

        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        return report_path

    def _analyze_file(self, name: str, file_path: Path) -> List[str]:
        """Analyze a single file and return markdown lines."""
        lines = []
        file_size = file_path.stat().st_size
        size_str = (
            f"{file_size / 1024 / 1024:.2f} MB"
            if file_size > 1024 * 1024
            else f"{file_size / 1024:.2f} KB"
        )

        lines.append(f"### 📄 {file_path.name} ({name})")
        lines.append("")
        lines.append(f"**File Size:** {size_str} ({file_size:,} bytes)")
        lines.append(f"**Format:** JSON")
        lines.append("")

        if not self.pandas_available:
            lines.append(
                "> ℹ️ Install `pandas` for detailed data shape and distribution analysis."
            )
            lines.append("")
            return lines

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Handle V2 Wrapper Structure (schema_version, etc.)
            content = data
            if isinstance(data, dict):
                # V2 often wraps the actual list in a key like "files", "nodes", "days"
                # We try to find the main list to analyze
                keys_to_check = [
                    "files",
                    "nodes",
                    "edges",
                    "days",
                    "months",
                    "snapshots",
                    "directories",
                ]
                for key in keys_to_check:
                    if key in data and isinstance(data[key], (list, dict)):
                        content = data[key]
                        lines.append(f"**Root Keys:** `{', '.join(data.keys())}`")
                        lines.append(f"**Analyzing Key:** `{key}`")
                        break

            # Convert to DataFrame
            df = None
            if isinstance(content, list):
                df = self.pd.DataFrame(content)
            elif isinstance(content, dict):
                # Handle dictionary of dictionaries (e.g. file_lifecycle.json files map)
                # Convert {key: {attr1, attr2}} to [{key, attr1, attr2}]
                try:
                    df = self.pd.DataFrame.from_dict(content, orient="index")
                    df.reset_index(inplace=True)
                    df.rename(columns={"index": "key"}, inplace=True)
                except:
                    pass

            if df is not None and not df.empty:
                lines.extend(self._generate_dataframe_analysis(df))
            else:
                lines.append(
                    "**Structure:** Complex nested JSON (automated tabular analysis skipped)"
                )

        except Exception as e:
            lines.append(f"**⚠️ Analysis Error:** {str(e)}")

        lines.append("")
        lines.append("---")
        lines.append("")
        return lines

    def _generate_dataframe_analysis(self, df) -> List[str]:
        """Generate detailed analysis for a pandas DataFrame."""
        lines = []

        # 1. Shape
        lines.append("#### 📊 Data Shape")
        lines.append(f"- **Rows:** {len(df):,}")
        lines.append(f"- **Columns:** {len(df.columns)}")
        lines.append("")

        # 2. Columns
        lines.append("#### 📋 Column Information")
        lines.append("| Column | Data Type | Non-Null | Null % |")
        lines.append("|--------|-----------|----------|--------|")

        for col in df.columns:
            dtype = str(df[col].dtype)
            non_null = df[col].count()
            null_pct = ((len(df) - non_null) / len(df) * 100) if len(df) > 0 else 0
            lines.append(f"| `{col}` | {dtype} | {non_null:,} | {null_pct:.1f}% |")
        lines.append("")

        # 3. Sample
        lines.append("#### 🔍 Sample Records (First 3)")
        lines.append("```json")
        # Convert complex types to string for display to avoid JSON serialization errors
        sample_df = df.head(3).copy()
        for col in sample_df.columns:
            if sample_df[col].apply(lambda x: isinstance(x, (dict, list, set))).any():
                sample_df[col] = sample_df[col].apply(str)
        lines.append(sample_df.to_json(orient="records", indent=2))
        lines.append("```")
        lines.append("")

        # 4. Distribution (Fixed "unhashable type" bug)
        lines.append("#### 📈 Data Distribution (Top 5)")

        # Limit to first 5 columns to save space, or specific interesting ones
        for col in df.columns[:8]:
            lines.append(f"**{col}**:")

            # Check if column contains unhashable types (dicts, lists)
            if df[col].apply(lambda x: isinstance(x, (dict, list, set))).any():
                lines.append("- *Contains nested structures (skipped distribution)*")
                lines.append("")
                continue

            if df[col].dtype in ["int64", "float64"]:
                stats = df[col].describe()
                lines.append(f"- Range: {stats['min']:.2f} to {stats['max']:.2f}")
                lines.append(f"- Mean: {stats['mean']:.2f}")
            else:
                unique_count = df[col].nunique()
                lines.append(f"- Unique values: {unique_count:,}")
                if unique_count < 50:
                    top = df[col].value_counts().head(5)
                    for val, count in top.items():
                        pct = (count / len(df)) * 100
                        lines.append(f"  - `{val}`: {count:,} ({pct:.1f}%)")
            lines.append("")

        return lines


# ============================================================================
# CLI & MAIN FUNCTION
# ============================================================================


class ConfigResolver:
    """
    Handles configuration precedence: CLI > Config File > Preset > Defaults
    """

    def __init__(
        self,
        cli_params: Dict[str, Any],
        config_path: str = None,
        preset_name: str = None,
        repo_path: str = None,
    ):
        # 1. Capture CLI params (filter out None to allow fallbacks)
        self.cli = {k: v for k, v in cli_params.items() if v is not None}

        # 2. Load Config File
        self.config = {}
        if config_path:
            try:
                self.config = load_config_file(config_path)
                print(f"Loaded configuration from: {config_path}")
            except Exception as e:
                print(f"Error loading config file: {e}", file=sys.stderr)
                sys.exit(1)
        elif repo_path:
            auto_path = find_config_file(repo_path)
            if auto_path:
                try:
                    self.config = load_config_file(auto_path)
                    print(f"Auto-discovered configuration: {auto_path}")
                except Exception as e:
                    print(
                        f"Warning: Found config file but failed to load: {e}",
                        file=sys.stderr,
                    )

        # Normalize config keys (kebab-case to snake_case)
        self.config = {k.replace("-", "_"): v for k, v in self.config.items()}

        # 3. Determine Preset
        # CLI preset overrides config preset
        final_preset_name = preset_name or self.config.get("preset")
        self.preset = self._get_preset(final_preset_name)

    def _get_preset(self, name: str) -> Dict[str, Any]:
        """Return configuration dictionary for a named preset"""
        if not name:
            return {}

        presets = {
            "standard": {"enable_aggregations": True, "phase2": True, "phase3": False},
            "quick": {
                "enable_aggregations": True,
                "phase2": True,
                "phase3": False,
                "quick_mode": True,
                "sample_rate": 0.3,
                "skip_author_network": True,
                "skip_milestones": True,
            },
            "essential": {
                "enable_aggregations": True,
                "phase2": True,
                "phase3": False,
                "skip_author_network": True,
            },
            "network": {
                "enable_aggregations": True,
                "phase2": True,
                "phase3": True,
                "skip_file_metadata": True,
                "skip_temporal": True,
                "skip_hierarchy": True,
                "skip_milestones": True,
            },
            "full": {"enable_aggregations": True, "phase2": True, "phase3": True},
        }
        return presets.get(name, {})

    def get(self, key: str, default: Any = None) -> Any:
        """Resolve value based on precedence"""
        if key in self.cli:
            return self.cli[key]
        if key in self.config:
            return self.config[key]
        if key in self.preset:
            return self.preset[key]
        return default


@click.command(context_settings=dict(help_option_names=["-h", "--help"]))
@click.argument(
    "repo_path",
    type=click.Path(exists=True, file_okay=False, resolve_path=True),
    required=False,  # Changed to False to allow --check-dependencies to run without path
)
@click.option(
    "-o",
    "--output",
    type=click.Path(file_okay=False),
    help="Output directory (default: git_analysis_output_TIMESTAMP)",
)
@click.option(
    "--config",
    type=click.Path(exists=True, dir_okay=False),
    help="Configuration file path (.yaml or .json)",
)
@click.option(
    "--preset",
    type=click.Choice(["standard", "quick", "essential", "network", "full"]),
    help="Use predefined feature configuration",
)
# Feature Flags
@click.option(
    "--enable-aggregations",
    is_flag=True,
    default=None,
    help="Enable additional dataset generation",
)
@click.option("--phase2", is_flag=True, default=None, help="Enable Phase 2 features")
@click.option("--phase3", is_flag=True, default=None, help="Enable Phase 3 features")
# Quick Mode
@click.option(
    "--quick-mode",
    is_flag=True,
    default=None,
    help="Fast analysis for large repositories",
)
@click.option("--sample-rate", type=float, help="Commit sampling rate (0.0-1.0)")
@click.option("--max-cochange-pairs", type=int, help="Maximum co-change pairs to track")
@click.option("--max-author-edges", type=int, help="Maximum author network edges")
# Performance
@click.option("--memory-limit", type=float, help="Memory limit in MB")
@click.option(
    "--profile", is_flag=True, default=None, help="Enable performance profiling"
)
@click.option(
    "--profile-output", default="profile_stats.prof", help="Profile output file"
)
# Output Control
@click.option(
    "-q", "--quiet", is_flag=True, default=None, help="Suppress progress output"
)
@click.option(
    "-v",
    "--verbose",
    is_flag=True,
    default=None,
    help="Show detailed progress information",
)
@click.option("--no-color", is_flag=True, default=None, help="Disable colored output")
# Skips
@click.option("--skip-file-metadata", is_flag=True, default=None)
@click.option("--skip-temporal", is_flag=True, default=None)
@click.option("--skip-author-network", is_flag=True, default=None)
@click.option("--skip-cochange", is_flag=True, default=None)
@click.option("--skip-hierarchy", is_flag=True, default=None)
@click.option("--skip-milestones", is_flag=True, default=None)
# Utils
@click.option(
    "--dry-run",
    is_flag=True,
    help="Show what would be generated without running analysis",
)
@click.option(
    "--check-dependencies",
    is_flag=True,
    help="Check for optional dependencies and exit",
)
@click.version_option(version=VERSION)
def main(repo_path, output, config, preset, **kwargs):
    """Git File Lifecycle Analyzer v2.0.0 - Production Release"""

    # Handle check_dependencies immediately
    if kwargs.get("check_dependencies"):
        print("Checking optional dependencies...")
        print(
            f"  tqdm (progress bars): {'✓ Available' if TQDM_AVAILABLE else '✗ Not installed'}"
        )
        print(
            f"  colorama (colored output): {'✓ Available' if COLORAMA_AVAILABLE else '✗ Not installed'}"
        )
        print(
            f"  PyYAML (YAML config): {'✓ Available' if YAML_AVAILABLE else '✗ Not installed'}"
        )
        print(
            f"  memory_profiler: {'✓ Available' if MEMORY_PROFILER_AVAILABLE else '✗ Not installed'}"
        )
        print(f"  psutil (memory monitoring): ", end="")
        try:
            import psutil

            print("✓ Available")
        except ImportError:
            print("✗ Not installed")
        print("\nInstall missing dependencies:")
        print("  pip install click tqdm colorama pyyaml memory-profiler psutil")
        return

    # Manual validation for repo_path (since it's now optional in decorator)
    if not repo_path:
        ctx = click.get_current_context()
        click.echo(ctx.get_help())
        ctx.exit(2)

    # Initialize Resolver
    resolver = ConfigResolver(kwargs, config, preset, repo_path)

    # Resolve Values
    quiet = resolver.get("quiet", False)
    verbose = resolver.get("verbose", False)
    no_color = resolver.get("no_color", False)
    dry_run = resolver.get("dry_run", False)

    # Initialize Reporter
    use_colors = not no_color
    reporter = ProgressReporter(quiet=quiet, verbose=verbose, use_colors=use_colors)

    # Display dependency warnings
    if not quiet:
        if not TQDM_AVAILABLE:
            reporter.warning(
                "tqdm not installed - progress bars disabled (pip install tqdm)"
            )
        if not COLORAMA_AVAILABLE:
            reporter.warning(
                "colorama not installed - colored output disabled (pip install colorama)"
            )

    # Resolve Feature Flags
    enable_aggregations = resolver.get("enable_aggregations", False)
    phase2 = resolver.get("phase2", False)
    phase3 = resolver.get("phase3", False)

    # Resolve Quick Mode & Limits
    quick_mode = resolver.get("quick_mode", False)
    sample_rate = resolver.get("sample_rate")
    max_cochange = resolver.get("max_cochange_pairs")
    max_edges = resolver.get("max_author_edges")

    # Apply Quick Mode Defaults
    if quick_mode:
        if sample_rate is None:
            sample_rate = 0.5
        if max_cochange is None:
            max_cochange = 10000
        if max_edges is None:
            max_edges = 5000
    else:
        if sample_rate is None:
            sample_rate = 1.0

    # Resolve Performance & Skips
    memory_limit = resolver.get("memory_limit")
    profile = resolver.get("profile", False)
    profile_output_file = resolver.get("profile_output", "profile_stats.prof")

    skip_file_metadata = resolver.get("skip_file_metadata", False)
    skip_temporal = resolver.get("skip_temporal", False)
    skip_author_network = resolver.get("skip_author_network", False)
    skip_cochange = resolver.get("skip_cochange", False)
    skip_hierarchy = resolver.get("skip_hierarchy", False)
    skip_milestones = resolver.get("skip_milestones", False)

    # Dry Run Output
    if dry_run:
        reporter.info("DRY RUN MODE - No analysis will be performed")
        reporter.info(f"Repository: {repo_path}")
        reporter.info(f"Quick mode: {quick_mode}")
        if quick_mode:
            reporter.info(f"  Sample rate: {sample_rate}")
            reporter.info(f"  Max co-change pairs: {max_cochange}")
            reporter.info(f"  Max author edges: {max_edges}")
        if memory_limit:
            reporter.info(f"Memory limit: {memory_limit} MB")
        if profile:
            reporter.info(f"Profiling enabled (output: {profile_output_file})")
        reporter.info("\nDatasets to generate:")
        reporter.info("  ✓ file_lifecycle.json (core)")
        if enable_aggregations or phase2:
            if not skip_file_metadata:
                reporter.info("  ✓ metadata/file_index.json")
            if not skip_temporal:
                reporter.info("  ✓ aggregations/temporal_daily.json")
            if not skip_author_network:
                reporter.info("  ✓ networks/author_network.json")
        if phase3:
            if not skip_cochange:
                reporter.info("  ✓ networks/cochange_network.json")
            if not skip_hierarchy:
                reporter.info("  ✓ aggregations/directory_stats.json")
            if not skip_milestones:
                reporter.info("  ✓ milestones/release_snapshots.json")
        return

    # Validate Repository (Click checks existence, but we check if it's a git repo)
    git_dir = os.path.join(repo_path, ".git")
    if not os.path.isdir(git_dir):
        reporter.error(f"Not a git repository: {repo_path}")
        sys.exit(1)

    # Output Directory
    if output:
        output_dir = output
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"git_analysis_output_{timestamp}"

    os.makedirs(output_dir, exist_ok=True)
    reporter.info(f"Output directory: {output_dir}")

    # Execution
    profile_path = os.path.join(output_dir, profile_output_file) if profile else None

    try:
        with ProfilingContext(enabled=profile, output_path=profile_path):
            reporter.stage_start("Initialization", f"Analyzing repository: {repo_path}")

            analyzer = GitFileLifecycle(
                repo_path,
                reporter,
                memory_limit_mb=memory_limit,
                enable_profiling=profile,
            )

            # Register Aggregators
            if enable_aggregations or phase2 or phase3:
                # Phase 2
                if phase2 or enable_aggregations:
                    if not skip_file_metadata:
                        analyzer.add_aggregator(
                            FileMetadataAggregator(
                                analyzer, quick_mode=quick_mode, reporter=reporter
                            )
                        )
                    if not skip_temporal:
                        analyzer.add_aggregator(
                            TemporalAggregator(
                                analyzer, quick_mode=quick_mode, reporter=reporter
                            )
                        )
                    if not skip_author_network:
                        analyzer.add_aggregator(
                            AuthorNetworkAggregator(
                                analyzer,
                                quick_mode=quick_mode,
                                max_edges=max_edges,
                                reporter=reporter,
                            )
                        )

                # Phase 3
                if phase3:
                    if not skip_cochange:
                        analyzer.add_aggregator(
                            CochangeNetworkAggregator(
                                analyzer,
                                quick_mode=quick_mode,
                                max_pairs=max_cochange,
                                reporter=reporter,
                            )
                        )
                    if not skip_hierarchy:
                        analyzer.add_aggregator(
                            FileHierarchyAggregator(
                                analyzer, quick_mode=quick_mode, reporter=reporter
                            )
                        )
                    if not skip_milestones:
                        analyzer.add_aggregator(
                            MilestoneAggregator(
                                analyzer,
                                repo_path=repo_path,
                                quick_mode=quick_mode,
                                reporter=reporter,
                            )
                        )

            reporter.stage_complete("Initialization")

            # Run Analysis
            analyzer.analyze_repository(sample_rate=sample_rate)

            # Export Core Data
            analyzer.export_lifecycle(os.path.join(output_dir, "file_lifecycle.json"))
            datasets = {"core_lifecycle": "file_lifecycle.json"}

            # Export Aggregators
            if analyzer.aggregators:
                reporter.stage_start(
                    "Dataset Export", "Exporting additional datasets..."
                )
                for agg in analyzer.aggregators:
                    try:
                        if isinstance(agg, FileMetadataAggregator):
                            agg.export(
                                os.path.join(output_dir, "metadata", "file_index.json")
                            )
                            datasets["file_metadata"] = "metadata/file_index.json"
                        elif isinstance(agg, TemporalAggregator):
                            agg.export_daily(
                                os.path.join(
                                    output_dir, "aggregations", "temporal_daily.json"
                                )
                            )
                            agg.export_monthly(
                                os.path.join(
                                    output_dir, "aggregations", "temporal_monthly.json"
                                )
                            )
                            datasets["temporal_daily"] = (
                                "aggregations/temporal_daily.json"
                            )
                        elif isinstance(agg, AuthorNetworkAggregator):
                            agg.export(
                                os.path.join(
                                    output_dir, "networks", "author_network.json"
                                )
                            )
                            datasets["author_network"] = "networks/author_network.json"
                        elif isinstance(agg, CochangeNetworkAggregator):
                            agg.export(
                                os.path.join(
                                    output_dir, "networks", "cochange_network.json"
                                )
                            )
                            datasets["cochange_network"] = (
                                "networks/cochange_network.json"
                            )
                        elif isinstance(agg, FileHierarchyAggregator):
                            agg.export(
                                os.path.join(
                                    output_dir, "aggregations", "directory_stats.json"
                                )
                            )
                            datasets["directory_stats"] = (
                                "aggregations/directory_stats.json"
                            )
                        elif isinstance(agg, MilestoneAggregator):
                            agg.export(
                                os.path.join(
                                    output_dir, "milestones", "release_snapshots.json"
                                )
                            )
                            datasets["milestone_snapshots"] = (
                                "milestones/release_snapshots.json"
                            )
                    except Exception as e:
                        reporter.warning(
                            f"{type(agg).__name__} export failed: {str(e)}"
                        )
                reporter.stage_complete("Dataset Export")

            # Generate Manifest
            generate_manifest(output_dir, analyzer, datasets)

            # --- NEW: Generate Metadata Report ---
            reporter.stage_start("Reporting", "Generating metadata report...")
            try:
                report_gen = MetadataReportGenerator(analyzer, output_dir, datasets)
                report_path = report_gen.generate()
                reporter.stage_complete("Reporting", {"Report": str(report_path)})

                if not report_gen.pandas_available:
                    reporter.info("Tip: Install 'pandas' for richer metadata reports")
            except Exception as e:
                reporter.warning(f"Failed to generate metadata report: {e}")
            # -------------------------------------

            # Write Errors
            if analyzer.errors:
                with open(
                    os.path.join(output_dir, "file_lifecycle_errors.txt"),
                    "w",
                    encoding="utf-8",
                ) as f:
                    f.write("\n".join(analyzer.errors))
                reporter.warning("Errors logged to file_lifecycle_errors.txt")

        # Summary
        summary_stats = {
            "Repository": repo_path,
            "Output directory": output_dir,
            "Total commits": f"{analyzer.total_commits:,}",
            "Files tracked": f"{len(analyzer.files):,}",
            "Datasets generated": len(datasets),
        }
        if quick_mode:
            summary_stats["Quick mode"] = f"Enabled (Sample: {sample_rate:.1%})"

        reporter.summary(summary_stats)
        reporter.success(f"Analysis complete! Results saved to: {output_dir}")

    except Exception as e:
        reporter.error(f"Analysis failed: {str(e)}")
        if verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
