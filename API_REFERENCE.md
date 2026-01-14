# API Reference - Repository Evolution Analyzer v2.0.0

Complete API documentation for developers extending and integrating the analyzer.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Core Classes](#2-core-classes)
3. [Aggregator System](#3-aggregator-system)
4. [Built-in Aggregators](#4-built-in-aggregators)
5. [Helper Classes](#5-helper-classes)
6. [Performance Tools](#6-performance-tools)
7. [Configuration](#7-configuration)
8. [Extension Guide](#8-extension-guide)
9. [Examples](#9-examples)

---

## 1. Overview

### Architecture

```
GitFileLifecycle (Core)
    ├── ProgressReporter (CLI feedback)
    ├── MemoryMonitor (Resource tracking)
    ├── PerformanceMetrics (Statistics)
    └── DatasetAggregator[] (Extensible plugins)
            ├── FileMetadataAggregator
            ├── TemporalAggregator
            ├── AuthorNetworkAggregator
            ├── CochangeNetworkAggregator
            ├── FileHierarchyAggregator
            └── MilestoneAggregator
```

### Design Principles

1. **Streaming Architecture**: Process commits one at a time, maintain O(1) memory per commit
2. **Plugin System**: Aggregators are independent, can be enabled/disabled
3. **Backward Compatibility**: Core output never changes
4. **Performance First**: Caching, memory monitoring, optional profiling

---

## 2. Core Classes

### 2.1 GitFileLifecycle

The main analyzer class responsible for processing git history.

#### Constructor

```python
GitFileLifecycle(
    repo_path: str,
    reporter: Optional[ProgressReporter] = None,
    memory_limit_mb: Optional[float] = None,
    enable_profiling: bool = False
)
```

**Parameters:**

- `repo_path` (str): Absolute or relative path to git repository
- `reporter` (ProgressReporter, optional): Custom progress reporter
- `memory_limit_mb` (float, optional): Memory limit in MB, raises MemoryError if exceeded
- `enable_profiling` (bool): Enable performance profiling

**Attributes:**

- `repo_path` (str): Absolute repository path
- `files` (Dict[str, List[Dict]]): File lifecycle events
- `total_commits` (int): Total commits processed
- `total_changes` (int): Total file change events
- `skipped_unmerged` (int): Unmerged files skipped
- `errors` (List[str]): Error messages
- `aggregators` (List[DatasetAggregator]): Registered aggregators
- `metrics` (PerformanceMetrics): Performance statistics
- `memory_monitor` (MemoryMonitor): Memory tracking
- `reporter` (ProgressReporter): Progress reporter

#### Methods

##### add_aggregator()

Register a dataset aggregator to receive commit events.

```python
def add_aggregator(self, aggregator: DatasetAggregator) -> None
```

**Parameters:**

- `aggregator` (DatasetAggregator): Aggregator instance to register

**Example:**

```python
analyzer = GitFileLifecycle('/path/to/repo')
analyzer.add_aggregator(FileMetadataAggregator(analyzer))
analyzer.add_aggregator(TemporalAggregator(analyzer))
```

##### analyze_repository()

Main analysis method. Streams git log and processes all commits.

```python
def analyze_repository(self, sample_rate: float = 1.0) -> None
```

**Parameters:**

- `sample_rate` (float): Fraction of commits to process (0.0-1.0). Use <1.0 for quick mode.

**Raises:**

- `RuntimeError`: If git command fails
- `MemoryError`: If memory limit exceeded

**Example:**

```python
# Full analysis
analyzer.analyze_repository()

# Quick mode (50% sampling)
analyzer.analyze_repository(sample_rate=0.5)
```

##### export_lifecycle()

Export core lifecycle data to JSON file.

```python
def export_lifecycle(self, output_path: str) -> Dict
```

**Parameters:**

- `output_path` (str): Path for output JSON file

**Returns:**

- `Dict`: The exported data structure

**Example:**

```python
data = analyzer.export_lifecycle('output/file_lifecycle.json')
print(f"Exported {len(data['files'])} files")
```

##### \_parse_commit_line()

Parse a git log commit header line. (Internal method)

```python
def _parse_commit_line(self, line: str) -> Optional[Dict]
```

**Parameters:**

- `line` (str): Commit line from git log (NULL-delimited format)

**Returns:**

- `Dict` with keys: commit_hash, timestamp, author_name, author_email, commit_subject
- `None` if parsing fails

##### \_parse_change_line()

Parse a file change line from git log. (Internal method)

```python
def _parse_change_line(self, line: str) -> Optional[Dict]
```

**Parameters:**

- `line` (str): Change line from git log --name-status

**Returns:**

- `Dict` with keys: operation, file_path, [old_path, similarity]
- `None` if parsing fails

##### \_process_commit()

Process a single commit and notify aggregators. (Internal method)

```python
def _process_commit(self, commit: Dict, changes: List[Dict]) -> None
```

**Parameters:**

- `commit` (Dict): Parsed commit metadata
- `changes` (List[Dict]): List of file changes in this commit

---

## 3. Aggregator System

### 3.1 DatasetAggregator (Base Class)

Abstract base class for all dataset aggregators. Extend this to create custom aggregators.

#### Constructor

```python
DatasetAggregator(
    analyzer: GitFileLifecycle,
    quick_mode: bool = False,
    reporter: Optional[ProgressReporter] = None
)
```

**Parameters:**

- `analyzer` (GitFileLifecycle): Reference to parent analyzer
- `quick_mode` (bool): Enable optimizations for large repositories
- `reporter` (ProgressReporter, optional): Progress reporter

**Attributes:**

- `analyzer` (GitFileLifecycle): Parent analyzer reference
- `quick_mode` (bool): Quick mode flag
- `reporter` (ProgressReporter): Progress reporter
- `data` (Dict): Aggregated data storage
- `processing_time` (float): Time spent processing

#### Methods to Override

##### process_commit()

Called for each commit during streaming. **Must override.**

```python
def process_commit(self, commit: dict, changes: list) -> None
```

**Parameters:**

- `commit` (dict): Commit metadata with keys:
  - `commit_hash` (str)
  - `timestamp` (int)
  - `author_name` (str)
  - `author_email` (str)
  - `commit_subject` (str)
- `changes` (list): File changes, each dict with:
  - `operation` (str): A/M/D/R/C/T
  - `file_path` (str)
  - `old_path` (str, optional)
  - `similarity` (int, optional)

**Example:**

```python
def process_commit(self, commit: dict, changes: list):
    for change in changes:
        file_path = change['file_path']
        # Aggregate data here
        self.data[file_path] = self.data.get(file_path, 0) + 1
```

##### finalize()

Called after all commits processed. Compute final aggregations. **Override if needed.**

```python
def finalize(self) -> dict
```

**Returns:**

- `dict`: Final aggregated data structure

**Example:**

```python
def finalize(self) -> dict:
    return {
        'schema_version': '1.0.0',
        'total_files': len(self.data),
        'files': self.data
    }
```

##### export()

Export aggregated data to file. **Default implementation provided.**

```python
def export(self, output_path: str) -> int
```

**Parameters:**

- `output_path` (str): Path for output JSON file

**Returns:**

- `int`: Size of exported data in bytes

---

## 4. Built-in Aggregators

### 4.1 FileMetadataAggregator

Tracks per-file statistics: first seen, last modified, author counts, etc.

#### Constructor

```python
FileMetadataAggregator(
    analyzer: GitFileLifecycle,
    quick_mode: bool = False,
    reporter: Optional[ProgressReporter] = None
)
```

#### Output Schema

See [SCHEMA_REFERENCE.md - File Metadata Index](#2-file-metadata-index)

#### Example

```python
aggregator = FileMetadataAggregator(analyzer)
analyzer.add_aggregator(aggregator)
analyzer.analyze_repository()
aggregator.export('metadata/file_index.json')
```

---

### 4.2 TemporalAggregator

Aggregates activity by day and month.

#### Constructor

```python
TemporalAggregator(
    analyzer: GitFileLifecycle,
    quick_mode: bool = False,
    reporter: Optional[ProgressReporter] = None
)
```

#### Methods

##### export_daily()

Export daily aggregation.

```python
def export_daily(self, output_path: str) -> None
```

##### export_monthly()

Export monthly aggregation.

```python
def export_monthly(self, output_path: str) -> None
```

#### Example

```python
aggregator = TemporalAggregator(analyzer)
analyzer.add_aggregator(aggregator)
analyzer.analyze_repository()
aggregator.export_daily('aggregations/temporal_daily.json')
aggregator.export_monthly('aggregations/temporal_monthly.json')
```

---

### 4.3 AuthorNetworkAggregator

Builds collaboration network based on shared file modifications.

#### Constructor

```python
AuthorNetworkAggregator(
    analyzer: GitFileLifecycle,
    quick_mode: bool = False,
    max_edges: Optional[int] = None,
    reporter: Optional[ProgressReporter] = None
)
```

**Parameters:**

- `max_edges` (int, optional): Maximum edges to track (for quick mode)

#### Example

```python
aggregator = AuthorNetworkAggregator(analyzer, max_edges=5000)
analyzer.add_aggregator(aggregator)
analyzer.analyze_repository()
aggregator.export('networks/author_network.json')
```

---

### 4.4 CochangeNetworkAggregator

Builds file coupling network (files modified together).

#### Constructor

```python
CochangeNetworkAggregator(
    analyzer: GitFileLifecycle,
    quick_mode: bool = False,
    max_pairs: Optional[int] = None,
    min_cochange_count: int = 2,
    reporter: Optional[ProgressReporter] = None
)
```

**Parameters:**

- `max_pairs` (int, optional): Maximum pairs to track (for quick mode)
- `min_cochange_count` (int): Minimum co-changes to include pair

#### Example

```python
aggregator = CochangeNetworkAggregator(
    analyzer,
    max_pairs=10000,
    min_cochange_count=3
)
analyzer.add_aggregator(aggregator)
analyzer.analyze_repository()
aggregator.export('networks/cochange_network.json')
```

---

### 4.5 FileHierarchyAggregator

Computes directory-level statistics.

#### Constructor

```python
FileHierarchyAggregator(
    analyzer: GitFileLifecycle,
    quick_mode: bool = False,
    reporter: Optional[ProgressReporter] = None
)
```

#### Example

```python
aggregator = FileHierarchyAggregator(analyzer)
analyzer.add_aggregator(aggregator)
analyzer.analyze_repository()
aggregator.export('aggregations/directory_stats.json')
```

---

### 4.6 MilestoneAggregator

Captures repository state at git tags.

#### Constructor

```python
MilestoneAggregator(
    analyzer: GitFileLifecycle,
    repo_path: str,
    quick_mode: bool = False,
    reporter: Optional[ProgressReporter] = None
)
```

**Parameters:**

- `repo_path` (str): Repository path (for tag lookup)

#### Example

```python
aggregator = MilestoneAggregator(analyzer, repo_path='/path/to/repo')
analyzer.add_aggregator(aggregator)
analyzer.analyze_repository()
aggregator.export('milestones/release_snapshots.json')
```

---

## 5. Helper Classes

### 5.1 ProgressReporter

Provides user feedback during analysis.

#### Constructor

```python
ProgressReporter(
    quiet: bool = False,
    verbose: bool = False,
    use_colors: bool = True
)
```

**Parameters:**

- `quiet` (bool): Suppress all output
- `verbose` (bool): Show detailed information
- `use_colors` (bool): Enable colored output (requires colorama)

#### Methods

##### stage_start()

Mark start of a processing stage.

```python
def stage_start(self, stage_name: str, message: str = "") -> None
```

##### stage_complete()

Mark completion of a stage.

```python
def stage_complete(self, stage_name: str, stats: Optional[Dict] = None) -> None
```

##### create_progress_bar()

Create a progress bar (requires tqdm).

```python
def create_progress_bar(self, total: int, desc: str = "Processing") -> Optional[tqdm]
```

**Returns:**

- `tqdm` object if available, `None` otherwise

##### info(), warning(), error(), success()

Display messages with appropriate formatting.

```python
def info(self, message: str) -> None
def warning(self, message: str) -> None
def error(self, message: str) -> None
def success(self, message: str) -> None
```

##### summary()

Display final summary statistics.

```python
def summary(self, stats: Dict[str, Any]) -> None
```

#### Example

```python
reporter = ProgressReporter(verbose=True, use_colors=True)
reporter.stage_start("Analysis", "Processing repository...")
reporter.info("Starting commit scan")
# ... work ...
reporter.stage_complete("Analysis", {'commits': 1000})
reporter.success("Analysis complete!")
```

---

### 5.2 PerformanceMetrics

Tracks performance statistics.

#### Attributes

```python
@dataclass
class PerformanceMetrics:
    commits_processed: int = 0
    files_tracked: int = 0
    aggregator_times: Dict[str, float] = field(default_factory=dict)
    memory_peak_mb: float = 0.0
    total_time: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    chunks_processed: int = 0
```

#### Methods

##### to_dict()

Convert metrics to dictionary.

```python
def to_dict(self) -> Dict
```

**Returns:**

- `Dict` with all metrics and computed statistics

---

## 6. Performance Tools

### 6.1 MemoryMonitor

Monitors memory usage and enforces limits.

#### Constructor

```python
MemoryMonitor(limit_mb: Optional[float] = None)
```

**Parameters:**

- `limit_mb` (float, optional): Memory limit in MB

#### Methods

##### check_memory()

Get current memory usage.

```python
def check_memory(self) -> float
```

**Returns:**

- `float`: Current memory usage in MB

**Raises:**

- `MemoryError`: If limit exceeded

##### get_peak()

Get peak memory usage.

```python
def get_peak(self) -> float
```

**Returns:**

- `float`: Peak memory usage in MB

#### Example

```python
monitor = MemoryMonitor(limit_mb=2048)
try:
    memory_mb = monitor.check_memory()
    print(f"Current memory: {memory_mb:.1f} MB")
except MemoryError as e:
    print(f"Memory limit exceeded: {e}")
```

---

### 6.2 ProfilingContext

Context manager for performance profiling.

#### Constructor

```python
ProfilingContext(
    enabled: bool = False,
    output_path: Optional[str] = None
)
```

**Parameters:**

- `enabled` (bool): Enable profiling
- `output_path` (str, optional): Path to save profile stats

#### Example

```python
with ProfilingContext(enabled=True, output_path='profile.prof'):
    # Your code here
    analyzer.analyze_repository()
# Profile summary printed automatically
```

---

### 6.3 Caching Functions

#### cached_file_hash()

LRU-cached hash generation for file identification.

```python
@lru_cache(maxsize=1024)
def cached_file_hash(file_path: str, commit_hash: str) -> str
```

**Parameters:**

- `file_path` (str): File path
- `commit_hash` (str): Commit hash

**Returns:**

- `str`: 16-character hash (truncated SHA-256)

**Example:**

```python
hash1 = cached_file_hash('src/main.py', 'abc123')
hash2 = cached_file_hash('src/main.py', 'abc123')  # Cache hit
```

#### chunk_iterator()

Memory-efficient chunking for large datasets.

```python
def chunk_iterator(items: List, chunk_size: int = 1000) -> Iterator[List]
```

**Parameters:**

- `items` (List): Items to chunk
- `chunk_size` (int): Size of each chunk

**Yields:**

- `List`: Chunks of specified size

**Example:**

```python
large_list = list(range(100000))
for chunk in chunk_iterator(large_list, chunk_size=1000):
    process_chunk(chunk)  # Process 1000 items at a time
```

---

## 7. Configuration

### 7.1 Configuration Management

#### load_config_file()

Load configuration from YAML or JSON.

```python
def load_config_file(config_path: str) -> Dict[str, Any]
```

**Parameters:**

- `config_path` (str): Path to configuration file

**Returns:**

- `Dict`: Configuration dictionary

**Raises:**

- `FileNotFoundError`: If file doesn't exist
- `ImportError`: If PyYAML not installed (for YAML files)
- `ValueError`: If unsupported format

#### find_config_file()

Auto-discover configuration file.

```python
def find_config_file(repo_path: str) -> Optional[str]
```

**Parameters:**

- `repo_path` (str): Repository path to search

**Returns:**

- `str`: Path to found config file
- `None`: If no config file found

#### ConfigResolver

Helper class to resolve configuration precedence (CLI > Config File > Preset > Defaults).

```python
class ConfigResolver:
    def __init__(
        self,
        cli_params: Dict[str, Any],
        config_path: Optional[str] = None,
        preset_name: Optional[str] = None,
        repo_path: Optional[str] = None
    )
```

**Methods:**

##### get()

Retrieve a configuration value based on precedence.

```python
def get(self, key: str, default: Any = None) -> Any
```

**Parameters:**

- `key` (str): Configuration key (snake_case)
- `default` (Any): Default value if key not found in any source

**Note:** CLI arguments take precedence over config file.

---

## 8. Extension Guide

### 8.1 Creating a Custom Aggregator

Step-by-step guide to create your own aggregator.

#### Step 1: Subclass DatasetAggregator

```python
from strata import DatasetAggregator

class MyCustomAggregator(DatasetAggregator):
    """Custom aggregator for specific analysis"""

    def __init__(self, analyzer, quick_mode=False, reporter=None):
        super().__init__(analyzer, quick_mode, reporter)
        # Initialize your data structures
        self.custom_data = {}
```

#### Step 2: Implement process_commit()

```python
    def process_commit(self, commit: dict, changes: list):
        """Process each commit"""
        for change in changes:
            file_path = change['file_path']
            operation = change['operation']

            # Your custom logic here
            if file_path.endswith('.py'):
                if file_path not in self.custom_data:
                    self.custom_data[file_path] = {
                        'modifications': 0,
                        'last_author': None
                    }

                self.custom_data[file_path]['modifications'] += 1
                self.custom_data[file_path]['last_author'] = commit['author_email']
```

#### Step 3: Override finalize() (Optional)

```python
    def finalize(self) -> dict:
        """Compute final aggregations"""
        # Calculate additional statistics
        total_mods = sum(d['modifications'] for d in self.custom_data.values())

        return {
            'schema_version': '1.0.0',
            'total_python_files': len(self.custom_data),
            'total_modifications': total_mods,
            'files': self.custom_data
        }
```

#### Step 4: Use Your Aggregator

```python
# Create analyzer
analyzer = GitFileLifecycle('/path/to/repo')

# Register your custom aggregator
my_agg = MyCustomAggregator(analyzer)
analyzer.add_aggregator(my_agg)

# Run analysis
analyzer.analyze_repository()

# Export results
my_agg.export('output/my_custom_analysis.json')
```

### 8.2 Best Practices

#### Memory Efficiency

```python
class EfficientAggregator(DatasetAggregator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Use efficient data structures
        from collections import defaultdict, Counter
        self.counters = defaultdict(Counter)

    def process_commit(self, commit, changes):
        # Aggregate incrementally, don't store all raw data
        for change in changes:
            self.counters[change['file_path']]['total'] += 1
```

#### Quick Mode Support

```python
class QuickModeAggregator(DatasetAggregator):
    def __init__(self, *args, max_items=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_items = max_items or (10000 if self.quick_mode else None)
        self.item_count = 0

    def process_commit(self, commit, changes):
        if self.max_items and self.item_count >= self.max_items:
            return  # Stop processing

        # Process normally
        self.item_count += len(changes)
```

#### Error Handling

```python
class RobustAggregator(DatasetAggregator):
    def process_commit(self, commit, changes):
        try:
            # Your processing logic
            pass
        except Exception as e:
            # Log error but don't crash
            if hasattr(self.analyzer, 'errors'):
                self.analyzer.errors.append(
                    f"MyAggregator failed on {commit['commit_hash']}: {e}"
                )
```

---

## 9. Examples

### 9.1 Basic Usage

```python
from strata import GitFileLifecycle, ProgressReporter

# Create analyzer
reporter = ProgressReporter(verbose=True)
analyzer = GitFileLifecycle('/path/to/repo', reporter=reporter)

# Run analysis
analyzer.analyze_repository()

# Export core data
analyzer.export_lifecycle('output/file_lifecycle.json')

print(f"Processed {analyzer.total_commits} commits")
print(f"Tracked {len(analyzer.files)} files")
```

### 9.2 Full Analysis with All Aggregators

```python
from strata import (
    GitFileLifecycle,
    FileMetadataAggregator,
    TemporalAggregator,
    AuthorNetworkAggregator,
    CochangeNetworkAggregator,
    FileHierarchyAggregator,
    MilestoneAggregator,
    ProgressReporter
)

# Setup
reporter = ProgressReporter(verbose=True, use_colors=True)
analyzer = GitFileLifecycle(
    repo_path='/path/to/repo',
    reporter=reporter,
    memory_limit_mb=4096
)

# Register all aggregators
analyzer.add_aggregator(FileMetadataAggregator(analyzer, reporter=reporter))
analyzer.add_aggregator(TemporalAggregator(analyzer, reporter=reporter))
analyzer.add_aggregator(AuthorNetworkAggregator(analyzer, reporter=reporter))
analyzer.add_aggregator(CochangeNetworkAggregator(analyzer, reporter=reporter))
analyzer.add_aggregator(FileHierarchyAggregator(analyzer, reporter=reporter))
analyzer.add_aggregator(MilestoneAggregator(analyzer, repo_path='/path/to/repo', reporter=reporter))

# Analyze
analyzer.analyze_repository()

# Export
analyzer.export_lifecycle('output/file_lifecycle.json')

for agg in analyzer.aggregators:
    agg_name = type(agg).__name__
    output_path = f'output/{agg_name.lower()}.json'
    agg.export(output_path)
    print(f"Exported {agg_name} to {output_path}")
```

### 9.3 Quick Mode for Large Repositories

```python
analyzer = GitFileLifecycle('/huge/repo')

# Use quick mode aggregators
analyzer.add_aggregator(
    FileMetadataAggregator(analyzer, quick_mode=True)
)
analyzer.add_aggregator(
    CochangeNetworkAggregator(analyzer, quick_mode=True, max_pairs=10000)
)

# Sample 30% of commits
analyzer.analyze_repository(sample_rate=0.3)

analyzer.export_lifecycle('output/file_lifecycle.json')
```

### 9.4 Performance Profiling

```python
from strata import ProfilingContext

with ProfilingContext(enabled=True, output_path='profile.prof'):
    analyzer = GitFileLifecycle('/path/to/repo')
    analyzer.analyze_repository()
    analyzer.export_lifecycle('output/file_lifecycle.json')

# Profile summary printed automatically
# Load profile with: python -m pstats profile.prof
```

### 9.5 Custom Aggregator Example

```python
class LanguageStatsAggregator(DatasetAggregator):
    """Track statistics by programming language"""

    EXTENSIONS = {
        '.py': 'Python',
        '.js': 'JavaScript',
        '.java': 'Java',
        '.cpp': 'C++',
        '.c': 'C',
        '.go': 'Go',
        '.rs': 'Rust'
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        from collections import defaultdict
        self.language_stats = defaultdict(lambda: {
            'files': set(),
            'commits': 0,
            'authors': set()
        })

    def process_commit(self, commit, changes):
        for change in changes:
            file_path = change['file_path']
            ext = '.' + file_path.split('.')[-1] if '.' in file_path else None

            if ext in self.EXTENSIONS:
                lang = self.EXTENSIONS[ext]
                stats = self.language_stats[lang]

                stats['files'].add(file_path)
                stats['commits'] += 1
                stats['authors'].add(commit['author_email'])

    def finalize(self):
        result = {}
        for lang, stats in self.language_stats.items():
            result[lang] = {
                'file_count': len(stats['files']),
                'commit_count': stats['commits'],
                'author_count': len(stats['authors']),
                'files': sorted(list(stats['files']))
            }

        return {
            'schema_version': '1.0.0',
            'languages': result,
            'total_languages': len(result)
        }

# Usage
analyzer = GitFileLifecycle('/path/to/repo')
lang_agg = LanguageStatsAggregator(analyzer)
analyzer.add_aggregator(lang_agg)
analyzer.analyze_repository()
lang_agg.export('output/language_stats.json')
```

### 9.6 Programmatic Configuration Resolution

```python
from strata import ConfigResolver, GitFileLifecycle

# Simulate CLI arguments (e.g., from another script or API call)
cli_params = {
    'repo_path': '/path/to/repo',
    'verbose': True,
    # None values allow the resolver to fall back to config/presets
    'memory_limit': None
}

# Initialize resolver
# This handles loading .git-lifecycle.yaml and applying the 'standard' preset
resolver = ConfigResolver(
    cli_params,
    config_path=None,  # Let it auto-discover
    preset_name='standard',
    repo_path=cli_params['repo_path']
)

# Resolve values (CLI > Config > Preset > Default)
memory_limit = resolver.get('memory_limit', 2048)
enable_aggregations = resolver.get('enable_aggregations', False)

# Initialize Analyzer
analyzer = GitFileLifecycle(
    cli_params['repo_path'],
    memory_limit_mb=memory_limit
)
```

## Version History

### v2.0.0 (2024-01-15)

- Initial API documentation
- Phase 4 enhancements (profiling, memory monitoring, caching)
- Configuration file support
- Enhanced progress reporting

---

## See Also

- [SCHEMA_REFERENCE.md](SCHEMA_REFERENCE.md) - Dataset schemas
- [QUICK_REFERENCE.md](QUICK_REFERENCE.md) - Quick start guide
- [PHASE4_COMPLETION.md](PHASE4_COMPLETION.md) - Phase 4 features

---

_Last Updated: 2024-01-15_  
_API Version: 2.0.0_
