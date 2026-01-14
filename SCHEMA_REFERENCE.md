# Schema Reference - Repository Evolution Analyzer v2.0.0

Complete schema documentation for all generated datasets.

---

## Table of Contents

1. [Core Lifecycle](#1-core-lifecycle)
2. [File Metadata Index](#2-file-metadata-index)
3. [Temporal Aggregations](#3-temporal-aggregations)
4. [Author Network](#4-author-network)
5. [Co-change Network](#5-co-change-network)
6. [Directory Hierarchy](#6-directory-hierarchy)
7. [Milestone Snapshots](#7-milestone-snapshots)
8. [Manifest](#8-manifest)
9. [Schema Versioning](#9-schema-versioning)
10. [Data Types Reference](#10-data-types-reference)

---

## 1. Core Lifecycle

**File**: `file_lifecycle.json`  
**Schema Version**: 1.0.0  
**Production Ready**: ✅ Yes

The core dataset containing complete file lifecycle history.

### Structure

```json
{
  "files": {
    "path/to/file.py": [
      {
        "commit_hash": "abc123...",
        "timestamp": 1704067200,
        "datetime": "2024-01-01T00:00:00+00:00",
        "operation": "A",
        "author_name": "John Doe",
        "author_email": "john@example.com",
        "commit_subject": "Add new feature",
        "old_path": "old/path.py",
        "similarity": 95
      }
    ]
  },
  "total_commits": 15234,
  "total_changes": 47892,
  "generated_at": "2024-01-15T10:30:00+00:00",
  "repository_path": "/path/to/repo",
  "skipped_unmerged": 0
}
```

### Field Definitions

#### Root Level

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `files` | Object | Yes | Map of file paths to event arrays |
| `total_commits` | Integer | Yes | Total number of commits processed |
| `total_changes` | Integer | Yes | Total file change events recorded |
| `generated_at` | ISO DateTime | Yes | Generation timestamp (UTC) |
| `repository_path` | String | Yes | Absolute path to analyzed repository |
| `skipped_unmerged` | Integer | Yes | Number of unmerged files skipped |

#### File Event Object

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `commit_hash` | String | Yes | Full 40-character SHA-1 hash |
| `timestamp` | Integer | Yes | Unix timestamp (seconds since epoch) |
| `datetime` | ISO DateTime | Yes | Human-readable timestamp (UTC) |
| `operation` | String | Yes | Operation type: A/M/D/R/C/T |
| `author_name` | String | Yes | Commit author's name |
| `author_email` | String | Yes | Commit author's email |
| `commit_subject` | String | Yes | First line of commit message |
| `old_path` | String | No | Original path (for R/C operations) |
| `similarity` | Integer | No | Similarity percentage (for R/C operations, 0-100) |

#### Operation Types

| Code | Meaning | Description |
|------|---------|-------------|
| `A` | Added | File created |
| `M` | Modified | File content changed |
| `D` | Deleted | File removed |
| `R` | Renamed | File moved/renamed |
| `C` | Copied | File copied |
| `T` | Type Changed | File type changed (e.g., symlink ↔ regular file) |

### Example

```json
{
  "files": {
    "src/main.py": [
      {
        "commit_hash": "a1b2c3d4e5f6789012345678901234567890abcd",
        "timestamp": 1704067200,
        "datetime": "2024-01-01T00:00:00+00:00",
        "operation": "A",
        "author_name": "Alice Developer",
        "author_email": "alice@company.com",
        "commit_subject": "Initial commit"
      },
      {
        "commit_hash": "b2c3d4e5f67890123456789012345678901abcde",
        "timestamp": 1704153600,
        "datetime": "2024-01-02T00:00:00+00:00",
        "operation": "M",
        "author_name": "Bob Engineer",
        "author_email": "bob@company.com",
        "commit_subject": "Fix bug in main"
      },
      {
        "commit_hash": "c3d4e5f678901234567890123456789012abcdef",
        "timestamp": 1704240000,
        "datetime": "2024-01-03T00:00:00+00:00",
        "operation": "R",
        "author_name": "Alice Developer",
        "author_email": "alice@company.com",
        "commit_subject": "Reorganize project structure",
        "old_path": "main.py",
        "similarity": 100
      }
    ]
  },
  "total_commits": 3,
  "total_changes": 3,
  "generated_at": "2024-01-15T10:30:00+00:00",
  "repository_path": "/home/user/project",
  "skipped_unmerged": 0
}
```

---

## 2. File Metadata Index

**File**: `metadata/file_index.json`  
**Schema Version**: 1.0.0  
**Production Ready**: ⚠️ Preview (Phase 2)

Per-file statistics and metadata aggregated from lifecycle history.

### Structure

```json
{
  "schema_version": "1.0.0",
  "total_files": 1250,
  "generation_method": "streaming",
  "files": {
    "path/to/file.py": {
      "first_seen": "2023-01-15T10:30:00+00:00",
      "last_modified": "2024-01-10T15:45:00+00:00",
      "total_commits": 45,
      "unique_authors": 7,
      "primary_author": {
        "email": "dev@example.com",
        "commit_count": 28,
        "percentage": 62.2
      },
      "operations": {
        "A": 1,
        "M": 43,
        "D": 0,
        "R": 1,
        "C": 0
      },
      "age_days": 360,
      "commits_per_day": 0.125,
      "lifecycle_event_count": 45
    }
  }
}
```

### Field Definitions

#### Root Level

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `schema_version` | String | Yes | Schema version (semver) |
| `total_files` | Integer | Yes | Total unique files tracked |
| `generation_method` | String | Yes | Always "streaming" |
| `files` | Object | Yes | Map of file paths to metadata |

#### File Metadata Object

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `first_seen` | ISO DateTime | Yes | First commit affecting this file |
| `last_modified` | ISO DateTime | Yes | Most recent commit affecting this file |
| `total_commits` | Integer | Yes | Number of commits touching this file |
| `unique_authors` | Integer | Yes | Number of distinct authors |
| `primary_author` | Object | Yes | Author with most commits |
| `operations` | Object | Yes | Count of each operation type |
| `age_days` | Float | Yes | Days between first seen and last modified |
| `commits_per_day` | Float | Yes | Average commit frequency |
| `lifecycle_event_count` | Integer | Yes | Total events (should equal total_commits) |

#### Primary Author Object

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `email` | String | Yes | Author's email address |
| `commit_count` | Integer | Yes | Number of commits by this author |
| `percentage` | Float | Yes | Percentage of total commits (0-100) |

---

## 3. Temporal Aggregations

### 3.1 Daily Aggregation

**File**: `aggregations/temporal_daily.json`  
**Schema Version**: 1.0.0

Daily activity statistics aggregated from commit history.

#### Structure

```json
{
  "schema_version": "1.0.0",
  "aggregation_level": "daily",
  "date_range": {
    "start": "2023-01-01",
    "end": "2024-01-15"
  },
  "total_days": 380,
  "days_with_activity": 245,
  "daily_stats": {
    "2024-01-01": {
      "commit_count": 15,
      "file_changes": 47,
      "unique_files": 23,
      "unique_authors": 5,
      "operations": {
        "A": 3,
        "M": 42,
        "D": 2
      },
      "authors": ["alice@example.com", "bob@example.com"]
    }
  }
}
```

#### Field Definitions

**Root Level**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `schema_version` | String | Yes | Schema version |
| `aggregation_level` | String | Yes | Always "daily" |
| `date_range` | Object | Yes | Start and end dates (ISO date) |
| `total_days` | Integer | Yes | Total days in range |
| `days_with_activity` | Integer | Yes | Days with at least one commit |
| `daily_stats` | Object | Yes | Map of ISO dates to day stats |

**Daily Stats Object**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `commit_count` | Integer | Yes | Commits on this day |
| `file_changes` | Integer | Yes | Total file change operations |
| `unique_files` | Integer | Yes | Distinct files modified |
| `unique_authors` | Integer | Yes | Distinct authors active |
| `operations` | Object | Yes | Breakdown by operation type |
| `authors` | Array[String] | Yes | List of author emails |

### 3.2 Monthly Aggregation

**File**: `aggregations/temporal_monthly.json`  
**Schema Version**: 1.0.0

Monthly activity statistics (same structure as daily, but aggregated by month).

#### Structure

```json
{
  "schema_version": "1.0.0",
  "aggregation_level": "monthly",
  "date_range": {
    "start": "2023-01",
    "end": "2024-01"
  },
  "total_months": 13,
  "months_with_activity": 13,
  "monthly_stats": {
    "2024-01": {
      "commit_count": 342,
      "file_changes": 1156,
      "unique_files": 287,
      "unique_authors": 12,
      "operations": {
        "A": 45,
        "M": 1089,
        "D": 22
      },
      "authors": ["alice@example.com", "bob@example.com"]
    }
  }
}
```

---

## 4. Author Network

**File**: `networks/author_network.json`  
**Schema Version**: 1.0.0  
**Production Ready**: ⚠️ Preview (Phase 2)

Collaboration network showing which authors work on the same files.

### Structure

```json
{
  "schema_version": "1.0.0",
  "network_type": "author_collaboration",
  "quick_mode": false,
  "edge_limit": null,
  "nodes": [
    {
      "id": "alice@example.com",
      "email": "alice@example.com",
      "commit_count": 342,
      "collaboration_count": 8
    }
  ],
  "edges": [
    {
      "source": "alice@example.com",
      "target": "bob@example.com",
      "weight": 45,
      "shared_files": 23
    }
  ],
  "statistics": {
    "total_authors": 12,
    "total_edges": 34,
    "density": 0.5152
  }
}
```

### Field Definitions

#### Root Level

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `schema_version` | String | Yes | Schema version |
| `network_type` | String | Yes | Always "author_collaboration" |
| `quick_mode` | Boolean | Yes | Whether quick mode was used |
| `edge_limit` | Integer/null | Yes | Max edges in quick mode |
| `nodes` | Array[Object] | Yes | Author nodes |
| `edges` | Array[Object] | Yes | Collaboration edges |
| `statistics` | Object | Yes | Network statistics |

#### Node Object

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `id` | String | Yes | Author email (unique identifier) |
| `email` | String | Yes | Author email (same as id) |
| `commit_count` | Integer | Yes | Total commits by this author |
| `collaboration_count` | Integer | Yes | Number of collaborators (edges) |

#### Edge Object

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `source` | String | Yes | Source author email |
| `target` | String | Yes | Target author email |
| `weight` | Integer | Yes | Number of shared file modifications |
| `shared_files` | Integer | Yes | Count of distinct shared files |

#### Statistics Object

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `total_authors` | Integer | Yes | Number of nodes |
| `total_edges` | Integer | Yes | Number of edges |
| `density` | Float | Yes | Network density (0-1) |

**Network Density Formula**: `edges / (nodes * (nodes - 1) / 2)`

---

## 5. Co-change Network

**File**: `networks/cochange_network.json`  
**Schema Version**: 1.0.0  
**Production Ready**: ⚠️ Preview (Phase 3)

File coupling network showing which files are frequently modified together.

### Structure

```json
{
  "schema_version": "1.0.0",
  "network_type": "file_cochange",
  "quick_mode": false,
  "pair_limit": null,
  "min_cochange_count": 2,
  "nodes": [
    {
      "id": "src/main.py",
      "file_path": "src/main.py",
      "total_commits": 156,
      "coupling_count": 12
    }
  ],
  "edges": [
    {
      "source": "src/main.py",
      "target": "src/config.py",
      "weight": 23,
      "confidence": 0.1474
    }
  ],
  "statistics": {
    "total_files": 234,
    "total_pairs": 456,
    "average_weight": 4.7,
    "max_weight": 45
  }
}
```

### Field Definitions

#### Root Level

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `schema_version` | String | Yes | Schema version |
| `network_type` | String | Yes | Always "file_cochange" |
| `quick_mode` | Boolean | Yes | Whether quick mode was used |
| `pair_limit` | Integer/null | Yes | Max pairs in quick mode |
| `min_cochange_count` | Integer | Yes | Minimum co-changes to include pair |
| `nodes` | Array[Object] | Yes | File nodes |
| `edges` | Array[Object] | Yes | Co-change edges |
| `statistics` | Object | Yes | Network statistics |

#### Node Object

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `id` | String | Yes | File path (unique identifier) |
| `file_path` | String | Yes | File path (same as id) |
| `total_commits` | Integer | Yes | Total commits affecting this file |
| `coupling_count` | Integer | Yes | Number of files coupled with |

#### Edge Object

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `source` | String | Yes | Source file path |
| `target` | String | Yes | Target file path |
| `weight` | Integer | Yes | Number of times modified together |
| `confidence` | Float | Yes | Coupling confidence (0-1) |

**Coupling Confidence Formula**: `weight / min(source_commits, target_commits)`

#### Statistics Object

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `total_files` | Integer | Yes | Number of nodes |
| `total_pairs` | Integer | Yes | Number of edges |
| `average_weight` | Float | Yes | Average co-change count |
| `max_weight` | Integer | Yes | Maximum co-change count |

---

## 6. Directory Hierarchy

**File**: `aggregations/directory_stats.json`  
**Schema Version**: 1.0.0  
**Production Ready**: ⚠️ Preview (Phase 3)

Hierarchical statistics showing activity by directory.

### Structure

```json
{
  "schema_version": "1.0.0",
  "hierarchy_type": "directory",
  "root_path": ".",
  "directories": {
    "src": {
      "path": "src",
      "depth": 1,
      "file_count": 45,
      "total_commits": 892,
      "unique_authors": 8,
      "last_modified": "2024-01-15T10:30:00+00:00",
      "subdirectories": ["src/components", "src/utils"],
      "top_files": [
        {
          "path": "src/main.py",
          "commit_count": 156
        }
      ]
    }
  },
  "statistics": {
    "total_directories": 23,
    "max_depth": 5,
    "total_files": 287,
    "average_files_per_directory": 12.5
  }
}
```

### Field Definitions

#### Root Level

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `schema_version` | String | Yes | Schema version |
| `hierarchy_type` | String | Yes | Always "directory" |
| `root_path` | String | Yes | Root directory analyzed |
| `directories` | Object | Yes | Map of directory paths to stats |
| `statistics` | Object | Yes | Overall statistics |

#### Directory Object

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `path` | String | Yes | Directory path relative to root |
| `depth` | Integer | Yes | Nesting depth (0 = root) |
| `file_count` | Integer | Yes | Direct files in this directory |
| `total_commits` | Integer | Yes | Sum of all file commits |
| `unique_authors` | Integer | Yes | Distinct authors in this directory |
| `last_modified` | ISO DateTime | Yes | Most recent commit timestamp |
| `subdirectories` | Array[String] | Yes | List of immediate subdirectory paths |
| `top_files` | Array[Object] | Yes | Files with most commits (top 10) |

#### Top File Object

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `path` | String | Yes | File path |
| `commit_count` | Integer | Yes | Number of commits |

---

## 7. Milestone Snapshots

**File**: `milestones/release_snapshots.json`  
**Schema Version**: 1.0.0  
**Production Ready**: ⚠️ Preview (Phase 3)

Repository state at important milestones (tags, releases).

### Structure

```json
{
  "schema_version": "1.0.0",
  "milestone_type": "git_tags",
  "repository_path": "/path/to/repo",
  "milestones": [
    {
      "name": "v1.0.0",
      "commit_hash": "abc123...",
      "timestamp": 1704067200,
      "datetime": "2024-01-01T00:00:00+00:00",
      "tagger": "release-bot@company.com",
      "message": "Release version 1.0.0",
      "snapshot": {
        "total_files": 234,
        "total_commits": 1456,
        "unique_authors": 12,
        "lines_of_code": 15678,
        "file_types": {
          ".py": 89,
          ".js": 45,
          ".md": 12
        }
      }
    }
  ],
  "statistics": {
    "total_milestones": 15,
    "date_range": {
      "first": "2023-01-01T00:00:00+00:00",
      "last": "2024-01-15T00:00:00+00:00"
    }
  }
}
```

### Field Definitions

#### Root Level

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `schema_version` | String | Yes | Schema version |
| `milestone_type` | String | Yes | Always "git_tags" |
| `repository_path` | String | Yes | Path to analyzed repository |
| `milestones` | Array[Object] | Yes | List of milestone snapshots |
| `statistics` | Object | Yes | Summary statistics |

#### Milestone Object

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | String | Yes | Tag/milestone name |
| `commit_hash` | String | Yes | Commit SHA-1 |
| `timestamp` | Integer | Yes | Unix timestamp |
| `datetime` | ISO DateTime | Yes | Human-readable timestamp |
| `tagger` | String | Yes | Person who created the tag |
| `message` | String | Yes | Tag annotation message |
| `snapshot` | Object | Yes | Repository state at this point |

#### Snapshot Object

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `total_files` | Integer | Yes | Number of tracked files |
| `total_commits` | Integer | Yes | Total commits up to this point |
| `unique_authors` | Integer | Yes | Distinct authors |
| `lines_of_code` | Integer | Yes | Total LOC (if available) |
| `file_types` | Object | Yes | Count by file extension |

---

## 8. Manifest

**File**: `manifest.json`  
**Schema Version**: 1.0.0  
**Production Ready**: ✅ Yes

Catalog of all generated datasets with metadata and checksums.

### Structure

```json
{
  "generator_version": "2.0.0",
  "schema_version": "1.0.0",
  "generated_at": "2024-01-15T10:30:00+00:00",
  "repository": "/path/to/repo",
  "performance_metrics": {
    "commits_processed": 15234,
    "files_tracked": 2567,
    "aggregator_times": {
      "FileMetadataAggregator": 2.34,
      "TemporalAggregator": 1.89
    },
    "memory_peak_mb": 342.5,
    "total_time_seconds": 135.42,
    "cache_statistics": {
      "hits": 12456,
      "misses": 2789,
      "hit_rate_percent": 81.7
    },
    "chunks_processed": 0
  },
  "datasets": {
    "core_lifecycle": {
      "file": "file_lifecycle.json",
      "schema_version": "1.0.0",
      "file_size_bytes": 2458624,
      "sha256": "abc123...",
      "production_ready": true
    },
    "file_metadata": {
      "file": "metadata/file_index.json",
      "schema_version": "1.0.0",
      "file_size_bytes": 156800,
      "sha256": "def456...",
      "production_ready": false
    }
  }
}
```

### Field Definitions

#### Root Level

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `generator_version` | String | Yes | Script version (semver) |
| `schema_version` | String | Yes | Manifest schema version |
| `generated_at` | ISO DateTime | Yes | Generation timestamp |
| `repository` | String | Yes | Repository path |
| `performance_metrics` | Object | Yes | Execution metrics |
| `datasets` | Object | Yes | Map of dataset names to metadata |

#### Dataset Object

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `file` | String | Yes | Relative path to dataset file |
| `schema_version` | String | Yes | Dataset schema version |
| `file_size_bytes` | Integer | Yes | File size in bytes |
| `sha256` | String | Yes | SHA-256 checksum (hex) |
| `production_ready` | Boolean | Yes | Whether dataset is production-stable |

---

## 9. Schema Versioning

All schemas follow **Semantic Versioning** (semver): `MAJOR.MINOR.PATCH`

### Version Changes

- **MAJOR**: Breaking changes (incompatible with previous version)
- **MINOR**: New fields added (backward compatible)
- **PATCH**: Bug fixes, documentation updates (fully compatible)

### Current Versions

| Dataset | Schema Version | Status |
|---------|---------------|--------|
| Core Lifecycle | 1.0.0 | Production |
| File Metadata | 1.0.0 | Preview |
| Temporal | 1.0.0 | Preview |
| Author Network | 1.0.0 | Preview |
| Co-change Network | 1.0.0 | Preview |
| Directory Hierarchy | 1.0.0 | Preview |
| Milestone Snapshots | 1.0.0 | Preview |
| Manifest | 1.0.0 | Production |

### Compatibility Guarantee

- **Core Lifecycle** (`file_lifecycle.json`): Schema locked at 1.0.0, will never change
- **Other datasets**: May evolve with minor/patch updates
- **Manifest**: Always includes `schema_version` field for each dataset

---

## 10. Data Types Reference

### String

UTF-8 encoded text. No length limit unless specified.

### Integer

Signed 64-bit integer. Range: -9,223,372,036,854,775,808 to 9,223,372,036,854,775,807

### Float

IEEE 754 double-precision floating-point.

### Boolean

`true` or `false`

### ISO DateTime

ISO 8601 format with timezone: `YYYY-MM-DDTHH:MM:SS+00:00`

Example: `2024-01-15T10:30:00+00:00`

### ISO Date

ISO 8601 date format: `YYYY-MM-DD`

Example: `2024-01-15`

### ISO Month

Year and month: `YYYY-MM`

Example: `2024-01`

### Unix Timestamp

Seconds since Unix epoch (January 1, 1970 00:00:00 UTC).

Example: `1704067200` = 2024-01-01 00:00:00 UTC

### SHA-1 Hash

40-character hexadecimal string (lowercase).

Example: `a1b2c3d4e5f6789012345678901234567890abcd`

### SHA-256 Hash

64-character hexadecimal string (lowercase).

Example: `abc123...` (truncated for readability in examples)

### Email Address

Standard email format: `user@domain.com`

### File Path

Relative path from repository root using forward slashes.

Example: `src/components/Button.js`

---

## Validation

### Schema Validation

Use JSON Schema validators to check conformance:

```bash
# Example with ajv-cli
npm install -g ajv-cli
ajv validate -s schema.json -d data.json
```

### Checksum Verification

Verify data integrity using SHA-256 checksums from manifest:

```bash
sha256sum file_lifecycle.json
# Compare with manifest.json -> datasets -> core_lifecycle -> sha256
```

---

## Changelog

### v1.0.0 (2024-01-15)
- Initial schema documentation
- Core lifecycle schema (production)
- Phase 2 and Phase 3 schemas (preview)
- Manifest schema (production)

---

## Support

For schema questions or issues:
- Check examples in this document
- Review actual generated files
- Refer to API_REFERENCE.md for implementation details
- Report schema bugs via your issue tracking system

---

*Last Updated: 2024-01-15*  
*Schema Version: 1.0.0*  
*Generator Version: 2.0.0*
