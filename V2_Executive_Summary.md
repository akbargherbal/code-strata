# Dataset Schema Migration Guide (v1 → v2)

**Effective Date:** January 14, 2026  
**Audience:** Data consumers, analysts, visualization developers  
**Purpose:** Schema changes and migration instructions for repository lifecycle datasets

---

## Executive Summary

**Critical Information:**

- ✅ **Core dataset unchanged**: `file_lifecycle.json` structure is identical
- ⚠️ **Four datasets relocated**: New directory structure
- 🆕 **Six new datasets**: Enhanced analytics capabilities
- 📋 **New metadata file**: `manifest.json` for integrity verification

**Action Required:** Update file paths and field mappings (details below)

---

## I. What Changed: Dataset Inventory

### Datasets That Remained Identical

| File (v1)             | File (v2)             | Status            |
| --------------------- | --------------------- | ----------------- |
| `file_lifecycle.json` | `file_lifecycle.json` | ✅ **No changes** |

**Action:** None required. Existing code works as-is.

---

### Datasets That Were Renamed/Restructured

| Old File (v1)           | New File (v2)                        | Status                 |
| ----------------------- | ------------------------------------ | ---------------------- |
| `timeline_monthly.json` | `aggregations/temporal_monthly.json` | ⚠️ **Schema enhanced** |
| `top_active_files.json` | `metadata/file_index.json`           | ⚠️ **Schema changed**  |
| `file_type_stats.json`  | `metadata/file_index.json`           | ⚠️ **Schema changed**  |
| `author_activity.json`  | `networks/author_network.json`       | ⚠️ **Schema changed**  |

**Action:** Update file paths and data extraction logic (see Section II)

---

### New Datasets (v2 Only)

| New File                            | Description                       |
| ----------------------------------- | --------------------------------- |
| `aggregations/temporal_daily.json`  | Daily activity aggregations       |
| `networks/cochange_network.json`    | File coupling analysis            |
| `aggregations/directory_stats.json` | Directory-level statistics        |
| `milestones/release_snapshots.json` | Repository state at tags/releases |
| `manifest.json`                     | Dataset catalog with checksums    |

**Action:** Optional integration for enhanced analytics

---

## II. Migration Instructions by Dataset

### 🟢 Dataset 1: Core Lifecycle (NO CHANGES)

**File:** `file_lifecycle.json`

**Schema (v1 and v2 - identical):**

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
        "commit_subject": "Add feature"
      }
    ]
  },
  "total_commits": 1234,
  "total_changes": 5678,
  "generated_at": "2024-01-15T10:30:00+00:00",
  "repository_path": "/path/to/repo",
  "skipped_unmerged": 0
}
```

**Migration:**

```javascript
// v1 code (still works)
const data = JSON.parse(fs.readFileSync("file_lifecycle.json"));
const events = data.files["src/main.py"];

// No changes needed ✅
```

---

### ⚠️ Dataset 2: Timeline/Temporal Data

**File Change:** `timeline_monthly.json` → `aggregations/temporal_monthly.json`

#### Schema Comparison

**v1 Schema:**

```json
{
  "timeline": [
    {
      "month": "2024-01",
      "adds": 45,
      "modifies": 234,
      "deletes": 12
    }
  ],
  "generated_at": "2024-01-15T10:30:00+00:00"
}
```

**v2 Schema:**

```json
{
  "schema_version": "1.0.0",
  "aggregation_level": "monthly",
  "total_months": 12,
  "months": {
    "2024-01": {
      "month": "2024-01",
      "commits": 342,
      "files_changed": 1156,
      "unique_authors": 12,
      "operations": {
        "A": 45,
        "M": 234,
        "D": 12
      }
    }
  }
}
```

#### Migration Code

**JavaScript/TypeScript:**

```javascript
// v1 code
const data = require("./timeline_monthly.json");
const janData = data.timeline.find((m) => m.month === "2024-01");
const adds = janData.adds;

// v2 migration
const data = require("./aggregations/temporal_monthly.json");
const janData = data.months["2024-01"];
const adds = janData.operations.A; // Changed: nested in 'operations'
```

**Python:**

```python
# v1 code
with open('timeline_monthly.json') as f:
    data = json.load(f)
jan_data = next(m for m in data['timeline'] if m['month'] == '2024-01')
adds = jan_data['adds']

# v2 migration
with open('aggregations/temporal_monthly.json') as f:
    data = json.load(f)
jan_data = data['months']['2024-01']
adds = jan_data['operations']['A']  # Changed: nested in 'operations'
```

**Key Changes:**
| Field (v1) | Field (v2) | Notes |
|------------|------------|-------|
| `timeline[].month` | `months[key].month` | Array → Object (keyed by month) |
| `timeline[].adds` | `months[key].operations.A` | Nested under `operations` |
| `timeline[].modifies` | `months[key].operations.M` | Nested under `operations` |
| `timeline[].deletes` | `months[key].operations.D` | Nested under `operations` |
| _(not present)_ | `months[key].commits` | **New field** |
| _(not present)_ | `months[key].unique_authors` | **New field** |

---

### ⚠️ Dataset 3: Top Active Files

**File Change:** `top_active_files.json` → `metadata/file_index.json`

#### Schema Comparison

**v1 Schema:**

```json
{
  "top_files": [
    {
      "path": "src/main.py",
      "change_count": 156,
      "first_seen": "2023-01-15T10:30:00+00:00",
      "last_seen": "2024-01-10T15:45:00+00:00",
      "operations": {
        "adds": 1,
        "modifies": 154,
        "deletes": 1
      }
    }
  ],
  "total_files_analyzed": 2567
}
```

**v2 Schema:**

```json
{
  "schema_version": "1.0.0",
  "total_files": 2567,
  "files": {
    "src/main.py": {
      "first_seen": "2023-01-15T10:30:00+00:00",
      "last_modified": "2024-01-10T15:45:00+00:00",
      "total_commits": 156,
      "unique_authors": 7,
      "primary_author": {
        "email": "dev@example.com",
        "commit_count": 98,
        "percentage": 62.8
      },
      "operations": {
        "A": 1,
        "M": 154,
        "D": 1
      },
      "age_days": 360.5,
      "commits_per_day": 0.433
    }
  }
}
```

#### Migration Code

**JavaScript/TypeScript:**

```javascript
// v1 code - Get top 10 files
const data = require("./top_active_files.json");
const topFiles = data.top_files.slice(0, 10);

// v2 migration - Need to sort manually
const data = require("./metadata/file_index.json");
const topFiles = Object.entries(data.files)
  .map(([path, stats]) => ({ path, ...stats }))
  .sort((a, b) => b.total_commits - a.total_commits)
  .slice(0, 10);

// Access data
const file = topFiles[0];
console.log(file.path);
console.log(file.total_commits); // Changed from 'change_count'
```

**Python:**

```python
# v1 code
with open('top_active_files.json') as f:
    data = json.load(f)
top_files = data['top_files'][:10]

# v2 migration
with open('metadata/file_index.json') as f:
    data = json.load(f)
top_files = sorted(
    [{'path': k, **v} for k, v in data['files'].items()],
    key=lambda x: x['total_commits'],
    reverse=True
)[:10]

# Access data
file = top_files[0]
print(file['path'])
print(file['total_commits'])  # Changed from 'change_count'
```

**Key Changes:**
| Field (v1) | Field (v2) | Notes |
|------------|------------|-------|
| `top_files[]` | `files{}` | Array → Object (keyed by path) |
| `.change_count` | `.total_commits` | Renamed |
| `.last_seen` | `.last_modified` | Renamed |
| `.operations.adds` | `.operations.A` | Changed to Git operation codes |
| `.operations.modifies` | `.operations.M` | Changed to Git operation codes |
| `.operations.deletes` | `.operations.D` | Changed to Git operation codes |
| _(not present)_ | `.unique_authors` | **New field** |
| _(not present)_ | `.primary_author` | **New field** |
| _(not present)_ | `.age_days` | **New field** |

---

### ⚠️ Dataset 4: File Type Statistics

**File Change:** `file_type_stats.json` → `metadata/file_index.json` _(merged)_

#### v1 Schema:

```json
{
  "file_types": [
    {
      "extension": ".py",
      "count": 89,
      "total_changes": 4523
    }
  ]
}
```

#### v2 Migration:

This data is no longer pre-aggregated. Compute from `file_index.json`:

**JavaScript/TypeScript:**

```javascript
const data = require("./metadata/file_index.json");

// Aggregate by extension
const fileTypes = {};
Object.entries(data.files).forEach(([path, stats]) => {
  const ext = path.includes(".") ? path.match(/\.[^.]+$/)[0] : "(no extension)";

  if (!fileTypes[ext]) {
    fileTypes[ext] = { extension: ext, count: 0, total_changes: 0 };
  }

  fileTypes[ext].count++;
  fileTypes[ext].total_changes += stats.total_commits;
});

const result = Object.values(fileTypes).sort(
  (a, b) => b.total_changes - a.total_changes
);
```

**Python:**

```python
from pathlib import Path
from collections import defaultdict

with open('metadata/file_index.json') as f:
    data = json.load(f)

file_types = defaultdict(lambda: {'count': 0, 'total_changes': 0})

for path, stats in data['files'].items():
    ext = Path(path).suffix or '(no extension)'
    file_types[ext]['extension'] = ext
    file_types[ext]['count'] += 1
    file_types[ext]['total_changes'] += stats['total_commits']

result = sorted(
    file_types.values(),
    key=lambda x: x['total_changes'],
    reverse=True
)
```

---

### ⚠️ Dataset 5: Author Activity

**File Change:** `author_activity.json` → `networks/author_network.json`

#### Schema Comparison

**v1 Schema:**

```json
{
  "authors": [
    {
      "author": "alice@example.com",
      "email": "alice@example.com",
      "unique_commits": 342,
      "files_touched": 156,
      "total_changes": 892
    }
  ],
  "total_authors": 12
}
```

**v2 Schema:**

```json
{
  "schema_version": "1.0.0",
  "network_type": "author_collaboration",
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

#### Migration Code

**JavaScript/TypeScript:**

```javascript
// v1 code
const data = require("./author_activity.json");
const authors = data.authors;
const alice = authors.find((a) => a.email === "alice@example.com");
console.log(alice.unique_commits, alice.files_touched);

// v2 migration
const data = require("./networks/author_network.json");
const authors = data.nodes;
const alice = authors.find((a) => a.email === "alice@example.com");
console.log(alice.commit_count, alice.collaboration_count);

// New: Access collaboration data
const aliceCollabs = data.edges.filter(
  (e) => e.source === "alice@example.com" || e.target === "alice@example.com"
);
```

**Key Changes:**
| Field (v1) | Field (v2) | Notes |
|------------|------------|-------|
| `authors[]` | `nodes[]` | Renamed for network structure |
| `.unique_commits` | `.commit_count` | Renamed |
| `.files_touched` | _(removed)_ | No longer available |
| `.total_changes` | _(removed)_ | No longer available |
| _(not present)_ | `.collaboration_count` | **New field** (# of collaborators) |
| _(not present)_ | `edges[]` | **New section** (collaboration relationships) |

**To get `files_touched` in v2:** Cross-reference with `file_index.json`:

```javascript
const fileIndex = require("./metadata/file_index.json");
const aliceFiles = new Set();

Object.entries(fileIndex.files).forEach(([path, stats]) => {
  // This requires walking lifecycle events (not in file_index)
  // Use file_lifecycle.json instead
});

const lifecycle = require("./file_lifecycle.json");
Object.entries(lifecycle.files).forEach(([path, events]) => {
  events.forEach((event) => {
    if (event.author_email === "alice@example.com") {
      aliceFiles.add(path);
    }
  });
});

console.log("Files touched:", aliceFiles.size);
```

---

## III. New Datasets Reference

### Daily Temporal Data

**File:** `aggregations/temporal_daily.json`

**Structure:**

```json
{
  "schema_version": "1.0.0",
  "aggregation_level": "daily",
  "total_days": 380,
  "days": {
    "2024-01-15": {
      "date": "2024-01-15",
      "commits": 15,
      "files_changed": 47,
      "unique_authors": 5,
      "operations": { "A": 3, "M": 42, "D": 2 }
    }
  }
}
```

**Usage:**

```javascript
const data = require("./aggregations/temporal_daily.json");
const jan15 = data.days["2024-01-15"];
console.log(`${jan15.commits} commits, ${jan15.unique_authors} authors`);
```

---

### Co-change Network

**File:** `networks/cochange_network.json`

**Structure:**

```json
{
  "schema_version": "1.0.0",
  "network_type": "file_cochange",
  "total_files": 234,
  "total_edges": 456,
  "edges": [
    {
      "source": "src/main.py",
      "target": "src/config.py",
      "cochange_count": 23,
      "coupling_strength": 0.6744
    }
  ]
}
```

**Usage (find files frequently changed with `main.py`):**

```javascript
const data = require("./networks/cochange_network.json");
const mainCoupling = data.edges
  .filter((e) => e.source === "src/main.py" || e.target === "src/main.py")
  .sort((a, b) => b.coupling_strength - a.coupling_strength);

console.log("Strongly coupled files:", mainCoupling.slice(0, 5));
```

---

### Directory Statistics

**File:** `aggregations/directory_stats.json`

**Structure:**

```json
{
  "schema_version": "1.0.0",
  "total_directories": 23,
  "directories": {
    "src": {
      "path": "src",
      "total_files": 45,
      "total_commits": 892,
      "unique_authors": 8,
      "operations": { "A": 12, "M": 856, "D": 24 },
      "activity_score": 19.82
    }
  }
}
```

**Usage:**

```javascript
const data = require("./aggregations/directory_stats.json");
const srcStats = data.directories["src"];
console.log(
  `src/: ${srcStats.total_commits} commits by ${srcStats.unique_authors} authors`
);
```

---

### Manifest (Checksums & Metadata)

**File:** `manifest.json`

**Structure:**

```json
{
  "generator_version": "2.0.0",
  "schema_version": "1.0.0",
  "generated_at": "2024-01-15T10:30:00+00:00",
  "datasets": {
    "core_lifecycle": {
      "file": "file_lifecycle.json",
      "schema_version": "1.0.0",
      "file_size_bytes": 2458624,
      "sha256": "abc123...",
      "production_ready": true
    }
  }
}
```

**Usage (verify integrity):**

```bash
# Bash
expected=$(jq -r '.datasets.core_lifecycle.sha256' manifest.json)
actual=$(sha256sum file_lifecycle.json | awk '{print $1}')
[ "$expected" = "$actual" ] && echo "✓ Valid" || echo "✗ Corrupted"
```

```javascript
// Node.js
const crypto = require("crypto");
const fs = require("fs");

const manifest = require("./manifest.json");
const fileContent = fs.readFileSync("file_lifecycle.json");
const actualHash = crypto
  .createHash("sha256")
  .update(fileContent)
  .digest("hex");
const expectedHash = manifest.datasets.core_lifecycle.sha256;

console.log(actualHash === expectedHash ? "✓ Valid" : "✗ Corrupted");
```

---

## IV. Quick Migration Checklist

### For `file_lifecycle.json` Consumers

- [ ] ✅ No changes required
- [ ] (Optional) Add integrity check using `manifest.json`

### For `timeline_monthly.json` Consumers

- [ ] Update path: `timeline_monthly.json` → `aggregations/temporal_monthly.json`
- [ ] Change data access: `timeline[]` → `months{}`
- [ ] Update field names: `adds` → `operations.A`, etc.

### For `top_active_files.json` Consumers

- [ ] Update path: `top_active_files.json` → `metadata/file_index.json`
- [ ] Add client-side sorting (data is no longer pre-sorted)
- [ ] Update field names: `change_count` → `total_commits`
- [ ] Handle new structure: array → object (keyed by path)

### For `file_type_stats.json` Consumers

- [ ] Update path: `file_type_stats.json` → `metadata/file_index.json`
- [ ] Implement aggregation logic (see Section II, Dataset 4)

### For `author_activity.json` Consumers

- [ ] Update path: `author_activity.json` → `networks/author_network.json`
- [ ] Change data access: `authors[]` → `nodes[]`
- [ ] Update field names: `unique_commits` → `commit_count`
- [ ] Decide: Use `file_lifecycle.json` if you need `files_touched` metric

---

## V. Schema Version Tracking

All v2 datasets include a `schema_version` field:

```javascript
// Defensive parsing
const data = require("./aggregations/temporal_monthly.json");

if (data.schema_version !== "1.0.0") {
  console.warn(`Unsupported schema: ${data.schema_version}`);
}

// Proceed with data access
```

**Current Schema Versions:**

- Core Lifecycle: `1.0.0` (locked, will never change)
- All other datasets: `1.0.0` (may evolve in future releases)

---

## VI. Support & Resources

**Complete Schema Documentation:**  
See `SCHEMA_REFERENCE.md` for detailed field definitions, validation rules, and examples.

**Dataset Previews:**  
Check `dataset_metadata.md` (auto-generated) for actual data shapes from your repository.

**Questions:**

- Field missing after migration? See Section II for field mapping tables
- Need data not in v2 datasets? Cross-reference with `file_lifecycle.json` (unchanged)
- Checksum validation failing? Ensure you're reading the correct file from `manifest.json`

---

**Last Updated:** January 14, 2026  
**Migration Guide Version:** 1.0.0
