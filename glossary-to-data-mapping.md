# Glossary Concepts → Strata Data Mapping

This document maps each concept from the Repository Metrics Glossary to the data structures available in `strata.py`, showing how to calculate or derive each metric.

**Legend:**

- ✅ **Direct**: Data directly available in output
- 🔄 **Derivable**: Can be calculated from available data
- 🔧 **Requires Processing**: Needs additional computation or aggregation
- ❌ **Not Available**: Would require additional data collection

---

## Core Metrics

### Change Frequency ✅

**Data Source:** `FileMetadataAggregator` → `files[file_path]["total_commits"]`  
**Also Available In:** Core lifecycle → `len(files[file_path])`

**Calculation:**

```python
change_frequency = len(files[file_path])  # Number of commits touching this file
commits_per_day = file_metadata["commits_per_day"]  # Pre-calculated rate
```

**Related Fields:**

- `files[file_path]` - array of all commit events for the file
- `total_commits` - count per file in FileMetadataAggregator
- `age_days` - file lifespan for normalization

---

### Churn 🔄

**Data Source:** Core lifecycle → `files[file_path][*]["lines_added"]` + `files[file_path][*]["lines_deleted"]`

**Calculation:**

```python
churn = sum(
    event["lines_added"] + event["lines_deleted"]
    for event in files[file_path]
)
```

**Per-Commit Churn:**

```python
commit_churn = event["lines_added"] + event["lines_deleted"]
```

**Available Since:** Phase 5 (v2.1.0) with `--numstat`

---

### Net Change 🔄

**Data Source:** Core lifecycle → `files[file_path][*]["lines_added"]` - `files[file_path][*]["lines_deleted"]`

**Calculation:**

```python
net_change = sum(
    event["lines_added"] - event["lines_deleted"]
    for event in files[file_path]
)
```

**Interpretation:**

- Positive = file grew
- Negative = file shrank
- Near zero with high churn = refactoring activity

---

### Bus Factor 🔧

**Data Source:** `AuthorNetworkAggregator` + `FileMetadataAggregator`

**Requires:**

1. Count files per author: `file_authors[file_path]` (in AuthorNetworkAggregator)
2. Identify single-owner files: `len(file_authors[file_path]) == 1`
3. Calculate critical file concentration per author

**Calculation Approach:**

```python
# Files modified by each author
author_files = defaultdict(set)
for file_path, authors in file_authors.items():
    for author in authors:
        author_files[author].add(file_path)

# Bus factor = minimum authors needed to cover all files
# Use set cover algorithm or count exclusive file ownership
```

---

## Activity Patterns

### Hot Files 🔄

**Data Source:** `FileMetadataAggregator` → sorted by `total_commits`

**Identification:**

```python
# Sort files by commit count
hot_files = sorted(
    file_metadata["files"].items(),
    key=lambda x: x[1]["total_commits"],
    reverse=True
)

# Top 10% or threshold-based
threshold = 50  # commits
hot_files = [f for f, meta in files.items() if meta["total_commits"] > threshold]
```

**Supporting Data:**

- `unique_authors` - multiple contributors indicator
- `commits_per_day` - activity rate
- Temporal data to check consistency across time

---

### Dormant Files 🔄

**Data Source:** `FileMetadataAggregator` → `last_modified` vs current date

**Identification:**

```python
from datetime import datetime, timedelta

dormancy_threshold = 180  # days
now = datetime.now(timezone.utc)

dormant_files = {
    file_path: meta
    for file_path, meta in file_metadata["files"].items()
    if (now - datetime.fromisoformat(meta["last_modified"])).days > dormancy_threshold
}
```

**Consider:**

- `age_days` - total file lifespan
- `total_commits` - historical activity level
- Recent activity in temporal aggregations

---

### Cold Zones 🔄

**Data Source:** `FileHierarchyAggregator` → directory-level statistics

**Identification:**

```python
# Low activity directories
cold_directories = {
    dir_path: stats
    for dir_path, stats in hierarchy["directories"].items()
    if stats["commits"] < threshold and stats["age_days"] > min_age
}
```

**Available Fields:**

- `commits` - total changes in directory
- `unique_files` - number of files
- `unique_authors` - contributor count
- `commits_per_file` - activity density

---

### Activity Heatmap 🔧

**Data Source:** Combination of `FileMetadataAggregator` + `FileHierarchyAggregator`

**Data Structure:**

```python
heatmap_data = {
    file_path: {
        "commits": meta["total_commits"],
        "authors": meta["unique_authors"],
        "directory": Path(file_path).parent,
        "intensity": meta["commits_per_day"]
    }
    for file_path, meta in file_metadata["files"].items()
}
```

**Visualization:** Requires external processing (treemap, grid layout)

---

## Temporal Analysis

### Story Arc ✅

**Data Source:** `TemporalAggregator` → `daily` and `monthly` aggregations

**Direct Access:**

```python
# Timeline of repository evolution
timeline = temporal["daily"]  # or temporal["monthly"]

phases = []
for date, stats in timeline.items():
    phases.append({
        "date": date,
        "commits": stats["commits"],
        "files_changed": stats["files_changed"],
        "active_authors": stats["unique_authors"],
        "operations": stats["operations"]  # A/M/D distribution
    })
```

**Phase Detection:** Analyze trends in operations (A=creation, M=modification, D=deletion)

---

### Creation Burst 🔄

**Data Source:** `TemporalAggregator` + operation types

**Identification:**

```python
# Find periods with high "A" (Add) operations
creation_bursts = []
for date, stats in temporal["daily"].items():
    if stats["operations"].get("A", 0) > threshold:
        creation_bursts.append({
            "date": date,
            "files_added": stats["operations"]["A"],
            "total_commits": stats["commits"]
        })
```

**Alternative:** Analyze `files[*][0]["timestamp"]` to find file creation dates clustering

---

### Stabilization Period 🔄

**Data Source:** `TemporalAggregator` + churn analysis over time

**Characteristics to Detect:**

```python
# Low churn, consistent commit rate, high M/D vs A ratio
for date, stats in temporal["monthly"].items():
    modifications = stats["operations"].get("M", 0)
    additions = stats["operations"].get("A", 0)

    if modifications > additions * 5 and stats["commits"] < avg_commits:
        # Potential stabilization period
        pass
```

**Supporting Data:**

- Core lifecycle for per-file churn calculation
- Commit message patterns (if available)

---

### Refactoring Event 🔄

**Data Source:** Core lifecycle → high churn + low net change

**Detection:**

```python
# Analyze specific time windows
from datetime import datetime

def analyze_period(start_date, end_date):
    period_events = [
        event for file_events in files.values()
        for event in file_events
        if start_date <= event["timestamp"] <= end_date
    ]

    total_churn = sum(e["lines_added"] + e["lines_deleted"] for e in period_events)
    net_change = sum(e["lines_added"] - e["lines_deleted"] for e in period_events)

    if total_churn > 1000 and abs(net_change) < total_churn * 0.1:
        return "Refactoring period"
```

---

### Migration Pattern 🔄

**Data Source:** `FileHierarchyAggregator` → track directory activity over time

**Analysis:**

```python
# Compare directory commit distribution across time periods
early_period = filter_by_date(commits, start, mid)
late_period = filter_by_date(commits, mid, end)

directory_shift = {}
for dir_path in directories:
    early_activity = count_commits(early_period, dir_path)
    late_activity = count_commits(late_period, dir_path)
    directory_shift[dir_path] = late_activity - early_activity
```

**Requires:** Combining temporal data with hierarchy data

---

### Release Cycle 🔄

**Data Source:** `TemporalAggregator` → detect periodic patterns

**Detection:**

```python
# Look for recurring peaks in commit activity
from scipy import signal  # External library needed

commit_series = [stats["commits"] for stats in temporal["daily"].values()]
peaks, _ = signal.find_peaks(commit_series, height=threshold, distance=min_days_between_releases)
```

**Alternative:** Analyze commit messages for release-related keywords

---

## Collaboration Metrics

### Collaboration Topology ✅

**Data Source:** `AuthorNetworkAggregator`

**Direct Access:**

```python
network = author_network_aggregator.finalize()

# Nodes: authors with commit counts
authors = network["nodes"]  # List of {id, email, commit_count, collaboration_count}

# Edges: collaborations via shared files
collaborations = network["edges"]  # List of {source, target, weight, shared_files}

# Statistics
density = network["statistics"]["density"]
total_authors = network["statistics"]["total_authors"]
```

---

### Collaboration Intensity ✅

**Data Source:** `FileMetadataAggregator` → `unique_authors` per file

**Direct Access:**

```python
high_collaboration_files = {
    file_path: meta
    for file_path, meta in file_metadata["files"].items()
    if meta["unique_authors"] >= intensity_threshold
}

# Average collaboration intensity
avg_authors_per_file = sum(
    meta["unique_authors"] for meta in file_metadata["files"].values()
) / len(file_metadata["files"])
```

---

### Knowledge Silo 🔄

**Data Source:** `AuthorNetworkAggregator` + `FileMetadataAggregator`

**Identification:**

```python
# Single or very few authors per file
knowledge_silos = {
    file_path: meta
    for file_path, meta in file_metadata["files"].items()
    if meta["unique_authors"] <= 2
}

# Critical silos: high change frequency + low author count
critical_silos = {
    path: meta
    for path, meta in knowledge_silos.items()
    if meta["total_commits"] > 20  # Actively maintained
}
```

---

### Handoff Event 🔧

**Data Source:** Core lifecycle → analyze author sequence over time

**Detection:**

```python
def detect_handoffs(file_path):
    events = files[file_path]
    handoffs = []

    # Group by time windows
    current_author = None
    current_window_start = None

    for event in sorted(events, key=lambda x: x["timestamp"]):
        author = event["author_email"]

        if current_author and author != current_author:
            time_gap = event["timestamp"] - current_window_start
            if time_gap > 30 * 86400:  # 30 days gap
                handoffs.append({
                    "from": current_author,
                    "to": author,
                    "timestamp": event["timestamp"]
                })

        current_author = author
        current_window_start = event["timestamp"]

    return handoffs
```

---

### Single-Owner Files ✅

**Data Source:** `FileMetadataAggregator` → `unique_authors == 1`

**Direct Identification:**

```python
single_owner_files = {
    file_path: meta
    for file_path, meta in file_metadata["files"].items()
    if meta["unique_authors"] == 1
}

# With author identification
for file_path, meta in single_owner_files.items():
    owner = meta["primary_author"]["email"]
    risk_score = meta["total_commits"]  # High commits = high risk
```

---

## Code Health Indicators

### Test Coverage Trend ❌

**Status:** Not directly available (requires code analysis tools)

**Possible Proxy:**

```python
# Track changes to test files over time
test_files = [
    f for f in files.keys()
    if "test" in f.lower() or f.endswith("_test.py") or f.endswith("Test.java")
]

test_activity = sum(len(files[f]) for f in test_files)
source_activity = sum(len(events) for f, events in files.items() if f not in test_files)
```

---

### Test-to-Source Ratio 🔄

**Data Source:** Core lifecycle → filter by file type

**Calculation:**

```python
def categorize_files(files):
    test_files = {}
    source_files = {}

    for file_path, events in files.items():
        if is_test_file(file_path):
            test_files[file_path] = events
        else:
            source_files[file_path] = events

    test_churn = sum(
        e["lines_added"] + e["lines_deleted"]
        for events in test_files.values()
        for e in events
    )
    source_churn = sum(
        e["lines_added"] + e["lines_deleted"]
        for events in source_files.values()
        for e in events
    )

    return test_churn / source_churn if source_churn > 0 else 0
```

---

### Documentation Ratio 🔄

**Data Source:** Core lifecycle → filter documentation files

**Calculation:**

```python
doc_patterns = [".md", ".rst", ".txt", "README", "docs/"]

doc_files = {
    f: events for f, events in files.items()
    if any(pattern in f for pattern in doc_patterns)
}

doc_commits = sum(len(events) for events in doc_files.values())
total_commits = sum(len(events) for events in files.values())

doc_ratio = doc_commits / total_commits
```

---

### Documentation Lag 🔧

**Data Source:** Core lifecycle → temporal analysis of doc vs source changes

**Detection:**

```python
def calculate_doc_lag(file_path, doc_file_path):
    source_events = sorted(files[file_path], key=lambda x: x["timestamp"])
    doc_events = sorted(files[doc_file_path], key=lambda x: x["timestamp"])

    lags = []
    for source_event in source_events:
        # Find next doc update after source change
        next_doc = next(
            (d for d in doc_events if d["timestamp"] > source_event["timestamp"]),
            None
        )
        if next_doc:
            lag_days = (next_doc["timestamp"] - source_event["timestamp"]) / 86400
            lags.append(lag_days)

    return sum(lags) / len(lags) if lags else float("inf")
```

---

### Dependency Churn 🔄

**Data Source:** Core lifecycle → filter dependency files

**Identification:**

```python
dependency_files = [
    "package.json", "requirements.txt", "Gemfile", "pom.xml",
    "build.gradle", "Cargo.toml", "go.mod", "composer.json"
]

dep_churn = {}
for dep_file in dependency_files:
    if dep_file in files:
        dep_churn[dep_file] = {
            "total_changes": len(files[dep_file]),
            "total_churn": sum(
                e["lines_added"] + e["lines_deleted"]
                for e in files[dep_file]
            ),
            "last_modified": files[dep_file][-1]["datetime"]
        }
```

---

### Velocity ✅

**Data Source:** `TemporalAggregator` + performance metrics

**Direct Calculation:**

```python
# Commits per time period
from temporal aggregator:
    monthly_velocity = [stats["commits"] for stats in temporal["monthly"].values()]

# Lines changed per period
def calculate_line_velocity(period_start, period_end):
    events = [
        e for file_events in files.values()
        for e in file_events
        if period_start <= e["timestamp"] <= period_end
    ]
    return sum(e["lines_added"] + e["lines_deleted"] for e in events)

# From performance metrics
commits_per_second = metrics["commits_processed"] / metrics["total_time"]
```

---

## Advanced Concepts

### Copy/Paste Detection 🔧

**Data Source:** Core lifecycle → analyze similar changes across files

**Approach:**

```python
def detect_similar_changes(time_window=3600):  # 1 hour
    # Group changes by commit time
    time_buckets = defaultdict(list)

    for file_path, events in files.items():
        for event in events:
            bucket = event["timestamp"] // time_window
            time_buckets[bucket].append((file_path, event))

    # Look for similar line changes in same bucket
    for bucket, changes in time_buckets.items():
        # Check if multiple files have identical lines_added/lines_deleted
        similar = defaultdict(list)
        for file_path, event in changes:
            key = (event["lines_added"], event["lines_deleted"])
            similar[key].append(file_path)

        # Report potential copy/paste
        for key, file_list in similar.items():
            if len(file_list) > 1:
                print(f"Potential copy/paste: {file_list}")
```

---

### Complexity Drift ❌

**Status:** Requires static code analysis (not available)

**Proxy Indicator:**

```python
# File size growth without splitting
def track_file_growth(file_path):
    events = files[file_path]

    cumulative_size = 0
    growth_over_time = []

    for event in sorted(events, key=lambda x: x["timestamp"]):
        cumulative_size += (event["lines_added"] - event["lines_deleted"])
        growth_over_time.append({
            "timestamp": event["timestamp"],
            "size": cumulative_size,
            "churn": event["lines_added"] + event["lines_deleted"]
        })

    return growth_over_time
```

---

### Abandonment/Resurrection 🔄

**Data Source:** Core lifecycle → analyze activity gaps

**Detection:**

```python
def find_abandonment_resurrection(file_path, gap_threshold=180):
    events = sorted(files[file_path], key=lambda x: x["timestamp"])

    patterns = []
    for i in range(len(events) - 1):
        gap_days = (events[i+1]["timestamp"] - events[i]["timestamp"]) / 86400

        if gap_days > gap_threshold:
            patterns.append({
                "file": file_path,
                "abandoned": events[i]["datetime"],
                "resurrected": events[i+1]["datetime"],
                "gap_days": gap_days,
                "resurrector": events[i+1]["author_email"]
            })

    return patterns
```

---

### Temporal Coupling 🔄

**Data Source:** `CochangeNetworkAggregator`

**Direct Access:**

```python
cochange_network = cochange_aggregator.finalize()

# Files that change together
temporal_couplings = [
    edge for edge in cochange_network["edges"]
    if edge["coupling_strength"] > threshold
]

# Sort by strength
temporal_couplings.sort(key=lambda x: x["coupling_strength"], reverse=True)

# Top coupled pairs
for edge in temporal_couplings[:10]:
    print(f"{edge['source']} <-> {edge['target']}: {edge['coupling_strength']}")
```

**Fields Available:**

- `source`, `target` - coupled file paths
- `cochange_count` - times changed together
- `coupling_strength` - normalized frequency

---

### Commit Message Analysis ✅

**Data Source:** Core lifecycle → `commit_subject`

**Basic Analysis:**

```python
from collections import Counter

# Extract all commit messages
commit_messages = [
    event["commit_subject"]
    for file_events in files.values()
    for event in file_events
]

# Categorize by keywords
categories = {
    "fix": [],
    "feature": [],
    "refactor": [],
    "test": [],
    "docs": []
}

for msg in commit_messages:
    msg_lower = msg.lower()
    for category in categories:
        if category in msg_lower or f"{category}:" in msg_lower:
            categories[category].append(msg)

# Most common words
all_words = " ".join(commit_messages).lower().split()
common_words = Counter(all_words).most_common(20)
```

---

## Repository Types

### Monorepo Detection 🔄

**Data Source:** `FileHierarchyAggregator`

**Identification:**

```python
# Look for multiple independent project roots
top_level_dirs = [
    dir_path for dir_path in hierarchy["directories"]
    if "/" not in dir_path or dir_path.count("/") == 0
]

# Check for project indicators in each
project_indicators = ["package.json", "pom.xml", "setup.py", "Cargo.toml"]

potential_projects = []
for dir_path in top_level_dirs:
    dir_files = hierarchy["directories"][dir_path]["unique_files"]
    if any(indicator in str(dir_files) for indicator in project_indicators):
        potential_projects.append(dir_path)

is_monorepo = len(potential_projects) > 1
```

---

## Statistical Analysis

### Median vs Mean ✅

**Data Source:** Any aggregated metric

**Calculation:**

```python
import statistics

commits_per_file = [
    meta["total_commits"]
    for meta in file_metadata["files"].values()
]

mean_commits = statistics.mean(commits_per_file)
median_commits = statistics.median(commits_per_file)
std_dev = statistics.stdev(commits_per_file)

print(f"Mean: {mean_commits:.2f}, Median: {median_commits:.2f}")
print(f"Standard Deviation: {std_dev:.2f}")
```

---

### Percentile Analysis 🔄

**Data Source:** Any metric distribution

**Calculation:**

```python
import numpy as np

commits_per_file = [
    meta["total_commits"]
    for meta in file_metadata["files"].values()
]

percentiles = {
    "50th": np.percentile(commits_per_file, 50),
    "75th": np.percentile(commits_per_file, 75),
    "90th": np.percentile(commits_per_file, 90),
    "95th": np.percentile(commits_per_file, 95),
    "99th": np.percentile(commits_per_file, 99)
}

# Files in 95th percentile (most active 5%)
threshold_95 = percentiles["95th"]
hot_files_95 = [
    f for f, meta in file_metadata["files"].items()
    if meta["total_commits"] >= threshold_95
]
```

---

## Time Windows & Temporal Labels ✅

**Data Source:** Core lifecycle → `temporal` field (Phase 5)

**Available Temporal Labels per Event:**

```python
event = files["some_file.py"][0]

temporal_labels = event["temporal"]
# {
#     "year": 2024,
#     "quarter": 2,
#     "month": 4,
#     "week_no": 15,
#     "day_of_week": 5,  # 0=Monday, 6=Sunday
#     "day_of_year": 103
# }
```

**Filtering by Time Windows:**

```python
# Get all events in Q2 2024
q2_2024_events = [
    event for file_events in files.values()
    for event in file_events
    if event["temporal"]["year"] == 2024 and event["temporal"]["quarter"] == 2
]

# Get weekend commits
weekend_commits = [
    event for file_events in files.values()
    for event in file_events
    if event["temporal"]["day_of_week"] in [5, 6]  # Saturday, Sunday
]

# Rolling 90-day window
from datetime import datetime, timedelta

def rolling_window_metrics(window_days=90):
    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=window_days)

    window_events = [
        event for file_events in files.values()
        for event in file_events
        if start_date.timestamp() <= event["timestamp"] <= end_date.timestamp()
    ]

    return {
        "commits": len(window_events),
        "files_changed": len(set(e["file_path"] for e in window_events)),
        "total_churn": sum(
            e["lines_added"] + e["lines_deleted"] for e in window_events
        )
    }
```

---

## Author Normalization ✅

**Data Source:** Core lifecycle → `author_id`, `author_domain` (Phase 5)

**Available Fields:**

```python
event = files["some_file.py"][0]

# Original author info
author_name = event["author_name"]
author_email = event["author_email"]

# Normalized identifiers (Phase 5)
author_id = event["author_id"]        # Normalized, lowercase, trimmed
author_domain = event["author_domain"]  # Email domain or "unknown"
```

**Analysis by Domain:**

```python
from collections import defaultdict

# Group commits by organization (domain)
domain_activity = defaultdict(lambda: {"commits": 0, "authors": set()})

for file_events in files.values():
    for event in file_events:
        domain = event["author_domain"]
        domain_activity[domain]["commits"] += 1
        domain_activity[domain]["authors"].add(event["author_id"])

# Report organizational contributions
for domain, stats in sorted(
    domain_activity.items(),
    key=lambda x: x[1]["commits"],
    reverse=True
):
    print(f"{domain}: {stats['commits']} commits by {len(stats['authors'])} authors")
```

---

## Position Metrics ✅

**Data Source:** Core lifecycle → `sequence`, `is_first`, `is_last` (Phase 5)

**Available Fields:**

```python
for event in files["some_file.py"]:
    sequence = event["sequence"]      # 1-indexed position in file history
    is_first = event["is_first"]      # True for file creation
    is_last = event["is_last"]        # True for most recent change

# Find file creation events
creation_events = [
    (file_path, events[0])
    for file_path, events in files.items()
    if events and events[0]["is_first"]
]

# Analyze file lifecycles
for file_path, events in files.items():
    lifecycle_length = len(events)
    created_by = events[0]["author_email"]
    last_modified_by = events[-1]["author_email"]

    print(f"{file_path}:")
    print(f"  Created by: {created_by}")
    print(f"  Last modified by: {last_modified_by}")
    print(f"  Total modifications: {lifecycle_length}")
```

---

## Summary: Data Availability

### ✅ Directly Available (No Calculation)

- Change frequency per file
- Commit timestamps and temporal labels (year, quarter, month, week)
- Author information (original + normalized)
- Collaboration networks
- Co-change networks
- File hierarchy statistics
- Daily/monthly temporal aggregations
- Lines added/deleted per commit
- Position metrics (sequence, first/last)

### 🔄 Easily Derivable (Simple Calculation)

- Churn (sum of additions + deletions)
- Net change (additions - deletions)
- Hot/cold files (threshold-based filtering)
- Dormant files (time-based filtering)
- Story arc phases (temporal pattern analysis)
- Refactoring events (churn vs net change ratio)
- Test-to-source ratio (file pattern matching)
- Velocity trends (temporal aggregation)
- Statistical measures (median, percentiles)

### 🔧 Requires Processing (Complex Computation)

- Bus factor (set cover algorithm)
- Handoff events (author sequence analysis)
- Documentation lag (temporal correlation)
- Copy/paste detection (similarity analysis)
- Abandonment/resurrection (gap analysis)
- Migration patterns (directory temporal analysis)
- Release cycles (periodicity detection)

### ❌ Not Available (Requires Additional Tools)

- Test coverage percentage (needs code analysis)
- Cyclomatic complexity (needs static analysis)
- Actual code quality metrics (needs linting tools)
- Build/CI integration data
- Issue tracker correlation

---

## Usage Examples

### Example 1: Identify High-Risk Knowledge Silos

```python
# Combine multiple metrics for risk assessment
risk_files = []

for file_path, meta in file_metadata["files"].items():
    if meta["unique_authors"] <= 2 and meta["total_commits"] > 20:
        # Calculate additional risk factors
        events = files[file_path]
        recent_activity = sum(
            1 for e in events
            if e["timestamp"] > (time.time() - 90*86400)
        )

        risk_score = (
            meta["total_commits"] * 0.4 +
            (3 - meta["unique_authors"]) * 30 +
            recent_activity * 0.2
        )

        risk_files.append({
            "file": file_path,
            "authors": meta["unique_authors"],
            "commits": meta["total_commits"],
            "recent_activity": recent_activity,
            "risk_score": risk_score
        })

# Sort by risk
risk_files.sort(key=lambda x: x["risk_score"], reverse=True)
```

### Example 2: Track Repository Evolution Story

```python
# Combine temporal data with operation types
story = []

for month, stats in temporal["monthly"].items():
    adds = stats["operations"].get("A", 0)
    mods = stats["operations"].get("M", 0)
    dels = stats["operations"].get("D", 0)

    # Classify phase
    if adds > mods * 2:
        phase = "Rapid Growth"
    elif dels > adds:
        phase = "Cleanup/Refactoring"
    elif mods > adds * 3:
        phase = "Stabilization"
    else:
        phase = "Normal Development"

    story.append({
        "month": month,
        "phase": phase,
        "commits": stats["commits"],
        "active_authors": stats["unique_authors"],
        "operations": {"adds": adds, "mods": mods, "dels": dels}
    })
```

### Example 3: Collaboration Health Dashboard

```python
# Multi-metric collaboration analysis
def generate_collab_health():
    network = author_network_aggregator.finalize()

    # Single-owner files (high risk)
    single_owner = sum(
        1 for meta in file_metadata["files"].values()
        if meta["unique_authors"] == 1
    )

    # Network density (team connectivity)
    density = network["statistics"]["density"]

    # Average collaborations per author
    avg_collabs = (
        sum(n["collaboration_count"] for n in network["nodes"])
        / len(network["nodes"])
    )

    return {
        "network_density": density,
        "avg_collaborations_per_author": avg_collabs,
        "single_owner_files": single_owner,
        "total_files": len(file_metadata["files"]),
        "knowledge_silo_percentage": (single_owner / len(file_metadata["files"])) * 100
    }
```

---

**Document Version:** 1.0  
**Last Updated:** January 2026  
**Strata Version:** 2.1.0 (Phase 5)
