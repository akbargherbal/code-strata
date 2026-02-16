# Repository Metrics Glossary → Dataset Mapping Guide

A practical guide showing which datasets illustrate each glossary concept, with Python/pandas code snippets for data extraction and analysis.

---

## Core Metrics

### Change Frequency

**Definition:** Number of times a file has been modified (committed).

**Best Dataset:** `file_lifecycle.json`, `file_index.json`

```python
import pandas as pd
import json

# From file_lifecycle.json
with open('output/file_lifecycle.json') as f:
    data = json.load(f)

# Calculate change frequency per file
change_freq = pd.DataFrame([
    {'file_path': path, 'change_frequency': len(events)}
    for path, events in data['files'].items()
])

# Top 10 most frequently changed files
hot_files = change_freq.nlargest(10, 'change_frequency')
```

**Alternative - file_index.json:**

```python
with open('output/metadata/file_index.json') as f:
    index = json.load(f)

df = pd.DataFrame.from_dict(index['files'], orient='index')
df = df.reset_index().rename(columns={'index': 'file_path'})
hot_files = df.nlargest(10, 'total_commits')
```

---

### Churn

**Definition:** Total volume of code change (lines added + lines deleted).

**Best Dataset:** `file_lifecycle.json`, `file_metrics_index.json`

```python
# From file_lifecycle.json
events_list = []
for path, events in data['files'].items():
    for event in events:
        events_list.append({
            'file_path': path,
            'lines_added': event.get('lines_added', 0),
            'lines_deleted': event.get('lines_deleted', 0)
        })

df = pd.DataFrame(events_list)

# Calculate churn per file
churn = df.groupby('file_path').agg({
    'lines_added': 'sum',
    'lines_deleted': 'sum'
}).reset_index()
churn['churn'] = churn['lines_added'] + churn['lines_deleted']

# Top churners
top_churn = churn.nlargest(10, 'churn')
```

**Alternative - file_metrics_index.json:**

```python
with open('output/frontend/file_metrics_index.json') as f:
    metrics = json.load(f)

churn_data = pd.DataFrame([
    {
        'file_path': path,
        'lines_added': m['volume']['lines_added'],
        'lines_deleted': m['volume']['lines_deleted'],
        'churn': m['volume']['lines_added'] + m['volume']['lines_deleted']
    }
    for path, m in metrics.items()
])
```

---

### Net Change

**Definition:** Difference between lines added and removed (growth indicator).

**Best Dataset:** `file_lifecycle.json`, `file_metrics_index.json`

```python
# Using churn dataframe from above
churn['net_change'] = churn['lines_added'] - churn['lines_deleted']

# Files with highest growth
growing = churn.nlargest(10, 'net_change')

# Files with highest reduction
shrinking = churn.nsmallest(10, 'net_change')

# Alternative from file_metrics_index.json
net_changes = pd.DataFrame([
    {
        'file_path': path,
        'net_change': m['volume']['net_change']
    }
    for path, m in metrics.items()
])
```

---

### Bus Factor

**Definition:** Minimum number of people who need to be unavailable before project is in trouble.

**Best Dataset:** `file_index.json`, `file_metrics_index.json`

```python
# From file_index.json
with open('output/metadata/file_index.json') as f:
    index = json.load(f)

files_df = pd.DataFrame.from_dict(index['files'], orient='index')
files_df = files_df.reset_index().rename(columns={'index': 'file_path'})

# Single-owner files (highest bus factor risk)
single_owner = files_df[files_df['unique_authors'] == 1]

# Calculate knowledge distribution
author_distribution = files_df['unique_authors'].describe()

# Files by author count
pd.cut(files_df['unique_authors'], bins=[0, 1, 2, 4, 100],
       labels=['High Risk (1)', 'Medium (2)', 'Medium (3-4)', 'Low (5+)']).value_counts()
```

---

## Activity Patterns

### Hot Files

**Definition:** Files with exceptionally high change frequency (>50 commits).

**Best Dataset:** `file_lifecycle.json`, `temporal_activity_map.json`

```python
# From file_lifecycle.json
with open('output/file_lifecycle.json') as f:
    data = json.load(f)

hot_threshold = 50
hot_files = pd.DataFrame([
    {
        'file_path': path,
        'commits': len(events),
        'unique_authors': len(set(e['author_email'] for e in events))
    }
    for path, events in data['files'].items()
    if len(events) > hot_threshold
])

# Sort by activity
hot_files_sorted = hot_files.sort_values('commits', ascending=False)
```

---

### Dormant Files

**Definition:** Files with zero or very low change frequency over extended period.

**Best Dataset:** `file_index.json`, `file_metrics_index.json`

```python
from datetime import datetime, timedelta

# From file_metrics_index.json
with open('output/frontend/file_metrics_index.json') as f:
    metrics = json.load(f)

dormant_files = pd.DataFrame([
    {
        'file_path': path,
        'last_modified': m['lifecycle']['last_modified_iso'],
        'is_dormant': m['lifecycle']['is_dormant'],
        'total_commits': m['volume']['total_commits']
    }
    for path, m in metrics.items()
    if m['lifecycle']['is_dormant']
])
```

---

### Cold Zones

**Definition:** Entire directories/modules with minimal activity.

**Best Dataset:** `directory_stats.json`, `temporal_activity_map.json`

```python
# From directory_stats.json
with open('output/aggregations/directory_stats.json') as f:
    dir_stats = json.load(f)

# Flatten directory hierarchy
def flatten_dirs(node, parent_path=''):
    path = f"{parent_path}/{node['name']}" if parent_path else node['name']
    result = [{
        'path': path,
        'commits': node['stats']['total_commits'],
        'unique_authors': node['stats']['unique_authors'],
        'files': node['stats']['total_files']
    }]
    for child in node.get('children', []):
        result.extend(flatten_dirs(child, path))
    return result

dirs_df = pd.DataFrame(flatten_dirs(dir_stats['directories']))

# Identify cold zones (bottom 10% by commits)
cold_zones = dirs_df[dirs_df['commits'] < dirs_df['commits'].quantile(0.1)]
```

---

### Activity Heatmap

**Definition:** Visual grid showing file activity intensity.

**Best Dataset:** `temporal_activity_map.json`

```python
# From temporal_activity_map.json
with open('output/frontend/temporal_activity_map.json') as f:
    activity_map = json.load(f)

# Convert to DataFrame for heatmap
rows = []
for dir_path, weeks in activity_map['data'].items():
    for week, values in weeks.items():
        commits, lines_changed, authors = values
        rows.append({
            'directory': dir_path,
            'week': week,
            'commits': commits,
            'lines_changed': lines_changed,
            'authors': authors
        })

heatmap_df = pd.DataFrame(rows)

# Pivot for visualization
heatmap_pivot = heatmap_df.pivot(index='directory',
                                  columns='week',
                                  values='commits').fillna(0)
```

---

## Temporal Analysis

### Story Arc

**Definition:** Narrative of repository evolution over time.

**Best Dataset:** `temporal_monthly.json`, `temporal_daily.json`

```python
# From temporal_monthly.json
with open('output/aggregations/temporal_monthly.json') as f:
    temporal = json.load(f)

months_df = pd.DataFrame.from_dict(temporal['months'], orient='index')
months_df = months_df.sort_values('month')

# Identify phases
months_df['phase'] = 'Unknown'
months_df.loc[months_df['files_changed'] > months_df['files_changed'].quantile(0.75), 'phase'] = 'High Activity'
months_df.loc[months_df['files_changed'] < months_df['files_changed'].quantile(0.25), 'phase'] = 'Stabilization'

# Calculate growth rate
months_df['growth_rate'] = months_df['files_changed'].pct_change()
```

---

### Creation Burst

**Definition:** Period where many new files are added rapidly.

**Best Dataset:** `file_lifecycle.json`, `temporal_daily.json`

```python
# From file_lifecycle.json
creation_events = []
for path, events in data['files'].items():
    first_event = events[0]  # First commit = creation
    if first_event['operation'] in ['A', 'M']:  # Add or first Modify
        creation_events.append({
            'date': first_event['datetime'][:10],
            'file_path': path
        })

creations = pd.DataFrame(creation_events)
daily_creations = creations.groupby('date').size().reset_index(name='files_created')

# Identify creation bursts (days with >X new files)
burst_threshold = daily_creations['files_created'].quantile(0.9)
creation_bursts = daily_creations[daily_creations['files_created'] > burst_threshold]
```

---

### Stabilization Period

**Definition:** Phase where change frequency decreases but size remains constant.

**Best Dataset:** `temporal_monthly.json`

```python
with open('output/aggregations/temporal_monthly.json') as f:
    temporal = json.load(f)

months_df = pd.DataFrame.from_dict(temporal['months'], orient='index')
months_df = months_df.sort_values('month')

# Calculate rolling statistics
months_df['commits_ma'] = months_df['commits'].rolling(window=3).mean()
months_df['files_changed_ma'] = months_df['files_changed'].rolling(window=3).mean()

# Stabilization: low change rate with consistent activity
months_df['is_stabilization'] = (
    (months_df['commits_ma'] < months_df['commits'].median()) &
    (months_df['commits_ma'].diff().abs() < months_df['commits'].std() * 0.5)
)

stabilization_periods = months_df[months_df['is_stabilization']]
```

---

### Refactoring Event

**Definition:** High churn with low net change (code restructuring).

**Best Dataset:** `file_lifecycle.json`

```python
# Analyze events by time period
events_list = []
for path, events in data['files'].items():
    for event in events:
        events_list.append({
            'file_path': path,
            'date': event['datetime'][:10],
            'week': event['temporal']['year'] * 100 + event['temporal']['week_no'],
            'lines_added': event.get('lines_added', 0),
            'lines_deleted': event.get('lines_deleted', 0)
        })

events_df = pd.DataFrame(events_list)

# Aggregate by week
weekly = events_df.groupby('week').agg({
    'lines_added': 'sum',
    'lines_deleted': 'sum'
}).reset_index()

weekly['churn'] = weekly['lines_added'] + weekly['lines_deleted']
weekly['net_change'] = weekly['lines_added'] - weekly['lines_deleted']
weekly['churn_to_net_ratio'] = weekly['churn'] / (weekly['net_change'].abs() + 1)

# Refactoring events: high churn, low net change
refactoring_weeks = weekly[
    (weekly['churn'] > weekly['churn'].quantile(0.75)) &
    (weekly['net_change'].abs() < weekly['churn'] * 0.1)
]
```

---

### Migration Pattern

**Definition:** Shift in development focus from one area to another over time.

**Best Dataset:** `temporal_activity_map.json`, `directory_stats.json`

```python
with open('output/frontend/temporal_activity_map.json') as f:
    activity_map = json.load(f)

# Track directory activity over time
timeline = []
for dir_path, weeks in activity_map['data'].items():
    for week, values in weeks.items():
        commits, lines_changed, authors = values
        timeline.append({
            'directory': dir_path,
            'week': week,
            'commits': commits
        })

timeline_df = pd.DataFrame(timeline)

# Calculate each directory's share of total activity per week
total_by_week = timeline_df.groupby('week')['commits'].sum()
timeline_df = timeline_df.merge(
    total_by_week.rename('total'),
    left_on='week',
    right_index=True
)
timeline_df['share'] = timeline_df['commits'] / timeline_df['total']

# Identify migrations (directories gaining/losing share)
migration = timeline_df.pivot(index='week', columns='directory', values='share').fillna(0)
migration_trends = migration.diff().mean().sort_values(ascending=False)
```

---

### Release Cycle

**Definition:** Recurring activity patterns aligned with releases.

**Best Dataset:** `release_snapshots.json`, `temporal_daily.json`

```python
# From release_snapshots.json
with open('output/milestones/release_snapshots.json') as f:
    releases = json.load(f)

releases_df = pd.DataFrame(releases['releases'])

# Extract release dates
release_dates = pd.to_datetime(releases_df['timestamp'].apply(
    lambda x: datetime.fromtimestamp(x, tz=timezone.utc)
))

# From temporal_daily.json for activity
with open('output/aggregations/temporal_daily.json') as f:
    daily = json.load(f)

daily_df = pd.DataFrame.from_dict(daily['days'], orient='index')
daily_df['date'] = pd.to_datetime(daily_df['date'])

# Annotate pre-release activity
for release_date in release_dates:
    window_start = release_date - timedelta(days=14)
    daily_df.loc[
        (daily_df['date'] >= window_start) & (daily_df['date'] < release_date),
        'pre_release'
    ] = True

pre_release_activity = daily_df[daily_df.get('pre_release', False)]
```

---

## Collaboration Metrics

### Collaboration Topology

**Definition:** Network of relationships between contributors.

**Best Dataset:** `author_network.json`

```python
with open('output/networks/author_network.json') as f:
    network = json.load(f)

# Nodes (authors)
authors_df = pd.DataFrame(network['nodes'])

# Edges (collaborations)
edges_df = pd.DataFrame(network['edges'])

# Calculate author centrality (degree)
author_connections = edges_df.groupby('source').size().reset_index(name='connections')
authors_with_centrality = authors_df.merge(
    author_connections,
    left_on='id',
    right_on='source',
    how='left'
).fillna(0)

# Most connected authors
central_authors = authors_with_centrality.nlargest(10, 'connections')
```

---

### Collaboration Intensity

**Definition:** Degree to which multiple contributors work on same files.

**Best Dataset:** `file_metrics_index.json`, `file_index.json`

```python
with open('output/frontend/file_metrics_index.json') as f:
    metrics = json.load(f)

collab_intensity = pd.DataFrame([
    {
        'file_path': path,
        'author_count': len(m['identifiers']['author_ids']),
        'total_commits': m['volume']['total_commits'],
        'primary_author_percentage': m['identifiers']['primary_author_percentage']
    }
    for path, m in metrics.items()
])

# High collaboration files (many authors)
high_collab = collab_intensity[collab_intensity['author_count'] > 5]

# Low primary author dominance = high collaboration
distributed_ownership = collab_intensity[
    collab_intensity['primary_author_percentage'] < 0.5
]
```

---

### Knowledge Silo

**Definition:** Files maintained by single contributor or small group.

**Best Dataset:** `file_metrics_index.json`, `file_index.json`

```python
# From file_metrics_index.json
knowledge_silos = pd.DataFrame([
    {
        'file_path': path,
        'primary_author': m['identifiers']['primary_author_id'],
        'primary_percentage': m['identifiers']['primary_author_percentage'],
        'author_count': len(m['identifiers']['author_ids'])
    }
    for path, m in metrics.items()
    if (len(m['identifiers']['author_ids']) <= 2 or
        m['identifiers']['primary_author_percentage'] > 0.8)
])

# Worst silos (single author with 100% ownership)
worst_silos = knowledge_silos[knowledge_silos['primary_percentage'] == 1.0]
```

---

### Single-Owner Files

**Definition:** Files modified by only one contributor (maximum risk).

**Best Dataset:** `file_index.json`

```python
with open('output/metadata/file_index.json') as f:
    index = json.load(f)

files_df = pd.DataFrame.from_dict(index['files'], orient='index')
files_df = files_df.reset_index().rename(columns={'index': 'file_path'})

# Single-owner files
single_owner = files_df[files_df['unique_authors'] == 1]

# Calculate risk: single-owner + high activity = high risk
single_owner['risk_score'] = single_owner['total_commits']
high_risk_silos = single_owner.nlargest(20, 'risk_score')
```

---

## Code Health Indicators

### Velocity

**Definition:** Rate of development activity (commits per time period).

**Best Dataset:** `temporal_daily.json`, `temporal_monthly.json`

```python
with open('output/aggregations/temporal_daily.json') as f:
    daily = json.load(f)

daily_df = pd.DataFrame.from_dict(daily['days'], orient='index')
daily_df['date'] = pd.to_datetime(daily_df['date'])
daily_df = daily_df.sort_values('date')

# Calculate rolling velocity (7-day average)
daily_df['velocity_commits'] = daily_df['commits'].rolling(window=7).mean()
daily_df['velocity_files'] = daily_df['files_changed'].rolling(window=7).mean()

# Monthly velocity trend
monthly_velocity = daily_df.resample('M', on='date').agg({
    'commits': 'sum',
    'files_changed': 'sum',
    'unique_authors': 'mean'
})
```

---

### Dependency Churn

**Definition:** Frequency of changes to dependency files.

**Best Dataset:** `file_lifecycle.json`

```python
# Identify dependency files
dependency_patterns = [
    'package.json', 'requirements.txt', 'Gemfile', 'pom.xml',
    'build.gradle', 'Cargo.toml', 'go.mod', 'composer.json'
]

dependency_files = {}
for path, events in data['files'].items():
    if any(pattern in path for pattern in dependency_patterns):
        dependency_files[path] = events

dep_churn = pd.DataFrame([
    {
        'file_path': path,
        'commits': len(events),
        'dates': [e['datetime'][:10] for e in events]
    }
    for path, events in dependency_files.items()
])

# Calculate churn rate
dep_churn['days_active'] = dep_churn['dates'].apply(
    lambda x: (pd.to_datetime(x[-1]) - pd.to_datetime(x[0])).days
)
dep_churn['churn_rate'] = dep_churn['commits'] / (dep_churn['days_active'] + 1)
```

---

## Advanced Concepts

### Temporal Coupling

**Definition:** Files frequently modified together in same commits.

**Best Dataset:** `cochange_network.json`, `file_metrics_index.json`

```python
# From cochange_network.json
with open('output/networks/cochange_network.json') as f:
    cochange = json.load(f)

# Edges represent temporal coupling
couplings = pd.DataFrame(cochange['edges'])
couplings = couplings.sort_values('weight', ascending=False)

# Top coupled pairs
top_couplings = couplings.head(20)

# Alternative from file_metrics_index.json
coupling_map = []
for path, metrics in metrics.items():
    for partner in metrics['coupling']['top_partners']:
        coupling_map.append({
            'file1': path,
            'file2': partner['path'],
            'strength': partner['strength']
        })

coupling_df = pd.DataFrame(coupling_map)
strong_couplings = coupling_df[coupling_df['strength'] > 0.5]
```

---

### Abandonment/Resurrection

**Definition:** Files that go dormant then become active again.

**Best Dataset:** `file_lifecycle.json`

```python
# Analyze time gaps between commits
resurrection_cases = []

for path, events in data['files'].items():
    if len(events) < 2:
        continue

    # Calculate time gaps
    timestamps = [e['timestamp'] for e in events]
    timestamps.sort()

    gaps = []
    for i in range(1, len(timestamps)):
        gap_days = (timestamps[i] - timestamps[i-1]) / 86400
        gaps.append(gap_days)

    # Find long dormancies (>180 days)
    max_gap = max(gaps) if gaps else 0
    if max_gap > 180:
        resurrection_cases.append({
            'file_path': path,
            'max_dormancy_days': max_gap,
            'total_commits': len(events),
            'first_seen': datetime.fromtimestamp(timestamps[0], tz=timezone.utc),
            'last_seen': datetime.fromtimestamp(timestamps[-1], tz=timezone.utc)
        })

resurrections = pd.DataFrame(resurrection_cases)
resurrections = resurrections.sort_values('max_dormancy_days', ascending=False)
```

---

## Visualization Concepts

### Treemap

**Definition:** Hierarchical visualization where rectangles represent files/directories.

**Best Dataset:** `project_hierarchy.json`

```python
with open('output/frontend/project_hierarchy.json') as f:
    hierarchy = json.load(f)

# Flatten tree for treemap data
def flatten_tree(node, parent_path=''):
    path = f"{parent_path}/{node['name']}" if parent_path else node['name']

    result = [{
        'path': path,
        'name': node['name'],
        'type': node['type'],
        'commits': node['stats']['total_commits'],
        'health_score': node['stats'].get('health_score_avg', 0),
        'files_count': node['stats']['total_files']
    }]

    for child in node.get('children', []):
        result.extend(flatten_tree(child, path))

    return result

treemap_data = pd.DataFrame(flatten_tree(hierarchy['tree']))

# Prepare for visualization (squarify, plotly treemap, etc.)
# Size by commits, color by health score
```

---

## Statistical Analysis

### Percentile Analysis

**Definition:** Identify files in top/bottom percentages of activity.

**Best Dataset:** `file_index.json`

```python
with open('output/metadata/file_index.json') as f:
    index = json.load(f)

files_df = pd.DataFrame.from_dict(index['files'], orient='index')
files_df = files_df.reset_index().rename(columns={'index': 'file_path'})

# Calculate percentiles
files_df['commit_percentile'] = files_df['total_commits'].rank(pct=True)

# Top 5% most active files
top_5_pct = files_df[files_df['commit_percentile'] > 0.95]

# Bottom 25% (dormant)
bottom_25_pct = files_df[files_df['commit_percentile'] <= 0.25]

# Percentile distribution
percentiles = files_df['total_commits'].quantile([0.25, 0.5, 0.75, 0.90, 0.95, 0.99])
print(percentiles)
```

---

## Concepts Not Directly Supported by Current Datasets

The following glossary items cannot be fully illustrated with the available datasets:

### 1. **Test Coverage Trend**

- Requires: Test execution results, coverage reports over time
- Not available in current git history analysis

### 2. **Test-to-Source Ratio**

- Requires: Test file identification and correlation with source files
- Partial support: Can identify test files by pattern, but no explicit test metadata

### 3. **Documentation Ratio / Documentation Lag**

- Requires: Documentation file identification
- Partial support: Can filter for .md files, but no semantic doc-code linking

### 4. **Complexity Drift**

- Requires: Code complexity metrics (cyclomatic, nesting depth)
- Not available: Need static analysis tools integration

### 5. **Copy/Paste Detection**

- Requires: Code similarity analysis
- Not available: Need specialized duplication detection tools

### 6. **Commit Message Analysis**

- Available: Commit subjects are in `file_lifecycle.json`
- Limitation: No pre-computed topic clustering or sentiment analysis

**Workaround for partial support:**

```python
# Example: Identify test files
test_files = files_df[
    files_df['file_path'].str.contains('test_|_test\.py|\.test\.|spec\.|\.spec\.')
]

# Example: Identify documentation files
doc_files = files_df[
    files_df['file_path'].str.endswith(('.md', '.rst', '.txt'))
]

# Example: Basic commit message analysis
subjects = []
for path, events in data['files'].items():
    subjects.extend([e['commit_subject'] for e in events])

import re
bug_commits = sum(1 for s in subjects if re.search(r'\b(fix|bug|issue)\b', s, re.I))
feature_commits = sum(1 for s in subjects if re.search(r'\b(feat|feature|add)\b', s, re.I))
```

---

## Complete Working Example

Here's a complete script demonstrating multiple concepts:

```python
import pandas as pd
import json
from datetime import datetime, timezone, timedelta

def analyze_repository_metrics(output_dir='output'):
    """Comprehensive repository metrics analysis"""

    # Load datasets
    with open(f'{output_dir}/file_lifecycle.json') as f:
        lifecycle = json.load(f)

    with open(f'{output_dir}/metadata/file_index.json') as f:
        file_index = json.load(f)

    with open(f'{output_dir}/frontend/file_metrics_index.json') as f:
        metrics_index = json.load(f)

    results = {}

    # 1. Hot Files (Change Frequency)
    change_freq = pd.DataFrame([
        {'file': p, 'commits': len(e)}
        for p, e in lifecycle['files'].items()
    ])
    results['hot_files'] = change_freq.nlargest(10, 'commits')

    # 2. High Churn Files
    churn_data = pd.DataFrame([
        {
            'file': path,
            'churn': m['volume']['lines_added'] + m['volume']['lines_deleted'],
            'net_change': m['volume']['net_change']
        }
        for path, m in metrics_index.items()
    ])
    results['high_churn'] = churn_data.nlargest(10, 'churn')

    # 3. Knowledge Silos
    silos = pd.DataFrame([
        {
            'file': path,
            'authors': len(m['identifiers']['author_ids']),
            'primary_pct': m['identifiers']['primary_author_percentage']
        }
        for path, m in metrics_index.items()
    ])
    results['silos'] = silos[silos['authors'] == 1]

    # 4. Temporal Coupling
    coupling = pd.DataFrame([
        {
            'file': path,
            'coupled_with': p['path'],
            'strength': p['strength']
        }
        for path, m in metrics_index.items()
        for p in m['coupling']['top_partners']
    ])
    results['strong_coupling'] = coupling[coupling['strength'] > 0.5]

    return results

# Run analysis
metrics = analyze_repository_metrics()

print("Top 10 Hot Files:")
print(metrics['hot_files'])

print("\nTop 10 High Churn Files:")
print(metrics['high_churn'])

print(f"\nKnowledge Silos: {len(metrics['silos'])} files")

print(f"\nStrong Couplings: {len(metrics['strong_coupling'])} pairs")
```

---

_Generated: January 2026_
_Compatible with: strata.py v2.2.0 datasets_
