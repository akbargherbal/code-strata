# Strata.py Dataset Reference Guide - Part 3

## Advanced Datasets and Common Workflows (Sections 4.6-5)

---

### 4.6 File Hierarchy and Directory Statistics

**File:** `aggregations/directory_stats.json`

Hierarchical view of the codebase with aggregated metrics at each directory level.

#### 4.6.1 Hierarchical Structure

```json
{
  "metadata": {
    "generated_at": "2024-11-15T10:32:30+00:00",
    "total_directories": 45,
    "max_depth": 6
  },
  "hierarchy": {
    "src": {
      "path": "src",
      "depth": 1,
      "file_count": 156,
      "subdirectory_count": 5,
      "total_commits": 1234,
      "unique_authors": 12,
      "total_churn": 45678,
      "lines_added": 28901,
      "lines_deleted": 16777,
      "average_file_age_days": 245,
      "average_churn_per_file": 292,
      "subdirectories": {
        "core": {
          /* nested stats */
        },
        "api": {
          /* nested stats */
        },
        "utils": {
          /* nested stats */
        }
      }
    }
  }
}
```

#### 4.6.2 Directory-Level Metrics

**Structure Metrics:**

```json
{
  "path": "src/api",
  "depth": 2,
  "file_count": 23, // Files directly in this directory
  "subdirectory_count": 3, // Immediate subdirectories
  "total_files_recursive": 45 // All files in subtree
}
```

**Activity Metrics:**

```json
{
  "total_commits": 567, // All commits affecting this directory
  "unique_authors": 8, // Authors who modified files here
  "author_diversity": 0.75, // Shannon entropy of author distribution
  "create_events": 12, // Files created in this directory
  "delete_events": 3 // Files deleted from this directory
}
```

**Churn Aggregates:**

```json
{
  "total_churn": 12345, // Sum of all file churn
  "lines_added": 7890,
  "lines_deleted": 4455,
  "average_churn_per_file": 537, // total_churn / file_count
  "max_file_churn": 2341 // Highest churn file in directory
}
```

**Age Metrics:**

```json
{
  "oldest_file_age_days": 456, // Age of oldest file
  "newest_file_age_days": 12, // Age of newest file
  "average_file_age_days": 234 // Mean age across all files
}
```

**Health Indicators:**

```json
{
  "average_file_health": 76.5, // Mean health score of files
  "files_low_health": 3, // Files with health < 50
  "bus_factor": 2 // Directory-level bus factor
}
```

#### 4.6.3 Usage Patterns

**Compare Module Activity:**

```python
import json
import pandas as pd

# Load directory stats
with open('output/aggregations/directory_stats.json', 'r') as f:
    dir_data = json.load(f)

# Flatten hierarchy for analysis
def flatten_hierarchy(hierarchy, parent_path=''):
    """Recursively flatten nested directory structure"""
    records = []
    for dir_name, stats in hierarchy.items():
        current_path = f"{parent_path}/{dir_name}" if parent_path else dir_name
        record = {
            'path': current_path,
            'depth': stats['depth'],
            'file_count': stats['file_count'],
            'total_commits': stats['total_commits'],
            'unique_authors': stats['unique_authors'],
            'total_churn': stats['total_churn'],
            'average_churn_per_file': stats['average_churn_per_file']
        }
        records.append(record)

        # Recurse into subdirectories
        if 'subdirectories' in stats:
            records.extend(flatten_hierarchy(stats['subdirectories'], current_path))

    return records

dirs_df = pd.DataFrame(flatten_hierarchy(dir_data['hierarchy']))

# Top-level modules comparison
top_level = dirs_df[dirs_df['depth'] == 1].sort_values('total_commits', ascending=False)

print("Top-level module activity:")
print(top_level[['path', 'file_count', 'total_commits', 'unique_authors', 'total_churn']])
```

**Find Hotspot Directories:**

```python
# Define hotspot criteria
def calculate_hotspot_score(row):
    # Normalize metrics
    commit_score = row['total_commits'] / dirs_df['total_commits'].max()
    churn_score = row['total_churn'] / dirs_df['total_churn'].max()
    author_score = row['unique_authors'] / dirs_df['unique_authors'].max()

    # Weighted combination
    return (commit_score * 0.4) + (churn_score * 0.4) + (author_score * 0.2)

dirs_df['hotspot_score'] = dirs_df.apply(calculate_hotspot_score, axis=1)

print("Top 10 hotspot directories:")
print(dirs_df.nlargest(10, 'hotspot_score')[
    ['path', 'hotspot_score', 'total_commits', 'total_churn', 'unique_authors']
])
```

**Calculate Directory Complexity:**

```python
# Complexity indicators
dirs_df['files_per_author'] = dirs_df['file_count'] / dirs_df['unique_authors']
dirs_df['churn_density'] = dirs_df['total_churn'] / dirs_df['file_count']
dirs_df['commit_density'] = dirs_df['total_commits'] / dirs_df['file_count']

# High complexity: many files, high churn, few authors
complexity_df = dirs_df[dirs_df['file_count'] > 10].copy()
complexity_df['complexity_score'] = (
    complexity_df['files_per_author'] *
    complexity_df['churn_density'] / 1000
)

print("Most complex directories:")
print(complexity_df.nlargest(10, 'complexity_score')[
    ['path', 'file_count', 'unique_authors', 'files_per_author', 'churn_density']
])
```

**Visualize Tree Structure:**

```python
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Treemap visualization
def plot_treemap(dirs_df, metric='total_churn'):
    """Create treemap of directory structure colored by metric"""
    import squarify  # pip install squarify

    # Filter to top level for clarity
    top_dirs = dirs_df[dirs_df['depth'] <= 2]

    # Prepare data
    sizes = top_dirs[metric].tolist()
    labels = top_dirs['path'].tolist()

    # Create color map
    colors = plt.cm.RdYlGn_r(top_dirs[metric] / top_dirs[metric].max())

    # Plot
    fig, ax = plt.subplots(figsize=(14, 10))
    squarify.plot(sizes=sizes, label=labels, color=colors, alpha=0.7, ax=ax)
    ax.set_title(f'Directory Treemap (by {metric})')
    ax.axis('off')
    plt.tight_layout()
    plt.show()

plot_treemap(dirs_df, 'total_churn')
```

---

### 4.7 Milestone Snapshots

**File:** `milestones/release_snapshots.json`

Point-in-time snapshots of repository metrics at important milestones (tags, releases, branches).

#### 4.7.1 Milestone Types

**Detected Milestones:**

- **Git Tags**: Version releases (v1.0.0, v2.1.3, etc.)
- **Branch Points**: Major branch creations
- **Significant Commits**: Large merges, major refactors

**Schema:**

```json
{
  "metadata": {
    "generated_at": "2024-11-15T10:33:00+00:00",
    "total_milestones": 15,
    "milestone_types": ["tag", "branch", "commit"]
  },
  "milestones": [
    {
      "name": "v2.0.0",
      "type": "tag",
      "commit_hash": "abc123...",
      "timestamp": "2024-06-15T10:00:00+00:00",
      "tagger": "release_bot",
      "message": "Release version 2.0.0"
    }
  ],
  "snapshots": [
    {
      "milestone_name": "v2.0.0",
      "snapshot_date": "2024-06-15T10:00:00+00:00",
      "metrics": {
        "total_files": 287,
        "total_commits": 1123,
        "unique_authors": 11,
        "total_churn": 34567,
        "average_file_health": 78.2,
        "bus_factor": 3,
        "top_files_by_churn": [
          { "file": "src/core/engine.py", "churn": 4521 },
          { "file": "src/api/handlers.py", "churn": 3890 }
        ],
        "author_breakdown": {
          "jane_developer_jane": 345,
          "john_smith_john": 289
        }
      },
      "delta_from_previous": {
        "files_added": 23,
        "files_deleted": 5,
        "commits_added": 156,
        "churn_increase": 4521,
        "authors_joined": 2,
        "authors_left": 1
      }
    }
  ]
}
```

#### 4.7.2 Snapshot Contents

**Milestone Identification:**

```json
{
  "milestone_name": "v2.0.0",
  "snapshot_date": "ISO datetime",
  "commit_hash": "SHA",
  "type": "tag|branch|commit"
}
```

**Point-in-Time Metrics:**

```json
{
  "metrics": {
    "total_files": 287,
    "active_files": 265,
    "deleted_files": 22,
    "total_commits": 1123,
    "unique_authors": 11,
    "total_churn": 34567,
    "lines_added": 21234,
    "lines_deleted": 13333,
    "average_file_health": 78.2,
    "bus_factor": 3,
    "coupling_edges": 234
  }
}
```

**Top Contributors/Files:**

```json
{
  "top_files_by_churn": [{ "file": "path", "churn": 4521 }],
  "top_authors_by_commits": [{ "author": "id", "commits": 345 }]
}
```

#### 4.7.3 Comparative Analysis

**Track Metric Evolution Across Releases:**

```python
import json
import pandas as pd
import matplotlib.pyplot as plt

# Load milestones
with open('output/milestones/release_snapshots.json', 'r') as f:
    milestones = json.load(f)

# Extract metrics from each snapshot
records = []
for snapshot in milestones['snapshots']:
    record = {
        'milestone': snapshot['milestone_name'],
        'date': snapshot['snapshot_date'],
        **snapshot['metrics']
    }
    records.append(record)

df = pd.DataFrame(records)
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date')

# Plot metric evolution
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Total files
axes[0, 0].plot(df['date'], df['total_files'], marker='o')
axes[0, 0].set_title('Total Files Over Releases')
axes[0, 0].set_ylabel('File Count')
axes[0, 0].grid(True, alpha=0.3)

# Churn
axes[0, 1].plot(df['date'], df['total_churn'], marker='o', color='orange')
axes[0, 1].set_title('Total Churn Over Releases')
axes[0, 1].set_ylabel('Churn')
axes[0, 1].grid(True, alpha=0.3)

# Health
axes[1, 0].plot(df['date'], df['average_file_health'], marker='o', color='green')
axes[1, 0].set_title('Average File Health Over Releases')
axes[1, 0].set_ylabel('Health Score')
axes[1, 0].grid(True, alpha=0.3)

# Authors
axes[1, 1].plot(df['date'], df['unique_authors'], marker='o', color='purple')
axes[1, 1].set_title('Unique Authors Over Releases')
axes[1, 1].set_ylabel('Author Count')
axes[1, 1].grid(True, alpha=0.3)

for ax in axes.flat:
    ax.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()
```

**Identify Regression Points:**

```python
# Calculate release-over-release changes
df['health_change'] = df['average_file_health'].diff()
df['churn_increase'] = df['total_churn'].diff()
df['files_added'] = df['total_files'].diff()

# Find releases with health degradation
regressions = df[df['health_change'] < -5]  # Health dropped >5 points

print("Releases with health regressions:")
print(regressions[['milestone', 'average_file_health', 'health_change', 'churn_increase']])

# Releases with excessive churn increase
high_churn_releases = df[df['churn_increase'] > df['churn_increase'].quantile(0.75)]

print("\nReleases with high churn increases:")
print(high_churn_releases[['milestone', 'total_churn', 'churn_increase']])
```

**Calculate Release Velocity:**

```python
# Time between releases
df['days_since_previous'] = df['date'].diff().dt.days

# Commits per release cycle
df['commits_per_release'] = df['total_commits'].diff()
df['velocity'] = df['commits_per_release'] / df['days_since_previous']

print("Release velocity (commits per day):")
print(df[['milestone', 'days_since_previous', 'commits_per_release', 'velocity']])

# Plot velocity trend
fig, ax = plt.subplots(figsize=(12, 6))
ax.bar(df['milestone'], df['velocity'], alpha=0.7)
ax.set_ylabel('Velocity (commits/day)')
ax.set_title('Development Velocity Between Releases')
ax.tick_params(axis='x', rotation=45)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

**Compare Release Quality:**

```python
# Quality score: health - (churn_rate * 10)
df['quality_score'] = df['average_file_health'] - (df['churn_increase'] / df['files_added']).fillna(0) * 10

print("Release quality ranking:")
print(df.nlargest(10, 'quality_score')[['milestone', 'quality_score', 'average_file_health', 'total_churn']])
```

---

### 4.8 Project Hierarchy (Frontend)

**File:** `frontend/project_hierarchy.json`

Nested tree structure optimized for treemap visualizations with health-based coloring.

#### 4.8.1 Treemap-Ready Format

```json
{
  "metadata": {
    "generated_at": "2024-11-15T10:33:30+00:00",
    "max_depth": 6,
    "total_nodes": 387,
    "health_scoring_enabled": true
  },
  "tree": {
    "name": "root",
    "type": "directory",
    "path": "",
    "size": 156234,
    "health": 75.3,
    "color": "#90EE90",
    "children": [
      {
        "name": "src",
        "type": "directory",
        "path": "src",
        "size": 98765,
        "health": 78.5,
        "color": "#90EE90",
        "metrics": {
          "file_count": 156,
          "total_commits": 1234,
          "total_churn": 45678,
          "unique_authors": 12,
          "average_file_health": 78.5
        },
        "children": [
          {
            "name": "engine.py",
            "type": "file",
            "path": "src/core/engine.py",
            "size": 5310,
            "health": 82.0,
            "color": "#90EE90",
            "metrics": {
              "total_commits": 87,
              "unique_authors": 5,
              "total_churn": 5310,
              "bus_factor": 2,
              "coupling_degree": 12
            }
          }
        ]
      }
    ]
  }
}
```

#### 4.8.2 Health Scoring System

**Components of Health Score:**

1. **Stability (40%)**: Inverse of churn rate
2. **Bus Factor (30%)**: Author diversity
3. **Coupling (20%)**: Inverse of coupling degree
4. **Age (10%)**: Maturity bonus

**For Files:**
$$\text{Health}_{\text{file}} = 0.4S + 0.3B + 0.2C + 0.1A$$

**For Directories:**
$$\text{Health}_{\text{dir}} = \frac{\sum \text{Health}_{\text{child}} \times \text{Size}_{\text{child}}}{\sum \text{Size}_{\text{child}}}$$

Weighted average of children, where size = churn.

**Weight Factors:**

| Component  | Weight | Rationale                 |
| ---------- | ------ | ------------------------- |
| Stability  | 40%    | Most predictive of issues |
| Bus Factor | 30%    | Knowledge risk            |
| Coupling   | 20%    | Maintenance impact        |
| Age        | 10%    | Maturity indicator        |

**Color Coding Thresholds:**

| Health Range | Color      | Hex     | Status    |
| ------------ | ---------- | ------- | --------- |
| 90-100       | Dark Green | #228B22 | Excellent |
| 75-89        | Green      | #90EE90 | Good      |
| 60-74        | Yellow     | #FFFF00 | Fair      |
| 40-59        | Orange     | #FFA500 | Poor      |
| 0-39         | Red        | #FF0000 | Critical  |

#### 4.8.3 Hierarchical Aggregation

**Parent-Child Relationships:**

```json
{
  "name": "parent_directory",
  "children": [
    { "name": "child1", "size": 100 },
    { "name": "child2", "size": 200 }
  ],
  "size": 300 // Sum of children
}
```

**Roll-up Calculations:**

**Size (Churn) Roll-up:**

```python
def calculate_size(node):
    if node['type'] == 'file':
        return node['metrics']['total_churn']
    else:  # directory
        return sum(calculate_size(child) for child in node['children'])
```

**Health Roll-up:**

```python
def calculate_health(node):
    if node['type'] == 'file':
        return calculate_file_health(node['metrics'])
    else:  # directory
        total_weight = sum(child['size'] for child in node['children'])
        weighted_health = sum(
            child['health'] * child['size']
            for child in node['children']
        )
        return weighted_health / total_weight if total_weight > 0 else 0
```

**Leaf vs. Branch Nodes:**

| Node Type          | Characteristics | Metrics                                          |
| ------------------ | --------------- | ------------------------------------------------ |
| Leaf (File)        | No children     | File-specific: commits, authors, churn, coupling |
| Branch (Directory) | Has children    | Aggregated: sum, average, or weighted metrics    |

#### 4.8.4 Visualization Examples

**Generate Treemap with D3.js:**

```html
<!DOCTYPE html>
<html>
  <head>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
      .node {
        stroke: #fff;
        stroke-width: 1px;
      }
      .node:hover {
        stroke: #000;
        stroke-width: 2px;
      }
    </style>
  </head>
  <body>
    <div id="treemap"></div>
    <script>
      // Load data
      d3.json("frontend/project_hierarchy.json").then((data) => {
        const width = 1200;
        const height = 800;

        const svg = d3
          .select("#treemap")
          .append("svg")
          .attr("width", width)
          .attr("height", height);

        const root = d3
          .hierarchy(data.tree)
          .sum((d) => (d.type === "file" ? d.size : 0))
          .sort((a, b) => b.value - a.value);

        d3.treemap().size([width, height]).padding(1)(root);

        const nodes = svg
          .selectAll("g")
          .data(root.leaves())
          .join("g")
          .attr("transform", (d) => `translate(${d.x0},${d.y0})`);

        nodes
          .append("rect")
          .attr("class", "node")
          .attr("width", (d) => d.x1 - d.x0)
          .attr("height", (d) => d.y1 - d.y0)
          .attr("fill", (d) => d.data.color);

        nodes
          .append("text")
          .attr("x", 3)
          .attr("y", 12)
          .text((d) => d.data.name)
          .attr("font-size", "10px")
          .attr("fill", "#000");

        nodes
          .append("title")
          .text(
            (d) =>
              `${d.data.path}\nHealth: ${d.data.health}\nSize: ${d.data.size}`,
          );
      });
    </script>
  </body>
</html>
```

**Python Treemap with Plotly:**

```python
import json
import plotly.express as px
import pandas as pd

# Load hierarchy
with open('output/frontend/project_hierarchy.json', 'r') as f:
    hierarchy = json.load(f)

# Flatten to DataFrame
def flatten_tree(node, records, parent=''):
    record = {
        'name': node['name'],
        'path': node['path'],
        'parent': parent,
        'size': node['size'],
        'health': node['health'],
        'type': node['type']
    }
    records.append(record)

    if 'children' in node:
        for child in node['children']:
            flatten_tree(child, records, node['path'] or 'root')

records = []
flatten_tree(hierarchy['tree'], records)
df = pd.DataFrame(records)

# Create treemap
fig = px.treemap(
    df,
    names='name',
    parents='parent',
    values='size',
    color='health',
    color_continuous_scale=['red', 'yellow', 'green'],
    title='Project Health Treemap',
    hover_data=['path', 'health', 'size']
)

fig.update_layout(width=1200, height=800)
fig.show()
```

---

### 4.9 File Metrics Index (Frontend)

**File:** `frontend/file_metrics_index.json`

Detailed per-file metrics with coupling analysis for drill-down panels.

#### 4.9.1 Per-File Detailed Metrics

```json
{
  "metadata": {
    "generated_at": "2024-11-15T10:34:00+00:00",
    "total_files": 342,
    "metrics_included": [
      "churn",
      "commits",
      "authors",
      "coupling",
      "health",
      "temporal"
    ]
  },
  "files": {
    "src/core/engine.py": {
      "basic": {
        "file_path": "src/core/engine.py",
        "extension": "py",
        "status": "active",
        "created_at": "2023-01-15T09:00:00+00:00",
        "last_modified": "2024-11-10T16:45:00+00:00",
        "age_days": 665
      },
      "activity": {
        "total_commits": 87,
        "modify_events": 85,
        "rename_events": 1,
        "touch_count": 87,
        "lines_added": 3420,
        "lines_deleted": 1890,
        "total_churn": 5310,
        "churn_per_commit": 61.0,
        "active_lifespan_days": 665,
        "change_frequency": 0.13
      },
      "authors": {
        "unique_authors": 5,
        "primary_author": "jane_developer_jane",
        "primary_author_percentage": 45.0,
        "author_diversity": 0.82,
        "bus_factor": 2,
        "contributors": {
          "jane_developer_jane": 39,
          "john_smith_john": 23,
          "bob_johnson_bob": 15,
          "sarah_lee_sarah": 8,
          "mike_wilson_mike": 2
        }
      },
      "coupling": {
        "coupling_degree": 12,
        "avg_coupling_strength": 0.45,
        "max_coupling_strength": 0.82,
        "strongly_coupled_files": [
          {
            "file": "src/api/handlers.py",
            "strength": 0.82,
            "commits_together": 45
          },
          {
            "file": "tests/test_engine.py",
            "strength": 0.75,
            "commits_together": 65
          }
        ]
      },
      "health": {
        "overall_score": 82.0,
        "stability_score": 85.0,
        "bus_factor_score": 75.0,
        "coupling_score": 88.0,
        "age_score": 100.0,
        "risk_level": "low"
      },
      "temporal": {
        "first_commit": "2023-01-15T09:00:00+00:00",
        "last_commit": "2024-11-10T16:45:00+00:00",
        "most_active_month": "2024-03",
        "commits_last_30_days": 5,
        "commits_last_90_days": 18,
        "dormant_periods": 1,
        "longest_gap_days": 45
      }
    }
  }
}
```

#### 4.9.2 Coupling Analysis

**Coupling Metrics:**

```json
{
  "coupling": {
    "coupling_degree": 12, // Number of files coupled
    "avg_coupling_strength": 0.45, // Mean coupling score
    "max_coupling_strength": 0.82, // Strongest coupling
    "inbound_coupling": 8, // Files that depend on this
    "outbound_coupling": 12, // Files this depends on
    "bidirectional_coupling": 4 // Mutual dependencies
  }
}
```

**Top Coupled Files:**

```json
{
  "strongly_coupled_files": [
    {
      "file": "src/api/handlers.py",
      "strength": 0.82,
      "commits_together": 45,
      "direction": "bidirectional",
      "last_cochange": "2024-11-10T16:45:00+00:00"
    }
  ]
}
```

**Coupling Direction:**

- **Bidirectional**: Both files frequently change together
- **Inbound**: This file causes changes in others
- **Outbound**: This file changes due to others

#### 4.9.3 Risk Indicators

**Composite Risk Score:**

```python
def calculate_risk_score(metrics):
    risk = 0

    # High churn
    if metrics['activity']['total_churn'] > 3000:
        risk += 25

    # Low bus factor
    if metrics['authors']['bus_factor'] == 1:
        risk += 30
    elif metrics['authors']['bus_factor'] == 2:
        risk += 15

    # High coupling
    if metrics['coupling']['coupling_degree'] > 15:
        risk += 20

    # Recent activity
    if metrics['temporal']['commits_last_30_days'] > 10:
        risk += 10

    # Age factor (very new or very old)
    age_days = metrics['basic']['age_days']
    if age_days < 30 or age_days > 730:
        risk += 15

    return min(risk, 100)
```

**Risk Categories:**

| Risk Score | Level    | Characteristics                      |
| ---------- | -------- | ------------------------------------ |
| 80-100     | Critical | Multiple red flags, urgent attention |
| 60-79      | High     | Significant issues, should refactor  |
| 40-59      | Medium   | Monitor closely, plan improvements   |
| 20-39      | Low      | Minor concerns, normal monitoring    |
| 0-19       | Minimal  | Healthy file, continue practices     |

#### 4.9.4 Deep-Dive Analysis

**Identify Technical Debt:**

```python
import json
import pandas as pd

# Load file metrics
with open('output/frontend/file_metrics_index.json', 'r') as f:
    metrics = json.load(f)

# Extract metrics into DataFrame
records = []
for filepath, file_metrics in metrics['files'].items():
    record = {
        'file': filepath,
        'churn': file_metrics['activity']['total_churn'],
        'commits': file_metrics['activity']['total_commits'],
        'bus_factor': file_metrics['authors']['bus_factor'],
        'coupling_degree': file_metrics['coupling']['coupling_degree'],
        'health': file_metrics['health']['overall_score'],
        'risk_level': file_metrics['health']['risk_level']
    }
    records.append(record)

df = pd.DataFrame(records)

# Technical debt indicators
debt_df = df[
    (df['churn'] > 2000) &
    (df['bus_factor'] <= 2) &
    (df['coupling_degree'] > 10)
].sort_values('churn', ascending=False)

print("Technical debt candidates:")
print(debt_df)
```

**Find Risky Files:**

```python
# Calculate custom risk score
def risk_score(row):
    score = 0
    if row['churn'] > 3000: score += 30
    if row['bus_factor'] == 1: score += 40
    if row['coupling_degree'] > 15: score += 30
    return score

df['risk_score'] = df.apply(risk_score, axis=1)

high_risk = df[df['risk_score'] >= 50].sort_values('risk_score', ascending=False)

print("High-risk files:")
print(high_risk[['file', 'risk_score', 'churn', 'bus_factor', 'coupling_degree']])
```

**Analyze Coupling Patterns:**

```python
# Load full metrics for coupling analysis
coupling_data = []
for filepath, file_metrics in metrics['files'].items():
    for coupled in file_metrics['coupling']['strongly_coupled_files']:
        coupling_data.append({
            'source': filepath,
            'target': coupled['file'],
            'strength': coupled['strength'],
            'commits': coupled['commits_together']
        })

coupling_df = pd.DataFrame(coupling_data)

# Find files with most outbound couplings
outbound_counts = coupling_df.groupby('source').size().sort_values(ascending=False)

print("Files with most couplings:")
print(outbound_counts.head(10))

# Files involved in strongest couplings
strongest = coupling_df.nlargest(10, 'strength')

print("\nStrongest coupling pairs:")
print(strongest)
```

---

### 4.10 Temporal Activity Map (Frontend)

**File:** `frontend/temporal_activity_map.json`

Heatmap-ready data showing activity patterns across weeks and days.

#### 4.10.1 Heatmap Data Structure

```json
{
  "metadata": {
    "generated_at": "2024-11-15T10:34:30+00:00",
    "date_range": {
      "start": "2023-01-15",
      "end": "2024-11-15"
    },
    "total_weeks": 96,
    "max_intensity": 45
  },
  "weekly_activity": [
    {
      "year": 2024,
      "week": 11,
      "start_date": "2024-03-11",
      "end_date": "2024-03-17",
      "by_day": {
        "monday": {
          "commits": 8,
          "files_modified": 12,
          "churn": 234,
          "intensity": 0.65
        },
        "tuesday": {
          "commits": 12,
          "files_modified": 18,
          "churn": 456,
          "intensity": 0.92
        },
        "wednesday": {
          "commits": 10,
          "files_modified": 15,
          "churn": 345,
          "intensity": 0.78
        },
        "thursday": {
          "commits": 15,
          "files_modified": 23,
          "churn": 567,
          "intensity": 1.0
        },
        "friday": {
          "commits": 6,
          "files_modified": 9,
          "churn": 178,
          "intensity": 0.45
        },
        "saturday": {
          "commits": 2,
          "files_modified": 3,
          "churn": 67,
          "intensity": 0.15
        },
        "sunday": {
          "commits": 1,
          "files_modified": 2,
          "churn": 23,
          "intensity": 0.08
        }
      },
      "weekly_total": {
        "commits": 54,
        "files_modified": 82,
        "churn": 1870
      }
    }
  ]
}
```

#### 4.10.2 Weekly Activity Patterns

**Day-of-Week Distribution:**

```json
{
  "day_of_week_stats": {
    "monday": { "avg_commits": 8.5, "total_weeks_active": 85 },
    "tuesday": { "avg_commits": 10.2, "total_weeks_active": 88 },
    "wednesday": { "avg_commits": 9.8, "total_weeks_active": 87 },
    "thursday": { "avg_commits": 11.5, "total_weeks_active": 90 },
    "friday": { "avg_commits": 7.2, "total_weeks_active": 82 },
    "saturday": { "avg_commits": 2.1, "total_weeks_active": 45 },
    "sunday": { "avg_commits": 1.3, "total_weeks_active": 32 }
  }
}
```

**Hour-of-Day Patterns** (if Git timestamps include time):

```json
{
  "hour_of_day_stats": {
    "09": { "avg_commits": 3.2 },
    "10": { "avg_commits": 5.7 },
    "11": { "avg_commits": 6.1 },
    "14": { "avg_commits": 7.8 },
    "15": { "avg_commits": 6.4 }
  }
}
```

#### 4.10.3 Activity Intensity Metrics

**Intensity Calculation:**

```python
def calculate_intensity(commits, max_commits):
    """Normalize to 0-1 scale"""
    return commits / max_commits if max_commits > 0 else 0
```

**Normalized Activity Scores:**

- 1.0: Peak activity (highest commit day)
- 0.5: Half of peak activity
- 0.0: No activity

**Comparative Scaling:**

```python
# Scale across entire dataset
global_max = max(day['commits'] for week in data
                for day in week['by_day'].values())

for week in data:
    for day in week['by_day'].values():
        day['intensity'] = day['commits'] / global_max
```

#### 4.10.4 Visualization Code

**Create Activity Heatmap with Seaborn:**

```python
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load activity map
with open('output/frontend/temporal_activity_map.json', 'r') as f:
    activity = json.load(f)

# Convert to matrix format
weeks = []
days = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']

for week_data in activity['weekly_activity']:
    week_row = []
    for day in days:
        intensity = week_data['by_day'][day]['intensity']
        week_row.append(intensity)
    weeks.append(week_row)

# Create DataFrame
df = pd.DataFrame(weeks, columns=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])

# Plot heatmap
fig, ax = plt.subplots(figsize=(12, 20))
sns.heatmap(df, cmap='YlOrRd', ax=ax, cbar_kws={'label': 'Activity Intensity'})
ax.set_xlabel('Day of Week')
ax.set_ylabel('Week Number')
ax.set_title('Development Activity Heatmap')
plt.tight_layout()
plt.show()
```

**Identify Working Patterns:**

```python
# Calculate average intensity by day of week
day_averages = {day: [] for day in days}

for week_data in activity['weekly_activity']:
    for day in days:
        day_averages[day].append(week_data['by_day'][day]['intensity'])

avg_by_day = {day: sum(intensities) / len(intensities)
              for day, intensities in day_averages.items()}

# Plot
fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(avg_by_day.keys(), avg_by_day.values())
ax.set_xlabel('Day of Week')
ax.set_ylabel('Average Activity Intensity')
ax.set_title('Average Activity by Day of Week')
ax.tick_params(axis='x', rotation=45)
plt.tight_layout()
plt.show()

print("Peak activity day:", max(avg_by_day, key=avg_by_day.get))
print("Lowest activity day:", min(avg_by_day, key=avg_by_day.get))
```

**Compare Periods:**

```python
# Split into time periods
import datetime

def get_period(week_data):
    date = datetime.datetime.fromisoformat(week_data['start_date'])
    if date < datetime.datetime(2023, 7, 1):
        return 'H1 2023'
    elif date < datetime.datetime(2024, 1, 1):
        return 'H2 2023'
    elif date < datetime.datetime(2024, 7, 1):
        return 'H1 2024'
    else:
        return 'H2 2024'

period_activity = {}
for week_data in activity['weekly_activity']:
    period = get_period(week_data)
    if period not in period_activity:
        period_activity[period] = []

    total = week_data['weekly_total']['commits']
    period_activity[period].append(total)

# Compare
for period, commits in period_activity.items():
    print(f"{period}: Avg {sum(commits)/len(commits):.1f} commits/week")
```

**Detect Anomalies:**

```python
import numpy as np

# Calculate z-scores for weekly activity
weekly_commits = [week['weekly_total']['commits']
                 for week in activity['weekly_activity']]

mean = np.mean(weekly_commits)
std = np.std(weekly_commits)
z_scores = [(x - mean) / std for x in weekly_commits]

# Find anomalous weeks (|z| > 2)
anomalies = []
for i, (week, z) in enumerate(zip(activity['weekly_activity'], z_scores)):
    if abs(z) > 2:
        anomalies.append({
            'week': f"{week['year']}-W{week['week']}",
            'commits': week['weekly_total']['commits'],
            'z_score': z,
            'type': 'high' if z > 0 else 'low'
        })

print("Anomalous weeks:")
for anom in sorted(anomalies, key=lambda x: abs(x['z_score']), reverse=True):
    print(f"{anom['week']}: {anom['commits']} commits (z={anom['z_score']:.2f}, {anom['type']})")
```

---

## 5. Common Analysis Workflows

### 5.1 Code Quality Assessment

**Objective:** Identify high-risk files that need attention based on multiple quality signals.

**Complete Workflow:**

```python
import json
import pandas as pd
import numpy as np

# Load datasets
with open('output/metadata/file_index.json', 'r') as f:
    metadata = json.load(f)

with open('output/networks/cochange_network.json', 'r') as f:
    cochange = json.load(f)

# Convert to DataFrames
files_df = pd.DataFrame(metadata['files'].values())
coupling_df = pd.DataFrame(cochange['edges'])

# Calculate coupling degree for each file
coupling_counts = coupling_df['source'].value_counts()
files_df['coupling_degree'] = files_df['file_path'].map(coupling_counts).fillna(0)

# Calculate risk score
def quality_risk_score(row):
    score = 0

    # High churn (30 points max)
    churn_norm = min(row['total_churn'] / 5000, 1.0)
    score += churn_norm * 30

    # Low bus factor (30 points max)
    if row['unique_authors'] == 1:
        score += 30
    elif row['unique_authors'] == 2:
        score += 20
    elif row['unique_authors'] == 3:
        score += 10

    # High coupling (25 points max)
    coupling_norm = min(row['coupling_degree'] / 20, 1.0)
    score += coupling_norm * 25

    # High frequency changes (15 points max)
    if row['total_commits'] > 100:
        score += 15
    elif row['total_commits'] > 50:
        score += 10
    elif row['total_commits'] > 25:
        score += 5

    return score

files_df['risk_score'] = files_df.apply(quality_risk_score, axis=1)

# Create quality categories
def categorize_quality(score):
    if score >= 70:
        return 'Critical'
    elif score >= 50:
        return 'High Risk'
    elif score >= 30:
        return 'Medium Risk'
    else:
        return 'Low Risk'

files_df['quality_category'] = files_df['risk_score'].apply(categorize_quality)

# Print results
print("Code Quality Assessment Summary:")
print(files_df['quality_category'].value_counts())

print("\nTop 10 highest risk files:")
high_risk = files_df.nlargest(10, 'risk_score')[
    ['file_path', 'risk_score', 'total_churn', 'unique_authors', 'coupling_degree', 'quality_category']
]
print(high_risk)

# Visualize
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Risk distribution
files_df['quality_category'].value_counts().plot(kind='bar', ax=axes[0, 0])
axes[0, 0].set_title('Quality Distribution')
axes[0, 0].set_ylabel('File Count')

# Churn vs Risk
axes[0, 1].scatter(files_df['total_churn'], files_df['risk_score'], alpha=0.5)
axes[0, 1].set_xlabel('Total Churn')
axes[0, 1].set_ylabel('Risk Score')
axes[0, 1].set_title('Churn vs Risk')

# Bus Factor vs Risk
axes[1, 0].scatter(files_df['unique_authors'], files_df['risk_score'], alpha=0.5)
axes[1, 0].set_xlabel('Unique Authors')
axes[1, 0].set_ylabel('Risk Score')
axes[1, 0].set_title('Bus Factor vs Risk')

# Coupling vs Risk
axes[1, 1].scatter(files_df['coupling_degree'], files_df['risk_score'], alpha=0.5)
axes[1, 1].set_xlabel('Coupling Degree')
axes[1, 1].set_ylabel('Risk Score')
axes[1, 1].set_title('Coupling vs Risk')

plt.tight_layout()
plt.show()

# Export results
files_df.to_csv('quality_assessment.csv', index=False)
print("\nResults exported to quality_assessment.csv")
```

### 5.2 Team Dynamics Analysis

**Objective:** Understand collaboration patterns, knowledge distribution, and team health.

**Complete Workflow:**

```python
import json
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from datetime import datetime

# Load datasets
with open('output/file_lifecycle.json', 'r') as f:
    lifecycle = json.load(f)

with open('output/networks/author_network.json', 'r') as f:
    author_network = json.load(f)

# Extract author contribution patterns
author_stats = {}
for file_obj in lifecycle['files']:
    for event in file_obj['lifecycle']:
        author_id = event.get('author_id')
        if not author_id:
            continue

        if author_id not in author_stats:
            author_stats[author_id] = {
                'commits': 0,
                'files': set(),
                'churn': 0,
                'first_commit': event['timestamp'],
                'last_commit': event['timestamp']
            }

        author_stats[author_id]['commits'] += 1
        author_stats[author_id]['files'].add(file_obj['file_path'])
        author_stats[author_id]['churn'] += event.get('lines_added', 0) + event.get('lines_deleted', 0)
        author_stats[author_id]['last_commit'] = event['timestamp']

# Convert to DataFrame
authors_df = pd.DataFrame([
    {
        'author_id': author,
        'commits': stats['commits'],
        'files_touched': len(stats['files']),
        'churn': stats['churn'],
        'first_commit': stats['first_commit'],
        'last_commit': stats['last_commit']
    }
    for author, stats in author_stats.items()
])

# Calculate tenure
authors_df['first_commit'] = pd.to_datetime(authors_df['first_commit'])
authors_df['last_commit'] = pd.to_datetime(authors_df['last_commit'])
authors_df['tenure_days'] = (authors_df['last_commit'] - authors_df['first_commit']).dt.days
authors_df['velocity'] = authors_df['commits'] / (authors_df['tenure_days'] + 1)

print("Team Composition:")
print(f"Total authors: {len(authors_df)}")
print(f"Active authors (>10 commits): {len(authors_df[authors_df['commits'] > 10])}")

print("\nTop Contributors:")
print(authors_df.nlargest(10, 'commits')[['author_id', 'commits', 'files_touched', 'churn']])

# Knowledge distribution
from scipy import stats as scipy_stats

# Calculate Gini coefficient (inequality measure)
def gini_coefficient(x):
    sorted_x = np.sort(x)
    n = len(x)
    cumsum = np.cumsum(sorted_x)
    return (2 * np.sum((n - np.arange(1, n+1) + 0.5) * sorted_x)) / (n * np.sum(sorted_x)) - 1

gini = gini_coefficient(authors_df['commits'].values)
print(f"\nKnowledge Distribution (Gini): {gini:.3f}")
print("(0 = perfect equality, 1 = perfect inequality)")

# Collaboration analysis
G = nx.Graph()
for node in author_network['nodes']:
    G.add_node(node['author_id'], **node)

for edge in author_network['edges']:
    G.add_edge(edge['source'], edge['target'], **edge)

# Network metrics
density = nx.density(G)
avg_clustering = nx.average_clustering(G)

print(f"\nCollaboration Network:")
print(f"Density: {density:.3f}")
print(f"Average clustering: {avg_clustering:.3f}")

# Identify isolated authors
isolated = [node for node in G.nodes() if G.degree(node) == 0]
if isolated:
    print(f"\nIsolated authors (no collaborations): {len(isolated)}")
    print(isolated)

# Community detection
communities = list(nx.community.louvain_communities(G))
print(f"\nDetected {len(communities)} collaboration communities")

# Onboarding/offboarding detection
now = datetime.now()
recent_threshold = 30  # days
leaving_threshold = 90  # days

authors_df['days_since_last'] = (now - authors_df['last_commit']).dt.days

new_authors = authors_df[authors_df['tenure_days'] < recent_threshold]
leaving_authors = authors_df[authors_df['days_since_last'] > leaving_threshold]

print(f"\nNew authors (tenure < {recent_threshold} days): {len(new_authors)}")
if len(new_authors) > 0:
    print(new_authors[['author_id', 'tenure_days', 'commits']])

print(f"\nPotentially leaving authors (inactive >{leaving_threshold} days): {len(leaving_authors)}")
if len(leaving_authors) > 0:
    print(leaving_authors[['author_id', 'days_since_last', 'commits']])

# Visualizations
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Commit distribution
axes[0, 0].hist(authors_df['commits'], bins=20)
axes[0, 0].set_xlabel('Commits')
axes[0, 0].set_ylabel('Author Count')
axes[0, 0].set_title('Commit Distribution')

# Tenure vs Commits
axes[0, 1].scatter(authors_df['tenure_days'], authors_df['commits'], alpha=0.6)
axes[0, 1].set_xlabel('Tenure (days)')
axes[0, 1].set_ylabel('Commits')
axes[0, 1].set_title('Tenure vs Productivity')

# Collaboration network
pos = nx.spring_layout(G, k=0.5, iterations=50)
node_sizes = [G.nodes[node]['total_commits'] * 2 for node in G.nodes()]
nx.draw(G, pos, ax=axes[1, 0], node_size=node_sizes, with_labels=False, alpha=0.7)
axes[1, 0].set_title('Collaboration Network')

# Velocity distribution
axes[1, 1].hist(authors_df['velocity'], bins=20)
axes[1, 1].set_xlabel('Velocity (commits/day)')
axes[1, 1].set_ylabel('Author Count')
axes[1, 1].set_title('Author Velocity Distribution')

plt.tight_layout()
plt.show()
```

---

This completes Part 3 of the Strata.py Dataset Reference Guide.

**Final Part:**

- Part 4: Advanced Topics, Troubleshooting, and Reference (Sections 6-8)
