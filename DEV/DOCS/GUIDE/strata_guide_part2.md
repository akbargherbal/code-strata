# Strata.py Dataset Reference Guide - Part 2

## Core Datasets Reference (Sections 4.1-4.5)

---

## 4. Dataset Reference

### 4.1 Core Lifecycle Dataset

**File:** `file_lifecycle.json`

This is the foundational dataset containing complete file history with all events, temporal metadata, and diff statistics.

#### 4.1.1 Schema Overview

**Top-Level Structure:**

```json
{
  "metadata": {
    "version": "2.2.0",
    "schema_version": "1.1.0",
    "repository": "/path/to/repo",
    "generated_at": "2024-11-15T10:30:45+00:00",
    "total_commits": 1523,
    "files_tracked": 342,
    "quick_mode": false
  },
  "files": [
    {
      "file_path": "src/core/engine.py",
      "lifecycle": [
        /* array of events */
      ]
    }
  ]
}
```

**Metadata Section Fields:**

| Field            | Type              | Description                          |
| ---------------- | ----------------- | ------------------------------------ |
| `version`        | string            | Strata.py tool version               |
| `schema_version` | string            | JSON schema version                  |
| `repository`     | string            | Absolute path to analyzed repository |
| `generated_at`   | ISO 8601 datetime | Timestamp of analysis run            |
| `total_commits`  | integer           | Total commits processed              |
| `files_tracked`  | integer           | Number of files in lifecycle array   |
| `quick_mode`     | boolean           | Whether sampling was used            |
| `sample_rate`    | float             | Sampling rate if quick_mode true     |

**Files Array Structure:**

```json
{
  "file_path": "relative/path/to/file.py",
  "lifecycle": [
    {
      /* event 1 */
    },
    {
      /* event 2 */
    },
    {
      /* event N */
    }
  ]
}
```

Each file object contains:

- `file_path`: Relative path from repository root
- `lifecycle`: Chronologically sorted array of events

#### 4.1.2 Event Types

All events share common fields plus type-specific fields.

**Common Fields:**

```json
{
  "event_type": "create|modify|rename|delete",
  "commit_hash": "a1b2c3d4e5f6...",
  "timestamp": "2024-03-15T14:30:00+00:00",
  "author_name": "Jane Developer",
  "author_email": "jane@example.com",
  "message": "Add new feature",
  "lines_added": 50,
  "lines_deleted": 10,
  "temporal_labels": {
    "year": 2024,
    "quarter": 1,
    "month": 3,
    "week_no": 11,
    "day_of_week": 4,
    "day_of_year": 75
  },
  "author_id": "jane_developer_jane",
  "author_domain": "example.com",
  "position_in_history": 15
}
```

**Create Event:**

```json
{
  "event_type": "create",
  "commit_hash": "abc123...",
  "timestamp": "2024-01-15T09:00:00+00:00",
  "author_name": "John Smith",
  "author_email": "john@company.com",
  "message": "Initial implementation",
  "lines_added": 200,
  "lines_deleted": 0,
  "temporal_labels": {
    /* ... */
  },
  "author_id": "john_smith_john",
  "author_domain": "company.com",
  "position_in_history": 0
}
```

**Modify Event:**

```json
{
  "event_type": "modify",
  "commit_hash": "def456...",
  "timestamp": "2024-02-10T14:30:00+00:00",
  "author_name": "Jane Developer",
  "author_email": "jane@company.com",
  "message": "Refactor validation logic",
  "lines_added": 45,
  "lines_deleted": 32,
  "temporal_labels": {
    /* ... */
  },
  "author_id": "jane_developer_jane",
  "author_domain": "company.com",
  "position_in_history": 3
}
```

**Rename Event:**

```json
{
  "event_type": "rename",
  "commit_hash": "ghi789...",
  "timestamp": "2024-03-05T11:15:00+00:00",
  "old_path": "utils/validator.py",
  "new_path": "core/validation.py",
  "author_name": "Bob Johnson",
  "author_email": "bob@company.com",
  "message": "Reorganize module structure",
  "lines_added": 0,
  "lines_deleted": 0,
  "temporal_labels": {
    /* ... */
  },
  "author_id": "bob_johnson_bob",
  "author_domain": "company.com",
  "position_in_history": 7
}
```

**Delete Event:**

```json
{
  "event_type": "delete",
  "commit_hash": "jkl012...",
  "timestamp": "2024-04-01T16:45:00+00:00",
  "author_name": "Sarah Lee",
  "author_email": "sarah@company.com",
  "message": "Remove deprecated module",
  "lines_added": 0,
  "lines_deleted": 247,
  "temporal_labels": {
    /* ... */
  },
  "author_id": "sarah_lee_sarah",
  "author_domain": "company.com",
  "position_in_history": 12
}
```

#### 4.1.3 Event Fields Explained

**commit_hash:**

- Full SHA-1 hash (40 characters) or abbreviated (7-10 characters)
- Unique identifier for the commit
- Use to cross-reference with Git: `git show <hash>`

**timestamp:**

- ISO 8601 format with timezone: `YYYY-MM-DDTHH:MM:SS+TZ`
- Represents author date (when changes were made)
- UTC timezone typically used (`+00:00`)

**author_name and author_email:**

- As recorded in Git commit
- May have variations for same person (see Author Normalization)
- Use `author_id` for consistent identification

**message:**

- First line of commit message
- May be truncated if very long
- Check for patterns: "fix", "feat", "refactor", etc.

**lines_added and lines_deleted:**

- From Git diff statistics
- `lines_added`: New lines introduced
- `lines_deleted`: Lines removed
- Both zero for pure renames
- Sum gives total churn: `churn = lines_added + lines_deleted`

**temporal_labels:**

Enriched time information for filtering and aggregation:

```json
{
  "year": 2024, // Calendar year
  "quarter": 2, // Q1-Q4 (1-4)
  "month": 6, // January=1, December=12
  "week_no": 23, // ISO week number (1-53)
  "day_of_week": 2, // Monday=0, Sunday=6
  "day_of_year": 153 // 1-366
}
```

**position_in_history:**

- Zero-indexed sequence number within file's lifecycle
- First event (create): 0
- Last event: N-1 where N = total events
- Useful for identifying early vs. late changes

#### 4.1.4 Pandas Code Examples

**Loading and Basic Exploration:**

```python
import pandas as pd
import json

# Load lifecycle data
with open('output/file_lifecycle.json', 'r') as f:
    data = json.load(f)

# Convert to flat DataFrame
records = []
for file_obj in data['files']:
    file_path = file_obj['file_path']
    for event in file_obj['lifecycle']:
        event['file_path'] = file_path
        records.append(event)

df = pd.DataFrame(records)

# Basic info
print(f"Total events: {len(df)}")
print(f"Unique files: {df['file_path'].nunique()}")
print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
print(f"\nEvent type distribution:")
print(df['event_type'].value_counts())
```

**Calculate File Lifespans:**

```python
# Ensure timestamp is datetime
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Calculate lifespan for each file
file_stats = df.groupby('file_path').agg({
    'timestamp': ['min', 'max'],
    'commit_hash': 'count',
    'lines_added': 'sum',
    'lines_deleted': 'sum'
}).reset_index()

file_stats.columns = ['file_path', 'created_at', 'last_modified',
                      'event_count', 'total_lines_added', 'total_lines_deleted']

# Calculate lifespan in days
file_stats['lifespan_days'] = (file_stats['last_modified'] -
                                file_stats['created_at']).dt.days

# Calculate total churn
file_stats['total_churn'] = (file_stats['total_lines_added'] +
                              file_stats['total_lines_deleted'])

# Display highest churn files
print(file_stats.nlargest(10, 'total_churn')[
    ['file_path', 'total_churn', 'event_count', 'lifespan_days']
])
```

**Identify Most Modified Files:**

```python
# Count modify events per file
modify_counts = df[df['event_type'] == 'modify'].groupby('file_path').size()
modify_counts = modify_counts.sort_values(ascending=False)

print("Top 10 most modified files:")
print(modify_counts.head(10))

# Combine with churn
hotspots = pd.DataFrame({
    'modify_count': modify_counts,
    'total_churn': file_stats.set_index('file_path')['total_churn']
})
hotspots['activity_score'] = hotspots['modify_count'] * hotspots['total_churn']

print("\nTop activity hotspots:")
print(hotspots.nlargest(10, 'activity_score'))
```

**Plot File Creation Timeline:**

```python
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Get creation events
creates = df[df['event_type'] == 'create'].copy()
creates['timestamp'] = pd.to_datetime(creates['timestamp'])
creates = creates.sort_values('timestamp')

# Group by month
creates['month'] = creates['timestamp'].dt.to_period('M')
monthly_creates = creates.groupby('month').size()

# Plot
fig, ax = plt.subplots(figsize=(12, 6))
monthly_creates.plot(kind='bar', ax=ax)
ax.set_xlabel('Month')
ax.set_ylabel('Files Created')
ax.set_title('File Creation Timeline')
ax.tick_params(axis='x', rotation=45)
plt.tight_layout()
plt.show()
```

**Analyze Author Contribution Patterns:**

```python
# Author statistics
author_stats = df.groupby('author_id').agg({
    'commit_hash': 'nunique',
    'file_path': 'nunique',
    'lines_added': 'sum',
    'lines_deleted': 'sum'
}).reset_index()

author_stats.columns = ['author_id', 'commits', 'files_touched',
                        'lines_added', 'lines_deleted']
author_stats['churn'] = author_stats['lines_added'] + author_stats['lines_deleted']

print("Top contributors:")
print(author_stats.nlargest(10, 'commits'))

# Author activity over time
author_timeline = df.groupby([df['timestamp'].dt.to_period('M'),
                               'author_id']).size().unstack(fill_value=0)
author_timeline.plot(figsize=(14, 7), title='Author Activity Over Time')
plt.ylabel('Commits')
plt.xlabel('Month')
plt.legend(title='Author', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()
```

---

### 4.2 File Metadata Index

**File:** `metadata/file_index.json`

Aggregated file-level statistics for quick lookups without processing full lifecycle.

#### 4.2.1 Schema and Structure

```json
{
  "metadata": {
    "generated_at": "2024-11-15T10:31:00+00:00",
    "total_files": 342,
    "active_files": 298,
    "deleted_files": 44
  },
  "files": {
    "src/core/engine.py": {
      "file_path": "src/core/engine.py",
      "extension": "py",
      "status": "active",
      "created_at": "2023-01-15T09:00:00+00:00",
      "last_modified": "2024-11-10T16:45:00+00:00",
      "deleted_at": null,
      "total_commits": 87,
      "unique_authors": 5,
      "lines_added": 3420,
      "lines_deleted": 1890,
      "total_churn": 5310,
      "modify_events": 85,
      "rename_events": 1,
      "current_path": "src/core/engine.py",
      "original_path": "utils/engine.py",
      "primary_author": "jane_developer_jane",
      "author_domain_diversity": 3
    }
  }
}
```

#### 4.2.2 Metadata Fields

**Status:**

- `"active"`: File exists in latest commit
- `"deleted"`: File was removed
- Use for filtering: `[f for f in files if f['status'] == 'active']`

**Temporal Fields:**

```json
{
  "created_at": "ISO timestamp of first commit",
  "last_modified": "ISO timestamp of most recent commit",
  "deleted_at": "ISO timestamp of deletion (null if active)"
}
```

**Activity Metrics:**

```json
{
  "total_commits": 87, // Unique commits affecting this file
  "modify_events": 85, // Number of modify operations
  "rename_events": 1 // Number of path changes
}
```

**Author Metrics:**

```json
{
  "unique_authors": 5, // Distinct author IDs
  "primary_author": "author_id", // Most frequent contributor
  "author_domain_diversity": 3 // Number of unique email domains
}
```

**Churn Aggregates:**

```json
{
  "lines_added": 3420, // Cumulative lines added
  "lines_deleted": 1890, // Cumulative lines deleted
  "total_churn": 5310 // Sum of added + deleted
}
```

**Path Tracking:**

```json
{
  "current_path": "src/core/engine.py", // Latest path
  "original_path": "utils/engine.py", // Initial path
  "file_path": "src/core/engine.py" // Same as current_path
}
```

**File Extension:**

```json
{
  "extension": "py" // Extracted from filename
}
```

#### 4.2.3 Usage Examples

**Find High-Churn Files:**

```python
import json
import pandas as pd

# Load metadata
with open('output/metadata/file_index.json', 'r') as f:
    index = json.load(f)

# Convert to DataFrame
files_df = pd.DataFrame(index['files'].values())

# Find top churn files
high_churn = files_df.nlargest(20, 'total_churn')[
    ['file_path', 'total_churn', 'total_commits', 'unique_authors']
]

print("High-churn files:")
print(high_churn)

# Calculate churn per commit
files_df['churn_per_commit'] = files_df['total_churn'] / files_df['total_commits']

# Find files with high churn rate
print("\nFiles with highest churn per commit:")
print(files_df.nlargest(20, 'churn_per_commit')[
    ['file_path', 'churn_per_commit', 'total_commits']
])
```

**Identify Single-Author Files (Bus Factor 1):**

```python
# Files with only one author
single_author_files = files_df[files_df['unique_authors'] == 1]

print(f"Files with single author: {len(single_author_files)}")
print(f"Percentage: {len(single_author_files) / len(files_df) * 100:.1f}%")

# High-risk: single author + high churn
high_risk = single_author_files[single_author_files['total_churn'] > 1000]

print(f"\nHigh-risk files (single author + high churn):")
print(high_risk[['file_path', 'primary_author', 'total_churn', 'total_commits']])
```

**Filter by File Type/Extension:**

```python
# Group by extension
extension_stats = files_df.groupby('extension').agg({
    'file_path': 'count',
    'total_churn': 'sum',
    'total_commits': 'sum'
}).reset_index()

extension_stats.columns = ['extension', 'file_count', 'total_churn', 'total_commits']
extension_stats = extension_stats.sort_values('file_count', ascending=False)

print("Files by extension:")
print(extension_stats)

# Analyze Python files specifically
py_files = files_df[files_df['extension'] == 'py']
print(f"\nPython files: {len(py_files)}")
print(f"Average churn: {py_files['total_churn'].mean():.0f}")
print(f"Average authors per file: {py_files['unique_authors'].mean():.1f}")
```

**Calculate Average File Age:**

```python
from datetime import datetime

# Ensure datetime types
files_df['created_at'] = pd.to_datetime(files_df['created_at'])
files_df['last_modified'] = pd.to_datetime(files_df['last_modified'])

# Calculate ages
now = datetime.now()
files_df['age_days'] = (now - files_df['created_at']).dt.days
files_df['days_since_modification'] = (now - files_df['last_modified']).dt.days

print(f"Average file age: {files_df['age_days'].mean():.0f} days")
print(f"Average days since last modification: {files_df['days_since_modification'].mean():.0f}")

# Find stale files (old + not recently modified)
stale_threshold = 90  # days
stale_files = files_df[files_df['days_since_modification'] > stale_threshold]

print(f"\nStale files (>90 days since modification): {len(stale_files)}")
print(stale_files.nlargest(10, 'days_since_modification')[
    ['file_path', 'days_since_modification', 'total_commits']
])
```

**Domain Diversity Analysis:**

```python
# Files with contributors from multiple organizations
multi_domain = files_df[files_df['author_domain_diversity'] > 1]

print(f"Files with multi-domain contributors: {len(multi_domain)}")
print(f"Percentage: {len(multi_domain) / len(files_df) * 100:.1f}%")

# Cross-tabulate
print("\nAuthor diversity distribution:")
print(files_df['author_domain_diversity'].value_counts().sort_index())
```

---

### 4.3 Temporal Aggregations

#### Daily Aggregation

**File:** `aggregations/temporal_daily.json`

Day-by-day activity summaries for fine-grained time-series analysis.

**Schema:**

```json
{
  "metadata": {
    "start_date": "2023-01-15",
    "end_date": "2024-11-15",
    "total_days": 670,
    "active_days": 423
  },
  "daily_activity": [
    {
      "date": "2024-03-15",
      "commits": 12,
      "files_modified": 23,
      "unique_authors": 4,
      "lines_added": 450,
      "lines_deleted": 230,
      "total_churn": 680,
      "create_events": 2,
      "modify_events": 19,
      "rename_events": 1,
      "delete_events": 1,
      "authors_active": [
        "jane_developer_jane",
        "john_smith_john",
        "bob_johnson_bob",
        "sarah_lee_sarah"
      ],
      "day_of_week": 4,
      "is_weekend": false
    }
  ]
}
```

**Key Metrics Per Day:**

| Field            | Type                | Description                 |
| ---------------- | ------------------- | --------------------------- |
| `date`           | string (YYYY-MM-DD) | Calendar date               |
| `commits`        | integer             | Unique commits on this date |
| `files_modified` | integer             | Distinct files touched      |
| `unique_authors` | integer             | Number of active authors    |
| `lines_added`    | integer             | Total lines added           |
| `lines_deleted`  | integer             | Total lines deleted         |
| `total_churn`    | integer             | Sum of added + deleted      |
| `create_events`  | integer             | Files created               |
| `modify_events`  | integer             | Modification events         |
| `rename_events`  | integer             | Rename operations           |
| `delete_events`  | integer             | Files deleted               |
| `authors_active` | array               | List of author IDs          |
| `day_of_week`    | integer             | 0=Monday, 6=Sunday          |
| `is_weekend`     | boolean             | Saturday or Sunday          |

#### Monthly Aggregation

**File:** `aggregations/temporal_monthly.json`

Month-by-month rollups for trend analysis and planning.

**Schema:**

```json
{
  "metadata": {
    "start_month": "2023-01",
    "end_month": "2024-11",
    "total_months": 23,
    "active_months": 23
  },
  "monthly_activity": [
    {
      "month": "2024-03",
      "year": 2024,
      "month_number": 3,
      "quarter": 1,
      "commits": 245,
      "files_modified": 187,
      "unique_authors": 8,
      "lines_added": 12450,
      "lines_deleted": 6780,
      "total_churn": 19230,
      "create_events": 15,
      "modify_events": 220,
      "rename_events": 8,
      "delete_events": 2,
      "active_days": 22,
      "working_days": 21,
      "weekend_commits": 18,
      "authors": {
        "jane_developer_jane": 89,
        "john_smith_john": 67,
        "bob_johnson_bob": 45,
        "sarah_lee_sarah": 23,
        "mike_wilson_mike": 15,
        "lisa_brown_lisa": 6
      },
      "top_files": [
        { "file": "src/core/engine.py", "commits": 18 },
        { "file": "src/api/handlers.py", "commits": 15 }
      ]
    }
  ]
}
```

**Additional Monthly Fields:**

| Field             | Description                                    |
| ----------------- | ---------------------------------------------- |
| `active_days`     | Days with at least one commit                  |
| `working_days`    | Weekdays in the month                          |
| `weekend_commits` | Commits made on weekends                       |
| `authors`         | Dictionary mapping author IDs to commit counts |
| `top_files`       | Most frequently modified files                 |

#### 4.3.3 Time-Series Metrics

**Loading and Analysis:**

```python
import json
import pandas as pd
import matplotlib.pyplot as plt

# Load daily data
with open('output/aggregations/temporal_daily.json', 'r') as f:
    daily_data = json.load(f)

daily_df = pd.DataFrame(daily_data['daily_activity'])
daily_df['date'] = pd.to_datetime(daily_df['date'])
daily_df = daily_df.set_index('date')

# Load monthly data
with open('output/aggregations/temporal_monthly.json', 'r') as f:
    monthly_data = json.load(f)

monthly_df = pd.DataFrame(monthly_data['monthly_activity'])
monthly_df['month'] = pd.to_datetime(monthly_df['month'])
monthly_df = monthly_df.set_index('month')
```

**Plot Commit Activity Timeline:**

```python
# Daily commits with moving average
fig, ax = plt.subplots(figsize=(14, 6))

ax.bar(daily_df.index, daily_df['commits'], alpha=0.5, label='Daily Commits')

# 7-day moving average
ma_7 = daily_df['commits'].rolling(window=7).mean()
ax.plot(daily_df.index, ma_7, color='red', linewidth=2, label='7-day MA')

ax.set_xlabel('Date')
ax.set_ylabel('Commits')
ax.set_title('Daily Commit Activity')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

**Identify Peak Development Periods:**

```python
# Monthly analysis
print("Top 5 most active months:")
print(monthly_df.nlargest(5, 'commits')[
    ['commits', 'files_modified', 'unique_authors', 'total_churn']
])

# Find months with highest velocity
monthly_df['velocity'] = monthly_df['commits'] / monthly_df['active_days']
print("\nHighest velocity months (commits per active day):")
print(monthly_df.nlargest(5, 'velocity')[['commits', 'active_days', 'velocity']])
```

**Calculate Moving Averages:**

```python
# Different window sizes
daily_df['ma_7'] = daily_df['commits'].rolling(window=7).mean()
daily_df['ma_30'] = daily_df['commits'].rolling(window=30).mean()

# Plot multiple moving averages
fig, ax = plt.subplots(figsize=(14, 6))
daily_df['commits'].plot(ax=ax, alpha=0.3, label='Daily')
daily_df['ma_7'].plot(ax=ax, label='7-day MA', linewidth=2)
daily_df['ma_30'].plot(ax=ax, label='30-day MA', linewidth=2)
ax.set_title('Commit Activity with Moving Averages')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

**Detect Seasonal Patterns:**

```python
# Day of week analysis
weekday_avg = daily_df.groupby('day_of_week')['commits'].mean()
weekday_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(weekday_names, weekday_avg)
ax.set_xlabel('Day of Week')
ax.set_ylabel('Average Commits')
ax.set_title('Commit Activity by Day of Week')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Weekend vs weekday
weekend_avg = daily_df[daily_df['is_weekend']]['commits'].mean()
weekday_avg_total = daily_df[~daily_df['is_weekend']]['commits'].mean()

print(f"Average commits - Weekdays: {weekday_avg_total:.1f}, Weekends: {weekend_avg:.1f}")
```

**Churn Analysis Over Time:**

```python
# Plot lines added vs deleted
fig, ax = plt.subplots(figsize=(14, 6))

monthly_df['lines_added'].plot(ax=ax, label='Lines Added', marker='o')
monthly_df['lines_deleted'].plot(ax=ax, label='Lines Deleted', marker='s')
monthly_df['total_churn'].plot(ax=ax, label='Total Churn', marker='^', linewidth=2)

ax.set_xlabel('Month')
ax.set_ylabel('Lines')
ax.set_title('Code Churn Over Time')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Net change
monthly_df['net_change'] = monthly_df['lines_added'] - monthly_df['lines_deleted']
monthly_df['net_change'].plot(kind='bar', figsize=(14, 6),
                               title='Net Lines Added per Month')
plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
plt.ylabel('Net Lines')
plt.tight_layout()
plt.show()
```

---

### 4.4 Author Network

**File:** `networks/author_network.json`

Graph representation of author collaborations based on shared file modifications.

#### 4.4.1 Network Structure

```json
{
  "metadata": {
    "generated_at": "2024-11-15T10:31:30+00:00",
    "total_authors": 12,
    "total_edges": 35,
    "density": 0.53
  },
  "nodes": [
    {
      "author_id": "jane_developer_jane",
      "author_name": "Jane Developer",
      "author_email": "jane@company.com",
      "author_domain": "company.com",
      "total_commits": 234,
      "files_touched": 87,
      "total_churn": 12450,
      "first_commit": "2023-01-15T09:00:00+00:00",
      "last_commit": "2024-11-10T16:45:00+00:00",
      "active_months": 23,
      "primary_files": ["src/core/engine.py", "src/api/handlers.py"]
    }
  ],
  "edges": [
    {
      "source": "jane_developer_jane",
      "target": "john_smith_john",
      "weight": 45,
      "shared_files": 28,
      "collaboration_strength": 0.65,
      "first_collaboration": "2023-02-01T10:30:00+00:00",
      "last_collaboration": "2024-11-08T14:20:00+00:00"
    }
  ]
}
```

#### 4.4.2 Author Node Attributes

**Identity Fields:**

```json
{
  "author_id": "normalized_id",
  "author_name": "Full Name",
  "author_email": "email@domain.com",
  "author_domain": "domain.com"
}
```

**Activity Metrics:**

```json
{
  "total_commits": 234, // Commits by this author
  "files_touched": 87, // Unique files modified
  "total_churn": 12450 // Cumulative lines changed
}
```

**Temporal Metrics:**

```json
{
  "first_commit": "ISO timestamp",
  "last_commit": "ISO timestamp",
  "active_months": 23 // Months with at least one commit
}
```

**Specialization:**

```json
{
  "primary_files": [
    // Most frequently modified files
    "src/core/engine.py",
    "src/api/handlers.py"
  ]
}
```

#### 4.4.3 Collaboration Metrics

**Edge Attributes:**

| Field                    | Type         | Description                          |
| ------------------------ | ------------ | ------------------------------------ |
| `source`                 | string       | Author ID of first collaborator      |
| `target`                 | string       | Author ID of second collaborator     |
| `weight`                 | integer      | Number of commits on shared files    |
| `shared_files`           | integer      | Files both authors modified          |
| `collaboration_strength` | float        | Normalized collaboration score (0-1) |
| `first_collaboration`    | ISO datetime | First shared file modification       |
| `last_collaboration`     | ISO datetime | Most recent collaboration            |

**Collaboration Strength Formula:**

$$\text{CollaborationStrength}(A, B) = \frac{\text{shared\_files}(A, B)}{\min(\text{files}(A), \text{files}(B))}$$

**Interpretation:**

- 0.8-1.0: Very strong collaboration (work on same files frequently)
- 0.6-0.79: Strong collaboration
- 0.4-0.59: Moderate collaboration
- 0.2-0.39: Weak collaboration
- 0.0-0.19: Minimal collaboration

#### 4.4.4 Analysis Examples

**Build NetworkX Graph:**

```python
import json
import networkx as nx
import matplotlib.pyplot as plt

# Load network data
with open('output/networks/author_network.json', 'r') as f:
    network = json.load(f)

# Create graph
G = nx.Graph()

# Add nodes with attributes
for node in network['nodes']:
    G.add_node(node['author_id'], **node)

# Add edges
for edge in network['edges']:
    G.add_edge(edge['source'], edge['target'],
               weight=edge['weight'],
               collaboration_strength=edge['collaboration_strength'])

print(f"Network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
```

**Calculate Centrality Measures:**

```python
# Degree centrality (number of collaborators)
degree_cent = nx.degree_centrality(G)

# Betweenness centrality (bridge between teams)
between_cent = nx.betweenness_centrality(G)

# Closeness centrality (average distance to others)
close_cent = nx.closeness_centrality(G)

# Eigenvector centrality (influence based on connections)
eigen_cent = nx.eigenvector_centrality(G, max_iter=1000)

# Combine into DataFrame
import pandas as pd
centrality_df = pd.DataFrame({
    'author': list(degree_cent.keys()),
    'degree': list(degree_cent.values()),
    'betweenness': list(between_cent.values()),
    'closeness': list(close_cent.values()),
    'eigenvector': list(eigen_cent.values())
})

print("Top 5 by degree centrality (most collaborators):")
print(centrality_df.nlargest(5, 'degree'))

print("\nTop 5 by betweenness (bridges between teams):")
print(centrality_df.nlargest(5, 'betweenness'))
```

**Identify Collaboration Clusters:**

```python
# Community detection (Louvain algorithm)
from networkx.algorithms import community

communities = community.louvain_communities(G, weight='collaboration_strength')

print(f"Detected {len(communities)} communities:")
for i, comm in enumerate(communities, 1):
    print(f"\nCommunity {i} ({len(comm)} members):")
    print(list(comm))

# Add community to nodes
for i, comm in enumerate(communities, 1):
    for node in comm:
        G.nodes[node]['community'] = i
```

**Visualize Author Relationships:**

```python
# Layout
pos = nx.spring_layout(G, k=0.5, iterations=50, seed=42)

# Node sizes by commit count
node_sizes = [G.nodes[node]['total_commits'] * 2 for node in G.nodes()]

# Node colors by community (if detected)
if 'community' in G.nodes[list(G.nodes())[0]]:
    node_colors = [G.nodes[node]['community'] for node in G.nodes()]
else:
    node_colors = 'lightblue'

# Edge widths by collaboration strength
edge_widths = [G[u][v]['collaboration_strength'] * 5
               for u, v in G.edges()]

# Draw
plt.figure(figsize=(14, 10))
nx.draw_networkx_nodes(G, pos, node_size=node_sizes,
                       node_color=node_colors, cmap='tab10', alpha=0.7)
nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.3)
nx.draw_networkx_labels(G, pos, font_size=8)

plt.title('Author Collaboration Network')
plt.axis('off')
plt.tight_layout()
plt.show()
```

**Analyze Collaboration Patterns:**

```python
# Strong collaborations (>0.6 strength)
strong_collabs = [(edge['source'], edge['target'], edge['collaboration_strength'])
                  for edge in network['edges']
                  if edge['collaboration_strength'] > 0.6]

print(f"Strong collaborations: {len(strong_collabs)}")
for source, target, strength in strong_collabs:
    print(f"{source} ↔ {target}: {strength:.2f}")

# Isolated authors (no collaborations)
all_authors = {node['author_id'] for node in network['nodes']}
collaborating_authors = {edge['source'] for edge in network['edges']}.union(
                        {edge['target'] for edge in network['edges']})
isolated = all_authors - collaborating_authors

if isolated:
    print(f"\nIsolated authors (no collaborations): {len(isolated)}")
    print(list(isolated))
```

---

### 4.5 Co-change Network

**File:** `networks/cochange_network.json`

File coupling relationships based on commit co-occurrence patterns.

#### 4.5.1 Network Schema

```json
{
  "metadata": {
    "generated_at": "2024-11-15T10:32:00+00:00",
    "total_files": 342,
    "total_couplings": 1247,
    "min_coupling_threshold": 0.3,
    "max_pairs_analyzed": 10000
  },
  "nodes": [
    {
      "file_path": "src/core/engine.py",
      "extension": "py",
      "total_commits": 87,
      "unique_authors": 5,
      "total_churn": 5310,
      "coupling_degree": 12
    }
  ],
  "edges": [
    {
      "source": "src/core/engine.py",
      "target": "src/api/handlers.py",
      "coupling_score": 0.72,
      "commits_together": 45,
      "total_commits_source": 87,
      "total_commits_target": 62,
      "first_cochange": "2023-02-10T11:30:00+00:00",
      "last_cochange": "2024-11-10T16:45:00+00:00",
      "cochange_authors": ["jane_developer_jane", "john_smith_john"]
    }
  ]
}
```

#### 4.5.2 Understanding Co-change

**What Triggers a Co-change:**
Two files are "co-changed" when they are modified in the same commit.

**Example:**

```bash
Commit abc123: Modified files
  - src/core/engine.py
  - src/api/handlers.py
  - tests/test_engine.py

Co-change pairs formed:
  (engine.py, handlers.py)
  (engine.py, test_engine.py)
  (handlers.py, test_engine.py)
```

**Coupling vs. Correlation:**

| Aspect      | Coupling                  | Correlation                  |
| ----------- | ------------------------- | ---------------------------- |
| Definition  | Frequency of co-changes   | Statistical relationship     |
| Causality   | May imply dependency      | Does not imply causation     |
| Measurement | Co-change count           | Pearson/Spearman coefficient |
| Use Case    | Refactoring, architecture | Pattern detection            |

**Temporal Windows:**
Strata.py uses **commit-based windows**: files changed in the same commit are coupled. Alternative approaches might use time windows (e.g., changes within 1 hour).

#### 4.5.3 Coupling Score Calculation

**Formula:**

$$\text{CouplingScore}(A, B) = \frac{\text{commits\_together}(A, B)}{\min(\text{commits}(A), \text{commits}(B))}$$

**Properties:**

- Range: [0, 1]
- Symmetric: Coupling(A, B) = Coupling(B, A)
- 1.0 = perfect coupling (always changed together)
- 0.0 = never changed together

**Normalization Rationale:**
Dividing by the minimum prevents bias toward files with many commits.

**Example:**

```python
File A: 100 commits
File B: 50 commits
Together: 30 commits

Coupling = 30 / min(100, 50) = 30 / 50 = 0.6
```

If we used max or sum, the score would be lower and less meaningful.

**Interpreting Scores:**

| Score Range | Strength    | Architectural Implication |
| ----------- | ----------- | ------------------------- |
| 0.9-1.0     | Perfect     | Should probably be merged |
| 0.7-0.89    | Very Strong | Tight logical coupling    |
| 0.5-0.69    | Strong      | Consider refactoring      |
| 0.3-0.49    | Moderate    | Monitor for patterns      |
| 0.1-0.29    | Weak        | Occasional co-change      |
| <0.1        | Negligible  | Independent files         |

**Thresholds for Analysis:**

```python
# Filter edges by coupling strength
strong_couplings = [edge for edge in network['edges']
                   if edge['coupling_score'] >= 0.6]

very_strong = [edge for edge in network['edges']
              if edge['coupling_score'] >= 0.8]
```

#### 4.5.4 Practical Applications

**Find Tightly Coupled Files:**

```python
import json
import pandas as pd

# Load co-change network
with open('output/networks/cochange_network.json', 'r') as f:
    cochange = json.load(f)

edges_df = pd.DataFrame(cochange['edges'])

# Top coupled pairs
top_couplings = edges_df.nlargest(20, 'coupling_score')[
    ['source', 'target', 'coupling_score', 'commits_together']
]

print("Top 20 tightly coupled file pairs:")
print(top_couplings)

# Files with highest coupling degree
nodes_df = pd.DataFrame(cochange['nodes'])
high_degree = nodes_df.nlargest(10, 'coupling_degree')[
    ['file_path', 'coupling_degree', 'total_commits']
]

print("\nFiles with most couplings:")
print(high_degree)
```

**Detect Architectural Boundaries:**

```python
# Group files by directory
def extract_module(filepath):
    parts = filepath.split('/')
    return parts[0] if len(parts) > 1 else 'root'

edges_df['source_module'] = edges_df['source'].apply(extract_module)
edges_df['target_module'] = edges_df['target'].apply(extract_module)

# Cross-module couplings
cross_module = edges_df[edges_df['source_module'] != edges_df['target_module']]

print(f"Cross-module couplings: {len(cross_module)}")
print("\nTop cross-module coupling:")
print(cross_module.nlargest(10, 'coupling_score')[
    ['source', 'target', 'coupling_score']
])

# Intra-module coupling strength
intra_module = edges_df[edges_df['source_module'] == edges_df['target_module']]
print(f"\nIntra-module couplings: {len(intra_module)}")
print(f"Average coupling strength: {intra_module['coupling_score'].mean():.2f}")
```

**Identify Refactoring Candidates:**

```python
# High coupling + high churn = refactoring opportunity
nodes_dict = {node['file_path']: node for node in cochange['nodes']}

refactor_candidates = []
for edge in cochange['edges']:
    if edge['coupling_score'] > 0.7:  # Strong coupling
        source_churn = nodes_dict[edge['source']]['total_churn']
        target_churn = nodes_dict[edge['target']]['total_churn']

        if source_churn > 2000 or target_churn > 2000:  # High churn
            refactor_candidates.append({
                'file_a': edge['source'],
                'file_b': edge['target'],
                'coupling': edge['coupling_score'],
                'churn_a': source_churn,
                'churn_b': target_churn,
                'risk_score': edge['coupling_score'] * (source_churn + target_churn)
            })

refactor_df = pd.DataFrame(refactor_candidates)
refactor_df = refactor_df.sort_values('risk_score', ascending=False)

print("Top refactoring candidates (high coupling + high churn):")
print(refactor_df.head(10))
```

**Build Dependency Graphs:**

```python
import networkx as nx
import matplotlib.pyplot as plt

# Create directed graph (treat as undirected for co-change)
G = nx.Graph()

# Add nodes
for node in cochange['nodes']:
    G.add_node(node['file_path'],
               commits=node['total_commits'],
               churn=node['total_churn'])

# Add edges (filter by threshold)
threshold = 0.5
for edge in cochange['edges']:
    if edge['coupling_score'] >= threshold:
        G.add_edge(edge['source'], edge['target'],
                   weight=edge['coupling_score'])

print(f"Coupling graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

# Find connected components (modules/clusters)
components = list(nx.connected_components(G))
print(f"Connected components: {len(components)}")

for i, comp in enumerate(sorted(components, key=len, reverse=True)[:5], 1):
    print(f"\nComponent {i} ({len(comp)} files):")
    print(list(comp)[:10])  # Show first 10 files

# Visualize largest component
largest_comp = max(components, key=len)
subG = G.subgraph(largest_comp)

pos = nx.spring_layout(subG, k=0.3, iterations=50)
plt.figure(figsize=(16, 12))

# Node sizes by churn
node_sizes = [subG.nodes[node]['churn'] / 50 for node in subG.nodes()]

# Edge widths by coupling
edge_widths = [subG[u][v]['weight'] * 3 for u, v in subG.edges()]

nx.draw_networkx_nodes(subG, pos, node_size=node_sizes, alpha=0.6)
nx.draw_networkx_edges(subG, pos, width=edge_widths, alpha=0.3)
nx.draw_networkx_labels(subG, pos, font_size=6)

plt.title('File Coupling Network (Largest Component)')
plt.axis('off')
plt.tight_layout()
plt.show()
```

---

This completes Part 2 of the Strata.py Dataset Reference Guide.

**Next Parts:**

- Part 3: Advanced Datasets and Workflows (Sections 4.6-5)
- Part 4: Advanced Topics, Troubleshooting, and Reference (Sections 6-8)
