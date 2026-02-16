# Strata.py Dataset Reference Guide - Part 1

## Introduction, Core Concepts, and Metrics Glossary

---

## 1. Introduction

### 1.1 Overview

Strata.py (version 2.2.0) is a comprehensive Git repository evolution analyzer that transforms your repository's history into structured JSON datasets for visualization, analysis, and insight generation. The tool tracks file lifecycles, measures code churn, analyzes collaboration patterns, detects coupling relationships, and identifies architectural patterns—all from your Git history.

**What Strata.py Does:**

- Extracts complete file lifecycle histories with create/modify/rename/delete events
- Calculates health metrics for files and directories
- Maps author collaboration networks and contribution patterns
- Identifies co-change relationships between files (logical coupling)
- Generates temporal aggregations for time-series analysis
- Produces frontend-ready datasets for visualization dashboards

**Key Features:**

- **Performance-optimized** with quick mode for large repositories
- **Rich temporal metadata** including quarters, ISO weeks, and position metrics
- **Author normalization** with domain extraction and ID generation
- **Diff statistics** tracking lines added/deleted per commit
- **Frontend integration** with treemap hierarchies, metrics indexes, and activity heatmaps
- **Backward compatible** with comprehensive error handling

### 1.2 Quick Start: Running the Analysis

**Basic Usage:**

```bash
python strata.py /path/to/your/repo
```

This creates an `output/` directory containing all generated datasets.

**Common Options:**

```bash
# Specify custom output directory
python strata.py /path/to/repo --output my_analysis

# Quick mode for large repositories (samples commits)
python strata.py /path/to/repo --quick --sample-rate 0.5

# Enable specific phases only
python strata.py /path/to/repo --phase1 --phase2

# Disable specific aggregators
python strata.py /path/to/repo --skip-cochange --skip-milestones

# Verbose output with profiling
python strata.py /path/to/repo --verbose --profile
```

**Using Presets:**

```bash
# Fast analysis (minimal datasets)
python strata.py /path/to/repo --preset fast

# Full analysis (all datasets)
python strata.py /path/to/repo --preset full
```

**Expected Runtime:**

- Small repos (<1,000 commits): seconds to minutes
- Medium repos (1,000-10,000 commits): minutes
- Large repos (>10,000 commits): minutes to hours (use `--quick` mode)

### 1.3 Output Structure and File Organization

After running the analysis, you'll find the following directory structure:

```
output/
├── file_lifecycle.json              # Core lifecycle dataset
├── file_lifecycle_errors.txt        # Error log (if any)
├── manifest.json                    # Metadata about all datasets
├── metadata_report.txt              # Human-readable summary
│
├── metadata/
│   └── file_index.json              # File-level metadata index
│
├── aggregations/
│   ├── temporal_daily.json          # Daily activity aggregations
│   ├── temporal_monthly.json        # Monthly activity aggregations
│   └── directory_stats.json         # Directory-level statistics
│
├── networks/
│   ├── author_network.json          # Author collaboration network
│   └── cochange_network.json        # File coupling network
│
├── milestones/
│   └── release_snapshots.json       # Version/release snapshots
│
└── frontend/
    ├── project_hierarchy.json       # Treemap-ready hierarchy
    ├── file_metrics_index.json      # Detailed file metrics
    └── temporal_activity_map.json   # Weekly activity heatmap
```

**Dataset Phases:**

- **Phase 1**: Core lifecycle and file metadata
- **Phase 2**: Temporal aggregations and author networks
- **Phase 3**: Co-change networks, hierarchies, and milestones
- **Phase 6**: Frontend-optimized datasets (hierarchy, metrics, activity maps)

### 1.4 Prerequisites and Setup

**Required Python Version:**

- Python 3.7 or higher

**Required Dependencies:**

```bash
pip install click
```

**Recommended Dependencies** (for enhanced features):

```bash
pip install tqdm colorama pandas numpy networkx matplotlib
```

**Optional Dependencies:**

```bash
# For YAML configuration files
pip install pyyaml

# For memory profiling
pip install memory-profiler psutil
```

**Git Requirements:**

- Git must be installed and accessible from command line
- Repository must have at least one commit
- Proper file permissions to read repository

**Validation:**

```python
# Test your setup
python -c "import click; print('Dependencies OK')"
```

---

## 2. Core Concepts and Terminology

### 2.1 What is File Lifecycle?

A **file lifecycle** is the complete history of a single file from creation to deletion (or present day) as recorded in Git commits. Each file's lifecycle consists of a chronological sequence of **events** that track how the file evolved.

**Lifecycle Components:**

- **Birth**: The commit where the file first appeared
- **Evolution**: All modification events throughout the file's existence
- **Transformations**: Rename and move operations
- **Death**: The commit where the file was deleted (if applicable)

**Example Lifecycle Sequence:**

```
File: src/utils/parser.py
1. Create    (commit abc123) - File introduced with 150 lines
2. Modify    (commit def456) - Refactored, +20 / -15 lines
3. Modify    (commit ghi789) - Bug fix, +5 / -3 lines
4. Rename    (commit jkl012) - Moved to src/core/parser.py
5. Modify    (commit mno345) - Feature addition, +80 / -10 lines
6. Delete    (commit pqr678) - File removed during restructure
```

**Why Lifecycle Matters:**

- Reveals file stability and volatility patterns
- Identifies high-churn files needing attention
- Tracks ownership and knowledge distribution
- Enables temporal analysis of development patterns
- Detects architectural changes through rename patterns

### 2.2 Understanding Git Events (create, modify, rename, delete)

Strata.py classifies every file change into one of four event types:

#### Create Events

**Definition**: First appearance of a file in repository history.

**Characteristics:**

- Marks the file's birth timestamp
- Initial `lines_added` count (no `lines_deleted`)
- Establishes the original author
- May include initial file path

**Example Event:**

```json
{
  "event_type": "create",
  "commit_hash": "a1b2c3d4",
  "timestamp": "2024-01-15T10:30:00+00:00",
  "author_name": "Jane Developer",
  "author_email": "jane@example.com",
  "lines_added": 200,
  "lines_deleted": 0,
  "message": "Initial implementation of parser module"
}
```

#### Modify Events

**Definition**: Any change to a file's content (additions, deletions, or edits).

**Characteristics:**

- Most common event type
- Both `lines_added` and `lines_deleted` typically present
- Tracks granular code churn
- May represent features, fixes, or refactoring

**Churn Calculation:**
Total churn for a modify event = `lines_added + lines_deleted`

**Example Event:**

```json
{
  "event_type": "modify",
  "commit_hash": "e5f6g7h8",
  "timestamp": "2024-02-20T14:45:00+00:00",
  "author_name": "John Smith",
  "author_email": "john@example.com",
  "lines_added": 35,
  "lines_deleted": 12,
  "message": "Optimize parsing algorithm"
}
```

#### Rename Events

**Definition**: File path change (move or rename) without content modification.

**Characteristics:**

- Preserves file identity across paths
- Important for tracking architectural refactoring
- `old_path` and `new_path` recorded
- May or may not include content changes

**Git Detection:**
Git's rename detection considers a file "renamed" if:

- > 50% content similarity (default threshold)
- Path change detected

**Example Event:**

```json
{
  "event_type": "rename",
  "commit_hash": "i9j0k1l2",
  "timestamp": "2024-03-10T09:15:00+00:00",
  "old_path": "utils/parser.py",
  "new_path": "core/parser.py",
  "author_name": "Jane Developer",
  "author_email": "jane@example.com",
  "lines_added": 0,
  "lines_deleted": 0,
  "message": "Restructure: move parser to core module"
}
```

#### Delete Events

**Definition**: File removal from the repository.

**Characteristics:**

- Marks file's end-of-life timestamp
- Total lines deleted equals final file size
- `lines_added` is zero
- File becomes "inactive" but lifecycle remains in history

**Example Event:**

```json
{
  "event_type": "delete",
  "commit_hash": "m3n4o5p6",
  "timestamp": "2024-04-05T16:00:00+00:00",
  "author_name": "Sarah Johnson",
  "author_email": "sarah@example.com",
  "lines_added": 0,
  "lines_deleted": 247,
  "message": "Remove deprecated parser implementation"
}
```

### 2.3 Temporal Concepts

#### Commit Timestamps vs. Author Dates

Git stores two timestamps for each commit:

1. **Author Date**: When changes were originally made
2. **Commit Date**: When changes were committed to repository

**Strata.py uses Author Date** for temporal analysis because it represents when work actually occurred, not when it was merged or pushed.

**Why This Matters:**

```
Scenario: Developer works offline Thursday, pushes Monday
- Author Date: Thursday (actual work time)
- Commit Date: Monday (push time)
- Strata.py records: Thursday
```

#### ISO Week Numbering

Strata.py uses **ISO 8601 week date system**:

**Rules:**

- Week starts on Monday (day 1), ends on Sunday (day 7)
- Week 1 is the first week with Thursday in the new year
- Weeks can span year boundaries

**Example:**

```
2024-01-01 (Monday) → Week 1 of 2024
2024-12-30 (Monday) → Week 1 of 2025 (contains 2025-01-02, a Thursday)
```

**Accessing Week Data:**

```python
temporal_labels = {
    "year": 2024,
    "week_no": 15,
    "day_of_week": 2  # 0=Monday, 2=Wednesday
}
```

**Use Cases:**

- Sprint planning (2-week sprints align with ISO weeks)
- Identifying weekly development patterns
- Comparing activity across consistent week boundaries

#### Quarters and Fiscal Periods

**Calendar Quarters:**

- Q1: January-March (months 1-3)
- Q2: April-June (months 4-6)
- Q3: July-September (months 7-9)
- Q4: October-December (months 10-12)

**Formula:**

```python
quarter = ((month - 1) // 3) + 1
```

**Accessing Quarter Data:**

```json
{
  "year": 2024,
  "quarter": 2,
  "month": 5
}
```

**Note on Fiscal Years:**
Strata.py uses calendar years by default. For fiscal year analysis, you'll need to post-process the temporal data based on your organization's fiscal calendar.

### 2.4 Author Identity and Normalization

#### Author IDs vs. Email Addresses

Git records authors as `name <email>` pairs, but the same person may appear with multiple identities:

**Common Variations:**

```
Jane Smith <jane.smith@company.com>
Jane Smith <jane@personal.com>
Jane Smith <jsmith@company.com>
Jane M. Smith <jane.smith@company.com>
jsmith <jane.smith@company.com>
```

**Strata.py's Normalization:**
Creates consistent `author_id` from name and email:

**Algorithm:**

1. Lowercase and normalize name: `jane_smith`
2. Extract email prefix (before @): `jane_smith` or `jsmith`
3. Remove special characters and Gmail + tags
4. Combine: `jane_smith_jane_smith` or `jane_smith_jsmith`
5. Truncate to 64 characters if needed

**Result:**

```python
{
  "author_id": "jane_smith_jsmith",
  "author_domain": "company.com"
}
```

#### Domain Extraction

**Email Domain** extracted as organization identifier:

**Examples:**

```python
"jane@company.com"     → "company.com"
"john@gmail.com"       → "gmail.com"
"bot@users.noreply.github.com" → "users.noreply.github.com"
```

**Use Cases:**

- Identify external contributors
- Analyze cross-organization collaboration
- Detect bot accounts
- Calculate organizational diversity

#### Handling Email Aliases

**Gmail + Tags:**

```
jane+work@gmail.com → Normalized to: jane@gmail.com
jane+feature-branch@gmail.com → Normalized to: jane@gmail.com
```

**Corporate Aliases:**

```
jane.smith@company.com
jsmith@company.com
j.smith@company.com
```

**Limitation:** Strata.py cannot automatically merge different email addresses for the same person. You may need to manually consolidate author statistics in post-processing if high precision is required.

**Best Practice:**
For the most accurate author analysis, encourage consistent Git configuration in your team:

```bash
git config --global user.name "Jane Smith"
git config --global user.email "jane.smith@company.com"
```

---

## 3. Metrics Glossary

### 3.1 File-Level Metrics

#### Churn

**Definition:** The total amount of code change in a file over time, measured by lines added and deleted.

**Mathematical Definition:**

For a file with $n$ commits, total churn is:

$$\text{Churn} = \sum_{i=1}^{n} (\text{lines_added}_i + \text{lines_deleted}_i)$$

**Types of Churn:**

1. **Absolute Churn** (raw sum):

   ```python
   absolute_churn = sum(event.lines_added + event.lines_deleted
                       for event in file_lifecycle)
   ```

2. **Net Change** (additions minus deletions):

   ```python
   net_change = sum(event.lines_added - event.lines_deleted
                   for event in file_lifecycle)
   ```

3. **Normalized Churn** (per commit):
   ```python
   normalized_churn = absolute_churn / commit_count
   ```

**Interpretation:**

| Churn Level       | Description                  | Action                 |
| ----------------- | ---------------------------- | ---------------------- |
| Low (<500)        | Stable file, minimal changes | Monitor as usual       |
| Medium (500-2000) | Normal evolution             | Review for patterns    |
| High (2000-5000)  | Active development area      | Increase test coverage |
| Very High (>5000) | Potential hotspot            | Refactor candidate     |

**Example Calculation:**

```python
File: src/core/engine.py
Commit 1: +100, -0    → Churn: 100
Commit 2: +20, -15    → Churn: 35
Commit 3: +50, -30    → Churn: 80
Commit 4: +10, -10    → Churn: 20
Total Churn: 235
Average per commit: 58.75
```

**Why Churn Matters:**

- High churn may indicate:
  - Complex or unclear requirements
  - Technical debt accumulation
  - Poor initial design
  - Active feature development
- Low churn suggests:
  - Stable, well-understood code
  - Potential neglect (if also low test coverage)
  - Simple or mature functionality

#### File Health

**Definition:** A composite score (0-100) representing overall file quality based on multiple factors.

**Components:**

1. **Stability** (40% weight): Inverse of churn rate
2. **Bus Factor** (30% weight): Author diversity
3. **Coupling** (20% weight): Inverse of co-change frequency
4. **Age** (10% weight): File maturity bonus

**Calculation Formula:**

$$\text{Health} = w_s \cdot S + w_b \cdot B + w_c \cdot C + w_a \cdot A$$

Where:

- $S$ = Stability score (0-100)
- $B$ = Bus factor score (0-100)
- $C$ = Coupling score (0-100)
- $A$ = Age score (0-100)
- $w_i$ = component weights

**Detailed Calculations:**

**Stability Score:**

```python
def calculate_stability(churn, commits):
    churn_per_commit = churn / max(commits, 1)
    # Normalize to 0-100 scale (lower churn = higher stability)
    stability = max(0, 100 - (churn_per_commit / 10))
    return min(100, stability)
```

**Bus Factor Score:**

```python
def calculate_bus_factor_score(unique_authors):
    # More authors = better score
    if unique_authors >= 5:
        return 100
    elif unique_authors >= 3:
        return 75
    elif unique_authors == 2:
        return 50
    else:
        return 25  # Single author = risk
```

**Coupling Score:**

```python
def calculate_coupling_score(coupling_strength):
    # Lower coupling = higher score
    # coupling_strength: 0.0 (none) to 1.0 (always together)
    return int((1 - coupling_strength) * 100)
```

**Age Score:**

```python
def calculate_age_score(days_old):
    # Bonus for mature files (plateaus at 180 days)
    return min(100, (days_old / 180) * 100)
```

**Example:**

```python
File: src/utils/validator.py
- Churn: 450, Commits: 30 → Churn/commit: 15 → Stability: 85
- Authors: 4 → Bus factor: 75
- Coupling: 0.2 → Coupling score: 80
- Age: 150 days → Age score: 83

Health = 0.4(85) + 0.3(75) + 0.2(80) + 0.1(83)
       = 34 + 22.5 + 16 + 8.3
       = 80.8
```

**Health Score Interpretation:**

| Range  | Status    | Color Code  | Action                |
| ------ | --------- | ----------- | --------------------- |
| 90-100 | Excellent | 🟢 Green    | Maintain practices    |
| 70-89  | Good      | 🟡 Yellow   | Monitor               |
| 50-69  | Fair      | 🟠 Orange   | Consider improvements |
| 30-49  | Poor      | 🔴 Red      | Refactor recommended  |
| 0-29   | Critical  | 🔴 Dark Red | Urgent attention      |

#### Touch Count

**Definition:** The number of commits that modified a file.

**Formula:**
$$\text{TouchCount} = |\{\text{commit} \mid \text{file modified in commit}\}|$$

**Important Distinction:**

- **Touch Count** ≠ **Modify Event Count**
- A single commit can include multiple modify operations (e.g., rename + modify)
- Touch count = unique commits affecting the file

**Calculation:**

```python
touch_count = len(set(event.commit_hash for event in file_lifecycle))
```

**Example:**

```
Commit abc: create file
Commit def: modify
Commit ghi: rename + modify (counts as 1 touch)
Commit jkl: modify
Commit mno: modify
Touch Count: 5
```

**Use Cases:**

- Identify frequently touched files (potential hotspots)
- Measure file volatility
- Combined with churn for comprehensive activity metric:
  ```python
  activity_score = touch_count * average_churn_per_commit
  ```

**Interpretation:**

| Touch Count | File Characteristic    |
| ----------- | ---------------------- |
| 1-5         | New or rarely modified |
| 6-20        | Normal evolution       |
| 21-50       | Active development     |
| 51-100      | Hotspot file           |
| >100        | Critical hotspot       |

#### Age and Lifespan

**Definitions:**

**Creation Date:** Timestamp of first commit introducing the file

```python
creation_date = file_lifecycle[0].timestamp  # First event
```

**Last Modified Date:** Timestamp of most recent commit affecting the file

```python
last_modified = file_lifecycle[-1].timestamp  # Last event
```

**Age:** Time elapsed from creation to now (or deletion)

```python
from datetime import datetime
age_days = (datetime.now() - creation_date).days
```

**Active Lifespan:** Time between first and last modification

```python
active_lifespan_days = (last_modified - creation_date).days
```

**Total Lifespan:** Age including dormant periods

- For active files: Age = Active Lifespan
- For deleted files: Total Lifespan = deletion_date - creation_date

**Example:**

```python
File: src/old_module.py
Created: 2022-01-15
Last Modified: 2023-06-20
Deleted: 2024-03-10

Age (from creation to deletion): 785 days
Active Lifespan: 521 days (creation to last modification)
Dormant Period: 264 days (last modification to deletion)
```

**Lifespan Metrics:**

1. **Activity Ratio:**
   $$\text{ActivityRatio} = \frac{\text{ActiveLifespan}}{\text{TotalLifespan}}$$

   High ratio (>0.8) = consistently active
   Low ratio (<0.3) = abandoned or completed

2. **Average Time Between Touches:**
   $$\text{AvgTimeBetweenTouches} = \frac{\text{ActiveLifespan}}{\text{TouchCount} - 1}$$

**Use Cases:**

- Identify stale code (old files, no recent touches)
- Track feature completion (files with early lifespan, now dormant)
- Find legacy code candidates (old + low activity)

#### Stability Metrics

**Definition:** Measures of how frequently and predictably a file changes.

**1. Change Frequency:**
$$\text{ChangeFrequency} = \frac{\text{TouchCount}}{\text{ActiveLifespanDays}}$$

Units: touches per day

**Interpretation:**

- High frequency (>0.5): Very volatile, changing multiple times per day
- Medium (0.1-0.5): Active development
- Low (0.01-0.1): Occasional updates
- Very low (<0.01): Stable/dormant

**2. Time Between Modifications (TBM):**

Mean TBM:
$$\text{MeanTBM} = \frac{1}{\text{ChangeFrequency}}$$

Standard Deviation of TBM:

```python
import numpy as np
timestamps = [event.timestamp for event in file_lifecycle]
time_deltas = np.diff(timestamps)  # Time between consecutive events
std_tbm = np.std(time_deltas)
```

**3. Stability Score:**

$$\text{StabilityScore} = 100 \times \frac{1}{1 + \text{ChangeFrequency}}$$

**Example:**

```python
File: src/config.py
Touch Count: 45
Active Lifespan: 300 days

Change Frequency: 45 / 300 = 0.15 touches/day
Mean TBM: 6.67 days
Stability Score: 100 / (1 + 0.15) = 86.96
```

**High Stability:**

- Predictable change patterns
- Long intervals between changes
- Suitable for: configuration, constants, utilities

**Low Stability:**

- Frequent, irregular changes
- Short intervals between changes
- May indicate: active feature work, technical debt, unclear requirements

---

### 3.2 Author Metrics

#### Bus Factor

**Definition:** The minimum number of contributors whose departure would significantly compromise the project's ability to continue.

**Formal Definition:**
The smallest set of contributors whose cumulative contribution exceeds 50% of the total work.

**Mathematical Formula:**

Let $C = \{c_1, c_2, ..., c_n\}$ be contributors sorted by contribution (descending).

$$\text{BusFactor} = \min\{k \mid \sum_{i=1}^{k} \text{contribution}(c_i) > 0.5 \times \text{TotalContribution}\}$$

**Calculation Algorithm:**

```python
def calculate_bus_factor(contributions):
    """
    contributions: dict mapping author_id to contribution count
    """
    total = sum(contributions.values())
    sorted_contribs = sorted(contributions.values(), reverse=True)

    cumsum = 0
    for i, contrib in enumerate(sorted_contribs, start=1):
        cumsum += contrib
        if cumsum > total / 2:
            return i
    return len(contributions)
```

**Example:**

```python
File: src/critical_module.py
Author A: 120 commits (60%)
Author B: 40 commits (20%)
Author C: 30 commits (15%)
Author D: 10 commits (5%)
Total: 200 commits

Cumulative:
A: 120 (60%) → exceeds 50%
Bus Factor: 1 ⚠️ CRITICAL
```

**Another Example:**

```python
File: src/shared_utils.py
Author A: 80 commits (32%)
Author B: 70 commits (28%)
Author C: 60 commits (24%)
Author D: 40 commits (16%)
Total: 250 commits

Cumulative:
A: 80 (32%)
A+B: 150 (60%) → exceeds 50%
Bus Factor: 2 ✓ Better
```

**Thresholds and Risk Levels:**

| Bus Factor | Risk Level  | Recommended Action                                            |
| ---------- | ----------- | ------------------------------------------------------------- |
| 1          | 🔴 Critical | Immediate knowledge transfer, pair programming, documentation |
| 2          | 🟠 High     | Cross-training, code reviews, rotate responsibilities         |
| 3-4        | 🟡 Medium   | Maintain collaboration, encourage participation               |
| 5+         | 🟢 Low      | Healthy distribution, continue practices                      |

**Project-Level Bus Factor:**

```python
# Across entire codebase
def project_bus_factor(all_commits_by_author):
    total_commits = sum(all_commits_by_author.values())
    sorted_authors = sorted(all_commits_by_author.items(),
                           key=lambda x: x[1], reverse=True)

    cumsum = 0
    for i, (author, commits) in enumerate(sorted_authors, start=1):
        cumsum += commits
        if cumsum > total_commits / 2:
            return i
```

**Why Bus Factor Matters:**

- **Risk Management**: Identifies knowledge concentration
- **Succession Planning**: Highlights critical dependencies
- **Team Health**: Measures collaboration quality
- **Onboarding Priority**: Indicates where training is needed

#### Author Diversity

**Definition:** The evenness of contribution distribution among authors, measured using Shannon entropy.

**Shannon Entropy Formula:**

$$H = -\sum_{i=1}^{n} p_i \log_2(p_i)$$

Where:

- $n$ = number of unique authors
- $p_i$ = proportion of total contributions from author $i$

**Calculation:**

```python
import math

def calculate_author_diversity(contributions):
    """
    contributions: dict mapping author_id to contribution count
    Returns: Shannon entropy (0 to log2(n))
    """
    total = sum(contributions.values())
    entropy = 0

    for count in contributions.values():
        if count > 0:
            p = count / total
            entropy -= p * math.log2(p)

    return entropy
```

**Normalized Diversity (0-1 scale):**

```python
def normalized_diversity(contributions):
    n = len(contributions)
    if n <= 1:
        return 0

    entropy = calculate_author_diversity(contributions)
    max_entropy = math.log2(n)  # Perfect uniform distribution
    return entropy / max_entropy
```

**Example 1: Single Author (No Diversity)**

```python
contributions = {"alice": 100}
H = -(1.0 * log2(1.0)) = 0
Normalized: 0.0
Interpretation: No diversity, single contributor risk
```

**Example 2: Equal Distribution (Perfect Diversity)**

```python
contributions = {"alice": 25, "bob": 25, "carol": 25, "dave": 25}
p = 0.25 for all
H = -4 * (0.25 * log2(0.25)) = -4 * (0.25 * -2) = 2.0
Max H = log2(4) = 2.0
Normalized: 1.0
Interpretation: Perfect diversity, healthy collaboration
```

**Example 3: Uneven Distribution**

```python
contributions = {"alice": 70, "bob": 20, "carol": 7, "dave": 3}
p_alice = 0.70, p_bob = 0.20, p_carol = 0.07, p_dave = 0.03

H = -(0.70*log2(0.70) + 0.20*log2(0.20) + 0.07*log2(0.07) + 0.03*log2(0.03))
  = -(0.70*(-0.515) + 0.20*(-2.322) + 0.07*(-3.836) + 0.03*(-5.059))
  = -(-0.361 - 0.464 - 0.269 - 0.152)
  = 1.246

Max H = log2(4) = 2.0
Normalized: 1.246 / 2.0 = 0.623
Interpretation: Moderate diversity, some concentration
```

**Interpretation Scale:**

| Normalized Diversity | Interpretation | Team Health            |
| -------------------- | -------------- | ---------------------- |
| 0.9 - 1.0            | Excellent      | Highly collaborative   |
| 0.7 - 0.89           | Good           | Balanced contributions |
| 0.5 - 0.69           | Fair           | Some concentration     |
| 0.3 - 0.49           | Poor           | High concentration     |
| 0.0 - 0.29           | Critical       | Dominated by few       |

**Relationship to Bus Factor:**

- Low diversity (high entropy) → Low bus factor
- High diversity (low entropy) → High bus factor

**Use Cases:**

- **Team Formation**: Ensure balanced skill distribution
- **Knowledge Silos**: Identify areas with single-expert reliance
- **Code Review**: Target reviews where diversity is low
- **Mentorship**: Pair experienced authors with areas of low diversity

#### Contribution Patterns

**Primary vs. Secondary Contributors:**

**Primary Contributor:**
Author with the highest number of commits to a file.

```python
def identify_primary_contributor(contributions):
    return max(contributions.items(), key=lambda x: x[1])
```

**Secondary Contributors:**
Authors ranked by contribution, excluding primary.

**Ownership Thresholds:**

```python
def classify_contributors(contributions):
    total = sum(contributions.values())
    sorted_contribs = sorted(contributions.items(),
                            key=lambda x: x[1], reverse=True)

    classifications = {}
    for author, commits in sorted_contribs:
        percentage = (commits / total) * 100

        if percentage >= 50:
            classifications[author] = "owner"
        elif percentage >= 25:
            classifications[author] = "major_contributor"
        elif percentage >= 10:
            classifications[author] = "contributor"
        else:
            classifications[author] = "minor_contributor"

    return classifications
```

**Ownership Concentration:**

**Herfindahl-Hirschman Index (HHI):**
$$\text{HHI} = \sum_{i=1}^{n} \left(\frac{\text{commits}_i}{\text{TotalCommits}}\right)^2$$

**Interpretation:**

- HHI = 1.0: Complete monopoly (single author)
- HHI = 1/n: Perfect distribution (n equal authors)
- HHI > 0.5: High concentration
- HHI < 0.25: Low concentration

**Example:**

```python
# File 1: High concentration
{"alice": 90, "bob": 10}
HHI = (0.9)^2 + (0.1)^2 = 0.81 + 0.01 = 0.82

# File 2: Distributed
{"alice": 30, "bob": 25, "carol": 25, "dave": 20}
HHI = (0.3)^2 + (0.25)^2 + (0.25)^2 + (0.2)^2
    = 0.09 + 0.0625 + 0.0625 + 0.04 = 0.255
```

**Contribution Velocity:**
Commits per unit time by author:

```python
def author_velocity(author_commits, timespan_days):
    return len(author_commits) / timespan_days
```

**Temporal Contribution Patterns:**

- **Bursty**: High activity in short periods, long gaps
- **Steady**: Consistent contributions over time
- **Declining**: Decreasing activity (potential departures)
- **Ramping**: Increasing activity (onboarding or specialization)

---

### 3.3 Network Metrics

#### Coupling Strength

**Definition:** The degree to which two files tend to change together in the same commits, indicating logical or structural dependency.

**Formula:**

$$\text{Coupling}(A,B) = \frac{\text{commits_together}(A, B)}{\min(\text{commits}(A), \text{commits}(B))}$$

**Rationale:**
Dividing by the minimum ensures:

- Coupling ∈ [0, 1]
- Symmetric: Coupling(A,B) = Coupling(B,A)
- Normalized by opportunity

**Example Calculation:**

```python
File A: modified in commits {1, 2, 3, 5, 7, 9} → 6 commits
File B: modified in commits {2, 3, 4, 7, 8} → 5 commits
Together: {2, 3, 7} → 3 commits

Coupling = 3 / min(6, 5) = 3 / 5 = 0.6
```

**Alternative Formulas:**

1. **Jaccard Similarity:**
   $$\text{Jaccard}(A,B) = \frac{|A \cap B|}{|A \cup B|}$$

   More conservative, penalizes files with many non-shared commits.

2. **Support-Based:**
   $$\text{Support}(A,B) = \frac{\text{commits_together}}{\text{total_commits}}$$

   Measures frequency relative to entire project.

**Temporal Proximity:**
Co-changes within a time window (e.g., 1 hour):

```python
def temporal_coupling(file_a_timestamps, file_b_timestamps, window_hours=1):
    coupling_count = 0
    for ts_a in file_a_timestamps:
        for ts_b in file_b_timestamps:
            if abs((ts_a - ts_b).total_seconds()) <= window_hours * 3600:
                coupling_count += 1
                break
    return coupling_count / min(len(file_a_timestamps),
                                 len(file_b_timestamps))
```

**Coupling Strength Interpretation:**

| Coupling Score | Strength    | Interpretation                 |
| -------------- | ----------- | ------------------------------ |
| 0.8 - 1.0      | Very Strong | Almost always changed together |
| 0.6 - 0.79     | Strong      | Frequently co-modified         |
| 0.4 - 0.59     | Moderate    | Regular co-changes             |
| 0.2 - 0.39     | Weak        | Occasional co-changes          |
| 0.0 - 0.19     | Very Weak   | Rarely changed together        |

**Why Coupling Matters:**

- **Refactoring Candidates**: High coupling may indicate missing abstraction
- **Test Coverage**: Coupled files should be tested together
- **Code Review**: Changes to one require reviewing the other
- **Architectural Boundaries**: Low coupling across modules is desirable

#### Coupling Types

**1. Logical Coupling:**
Files that change together due to shared functionality, not explicit imports.

**Characteristics:**

- Detected through co-change analysis
- May span different modules or layers
- Often indicates hidden dependencies

**Example:**

```
src/api/users_controller.py
src/models/user.py
src/views/user_profile.html
```

No direct imports between controller and view, but they change together when user features are modified.

**2. Structural Coupling:**
Files with explicit dependencies (imports, includes).

**Detection:**

```python
# From import analysis (not in base strata.py)
import ast
def find_imports(filepath):
    with open(filepath) as f:
        tree = ast.parse(f.read())
    return [node.name for node in ast.walk(tree)
            if isinstance(node, ast.Import)]
```

**Combined Analysis:**

```python
# High logical + High structural = expected
# High logical + Low structural = hidden dependency
# Low logical + High structural = stable interface
```

**3. Symmetric vs. Asymmetric Dependencies:**

**Symmetric:** Files change together at similar rates

```
File A commits: 10
File B commits: 12
Together: 8
Coupling(A,B) = Coupling(B,A) = 0.8
```

**Asymmetric:** One file drives changes in another

```
File A (interface): 20 commits
File B (implementation): 40 commits
Together: 18
Coupling(A→B) = 18/20 = 0.9  # Interface drives implementation
Coupling(B→A) = 18/40 = 0.45 # Implementation less impacts interface
```

**Detecting Asymmetry:**

```python
def asymmetric_coupling(a_commits, b_commits, together):
    coupling_a_to_b = together / len(a_commits)
    coupling_b_to_a = together / len(b_commits)
    asymmetry = abs(coupling_a_to_b - coupling_b_to_a)
    return coupling_a_to_b, coupling_b_to_a, asymmetry
```

#### Network Centrality

**Definition:** Metrics quantifying a file's importance or influence within the codebase graph.

**1. Degree Centrality:**
Number of files directly coupled to a given file.

$$\text{DegreeCentrality}(f) = \frac{|\{g \mid \text{Coupling}(f,g) > \theta\}|}{n-1}$$

Where $\theta$ is coupling threshold (e.g., 0.5).

**Interpretation:**

- High degree → Hub file, many dependencies
- Low degree → Peripheral file, isolated

**2. Betweenness Centrality:**
Frequency with which a file lies on shortest paths between other files.

$$\text{Betweenness}(f) = \sum_{s \neq f \neq t} \frac{\sigma_{st}(f)}{\sigma_{st}}$$

Where:

- $\sigma_{st}$ = number of shortest paths from $s$ to $t$
- $\sigma_{st}(f)$ = number of those paths passing through $f$

**Implementation with NetworkX:**

```python
import networkx as nx

def build_coupling_graph(coupling_data, threshold=0.5):
    G = nx.Graph()
    for (file_a, file_b), coupling in coupling_data.items():
        if coupling >= threshold:
            G.add_edge(file_a, file_b, weight=coupling)
    return G

def calculate_centralities(G):
    degree_cent = nx.degree_centrality(G)
    between_cent = nx.betweenness_centrality(G, weight='weight')
    close_cent = nx.closeness_centrality(G, distance='weight')

    return {
        'degree': degree_cent,
        'betweenness': between_cent,
        'closeness': close_cent
    }
```

**3. Closeness Centrality:**
Average shortest path distance to all other files.

$$\text{Closeness}(f) = \frac{n-1}{\sum_{g \neq f} d(f,g)}$$

High closeness → File can quickly "reach" others through coupling chains.

**Hub Files vs. Peripheral Files:**

| Characteristic | Hub Files                    | Peripheral Files       |
| -------------- | ---------------------------- | ---------------------- |
| Degree         | High (>10 connections)       | Low (<3 connections)   |
| Betweenness    | High (bridge)                | Low (leaf)             |
| Risk Impact    | Changes affect many          | Isolated impact        |
| Refactoring    | Complex, risky               | Simpler, safer         |
| Examples       | Core utilities, base classes | UI components, scripts |

**Visualization:**

```python
import matplotlib.pyplot as plt

def visualize_coupling_network(G, centrality_scores):
    pos = nx.spring_layout(G, k=0.5, iterations=50)
    node_sizes = [centrality_scores[node] * 3000 for node in G.nodes()]

    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, alpha=0.7)
    nx.draw_networkx_edges(G, pos, alpha=0.3)
    nx.draw_networkx_labels(G, pos, font_size=8)
    plt.title("File Coupling Network")
    plt.axis('off')
    plt.show()
```

---

### 3.4 Temporal Metrics

#### Temporal Entropy

**Definition:** Measure of activity distribution uniformity across time periods.

**Formula (Shannon Entropy applied to time):**

$$H_{temporal} = -\sum_{t=1}^{T} p_t \log_2(p_t)$$

Where:

- $T$ = number of time periods (days, weeks, months)
- $p_t$ = proportion of total activity in period $t$

**Calculation:**

```python
def calculate_temporal_entropy(activity_by_period):
    """
    activity_by_period: dict mapping period to commit count
    """
    total = sum(activity_by_period.values())
    entropy = 0

    for count in activity_by_period.values():
        if count > 0:
            p = count / total
            entropy -= p * math.log2(p)

    return entropy
```

**Interpretation:**

**High Entropy (even distribution):**

```python
# Activity spread evenly across 8 weeks
activity = {week: 10 for week in range(8)}
H = log2(8) = 3.0
Pattern: Steady development
```

**Low Entropy (concentrated activity):**

```python
# Activity in 2 burst weeks
activity = {week: 0 for week in range(8)}
activity[2] = 40
activity[5] = 40
H ≈ 1.0
Pattern: Bursty development
```

**Use Cases:**

- **Sprint Planning**: High entropy suggests consistent velocity
- **Crunch Detection**: Low entropy may indicate deadline crunches
- **Team Health**: Very low entropy can signal burnout risk

**Normalized Temporal Entropy:**

```python
def normalized_temporal_entropy(activity_by_period):
    n = len([p for p in activity_by_period.values() if p > 0])
    if n <= 1:
        return 0

    entropy = calculate_temporal_entropy(activity_by_period)
    max_entropy = math.log2(n)
    return entropy / max_entropy
```

#### Active Periods

**Definition:** Time intervals where development activity occurred.

**Active Week/Month:**
A period with at least one commit.

**Calculation:**

```python
def identify_active_periods(timestamps, granularity='week'):
    """
    timestamps: list of datetime objects
    granularity: 'day', 'week', 'month', 'quarter'
    """
    if granularity == 'week':
        periods = {ts.isocalendar()[1] for ts in timestamps}
    elif granularity == 'month':
        periods = {(ts.year, ts.month) for ts in timestamps}
    elif granularity == 'day':
        periods = {ts.date() for ts in timestamps}

    return sorted(periods)
```

**Activity Ratio:**
$$\text{ActivityRatio} = \frac{\text{ActivePeriods}}{\text{TotalPeriods}}$$

**Example:**

```python
Project Duration: 24 weeks
Active Weeks: 18
Gaps: 6 weeks (no commits)

Activity Ratio: 18 / 24 = 0.75 (75% active)
```

**Gap Analysis:**

```python
def find_dormancy_periods(active_periods, threshold_days=14):
    """Find gaps longer than threshold"""
    gaps = []
    for i in range(len(active_periods) - 1):
        current = active_periods[i]
        next_period = active_periods[i + 1]
        gap_days = (next_period - current).days

        if gap_days > threshold_days:
            gaps.append({
                'start': current,
                'end': next_period,
                'duration_days': gap_days
            })

    return gaps
```

**Longest Dormancy:**
Maximum gap between commits:

```python
longest_gap = max((gaps[i]['duration_days'] for i in range(len(gaps))),
                  default=0)
```

**Use Cases:**

- **Abandoned Projects**: Long gaps may indicate neglect
- **Seasonal Patterns**: Identify vacation periods or quarterly cycles
- **Feature Completion**: Gaps after intense activity suggest milestones
- **Reactivation Events**: Detect when dormant files become active again

#### Velocity Metrics

**Definition:** Rate of development activity over time.

**1. Commit Velocity:**
Commits per unit time.

$$v_{commits} = \frac{\text{CommitCount}}{\text{TimeSpan}}$$

**Example:**

```python
Sprint 1 (2 weeks): 45 commits → 22.5 commits/week
Sprint 2 (2 weeks): 38 commits → 19 commits/week
Sprint 3 (2 weeks): 52 commits → 26 commits/week
```

**2. Churn Velocity:**
Lines changed per unit time.

$$v_{churn} = \frac{\sum(\text{lines_added} + \text{lines_deleted})}{\text{TimeSpan}}$$

**3. File Velocity:**
Files modified per unit time.

$$v_{files} = \frac{|\{\text{unique files modified}\}|}{\text{TimeSpan}}$$

**Acceleration:**
Change in velocity over time.

$$a = \frac{v_2 - v_1}{\Delta t}$$

**Example:**

```python
Month 1: 100 commits/month
Month 2: 120 commits/month  → acceleration = +20
Month 3: 140 commits/month  → acceleration = +20 (constant growth)
Month 4: 135 commits/month  → acceleration = -5 (deceleration)
```

**Moving Average Velocity:**

```python
def calculate_moving_average_velocity(commits_per_period, window=3):
    """
    commits_per_period: list of commit counts per time period
    window: number of periods for moving average
    """
    if len(commits_per_period) < window:
        return commits_per_period

    moving_avg = []
    for i in range(len(commits_per_period) - window + 1):
        window_avg = sum(commits_per_period[i:i+window]) / window
        moving_avg.append(window_avg)

    return moving_avg
```

**Velocity Trends:**

```python
from scipy import stats

def detect_velocity_trend(velocity_over_time):
    """
    Returns: slope, p-value
    - Positive slope: increasing velocity
    - Negative slope: decreasing velocity
    - p-value < 0.05: statistically significant trend
    """
    x = range(len(velocity_over_time))
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, velocity_over_time)
    return slope, p_value
```

**Interpretation:**

| Velocity Pattern | Interpretation       | Possible Causes                 |
| ---------------- | -------------------- | ------------------------------- |
| Increasing       | Project acceleration | Team growth, momentum           |
| Steady           | Mature velocity      | Sustainable pace                |
| Decreasing       | Slowing down         | Technical debt, departures      |
| Volatile         | Unstable             | Poor planning, external factors |

**Optimal Velocity:**
Not too fast (indicates rushing, quality issues) and not too slow (indicates obstacles). Aim for steady, sustainable pace.

---

This completes Part 1 of the Strata.py Dataset Reference Guide.

**Next Parts:**

- Part 2: Core Datasets Reference (Sections 4.1-4.5)
- Part 3: Advanced Datasets and Workflows (Sections 4.6-5)
- Part 4: Advanced Topics, Troubleshooting, and Reference (Sections 6-8)
