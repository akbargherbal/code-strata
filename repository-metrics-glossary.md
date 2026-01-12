# Repository Evolution Metrics — Glossary

A comprehensive reference guide for understanding Git repository analytics, development patterns, and code health indicators.

---

## Core Metrics

### Change Frequency
The number of times a file has been modified (committed) within a given time period. High change frequency may indicate either critical infrastructure that requires constant attention or configuration files that are frequently tweaked.

**Related concepts:** Commit count, modification rate, touch frequency

### Churn
The total volume of code change in a file, measured as the sum of lines added and lines removed (|additions| + |deletions|). A file with +40/-40 lines has a churn of 80, even though the net change is zero. High churn indicates significant work invested, regardless of growth.

**Formula:** `Churn = Lines Added + Lines Removed`

**Example:** +120 lines, -80 lines = 200 churn

### Net Change
The difference between lines added and lines removed (additions - deletions). Positive values indicate code growth, negative values indicate code reduction. A file can have high churn but low net change during refactoring.

**Formula:** `Net Change = Lines Added - Lines Removed`

**Example:** +120 lines, -80 lines = +40 net change

### Bus Factor
The minimum number of team members who would need to be unavailable (hit by a bus) before a project is in serious trouble due to knowledge loss. Calculated by analyzing how many files are only understood or maintained by a small number of developers.

**Risk Levels:**
- **High Risk:** Bus factor < 2 (knowledge concentrated in 1-2 people)
- **Medium Risk:** Bus factor 2-4
- **Low Risk:** Bus factor > 4 (knowledge well distributed)

---

## Activity Patterns

### Hot Files
Files with exceptionally high change frequency (typically >50 commits in the analysis period). These are often core infrastructure, API endpoints, or problematic areas requiring frequent fixes. Also called "hotspots" or "change-prone files."

**Indicators:**
- High commit count
- Multiple contributors
- Consistent activity across time periods

### Dormant Files
Files with zero or very low change frequency over an extended period. May represent stable, well-tested code OR deprecated functionality that developers avoid touching.

**Analysis considerations:**
- Location in codebase (core vs. experimental)
- Last modification date
- Current test coverage

### Cold Zones
Entire directories or modules with minimal activity. May indicate deprecated features, stable subsystems, or areas of technical debt that teams avoid.

### Activity Heatmap
A visual representation (often grid-based) where each cell represents a file, and color intensity indicates change frequency. Allows quick identification of hot files and cold zones across the entire codebase.

---

## Temporal Analysis

### Story Arc
The narrative of how a repository evolves over time, typically visualized as a timeline showing different development phases: initial growth, stabilization, major refactoring, feature additions, or decline.

**Common phases:**
1. **Creation Burst** — Rapid file creation, high growth
2. **Feature Development** — Steady additions with positive net change
3. **Stabilization** — Bug fixes, low churn, minimal growth
4. **Refactoring** — High churn, low or negative net change
5. **Maintenance** — Sporadic updates, focused on specific areas

### Creation Burst
A period where many new files are added in a short timeframe, indicating new features, modules, or major architectural additions. Often visible as spikes in file creation graphs.

### Stabilization Period
A phase where change frequency decreases but the codebase size remains relatively constant. Characterized by bug fixes, small tweaks, and polish rather than major feature development.

**Indicators:**
- Lower churn rates
- High ratio of bug fix commits
- Decreased net change
- Increased test activity

### Refactoring Event
A period of high churn with low net change across multiple files. Indicates code restructuring, optimization, or cleanup without significant feature additions.

**Characteristics:**
- High churn across many files
- Net change near zero or negative
- Often affects related files simultaneously
- May coincide with architecture updates

### Migration Pattern
A shift in development focus from one area of the codebase to another over time. For example, early focus on backend infrastructure followed by frontend development.

**Detection methods:**
- Directory-level activity analysis
- Commit message topic clustering
- Contributor focus changes

### Release Cycle
Recurring patterns of activity that align with version releases. Often shows increased activity (commits, churn) in the weeks before a release, followed by quieter periods.

---

## Collaboration Metrics

### Collaboration Topology
The network of relationships between contributors based on shared file modifications. Reveals who works together, knowledge silos, and communication patterns.

**Visualization:** Graph with nodes (contributors) and edges (shared files)

### Collaboration Intensity
The degree to which multiple contributors work on the same files. High collaboration intensity on a file suggests it's critical infrastructure or a cross-cutting concern.

**Measurement:** Number of unique contributors per file

### Knowledge Silo
A file or module maintained primarily by a single contributor or small group, creating a knowledge bottleneck and increasing project risk.

**Warning signs:**
- Single-owner files (1 contributor)
- Infrequent contributions from others
- Long time between different contributors touching the file

### Handoff Event
When the primary maintainer of a file or module changes from one contributor to another. Indicates knowledge transfer, role changes, or team transitions.

**Detection:** Significant gap in time followed by different primary contributor

### Single-Owner Files
Files that have only been modified by one contributor. These create maximum bus factor risk as knowledge is completely concentrated.

**Risk mitigation:** Code reviews, documentation, pair programming

---

## Code Health Indicators

### Test Coverage Trend
The percentage of code covered by automated tests, tracked over time. Rising coverage suggests improving code health; declining coverage may indicate technical debt accumulation.

**Healthy pattern:** Steady increase or maintenance at >70%

### Test-to-Source Ratio
The relationship between test file changes and source code changes. Ideally, when source code changes significantly, tests should change proportionally.

**Formula:** `Test Changes / Source Changes`

**Ideal ratio:** Close to 1.0 (tests change in sync with code)

### Documentation Ratio
The proportion of documentation updates relative to code changes. Documentation that "lags behind" code suggests incomplete knowledge capture.

**Calculation methods:**
- Doc commits / Total commits
- Doc file changes / Source file changes
- Lines in docs / Lines in source

### Documentation Lag
The time delay between source code changes and corresponding documentation updates. High lag indicates docs are falling out of sync with reality.

### Dependency Churn
The frequency of changes to dependency files (package.json, requirements.txt, Gemfile, etc.). High dependency churn may indicate:
- Active library updates (positive)
- Dependency instability (negative)
- Security patching (necessary)

### Velocity
The rate of development activity, typically measured as commits per unit time or lines changed per unit time. Increasing velocity suggests growing team or accelerating development; decreasing velocity may indicate stabilization or slowdown.

**Metrics:**
- Commits per week/month
- Lines changed per sprint
- Features delivered per quarter

---

## Advanced Concepts

### Displacive Summary
A summary so complete that it eliminates the need to read the original source material. In repository analysis, overly detailed summaries of code changes that essentially reproduce the diffs themselves.

**Best practice:** Provide high-level insights, not line-by-line reproduction

### Copy/Paste Detection
Identifying code duplication by tracking similar changes across multiple files. Useful for finding:
- Repeated patterns that should be abstracted
- Copy/paste errors
- Opportunities for refactoring

### Complexity Drift
The tendency for code complexity to increase over time without corresponding refactoring. Measured using metrics like cyclomatic complexity, function length, or nesting depth.

**Warning signs:**
- Increasing lines per function over time
- Growing file sizes without splitting
- Rising complexity metrics

### Abandonment/Resurrection
When a file or module goes dormant for an extended period and then suddenly becomes active again. May indicate:
- Legacy code being revisited
- Bug discovery in old code
- Feature revival
- Technical debt paydown

### Temporal Coupling
Files that are frequently modified together in the same commits, even if they're not directly related by imports or dependencies. Suggests hidden relationships or cross-cutting concerns.

**Example:** Frontend component always changes with backend API route

### Commit Message Analysis
Mining commit messages for patterns, topics, or sentiment. Can reveal:
- Types of work (features, bugs, refactoring)
- Problem areas (frequent "fix" messages)
- Team communication patterns

---

## Repository Types

### Monorepo
A single repository containing multiple projects or modules. Requires special analysis techniques to understand activity patterns within each sub-project.

**Analysis challenges:**
- Separating project-specific metrics
- Understanding cross-project dependencies
- Directory-based segmentation

### Polyrepo
Multiple repositories, each containing a single project. Activity analysis must consider the entire ecosystem, not just individual repos.

---

## Time Windows

### Analysis Period
The specific date range being examined for metrics calculation. Common periods: last 30 days, last quarter, last year, entire repository history.

### Rolling Window
A moving time period used for trend analysis. For example, a 90-day rolling window recalculates metrics for the most recent 90 days continuously.

### Snapshot
A point-in-time view of repository state. Useful for comparing repository health at different stages.

---

## Statistical Terms

### Median vs. Mean
- **Mean (Average):** Total changes / number of files — skewed by outliers (hot files)
- **Median:** The middle value — better represents "typical" file activity

**Example:** If 10% of files have 90% of changes, median gives clearer picture than mean

### Percentile
The value below which a percentage of observations fall. For example, the 90th percentile of file changes means 90% of files have fewer changes than this value.

**Use case:** "Files in the 95th percentile" identifies the most active 5%

### Standard Deviation
Measure of variability in changes across files. High standard deviation indicates some files change much more than others.

---

## Visualization Concepts

### Treemap
A hierarchical visualization where each file/directory is represented by a rectangle, with size proportional to a metric (file size, change frequency, etc.).

### Time-Series Graph
A chart showing how metrics change over time, typically with date on x-axis and metric value on y-axis.

### Network Graph
A node-and-edge diagram showing relationships between entities (contributors, files, modules).

### Heatmap
A grid where each cell's color intensity represents a metric value. Allows quick visual scanning for patterns.

---

## Best Practices

### Balanced Metrics
Using multiple metrics together to avoid misleading conclusions. For example, examining both change frequency AND contributor count to distinguish critical files from problematic ones.

### Context-Aware Analysis
Considering the type of file, location in codebase, and project phase when interpreting metrics. A configuration file with high change frequency has different implications than a core module with the same pattern.

### Longitudinal Analysis
Tracking metrics over extended periods to identify trends rather than point-in-time snapshots.

---

## Common Pitfalls

### Metric Gaming
When developers change behavior to optimize metrics rather than actual code quality. For example, making many small commits to increase commit counts.

### False Positives
Metric patterns that suggest problems but are actually benign. For example, high churn in a test data generator file.

### Correlation vs. Causation
Two metrics moving together doesn't mean one causes the other. High bug count and high change frequency may both stem from poor initial design.

---

*Last updated: January 2026*
