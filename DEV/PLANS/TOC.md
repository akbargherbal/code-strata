# Strata.py Dataset Reference Guide

## Complete Documentation for Git Repository Evolution Analysis

---

## Table of Contents

### 1. Introduction

- 1.1 Overview
- 1.2 Quick Start: Running the Analysis
- 1.3 Output Structure and File Organization
- 1.4 Prerequisites and Setup (pandas, numpy, etc.)

### 2. Core Concepts and Terminology

- 2.1 What is File Lifecycle?
- 2.2 Understanding Git Events (create, modify, rename, delete)
- 2.3 Temporal Concepts
  - Commit timestamps vs. author dates
  - ISO week numbering
  - Quarters and fiscal periods
- 2.4 Author Identity and Normalization
  - Author IDs vs. email addresses
  - Domain extraction
  - Handling email aliases

### 3. Metrics Glossary

#### 3.1 File-Level Metrics

- **Churn** (with mathematical definition)
  - Line churn vs. commit churn
  - Absolute vs. normalized churn
  - Formula: $\text{Churn} = \sum_{i=1}^{n} (\text{lines\_added}_i + \text{lines\_deleted}_i)$
- **File Health**
  - Definition and calculation
  - Health score components
  - Interpretation guidelines
- **Touch Count**
  - What constitutes a "touch"
  - Relationship to commit frequency
- **Age and Lifespan**
  - Creation date vs. first commit
  - Active lifespan vs. total lifespan
- **Stability Metrics**
  - Change frequency
  - Time between modifications
  - Stability score calculation

#### 3.2 Author Metrics

- **Bus Factor**
  - Formal definition
  - Calculation methodology
  - Formula: Minimum number of contributors whose combined contributions exceed 50% of total
  - Critical thresholds and warnings
- **Author Diversity**
  - Shannon entropy for author distribution
  - Formula: $H = -\sum_{i=1}^{n} p_i \log_2(p_i)$
  - Interpretation scale
- **Contribution Patterns**
  - Primary vs. secondary contributors
  - Ownership concentration

#### 3.3 Network Metrics

- **Coupling Strength**
  - Co-change frequency
  - Temporal proximity
  - Formula: $\text{Coupling}(A,B) = \frac{\text{commits\_together}}{\min(\text{commits}_A, \text{commits}_B)}$
- **Coupling Types**
  - Logical coupling vs. structural coupling
  - Symmetric vs. asymmetric dependencies
- **Network Centrality**
  - File importance in the codebase graph
  - Hub files vs. peripheral files

#### 3.4 Temporal Metrics

- **Temporal Entropy**
  - Activity distribution over time
  - Bursty vs. steady development patterns
- **Active Periods**
  - Definition of "active" weeks/months
  - Gaps and dormancy periods
- **Velocity Metrics**
  - Changes per unit time
  - Acceleration and deceleration

---

### 4. Dataset Reference

#### 4.1 Core Lifecycle Dataset

**File:** `file_lifecycle.json`

- **4.1.1 Schema Overview**
  - Top-level structure
  - Metadata section
  - Files array structure
- **4.1.2 Event Types**
  - `create` events
  - `modify` events
  - `rename` events
  - `delete` events
- **4.1.3 Event Fields Explained**
  - `commit_hash`: SHA identifier
  - `timestamp`: ISO 8601 format
  - `author_name` and `author_email`
  - `message`: Commit message
  - `lines_added` and `lines_deleted`
  - `temporal_labels`: Year, quarter, month, week
  - `position_in_history`: Sequence number
- **4.1.4 Pandas Code Examples**
  ```python
  # Load and explore lifecycle data
  # Calculate file lifespans
  # Identify most modified files
  # Plot file creation timeline
  ```

#### 4.2 File Metadata Index

**File:** `metadata/file_index.json`

- **4.2.1 Schema and Structure**
- **4.2.2 Metadata Fields**
  - File path and extension
  - Creation and last modification dates
  - Total commits and authors
  - Aggregate statistics
- **4.2.3 Usage Examples**
  ```python
  # Find files with highest churn
  # Identify files with single authors
  # Filter by file type/extension
  # Calculate average file age
  ```

#### 4.3 Temporal Aggregations

**Files:** `aggregations/temporal_daily.json`, `aggregations/temporal_monthly.json`

- **4.3.1 Daily Aggregation Schema**
- **4.3.2 Monthly Aggregation Schema**
- **4.3.3 Time-Series Metrics**
  - Commits per day/month
  - Active files per period
  - Author activity over time
  - Lines changed trends
- **4.3.4 Code Examples**
  ```python
  # Plot commit activity timeline
  # Identify peak development periods
  # Calculate moving averages
  # Detect seasonal patterns
  ```

#### 4.4 Author Network

**File:** `networks/author_network.json`

- **4.4.1 Network Structure**
  - Nodes (authors)
  - Edges (collaborations)
  - Edge weights
- **4.4.2 Author Node Attributes**
  - Author ID and metadata
  - Total commits and files touched
  - Email domain
  - Active periods
- **4.4.3 Collaboration Metrics**
  - Co-authorship strength
  - File overlap
  - Temporal co-occurrence
- **4.4.4 Analysis Examples**
  ```python
  # Build NetworkX graph
  # Calculate centrality measures
  # Identify collaboration clusters
  # Visualize author relationships
  ```

#### 4.5 Co-change Network

**File:** `networks/cochange_network.json`

- **4.5.1 Network Schema**
  - File nodes
  - Co-change edges
  - Coupling scores
- **4.5.2 Understanding Co-change**
  - What triggers a co-change
  - Coupling vs. correlation
  - Temporal windows
- **4.5.3 Coupling Score Calculation**
  - Formula and normalization
  - Interpreting scores (0.0 to 1.0)
  - Thresholds for strong coupling
- **4.5.4 Practical Applications**
  ```python
  # Find tightly coupled files
  # Detect architectural boundaries
  # Identify refactoring candidates
  # Build dependency graphs
  ```

#### 4.6 File Hierarchy and Directory Statistics

**File:** `aggregations/directory_stats.json`

- **4.6.1 Hierarchical Structure**
- **4.6.2 Directory-Level Metrics**
  - Total files and subdirectories
  - Aggregate churn
  - Author diversity per directory
  - Average file health
- **4.6.3 Usage Patterns**
  ```python
  # Compare module activity
  # Find hotspot directories
  # Calculate directory complexity
  # Visualize tree structure
  ```

#### 4.7 Milestone Snapshots

**File:** `milestones/release_snapshots.json`

- **4.7.1 Milestone Types**
  - Tagged releases
  - Branch points
  - Major commits
- **4.7.2 Snapshot Contents**
  - File states at milestone
  - Metrics at point in time
  - Delta from previous milestone
- **4.7.3 Comparative Analysis**
  ```python
  # Track metric evolution across releases
  # Identify regression points
  # Calculate release velocity
  # Compare release quality
  ```

#### 4.8 Project Hierarchy (Frontend)

**File:** `frontend/project_hierarchy.json`

- **4.8.1 Treemap-Ready Format**
- **4.8.2 Health Scoring System**
  - Components of health score
  - Weight factors
  - Color coding thresholds
- **4.8.3 Hierarchical Aggregation**
  - Parent-child relationships
  - Roll-up calculations
  - Leaf vs. branch nodes
- **4.8.4 Visualization Examples**
  ```python
  # Generate treemap visualization
  # Filter by health threshold
  # Drill-down navigation
  ```

#### 4.9 File Metrics Index (Frontend)

**File:** `frontend/file_metrics_index.json`

- **4.9.1 Per-File Detailed Metrics**
- **4.9.2 Coupling Analysis**
  - Top coupled files
  - Coupling direction (inbound/outbound)
  - Coupling strength distribution
- **4.9.3 Risk Indicators**
  - High churn + low coverage
  - Single author ownership
  - Coupling hotspots
- **4.9.4 Deep-Dive Analysis**
  ```python
  # Identify technical debt
  # Find risky files
  # Analyze coupling patterns
  # Generate file reports
  ```

#### 4.10 Temporal Activity Map (Frontend)

**File:** `frontend/temporal_activity_map.json`

- **4.10.1 Heatmap Data Structure**
- **4.10.2 Weekly Activity Patterns**
  - Day-of-week distributions
  - Hour-of-day patterns (if available)
  - Weekend vs. weekday activity
- **4.10.3 Activity Intensity Metrics**
  - Normalized activity scores
  - Comparative scaling
- **4.10.4 Visualization Code**
  ```python
  # Create activity heatmap
  # Identify working patterns
  # Compare periods
  # Detect anomalies
  ```

---

### 5. Common Analysis Workflows

#### 5.1 Code Quality Assessment

- Identifying high-risk files
- Combining churn, coupling, and bus factor
- Creating composite quality scores
- **Complete workflow with code**

#### 5.2 Team Dynamics Analysis

- Author contribution patterns
- Collaboration network analysis
- Knowledge distribution assessment
- Onboarding/offboarding detection
- **Complete workflow with code**

#### 5.3 Architectural Insight

- Module boundary detection
- Identifying architectural hotspots
- Coupling-based refactoring suggestions
- Monolith vs. modular patterns
- **Complete workflow with code**

#### 5.4 Technical Debt Identification

- Combining multiple risk signals
- Prioritization matrices
- Debt accumulation trends
- Remediation tracking
- **Complete workflow with code**

#### 5.5 Development Velocity Tracking

- Sprint-over-sprint comparisons
- Team productivity metrics
- Bottleneck identification
- **Complete workflow with code**

---

### 6. Advanced Topics

#### 6.1 Cross-Dataset Joins

- Combining lifecycle with network data
- Enriching temporal data with metadata
- Multi-dimensional analysis
- **Code examples**

#### 6.2 Custom Metric Derivation

- Creating composite scores
- Normalization techniques
- Statistical methods for comparison
- **Formulas and implementations**

#### 6.3 Time-Series Analysis

- Trend detection
- Seasonality and cycles
- Forecasting development patterns
- **Using statsmodels and prophet**

#### 6.4 Network Analysis Deep-Dive

- Community detection in author/file networks
- PageRank for file importance
- Structural holes and bridges
- **Using NetworkX**

#### 6.5 Visualization Best Practices

- Choosing the right chart type
- Interactive dashboards
- D3.js integration patterns
- **Example implementations**

---

### 7. Troubleshooting and FAQ

#### 7.1 Common Issues

- Missing or incomplete data
- Handling renames and moves
- Large repository performance
- Timezone considerations

#### 7.2 Data Quality Checks

- Validating completeness
- Detecting anomalies
- Consistency verification
- **Validation scripts**

#### 7.3 Interpretation Pitfalls

- Correlation vs. causation
- Sample bias in quick mode
- Metric limitations
- Context-dependent thresholds

---

### 8. Reference

#### 8.1 Mathematical Formulas Summary

- All metrics in formal notation
- Normalization functions
- Statistical tests

#### 8.2 Dataset Schema Reference

- Complete JSON schemas
- Field data types
- Null handling

#### 8.3 Command-Line Options

- All CLI flags explained
- Preset configurations
- Performance tuning

#### 8.4 External Resources

- Related research papers
- Visualization libraries
- Git analytics tools
- Further reading

---
