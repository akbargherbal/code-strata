# Strata.py Dataset Reference Guide - Part 4

## Advanced Topics, Troubleshooting, and Reference (Sections 6-8)

---

## 5.3 Architectural Insight

**Objective:** Detect module boundaries, identify hotspots, and suggest refactoring based on coupling patterns.

**Complete Workflow:**

```python
import json
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict

# Load coupling network
with open('output/networks/cochange_network.json', 'r') as f:
    cochange = json.load(f)

# Load file metadata
with open('output/metadata/file_index.json', 'r') as f:
    metadata = json.load(f)

# Build coupling graph
G = nx.Graph()
for node in cochange['nodes']:
    G.add_node(node['file_path'], **node)

threshold = 0.5  # Only strong couplings
for edge in cochange['edges']:
    if edge['coupling_score'] >= threshold:
        G.add_edge(edge['source'], edge['target'],
                   weight=edge['coupling_score'])

# Module detection
def extract_module(filepath):
    """Extract top-level module from path"""
    parts = filepath.split('/')
    return parts[0] if len(parts) > 1 else 'root'

# Add module attribute to nodes
for node in G.nodes():
    G.nodes[node]['module'] = extract_module(node)

# Detect communities (architectural modules)
communities = nx.community.louvain_communities(G, weight='weight')

print(f"Detected {len(communities)} architectural communities")
for i, comm in enumerate(sorted(communities, key=len, reverse=True), 1):
    print(f"\nCommunity {i} ({len(comm)} files):")

    # Identify primary module
    modules = defaultdict(int)
    for file in comm:
        modules[G.nodes[file]['module']] += 1

    primary_module = max(modules.items(), key=lambda x: x[1])
    print(f"  Primary module: {primary_module[0]} ({primary_module[1]}/{len(comm)} files)")
    print(f"  Files: {list(comm)[:5]}...")

# Architectural hotspots
# Calculate betweenness centrality (files that bridge modules)
betweenness = nx.betweenness_centrality(G, weight='weight')
hotspots = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:10]

print("\n\nArchitectural Hotspots (High Betweenness):")
print("Files that bridge multiple modules:")
for file, score in hotspots:
    module = G.nodes[file]['module']
    degree = G.degree(file)
    print(f"  {file} (module: {module}, connections: {degree}, betweenness: {score:.3f})")

# Module boundary violations
# Files with high cross-module coupling
cross_module_violations = []
for node in G.nodes():
    node_module = G.nodes[node]['module']
    neighbors = list(G.neighbors(node))

    cross_module = [n for n in neighbors if G.nodes[n]['module'] != node_module]

    if len(cross_module) > len(neighbors) * 0.5:  # >50% cross-module
        violation_score = len(cross_module) / max(len(neighbors), 1)
        cross_module_violations.append({
            'file': node,
            'module': node_module,
            'total_coupling': len(neighbors),
            'cross_module_coupling': len(cross_module),
            'violation_score': violation_score
        })

violations_df = pd.DataFrame(cross_module_violations)
violations_df = violations_df.sort_values('violation_score', ascending=False)

print("\n\nModule Boundary Violations:")
print("Files with excessive cross-module coupling:")
print(violations_df.head(10))

# Coupling-based refactoring suggestions
refactor_candidates = []
for edge in cochange['edges']:
    if edge['coupling_score'] >= 0.8:  # Very strong coupling
        source_module = extract_module(edge['source'])
        target_module = extract_module(edge['target'])

        if source_module != target_module:
            refactor_candidates.append({
                'file_a': edge['source'],
                'file_b': edge['target'],
                'module_a': source_module,
                'module_b': target_module,
                'coupling': edge['coupling_score'],
                'suggestion': 'Consider merging modules or extracting shared abstraction'
            })

if refactor_candidates:
    refactor_df = pd.DataFrame(refactor_candidates)
    print("\n\nRefactoring Suggestions:")
    print(refactor_df)

# Visualize module structure
fig, axes = plt.subplots(1, 2, figsize=(16, 8))

# Full graph with module colors
pos = nx.spring_layout(G, k=0.5, iterations=50, seed=42)
modules = list(set(G.nodes[node]['module'] for node in G.nodes()))
colors = plt.cm.tab10(range(len(modules)))
module_color_map = dict(zip(modules, colors))

node_colors = [module_color_map[G.nodes[node]['module']] for node in G.nodes()]
node_sizes = [G.degree(node) * 50 for node in G.nodes()]

nx.draw(G, pos, ax=axes[0], node_color=node_colors, node_size=node_sizes,
        alpha=0.7, with_labels=False, edge_color='gray', width=0.5)
axes[0].set_title('File Coupling Network (by Module)')

# Module-level graph
module_graph = nx.Graph()
for u, v in G.edges():
    module_u = G.nodes[u]['module']
    module_v = G.nodes[v]['module']

    if module_u != module_v:
        if module_graph.has_edge(module_u, module_v):
            module_graph[module_u][module_v]['weight'] += 1
        else:
            module_graph.add_edge(module_u, module_v, weight=1)

pos_module = nx.spring_layout(module_graph, k=1, iterations=50)
edge_widths = [module_graph[u][v]['weight'] / 5 for u, v in module_graph.edges()]
node_sizes_module = [G.subgraph([n for n in G.nodes() if G.nodes[n]['module'] == m]).number_of_nodes() * 100
                     for m in module_graph.nodes()]

nx.draw(module_graph, pos_module, ax=axes[1], node_size=node_sizes_module,
        with_labels=True, font_size=8, width=edge_widths, alpha=0.8)
axes[1].set_title('Module-Level Coupling')

plt.tight_layout()
plt.show()

# Monolith vs Modular assessment
total_edges = G.number_of_edges()
cross_module_edges = module_graph.number_of_edges()
modularity_score = 1 - (cross_module_edges / max(total_edges, 1))

print(f"\n\nArchitectural Assessment:")
print(f"Modularity score: {modularity_score:.3f}")
print(f"(0 = highly entangled/monolith, 1 = perfectly modular)")

if modularity_score < 0.3:
    print("Assessment: Highly entangled, consider major refactoring")
elif modularity_score < 0.6:
    print("Assessment: Moderate modularity, room for improvement")
else:
    print("Assessment: Well-modularized architecture")
```

### 5.4 Technical Debt Identification

**Objective:** Combine multiple risk signals to identify and prioritize technical debt.

**Complete Workflow:**

```python
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Load datasets
with open('output/metadata/file_index.json', 'r') as f:
    metadata = json.load(f)

with open('output/networks/cochange_network.json', 'r') as f:
    cochange = json.load(f)

with open('output/file_lifecycle.json', 'r') as f:
    lifecycle = json.load(f)

# Build comprehensive file metrics
files_data = []
for filepath, meta in metadata['files'].items():
    # Find coupling degree
    coupling_degree = sum(1 for edge in cochange['edges']
                         if edge['source'] == filepath or edge['target'] == filepath)

    # Find file lifecycle
    file_lifecycle = next((f for f in lifecycle['files'] if f['file_path'] == filepath), None)

    # Calculate temporal metrics
    if file_lifecycle:
        events = file_lifecycle['lifecycle']
        timestamps = [datetime.fromisoformat(e['timestamp'].replace('+0000', '+00:00'))
                     for e in events]

        if len(timestamps) > 1:
            time_diffs = [(timestamps[i+1] - timestamps[i]).days
                         for i in range(len(timestamps)-1)]
            avg_time_between_changes = np.mean(time_diffs)
            max_gap = max(time_diffs)
        else:
            avg_time_between_changes = 0
            max_gap = 0

        # Recent activity
        now = datetime.now()
        recent_30d = sum(1 for ts in timestamps if (now - ts).days <= 30)
        recent_90d = sum(1 for ts in timestamps if (now - ts).days <= 90)
    else:
        avg_time_between_changes = 0
        max_gap = 0
        recent_30d = 0
        recent_90d = 0

    files_data.append({
        'file_path': filepath,
        'extension': meta['extension'],
        'status': meta['status'],
        'total_commits': meta['total_commits'],
        'unique_authors': meta['unique_authors'],
        'total_churn': meta['total_churn'],
        'churn_per_commit': meta['total_churn'] / max(meta['total_commits'], 1),
        'coupling_degree': coupling_degree,
        'avg_time_between_changes': avg_time_between_changes,
        'max_gap_days': max_gap,
        'recent_30d_commits': recent_30d,
        'recent_90d_commits': recent_90d
    })

df = pd.DataFrame(files_data)

# Calculate technical debt indicators
def calculate_debt_score(row):
    """
    Multi-dimensional technical debt scoring
    """
    score = 0
    reasons = []

    # 1. High churn (20 points)
    if row['total_churn'] > 3000:
        score += 20
        reasons.append('High churn')
    elif row['total_churn'] > 1500:
        score += 10

    # 2. Low bus factor (25 points)
    if row['unique_authors'] == 1:
        score += 25
        reasons.append('Single author (bus factor 1)')
    elif row['unique_authors'] == 2:
        score += 15
        reasons.append('Low bus factor (2)')

    # 3. High coupling (20 points)
    if row['coupling_degree'] > 15:
        score += 20
        reasons.append('High coupling')
    elif row['coupling_degree'] > 10:
        score += 10

    # 4. Recent high activity (15 points)
    if row['recent_30d_commits'] > 10:
        score += 15
        reasons.append('Very active recently')
    elif row['recent_30d_commits'] > 5:
        score += 8

    # 5. Volatile change pattern (10 points)
    if row['avg_time_between_changes'] < 7 and row['total_commits'] > 20:
        score += 10
        reasons.append('Frequent changes')

    # 6. High churn rate (10 points)
    if row['churn_per_commit'] > 100:
        score += 10
        reasons.append('High churn per commit')

    return score, '; '.join(reasons)

df['debt_score'], df['debt_reasons'] = zip(*df.apply(calculate_debt_score, axis=1))

# Prioritization matrix
def categorize_priority(row):
    """
    Priority = Impact × Urgency
    Impact = churn + coupling
    Urgency = recent activity
    """
    impact = (row['total_churn'] / 5000) + (row['coupling_degree'] / 20)
    urgency = row['recent_30d_commits'] / 10
    priority = impact * urgency

    if priority > 1.5:
        return 'Critical'
    elif priority > 0.8:
        return 'High'
    elif priority > 0.4:
        return 'Medium'
    else:
        return 'Low'

df['priority'] = df.apply(categorize_priority, axis=1)

# Results
print("Technical Debt Summary:")
print(df['priority'].value_counts())

print("\n\nTop 20 Technical Debt Items:")
debt_items = df.nlargest(20, 'debt_score')[
    ['file_path', 'debt_score', 'priority', 'total_churn',
     'unique_authors', 'coupling_degree', 'debt_reasons']
]
print(debt_items)

# Debt accumulation trends
with open('output/aggregations/temporal_monthly.json', 'r') as f:
    temporal = json.load(f)

monthly_df = pd.DataFrame(temporal['monthly_activity'])
monthly_df['month'] = pd.to_datetime(monthly_df['month'])

# Approximate debt accumulation (using churn as proxy)
monthly_df['debt_proxy'] = monthly_df['total_churn'].cumsum()

plt.figure(figsize=(12, 6))
plt.plot(monthly_df['month'], monthly_df['debt_proxy'], marker='o')
plt.xlabel('Month')
plt.ylabel('Cumulative Churn (Debt Proxy)')
plt.title('Technical Debt Accumulation Over Time')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Remediation tracking
# Suggest refactoring efforts by priority
remediation_plan = df[df['debt_score'] > 30].sort_values('debt_score', ascending=False)

print("\n\nSuggested Remediation Plan:")
for i, (idx, row) in enumerate(remediation_plan.iterrows(), 1):
    print(f"\n{i}. {row['file_path']}")
    print(f"   Priority: {row['priority']}")
    print(f"   Debt Score: {row['debt_score']}")
    print(f"   Issues: {row['debt_reasons']}")
    print(f"   Recommended Actions:")

    if row['unique_authors'] <= 2:
        print(f"     - Pair programming / knowledge transfer")
    if row['coupling_degree'] > 10:
        print(f"     - Refactor to reduce coupling (currently coupled with {row['coupling_degree']} files)")
    if row['total_churn'] > 2000:
        print(f"     - Consider splitting into smaller modules")
    if row['recent_30d_commits'] > 5:
        print(f"     - Stabilize interface before next sprint")

    if i >= 10:  # Limit to top 10
        break

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Debt distribution
df['debt_score'].hist(bins=20, ax=axes[0, 0])
axes[0, 0].set_xlabel('Debt Score')
axes[0, 0].set_ylabel('File Count')
axes[0, 0].set_title('Technical Debt Distribution')

# Churn vs Bus Factor (colored by debt)
scatter = axes[0, 1].scatter(df['unique_authors'], df['total_churn'],
                            c=df['debt_score'], cmap='RdYlGn_r', alpha=0.6)
axes[0, 1].set_xlabel('Unique Authors (Bus Factor)')
axes[0, 1].set_ylabel('Total Churn')
axes[0, 1].set_title('Churn vs Bus Factor')
plt.colorbar(scatter, ax=axes[0, 1], label='Debt Score')

# Priority distribution
df['priority'].value_counts().plot(kind='bar', ax=axes[1, 0])
axes[1, 0].set_xlabel('Priority')
axes[1, 0].set_ylabel('File Count')
axes[1, 0].set_title('Debt Priority Distribution')
axes[1, 0].tick_params(axis='x', rotation=45)

# Coupling vs Debt
axes[1, 1].scatter(df['coupling_degree'], df['debt_score'], alpha=0.6)
axes[1, 1].set_xlabel('Coupling Degree')
axes[1, 1].set_ylabel('Debt Score')
axes[1, 1].set_title('Coupling vs Technical Debt')

plt.tight_layout()
plt.show()

# Export
df.to_csv('technical_debt_analysis.csv', index=False)
print("\n\nDetailed analysis exported to technical_debt_analysis.csv")
```

### 5.5 Development Velocity Tracking

**Objective:** Measure and compare velocity across sprints/periods to identify bottlenecks and trends.

**Complete Workflow:**

```python
import json
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np

# Load temporal data
with open('output/aggregations/temporal_monthly.json', 'r') as f:
    monthly = json.load(f)

with open('output/aggregations/temporal_daily.json', 'r') as f:
    daily = json.load(f)

# Convert to DataFrames
monthly_df = pd.DataFrame(monthly['monthly_activity'])
monthly_df['month'] = pd.to_datetime(monthly_df['month'])
monthly_df = monthly_df.sort_values('month')

daily_df = pd.DataFrame(daily['daily_activity'])
daily_df['date'] = pd.to_datetime(daily_df['date'])
daily_df = daily_df.sort_values('date')

# Calculate velocity metrics
monthly_df['commits_per_day'] = monthly_df['commits'] / monthly_df['active_days']
monthly_df['churn_per_day'] = monthly_df['total_churn'] / monthly_df['active_days']
monthly_df['files_per_commit'] = monthly_df['files_modified'] / monthly_df['commits']

# Sprint-over-sprint comparison (assuming 2-week sprints)
# Group into 2-week periods
daily_df['sprint'] = ((daily_df['date'] - daily_df['date'].min()).dt.days // 14)

sprint_stats = daily_df.groupby('sprint').agg({
    'commits': 'sum',
    'files_modified': 'sum',
    'unique_authors': 'mean',
    'total_churn': 'sum',
    'date': ['min', 'max']
})

sprint_stats.columns = ['commits', 'files', 'avg_authors', 'churn', 'start', 'end']
sprint_stats['velocity'] = sprint_stats['commits'] / 14  # commits per day
sprint_stats['churn_rate'] = sprint_stats['churn'] / 14

print("Sprint Velocity Summary:")
print(sprint_stats[['commits', 'velocity', 'churn_rate', 'avg_authors']])

# Velocity trend
slope, intercept, r_value, p_value, std_err = stats.linregress(
    range(len(sprint_stats)), sprint_stats['velocity']
)

print(f"\n\nVelocity Trend:")
print(f"Slope: {slope:.4f} (commits/day per sprint)")
print(f"R-squared: {r_value**2:.4f}")
print(f"P-value: {p_value:.4f}")

if p_value < 0.05:
    if slope > 0:
        print("Trend: Significantly INCREASING velocity ✓")
    else:
        print("Trend: Significantly DECREASING velocity ⚠")
else:
    print("Trend: No significant trend (stable velocity)")

# Team productivity metrics
# Commits per author
with open('output/file_lifecycle.json', 'r') as f:
    lifecycle = json.load(f)

author_productivity = {}
for file_obj in lifecycle['files']:
    for event in file_obj['lifecycle']:
        author = event.get('author_id')
        timestamp = pd.to_datetime(event['timestamp'])
        sprint_num = ((timestamp - daily_df['date'].min()).days // 14)

        key = (author, sprint_num)
        if key not in author_productivity:
            author_productivity[key] = 0
        author_productivity[key] += 1

# Convert to DataFrame
productivity_records = [
    {'author': author, 'sprint': sprint, 'commits': commits}
    for (author, sprint), commits in author_productivity.items()
]
prod_df = pd.DataFrame(productivity_records)

# Average per author per sprint
avg_productivity = prod_df.groupby('sprint')['commits'].mean()

print("\n\nAverage Commits per Author per Sprint:")
print(avg_productivity)

# Bottleneck identification
# Look for correlation between factors and velocity
factors = monthly_df[['commits', 'unique_authors', 'files_modified', 'total_churn']].copy()
correlation = factors.corr()

print("\n\nVelocity Correlation Analysis:")
print("Correlations with commit count:")
print(correlation['commits'].sort_values(ascending=False))

# Bottleneck indicators
bottleneck_months = monthly_df[
    (monthly_df['commits'] < monthly_df['commits'].quantile(0.25)) &
    (monthly_df['unique_authors'] < monthly_df['unique_authors'].median())
]

if len(bottleneck_months) > 0:
    print("\n\nPotential Bottleneck Periods:")
    print(bottleneck_months[['month', 'commits', 'unique_authors', 'files_modified']])

# Visualizations
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Monthly velocity
axes[0, 0].plot(monthly_df['month'], monthly_df['commits_per_day'], marker='o')
axes[0, 0].set_ylabel('Commits per Day')
axes[0, 0].set_title('Monthly Velocity Trend')
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].tick_params(axis='x', rotation=45)

# Sprint velocity with trend line
x = range(len(sprint_stats))
y = sprint_stats['velocity']
axes[0, 1].bar(x, y, alpha=0.6)
axes[0, 1].plot(x, slope * np.array(x) + intercept, 'r--', label='Trend')
axes[0, 1].set_xlabel('Sprint')
axes[0, 1].set_ylabel('Velocity (commits/day)')
axes[0, 1].set_title('Sprint Velocity with Trend')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Authors vs Velocity
axes[1, 0].scatter(monthly_df['unique_authors'], monthly_df['commits'], alpha=0.6)
axes[1, 0].set_xlabel('Unique Authors')
axes[1, 0].set_ylabel('Commits')
axes[1, 0].set_title('Team Size vs Output')
axes[1, 0].grid(True, alpha=0.3)

# Churn rate over time
axes[1, 1].plot(monthly_df['month'], monthly_df['churn_per_day'],
                marker='o', color='orange')
axes[1, 1].set_ylabel('Churn per Day')
axes[1, 1].set_title('Code Churn Rate')
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()

# Export sprint summary
sprint_stats.to_csv('velocity_analysis.csv')
print("\n\nVelocity analysis exported to velocity_analysis.csv")
```

---

## 6. Advanced Topics

### 6.1 Cross-Dataset Joins

**Combining Lifecycle with Network Data:**

```python
import json
import pandas as pd

# Load multiple datasets
with open('output/file_lifecycle.json', 'r') as f:
    lifecycle = json.load(f)

with open('output/metadata/file_index.json', 'r') as f:
    metadata = json.load(f)

with open('output/networks/cochange_network.json', 'r') as f:
    cochange = json.load(f)

# Example: Enrich lifecycle events with file metadata
enriched_events = []
for file_obj in lifecycle['files']:
    filepath = file_obj['file_path']
    file_meta = metadata['files'].get(filepath, {})

    for event in file_obj['lifecycle']:
        enriched_event = {
            **event,
            'file_path': filepath,
            'file_extension': file_meta.get('extension'),
            'file_total_commits': file_meta.get('total_commits'),
            'file_unique_authors': file_meta.get('unique_authors')
        }
        enriched_events.append(enriched_event)

events_df = pd.DataFrame(enriched_events)

# Join with coupling data
coupling_dict = {}
for edge in cochange['edges']:
    source = edge['source']
    if source not in coupling_dict:
        coupling_dict[source] = []
    coupling_dict[source].append({
        'coupled_file': edge['target'],
        'coupling_score': edge['coupling_score']
    })

# Add coupling info to events
events_df['coupled_files'] = events_df['file_path'].map(
    lambda x: len(coupling_dict.get(x, []))
)

print("Enriched events sample:")
print(events_df.head())
```

**Enriching Temporal Data with Metadata:**

```python
# Load temporal and metadata
with open('output/aggregations/temporal_monthly.json', 'r') as f:
    temporal = json.load(f)

with open('output/networks/author_network.json', 'r') as f:
    author_net = json.load(f)

# Create author lookup
author_info = {node['author_id']: node for node in author_net['nodes']}

# Enrich temporal data with author details
monthly_df = pd.DataFrame(temporal['monthly_activity'])

for idx, row in monthly_df.iterrows():
    author_dict = row.get('authors', {})

    # Add author domains
    domains = []
    for author_id in author_dict.keys():
        if author_id in author_info:
            domains.append(author_info[author_id].get('author_domain', ''))

    monthly_df.at[idx, 'unique_domains'] = len(set(domains))

print("Enriched monthly data:")
print(monthly_df[['month', 'commits', 'unique_authors', 'unique_domains']])
```

### 6.2 Custom Metric Derivation

**Creating Composite Scores:**

```python
import pandas as pd
import numpy as np

# Load data
with open('output/metadata/file_index.json', 'r') as f:
    metadata = json.load(f)

files_df = pd.DataFrame(metadata['files'].values())

# Define custom metric: Maintainability Index
def calculate_maintainability_index(row):
    """
    Custom composite score combining multiple factors

    MI = 100 - (Complexity + Churn_Rate + Bus_Factor_Risk)
    """
    # Normalize components to 0-100 scale

    # Complexity (based on coupling and commits)
    complexity = min(
        (row['total_commits'] / 100) * 30 +
        (row.get('coupling_degree', 0) / 20) * 20,
        50
    )

    # Churn rate risk
    churn_rate = row['total_churn'] / max(row['total_commits'], 1)
    churn_risk = min((churn_rate / 100) * 30, 30)

    # Bus factor risk
    if row['unique_authors'] == 1:
        bus_risk = 20
    elif row['unique_authors'] == 2:
        bus_risk = 10
    else:
        bus_risk = 0

    mi = max(0, 100 - (complexity + churn_risk + bus_risk))
    return mi

files_df['maintainability_index'] = files_df.apply(calculate_maintainability_index, axis=1)

print("Maintainability Index Distribution:")
print(files_df['maintainability_index'].describe())
```

**Normalization Techniques:**

```python
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Min-Max normalization (0-1 scale)
scaler = MinMaxScaler()
files_df['churn_normalized'] = scaler.fit_transform(files_df[['total_churn']])

# Z-score normalization (mean=0, std=1)
scaler_std = StandardScaler()
files_df['commits_zscore'] = scaler_std.fit_transform(files_df[['total_commits']])

# Quantile-based normalization
files_df['churn_quantile'] = files_df['total_churn'].rank(pct=True)

print("Normalized metrics:")
print(files_df[['total_churn', 'churn_normalized', 'churn_quantile']].head())
```

### 6.3 Time-Series Analysis

**Trend Detection:**

```python
from scipy import stats
import pandas as pd

# Load temporal data
with open('output/aggregations/temporal_daily.json', 'r') as f:
    daily = json.load(f)

daily_df = pd.DataFrame(daily['daily_activity'])
daily_df['date'] = pd.to_datetime(daily_df['date'])

# Mann-Kendall trend test
def mann_kendall_test(data):
    n = len(data)
    s = 0
    for i in range(n-1):
        for j in range(i+1, n):
            s += np.sign(data[j] - data[i])

    var_s = n * (n-1) * (2*n+5) / 18
    z = (s - np.sign(s)) / np.sqrt(var_s) if var_s > 0 else 0
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))

    return {'statistic': s, 'z_score': z, 'p_value': p_value}

result = mann_kendall_test(daily_df['commits'].values)
print("Mann-Kendall Trend Test:")
print(f"Z-score: {result['z_score']:.4f}")
print(f"P-value: {result['p_value']:.4f}")
```

**Seasonality Detection:**

```python
from statsmodels.tsa.seasonal import seasonal_decompose

# Ensure regular frequency
monthly_df = pd.DataFrame(monthly['monthly_activity'])
monthly_df['month'] = pd.to_datetime(monthly_df['month'])
monthly_df = monthly_df.set_index('month')
monthly_df = monthly_df.asfreq('MS')  # Month start frequency

# Decompose time series
decomposition = seasonal_decompose(
    monthly_df['commits'].fillna(method='ffill'),
    model='additive',
    period=12  # Annual seasonality
)

# Plot
fig, axes = plt.subplots(4, 1, figsize=(12, 10))

monthly_df['commits'].plot(ax=axes[0], title='Original')
decomposition.trend.plot(ax=axes[1], title='Trend')
decomposition.seasonal.plot(ax=axes[2], title='Seasonal')
decomposition.resid.plot(ax=axes[3], title='Residual')

plt.tight_layout()
plt.show()
```

**Forecasting with Prophet:**

```python
try:
    from prophet import Prophet

    # Prepare data
    forecast_df = monthly_df.reset_index()[['month', 'commits']]
    forecast_df.columns = ['ds', 'y']

    # Fit model
    model = Prophet()
    model.fit(forecast_df)

    # Make predictions
    future = model.make_future_dataframe(periods=6, freq='MS')
    forecast = model.predict(future)

    # Plot
    fig = model.plot(forecast)
    plt.title('Commit Forecast (6 months)')
    plt.show()

except ImportError:
    print("Prophet not installed. Install with: pip install prophet")
```

### 6.4 Network Analysis Deep-Dive

**Community Detection:**

```python
import networkx as nx
from networkx.algorithms import community

# Load network
with open('output/networks/cochange_network.json', 'r') as f:
    network = json.load(f)

G = nx.Graph()
for node in network['nodes']:
    G.add_node(node['file_path'], **node)

for edge in network['edges']:
    if edge['coupling_score'] >= 0.5:
        G.add_edge(edge['source'], edge['target'],
                   weight=edge['coupling_score'])

# Louvain communities
communities_louvain = community.louvain_communities(G, weight='weight', seed=42)

# Girvan-Newman (hierarchical)
communities_gn = list(community.girvan_newman(G))

print(f"Louvain: {len(communities_louvain)} communities")
print(f"Girvan-Newman (level 1): {len(communities_gn[0])} communities")

# Modularity score
modularity = community.modularity(G, communities_louvain, weight='weight')
print(f"Modularity: {modularity:.3f}")
```

**PageRank for File Importance:**

```python
# Calculate PageRank
pagerank = nx.pagerank(G, weight='weight')

# Sort by importance
ranked_files = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)

print("Top 10 most important files (PageRank):")
for file, score in ranked_files[:10]:
    print(f"{file}: {score:.4f}")
```

**Structural Holes and Bridges:**

```python
# Betweenness centrality (bridges)
betweenness = nx.betweenness_centrality(G, weight='weight')

# Constraint (structural holes - lower is better)
try:
    constraint = nx.constraint(G)

    # Low constraint = spans structural holes
    low_constraint = sorted(constraint.items(), key=lambda x: x[1])[:10]

    print("\nFiles spanning structural holes:")
    for file, c in low_constraint:
        print(f"{file}: constraint={c:.3f}")
except:
    print("Constraint calculation requires connected graph")
```

### 6.5 Visualization Best Practices

**Choosing the Right Chart Type:**

| Data Type     | Best Visualization     | Use Case             |
| ------------- | ---------------------- | -------------------- |
| Time series   | Line chart, area chart | Trends over time     |
| Distributions | Histogram, box plot    | Spread and outliers  |
| Comparisons   | Bar chart, grouped bar | Category comparisons |
| Relationships | Scatter plot, bubble   | Correlations         |
| Hierarchies   | Treemap, sunburst      | Nested structures    |
| Networks      | Force-directed graph   | Relationships        |
| Proportions   | Pie chart, donut       | Parts of whole       |

**Interactive Dashboards with Plotly:**

```python
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Create dashboard
fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=('Commits Over Time', 'Churn Distribution',
                   'Authors vs Files', 'Health Scores'),
    specs=[[{'type': 'scatter'}, {'type': 'histogram'}],
           [{'type': 'scatter'}, {'type': 'box'}]]
)

# Add traces
fig.add_trace(
    go.Scatter(x=monthly_df['month'], y=monthly_df['commits'], mode='lines+markers'),
    row=1, col=1
)

fig.add_trace(
    go.Histogram(x=files_df['total_churn']),
    row=1, col=2
)

fig.add_trace(
    go.Scatter(x=files_df['unique_authors'], y=files_df['total_commits'],
               mode='markers', marker=dict(size=5, opacity=0.6)),
    row=2, col=1
)

fig.add_trace(
    go.Box(y=files_df['health_score']),
    row=2, col=2
)

fig.update_layout(height=800, showlegend=False, title_text="Repository Dashboard")
fig.show()
```

---

## 7. Troubleshooting and FAQ

### 7.1 Common Issues

**Missing or Incomplete Data:**

**Problem:** Some files appear in lifecycle but not in metadata.

**Solution:**

```python
# Check for missing files
lifecycle_files = {f['file_path'] for f in lifecycle['files']}
metadata_files = set(metadata['files'].keys())

missing_in_metadata = lifecycle_files - metadata_files
missing_in_lifecycle = metadata_files - lifecycle_files

print(f"Missing in metadata: {len(missing_in_metadata)}")
print(f"Missing in lifecycle: {len(missing_in_lifecycle)}")
```

**Handling Renames and Moves:**

**Problem:** File appears multiple times with different paths.

**Solution:** Use `original_path` and `current_path` fields:

```python
# Track file identity across renames
def get_file_identity(file_meta):
    return file_meta.get('original_path', file_meta['file_path'])

# Group by identity
from collections import defaultdict
identity_groups = defaultdict(list)
for filepath, meta in metadata['files'].items():
    identity = get_file_identity(meta)
    identity_groups[identity].append(filepath)

# Find files with renames
renamed_files = {k: v for k, v in identity_groups.items() if len(v) > 1}
print(f"Files with renames: {len(renamed_files)}")
```

**Large Repository Performance:**

**Problem:** Analysis takes too long or runs out of memory.

**Solutions:**

1. Use quick mode: `python strata.py --quick --sample-rate 0.5`
2. Disable expensive aggregators: `--skip-cochange --skip-milestones`
3. Increase memory limit: `--memory-limit 8192`
4. Process in batches:

```python
# Process subset of files
files_subset = lifecycle['files'][:1000]
```

**Timezone Considerations:**

**Problem:** Timestamps show unexpected dates.

**Solution:**

```python
from datetime import timezone, datetime

# Ensure UTC
def parse_utc(timestamp_str):
    dt = datetime.fromisoformat(timestamp_str.replace('+0000', '+00:00'))
    return dt.astimezone(timezone.utc)

# Or convert to local timezone
import pytz
local_tz = pytz.timezone('America/New_York')
def to_local(timestamp_str):
    dt = parse_utc(timestamp_str)
    return dt.astimezone(local_tz)
```

### 7.2 Data Quality Checks

**Validating Completeness:**

```python
import json

def validate_lifecycle_data(filepath):
    with open(filepath, 'r') as f:
        data = json.load(f)

    issues = []

    # Check metadata
    required_meta = ['version', 'generated_at', 'total_commits', 'files_tracked']
    for field in required_meta:
        if field not in data['metadata']:
            issues.append(f"Missing metadata field: {field}")

    # Check files
    for file_obj in data['files']:
        if 'file_path' not in file_obj:
            issues.append("File object missing file_path")

        if 'lifecycle' not in file_obj or not file_obj['lifecycle']:
            issues.append(f"Empty lifecycle for {file_obj.get('file_path', 'unknown')}")

        # Check events
        for event in file_obj.get('lifecycle', []):
            required_fields = ['event_type', 'commit_hash', 'timestamp']
            for field in required_fields:
                if field not in event:
                    issues.append(f"Event missing {field}")

    return issues

issues = validate_lifecycle_data('output/file_lifecycle.json')
if issues:
    print("Data quality issues found:")
    for issue in issues[:10]:  # Show first 10
        print(f"  - {issue}")
else:
    print("✓ Data validation passed")
```

**Detecting Anomalies:**

```python
# Statistical outlier detection
from scipy import stats

# Z-score method
z_scores = np.abs(stats.zscore(files_df['total_churn']))
outliers = files_df[z_scores > 3]

print(f"Churn outliers (>3 std dev): {len(outliers)}")
print(outliers[['file_path', 'total_churn']])

# IQR method
Q1 = files_df['total_commits'].quantile(0.25)
Q3 = files_df['total_commits'].quantile(0.75)
IQR = Q3 - Q1
outliers_iqr = files_df[
    (files_df['total_commits'] < Q1 - 1.5*IQR) |
    (files_df['total_commits'] > Q3 + 1.5*IQR)
]

print(f"\nCommit outliers (IQR method): {len(outliers_iqr)}")
```

**Consistency Verification:**

```python
# Cross-check totals
metadata_total_commits = sum(f['total_commits'] for f in metadata['files'].values())
lifecycle_total_events = sum(len(f['lifecycle']) for f in lifecycle['files'])

print(f"Metadata total commits: {metadata_total_commits}")
print(f"Lifecycle total events: {lifecycle_total_events}")

if abs(metadata_total_commits - lifecycle_total_events) > 100:
    print("⚠ Warning: Significant mismatch between metadata and lifecycle")
```

### 7.3 Interpretation Pitfalls

**Correlation vs. Causation:**

Files that change together (high coupling) doesn't necessarily mean one causes changes in the other. They might both be affected by a third factor (e.g., shared feature implementation).

**Sample Bias in Quick Mode:**

When using `--quick` mode, remember:

- Metrics are approximations based on sampled commits
- Rare events may be missed
- Use for exploration, not production metrics

**Metric Limitations:**

| Metric           | Limitation                                 | Mitigation                             |
| ---------------- | ------------------------------------------ | -------------------------------------- |
| Bus Factor       | Doesn't account for expertise depth        | Review code quality, not just quantity |
| Churn            | High churn may indicate active development | Combine with health score              |
| Coupling         | May miss runtime dependencies              | Use with static analysis tools         |
| Author Diversity | Doesn't measure contribution quality       | Check PR reviews, not just commits     |

**Context-Dependent Thresholds:**

Default thresholds (e.g., bus factor < 3 = risk) may not apply to your project:

- Adjust based on team size
- Consider project criticality
- Factor in domain complexity

---

## 8. Reference

### 8.1 Mathematical Formulas Summary

**Churn:**
$$\text{Churn} = \sum_{i=1}^{n} (\text{lines\_added}_i + \text{lines\_deleted}_i)$$

**Bus Factor:**
$$\text{BusFactor} = \min\{k \mid \sum_{i=1}^{k} \text{contribution}(c_i) > 0.5 \times \text{Total}\}$$

**Shannon Entropy (Author Diversity):**
$$H = -\sum_{i=1}^{n} p_i \log_2(p_i)$$

**Coupling Score:**
$$\text{Coupling}(A,B) = \frac{\text{commits\_together}(A, B)}{\min(\text{commits}(A), \text{commits}(B))}$$

**Health Score:**
$$\text{Health} = 0.4 \times \text{Stability} + 0.3 \times \text{BusFactor} + 0.2 \times \text{Coupling} + 0.1 \times \text{Age}$$

**Temporal Entropy:**
$$H_{temporal} = -\sum_{t=1}^{T} p_t \log_2(p_t)$$

**Gini Coefficient:**
$$G = \frac{2\sum_{i=1}^{n}(n - i + 0.5)x_i}{n\sum_{i=1}^{n}x_i} - 1$$

**Herfindahl-Hirschman Index:**
$$\text{HHI} = \sum_{i=1}^{n} \left(\frac{\text{commits}_i}{\text{Total}}\right)^2$$

### 8.2 Dataset Schema Reference

**file_lifecycle.json:**

```json
{
  "metadata": {
    "version": "string",
    "schema_version": "string",
    "repository": "string",
    "generated_at": "ISO datetime",
    "total_commits": "integer",
    "files_tracked": "integer",
    "quick_mode": "boolean",
    "sample_rate": "float | null"
  },
  "files": [
    {
      "file_path": "string",
      "lifecycle": [
        {
          "event_type": "create|modify|rename|delete",
          "commit_hash": "string",
          "timestamp": "ISO datetime",
          "author_name": "string",
          "author_email": "string",
          "message": "string",
          "lines_added": "integer",
          "lines_deleted": "integer",
          "temporal_labels": {
            "year": "integer",
            "quarter": "integer (1-4)",
            "month": "integer (1-12)",
            "week_no": "integer (1-53)",
            "day_of_week": "integer (0-6)",
            "day_of_year": "integer (1-366)"
          },
          "author_id": "string",
          "author_domain": "string",
          "position_in_history": "integer",
          "old_path": "string | null (rename only)",
          "new_path": "string | null (rename only)"
        }
      ]
    }
  ]
}
```

### 8.3 Command-Line Options

**Core Options:**

```bash
python strata.py [REPO_PATH] [OPTIONS]
```

| Option                | Description             | Default    |
| --------------------- | ----------------------- | ---------- |
| `--output DIR`        | Output directory        | `./output` |
| `--quick`             | Enable sampling mode    | False      |
| `--sample-rate FLOAT` | Sampling rate (0.0-1.0) | 1.0        |
| `--verbose`           | Detailed logging        | False      |
| `--profile`           | Enable CPU profiling    | False      |
| `--memory-limit MB`   | Max memory usage        | None       |

**Phase Control:**

```bash
--phase1        # Core lifecycle + metadata
--phase2        # Add temporal + author networks
--phase3        # Add cochange + hierarchies + milestones
```

**Aggregator Control:**

```bash
--skip-metadata        # Disable file metadata aggregator
--skip-temporal        # Disable temporal aggregations
--skip-author-network  # Disable author network
--skip-cochange        # Disable co-change network
--skip-hierarchy       # Disable directory hierarchy
--skip-milestones      # Disable milestone snapshots
```

**Performance Tuning:**

```bash
--max-edges INT       # Limit author network edges (default: 10000)
--max-cochange INT    # Limit coupling pairs (default: 50000)
```

**Presets:**

```bash
--preset fast         # Minimal datasets, quick mode
--preset full         # All datasets, no sampling
--preset balanced     # Good balance of speed/detail
```

### 8.4 External Resources

**Related Research Papers:**

- "Predicting Faults from Cached History" (Kim et al., 2007)
- "Mining Software Repositories" (Hassan, 2008)
- "Code Ownership and Software Quality" (Bird et al., 2011)

**Visualization Libraries:**

- D3.js: https://d3js.org/
- Plotly: https://plotly.com/python/
- NetworkX: https://networkx.org/
- Seaborn: https://seaborn.pydata.org/

**Git Analytics Tools:**

- CodeScene: https://codescene.io/
- GitPrime (now Pluralsight Flow): https://www.pluralsight.com/product/flow
- Code Climate: https://codeclimate.com/

**Further Reading:**

- "Your Code as a Crime Scene" by Adam Tornhill
- "Software Design X-Rays" by Adam Tornhill
- "Working Effectively with Legacy Code" by Michael Feathers

---

## Conclusion

This comprehensive guide covers all aspects of working with Strata.py's dataset outputs. You should now be able to:

✓ Understand the structure and meaning of each dataset
✓ Load and analyze data using pandas and other Python tools
✓ Combine multiple datasets for deeper insights
✓ Calculate custom metrics tailored to your needs
✓ Visualize findings effectively
✓ Identify code quality issues and technical debt
✓ Assess team dynamics and collaboration patterns
✓ Make data-driven architectural decisions

**Next Steps:**

1. Run Strata.py on your repository
2. Explore the generated datasets
3. Start with simple analyses from Section 5
4. Gradually incorporate advanced techniques from Section 6
5. Share insights with your team

**Getting Help:**

- Review the troubleshooting section (7) for common issues
- Check validation scripts (7.2) for data quality
- Experiment with code examples—they're designed to be executed directly

**Contributing:**
If you develop useful analysis workflows or discover interesting patterns, consider documenting and sharing them with the community.

---

_This guide is based on Strata.py version 2.2.0. For the latest documentation and updates, refer to the tool's repository._
