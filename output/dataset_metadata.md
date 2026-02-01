# Git File Lifecycle Analysis - Dataset Metadata

**Generated:** 2026-02-01T07:48:21.636959+00:00
**Repository:** `/home/akbar/Jupyter_Notebooks/SOFTWARE_ARCHEOLOGY/excalidraw`
**Generator Version:** 2.2.0
**Schema Version:** 1.1.0

---

## 📊 Analysis Overview

- **Total Commits:** 5,137
- **Total Files Tracked:** 3,484
- **Total Changes Recorded:** 26,822

### Phase 5 Enhancements ✨

This dataset includes rich metadata for each commit:
- **Temporal Labels**: Year, quarter, month, week number, day of week, day of year
- **Author Normalization**: Consistent author IDs and email domains
- **Diff Statistics**: Lines added and deleted per change
- **Position Metrics**: Sequence numbers and first/last commit flags

### Performance Metrics

- **Execution Time:** 4.97s
- **Peak Memory:** 0.0 MB
- **Cache Hit Rate:** 0.0%

---

## 📂 Dataset Files

### `networks/author_network.json`

**Size:** 1,854,794 bytes

### `networks/cochange_network.json`

**Size:** 16,301,257 bytes

### `file_lifecycle.json`

**Size:** 20,817,664 bytes

### `aggregations/directory_stats.json`

**Size:** 1,137,989 bytes

### `metadata/file_index.json`

**Size:** 2,530,269 bytes

### `frontend/file_metrics_index.json`

**Size:** 3,946,139 bytes

### `milestones/release_snapshots.json`

**Size:** 10,657 bytes

### `frontend/project_hierarchy.json`

**Size:** 2,562,139 bytes

### `frontend/temporal_activity_map.json`

**Size:** 393,253 bytes

### `aggregations/temporal_daily.json`

**Size:** 241,078 bytes

---

## 🔍 Enhanced Metadata Example

Each commit in `file_lifecycle.json` now includes:

```json
{
  "commit_hash": "abc123...",
  "timestamp": 1712927658,
  "datetime": "2024-04-12T13:14:18+00:00",
  "operation": "M",
  "author_name": "John Doe",
  "author_email": "john@example.com",
  "author_id": "john_doe_john",
  "author_domain": "example.com",
  "temporal": {
    "year": 2024,
    "quarter": 2,
    "month": 4,
    "week_no": 15,
    "day_of_week": 4,
    "day_of_year": 103
  },
  "lines_added": 15,
  "lines_deleted": 3,
  "sequence": 42,
  "is_first": false,
  "is_last": false
}
```

---

## 📖 Usage Notes

### Temporal Analysis
Use the `temporal` object to:
- Group commits by week, month, quarter, or year
- Analyze activity patterns by day of week
- Track development velocity over time

### Author Analysis
Use `author_id` for:
- Consistent author identification across email variations
- Grouping contributions by author
- Use `author_domain` to identify organization contributions

### Diff Statistics
Use `lines_added` and `lines_deleted` to:
- Calculate code churn
- Identify hotspots of activity
- Measure commit size and complexity

### Position Metrics
Use `sequence`, `is_first`, and `is_last` to:
- Identify file creation and final modifications
- Track commit order within file history
- Build timeline visualizations
