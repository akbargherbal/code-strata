### 1. `project_hierarchy.json`
**Purpose:** Powers the **Treemap Explorer**.
**Requirement:** A fully recursive, nested JSON structure. The frontend should simply pass this object to D3 without any parsing, sorting, or tree-building logic.

#### JSON Structure
```json
{
  "meta": {
    "generated_at": "2026-01-30T09:40:53Z",
    "root_path": "/",
    "repository_name": "excalidraw"
  },
  "tree": {
    "name": "root",
    "path": "",
    "type": "directory",
    "stats": {
      "total_commits": 5137,
      "health_score_avg": 72
    },
    "children": [
      {
        "name": "src",
        "path": "src",
        "type": "directory",
        "stats": {
          "total_commits": 4200,
          "health_score_avg": 68
        },
        "children": [
          {
            "name": "App.tsx",
            "path": "src/App.tsx",
            "type": "file",
            "value": 150, 
            "attributes": {
              "health_score": 45,
              "health_category": "critical",
              "bus_factor_status": "high-risk",
              "churn_rate": 0.85,
              "primary_author_id": "john_doe",
              "last_modified_age_days": 2,
              "created_at_iso": "2023-01-15T00:00:00Z"
            }
          }
          // ... more files/directories
        ]
      }
    ]
  }
}
```

#### Field Definitions
*   **`value`**: The integer used to size the Treemap cell (usually `total_commits`).
*   **`attributes.health_score`**: An integer (0-100) pre-calculated by the backend.
*   **`attributes.health_category`**: One of `"healthy"`, `"medium"`, `"critical"`.
*   **`attributes.bus_factor_status`**: One of `"low-risk"`, `"medium-risk"`, `"high-risk"`.
*   **`attributes.churn_rate`**: A float (0.0 - 1.0) representing code turbulence.

---

### 2. `file_metrics_index.json`
**Purpose:** Powers the **Detail Panel** and **Tooltips**.
**Requirement:** A flat Key-Value map for O(1) lookup. This prevents the hierarchy file from becoming too bloated with details that are only needed when a user clicks a file.

#### JSON Structure
```json
{
  "src/App.tsx": {
    "identifiers": {
      "author_ids": ["john_doe", "jane_smith", "bob_builder"],
      "primary_author_id": "john_doe",
      "primary_author_percentage": 0.65
    },
    "volume": {
      "lines_added": 1250,
      "lines_deleted": 800,
      "net_change": 450,
      "total_commits": 150
    },
    "coupling": {
      "max_strength": 0.75,
      "top_partners": [
        { "path": "src/components/Button.tsx", "strength": 0.75 },
        { "path": "src/utils/helpers.ts", "strength": 0.40 },
        { "path": "src/types.ts", "strength": 0.25 }
      ]
    },
    "lifecycle": {
      "created_iso": "2023-01-15T10:00:00Z",
      "last_modified_iso": "2026-01-28T14:30:00Z",
      "is_dormant": false
    }
  },
  "src/utils/legacy.ts": {
    // ... metrics for next file
  }
}
```

#### Field Definitions
*   **`identifiers`**: Uses the normalized `author_id` from Phase 5, not raw email strings.
*   **`volume`**: Exact line counts (derived from the new diff stats).
*   **`coupling`**: Pre-calculated top 5-10 coupled files. This removes the need for the frontend to load the entire network graph and traverse edges.

---

### 3. `temporal_activity_map.json`
**Purpose:** Powers the **Timeline Heatmap**.
**Requirement:** A pre-aggregated time-series matrix. The frontend should not have to iterate over raw events to count commits per week.

#### JSON Structure
```json
{
  "meta": {
    "granularity": "week",
    "start_date": "2023-01-01",
    "end_date": "2026-01-30",
    "data_schema": ["commits", "lines_changed", "unique_authors"]
  },
  "data": {
    "src/components": {
      "2024-W01": [5, 120, 2], 
      "2024-W02": [12, 450, 3],
      "2024-W03": [0, 0, 0],
      "2024-W04": [8, 80, 1]
    },
    "src/utils": {
      "2024-W01": [1, 10, 1],
      "2024-W02": [0, 0, 0]
    }
  }
}
```

#### Field Definitions
*   **Keys (`src/components`)**: Directory paths.
*   **Sub-Keys (`2024-W01`)**: ISO 8601 Year-Week format.
*   **Values (`[5, 120, 2]`)**: An array corresponding to `data_schema`:
    1.  Total Commits in that week.
    2.  Total Lines Changed (Added + Deleted) in that week.
    3.  Count of Unique Authors in that week.

---

### Summary of "Missing" Logic to be Provided by Backend

To generate these files, the backend must perform the following logic that is currently done in the browser:

1.  **Health Scoring**:
    *   Calculate a 0-100 score for every file based on Churn, Age, and Author Diversity.
    *   Assign categorical labels ("critical", "healthy").

2.  **Tree Construction**:
    *   Convert the flat list of file paths into the recursive `children` array structure.
    *   Aggregate stats (like total commits) from files up to their parent directories.

3.  **Coupling Analysis**:
    *   Identify the top N coupled partners for every file and store them directly in the file's record (so the frontend doesn't need the full graph).

4.  **Temporal Aggregation**:
    *   Group all raw events by Directory AND Week.
    *   Sum the metrics (commits, lines, authors) for each group.