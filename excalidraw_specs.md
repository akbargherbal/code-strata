# Git Repository Evolution: Animation Specifications

## Overview
Create 2-3 browser-based visualizations that reveal how the Excalidraw repository evolved from 2020-2025, focusing on structural changes, activity patterns, and file lifecycle dynamics. Target audience: developers onboarding to the codebase or stakeholders reviewing project history.

## Animation Type 1: Timeline Heatmap (Primary Focus)

**Concept**: A chronological view showing repository activity across time and directory structure

**Visual Elements**:
- **Y-axis**: Major directory paths (collapsible tree structure, ~15-20 visible paths)
- **X-axis**: Time (daily/weekly/monthly bins depending on zoom level)
- **Heatmap cells**: Color intensity = commit frequency + line changes
- **Overlay markers**: File creations (green dot), deletions (red X), major refactors (yellow star)

**Key Interactions**:
- Scrub timeline to see repository state at any point
- Click cell to see commit details, affected files, and authors
- Filter by author, file type, or directory
- Toggle between "commits" vs "lines changed" vs "unique authors" modes

**Data Requirements**: file_lifecycle_events.csv (all we need)

**Success Criteria**: User can identify "hot zones" of development, major refactoring periods, and dormant areas within 30 seconds

---

## Animation Type 2: Repository Structure Treemap Animation

**Concept**: Animated treemap showing file/directory sizes and change frequency over time

**Visual Elements**:
- **Rectangles**: File size (area) × change frequency (color warmth)
- **Hierarchy**: Nested boxes for directory structure
- **Animation**: Morph between time periods (quarterly snapshots)
- **Labels**: Show on hover or for large files only

**Key Interactions**:
- Play/pause time-lapse (auto-advance quarterly)
- Jump to specific date
- Click file to see lifecycle stats
- Toggle metrics: net_lines, total_commits, or file count

**Data Requirements**: file_lifecycle_summary.csv + derive directory aggregates

**Success Criteria**: User can visually identify which parts of the codebase grew, shrank, or stabilized over time

---

## Animation Type 3: Contributor Flow Diagram (Optional, High Impact)

**Concept**: Sankey-style flow showing how authors contribute to different parts of the repository over time

**Visual Elements**:
- **Left side**: Authors (sorted by contribution volume)
- **Right side**: Top 10-15 directories
- **Flows**: Curved bands showing commit volume from author → directory
- **Time slider**: Adjust to see different time periods
- **Color**: By author (consistent across views)

**Key Interactions**:
- Filter to specific time range
- Highlight specific author or directory
- Click flow to see commit details
- Toggle between "commits" vs "lines changed"

**Data Requirements**: file_lifecycle_events.csv grouped by author + directory

**Success Criteria**: User can identify primary contributors for any area of the codebase and spot ownership changes

---

## Implementation Priorities

### Must Have (MVP):
1. **Timeline Heatmap** - Delivers maximum insight with minimal complexity
2. **Working time controls** - Scrubbing, play/pause, date selection
3. **Basic filtering** - By directory, author, or date range

### Should Have:
4. **Treemap Animation** - Adds structural understanding
5. **Linked views** - Clicking in one view updates others
6. **Export capabilities** - Screenshot or data export

### Could Have:
7. **Contributor Flow** - If Heatmap + Treemap prove insufficient for understanding team dynamics
8. **3D timeline** - Only if 2D proves limiting (unlikely)

---

## Design Constraints

- **Performance**: Handle 44K events, 3.9K files without lag
- **Responsive**: Work on laptop screens (1366×768 minimum)
- **Load time**: Initial render < 3 seconds
- **Data processing**: Pre-aggregate in CSV, minimal runtime computation
- **Browser support**: Modern Chrome/Firefox/Safari (no IE)

---

## Evaluation Criteria (Fail-Fast)

After building Timeline Heatmap prototype:
- **Can new developers identify the most active parts of the codebase?** (Yes/No)
- **Can they spot when major refactors occurred?** (Yes/No)
- **Is interaction intuitive without documentation?** (Yes/No)
- **Does it reveal insights not obvious from GitHub?** (Yes/No)

**If 3+ = Yes**: Proceed to Treemap
**If 2 or fewer = Yes**: Rethink approach or adjust Heatmap design

---

## Technical Stack Recommendation

- **Framework**: React (for state management)
- **Visualization**: D3.js or Recharts (Timeline), D3-hierarchy (Treemap)
- **Data**: Load CSVs directly, use PapaParse
- **Storage**: None needed - stateless visualization
- **Performance**: Canvas rendering for heatmap if >10K cells

---

## Key Decisions to Defer

- Exact color schemes (test with users)
- Animation duration/easing (requires prototyping)
- Mobile support (desktop-first)
- Advanced filtering UI (start with simple dropdowns)
- Statistical overlays (correlation, trends) - add only if requested

**Next Step**: Build Timeline Heatmap prototype with file_lifecycle_events.csv, test with 2-3 developers, iterate or pivot based on feedback.