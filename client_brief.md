# Git Repository Evolution Visualization: Requirements Document

## Project Overview

Create 2-3 interactive browser-based visualizations that reveal the structural and temporal evolution of the Excalidraw repository (2020-2025). The goal is to help developers onboard to the codebase and provide stakeholders with actionable insights about project history, development patterns, and team dynamics.

## Target Audience

- **Primary**: Developers onboarding to the Excalidraw codebase
- **Secondary**: Project stakeholders reviewing development history
- **Technical Level**: Assumes familiarity with software development concepts

## Data Sources

### Available Datasets

1. **file_lifecycle_events.csv** (9.84 MB, 44,074 rows)
   - Event-level granularity: every commit affecting every file
   - Columns: filepath, event_type, event_datetime, commit_hash, author_name, additions, deletions, commit_subject, etc.
2. **file_lifecycle_summary.csv** (1.05 MB, 3,892 rows)
   - File-level aggregates across entire repository history
   - Columns: filepath, created_at, total_commits, net_lines, unique_authors, is_deleted, etc.

### Data Characteristics

- 3,892 files tracked
- 44,074 lifecycle events
- Time span: 2020-2025
- No missing critical data (null values only in unused columns)

## Core Requirements

### Visualization 1: Timeline Heatmap (MUST HAVE - Priority 1)

#### Functional Requirements

- Display repository activity across time and directory structure
- Support temporal navigation (scrubbing, play/pause, date selection)
- Enable users to identify "hot zones" of development within 30 seconds
- Reveal major refactoring periods and dormant areas
- Show file lifecycle events (creations, deletions, major changes)

#### Visual Requirements

- **Y-axis**: Major directory paths (15-20 visible, with hierarchy)
- **X-axis**: Time dimension (adaptable bins: daily/weekly/monthly)
- **Cells**: Color-coded by intensity (commit frequency + line changes)
- **Markers**: Visual indicators for file creations (births), deletions (deaths), major refactors

#### Interaction Requirements

- Click cell → show commit details, affected files, authors
- Filter by: author, file type, directory
- Toggle metrics: commits vs. lines changed vs. unique authors
- Scrub timeline to see repository state at any historical point

#### Success Criteria

- User can identify most active codebase areas: YES/NO
- User can spot when major refactors occurred: YES/NO
- Interaction is intuitive without documentation: YES/NO
- Reveals insights not obvious from GitHub interface: YES/NO
- **Minimum 3/4 YES required to proceed to Visualization 2**

---

### Visualization 2: Repository Structure Treemap Animation (SHOULD HAVE - Priority 2)

#### Functional Requirements

- Show file/directory sizes and change frequency over time
- Animate transitions between time periods (quarterly snapshots recommended)
- Enable exploration of how codebase areas grew, shrank, or stabilized

#### Visual Requirements

- **Rectangles**: Size = file size (or line count), Color intensity = change frequency
- **Hierarchy**: Nested boxes representing directory structure
- **Animation**: Smooth morphing between time periods
- **Labels**: Visible on hover or for significant files/directories

#### Interaction Requirements

- Play/pause time-lapse (auto-advance through periods)
- Jump to specific date/period
- Click file → lifecycle statistics
- Toggle metrics: net_lines, total_commits, or file count

#### Success Criteria

- User can visually identify which codebase parts grew/shrank over time
- User can spot structural refactoring (file/directory reorganizations)

---

### Visualization 3: Contributor Flow Diagram (OPTIONAL - Priority 3)

#### Functional Requirements

- Show how authors contribute to different repository areas over time
- Identify primary contributors for any codebase area
- Spot ownership changes over time

#### Visual Requirements

- **Sankey-style flow diagram**
- **Left side**: Authors (ranked by contribution volume)
- **Right side**: Top 10-15 directories
- **Flows**: Curved bands showing commit volume (author → directory)
- **Color**: Consistent per-author coloring

#### Interaction Requirements

- Filter to specific time ranges
- Highlight specific author or directory
- Click flow → commit details
- Toggle between commits vs. lines changed

#### Success Criteria

- User can identify primary contributors for any area
- User can spot when ownership transferred between contributors

---

## Non-Functional Requirements

### Performance Constraints

- Handle 44K events and 3.9K files without perceptible lag
- Initial render < 3 seconds
- Smooth interactions (>30 fps for animations)
- Data processing should be pre-aggregated where possible to minimize runtime computation

### Compatibility Requirements

- **Screen Resolution**: Minimum 1366×768 (laptop standard)
- **Browsers**: Modern Chrome, Firefox, Safari (no IE11 support required)
- **Responsive**: Must work on laptop screens; mobile support deferred

### Usability Requirements

- Intuitive controls requiring no documentation for basic usage
- Visual feedback for all interactions (hover states, loading indicators)
- Consistent color schemes and interaction patterns across visualizations

### Data Processing Requirements

- Load CSVs efficiently (streaming/chunking if needed)
- Parse dates/timestamps correctly
- Handle edge cases: binary files, deleted files, renamed paths
- Minimal runtime aggregation (prefer pre-computed summaries)

---

## Scope Boundaries

### In Scope

- Desktop browser-based visualizations
- Basic filtering and temporal controls
- Export capabilities (screenshots or data subsets)
- Linked views (clicking in one updates others, if applicable)

### Out of Scope (Deferred)

- Mobile-specific interfaces
- Advanced statistical overlays (correlation analysis, trend prediction)
- Real-time data updates or live Git integration
- 3D visualizations (unless 2D proves insufficient)
- Detailed animation timing/easing (iterate based on user testing)
- User accounts or saved preferences

---

## Implementation Approach

### Development Strategy

**Fail-Fast Iteration**: Build Timeline Heatmap first, evaluate against success criteria, then decide whether to proceed with additional visualizations.

### Suggested Considerations (Non-Prescriptive)

- State management may be important for complex interactions
- Consider performance trade-offs: SVG vs. Canvas vs. WebGL
- CSV parsing libraries can simplify data loading
- Date/time libraries may help with temporal binning
- Component-based architecture could enable reusable controls

### Technical Constraints

- Must load and parse provided CSV files
- Cannot require server-side processing (client-side only)
- Should work offline after initial load (no external API dependencies during runtime)

---

## Evaluation & Decision Points

### Checkpoint 1: After Timeline Heatmap Prototype

**Evaluate against 4 success criteria** (see Visualization 1)

- **If 3+ YES**: Proceed to Treemap Animation
- **If ≤2 YES**: Rethink approach or refine Heatmap design

### Checkpoint 2: After Treemap Animation

- **Does Treemap add significant value beyond Heatmap?**
- **Are two visualizations sufficient for user needs?**
- **Decide**: Build Contributor Flow (Priority 3) or refine existing visualizations

### Checkpoint 3: Final Review

- Test with 2-3 developers from target audience
- Gather qualitative feedback on insights gained
- Identify any critical missing features

---

## Deliverables

1. **Functional Timeline Heatmap** (Visualization 1)
2. **Functional Treemap Animation** (Visualization 2, if Checkpoint 1 passes)
3. **Contributor Flow Diagram** (Visualization 3, if needed and time permits)
4. **Basic user documentation** (embedded tooltips/help text preferred)
5. **README or usage guide** (brief, focused on key interactions)

---

## Open Questions for Developer

1. **Output Format**: Single-page application? Multiple pages? Embedded widgets?
2. **Color Scheme**: Light/dark mode? Colorblind-friendly palettes?
3. **Time Binning Strategy**: How to handle sparse vs. dense periods?
4. **Directory Depth**: How many levels to show by default? Expand/collapse strategy?
5. **Performance Optimization**: When to use Canvas? How to handle large datasets?
6. **Export Format**: PNG/SVG screenshots? CSV data exports? Both?

---

## Key Decisions Deferred to Developer

- Technology stack (framework, libraries, rendering approach)
- Exact color schemes and visual styling
- Animation durations and easing functions
- Specific UI layout and control placement
- Data structure and state management patterns
- Build system and deployment approach
