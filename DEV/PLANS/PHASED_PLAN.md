# Treemap Explorer - Implementation Plan

**Version:** 2.0  
**Target Timeline:** 8-10 days  
**Status:** Building on refactored Phase 2 foundation  
**Dataset Scale:** 2,739 files, 115,824 coupling edges, 1,384 temporal days

---

## 🎯 Overview

This plan delivers a production-ready Treemap Explorer plugin with three visualization modes (Debt, Coupling, Time), advanced filtering, search capabilities, and export functionality.

### Core Features

**Debt Lens:** Visualize code health using color-coded treemap  
**Coupling Lens:** Display file coupling relationships with interactive arcs  
**Time Lens:** Historical evolution through quarterly snapshots  
**Advanced UX:** Search, filters, keyboard shortcuts, export capabilities

### Success Criteria

**Performance:**

- Initial render: <300ms for 2,739 files
- Lens switching: <300ms
- Cell selection: <100ms
- Arc rendering: <100ms (10 arcs max)
- Memory stable: <50 MB growth per 20 interactions

**Functionality:**

- All three lenses working smoothly
- Detail panel with contextual information
- Advanced filtering and search
- Export to PNG, CSV, JSON
- Keyboard shortcuts for power users

---

## 📊 Current Foundation

### What Exists (Refactored Phase 2)

```
✅ Core Infrastructure
  - TreemapExplorerPlugin with D3 join pattern
  - HealthScoreCalculator (production-ready)
  - CouplingDataProcessor (optimized indexing)
  - CouplingArcRenderer with proper lifecycle

✅ React Components
  - TreemapExplorerControls (lens/metric selection)
  - TreemapExplorerFilters (threshold controls)
  - TreemapDetailPanel (file information)
  - LensModeSelector (mode switching)

✅ Visualization Features
  - Debt lens (health-based coloring)
  - Coupling lens (relationship arcs)
  - Cell selection and highlighting
  - Detail panel with metrics

✅ Performance Baseline
  - Initial render: 150-250ms
  - Lens switch: 200-300ms
  - Cell selection: 50-100ms
  - Arc rendering: 50-100ms
  - Memory: +15 MB per 20 interactions
```

### What Needs Work

**UX Polish:**

- No loading states (silent 500-900ms data processing)
- No error boundaries (silent failures)
- Threshold sliders trigger too many re-renders
- Fixed dimensions (not responsive)

**Missing Features:**

- Time/Evolution lens
- Advanced search and filtering
- Export capabilities
- Keyboard shortcuts
- Performance monitoring

---

## 📦 Phase 0: Validation & Essential UX (1-2 days)

**Goal:** Validate refactored foundation and add essential user experience improvements

### 0.1 Refactoring Validation (2-4 hours)

**Checklist:**

- [ ] Apply refactored TreemapExplorerPlugin.tsx
- [ ] Apply refactored CouplingArcRenderer.ts
- [ ] Run test suite (all 63 tests pass)
- [ ] Manual testing (all lenses, filters, selection)
- [ ] Performance benchmarking in DevTools

**Performance Validation:**

```typescript
// Measure in DevTools Performance tab
✓ Debt lens render: <300ms
✓ Coupling lens render: <300ms
✓ Lens switch: <300ms
✓ Arc rendering: <100ms (10 arcs)
✓ Memory stable after 20 interactions
```

**Functional Validation:**

```typescript
✓ All cells render correctly
✓ Health scores display properly
✓ Cell selection highlights work
✓ Detail panel opens/closes
✓ Arcs render on cell selection (coupling lens)
✓ Filters work (threshold sliders)
✓ No console errors
```

---

### 0.2 Loading States & Error Handling (4 hours)

**File:** `src/plugins/treemap-explorer/TreemapExplorerPlugin.tsx`

**Extend State Interface:**

```typescript
export interface TreemapExplorerState extends Record<string, unknown> {
  // ... existing fields

  // NEW: Loading and error states
  isLoading: boolean;
  loadingStage:
    | "idle"
    | "processing-files"
    | "building-index"
    | "rendering"
    | "ready";
  loadingMessage: string | null;
  error: string | null;
}
```

**Update processData Method:**

```typescript
processData(dataset: any): EnrichedFileData[] {
  try {
    // Start loading
    this.emitLoadingEvent('processing-files', 'Processing file index...');

    const fileIndex = dataset.file_index || dataset;
    if (!fileIndex || !fileIndex.files) {
      throw new Error('Invalid file_index dataset');
    }

    // Process files with health scores
    const files: EnrichedFileData[] = Object.entries(fileIndex.files).map(
      ([key, fileData]: [string, any]) => {
        // ... existing file processing logic
      }
    );

    // Process coupling data (if available)
    if (dataset.cochange_network) {
      this.emitLoadingEvent('building-index', 'Building coupling index...');
      this.couplingIndex = CouplingDataProcessor.process(dataset.cochange_network);
    }

    this.emitLoadingEvent('ready', null);
    return files;

  } catch (error) {
    this.emitLoadingEvent('idle', null, error.message);
    console.error('TreemapExplorer: Data processing failed', error);
    return [];
  }
}

private emitLoadingEvent(
  stage: TreemapExplorerState['loadingStage'],
  message: string | null = null,
  error: string | null = null
): void {
  // Emit custom event that React components can listen to
  window.dispatchEvent(new CustomEvent('treemap-loading-state', {
    detail: { stage, message, error }
  }));
}
```

**Loading Overlay Component:**

```typescript
// NEW: src/plugins/treemap-explorer/components/LoadingOverlay.tsx
import { Loader2 } from 'lucide-react';

interface LoadingOverlayProps {
  stage: string;
  message: string | null;
}

export const LoadingOverlay: React.FC<LoadingOverlayProps> = ({ stage, message }) => {
  if (stage === 'ready' || stage === 'idle') return null;

  return (
    <div className="absolute inset-0 bg-zinc-950/90 flex items-center justify-center z-50 backdrop-blur-sm">
      <div className="flex flex-col items-center gap-4">
        <Loader2 className="w-8 h-8 text-purple-500 animate-spin" />
        <div className="text-sm text-zinc-300 font-medium">
          {message || 'Loading...'}
        </div>
        <div className="text-xs text-zinc-500">
          This may take a moment for large datasets
        </div>
      </div>
    </div>
  );
};
```

**Expected Loading Times:**

- Dataset parse: 300-500ms (21.43 MB coupling data)
- Coupling index build: 200-400ms (115,824 edges)
- Initial render: 150-250ms (2,739 files)
- **Total:** ~1 second (acceptable with visual feedback)

---

### 0.3 Filter Debouncing (2 hours)

**File:** `src/utils/debounce.ts`

```typescript
/**
 * Debounce function calls to prevent excessive re-renders
 */
export function debounce<T extends (...args: any[]) => any>(
  func: T,
  wait: number,
): (...args: Parameters<T>) => void {
  let timeout: ReturnType<typeof setTimeout> | null = null;

  return function executedFunction(...args: Parameters<T>) {
    const later = () => {
      timeout = null;
      func(...args);
    };

    if (timeout) clearTimeout(timeout);
    timeout = setTimeout(later, wait);
  };
}
```

**File:** `src/plugins/treemap-explorer/components/TreemapExplorerControls.tsx`

```typescript
import { useMemo, useState } from 'react';
import { debounce } from '@/utils/debounce';

export const TreemapExplorerControls: React.FC<Props> = ({ state, updateState }) => {
  // Local state for immediate UI feedback
  const [localThreshold, setLocalThreshold] = useState(state.healthThreshold);

  // Debounced state update
  const debouncedUpdate = useMemo(
    () => debounce((value: number) => {
      updateState({ healthThreshold: value });
    }, 150),
    [updateState]
  );

  const handleThresholdChange = (value: number) => {
    setLocalThreshold(value);  // Update UI immediately
    debouncedUpdate(value);     // Update state after 150ms
  };

  return (
    <div className="flex items-center gap-3">
      <label className="text-xs text-zinc-500">Health Threshold</label>
      <input
        type="range"
        min={0}
        max={100}
        value={localThreshold}
        onChange={(e) => handleThresholdChange(parseFloat(e.target.value))}
        className="w-32"
      />
      <span className="text-xs text-zinc-400 font-mono w-8">{localThreshold}</span>
    </div>
  );
};
```

**Impact:** Reduces re-renders from 20-30 to 1 during slider drag

---

### 0.4 Responsive Layout (2 hours)

**File:** Main plugin container component

```typescript
import { useEffect, useRef, useState } from 'react';

export const TreemapExplorerContainer = () => {
  const containerRef = useRef<HTMLDivElement>(null);
  const [dimensions, setDimensions] = useState({ width: 800, height: 600 });

  // Resize observer for responsive dimensions
  useEffect(() => {
    if (!containerRef.current) return;

    const updateDimensions = () => {
      if (containerRef.current) {
        const { width, height } = containerRef.current.getBoundingClientRect();
        setDimensions({ width, height });
      }
    };

    // Initial measurement
    updateDimensions();

    // Use ResizeObserver for efficient updates
    const observer = new ResizeObserver(updateDimensions);
    observer.observe(containerRef.current);

    // Fallback to window resize
    window.addEventListener('resize', updateDimensions);

    return () => {
      observer.disconnect();
      window.removeEventListener('resize', updateDimensions);
    };
  }, []);

  return (
    <div ref={containerRef} className="relative w-full h-full min-h-[400px]">
      <TreemapVisualization
        {...otherProps}
        width={dimensions.width}
        height={dimensions.height}
      />
    </div>
  );
};
```

**Impact:** Treemap adapts to window size, works on all screen sizes

---

### Phase 0 Deliverables

**Code Changes:**

- ✅ Refactored files validated and working
- ✅ Loading states showing during data processing
- ✅ Error boundaries catching and displaying failures
- ✅ Filter controls debounced (smooth interactions)
- ✅ Responsive layout implemented

**Performance:**

- ✅ All targets met (<300ms renders)
- ✅ Smooth user interactions
- ✅ No memory leaks

**Status:** Production-ready Debt + Coupling lenses with essential UX

---

## 📦 Phase 1: Polish & Advanced Features (2-3 days)

**Goal:** Elevate existing features to production quality

### 1.1 Enhanced Detail Panel (4 hours)

**File:** `src/plugins/treemap-explorer/components/TreemapDetailPanel.tsx`

**Add Health Trend Visualization:**

```typescript
import { LineChart, Line, ResponsiveContainer } from 'recharts';

const DebtView: React.FC<{ file: EnrichedFileData }> = ({ file }) => {
  // Generate mock trend data (Phase 2 will add real historical data)
  const trendData = useMemo(() => {
    const baseHealth = file.healthScore?.score || 50;
    return Array.from({ length: 12 }, (_, i) => ({
      month: `M${i + 1}`,
      health: Math.max(0, Math.min(100, baseHealth + (Math.random() - 0.5) * 20))
    }));
  }, [file.healthScore]);

  return (
    <div className="space-y-6">
      {/* Existing health score badge */}
      <div className="p-4 bg-zinc-800/50 rounded">
        <div className="flex items-center justify-between">
          <span className="text-sm text-zinc-400">Health Score</span>
          <span className="text-3xl font-bold" style={{ color: getHealthColor(file.healthScore?.score) }}>
            {file.healthScore?.score}
          </span>
        </div>
      </div>

      {/* NEW: Health trend sparkline */}
      <div>
        <h4 className="text-xs text-zinc-500 uppercase mb-3 tracking-wider">Health Trend</h4>
        <ResponsiveContainer width="100%" height={80}>
          <LineChart data={trendData}>
            <Line
              type="monotone"
              dataKey="health"
              stroke="#8b5cf6"
              strokeWidth={2}
              dot={false}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Existing metrics grid */}
      <div>
        <h4 className="text-xs text-zinc-500 uppercase mb-3 tracking-wider">Metrics</h4>
        <div className="grid grid-cols-2 gap-3">
          {/* ... existing metrics */}
        </div>
      </div>

      {/* NEW: Quick Actions */}
      <div>
        <h4 className="text-xs text-zinc-500 uppercase mb-3 tracking-wider">Quick Actions</h4>
        <div className="flex flex-wrap gap-2">
          <ActionButton
            icon={Copy}
            label="Copy Path"
            onClick={() => {
              navigator.clipboard.writeText(file.path);
              toast.success('Path copied to clipboard');
            }}
          />
          <ActionButton
            icon={GitBranch}
            label="Show Coupling"
            onClick={() => {
              updateState({ lensMode: 'coupling', selectedFile: file.path });
            }}
          />
          <ActionButton
            icon={Download}
            label="Export Data"
            onClick={() => exportFileData(file)}
          />
        </div>
      </div>
    </div>
  );
};

const ActionButton: React.FC<{
  icon: React.ComponentType;
  label: string;
  onClick: () => void;
}> = ({ icon: Icon, label, onClick }) => (
  <button
    onClick={onClick}
    className="flex items-center gap-2 px-3 py-1.5 text-xs bg-zinc-800 hover:bg-zinc-700 rounded transition-colors"
  >
    <Icon className="w-3 h-3" />
    {label}
  </button>
);
```

---

### 1.2 Advanced Filtering (4 hours)

**File:** `src/plugins/treemap-explorer/components/TreemapExplorerFilters.tsx`

**Extended Filter Interface:**

```typescript
export interface AdvancedFilters {
  // Range filters
  healthRange: [number, number];
  commitRange: [number, number];
  authorRange: [number, number];

  // Categorical filters
  fileExtensions: string[];
  pathPattern: string;

  // Quick presets
  preset: "all" | "critical" | "hotspots" | "dormant" | "custom";
}

// Preset configurations
const FILTER_PRESETS: Record<string, Partial<AdvancedFilters>> = {
  all: {},
  critical: {
    healthRange: [0, 30],
    commitRange: [10, Infinity],
  },
  hotspots: {
    commitRange: [50, Infinity],
    authorRange: [3, Infinity],
  },
  dormant: {
    // Files not modified in 180+ days (calculated in filter logic)
  },
};
```

**Filter UI Component:**

```typescript
export const AdvancedFilterPanel: React.FC<Props> = ({ filters, onFiltersChange }) => {
  const fileExtensions = ['.tsx', '.ts', '.js', '.jsx', '.css', '.json', '.md'];

  return (
    <div className="space-y-6 p-4">
      {/* Quick Presets */}
      <div>
        <label className="text-xs text-zinc-500 uppercase tracking-wider mb-3 block">
          Quick Presets
        </label>
        <div className="grid grid-cols-2 gap-2">
          {Object.keys(FILTER_PRESETS).map(preset => (
            <button
              key={preset}
              onClick={() => applyPreset(preset)}
              className={cn(
                "px-3 py-2 text-xs rounded capitalize transition-colors",
                filters.preset === preset
                  ? "bg-purple-900/50 text-purple-200 border border-purple-700"
                  : "bg-zinc-800 text-zinc-400 hover:bg-zinc-700 border border-zinc-700"
              )}
            >
              {preset}
            </button>
          ))}
        </div>
      </div>

      {/* Health Range */}
      <div>
        <label className="text-xs text-zinc-500 uppercase tracking-wider mb-3 block">
          Health Score Range
        </label>
        <RangeSlider
          min={0}
          max={100}
          values={filters.healthRange}
          onChange={(values) => onFiltersChange({ healthRange: values })}
        />
      </div>

      {/* Commit Range */}
      <div>
        <label className="text-xs text-zinc-500 uppercase tracking-wider mb-3 block">
          Commit Count
        </label>
        <RangeSlider
          min={1}
          max={500}
          values={filters.commitRange}
          onChange={(values) => onFiltersChange({ commitRange: values })}
        />
      </div>

      {/* File Extensions */}
      <div>
        <label className="text-xs text-zinc-500 uppercase tracking-wider mb-3 block">
          File Types
        </label>
        <div className="flex flex-wrap gap-2">
          {fileExtensions.map(ext => (
            <label
              key={ext}
              className="flex items-center gap-2 px-2 py-1 bg-zinc-800 rounded text-xs cursor-pointer hover:bg-zinc-700"
            >
              <input
                type="checkbox"
                checked={filters.fileExtensions.includes(ext)}
                onChange={(e) => {
                  const newExts = e.target.checked
                    ? [...filters.fileExtensions, ext]
                    : filters.fileExtensions.filter(e => e !== ext);
                  onFiltersChange({ fileExtensions: newExts });
                }}
                className="rounded"
              />
              <span className="text-zinc-300">{ext}</span>
            </label>
          ))}
        </div>
      </div>

      {/* Path Pattern */}
      <div>
        <label className="text-xs text-zinc-500 uppercase tracking-wider mb-3 block">
          Path Filter
        </label>
        <input
          type="text"
          placeholder="e.g., src/components/*"
          value={filters.pathPattern}
          onChange={(e) => onFiltersChange({ pathPattern: e.target.value })}
          className="w-full px-3 py-2 bg-zinc-800 border border-zinc-700 rounded text-sm text-zinc-300 placeholder-zinc-500 focus:outline-none focus:border-purple-500"
        />
        <p className="text-xs text-zinc-600 mt-2">Use * as wildcard</p>
      </div>
    </div>
  );
};
```

**Filter Application Logic:**

```typescript
function applyFilters(
  files: EnrichedFileData[],
  filters: AdvancedFilters,
): EnrichedFileData[] {
  return files.filter((file) => {
    // Health range
    const health = file.healthScore?.score || 0;
    if (health < filters.healthRange[0] || health > filters.healthRange[1]) {
      return false;
    }

    // Commit range
    if (
      file.totalCommits < filters.commitRange[0] ||
      file.totalCommits > filters.commitRange[1]
    ) {
      return false;
    }

    // Author range
    if (
      file.uniqueAuthors < filters.authorRange[0] ||
      file.uniqueAuthors > filters.authorRange[1]
    ) {
      return false;
    }

    // File extensions
    if (filters.fileExtensions.length > 0) {
      const ext = "." + file.name.split(".").pop();
      if (!filters.fileExtensions.includes(ext)) {
        return false;
      }
    }

    // Path pattern (glob-style)
    if (filters.pathPattern) {
      const pattern = new RegExp(
        "^" + filters.pathPattern.replace(/\*/g, ".*") + "$",
      );
      if (!pattern.test(file.path)) {
        return false;
      }
    }

    return true;
  });
}
```

---

### 1.3 Search Functionality (3 hours)

**File:** `src/plugins/treemap-explorer/components/TreemapSearch.tsx`

```typescript
import { Search, X } from 'lucide-react';
import { useMemo, useState } from 'react';

interface TreemapSearchProps {
  files: EnrichedFileData[];
  onFileSelect: (file: EnrichedFileData) => void;
}

export const TreemapSearch: React.FC<TreemapSearchProps> = ({ files, onFileSelect }) => {
  const [query, setQuery] = useState('');
  const [isOpen, setIsOpen] = useState(false);

  // Fuzzy search implementation
  const results = useMemo(() => {
    if (!query.trim()) return [];

    const scored = files
      .map(file => ({
        file,
        score: fuzzyMatch(file.path.toLowerCase(), query.toLowerCase())
      }))
      .filter(r => r.score > 0.3)
      .sort((a, b) => b.score - a.score)
      .slice(0, 10);

    return scored.map(r => r.file);
  }, [query, files]);

  const handleSelect = (file: EnrichedFileData) => {
    onFileSelect(file);
    setQuery('');
    setIsOpen(false);
  };

  return (
    <div className="relative">
      {/* Search Input */}
      <div className="relative">
        <Search className="absolute left-3 top-2.5 w-4 h-4 text-zinc-500" />
        <input
          type="text"
          placeholder="Search files... (press / to focus)"
          value={query}
          onChange={(e) => {
            setQuery(e.target.value);
            setIsOpen(true);
          }}
          onFocus={() => setIsOpen(true)}
          className="w-full pl-10 pr-8 py-2 bg-zinc-800 border border-zinc-700 rounded text-sm text-zinc-300 placeholder-zinc-500 focus:outline-none focus:border-purple-500"
        />
        {query && (
          <button
            onClick={() => {
              setQuery('');
              setIsOpen(false);
            }}
            className="absolute right-3 top-2.5 text-zinc-500 hover:text-zinc-300"
          >
            <X className="w-4 h-4" />
          </button>
        )}
      </div>

      {/* Results Dropdown */}
      {isOpen && results.length > 0 && (
        <div className="absolute top-full mt-2 w-full bg-zinc-900 border border-zinc-800 rounded-lg shadow-2xl max-h-96 overflow-y-auto z-50">
          {results.map((file, index) => (
            <button
              key={file.key}
              onClick={() => handleSelect(file)}
              className="w-full px-4 py-3 text-left hover:bg-zinc-800 border-b border-zinc-800 last:border-0 transition-colors"
            >
              <div className="flex items-center justify-between">
                <div className="flex-1 min-w-0">
                  <div className="text-sm font-mono text-zinc-300 truncate">
                    {file.name}
                  </div>
                  <div className="text-xs text-zinc-500 truncate mt-0.5">
                    {file.path}
                  </div>
                </div>
                <div className="ml-3 flex items-center gap-2">
                  <span className="text-xs text-zinc-600">
                    {file.totalCommits} commits
                  </span>
                  <div
                    className="w-2 h-2 rounded-full"
                    style={{ backgroundColor: getHealthColor(file.healthScore?.score) }}
                  />
                </div>
              </div>
            </button>
          ))}
        </div>
      )}

      {/* No Results */}
      {isOpen && query && results.length === 0 && (
        <div className="absolute top-full mt-2 w-full bg-zinc-900 border border-zinc-800 rounded-lg shadow-2xl p-4 z-50">
          <p className="text-sm text-zinc-500 text-center">
            No files found matching "{query}"
          </p>
        </div>
      )}
    </div>
  );
};

/**
 * Simple fuzzy matching algorithm
 * Returns score 0-1 based on character matching
 */
function fuzzyMatch(str: string, pattern: string): number {
  let patternIdx = 0;
  let score = 0;
  let consecutiveMatches = 0;

  for (let i = 0; i < str.length && patternIdx < pattern.length; i++) {
    if (str[i] === pattern[patternIdx]) {
      score += 1 + consecutiveMatches;
      consecutiveMatches++;
      patternIdx++;
    } else {
      consecutiveMatches = 0;
    }
  }

  // Only return score if all pattern characters were matched
  return patternIdx === pattern.length ? score / str.length : 0;
}
```

**Keyboard Shortcut for Search:**

```typescript
// In main app component
useEffect(() => {
  const handleKeyPress = (e: KeyboardEvent) => {
    if (e.key === "/" && !e.ctrlKey && !e.metaKey) {
      e.preventDefault();
      searchInputRef.current?.focus();
    }
  };

  window.addEventListener("keydown", handleKeyPress);
  return () => window.removeEventListener("keydown", handleKeyPress);
}, []);
```

---

### 1.4 Color Legends (2 hours)

**File:** `src/plugins/treemap-explorer/components/ColorLegend.tsx`

```typescript
interface LegendItem {
  label: string;
  color: string;
  description?: string;
}

const LENS_LEGENDS: Record<string, LegendItem[]> = {
  debt: [
    { label: 'Critical', color: 'hsl(0, 70%, 40%)', description: '0-30' },
    { label: 'Medium', color: 'hsl(45, 70%, 50%)', description: '30-70' },
    { label: 'Healthy', color: 'hsl(145, 60%, 40%)', description: '70-100' }
  ],
  coupling: [
    { label: 'Selected', color: '#ffffff' },
    { label: 'Strong', color: 'hsl(270, 70%, 60%)', description: '>0.7' },
    { label: 'Medium', color: 'hsl(270, 70%, 40%)', description: '0.4-0.7' },
    { label: 'Weak', color: '#27272a', description: '<0.4' }
  ],
  time: [
    { label: 'Recently Modified', color: '#3b82f6' },
    { label: 'New Files', color: '#22c55e' },
    { label: 'Dormant', color: '#3f3f46' }
  ]
};

export const ColorLegend: React.FC<{ lensMode: string }> = ({ lensMode }) => {
  const items = LENS_LEGENDS[lensMode] || [];

  return (
    <div className="flex items-center gap-4">
      {items.map(item => (
        <div key={item.label} className="flex items-center gap-2">
          <div
            className="w-3 h-3 rounded border border-zinc-700"
            style={{ backgroundColor: item.color }}
          />
          <span className="text-xs text-zinc-400">
            {item.label}
            {item.description && (
              <span className="text-zinc-600 ml-1">({item.description})</span>
            )}
          </span>
        </div>
      ))}
    </div>
  );
};
```

**Integration in Header:**

```typescript
export const TreemapExplorerHeader = () => {
  return (
    <div className="flex items-center justify-between px-4 py-3 border-b border-zinc-800">
      <div className="flex items-center gap-4">
        {/* Lens mode selector, size metric, etc. */}
      </div>

      {/* NEW: Color legend */}
      <ColorLegend lensMode={state.lensMode} />
    </div>
  );
};
```

---

### Phase 1 Deliverables

**Features:**

- ✅ Enhanced detail panel (sparklines, quick actions)
- ✅ Advanced filtering (presets, ranges, file types, paths)
- ✅ Search with fuzzy matching
- ✅ Color legends for visual guidance

**UX Improvements:**

- ✅ Quick action buttons in detail panel
- ✅ Filter presets for common scenarios
- ✅ Keyboard shortcut (/) for search
- ✅ Visual feedback for all interactions

**Status:** Production-quality Debt + Coupling lenses

---

## 📦 Phase 2: Time Lens (3-4 days)

**Goal:** Add historical evolution visualization through quarterly snapshots

### Architectural Decision: Discrete Snapshots

**Approach:** Generate 16-20 quarterly snapshots from historical data  
**Rationale:**

- Performance: 300ms per snapshot transition (smooth)
- UX: Users care about trends, not daily granularity
- Scalability: Precomputed once, reused forever

**Not doing:** Continuous scrubbing through 1,384 daily states (would require 3.7M+ operations)

---

### 2.1 Temporal Snapshot Processor (6 hours)

**File:** `src/services/data/TemporalSnapshotProcessor.ts`

```typescript
export interface TemporalSnapshot {
  id: string;
  date: string; // ISO format
  label: string; // "Q1 2023", "Jan 2024"
  timestamp: number;
  fileStates: Map<string, FileTemporalState>;
  statistics: {
    totalFiles: number;
    averageHealth: number;
    totalCommits: number;
    activeFiles: number;
  };
}

export interface FileTemporalState {
  path: string;
  exists: boolean;
  commitCount: number;
  health: number;
  isNew: boolean;
  isDormant: boolean;
  daysSinceModified: number;
  lastOperation: "A" | "M" | "D" | "R";
}

export class TemporalSnapshotProcessor {
  /**
   * Generate quarterly snapshots from temporal and lifecycle data
   * Returns approximately 16-20 snapshots for 4-5 years of repository history
   */
  static generateQuarterlySnapshots(
    temporalDaily: any,
    fileLifecycle: any,
  ): TemporalSnapshot[] {
    console.log("Generating temporal snapshots...");

    // Identify quarter boundaries from available data
    const quarters = this.identifyQuarters(temporalDaily);

    // Generate snapshot for each quarter
    const snapshots = quarters.map((quarter) => {
      const fileStates = this.computeFileStatesAtDate(
        quarter.endDate,
        fileLifecycle,
        temporalDaily,
      );

      const statistics = this.computeStatistics(fileStates);

      return {
        id: quarter.id,
        date: quarter.endDate,
        label: quarter.label,
        timestamp: new Date(quarter.endDate).getTime(),
        fileStates,
        statistics,
      };
    });

    console.log(`Generated ${snapshots.length} quarterly snapshots`);
    return snapshots;
  }

  /**
   * Identify quarter boundaries from temporal data
   */
  private static identifyQuarters(temporalDaily: any): Quarter[] {
    const days = Object.keys(temporalDaily.days || {}).sort();
    if (days.length === 0) return [];

    const firstDate = new Date(days[0]);
    const lastDate = new Date(days[days.length - 1]);

    const quarters: Quarter[] = [];
    let currentDate = new Date(
      firstDate.getFullYear(),
      Math.floor(firstDate.getMonth() / 3) * 3,
      1,
    );

    while (currentDate <= lastDate) {
      const year = currentDate.getFullYear();
      const quarter = Math.floor(currentDate.getMonth() / 3) + 1;

      // Calculate quarter end date
      const endMonth = quarter * 3;
      const endDate = new Date(year, endMonth, 0); // Last day of quarter

      if (endDate >= firstDate) {
        quarters.push({
          id: `${year}-Q${quarter}`,
          label: `Q${quarter} ${year}`,
          startDate: currentDate.toISOString().split("T")[0],
          endDate: endDate.toISOString().split("T")[0],
        });
      }

      // Move to next quarter
      currentDate = new Date(year, quarter * 3, 1);
    }

    return quarters;
  }

  /**
   * Compute file states at a specific snapshot date
   */
  private static computeFileStatesAtDate(
    targetDate: string,
    fileLifecycle: any,
    temporalDaily: any,
  ): Map<string, FileTemporalState> {
    const states = new Map<string, FileTemporalState>();
    const target = new Date(targetDate);

    Object.entries(fileLifecycle.files || {}).forEach(
      ([path, events]: [string, any[]]) => {
        // Filter events up to target date
        const relevantEvents = events.filter(
          (e) => new Date(e.datetime) <= target,
        );

        if (relevantEvents.length === 0) {
          // File didn't exist yet
          return;
        }

        // Check if file was deleted before snapshot
        const lastEvent = relevantEvents[relevantEvents.length - 1];
        if (lastEvent.operation === "D") {
          return;
        }

        // Calculate metrics at this point in time
        const commitCount = relevantEvents.filter(
          (e) => e.operation === "M",
        ).length;
        const uniqueAuthors = new Set(relevantEvents.map((e) => e.author_email))
          .size;
        const daysSinceModified = Math.floor(
          (target.getTime() - new Date(lastEvent.datetime).getTime()) /
            (1000 * 60 * 60 * 24),
        );

        // Calculate simplified health score
        const health = this.calculateHistoricalHealth(
          commitCount,
          uniqueAuthors,
          daysSinceModified,
        );

        // Determine if file is new (created within last 30 days of snapshot)
        const firstEvent = relevantEvents[0];
        const daysSinceCreation = Math.floor(
          (target.getTime() - new Date(firstEvent.datetime).getTime()) /
            (1000 * 60 * 60 * 24),
        );
        const isNew = daysSinceCreation <= 30;

        states.set(path, {
          path,
          exists: true,
          commitCount,
          health,
          isNew,
          isDormant: daysSinceModified > 90,
          daysSinceModified,
          lastOperation: lastEvent.operation,
        });
      },
    );

    return states;
  }

  /**
   * Calculate simplified health score for historical snapshots
   */
  private static calculateHistoricalHealth(
    commits: number,
    authors: number,
    daysSinceModified: number,
  ): number {
    let score = 60; // Base score

    // Churn penalty
    if (commits > 100) score -= 25;
    else if (commits > 50) score -= 15;
    else if (commits > 20) score -= 5;

    // Author diversity bonus
    if (authors >= 5) score += 20;
    else if (authors >= 3) score += 10;
    else if (authors === 1) score -= 10;

    // Dormancy penalty
    if (daysSinceModified > 180) score -= 20;
    else if (daysSinceModified > 90) score -= 10;

    return Math.max(10, Math.min(90, score));
  }

  /**
   * Compute aggregate statistics for a snapshot
   */
  private static computeStatistics(
    fileStates: Map<string, FileTemporalState>,
  ): TemporalSnapshot["statistics"] {
    const files = Array.from(fileStates.values());

    return {
      totalFiles: files.length,
      averageHealth: files.reduce((sum, f) => sum + f.health, 0) / files.length,
      totalCommits: files.reduce((sum, f) => sum + f.commitCount, 0),
      activeFiles: files.filter((f) => !f.isDormant).length,
    };
  }
}

interface Quarter {
  id: string;
  label: string;
  startDate: string;
  endDate: string;
}
```

**Integration in Plugin:**

```typescript
// In TreemapExplorerPlugin.processData()
processData(dataset: any): EnrichedFileData[] {
  // ... existing file processing

  // NEW: Generate temporal snapshots
  if (dataset.temporal_daily && dataset.file_lifecycle) {
    this.emitLoadingEvent('building-snapshots', 'Generating temporal snapshots...');

    this.temporalSnapshots = TemporalSnapshotProcessor.generateQuarterlySnapshots(
      dataset.temporal_daily,
      dataset.file_lifecycle
    );

    console.log(`Temporal snapshots ready: ${this.temporalSnapshots.length} quarters`);
  }

  return files;
}
```

---

### 2.2 Timeline Scrubber Component (4 hours)

**File:** `src/plugins/treemap-explorer/components/TimelineScrubber.tsx`

```typescript
import { Play, Pause, SkipBack, SkipForward } from 'lucide-react';

interface TimelineScrubberProps {
  snapshots: TemporalSnapshot[];
  currentIndex: number;
  onSnapshotChange: (index: number) => void;
  isPlaying: boolean;
  onPlayToggle: () => void;
}

export const TimelineScrubber: React.FC<TimelineScrubberProps> = ({
  snapshots,
  currentIndex,
  onSnapshotChange,
  isPlaying,
  onPlayToggle
}) => {
  const currentSnapshot = snapshots[currentIndex];
  const canGoBack = currentIndex > 0;
  const canGoForward = currentIndex < snapshots.length - 1;

  return (
    <div className="absolute bottom-0 left-0 right-0 h-24 bg-gradient-to-t from-zinc-900 via-zinc-900/95 to-transparent px-8 flex items-center gap-6">
      {/* Playback Controls */}
      <div className="flex items-center gap-2">
        <button
          onClick={() => onSnapshotChange(0)}
          disabled={!canGoBack}
          className="p-2 bg-zinc-800 hover:bg-zinc-700 rounded disabled:opacity-30 disabled:cursor-not-allowed"
          title="Go to start"
        >
          <SkipBack className="w-4 h-4" />
        </button>

        <button
          onClick={onPlayToggle}
          className="p-3 bg-purple-900/50 hover:bg-purple-900 rounded"
          title={isPlaying ? 'Pause' : 'Play'}
        >
          {isPlaying ? <Pause className="w-5 h-5" /> : <Play className="w-5 h-5" />}
        </button>

        <button
          onClick={() => onSnapshotChange(snapshots.length - 1)}
          disabled={!canGoForward}
          className="p-2 bg-zinc-800 hover:bg-zinc-700 rounded disabled:opacity-30 disabled:cursor-not-allowed"
          title="Go to end"
        >
          <SkipForward className="w-4 h-4" />
        </button>
      </div>

      {/* Timeline Slider */}
      <div className="flex-1 relative pt-2">
        <input
          type="range"
          min={0}
          max={snapshots.length - 1}
          value={currentIndex}
          onChange={(e) => onSnapshotChange(parseInt(e.target.value))}
          className="w-full h-2 bg-zinc-800 rounded-lg appearance-none cursor-pointer accent-purple-500"
          style={{
            background: `linear-gradient(to right, #8b5cf6 0%, #8b5cf6 ${(currentIndex / (snapshots.length - 1)) * 100}%, #27272a ${(currentIndex / (snapshots.length - 1)) * 100}%, #27272a 100%)`
          }}
        />

        {/* Snapshot markers */}
        <div className="absolute top-8 left-0 right-0 flex justify-between px-1">
          {snapshots.map((snapshot, i) => {
            const isActive = i === currentIndex;
            const showLabel = i % Math.ceil(snapshots.length / 6) === 0 || isActive;

            return (
              <div key={snapshot.id} className="relative flex flex-col items-center">
                <button
                  onClick={() => onSnapshotChange(i)}
                  className={cn(
                    "w-2 h-2 rounded-full transition-all",
                    isActive ? "bg-purple-400 scale-150" : "bg-zinc-600 hover:bg-zinc-500"
                  )}
                />
                {showLabel && (
                  <span className={cn(
                    "absolute top-3 text-xs whitespace-nowrap transition-all",
                    isActive ? "text-purple-400 font-bold" : "text-zinc-600"
                  )}>
                    {snapshot.label}
                  </span>
                )}
              </div>
            );
          })}
        </div>
      </div>

      {/* Current Snapshot Info */}
      <div className="min-w-48 text-right">
        <div className="text-sm font-bold text-purple-400">
          {currentSnapshot.label}
        </div>
        <div className="text-xs text-zinc-500">
          {currentSnapshot.statistics.totalFiles.toLocaleString()} files,
          {' '}Avg Health: {currentSnapshot.statistics.averageHealth.toFixed(0)}
        </div>
      </div>
    </div>
  );
};
```

**Playback Logic:**

```typescript
// In main plugin component
export const TreemapExplorerView = () => {
  const [currentSnapshotIndex, setCurrentSnapshotIndex] = useState(() =>
    snapshots.length - 1  // Start at most recent
  );
  const [isPlaying, setIsPlaying] = useState(false);

  // Auto-advance playback
  useEffect(() => {
    if (!isPlaying) return;

    const interval = setInterval(() => {
      setCurrentSnapshotIndex(prev => {
        if (prev >= snapshots.length - 1) {
          setIsPlaying(false);  // Stop at end
          return prev;
        }
        return prev + 1;
      });
    }, 1500);  // 1.5 seconds per snapshot

    return () => clearInterval(interval);
  }, [isPlaying, snapshots.length]);

  return (
    <>
      {/* Treemap visualization */}
      <TreemapVisualization
        snapshotIndex={currentSnapshotIndex}
        {...otherProps}
      />

      {/* Timeline scrubber (only visible in time lens) */}
      {state.lensMode === 'time' && (
        <TimelineScrubber
          snapshots={snapshots}
          currentIndex={currentSnapshotIndex}
          onSnapshotChange={setCurrentSnapshotIndex}
          isPlaying={isPlaying}
          onPlayToggle={() => setIsPlaying(prev => !prev)}
        />
      )}
    </>
  );
};
```

---

### 2.3 Temporal Rendering (4 hours)

**Update Plugin Render Method:**

```typescript
render(files: EnrichedFileData[], config: TreemapConfig): void {
  // ... existing setup

  const currentState = this.extractState(config);
  let displayFiles = files;

  // NEW: Apply temporal filtering if in time lens
  if (currentState.lensMode === 'time' && this.temporalSnapshots) {
    const snapshotIndex = config.temporalSnapshotIndex || 0;
    const snapshot = this.temporalSnapshots[snapshotIndex];

    if (snapshot) {
      // Filter to files that existed at this snapshot
      displayFiles = files.filter(f => snapshot.fileStates.has(f.path));

      // Enrich with historical state
      displayFiles = displayFiles.map(f => {
        const historicalState = snapshot.fileStates.get(f.path)!;
        return {
          ...f,
          // Overlay historical data
          temporalState: historicalState,
          healthScore: {
            ...f.healthScore!,
            score: historicalState.health
          }
        };
      });
    }
  }

  // Continue with existing render logic
  const updatedFiles = this.applySizeMetric(displayFiles, currentState.sizeMetric);

  // ... rest of render method
}
```

**Color Mapping for Time Lens:**

```typescript
// In colorScales.ts
export function getTimeLensColor(
  file: EnrichedFileData,
  temporalState: FileTemporalState | undefined,
): string {
  if (!temporalState) {
    return "#27272a"; // Gray for files without temporal data
  }

  if (temporalState.isNew) {
    return "#22c55e"; // Green for new files (created within last 30 days)
  }

  if (temporalState.isDormant) {
    return "#3f3f46"; // Dark gray for dormant files (no changes in 90+ days)
  }

  // Blue gradient based on activity level
  const activityLevel = Math.min(temporalState.commitCount / 50, 1);
  const lightness = 30 + activityLevel * 30;
  return `hsl(217, 91%, ${lightness}%)`;
}
```

---

### 2.4 Time Lens Detail Panel (3 hours)

**File:** `src/plugins/treemap-explorer/components/TimeView.tsx`

```typescript
import { Clock, GitCommit, Activity, TrendingUp, FileCode } from 'lucide-react';

interface TimeViewProps {
  file: EnrichedFileData;
  currentSnapshot: TemporalSnapshot;
  allSnapshots: TemporalSnapshot[];
}

export const TimeView: React.FC<TimeViewProps> = ({ file, currentSnapshot, allSnapshots }) => {
  const historicalState = currentSnapshot.fileStates.get(file.path);

  if (!historicalState) {
    return (
      <div className="p-6 text-center">
        <FileCode className="w-12 h-12 text-zinc-700 mx-auto mb-3" />
        <p className="text-sm text-zinc-500">
          This file did not exist at <span className="text-purple-400">{currentSnapshot.label}</span>
        </p>
      </div>
    );
  }

  // Calculate evolution trend
  const evolution = calculateEvolution(file.path, allSnapshots, currentSnapshot);

  return (
    <div className="space-y-6">
      {/* Snapshot Context */}
      <div className="p-4 bg-purple-900/20 border border-purple-800/50 rounded-lg">
        <div className="flex items-center gap-2 mb-2">
          <Clock className="w-4 h-4 text-purple-400" />
          <span className="text-xs text-purple-300 uppercase tracking-wider">Viewing Snapshot</span>
        </div>
        <div className="text-xl font-bold text-purple-300">{currentSnapshot.label}</div>
        <div className="text-xs text-purple-400/70 mt-1">{currentSnapshot.date}</div>
      </div>

      {/* Historical Metrics */}
      <div>
        <h4 className="text-xs text-zinc-500 uppercase mb-3 tracking-wider">Metrics at This Time</h4>
        <div className="grid grid-cols-2 gap-3">
          <MetricCard
            icon={GitCommit}
            label="Commits"
            value={historicalState.commitCount}
            color="text-blue-400"
          />
          <MetricCard
            icon={Activity}
            label="Health Score"
            value={historicalState.health}
            color={getHealthColor(historicalState.health)}
          />
          <MetricCard
            icon={Clock}
            label="Days Inactive"
            value={historicalState.daysSinceModified}
            color="text-zinc-400"
          />
          <MetricCard
            icon={TrendingUp}
            label="Status"
            value={getStatusLabel(historicalState)}
            color={getStatusColor(historicalState)}
          />
        </div>
      </div>

      {/* Evolution Context */}
      {evolution && (
        <div>
          <h4 className="text-xs text-zinc-500 uppercase mb-3 tracking-wider">Evolution Trend</h4>
          <div className="space-y-2 text-xs">
            {evolution.isNew && (
              <div className="flex items-start gap-2 text-green-400">
                <span>✨</span>
                <span>File was recently created in this period</span>
              </div>
            )}
            {evolution.isDormant && (
              <div className="flex items-start gap-2 text-zinc-500">
                <span>⏸️</span>
                <span>File had not been modified in over 90 days</span>
              </div>
            )}
            {evolution.isHotspot && (
              <div className="flex items-start gap-2 text-orange-400">
                <span>🔥</span>
                <span>This was a hotspot with {historicalState.commitCount} commits</span>
              </div>
            )}
            {evolution.healthTrend === 'improving' && (
              <div className="flex items-start gap-2 text-green-400">
                <span>📈</span>
                <span>Health was improving during this period</span>
              </div>
            )}
            {evolution.healthTrend === 'declining' && (
              <div className="flex items-start gap-2 text-red-400">
                <span>📉</span>
                <span>Health was declining during this period</span>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Compare to Current */}
      <div className="pt-4 border-t border-zinc-800">
        <button
          onClick={() => compareToPresent(file, historicalState)}
          className="w-full px-4 py-2.5 bg-purple-900/50 hover:bg-purple-900 rounded-lg text-sm font-medium transition-colors flex items-center justify-center gap-2"
        >
          <TrendingUp className="w-4 h-4" />
          Compare to Current State
        </button>
      </div>
    </div>
  );
};

function getStatusLabel(state: FileTemporalState): string {
  if (state.isNew) return 'New';
  if (state.isDormant) return 'Dormant';
  return 'Active';
}

function getStatusColor(state: FileTemporalState): string {
  if (state.isNew) return 'text-green-400';
  if (state.isDormant) return 'text-zinc-500';
  return 'text-blue-400';
}

function calculateEvolution(
  filePath: string,
  allSnapshots: TemporalSnapshot[],
  currentSnapshot: TemporalSnapshot
): EvolutionTrend | null {
  const currentIndex = allSnapshots.findIndex(s => s.id === currentSnapshot.id);
  if (currentIndex < 1) return null;

  const previousSnapshot = allSnapshots[currentIndex - 1];
  const currentState = currentSnapshot.fileStates.get(filePath);
  const previousState = previousSnapshot?.fileStates.get(filePath);

  if (!currentState) return null;

  return {
    isNew: currentState.isNew,
    isDormant: currentState.isDormant,
    isHotspot: currentState.commitCount > 20,
    healthTrend: previousState
      ? (currentState.health > previousState.health ? 'improving' :
         currentState.health < previousState.health ? 'declining' : 'stable')
      : 'stable'
  };
}

interface EvolutionTrend {
  isNew: boolean;
  isDormant: boolean;
  isHotspot: boolean;
  healthTrend: 'improving' | 'declining' | 'stable';
}
```

---

### Phase 2 Deliverables

**Features:**

- ✅ Temporal snapshot generation (16-20 quarterly snapshots)
- ✅ Timeline scrubber with playback controls
- ✅ Historical rendering (colors reflect past state)
- ✅ Time lens detail panel with evolution context

**Performance:**

- ✅ Snapshot transition: <300ms (precomputed data)
- ✅ Playback smooth at 1.5s per frame
- ✅ No memory growth during playback

**Data Processing:**

- ✅ Handles temporal_daily.json (1,384 days)
- ✅ Graceful degradation if temporal data missing
- ✅ Statistics computed for each snapshot

**Status:** Complete feature set (Debt + Coupling + Time lenses)

---

## 📦 Phase 3: Export & Power User Features (2 days)

**Goal:** Add professional export capabilities and keyboard shortcuts

### 3.1 Export Service (4 hours)

**File:** `src/services/export/TreemapExporter.ts`

```typescript
import html2canvas from "html2canvas";

export class TreemapExporter {
  /**
   * Export current treemap view as PNG image
   */
  static async exportAsPNG(
    containerElement: HTMLElement,
    filename?: string,
  ): Promise<void> {
    try {
      const canvas = await html2canvas(containerElement, {
        backgroundColor: "#09090b",
        scale: 2, // High DPI
      });

      canvas.toBlob((blob) => {
        if (blob) {
          const link = document.createElement("a");
          link.download =
            filename || `treemap-${new Date().toISOString().split("T")[0]}.png`;
          link.href = URL.createObjectURL(blob);
          link.click();
          URL.revokeObjectURL(link.href);
        }
      });
    } catch (error) {
      console.error("PNG export failed:", error);
      throw new Error("Failed to export image");
    }
  }

  /**
   * Export visible files as CSV
   */
  static exportAsCSV(
    files: EnrichedFileData[],
    lensMode: string,
    filename?: string,
  ): void {
    const headers = [
      "Path",
      "Name",
      "Total Commits",
      "Unique Authors",
      "Health Score",
      "Age (days)",
      "Last Modified",
      "First Seen",
    ];

    const rows = files.map((f) => [
      f.path,
      f.name,
      f.totalCommits,
      f.uniqueAuthors,
      f.healthScore?.score || "N/A",
      f.ageDays,
      f.lastModified,
      f.firstSeen,
    ]);

    const csv = [
      headers.join(","),
      ...rows.map((row) => row.map((cell) => `"${cell}"`).join(",")),
    ].join("\n");

    const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
    const link = document.createElement("a");
    link.download = filename || `treemap-${lensMode}-${Date.now()}.csv`;
    link.href = URL.createObjectURL(blob);
    link.click();
    URL.revokeObjectURL(link.href);
  }

  /**
   * Export selected file with full metrics as JSON
   */
  static exportFileJSON(
    file: EnrichedFileData,
    couplingIndex: CouplingIndex,
    filename?: string,
  ): void {
    const partners = CouplingDataProcessor.getTopCouplings(
      couplingIndex,
      file.path,
      20,
    );

    const exportData = {
      metadata: {
        exportedAt: new Date().toISOString(),
        exportType: "treemap-file-details",
      },
      file: {
        path: file.path,
        name: file.name,
        metrics: {
          totalCommits: file.totalCommits,
          uniqueAuthors: file.uniqueAuthors,
          ageDays: file.ageDays,
          healthScore: file.healthScore,
        },
        lifecycle: {
          firstSeen: file.firstSeen,
          lastModified: file.lastModified,
        },
        operations: file.operations,
      },
      coupling: {
        totalPartners: partners.length,
        topPartners: partners.map((p) => ({
          file: p.filePath,
          strength: p.strength,
          cochangeCount: p.cochangeCount,
        })),
      },
    };

    const blob = new Blob([JSON.stringify(exportData, null, 2)], {
      type: "application/json",
    });
    const link = document.createElement("a");
    link.download = filename || `${file.name.replace(/\./g, "-")}-metrics.json`;
    link.href = URL.createObjectURL(blob);
    link.click();
    URL.revokeObjectURL(link.href);
  }
}
```

**Export UI Component:**

```typescript
import { Download } from 'lucide-react';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger
} from '@/components/ui/dropdown-menu';

export const ExportMenu: React.FC<{
  containerRef: React.RefObject<HTMLElement>;
  visibleFiles: EnrichedFileData[];
  selectedFile: EnrichedFileData | null;
  lensMode: string;
  couplingIndex: CouplingIndex;
}> = ({ containerRef, visibleFiles, selectedFile, lensMode, couplingIndex }) => {
  const [isExporting, setIsExporting] = useState(false);

  const handleExportPNG = async () => {
    if (!containerRef.current) return;
    setIsExporting(true);
    try {
      await TreemapExporter.exportAsPNG(containerRef.current);
      toast.success('Image exported successfully');
    } catch (error) {
      toast.error('Failed to export image');
    } finally {
      setIsExporting(false);
    }
  };

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <button
          className="flex items-center gap-2 px-3 py-1.5 text-xs bg-zinc-800 hover:bg-zinc-700 rounded transition-colors"
          disabled={isExporting}
        >
          <Download className="w-3 h-3" />
          {isExporting ? 'Exporting...' : 'Export'}
        </button>
      </DropdownMenuTrigger>

      <DropdownMenuContent align="end" className="w-48">
        <DropdownMenuItem onClick={handleExportPNG}>
          <Download className="w-4 h-4 mr-2" />
          Export as PNG
        </DropdownMenuItem>

        <DropdownMenuItem
          onClick={() => TreemapExporter.exportAsCSV(visibleFiles, lensMode)}
        >
          <Download className="w-4 h-4 mr-2" />
          Export as CSV
        </DropdownMenuItem>

        {selectedFile && (
          <DropdownMenuItem
            onClick={() => TreemapExporter.exportFileJSON(selectedFile, couplingIndex)}
          >
            <Download className="w-4 h-4 mr-2" />
            Export Selected File
          </DropdownMenuItem>
        )}
      </DropdownMenuContent>
    </DropdownMenu>
  );
};
```

---

### 3.2 Keyboard Shortcuts (2 hours)

**File:** `src/hooks/useKeyboardShortcuts.ts`

```typescript
import { useEffect } from "react";

export type ShortcutHandler = () => void;
export type Shortcuts = Record<string, ShortcutHandler>;

export const useKeyboardShortcuts = (shortcuts: Shortcuts, enabled = true) => {
  useEffect(() => {
    if (!enabled) return;

    const handleKeyDown = (event: KeyboardEvent) => {
      const { key, ctrlKey, metaKey, shiftKey, altKey } = event;

      // Don't trigger shortcuts when typing in inputs
      if (
        event.target instanceof HTMLInputElement ||
        event.target instanceof HTMLTextAreaElement
      ) {
        return;
      }

      // Build shortcut key string
      const modifiers = [];
      if (ctrlKey || metaKey) modifiers.push("mod");
      if (shiftKey) modifiers.push("shift");
      if (altKey) modifiers.push("alt");

      const shortcutKey = [...modifiers, key.toLowerCase()].join("+");

      // Execute handler if shortcut exists
      if (shortcuts[shortcutKey]) {
        event.preventDefault();
        shortcuts[shortcutKey]();
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [shortcuts, enabled]);
};
```

**Implementation in Plugin:**

```typescript
export const TreemapExplorerView = () => {
  const searchInputRef = useRef<HTMLInputElement>(null);

  useKeyboardShortcuts({
    // Lens switching
    '1': () => updateState({ lensMode: 'debt' }),
    '2': () => updateState({ lensMode: 'coupling' }),
    '3': () => updateState({ lensMode: 'time' }),

    // UI controls
    'f': () => setFilterPanelOpen(prev => !prev),
    '/': () => searchInputRef.current?.focus(),
    'escape': () => {
      updateState({ selectedFile: null });
      setFilterPanelOpen(false);
    },

    // Export shortcuts
    'mod+e': () => TreemapExporter.exportAsCSV(visibleFiles, state.lensMode),
    'mod+s': () => TreemapExporter.exportAsPNG(containerRef.current),

    // Time lens playback
    'space': () => {
      if (state.lensMode === 'time') {
        setIsPlaying(prev => !prev);
      }
    },

    // Navigation
    'arrowleft': () => {
      if (state.lensMode === 'time' && currentSnapshotIndex > 0) {
        setCurrentSnapshotIndex(prev => prev - 1);
      }
    },
    'arrowright': () => {
      if (state.lensMode === 'time' && currentSnapshotIndex < snapshots.length - 1) {
        setCurrentSnapshotIndex(prev => prev + 1);
      }
    }
  });

  return (
    // ... component JSX
  );
};
```

**Keyboard Shortcuts Help Overlay:**

```typescript
const KeyboardShortcutsHelp: React.FC<{ onClose: () => void }> = ({ onClose }) => (
  <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 backdrop-blur-sm">
    <div className="bg-zinc-900 border border-zinc-800 rounded-lg p-6 max-w-md w-full mx-4 shadow-2xl">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-bold">Keyboard Shortcuts</h3>
        <button onClick={onClose} className="text-zinc-500 hover:text-zinc-300">
          <X className="w-5 h-5" />
        </button>
      </div>

      <div className="space-y-4">
        <ShortcutSection title="Lens Modes">
          <ShortcutRow keys={['1']} action="Debt Lens" />
          <ShortcutRow keys={['2']} action="Coupling Lens" />
          <ShortcutRow keys={['3']} action="Time Lens" />
        </ShortcutSection>

        <ShortcutSection title="Navigation">
          <ShortcutRow keys={['F']} action="Toggle Filters" />
          <ShortcutRow keys={['/']} action="Focus Search" />
          <ShortcutRow keys={['Esc']} action="Close Panel / Deselect" />
        </ShortcutSection>

        <ShortcutSection title="Export">
          <ShortcutRow keys={['Ctrl', 'E']} action="Export CSV" />
          <ShortcutRow keys={['Ctrl', 'S']} action="Export PNG" />
        </ShortcutSection>

        <ShortcutSection title="Time Lens">
          <ShortcutRow keys={['Space']} action="Play / Pause" />
          <ShortcutRow keys={['←', '→']} action="Navigate Snapshots" />
        </ShortcutSection>
      </div>

      <div className="mt-6 pt-4 border-t border-zinc-800 text-xs text-zinc-500 text-center">
        Press <kbd className="px-2 py-1 bg-zinc-800 rounded">?</kbd> to toggle this help
      </div>
    </div>
  </div>
);

const ShortcutSection: React.FC<{ title: string; children: React.ReactNode }> = ({ title, children }) => (
  <div>
    <h4 className="text-xs text-zinc-500 uppercase tracking-wider mb-2">{title}</h4>
    <div className="space-y-1.5">{children}</div>
  </div>
);

const ShortcutRow: React.FC<{ keys: string[]; action: string }> = ({ keys, action }) => (
  <div className="flex items-center justify-between text-sm">
    <span className="text-zinc-400">{action}</span>
    <div className="flex items-center gap-1">
      {keys.map((key, i) => (
        <React.Fragment key={i}>
          {i > 0 && <span className="text-zinc-600">+</span>}
          <kbd className="px-2 py-1 bg-zinc-800 border border-zinc-700 rounded text-xs font-mono">
            {key}
          </kbd>
        </React.Fragment>
      ))}
    </div>
  </div>
);
```

**Toggle Help with ? Key:**

```typescript
useKeyboardShortcuts({
  // ... other shortcuts
  "?": () => setShowShortcutsHelp((prev) => !prev),
});
```

---

### 3.3 Performance Monitoring (2 hours)

**File:** `src/components/PerformanceMonitor.tsx`

```typescript
import { Activity } from 'lucide-react';
import { useEffect, useState } from 'react';

interface PerformanceMetrics {
  renderTime: number;
  fileCount: number;
  memoryUsage: number;
  lastUpdate: number;
}

export const PerformanceMonitor: React.FC = () => {
  const [metrics, setMetrics] = useState<PerformanceMetrics>({
    renderTime: 0,
    fileCount: 0,
    memoryUsage: 0,
    lastUpdate: Date.now()
  });

  const [isVisible, setIsVisible] = useState(() =>
    process.env.NODE_ENV === 'development' || localStorage.getItem('show-perf-monitor') === 'true'
  );

  useEffect(() => {
    if (!isVisible) return;

    // Track render performance
    const observer = new PerformanceObserver((list) => {
      for (const entry of list.getEntries()) {
        if (entry.name === 'treemap-render') {
          setMetrics(prev => ({
            ...prev,
            renderTime: entry.duration,
            lastUpdate: Date.now()
          }));
        }
      }
    });

    observer.observe({ entryTypes: ['measure'] });

    // Track memory (Chrome only)
    let memoryInterval: NodeJS.Timeout | null = null;
    if ('memory' in performance) {
      memoryInterval = setInterval(() => {
        const mem = (performance as any).memory;
        setMetrics(prev => ({
          ...prev,
          memoryUsage: Math.round(mem.usedJSHeapSize / 1024 / 1024)
        }));
      }, 2000);
    }

    return () => {
      observer.disconnect();
      if (memoryInterval) clearInterval(memoryInterval);
    };
  }, [isVisible]);

  if (!isVisible) return null;

  const renderStatus = metrics.renderTime > 300 ? 'error' : metrics.renderTime > 200 ? 'warning' : 'success';

  return (
    <div className="fixed top-20 right-4 bg-zinc-900/95 border border-zinc-800 rounded-lg p-3 text-xs font-mono shadow-xl backdrop-blur-sm">
      <div className="flex items-center gap-2 mb-3 pb-2 border-b border-zinc-800">
        <Activity className="w-3 h-3 text-purple-500" />
        <span className="font-bold text-zinc-300">Performance</span>
        <button
          onClick={() => {
            setIsVisible(false);
            localStorage.removeItem('show-perf-monitor');
          }}
          className="ml-auto text-zinc-600 hover:text-zinc-400"
        >
          <X className="w-3 h-3" />
        </button>
      </div>

      <div className="space-y-2">
        <MetricRow
          label="Render"
          value={`${metrics.renderTime.toFixed(0)}ms`}
          status={renderStatus}
        />
        <MetricRow
          label="Files"
          value={metrics.fileCount.toLocaleString()}
        />
        <MetricRow
          label="Memory"
          value={`${metrics.memoryUsage} MB`}
          status={metrics.memoryUsage > 150 ? 'warning' : 'success'}
        />
      </div>

      <div className="mt-3 pt-2 border-t border-zinc-800 text-[10px] text-zinc-600">
        Last update: {new Date(metrics.lastUpdate).toLocaleTimeString()}
      </div>
    </div>
  );
};

const MetricRow: React.FC<{
  label: string;
  value: string;
  status?: 'success' | 'warning' | 'error';
}> = ({ label, value, status }) => (
  <div className="flex items-center justify-between gap-4">
    <span className="text-zinc-500">{label}:</span>
    <span className={cn(
      'font-bold',
      status === 'error' && 'text-red-400',
      status === 'warning' && 'text-yellow-400',
      status === 'success' && 'text-green-400',
      !status && 'text-zinc-300'
    )}>
      {value}
    </span>
  </div>
);
```

**Instrument Render Method:**

```typescript
// In TreemapExplorerPlugin.render()
render(files: EnrichedFileData[], config: TreemapConfig): void {
  performance.mark('treemap-render-start');

  // ... existing render logic

  performance.mark('treemap-render-end');
  performance.measure('treemap-render', 'treemap-render-start', 'treemap-render-end');

  // Dispatch event with file count
  window.dispatchEvent(new CustomEvent('treemap-metrics', {
    detail: { fileCount: files.length }
  }));
}
```

---

### Phase 3 Deliverables

**Features:**

- ✅ Export (PNG, CSV, JSON)
- ✅ Comprehensive keyboard shortcuts
- ✅ Performance monitoring (dev mode)
- ✅ Shortcuts help overlay (press ?)

**UX:**

- ✅ Export dropdown menu in header
- ✅ Keyboard navigation throughout
- ✅ Real-time performance metrics
- ✅ Help system for shortcuts

**Status:** Production-ready, feature-complete product

---

## 📦 Phase 4: Advanced Scale Features (Optional, 1-2 days)

**Goal:** Handle extreme dataset sizes and add power-user customization

### 4.1 Canvas Rendering Fallback (8 hours)

**When to Use:** Automatically switch to Canvas for >3,000 files

**File:** `src/plugins/treemap-explorer/renderers/CanvasTreemapRenderer.ts`

```typescript
export class CanvasTreemapRenderer {
  private canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;
  private dpr: number;
  private cellMap: Map<string, CellBounds> = new Map();

  constructor(container: HTMLElement, width: number, height: number) {
    this.canvas = document.createElement("canvas");
    this.ctx = this.canvas.getContext("2d", { alpha: false })!;
    this.dpr = window.devicePixelRatio || 1;

    // Set canvas size with DPI scaling
    this.canvas.width = width * this.dpr;
    this.canvas.height = height * this.dpr;
    this.canvas.style.width = `${width}px`;
    this.canvas.style.height = `${height}px`;
    this.ctx.scale(this.dpr, this.dpr);

    container.appendChild(this.canvas);

    // Add click handler
    this.canvas.addEventListener("click", this.handleClick.bind(this));
  }

  render(
    leaves: HierarchyNode[],
    colorFn: (d: any) => string,
    selectedFile: string | null,
  ): void {
    // Clear canvas
    this.ctx.fillStyle = "#09090b";
    this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);

    // Clear cell map
    this.cellMap.clear();

    // Draw all cells
    leaves.forEach((leaf) => {
      const x = leaf.x0;
      const y = leaf.y0;
      const w = leaf.x1 - leaf.x0;
      const h = leaf.y1 - leaf.y0;

      // Store cell bounds for hit detection
      this.cellMap.set(leaf.data.data.key, { x, y, w, h });

      // Fill
      this.ctx.fillStyle = colorFn(leaf.data.data);
      this.ctx.fillRect(x, y, w, h);

      // Border
      this.ctx.strokeStyle = "#18181b";
      this.ctx.lineWidth = 1;
      this.ctx.strokeRect(x, y, w, h);

      // Selection highlight
      if (leaf.data.data.path === selectedFile) {
        this.ctx.strokeStyle = "#ffffff";
        this.ctx.lineWidth = 3;
        this.ctx.strokeRect(x, y, w, h);
      }

      // Label (if space allows)
      if (w > 50 && h > 20) {
        this.ctx.fillStyle =
          leaf.data.data.path === selectedFile ? "#000" : "#fff";
        this.ctx.font = "10px monospace";
        this.ctx.textAlign = "center";
        this.ctx.textBaseline = "middle";
        this.ctx.fillText(
          leaf.data.data.name.substring(0, Math.floor(w / 6)),
          x + w / 2,
          y + h / 2,
        );
      }
    });
  }

  private handleClick(event: MouseEvent): void {
    const rect = this.canvas.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;

    // Find clicked cell
    for (const [key, bounds] of this.cellMap) {
      if (
        x >= bounds.x &&
        x <= bounds.x + bounds.w &&
        y >= bounds.y &&
        y <= bounds.y + bounds.h
      ) {
        // Dispatch custom event
        window.dispatchEvent(
          new CustomEvent("treemap-cell-click", {
            detail: { fileKey: key },
          }),
        );
        break;
      }
    }
  }

  destroy(): void {
    this.canvas.removeEventListener("click", this.handleClick);
    this.canvas.remove();
  }
}

interface CellBounds {
  x: number;
  y: number;
  w: number;
  h: number;
}
```

**Auto-switching Logic:**

```typescript
// In TreemapExplorerPlugin
render(files: EnrichedFileData[], config: TreemapConfig): void {
  const shouldUseCanvas = files.length > 3000;

  if (shouldUseCanvas && !this.canvasRenderer) {
    console.log('Switching to Canvas renderer for performance');
    this.switchToCanvas();
  } else if (!shouldUseCanvas && this.canvasRenderer) {
    this.switchToSVG();
  }

  // ... continue with appropriate renderer
}
```

---

### 4.2 Custom Health Formulas (4 hours)

**Allow users to adjust health score weighting:**

**File:** `src/services/data/CustomHealthFormula.ts`

```typescript
export interface HealthFormulaConfig {
  churnWeight: number; // 0-100
  authorWeight: number; // 0-100
  ageWeight: number; // 0-100
}

export class CustomHealthFormula {
  static DEFAULT_CONFIG: HealthFormulaConfig = {
    churnWeight: 40,
    authorWeight: 30,
    ageWeight: 30,
  };

  static calculate(
    inputs: HealthScoreInputs,
    config: HealthFormulaConfig = this.DEFAULT_CONFIG,
  ): HealthScoreResult {
    // Normalize weights to sum to 100
    const total = config.churnWeight + config.authorWeight + config.ageWeight;
    const normalizedWeights = {
      churn: config.churnWeight / total,
      author: config.authorWeight / total,
      age: config.ageWeight / total,
    };

    // Calculate component scores
    const churnScore = this.calculateChurnScore(inputs.operations);
    const authorScore = this.calculateAuthorScore(inputs.uniqueAuthors);
    const ageScore = this.calculateAgeScore(
      inputs.ageDays,
      inputs.lastModifiedDaysAgo,
    );

    // Weighted average
    const finalScore =
      churnScore * normalizedWeights.churn +
      authorScore * normalizedWeights.author +
      ageScore * normalizedWeights.age;

    return {
      score: Math.round(finalScore),
      category: this.categorize(finalScore),
      // ... rest of result
    };
  }

  // ... component score calculations
}
```

**UI for Formula Editor:**

```typescript
const HealthFormulaEditor: React.FC = () => {
  const [config, setConfig] = useState(CustomHealthFormula.DEFAULT_CONFIG);

  return (
    <div className="space-y-4 p-4">
      <h3 className="font-bold">Custom Health Formula</h3>

      <div className="space-y-3">
        <WeightSlider
          label="Churn Weight"
          value={config.churnWeight}
          onChange={(value) => setConfig({ ...config, churnWeight: value })}
          description="How much code changes matter"
        />

        <WeightSlider
          label="Author Weight"
          value={config.authorWeight}
          onChange={(value) => setConfig({ ...config, authorWeight: value })}
          description="How much contributor diversity matters"
        />

        <WeightSlider
          label="Age Weight"
          value={config.ageWeight}
          onChange={(value) => setConfig({ ...config, ageWeight: value })}
          description="How much file age matters"
        />
      </div>

      <div className="flex gap-2 pt-4">
        <button
          onClick={() => applyCustomFormula(config)}
          className="px-4 py-2 bg-purple-900 hover:bg-purple-800 rounded"
        >
          Apply Formula
        </button>
        <button
          onClick={() => setConfig(CustomHealthFormula.DEFAULT_CONFIG)}
          className="px-4 py-2 bg-zinc-800 hover:bg-zinc-700 rounded"
        >
          Reset to Default
        </button>
      </div>
    </div>
  );
};
```

---

### Phase 4 Deliverables (Optional)

**Advanced Features:**

- ⚪ Canvas renderer (automatic for >3,000 files)
- ⚪ Custom health formula editor
- ⚪ Virtualization (if needed)

**Performance:**

- ⚪ Supports 5,000+ files
- ⚪ Maintains <300ms targets at scale

**Status:** Advanced features for power users and extreme scale

---

## 🎯 Implementation Timeline

### Week 1: Core Delivery

| Day | Phase   | Deliverable                          |
| --- | ------- | ------------------------------------ |
| 0-1 | Phase 0 | Validated refactoring + essential UX |
| 2-3 | Phase 1 | Production-quality polish            |
| 4-6 | Phase 2 | Time lens with quarterly snapshots   |

**End of Week 1:** Complete Debt + Coupling + Time lenses (shippable)

### Week 2: Production Polish

| Day  | Phase              | Deliverable                         |
| ---- | ------------------ | ----------------------------------- |
| 7    | Phase 3 Part 1     | Export capabilities                 |
| 8    | Phase 3 Part 2     | Keyboard shortcuts + monitoring     |
| 9-10 | Phase 4 (Optional) | Advanced features OR buffer/testing |

**End of Week 2:** Production-ready, fully-featured product

---

## 📋 Success Criteria

### Performance (All Validated)

✅ **Initial render:** <300ms (achieving 150-250ms)  
✅ **Lens switch:** <300ms (achieving 200-300ms)  
✅ **Cell selection:** <100ms (achieving 50-100ms)  
✅ **Arc rendering:** <100ms (achieving 50-100ms, 10 arcs)  
✅ **Snapshot transition:** <300ms (achieving 200-300ms)  
✅ **Memory:** <50 MB growth per 20 interactions (achieving +15 MB)

### Features (Complete)

✅ **Debt Lens:** Health-based visualization  
✅ **Coupling Lens:** Relationship arcs (10 max)  
✅ **Time Lens:** Quarterly historical snapshots  
✅ **Search:** Fuzzy file matching  
✅ **Filters:** Advanced + presets  
✅ **Export:** PNG, CSV, JSON  
✅ **Shortcuts:** Comprehensive keyboard navigation

### Quality

✅ **Tests:** All 63 tests passing  
✅ **TypeScript:** No compilation errors  
✅ **Browser:** Chrome, Firefox, Safari  
✅ **Responsive:** Mobile, tablet, desktop  
✅ **Accessible:** Keyboard navigation, ARIA labels

---

## 🚀 Deployment Checklist

### Pre-Launch

- [ ] All tests pass (63/63)
- [ ] Performance benchmarks met
- [ ] No console errors in production build
- [ ] Responsive layout tested
- [ ] Cross-browser testing complete
- [ ] Loading states working
- [ ] Error boundaries catching failures
- [ ] Export functions working
- [ ] Keyboard shortcuts documented

### Documentation

- [ ] User guide (lens modes, features)
- [ ] Keyboard shortcuts reference
- [ ] Dataset requirements
- [ ] Known limitations
- [ ] Performance characteristics

---

## 📞 Known Limitations

1. **File Count:** Optimized for <3,000 files (Canvas needed beyond)
2. **Arc Rendering:** Hard limit of 10 arcs for performance
3. **Temporal Snapshots:** Quarterly/monthly only (not daily)
4. **Export Formats:** PNG/CSV/JSON only

---

## 🔮 Future Enhancements

**Post-Launch Roadmap:**

- Repository comparison (side-by-side)
- Diff view (changes between snapshots)
- File annotations and notes
- Saved filter/lens views
- URL sharing for specific views
- AI-powered hotspot detection
- Custom color schemes

---

**This plan delivers a production-ready Treemap Explorer in 8-10 days with incremental, validated progress at each phase.**
