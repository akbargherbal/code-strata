# Phase 4 Features - Quick Reference

## 🚀 Getting Started

### Basic Usage (Unchanged from v1.0)
```bash
python strata.py /path/to/repo
```

### Recommended (New in v2.0)
```bash
python strata.py /path/to/repo --preset=standard
```

---

## 📋 New Presets

```bash
# Standard - Recommended for most cases
python strata.py /repo --preset=standard

# Quick - Fast analysis for large repos (30% sampling)
python strata.py /repo --preset=quick

# Full - All features enabled
python strata.py /repo --preset=full

# Essential - Core metrics only
python strata.py /repo --preset=essential

# Network - Collaboration analysis
python strata.py /repo --preset=network
```

---

## ⚙️ Configuration Files

### Create Config File
```yaml
# .git-lifecycle.yaml
preset: standard
memory-limit: 4096
verbose: true
```

### Use Config File
```bash
# Auto-discover (looks for .git-lifecycle.yaml in repo)
python strata.py /repo

# Explicit path
python strata.py /repo --config=my-config.yaml
```

---

## 🔍 Performance Profiling

### Enable Profiling
```bash
python strata.py /repo --profile --preset=full
```

### With Memory Limit
```bash
python strata.py /repo --profile --memory-limit=2048
```

### Verbose + Memory Monitoring
```bash
python strata.py /repo --preset=standard -v --memory-limit=4096
```

---

## 🎨 Output Control

### Colored Output (Default)
```bash
python strata.py /repo --preset=standard
```

### Disable Colors (for CI/CD)
```bash
python strata.py /repo --preset=standard --no-color
```

### Quiet Mode (scripting)
```bash
python strata.py /repo --preset=standard -q
```

### Verbose Mode (detailed)
```bash
python strata.py /repo --preset=standard -v
```

---

## 📊 Progress Display

With tqdm installed:
```
Analyzing commits: |████████████████| 15234/15234 [02:15<00:00, 112 commits/s]
```

Without tqdm:
```
   Processing commits (15234)
```

---

## 🔧 Dependency Management

### Check Dependencies
```bash
python strata.py --check-dependencies
```

Output:
```
Checking optional dependencies...
  tqdm (progress bars): ✓ Available
  colorama (colored output): ✓ Available
  PyYAML (YAML config): ✓ Available
  memory_profiler: ✗ Not installed
  psutil (memory monitoring): ✓ Available
```

### Install All Optional Dependencies
```bash
pip install tqdm colorama pyyaml memory-profiler psutil
```

### Install from Requirements
```bash
pip install -r requirements.txt
```

---

## 🎯 Common Use Cases

### Large Repository (Quick Analysis)
```bash
python strata.py /huge-repo --preset=quick
# Uses 30% sampling, limited features, optimized for speed
```

### Memory-Constrained Environment
```bash
python strata.py /repo --preset=standard --memory-limit=1024
# Aborts if memory usage exceeds 1GB
```

### Performance Debugging
```bash
python strata.py /repo --preset=full --profile -v
# Full profiling with verbose output and cache statistics
```

### CI/CD Pipeline
```bash
python strata.py /repo --preset=essential --no-color -q -o output/
# No colors, quiet mode, specified output directory
```

### Custom Configuration
```yaml
# .git-lifecycle.yaml
enable-aggregations: true
phase2: true
phase3: true
skip-milestones: true
memory-limit: 2048
verbose: true
```

```bash
python strata.py /repo
# Automatically uses .git-lifecycle.yaml
```

---

## 📈 Performance Metrics

After analysis completes, you'll see:
```
📊 ANALYSIS SUMMARY
======================================================================
   Repository: /path/to/repo
   Output directory: git_analysis_output_20250114_123456
   Total commits: 15,234
   Files tracked: 2,567
   Total changes: 47,892
   Datasets generated: 5
   Errors: 0
   Peak memory: 342.5 MB
   Cache hit rate: 87.3%

⏱️  Total time: 135.42s
======================================================================
```

---

## 🛠️ Troubleshooting

### Progress Bar Not Showing?
```bash
pip install tqdm
```

### No Colors?
```bash
pip install colorama
```

### YAML Config Not Loading?
```bash
pip install pyyaml
# Or use JSON: .git-lifecycle.json
```

### Memory Errors?
```bash
# Use quick mode
python strata.py /repo --preset=quick

# Or set memory limit
python strata.py /repo --memory-limit=2048
```

---

## 📝 Preset Comparison

| Feature | standard | quick | essential | network | full |
|---------|----------|-------|-----------|---------|------|
| File metadata | ✓ | ✗ | ✓ | ✗ | ✓ |
| Temporal | ✓ | ✗ | ✓ | ✗ | ✓ |
| Author network | ✓ | ✗ | ✗ | ✓ | ✓ |
| Co-change | ✗ | ✗ | ✗ | ✓ | ✓ |
| Hierarchy | ✗ | ✗ | ✗ | ✗ | ✓ |
| Milestones | ✗ | ✗ | ✗ | ✗ | ✓ |
| Sampling | 100% | 30% | 100% | 100% | 100% |

---

## 🔗 Related Files

- `strata.py` - Enhanced script
- `requirements.txt` - Optional dependencies
- `.git-lifecycle.example.yaml` - Configuration template
- `PHASE4_COMPLETION.md` - Detailed documentation

---

## 💡 Pro Tips

1. **Use presets** for common scenarios instead of many flags
2. **Create config files** for repeated analysis
3. **Enable verbose mode** (`-v`) when diagnosing issues
4. **Use quick mode** for initial exploration of large repos
5. **Monitor memory** with `--memory-limit` on constrained systems
6. **Profile performance** with `--profile` to optimize workflows
