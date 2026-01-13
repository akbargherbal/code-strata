# Schema Compatibility Contract v1.0

## Executive Summary

This document defines the **immutable contract** for `file_lifecycle.json` that all production systems depend on. Any changes to this schema constitute a **BREAKING CHANGE** and require major version bump and migration plan.

**Schema Version:** 1.0.0  
**Last Updated:** 2024-01-15  
**Status:** ✅ PRODUCTION - DO NOT MODIFY

---

## Critical Fields (IMMUTABLE)

These fields are used by production consumers and **MUST NEVER CHANGE**:

### Root Level

| Field              | Type    | Required | Description                             | Used By       |
| ------------------ | ------- | -------- | --------------------------------------- | ------------- |
| `files`            | Object  | ✅       | Map of file paths to lifecycle events   | All consumers |
| `total_commits`    | Integer | ✅       | Total commits analyzed                  | Dashboard     |
| `total_changes`    | Integer | ✅       | Total file changes recorded             | Dashboard     |
| `generated_at`     | String  | ✅       | ISO 8601 timestamp                      | Audit log     |
| `repository_path`  | String  | ✅       | Absolute path to repository             | All consumers |
| `skipped_unmerged` | Integer | ✅       | Number of unmerged changes skipped      | Quality check |

### Lifecycle Event Schema

Each event in `files[path]` array **MUST** contain:

| Field            | Type    | Required | Description                       | Format                |
| ---------------- | ------- | -------- | --------------------------------- | --------------------- |
| `commit_hash`    | String  | ✅       | Full git SHA-1 hash               | 40 hex characters     |
| `timestamp`      | Integer | ✅       | Unix timestamp                    | Seconds since epoch   |
| `datetime`       | String  | ✅       | ISO 8601 datetime                 | UTC timezone          |
| `operation`      | String  | ✅       | Git operation code                | A, M, D, R, C, or T   |
| `author_name`    | String  | ✅       | Commit author name                | UTF-8 string          |
| `author_email`   | String  | ✅       | Commit author email               | UTF-8 string          |
| `commit_subject` | String  | ✅       | Commit message first line         | UTF-8 string          |
| `old_path`       | String  | ❌       | Original path (rename/copy only)  | UTF-8 string          |
| `similarity`     | Integer | ❌       | Similarity % (rename/copy only)   | 0-100                 |

---

## Operation Codes

| Code | Meaning     | Description                        | old_path Present? |
| ---- | ----------- | ---------------------------------- | ----------------- |
| A    | Add         | File created                       | No                |
| M    | Modify      | File content changed               | No                |
| D    | Delete      | File removed                       | No                |
| R    | Rename      | File moved/renamed                 | Yes               |
| C    | Copy        | File copied                        | Yes               |
| T    | Type change | File type changed (rare)           | No                |

---

## Data Type Guarantees

### Strings
- **Encoding:** UTF-8 with invalid byte replacement
- **Special Characters:** Preserved (including newlines, tabs, emoji)
- **Null Values:** Never null, always present (may be empty string)

### Integers
- **Range:** All integers are non-negative (>= 0)
- **Null Values:** Never null, always present

### Arrays
- **Empty Arrays:** Possible for files with no events (edge case)
- **Null Values:** Never null, always present
- **Order:** Events are chronologically ordered by timestamp (ascending)

---

## Invariants & Guarantees

### Global Invariants

1. **Chronological Order:** Events within each file array are sorted by `timestamp` (ascending)
2. **Referential Integrity:** Every `commit_hash` is valid and exists in repository
3. **Event Uniqueness:** No duplicate events (same commit + file + operation)
4. **Total Accuracy:** `total_changes` equals sum of all event counts across all files

### Per-File Invariants

1. **First Event:** First event for each file has `operation` of 'A' (Add)
2. **Last Event:** Files with operation 'D' (Delete) should not have subsequent events
3. **Rename Chain:** Renames preserve history (events follow renamed paths)

### Edge Cases

1. **Empty Files Object:** Possible if repository has no trackable files (very rare)
2. **Zero Commits:** `total_commits` can be 0 for empty repositories
3. **High Skipped Count:** `skipped_unmerged` can be high for active merge-heavy repos

---

## Schema Validation

### Automated Validation

Use JSON Schema validation before consuming data:

```python
import jsonschema
import json

# Load schema
with open('schema_v1.0_contract.json') as f:
    schema = json.load(f)

# Validate data
with open('file_lifecycle.json') as f:
    data = json.load(f)
    
jsonschema.validate(instance=data, schema=schema)
```

### Quick Sanity Checks

```python
# Check 1: Required fields present
assert 'files' in data
assert 'total_commits' in data
assert 'total_changes' in data

# Check 2: Counts match
calculated_changes = sum(len(events) for events in data['files'].values())
assert calculated_changes == data['total_changes']

# Check 3: Events are chronologically ordered
for path, events in data['files'].items():
    timestamps = [e['timestamp'] for e in events]
    assert timestamps == sorted(timestamps), f"Events not sorted for {path}"
```

---

## Production Consumers

### Known Systems Using This Schema

1. **Interactive Visualization Dashboard**
   - Fields Used: `files`, `total_commits`, `generated_at`
   - Dependency: All event fields
   - Update Frequency: Daily

2. **Analytics Pipeline**
   - Fields Used: All root and event fields
   - Dependency: Operation codes, timestamps
   - Update Frequency: Real-time

3. **Compliance Auditing**
   - Fields Used: `author_email`, `datetime`, `commit_hash`
   - Dependency: Chronological ordering
   - Update Frequency: On-demand

---

## Change Policy

### Allowed Changes (Non-Breaking)

✅ **Can Do Without Breaking Compatibility:**
- Add new optional fields to root level
- Add new optional fields to events (not required)
- Add new operation codes (for future git features)
- Improve performance without changing output
- Fix bugs that cause schema violations

### Forbidden Changes (Breaking)

❌ **NEVER Do Without Major Version Bump:**
- Remove any required field
- Change field data types
- Rename fields
- Change chronological ordering
- Modify operation code meanings
- Change timestamp format
- Break referential integrity

### Version Bump Requirements

| Change Type                  | Version Bump | Migration Required? |
| ---------------------------- | ------------ | ------------------- |
| Add optional field           | Minor (1.1)  | No                  |
| Change required field        | Major (2.0)  | Yes                 |
| Remove any field             | Major (2.0)  | Yes                 |
| Add new operation code       | Minor (1.1)  | No                  |
| Change operation code format | Major (2.0)  | Yes                 |

---

## Testing Requirements

### Before Any Schema Change

1. ✅ Run full regression test suite
2. ✅ Validate against JSON schema
3. ✅ Check all production consumer compatibility
4. ✅ Generate byte-identical output (if no schema change)
5. ✅ Test on diverse repositories (small, medium, large)

### Test Repositories

- **Small:** <100 commits, few files
- **Medium:** 1,000-10,000 commits, mixed operations
- **Large:** 50,000+ commits, merge-heavy, renames

---

## Migration Path (If Breaking Change Required)

If a breaking change is absolutely necessary:

1. **Announce:** 3 months notice to all consumers
2. **Dual Output:** Support both v1.0 and v2.0 simultaneously
3. **Migration Tool:** Provide automatic converter
4. **Deprecation Period:** 6 months minimum
5. **Documentation:** Complete migration guide
6. **Rollback Plan:** Easy revert to v1.0

---

## Contact & Support

**Schema Owner:** Git Lifecycle Team  
**Review Required:** Any change to this document  
**Last Validated:** 2024-01-15  

**Emergency Contact:** If production issues related to schema changes  
**Rollback Procedure:** See `ROLLBACK_PROCEDURE.md`

---

## Appendix A: Example Valid Data

```json
{
  "files": {
    "src/main.py": [
      {
        "commit_hash": "a1b2c3d4e5f6789012345678901234567890abcd",
        "timestamp": 1705324800,
        "datetime": "2024-01-15T12:00:00+00:00",
        "operation": "A",
        "author_name": "Jane Developer",
        "author_email": "jane@example.com",
        "commit_subject": "Initial commit"
      },
      {
        "commit_hash": "b2c3d4e5f67890123456789012345678901abcde",
        "timestamp": 1705411200,
        "datetime": "2024-01-16T12:00:00+00:00",
        "operation": "M",
        "author_name": "Jane Developer",
        "author_email": "jane@example.com",
        "commit_subject": "Fix bug in parser"
      }
    ]
  },
  "total_commits": 2,
  "total_changes": 2,
  "generated_at": "2024-01-16T14:30:00+00:00",
  "repository_path": "/home/user/project",
  "skipped_unmerged": 0
}
```

## Appendix B: Schema Hash

For verification purposes:

```
SHA-256: [To be generated after finalization]
MD5: [To be generated after finalization]
```

---

**Document Status:** ✅ APPROVED FOR PRODUCTION  
**Version:** 1.0.0  
**Effective Date:** 2024-01-15  
**Next Review:** 2024-04-15 (or before any schema modification)
