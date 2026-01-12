#!/usr/bin/env python3
"""
Git File Lifecycle Analyzer - Improved Version

Fixes:
- Performance: Streaming processing instead of loading all commits into memory
- Accuracy: Null-terminated output, timezone handling, proper encoding
- Better error handling and logging
"""

import subprocess
import json
import csv
import click
import pandas as pd
from datetime import datetime, timezone
from collections import defaultdict
from pathlib import Path
from typing import Optional, Dict, List, Generator


class GitFileLifecycle:
    def __init__(self, repo_path):
        self.repo_path = Path(repo_path).resolve()
        self.file_lifecycle = defaultdict(list)
        self.errors = []
        self.skipped_unmerged = 0
        self.total_commits = 0
        self.total_changes = 0

    def run_git_command(self, *args, stream=False):
        """Execute a git command and return the output or stream."""
        try:
            if stream:
                # For streaming large outputs
                process = subprocess.Popen(
                    ["git", "-C", str(self.repo_path)] + list(args),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                )
                return process
            else:
                result = subprocess.run(
                    ["git", "-C", str(self.repo_path)] + list(args),
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    check=True,
                )
                return result.stdout.strip()
        except subprocess.CalledProcessError as e:
            error_msg = f"Error running git command: {e}\nstderr: {e.stderr}"
            self.errors.append(error_msg)
            click.echo(click.style(error_msg, fg="red"), err=True)
            return None

    def validate_repository(self):
        """Check if the path is a valid git repository."""
        result = self.run_git_command("rev-parse", "--is-inside-work-tree")
        if result is None or result.lower() != "true":
            raise ValueError(f"Not a valid git repository: {self.repo_path}")

    def stream_commits(self) -> Generator[Dict, None, None]:
        """
        Stream commits one at a time using null-terminated output.
        This avoids loading all commits into memory.
        """
        # Use -z for null-terminated records (commit records separated by null bytes)
        process = self.run_git_command(
            "log",
            "--all",
            "-z",
            "--format=%H%x00%at%x00%an%x00%ae%x00%s",
            "--reverse",
            stream=True,
        )

        if process is None:
            return

        buffer = ""
        try:
            for chunk in iter(lambda: process.stdout.read(4096), ""):
                buffer += chunk

                # With -z, each commit record is terminated by \0
                # Our format has internal \0 between fields: hash\0timestamp\0name\0email\0subject
                # Then -z adds another \0, so we get: hash\0timestamp\0name\0email\0subject\0
                # We need to collect 5 fields per commit

                while buffer.count("\x00") >= 5:
                    # Extract one complete record (5 null-separated fields)
                    parts = []
                    remaining = buffer

                    for _ in range(5):
                        if "\x00" in remaining:
                            field, remaining = remaining.split("\x00", 1)
                            parts.append(field)
                        else:
                            break

                    if len(parts) == 5:
                        buffer = remaining
                        commit_hash, timestamp, author_name, author_email, subject = (
                            parts
                        )

                        if commit_hash:  # Skip empty records
                            try:
                                yield {
                                    "hash": commit_hash,
                                    "timestamp": int(timestamp),
                                    "datetime": datetime.fromtimestamp(
                                        int(timestamp), tz=timezone.utc
                                    ).isoformat(),
                                    "author_name": author_name,
                                    "author_email": author_email,
                                    "subject": subject,
                                }
                            except (ValueError, IndexError) as e:
                                self.errors.append(
                                    f"Error parsing commit: {commit_hash[:8] if commit_hash else 'unknown'} - {e}"
                                )
                    else:
                        break

        finally:
            process.stdout.close()
            process.wait()

    def get_commit_changes(self, commit_hash):
        """
        Get all file changes using --raw format with -z for null termination.
        """
        # Use -z for null-terminated output to handle special characters in filenames
        output = self.run_git_command(
            "show", "--raw", "-z", "--no-abbrev", "--format=", commit_hash
        )

        if output is None:
            return []

        changes = []

        # Split on null bytes - with -z, format is:
        # :mode1 mode2 sha1 sha2 status\0path1\0[path2\0]
        parts = output.split("\x00")

        i = 0
        while i < len(parts):
            part = parts[i].strip()

            if not part:
                i += 1
                continue

            if part.startswith(":"):
                # This is a raw diff line
                # The paths follow in the next parts
                parsed = self.parse_raw_diff_line_with_paths(
                    part, parts[i + 1 :], commit_hash
                )
                if parsed:
                    changes.append(parsed["change"])
                    i += parsed["consumed"]  # Skip the path parts we consumed
                else:
                    i += 1
            else:
                i += 1

        return changes

    def parse_raw_diff_line_with_paths(self, metadata_line, path_parts, commit_hash):
        """
        Parse a raw diff metadata line along with its associated path parts.

        Args:
            metadata_line: The ":mode mode sha sha status" line
            path_parts: List of remaining parts (paths)
            commit_hash: Current commit hash for error reporting

        Returns:
            Dict with 'change' (the parsed change) and 'consumed' (number of path parts used)
        """
        try:
            # Parse metadata: :old_mode new_mode old_sha new_sha status
            metadata = metadata_line[1:].split()  # Remove leading colon
            if len(metadata) < 5:
                return None

            old_mode = metadata[0]
            new_mode = metadata[1]
            old_sha = metadata[2]
            new_sha = metadata[3]
            status_with_score = metadata[4]

            # Extract the status letter and optional score
            status_letter = status_with_score[0]
            score = status_with_score[1:] if len(status_with_score) > 1 else None

            # Parse based on status letter
            if status_letter == "A":  # Added
                if len(path_parts) < 1:
                    return None
                return {
                    "change": {
                        "status": "added",
                        "path": path_parts[0],
                        "old_mode": old_mode,
                        "new_mode": new_mode,
                        "old_sha": old_sha,
                        "new_sha": new_sha,
                    },
                    "consumed": 2,  # metadata line + 1 path
                }

            elif status_letter == "M":  # Modified
                if len(path_parts) < 1:
                    return None
                return {
                    "change": {
                        "status": "modified",
                        "path": path_parts[0],
                        "old_mode": old_mode,
                        "new_mode": new_mode,
                        "old_sha": old_sha,
                        "new_sha": new_sha,
                    },
                    "consumed": 2,
                }

            elif status_letter == "D":  # Deleted
                if len(path_parts) < 1:
                    return None
                return {
                    "change": {
                        "status": "deleted",
                        "path": path_parts[0],
                        "old_mode": old_mode,
                        "new_mode": new_mode,
                        "old_sha": old_sha,
                        "new_sha": new_sha,
                    },
                    "consumed": 2,
                }

            elif status_letter == "R":  # Renamed
                if len(path_parts) < 2:
                    self.errors.append(
                        f"Invalid rename in commit {commit_hash[:8]}: not enough paths"
                    )
                    return None
                return {
                    "change": {
                        "status": "renamed",
                        "old_path": path_parts[0],
                        "new_path": path_parts[1],
                        "similarity": score if score else "100",
                        "old_mode": old_mode,
                        "new_mode": new_mode,
                        "old_sha": old_sha,
                        "new_sha": new_sha,
                    },
                    "consumed": 3,  # metadata line + 2 paths
                }

            elif status_letter == "C":  # Copied
                if len(path_parts) < 2:
                    self.errors.append(
                        f"Invalid copy in commit {commit_hash[:8]}: not enough paths"
                    )
                    return None
                return {
                    "change": {
                        "status": "copied",
                        "source_path": path_parts[0],
                        "path": path_parts[1],
                        "similarity": score if score else "100",
                        "old_mode": old_mode,
                        "new_mode": new_mode,
                        "old_sha": old_sha,
                        "new_sha": new_sha,
                    },
                    "consumed": 3,
                }

            elif status_letter == "T":  # Type change
                if len(path_parts) < 1:
                    return None
                return {
                    "change": {
                        "status": "type_changed",
                        "path": path_parts[0],
                        "old_mode": old_mode,
                        "new_mode": new_mode,
                        "old_sha": old_sha,
                        "new_sha": new_sha,
                    },
                    "consumed": 2,
                }

            elif status_letter == "U":  # Unmerged
                self.skipped_unmerged += 1
                if self.skipped_unmerged <= 10:
                    self.errors.append(
                        f"Skipped unmerged file in commit {commit_hash[:8]}: {path_parts[0] if path_parts else 'unknown'}"
                    )
                return None

            elif status_letter == "X":  # Unknown
                self.errors.append(f"Unknown change type in commit {commit_hash[:8]}")
                return None

            else:
                self.errors.append(
                    f"Unexpected status letter '{status_letter}' in commit {commit_hash[:8]}"
                )
                return None

        except (IndexError, ValueError) as e:
            self.errors.append(
                f"Error parsing raw diff in commit {commit_hash[:8]}: {e}"
            )
            return None

    def analyze_repository(self):
        """
        Analyze the repository by streaming commits.
        This version processes commits incrementally to avoid memory issues.
        """
        click.echo("Analyzing repository...")
        click.echo(f"Repository: {click.style(str(self.repo_path), fg='cyan')}")
        click.echo()
        click.echo("Processing commits (streaming)...")

        for commit in self.stream_commits():
            self.total_commits += 1

            # Show progress every 100 commits
            if self.total_commits % 100 == 0:
                click.echo(f"  Processed {self.total_commits:,} commits...", nl=False)
                click.echo("\r", nl=False)

            changes = self.get_commit_changes(commit["hash"])

            for change in changes:
                self.total_changes += 1

                # Get the primary path for this change
                if change["status"] == "renamed":
                    # Track both old and new paths
                    old_path = change["old_path"]
                    new_path = change["new_path"]

                    self.file_lifecycle[old_path].append(
                        {
                            **change,
                            "commit_hash": commit["hash"],
                            "commit_timestamp": commit["timestamp"],
                            "commit_datetime": commit["datetime"],
                            "author_name": commit["author_name"],
                            "author_email": commit["author_email"],
                            "commit_subject": commit["subject"],
                        }
                    )

                    self.file_lifecycle[new_path].append(
                        {
                            **change,
                            "commit_hash": commit["hash"],
                            "commit_timestamp": commit["timestamp"],
                            "commit_datetime": commit["datetime"],
                            "author_name": commit["author_name"],
                            "author_email": commit["author_email"],
                            "commit_subject": commit["subject"],
                        }
                    )

                elif change["status"] == "copied":
                    source_path = change["source_path"]
                    dest_path = change["path"]

                    self.file_lifecycle[source_path].append(
                        {
                            **change,
                            "commit_hash": commit["hash"],
                            "commit_timestamp": commit["timestamp"],
                            "commit_datetime": commit["datetime"],
                            "author_name": commit["author_name"],
                            "author_email": commit["author_email"],
                            "commit_subject": commit["subject"],
                        }
                    )

                    self.file_lifecycle[dest_path].append(
                        {
                            **change,
                            "commit_hash": commit["hash"],
                            "commit_timestamp": commit["timestamp"],
                            "commit_datetime": commit["datetime"],
                            "author_name": commit["author_name"],
                            "author_email": commit["author_email"],
                            "commit_subject": commit["subject"],
                        }
                    )

                else:
                    path = change.get("path")
                    if path:
                        self.file_lifecycle[path].append(
                            {
                                **change,
                                "commit_hash": commit["hash"],
                                "commit_timestamp": commit["timestamp"],
                                "commit_datetime": commit["datetime"],
                                "author_name": commit["author_name"],
                                "author_email": commit["author_email"],
                                "commit_subject": commit["subject"],
                            }
                        )

        click.echo()
        click.echo(f"✓ Processed {self.total_commits:,} commits")
        click.echo(f"✓ Found {self.total_changes:,} file changes")
        click.echo(f"✓ Tracked {len(self.file_lifecycle):,} unique files")

        if self.skipped_unmerged > 0:
            click.echo(
                click.style(
                    f"⚠ Skipped {self.skipped_unmerged:,} unmerged files (merge conflicts)",
                    fg="yellow",
                )
            )

    def create_output_directory(self):
        """Create output directory named DATASETS_<repo_name>."""
        repo_name = self.repo_path.name
        output_dir = Path(f"DATASETS_{repo_name}")
        output_dir.mkdir(exist_ok=True)

        click.echo(
            f"✓ Created output directory: {click.style(str(output_dir), fg='cyan')}"
        )
        return output_dir

    def export_to_json(self, output_path):
        """Export the lifecycle data to JSON."""
        click.echo(f"Exporting to JSON: {output_path}")

        data = {
            "analyzed_at": datetime.now(timezone.utc).isoformat(),
            "repository_path": str(self.repo_path),
            "total_files": len(self.file_lifecycle),
            "total_commits": self.total_commits,
            "total_changes": self.total_changes,
            "skipped_unmerged": self.skipped_unmerged,
            "total_errors": len(self.errors),
            "file_lifecycle": dict(self.file_lifecycle),
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        click.echo(f"✓ Exported lifecycle data: {click.style(output_path, fg='green')}")

    def export_to_csv(self, output_path):
        """Export summary data to CSV with proper quoting."""
        click.echo(f"Exporting summary to CSV: {output_path}")

        rows = []
        for file_path, events in self.file_lifecycle.items():
            # Calculate summary statistics
            added_count = sum(1 for e in events if e["status"] == "added")
            modified_count = sum(1 for e in events if e["status"] == "modified")
            deleted_count = sum(1 for e in events if e["status"] == "deleted")
            renamed_count = sum(1 for e in events if e["status"] == "renamed")
            copied_count = sum(1 for e in events if e["status"] == "copied")
            type_changed_count = sum(1 for e in events if e["status"] == "type_changed")

            first_event = events[0] if events else None
            last_event = events[-1] if events else None

            rows.append(
                {
                    "file_path": file_path,
                    "total_events": len(events),
                    "added_count": added_count,
                    "modified_count": modified_count,
                    "deleted_count": deleted_count,
                    "renamed_count": renamed_count,
                    "copied_count": copied_count,
                    "type_changed_count": type_changed_count,
                    "first_seen": (
                        first_event["commit_datetime"] if first_event else None
                    ),
                    "last_seen": last_event["commit_datetime"] if last_event else None,
                    "first_commit": first_event["commit_hash"] if first_event else None,
                    "last_commit": last_event["commit_hash"] if last_event else None,
                }
            )

        if not rows:
            click.echo(click.style("⚠ No data to export", fg="yellow"))
            return

        fieldnames = [
            "file_path",
            "total_events",
            "added_count",
            "modified_count",
            "deleted_count",
            "renamed_count",
            "copied_count",
            "type_changed_count",
            "first_seen",
            "last_seen",
            "first_commit",
            "last_commit",
        ]

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            # Use QUOTE_NONNUMERIC to properly handle special characters
            writer = csv.DictWriter(
                f, fieldnames=fieldnames, quoting=csv.QUOTE_NONNUMERIC
            )
            writer.writeheader()
            writer.writerows(rows)

        click.echo(
            f"✓ Exported summary CSV: {click.style(output_path, fg='green')} ({len(rows):,} files)"
        )

    def export_events_to_csv(self, output_path):
        """Export detailed event data to CSV with proper quoting."""
        click.echo(f"Exporting events to CSV: {output_path}")

        rows = []
        for file_path, events in self.file_lifecycle.items():
            for event in events:
                row = {
                    "file_path": file_path,
                    "status": event["status"],
                    "commit_hash": event["commit_hash"],
                    "commit_datetime": event["commit_datetime"],
                    "commit_timestamp": event["commit_timestamp"],
                    "author_name": event["author_name"],
                    "author_email": event["author_email"],
                    "commit_subject": event["commit_subject"],
                    "old_mode": event.get("old_mode", ""),
                    "new_mode": event.get("new_mode", ""),
                    "old_sha": event.get("old_sha", ""),
                    "new_sha": event.get("new_sha", ""),
                }

                # Add rename/copy specific fields
                if event["status"] == "renamed":
                    row["old_path"] = event.get("old_path", "")
                    row["new_path"] = event.get("new_path", "")
                    row["similarity"] = event.get("similarity", "")
                elif event["status"] == "copied":
                    row["source_path"] = event.get("source_path", "")
                    row["similarity"] = event.get("similarity", "")

                rows.append(row)

        if not rows:
            click.echo(click.style("⚠ No events to export", fg="yellow"))
            return

        # Determine all possible fieldnames
        fieldnames = [
            "file_path",
            "status",
            "commit_hash",
            "commit_datetime",
            "commit_timestamp",
            "author_name",
            "author_email",
            "commit_subject",
            "old_mode",
            "new_mode",
            "old_sha",
            "new_sha",
            "old_path",
            "new_path",
            "source_path",
            "similarity",
        ]

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            # Use QUOTE_NONNUMERIC to properly handle special characters
            writer = csv.DictWriter(
                f,
                fieldnames=fieldnames,
                quoting=csv.QUOTE_NONNUMERIC,
                extrasaction="ignore",
            )
            writer.writeheader()
            writer.writerows(rows)

        click.echo(
            f"✓ Exported events CSV: {click.style(output_path, fg='green')} ({len(rows):,} events)"
        )

    def export_errors(self, output_path):
        """Export errors to a text file."""
        click.echo(f"Exporting errors to: {output_path}")

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(f"Total Errors: {len(self.errors)}\n")
            f.write(f"Skipped Unmerged Files: {self.skipped_unmerged}\n")
            f.write(f"Generated at: {datetime.now(timezone.utc).isoformat()}\n")
            f.write("\n" + "=" * 80 + "\n\n")

            for i, error in enumerate(self.errors, 1):
                f.write(f"Error {i}:\n{error}\n\n")

        click.echo(
            f"✓ Exported errors: {click.style(output_path, fg='green')} ({len(self.errors)} errors)"
        )

    def generate_metadata_report(self, generated_files: List[str], output_path: str):
        """Generate a metadata report with file size limits."""
        click.echo(f"Generating metadata report: {output_path}")

        report_lines = [
            "# Git File Lifecycle Analysis - Dataset Metadata",
            "",
            f"**Generated:** {datetime.now(timezone.utc).isoformat()}",
            f"**Repository:** `{self.repo_path}`",
            "",
            "## Analysis Summary",
            "",
            f"- **Total Commits Processed:** {self.total_commits:,}",
            f"- **Total File Changes:** {self.total_changes:,}",
            f"- **Unique Files Tracked:** {len(self.file_lifecycle):,}",
            f"- **Errors Encountered:** {len(self.errors):,}",
            f"- **Unmerged Files Skipped:** {self.skipped_unmerged:,}",
            "",
            "## Generated Files",
            "",
        ]

        for file_path in generated_files:
            file_path_obj = Path(file_path)

            if not file_path_obj.exists():
                continue

            file_name = file_path_obj.name
            file_size = file_path_obj.stat().st_size
            file_size_mb = file_size / (1024 * 1024)

            report_lines.append(f"### {file_name}")
            report_lines.append("")
            report_lines.append(
                f"**File Size:** {file_size:,} bytes ({file_size_mb:.2f} MB)"
            )
            report_lines.append(f"**Path:** `{file_path}`")
            report_lines.append("")

            # Only analyze files under 50MB to avoid memory issues
            MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB

            if file_size > MAX_FILE_SIZE:
                report_lines.append(
                    f"**Note:** File exceeds {MAX_FILE_SIZE // (1024*1024)}MB limit. Skipping detailed analysis."
                )
                report_lines.append("")
                report_lines.append("---")
                report_lines.append("")
                continue

            # Analyze CSV files
            if file_path.endswith(".csv"):
                try:
                    # Use low_memory=False and specify dtypes for better accuracy
                    df = pd.read_csv(file_path, low_memory=False, encoding="utf-8")

                    report_lines.append(f"**Format:** CSV (Comma-Separated Values)")
                    report_lines.append(f"**Rows:** {len(df):,}")
                    report_lines.append(f"**Columns:** {len(df.columns)}")
                    report_lines.append("")

                    # Column information
                    report_lines.append("**Columns:**")
                    for col in df.columns:
                        dtype = df[col].dtype
                        non_null = df[col].count()
                        report_lines.append(
                            f"- `{col}`: {dtype} ({non_null:,} non-null)"
                        )
                    report_lines.append("")

                    # Missing data
                    null_counts = df.isnull().sum()
                    null_counts = null_counts[null_counts > 0]
                    if len(null_counts) > 0:
                        report_lines.append("**Missing Data:**")
                        for col, count in null_counts.items():
                            pct = (count / len(df)) * 100
                            report_lines.append(
                                f"  - `{col}`: {count:,} nulls ({pct:.1f}%)"
                            )
                    else:
                        report_lines.append("**Missing Data:** No null values")
                    report_lines.append("")

                    # Show value ranges for numeric columns
                    numeric_cols = df.select_dtypes(
                        include=["int64", "float64"]
                    ).columns
                    if len(numeric_cols) > 0:
                        report_lines.append("**Numeric Column Ranges:**")
                        for col in numeric_cols:
                            min_val = df[col].min()
                            max_val = df[col].max()
                            mean_val = df[col].mean()
                            report_lines.append(
                                f"- `{col}`: min={min_val:,.2f}, max={max_val:,.2f}, mean={mean_val:,.2f}"
                            )
                        report_lines.append("")

                    # Show sample data (first 3 rows only)
                    report_lines.append("**Sample Data (first 3 rows):**")
                    report_lines.append("```")
                    report_lines.append(df.head(3).to_string(index=False))
                    report_lines.append("```")

                except Exception as e:
                    report_lines.append(f"**Error analyzing CSV:** {str(e)}")

            # Analyze JSON files
            elif file_path.endswith(".json"):
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        data = json.load(f)

                    report_lines.append(
                        f"**Format:** JSON (JavaScript Object Notation)"
                    )
                    report_lines.append("")

                    if isinstance(data, dict):
                        report_lines.append(f"**Top-level keys:** {len(data)} keys")
                        report_lines.append("**Structure:**")
                        for key in list(data.keys())[:10]:  # Show first 10 keys
                            value = data[key]
                            value_type = type(value).__name__
                            if isinstance(value, (list, dict)):
                                size_info = f" ({len(value)} items)"
                                report_lines.append(
                                    f"- `{key}`: {value_type}{size_info}"
                                )
                            else:
                                report_lines.append(f"- `{key}`: {value_type}")

                        if len(data) > 10:
                            report_lines.append(f"- *(and {len(data) - 10} more keys)*")
                    elif isinstance(data, list):
                        report_lines.append(
                            f"**Structure:** Array with {len(data)} items"
                        )

                    report_lines.append("")

                    # Special handling for our lifecycle JSON structure
                    if isinstance(data, dict) and "total_files" in data:
                        report_lines.append("**Dataset Statistics:**")
                        report_lines.append(
                            f"- Total files analyzed: {data.get('total_files', 0):,}"
                        )
                        report_lines.append(
                            f"- Analysis timestamp: {data.get('analyzed_at', 'N/A')}"
                        )
                        if data.get("total_errors", 0) > 0:
                            report_lines.append(
                                f"- Errors encountered: {data.get('total_errors', 0)}"
                            )
                        if data.get("skipped_unmerged", 0) > 0:
                            report_lines.append(
                                f"- Unmerged files skipped: {data.get('skipped_unmerged', 0)}"
                            )

                except Exception as e:
                    report_lines.append(f"**Error analyzing JSON:** {str(e)}")

            # Analyze text/error files
            elif file_path.endswith(".txt"):
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        lines = f.readlines()

                    report_lines.append(f"**Format:** Plain Text")
                    report_lines.append(f"**Line Count:** {len(lines):,} lines")
                    report_lines.append("")

                    if "error" in file_name.lower():
                        report_lines.append(
                            "**Content:** Error log file containing parsing and processing errors"
                        )

                except Exception as e:
                    report_lines.append(f"**Error analyzing text file:** {str(e)}")

            report_lines.append("")
            report_lines.append("---")
            report_lines.append("")

        # Write the report
        report_content = "\n".join(report_lines)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report_content)

        click.echo(
            f"✓ Generated metadata report: {click.style(str(output_path), fg='green')}"
        )


@click.command()
@click.argument(
    "repo_path", type=click.Path(exists=True, file_okay=False, dir_okay=True)
)
@click.option(
    "--output-json",
    default="file_lifecycle.json",
    help="Output JSON file path",
    show_default=True,
)
@click.option(
    "--output-summary-csv",
    default="file_lifecycle_summary.csv",
    help="Output summary CSV file path",
    show_default=True,
)
@click.option(
    "--output-events-csv",
    default="file_lifecycle_events.csv",
    help="Output detailed events CSV file path",
    show_default=True,
)
@click.option(
    "--output-errors",
    default="file_lifecycle_errors.txt",
    help="Output errors file path",
    show_default=True,
)
@click.option("--no-json", is_flag=True, help="Skip JSON export")
@click.option("--no-csv", is_flag=True, help="Skip CSV exports")
def main(
    repo_path,
    output_json,
    output_summary_csv,
    output_events_csv,
    output_errors,
    no_json,
    no_csv,
):
    """Analyze git repository using native structured output.

    REPO_PATH: Path to the git repository to analyze
    """
    click.echo(
        click.style("Git File Lifecycle Analyzer - Improved", fg="cyan", bold=True)
    )
    click.echo(click.style("Using Null-Terminated Output for Accuracy", fg="cyan"))
    click.echo()

    # Initialize analyzer
    analyzer = GitFileLifecycle(repo_path)

    # Validate repository
    try:
        analyzer.validate_repository()
    except ValueError as e:
        click.echo(click.style(f"✗ {e}", fg="red", bold=True))
        return

    # Create output directory
    output_dir = analyzer.create_output_directory()

    # Update output paths to use the new directory
    output_json = str(output_dir / output_json)
    output_summary_csv = str(output_dir / output_summary_csv)
    output_events_csv = str(output_dir / output_events_csv)
    output_errors = str(output_dir / output_errors)
    metadata_report = str(output_dir / "dataset_metadata.md")

    # Analyze repository
    analyzer.analyze_repository()

    # Track generated files for metadata report
    generated_files = []

    # Export data
    if not no_json:
        click.echo()
        analyzer.export_to_json(output_json)
        generated_files.append(output_json)

    if not no_csv:
        click.echo()
        analyzer.export_to_csv(output_summary_csv)
        generated_files.append(output_summary_csv)
        analyzer.export_events_to_csv(output_events_csv)
        generated_files.append(output_events_csv)

    # Export errors if any
    if analyzer.errors or analyzer.skipped_unmerged > 0:
        click.echo()
        analyzer.export_errors(output_errors)
        generated_files.append(output_errors)

    # Generate metadata report
    click.echo()
    analyzer.generate_metadata_report(generated_files, metadata_report)

    click.echo()
    click.secho("✓ Analysis complete!", fg="green", bold=True)
    click.echo(
        f"All outputs saved to: {click.style(str(output_dir), fg='cyan', bold=True)}"
    )


if __name__ == "__main__":
    main()
