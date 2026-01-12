#!/usr/bin/env python3
"""
Git File Lifecycle Analyzer - Using Git's Native Structured Output

This version uses --raw format which provides structured, unambiguous output
instead of parsing text-based --name-status.
"""

import subprocess
import json
import csv
import click
import pandas as pd
from datetime import datetime
from collections import defaultdict
from pathlib import Path


class GitFileLifecycle:
    def __init__(self, repo_path):
        self.repo_path = Path(repo_path).resolve()
        self.file_lifecycle = defaultdict(list)
        self.errors = []

    def run_git_command(self, *args):
        """Execute a git command and return the output."""
        try:
            result = subprocess.run(
                ["git", "-C", str(self.repo_path)] + list(args),
                capture_output=True,
                text=True,
                check=True,
            )
            return result.stdout.strip()
        except subprocess.CalledProcessError as e:
            error_msg = f"Error running git command: {e}\nstderr: {e.stderr}"
            self.errors.append(error_msg)
            click.echo(click.style(error_msg, fg="red"), err=True)
            return None

    def get_all_commits(self):
        """Get all commits in reverse chronological order."""
        output = self.run_git_command(
            "log", "--all", "--format=%H|%at|%an|%ae|%s", "--reverse"
        )

        if not output:
            return []

        commits = []
        for line in output.split("\n"):
            if line:
                try:
                    commit_hash, timestamp, author_name, author_email, subject = (
                        line.split("|", 4)
                    )
                    commits.append(
                        {
                            "hash": commit_hash,
                            "timestamp": int(timestamp),
                            "datetime": datetime.fromtimestamp(
                                int(timestamp)
                            ).isoformat(),
                            "author_name": author_name,
                            "author_email": author_email,
                            "subject": subject,
                        }
                    )
                except ValueError as e:
                    self.errors.append(
                        f"Error parsing commit line: {line[:100]}... Error: {e}"
                    )
                    continue

        return commits

    def parse_raw_diff_line(self, line, commit_hash):
        """
        Parse a single line from git's --raw output.

        Raw format is:
        :old_mode new_mode old_sha new_sha status[score]<tab>old_path<tab>new_path

        For example:
        :100644 100644 abc123... def456... M	file.txt
        :100644 100644 abc123... def456... R100	old.txt	new.txt
        :000000 100644 000000... abc123... A	new_file.txt

        The status is a single letter (A, M, D, R, C, T, U, X) followed by
        an optional similarity score for R and C.
        """
        if not line.strip() or not line.startswith(":"):
            return None

        try:
            # Split on colon first to separate the metadata from the paths
            # Format: :old_mode new_mode old_sha new_sha status\tpaths
            parts = line[1:].split("\t")  # Remove leading colon and split on tabs

            if len(parts) < 2:
                return None

            # First part contains: old_mode new_mode old_sha new_sha status
            metadata = parts[0].split()
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
                return {
                    "status": "added",
                    "path": parts[1],
                    "old_mode": old_mode,
                    "new_mode": new_mode,
                    "old_sha": old_sha,
                    "new_sha": new_sha,
                }

            elif status_letter == "M":  # Modified
                return {
                    "status": "modified",
                    "path": parts[1],
                    "old_mode": old_mode,
                    "new_mode": new_mode,
                    "old_sha": old_sha,
                    "new_sha": new_sha,
                }

            elif status_letter == "D":  # Deleted
                return {
                    "status": "deleted",
                    "path": parts[1],
                    "old_mode": old_mode,
                    "new_mode": new_mode,
                    "old_sha": old_sha,
                    "new_sha": new_sha,
                }

            elif status_letter == "R":  # Renamed
                if len(parts) < 3:
                    self.errors.append(
                        f"Invalid rename in commit {commit_hash[:8]}: {line}"
                    )
                    return None
                return {
                    "status": "renamed",
                    "old_path": parts[1],
                    "new_path": parts[2],
                    "similarity": score or "100",
                    "old_mode": old_mode,
                    "new_mode": new_mode,
                    "old_sha": old_sha,
                    "new_sha": new_sha,
                }

            elif status_letter == "C":  # Copied
                if len(parts) < 3:
                    self.errors.append(
                        f"Invalid copy in commit {commit_hash[:8]}: {line}"
                    )
                    return None
                return {
                    "status": "copied",
                    "source_path": parts[1],
                    "path": parts[2],
                    "similarity": score or "100",
                    "old_mode": old_mode,
                    "new_mode": new_mode,
                    "old_sha": old_sha,
                    "new_sha": new_sha,
                }

            elif status_letter == "T":  # Type change
                return {
                    "status": "type_changed",
                    "path": parts[1],
                    "old_mode": old_mode,
                    "new_mode": new_mode,
                    "old_sha": old_sha,
                    "new_sha": new_sha,
                }

            elif status_letter == "U":  # Unmerged
                # Skip unmerged files (merge conflicts)
                return None

            elif status_letter == "X":  # Unknown
                self.errors.append(
                    f"Unknown change type in commit {commit_hash[:8]}: {line}"
                )
                return None

            else:
                self.errors.append(
                    f"Unexpected status letter '{status_letter}' in commit {commit_hash[:8]}: {line}"
                )
                return None

        except (IndexError, ValueError) as e:
            self.errors.append(
                f"Error parsing raw diff in commit {commit_hash[:8]}: {line[:100]}... Error: {e}"
            )
            return None

    def get_commit_changes(self, commit_hash):
        """
        Get all file changes using --raw format.

        --raw provides structured output with:
        - File modes (permissions)
        - Object SHAs (Git's internal IDs)
        - Status letter + optional score
        - File paths

        This is Git's native structured format, not text parsing!
        """
        output = self.run_git_command(
            "show",
            "--raw",
            "--numstat",
            "--format=",
            commit_hash,
        )

        if not output:
            return []

        changes = []
        lines = output.split("\n")

        # Build mappings from --raw output first
        raw_changes = {}
        for line in lines:
            if line.startswith(":"):
                parsed = self.parse_raw_diff_line(line, commit_hash)
                if parsed:
                    # Use the final path as the key
                    key_path = parsed.get("new_path") or parsed.get("path")
                    if key_path:
                        raw_changes[key_path] = parsed

        # Then process --numstat output
        for line in lines:
            if not line.strip() or line.startswith(":"):
                continue

            parts = line.split("\t")
            if len(parts) >= 3:
                additions_str = parts[0]
                deletions_str = parts[1]
                filepath = parts[2]

                # Get the corresponding raw change
                raw_change = raw_changes.get(filepath, {})

                # Handle binary files
                is_binary = additions_str == "-" and deletions_str == "-"

                change = {
                    "filepath": filepath,
                    "additions": (
                        0
                        if is_binary
                        else (int(additions_str) if additions_str.isdigit() else 0)
                    ),
                    "deletions": (
                        0
                        if is_binary
                        else (int(deletions_str) if deletions_str.isdigit() else 0)
                    ),
                    "is_binary": is_binary,
                }

                # Merge with raw change data
                change.update(raw_change)
                changes.append(change)

        return changes

    def analyze_repository(self):
        """Analyze the entire repository."""
        click.echo(
            f"Analyzing repository: {click.style(str(self.repo_path), fg='cyan')}"
        )

        commits = self.get_all_commits()
        if not commits:
            click.echo(click.style("No commits found in repository", fg="yellow"))
            return

        click.echo(f"Found {len(commits)} commits to analyze...")

        with click.progressbar(commits, label="Processing commits") as bar:
            for commit in bar:
                changes = self.get_commit_changes(commit["hash"])

                for change in changes:
                    status = change.get("status", "modified")
                    filepath = (
                        change.get("filepath")
                        or change.get("new_path")
                        or change.get("path")
                    )

                    if not filepath:
                        continue

                    event = {
                        "event_type": status,
                        "datetime": commit["datetime"],
                        "timestamp": commit["timestamp"],
                        "commit_hash": commit["hash"],
                        "author_name": commit["author_name"],
                        "author_email": commit["author_email"],
                        "commit_subject": commit["subject"],
                        "additions": change.get("additions", 0),
                        "deletions": change.get("deletions", 0),
                        "is_binary": change.get("is_binary", False),
                    }

                    # Add status-specific fields
                    if status == "renamed":
                        event["old_path"] = change.get("old_path")
                        event["new_path"] = change.get("new_path")
                    elif status == "copied":
                        event["source_path"] = change.get("source_path")

                    self.file_lifecycle[filepath].append(event)

        click.echo(
            f"\n✓ Analyzed {len(self.file_lifecycle)} unique files across {len(commits)} commits"
        )

    def get_file_summary(self, filepath, events):
        """Generate a summary for a file's lifecycle."""
        if not events:
            return None

        first_event = events[0]
        last_event = events[-1]

        total_additions = sum(e.get("additions", 0) for e in events)
        total_deletions = sum(e.get("deletions", 0) for e in events)

        # Count different event types
        modifications = sum(1 for e in events if e["event_type"] == "modified")
        renames = sum(1 for e in events if e["event_type"] == "renamed")

        # Check if file is deleted
        is_deleted = last_event["event_type"] == "deleted"

        # Check if any event involved binary data
        has_binary = any(e.get("is_binary", False) for e in events)

        # Get unique authors
        authors = set(e["author_name"] for e in events)

        return {
            "filepath": filepath,
            "created_at": first_event["datetime"],
            "created_by": first_event["author_name"],
            "first_commit": first_event["commit_hash"],
            "last_modified_at": last_event["datetime"],
            "last_modified_by": last_event["author_name"],
            "last_commit": last_event["commit_hash"],
            "is_deleted": is_deleted,
            "has_binary": has_binary,
            "total_commits": len(events),
            "total_modifications": modifications,
            "total_renames": renames,
            "total_additions": total_additions,
            "total_deletions": total_deletions,
            "net_lines": total_additions - total_deletions,
            "unique_authors": len(authors),
            "authors": sorted(authors),
        }

    def export_to_json(self, output_file):
        """Export complete lifecycle data to JSON."""
        data = {
            "repository": str(self.repo_path),
            "analyzed_at": datetime.now().isoformat(),
            "total_files": len(self.file_lifecycle),
            "total_errors": len(self.errors),
            "files": {},
        }

        for filepath, events in self.file_lifecycle.items():
            data["files"][filepath] = {
                "summary": self.get_file_summary(filepath, events),
                "events": events,
            }

        with open(output_file, "w") as f:
            json.dump(data, f, indent=2)

        click.echo(
            f"✓ Exported lifecycle data to {click.style(output_file, fg='green')}"
        )

    def export_to_csv(self, output_file):
        """Export summary data to CSV."""
        with open(output_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)

            writer.writerow(
                [
                    "filepath",
                    "created_at",
                    "created_by",
                    "first_commit",
                    "datetime_last_modified",
                    "last_modified_by",
                    "last_commit",
                    "is_deleted",
                    "has_binary",
                    "total_commits",
                    "total_modifications",
                    "total_renames",
                    "total_additions",
                    "total_deletions",
                    "net_lines",
                    "unique_authors",
                    "authors",
                ]
            )

            for filepath, events in sorted(self.file_lifecycle.items()):
                summary = self.get_file_summary(filepath, events)
                if summary:
                    writer.writerow(
                        [
                            summary["filepath"],
                            summary["created_at"],
                            summary["created_by"],
                            summary["first_commit"],
                            summary["last_modified_at"],
                            summary["last_modified_by"],
                            summary["last_commit"],
                            summary["is_deleted"],
                            summary["has_binary"],
                            summary["total_commits"],
                            summary["total_modifications"],
                            summary["total_renames"],
                            summary["total_additions"],
                            summary["total_deletions"],
                            summary["net_lines"],
                            summary["unique_authors"],
                            "; ".join(summary["authors"]),
                        ]
                    )

        click.echo(f"✓ Exported summary data to {click.style(output_file, fg='green')}")

    def export_events_to_csv(self, output_file):
        """Export all events to a detailed CSV."""
        with open(output_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)

            writer.writerow(
                [
                    "filepath",
                    "event_type",
                    "event_datetime",
                    "datetime_last_modified",
                    "commit_hash",
                    "author_name",
                    "author_email",
                    "commit_subject",
                    "additions",
                    "deletions",
                    "is_binary",
                    "old_path",
                    "new_path",
                    "source_path",
                ]
            )

            for filepath, events in sorted(self.file_lifecycle.items()):
                last_modified = events[-1]["datetime"] if events else ""

                for event in events:
                    writer.writerow(
                        [
                            filepath,
                            event["event_type"],
                            event["datetime"],
                            last_modified,
                            event["commit_hash"],
                            event["author_name"],
                            event["author_email"],
                            event["commit_subject"],
                            event.get("additions", ""),
                            event.get("deletions", ""),
                            event.get("is_binary", ""),
                            event.get("old_path", ""),
                            event.get("new_path", ""),
                            event.get("source_path", ""),
                        ]
                    )

        click.echo(
            f"✓ Exported detailed events to {click.style(output_file, fg='green')}"
        )

    def export_errors(self, output_file):
        """Export errors to a text file."""
        if not self.errors:
            return

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(f"Git File Lifecycle Analyzer - Error Log\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n")
            f.write(f"Repository: {self.repo_path}\n")
            f.write(f"Total Errors: {len(self.errors)}\n")
            f.write("=" * 80 + "\n\n")

            for i, error in enumerate(self.errors, 1):
                f.write(f"{i}. {error}\n\n")

        click.echo(f"✓ Exported errors to {click.style(output_file, fg='yellow')}")

    def create_output_directory(self):
        """Create output directory named DATASETS_<repo_name>."""
        repo_name = self.repo_path.name
        output_dir = Path(f"DATASETS_{repo_name}")
        output_dir.mkdir(exist_ok=True)

        click.echo(
            f"✓ Created output directory: {click.style(str(output_dir), fg='cyan')}"
        )
        return output_dir

    def generate_metadata_report(self, output_files, output_path):
        """Generate a metadata report for all exported datasets."""
        report_lines = [
            "# Dataset Metadata Report",
            "",
            f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Repository:** {self.repo_path.name}",
            f"**Repository Path:** {self.repo_path}",
            "",
            "---",
            "",
        ]

        for file_path in output_files:
            if not Path(file_path).exists():
                continue

            file_name = Path(file_path).name
            file_size = Path(file_path).stat().st_size
            file_size_mb = file_size / (1024 * 1024)

            report_lines.append(f"## {file_name}")
            report_lines.append("")
            report_lines.append(
                f"**File Size:** {file_size:,} bytes ({file_size_mb:.2f} MB)"
            )
            report_lines.append("")

            # Analyze CSV files
            if file_path.endswith(".csv"):
                try:
                    df = pd.read_csv(file_path)

                    report_lines.append(f"**Format:** CSV (Comma-Separated Values)")
                    report_lines.append(
                        f"**Shape:** {df.shape[0]:,} rows × {df.shape[1]} columns"
                    )
                    report_lines.append("")

                    report_lines.append("**Columns:**")
                    for col in df.columns:
                        report_lines.append(f"- `{col}`")
                    report_lines.append("")

                    # Check for null values
                    null_counts = df.isnull().sum()
                    total_nulls = null_counts.sum()

                    if total_nulls > 0:
                        report_lines.append(
                            f"**Missing Data:** {total_nulls:,} null values found"
                        )
                        cols_with_nulls = null_counts[null_counts > 0]
                        for col, count in cols_with_nulls.items():
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

                    # Show sample data
                    report_lines.append("**Sample Data (first 3 rows):**")
                    report_lines.append("```")
                    report_lines.append(df.head(3).to_string(index=False))
                    report_lines.append("```")

                except Exception as e:
                    report_lines.append(f"**Error analyzing CSV:** {str(e)}")

            # Analyze JSON files
            elif file_path.endswith(".json"):
                try:
                    with open(file_path, "r") as f:
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
                                size_info = (
                                    f" ({len(value)} items)"
                                    if isinstance(value, (list, dict))
                                    else ""
                                )
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
                    if "total_files" in data:
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
    click.echo(click.style("Git File Lifecycle Analyzer", fg="cyan", bold=True))
    click.echo(click.style("Using Git's Native Structured Format (--raw)", fg="cyan"))
    click.echo()

    # Initialize analyzer
    analyzer = GitFileLifecycle(repo_path)

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
    if analyzer.errors:
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
