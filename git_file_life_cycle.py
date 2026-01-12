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
        if not line.strip() or not line.startswith(':'):
            return None
        
        try:
            # Split on colon first to separate the metadata from the paths
            # Format: :old_mode new_mode old_sha new_sha status\tpaths
            parts = line[1:].split('\t')  # Remove leading colon and split on tabs
            
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
            if status_letter == 'A':  # Added
                return {
                    'status': 'added',
                    'path': parts[1],
                    'old_mode': old_mode,
                    'new_mode': new_mode,
                    'old_sha': old_sha,
                    'new_sha': new_sha,
                }
            
            elif status_letter == 'M':  # Modified
                return {
                    'status': 'modified',
                    'path': parts[1],
                    'old_mode': old_mode,
                    'new_mode': new_mode,
                    'old_sha': old_sha,
                    'new_sha': new_sha,
                }
            
            elif status_letter == 'D':  # Deleted
                return {
                    'status': 'deleted',
                    'path': parts[1],
                    'old_mode': old_mode,
                    'new_mode': new_mode,
                    'old_sha': old_sha,
                    'new_sha': new_sha,
                }
            
            elif status_letter == 'R':  # Renamed
                if len(parts) < 3:
                    self.errors.append(
                        f"Invalid rename in commit {commit_hash[:8]}: {line}"
                    )
                    return None
                return {
                    'status': 'renamed',
                    'old_path': parts[1],
                    'new_path': parts[2],
                    'similarity': score or '100',
                    'old_mode': old_mode,
                    'new_mode': new_mode,
                    'old_sha': old_sha,
                    'new_sha': new_sha,
                }
            
            elif status_letter == 'C':  # Copied
                if len(parts) < 3:
                    self.errors.append(
                        f"Invalid copy in commit {commit_hash[:8]}: {line}"
                    )
                    return None
                return {
                    'status': 'copied',
                    'source_path': parts[1],
                    'path': parts[2],
                    'similarity': score or '100',
                    'old_mode': old_mode,
                    'new_mode': new_mode,
                    'old_sha': old_sha,
                    'new_sha': new_sha,
                }
            
            elif status_letter == 'T':  # Type change
                return {
                    'status': 'type_changed',
                    'path': parts[1],
                    'old_mode': old_mode,
                    'new_mode': new_mode,
                    'old_sha': old_sha,
                    'new_sha': new_sha,
                }
            
            elif status_letter == 'U':  # Unmerged
                # Skip unmerged files (merge conflicts)
                return None
            
            elif status_letter == 'X':  # Unknown
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
            "diff-tree",
            "--no-commit-id",  # Don't show commit ID
            "--raw",           # Use structured raw format
            "-r",              # Recursive into subdirectories
            "-M",              # Detect renames
            "-C",              # Detect copies (optional, can remove if not needed)
            commit_hash,
        )

        if not output:
            return []

        changes = []
        for line in output.split("\n"):
            change = self.parse_raw_diff_line(line, commit_hash)
            if change:
                changes.append(change)

        return changes

    def get_file_stats(self, commit_hash, filepath):
        """Get line addition/deletion stats for a file in a commit."""
        try:
            output = self.run_git_command(
                "show", "--numstat", "--format=", commit_hash, "--", filepath
            )

            if output:
                for line in output.split("\n"):
                    if not line.strip():
                        continue
                    parts = line.split("\t")
                    if len(parts) >= 2:
                        additions = parts[0]
                        deletions = parts[1]

                        # Handle binary files (marked with '-')
                        if additions == "-" or deletions == "-":
                            return {"additions": 0, "deletions": 0, "is_binary": True}

                        return {
                            "additions": int(additions),
                            "deletions": int(deletions),
                            "is_binary": False,
                        }
        except (ValueError, IndexError) as e:
            self.errors.append(
                f"Error getting stats for {filepath} in commit {commit_hash[:8]}: {e}"
            )

        return {"additions": 0, "deletions": 0, "is_binary": False}

    def analyze_repository(self):
        """Analyze the entire repository and build file lifecycle data."""
        click.echo(
            f"Analyzing repository: {click.style(str(self.repo_path), fg='yellow')}"
        )

        commits = self.get_all_commits()
        click.echo(
            f"Found {click.style(str(len(commits)), fg='cyan', bold=True)} commits"
        )

        if not commits:
            click.echo(
                click.style("No commits found or error reading repository", fg="red")
            )
            return self.file_lifecycle

        click.echo()

        # Track current file paths (handle renames)
        file_mappings = {}  # old_path -> current_path

        with click.progressbar(
            commits, label="Processing commits", show_pos=True, show_percent=True
        ) as bar:
            for commit in bar:
                try:
                    changes = self.get_commit_changes(commit["hash"])

                    for change in changes:
                        event = {
                            "commit_hash": commit["hash"],
                            "timestamp": commit["timestamp"],
                            "datetime": commit["datetime"],
                            "author_name": commit["author_name"],
                            "author_email": commit["author_email"],
                            "commit_subject": commit["subject"],
                        }

                        if change["status"] == "renamed":
                            old_path = change["old_path"]
                            new_path = change["new_path"]

                            # Update file mapping
                            if old_path in file_mappings:
                                original_path = file_mappings[old_path]
                                file_mappings[new_path] = original_path
                                del file_mappings[old_path]
                                tracked_path = original_path
                            else:
                                file_mappings[new_path] = old_path
                                tracked_path = old_path

                            event.update(
                                {
                                    "event_type": "renamed",
                                    "old_path": old_path,
                                    "new_path": new_path,
                                    "similarity": change.get("similarity", "100"),
                                }
                            )

                            self.file_lifecycle[tracked_path].append(event)

                        elif change["status"] == "copied":
                            source_path = change["source_path"]
                            dest_path = change["path"]

                            stats = self.get_file_stats(commit["hash"], dest_path)
                            event.update(
                                {
                                    "event_type": "copied",
                                    "source_path": source_path,
                                    "similarity": change.get("similarity", "100"),
                                    **stats,
                                }
                            )

                            self.file_lifecycle[dest_path].append(event)

                        elif change["status"] in ["added", "modified", "type_changed"]:
                            filepath = change["path"]

                            if filepath in file_mappings:
                                tracked_path = file_mappings[filepath]
                            else:
                                tracked_path = filepath

                            stats = self.get_file_stats(commit["hash"], filepath)
                            event.update(
                                {
                                    "event_type": change["status"],
                                    **stats,
                                }
                            )

                            self.file_lifecycle[tracked_path].append(event)

                        elif change["status"] == "deleted":
                            filepath = change["path"]

                            if filepath in file_mappings:
                                tracked_path = file_mappings[filepath]
                            else:
                                tracked_path = filepath

                            event.update({"event_type": "deleted"})

                            self.file_lifecycle[tracked_path].append(event)

                except Exception as e:
                    self.errors.append(
                        f"Error processing commit {commit['hash'][:8]}: {str(e)}"
                    )
                    continue

        click.echo()
        click.echo(
            f"Found {click.style(str(len(self.file_lifecycle)), fg='cyan', bold=True)} unique files"
        )

        if self.errors:
            click.echo()
            click.echo(
                click.style(
                    f"⚠ Encountered {len(self.errors)} errors during processing",
                    fg="yellow",
                )
            )
            click.echo("First few errors:")
            for error in self.errors[:5]:
                click.echo(f"  - {error}")

        return self.file_lifecycle

    def get_file_summary(self, filepath, events):
        """Generate summary statistics for a file."""
        if not events:
            return None

        first_event = events[0]
        last_event = events[-1]

        total_modifications = sum(
            1 for e in events if e["event_type"] in ["added", "modified"]
        )
        total_renames = sum(1 for e in events if e["event_type"] == "renamed")

        total_additions = sum(e.get("additions", 0) for e in events)
        total_deletions = sum(e.get("deletions", 0) for e in events)

        authors = set(e["author_email"] for e in events)

        has_binary = any(e.get("is_binary", False) for e in events)

        return {
            "filepath": filepath,
            "created_at": first_event["datetime"],
            "created_by": first_event["author_email"],
            "first_commit": first_event["commit_hash"],
            "last_modified_at": last_event["datetime"],
            "last_modified_by": last_event["author_email"],
            "last_commit": last_event["commit_hash"],
            "is_deleted": last_event["event_type"] == "deleted",
            "has_binary": has_binary,
            "total_commits": len(events),
            "total_modifications": total_modifications,
            "total_renames": total_renames,
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

    # Analyze repository
    analyzer.analyze_repository()

    # Export data
    if not no_json:
        click.echo()
        analyzer.export_to_json(output_json)

    if not no_csv:
        click.echo()
        analyzer.export_to_csv(output_summary_csv)
        analyzer.export_events_to_csv(output_events_csv)

    # Export errors if any
    if analyzer.errors:
        click.echo()
        analyzer.export_errors(output_errors)

    click.echo()
    click.secho("✓ Analysis complete!", fg="green", bold=True)


if __name__ == "__main__":
    main()