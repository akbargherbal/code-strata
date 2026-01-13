#!/usr/bin/env python3
"""
Git File Lifecycle Analyzer - JSON-Only Version
REFACTORING NOTES - This script took 5 days to stabilize.

Critical areas that cause silent data corruption if modified incorrectly:
1. Combined diff parsing (merge commits) - the formula `n_parents = (num_tokens - 3) // 2` is derived from git's internal format
2. Null-terminated stream parsing - field count (5) must match format string exactly
3. Path consumption counter - `i += parsed["consumed"]` must match actual paths consumed (renames=2, others=1)
4. All `-z` flags and `\x00` splitting - architectural decision for handling special chars in filenames

If something looks counterintuitive, there's a reason. Test on repos with merge commits, renames, and unusual filenames before committing changes.
"""

import subprocess
import json
import click
from datetime import datetime, timezone
from collections import defaultdict
from pathlib import Path
from typing import Optional, Dict, List, Generator, Set, Any, Tuple


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
        Handles both standard diffs (:...) and combined diffs (::...) for merges.
        """
        try:
            # Check for combined diff (merge commit) which starts with multiple colons
            is_combined = metadata_line.startswith("::")

            if is_combined:
                # --- COMBINED DIFF PARSING (MERGE COMMITS) ---
                # Format: ::mode1 mode2 mode3 sha1 sha2 sha3 status
                # Remove all leading colons
                clean_line = metadata_line.lstrip(":")
                metadata = clean_line.split()

                # Combined diffs don't usually show renames, so we consume 1 path
                if len(path_parts) < 1:
                    return None

                # The last item is the status (e.g., "MM", "AM")
                status_with_score = metadata[-1]

                # Check for unmerged status
                if "U" in status_with_score:
                    self.skipped_unmerged += 1
                    return None

                # Calculate indices based on number of parents
                # Structure: N parents -> N+1 modes, N+1 shas, 1 status
                # Total tokens = 2(N+1) + 1 = 2N + 3
                num_tokens = len(metadata)
                n_parents = (num_tokens - 3) // 2

                # Extract result state (last mode and last sha)
                result_mode_idx = n_parents
                result_sha_idx = 2 * n_parents + 1

                mode = (
                    metadata[result_mode_idx]
                    if result_mode_idx < len(metadata)
                    else "000000"
                )
                sha = (
                    metadata[result_sha_idx]
                    if result_sha_idx < len(metadata)
                    else "0" * 40
                )

                # Use first character of status for operation type (A, M, D, etc.)
                status = status_with_score[0] if status_with_score else "M"

                file_path = path_parts[0]

                return {
                    "change": {
                        "operation": status,
                        "path": file_path,
                        "mode": mode,
                        "sha": sha,
                    },
                    "consumed": 2,  # metadata line + 1 path
                }

            else:
                # --- STANDARD DIFF PARSING (NORMAL COMMITS) ---
                # Format: :mode1 mode2 sha1 sha2 status
                clean_line = metadata_line.lstrip(":")
                tokens = clean_line.split()

                if len(tokens) < 5:
                    return None

                # Extract components
                mode1, mode2, sha1, sha2, status_with_score = tokens[:5]

                # Extract operation and score
                status = status_with_score[0]
                score = None
                if len(status_with_score) > 1:
                    try:
                        score = int(status_with_score[1:])
                    except ValueError:
                        score = None

                # Determine paths consumed
                if status in ["R", "C"]:
                    # Rename/Copy: consume 2 paths (old and new)
                    if len(path_parts) < 2:
                        return None
                    old_path = path_parts[0]
                    new_path = path_parts[1]
                    paths_consumed = 3  # metadata + 2 paths
                    file_path = new_path
                else:
                    # Add/Modify/Delete: consume 1 path
                    if len(path_parts) < 1:
                        return None
                    file_path = path_parts[0]
                    paths_consumed = 2  # metadata + 1 path
                    old_path = None

                # Check for unmerged status
                if status == "U":
                    self.skipped_unmerged += 1
                    return None

                change = {
                    "operation": status,
                    "path": file_path,
                    "mode": mode2,
                    "sha": sha2,
                }

                if old_path:
                    change["old_path"] = old_path
                if score is not None:
                    change["score"] = score

                return {"change": change, "consumed": paths_consumed}

        except (ValueError, IndexError) as e:
            self.errors.append(
                f"Error parsing raw diff line in commit {commit_hash[:8]}: {metadata_line} - {e}"
            )
            return None

    def analyze_repository(self):
        """Main analysis loop - stream commits and build lifecycle data."""
        click.echo("Analyzing repository...")

        with click.progressbar(
            self.stream_commits(), label="Processing commits", show_pos=True
        ) as commits:
            for commit in commits:
                self.total_commits += 1

                # Get changes for this commit
                changes = self.get_commit_changes(commit["hash"])

                # Process each change
                for change in changes:
                    file_path = change["path"]

                    # Add lifecycle event
                    self.file_lifecycle[file_path].append(
                        {
                            "commit": commit["hash"],
                            "timestamp": commit["timestamp"],
                            "datetime": commit["datetime"],
                            "operation": change["operation"],
                            "author": commit["author_name"],
                            "author_email": commit["author_email"],
                            "subject": commit["subject"],
                        }
                    )

                    self.total_changes += 1

        click.echo(
            f"✓ Analyzed {click.style(str(self.total_commits), fg='green', bold=True)} commits"
        )
        click.echo(
            f"✓ Tracked {click.style(str(len(self.file_lifecycle)), fg='green', bold=True)} unique files"
        )
        click.echo(
            f"✓ Recorded {click.style(str(self.total_changes), fg='green', bold=True)} changes"
        )

        if self.skipped_unmerged > 0:
            click.echo(
                f"⚠ Skipped {click.style(str(self.skipped_unmerged), fg='yellow')} unmerged entries"
            )

    def create_output_directory(self):
        """Create output directory named DATASETS_<repo_name>."""
        repo_name = self.repo_path.name
        output_dir = Path(f"DATASETS_{repo_name}")
        output_dir.mkdir(exist_ok=True)
        return output_dir

    def export_to_json(self, output_path):
        """Export the file lifecycle data to JSON format."""
        click.echo(f"Exporting to JSON: {output_path}")

        output_data = {
            "analyzed_at": datetime.now(timezone.utc).isoformat(),
            "repository": str(self.repo_path),
            "total_files": len(self.file_lifecycle),
            "total_commits": self.total_commits,
            "total_changes": self.total_changes,
            "total_errors": len(self.errors),
            "skipped_unmerged": self.skipped_unmerged,
            "files": dict(self.file_lifecycle),
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        click.echo(
            f"✓ Exported {len(self.file_lifecycle):,} files to {click.style(str(output_path), fg='green')}"
        )

    def export_errors(self, output_path):
        """Export errors to a text file."""
        click.echo(f"Exporting errors: {output_path}")

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(f"Git File Lifecycle Analyzer - Error Log\n")
            f.write(f"Generated: {datetime.now(timezone.utc).isoformat()}\n")
            f.write(f"Repository: {self.repo_path}\n")
            f.write(f"\n{'=' * 80}\n\n")

            if self.skipped_unmerged > 0:
                f.write(f"Unmerged Entries Skipped: {self.skipped_unmerged}\n")
                f.write(
                    "Note: Unmerged entries (conflict markers) are skipped automatically.\n\n"
                )

            if self.errors:
                f.write(f"Total Errors: {len(self.errors)}\n\n")
                for i, error in enumerate(self.errors, 1):
                    f.write(f"Error {i}:\n{error}\n\n")
            else:
                f.write("No errors encountered.\n")

        click.echo(
            f"✓ Exported {len(self.errors)} errors to {click.style(str(output_path), fg='yellow')}"
        )

    def generate_frontend_assets(self, output_dir):
        """Generate pre-aggregated JSON files for frontend visualization."""
        click.echo("Generating frontend-optimized datasets...")

        generated_files = []

        # 1. File Activity Timeline (aggregated by month)
        timeline_file = output_dir / "timeline_monthly.json"
        timeline_data = self._aggregate_timeline_monthly()
        with open(timeline_file, "w", encoding="utf-8") as f:
            json.dump(timeline_data, f, indent=2)
        generated_files.append(str(timeline_file))
        click.echo(f"  ✓ Generated: {timeline_file.name}")

        # 2. Top Active Files
        top_files_file = output_dir / "top_active_files.json"
        top_files_data = self._get_top_active_files(limit=100)
        with open(top_files_file, "w", encoding="utf-8") as f:
            json.dump(top_files_data, f, indent=2)
        generated_files.append(str(top_files_file))
        click.echo(f"  ✓ Generated: {top_files_file.name}")

        # 3. File Type Statistics
        file_types_file = output_dir / "file_type_stats.json"
        file_types_data = self._get_file_type_stats()
        with open(file_types_file, "w", encoding="utf-8") as f:
            json.dump(file_types_data, f, indent=2)
        generated_files.append(str(file_types_file))
        click.echo(f"  ✓ Generated: {file_types_file.name}")

        # 4. Author Activity
        author_stats_file = output_dir / "author_activity.json"
        author_data = self._get_author_activity()
        with open(author_stats_file, "w", encoding="utf-8") as f:
            json.dump(author_data, f, indent=2)
        generated_files.append(str(author_stats_file))
        click.echo(f"  ✓ Generated: {author_stats_file.name}")

        return generated_files

    def _aggregate_timeline_monthly(self):
        """Aggregate file changes by month."""
        monthly_activity = defaultdict(lambda: {"adds": 0, "modifies": 0, "deletes": 0})

        for file_path, events in self.file_lifecycle.items():
            for event in events:
                # Parse datetime and extract year-month
                dt = datetime.fromisoformat(event["datetime"])
                month_key = dt.strftime("%Y-%m")

                op = event["operation"]
                if op == "A":
                    monthly_activity[month_key]["adds"] += 1
                elif op == "M":
                    monthly_activity[month_key]["modifies"] += 1
                elif op == "D":
                    monthly_activity[month_key]["deletes"] += 1

        # Convert to sorted list
        timeline = [
            {"month": month, **stats}
            for month, stats in sorted(monthly_activity.items())
        ]

        return {
            "timeline": timeline,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

    def _get_top_active_files(self, limit=100):
        """Get files with the most changes."""
        file_activity = [
            {
                "path": file_path,
                "change_count": len(events),
                "first_seen": min(e["datetime"] for e in events),
                "last_seen": max(e["datetime"] for e in events),
                "operations": {
                    "adds": sum(1 for e in events if e["operation"] == "A"),
                    "modifies": sum(1 for e in events if e["operation"] == "M"),
                    "deletes": sum(1 for e in events if e["operation"] == "D"),
                },
            }
            for file_path, events in self.file_lifecycle.items()
        ]

        # Sort by change count
        file_activity.sort(key=lambda x: x["change_count"], reverse=True)

        return {
            "top_files": file_activity[:limit],
            "total_files_analyzed": len(self.file_lifecycle),
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

    def _get_file_type_stats(self):
        """Aggregate statistics by file extension."""
        extension_stats = defaultdict(lambda: {"count": 0, "total_changes": 0})

        for file_path, events in self.file_lifecycle.items():
            # Extract extension
            ext = (
                Path(file_path).suffix.lower()
                if Path(file_path).suffix
                else "(no extension)"
            )

            extension_stats[ext]["count"] += 1
            extension_stats[ext]["total_changes"] += len(events)

        # Convert to sorted list
        stats_list = [
            {"extension": ext, **stats}
            for ext, stats in sorted(
                extension_stats.items(),
                key=lambda x: x[1]["total_changes"],
                reverse=True,
            )
        ]

        return {
            "file_types": stats_list,
            "total_extensions": len(extension_stats),
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

    def _get_author_activity(self):
        """Aggregate activity by author."""
        author_stats = defaultdict(
            lambda: {"commits": set(), "files_touched": set(), "total_changes": 0}
        )

        for file_path, events in self.file_lifecycle.items():
            for event in events:
                author = event["author_email"]
                author_stats[author]["commits"].add(event["commit"])
                author_stats[author]["files_touched"].add(file_path)
                author_stats[author]["total_changes"] += 1

        # Convert sets to counts and create list
        authors_list = [
            {
                "author": author,
                "email": author,
                "unique_commits": len(stats["commits"]),
                "files_touched": len(stats["files_touched"]),
                "total_changes": stats["total_changes"],
            }
            for author, stats in author_stats.items()
        ]

        # Sort by total changes
        authors_list.sort(key=lambda x: x["total_changes"], reverse=True)

        return {
            "authors": authors_list,
            "total_authors": len(authors_list),
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

    def generate_metadata_report(self, generated_files: List[str], output_path: str):
        """Generate a metadata report with Pandas-powered insights for JSON files."""
        click.echo(f"Generating metadata report: {output_path}")

        # Import pandas here (only when needed)
        try:
            import pandas as pd

            pandas_available = True
        except ImportError:
            pandas_available = False
            click.echo(
                click.style("⚠ Pandas not available - basic analysis only", fg="yellow")
            )

        report_lines = [
            "# Git File Lifecycle Analysis - Dataset Metadata",
            "",
            f"**Generated:** {datetime.now(timezone.utc).isoformat()}",
            f"**Repository:** `{self.repo_path}`",
            "",
            "---",
            "",
            "## Overview",
            "",
            f"This dataset contains Git file lifecycle analysis data exported in JSON format.",
            f"The analysis tracked **{len(self.file_lifecycle):,} unique files** across **{self.total_commits:,} commits**.",
            "",
            "---",
            "",
            "## Dataset Files",
            "",
        ]

        for file_path in sorted(generated_files):
            file_name = Path(file_path).name
            file_size = Path(file_path).stat().st_size
            size_mb = file_size / (1024 * 1024)

            report_lines.append(f"### 📄 {file_name}")
            report_lines.append("")
            report_lines.append(
                f"**File Size:** {size_mb:.2f} MB ({file_size:,} bytes)"
            )
            report_lines.append("")

            # Analyze JSON files with Pandas
            if file_path.endswith(".json") and pandas_available:
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        data = json.load(f)

                    # Basic structure info
                    report_lines.append(f"**Format:** JSON")
                    report_lines.append("")

                    # Handle different JSON structures
                    if isinstance(data, dict):
                        # Try to convert to DataFrame for analysis
                        df = None
                        nested_analysis = None

                        # Strategy 1: Direct dictionary (e.g., top_active_files, timeline_monthly)
                        if any(
                            key in data
                            for key in [
                                "top_files",
                                "timeline",
                                "file_types",
                                "authors",
                            ]
                        ):
                            for key in [
                                "top_files",
                                "timeline",
                                "file_types",
                                "authors",
                            ]:
                                if key in data and isinstance(data[key], list):
                                    try:
                                        df = pd.DataFrame(data[key])
                                        break
                                    except:
                                        pass

                        # Strategy 2: Main lifecycle file with nested structure
                        elif "files" in data and isinstance(data["files"], dict):
                            # This is the main lifecycle file - special handling
                            nested_analysis = self._analyze_nested_lifecycle(data, pd)

                        # If we have a DataFrame, analyze it
                        if df is not None and not df.empty:
                            report_lines.extend(
                                self._generate_dataframe_analysis(df, file_name)
                            )

                        # If we have nested analysis, add it
                        elif nested_analysis:
                            report_lines.extend(nested_analysis)

                        # Fallback to basic structure
                        else:
                            report_lines.append(
                                f"**Structure:** Dictionary with {len(data)} top-level keys"
                            )
                            report_lines.append("")
                            report_lines.append("**Top-level keys:**")
                            for key in list(data.keys())[:10]:
                                value = data[key]
                                value_type = type(value).__name__
                                if isinstance(value, (list, dict)):
                                    size_info = f" ({len(value)} items)"
                                    report_lines.append(
                                        f"- `{key}`: {value_type}{size_info}"
                                    )
                                else:
                                    report_lines.append(
                                        f"- `{key}`: {value_type} = {value}"
                                    )

                            if len(data) > 10:
                                report_lines.append(
                                    f"- *(and {len(data) - 10} more keys)*"
                                )

                    elif isinstance(data, list):
                        try:
                            df = pd.DataFrame(data)
                            report_lines.extend(
                                self._generate_dataframe_analysis(df, file_name)
                            )
                        except:
                            report_lines.append(
                                f"**Structure:** Array with {len(data)} items"
                            )

                    report_lines.append("")

                except Exception as e:
                    report_lines.append(f"**Error analyzing JSON:** {str(e)}")
                    report_lines.append("")

            # Analyze JSON files without Pandas (fallback)
            elif file_path.endswith(".json") and not pandas_available:
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        data = json.load(f)

                    report_lines.append(f"**Format:** JSON")
                    report_lines.append("")

                    if isinstance(data, dict):
                        report_lines.append(
                            f"**Structure:** Dictionary with {len(data)} keys"
                        )
                        if "total_files" in data:
                            report_lines.append("")
                            report_lines.append("**Summary Statistics:**")
                            for key in [
                                "total_files",
                                "total_commits",
                                "total_changes",
                                "total_errors",
                            ]:
                                if key in data:
                                    report_lines.append(
                                        f"- {key.replace('_', ' ').title()}: {data[key]:,}"
                                    )
                    elif isinstance(data, list):
                        report_lines.append(
                            f"**Structure:** Array with {len(data)} items"
                        )

                    report_lines.append("")

                except Exception as e:
                    report_lines.append(f"**Error analyzing JSON:** {str(e)}")
                    report_lines.append("")

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

    def _generate_dataframe_analysis(
        self, df: "pd.DataFrame", file_name: str
    ) -> List[str]:
        """Generate detailed analysis for a pandas DataFrame."""
        lines = []

        # 1. Shape and Size
        lines.append("#### 📊 Data Shape & Size")
        lines.append("")
        lines.append(f"- **Rows:** {len(df):,}")
        lines.append(f"- **Columns:** {len(df.columns)}")
        lines.append(
            f"- **Memory Usage:** {df.memory_usage(deep=True).sum() / 1024:.2f} KB"
        )
        lines.append("")

        # 2. Column Information
        lines.append("#### 📋 Column Information")
        lines.append("")
        lines.append("| Column | Data Type | Non-Null Count | Null % |")
        lines.append("|--------|-----------|----------------|---------|")

        for col in df.columns:
            dtype = str(df[col].dtype)
            non_null = df[col].count()
            null_pct = ((len(df) - non_null) / len(df) * 100) if len(df) > 0 else 0
            lines.append(f"| `{col}` | {dtype} | {non_null:,} | {null_pct:.1f}% |")

        lines.append("")

        # 3. Sample Records (first 3 rows)
        lines.append("#### 🔍 Sample Records")
        lines.append("")
        lines.append("First 3 records (showing key fields):")
        lines.append("")
        lines.append("```json")

        # Show first 3 records, but limit to key fields if many columns
        sample_df = df.head(3)
        if len(df.columns) > 10:
            # Show only first 6 columns
            sample_df = sample_df.iloc[:, :6]
            lines.append(f"// Showing first 6 of {len(df.columns)} columns")

        lines.append(sample_df.to_json(orient="records", indent=2))
        lines.append("```")
        lines.append("")

        # 4. Value Distributions for key columns
        lines.append("#### 📈 Data Distribution")
        lines.append("")

        # Analyze up to 5 interesting columns
        analyzed_cols = 0
        for col in df.columns:
            if analyzed_cols >= 5:
                break

            # For numeric columns
            if df[col].dtype in ["int64", "float64"]:
                stats = df[col].describe()
                lines.append(f"**{col}** (numeric):")
                lines.append(f"- Range: {stats['min']:.2f} to {stats['max']:.2f}")
                lines.append(f"- Mean: {stats['mean']:.2f}, Median: {stats['50%']:.2f}")
                lines.append("")
                analyzed_cols += 1

            # For categorical/object columns
            elif df[col].dtype == "object" or df[col].nunique() < 50:
                unique_count = df[col].nunique()
                lines.append(f"**{col}**:")
                lines.append(f"- Unique values: {unique_count:,}")

                if unique_count <= 10:
                    # Show all values with counts
                    value_counts = df[col].value_counts().head(10)
                    lines.append("- Distribution:")
                    for val, count in value_counts.items():
                        pct = count / len(df) * 100
                        lines.append(f"  - `{val}`: {count:,} ({pct:.1f}%)")
                else:
                    # Show top 5 values
                    value_counts = df[col].value_counts().head(5)
                    lines.append("- Top 5 values:")
                    for val, count in value_counts.items():
                        pct = count / len(df) * 100
                        lines.append(f"  - `{val}`: {count:,} ({pct:.1f}%)")

                lines.append("")
                analyzed_cols += 1

        return lines

    def _analyze_nested_lifecycle(self, data: dict, pd: "module") -> List[str]:
        """Analyze the nested lifecycle file structure."""
        lines = []

        lines.append("#### 📊 Lifecycle Data Structure")
        lines.append("")
        lines.append(
            "This file contains the complete lifecycle history for all tracked files."
        )
        lines.append("")

        # Extract summary stats
        files = data.get("files", {})
        lines.append(f"- **Total Files Tracked:** {len(files):,}")
        lines.append(f"- **Total Commits Analyzed:** {data.get('total_commits', 0):,}")
        lines.append(f"- **Total Changes Recorded:** {data.get('total_changes', 0):,}")
        lines.append("")

        # Analyze event structure
        if files:
            lines.append("#### 🔍 Event Structure")
            lines.append("")

            # Calculate statistics about events per file
            events_per_file = [len(events) for events in files.values()]

            import statistics

            lines.append(
                f"- **Average events per file:** {statistics.mean(events_per_file):.1f}"
            )
            lines.append(
                f"- **Median events per file:** {statistics.median(events_per_file):.0f}"
            )
            lines.append(f"- **Max events (single file):** {max(events_per_file):,}")
            lines.append(f"- **Min events (single file):** {min(events_per_file):,}")
            lines.append("")

            # Sample event structure
            lines.append("#### 📝 Event Record Example")
            lines.append("")
            lines.append("Each file has an array of lifecycle events:")
            lines.append("")
            lines.append("```json")

            # Get first file with events
            for file_path, events in list(files.items())[:1]:
                if events:
                    sample_event = {
                        "file": file_path,
                        "events": events[:2],  # Show first 2 events
                    }
                    lines.append(json.dumps(sample_event, indent=2))

            lines.append("```")
            lines.append("")

            # Operation distribution
            lines.append("#### 📈 Operation Distribution")
            lines.append("")

            operation_counts = {"A": 0, "M": 0, "D": 0, "R": 0, "C": 0}
            for events in files.values():
                for event in events:
                    op = event.get("operation", "M")
                    if op in operation_counts:
                        operation_counts[op] += 1
                    else:
                        operation_counts["M"] += 1  # Default unknown to modify

            total_ops = sum(operation_counts.values())
            if total_ops > 0:
                lines.append("| Operation | Count | Percentage |")
                lines.append("|-----------|-------|------------|")
                for op, count in sorted(
                    operation_counts.items(), key=lambda x: x[1], reverse=True
                ):
                    if count > 0:
                        pct = count / total_ops * 100
                        op_name = {
                            "A": "Add",
                            "M": "Modify",
                            "D": "Delete",
                            "R": "Rename",
                            "C": "Copy",
                        }.get(op, op)
                        lines.append(f"| {op_name} ({op}) | {count:,} | {pct:.1f}% |")
                lines.append("")

        lines.append("#### ⚠️ Data Considerations")
        lines.append("")
        lines.append(
            "- **Nested Structure:** Each file path maps to an array of lifecycle events"
        )
        lines.append(
            "- **Query Complexity:** Deep nesting requires recursive traversal"
        )
        lines.append(
            "- **Memory Requirements:** Large repositories may produce multi-MB files"
        )
        lines.append("- **Timestamp Format:** All timestamps in ISO 8601 UTC format")
        lines.append("")

        return lines


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
    "--output-errors",
    default="file_lifecycle_errors.txt",
    help="Output errors file path",
    show_default=True,
)
@click.option("--no-json", is_flag=True, help="Skip JSON export")
def main(
    repo_path,
    output_json,
    output_errors,
    no_json,
):
    """Analyze git repository and export to JSON.

    REPO_PATH: Path to the git repository to analyze
    """
    click.echo(
        click.style("Git File Lifecycle Analyzer - JSON-Only", fg="cyan", bold=True)
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
    output_errors = str(output_dir / output_errors)
    metadata_report = str(output_dir / "dataset_metadata.md")

    # Analyze repository
    analyzer.analyze_repository()

    # Track generated files for metadata report
    generated_files = []

    # Generate Frontend Assets
    click.echo()
    frontend_files = analyzer.generate_frontend_assets(output_dir)
    generated_files.extend(frontend_files)

    # Export JSON
    if not no_json:
        click.echo()
        analyzer.export_to_json(output_json)
        generated_files.append(output_json)

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
