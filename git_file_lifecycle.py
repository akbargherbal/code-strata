#!/usr/bin/env python3
"""
Git File Lifecycle Analyzer - Enhanced Version with Phase 1 Features
Adds sidecar file infrastructure while maintaining 100% backward compatibility.

NEW IN PHASE 1:
- Multi-file output directory structure
- Manifest system for dataset catalog
- Base aggregator framework
- Feature flags for new functionality

BACKWARD COMPATIBILITY: 
- file_lifecycle.json remains UNCHANGED
- All new features behind --enable-aggregations flag (default: False)
"""

import subprocess
import json
import click
import hashlib
from datetime import datetime, timezone
from collections import defaultdict
from pathlib import Path
from typing import Optional, Dict, List, Generator, Set, Any, Tuple
from abc import ABC, abstractmethod


# ============================================================================
# Phase 1.3: Base Aggregator Class
# ============================================================================

class DatasetAggregator(ABC):
    """
    Base class for generating additional datasets.
    All aggregators must inherit from this class.
    """
    
    def __init__(self, analyzer: "GitFileLifecycle"):
        self.analyzer = analyzer
        self.data = {}
        self.errors = []
        
    @abstractmethod
    def get_name(self) -> str:
        """Return the name of this aggregator"""
        pass
    
    @abstractmethod
    def get_output_path(self, output_dir: Path) -> Path:
        """Return the output file path for this aggregator"""
        pass
    
    def process_commit(self, commit: dict, changes: list):
        """
        Called for each commit during streaming.
        Subclasses can override to process commit data.
        """
        pass
    
    def finalize(self) -> dict:
        """
        Called after all commits processed.
        Subclasses should override to compute final aggregations.
        Returns: Dictionary of aggregated data
        """
        return self.data
    
    def export(self, output_path: Path):
        """Export aggregated data to JSON file"""
        data = self.finalize()
        
        # Ensure parent directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)


# ============================================================================
# Phase 1.2: Manifest System
# ============================================================================

class ManifestGenerator:
    """
    Generates manifest.json that catalogs all generated datasets.
    Tracks metadata, dependencies, and integrity information.
    """
    
    def __init__(self, generator_version: str, repository_path: str):
        self.generator_version = generator_version
        self.repository_path = repository_path
        self.generated_at = datetime.now(timezone.utc).isoformat()
        self.datasets = {}
    
    def add_dataset(
        self,
        dataset_id: str,
        file_path: Path,
        schema_version: str,
        format_type: str,
        record_count: int,
        production_ready: bool = True,
        depends_on: List[str] = None,
        preview: bool = False
    ):
        """Add a dataset to the manifest"""
        
        # Calculate file size and checksum
        file_size = file_path.stat().st_size if file_path.exists() else 0
        sha256_hash = self._calculate_sha256(file_path) if file_path.exists() else None
        
        self.datasets[dataset_id] = {
            "file": str(file_path.name),
            "schema_version": schema_version,
            "format": format_type,
            "record_count": record_count,
            "file_size_bytes": file_size,
            "sha256": sha256_hash,
            "production_ready": production_ready,
            "depends_on": depends_on or [],
            "preview": preview
        }
    
    def _calculate_sha256(self, file_path: Path) -> str:
        """Calculate SHA-256 hash of file"""
        sha256 = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                sha256.update(chunk)
        return sha256.hexdigest()
    
    def export(self, output_path: Path):
        """Export manifest to JSON"""
        manifest = {
            "generator_version": self.generator_version,
            "generated_at": self.generated_at,
            "repository": self.repository_path,
            "datasets": self.datasets
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)


# ============================================================================
# Original GitFileLifecycle Class (with Phase 1 enhancements)
# ============================================================================

class GitFileLifecycle:
    VERSION = "2.0.0-phase1"  # Phase 1 version
    
    def __init__(self, repo_path, enable_aggregations=False):
        self.repo_path = Path(repo_path).resolve()
        self.file_lifecycle = defaultdict(list)
        self.errors = []
        self.skipped_unmerged = 0
        self.total_commits = 0
        self.total_changes = 0
        
        # Phase 1: Aggregator support
        self.enable_aggregations = enable_aggregations
        self.aggregators: List[DatasetAggregator] = []
        self.manifest: Optional[ManifestGenerator] = None

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

    def create_output_directory(self) -> Path:
        """
        Phase 1.1: Enhanced output directory structure.
        Creates subdirectories for organized dataset storage.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path.cwd() / f"git_analysis_output_{timestamp}"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Phase 1: Create subdirectories for new datasets
        if self.enable_aggregations:
            (output_dir / "metadata").mkdir(exist_ok=True)
            (output_dir / "aggregations").mkdir(exist_ok=True)
            (output_dir / "networks").mkdir(exist_ok=True)
        
        click.echo(f"Output directory: {click.style(str(output_dir), fg='cyan')}")
        return output_dir

    def register_aggregator(self, aggregator: DatasetAggregator):
        """Phase 1.4: Register an aggregator for processing"""
        self.aggregators.append(aggregator)

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

                if n_parents < 1:
                    return None

                # Parse the status from combined diff
                status = status_with_score[0] if status_with_score else "M"

                # Get the file path
                file_path = path_parts[0]

                return {
                    "change": {
                        "path": file_path,
                        "operation": status,
                    },
                    "consumed": 1,
                }

            else:
                # --- STANDARD DIFF PARSING ---
                # Format: :oldmode newmode oldsha newsha status[similarity]\0path[\0path2]
                clean_line = metadata_line[1:]  # Remove leading ':'
                metadata = clean_line.split()

                if len(metadata) < 5:
                    return None

                status_with_score = metadata[4]

                # Check for unmerged status
                if status_with_score.startswith("U"):
                    self.skipped_unmerged += 1
                    return None

                # Parse status and similarity score
                status = status_with_score[0]
                similarity = None

                # Extract similarity score if present (e.g., R095 -> 95)
                if len(status_with_score) > 1 and status in ("R", "C"):
                    try:
                        similarity = int(status_with_score[1:])
                    except ValueError:
                        pass

                # Determine how many paths to consume
                if status in ("R", "C"):  # Rename or Copy
                    if len(path_parts) < 2:
                        return None
                    old_path = path_parts[0]
                    new_path = path_parts[1]

                    return {
                        "change": {
                            "path": new_path,
                            "operation": status,
                            "old_path": old_path,
                            "similarity": similarity,
                        },
                        "consumed": 2,
                    }
                else:  # A, M, D, T
                    if len(path_parts) < 1:
                        return None

                    file_path = path_parts[0]

                    return {
                        "change": {
                            "path": file_path,
                            "operation": status,
                        },
                        "consumed": 1,
                    }

        except (IndexError, ValueError) as e:
            self.errors.append(
                f"Error parsing diff line for commit {commit_hash[:8]}: {e}"
            )
            return None

    def analyze_repository(self):
        """
        Phase 1.4: Enhanced analysis with aggregator support.
        Main analysis loop - maintains backward compatibility.
        """
        click.echo(click.style("\n📊 Analyzing repository...", fg="cyan", bold=True))

        with click.progressbar(
            self.stream_commits(),
            label="Processing commits",
            item_show_func=lambda x: x["hash"][:8] if x else None,
        ) as commits:
            for commit in commits:
                self.total_commits += 1

                # Get changes for this commit
                changes = self.get_commit_changes(commit["hash"])

                # Process changes for lifecycle tracking (original behavior)
                for change in changes:
                    self.total_changes += 1
                    file_path = change["path"]

                    event = {
                        "commit_hash": commit["hash"],
                        "timestamp": commit["timestamp"],
                        "datetime": commit["datetime"],
                        "operation": change["operation"],
                        "author_name": commit["author_name"],
                        "author_email": commit["author_email"],
                        "commit_subject": commit["subject"],
                    }

                    # Add optional fields if present
                    if "old_path" in change:
                        event["old_path"] = change["old_path"]
                    if "similarity" in change:
                        event["similarity"] = change["similarity"]

                    self.file_lifecycle[file_path].append(event)

                # Phase 1.4: Call aggregators if enabled
                if self.enable_aggregations:
                    for aggregator in self.aggregators:
                        try:
                            aggregator.process_commit(commit, changes)
                        except Exception as e:
                            self.errors.append(
                                f"Aggregator {aggregator.get_name()} error: {e}"
                            )

        click.echo(
            f"\n✓ Analyzed {click.style(str(self.total_commits), fg='green', bold=True)} commits"
        )
        click.echo(
            f"✓ Tracked {click.style(str(self.total_changes), fg='green', bold=True)} file changes"
        )
        click.echo(
            f"✓ Monitoring {click.style(str(len(self.file_lifecycle)), fg='green', bold=True)} unique files"
        )

        if self.skipped_unmerged > 0:
            click.echo(
                click.style(
                    f"⚠ Skipped {self.skipped_unmerged} unmerged changes",
                    fg="yellow",
                )
            )

    def export_to_json(self, output_file):
        """
        Export lifecycle data to JSON.
        UNCHANGED - maintains production schema.
        """
        click.echo(click.style("\n💾 Exporting to JSON...", fg="cyan"))

        # Convert defaultdict to regular dict for JSON serialization
        data = {
            "files": dict(self.file_lifecycle),
            "total_commits": self.total_commits,
            "total_changes": self.total_changes,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "repository_path": str(self.repo_path),
            "skipped_unmerged": self.skipped_unmerged,
        }

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        file_size = Path(output_file).stat().st_size
        click.echo(
            f"✓ Exported: {click.style(output_file, fg='green')} ({file_size:,} bytes)"
        )

    def export_aggregators(self, output_dir: Path) -> List[Path]:
        """Phase 1.4: Export all aggregator outputs"""
        if not self.enable_aggregations:
            return []
        
        click.echo(click.style("\n📦 Exporting aggregations...", fg="cyan"))
        exported_files = []
        
        for aggregator in self.aggregators:
            try:
                output_path = aggregator.get_output_path(output_dir)
                aggregator.export(output_path)
                click.echo(f"✓ {aggregator.get_name()}: {output_path.name}")
                exported_files.append(output_path)
            except Exception as e:
                self.errors.append(f"Failed to export {aggregator.get_name()}: {e}")
                click.echo(click.style(f"✗ {aggregator.get_name()}: {e}", fg="red"))
        
        return exported_files

    def generate_manifest(self, output_dir: Path, generated_files: List[Path]):
        """Phase 1.2: Generate manifest.json"""
        if not self.enable_aggregations:
            return None
        
        click.echo(click.style("\n📋 Generating manifest...", fg="cyan"))
        
        self.manifest = ManifestGenerator(
            generator_version=self.VERSION,
            repository_path=str(self.repo_path)
        )
        
        # Add core lifecycle dataset
        core_file = output_dir / "file_lifecycle.json"
        if core_file.exists():
            self.manifest.add_dataset(
                dataset_id="core_lifecycle",
                file_path=core_file,
                schema_version="1.0.0",
                format_type="nested",
                record_count=self.total_changes,
                production_ready=True
            )
        
        # Add aggregator datasets
        for aggregator, file_path in zip(self.aggregators, generated_files):
            self.manifest.add_dataset(
                dataset_id=aggregator.get_name(),
                file_path=file_path,
                schema_version="1.0.0",
                format_type="json",
                record_count=0,  # Aggregators should provide this
                production_ready=False,
                preview=True,
                depends_on=["core_lifecycle"]
            )
        
        # Export manifest
        manifest_path = output_dir / "manifest.json"
        self.manifest.export(manifest_path)
        click.echo(f"✓ Manifest: {manifest_path.name}")
        
        return manifest_path

    def export_errors(self, output_file):
        """Export errors to file"""
        if not self.errors and self.skipped_unmerged == 0:
            return

        click.echo(click.style("\n⚠️  Writing errors log...", fg="yellow"))

        with open(output_file, "w", encoding="utf-8") as f:
            f.write("Git File Lifecycle Analyzer - Error Log\n")
            f.write("=" * 60 + "\n\n")

            if self.skipped_unmerged > 0:
                f.write(f"Skipped unmerged changes: {self.skipped_unmerged}\n\n")

            if self.errors:
                f.write(f"Errors encountered: {len(self.errors)}\n\n")
                for i, error in enumerate(self.errors, 1):
                    f.write(f"{i}. {error}\n")

        click.echo(f"✓ Errors logged: {click.style(output_file, fg='yellow')}")


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
@click.option(
    "--enable-aggregations",
    is_flag=True,
    default=False,
    help="Enable new aggregation features (Phase 1+)"
)
@click.option("--version", is_flag=True, help="Show version and exit")
def main(
    repo_path,
    output_json,
    output_errors,
    no_json,
    enable_aggregations,
    version,
):
    """
    Analyze git repository and export to JSON.

    REPO_PATH: Path to the git repository to analyze
    
    Phase 1 Features:
    - Sidecar file infrastructure
    - Manifest system
    - Base aggregator framework
    """
    
    if version:
        click.echo(f"Git File Lifecycle Analyzer v{GitFileLifecycle.VERSION}")
        return
    
    click.echo(
        click.style("Git File Lifecycle Analyzer", fg="cyan", bold=True)
    )
    click.echo(click.style(f"Version: {GitFileLifecycle.VERSION}", fg="cyan"))
    
    if enable_aggregations:
        click.echo(click.style("⚡ Aggregations enabled (Phase 1)", fg="yellow"))
    
    click.echo()

    # Initialize analyzer
    analyzer = GitFileLifecycle(repo_path, enable_aggregations=enable_aggregations)

    # Validate repository
    try:
        analyzer.validate_repository()
    except ValueError as e:
        click.echo(click.style(f"✗ {e}", fg="red", bold=True))
        return

    # Create output directory
    output_dir = analyzer.create_output_directory()

    # Update output paths to use the new directory
    output_json_path = output_dir / output_json
    output_errors_path = output_dir / output_errors

    # Analyze repository
    analyzer.analyze_repository()

    # Export core lifecycle JSON (UNCHANGED)
    if not no_json:
        click.echo()
        analyzer.export_to_json(str(output_json_path))

    # Phase 1: Export aggregators
    generated_files = []
    if enable_aggregations:
        aggregator_files = analyzer.export_aggregators(output_dir)
        generated_files.extend(aggregator_files)
        
        # Generate manifest
        if aggregator_files:
            manifest_path = analyzer.generate_manifest(output_dir, aggregator_files)
            if manifest_path:
                generated_files.append(manifest_path)

    # Export errors if any
    if analyzer.errors or analyzer.skipped_unmerged > 0:
        click.echo()
        analyzer.export_errors(str(output_errors_path))

    click.echo()
    click.secho("✓ Analysis complete!", fg="green", bold=True)
    click.echo(
        f"All outputs saved to: {click.style(str(output_dir), fg='cyan', bold=True)}"
    )


if __name__ == "__main__":
    main()
