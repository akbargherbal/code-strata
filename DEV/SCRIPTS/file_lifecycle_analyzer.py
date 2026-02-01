#!/usr/bin/env python3
"""
File Lifecycle Analyzer
Analyzes git repository file lifecycle data and generates aggregated reports.
"""

import sys
import logging
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from datetime import datetime, timezone
from itertools import chain
from collections import Counter

import click
import pandas as pd
import regex as re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


class FileLifecycleAnalyzer:
    """Analyzes file lifecycle data from git repositories."""

    def __init__(self, input_file: Path, output_dir: Path):
        """
        Initialize the analyzer.

        Args:
            input_file: Path to input JSON file
            output_dir: Directory to save output files
        """
        self.input_file = input_file
        self.output_dir = output_dir
        self.df_file_lifecycle: Optional[pd.DataFrame] = None
        self.df_commit_details: Optional[pd.DataFrame] = None
        self.df_period: Optional[pd.DataFrame] = None

    def load_data(self) -> None:
        """Load and validate input data."""
        try:
            logger.info(f"Loading data from {self.input_file}")
            self.df_file_lifecycle = pd.read_json(self.input_file)

            # Validate required columns
            required_cols = [
                "files",
                "total_commits",
                "total_changes",
                "generated_at",
                "repository_path",
                "skipped_unmerged",
            ]
            missing_cols = set(required_cols) - set(self.df_file_lifecycle.columns)
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")

            logger.info(f"Successfully loaded {len(self.df_file_lifecycle)} records")

        except FileNotFoundError:
            logger.error(f"Input file not found: {self.input_file}")
            raise
        except ValueError as e:
            logger.error(f"Invalid JSON format: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error loading data: {e}")
            raise

    def prepare_file_lifecycle(self) -> None:
        """Prepare and transform file lifecycle dataframe."""
        try:
            logger.info("Preparing file lifecycle data")

            # Reset index and rename
            self.df_file_lifecycle = self.df_file_lifecycle.reset_index()
            self.df_file_lifecycle = self.df_file_lifecycle.rename(
                columns={"index": "FILE_PATH"}
            )

            # Add parent directory column
            self.df_file_lifecycle["PARENT_DIR"] = self.df_file_lifecycle[
                "FILE_PATH"
            ].apply(lambda x: Path(x).parent.as_posix())

            # Create commit details dictionary
            self._extract_commit_details()

            logger.info("File lifecycle data prepared successfully")

        except Exception as e:
            logger.error(f"Error preparing file lifecycle data: {e}")
            raise

    def _extract_commit_details(self) -> None:
        """Extract commit details from file lifecycle data."""
        try:
            logger.info("Extracting commit details")

            # Build commit details dictionary
            list_commits = list(self.df_file_lifecycle["files"])
            dict_commits = {}

            for file_commits in list_commits:
                for commit in file_commits:
                    commit_hash = commit["commit_hash"]
                    if commit_hash not in dict_commits:
                        dict_commits[commit_hash] = {
                            "timestamp": commit["timestamp"],
                            "datetime": commit["datetime"],
                            "author_name": commit["author_name"],
                            "author_email": commit["author_email"],
                        }

            # Create commit details dataframe
            df_commits = pd.DataFrame(dict_commits.items())
            df_commits.columns = ["COMMIT_HASH", "DETAILS"]

            self.df_commit_details = df_commits.copy()
            self.df_commit_details["DATETIME"] = self.df_commit_details[
                "DETAILS"
            ].apply(lambda x: x["datetime"])
            self.df_commit_details["AUTHOR_NAME"] = self.df_commit_details[
                "DETAILS"
            ].apply(lambda x: x["author_name"])
            self.df_commit_details["AUTHOR_EMAIL"] = self.df_commit_details[
                "DETAILS"
            ].apply(lambda x: x["author_email"])

            logger.info(f"Extracted {len(self.df_commit_details)} unique commits")

        except KeyError as e:
            logger.error(f"Missing expected key in commit data: {e}")
            raise
        except Exception as e:
            logger.error(f"Error extracting commit details: {e}")
            raise

    def generate_period_labels(self, period: str = "weekly") -> None:
        """
        Generate time period labels for commits.

        Args:
            period: Aggregation period ('weekly', 'monthly', 'quarterly')
        """
        try:
            logger.info(f"Generating {period} period labels")

            # Convert datetime column to pandas datetime
            self.df_commit_details["DATETIME"] = pd.to_datetime(
                self.df_commit_details["DATETIME"], utc=True
            )

            # Get date range
            min_date = self.df_commit_details["DATETIME"].min()
            max_date = self.df_commit_details["DATETIME"].max()

            logger.info(f"Date range: {min_date} to {max_date}")

            if period == "weekly":
                self._generate_weekly_labels(min_date, max_date)
            elif period == "monthly":
                self._generate_monthly_labels(min_date, max_date)
            elif period == "quarterly":
                self._generate_quarterly_labels(min_date, max_date)
            else:
                raise ValueError(
                    f"Invalid period: {period}. Must be 'weekly', 'monthly', or 'quarterly'"
                )

            # Map commits to periods
            self._map_commits_to_periods()

            # Add period labels to file lifecycle
            self._add_period_labels_to_files()

            logger.info(f"Period labels generated successfully")

        except Exception as e:
            logger.error(f"Error generating period labels: {e}")
            raise

    def _generate_weekly_labels(
        self, min_date: pd.Timestamp, max_date: pd.Timestamp
    ) -> None:
        """Generate weekly period labels."""
        # Start from the Monday on or before min_date
        days_since_monday = min_date.weekday()  # Monday = 0
        week_start_date = min_date.normalize() - pd.Timedelta(days=days_since_monday)

        date_range = pd.date_range(
            start=week_start_date,
            end=max_date.normalize() + pd.Timedelta(days=7),
            freq="W-MON",
        )

        periods = []
        for i in range(len(date_range) - 1):
            week_start = date_range[i]
            week_end = date_range[i + 1] - pd.Timedelta(microseconds=1)
            year = week_start.year
            week = week_start.isocalendar()[1]
            periods.append(
                {
                    "period_start": week_start,
                    "period_end": week_end,
                    "year": year,
                    "period_label": (year, week),
                }
            )

        self.df_period = pd.DataFrame(periods)

    def _generate_monthly_labels(
        self, min_date: pd.Timestamp, max_date: pd.Timestamp
    ) -> None:
        """Generate monthly period labels."""
        # Start from the first day of the month containing min_date
        month_start_date = min_date.normalize().replace(day=1)

        date_range = pd.date_range(
            start=month_start_date,
            end=max_date.normalize() + pd.DateOffset(months=1),
            freq="MS",
        )

        periods = []
        for i in range(len(date_range) - 1):
            month_start = date_range[i]
            month_end = date_range[i + 1] - pd.Timedelta(microseconds=1)

            year = month_start.year
            month = month_start.month
            periods.append(
                {
                    "period_start": month_start,
                    "period_end": month_end,
                    "year": year,
                    "period_label": (year, month),
                }
            )

        self.df_period = pd.DataFrame(periods)

    def _generate_quarterly_labels(
        self, min_date: pd.Timestamp, max_date: pd.Timestamp
    ) -> None:
        """Generate quarterly period labels."""
        # Start from the first day of the quarter containing min_date
        quarter_month = ((min_date.month - 1) // 3) * 3 + 1
        quarter_start_date = min_date.normalize().replace(day=1, month=quarter_month)

        date_range = pd.date_range(
            start=quarter_start_date,
            end=max_date.normalize() + pd.DateOffset(months=3),
            freq="QS",
        )

        periods = []
        for i in range(len(date_range) - 1):
            quarter_start = date_range[i]
            quarter_end = date_range[i + 1] - pd.Timedelta(microseconds=1)

            year = quarter_start.year
            quarter = (quarter_start.month - 1) // 3 + 1
            periods.append(
                {
                    "period_start": quarter_start,
                    "period_end": quarter_end,
                    "year": year,
                    "period_label": (year, f"Q{quarter}"),
                }
            )

        self.df_period = pd.DataFrame(periods)

    def _get_period_label(self, commit_date: pd.Timestamp) -> Tuple:
        """Get period label for a given commit date."""
        try:
            cond_1 = self.df_period["period_start"] <= commit_date
            cond_2 = self.df_period["period_end"] >= commit_date
            result = self.df_period[cond_1 & cond_2]

            if len(result) > 0:
                return result.reset_index(drop=True)["period_label"][0]
            else:
                # This should be very rare now with improved period generation
                logger.debug(f"No period found for date {commit_date}, using fallback")
                year = commit_date.year
                month = commit_date.month
                return (year, month)
        except Exception as e:
            logger.debug(f"Error getting period label for {commit_date}: {e}")
            return None

    def _map_commits_to_periods(self) -> None:
        """Map each commit to its period label."""
        logger.info("Mapping commits to periods")
        self.df_commit_details["PERIOD_LABEL"] = self.df_commit_details[
            "DATETIME"
        ].apply(self._get_period_label)

        # Check for any commits that couldn't be mapped
        null_labels = self.df_commit_details["PERIOD_LABEL"].isna().sum()
        if null_labels > 0:
            logger.info(
                f"Note: {null_labels} commits could not be mapped to periods (using fallback labels)"
            )
            self.df_commit_details = self.df_commit_details.dropna(
                subset=["PERIOD_LABEL"]
            )

    def _add_period_labels_to_files(self) -> None:
        """Add period labels to file lifecycle data."""
        logger.info("Adding period labels to file lifecycle")

        # Create mapping dictionary
        dict_commit_period_label = dict(
            zip(
                self.df_commit_details["COMMIT_HASH"],
                self.df_commit_details["PERIOD_LABEL"],
            )
        )

        # Extract commit hashes from files
        self.df_file_lifecycle["LIST_COMMIT_HASHES"] = self.df_file_lifecycle[
            "files"
        ].apply(lambda x: [i["commit_hash"] for i in x])

        # Map to period labels
        self.df_file_lifecycle["LIST_PERIOD_LABELS"] = self.df_file_lifecycle[
            "LIST_COMMIT_HASHES"
        ].apply(
            lambda x: [
                dict_commit_period_label.get(commit_hash)
                for commit_hash in x
                if commit_hash in dict_commit_period_label
            ]
        )

        # Sort period labels
        self.df_file_lifecycle["LIST_PERIOD_LABELS"] = self.df_file_lifecycle[
            "LIST_PERIOD_LABELS"
        ].apply(lambda x: sorted([label for label in x if label is not None]))

        # Add change count
        self.df_file_lifecycle["CHANGE_COUNT"] = self.df_file_lifecycle[
            "LIST_PERIOD_LABELS"
        ].apply(len)

        # Sort by change count
        self.df_file_lifecycle = self.df_file_lifecycle.sort_values(
            ["CHANGE_COUNT"], ascending=False
        ).reset_index(drop=True)

    def generate_aggregated_report(self) -> pd.DataFrame:
        """Generate aggregated report by period."""
        try:
            logger.info("Generating aggregated report")

            # Explode by period labels and aggregate
            df_aggregated = (
                self.df_file_lifecycle.explode(["LIST_PERIOD_LABELS"])
                .groupby(["LIST_PERIOD_LABELS"], as_index=False)
                .agg({"FILE_PATH": lambda x: list(x), "PARENT_DIR": lambda x: list(x)})
            )

            df_aggregated = df_aggregated.rename(
                columns={"LIST_PERIOD_LABELS": "PERIOD_LABEL"}
            )

            # Replace '.' with 'root' in parent directories
            df_aggregated["PARENT_DIR"] = df_aggregated["PARENT_DIR"].apply(
                lambda x: ["root" if i == "." else i for i in x]
            )

            # Count occurrences
            df_aggregated["FILE_PATH_COUNT"] = df_aggregated["FILE_PATH"].apply(
                lambda x: dict(Counter(x).most_common())
            )
            df_aggregated["PARENT_DIR_COUNT"] = df_aggregated["PARENT_DIR"].apply(
                lambda x: dict(Counter(x).most_common())
            )

            # Drop raw lists
            df_aggregated = df_aggregated.drop(columns=["FILE_PATH", "PARENT_DIR"])

            # Add period details
            dict_period_label = dict(
                zip(
                    self.df_period["period_label"],
                    self.df_period.apply(
                        lambda x: {
                            "period_start": x["period_start"],
                            "period_end": x["period_end"],
                        },
                        axis=1,
                    ),
                )
            )

            df_aggregated["PERIOD_DETAILS"] = df_aggregated["PERIOD_LABEL"].apply(
                lambda x: dict_period_label.get(x, {})
            )

            logger.info(
                f"Generated aggregated report with {len(df_aggregated)} periods"
            )

            return df_aggregated

        except Exception as e:
            logger.error(f"Error generating aggregated report: {e}")
            raise

    def save_outputs(
        self, df_aggregated: pd.DataFrame, period: str, formats: Set[str]
    ) -> None:
        """
        Save all output dataframes in specified formats.

        Args:
            df_aggregated: Aggregated report dataframe
            period: Period type for filename
            formats: Set of output formats ('pkl', 'json', 'csv', 'parquet')
        """
        try:
            logger.info(f"Saving outputs to {self.output_dir}")
            logger.info(f"Output formats: {', '.join(sorted(formats))}")

            # Create output directory if it doesn't exist
            self.output_dir.mkdir(parents=True, exist_ok=True)

            files_saved = []

            # Define datasets to save
            datasets = {
                "commit_details": self.df_commit_details,
                "file_lifecycle": self.df_file_lifecycle,
                "period_data": self.df_period,
                "aggregated_report": df_aggregated,
            }

            # Save each dataset in requested formats
            for dataset_name, df in datasets.items():
                for fmt in formats:
                    output_file = self.output_dir / f"{period}_{dataset_name}.{fmt}"

                    try:
                        if fmt == "pkl":
                            df.to_pickle(output_file, protocol=4)
                            files_saved.append(output_file)
                            logger.info(f"Saved {dataset_name}: {output_file}")

                        elif fmt == "json":
                            # Convert DataFrame to JSON
                            # Handle special columns that may not be JSON serializable
                            df_json = df.copy()

                            # Convert datetime columns to ISO format strings
                            for col in df_json.columns:
                                if pd.api.types.is_datetime64_any_dtype(df_json[col]):
                                    df_json[col] = df_json[col].dt.strftime(
                                        "%Y-%m-%d %H:%M:%S%z"
                                    )

                            df_json.to_json(output_file, orient="records", indent=2)
                            files_saved.append(output_file)
                            logger.info(f"Saved {dataset_name}: {output_file}")

                        elif fmt == "csv":
                            # For CSV, convert complex types to strings
                            df_csv = df.copy()
                            for col in df_csv.columns:
                                if df_csv[col].dtype == "object":
                                    # Check if column contains dicts or lists
                                    if (
                                        df_csv[col]
                                        .apply(lambda x: isinstance(x, (dict, list)))
                                        .any()
                                    ):
                                        df_csv[col] = df_csv[col].astype(str)

                            df_csv.to_csv(output_file, index=False)
                            files_saved.append(output_file)
                            logger.info(f"Saved {dataset_name}: {output_file}")

                        elif fmt == "parquet":
                            # For Parquet, convert complex types that aren't supported
                            df_parquet = df.copy()
                            for col in df_parquet.columns:
                                if df_parquet[col].dtype == "object":
                                    # Check if column contains dicts or lists
                                    if (
                                        df_parquet[col]
                                        .apply(lambda x: isinstance(x, (dict, list)))
                                        .any()
                                    ):
                                        df_parquet[col] = df_parquet[col].astype(str)

                            df_parquet.to_parquet(output_file, index=False)
                            files_saved.append(output_file)
                            logger.info(f"Saved {dataset_name}: {output_file}")

                    except Exception as e:
                        logger.warning(f"Failed to save {dataset_name} as {fmt}: {e}")

            # Always save summary statistics
            summary_file = self.output_dir / f"{period}_summary.txt"
            with open(summary_file, "w") as f:
                f.write(f"File Lifecycle Analysis Summary\n")
                f.write(f"{'=' * 50}\n\n")
                f.write(f"Analysis Period: {period}\n")
                f.write(f"Generated at: {datetime.now().isoformat()}\n\n")
                f.write(f"Total Files: {len(self.df_file_lifecycle)}\n")
                f.write(f"Total Commits: {len(self.df_commit_details)}\n")
                f.write(f"Total Periods: {len(df_aggregated)}\n\n")
                f.write(f"Date Range:\n")
                f.write(f"  Start: {self.df_commit_details['DATETIME'].min()}\n")
                f.write(f"  End: {self.df_commit_details['DATETIME'].max()}\n\n")
                f.write(f"Top 10 Most Changed Files:\n")
                for idx, row in self.df_file_lifecycle.head(10).iterrows():
                    f.write(f"  {row['FILE_PATH']}: {row['CHANGE_COUNT']} changes\n")
            files_saved.append(summary_file)
            logger.info(f"Saved summary: {summary_file}")

            logger.info(f"Successfully saved {len(files_saved)} output files")

        except Exception as e:
            logger.error(f"Error saving outputs: {e}")
            raise


@click.command()
@click.option(
    "--input-file",
    "-i",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to input JSON file containing file lifecycle data",
)
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(path_type=Path),
    required=True,
    help="Directory to save output files",
)
@click.option(
    "--period",
    "-p",
    type=click.Choice(["weekly", "monthly", "quarterly"], case_sensitive=False),
    default="weekly",
    help="Aggregation period (default: weekly)",
)
@click.option(
    "--format",
    "-f",
    "formats",
    type=click.Choice(["pkl", "json", "csv", "parquet"], case_sensitive=False),
    multiple=True,
    default=["pkl"],
    help="Output format(s) - can be specified multiple times (default: pkl)",
)
@click.option(
    "--log-level",
    "-l",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"], case_sensitive=False),
    default="INFO",
    help="Logging level (default: INFO)",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Enable verbose output (equivalent to --log-level DEBUG)",
)
def main(
    input_file: Path,
    output_dir: Path,
    period: str,
    formats: tuple,
    log_level: str,
    verbose: bool,
):
    """
    Analyze git repository file lifecycle data and generate aggregated reports.

    This tool processes file lifecycle data from git repositories and generates
    reports aggregated by week, month, or quarter. It tracks file changes,
    commit activity, and provides insights into repository evolution over time.

    Example usage:

        # Save as pickle (default)
        python file_lifecycle_analyzer.py -i data/file_lifecycle.json -o output/ -p monthly

        # Save as CSV
        python file_lifecycle_analyzer.py -i data.json -o output/ -f csv

        # Save as multiple formats
        python file_lifecycle_analyzer.py -i data.json -o output/ -f pkl -f csv -f json
    """
    try:
        # Set log level
        if verbose:
            log_level = "DEBUG"
        logger.setLevel(getattr(logging, log_level.upper()))

        # Convert formats tuple to set
        format_set = set(formats)

        # Log start
        logger.info("=" * 60)
        logger.info("File Lifecycle Analyzer")
        logger.info("=" * 60)
        logger.info(f"Input file: {input_file}")
        logger.info(f"Output directory: {output_dir}")
        logger.info(f"Period: {period}")
        logger.info(f"Output formats: {', '.join(sorted(format_set))}")
        logger.info("=" * 60)

        # Initialize analyzer
        analyzer = FileLifecycleAnalyzer(input_file, output_dir)

        # Run analysis pipeline
        analyzer.load_data()
        analyzer.prepare_file_lifecycle()
        analyzer.generate_period_labels(period)
        df_aggregated = analyzer.generate_aggregated_report()
        analyzer.save_outputs(df_aggregated, period, format_set)

        # Log completion
        logger.info("=" * 60)
        logger.info("Analysis completed successfully!")
        logger.info(f"Output files saved to: {output_dir}")
        logger.info("=" * 60)

        return 0

    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
