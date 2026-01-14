#!/usr/bin/env python3
"""
Test Suite for Phase 4 Features - Repository Evolution Analyzer v2.0.0

Tests for:
- Phase 4.1: Performance Optimization (memory monitoring, profiling, caching)
- Phase 4.2: CLI Enhancements (progress bars, colors, dependency checking)
- Phase 4.3: Presets & Configuration (config files, presets, auto-discovery)

Requirements:
    pip install pytest pytest-mock pyyaml
"""

import pytest
import os
import sys
import json
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, call
from io import StringIO
from click.testing import CliRunner

# Add parent directory to path to import the module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from strata import (
    MemoryMonitor,
    ProfilingContext,
    PerformanceMetrics,
    cached_file_hash,
    chunk_iterator,
    ProgressReporter,
    load_config_file,
    find_config_file,
    ConfigResolver,  # New class
    main,  # Click command
    GitFileLifecycle,
    DatasetAggregator,
)

# ============================================================================
# PHASE 4.1: PERFORMANCE OPTIMIZATION TESTS
# ============================================================================


class TestMemoryMonitor:
    """Test memory monitoring and limit enforcement"""

    def test_memory_monitor_creation(self):
        """Test MemoryMonitor initialization"""
        monitor = MemoryMonitor(limit_mb=1024)
        assert monitor.limit_mb == 1024
        assert monitor.peak_mb == 0.0

    def test_memory_monitor_no_limit(self):
        """Test MemoryMonitor without limit"""
        monitor = MemoryMonitor()
        assert monitor.limit_mb is None
        memory = monitor.check_memory()
        assert memory >= 0.0  # Should return non-negative value

    def test_memory_check_with_psutil(self):
        """Test memory checking with psutil available"""
        mock_psutil = MagicMock()
        mock_process = Mock()
        mock_process.memory_info.return_value.rss = 512 * 1024 * 1024  # 512 MB
        mock_psutil.Process.return_value = mock_process

        # Patch sys.modules to intercept the local import inside check_memory
        with patch.dict(sys.modules, {"psutil": mock_psutil}):
            monitor = MemoryMonitor()
            memory = monitor.check_memory()

        assert memory == 512.0
        assert monitor.peak_mb == 512.0

    def test_memory_limit_enforcement(self):
        """Test that memory limit raises MemoryError"""
        mock_psutil = MagicMock()
        mock_process = Mock()
        mock_process.memory_info.return_value.rss = 2048 * 1024 * 1024  # 2048 MB
        mock_psutil.Process.return_value = mock_process

        monitor = MemoryMonitor(limit_mb=1024)

        with patch.dict(sys.modules, {"psutil": mock_psutil}):
            with pytest.raises(MemoryError, match="Memory limit exceeded"):
                monitor.check_memory()

    def test_peak_memory_tracking(self):
        """Test that peak memory is tracked correctly"""
        mock_psutil = MagicMock()
        mock_process = Mock()
        mock_psutil.Process.return_value = mock_process

        monitor = MemoryMonitor()

        with patch.dict(sys.modules, {"psutil": mock_psutil}):
            # Simulate increasing memory usage
            mock_process.memory_info.return_value.rss = 100 * 1024 * 1024
            monitor.check_memory()
            assert monitor.get_peak() == 100.0

            mock_process.memory_info.return_value.rss = 200 * 1024 * 1024
            monitor.check_memory()
            assert monitor.get_peak() == 200.0

            mock_process.memory_info.return_value.rss = 150 * 1024 * 1024
            monitor.check_memory()
            assert monitor.get_peak() == 200.0  # Peak should not decrease


class TestProfilingContext:
    """Test performance profiling context manager"""

    def test_profiling_disabled(self):
        """Test that profiling does nothing when disabled"""
        with ProfilingContext(enabled=False) as ctx:
            result = 2 + 2

        assert result == 4
        assert ctx.profiler is None

    def test_profiling_enabled(self):
        """Test that profiling captures data when enabled"""
        with tempfile.NamedTemporaryFile(suffix=".prof", delete=False) as f:
            profile_path = f.name

        try:
            with ProfilingContext(enabled=True, output_path=profile_path):
                # Do some work
                sum([i**2 for i in range(1000)])

            # Check that profile file was created
            assert os.path.exists(profile_path)
            assert os.path.getsize(profile_path) > 0
        finally:
            if os.path.exists(profile_path):
                os.unlink(profile_path)


class TestPerformanceMetrics:
    """Test performance metrics tracking"""

    def test_metrics_initialization(self):
        """Test PerformanceMetrics default values"""
        metrics = PerformanceMetrics()

        assert metrics.commits_processed == 0
        assert metrics.files_tracked == 0
        assert metrics.memory_peak_mb == 0.0
        assert metrics.total_time == 0.0
        assert metrics.cache_hits == 0
        assert metrics.cache_misses == 0

    def test_metrics_to_dict(self):
        """Test conversion to dictionary"""
        metrics = PerformanceMetrics(
            commits_processed=1000,
            files_tracked=250,
            memory_peak_mb=512.5,
            total_time=45.67,
            cache_hits=800,
            cache_misses=200,
        )

        data = metrics.to_dict()

        assert data["commits_processed"] == 1000
        assert data["files_tracked"] == 250
        assert data["memory_peak_mb"] == 512.5
        assert data["total_time_seconds"] == 45.67
        assert data["cache_statistics"]["hits"] == 800
        assert data["cache_statistics"]["misses"] == 200
        assert data["cache_statistics"]["hit_rate_percent"] == 80.0

    def test_cache_hit_rate_zero_division(self):
        """Test cache hit rate with no cache activity"""
        metrics = PerformanceMetrics(cache_hits=0, cache_misses=0)
        data = metrics.to_dict()

        assert data["cache_statistics"]["hit_rate_percent"] == 0.0


class TestCaching:
    """Test caching functions"""

    def test_cached_file_hash(self):
        """Test that file hash caching works"""
        # Clear cache
        cached_file_hash.cache_clear()

        # First call
        hash1 = cached_file_hash("src/main.py", "abc123")
        assert isinstance(hash1, str)
        assert len(hash1) == 16

        # Second call with same args should return same hash
        hash2 = cached_file_hash("src/main.py", "abc123")
        assert hash1 == hash2

        # Different args should return different hash
        hash3 = cached_file_hash("src/main.py", "def456")
        assert hash1 != hash3

        # Check cache statistics
        cache_info = cached_file_hash.cache_info()
        assert cache_info.hits >= 1
        assert cache_info.misses >= 2

    def test_chunk_iterator(self):
        """Test memory-efficient chunking"""
        items = list(range(100))
        chunks = list(chunk_iterator(items, chunk_size=25))

        assert len(chunks) == 4
        assert chunks[0] == list(range(0, 25))
        assert chunks[1] == list(range(25, 50))
        assert chunks[2] == list(range(50, 75))
        assert chunks[3] == list(range(75, 100))

    def test_chunk_iterator_uneven(self):
        """Test chunking with uneven division"""
        items = list(range(10))
        chunks = list(chunk_iterator(items, chunk_size=3))

        assert len(chunks) == 4
        assert chunks[0] == [0, 1, 2]
        assert chunks[1] == [3, 4, 5]
        assert chunks[2] == [6, 7, 8]
        assert chunks[3] == [9]


# ============================================================================
# PHASE 4.2: CLI ENHANCEMENTS TESTS
# ============================================================================


class TestProgressReporter:
    """Test enhanced progress reporting"""

    def test_reporter_initialization(self):
        """Test ProgressReporter initialization"""
        reporter = ProgressReporter(quiet=False, verbose=True, use_colors=False)

        assert reporter.quiet is False
        assert reporter.verbose is True
        assert reporter.use_colors is False

    def test_quiet_mode_suppresses_output(self, capsys):
        """Test that quiet mode suppresses output"""
        reporter = ProgressReporter(quiet=True)

        reporter.info("This should not appear")
        reporter.warning("Neither should this")
        reporter.success("Or this")

        captured = capsys.readouterr()
        assert captured.out == ""

    def test_error_always_shown(self, capsys):
        """Test that errors are always shown even in quiet mode"""
        reporter = ProgressReporter(quiet=True)

        reporter.error("This error must appear")

        captured = capsys.readouterr()
        assert "ERROR" in captured.err
        assert "This error must appear" in captured.err

    def test_colorize_with_colors_disabled(self):
        """Test colorization when colors are disabled"""
        reporter = ProgressReporter(use_colors=False)

        colored = reporter._colorize("Test", "RED")
        assert colored == "Test"  # No color codes added

    @patch("strata.COLORAMA_AVAILABLE", True)
    def test_colorize_with_colors_enabled(self):
        """Test colorization when colors are enabled"""
        reporter = ProgressReporter(use_colors=True)

        colored = reporter._colorize("Test", "\x1b[31m")  # Red color
        assert "\x1b[31m" in colored or colored == "Test"

    def test_stage_start_and_complete(self, capsys):
        """Test stage timing"""
        reporter = ProgressReporter(quiet=False, use_colors=False)

        reporter.stage_start("Testing", "Running tests...")
        captured = capsys.readouterr()
        assert "Testing" in captured.out
        assert "Running tests..." in captured.out

        reporter.stage_complete("Testing", {"Tests passed": 10})
        captured = capsys.readouterr()
        assert "complete" in captured.out

    @patch("strata.TQDM_AVAILABLE", True)
    @patch("strata.tqdm")
    def test_create_progress_bar_available(self, mock_tqdm):
        """Test progress bar creation when tqdm is available"""
        reporter = ProgressReporter(quiet=False)

        bar = reporter.create_progress_bar(total=100, desc="Processing")

        mock_tqdm.assert_called_once()
        call_kwargs = mock_tqdm.call_args[1]
        assert call_kwargs["total"] == 100

    @patch("strata.TQDM_AVAILABLE", False)
    def test_create_progress_bar_unavailable(self):
        """Test progress bar creation when tqdm is not available"""
        reporter = ProgressReporter(quiet=False)

        bar = reporter.create_progress_bar(total=100, desc="Processing")

        assert bar is None

    def test_summary_display(self, capsys):
        """Test summary statistics display"""
        reporter = ProgressReporter(quiet=False, use_colors=False)

        stats = {"Total commits": "1,234", "Files tracked": "567", "Errors": 0}

        reporter.summary(stats)
        captured = capsys.readouterr()

        assert "ANALYSIS SUMMARY" in captured.out or "SUMMARY" in captured.out
        assert "1,234" in captured.out
        assert "567" in captured.out


# ============================================================================
# PHASE 4.3: PRESETS & CONFIGURATION TESTS
# ============================================================================


class TestConfigurationLoading:
    """Test configuration file loading"""

    def test_load_yaml_config(self):
        """Test loading YAML configuration file"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(
                """
enable-aggregations: true
phase2: true
quick-mode: false
sample-rate: 0.8
memory-limit: 2048
verbose: true
            """
            )
            config_path = f.name

        try:
            config = load_config_file(config_path)

            assert config["enable-aggregations"] is True
            assert config["phase2"] is True
            assert config["quick-mode"] is False
            assert config["sample-rate"] == 0.8
            assert config["memory-limit"] == 2048
            assert config["verbose"] is True
        finally:
            os.unlink(config_path)

    def test_load_json_config(self):
        """Test loading JSON configuration file"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(
                {"enable-aggregations": True, "phase2": True, "memory-limit": 4096}, f
            )
            config_path = f.name

        try:
            config = load_config_file(config_path)

            assert config["enable-aggregations"] is True
            assert config["phase2"] is True
            assert config["memory-limit"] == 4096
        finally:
            os.unlink(config_path)

    def test_load_config_file_not_found(self):
        """Test loading non-existent config file"""
        with pytest.raises(FileNotFoundError):
            load_config_file("/nonexistent/config.yaml")

    def test_load_config_unsupported_format(self):
        """Test loading config with unsupported format"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("not a config file")
            config_path = f.name

        try:
            with pytest.raises(ValueError, match="Unsupported config file format"):
                load_config_file(config_path)
        finally:
            os.unlink(config_path)


class TestConfigFileDiscovery:
    """Test automatic config file discovery"""

    def test_find_config_in_repo(self):
        """Test finding config file in repository directory"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, ".git-lifecycle.yaml")
            with open(config_path, "w") as f:
                f.write("test: true")

            found = find_config_file(tmpdir)
            assert found == config_path

    def test_find_config_priority(self):
        """Test config file priority (.yaml > .yml > .json)"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create all three types
            yaml_path = os.path.join(tmpdir, ".git-lifecycle.yaml")
            yml_path = os.path.join(tmpdir, ".git-lifecycle.yml")
            json_path = os.path.join(tmpdir, ".git-lifecycle.json")

            with open(yml_path, "w") as f:
                f.write("test: yml")
            with open(json_path, "w") as f:
                f.write('{"test": "json"}')

            # .yaml should take priority (even though it doesn't exist yet)
            found = find_config_file(tmpdir)
            assert found == yml_path

            # Now create .yaml
            with open(yaml_path, "w") as f:
                f.write("test: yaml")

            found = find_config_file(tmpdir)
            assert found == yaml_path

    def test_find_config_not_found(self):
        """Test when no config file exists"""
        with tempfile.TemporaryDirectory() as tmpdir:
            found = find_config_file(tmpdir)
            assert found is None


class TestConfigResolver:
    """Test configuration precedence logic (CLI > Config > Preset > Default)"""

    def test_precedence_cli_over_config(self):
        """Ensure CLI args override config file"""
        cli_params = {"memory_limit": 4096}

        # Mock load_config_file to return a different value
        with patch(
            "strata.load_config_file", return_value={"memory_limit": 2048}
        ):
            resolver = ConfigResolver(cli_params, config_path="dummy.yaml")
            assert resolver.get("memory_limit") == 4096

    def test_precedence_config_over_preset(self):
        """Ensure Config file overrides Preset"""
        cli_params = {}
        config_mock = {"sample_rate": 0.5}

        # Preset 'quick' usually has sample_rate 0.3
        with patch("strata.load_config_file", return_value=config_mock):
            resolver = ConfigResolver(
                cli_params, config_path="dummy.yaml", preset_name="quick"
            )
            assert resolver.get("sample_rate") == 0.5

    def test_preset_application(self):
        """Ensure Preset values are applied when nothing else is set"""
        resolver = ConfigResolver({}, preset_name="quick")
        assert resolver.get("sample_rate") == 0.3
        assert resolver.get("quick_mode") is True

    def test_config_key_normalization(self):
        """Test that kebab-case keys in config are converted to snake_case"""
        config_mock = {"enable-aggregations": True, "memory-limit": 1024}

        with patch("strata.load_config_file", return_value=config_mock):
            resolver = ConfigResolver({}, config_path="dummy.yaml")
            assert resolver.get("enable_aggregations") is True
            assert resolver.get("memory_limit") == 1024


# ============================================================================
# INTEGRATION TESTS
# ============================================================================


class TestClickIntegration:
    """Integration tests for Click CLI"""

    def test_basic_execution(self, temp_git_repo):
        """Test basic dry-run execution"""
        runner = CliRunner()
        result = runner.invoke(main, [temp_git_repo, "--preset=standard", "--dry-run"])

        assert result.exit_code == 0
        assert "DRY RUN MODE" in result.output
        assert "file_lifecycle.json" in result.output

    def test_invalid_path_validation(self):
        """Test Click's built-in path validation"""
        runner = CliRunner()
        result = runner.invoke(main, ["/non/existent/path"])

        assert result.exit_code != 0
        # Updated to match Click's output for optional arguments (brackets added)
        assert "Invalid value for '[REPO_PATH]'" in result.output

    def test_check_dependencies(self):
        """Test dependency check flag"""
        runner = CliRunner()
        result = runner.invoke(main, ["--check-dependencies"])

        assert result.exit_code == 0
        assert "Checking optional dependencies" in result.output


class TestGitFileLifecycleIntegration:
    """Integration tests for GitFileLifecycle with Phase 4 features"""

    def test_analyzer_with_memory_monitor(self):
        """Test that analyzer properly initializes with memory monitor"""
        import subprocess

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a minimal git repo
            subprocess.run(
                ["git", "init", "--initial-branch=main"],
                cwd=tmpdir,
                check=True,
                capture_output=True,
            )
            subprocess.run(
                ["git", "config", "user.name", "Test"],
                cwd=tmpdir,
                check=True,
                capture_output=True,
            )
            subprocess.run(
                ["git", "config", "user.email", "test@test.com"],
                cwd=tmpdir,
                check=True,
                capture_output=True,
            )

            with open(os.path.join(tmpdir, "test.txt"), "w") as f:
                f.write("test")

            subprocess.run(
                ["git", "add", "test.txt"], cwd=tmpdir, check=True, capture_output=True
            )
            subprocess.run(
                ["git", "commit", "-m", "Initial commit"],
                cwd=tmpdir,
                check=True,
                capture_output=True,
            )

            reporter = ProgressReporter(quiet=True)
            analyzer = GitFileLifecycle(
                tmpdir, reporter=reporter, memory_limit_mb=1024, enable_profiling=False
            )

            assert analyzer.memory_monitor is not None
            assert analyzer.memory_monitor.limit_mb == 1024
            assert analyzer.enable_profiling is False

    def test_analyzer_performance_metrics(self):
        """Test that performance metrics are tracked"""
        import subprocess

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a minimal git repo using subprocess for safety
            subprocess.run(
                ["git", "init", "--initial-branch=main"],
                cwd=tmpdir,
                check=True,
                capture_output=True,
            )
            subprocess.run(
                ["git", "config", "user.name", "Test"],
                cwd=tmpdir,
                check=True,
                capture_output=True,
            )
            subprocess.run(
                ["git", "config", "user.email", "test@test.com"],
                cwd=tmpdir,
                check=True,
                capture_output=True,
            )

            with open(os.path.join(tmpdir, "test.txt"), "w") as f:
                f.write("test")

            subprocess.run(
                ["git", "add", "test.txt"], cwd=tmpdir, check=True, capture_output=True
            )
            subprocess.run(
                ["git", "commit", "-m", "Initial commit"],
                cwd=tmpdir,
                check=True,
                capture_output=True,
            )

            reporter = ProgressReporter(quiet=True)
            analyzer = GitFileLifecycle(tmpdir, reporter=reporter)

            analyzer.analyze_repository()

            assert analyzer.metrics.commits_processed > 0
            assert analyzer.metrics.total_time > 0
            assert analyzer.metrics.files_tracked > 0


# ============================================================================
# PYTEST FIXTURES
# ============================================================================


@pytest.fixture
def temp_git_repo():
    """Create a temporary git repository for testing"""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Initialize repo
        os.system(f"cd {tmpdir} && git init --initial-branch=main > /dev/null 2>&1")
        os.system(f'cd {tmpdir} && git config user.name "Test User"')
        os.system(f'cd {tmpdir} && git config user.email "test@example.com"')

        # Create some files and commits
        for i in range(3):
            file_path = os.path.join(tmpdir, f"file{i}.txt")
            with open(file_path, "w") as f:
                f.write(f"Content {i}")
            os.system(f"cd {tmpdir} && git add file{i}.txt")
            os.system(f'cd {tmpdir} && git commit -m "Add file{i}" > /dev/null 2>&1')

        yield tmpdir


# ============================================================================
# TEST RUNNER
# ============================================================================

if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"])
