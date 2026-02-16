import pytest
import json
import sys
import os
import io
from unittest.mock import MagicMock, patch, mock_open
from click.testing import CliRunner

# Import from local strata.py
from strata import (
    TemporalLabeler, AuthorNormalizer, cached_file_hash, chunk_iterator,
    MemoryMonitor, ProfilingContext, load_config_file, find_config_file,
    ConfigResolver, ProgressReporter, PerformanceMetrics, HealthScorer,
    FileMetadataAggregator, TemporalAggregator, AuthorNetworkAggregator,
    CochangeNetworkAggregator, FileHierarchyAggregator, MilestoneAggregator,
    ProjectHierarchyAggregator, FileMetricsIndexAggregator, TemporalActivityMapAggregator,
    GitFileLifecycle, generate_manifest, MetadataReportGenerator, main,
    DatasetAggregator, YAML_AVAILABLE
)

# ============================================================================
# UNIT TESTS: UTILITIES & HELPERS
# ============================================================================

def test_temporal_labeler_standard():
    tl = TemporalLabeler()
    res = tl.get_temporal_labels("2024-04-12T13:14:18+00:00")
    assert res["year"] == 2024
    assert res["quarter"] == 2
    assert res["month"] == 4
    assert res["day_of_week"] == 4
    assert res["day_of_year"] == 103

def test_temporal_labeler_timezone_variant():
    tl = TemporalLabeler()
    res = tl.get_temporal_labels("2024-04-12T13:14:18+0000")
    assert res["year"] == 2024

def test_author_normalizer():
    an = AuthorNormalizer()
    res = an.normalize("Alice Smith", "alice+work@example.com")
    # Implementation replaces spaces with underscores
    assert res["author_id"] == "alice_smith_alice"
    assert res["author_domain"] == "example.com"
    
    res_empty = an.normalize("", "")
    assert res_empty["author_id"] == "unknown"
    
    # Test truncation
    long_name = "a" * 100
    res_long = an.normalize(long_name, "test@test.com")
    assert len(res_long["author_id"]) <= 64

def test_cached_file_hash():
    h1 = cached_file_hash("file1.py", "abc")
    h2 = cached_file_hash("file2.py", "abc")
    assert h1 != h2
    assert len(h1) == 16

def test_chunk_iterator():
    items = list(range(10))
    chunks = list(chunk_iterator(items, 3))
    assert len(chunks) == 4
    assert len(chunks[0]) == 3
    assert len(chunks[3]) == 1

def test_memory_monitor(monkeypatch):
    # Test with psutil
    mock_psutil = MagicMock()
    mock_psutil.Process.return_value.memory_info.return_value.rss = 100 * 1024 * 1024
    with patch.dict(sys.modules, {'psutil': mock_psutil}):
        mm = MemoryMonitor()
        assert mm.check_memory() == 100.0
        assert mm.get_peak() == 100.0
        
        mm_limit = MemoryMonitor(limit_mb=50)
        with pytest.raises(MemoryError):
            mm_limit.check_memory()

def test_memory_monitor_no_psutil():
    # Test fallback when psutil is missing
    with patch.dict(sys.modules, {'psutil': None}):
        mm = MemoryMonitor()
        assert mm.check_memory() == 0.0

def test_memory_monitor_import_error(monkeypatch):
    # Test fallback when psutil import fails (simulated by sys.modules)
    with patch.dict(sys.modules, {'psutil': None}):
        mm = MemoryMonitor()
        assert mm.check_memory() == 0.0

def test_config_loading(tmp_path):
    # JSON
    f = tmp_path / "config.json"
    f.write_text('{"key": "value"}', encoding='utf-8')
    data = load_config_file(str(f))
    assert data["key"] == "value"
    
    # YAML (if available)
    if YAML_AVAILABLE:
        y = tmp_path / "config.yaml"
        y.write_text('key: value', encoding='utf-8')
        data_y = load_config_file(str(y))
        assert data_y["key"] == "value"
    
    # Missing file
    with pytest.raises(FileNotFoundError):
        load_config_file("nonexistent.json")
        
    # Unsupported extension
    bad = tmp_path / "config.txt"
    bad.touch()
    with pytest.raises(ValueError):
        load_config_file(str(bad))

def test_config_loading_no_yaml(monkeypatch, tmp_path):
    monkeypatch.setattr("strata.YAML_AVAILABLE", False)
    f = tmp_path / "config.yaml"
    f.write_text("key: value", encoding="utf-8")
    with pytest.raises(ImportError):
        load_config_file(str(f))

def test_config_resolver(tmp_path):
    # Test auto-discovery
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".git-lifecycle.json").write_text('{"preset": "quick"}', encoding='utf-8')
    
    cr = ConfigResolver({}, None, None, str(repo))
    assert cr.get("preset") == "quick"
    
    # Test precedence: CLI > Config > Preset
    cr = ConfigResolver({"quick_mode": True}, None, "standard", str(repo))
    assert cr.get("quick_mode") is True

def test_health_scorer():
    # Perfect health
    s = HealthScorer.calculate_health_score(10, 3, 0.0, 180, "f.py")
    assert s["health_score"] == 100
    assert s["health_category"] == "healthy"
    
    # High churn penalty
    s = HealthScorer.calculate_health_score(10, 3, 1.0, 180, "f.py")
    assert s["health_score"] <= 60
    
    # Bus factor risk (1 author)
    s = HealthScorer.calculate_health_score(10, 1, 0.0, 180, "f.py")
    assert s["bus_factor_status"] == "high-risk"

def test_progress_reporter(capsys):
    # Test verbose output
    pr = ProgressReporter(quiet=False, verbose=True, use_colors=False)
    pr.stage_start("Stage 1", "Details")
    pr.progress("Working", 10)
    pr.info("Info")
    pr.warning("Warn")
    pr.success("Done")
    pr.stage_complete("Stage 1", {"Stat": 1})
    pr.summary({"Total": 10})
    
    captured = capsys.readouterr()
    assert "Stage 1" in captured.out
    assert "Warn" in captured.out
    assert "Done" in captured.out

def test_progress_reporter_dependencies(monkeypatch):
    # Test missing tqdm
    monkeypatch.setattr("strata.TQDM_AVAILABLE", False)
    pr = ProgressReporter(quiet=False)
    assert pr.create_progress_bar(100) is None

def test_progress_reporter_no_tqdm(monkeypatch):
    monkeypatch.setattr("strata.TQDM_AVAILABLE", False)
    pr = ProgressReporter()
    assert pr.create_progress_bar(10) is None

def test_profiling_context(tmp_path):
    # Enabled
    prof_out = tmp_path / "test.prof"
    with ProfilingContext(enabled=True, output_path=str(prof_out)):
        _ = [x*x for x in range(1000)]
    assert prof_out.exists()
    
    # Disabled
    with ProfilingContext(enabled=False):
        pass

# ============================================================================
# UNIT TESTS: AGGREGATORS
# ============================================================================

def test_dataset_aggregator_base(mock_analyzer, tmp_path):
    # Test base class methods
    agg = DatasetAggregator(mock_analyzer)
    agg.process_commit({}, [])
    assert agg.finalize() == {}
    
    # Test export
    out_file = tmp_path / "dummy.json"
    agg.export(str(out_file))
    assert out_file.exists()

def test_file_metadata_aggregator(mock_analyzer, quiet_reporter, sample_commits, sample_changes):
    agg = FileMetadataAggregator(mock_analyzer, reporter=quiet_reporter)
    for i, commit in enumerate(sample_commits):
        agg.process_commit(commit, sample_changes[i])
    
    res = agg.finalize()
    assert res["total_files"] == 3
    assert res["files"]["src/main.py"]["total_commits"] == 3
    assert res["files"]["src/main.py"]["unique_authors"] == 2

def test_temporal_aggregator(mock_analyzer, quiet_reporter, sample_commits, sample_changes, tmp_path):
    agg = TemporalAggregator(mock_analyzer, reporter=quiet_reporter)
    for i, commit in enumerate(sample_commits):
        agg.process_commit(commit, sample_changes[i])
    
    # Check internal state before export
    assert len(agg.daily_stats) == 3
    assert len(agg.monthly_stats) == 1
    
    # Test exports
    agg.export_daily(str(tmp_path / "daily.json"))
    agg.export_monthly(str(tmp_path / "monthly.json"))
    assert (tmp_path / "daily.json").exists()

def test_author_network_aggregator(mock_analyzer, quiet_reporter, sample_commits, sample_changes):
    agg = AuthorNetworkAggregator(mock_analyzer, reporter=quiet_reporter)
    for i, commit in enumerate(sample_commits):
        agg.process_commit(commit, sample_changes[i])
    
    res = agg.finalize()
    assert len(res["nodes"]) == 2
    # Alice and Bob both touched src/main.py
    assert len(res["edges"]) >= 1
    
    # Test quick mode limit with max_edges=1 (truthy)
    # Since we only have 1 edge in sample data, it should be preserved
    agg_limit = AuthorNetworkAggregator(mock_analyzer, max_edges=1, reporter=quiet_reporter)
    for i, commit in enumerate(sample_commits):
        agg_limit.process_commit(commit, sample_changes[i])
    res_limit = agg_limit.finalize()
    assert len(res_limit["edges"]) == 1

def test_cochange_network_aggregator(mock_analyzer, quiet_reporter, sample_commits, sample_changes):
    # min_cochange_count=1 to capture the co-changes
    agg = CochangeNetworkAggregator(mock_analyzer, min_cochange_count=1, reporter=quiet_reporter)
    for i, commit in enumerate(sample_commits):
        agg.process_commit(commit, sample_changes[i])
    
    res = agg.finalize()
    assert len(res["edges"]) >= 1
    # src/main.py and src/utils.py change together in Commit 0 and Commit 2
    assert res["edges"][0]["cochange_count"] == 2
    
    # Test quick mode limit with max_pairs=1 (truthy)
    # Set min_cochange_count=1 because max_pairs=1 will stop processing after 1 occurrence
    agg_limit = CochangeNetworkAggregator(mock_analyzer, max_pairs=1, min_cochange_count=1, reporter=quiet_reporter)
    for i, commit in enumerate(sample_commits):
        agg_limit.process_commit(commit, sample_changes[i])
    res_limit = agg_limit.finalize()
    assert len(res_limit["edges"]) == 1

def test_project_hierarchy_aggregator(mock_analyzer, quiet_reporter, sample_commits, sample_changes):
    agg = ProjectHierarchyAggregator(mock_analyzer, reporter=quiet_reporter)
    for i, commit in enumerate(sample_commits):
        agg.process_commit(commit, sample_changes[i])
    
    res = agg.finalize()
    root = res["tree"]
    # Sum of commits per file: main(3) + utils(2) + test_main(1) = 6
    assert root["stats"]["total_commits"] == 6
    assert len(root["children"]) > 0

def test_file_metrics_index_aggregator(mock_analyzer, quiet_reporter, sample_commits, sample_changes):
    agg = FileMetricsIndexAggregator(mock_analyzer, reporter=quiet_reporter)
    for i, commit in enumerate(sample_commits):
        agg.process_commit(commit, sample_changes[i])
    
    res = agg.finalize()
    assert "src/main.py" in res
    metrics = res["src/main.py"]
    assert metrics["volume"]["total_commits"] == 3
    assert metrics["volume"]["net_change"] == (50+10+5) - (0+3+8)

def test_temporal_activity_map_aggregator(mock_analyzer, quiet_reporter, sample_commits, sample_changes):
    agg = TemporalActivityMapAggregator(mock_analyzer, reporter=quiet_reporter)
    for i, commit in enumerate(sample_commits):
        agg.process_commit(commit, sample_changes[i])
    
    res = agg.finalize()
    # Sample files are in 'src' and 'tests', so 'src' should be a key
    assert "src" in res["data"]
    assert len(res["data"]["src"]) > 0

def test_milestone_aggregator(mock_analyzer, quiet_reporter, sample_commits, sample_changes):
    # Mock git tags
    with patch("subprocess.run") as mock_run:
        mock_run.return_value.stdout = "v1.0 aaa111\n"
        mock_run.return_value.returncode = 0
        
        agg = MilestoneAggregator(mock_analyzer, repo_path="/fake", reporter=quiet_reporter)
        assert agg.tags["aaa111"] == "v1.0"
        
        for i, commit in enumerate(sample_commits):
            agg.process_commit(commit, sample_changes[i])
            
        res = agg.finalize()
        assert len(res["snapshots"]) > 0
        assert res["snapshots"][0]["tag"] == "v1.0"

def test_file_hierarchy_aggregator(mock_analyzer, quiet_reporter, sample_commits, sample_changes):
    agg = FileHierarchyAggregator(mock_analyzer, reporter=quiet_reporter)
    for i, commit in enumerate(sample_commits):
        agg.process_commit(commit, sample_changes[i])
    res = agg.finalize()
    assert "src" in res["directories"]

# ============================================================================
# UNIT TESTS: CORE LIFECYCLE
# ============================================================================

def test_git_file_lifecycle_parsing(mock_analyzer):
    gfl = GitFileLifecycle("/fake")
    
    # Commit parsing
    line = "aaa\x00123\x00Alice\x00a@b.com\x00msg"
    res = gfl._parse_commit_line(line)
    assert res["commit_hash"] == "aaa"
    assert gfl._parse_commit_line("broken") is None
    
    # Change parsing (numstat)
    res = gfl._parse_change_line_with_numstat("10\t5\tsrc/f.py")
    assert res["lines_added"] == 10
    assert res["lines_deleted"] == 5
    
    # Change parsing (binary)
    res = gfl._parse_change_line_with_numstat("-\t-\timg.png")
    assert res["lines_added"] == 0
    
    # Change parsing (rename legacy)
    # This triggers the ValueError in int() conversion and falls back to legacy
    res = gfl._parse_change_line_with_numstat("R100\told.py\tnew.py")
    assert res["operation"] == "R"
    assert res["file_path"] == "new.py"
    
    # Change parsing (copy legacy)
    res = gfl._parse_change_line_with_numstat("C100\tsrc.py\tdst.py")
    assert res["operation"] == "C"
    
    # Change parsing (simple status)
    res = gfl._parse_change_line_with_numstat("M\tfile.py")
    assert res["operation"] == "M"
    
    # Malformed
    assert gfl._parse_change_line_with_numstat("garbage") is None

def test_git_file_lifecycle_processing(mock_analyzer, sample_commits, sample_changes):
    gfl = GitFileLifecycle("/fake")
    gfl._process_commit(sample_commits[0], sample_changes[0])
    
    assert "src/main.py" in gfl.files
    event = gfl.files["src/main.py"][0]
    assert event["lines_added"] == 50
    assert "temporal" in event
    assert "author_id" in event
    
    # Test aggregator failure handling
    bad_agg = MagicMock()
    bad_agg.process_commit.side_effect = Exception("Boom")
    gfl.add_aggregator(bad_agg)
    gfl._process_commit(sample_commits[1], sample_changes[1])
    assert len(gfl.errors) > 0

def test_manifest_generation(tmp_path):
    gfl = GitFileLifecycle("/fake")
    (tmp_path / "data.json").write_text("content", encoding='utf-8')
    datasets = {"core": "data.json"}
    m = generate_manifest(str(tmp_path), gfl, datasets)
    assert "core" in m["datasets"]

def test_metadata_report_generator(tmp_path, mock_analyzer):
    mock_analyzer.total_commits = 100
    mock_analyzer.files = {"a": 1}
    mock_analyzer.total_changes = 500
    mock_analyzer.metrics = PerformanceMetrics()
    
    datasets = {"core": "core.json"}
    (tmp_path / "core.json").touch()
    
    gen = MetadataReportGenerator(mock_analyzer, str(tmp_path), datasets)
    report_path = gen.generate()
    
    assert os.path.exists(report_path)
    content = open(report_path).read()
    assert "Git File Lifecycle Analysis" in content

def test_metadata_report_no_pandas(monkeypatch, mock_analyzer, tmp_path):
    # Setup mock attributes required for formatting
    mock_analyzer.total_commits = 100
    mock_analyzer.total_changes = 500
    mock_analyzer.files = {"a": 1}
    mock_analyzer.metrics = PerformanceMetrics()

    gen = MetadataReportGenerator(mock_analyzer, str(tmp_path), {})
    gen.pandas_available = False
    gen.generate()

# ============================================================================
# INTEGRATION & CLI TESTS
# ============================================================================

def test_full_integration(git_repo, tmp_path):
    gfl = GitFileLifecycle(git_repo)
    
    # Add aggregators
    gfl.add_aggregator(FileMetadataAggregator(gfl))
    gfl.add_aggregator(MilestoneAggregator(gfl, repo_path=git_repo))
    
    # Run analysis
    gfl.analyze_repository()
    
    assert gfl.total_commits >= 4
    assert len(gfl.files) >= 3
    
    # Export
    out = tmp_path / "lifecycle.json"
    gfl.export_lifecycle(str(out))
    assert out.exists()

def test_analyze_repository_mocked(mock_analyzer, quiet_reporter):
    # Test the analyze_repository loop with mocked git log output
    gfl = GitFileLifecycle("/fake", reporter=quiet_reporter)
    
    git_log_output = (
        "aaa111\x001700000000\x00Alice\x00a@b.com\x00msg\n"
        "10\t5\tsrc/main.py\n"
        "bbb222\x001700100000\x00Bob\x00b@b.com\x00msg2\n"
        "-\t-\timg.png\n"
    )
    
    with patch("subprocess.Popen") as mock_popen:
        process_mock = MagicMock()
        process_mock.stdout = io.StringIO(git_log_output)
        process_mock.stderr = io.StringIO("")
        process_mock.returncode = 0
        process_mock.wait.return_value = None
        mock_popen.return_value = process_mock
        
        with patch("subprocess.run") as mock_run:
            # Mock rev-list count
            mock_run.return_value.stdout = "2"
            mock_run.return_value.returncode = 0
            
            gfl.analyze_repository()
            
            assert gfl.total_commits == 2
            assert "src/main.py" in gfl.files

def test_analyze_repository_failure(mock_analyzer, quiet_reporter):
    gfl = GitFileLifecycle("/fake", reporter=quiet_reporter)
    
    with patch("subprocess.Popen") as mock_popen:
        process_mock = MagicMock()
        process_mock.stdout = io.StringIO("")
        process_mock.stderr = io.StringIO("Git error")
        process_mock.returncode = 1
        mock_popen.return_value = process_mock
        
        with patch("subprocess.run") as mock_run:
            mock_run.return_value.stdout = "0"
            
            with pytest.raises(RuntimeError, match="Git command failed"):
                gfl.analyze_repository()

def test_cli_dry_run(git_repo):
    runner = CliRunner()
    result = runner.invoke(main, [git_repo, "--dry-run", "--phase3", "--quick-mode", "--profile"])
    assert result.exit_code == 0
    assert "DRY RUN MODE" in result.output
    assert "project_hierarchy.json" in result.output

def test_cli_check_dependencies():
    runner = CliRunner()
    result = runner.invoke(main, ["--check-dependencies"])
    assert result.exit_code == 0
    assert "Checking optional dependencies" in result.output

def test_check_dependencies_missing(monkeypatch):
    monkeypatch.setattr("strata.TQDM_AVAILABLE", False)
    runner = CliRunner()
    result = runner.invoke(main, ["--check-dependencies"])
    assert "Not installed" in result.output

def test_cli_invalid_repo(tmp_path):
    runner = CliRunner()
    result = runner.invoke(main, [str(tmp_path)])
    assert result.exit_code == 1
    assert "Not a git repository" in result.output

def test_cli_presets_and_execution(tmp_path):
    runner = CliRunner()
    
    # Create a fake repo structure
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".git").mkdir()
    
    # Mock the actual analysis to speed up tests
    with patch("strata.GitFileLifecycle") as MockLifecycle:
        instance = MockLifecycle.return_value
        instance.files = {}
        instance.total_commits = 0
        instance.total_changes = 0
        instance.aggregators = []
        instance.metrics = PerformanceMetrics()
        instance.errors = []
        instance.repo_path = str(repo)  # Fix: Required for manifest generation
        
        # Test Quick Preset
        result = runner.invoke(main, [str(repo), "--preset", "quick", "-o", str(tmp_path)])
        assert result.exit_code == 0
        
        # Test Network Preset
        result = runner.invoke(main, [str(repo), "--preset", "network", "-o", str(tmp_path)])
        assert result.exit_code == 0
        
        # Test Full Preset with Config
        config_file = tmp_path / "config.json"
        config_file.write_text('{"preset": "full"}', encoding='utf-8')
        result = runner.invoke(main, [str(repo), "--config", str(config_file), "-o", str(tmp_path)])
        assert result.exit_code == 0
        
        # Test with errors
        instance.errors = ["Some error"]
        result = runner.invoke(main, [str(repo), "-o", str(tmp_path)])
        assert result.exit_code == 0
        assert "Errors logged" in result.output

def test_main_analysis_error(tmp_path):
    runner = CliRunner()
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".git").mkdir()
    
    with patch("strata.GitFileLifecycle.analyze_repository", side_effect=RuntimeError("Fail")):
        result = runner.invoke(main, [str(repo), "-o", str(tmp_path)])
        assert result.exit_code == 1
        assert "Analysis failed" in result.output