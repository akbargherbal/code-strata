import pytest
import json
import os
import subprocess
import tempfile
from unittest.mock import MagicMock
from strata import (
    ProgressReporter, TemporalLabeler, AuthorNormalizer,
    GitFileLifecycle, PerformanceMetrics,
)

@pytest.fixture
def quiet_reporter():
    return ProgressReporter(quiet=True)

@pytest.fixture
def mock_analyzer():
    a = MagicMock()
    a.repo_path = "/fake/repo"
    return a

@pytest.fixture
def sample_commits():
    """Three commits touching overlapping files, two authors, spanning two days."""
    return [
        {
            "commit_hash": "aaa111",
            "timestamp": 1_700_000_000,   # 2023-11-14 22:13:20 UTC
            "author_name": "Alice Smith",
            "author_email": "alice@example.com",
            "author_id": "alice_smith_alice",
            "commit_subject": "initial commit",
        },
        {
            "commit_hash": "bbb222",
            "timestamp": 1_700_100_000,   # 2023-11-16 03:46:40 UTC
            "author_name": "Bob Jones",
            "author_email": "bob@example.com",
            "author_id": "bob_jones_bob",
            "commit_subject": "add feature",
        },
        {
            "commit_hash": "ccc333",
            "timestamp": 1_700_200_000,   # 2023-11-17 09:06:40 UTC
            "author_name": "Alice Smith",
            "author_email": "alice@example.com",
            "author_id": "alice_smith_alice",
            "commit_subject": "fix bug",
        },
    ]

@pytest.fixture
def sample_changes():
    """Parallel change-sets for the three commits above."""
    return [
        # Commit 0: two files created
        [
            {"file_path": "src/main.py",   "operation": "A", "lines_added": 50,  "lines_deleted": 0},
            {"file_path": "src/utils.py",  "operation": "A", "lines_added": 30,  "lines_deleted": 0},
        ],
        # Commit 1: one file modified, one new file
        [
            {"file_path": "src/main.py",   "operation": "M", "lines_added": 10,  "lines_deleted": 3},
            {"file_path": "tests/test_main.py", "operation": "A", "lines_added": 40, "lines_deleted": 0},
        ],
        # Commit 2: two files modified (triggers co-change)
        [
            {"file_path": "src/main.py",   "operation": "M", "lines_added": 5,   "lines_deleted": 8},
            {"file_path": "src/utils.py",  "operation": "M", "lines_added": 2,   "lines_deleted": 1},
        ],
    ]

@pytest.fixture
def git_repo(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()

    def run(*args):
        subprocess.run(["git", "-C", str(repo)] + list(args),
                       check=True, capture_output=True)

    run("init")
    run("config", "user.email", "tester@test.com")
    run("config", "user.name",  "Tester")

    # Commit 1 — add two files
    (repo / "app.py").write_text("print('hello')\n", encoding='utf-8')
    (repo / "lib.py").write_text("def helper(): pass\n", encoding='utf-8')
    run("add", ".")
    run("commit", "-m", "initial")

    # Commit 2 — modify both (co-change)
    (repo / "app.py").write_text("print('hello')\nprint('world')\n", encoding='utf-8')
    (repo / "lib.py").write_text("def helper(): return 1\n", encoding='utf-8')
    run("add", ".")
    run("commit", "-m", "update both")

    # Commit 3 — add new file, tag this commit
    (repo / "readme.md").write_text("# App\n", encoding='utf-8')
    run("add", ".")
    run("commit", "-m", "add readme")
    run("tag", "v1.0")

    # Commit 4 — modify app only
    (repo / "app.py").write_text("print('hello')\nprint('world')\nprint('!')\n", encoding='utf-8')
    run("add", ".")
    run("commit", "-m", "tweak app")

    return str(repo)