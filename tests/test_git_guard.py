from pathlib import Path

from corr2surrogate.security.git_guard import scan_files_for_leaks


def test_git_guard_detects_token_pattern(tmp_path: Path) -> None:
    sample = tmp_path / "sample.txt"
    sample.write_text("token = sk-abcdefghijklmnopqrstuvwxyz123456", encoding="utf-8")
    findings = scan_files_for_leaks([sample])
    assert findings


def test_git_guard_skips_safe_text(tmp_path: Path) -> None:
    sample = tmp_path / "safe.txt"
    sample.write_text("hello world", encoding="utf-8")
    findings = scan_files_for_leaks([sample])
    assert not findings
