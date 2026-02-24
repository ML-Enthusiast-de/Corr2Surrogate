"""Lightweight secret/system-info leak scanner for git safety checks."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class LeakFinding:
    """One suspicious finding in a file."""

    path: str
    line_number: int
    pattern_name: str
    excerpt: str


LEAK_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("openai_key", re.compile(r"\bsk-[A-Za-z0-9]{20,}\b")),
    ("github_token", re.compile(r"\bghp_[A-Za-z0-9]{20,}\b")),
    ("aws_access_key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("slack_token", re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b")),
    (
        "generic_secret_assignment",
        re.compile(
            r"(?i)\b(api[_-]?key|secret|token|password)\b\s*[:=]\s*['\"]?[A-Za-z0-9_\-]{10,}"
        ),
    ),
    ("windows_user_path", re.compile(r"C:\\Users\\[^\\\s]+\\", re.IGNORECASE)),
    ("mac_user_path", re.compile(r"/Users/[^/\s]+/")),
    ("linux_user_path", re.compile(r"/home/[^/\s]+/")),
]


IGNORE_SUFFIXES = {
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".pdf",
    ".zip",
    ".gz",
    ".pkl",
    ".pt",
    ".pth",
}

DEFAULT_EXCLUDED_PATHS = {
    "src/corr2surrogate/security/git_guard.py",
    "tests/test_git_guard.py",
}


def scan_files_for_leaks(paths: Iterable[str | Path]) -> list[LeakFinding]:
    """Scan files for probable secrets and system-specific path leaks."""
    findings: list[LeakFinding] = []
    for raw_path in paths:
        path = Path(raw_path)
        normalized = _normalize_path(path)
        if normalized in DEFAULT_EXCLUDED_PATHS:
            continue
        if not path.is_file():
            continue
        if path.suffix.lower() in IGNORE_SUFFIXES:
            continue
        text = _safe_read_text(path)
        if text is None:
            continue
        findings.extend(_scan_text(path, text))
    return findings


def scan_tracked_files_for_leaks() -> list[LeakFinding]:
    """Scan all git-tracked files in the current repository."""
    files = _git_tracked_files()
    return scan_files_for_leaks(files)


def _scan_text(path: Path, text: str) -> list[LeakFinding]:
    findings: list[LeakFinding] = []
    lines = text.splitlines()
    for idx, line in enumerate(lines, start=1):
        for name, pattern in LEAK_PATTERNS:
            if pattern.search(line):
                findings.append(
                    LeakFinding(
                        path=str(path),
                        line_number=idx,
                        pattern_name=name,
                        excerpt=line[:220],
                    )
                )
    return findings


def _safe_read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        try:
            return path.read_text(encoding="latin-1")
        except UnicodeDecodeError:
            return None


def _git_tracked_files() -> list[str]:
    completed = subprocess.run(
        ["git", "ls-files"],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"git ls-files failed with code {completed.returncode}: {completed.stderr.strip()}"
        )
    return [line.strip() for line in completed.stdout.splitlines() if line.strip()]


def _normalize_path(path: Path) -> str:
    return str(path).replace("\\", "/").lstrip("./")


def main(argv: list[str] | None = None) -> int:
    """CLI entry for local leak checks before commit/push."""
    parser = argparse.ArgumentParser(description="Scan repository files for leak patterns.")
    parser.add_argument(
        "--tracked",
        action="store_true",
        help="Scan git-tracked files (default behavior when no paths are provided).",
    )
    parser.add_argument(
        "paths",
        nargs="*",
        help="Optional explicit files/directories to scan.",
    )
    args = parser.parse_args(argv)

    if args.paths:
        paths: list[str] = []
        for raw in args.paths:
            p = Path(raw)
            if p.is_dir():
                paths.extend(str(x) for x in p.rglob("*") if x.is_file())
            else:
                paths.append(str(p))
        findings = scan_files_for_leaks(paths)
    else:
        findings = scan_tracked_files_for_leaks()

    if not findings:
        print("No leak patterns detected.")
        return 0

    print("Potential leak findings:")
    for finding in findings:
        print(
            f"- {finding.path}:{finding.line_number} [{finding.pattern_name}] {finding.excerpt}"
        )
    return 1


if __name__ == "__main__":
    sys.exit(main())
