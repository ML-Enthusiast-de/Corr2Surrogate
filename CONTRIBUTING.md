# Contributing

## Development Setup
1. Create a virtual environment:
   - Windows: `py -3.11 -m venv .venv`
   - macOS/Linux: `python3 -m venv .venv`
2. Activate it and install:
   - `python -m pip install --upgrade pip`
   - `python -m pip install -e ".[dev,stats,viz]"`

## Quality Gate (Required Before PR)
1. Run tests: `python -m pytest -q`
2. Run leak scan: `python -m corr2surrogate.ui.cli scan-git-safety`
3. Ensure no private/local outputs are tracked:
   - `git status --short`
   - Confirm `data/private/`, `reports/`, `artifacts/`, and `models/` are clean or ignored.

## Scope Rules
- Keep deterministic analytics/modeling as source of truth.
- Keep LLM behavior bounded to orchestration, explanation, and tool selection.
- Do not commit secrets, machine-specific paths, or private datasets.
- Maintain backward-compatible CLI behavior unless a breaking change is explicitly planned.

## PR Expectations
- Include tests for behavioral changes.
- Update `README.md` when commands, workflow, or capabilities change.
- Keep user-facing language concise, scientific, and reproducible.
