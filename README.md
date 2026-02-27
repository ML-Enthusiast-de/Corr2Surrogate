# Corr2Surrogate

Corr2Surrogate is a local-first framework for converting real-world sensor data into validated surrogate-model candidates.

## What It Does
- Agent 1 (`Analyst`) ingests CSV/XLS/XLSX data, checks data quality, runs correlation analysis, and ranks surrogate candidates.
- Agent 2 (`Modeler`) is designed to train/evaluate surrogate models with reproducible artifacts and checkpoints.

## Key Capabilities
- CSV/XLS/XLSX ingestion with sheet selection and header/data-start inference
- Quality checks (missingness, duplicates, outliers, timestamp integrity)
- Correlation analysis (Pearson, Spearman, Kendall, distance, lag-aware)
- Feature-engineering scans (including `rate_change`)
- Dependency-aware surrogate ranking
- Dataset-scoped reports and artifact export
- Local runtime by default; optional API mode via explicit opt-in

## Privacy Defaults
- Keep private datasets in `data/private/`.
- `data/private/`, `reports/`, and `artifacts/` are git-ignored by default.

## Prerequisites
- Python `>=3.10` (project default tested with Python 3.11)
- Git
- For local LLM mode:
  - Windows: `llama.cpp` or Ollama
  - macOS: `llama.cpp` (recommended) or Ollama

## Quickstart (Windows)
1. Clone and enter repo:
```powershell
git clone <your-repo-url>
cd Corr2Surrogate
```
2. Create virtual environment and install:
```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```
3. Set up local LLM runtime (Qwen 4B local profile):
```powershell
& .\.venv\Scripts\corr2surrogate.exe setup-local-llm --provider llama_cpp --install-provider
```
4. Start analyst session:
```powershell
# works even if you stay in conda base/ml
& .\.venv\Scripts\corr2surrogate.exe run-agent-session --agent analyst
```

## Quickstart (macOS)
1. Install `llama.cpp`:
```bash
brew install llama.cpp
```
2. Clone and enter repo:
```bash
git clone <your-repo-url>
cd Corr2Surrogate
```
3. Create virtual environment and install:
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```
4. Set up local LLM runtime:
```bash
corr2surrogate setup-local-llm --provider llama_cpp
```
5. Start analyst session:
```bash
./.venv/bin/corr2surrogate run-agent-session --agent analyst
```

## Analyst Session Usage
At startup you can:
- paste a `.csv/.xlsx/.xls` path, or
- type `default` to run the built-in public test dataset.

Default dataset note:
- `data/public/public_testbench_dataset_20k_minmax.csv` is sanitized and intentionally includes ~5% missing values in three representative signals (`B`, `D`, `F`), while all other signals and `time` remain complete. This lets users test missing-data handling and leakage warnings without collapsing row count when trying `drop_rows`.

Useful commands:
- `/help`
- `/context`
- `/reset`
- `/exit`

Target selection shortcuts:
- `list`
- `list <filter>`
- `all`
- comma-separated signal names
- numeric index

Hypothesis syntax:
- Correlation: `hypothesis corr target:pred1,pred2; target2:pred3`
- Feature: `hypothesis feature target:signal->rate_change; signal2->square`

## Optional API Mode (Explicit Opt-In)
Default policy is local-only. API mode must be explicitly enabled.

### PowerShell
```powershell
$env:C2S_PROVIDER="openai"
$env:C2S_REQUIRE_LOCAL_MODELS="false"
$env:C2S_BLOCK_REMOTE_ENDPOINTS="false"
$env:C2S_API_CALLS_ALLOWED="true"
$env:C2S_OFFLINE_MODE="false"
$env:C2S_API_KEY="<your_api_key>"
$env:C2S_MODEL="gpt-4.1-mini"
# optional:
# $env:C2S_ENDPOINT="https://api.openai.com/v1/chat/completions"

& .\.venv\Scripts\corr2surrogate.exe setup-local-llm --provider openai
```

### Bash/Zsh
```bash
export C2S_PROVIDER=openai
export C2S_REQUIRE_LOCAL_MODELS=false
export C2S_BLOCK_REMOTE_ENDPOINTS=false
export C2S_API_CALLS_ALLOWED=true
export C2S_OFFLINE_MODE=false
export C2S_API_KEY=<your_api_key>
export C2S_MODEL=gpt-4.1-mini
# optional:
# export C2S_ENDPOINT=https://api.openai.com/v1/chat/completions

corr2surrogate setup-local-llm --provider openai
```

## Deterministic Agent 1 (No LLM Call)
```bash
corr2surrogate run-agent1-analysis --data-path data/private/run1.csv --timestamp-column time
```

Example with advanced options:
```bash
corr2surrogate run-agent1-analysis \
  --data-path data/private/run1.csv \
  --timestamp-column time \
  --target-signals y \
  --max-lag 12 \
  --confidence-top-k 10 \
  --bootstrap-rounds 40 \
  --stability-windows 4 \
  --strategy-search-candidates 4
```

## Behavior and Prompt Control
- Runtime defaults: `configs/default.yaml`
- Analyst prompt: `src/corr2surrogate/agents/prompts/analyst_system.txt`
- Modeler prompt: `src/corr2surrogate/agents/prompts/modeler_system.txt`
- Optional prompt overrides:
  - `prompts.analyst_system_path`
  - `prompts.modeler_system_path`
  - `prompts.extra_instructions`

## Output Structure
Outputs are grouped by dataset slug under `reports/<dataset_slug>/`:
- Markdown report: `agent1_<timestamp>.md`
- Lineage JSON: `agent1_<timestamp>.lineage.json`
- Artifact directory: `agent1_<timestamp>_artifacts/`

## Quality and Security Checks
Run tests:
```bash
./.venv/bin/python -m pytest
```
Run leak scan before commit/push:
```bash
./.venv/bin/c2s-guard
# or
./.venv/bin/python -m corr2surrogate.ui.cli scan-git-safety
```

Windows equivalents:
```powershell
& .\.venv\Scripts\python.exe -m pytest
& .\.venv\Scripts\c2s-guard.exe
# or
& .\.venv\Scripts\python.exe -m corr2surrogate.ui.cli scan-git-safety
```

## Troubleshooting
- `corr2surrogate` without path only works when the venv is active or the script is on `PATH`.
- If `corr2surrogate` is not recognized on Windows, use:
```powershell
& .\.venv\Scripts\corr2surrogate.exe --help
```
- If local provider is not reachable, run:
```bash
corr2surrogate setup-local-llm
```

## License
MIT (`LICENSE`)
