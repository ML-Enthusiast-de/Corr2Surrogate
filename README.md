# Corr2Surrogate

Local-first, privacy-preserving framework that converts real-world sensor data into scientifically validated surrogate models.

## Core Workflow
- Agent 1 (`Analyst`) handles ingestion, quality checks, correlation/dependency analysis, and surrogateability ranking.
- Agent 2 (`Modeler`) trains surrogate models, validates against acceptance criteria, and saves reproducible artifacts.

## Key Features
- Input formats: `.csv`, `.xlsx`, `.xls`
- Multi-sheet Excel support with explicit sheet selection prompt
- Header/data-start inference with user confirmation on low confidence
- Optional Min-Max normalization with saved state for inference de-normalization
- Saved optimization outputs (`model_params.json`) and normalization state
- Agent 1 complete analysis pipeline:
  - quality checks (missingness, duplicates, outliers, timestamp integrity)
  - stationarity assessment per signal
  - multi-technique correlation scan (Pearson/Spearman/Kendall/distance/lagged)
  - feature-engineering opportunity detection (transforms/interactions/lag features)
  - dependency-aware surrogate ranking
- Model savepoints/checkpoints:
  - Load an existing trained model
  - Add additional data and retrain from checkpoint
  - Create child checkpoints for traceable model evolution
- Post-test diagnostics:
  - Detect where model performs poorly
  - Suggest concrete new lab/testbench data trajectories to improve coverage
- User-forced modeling directives:
  - Model target signal `A` with user-defined predictors `[B, C, ...]`
  - Run even if correlation ranking is weak
- Dependency-aware ranking:
  - Penalizes or blocks candidates that depend on other virtualized signals without stable physical anchors
- System knowledge injection:
  - Mark critical signals
  - Mark physically required / non-virtualizable signals
- Agentic loops:
  - If quality is below criteria, continue iterations with guidance
  - Recommend more data, architecture changes, or feature changes when stalled

## Privacy
- Put sensitive data in `data/private/`
- `data/private/`, generated `artifacts/*`, and `reports/*` are git-ignored by default

## Design Principles
- Local-first execution (no data leaves the machine)
- Deterministic scientific tooling behind agent decisions
- Reproducibility through run artifacts and policy-controlled orchestration
- Transparent user communication at each critical decision point

## Project Docs
- Blueprint: `PROJECT_LAYOUT.md`
- Runtime defaults: `configs/default.yaml`
- Harness contract (agents + tool use + local runtime): `HARNESS_CONTRACT.md`

## Agent Behavior Control
Default behavior is controlled by system prompts and config, not fine-tuning.

Where to edit:
- System prompts:
  - `src/corr2surrogate/agents/prompts/analyst_system.txt`
  - `src/corr2surrogate/agents/prompts/modeler_system.txt`
- Prompt overrides in config:
  - `prompts.analyst_system_path`
  - `prompts.modeler_system_path`
  - `prompts.extra_instructions`
- Runtime behavior:
  - `runtime.temperature`
  - `runtime.provider`
  - `runtime.profiles.*`
  - `runtime.endpoints.*`

Fine-tuning is optional and not required for MVP. Start with strong system prompts + strict tool contracts.

## Portable Setup
1. Clone the repo:
```bash
git clone <your-repo-url>
cd Corr2Surrogate
```
2. Create and activate a virtual environment:
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate
```
3. Install dependencies:
```bash
pip install -e ".[dev]"
```
4. Run tests:
```bash
python -m pytest
```

## Harness Modules
- `src/corr2surrogate/orchestration/tool_registry.py`
- `src/corr2surrogate/orchestration/agent_loop.py`
- `src/corr2surrogate/orchestration/runtime_policy.py`
- `src/corr2surrogate/orchestration/local_provider.py`
- `src/corr2surrogate/orchestration/default_tools.py`
- `src/corr2surrogate/orchestration/harness_runner.py`

These modules are OS-agnostic and avoid machine-specific paths. Runtime behavior is controlled through `configs/default.yaml` and optional environment overrides:
- `C2S_CONFIG_PATH`
- `C2S_PROVIDER`
- `C2S_PROFILE`
- `C2S_OFFLINE_MODE`
- `C2S_MODEL`
- `C2S_ENDPOINT`

## CLI
Set up local LLM runtime (recommended before first agent run):
```bash
# Use configured provider/profile from config
corr2surrogate setup-local-llm

# Lightweight CPU path (llama.cpp + small GGUF)
corr2surrogate setup-local-llm --provider llama_cpp --install-provider
```

Run one local agent turn:
```bash
# Optional provider/model overrides (PowerShell)
$env:C2S_PROVIDER="llama_cpp"
$env:C2S_MODEL="c2s-local"

corr2surrogate run-agent-once --agent analyst --message "Load data/private/run1.csv and tell me next step"
```

Run interactive multi-turn session:
```bash
corr2surrogate run-agent-session --agent analyst
```
At startup, analyst session now asks for dataset choice:
- paste a new `.csv` / `.xlsx` path, or
- type `default` to use `data/public/public_testbench_dataset_20k_minmax.csv` (if present).

Session commands:
- `/help`
- `/context`
- `/reset`
- `/exit`

Run Analyst on XLSX (PowerShell):
```powershell
.\.venv\Scripts\Activate.ps1
corr2surrogate run-agent-session --agent analyst
# then paste:
# C:\path\to\Corr2Surrogate\data\private\your_data.xlsx
```
Behavior:
- Agent detects the file path and runs deterministic ingestion + Agent 1 analysis.
- If sheet/header is ambiguous, Agent asks in CLI and continues after your input.
- In sheet/header/target prompts, casual chat still works and the agent steers you back to the pending decision.
- For large datasets, Agent asks if you want full data or a sample subset (uniform/head/tail + row count).
- If NaN or uneven row coverage is detected, Agent asks how to proceed (`keep`, `drop_rows`, `fill_median`, `fill_constant`, `drop_sparse_rows`, `trim_dense_window`, `manual_range`).
- For wide datasets, Agent asks for target focus; type `list` (or `list <filter>`) to show signal names.
- At target selection, you can add user hypotheses inline:
  - correlation: `hypothesis corr target:pred1,pred2; target2:pred3`
  - feature engineering: `hypothesis feature target:signal->rate_change; signal2->square`
- For time-like datasets, Agent asks whether lag analysis is expected and lets you choose lag dimension (`samples` or `seconds`) and search window.
- Report is saved in dataset-related folder:
  - `reports/<dataset_slug>/agent1_<timestamp>.md`
  - example: `reports/rde_v19_1_143_kopie/agent1_20260225_120501.md`
- Report content includes per-target top 10 predictors with correlation type/strength (pearson, spearman, kendall, distance, lagged) plus top feature-engineering opportunities.
- User hypotheses are investigated additionally and reported explicitly (including requested `rate_change` feature checks).
- Agent 1 now also includes:
  - confidence + stability layer for top predictors (bootstrap CI, approximate p-value, window stability),
  - confounder-aware stats (partial correlation + conditional MI for top predictors),
  - planner/critic preprocessing strategy search (candidate plans scored before final run),
  - sensor diagnostics and trust scoring (saturation/quantization/drift/dropout/stuck),
  - experiment trajectory recommendations for targeted new data collection,
  - run lineage (`.lineage.json`) for reproducibility,
  - artifact export folder with CSV/JSON (and optional plot when `matplotlib` is available).

Run deterministic Agent 1 analysis directly:
```bash
corr2surrogate run-agent1-analysis --data-path data/private/run1.csv --timestamp-column time
```
Useful flags for advanced analysis:
```bash
corr2surrogate run-agent1-analysis \
  --data-path data/private/run1.csv \
  --timestamp-column time \
  --confidence-top-k 10 \
  --bootstrap-rounds 40 \
  --stability-windows 4 \
  --strategy-search-candidates 4
```
Optional hypothesis JSON args:
```bash
corr2surrogate run-agent1-analysis \
  --data-path data/private/run1.csv \
  --user-hypotheses-json '[{"target_signal":"y","predictor_signals":["x1","x2"],"user_reason":"lab hypothesis"}]' \
  --feature-hypotheses-json '[{"target_signal":"y","base_signal":"x1","transformation":"rate_change","user_reason":"dynamics expected"}]'
```
Outputs are saved under `reports/<dataset_slug>/`:
- markdown report: `agent1_<timestamp>.md`
- run lineage: `agent1_<timestamp>.lineage.json`
- artifacts folder: `agent1_<timestamp>_artifacts/`

## Troubleshooting (Windows)
If PowerShell says `corr2surrogate` is not recognized:
1. Make sure you are using Python 3.10+ (project requires `>=3.10`).
2. Create a local venv and install:
```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```
3. Run CLI:
```powershell
corr2surrogate --help
```
If execution policy blocks activation, run without activation:
```powershell
.\.venv\Scripts\corr2surrogate.exe run-agent-session --agent analyst
```
If local chat turns fail due provider endpoint not reachable, start/check local runtime:
```powershell
corr2surrogate setup-local-llm
```
Session mode also performs a best-effort local-runtime recovery and retries once before returning an error.

Run leak scan before commit/push:
```bash
c2s-guard
```

Or via module:
```bash
python -m corr2surrogate.ui.cli scan-git-safety
```

Enable automatic pre-commit leak checks:
```bash
git config core.hooksPath .githooks
chmod +x .githooks/pre-commit
```

## License
MIT (`LICENSE`)
