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

## CLI
Run one local agent turn:
```bash
corr2surrogate run-agent-once --agent analyst --message "Load data/private/run1.csv and tell me next step"
```

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
