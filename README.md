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

## License
MIT (`LICENSE`)
