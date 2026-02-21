# Corr2Surrogate

Local-first, privacy-preserving framework to turn real-world sensor CSV data into scientifically validated surrogate models.

## Why this project
Lab and industrial data is messy: different signal counts, changing names, mixed sampling, and varying quality. Corr2Surrogate is designed so users can inject CSV data without manually rebuilding pipelines each time.

## Core concept
Two-agent workflow:
- Agent 1 (`Analyst`): data intake, quality checks, stationarity/outlier checks, correlation/dependency analysis, and surrogateability ranking.
- Agent 2 (`Modeler`): model selection, leakage-safe splitting, training, Optuna tuning, and final surrogate export.

## Is this feasible?
Yes. This is technically feasible with current local tooling and open-weight models if the system is built with deterministic ML modules and strict data contracts.

Feasibility conditions:
- Keep scientific logic in deterministic Python tools.
- Use LLM agents for orchestration, explanations, and user interaction.
- Enforce schema checks, split governance, and run tracking from day one.

## Is it useful?
Yes, especially for teams that need:
- Private/local processing of sensitive measurement data.
- Repeatable analysis from heterogeneous CSVs.
- Fast identification of signals that can be replaced by surrogates.
- Transparent model quality reporting before deployment.


## Are the two agents good enough?
The two-agent split is good and practical for an MVP.

To keep it robust:
- Add a strict JSON handoff contract between agents.
- Require evidence/confidence for agent claims.
- Keep outlier handling and split decisions auditable.
- Gate model promotion by fixed acceptance criteria.

## Current status
Planning phase. Initial architecture is documented in `PROJECT_LAYOUT.md`.

## Planned roadmap
1. MVP:
- CSV ingestion + schema inference.
- Agent 1 quality and correlation analysis.
- Agent 2 baseline surrogate models.
- Basic scientific report output.

2. Modeling upgrade:
- LSTM pipeline for temporal dependencies.
- Optuna-based tuning.
- Stronger residual diagnostics.

3. Production hardening:
- Uncertainty quantification.
- Drift detection and retraining policy.
- Better explainability and governance controls.

## Design principles
- Local-first and private by default.
- Reproducible runs and artifact tracking.
- No raw-data mutation.
- Leakage-safe train/val/test strategy.
- Correlation is evidence, not causality.

## Tech direction
- Orchestration: Python workflow + tool-driven agents.
- Local LLM runtime: Ollama or llama.cpp.
- Data/ML stack: pandas, scipy/statsmodels, scikit-learn, PyTorch, Optuna.

## Documentation
- Architecture and implementation blueprint: `PROJECT_LAYOUT.md`

## License
MIT (`LICENSE`).
