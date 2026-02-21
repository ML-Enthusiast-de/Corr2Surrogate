# Corr2Surrogate - Project Layout and Architecture

## 1. Vision and Scope
Build a local-first, privacy-preserving system that takes heterogeneous sensor CSV data and produces scientifically reliable surrogate models.

Core idea:
- Agent 1 analyzes and explains the data.
- Agent 2 builds and validates surrogate models based on Agent 1 output.

Primary input:
- CSV files with varying numbers of signals, naming styles, units, and sampling behavior.
- Data can be time-series (dynamic) or steady-state / averaged snapshots.

Primary output:
- Human-readable scientific report.
- Machine-readable handoff to modeling.
- Trained surrogate model plus reproducible evaluation artifacts.

## 2. Product Requirements (Non-Negotiable)
- Local execution only. No data leaves the machine.
- Open-weight models only for agent reasoning.
- Deterministic data and modeling pipeline for scientific reliability.
- Full transparency: user sees what is happening, decisions, and confidence.
- Reproducibility: runs are logged and rerunnable.

## 3. Agent Responsibilities

### Agent 1: Data Analyst and Scientific Guide
Responsibilities:
- Greet user and explain capabilities.
- Guide user to data injection and expected formats.
- Auto-profile dataset structure and infer data type:
  - Time-based vs steady-state.
  - Single run vs multi-run.
- Ask user for focus target(s) or proceed with full scan.
- Run quality checks:
  - Completeness / missing values.
  - Duplicate rows and timestamp consistency.
  - Extreme outliers (global and local).
  - Basic noise and drift checks.
  - Stationarity checks for time series (ADF + KPSS).
- Correlation and dependency analysis:
  - Linear: Pearson.
  - Monotonic: Spearman, Kendall.
  - Nonlinear: Mutual Information, Distance Correlation.
  - Time-lag relationships: cross-correlation by lag.
  - Multi-signal effects: partial correlation, multivariate regressors.
- Rank target signals by "surrogateability".
- Produce:
  - Scientist-facing report.
  - Structured handoff package for Agent 2.

### Agent 2: Surrogate Model Builder
Responsibilities:
- Read Agent 1 handoff and user constraints.
- Select modeling approach by complexity and data type.
- Build baseline first, then advanced models when needed.
- Use appropriate splits:
  - Time-aware splitting for time series.
  - Group-aware splitting for multiple runs/batches.
- Train, validate, and test.
- Optimize hyperparameters with Optuna.
- Compare candidates and select best model using predefined criteria.
- Provide model diagnostics and explainability outputs.
- Export artifact bundle for inference and audit.

## 4. Critical Missing Pieces to Add Early
These are usually missed in first drafts and should be designed now:

- Canonical internal data contract:
  - Standard columns (`timestamp`, `run_id`, signals, optional metadata).
  - Units and sensor metadata registry.
- Outlier governance policy:
  - Detect, then ask user whether to cap/remove/keep.
  - Preserve both raw and treated variants.
- Split governance:
  - Prevent leakage for lagged and temporally adjacent samples.
- Experiment tracking:
  - Parameters, metrics, artifacts, data hash, code version.
- Uncertainty reporting:
  - Prediction intervals / quantile models / conformal calibration.
- Explainability:
  - Feature importance and sensitivity for trust.
- Failure policy:
  - What if data is too short, too noisy, or inconsistent.
- Model lifecycle:
  - Retrain triggers and data drift checks over time.

## 5. Recommended Local Stack (Open Weight + Private)

LLM runtime (local):
- `Ollama` or `llama.cpp` with quantized GGUF models.

Candidate agent models:
- `Qwen2.5-7B-Instruct` (good reasoning per compute).
- `Mistral-7B-Instruct` (strong general baseline).
- `Llama-3.1-8B-Instruct` (if license and hardware fit).

Data/ML stack:
- Python, `pandas`, `numpy`, `scipy`, `statsmodels`, `scikit-learn`.
- `optuna` for hyperparameter search.
- `pytorch` for LSTM/sequence models.
- Optional: `xgboost`/`lightgbm` for strong non-sequence baselines.

Orchestration:
- Keep core logic deterministic in Python.
- Use LLM agents for interaction, explanations, and decision prompts.
- Avoid letting LLM directly execute modeling math without tool wrappers.

## 6. Handoff Contract (Agent 1 -> Agent 2)
Define a strict JSON schema. Example sections:

- `dataset_profile`:
  - row count, signal list, inferred sampling, missingness summary.
- `data_type_assessment`:
  - time-series vs steady-state, confidence, evidence.
- `quality_flags`:
  - outlier signals, stationarity flags, drift warnings.
- `target_ranking`:
  - ordered targets with surrogateability scores.
- `candidate_features`:
  - recommended feature subsets and lag features.
- `constraints`:
  - user constraints, compute budget, latency requirements.
- `acceptance_criteria`:
  - metrics thresholds and mandatory diagnostics.

## 7. Model Selection Policy
Use policy rules before heavy optimization:

- If steady-state + strong linear relation:
  - OLS -> Ridge/Lasso.
- If steady-state + nonlinear relation:
  - Random Forest / Gradient Boosting.
- If time-series with short memory:
  - Lagged regression / tree models with lag features.
- If time-series with long temporal dependencies:
  - LSTM (then possibly TCN/Transformer only if needed).

Always run baselines first to create a performance floor.

## 8. Evaluation and Scientific Reporting
Minimum report sections:
- Data summary and assumptions.
- Quality checks and user decisions (especially outliers).
- Correlation findings with caveats (correlation is not causation).
- Model comparison table.
- Final model metrics:
  - Regression: MAE, RMSE, R2.
  - Time-series: horizon-wise MAE/RMSE, residual autocorrelation.
- Error analysis:
  - where model fails (ranges, regimes, transients).
- Uncertainty and confidence statements.

## 9. Suggested Repository Layout
```text
Corr2Surrogate/
  README.md
  PROJECT_LAYOUT.md
  pyproject.toml
  src/corr2surrogate/
    agents/
      agent1_analyst.py
      agent2_modeler.py
      prompts/
    core/
      config.py
      schemas.py
      logging.py
    ingestion/
      csv_loader.py
      schema_inference.py
      profiler.py
    analytics/
      quality_checks.py
      stationarity.py
      correlations.py
      ranking.py
      reporting.py
    modeling/
      splitters.py
      baselines.py
      sequence_lstm.py
      optuna_tuning.py
      evaluation.py
      uncertainty.py
    orchestration/
      workflow.py
      handoff_contract.py
    ui/
      cli.py
      api.py
    persistence/
      run_store.py
      artifact_store.py
  configs/
    default.yaml
  reports/
  artifacts/
  tests/
    test_ingestion.py
    test_stationarity.py
    test_splitting.py
    test_modeling_pipeline.py
```

## 10. End-to-End Workflow
1. User uploads CSV(s).
2. Agent 1 profiles structure and classifies data type.
3. Agent 1 runs quality checks and asks user decisions where required.
4. Agent 1 computes dependencies and target ranking.
5. Agent 1 generates report + handoff JSON.
6. Agent 2 selects modeling path and builds baselines.
7. Agent 2 runs Optuna for candidate improvements.
8. Agent 2 evaluates on held-out test with leakage-safe split.
9. Agent 2 exports model and diagnostics.
10. System presents final recommendation and artifacts.

## 11. Engineering Guardrails
- Never mutate raw data in place.
- Every transformation is logged and reproducible.
- Time-based split enforcement is mandatory for temporal data.
- Agent statements should include confidence and evidence source.
- Keep prompts versioned for auditability.

## 12. Phased Build Plan

Phase 1 (MVP):
- CSV ingestion + schema inference.
- Agent 1 quality checks + correlation ranking.
- Agent 2 baseline regressors.
- Basic report generation.

Phase 2:
- LSTM pipeline + Optuna tuning.
- Handoff JSON schema enforcement.
- Better diagnostics and residual analysis.

Phase 3:
- Uncertainty quantification.
- Drift monitoring and retrain recommendation.
- Enhanced explainability and user policy controls.

## 13. Practical First Implementation Decisions
- Start with CLI before full UI.
- Support one CSV schema first, then generalize using adapters.
- Enforce one canonical dataframe representation internally.
- Use YAML config for thresholds and policy switches.
- Add synthetic test datasets to validate behavior on known patterns.

## 14. Risks and Mitigations
- Risk: Over-reliance on correlation for causality.
  - Mitigation: explicit warnings and optional causal discovery module later.
- Risk: Data leakage in temporal tasks.
  - Mitigation: strict splitter module and tests.
- Risk: Too much LLM autonomy in scientific steps.
  - Mitigation: deterministic tool calls and bounded agent actions.
- Risk: User confusion from too many options.
  - Mitigation: progressive disclosure and sensible defaults.


