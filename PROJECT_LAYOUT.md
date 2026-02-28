# Corr2Surrogate - Project Layout and Architecture

## 1. Vision
Build a local-first system that ingests heterogeneous lab/industrial sensor data and produces trustworthy surrogate models with full scientific traceability.

## 2. Inputs and Outputs
Inputs:
- `.csv`, `.xlsx`, `.xls`
- Time-series and steady-state datasets
- Optional user system knowledge (critical/non-virtualizable signals)

Outputs:
- Scientist-facing analysis report
- Agent handoff payload (machine-readable)
- Trained model artifacts, optimization parameters, normalization state
- Iteration guidance when quality is insufficient

## 3. Agent Roles
### Agent 1 (`Analyst`)
- Data intake and structure inference
- Sheet selection handling for multi-sheet Excel
- Header/data-start inference with confidence checks
- Data quality checks and correlation/dependency analysis
- Lightweight probe-model screening on shortlisted predictors:
  - linear ridge baseline
  - interaction-aware ridge
  - piecewise tree proxy
  - lagged linear probe for time-series
- Residual nonlinearity, regime, and lag diagnostics
- Ranking of surrogate candidates with dependency-awareness
- Supports user-forced directives:
  - Model specific target(s) with specific predictor sensors, even with weak correlation
- Produces handoff with:
  - ranking, forced directives, system knowledge, normalization plan, loop policy
  - model-strategy prior for Agent 2 (search order, tree-worth-testing, sequence-worth-testing)

### Agent 2 (`Modeler`)
- Reads handoff and enforces constraints/policies
- Builds baseline and advanced models (including temporal models)
- Applies split strategy with leakage safeguards
- Runs Optuna optimization
- Saves:
  - tuned model params
  - scaler/normalization state
  - metrics and metadata
- Runs agentic loops if criteria not met
- Explains whether next step should be:
  - more data
  - different architecture
  - feature/lag redesign
- Planned model order:
  - linear / lagged linear baseline first
  - tree ensembles next when Agent 1 finds interaction or regime evidence
  - sequence models only after simpler lagged/tabular baselines fail

## 4. Critical Behavioral Requirements
- Correlation is not the only trigger for modeling:
  - user can force target/predictor combinations
- Ranking cannot assume independent virtualization:
  - avoid selecting targets that rely on other virtualized targets without physical anchors
- System knowledge must override automation when needed:
  - critical signals may be required physically
  - non-virtualizable signals must not become surrogate targets
- If quality is below acceptance:
  - loop with bounded retries
  - present explicit remediation options to user

## 5. Handoff Contract Essentials
`Agent2Handoff` includes:
- `target_signal`, `feature_signals`
- `forced_modeling_requests`
- `dependency_map`
- `system_knowledge`
- `normalization`
- `acceptance_criteria`
- `loop_policy`
- `model_strategy_recommendation`

## 6. Ranking Strategy
Dependency-aware ranking logic:
- Start from base surrogateability score
- Penalize dependencies on signals also marked for virtualization
- Mark infeasible when required dependencies lack stable physical path
- Keep forced user directives separate and always executable (unless hard policy blocks)

## 7. Agentic Loop Strategy
Each iteration:
1. Evaluate metrics vs acceptance criteria
2. If met: stop and export
3. If unmet and attempts remain: continue with recommendations
4. If max attempts reached: stop with failure guidance and explicit next actions

Guidance includes:
- collect more representative data
- switch architecture class
- adjust regularization / feature engineering / temporal windowing

Model-selection rule:
- Agent 2 must treat Agent 1's recommendation as a prior, not a hard decision.
- If a simple baseline already performs well, keep it unless the higher-capacity model delivers meaningful validated improvement.
- Time-series does not automatically imply LSTM.

## 8. Persistence and Reproducibility
Persist per run:
- trained model artifact(s)
- `model_params.json` (best params + metrics + split info)
- `normalization_state.json` (for inverse transform at inference)
- user/system constraints used in training

## 9. Repository Layout (Current Direction)
```text
Corr2Surrogate/
  README.md
  PROJECT_LAYOUT.md
  configs/default.yaml
  data/private/
  artifacts/
  reports/
  src/corr2surrogate/
    agents/
    analytics/
    ingestion/
    modeling/
    orchestration/
    persistence/
    ui/
  tests/
```

## 10. Implementation Phases
Phase 1:
- ingestion + profiling + ranking + forced directives
- probe-model screening + model-family recommendation prior
- baseline training + artifact persistence

Phase 2:
- tree ensembles + lagged tabular modeling + robust optimization loops
- stronger diagnostics and failure guidance

Phase 3:
- sequence models + drift monitoring + uncertainty calibration + governance hardening
