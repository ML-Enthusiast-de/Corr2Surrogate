# Corr2Surrogate - Project Layout and Architecture

## 1. Vision
Build a local-first system that ingests heterogeneous lab/industrial sensor data and produces trustworthy surrogate models with full scientific traceability.

## 2. Inputs and Outputs
Inputs:
- `.csv`, `.xlsx`, `.xls`
- Time-series and steady-state datasets
- Optional user system knowledge (critical/non-virtualizable signals)
- Optional user task override (`regression`, `binary_classification`, `fraud_detection`, etc.)

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
- Detects whether each target behaves like regression, classification, or fraud/anomaly screening
- Lightweight probe-model screening on shortlisted predictors:
  - linear ridge baseline
  - interaction-aware ridge
  - tiny regression tree baseline
  - lagged linear probe for time-series
- Residual nonlinearity, regime, and lag diagnostics
- Ranking of surrogate candidates with dependency-awareness
- Supports user-forced directives:
  - Model specific target(s) with specific predictor sensors, even with weak correlation
- Produces handoff with:
  - ranking, forced directives, system knowledge, normalization plan, loop policy
  - evidence-backed model-strategy prior for Agent 2 (probe inputs, quick metrics, confidence, search order)
- May be bypassed when the user chooses direct modeler mode with explicit target/features/model

### Agent 2 (`Modeler`)
- Reads handoff and enforces constraints/policies
- Must also support direct invocation without handoff when the user explicitly specifies:
  - target signal
  - predictor inputs
  - model family / architecture
- Builds baseline and advanced models (including temporal models)
- Applies split strategy with leakage safeguards:
  - time-series -> ordered blocked split
  - steady-state classification/fraud -> stratified deterministic split
  - steady-state regression -> deterministic modulo split
- Tunes binary classifier decision thresholds on the validation split for classification/fraud tasks before final test reporting
- Planned next step: run Optuna optimization after the deterministic baseline path is stable
- Saves:
  - selected model params (and later tuned params)
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
- Current executable baseline path:
  - direct modeler mode and Agent 1 structured-report handoff both reach a split-safe trainer
  - executable families today: `linear_ridge`, `logistic_regression`, `lagged_logistic_regression`, `bagged_tree_classifier`, `lagged_linear`, `lagged_tree_classifier`, `lagged_tree_ensemble`, `bagged_tree_ensemble`
  - classification / fraud targets now train local classifier baselines with task-aware acceptance checks instead of stopping at detection
  - modeler compares available candidates, runs a bounded acceptance check, can retry with the next safe family when policy allows it, and can also expand the feature set, widen the lag window, or retune binary thresholds inside the same bounded loop
  - if retries stall, modeler emits concrete next experiment / data-collection trajectory suggestions
  - persists artifacts, then lets the LLM interpret the measured result
- User control:
  - if a handoff exists, user can accept the recommendation or override target, predictors, and architecture
  - explicit user overrides win unless blocked by hard policy / safety constraints

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
- `task_type`
- `target_signal`, `feature_signals`
- `forced_modeling_requests`
- `dependency_map`
- `system_knowledge`
- `normalization`
- `acceptance_criteria`
- `loop_policy`
- `model_strategy_recommendation`

Direct Agent 2 invocation must also accept an equivalent explicit request payload when no handoff exists:
- `target_signal`
- `feature_signals`
- `model_family`
- optional normalization / split preferences

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
- If the user explicitly requests a model family, Agent 2 should honor that request and still report whether the result agrees or conflicts with the recommended search order.

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

## 10. Concrete Implementation Sequence
1. Execute `Agent2Handoff` directly:
Agent 2 consumes Agent 1 outputs as an actionable training contract.

2. Build split-safe preprocessing:
Fit imputation / normalization on train only, then apply frozen transforms to validation/test.

3. Implement first end-to-end baselines:
Steady-state: `Ridge` / `ElasticNet`. Time-series: `lagged linear`.

4. Add direct modeler mode:
Allow immediate Agent 2 training from explicit user target / inputs / architecture with no Agent 1 handoff required.

5. Add first nonlinear baseline:
`HistGradientBoostingRegressor` or `ExtraTreesRegressor`.

6. Add model comparison + acceptance logic:
Compare simple vs higher-capacity models and only escalate when validated improvement is meaningful.

7. Persist full reproducibility payload:
Model parameters, split metadata, normalization state, feature list / lag schema, lineage, and metrics.

8. Add post-model failure analysis:
Identify high-error operating regions and recommend concrete new lab/testbench trajectories.

9. Add bounded optimization loops:
Add Optuna only after deterministic training is stable.

10. Add advanced temporal models:
Add GRU / LSTM only after lagged and tree-based baselines are in place.

Current status:
- steps 1-7 are now implemented in the first production path
- bounded acceptance-loop retries are already active across model family, feature set, lag horizon, and binary threshold policy when allowed by loop policy
- a first post-stall experiment-guidance layer is implemented; remaining work is stronger boosted-tree families, deeper region-aware experiment design, inference-first workflows, uncertainty estimation, and later Optuna

## 11. Bounded LLM Autonomy
The LLM should act as an agent, but not as an uncontrolled replacement for deterministic analytics.

Give the LLM freedom to:
- ask clarifying questions when intent is ambiguous
- choose the next safe tool step from the registry
- summarize evidence and explain tradeoffs to the user
- interpret final analysis/modeling outputs and propose next actions

Do not give the LLM freedom to:
- invent metrics, artifacts, or preprocessing state
- bypass split-safety rules
- replace measured validation/test comparisons with intuition
- override explicit user constraints or hard safety rules

Prompt design should always include:
- current workflow stage
- recent user turns (at least the last 5 when available)
- compact state from the last tool result
- explicit hard rules:
  - correlation is not causality
  - user overrides beat recommendations unless blocked by policy
  - time-series does not automatically imply sequence models
