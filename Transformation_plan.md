# TRANSFORMATION_PLAN.md

# Corr2Surrogate Transformation Plan

## Goal

Transform this repository from a “data -> surrogate model” framework into a **local-first autonomous ML workbench for structured data** that:

- runs fully locally by default
- can be steered when the user wants control
- can act autonomously when the user prefers delegation
- supports multiple modeling routes, not just one narrow pipeline
- produces reproducible artifacts, reports, and deployable outputs
- remains deterministic, testable, and auditable

The repository already has strong foundations for this direction: local-first policy, explicit remote opt-in, deterministic training and evaluation, CLI entry points, artifact generation, and test/CI structure.

---

## 1. Product definition

### New product identity

Corr2Surrogate becomes:

> A local-first autonomous ML workbench for structured data, where agents can profile data, choose evaluation strategy, propose and run experiments, audit results, and produce deployment-grade artifacts.

### Core principles

1. **Local-first always**
   - No remote model or API use unless explicitly enabled.
   - The current runtime policy remains and becomes more visible and more enforceable.

2. **Steerable autonomy**
   - Support three execution modes:
     - `manual`
     - `guided`
     - `autonomous`
   - In autonomous mode, the system may propose and run experiments until stop criteria are met.

3. **Policy-aware optimization**
   - Optimize not only for score, but also for constraints such as:
     - privacy
     - latency
     - interpretability
     - robustness
     - uncertainty requirements
     - compute budget

4. **Workflow search, not just hyperparameter search**
   - The system should choose:
     - task route
     - feature strategy
     - split strategy
     - model family
     - calibration method
     - search budget
     - stopping rule

5. **Reproducibility by default**
   - Every decision, experiment, artifact, and result must be persisted.

---

## 2. Scope changes

### Keep and build on

Preserve these strengths:

- local-only runtime policy and explicit remote opt-in
- the current analyst/modeler flow as the seed of a broader agent system
- deterministic training and evaluation
- artifact and report generation
- current CLI surface where practical
- current tests and CI as the baseline quality bar

### Replace or expand

- replace the public two-agent framing with a multi-role planner/executor system
- make surrogate modeling one route among several
- replace ad hoc experimentation flow with an experiment graph plus run registry
- upgrade reports into full decision artifacts
- make uncertainty, calibration, robustness, and drift first-class outputs

---

## 3. Target architecture

### Agent roles

Replace the current public framing with five internal roles.

#### 3.1 Profiler
Responsible for:
- schema inference
- target detection and task inference
- missingness profiling
- cardinality profiling
- leakage checks
- time, group, and entity detection
- dataset warnings

#### 3.2 Strategist
Responsible for:
- choosing task route
- choosing metric
- choosing split strategy
- building candidate workflow plans
- assigning budget
- deciding whether uncertainty or calibration is required
- deciding whether further experimentation is worth it

#### 3.3 Experimenter
Responsible for:
- executing workflow candidates
- tracking trials
- early-stopping weak branches
- collecting metrics
- persisting artifacts

#### 3.4 Auditor
Responsible for:
- calibration
- uncertainty
- slice analysis
- robustness checks
- drift and OOD checks
- repeatability validation
- model card generation

#### 3.5 Reporter
Responsible for:
- run summary
- leaderboard
- executive report
- machine-readable artifact index
- recommendation memo
- deployment bundle description

---

## 4. Package structure

Refactor toward this structure:

```text
src/corr2surrogate/
  agents/
    profiler.py
    strategist.py
    experimenter.py
    auditor.py
    reporter.py
    coordinator.py
    schemas.py
    policies.py
  autonomy/
    controller.py
    stop_rules.py
    approval.py
    budgets.py
  datasets/
    schema.py
    profiling.py
    quality.py
    leakage.py
    slicing.py
    time_semantics.py
  planning/
    task_inference.py
    route_selection.py
    metric_selection.py
    split_selection.py
    search_space.py
    workflow_graph.py
  experiments/
    registry.py
    tracker.py
    execution.py
    leaderboard.py
    replay.py
  features/
    tabular_basic.py
    tabular_interactions.py
    time_lags.py
    time_windows.py
    encoders.py
    selectors.py
  modeling/
    families/
      linear.py
      tree.py
      ensemble.py
      baseline.py
    calibration.py
    uncertainty.py
    robustness.py
    training.py
    inference.py
  evaluation/
    metrics.py
    diagnostics.py
    drift.py
    ood.py
    conformal.py
    stability.py
  artifacts/
    manifests.py
    model_card.py
    reports.py
    packaging.py
  serve/
    app.py
    schemas.py
    monitoring.py
  ui/
    cli.py
    api.py
    streamlit_app.py
```

### Refactor rule

Do not do this as a big-bang rewrite.

Move code incrementally and keep compatibility shims for existing CLI commands until the new system is stable.

---

## 5. Execution modes

Add one explicit concept: `execution_mode`.

Supported values:

- `manual`
- `guided`
- `autonomous`

### 5.1 Manual mode
The system only analyzes and proposes.
No experiment runs without explicit approval.

### 5.2 Guided mode
The system proposes a plan and asks for approval at checkpoints:
- before search
- before expensive branches
- before final selection

### 5.3 Autonomous mode
The system may:
- propose workflows
- run experiments
- expand or prune search
- stop when stop criteria are met

### Required CLI and API support

All major commands must accept:

```bash
--execution-mode manual|guided|autonomous
```

---

## 6. Policy system

Promote the current runtime policy into a full **ML policy layer**.

### Policy schema

```yaml
policy:
  locality:
    remote_allowed: false
    local_models_only: true
  autonomy:
    execution_mode: autonomous
    allow_auto_run: true
    approval_required_for_expensive_runs: true
  compute:
    max_wall_clock_minutes: 120
    max_trials: 200
    max_parallel_trials: 4
    max_memory_gb: 24
  optimization:
    objective: best_robust_score
    primary_metric: auto
    secondary_metrics: [calibration, stability]
  constraints:
    latency_ms_max: null
    interpretability: medium
    uncertainty_required: true
    reproducibility_required: true
  safety:
    strict_leakage_checks: true
    strict_time_ordering: true
    reject_unstable_models: true
  reporting:
    create_model_card: true
    create_risk_report: true
    create_experiment_graph: true
```

### Acceptance criteria

- policy loads from YAML and can be overridden by CLI flags
- every experiment stores the resolved effective policy
- planner decisions cite the policy fields that influenced them

---

## 7. Task routes

Generalize the system into routes.

### Initial supported routes

- `tabular_regression`
- `tabular_classification`
- `time_series_forecasting`
- `time_aware_prediction`
- `surrogate`

Keep current surrogate capabilities inside the `surrogate` route instead of making them the whole identity of the project.

### Route inference

Profiler plus Strategist decide route using:
- target type
- timestamp presence
- group or entity columns
- lag structure
- dataset size
- missingness
- user hints

### Acceptance criteria

- route is inferred automatically when possible
- user can override route explicitly
- route selection is written to `plan.json`

---

## 8. Planning system

Introduce a real plan object.

### New artifact: `plan.json`

Example:

```json
{
  "task_type": "tabular_classification",
  "route": "tabular_classification",
  "primary_metric": "roc_auc",
  "split_strategy": {
    "type": "stratified_holdout",
    "seed": 42
  },
  "feature_recipes": [
    "basic_numeric",
    "categorical_frequency",
    "missingness_indicators"
  ],
  "candidate_families": [
    "logistic",
    "random_forest",
    "gradient_boosted_tree"
  ],
  "calibration_plan": "platt_or_isotonic_if_needed",
  "uncertainty_plan": "conformal_if_supported",
  "stop_criteria": {
    "max_trials": 60,
    "min_improvement_delta": 0.002,
    "patience": 12
  }
}
```

### Planner requirements

Planner must produce:
- one primary plan
- at least two alternative plans
- rationale for why each plan exists
- estimated cost per plan
- notes for rejected routes

---

## 9. Autonomous experimentation loop

This is the core change.

### New orchestration loop

```text
profile -> plan -> execute batch -> audit -> decide next step
```

### Required loop decisions

After each experiment batch, the Strategist decides one of:

- `expand_search`
- `refine_promising_branch`
- `calibrate_top_models`
- `run_robustness_checks`
- `stop_and_report`

### Stop conditions

Support these stop rules:

- budget exhausted
- no meaningful improvement for N trials
- robust winner found
- policy constraints satisfied and confidence threshold met
- user interruption

### Definition of “model is great”

Make this explicit and configurable:

```yaml
autonomy:
  success_criteria:
    score_threshold: null
    min_margin_over_baseline: 0.03
    max_instability: 0.02
    calibration_max_error: 0.05
    coverage_minimum: 0.9
```

The system should stop only when:
- the selected model beats the baseline
- stability is acceptable
- calibration or uncertainty criteria are satisfied when required
- no major leakage or catastrophic failure remains

---

## 10. Experiment graph

Add a structured experiment graph instead of flat runs.

### New artifact: `experiment_graph.json`

Each node may represent:
- split strategy
- feature recipe
- model family
- hyperparameter config
- calibration config
- robustness test
- final selection decision

Each edge may represent:
- expanded from
- pruned because
- promoted because
- calibrated because
- rejected because

### Acceptance criteria

- every run has a graph
- graph can be rendered as JSON and markdown
- any final result can be traced back to the decisions that produced it

---

## 11. Model family system

Keep the current deterministic, lightweight modeling base and expand in layers.

### Baseline family layer
Must always include:
- linear regression or ridge
- logistic regression
- simple decision tree
- majority or mean baseline
- last-value and seasonal naive for time series

### Standard family layer
Add:
- random forest
- extra trees
- gradient boosting
- histogram boosting where practical

### Optional plugin family layer
Keep the core dependency-light.
Optional extras may later expose:
- surrogate-specific libraries
- heavier AutoML-style families
- optional external libraries

---

## 12. Features system

Feature generation must become route-aware and policy-aware.

### Required feature modules

#### Tabular basic
- imputation
- scaling when needed
- missingness flags
- category frequency or count encoding

#### Tabular interactions
- bounded pairwise interactions
- ratio and delta transforms
- monotonic-safe transforms where justified

#### Time features
- lags
- rolling mean, std, min, max
- seasonal offsets
- trend deltas
- gap indicators

### Rules

- feature generation must be deterministic
- all generated features must be logged to `feature_manifest.json`
- feature selection decisions must be explainable

---

## 13. Evaluation redesign

Expand evaluation into a richer layer.

### Required outputs per candidate

#### Performance
- primary metric
- secondary metrics
- train/validation gap

#### Calibration
- Brier score for probabilistic classification
- expected calibration error
- calibration curve bins

#### Stability
- variance across seeds or folds
- sensitivity to row perturbation
- sensitivity to missingness perturbation

#### Robustness
- slice-level performance
- degradation under noise
- degradation under reduced features

#### Uncertainty
- conformal intervals or sets where supported
- empirical coverage
- interval width statistics

#### Operational
- fit time
- predict time
- artifact size
- memory estimate

---

## 14. Uncertainty and calibration

Add this as a first-class subsystem.

### Minimum implementation

#### Classification
- probability calibration
  - Platt scaling
  - isotonic when justified
- reliability report

#### Regression and forecasting
- split conformal prediction intervals
- coverage report

### New artifacts

- `uncertainty_report.json`
- `calibration_report.json`

### Acceptance criteria

- any model marked `recommended` must include uncertainty and calibration outputs when policy requires them
- the final report must state whether uncertainty requirements were satisfied

---

## 15. Drift, OOD, and monitoring

Make monitoring structured and visible.

### Monitoring modules

- feature drift
- target drift when labels are available
- schema drift
- missingness drift
- simple OOD score for tabular inputs
- time distribution drift

### Required artifact

- `drift_report.json`

### Serving integration

When local serving is enabled, log:
- schema mismatches
- missing features
- OOD score summary
- drift snapshots

---

## 16. Artifacts contract

Standardize the output layout.

### New run directory layout

```text
artifacts/run_<timestamp>/
  manifest.json
  policy_resolved.yaml
  dataset_profile.json
  plan.json
  alternatives.json
  experiment_graph.json
  leaderboard.csv
  best_model/
    model.pkl
    model_config.json
    feature_manifest.json
  challenger_models/
  reports/
    summary.md
    technical_report.md
    model_card.md
    risk_report.md
  diagnostics/
    calibration_report.json
    uncertainty_report.json
    drift_report.json
    stability_report.json
    slice_report.json
  logs/
    agent_trace.jsonl
    execution.log
```

### Rule

Every emitted file must be listed in `manifest.json`.

---

## 17. Reporting redesign

Make reports layered.

### Required reports

#### `summary.md`
Contains:
- what the system found
- the recommended model
- why it was chosen
- confidence in the recommendation
- whether more search is likely to help

#### `technical_report.md`
Contains:
- dataset profile
- route
- split choice
- search path
- model comparison
- calibration and uncertainty
- robustness findings

#### `model_card.md`
Contains:
- intended use
- limitations
- training setup
- metrics
- risks

#### `risk_report.md`
Contains:
- leakage risk
- instability
- drift sensitivity
- missingness sensitivity
- unsupported use cases

---

## 18. CLI redesign

Preserve current commands, but add a unified surface.

### New primary CLI

```bash
corr2surrogate run \
  --data-path data.csv \
  --target target \
  --execution-mode autonomous \
  --policy configs/policies/default.yaml \
  --goal best_robust_score
```

### Required subcommands

```bash
corr2surrogate profile
corr2surrogate plan
corr2surrogate experiment
corr2surrogate audit
corr2surrogate report
corr2surrogate serve
corr2surrogate replay
```

### Backward compatibility

Keep current commands working for one transition cycle and mark them as legacy in help text.

---

## 19. API redesign

Turn the current API entrypoint into a full local API surface.

### Endpoints

- `POST /profile`
- `POST /plan`
- `POST /run`
- `POST /audit`
- `POST /report`
- `POST /predict`
- `GET /runs/{run_id}`
- `GET /runs/{run_id}/artifacts`
- `GET /health`

### Requirements

- local only by default
- no remote dependencies required
- OpenAPI schema must reflect execution mode and policy objects

---

## 20. Streamlit interface

Add a local UI as an operator console.

### Screens

- upload / dataset selection
- profile view
- plan view
- experiment progress
- leaderboard
- audit view
- final report view
- serve / inference view

### Behavior

- user can inspect and override the plan
- user can switch between guided and autonomous mode
- user can stop runs safely
- user can compare top models visually

---

## 21. LLM and agent contract redesign

Make the agent layer stricter and schema-driven.

### Required changes

#### Structured action schemas
Each agent emits one structured action validated against Pydantic models.

#### Example action types
- `inspect_dataset`
- `choose_route`
- `choose_split`
- `queue_experiments`
- `request_calibration`
- `request_robustness_check`
- `stop_with_recommendation`

#### Refusal and fallback behavior
If an agent output is invalid:
- retry once
- fall back to deterministic planner logic
- log the failure

### Important rule

The system must remain fully functional without any LLM.
Local LLM assistance is an enhancement, not a hard dependency.

---

## 22. Benchmark harness

Add a benchmark package.

### New directory

```text
bench/
  configs/
  datasets/
  runners/
  reports/
```

### Benchmark tracks

#### Track 1: tabular classification and regression
OpenML provides benchmark suites commonly used for structured-data evaluation.

#### Track 2: time-series classification
UEA/UCR-style benchmark collections provide a standard comparison track.

#### Track 3: forecasting
M4-style forecasting benchmarks provide a standard evaluation route.

### Benchmark outputs

- `benchmark_results.csv`
- `benchmark_summary.md`
- `ablations.csv`

### Required ablations

- no agent planning vs agent planning
- no feature engineering vs feature engineering
- no uncertainty vs uncertainty-aware selection
- minimal search vs autonomous search
- strict policy vs permissive policy

---

## 23. Testing expansion

Extend test coverage around the new architecture.

### Unit tests
- route inference
- split selection
- policy resolution
- stop rules
- artifact manifests
- calibration utilities
- conformal utilities

### Integration tests
- profile -> plan
- plan -> execute
- execute -> audit
- audit -> report

### End-to-end tests
- one small tabular dataset
- one small time-series dataset
- one surrogate-route dataset

### Failure tests
- malformed schema
- bad target selection
- no valid model family
- invalid agent output
- budget exhaustion
- user stop mid-run

### Regression tests
Golden artifact snapshots for:
- `plan.json`
- `leaderboard.csv`
- `summary.md`

---

## 24. CI and repo health

Expand the quality gate.

### CI jobs to add

- formatting check
- lint check
- type check
- tests
- coverage
- benchmark smoke test
- artifact contract test
- package build test

### Repo files to add

- `SECURITY.md`
- `CHANGELOG.md`
- `CODE_OF_CONDUCT.md`

---

## 25. Migration plan

### Phase 1: architecture foundations
Implement first:
1. policy schema
2. plan object
3. execution modes
4. experiment registry
5. artifact contract
6. compatibility shims

### Phase 2: agent upgrade
Implement:
1. profiler
2. strategist
3. experimenter
4. coordinator loop
5. schema-validated actions

### Phase 3: evaluation upgrade
Implement:
1. calibration
2. conformal uncertainty
3. stability checks
4. slice analysis
5. drift report

### Phase 4: product surfaces
Implement:
1. unified CLI
2. local API
3. Streamlit app
4. local serve mode

### Phase 5: benchmark and polish
Implement:
1. benchmark harness
2. ablations
3. docs rewrite
4. end-to-end demos

---

## 26. Implementation rules for Codex

1. Do not remove local-first defaults.
2. Do not make remote APIs required.
3. Preserve existing commands where practical using compatibility wrappers.
4. Prefer additive refactors over destructive rewrites.
5. Keep deterministic behavior where already present.
6. Add tests with each major subsystem.
7. Persist every important decision as an artifact.
8. Do not introduce heavyweight dependencies into the core package unless behind optional extras.
9. Maintain clean separation between:
   - planning
   - execution
   - evaluation
   - reporting
10. The system must work without LLMs.

### Required deliverables from Codex

- refactored package structure
- new policy system
- new execution mode system
- new plan object and planner
- experiment graph and registry
- uncertainty and calibration modules
- standardized artifacts
- unified CLI
- expanded tests
- rewritten README and docs
- one complete local demo path

---

## 27. README rewrite requirements

The README must present the system as:

- local-first
- autonomous but steerable
- route-based
- reproducible
- policy-aware
- artifact-rich

### Required README sections

1. What it is
2. Why local-first
3. Execution modes
4. Supported routes
5. Quickstart
6. Example outputs
7. Artifact structure
8. Policy examples
9. Benchmarks
10. Development and tests

---

## 28. Definition of done

The transformation is complete when all of the following are true:

1. A user can run one command locally on a CSV and get:
   - dataset profile
   - plan
   - experiments
   - leaderboard
   - calibrated and uncertainty-aware best model
   - final report

2. The system supports:
   - manual
   - guided
   - autonomous

3. Autonomous mode can keep experimenting until:
   - budget is exhausted, or
   - success criteria are met

4. Every run produces:
   - `policy_resolved.yaml`
   - `dataset_profile.json`
   - `plan.json`
   - `experiment_graph.json`
   - `leaderboard.csv`
   - summary and technical reports
   - uncertainty, calibration, and drift artifacts

5. The system works fully offline by default.

6. Tests cover the new planner, executor, and auditor flow.

7. Existing deterministic behavior remains intact where applicable.

---

## 29. First implementation sprint

### Sprint 1
- add policy schema
- add execution mode support
- create `plan.json`
- create experiment registry
- refactor current analyst/modeler flow into `profiler` + `strategist` + `experimenter`
- standardize artifact directory
- add compatibility shims
- update CLI with `run`, `profile`, `plan`

### Sprint 2
- add experiment graph
- add calibration
- add conformal uncertainty
- add stability checks
- add `summary.md`, `technical_report.md`, `model_card.md`

### Sprint 3
- add API
- add Streamlit app
- add serve mode and local monitoring
- add benchmark harness

---

## 30. Codex prompt to paste

```text
Read TRANSFORMATION_PLAN.md and implement it incrementally.

Constraints:
- Keep local-first defaults and existing runtime policy.
- Remote APIs must remain explicit opt-in only.
- The system must work without any LLM.
- Preserve existing functionality where possible through compatibility wrappers.
- Refactor incrementally, not as a destructive rewrite.
- Keep deterministic behavior where it already exists.
- Add tests for every new subsystem.

Main goals:
1. Replace the public two-agent framing with a multi-role architecture:
   profiler, strategist, experimenter, auditor, reporter.
2. Add execution modes: manual, guided, autonomous.
3. Add a policy-aware planning layer that chooses route, metric, split strategy, feature plan, candidate families, and stop criteria.
4. Generalize the system into routes:
   tabular_regression, tabular_classification, time_series_forecasting, time_aware_prediction, surrogate.
5. Add an autonomous loop:
   profile -> plan -> execute batch -> audit -> decide next step.
6. Add experiment registry and experiment graph artifacts.
7. Add calibration, conformal uncertainty, robustness, slice analysis, and drift reports.
8. Standardize artifacts and reports.
9. Add a unified CLI:
   run, profile, plan, experiment, audit, report, serve, replay.
10. Add a real local API and a local Streamlit operator UI.
11. Add benchmark harness and ablations.
12. Rewrite README to match the new architecture.

Required outputs per run:
- policy_resolved.yaml
- dataset_profile.json
- plan.json
- alternatives.json
- experiment_graph.json
- leaderboard.csv
- best_model/*
- reports/summary.md
- reports/technical_report.md
- reports/model_card.md
- reports/risk_report.md
- diagnostics/calibration_report.json
- diagnostics/uncertainty_report.json
- diagnostics/drift_report.json
- diagnostics/stability_report.json
- logs/agent_trace.jsonl

Implementation order:
Sprint 1:
- policy schema
- execution modes
- planner + plan.json
- profiler/strategist/experimenter
- artifact contract
- CLI compatibility shims

Sprint 2:
- experiment graph
- calibration
- conformal uncertainty
- robustness/stability/slice analysis
- richer reports

Sprint 3:
- API
- Streamlit UI
- serve mode with local monitoring
- benchmark harness

Do not add heavy dependencies to the core package unless placed behind optional extras.
```
