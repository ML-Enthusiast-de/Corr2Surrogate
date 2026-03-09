# TRANSFORMATION_PLAN.md

# Relaytic Transformation Plan

## Branding and naming

### Working brand

**Relaytic**

### Product descriptor

**The Relay Inference Lab**

### Positioning line

**Relaytic is a local-first autonomous inference engineering system for structured data.**

### Naming strategy

Use a two-layer naming system throughout the repository and docs:

- **Brand:** Relaytic
- **Descriptor:** The Relay Inference Lab

This keeps the product name short and brandable while preserving the more explanatory system description.

### Repository and package guidance

- Public repo / product name: `Relaytic`
- Product subtitle: `The Relay Inference Lab`
- Python package path may remain transitional during refactor, but the target import/package identity should align with the new brand over time.
- In README and docs, use:
  - `Relaytic`
  - `Relaytic — The Relay Inference Lab`
  - `Relaytic, a local-first autonomous inference engineering system for structured data`

### Documentation rule

All top-level docs should present the project as:

> **Relaytic — The Relay Inference Lab**

On first mention, explain that Relaytic is the product brand and The Relay Inference Lab is the descriptor for the multi-specialist local system.

---

## Goal

Transform this repository from a “data -> surrogate model” framework into a **local-first autonomous inference engineering system for structured data** that:

- runs fully locally by default
- can be steered when the user wants control
- can act autonomously when the user prefers delegation
- uses a coordinated team of specialist agents rather than a single planner
- independently inspects and challenges unknown datasets before committing to modeling assumptions
- supports multiple modeling routes, not just one narrow pipeline
- can propose additional data collection opportunities regardless of domain
- optimizes for strong real-world inference, not just benchmark score
- produces reproducible artifacts, reports, traces, and deployable outputs
- exposes itself as a reusable local tool surface for other agents
- remains deterministic, testable, auditable, and policy-governed

The repository already has strong foundations for this direction: local-first policy, explicit remote opt-in, deterministic training and evaluation, CLI entry points, artifact generation, and test/CI structure.

---

## 1. Product definition

### New product identity

The project becomes:

> A local-first autonomous inference engineering system for structured data, where multiple specialist agents independently inspect data, form and challenge hypotheses, choose workflows, run experiments, audit reliability, recommend additional data, and produce deployment-grade inference systems and artifacts.

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
     - memory budget
     - artifact size
     - approval boundaries

4. **Workflow search, not just hyperparameter search**
   - The system should choose:
     - task route
     - feature strategy
     - split strategy
     - model family
     - calibration method
     - uncertainty method
     - search budget
     - stopping rule
     - escalation path
     - data-collection recommendations

5. **Reproducibility by default**
   - Every decision, experiment, artifact, and result must be persisted.

6. **Specialist disagreement by design**
   - The system must not rely on a single planner.
   - Multiple specialist agents should independently inspect the dataset, propose competing interpretations, challenge each other’s assumptions, and converge only after evidence is collected.
   - Challenger behavior is mandatory, not optional.

7. **Inference-first**
   - The system’s goal is not merely to train a model.
   - Its goal is to produce a robust inference system with clear reliability characteristics, operational constraints, and failure boundaries.

8. **Agent interoperability**
   - The system should be usable directly by humans and consumable by other agents as a plugin/tool server.


9. **Showcaseability and operator experience**
   - The system must be easy to demo, inspect, and adopt.
   - Every major capability should have a polished local UI flow, a one-command demo path, and README-first documentation.
   - Outputs should be visually and structurally compelling enough that users and external agent systems immediately understand what the tool did and why it matters.


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

- replace the public two-agent framing with a multi-role specialist system
- make surrogate modeling one route among several
- replace ad hoc experimentation flow with an experiment graph plus run registry
- upgrade reports into full decision and evidence artifacts
- make uncertainty, calibration, robustness, drift, and abstention first-class outputs
- add independent profile generation, disagreement resolution, and challenger search
- add additional-data recommendation as a first-class subsystem
- add an interoperability layer so other agents can invoke the system as a local tool

---

## 3. Target architecture

### Specialist roles

Replace the current public framing with a specialist team.

#### 3.1 Scout
Responsible for:
- independent dataset inspection
- schema inference
- hidden key detection
- leakage pattern discovery
- time/entity/group detection
- missingness profiling
- suspicious-column surfacing
- dataset weirdness logging

#### 3.2 Scientist
Responsible for:
- task framing
- route hypotheses
- target ambiguity resolution
- feature hypotheses
- split hypotheses
- additional-data hypotheses
- missing-signal hypotheses
- problem restatement in operational terms

#### 3.3 Strategist
Responsible for:
- choosing routes under policy
- selecting metrics
- selecting split strategies
- composing workflow plans
- assigning budget
- selecting search controller behavior
- deciding whether more experimentation is worth it
- deciding whether success criteria are met

#### 3.4 Builder
Responsible for:
- executing workflow candidates
- tracking trials
- early-stopping weak branches
- managing search budgets
- collecting metrics
- persisting artifacts

#### 3.5 Challenger
Responsible for:
- beating the incumbent plan/model
- proposing alternative assumptions
- testing anti-leakage variants
- testing different split semantics
- testing reduced-feature and robustness variants
- surfacing whether the current winner is fragile

#### 3.6 Auditor
Responsible for:
- calibration
- uncertainty
- slice analysis
- robustness checks
- drift and OOD checks
- repeatability validation
- abstention/defer policy evaluation
- model card generation
- risk report generation

#### 3.7 Synthesizer
Responsible for:
- comparing competing interpretations
- integrating evidence across specialists
- writing final recommendation rationale
- producing a machine-readable recommendation bundle

#### 3.8 Broker
Responsible for:
- exposing the system as local tools for external agents
- serving MCP-compatible tool interfaces
- exposing machine-readable capability manifests
- packaging runs for consumption by other systems

---

## 4. Package structure

Refactor toward this structure:

```text
src/relay_inference_lab/
  agents/
    scout.py
    scientist.py
    strategist.py
    builder.py
    challenger.py
    auditor.py
    synthesizer.py
    broker.py
    coordinator.py
    schemas.py
    policies.py
    handoffs.py
    disagreements.py
  autonomy/
    controller.py
    stop_rules.py
    approval.py
    budgets.py
    success_criteria.py
    branch_promotion.py
  datasets/
    schema.py
    profiling.py
    quality.py
    leakage.py
    slicing.py
    time_semantics.py
    entity_detection.py
    target_semantics.py
    additional_data.py
  planning/
    task_inference.py
    route_selection.py
    metric_selection.py
    split_selection.py
    search_space.py
    workflow_graph.py
    hypotheses.py
    portfolio_warmstart.py
    pareto.py
  experiments/
    registry.py
    tracker.py
    execution.py
    leaderboard.py
    replay.py
    multi_fidelity.py
    challenger_runs.py
  features/
    tabular_basic.py
    tabular_interactions.py
    time_lags.py
    time_windows.py
    encoders.py
    selectors.py
    feature_hypotheses.py
  modeling/
    families/
      linear.py
      tree.py
      ensemble.py
      baseline.py
      tabpfn_backend.py
    calibration.py
    uncertainty.py
    robustness.py
    abstention.py
    training.py
    inference.py
  evaluation/
    metrics.py
    diagnostics.py
    drift.py
    ood.py
    conformal.py
    stability.py
    calibration_plots.py
    conditional_coverage.py
  interoperability/
    api_tools.py
    mcp_server.py
    capability_manifest.py
    artifact_exports.py
  observability/
    traces.py
    spans.py
    telemetry.py
    handoff_graph.py
  artifacts/
    manifests.py
    model_card.py
    reports.py
    packaging.py
  serve/
    app.py
    schemas.py
    monitoring.py
    abstention.py
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
The system analyzes, proposes, and audits.
No experiment runs without explicit approval.

### 5.2 Guided mode
The system proposes a plan and asks for approval at checkpoints:
- before search
- before expensive branches
- before external optional backends
- before final recommendation
- before packaging or serving

### 5.3 Autonomous mode
The system may:
- inspect data independently with multiple specialists
- propose competing interpretations
- run experiments
- expand or prune search
- promote promising branches
- run challenger branches
- calibrate and uncertainty-wrap top candidates
- recommend additional data
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
    allow_optional_remote_baselines: false
  autonomy:
    execution_mode: autonomous
    allow_auto_run: true
    approval_required_for_expensive_runs: true
    approval_required_for_optional_backends: true
  compute:
    max_wall_clock_minutes: 120
    max_trials: 200
    max_parallel_trials: 4
    max_memory_gb: 24
    multi_fidelity_enabled: true
  optimization:
    objective: best_robust_pareto_front
    primary_metric: auto
    secondary_metrics: [calibration, stability, latency]
    pareto_selection: true
  constraints:
    latency_ms_max: null
    interpretability: medium
    uncertainty_required: true
    reproducibility_required: true
    abstention_allowed: true
  safety:
    strict_leakage_checks: true
    strict_time_ordering: true
    reject_unstable_models: true
    require_disagreement_resolution: true
  interoperability:
    enable_mcp_server: true
    expose_tool_contracts: true
  reporting:
    create_model_card: true
    create_risk_report: true
    create_experiment_graph: true
    create_handoff_graph: true
    create_data_recommendations: true
```

### Acceptance criteria

- policy loads from YAML and can be overridden by CLI flags
- every experiment stores the resolved effective policy
- planner decisions cite the policy fields that influenced them
- optional frontier backends are policy-gated
- approval boundaries are explicit and logged

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

Scout + Scientist + Strategist decide route using:
- target type
- timestamp presence
- group or entity columns
- lag structure
- dataset size
- missingness
- target semantics
- suspicious post-target columns
- user hints

### Acceptance criteria

- route is inferred automatically when possible
- user can override route explicitly
- route selection is written to `plan.json`
- rejected route rationales are persisted
- disagreements about route are logged explicitly

---

## 8. Independent data understanding

Before major experiment search begins, at least two specialists must independently inspect the dataset.

### Required independent outputs

Each independent profile must include:
- dataset interpretation
- inferred task framing
- likely leakage risks
- likely causal or temporal risks
- feature-type corrections
- suspicious columns
- target alternatives if target ambiguity exists
- required missing-data interventions
- additional-data suggestions
- confidence level

### Required artifacts

```text
independent_profiles/
  profile_a.json
  profile_b.json
  disagreement_report.json
  resolved_understanding.json
```

### Acceptance criteria

- two independent profiles are created in guided/autonomous modes
- disagreements are compared before planning
- the final resolved understanding cites both agreements and conflicts
- no final recommendation can be produced without disagreement resolution when policy requires it

---

## 9. Planning system

Introduce a real plan object plus explicit hypotheses.

### New artifacts

- `plan.json`
- `alternatives.json`
- `hypotheses.json`
- `rejected_routes.json`

### Example `plan.json`

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
  "calibration_plan": "temperature_or_isotonic_if_needed",
  "uncertainty_plan": "conformal_or_cqr_if_supported",
  "search_controller": {
    "type": "multi_fidelity_portfolio_search",
    "warmstart": true,
    "pareto_selection": true
  },
  "stop_criteria": {
    "max_trials": 60,
    "min_improvement_delta": 0.002,
    "patience": 12
  }
}
```

### Example `hypotheses.json`

Each hypothesis must include:
- id
- claim
- rationale
- supporting evidence
- expected effect
- experimental plan
- success/failure criterion
- estimated cost
- responsible agent
- challenger counterpart if applicable

### Planner requirements

Planner must produce:
- one primary plan
- at least two alternative plans
- explicit rationale for why each plan exists
- estimated cost per plan
- notes for rejected routes
- at least one challenger plan
- at least one additional-data hypothesis when relevant
- at least one leakage hypothesis when relevant
- at least one split-validity hypothesis when relevant

---

## 10. Advanced search controller

Search must feel like a real autonomous optimization system, not a simple grid/random wrapper.

### Required search features

- multi-fidelity pruning
- Successive Halving / ASHA-style branch pruning
- portfolio warm-starts
- branch promotion
- compute-aware scheduling
- per-branch stop criteria
- incumbent vs challenger tracking
- Pareto-front model selection across:
  - score
  - calibration
  - stability
  - latency
  - artifact size

### Acceptance criteria

- weak branches can be pruned before full evaluation
- promising branches can be promoted to deeper search
- warm-start workflows are chosen from dataset archetypes
- a Pareto frontier is materialized as an artifact
- challenger branches are visible in the experiment graph

---

## 11. Autonomous experimentation loop

This is the core change.

### New orchestration loop

```text
independent inspect -> resolve understanding -> hypothesize -> plan -> execute batch -> challenge -> audit -> decide next step
```

### Required loop decisions

After each batch, the Strategist or Synthesizer decides one of:

- `expand_search`
- `refine_promising_branch`
- `promote_branch`
- `queue_challenger_branch`
- `calibrate_top_models`
- `run_uncertainty_wrap`
- `run_robustness_checks`
- `request_more_data`
- `stop_and_report`

### Stop conditions

Support these stop rules:

- budget exhausted
- no meaningful improvement for N trials
- robust winner found
- policy constraints satisfied and confidence threshold met
- user interruption
- challenger fails to materially improve incumbent after defined attempts

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
    abstention_cost_limit: 0.1
```

The system should stop only when:
- the selected model beats the baseline
- stability is acceptable
- calibration and uncertainty criteria are satisfied when required
- no major leakage or catastrophic failure remains
- a challenger has failed to reveal a materially better or safer alternative

---

## 12. Experiment graph and handoff graph

Add structured graphs instead of flat runs.

### New artifacts

- `experiment_graph.json`
- `handoff_graph.json`
- `pareto_front.json`

### Experiment graph nodes may represent

- split strategy
- feature recipe
- model family
- hyperparameter config
- calibration config
- uncertainty config
- robustness test
- challenger branch
- final selection decision

### Experiment graph edges may represent

- expanded from
- pruned because
- promoted because
- challenged because
- calibrated because
- rejected because

### Handoff graph must represent

- which specialist produced which action
- which agent handed off to which other agent
- what evidence triggered the handoff
- where disagreement occurred
- where fallback logic was used

### Acceptance criteria

- every run has an experiment graph
- every run has a handoff graph
- graphs can be rendered as JSON and markdown
- any final result can be traced back to the decisions and handoffs that produced it

---

## 13. Model family system

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

### Frontier optional backend layer
Keep the core dependency-light.
Optional extras may expose:
- TabPFN backend for small-to-medium structured data
- benchmark reference runners for AutoGluon
- benchmark reference runners for FLAML
- benchmark reference runners for auto-sklearn
- surrogate-specific libraries
- optional heavier local libraries

### Rule

Optional frontier backends must never become required for core offline operation.

---

## 14. Features system

Feature generation must become route-aware, policy-aware, and hypothesis-aware.

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

#### Hypothesis-driven feature modules
- feature ablations
- suspicious-feature exclusion
- proxy-leakage exclusion
- challenger feature recipes
- minimal viable feature sets

### Rules

- feature generation must be deterministic
- all generated features must be logged to `feature_manifest.json`
- feature selection decisions must be explainable
- challenger branches must be able to exclude suspicious features deliberately

---

## 15. Evaluation redesign

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
- temperature scaling result where applicable

#### Stability
- variance across seeds or folds
- sensitivity to row perturbation
- sensitivity to missingness perturbation

#### Robustness
- slice-level performance
- degradation under noise
- degradation under reduced features
- challenger-vs-incumbent deltas

#### Uncertainty
- conformal intervals or sets where supported
- conformalized quantile regression when applicable
- empirical coverage
- interval width statistics
- conditional coverage by slice

#### Abstention / defer
- abstention threshold candidates
- retained-coverage tradeoff
- abstention cost estimate

#### Operational
- fit time
- predict time
- artifact size
- memory estimate

---

## 16. Uncertainty, calibration, and abstention

Add this as a first-class subsystem.

### Minimum implementation

#### Classification
- probability calibration
  - temperature scaling
  - Platt scaling
  - isotonic when justified
- reliability report
- abstention policy support

#### Regression and forecasting
- split conformal prediction intervals
- conformalized quantile regression where supported
- coverage report
- width-vs-coverage diagnostics

### New artifacts

- `uncertainty_report.json`
- `calibration_report.json`
- `abstention_report.json`

### Acceptance criteria

- any model marked `recommended` must include uncertainty and calibration outputs when policy requires them
- any abstaining model must include abstention policy outputs
- the final report must state whether uncertainty requirements were satisfied
- conditional coverage by slice must be available when feasible

---

## 17. Drift, OOD, monitoring, and observability

Make monitoring structured and visible.

### Monitoring modules

- feature drift
- target drift when labels are available
- schema drift
- missingness drift
- simple built-in OOD score for tabular inputs
- time distribution drift

### Optional monitoring integrations

- Evidently integration
- Alibi Detect integration
- whylogs integration
- OpenTelemetry traces/logs/metrics

### Required artifacts

- `drift_report.json`
- `telemetry_summary.json`
- `fallback_events.json`

### Serving integration

When local serving is enabled, log:
- schema mismatches
- missing features
- OOD score summary
- drift snapshots
- abstention events
- model confidence / interval summary

### Agent observability

Every specialist action, handoff, disagreement, retry, fallback, approval gate, and branch decision must be traced.

Required outputs:
- agent span traces
- handoff graph
- structured event log
- fallback events
- challenger win rate
- autonomy efficiency metrics

---

## 18. Additional data recommendation

The system must be able to propose additional data collection opportunities regardless of domain.

### Required behaviors

- identify likely missing explanatory factors
- identify weakly observed entities or slices
- suggest temporal extension when forecast uncertainty is too high
- suggest external covariates when residual structure indicates missing signal
- distinguish between:
  - must-have additional data
  - high-value optional data
  - speculative exploratory data

### Required artifact

- `data_recommendations.json`

### Each recommendation must include

- proposed data
- why it likely matters
- which failure mode it addresses
- expected impact
- confidence level
- collection difficulty estimate
- evidence source
- responsible specialist

---

## 19. Artifacts contract

Standardize the output layout.

### New run directory layout

```text
artifacts/run_<timestamp>/
  manifest.json
  policy_resolved.yaml
  dataset_profile.json
  independent_profiles/
    profile_a.json
    profile_b.json
    disagreement_report.json
    resolved_understanding.json
  plan.json
  alternatives.json
  hypotheses.json
  rejected_routes.json
  experiment_graph.json
  handoff_graph.json
  pareto_front.json
  leaderboard.csv
  data_recommendations.json
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
    interoperability_report.md
  diagnostics/
    calibration_report.json
    uncertainty_report.json
    abstention_report.json
    drift_report.json
    stability_report.json
    slice_report.json
  logs/
    agent_trace.jsonl
    telemetry_summary.json
    fallback_events.json
    execution.log
```

### Rule

Every emitted file must be listed in `manifest.json`.

---

## 20. Reporting redesign

Make reports layered.

### Required reports

#### `summary.md`
Contains:
- what the system found
- the recommended model
- why it was chosen
- confidence in the recommendation
- whether more search is likely to help
- whether more data is likely to help

#### `technical_report.md`
Contains:
- dataset profile
- independent interpretations
- disagreement resolution
- route
- split choice
- search path
- model comparison
- calibration and uncertainty
- robustness findings
- challenger findings

#### `model_card.md`
Contains:
- intended use
- limitations
- training setup
- metrics
- risks
- abstention behavior if applicable

#### `risk_report.md`
Contains:
- leakage risk
- instability
- drift sensitivity
- missingness sensitivity
- unsupported use cases
- residual uncertainty

#### `interoperability_report.md`
Contains:
- available tool endpoints
- artifact export structure
- machine-readable actions
- MCP/tool compatibility notes

---

## 21. CLI redesign

Preserve current commands, but add a unified surface.

### New primary CLI

```bash
corr2surrogate run   --data-path data.csv   --target target   --execution-mode autonomous   --policy configs/policies/default.yaml   --goal best_robust_pareto_front
```

### Required subcommands

```bash
relay-inference-lab inspect
relay-inference-lab hypothesize
relay-inference-lab plan
relay-inference-lab experiment
relay-inference-lab challenge
relay-inference-lab audit
relay-inference-lab report
relay-inference-lab serve
relay-inference-lab replay
relay-inference-lab tool-server
```

### Backward compatibility

Keep current commands working for one transition cycle and mark them as legacy in help text.

---

## 22. API redesign

Turn the current API entrypoint into a full local API surface.

### Endpoints

- `POST /inspect`
- `POST /hypothesize`
- `POST /plan`
- `POST /run`
- `POST /challenge`
- `POST /audit`
- `POST /report`
- `POST /predict`
- `POST /recommend-data`
- `GET /runs/{run_id}`
- `GET /runs/{run_id}/artifacts`
- `GET /health`

### Requirements

- local only by default
- no remote dependencies required
- OpenAPI schema must reflect execution mode and policy objects
- machine-readable action schemas must be exposed
- API responses must reference artifact locations when applicable

---

## 23. Agent interoperability layer

The system must expose itself as a reusable tool surface for other agents.

### Required surfaces

- local HTTP API
- schema-defined tool actions
- MCP-compatible server mode
- machine-readable artifact manifests
- tool descriptions for external agent frameworks

### Required exported capabilities

- `inspect_dataset`
- `propose_hypotheses`
- `generate_plan`
- `run_experiments`
- `run_challenger_branch`
- `audit_model`
- `compare_models`
- `recommend_additional_data`
- `package_model`
- `summarize_run`

### Acceptance criteria

- another local agent can call the system through a stable tool contract
- capability manifests are emitted automatically
- tool outputs are schema-validated
- external agents can consume artifacts without reading ad hoc logs

---

## 24. Streamlit interface

Add a local UI as an operator console.

### Screens

- upload / dataset selection
- independent profile comparison
- hypothesis view
- plan view
- experiment progress
- challenger comparison
- leaderboard
- audit view
- final report view
- serve / inference view
- tool-server status view

### Behavior

- user can inspect and override the plan
- user can switch between guided and autonomous mode
- user can stop runs safely
- user can compare top models visually
- user can inspect disagreements between specialists
- user can inspect additional-data suggestions
- user can inspect handoff traces
- user can launch golden demos directly from the UI
- user can export a showcase-ready report bundle
- user can inspect tool-server status and external-agent calls

---


## 24A. Demo, showcase, and adoption layer

The system must be easy to showcase locally and easy for new users to adopt.

### Required demos

- **Golden tabular demo**
  - one public dataset
  - full inspect -> hypothesize -> plan -> experiment -> challenge -> audit -> report flow
  - polished artifacts committed or reproducibly generated

- **Golden time-series demo**
  - one public time-series dataset
  - temporal route selection
  - challenger branch and uncertainty outputs

- **Plugin / tool-server demo**
  - one example where another local agent or script invokes Relay Inference Lab through the tool server or API
  - returns machine-readable artifacts and a final recommendation

### Required showcase assets

- architecture diagram
- specialist handoff diagram
- screenshot set or GIFs of the UI
- sample report bundle
- example artifact tree
- benchmark summary figure
- one “why the model was chosen” visual

### Acceptance criteria

- a new user can run one command and reproduce a polished demo
- the README links directly to demo commands and expected outputs
- the Streamlit/operator UI can be used live for a walkthrough
- at least one demo shows external-agent interoperability

---

## 25. LLM and agent contract redesign

Make the agent layer stricter, schema-driven, and optional.

### Required changes

#### Structured action schemas
Each agent emits one structured action validated against Pydantic models.

#### Example action types
- `inspect_dataset`
- `flag_suspicious_column`
- `choose_route`
- `choose_split`
- `propose_hypothesis`
- `queue_experiments`
- `request_calibration`
- `request_uncertainty_wrap`
- `request_robustness_check`
- `request_additional_data`
- `challenge_incumbent`
- `stop_with_recommendation`

#### Refusal and fallback behavior
If an agent output is invalid:
- retry once
- fall back to deterministic planner logic
- log the failure

### Important rule

The system must remain fully functional without any LLM.
Local LLM assistance is an enhancement, not a hard dependency.

### Local LLM enhancement path

When a local LLM is available, it may improve:
- dataset interpretation quality
- hypothesis generation
- route selection explanation
- additional-data recommendations
- report quality
- challenger creativity

But the system must still:
- validate every action
- log every action
- fall back safely

---

## 26. Benchmark harness

Add a benchmark package.

### New directory

```text
bench/
  configs/
  datasets/
  runners/
  reports/
  baselines/
```

### Benchmark tracks

#### Track 1: tabular classification and regression
Use OpenML benchmark suites commonly used for structured-data evaluation.

#### Track 2: time-series classification
Use UEA/UCR-style benchmark collections.

#### Track 3: forecasting
Use M4-style and broader forecasting-archive evaluation routes.

### Benchmark baselines

Include benchmark runners for:
- internal core system
- optional TabPFN backend
- optional AutoGluon runner
- optional FLAML runner
- optional auto-sklearn runner

### Benchmark outputs

- `benchmark_results.csv`
- `benchmark_summary.md`
- `ablations.csv`

### Required ablations

- no multi-agent disagreement vs multi-agent disagreement
- no feature engineering vs feature engineering
- no uncertainty vs uncertainty-aware selection
- split conformal vs CQR
- minimal search vs autonomous search
- no challenger vs challenger enabled
- strict policy vs permissive policy
- no additional-data recommender vs recommender enabled
- no warm-start vs warm-start
- no multi-fidelity vs multi-fidelity

---

## 27. Testing expansion

Extend test coverage around the new architecture.

### Unit tests
- route inference
- split selection
- policy resolution
- stop rules
- artifact manifests
- calibration utilities
- conformal utilities
- CQR utilities
- disagreement resolution
- handoff schemas

### Integration tests
- inspect -> hypothesize
- hypothesize -> plan
- plan -> execute
- execute -> challenge
- challenge -> audit
- audit -> report
- tool-server capability exposure

### End-to-end tests
- one small tabular dataset
- one small time-series dataset
- one surrogate-route dataset
- one agent-interoperability tool invocation

### Failure tests
- malformed schema
- bad target selection
- no valid model family
- invalid agent output
- budget exhaustion
- user stop mid-run
- challenger failure
- missing optional backend
- conflict between specialists

### Regression tests
Golden artifact snapshots for:
- `plan.json`
- `hypotheses.json`
- `leaderboard.csv`
- `summary.md`
- `data_recommendations.json`

---

## 28. CI and repo health

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
- optional backend smoke test
- tool-server contract test

### Repo files to add

- `SECURITY.md`
- `CHANGELOG.md`
- `CODE_OF_CONDUCT.md`

### Optional repo-health upgrades

- SBOM generation
- provenance/build metadata
- benchmark result publishing
- demo artifact publishing

---

## 29. Migration plan

### Phase 1: architecture foundations
Implement first:
1. policy schema
2. execution modes
3. independent inspect and disagreement artifacts
4. plan object
5. experiment registry
6. artifact contract
7. compatibility shims

### Phase 2: specialist upgrade
Implement:
1. scout
2. scientist
3. strategist
4. builder
5. challenger
6. coordinator loop
7. schema-validated actions

### Phase 3: evaluation and inference reliability
Implement:
1. calibration
2. conformal uncertainty
3. CQR
4. stability checks
5. slice analysis
6. drift report
7. abstention policy outputs

### Phase 4: product and interoperability surfaces
Implement:
1. unified CLI
2. local API
3. MCP/tool server
4. Streamlit app
5. local serve mode

### Phase 5: benchmark and frontier polish
Implement:
1. benchmark harness
2. ablations
3. optional frontier backends
4. docs rewrite
5. end-to-end demos

---

## 30. Implementation rules for Codex

1. Do not remove local-first defaults.
2. Do not make remote APIs required.
3. Preserve existing commands where practical using compatibility wrappers.
4. Prefer additive refactors over destructive rewrites.
5. Keep deterministic behavior where already present.
6. Add tests with each major subsystem.
7. Persist every important decision as an artifact.
8. Do not introduce heavyweight dependencies into the core package unless behind optional extras.
9. Maintain clean separation between:
   - inspection
   - hypotheses
   - planning
   - execution
   - challenge
   - evaluation
   - reporting
   - interoperability
10. The system must work without LLMs.
11. The system must support local LLM enhancement without becoming LLM-dependent.
12. Challenger behavior is required in autonomous mode.
13. Additional-data recommendation is required in guided/autonomous modes.
14. Every important handoff must be logged.

### Required deliverables from Codex

- refactored package structure
- new policy system
- new execution mode system
- independent inspect/disagreement layer
- new plan object and planner
- hypothesis subsystem
- experiment graph and handoff graph
- challenger subsystem
- uncertainty, calibration, abstention, and CQR modules
- additional-data recommendation subsystem
- standardized artifacts
- unified CLI
- local API and tool-server
- expanded tests
- rewritten README and docs
- branding integrated across README, CLI help text, UI copy, and demo assets
- one complete local demo path
- one polished tabular golden demo
- one polished time-series golden demo
- one external-agent integration demo path
- showcase screenshots/GIFs and demo assets

---

## 31. README rewrite requirements

The README must present the system as:

- local-first
- autonomous but steerable
- multi-specialist
- hypothesis-driven
- inference-first
- policy-aware
- artifact-rich
- consumable by other agents

### Required README sections

1. What Relaytic is
2. Why local-first
3. Why multiple specialists
4. Execution modes
5. Supported routes
6. Hypothesis-driven workflow
7. Quickstart
8. Golden demos
9. Example outputs
10. Artifact structure
11. Policy examples
12. Tool-server / plugin usage
13. Benchmarks
14. Development and tests

### README quality requirements

- open with a one-paragraph product identity statement using the format: `Relaytic — The Relay Inference Lab`
- include a fast 60-second quickstart
- include one copy-paste demo command
- include screenshots or GIFs of the UI
- include a sample artifact tree
- include a clear explanation of how external agents can use the tool server
- include benchmark and ablation highlights
- include “why this is different from plain AutoML”

---

## 32. Definition of done

The transformation is complete when all of the following are true:

1. A user can run one command locally on a CSV and get:
   - dataset profile
   - independent specialist interpretations
   - hypotheses
   - plan
   - experiments
   - challenger comparison
   - leaderboard
   - calibrated and uncertainty-aware best model
   - additional-data recommendations
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
   - independent profile artifacts
   - `plan.json`
   - `hypotheses.json`
   - `experiment_graph.json`
   - `handoff_graph.json`
   - `leaderboard.csv`
   - summary and technical reports
   - uncertainty, calibration, abstention, and drift artifacts
   - `data_recommendations.json`

5. The system works fully offline by default.

6. Tests cover the new inspect, planner, builder, challenger, and auditor flow.

7. Existing deterministic behavior remains intact where applicable.

8. Another local agent can invoke the system through a stable tool contract.
9. The repository includes at least:
   - one polished tabular golden demo
   - one polished time-series golden demo
   - one external-agent / tool-server demo
10. The README and UI are strong enough to support a live walkthrough without extra explanation.

---

## 33. First implementation sprint

### Sprint 1
- add policy schema
- add execution mode support
- create independent inspect outputs
- create `plan.json`
- create `hypotheses.json`
- create experiment registry
- refactor current analyst/modeler flow into `scout` + `scientist` + `strategist` + `builder`
- standardize artifact directory
- add compatibility shims
- update CLI with `inspect`, `hypothesize`, `plan`

### Sprint 2
- add experiment graph
- add handoff graph
- add challenger subsystem
- add calibration
- add conformal uncertainty
- add CQR
- add stability checks
- add `summary.md`, `technical_report.md`, `model_card.md`, `risk_report.md`
- add `data_recommendations.json`

### Sprint 3
- add API
- add MCP/tool server
- add Streamlit app
- add serve mode and local monitoring
- add benchmark harness
- add optional frontier backends
- add golden demos
- add showcase assets
- rewrite README for fast onboarding and live demo use
- integrate Relaytic branding across docs, UI, CLI, and demo assets

---

## 34. Codex prompt to paste

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
- Keep the core dependency-light.
- Put frontier backends behind optional extras.

Main goals:
1. Replace the public two-agent framing with a multi-specialist architecture:
   scout, scientist, strategist, builder, challenger, auditor, synthesizer, broker.
2. Add execution modes: manual, guided, autonomous.
3. Add independent dataset inspection and disagreement resolution before planning.
4. Add a policy-aware planning layer that chooses route, metric, split strategy, feature plan, candidate families, uncertainty plan, and stop criteria.
5. Generalize the system into routes:
   tabular_regression, tabular_classification, time_series_forecasting, time_aware_prediction, surrogate.
6. Add a hypothesis-driven autonomous loop:
   inspect -> resolve understanding -> hypothesize -> plan -> execute batch -> challenge -> audit -> decide next step.
7. Add multi-fidelity search, branch promotion, challenger branches, and Pareto-front selection.
8. Add experiment graph and handoff graph artifacts.
9. Add calibration, conformal uncertainty, CQR, abstention, robustness, slice analysis, and drift reports.
10. Add additional-data recommendation as a first-class artifact.
11. Standardize artifacts and reports.
12. Add a unified CLI:
   inspect, hypothesize, plan, experiment, challenge, audit, report, serve, replay, tool-server.
13. Add a real local API and a local MCP/tool server.
14. Add a local Streamlit operator UI.
15. Add benchmark harness and ablations.
16. Rewrite README to match the new architecture.
17. Integrate the Relaytic brand and the descriptor `The Relay Inference Lab` consistently across docs, CLI help, UI copy, tool-server descriptions, and demo assets.

Required outputs per run:
- policy_resolved.yaml
- dataset_profile.json
- independent_profiles/*
- plan.json
- alternatives.json
- hypotheses.json
- experiment_graph.json
- handoff_graph.json
- pareto_front.json
- leaderboard.csv
- data_recommendations.json
- best_model/*
- reports/summary.md
- reports/technical_report.md
- reports/model_card.md
- reports/risk_report.md
- reports/interoperability_report.md
- diagnostics/calibration_report.json
- diagnostics/uncertainty_report.json
- diagnostics/abstention_report.json
- diagnostics/drift_report.json
- diagnostics/stability_report.json
- logs/agent_trace.jsonl
- logs/telemetry_summary.json
- logs/fallback_events.json

Implementation order:
Sprint 1:
- policy schema
- execution modes
- independent inspect/disagreement layer
- planner + plan.json
- hypotheses subsystem
- scout/scientist/strategist/builder
- artifact contract
- CLI compatibility shims

Sprint 2:
- experiment graph
- handoff graph
- challenger subsystem
- calibration
- conformal uncertainty
- CQR
- robustness/stability/slice analysis
- additional-data recommendations
- richer reports

Sprint 3:
- API
- MCP/tool server
- Streamlit UI
- serve mode with local monitoring
- benchmark harness
- optional frontier backends

Do not add heavy dependencies to the core package unless placed behind optional extras.
```
