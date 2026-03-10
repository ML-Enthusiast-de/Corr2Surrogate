# RELAYTIC_TRANSFORMATION_PLAN.md

# Relaytic Transformation Plan

## Branding and naming

### Product brand

**Relaytic**

### Product descriptor

**The Relay Inference Lab**

### Positioning line

**Relaytic is a local-first inference engineering system for structured data, built around specialist agents that investigate data, form competing hypotheses, run challenger science, and expose their judgment as reusable tools.**

### Naming strategy

Use a two-layer naming system throughout the repository and docs:

- **Brand:** Relaytic
- **Descriptor:** The Relay Inference Lab

This keeps the product name short and ownable while preserving the explanatory system description.

### Documentation rule

All top-level docs should present the project as:

> **Relaytic — The Relay Inference Lab**

On first mention, explain that Relaytic is the product brand and The Relay Inference Lab is the descriptor for the multi-specialist local system.

---

## Goal

Transform this repository from a “data -> surrogate model” framework into a **local-first inference engineering system** for structured data that:

- runs fully locally by default
- can be steered when the user wants control
- can act autonomously when the user prefers delegation
- uses a coordinated team of specialist agents rather than a single planner
- investigates unknown datasets before committing to modeling assumptions
- supports multiple modeling routes, not just one narrow pipeline
- can recommend additional data collection opportunities regardless of domain
- optimizes for robust inference, not merely leaderboard score
- produces reproducible artifacts, reports, traces, and deployable outputs
- learns from prior runs and uses memory to improve future planning
- exposes itself as a reusable tool surface for other agents
- remains deterministic, testable, auditable, and policy-governed

The repository already has strong foundations for this direction: local-first policy, explicit remote opt-in, deterministic training and evaluation, CLI entry points, artifact generation, and test/CI structure.

---

## 1. Product definition

### New product identity

Relaytic becomes:

> A local-first inference engineering system for structured data, where multiple specialist agents independently inspect data, form and challenge hypotheses, run evidence-driven experiments, quantify reliability, recommend additional data, and produce reusable inference systems and artifacts.

### What Relaytic is not

Relaytic is **not**:
- a thin AutoML wrapper
- a generic “chat with your CSV” app
- a model zoo launcher
- a benchmark-only leaderboard generator

Relaytic is:
- an investigator of unknown structured datasets
- a builder of robust inference systems
- a generator of evidence, not just metrics
- a local tool that other agents can call when they need judgment about data, modeling, reliability, or missing information

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
     - split semantics
     - feature strategy
     - model family
     - calibration method
     - uncertainty method
     - abstention/defer policy
     - search budget
     - stopping rule
     - data-collection recommendations

5. **Reproducibility by default**
   - Every decision, experiment, artifact, and result must be persisted.

6. **Specialist disagreement by design**
   - The system must not rely on a single planner.
   - Multiple specialist agents should independently inspect the dataset, propose competing interpretations, challenge each other’s assumptions, and converge only after evidence is collected.
   - Challenger behavior is mandatory, not optional.

7. **Evidence over vibes**
   - Every major recommendation must be backed by:
     - experiments
     - ablations
     - uncertainty analysis
     - challenger results
     - explicit rationale

8. **Inference-first**
   - The goal is not merely to train a model.
   - The goal is to produce a robust inference system with clear reliability characteristics, operating boundaries, and failure modes.

9. **Memory and priors**
   - The system should learn from prior runs.
   - Planning should improve over time through retrieved priors, plan archetypes, and remembered failure modes.

10. **Agent interoperability**
   - The system should be usable directly by humans and consumable by other agents as a plugin/tool server.

11. **Showcaseability and operator experience**
   - Every major capability should have:
     - a polished local UI flow
     - a one-command demo path
     - README-first documentation
     - inspectable artifacts
   - Outputs should be strong enough that users and external agent systems immediately understand what the tool did and why it matters.

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
- replace ad hoc experimentation flow with:
  - an investigation phase
  - a hypothesis phase
  - an experiment graph
  - a handoff graph
  - a run memory layer
- upgrade reports into full decision and evidence artifacts
- make uncertainty, calibration, robustness, drift, and abstention first-class outputs
- add independent profile generation, disagreement resolution, and challenger science
- add additional-data recommendation and value-of-information estimation
- add a judgment-oriented interoperability layer so other agents can invoke the system as a local tool
- add learned-prior route selection and frontier backends

---

## 3. Innovation thesis

Relaytic should stand out because it combines five ideas that most projects do not combine well:

### 3.1 Specialist disagreement before model search
At least two specialists independently inspect the data and argue about task framing, leakage, split semantics, and missing signal before the main search begins.

### 3.2 Challenger science, not just challenger models
The system must challenge not only the incumbent model, but also:
- the split strategy
- the route
- suspicious features
- temporal assumptions
- low-data robustness
- missingness sensitivity

### 3.3 Additional data planning as a first-class output
The system should not merely say “best model found.”
It should also say:
- what data is missing
- why it matters
- what collection plan is likely to reduce uncertainty most
- whether more data is better than more search

### 3.4 Memory-guided planning
The system should learn from prior runs by storing:
- dataset fingerprints
- winning plan archetypes
- failure archetypes
- challenger win patterns
- calibration/drift failure patterns

### 3.5 Judgment tools for other agents
Relaytic should expose high-level tools like:
- `investigate_dataset`
- `challenge_incumbent_model`
- `generate_data_collection_plan`
- `produce_decision_memo`

This makes it useful as a plugin for OpenClaw-style systems and other agent frameworks.

---

## 4. Target architecture

### Specialist roles

Replace the public framing with a specialist team.

#### 4.1 Scout
Responsible for:
- independent dataset inspection
- schema inference
- hidden key detection
- leakage pattern discovery
- time/entity/group detection
- missingness profiling
- suspicious-column surfacing
- dataset weirdness logging

#### 4.2 Scientist
Responsible for:
- task framing
- route hypotheses
- target ambiguity resolution
- feature hypotheses
- split hypotheses
- additional-data hypotheses
- missing-signal hypotheses
- problem restatement in operational terms

#### 4.3 Strategist
Responsible for:
- choosing routes under policy
- selecting metrics
- selecting split strategies
- composing workflow plans
- assigning budget
- selecting search-controller behavior
- deciding whether more experimentation is worth it
- deciding whether success criteria are met

#### 4.4 Builder
Responsible for:
- executing workflow candidates
- tracking trials
- early-stopping weak branches
- managing search budgets
- collecting metrics
- persisting artifacts

#### 4.5 Challenger
Responsible for:
- beating the incumbent plan/model
- proposing alternative assumptions
- testing anti-leakage variants
- testing different split semantics
- testing reduced-feature and robustness variants
- surfacing whether the current winner is fragile

#### 4.6 Ablation Judge
Responsible for:
- structured ablations
- fragility analysis
- identifying which assumptions actually drove performance
- separating robust gains from inflated gains
- generating belief updates from ablation evidence

#### 4.7 Auditor
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

#### 4.8 Synthesizer
Responsible for:
- comparing competing interpretations
- integrating evidence across specialists
- writing final recommendation rationale
- producing a machine-readable recommendation bundle
- deciding whether the answer is:
  - ship model
  - collect more data
  - continue search
  - reject deployment

#### 4.9 Memory Keeper
Responsible for:
- storing run fingerprints
- retrieving similar prior runs
- proposing plan archetypes
- remembering failure archetypes
- improving warm starts over time

#### 4.10 Broker
Responsible for:
- exposing the system as local tools for external agents
- serving MCP-compatible tool interfaces
- exposing machine-readable capability manifests
- packaging runs for consumption by other systems

---

## 5. Package structure

Refactor toward this structure:

```text
src/relaytic/
  agents/
    scout.py
    scientist.py
    strategist.py
    builder.py
    challenger.py
    ablation_judge.py
    auditor.py
    synthesizer.py
    memory_keeper.py
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
    unknown_domain.py
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
    plan_archetypes.py
  experiments/
    registry.py
    tracker.py
    execution.py
    leaderboard.py
    replay.py
    multi_fidelity.py
    challenger_runs.py
    ablations.py
  memory/
    fingerprints.py
    retrieval.py
    run_memory.py
    failure_memory.py
    priors.py
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
    value_of_information.py
  interoperability/
    api_tools.py
    mcp_server.py
    capability_manifest.py
    tool_cards.py
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

## 6. Execution modes

Add one explicit concept: `execution_mode`.

Supported values:

- `manual`
- `guided`
- `autonomous`

### 6.1 Manual mode
The system analyzes, proposes, and audits.
No experiment runs without explicit approval.

### 6.2 Guided mode
The system proposes a plan and asks for approval at checkpoints:
- before search
- before expensive branches
- before external optional backends
- before final recommendation
- before packaging or serving

### 6.3 Autonomous mode
The system may:
- inspect data independently with multiple specialists
- propose competing interpretations
- run experiments
- expand or prune search
- promote promising branches
- run challenger branches
- run ablations
- calibrate and uncertainty-wrap top candidates
- recommend additional data
- stop when stop criteria are met

### Required CLI and API support

All major commands must accept:

```bash
--execution-mode manual|guided|autonomous
```

---

## 7. Policy system

Promote the current runtime policy into a full **inference engineering policy layer**.

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
    require_challenger_science: true
  memory:
    enable_run_memory: true
    allow_prior_retrieval: true
  interoperability:
    enable_mcp_server: true
    expose_tool_contracts: true
  reporting:
    create_model_card: true
    create_risk_report: true
    create_experiment_graph: true
    create_handoff_graph: true
    create_data_recommendations: true
    create_decision_memo: true
```

### Acceptance criteria

- policy loads from YAML and can be overridden by CLI flags
- every experiment stores the resolved effective policy
- planner decisions cite the policy fields that influenced them
- optional frontier backends are policy-gated
- approval boundaries are explicit and logged

---

## 8. Unknown-domain investigation

Relaytic must be strong when the domain is not known in advance.

### Required behaviors

Before major modeling begins, the system must infer:
- task framing
- target semantics
- whether the dataset is event-based, entity-based, or time-indexed
- whether post-outcome leakage is likely
- whether the user likely wants:
  - forecasting
  - scoring
  - ranking
  - classification
  - surrogate/emulation
  - anomaly triage
- whether there is evidence of label lag
- whether additional business/process context is required

### Required artifact

- `domain_memo.json`

### Acceptance criteria

- the system can produce a structured memo even when domain labels are absent
- route selection must cite unknown-domain reasoning
- the memo must identify what the system is confident about vs unsure about

---

## 9. Independent data understanding

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

## 10. Planning system

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
- one learned-prior route when appropriate
- one classical search route when appropriate
- one hybrid route when appropriate

---

## 11. Run memory and meta-priors

Relaytic should improve over time through memory.

### Required stored memory

For each run, store:
- dataset fingerprint / meta-features
- selected route
- winning plan archetype
- challenger outcomes
- fragility patterns
- calibration outcomes
- drift warnings
- additional-data recommendations
- final deployment decision

### Required capabilities

- retrieve similar prior runs
- warm-start plan generation from archetypes
- retrieve similar failure modes
- propose prior-informed challenger plans
- compare current run against historical analogs

### Required artifacts

- `memory_retrieval.json`
- `plan_archetype.json`
- `historical_analogs.json`

### Acceptance criteria

- planning can use prior memory when enabled
- priors are cited explicitly in planning rationale
- memory use is optional and policy-controlled
- the system remains fully functional when memory is empty

---

## 12. Advanced search controller

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

## 13. Autonomous experimentation loop

This is the core change.

### New orchestration loop

```text
investigate -> resolve understanding -> hypothesize -> plan -> execute batch -> challenge -> ablate -> audit -> decide next step
```

### Required loop decisions

After each batch, the Strategist or Synthesizer decides one of:

- `expand_search`
- `refine_promising_branch`
- `promote_branch`
- `queue_challenger_branch`
- `run_ablation_suite`
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
- ablation evidence suggests current gains are fragile

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
- ablation evidence does not indicate obvious fragility

---

## 14. Experiment graph and handoff graph

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
- ablation suite
- final selection decision

### Experiment graph edges may represent

- expanded from
- pruned because
- promoted because
- challenged because
- ablated because
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

## 15. Model family system

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

### Learned-prior / frontier route layer
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

### Strategy rule

The Strategist must explicitly consider:
- classical search route
- learned-prior route
- hybrid route

when the dataset regime supports them.

---

## 16. Features system

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

## 17. Evidence and ablation engine

Relaytic must generate evidence about why performance happened.

### Required ablations

- leakage ablations
- suspicious-feature removal
- temporal split ablations
- reduced-data ablations
- missingness stress tests
- entity holdout ablations
- label-noise sensitivity
- feature-family ablations
- route ablations

### Required outputs

- `ablation_report.json`
- `belief_update.json`

### Belief update must include

- which assumptions likely drove gains
- which assumptions likely inflated gains
- whether the incumbent is fragile
- what evidence most strongly supports the final recommendation
- what evidence remains uncertain

### Acceptance criteria

- ablations are runnable automatically in guided/autonomous modes
- the final recommendation must cite ablation findings
- fragile gains must be surfaced clearly

---

## 18. Evaluation redesign

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

## 19. Uncertainty, calibration, and abstention

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

## 20. Drift, OOD, monitoring, and observability

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

## 21. Additional data planning and value-of-information

Relaytic must be able to propose additional data collection opportunities regardless of domain.

### Required behaviors

- identify likely missing explanatory factors
- identify weakly observed entities or slices
- suggest temporal extension when forecast uncertainty is too high
- suggest external covariates when residual structure indicates missing signal
- distinguish between:
  - must-have additional data
  - high-value optional data
  - speculative exploratory data

### Value-of-information requirements

For each recommendation, estimate:
- expected uncertainty reduction
- expected robustness improvement
- expected calibration improvement
- relative collection cost
- whether more data is better than more search

### Required artifacts

- `data_recommendations.json`
- `voi_report.json`

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

## 22. Artifacts contract

Standardize the output layout.

### New run directory layout

```text
artifacts/run_<timestamp>/
  manifest.json
  policy_resolved.yaml
  dataset_profile.json
  domain_memo.json
  independent_profiles/
    profile_a.json
    profile_b.json
    disagreement_report.json
    resolved_understanding.json
  memory_retrieval.json
  plan_archetype.json
  historical_analogs.json
  plan.json
  alternatives.json
  hypotheses.json
  rejected_routes.json
  experiment_graph.json
  handoff_graph.json
  pareto_front.json
  leaderboard.csv
  ablation_report.json
  belief_update.json
  data_recommendations.json
  voi_report.json
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
    decision_memo.md
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

## 23. Reporting redesign

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
- domain memo
- independent interpretations
- disagreement resolution
- route
- split choice
- search path
- model comparison
- calibration and uncertainty
- robustness findings
- challenger findings
- ablation findings

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

#### `decision_memo.md`
Contains:
- ship / continue search / collect data / reject recommendation
- rationale
- strongest supporting evidence
- strongest remaining uncertainty
- what next step would most change confidence

#### `interoperability_report.md`
Contains:
- available tool endpoints
- artifact export structure
- machine-readable actions
- MCP/tool compatibility notes

---

## 24. CLI redesign

Preserve current commands, but add a unified surface.

### New primary CLI

```bash
relaytic run \
  --data-path data.csv \
  --target target \
  --execution-mode autonomous \
  --policy configs/policies/default.yaml \
  --goal best_robust_pareto_front
```

### Required subcommands

```bash
relaytic investigate
relaytic hypothesize
relaytic plan
relaytic experiment
relaytic challenge
relaytic ablate
relaytic audit
relaytic report
relaytic serve
relaytic replay
relaytic tool-server
```

### Backward compatibility

Keep current commands working for one transition cycle and mark them as legacy in help text.

---

## 25. API redesign

Turn the current API entrypoint into a full local API surface.

### Endpoints

- `POST /investigate`
- `POST /hypothesize`
- `POST /plan`
- `POST /run`
- `POST /challenge`
- `POST /ablate`
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

## 26. Agent interoperability layer

The system must expose itself as a reusable tool surface for other agents.

### Required surfaces

- local HTTP API
- schema-defined tool actions
- MCP-compatible server mode
- machine-readable artifact manifests
- tool descriptions for external agent frameworks
- tool cards for judgment-oriented tasks

### Required exported capabilities

- `investigate_dataset`
- `propose_hypotheses`
- `generate_plan`
- `run_experiments`
- `run_challenger_branch`
- `run_ablation_suite`
- `audit_model`
- `compare_models`
- `recommend_additional_data`
- `generate_data_collection_plan`
- `produce_decision_memo`
- `package_model`
- `summarize_run`

### Acceptance criteria

- another local agent can call the system through a stable tool contract
- capability manifests are emitted automatically
- tool outputs are schema-validated
- external agents can consume artifacts without reading ad hoc logs

---

## 27. Streamlit interface

Add a local UI as an operator console.

### Screens

- upload / dataset selection
- domain memo
- independent profile comparison
- hypothesis view
- plan view
- experiment progress
- challenger comparison
- ablation findings
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

## 28. Demo, showcase, and adoption layer

The system must be easy to showcase locally and easy for new users to adopt.

### Required demos

- **Golden tabular demo**
  - one public dataset
  - full investigate -> hypothesize -> plan -> experiment -> challenge -> ablate -> audit -> report flow
  - polished artifacts committed or reproducibly generated

- **Golden time-series demo**
  - one public time-series dataset
  - temporal route selection
  - challenger branch, ablation findings, and uncertainty outputs

- **Plugin / tool-server demo**
  - one example where another local agent or script invokes Relaytic through the tool server or API
  - returns machine-readable artifacts and a final recommendation

### Required showcase assets

- architecture diagram
- specialist handoff diagram
- screenshot set or GIFs of the UI
- sample report bundle
- example artifact tree
- benchmark summary figure
- one “why the model was chosen” visual
- one “more data vs more search” visual

### Acceptance criteria

- a new user can run one command and reproduce a polished demo
- the README links directly to demo commands and expected outputs
- the Streamlit/operator UI can be used live for a walkthrough
- at least one demo shows external-agent interoperability

---

## 29. LLM and agent contract redesign

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
- `run_ablation_suite`
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

## 30. Benchmark harness

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
- no run memory vs run memory
- no ablation judge vs ablation judge

---

## 31. Testing expansion

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
- run-memory retrieval
- value-of-information estimation

### Integration tests
- investigate -> hypothesize
- hypothesize -> plan
- plan -> execute
- execute -> challenge
- challenge -> ablate
- ablate -> audit
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
- `decision_memo.md`

---

## 32. CI and repo health

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

## 33. Migration plan

### Phase 1: architecture foundations
Implement first:
1. policy schema
2. execution modes
3. unknown-domain memo
4. independent inspect and disagreement artifacts
5. plan object
6. experiment registry
7. artifact contract
8. compatibility shims

### Phase 2: specialist upgrade
Implement:
1. scout
2. scientist
3. strategist
4. builder
5. challenger
6. ablation judge
7. memory keeper
8. coordinator loop
9. schema-validated actions

### Phase 3: evidence and inference reliability
Implement:
1. calibration
2. conformal uncertainty
3. CQR
4. stability checks
5. slice analysis
6. drift report
7. abstention policy outputs
8. ablation engine
9. value-of-information estimates

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
4. run memory priors
5. docs rewrite
6. end-to-end demos

---

## 34. Implementation rules for Codex

1. Do not remove local-first defaults.
2. Do not make remote APIs required.
3. Preserve existing commands where practical using compatibility wrappers.
4. Prefer additive refactors over destructive rewrites.
5. Keep deterministic behavior where already present.
6. Add tests with each major subsystem.
7. Persist every important decision as an artifact.
8. Do not introduce heavyweight dependencies into the core package unless behind optional extras.
9. Maintain clean separation between:
   - investigation
   - hypotheses
   - planning
   - execution
   - challenge
   - ablation
   - evaluation
   - reporting
   - memory
   - interoperability
10. The system must work without LLMs.
11. The system must support local LLM enhancement without becoming LLM-dependent.
12. Challenger behavior is required in autonomous mode.
13. Additional-data recommendation is required in guided/autonomous modes.
14. Every important handoff must be logged.
15. The system must generate evidence, not just metrics.
16. Judgment-oriented tools must be exposed for external agents.

### Required deliverables from Codex

- refactored package structure
- new policy system
- new execution mode system
- unknown-domain memo layer
- independent inspect/disagreement layer
- new plan object and planner
- hypothesis subsystem
- experiment graph and handoff graph
- challenger subsystem
- ablation engine and belief-update engine
- run-memory subsystem
- uncertainty, calibration, abstention, and CQR modules
- additional-data recommendation subsystem
- value-of-information subsystem
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

## 35. README rewrite requirements

The README must present the system as:

- local-first
- autonomous but steerable
- multi-specialist
- hypothesis-driven
- evidence-driven
- inference-first
- memory-guided
- policy-aware
- artifact-rich
- consumable by other agents

### Required README sections

1. What Relaytic is
2. Why local-first
3. Why multiple specialists
4. Why this is different from plain AutoML
5. Execution modes
6. Supported routes
7. Hypothesis-driven workflow
8. Evidence and challenger science
9. Golden demos
10. Example outputs
11. Artifact structure
12. Policy examples
13. Tool-server / plugin usage
14. Benchmarks
15. Development and tests

### README quality requirements

- open with a one-paragraph product identity statement using the format: `Relaytic — The Relay Inference Lab`
- include a fast 60-second quickstart
- include one copy-paste demo command
- include screenshots or GIFs of the UI
- include a sample artifact tree
- include a clear explanation of how external agents can use the tool server
- include benchmark and ablation highlights
- include a concise statement explaining why Relaytic is not just AutoML

### Required differentiator sentence

The README must contain a line close to:

> Relaytic is not an AutoML wrapper. It is a local-first inference engineering system that investigates data, forms competing hypotheses, runs challenger science, quantifies uncertainty, recommends missing data, and exposes its judgment as reusable tools for other agents.

---

## 36. Definition of done

The transformation is complete when all of the following are true:

1. A user can run one command locally on a CSV and get:
   - dataset profile
   - domain memo
   - independent specialist interpretations
   - hypotheses
   - plan
   - experiments
   - challenger comparison
   - ablation findings
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
   - `domain_memo.json`
   - independent profile artifacts
   - `plan.json`
   - `hypotheses.json`
   - `experiment_graph.json`
   - `handoff_graph.json`
   - `ablation_report.json`
   - `belief_update.json`
   - `leaderboard.csv`
   - summary and technical reports
   - uncertainty, calibration, abstention, and drift artifacts
   - `data_recommendations.json`
   - `voi_report.json`

5. The system works fully offline by default.

6. Tests cover the new investigate, planner, builder, challenger, ablation judge, and auditor flow.

7. Existing deterministic behavior remains intact where applicable.

8. Another local agent can invoke the system through a stable tool contract.

9. The repository includes at least:
   - one polished tabular golden demo
   - one polished time-series golden demo
   - one external-agent / tool-server demo

10. The README and UI are strong enough to support a live walkthrough without extra explanation.

---

## 37. First implementation sprint

### Sprint 1
- add policy schema
- add execution mode support
- create unknown-domain memo
- create independent inspect outputs
- create `plan.json`
- create `hypotheses.json`
- create experiment registry
- refactor current analyst/modeler flow into `scout` + `scientist` + `strategist` + `builder`
- standardize artifact directory
- add compatibility shims
- update CLI with `investigate`, `hypothesize`, `plan`

### Sprint 2
- add experiment graph
- add handoff graph
- add challenger subsystem
- add ablation engine
- add calibration
- add conformal uncertainty
- add CQR
- add stability checks
- add `summary.md`, `technical_report.md`, `model_card.md`, `risk_report.md`, `decision_memo.md`
- add `data_recommendations.json`
- add `voi_report.json`

### Sprint 3
- add API
- add MCP/tool server
- add Streamlit app
- add serve mode and local monitoring
- add benchmark harness
- add optional frontier backends
- add run memory priors
- add golden demos
- add showcase assets
- rewrite README for fast onboarding and live demo use
- integrate Relaytic branding across docs, UI, CLI, and demo assets

---

## 38. Codex prompt to paste

```text
Read RELAYTIC_TRANSFORMATION_PLAN.md and implement it incrementally.

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
   scout, scientist, strategist, builder, challenger, ablation_judge, auditor, synthesizer, memory_keeper, broker.
2. Add execution modes: manual, guided, autonomous.
3. Add unknown-domain investigation and disagreement resolution before planning.
4. Add a policy-aware planning layer that chooses route, metric, split strategy, feature plan, candidate families, uncertainty plan, and stop criteria.
5. Generalize the system into routes:
   tabular_regression, tabular_classification, time_series_forecasting, time_aware_prediction, surrogate.
6. Add a hypothesis-driven autonomous loop:
   investigate -> resolve understanding -> hypothesize -> plan -> execute batch -> challenge -> ablate -> audit -> decide next step.
7. Add multi-fidelity search, branch promotion, challenger branches, and Pareto-front selection.
8. Add experiment graph and handoff graph artifacts.
9. Add run-memory priors and historical analog retrieval.
10. Add calibration, conformal uncertainty, CQR, abstention, robustness, slice analysis, drift reports, and value-of-information estimates.
11. Add additional-data recommendation as a first-class artifact.
12. Add structured ablation science and belief updates.
13. Standardize artifacts and reports.
14. Add a unified CLI:
   investigate, hypothesize, plan, experiment, challenge, ablate, audit, report, serve, replay, tool-server.
15. Add a real local API and a local MCP/tool server.
16. Add a local Streamlit operator UI.
17. Add benchmark harness and ablations.
18. Rewrite README to match the new architecture.
19. Integrate the Relaytic brand and the descriptor `The Relay Inference Lab` consistently across docs, CLI help, UI copy, tool-server descriptions, and demo assets.

Required outputs per run:
- policy_resolved.yaml
- dataset_profile.json
- domain_memo.json
- independent_profiles/*
- memory_retrieval.json
- plan_archetype.json
- historical_analogs.json
- plan.json
- alternatives.json
- hypotheses.json
- experiment_graph.json
- handoff_graph.json
- pareto_front.json
- leaderboard.csv
- ablation_report.json
- belief_update.json
- data_recommendations.json
- voi_report.json
- best_model/*
- reports/summary.md
- reports/technical_report.md
- reports/model_card.md
- reports/risk_report.md
- reports/decision_memo.md
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
- unknown-domain memo
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
- ablation engine
- calibration
- conformal uncertainty
- CQR
- robustness/stability/slice analysis
- additional-data recommendations
- value-of-information estimates
- richer reports

Sprint 3:
- API
- MCP/tool server
- Streamlit UI
- serve mode with local monitoring
- benchmark harness
- optional frontier backends
- run memory priors
- golden demos
- showcase assets

Do not add heavy dependencies to the core package unless placed behind optional extras.
```