# Corr2Surrogate

Corr2Surrogate is a local-first framework for converting real-world sensor data into validated surrogate-model candidates.

## What It Does
- Agent 1 (`Analyst`) ingests CSV/XLS/XLSX data, checks data quality, runs correlation analysis, ranks surrogate candidates, and performs lightweight probe-model screening to recommend which model families Agent 2 should test first.
- Agent 2 (`Modeler`) now has a first executable training path: split-safe train/validation/test modeling for both regression and classification, with local linear/logistic baselines, local tree baselines, reproducible artifacts, and result interpretation. The intended search order remains pragmatic: start with linear or lagged baselines, escalate to tree ensembles when interaction or piecewise evidence is real, and only test sequence models when temporal probes justify it.

## Key Capabilities
- CSV/XLS/XLSX ingestion with sheet selection and header/data-start inference
- Quality checks (missingness, duplicates, outliers, timestamp integrity)
- Correlation analysis (Pearson, Spearman, Kendall, distance, lag-aware)
- Feature-engineering scans (including `rate_change`)
- Probe-model screening (linear, interaction-aware, tiny regression tree, lagged linear)
- Residual nonlinearity and regime diagnostics to support model-family selection
- Model-family recommendations for Agent 2 search order
- Evidence-backed recommendation blocks with probe inputs, quick metrics, and confidence
- Dependency-aware surrogate ranking
- Dataset-scoped reports and artifact export
- Task-type detection for regression, balanced classification, and fraud/anomaly-style labels
- User task override command: `task regression`, `task binary_classification`, `task fraud_detection`, `task auto`
- Split-safe Agent 2 training (train-only preprocessing, time-ordered splits for time series, stratified steady-state splits for classification labels)
- Executable classifier baselines: `logistic_regression` and `bagged_tree_classifier`
- Executable time-series classifier baselines: `lagged_logistic_regression` and `lagged_tree_classifier`
- Validation-tuned decision thresholds for binary classification and fraud screening
- Adaptive bounded optimization in Agent 2:
  - safe model-family switches
  - feature-set expansion
  - lag-window expansion
  - threshold-policy retuning for binary classifiers
- Post-stall experiment guidance:
  - when retries stop, Agent 2 proposes concrete next data-collection trajectories instead of only saying "collect more data"
- First nonlinear modeling baseline: local bagged tree ensemble / tree classifier
- Candidate comparison with LLM-assisted model interpretation in the CLI
- Local runtime by default; optional API mode via explicit opt-in

## Privacy Defaults
- Keep private datasets in `data/private/`.
- `data/private/`, `reports/`, and `artifacts/` are git-ignored by default.

## Prerequisites
- Python `>=3.10` (project default tested with Python 3.11)
- Git
- For local LLM mode:
  - Windows: `llama.cpp` or Ollama
  - macOS: `llama.cpp` (recommended) or Ollama

## Quickstart (Windows)
1. Clone and enter repo:
```powershell
git clone <your-repo-url>
cd Corr2Surrogate
```
2. Create virtual environment and install:
```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```
3. Set up local LLM runtime (Qwen 4B local profile):
```powershell
& .\.venv\Scripts\corr2surrogate.exe setup-local-llm --provider llama_cpp --install-provider
```
4. Start analyst session:
```powershell
# works even if you stay in conda base/ml
& .\.venv\Scripts\corr2surrogate.exe run-agent-session --agent analyst
```

## Quickstart (macOS)
1. Install `llama.cpp`:
```bash
brew install llama.cpp
```
2. Clone and enter repo:
```bash
git clone <your-repo-url>
cd Corr2Surrogate
```
3. Create virtual environment and install:
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```
4. Set up local LLM runtime:
```bash
corr2surrogate setup-local-llm --provider llama_cpp
```
5. Start analyst session:
```bash
./.venv/bin/corr2surrogate run-agent-session --agent analyst
```

## Analyst Session Usage
At startup you can:
- paste a `.csv/.xlsx/.xls` path, or
- type `default` to run the built-in public test dataset.

Default dataset note:
- `data/public/public_testbench_dataset_20k_minmax.csv` is sanitized and intentionally includes ~5% missing values in three representative signals (`B`, `D`, `F`), while all other signals and `time` remain complete. This lets users test missing-data handling and leakage warnings without collapsing row count when trying `drop_rows`.
- Additional synthetic public datasets are available for task detection checks:
  - `data/public/public_binary_classification_dataset.csv`
  - `data/public/public_fraud_screening_dataset.csv`

Useful commands:
- `/help`
- `/context`
- `/reset`
- `/exit`
- `task regression`
- `task binary_classification`
- `task fraud_detection`
- `task auto`

Target selection shortcuts:
- `list`
- `list <filter>`
- `all`
- comma-separated signal names
- numeric index

Hypothesis syntax:
- Correlation: `hypothesis corr target:pred1,pred2; target2:pred3`
- Feature: `hypothesis feature target:signal->rate_change; signal2->square`

Agent 1 also produces model-strategy guidance for Agent 2:
- start with `linear_ridge` or `lagged_linear` baselines
- treat `tree_ensemble_candidate` as "worth testing next" only when probe gains or regime evidence are material
- only test `sequence_model_candidate` when temporal probes justify it after lagged/tabular baselines
- after Agent 1 finishes in the analyst session, the CLI now prints the structured handoff path and can immediately continue into Agent 2 without restarting a separate session

Interpretation rule:
- a detected nonlinear dependence alone is not enough to recommend trees or sequence models
- Agent 1 uses cheap validation probes to decide whether those heavier families are likely to outperform the linear baseline
- Agent 1 also infers whether a target behaves like regression, classification, or fraud/anomaly screening and records the recommended split policy for Agent 2

## Optional API Mode (Explicit Opt-In)
Default policy is local-only. API mode must be explicitly enabled.

### PowerShell
```powershell
$env:C2S_PROVIDER="openai"
$env:C2S_REQUIRE_LOCAL_MODELS="false"
$env:C2S_BLOCK_REMOTE_ENDPOINTS="false"
$env:C2S_API_CALLS_ALLOWED="true"
$env:C2S_OFFLINE_MODE="false"
$env:C2S_API_KEY="<your_api_key>"
$env:C2S_MODEL="gpt-4.1-mini"
# optional:
# $env:C2S_ENDPOINT="https://api.openai.com/v1/chat/completions"

& .\.venv\Scripts\corr2surrogate.exe setup-local-llm --provider openai
```

### Bash/Zsh
```bash
export C2S_PROVIDER=openai
export C2S_REQUIRE_LOCAL_MODELS=false
export C2S_BLOCK_REMOTE_ENDPOINTS=false
export C2S_API_CALLS_ALLOWED=true
export C2S_OFFLINE_MODE=false
export C2S_API_KEY=<your_api_key>
export C2S_MODEL=gpt-4.1-mini
# optional:
# export C2S_ENDPOINT=https://api.openai.com/v1/chat/completions

corr2surrogate setup-local-llm --provider openai
```

## Deterministic Agent 1 (No LLM Call)
```bash
corr2surrogate run-agent1-analysis --data-path data/private/run1.csv --timestamp-column time
```

Example with advanced options:
```bash
corr2surrogate run-agent1-analysis \
  --data-path data/private/run1.csv \
  --timestamp-column time \
  --target-signals y \
  --max-lag 12 \
  --confidence-top-k 10 \
  --bootstrap-rounds 40 \
  --stability-windows 4 \
  --strategy-search-candidates 4
```

## Planned Agent 2 Modeling Roadmap
Agent 1 now emits a model-strategy prior for each target. Agent 2 should treat that as a search-order hint, not a guarantee.

Current executable model families:
- `linear_ridge`
- `logistic_regression`
- `lagged_logistic_regression`
- `bagged_tree_classifier`
- `lagged_linear`
- `lagged_tree_classifier`
- `lagged_tree_ensemble`
- `bagged_tree_ensemble`

Current classification note:
- Agent 2 now detects classification / fraud-like targets, applies the correct split policy (stratified for steady-state labels, ordered for time-based labels), and trains the local classifier baselines instead of falling back to regression.
- Default acceptance checks are task-aware:
  - balanced classification defaults to `f1` + `accuracy`
  - fraud / anomaly screening defaults to `recall` + `pr_auc`
- Binary classifier thresholds are validation-tuned by default, and users can override the next run with:
  - `threshold favor_recall`
  - `threshold favor_precision`
  - `threshold favor_f1`
  - `threshold favor_pr_auc`
  - `threshold 0.35`
  - `threshold auto`

Recommended model families to implement next:
- steady-state / tabular: add `ElasticNet`
- stronger tabular baseline: add boosted-tree alternatives beyond the current bagged tree path
- sequence models: `GRU`/`LSTM` only when lagged/tabular probes still leave meaningful residual dynamics

Operational rule:
- do not jump directly to LSTM just because a target is time-based
- require evidence from lag benefit, autocorrelation, and failed simpler baselines first

## Concrete Implementation Sequence
This is the planned implementation order to maximize practical value while keeping the system scientifically defensible.

1. Build executable `Agent2Handoff` consumption:
   - Agent 2 must accept Agent 1 target, predictor, normalization, and constraint payloads directly.
2. Implement split-safe preprocessing:
   - split first
   - fit imputation/normalization on train only
   - apply unchanged transforms to validation/test
3. Implement the first true end-to-end baseline trainers:
   - steady-state: `Ridge` / `ElasticNet`
   - time-series: `lagged linear`
4. Add the first nonlinear baseline:
   - `HistGradientBoostingRegressor` or `ExtraTreesRegressor`
5. Add model comparison and acceptance logic:
   - compare baseline vs higher-capacity models on validation/test
   - promote only when the more complex model shows meaningful validated gain
6. Persist reproducible artifacts:
   - model weights/parameters
   - feature list / lag schema
   - split metadata
   - normalization state
   - metrics and lineage
7. Add post-model failure analysis:
   - identify operating regions with high error
   - recommend concrete new lab/testbench trajectories
8. Add bounded optimization loops:
   - Optuna only after the deterministic training pipeline is correct

Current state:
- steps 1-7 are now implemented in the first production path
- the current modeler loop already performs bounded retries across model family, feature set, lag horizon, and binary threshold policy when the loop policy allows it
- the current loop also prints concrete next experiment trajectories when retries stall
- the next highest-value gaps are stronger boosted-tree families, deeper region-aware experiment design, inference workflows, and uncertainty reporting

## Modeling Entry Modes
Two user entry paths are part of the intended product behavior:

1. Handoff-driven mode:
   - run Agent 1 analysis first
   - pass Agent 1 recommendations into Agent 2 as a prior
2. Direct modeler mode (skip Agent 1):
   - user may immediately ask Agent 2 to build a model with explicit inputs and target
   - current executable implementation supports `auto`, `linear_ridge`, `logistic_regression`, `lagged_logistic_regression`, `bagged_tree_classifier`, `lagged_linear`, `lagged_tree_classifier`, `lagged_tree_ensemble`, and `bagged_tree_ensemble`
   - example intent: "build model `linear_ridge` with inputs `A,B,C` and target `D`"
   - this fast path should not require a prior Agent 1 handoff

Session entry examples:
```powershell
& .\.venv\Scripts\corr2surrogate.exe run-agent-session --agent modeler
```

Example direct modeler request:
- `build model linear_ridge with inputs A,B,C and target D`
- `build model tree with inputs torque,temp,flow and target pressure`

Current implementation note:
- the direct modeler session currently executes `auto`, `linear_ridge` / `ridge` / `linear`, `logistic_regression` / `logistic` / `logit` / `linear_classifier`, `lagged_logistic_regression` / `lagged_logistic` / `lagged_logit` / `temporal_classifier`, `bagged_tree_classifier` / `tree_classifier`, `lagged_linear` / `lagged` / `temporal_linear` / `arx`, `lagged_tree_classifier` / `temporal_tree_classifier`, `lagged_tree_ensemble` / `lagged_tree` / `lag_window_tree` / `temporal_tree`, and `bagged_tree_ensemble` / `tree`
- the modeler CLI now runs a split-safe comparison between the linear baseline and the available temporal/nonlinear comparators, performs a bounded acceptance check, and can retry with the next safe model family, expanded feature set, wider lag window, or retuned binary threshold when policy allows it
- if the selected target looks like classification or fraud detection, the modeler reports that explicitly, applies the correct split policy, trains the local classifier candidates, and evaluates them with classification-aware metrics
- if retries stop without meeting quality, the modeler prints concrete experiment recommendations for the next data collection pass
- before a modeler run, users can steer binary decision policy with:
  - `threshold favor_recall`
  - `threshold favor_precision`
  - `threshold favor_f1`
  - `threshold favor_pr_auc`
  - `threshold 0.35`
  - `threshold auto`

## User Override Rules For Agent 2
Even when a valid Agent 1 handoff exists, the user must remain in control of the modeling scope.

Agent 2 should offer these choices:
- use Agent 1 recommended target/predictors/model family
- choose a specific target manually
- choose specific predictor inputs manually (with `list` / filtered list support)
- choose a specific model architecture manually

Priority rule:
- explicit user choices override Agent 1 recommendations unless they violate a hard policy or missing-data / split-safety constraint

Example override intents after handoff:
- `use the recommended target but only inputs A,C,F`
- `show signal list`
- `train a tree ensemble for target D using inputs A,B,C`

## LLM Role And Boundaries
The LLM should be used as an agent, but only inside a bounded control loop.

Use the LLM for:
- conversational steering and clarification
- choosing the next safe tool call among registered tools
- explaining analysis and modeling results in plain scientific language
- proposing next experiments, architecture changes, or feature hypotheses

Keep deterministic tools as the source of truth for:
- data loading and schema inference
- correlation metrics and probe-model scores
- split creation, preprocessing, training, and persisted artifacts
- acceptance decisions based on measured validation/test metrics

Prompting guidance to improve agent quality:
- always include the current `workflow_stage`
- include the last 5 user prompts plus the latest compact tool result
- include hard constraints (critical signals, non-virtualizable signals, accepted overrides)
- explicitly remind the model that correlation is not causality and time-series does not imply LSTM by default
- require the assistant to answer naturally, then steer back to the pending required input or next tool decision

## Behavior and Prompt Control
- Runtime defaults: `configs/default.yaml`
- Analyst prompt: `src/corr2surrogate/agents/prompts/analyst_system.txt`
- Modeler prompt: `src/corr2surrogate/agents/prompts/modeler_system.txt`
- Optional prompt overrides:
  - `prompts.analyst_system_path`
  - `prompts.modeler_system_path`
  - `prompts.extra_instructions`

## Output Structure
Outputs are grouped by dataset slug under `reports/<dataset_slug>/`:
- Markdown report: `agent1_<timestamp>.md`
- Lineage JSON: `agent1_<timestamp>.lineage.json`
- Artifact directory: `agent1_<timestamp>_artifacts/`
- Agent 1 report now includes a dedicated "Model Strategy Recommendations (Agent 2 Planning)" section with probe-model scores and suggested search order.

## Quality and Security Checks
Run tests:
```bash
./.venv/bin/python -m pytest
```
Run leak scan before commit/push:
```bash
./.venv/bin/c2s-guard
# or
./.venv/bin/python -m corr2surrogate.ui.cli scan-git-safety
```

Windows equivalents:
```powershell
& .\.venv\Scripts\python.exe -m pytest
& .\.venv\Scripts\c2s-guard.exe
# or
& .\.venv\Scripts\python.exe -m corr2surrogate.ui.cli scan-git-safety
```

## Troubleshooting
- `corr2surrogate` without path only works when the venv is active or the script is on `PATH`.
- If `corr2surrogate` is not recognized on Windows, use:
```powershell
& .\.venv\Scripts\corr2surrogate.exe --help
```
- If local provider is not reachable, run:
```bash
corr2surrogate setup-local-llm
```

## License
MIT (`LICENSE`)
