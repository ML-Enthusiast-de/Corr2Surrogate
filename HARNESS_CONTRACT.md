# Corr2Surrogate Harness Contract

## 1. Integration Point
Integrate LLM agents now at the orchestration layer, not inside analytics/modeling math.

- Deterministic modules remain source of truth:
  - ingestion, profiling, correlations, ranking, splitting, training, evaluation.
- LLM responsibility:
  - decide next step,
  - ask clarifying questions,
  - call allowed tools,
  - explain outcomes to user.

## 2. Runtime and Local-Only Policy
Default runtime constraints:
- Provider: local by default (`ollama` or `llama.cpp`).
- API usage: disabled by default; remote API mode is optional explicit opt-in.
- Telemetry/upload: disabled.
- Network calls: blocked by default for agent runtime.
- Model files: local GGUF only when running in local model mode.

CPU-first model profiles:
- `small_cpu`: 3B-4B quantized (Q4/K_M), no GPU required.
- `balanced`: 7B quantized (Q4/K_M), optional GPU.

Fallback policy:
- If hardware cannot sustain configured profile, downshift model profile automatically.

## 3. Agent/Tool Message Contract
Agent output must be strict JSON with exactly one action:

```json
{
  "action": "tool_call",
  "tool_name": "run_quality_checks",
  "arguments": {
    "dataset_id": "run_20260223_001",
    "target_signal": "pressure_out"
  },
  "reason": "Need completeness/outlier status before ranking."
}
```

or

```json
{
  "action": "respond",
  "message": "Header detection is uncertain. Please confirm header row and data start row."
}
```

Rules:
- No free-form chain-of-thought in output.
- One action per turn.
- Unknown tools or invalid args are rejected and returned as structured errors.

## 4. Tool Registry Contract
Each tool is registered with:
- `name`
- `description`
- `input_schema` (JSON Schema)
- `handler` (Python callable)
- `risk_level` (`low`, `confirm`, `blocked`)

Example minimal tool set:
- `load_tabular_data`
- `run_agent1_analysis`
- `run_quality_checks`
- `compute_correlations`
- `rank_surrogate_candidates`
- `build_modeling_directives`
- `train_incremental_linear_surrogate`
- `resume_incremental_linear_surrogate`
- `list_model_checkpoints`
- `analyze_model_checkpoint_performance`
- `save_artifacts`

## 5. Orchestration Loop Contract
Per turn:
1. Build prompt context from state.
2. Ask active agent for one JSON action.
3. Validate action + arguments against registry schema.
4. Execute tool or send user response.
5. Append structured event to run log.
6. Stop on terminal state, max turns, or explicit user wait state.

Hard limits:
- `max_turns_per_phase`
- `max_tool_calls_per_turn` = 1
- `max_retries_on_invalid_action`

## 6. Agent Roles in Harness
Agent 1 (`Analyst`) terminal outputs:
- ranked candidates with dependency risk,
- forced requests acknowledged,
- handoff payload for Agent 2.

Agent 2 (`Modeler`) terminal outputs:
- selected model strategy,
- loop evaluation status,
- artifact paths + recommendations if quality unmet,
- savepoint/checkpoint ids for resume-retraining,
- post-test bad-region diagnostics + suggested lab trajectories.

## 7. Handoff Schema Requirements
The Agent 1 to Agent 2 handoff must include:
- `forced_modeling_requests`
- `dependency_map`
- `system_knowledge`
- `normalization`
- `acceptance_criteria`
- `loop_policy`

Any missing required section blocks Agent 2 start.

## 8. Agentic Loop Contract
After each training attempt:
- evaluate metrics vs acceptance criteria.
- if passed: finalize.
- if failed and attempts remain: continue with updated strategy.
- if failed and max attempts reached: stop and emit recommendations.

Recommendation categories:
- data sufficiency (`collect more representative data`)
- architecture change (`switch baseline/tree/LSTM class`)
- feature redesign (`lags/windows/derived signals`)

## 9. Folder Mapping (Current Project)
Use current modules as the harness backbone:
- `src/corr2surrogate/orchestration/workflow.py`
- `src/corr2surrogate/orchestration/handoff_contract.py`
- `src/corr2surrogate/agents/agent1_analyst.py`
- `src/corr2surrogate/agents/agent2_modeler.py`
- `src/corr2surrogate/analytics/`
- `src/corr2surrogate/modeling/`
- `src/corr2surrogate/persistence/`

Recommended next files:
- `src/corr2surrogate/orchestration/tool_registry.py`
- `src/corr2surrogate/orchestration/agent_loop.py`
- `src/corr2surrogate/orchestration/runtime_policy.py`
- `src/corr2surrogate/orchestration/default_tools.py`
- `src/corr2surrogate/orchestration/harness_runner.py`

## 10. LangChain Decision
Recommendation: do not use LangChain for MVP.

Reason:
- you already have domain-specific deterministic modules,
- strict schema + policy control is easier with a thin custom harness,
- lower overhead and easier auditability.

Re-evaluate LangGraph/LangChain only if orchestration graph complexity grows beyond what your custom loop can maintain.

## 11. Behavior Tuning Path
Default approach:
- tune system prompts + runtime settings first.
- use fine-tuning only if prompt-level behavior control is insufficient at scale.

Prompt edit points:
- `src/corr2surrogate/agents/prompts/analyst_system.txt`
- `src/corr2surrogate/agents/prompts/modeler_system.txt`
- config overrides in `configs/default.yaml` under `prompts.*`
