"""Command line interface for local harness operations."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Callable

from corr2surrogate.core.json_utils import dumps_json
from corr2surrogate.modeling import normalize_candidate_model_family
from corr2surrogate.orchestration.default_tools import build_default_registry
from corr2surrogate.orchestration.handoff_contract import build_agent2_handoff_from_report_payload
from corr2surrogate.orchestration.harness_runner import run_local_agent_once
from corr2surrogate.orchestration.local_llm_setup import setup_local_llm
from corr2surrogate.security.git_guard import main as git_guard_main


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="corr2surrogate", description="Corr2Surrogate CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    run_agent = sub.add_parser(
        "run-agent-once",
        help="Run one local agent loop (tool-calling + response).",
    )
    run_agent.add_argument(
        "--agent",
        choices=["analyst", "modeler"],
        required=True,
        help="Which agent system prompt to use.",
    )
    run_agent.add_argument(
        "--message",
        required=True,
        help="User message injected into loop context.",
    )
    run_agent.add_argument(
        "--context-json",
        default="{}",
        help="Additional JSON context object passed to the loop.",
    )
    run_agent.add_argument(
        "--config",
        default=None,
        help="Optional config path (otherwise default config resolution is used).",
    )

    run_session = sub.add_parser(
        "run-agent-session",
        help="Run interactive multi-turn local agent session.",
    )
    run_session.add_argument(
        "--agent",
        choices=["analyst", "modeler"],
        required=True,
        help="Which agent system prompt to use.",
    )
    run_session.add_argument(
        "--context-json",
        default="{}",
        help="Base JSON context object persisted across turns.",
    )
    run_session.add_argument(
        "--config",
        default=None,
        help="Optional config path (otherwise default config resolution is used).",
    )
    run_session.add_argument(
        "--show-json",
        action="store_true",
        help="Print full JSON payload for each turn in addition to assistant text.",
    )
    run_session.add_argument(
        "--max-turns",
        type=int,
        default=0,
        help="Optional positive turn cap. 0 means unlimited until /exit.",
    )

    setup_llm = sub.add_parser(
        "setup-local-llm",
        help="Install/check local LLM runtime and ensure model availability.",
    )
    setup_llm.add_argument(
        "--provider",
        choices=["ollama", "llama_cpp", "llama.cpp", "openai", "openai_compatible"],
        default=None,
        help="Override provider. Defaults to configured runtime provider.",
    )
    setup_llm.add_argument(
        "--profile",
        default=None,
        help="Optional runtime profile name.",
    )
    setup_llm.add_argument(
        "--model",
        default=None,
        help="Optional model override (ollama tag or llama.cpp alias/path).",
    )
    setup_llm.add_argument(
        "--endpoint",
        default=None,
        help="Optional endpoint override.",
    )
    setup_llm.add_argument(
        "--config",
        default=None,
        help="Optional config path.",
    )
    setup_llm.add_argument(
        "--install-provider",
        action="store_true",
        help="Attempt to install runtime provider if missing (via winget on Windows).",
    )
    setup_llm.add_argument(
        "--no-start-runtime",
        action="store_true",
        help="Skip runtime auto-start.",
    )
    setup_llm.add_argument(
        "--no-pull-model",
        action="store_true",
        help="Skip `ollama pull` when provider is ollama.",
    )
    setup_llm.add_argument(
        "--no-download-model",
        action="store_true",
        help="Skip GGUF download when provider is llama.cpp and file is missing.",
    )
    setup_llm.add_argument(
        "--llama-model-path",
        default=None,
        help="Override local GGUF path for llama.cpp setup.",
    )
    setup_llm.add_argument(
        "--llama-model-url",
        default=None,
        help="Optional GGUF download URL for llama.cpp setup.",
    )
    setup_llm.add_argument(
        "--timeout-seconds",
        type=int,
        default=120,
        help="HTTP timeout for setup checks.",
    )

    run_agent1 = sub.add_parser(
        "run-agent1-analysis",
        help="Run deterministic Agent 1 analysis directly (no LLM call).",
    )
    run_agent1.add_argument("--data-path", required=True, help="CSV/XLSX data file path.")
    run_agent1.add_argument("--sheet-name", default=None, help="Excel sheet name if needed.")
    run_agent1.add_argument("--header-row", type=int, default=None, help="Optional header row.")
    run_agent1.add_argument(
        "--data-start-row",
        type=int,
        default=None,
        help="Optional data start row.",
    )
    run_agent1.add_argument("--timestamp-column", default=None, help="Timestamp column name.")
    run_agent1.add_argument(
        "--target-signals",
        nargs="*",
        default=None,
        help="Optional target signal list.",
    )
    run_agent1.add_argument(
        "--predictor-map-json",
        default="{}",
        help="JSON object mapping target->predictor list, optional wildcard '*' key.",
    )
    run_agent1.add_argument(
        "--forced-requests-json",
        default="[]",
        help="JSON array of forced requests: [{target_signal,predictor_signals,user_reason}]",
    )
    run_agent1.add_argument(
        "--user-hypotheses-json",
        default="[]",
        help="JSON array of correlation hypotheses: [{target_signal,predictor_signals,user_reason}]",
    )
    run_agent1.add_argument(
        "--feature-hypotheses-json",
        default="[]",
        help=(
            "JSON array of feature hypotheses: "
            "[{target_signal?,base_signal,transformation,user_reason}]"
        ),
    )
    run_agent1.add_argument("--max-lag", type=int, default=8)
    run_agent1.add_argument("--no-feature-engineering", action="store_true")
    run_agent1.add_argument("--feature-gain-threshold", type=float, default=0.05)
    run_agent1.add_argument("--confidence-top-k", type=int, default=10)
    run_agent1.add_argument("--bootstrap-rounds", type=int, default=40)
    run_agent1.add_argument("--stability-windows", type=int, default=4)
    run_agent1.add_argument("--max-samples", type=int, default=None)
    run_agent1.add_argument(
        "--sample-selection",
        choices=["uniform", "head", "tail"],
        default="uniform",
    )
    run_agent1.add_argument(
        "--missing-data-strategy",
        choices=["keep", "drop_rows", "fill_median", "fill_constant"],
        default="keep",
    )
    run_agent1.add_argument("--fill-constant-value", type=float, default=None)
    run_agent1.add_argument(
        "--row-coverage-strategy",
        choices=["keep", "drop_sparse_rows", "trim_dense_window", "manual_range"],
        default="keep",
    )
    run_agent1.add_argument("--sparse-row-min-fraction", type=float, default=0.8)
    run_agent1.add_argument("--row-range-start", type=int, default=None)
    run_agent1.add_argument("--row-range-end", type=int, default=None)
    run_agent1.add_argument("--no-strategy-search", action="store_true")
    run_agent1.add_argument("--strategy-search-candidates", type=int, default=4)
    run_agent1.add_argument("--no-save-artifacts", action="store_true")
    run_agent1.add_argument("--run-id", default=None)
    run_agent1.add_argument("--no-save-report", action="store_true")

    scan = sub.add_parser(
        "scan-git-safety",
        help="Scan repository for potential secret/system-path leaks.",
    )
    scan.add_argument(
        "paths",
        nargs="*",
        help="Optional files/directories. If omitted, scans git-tracked files.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "scan-git-safety":
        passthrough = list(args.paths)
        return git_guard_main(passthrough)

    if args.command == "run-agent-once":
        try:
            context = _parse_context(args.context_json)
        except ValueError as exc:
            parser.error(str(exc))
            return 2
        try:
            result = _invoke_agent_once_with_recovery(
                agent=args.agent,
                user_message=args.message,
                context=context,
                config_path=args.config,
            )
        except Exception as exc:
            message = _runtime_error_fallback_message(
                agent=args.agent,
                user_message=args.message,
                error=exc,
            )
            print(dumps_json({"status": "error", "message": message}, indent=2))
            return 1
        print(dumps_json(result, indent=2))
        return 0

    if args.command == "run-agent-session":
        try:
            context = _parse_context(args.context_json)
        except ValueError as exc:
            parser.error(str(exc))
            return 2
        return _run_agent_session(
            agent=args.agent,
            base_context=context,
            config_path=args.config,
            show_json=bool(args.show_json),
            max_turns=int(args.max_turns),
        )

    if args.command == "setup-local-llm":
        try:
            result = setup_local_llm(
                config_path=args.config,
                provider=args.provider,
                profile_name=args.profile,
                model=args.model,
                endpoint=args.endpoint,
                install_provider=bool(args.install_provider),
                start_runtime=not bool(args.no_start_runtime),
                pull_model=not bool(args.no_pull_model),
                download_model=not bool(args.no_download_model),
                llama_model_path=args.llama_model_path,
                llama_model_url=args.llama_model_url,
                timeout_seconds=int(args.timeout_seconds),
            )
        except Exception as exc:
            print(dumps_json({"ready": False, "error": str(exc)}, indent=2))
            return 1
        print(dumps_json(result, indent=2))
        return 0

    if args.command == "run-agent1-analysis":
        try:
            predictor_map = _parse_json_object(args.predictor_map_json, arg_name="--predictor-map-json")
            forced_requests = _parse_json_array(
                args.forced_requests_json, arg_name="--forced-requests-json"
            )
            user_hypotheses = _parse_json_array(
                args.user_hypotheses_json, arg_name="--user-hypotheses-json"
            )
            feature_hypotheses = _parse_json_array(
                args.feature_hypotheses_json, arg_name="--feature-hypotheses-json"
            )
        except ValueError as exc:
            parser.error(str(exc))
            return 2
        registry = build_default_registry()
        tool_args: dict[str, Any] = {
            "data_path": args.data_path,
            "predictor_signals_by_target": predictor_map,
            "forced_requests": forced_requests,
            "user_hypotheses": user_hypotheses,
            "feature_hypotheses": feature_hypotheses,
            "max_lag": int(args.max_lag),
            "include_feature_engineering": not args.no_feature_engineering,
            "feature_gain_threshold": float(args.feature_gain_threshold),
            "confidence_top_k": int(args.confidence_top_k),
            "bootstrap_rounds": int(args.bootstrap_rounds),
            "stability_windows": int(args.stability_windows),
            "max_samples": args.max_samples,
            "sample_selection": args.sample_selection,
            "missing_data_strategy": args.missing_data_strategy,
            "fill_constant_value": args.fill_constant_value,
            "row_coverage_strategy": args.row_coverage_strategy,
            "sparse_row_min_fraction": float(args.sparse_row_min_fraction),
            "row_range_start": args.row_range_start,
            "row_range_end": args.row_range_end,
            "enable_strategy_search": not args.no_strategy_search,
            "strategy_search_candidates": int(args.strategy_search_candidates),
            "save_artifacts": not args.no_save_artifacts,
            "save_report": not args.no_save_report,
        }
        if args.sheet_name:
            tool_args["sheet_name"] = args.sheet_name
        if args.header_row is not None:
            tool_args["header_row"] = int(args.header_row)
        if args.data_start_row is not None:
            tool_args["data_start_row"] = int(args.data_start_row)
        if args.timestamp_column:
            tool_args["timestamp_column"] = args.timestamp_column
        if args.target_signals:
            tool_args["target_signals"] = args.target_signals
        if args.run_id:
            tool_args["run_id"] = args.run_id

        result = registry.execute("run_agent1_analysis", _drop_none_fields(tool_args))
        print(dumps_json(result.output, indent=2))
        return 0

    parser.error(f"Unsupported command '{args.command}'.")
    return 2


def _parse_context(raw: str) -> dict[str, Any]:
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid --context-json value: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError("--context-json must decode to a JSON object.")
    return parsed


def _parse_json_object(raw: str, *, arg_name: str) -> dict[str, Any]:
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid {arg_name} value: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"{arg_name} must decode to a JSON object.")
    return parsed


def _parse_json_array(raw: str, *, arg_name: str) -> list[Any]:
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid {arg_name} value: {exc}") from exc
    if not isinstance(parsed, list):
        raise ValueError(f"{arg_name} must decode to a JSON array.")
    return parsed


def _drop_none_fields(payload: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if value is not None}


def _run_agent_session(
    *,
    agent: str,
    base_context: dict[str, Any],
    config_path: str | None,
    show_json: bool,
    max_turns: int,
) -> int:
    if max_turns < 0:
        print("max-turns must be >= 0")
        return 2

    print(f"agent> Welcome to Corr2Surrogate ({agent} session).")
    if agent == "analyst":
        default_dataset_path = _resolve_default_public_dataset_path()
        print(
            "agent> I can chat, inspect CSV/XLSX data, run Agent 1 analysis, "
            "save reports, and interpret the results."
        )
        print(
            "agent> Runtime mode: local LLM by default. Optional API mode: set "
            "C2S_PROVIDER=openai, C2S_API_CALLS_ALLOWED=true, "
            "C2S_REQUIRE_LOCAL_MODELS=false, C2S_OFFLINE_MODE=false, C2S_API_KEY=<key>."
        )
        print(
            "agent> Useful commands: /help, /context, /reset, /exit. "
            "At target selection use: list, list <filter>, all, or comma-separated names."
        )
        print(
            "agent> Hypothesis syntax: "
            "`hypothesis corr target:pred1,pred2; target2:pred3` and "
            "`hypothesis feature target:signal->rate_change; signal2->square`."
        )
        if default_dataset_path is not None:
            print(
                "agent> Dataset choice: paste a CSV/XLSX path or type `default` "
                "to run the built-in test dataset: "
                f"`{_path_for_display(default_dataset_path)}`."
            )
        else:
            print(
                "agent> Dataset choice: paste a CSV/XLSX path. "
                "(`default` is unavailable because no public test dataset was found.)"
            )
    else:
        default_dataset_path = _resolve_default_public_dataset_path()
        print(
            "agent> I can chat, load CSV/XLSX data, run direct modeler workflows, "
            "and explain the training outcome."
        )
        print(
            "agent> Useful commands: /help, /context, /reset, /exit. "
            "Use `list` after loading data to inspect available signals."
        )
        print(
            "agent> Direct build syntax: "
            "`build model linear_ridge with inputs A,B,C and target D` "
            "or `build model lagged_linear with inputs A,B and target C`."
        )
        print(
            "agent> Current executable models: `auto`, `linear_ridge`, "
            "`lagged_linear`, `lagged_tree_ensemble`, `bagged_tree_ensemble` "
            "(aliases include `ridge`, `linear`, `lagged`, `temporal_linear`, `arx`, "
            "`lagged_tree`, `lag_window_tree`, `temporal_tree`, `tree`, `tree_ensemble`, "
            "`extra_trees`, `hist_gradient_boosting`)."
        )
        print(
            "agent> During training I will print staged progress, compare candidates on "
            "validation/test, and then give an LLM interpretation grounded in those metrics."
        )
        print(
            "agent> Handoff syntax: "
            "`use handoff path\\\\to\\\\structured_report.json` "
            "to load an Agent 1 structured report and then confirm/override target, inputs, and model."
        )
        if default_dataset_path is not None:
            print(
                "agent> Dataset choice: paste a CSV/XLSX path, type `default`, "
                "or give a direct build request first and then provide the dataset."
            )
        else:
            print(
                "agent> Dataset choice: paste a CSV/XLSX path, "
                "or give a direct build request first and then provide the dataset."
            )
    session_messages: list[dict[str, str]] = []
    session_context = dict(base_context)
    if agent == "analyst":
        session_context.setdefault("workflow_stage", "awaiting_dataset_path")
    else:
        session_context.setdefault("workflow_stage", "awaiting_modeler_request_or_dataset")
    registry = build_default_registry()
    turns = 0

    while True:
        try:
            raw_user = input("you> ")
        except EOFError:
            print("\nSession ended.")
            return 0
        except KeyboardInterrupt:
            print("\nSession interrupted.")
            return 0

        user_message = raw_user.strip()
        if not user_message:
            continue

        command = user_message.lower()
        if command in {"/exit", "/quit"}:
            print("Session ended.")
            return 0
        if command == "/help":
            if agent == "analyst":
                default_dataset_path = _resolve_default_public_dataset_path()
                print(
                    "agent> Commands: /help, /context, /reset, /exit. "
                    "For data analysis paste a .csv/.xlsx path."
                )
                print(
                    "agent> During target selection: type list, list <filter>, "
                    "all, numeric index, or comma-separated signal names."
                )
                print(
                    "agent> You can also add hypotheses: "
                    "`hypothesis corr target:pred1,pred2` or "
                    "`hypothesis feature target:signal->rate_change`."
                )
                print(
                    "agent> Local runtime setup: "
                    "`corr2surrogate setup-local-llm --provider llama_cpp --install-provider` "
                    "(Windows) or `corr2surrogate setup-local-llm --provider llama_cpp` (macOS/Linux)."
                )
                print(
                    "agent> API mode (optional): set C2S_PROVIDER=openai, "
                    "C2S_API_CALLS_ALLOWED=true, C2S_REQUIRE_LOCAL_MODELS=false, "
                    "C2S_OFFLINE_MODE=false, C2S_API_KEY=<key>, then restart or /reset."
                )
                if default_dataset_path is not None:
                    print(
                        "agent> Type `default` to analyze the built-in public test dataset."
                    )
            else:
                default_dataset_path = _resolve_default_public_dataset_path()
                print("agent> Commands: /help, /context, /reset, /exit.")
                print(
                    "agent> Load a dataset by pasting a CSV/XLSX path, "
                    "or type `default` to use the built-in public test dataset."
                    if default_dataset_path is not None
                    else "agent> Load a dataset by pasting a CSV/XLSX path."
                )
                print(
                    "agent> Direct build syntax: "
                    "`build model linear_ridge with inputs A,B,C and target D` "
                    "or `build model lagged_linear with inputs A,B and target C`."
                )
                print(
                    "agent> Current executable models: `auto`, `linear_ridge`, "
                    "`lagged_linear`, `lagged_tree_ensemble`, `bagged_tree_ensemble` "
                    "(aliases include `ridge`, `linear`, `lagged`, `temporal_linear`, `arx`, "
                    "`lagged_tree`, `lag_window_tree`, `temporal_tree`, `tree`, `tree_ensemble`, "
                    "`extra_trees`, `hist_gradient_boosting`)."
                )
                print(
                    "agent> You can also load an Agent 1 handoff via: "
                    "`use handoff path\\\\to\\\\structured_report.json`."
                )
            continue
        if command == "/context":
            snapshot = dict(session_context)
            snapshot["session_messages"] = session_messages
            print(dumps_json(snapshot, indent=2))
            continue
        if command == "/reset":
            session_context = dict(base_context)
            if agent == "analyst":
                session_context["workflow_stage"] = "awaiting_dataset_path"
            session_messages = []
            print("Session state reset.")
            continue

        if agent == "analyst":
            def _chat_reply_only(detour_user_message: str) -> str:
                return _llm_chat_detour(
                    agent=agent,
                    user_message=detour_user_message,
                    session_context=session_context,
                    session_messages=session_messages,
                    config_path=config_path,
                )

            def _chat_reply_internal(detour_user_message: str) -> str:
                return _llm_chat_detour(
                    agent=agent,
                    user_message=detour_user_message,
                    session_context=session_context,
                    session_messages=session_messages,
                    config_path=config_path,
                    record_in_history=False,
                )

            def _chat_detour_with_reprompt(detour_user_message: str, reminder: str) -> None:
                reply = _chat_reply_only(detour_user_message)
                if reply:
                    print(f"agent> {reply}")
                print(f"agent> {reminder}")

            try:
                autopilot = _run_analyst_autopilot_turn(
                    user_message=user_message,
                    registry=registry,
                    chat_detour=_chat_detour_with_reprompt,
                    chat_reply_only=_chat_reply_internal,
                )
            except Exception as exc:
                response = _runtime_error_fallback_message(
                    agent=agent,
                    user_message=user_message,
                    error=exc,
                )
                print(f"agent> {response}")
                session_messages.append({"role": "user", "content": user_message})
                session_messages.append({"role": "assistant", "content": response})
                session_messages = session_messages[-20:]
                session_context["last_event"] = _compact_event_for_context(
                    {"status": "respond", "message": response, "error": "runtime_error"}
                )
                turns += 1
                if max_turns > 0 and turns >= max_turns:
                    print(f"Reached max turns ({max_turns}). Session ended.")
                    return 0
                continue
            if autopilot is not None:
                response = autopilot["response"]
                summary_event = autopilot["event"]
                if summary_event.get("error"):
                    print(f"agent> {response}")
                    session_context["workflow_stage"] = "awaiting_dataset_path"
                session_messages.append({"role": "user", "content": user_message})
                session_messages.append({"role": "assistant", "content": response})
                session_messages = session_messages[-20:]
                session_context["last_event"] = _compact_event_for_context(summary_event)
                if not summary_event.get("error"):
                    session_context["workflow_stage"] = "analysis_completed"
                    tool_output = summary_event.get("tool_output")
                    if isinstance(tool_output, dict):
                        report_path = tool_output.get("report_path")
                        if isinstance(report_path, str) and report_path.strip():
                            session_context["last_report_path"] = report_path
                turns += 1
                if max_turns > 0 and turns >= max_turns:
                    print(f"Reached max turns ({max_turns}). Session ended.")
                    return 0
                continue

        if agent == "modeler":
            def _modeler_chat_reply_internal(detour_user_message: str) -> str:
                return _llm_chat_detour(
                    agent=agent,
                    user_message=detour_user_message,
                    session_context=session_context,
                    session_messages=session_messages,
                    config_path=config_path,
                    record_in_history=False,
                )

            try:
                autopilot = _run_modeler_autopilot_turn(
                    user_message=user_message,
                    registry=registry,
                    session_context=session_context,
                    chat_reply_only=_modeler_chat_reply_internal,
                )
            except Exception as exc:
                response = _runtime_error_fallback_message(
                    agent=agent,
                    user_message=user_message,
                    error=exc,
                )
                print(f"agent> {response}")
                session_messages.append({"role": "user", "content": user_message})
                session_messages.append({"role": "assistant", "content": response})
                session_messages = session_messages[-20:]
                session_context["last_event"] = _compact_event_for_context(
                    {"status": "respond", "message": response, "error": "runtime_error"}
                )
                turns += 1
                if max_turns > 0 and turns >= max_turns:
                    print(f"Reached max turns ({max_turns}). Session ended.")
                    return 0
                continue
            if autopilot is not None:
                response = autopilot["response"]
                summary_event = autopilot["event"]
                session_messages.append({"role": "user", "content": user_message})
                session_messages.append({"role": "assistant", "content": response})
                session_messages = session_messages[-20:]
                session_context["last_event"] = _compact_event_for_context(summary_event)
                turns += 1
                if max_turns > 0 and turns >= max_turns:
                    print(f"Reached max turns ({max_turns}). Session ended.")
                    return 0
                continue

        turn_context = dict(session_context)
        turn_context["session_messages"] = list(session_messages)
        turn_context["recent_user_prompts"] = _recent_user_prompts(
            session_messages=session_messages,
            limit=5,
        )
        try:
            result = _invoke_agent_once_with_recovery(
                agent=agent,
                user_message=user_message,
                context=turn_context,
                config_path=config_path,
            )
        except Exception as exc:
            response = _runtime_error_fallback_message(
                agent=agent,
                user_message=user_message,
                error=exc,
            )
            print(f"agent> {response}")
            session_messages.append({"role": "user", "content": user_message})
            session_messages.append({"role": "assistant", "content": response})
            session_messages = session_messages[-20:]
            session_context["last_event"] = _compact_event_for_context(
                {"status": "respond", "message": response, "error": "runtime_error"}
            )
            turns += 1
            if max_turns > 0 and turns >= max_turns:
                print(f"Reached max turns ({max_turns}). Session ended.")
                return 0
            continue

        event = result.get("event", {})
        response = str(event.get("message", "")).strip() or "[empty response]"
        response = _rewrite_unhelpful_response(
            agent=agent,
            user_message=user_message,
            response=response,
            chat_detour=(
                _chat_reply_only
                if agent == "analyst"
                else None
            ),
        )
        stage_reminder = _analyst_stage_reprompt_message(
            agent=agent,
            session_context=session_context,
            user_message=user_message,
        )
        print(f"agent> {response}")
        if stage_reminder:
            print(f"agent> {stage_reminder}")
            response = f"{response}\n{stage_reminder}"
        if show_json:
            print(dumps_json(result, indent=2))

        session_messages.append({"role": "user", "content": user_message})
        session_messages.append({"role": "assistant", "content": response})
        session_messages = session_messages[-20:]
        session_context["last_event"] = _compact_event_for_context(event)

        turns += 1
        if max_turns > 0 and turns >= max_turns:
            print(f"Reached max turns ({max_turns}). Session ended.")
            return 0


def _analyst_stage_reprompt_message(
    *,
    agent: str,
    session_context: dict[str, Any],
    user_message: str,
) -> str:
    stage = str(session_context.get("workflow_stage", "")).strip().lower()
    lowered = user_message.strip().lower()
    if agent == "analyst":
        if stage != "awaiting_dataset_path":
            return ""
        if lowered == "default":
            return ""
        if _extract_first_data_path(user_message) is not None:
            return ""
        return (
            "To continue, paste a CSV/XLSX path or type `default` "
            "to run the built-in test dataset."
        )
    if agent == "modeler":
        if stage == "awaiting_modeler_dataset_path":
            if lowered == "default":
                return ""
            if _extract_first_data_path(user_message) is not None:
                return ""
            return (
                "To continue, paste a CSV/XLSX path or type `default` "
                "so I can load data for the pending model request."
            )
        if stage == "modeler_dataset_ready":
            return (
                "To continue, type `list` to inspect signals or "
                "`build model linear_ridge with inputs A,B and target C`."
            )
    return ""


def _run_analyst_autopilot_turn(
    *,
    user_message: str,
    registry: Any,
    chat_detour: Callable[[str, str], None] | None = None,
    chat_reply_only: Callable[[str], str] | None = None,
) -> dict[str, Any] | None:
    detected: Path | None = None
    if user_message.strip().lower() == "default":
        detected = _resolve_default_public_dataset_path()
        if detected is None:
            response = (
                "Default dataset is not available. "
                "Please paste a CSV/XLSX path from your machine."
            )
            return {
                "response": response,
                "event": {
                    "status": "respond",
                    "message": response,
                    "error": "default_dataset_missing",
                },
            }
    else:
        detected = _extract_first_data_path(user_message)
    if detected is None:
        return None

    data_path = str(detected.resolve())
    if not detected.exists():
        response = f"Detected data path but file does not exist: {data_path}"
        return {
            "response": response,
            "event": {"status": "respond", "message": response, "error": "missing_path"},
        }

    print(f"agent> Detected data file: {_path_for_display(Path(data_path))}")
    preflight_args: dict[str, Any] = {"path": data_path}
    preflight = _execute_registry_tool(registry, "prepare_ingestion_step", preflight_args)
    print(f"agent> Ingestion check: {preflight.get('message', '')}")

    if preflight.get("status") == "needs_user_input":
        options = preflight.get("options") or preflight.get("available_sheets") or []
        selected_sheet = _prompt_sheet_selection(options, chat_detour=chat_detour)
        if selected_sheet is None:
            response = "Sheet selection aborted. Please provide a valid sheet."
            return {
                "response": response,
                "event": {"status": "respond", "message": response, "error": "sheet_selection_aborted"},
            }
        preflight_args["sheet_name"] = selected_sheet
        preflight = _execute_registry_tool(registry, "prepare_ingestion_step", preflight_args)
        print(f"agent> Ingestion check: {preflight.get('message', '')}")

    if preflight.get("status") != "ok":
        response = preflight.get("message") or "Ingestion failed."
        return {
            "response": response,
            "event": {"status": "respond", "message": response, "error": "ingestion_error"},
        }

    selected_sheet = preflight.get("selected_sheet")
    header_row = preflight.get("header_row")
    data_start_row = preflight.get("data_start_row")

    inferred_header_row = preflight.get("header_row")
    wide_table = int(preflight.get("column_count") or 0) >= 50
    force_header_check = (
        isinstance(inferred_header_row, int) and inferred_header_row > 0 and wide_table
    )
    if bool(preflight.get("needs_user_confirmation")) or force_header_check:
        confidence = preflight.get("header_confidence")
        if bool(preflight.get("needs_user_confirmation")):
            print(
                "agent> Header detection confidence is low "
                f"({confidence if confidence is not None else 'n/a'})."
            )
        if force_header_check and not bool(preflight.get("needs_user_confirmation")):
            print(
                "agent> Wide table with non-zero header row detected; "
                "please confirm inferred rows before analysis."
            )
        _print_header_preview(preflight)
        resolved_header_row = int(header_row) if isinstance(header_row, int) else 0
        resolved_data_start = (
            int(data_start_row)
            if isinstance(data_start_row, int)
            else max(resolved_header_row + 1, 1)
        )
        header_row, data_start_row = _prompt_header_confirmation(
            header_row=resolved_header_row,
            data_start_row=resolved_data_start,
            chat_detour=chat_detour,
        )

        preflight_args["header_row"] = int(header_row)
        preflight_args["data_start_row"] = int(data_start_row)
        preflight = _execute_registry_tool(registry, "prepare_ingestion_step", preflight_args)
        print(f"agent> Ingestion check: {preflight.get('message', '')}")
        _print_header_preview(preflight)
        if preflight.get("status") != "ok":
            response = preflight.get("message") or "Ingestion confirmation failed."
            return {
                "response": response,
                "event": {
                    "status": "respond",
                    "message": response,
                    "error": "ingestion_confirmation_error",
                },
            }

    print("agent> Running Agent 1 analysis...")
    analysis_args: dict[str, Any] = {
        "data_path": data_path,
        "save_report": True,
        "save_artifacts": True,
        "enable_strategy_search": True,
        "strategy_search_candidates": 4,
        "include_feature_engineering": True,
        "top_k_predictors": 10,
        "feature_scan_predictors": 10,
        "max_feature_opportunities": 20,
        "confidence_top_k": 10,
        "bootstrap_rounds": 40,
        "stability_windows": 4,
    }
    hypothesis_state: dict[str, list[dict[str, Any]]] = {
        "user_hypotheses": [],
        "feature_hypotheses": [],
    }
    row_count = int(preflight.get("row_count") or 0)
    sample_plan = _prompt_sample_budget(row_count=row_count, chat_detour=chat_detour)
    analysis_args.update(sample_plan)

    data_issue_plan = _prompt_data_issue_handling(preflight=preflight, chat_detour=chat_detour)
    analysis_args.update(data_issue_plan)

    numeric_signals = [str(item) for item in (preflight.get("numeric_signal_columns") or [])]
    if len(numeric_signals) > 40:
        print(
            "agent> Detected "
            f"{len(numeric_signals)} numeric signals. Full all-signal correlation can take a long time."
        )
        print(
            "agent> Enter comma-separated target signals to focus, "
            "'all' for full run, 'list' to show signal names, "
            "or `hypothesis ...` to add correlation/feature hypotheses; "
            "or press Enter to use a quick default subset."
        )
        selected_targets = _prompt_target_selection(
            available_signals=numeric_signals,
            default_count=5,
            hypothesis_state=hypothesis_state,
            chat_detour=chat_detour,
        )
        if selected_targets is not None:
            analysis_args["target_signals"] = selected_targets
            print(f"agent> Using focused targets: {selected_targets}")
            print(
                "agent> Focused target mode enabled with full analysis "
                "(multi-technique correlations + feature engineering)."
            )
        else:
            print("agent> Running full all-signal analysis as requested.")

    inline_hypotheses = _parse_inline_hypothesis_command(
        user_message=user_message,
        available_signals=numeric_signals,
    )
    if inline_hypotheses["user_hypotheses"] or inline_hypotheses["feature_hypotheses"]:
        _merge_hypothesis_state(hypothesis_state, inline_hypotheses)
        print(
            "agent> Parsed inline hypotheses from your request: "
            f"correlation={len(inline_hypotheses['user_hypotheses'])}, "
            f"feature={len(inline_hypotheses['feature_hypotheses'])}."
        )

    if hypothesis_state["user_hypotheses"] or hypothesis_state["feature_hypotheses"]:
        analysis_args["user_hypotheses"] = hypothesis_state["user_hypotheses"]
        analysis_args["feature_hypotheses"] = hypothesis_state["feature_hypotheses"]
        print(
            "agent> User hypotheses will be investigated additionally: "
            f"correlation={len(hypothesis_state['user_hypotheses'])}, "
            f"feature={len(hypothesis_state['feature_hypotheses'])}."
        )

    timestamp_hint = str(preflight.get("timestamp_column_hint") or "").strip()
    estimated_sample_period = _safe_float_or_none(preflight.get("estimated_sample_period_seconds"))
    if timestamp_hint:
        lag_plan = _prompt_lag_preferences(
            timestamp_column_hint=timestamp_hint,
            estimated_sample_period_seconds=estimated_sample_period,
            chat_detour=chat_detour,
        )
        analysis_args["timestamp_column"] = timestamp_hint
        analysis_args["max_lag"] = int(lag_plan["max_lag"])
        print(
            "agent> Lag plan: "
            f"enabled={lag_plan['enabled']}, "
            f"dimension={lag_plan['dimension']}, "
            f"max_lag_samples={lag_plan['max_lag']}."
        )

    if selected_sheet:
        analysis_args["sheet_name"] = str(selected_sheet)
    if header_row is not None:
        analysis_args["header_row"] = int(header_row)
    if data_start_row is not None:
        analysis_args["data_start_row"] = int(data_start_row)

    analysis = _execute_registry_tool(registry, "run_agent1_analysis", analysis_args)
    summary = (
        "Analysis complete: "
        f"data_mode={analysis.get('data_mode', 'unknown')}, "
        f"targets={analysis.get('target_count', 'n/a')}, "
        f"candidates={analysis.get('candidate_count', 'n/a')}."
    )
    report_path = str(analysis.get("report_path", "n/a"))
    report_line = f"Report saved: {report_path}"
    print(f"agent> {summary}")
    print(f"agent> {report_line}")
    top3_correlations = _extract_top3_correlations_global(analysis)
    if chat_reply_only is not None:
        interpretation = _generate_analysis_interpretation(
            analysis=analysis,
            chat_reply_only=chat_reply_only,
        )
        if interpretation:
            print("agent> LLM interpretation:")
            for line in interpretation.splitlines():
                text = line.strip()
                if text:
                    print(f"agent> {text}")
            if top3_correlations and not _interpretation_mentions_top3(
                interpretation=interpretation,
                top3=top3_correlations,
            ):
                print(f"agent> {_format_top3_correlations_line(top3_correlations)}")
        elif top3_correlations:
            print(
                "agent> LLM interpretation unavailable for this turn. "
                "Showing deterministic correlation summary."
            )
            print(f"agent> {_format_top3_correlations_line(top3_correlations)}")

    response = f"{summary} {report_line}"
    event = {
        "status": "respond",
        "message": response,
        "tool_output": {
            "data_mode": analysis.get("data_mode"),
            "target_count": analysis.get("target_count"),
            "candidate_count": analysis.get("candidate_count"),
            "report_path": report_path,
        },
    }
    return {"response": response, "event": event}


def _run_modeler_autopilot_turn(
    *,
    user_message: str,
    registry: Any,
    session_context: dict[str, Any],
    chat_reply_only: Callable[[str], str] | None = None,
) -> dict[str, Any] | None:
    stripped = user_message.strip()
    lowered = stripped.lower()

    if lowered.startswith(("use handoff", "load handoff")):
        handoff_path = _extract_first_json_path(user_message)
        if handoff_path is None:
            response = (
                "I did not detect a handoff JSON path. "
                "Use: `use handoff path\\to\\structured_report.json`."
            )
            print(f"agent> {response}")
            return {
                "response": response,
                "event": {"status": "respond", "message": response, "error": "handoff_path_missing"},
            }
        handoff = _load_modeler_handoff_payload(handoff_path)
        if isinstance(handoff.get("error"), str):
            response = str(handoff["error"])
            print(f"agent> {response}")
            return {
                "response": response,
                "event": {"status": "respond", "message": response, "error": "handoff_load_error"},
            }

        data_path = str(handoff.get("data_path", "")).strip()
        if not data_path:
            response = (
                "The handoff file does not contain a usable `data_path`. "
                "Provide a dataset path directly or generate a newer Agent 1 structured report."
            )
            print(f"agent> {response}")
            return {
                "response": response,
                "event": {"status": "respond", "message": response, "error": "handoff_missing_data_path"},
            }

        dataset_result = _prepare_modeler_dataset_for_session(
            path=data_path,
            registry=registry,
            session_context=session_context,
        )
        if dataset_result.get("error"):
            response = str(dataset_result["error"])
            return {
                "response": response,
                "event": {"status": "respond", "message": response, "error": "modeler_dataset_error"},
            }

        dataset = _modeler_loaded_dataset(session_context)
        if dataset is None:
            response = "Modeler dataset state could not be initialized."
            print(f"agent> {response}")
            return {
                "response": response,
                "event": {"status": "respond", "message": response, "error": "modeler_state_error"},
            }

        build_request = _modeler_request_from_handoff(
            payload=handoff["payload"],
            handoff=handoff["handoff"],
            available_signals=list(dataset.get("signal_columns", [])),
        )
        if build_request is None:
            response = (
                "I could not derive a usable modeling request from that handoff. "
                "Please specify target, inputs, and model directly."
            )
            print(f"agent> {response}")
            return {
                "response": response,
                "event": {"status": "respond", "message": response, "error": "handoff_parse_error"},
            }

        session_context["active_handoff"] = handoff["handoff"]
        handoff_info = handoff["handoff"]
        print(
            "agent> Handoff contract: "
            f"data_mode=`{handoff_info.get('dataset_profile', {}).get('data_mode', 'n/a')}`, "
            f"split=`{handoff_info.get('split_strategy', 'n/a')}`, "
            f"acceptance={handoff_info.get('acceptance_criteria', {})}."
        )
        print(
            "agent> Handoff suggestion: "
            f"target=`{build_request['target_raw']}`, "
            f"inputs={build_request['feature_raw']}, "
            f"recommended_model=`{build_request['requested_model_family']}`."
        )
        build_request = _prompt_modeler_overrides(
            request=build_request,
            available_signals=list(dataset.get("signal_columns", [])),
        )
        return _execute_modeler_build_request(
            build_request=build_request,
            registry=registry,
            session_context=session_context,
            chat_reply_only=chat_reply_only,
        )

    parsed_request = _parse_modeler_build_request(user_message)
    if parsed_request is not None:
        requested_data_path = str(parsed_request.get("data_path", "")).strip()
        if requested_data_path:
            dataset_result = _prepare_modeler_dataset_for_session(
                path=requested_data_path,
                registry=registry,
                session_context=session_context,
            )
            if dataset_result.get("error"):
                response = str(dataset_result["error"])
                return {
                    "response": response,
                    "event": {
                        "status": "respond",
                        "message": response,
                        "error": "modeler_dataset_error",
                    },
                }

        if _modeler_loaded_dataset(session_context) is None:
            session_context["pending_model_request"] = parsed_request
            session_context["workflow_stage"] = "awaiting_modeler_dataset_path"
            response = (
                "I parsed your model request. To continue, paste a CSV/XLSX path "
                "or type `default` so I can load the training dataset first."
            )
            print(f"agent> {response}")
            return {
                "response": response,
                "event": {"status": "respond", "message": response},
            }
        return _execute_modeler_build_request(
            build_request=parsed_request,
            registry=registry,
            session_context=session_context,
            chat_reply_only=chat_reply_only,
        )

    if lowered.startswith("list"):
        dataset = _modeler_loaded_dataset(session_context)
        if dataset is None:
            response = (
                "Load a dataset first, then I can show available signal names. "
                "Paste a CSV/XLSX path or type `default`."
            )
            print(f"agent> {response}")
            return {
                "response": response,
                "event": {"status": "respond", "message": response},
            }
        query = stripped[4:].strip()
        _print_signal_names(list(dataset.get("signal_columns", [])), query=query)
        response = "Signal list displayed."
        return {
            "response": response,
            "event": {"status": "respond", "message": response},
        }

    detected: Path | None = None
    if lowered == "default":
        detected = _resolve_default_public_dataset_path()
        if detected is None:
            response = (
                "Default dataset is not available. "
                "Please paste a CSV/XLSX path from your machine."
            )
            print(f"agent> {response}")
            return {
                "response": response,
                "event": {"status": "respond", "message": response, "error": "default_dataset_missing"},
            }
    else:
        detected = _extract_first_data_path(user_message)

    if detected is None:
        return None

    dataset_result = _prepare_modeler_dataset_for_session(
        path=str(detected),
        registry=registry,
        session_context=session_context,
    )
    if dataset_result.get("error"):
        response = str(dataset_result["error"])
        return {
            "response": response,
            "event": {"status": "respond", "message": response, "error": "modeler_dataset_error"},
        }

    pending_request = session_context.pop("pending_model_request", None)
    if isinstance(pending_request, dict):
        print("agent> Continuing with your pending model request.")
        return _execute_modeler_build_request(
            build_request=pending_request,
            registry=registry,
            session_context=session_context,
            chat_reply_only=chat_reply_only,
        )

    response = (
        "Dataset ready. Type `list` to inspect signals, "
        "or `build model linear_ridge with inputs A,B and target C` to train."
    )
    print(f"agent> {response}")
    return {
        "response": response,
        "event": {
            "status": "respond",
            "message": response,
            "tool_output": {
                "data_path": str((detected if lowered != 'default' else _resolve_default_public_dataset_path()) or detected),
                "signal_count": len((_modeler_loaded_dataset(session_context) or {}).get("signal_columns", [])),
            },
        },
    }


def _prepare_modeler_dataset_for_session(
    *,
    path: str,
    registry: Any,
    session_context: dict[str, Any],
) -> dict[str, Any]:
    data_path = str(Path(path).expanduser())
    detected = Path(data_path)
    if not detected.exists():
        response = f"Detected data path but file does not exist: {data_path}"
        print(f"agent> {response}")
        return {"error": response}

    print(f"agent> Detected data file: {_path_for_display(detected)}")
    preflight_args: dict[str, Any] = {"path": data_path}
    preflight = _execute_registry_tool(registry, "prepare_ingestion_step", preflight_args)
    print(f"agent> Ingestion check: {preflight.get('message', '')}")

    if preflight.get("status") == "needs_user_input":
        options = preflight.get("options") or preflight.get("available_sheets") or []
        selected_sheet = _prompt_sheet_selection(options)
        if selected_sheet is None:
            return {"error": "Sheet selection aborted. Please provide a valid sheet."}
        preflight_args["sheet_name"] = selected_sheet
        preflight = _execute_registry_tool(registry, "prepare_ingestion_step", preflight_args)
        print(f"agent> Ingestion check: {preflight.get('message', '')}")

    if preflight.get("status") != "ok":
        return {"error": preflight.get("message") or "Ingestion failed."}

    selected_sheet = preflight.get("selected_sheet")
    header_row = preflight.get("header_row")
    data_start_row = preflight.get("data_start_row")
    inferred_header_row = preflight.get("header_row")
    wide_table = int(preflight.get("column_count") or 0) >= 50
    force_header_check = isinstance(inferred_header_row, int) and inferred_header_row > 0 and wide_table
    if bool(preflight.get("needs_user_confirmation")) or force_header_check:
        if bool(preflight.get("needs_user_confirmation")):
            print(
                "agent> Header detection confidence is low "
                f"({preflight.get('header_confidence', 'n/a')})."
            )
        if force_header_check and not bool(preflight.get("needs_user_confirmation")):
            print(
                "agent> Wide table with non-zero header row detected; "
                "please confirm inferred rows before modeling."
            )
        _print_header_preview(preflight)
        resolved_header_row = int(header_row) if isinstance(header_row, int) else 0
        resolved_data_start = (
            int(data_start_row)
            if isinstance(data_start_row, int)
            else max(resolved_header_row + 1, 1)
        )
        header_row, data_start_row = _prompt_header_confirmation(
            header_row=resolved_header_row,
            data_start_row=resolved_data_start,
        )
        preflight_args["header_row"] = int(header_row)
        preflight_args["data_start_row"] = int(data_start_row)
        preflight = _execute_registry_tool(registry, "prepare_ingestion_step", preflight_args)
        print(f"agent> Ingestion check: {preflight.get('message', '')}")
        _print_header_preview(preflight)
        if preflight.get("status") != "ok":
            return {"error": preflight.get("message") or "Ingestion confirmation failed."}

    dataset = {
        "data_path": data_path,
        "sheet_name": selected_sheet,
        "header_row": header_row,
        "data_start_row": data_start_row,
        "timestamp_column_hint": str(preflight.get("timestamp_column_hint") or "").strip() or None,
        "signal_columns": [str(item) for item in (preflight.get("signal_columns") or [])],
        "numeric_signal_columns": [
            str(item) for item in (preflight.get("numeric_signal_columns") or [])
        ],
        "row_count": int(preflight.get("row_count") or 0),
    }
    session_context["modeler_dataset"] = dataset
    session_context["workflow_stage"] = "modeler_dataset_ready"
    print(
        "agent> Dataset ready: "
        f"rows={dataset['row_count']}, "
        f"signals={len(dataset['signal_columns'])}, "
        f"numeric_signals={len(dataset['numeric_signal_columns'])}."
    )
    return {"dataset": dataset}


def _modeler_loaded_dataset(session_context: dict[str, Any]) -> dict[str, Any] | None:
    dataset = session_context.get("modeler_dataset")
    return dataset if isinstance(dataset, dict) else None


def _parse_modeler_build_request(user_message: str) -> dict[str, Any] | None:
    text = user_message.strip()
    pattern = re.compile(
        r"^\s*(?:build|train)(?:\s+me)?\s+model\s+(?P<model>[A-Za-z0-9_\-]+)\s+"
        r"with\s+(?:inputs|inouts|features|predictors)\s+(?P<inputs>.+?)\s+"
        r"and\s+target\s+(?P<target>.+?)"
        r"(?:\s+(?:using|from|on)(?:\s+data)?\s+.+)?\s*$",
        flags=re.IGNORECASE,
    )
    match = pattern.match(text)
    if not match:
        return None
    inputs_raw = match.group("inputs").strip()
    target_raw = _strip_wrapping_quotes(match.group("target").strip())
    feature_raw = _split_modeler_input_tokens(inputs_raw)
    if not feature_raw or not target_raw:
        return None
    data_path = _extract_first_data_path(user_message)
    requested_model_family = match.group("model").strip()
    normalized_model = _normalize_modeler_model_family(requested_model_family)
    return {
        "requested_model_family": requested_model_family,
        "feature_raw": feature_raw,
        "target_raw": target_raw,
        "data_path": str(data_path) if data_path is not None else "",
        "acceptance_criteria": {"r2": 0.70},
        "loop_policy": {
            "enabled": True,
            "max_attempts": 2,
            "allow_architecture_switch": True,
        },
        "user_locked_model_family": normalized_model not in {None, "auto"},
        "source": "direct",
    }


def _split_modeler_input_tokens(raw: str) -> list[str]:
    text = raw.strip()
    if not text:
        return []
    if "," in text or "|" in text:
        parts = [item.strip() for item in re.split(r"[,|]", text) if item.strip()]
    else:
        parts = [item.strip() for item in text.split() if item.strip()]
    return [_strip_wrapping_quotes(item) for item in parts if _strip_wrapping_quotes(item)]


def _strip_wrapping_quotes(value: str) -> str:
    text = value.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {"'", '"'}:
        return text[1:-1].strip()
    return text


def _execute_modeler_build_request(
    *,
    build_request: dict[str, Any],
    registry: Any,
    session_context: dict[str, Any],
    chat_reply_only: Callable[[str], str] | None = None,
) -> dict[str, Any]:
    dataset = _modeler_loaded_dataset(session_context)
    if dataset is None:
        response = "No dataset is loaded. Paste a CSV/XLSX path or type `default` first."
        print(f"agent> {response}")
        session_context["workflow_stage"] = "awaiting_modeler_dataset_path"
        return {
            "response": response,
            "event": {"status": "respond", "message": response, "error": "dataset_missing"},
        }

    numeric_signals = list(dataset.get("numeric_signal_columns", []))
    if not numeric_signals:
        response = (
            "The loaded dataset does not expose usable numeric signals for training. "
            "Please load another dataset or adjust the header selection."
        )
        print(f"agent> {response}")
        return {
            "response": response,
            "event": {"status": "respond", "message": response, "error": "no_numeric_signals"},
        }

    target = _resolve_signal_name(str(build_request.get("target_raw", "")).strip(), numeric_signals)
    if target is None:
        response = (
            "I could not resolve the requested target signal in the loaded dataset. "
            "Type `list` to inspect signal names and retry."
        )
        print(f"agent> {response}")
        return {
            "response": response,
            "event": {"status": "respond", "message": response, "error": "target_unresolved"},
        }

    raw_features = [str(item).strip() for item in build_request.get("feature_raw", [])]
    features: list[str] = []
    unknown_features: list[str] = []
    seen: set[str] = set()
    for raw in raw_features:
        resolved = _resolve_signal_name(raw, numeric_signals)
        if resolved is None:
            unknown_features.append(raw)
            continue
        if resolved == target or resolved in seen:
            continue
        features.append(resolved)
        seen.add(resolved)
    if unknown_features:
        print(f"agent> Ignoring unknown input signals: {unknown_features}")
    if not features:
        response = (
            "I did not resolve any usable numeric input signals after validation. "
            "Type `list` to inspect signal names and retry."
        )
        print(f"agent> {response}")
        return {
            "response": response,
            "event": {"status": "respond", "message": response, "error": "features_unresolved"},
        }

    requested_model = str(build_request.get("requested_model_family", "")).strip() or "linear_ridge"
    resolved_model = _normalize_modeler_model_family(requested_model)
    if resolved_model is None:
        response = (
            f"Requested model `{requested_model}` is not implemented yet. "
            "Currently available: `auto`, `linear_ridge` "
            "(aliases: `ridge`, `linear`, `incremental_linear_surrogate`), "
            "`lagged_linear` (aliases: `lagged`, `temporal_linear`, `arx`), "
            "`lagged_tree_ensemble` (aliases: `lagged_tree`, `lag_window_tree`, `temporal_tree`), and "
            "`bagged_tree_ensemble` (aliases: `tree`, `tree_ensemble`, `extra_trees`, `hist_gradient_boosting`)."
        )
        print(f"agent> {response}")
        session_context["workflow_stage"] = "modeler_dataset_ready"
        return {
            "response": response,
            "event": {"status": "respond", "message": response, "error": "model_not_implemented"},
        }

    requested_normalize = bool(build_request.get("normalize", True))
    missing_data_strategy = str(build_request.get("missing_data_strategy", "fill_median")).strip() or "fill_median"
    fill_constant_value = build_request.get("fill_constant_value")
    compare_against_baseline = bool(build_request.get("compare_against_baseline", True))
    lag_horizon_samples = build_request.get("lag_horizon_samples")
    timestamp_column = str(
        build_request.get("timestamp_column")
        or dataset.get("timestamp_column_hint")
        or ""
    ).strip() or None
    acceptance_criteria = _safe_acceptance_criteria(build_request.get("acceptance_criteria"))
    loop_policy = _safe_loop_policy(build_request.get("loop_policy"))
    user_locked_model_family = bool(build_request.get("user_locked_model_family", False))
    raw_search_order = [str(item).strip() for item in build_request.get("model_search_order", []) if str(item).strip()]

    print(
        "agent> Training request: "
        f"model=`{resolved_model}`, target=`{target}`, inputs={features}."
    )
    if build_request.get("source") == "handoff" and raw_search_order:
        print(
            "agent> Handoff prior: "
            f"search_order={raw_search_order}, normalize={requested_normalize}, "
            f"missing_data={missing_data_strategy}."
        )
    if timestamp_column:
        print(f"agent> Timestamp context: using `{timestamp_column}` for data-mode-aware splitting.")

    attempt = 1
    max_attempts = int(loop_policy.get("max_attempts", 2))
    allow_loop = bool(loop_policy.get("enabled", True))
    allow_architecture_switch = bool(loop_policy.get("allow_architecture_switch", True))
    current_requested_model = resolved_model
    tried_models: set[str] = set()
    last_training: dict[str, Any] | None = None
    last_loop_eval: dict[str, Any] | None = None

    while True:
        tried_models.add(current_requested_model)
        print(
            f"agent> Attempt {attempt}/{max_attempts}: requested candidate family `{current_requested_model}`."
        )
        print("agent> Step 1/3: building split-safe train/validation/test partitions.")
        print("agent> Step 2/3: fitting train-only preprocessing (missing-data handling and optional normalization).")
        print(
            "agent> Step 3/3: training the linear baseline and available temporal/nonlinear comparators, "
            "then selecting the requested/best candidate."
        )
        tool_args = _modeler_training_tool_args(
            dataset=dataset,
            target=target,
            features=features,
            requested_model_family=current_requested_model,
            timestamp_column=timestamp_column,
            requested_normalize=requested_normalize,
            missing_data_strategy=missing_data_strategy,
            fill_constant_value=fill_constant_value,
            compare_against_baseline=compare_against_baseline,
            lag_horizon_samples=lag_horizon_samples,
            checkpoint_tag=f"modeler_session_attempt_{attempt}",
        )
        try:
            training = _execute_registry_tool(registry, "train_surrogate_candidates", tool_args)
        except Exception as exc:
            response = str(exc).strip() or _runtime_error_fallback_message(
                agent="modeler",
                user_message=f"train {current_requested_model}",
                error=exc,
            )
            print(f"agent> {response}")
            return {
                "response": response,
                "event": {"status": "respond", "message": response, "error": "training_runtime_error"},
            }
        if str(training.get("status", "")).lower() != "ok":
            response = str(training.get("message") or "Model training failed.")
            print(f"agent> {response}")
            return {
                "response": response,
                "event": {"status": "respond", "message": response, "error": "training_failed"},
            }

        last_training = training
        _print_modeler_training_summary(training=training)
        metrics_payload = _build_model_loop_metrics(training)
        try:
            loop_eval = _execute_registry_tool(
                registry,
                "evaluate_training_iteration",
                {
                    "metrics": metrics_payload,
                    "acceptance_criteria": acceptance_criteria,
                    "attempt": attempt,
                    "max_attempts": max_attempts,
                },
            )
        except Exception:
            loop_eval = {
                "should_continue": False,
                "attempt": attempt,
                "max_attempts": max_attempts,
                "unmet_criteria": [],
                "recommendations": [],
                "summary": "Acceptance loop evaluation failed; keeping the current measured result.",
            }
        last_loop_eval = loop_eval if isinstance(loop_eval, dict) else {}
        if last_loop_eval:
            summary_line = str(last_loop_eval.get("summary", "")).strip()
            if summary_line:
                print(f"agent> Acceptance check: {summary_line}")
            unmet = last_loop_eval.get("unmet_criteria")
            if isinstance(unmet, list) and unmet:
                print(f"agent> Unmet criteria: {', '.join(str(item) for item in unmet)}.")
            recommendations = last_loop_eval.get("recommendations")
            if isinstance(recommendations, list):
                for rec in recommendations[:3]:
                    text = str(rec).strip()
                    if text:
                        print(f"agent> Loop recommendation: {text}")

        should_continue = bool((last_loop_eval or {}).get("should_continue", False))
        if not allow_loop:
            should_continue = False
        if user_locked_model_family and current_requested_model != "auto":
            if should_continue:
                print(
                    "agent> Architecture auto-switch is disabled because this model family was explicitly chosen by the user."
                )
            should_continue = False
        if not allow_architecture_switch:
            should_continue = False
        if not should_continue:
            break

        next_model = _choose_model_retry_candidate(
            training=training,
            current_model_family=current_requested_model,
            model_search_order=raw_search_order,
            tried_models=tried_models,
        )
        if next_model is None:
            print(
                "agent> No additional safe model-family retry is available from the current comparison set."
            )
            break
        attempt += 1
        if attempt > max_attempts:
            break
        current_requested_model = next_model
        print(
            f"agent> Continuing bounded optimization loop with `{current_requested_model}` as the next retry."
        )

    if last_training is None:
        response = "Model training did not produce a usable result."
        print(f"agent> {response}")
        return {
            "response": response,
            "event": {"status": "respond", "message": response, "error": "training_unavailable"},
        }

    training = last_training
    selected_metrics_bundle = (
        training.get("selected_metrics") if isinstance(training.get("selected_metrics"), dict) else {}
    )
    test_metrics = (
        selected_metrics_bundle.get("test")
        if isinstance(selected_metrics_bundle.get("test"), dict)
        else {}
    )
    comparison = training.get("comparison") if isinstance(training.get("comparison"), list) else []
    summary = (
        "Model build complete: "
        f"requested_model={resolved_model}, "
        f"final_attempt_model={current_requested_model}, "
        f"selected_model={training.get('selected_model_family', 'n/a')}, "
        f"best_validation_model={training.get('best_validation_model_family', 'n/a')}, "
        f"target={target}, "
        f"inputs={len(features)}, "
        f"rows_used={training.get('rows_used', 'n/a')}, "
        f"test_r2={_fmt_metric(test_metrics.get('r2'))}, "
        f"test_mae={_fmt_metric(test_metrics.get('mae'))}."
    )
    checkpoint_line = f"Checkpoint saved: {training.get('checkpoint_id', 'n/a')}"
    run_dir_line = f"Artifacts: {training.get('run_dir', 'n/a')}"
    print(f"agent> {summary}")
    print(f"agent> {checkpoint_line}")
    print(f"agent> {run_dir_line}")
    if chat_reply_only is not None:
        interpretation = _generate_modeling_interpretation(
            training=training,
            target_signal=target,
            requested_model_family=current_requested_model,
            chat_reply_only=chat_reply_only,
        )
        if interpretation:
            print("agent> LLM interpretation:")
            for line in interpretation.splitlines():
                text = line.strip()
                if text:
                    print(f"agent> {text}")
    session_context["workflow_stage"] = "model_training_completed"
    session_context["last_model_request"] = {
        "target_signal": target,
        "feature_signals": features,
        "requested_model_family": requested_model,
        "final_attempt_model_family": current_requested_model,
        "resolved_model_family": training.get("selected_model_family", current_requested_model),
        "checkpoint_id": training.get("checkpoint_id"),
        "run_dir": training.get("run_dir"),
        "lag_horizon_samples": int(training.get("lag_horizon_samples") or 0),
        "acceptance_check": last_loop_eval or {},
    }
    response = f"{summary} {checkpoint_line}"
    return {
        "response": response,
        "event": {
            "status": "respond",
            "message": response,
            "tool_output": {
                "target_signal": target,
                "feature_signals": features,
                "resolved_model_family": training.get("selected_model_family", current_requested_model),
                "checkpoint_id": training.get("checkpoint_id"),
                "run_dir": training.get("run_dir"),
                "metrics": test_metrics,
                "comparison": comparison,
                "acceptance_check": last_loop_eval or {},
            },
        },
    }


def _normalize_modeler_model_family(requested_model: str) -> str | None:
    return normalize_candidate_model_family(requested_model)


def _fmt_metric(value: Any) -> str:
    parsed = _float_value_or_none(value)
    if parsed is None:
        return "n/a"
    return f"{parsed:.4f}"


def _modeler_training_tool_args(
    *,
    dataset: dict[str, Any],
    target: str,
    features: list[str],
    requested_model_family: str,
    timestamp_column: str | None,
    requested_normalize: bool,
    missing_data_strategy: str,
    fill_constant_value: Any,
    compare_against_baseline: bool,
    lag_horizon_samples: Any,
    checkpoint_tag: str,
) -> dict[str, Any]:
    return {
        "data_path": str(dataset.get("data_path", "")),
        "target_column": target,
        "feature_columns": features,
        "requested_model_family": requested_model_family,
        "sheet_name": dataset.get("sheet_name"),
        "timestamp_column": timestamp_column,
        "checkpoint_tag": checkpoint_tag,
        "normalize": requested_normalize,
        "missing_data_strategy": missing_data_strategy,
        "fill_constant_value": fill_constant_value,
        "compare_against_baseline": compare_against_baseline,
        "lag_horizon_samples": lag_horizon_samples,
    }


def _print_modeler_training_summary(*, training: dict[str, Any]) -> None:
    split_info = training.get("split") if isinstance(training.get("split"), dict) else {}
    preprocessing = training.get("preprocessing") if isinstance(training.get("preprocessing"), dict) else {}
    comparison = training.get("comparison") if isinstance(training.get("comparison"), list) else []
    print(
        "agent> Split-safe pipeline: "
        f"strategy={split_info.get('strategy', 'n/a')}, "
        f"train={split_info.get('train_size', 'n/a')}, "
        f"val={split_info.get('validation_size', 'n/a')}, "
        f"test={split_info.get('test_size', 'n/a')}."
    )
    print(
        "agent> Preprocessing policy: "
        f"requested={preprocessing.get('missing_data_strategy_requested', 'n/a')}, "
        f"effective={preprocessing.get('missing_data_strategy_effective', 'n/a')}, "
        f"normalization={((training.get('normalization') or {}).get('method', 'none'))}."
    )
    if int(training.get("lag_horizon_samples") or 0) > 0:
        print(
            "agent> Temporal feature plan: "
            f"lag_horizon_samples={int(training.get('lag_horizon_samples') or 0)}."
        )
    for item in comparison:
        if not isinstance(item, dict):
            continue
        val_metrics = item.get("validation_metrics") if isinstance(item.get("validation_metrics"), dict) else {}
        test_metrics = item.get("test_metrics") if isinstance(item.get("test_metrics"), dict) else {}
        print(
            "agent> Candidate "
            f"`{item.get('model_family', 'n/a')}`: "
            f"val_r2={_fmt_metric(val_metrics.get('r2'))}, "
            f"val_mae={_fmt_metric(val_metrics.get('mae'))}, "
            f"test_r2={_fmt_metric(test_metrics.get('r2'))}, "
            f"test_mae={_fmt_metric(test_metrics.get('mae'))}."
        )


def _safe_acceptance_criteria(raw: Any) -> dict[str, float]:
    if not isinstance(raw, dict):
        return {"r2": 0.70}
    criteria: dict[str, float] = {}
    for key, value in raw.items():
        name = str(key).strip()
        if not name:
            continue
        try:
            criteria[name] = float(value)
        except (TypeError, ValueError):
            continue
    return criteria or {"r2": 0.70}


def _safe_loop_policy(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        return {"enabled": True, "max_attempts": 2, "allow_architecture_switch": True}
    enabled = bool(raw.get("enabled", True))
    try:
        max_attempts = max(1, int(raw.get("max_attempts", 2)))
    except (TypeError, ValueError):
        max_attempts = 2
    allow_architecture_switch = bool(raw.get("allow_architecture_switch", True))
    return {
        "enabled": enabled,
        "max_attempts": max_attempts,
        "allow_architecture_switch": allow_architecture_switch,
    }


def _build_model_loop_metrics(training: dict[str, Any]) -> dict[str, float]:
    selected = training.get("selected_metrics") if isinstance(training.get("selected_metrics"), dict) else {}
    train_metrics = selected.get("train") if isinstance(selected.get("train"), dict) else {}
    val_metrics = selected.get("validation") if isinstance(selected.get("validation"), dict) else {}
    test_metrics = selected.get("test") if isinstance(selected.get("test"), dict) else {}
    metrics: dict[str, float] = {}
    for key, value in test_metrics.items():
        try:
            metrics[str(key)] = float(value)
        except (TypeError, ValueError):
            continue
    for source, prefix in ((train_metrics, "train_"), (val_metrics, "val_")):
        for key, value in source.items():
            try:
                metrics[f"{prefix}{key}"] = float(value)
            except (TypeError, ValueError):
                continue
    try:
        metrics["n_samples"] = float(training.get("rows_used", 0))
    except (TypeError, ValueError):
        metrics["n_samples"] = 0.0
    return metrics


def _choose_model_retry_candidate(
    *,
    training: dict[str, Any],
    current_model_family: str,
    model_search_order: list[str],
    tried_models: set[str],
) -> str | None:
    normalized_order: list[str] = []
    for item in model_search_order:
        normalized = _normalize_modeler_model_family(str(item))
        if normalized is not None and normalized not in normalized_order:
            normalized_order.append(normalized)
    best_validation = _normalize_modeler_model_family(
        str(training.get("best_validation_model_family", "")).strip()
    )
    if best_validation is not None and best_validation not in normalized_order:
        normalized_order.append(best_validation)
    comparison = training.get("comparison") if isinstance(training.get("comparison"), list) else []
    for item in comparison:
        if not isinstance(item, dict):
            continue
        candidate = _normalize_modeler_model_family(str(item.get("model_family", "")).strip())
        if candidate is not None and candidate not in normalized_order:
            normalized_order.append(candidate)

    current_normalized = _normalize_modeler_model_family(current_model_family) or current_model_family
    for candidate in normalized_order:
        if candidate == current_normalized:
            continue
        if candidate in tried_models:
            continue
        return candidate
    return None


def _extract_first_json_path(user_message: str) -> Path | None:
    quoted = re.findall(r"[\"']([^\"']+\.json)[\"']", user_message, flags=re.IGNORECASE)
    candidates: list[str] = list(quoted)
    absolute_windows = re.findall(
        r"([A-Za-z]:\\[^\n]+?\.json)",
        user_message,
        flags=re.IGNORECASE,
    )
    candidates.extend(absolute_windows)
    plain = re.findall(r"([^\s\"']+\.json)", user_message, flags=re.IGNORECASE)
    candidates.extend(plain)
    for raw in candidates:
        cleaned = raw.strip().rstrip(".,;:)")
        if not cleaned:
            continue
        path = Path(cleaned).expanduser()
        if path.exists():
            return path
    for raw in candidates:
        cleaned = raw.strip().rstrip(".,;:)")
        if cleaned:
            return Path(cleaned).expanduser()
    return None


def _load_modeler_handoff_payload(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {"error": f"Handoff file does not exist: {path}"}
    except json.JSONDecodeError as exc:
        return {"error": f"Handoff file is not valid JSON: {exc}"}
    if not isinstance(payload, dict):
        return {"error": "Handoff JSON must be an object."}
    handoff = build_agent2_handoff_from_report_payload(payload)
    if handoff is None:
        return {
            "error": (
                "I could not derive a valid Agent 2 handoff from that report. "
                "Ensure it is an Agent 1 structured report with target and predictor recommendations."
            )
        }
    dataset_profile = handoff.dataset_profile if isinstance(handoff.dataset_profile, dict) else {}
    return {
        "payload": payload,
        "handoff": handoff.to_dict(),
        "data_path": dataset_profile.get("data_path", "") or payload.get("data_path", ""),
    }


def _modeler_request_from_handoff(
    *,
    payload: dict[str, Any],
    handoff: dict[str, Any],
    available_signals: list[str],
) -> dict[str, Any] | None:
    target_raw = str(handoff.get("target_signal", "")).strip()
    if not target_raw:
        return None

    feature_raw = [
        str(item).strip()
        for item in handoff.get("feature_signals", [])
        if str(item).strip()
    ]
    if not feature_raw and available_signals:
        fallback = [name for name in available_signals if name != target_raw]
        feature_raw = fallback[: min(3, len(fallback))]

    if not feature_raw:
        return None

    preprocessing = payload.get("preprocessing") if isinstance(payload.get("preprocessing"), dict) else {}
    missing_plan = (
        preprocessing.get("missing_data_plan")
        if isinstance(preprocessing.get("missing_data_plan"), dict)
        else {}
    )
    return {
        "requested_model_family": str(handoff.get("recommended_model_family", "linear_ridge")).strip() or "linear_ridge",
        "feature_raw": feature_raw,
        "target_raw": target_raw,
        "data_path": str(payload.get("data_path", "")).strip(),
        "timestamp_column": str(handoff.get("dataset_profile", {}).get("timestamp_column", "")).strip() or None,
        "normalize": bool(handoff.get("normalization", {}).get("enabled", True)),
        "missing_data_strategy": str(missing_plan.get("strategy", "keep")).strip() or "keep",
        "fill_constant_value": missing_plan.get("fill_constant_value"),
        "compare_against_baseline": True,
        "lag_horizon_samples": int(handoff.get("lag_horizon_samples", 0) or 0) or None,
        "acceptance_criteria": (
            dict(handoff.get("acceptance_criteria"))
            if isinstance(handoff.get("acceptance_criteria"), dict)
            else {"r2": 0.70}
        ),
        "loop_policy": (
            dict(handoff.get("loop_policy"))
            if isinstance(handoff.get("loop_policy"), dict)
            else {"enabled": True, "max_attempts": 3, "allow_architecture_switch": True}
        ),
        "user_locked_model_family": False,
        "model_search_order": [
            str(item).strip()
            for item in handoff.get("model_search_order", [])
            if str(item).strip()
        ],
        "source": "handoff",
    }


def _prompt_modeler_overrides(
    *,
    request: dict[str, Any],
    available_signals: list[str],
) -> dict[str, Any]:
    target_raw = _prompt_modeler_target_override(
        default_target=str(request.get("target_raw", "")).strip(),
        available_signals=available_signals,
    )
    feature_raw = _prompt_modeler_feature_override(
        default_features=[str(item) for item in request.get("feature_raw", [])],
        target_signal=target_raw,
        available_signals=available_signals,
    )
    requested_model_family, user_locked_model_family = _prompt_modeler_model_override(
        default_model=str(request.get("requested_model_family", "linear_ridge")).strip(),
    )
    return {
        **request,
        "target_raw": target_raw,
        "feature_raw": feature_raw,
        "requested_model_family": requested_model_family,
        "user_locked_model_family": user_locked_model_family,
    }


def _prompt_modeler_target_override(
    *,
    default_target: str,
    available_signals: list[str],
) -> str:
    while True:
        print(
            "agent> Press Enter to use the recommended target "
            f"`{default_target}`, type `list` to show signals, or enter a target name/index."
        )
        answer = input("you> ").strip()
        lowered = answer.lower()
        if not answer:
            return default_target
        if lowered.startswith("list"):
            _print_signal_names(available_signals, query=answer[4:].strip())
            continue
        resolved = _resolve_signal_name(answer, available_signals)
        if resolved is not None:
            return resolved
        print("agent> I did not resolve that target. Type `list` to inspect signal names.")


def _prompt_modeler_feature_override(
    *,
    default_features: list[str],
    target_signal: str,
    available_signals: list[str],
) -> list[str]:
    default_display = ",".join(default_features)
    while True:
        print(
            "agent> Press Enter to use the recommended inputs "
            f"`{default_display}`, type `list` to show signals, or enter comma-separated inputs."
        )
        answer = input("you> ").strip()
        lowered = answer.lower()
        if not answer:
            return [item for item in default_features if item != target_signal]
        if lowered.startswith("list"):
            _print_signal_names(available_signals, query=answer[4:].strip())
            continue
        requested = _split_modeler_input_tokens(answer)
        resolved: list[str] = []
        unknown: list[str] = []
        seen: set[str] = set()
        for raw in requested:
            match = _resolve_signal_name(raw, available_signals)
            if match is None:
                unknown.append(raw)
                continue
            if match == target_signal or match in seen:
                continue
            resolved.append(match)
            seen.add(match)
        if unknown:
            print(f"agent> Ignoring unknown inputs: {unknown}")
        if resolved:
            return resolved
        print("agent> I did not resolve any usable inputs. Type `list` to inspect signal names.")


def _prompt_modeler_model_override(*, default_model: str) -> tuple[str, bool]:
    available = (
        "auto, linear_ridge (aliases: ridge, linear, incremental_linear_surrogate), "
        "lagged_linear (aliases: lagged, temporal_linear, arx), "
        "lagged_tree_ensemble (aliases: lagged_tree, lag_window_tree, temporal_tree), "
        "bagged_tree_ensemble (aliases: tree, tree_ensemble, extra_trees, hist_gradient_boosting)"
    )
    recommended_supported = _normalize_modeler_model_family(default_model)
    while True:
        default_text = default_model or "linear_ridge"
        if recommended_supported is None:
            print(
                "agent> The recommended model "
                f"`{default_text}` is not implemented yet. "
                "Press Enter to use `auto`, or type an available model."
            )
        else:
            print(
                "agent> Press Enter to use the recommended model "
                f"`{default_text}`, or type an available model."
            )
        print(f"agent> Currently implemented: {available}.")
        answer = input("you> ").strip()
        if not answer:
            return recommended_supported or "auto", False
        if _normalize_modeler_model_family(answer) is not None:
            return answer, True
        print("agent> That model is not implemented yet. Please choose an available model.")


def _generate_analysis_interpretation(
    *,
    analysis: dict[str, Any],
    chat_reply_only: Callable[[str], str],
) -> str:
    primary_prompt = _build_analysis_interpretation_prompt(analysis)
    primary = chat_reply_only(primary_prompt).strip()
    if primary and not _looks_like_llm_failure_message(primary):
        return primary

    compact_prompt = _build_compact_analysis_interpretation_prompt(analysis)
    compact = chat_reply_only(compact_prompt).strip()
    if compact and not _looks_like_llm_failure_message(compact):
        return compact
    return ""


def _generate_modeling_interpretation(
    *,
    training: dict[str, Any],
    target_signal: str,
    requested_model_family: str,
    chat_reply_only: Callable[[str], str],
) -> str:
    prompt = _build_modeling_interpretation_prompt(
        training=training,
        target_signal=target_signal,
        requested_model_family=requested_model_family,
    )
    primary = chat_reply_only(prompt).strip()
    if primary and not _looks_like_llm_failure_message(primary):
        return primary
    compact = (
        "Interpret this model training result in 4 concise bullets: "
        "1) whether the selected model is scientifically credible, "
        "2) whether the requested model agreed with the best validation result, "
        "3) main risks, "
        "4) what to try next.\n"
        f"SUMMARY={dumps_json(_compact_modeling_summary(training, target_signal, requested_model_family), ensure_ascii=False)}"
    )
    secondary = chat_reply_only(compact).strip()
    if secondary and not _looks_like_llm_failure_message(secondary):
        return secondary
    return ""


def _build_modeling_interpretation_prompt(
    *,
    training: dict[str, Any],
    target_signal: str,
    requested_model_family: str,
) -> str:
    return (
        "Interpret this Agent 2 model training result for a lab engineer. "
        "Use 5-7 concise bullets. Cover: "
        "overall result quality, whether the requested model matched the best validation model, "
        "what the train/validation/test split implies, main risks, and immediate next actions. "
        "Do not invent values.\n"
        f"RESULTS_JSON={dumps_json(_compact_modeling_summary(training, target_signal, requested_model_family), ensure_ascii=False)}"
    )


def _compact_modeling_summary(
    training: dict[str, Any],
    target_signal: str,
    requested_model_family: str,
) -> dict[str, Any]:
    comparison_out: list[dict[str, Any]] = []
    comparison = training.get("comparison") if isinstance(training.get("comparison"), list) else []
    for item in comparison[:4]:
        if not isinstance(item, dict):
            continue
        comparison_out.append(
            {
                "model_family": item.get("model_family"),
                "validation_metrics": item.get("validation_metrics"),
                "test_metrics": item.get("test_metrics"),
            }
        )
    return {
        "target_signal": target_signal,
        "requested_model_family": requested_model_family,
        "selected_model_family": training.get("selected_model_family"),
        "best_validation_model_family": training.get("best_validation_model_family"),
        "lag_horizon_samples": training.get("lag_horizon_samples"),
        "split": training.get("split"),
        "preprocessing": training.get("preprocessing"),
        "normalization": training.get("normalization"),
        "selected_metrics": training.get("selected_metrics"),
        "comparison": comparison_out,
    }


def _build_analysis_interpretation_prompt(analysis: dict[str, Any]) -> str:
    quality = analysis.get("quality") if isinstance(analysis.get("quality"), dict) else {}
    ranking = analysis.get("ranking") if isinstance(analysis.get("ranking"), list) else []
    correlations = analysis.get("correlations")
    target_analyses = []
    if isinstance(correlations, dict):
        maybe_targets = correlations.get("target_analyses")
        if isinstance(maybe_targets, list):
            target_analyses = maybe_targets

    top_ranked: list[dict[str, Any]] = []
    for item in ranking[:3]:
        if not isinstance(item, dict):
            continue
        top_ranked.append(
            {
                "target_signal": item.get("target_signal"),
                "adjusted_score": item.get("adjusted_score"),
                "feasible": item.get("feasible"),
                "rationale": item.get("rationale"),
            }
        )

    top_predictors = _extract_top3_correlations_global(analysis)

    warnings = quality.get("warnings") if isinstance(quality, dict) else []
    if not isinstance(warnings, list):
        warnings = []

    summary = {
        "data_mode": analysis.get("data_mode"),
        "target_count": analysis.get("target_count"),
        "candidate_count": analysis.get("candidate_count"),
        "quality": {
            "rows": quality.get("rows"),
            "columns": quality.get("columns"),
            "completeness_score": quality.get("completeness_score"),
            "warnings": [str(item) for item in warnings[:6]],
        },
        "top_ranked_signals": top_ranked,
        "top_3_correlated_predictors": top_predictors,
    }

    return (
        "Interpret these Agent 1 results for a lab engineer. "
        "Give a concise scientific readout in 5-8 bullets: "
        "overall assessment, strongest evidence, risks/uncertainties, and immediate next actions. "
        "Mandatory: include one bullet that starts with `Top 3 correlated predictors:` and list "
        "the predictor_signal, target_signal, best_method, and best_abs_score from "
        "`top_3_correlated_predictors`. "
        "Do not invent values.\n"
        f"RESULTS_JSON={dumps_json(summary, ensure_ascii=False)}"
    )


def _build_compact_analysis_interpretation_prompt(analysis: dict[str, Any]) -> str:
    quality = analysis.get("quality") if isinstance(analysis.get("quality"), dict) else {}
    top3 = _extract_top3_correlations_global(analysis)
    rows = quality.get("rows", "n/a")
    completeness = quality.get("completeness_score", "n/a")
    warnings = quality.get("warnings") if isinstance(quality.get("warnings"), list) else []
    top3_line = _format_top3_correlations_line(top3) if top3 else "Top 3 correlated predictors: n/a"
    return (
        "You are Agent 1's scientific narrator. Use concise plain text.\n"
        "Summarize in 4 numbered points:\n"
        "1) data quality,\n"
        "2) strongest evidence,\n"
        "3) key risks,\n"
        "4) immediate next actions.\n"
        "Mandatory: include one bullet that starts with `Top 3 correlated predictors:` exactly.\n"
        f"rows={rows}\n"
        f"completeness_score={completeness}\n"
        f"warnings={warnings[:5]}\n"
        f"{top3_line}\n"
    )


def _extract_top3_correlations_global(analysis: dict[str, Any]) -> list[dict[str, Any]]:
    correlations = analysis.get("correlations")
    target_analyses: list[dict[str, Any]] = []
    if isinstance(correlations, dict):
        maybe_targets = correlations.get("target_analyses")
        if isinstance(maybe_targets, list):
            target_analyses = [row for row in maybe_targets if isinstance(row, dict)]

    flattened: list[dict[str, Any]] = []
    for target in target_analyses:
        target_signal = str(target.get("target_signal", "")).strip()
        predictor_rows = target.get("predictor_results")
        if not isinstance(predictor_rows, list):
            continue
        for row in predictor_rows:
            if not isinstance(row, dict):
                continue
            score = _float_value_or_none(row.get("best_abs_score"))
            predictor_signal = str(row.get("predictor_signal", "")).strip()
            best_method = str(row.get("best_method", "")).strip()
            if not predictor_signal or score is None:
                continue
            flattened.append(
                {
                    "target_signal": target_signal,
                    "predictor_signal": predictor_signal,
                    "best_method": best_method or "n/a",
                    "best_abs_score": float(score),
                    "sample_count": row.get("sample_count"),
                }
            )

    flattened.sort(
        key=lambda item: float(item.get("best_abs_score", 0.0)),
        reverse=True,
    )
    return flattened[:3]


def _format_top3_correlations_line(top3: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for row in top3:
        predictor = str(row.get("predictor_signal", "n/a"))
        target = str(row.get("target_signal", "n/a"))
        method = str(row.get("best_method", "n/a"))
        score = _float_value_or_none(row.get("best_abs_score"))
        score_text = f"{score:.3f}" if score is not None else "n/a"
        parts.append(f"{predictor}->{target} ({method}, abs={score_text})")
    return "Top 3 correlated predictors: " + "; ".join(parts)


def _interpretation_mentions_top3(*, interpretation: str, top3: list[dict[str, Any]]) -> bool:
    lowered = interpretation.lower()
    return all(
        str(row.get("predictor_signal", "")).strip().lower() in lowered
        for row in top3
        if str(row.get("predictor_signal", "")).strip()
    )


def _execute_registry_tool(registry: Any, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    result = registry.execute(tool_name, _drop_none_fields(arguments))
    output = result.output
    if not isinstance(output, dict):
        return {"status": "error", "message": f"Tool '{tool_name}' returned non-object output."}
    return output


def _prompt_sheet_selection(
    options: list[str],
    *,
    chat_detour: Callable[[str, str], None] | None = None,
) -> str | None:
    if not options:
        return None
    _print_sheet_options(options)
    lowered_to_name = {str(name).lower(): str(name) for name in options}
    while True:
        selection = input("you> ").strip()
        lowered = selection.lower()
        if lowered in {"cancel", "abort"}:
            return None
        if lowered in {"list", "show", "help", "?"}:
            _print_sheet_options(options)
            continue
        if _looks_like_small_talk(selection):
            if chat_detour is not None:
                chat_detour(
                    selection,
                    "To continue, please enter a sheet number/name, type 'list', or 'cancel'.",
                )
            else:
                print(
                    "agent> I can chat, and we are selecting an Excel sheet now. "
                    "Please enter a sheet number/name, type 'list' to show sheets again, "
                    "or 'cancel' to abort."
                )
            continue
        if selection.isdigit():
            index = int(selection)
            if 1 <= index <= len(options):
                return str(options[index - 1])
        resolved = lowered_to_name.get(lowered)
        if resolved is not None:
            return resolved
        print(
            "agent> Invalid selection. Enter a sheet number/name, "
            "type 'list' to show sheets, or 'cancel' to abort."
        )


def _print_sheet_options(options: list[str]) -> None:
    print("agent> Multiple sheets detected. Please choose one:")
    for idx, name in enumerate(options, start=1):
        print(f"agent>   {idx}. {name}")


def _prompt_header_confirmation(
    *,
    header_row: int,
    data_start_row: int,
    chat_detour: Callable[[str, str], None] | None = None,
) -> tuple[int, int]:
    while True:
        print(
            "agent> Use inferred rows "
            f"header={header_row}, data_start={data_start_row}? [Y/n or custom 'h,d']"
        )
        answer = input("you> ").strip()
        lowered = answer.lower()
        parsed_override = _parse_header_override(answer)
        if parsed_override is not None:
            return parsed_override
        if lowered in {"", "y", "yes"}:
            return header_row, data_start_row
        if lowered in {"n", "no"}:
            return _prompt_header_override(
                default_header_row=header_row,
                default_data_start_row=data_start_row,
                chat_detour=chat_detour,
            )
        if _looks_like_small_talk(answer):
            if chat_detour is not None:
                chat_detour(
                    answer,
                    "To continue, reply with Y/Enter to keep inferred rows, N to change, or 'h,d'.",
                )
            else:
                print(
                    "agent> I can chat, and we are confirming inferred rows now. "
                    "Reply with Y/Enter to keep inferred rows, N to change, "
                    "or 'h,d' (for example 2,3)."
                )
            continue
        print(
            "agent> I did not parse that. Reply Y/Enter to keep inferred rows, "
            "N to change, or 'h,d' (for example 2,3)."
        )


def _prompt_header_override(
    *,
    default_header_row: int,
    default_data_start_row: int,
    chat_detour: Callable[[str, str], None] | None = None,
) -> tuple[int, int]:
    while True:
        print("agent> Enter header_row,data_start_row (e.g. 2,3). Press Enter to keep inferred.")
        second = input("you> ").strip()
        lowered = second.lower()
        if not second or lowered in {"y", "yes", "keep"}:
            return default_header_row, default_data_start_row
        parsed_second = _parse_header_override(second)
        if parsed_second is not None:
            return parsed_second
        if _looks_like_small_talk(second):
            if chat_detour is not None:
                chat_detour(
                    second,
                    "To continue, please enter 'header_row,data_start_row' or press Enter to keep inferred rows.",
                )
            else:
                print(
                    "agent> I can chat, and we are choosing explicit row numbers now. "
                    "Please enter 'header_row,data_start_row' (for example 2,3), "
                    "or press Enter to keep inferred rows."
                )
            continue
        print(
            "agent> Invalid format. Use 'header_row,data_start_row' with "
            "data_start_row greater than header_row."
        )


def _parse_header_override(raw: str) -> tuple[int, int] | None:
    text = raw.strip()
    if not text:
        return None
    match = re.match(r"^\s*(\d+)\s*,\s*(\d+)\s*$", text)
    if not match:
        return None
    header_row = int(match.group(1))
    data_start_row = int(match.group(2))
    if data_start_row <= header_row:
        return None
    return header_row, data_start_row


def _extract_first_data_path(user_message: str) -> Path | None:
    quoted = re.findall(r"[\"']([^\"']+\.(?:csv|xlsx|xls))[\"']", user_message, flags=re.IGNORECASE)
    candidates: list[str] = list(quoted)

    absolute_windows = re.findall(
        r"([A-Za-z]:\\[^\n]+?\.(?:csv|xlsx|xls))",
        user_message,
        flags=re.IGNORECASE,
    )
    candidates.extend(absolute_windows)

    plain = re.findall(r"([^\s\"']+\.(?:csv|xlsx|xls))", user_message, flags=re.IGNORECASE)
    candidates.extend(plain)

    for raw in candidates:
        cleaned = raw.strip().rstrip(".,;:)")
        if not cleaned:
            continue
        path = Path(cleaned).expanduser()
        if path.exists():
            return path
    for raw in candidates:
        cleaned = raw.strip().rstrip(".,;:)")
        if not cleaned:
            continue
        return Path(cleaned).expanduser()
    return None


def _resolve_default_public_dataset_path() -> Path | None:
    candidates: list[Path] = []
    candidates.append(Path("data/public/public_testbench_dataset_20k_minmax.csv"))
    candidates.append(Path("data/public/public_testbench_dataset_20k_minmax.xlsx"))
    repo_root_guess = Path(__file__).resolve().parents[3]
    candidates.append(
        repo_root_guess / "data" / "public" / "public_testbench_dataset_20k_minmax.csv"
    )
    candidates.append(
        repo_root_guess / "data" / "public" / "public_testbench_dataset_20k_minmax.xlsx"
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None


def _path_for_display(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(resolved)


def _compact_event_for_context(event: dict[str, Any]) -> dict[str, Any]:
    compact = {
        "status": event.get("status"),
        "message": _truncate_text(str(event.get("message", "")), 500),
        "error": _truncate_text(str(event.get("error", "")), 300),
    }
    action = event.get("action")
    if isinstance(action, dict):
        compact["action"] = {
            "action": action.get("action"),
            "tool_name": action.get("tool_name"),
        }
    tool_output = event.get("tool_output")
    if isinstance(tool_output, dict):
        compact["tool_output"] = {
            "data_mode": tool_output.get("data_mode"),
            "target_count": tool_output.get("target_count"),
            "candidate_count": tool_output.get("candidate_count"),
            "report_path": tool_output.get("report_path"),
        }
    return compact


def _truncate_text(value: str, max_len: int) -> str:
    if len(value) <= max_len:
        return value
    return value[: max_len - 3] + "..."


def _parse_target_selection(
    *,
    target_answer: str,
    available_signals: list[str],
    default_count: int,
) -> list[str]:
    if not target_answer.strip():
        return list(available_signals[:default_count])
    requested = [item.strip() for item in target_answer.split(",") if item.strip()]
    selected = [item for item in requested if item in available_signals]
    if selected:
        return selected
    return list(available_signals[:default_count])


def _prompt_target_selection(
    *,
    available_signals: list[str],
    default_count: int,
    hypothesis_state: dict[str, list[dict[str, Any]]] | None = None,
    chat_detour: Callable[[str, str], None] | None = None,
) -> list[str] | None:
    while True:
        answer = input("you> ").strip()
        lowered = answer.lower()
        if lowered == "all":
            return None
        if lowered.startswith("hypothesis"):
            parsed = _parse_inline_hypothesis_command(
                user_message=answer,
                available_signals=available_signals,
            )
            if parsed["user_hypotheses"] or parsed["feature_hypotheses"]:
                if hypothesis_state is not None:
                    _merge_hypothesis_state(hypothesis_state, parsed)
                print(
                    "agent> Hypotheses noted. "
                    f"correlation={len(parsed['user_hypotheses'])}, "
                    f"feature={len(parsed['feature_hypotheses'])}. "
                    "Now continue with target selection."
                )
                print(
                    "agent> Enter comma-separated target signals, "
                    "'all' for full run, `hypothesis ...` to add more, "
                    "or press Enter for quick default subset."
                )
                continue
            print(
                "agent> Hypothesis format not recognized. "
                "Use: `hypothesis corr target:pred1,pred2; target2:pred3` or "
                "`hypothesis feature target:signal->rate_change; signal2->square`."
            )
            continue
        if lowered.startswith("list"):
            query = answer[4:].strip()
            _print_signal_names(available_signals, query=query)
            print(
                "agent> Enter comma-separated target signals, "
                "'all' for full run, `hypothesis ...` to add hypotheses, "
                "or press Enter for quick default subset."
            )
            continue
        selected, unknown = _parse_target_selection_with_unknowns(
            target_answer=answer,
            available_signals=available_signals,
            default_count=default_count,
        )
        if unknown and not selected:
            if _looks_like_small_talk(answer):
                if chat_detour is not None:
                    chat_detour(
                        answer,
                        "To continue, type 'list' to show names, 'all' for full run, or provide target signals.",
                    )
                else:
                    print(
                        "agent> I am good and ready to continue. "
                        "We are currently selecting target signals. "
                        "If useful, add hypotheses via `hypothesis ...`. "
                        "Type 'list' to show names, 'all' for full run, "
                        "or press Enter for a quick subset."
                    )
                continue
            print(
                "agent> No matching signal names found. "
                "Type 'list' to display available names."
            )
            continue
        if unknown:
            print(f"agent> Ignoring unknown signals: {unknown}")
        return selected


def _parse_target_selection_with_unknowns(
    *,
    target_answer: str,
    available_signals: list[str],
    default_count: int,
) -> tuple[list[str], list[str]]:
    if not target_answer.strip():
        return list(available_signals[:default_count]), []

    requested = [item.strip() for item in target_answer.split(",") if item.strip()]
    available_lookup = {name.lower(): name for name in available_signals}
    selected: list[str] = []
    unknown: list[str] = []
    for item in requested:
        resolved: str | None = None
        if item.isdigit():
            idx = int(item)
            if 1 <= idx <= len(available_signals):
                resolved = available_signals[idx - 1]
        if resolved is None:
            resolved = available_lookup.get(item.lower())
        if resolved is None:
            fuzzy = [name for name in available_signals if item.lower() in name.lower()]
            if len(fuzzy) == 1:
                resolved = fuzzy[0]
        if resolved:
            selected.append(resolved)
        else:
            unknown.append(item)

    deduped: list[str] = []
    seen: set[str] = set()
    for name in selected:
        if name not in seen:
            deduped.append(name)
            seen.add(name)

    return deduped, unknown


def _parse_inline_hypothesis_command(
    *,
    user_message: str,
    available_signals: list[str],
) -> dict[str, list[dict[str, Any]]]:
    parsed: dict[str, list[dict[str, Any]]] = {
        "user_hypotheses": [],
        "feature_hypotheses": [],
    }
    text = user_message.strip()
    lowered = text.lower()
    if "hypothesis" not in lowered:
        return parsed

    payload = text
    if lowered.startswith("hypothesis"):
        payload = text[len("hypothesis") :].strip()
    if not payload:
        return parsed

    segments = [part.strip() for part in payload.split(";") if part.strip()]
    for segment in segments:
        seg_lower = segment.lower()
        if seg_lower.startswith("corr"):
            seg_body = segment[4:].strip()
            corr = _parse_correlation_hypothesis_segment(
                segment=seg_body,
                available_signals=available_signals,
            )
            if corr:
                parsed["user_hypotheses"].append(corr)
            continue
        if seg_lower.startswith("feature"):
            seg_body = segment[7:].strip()
            features = _parse_feature_hypothesis_segment(
                segment=seg_body,
                available_signals=available_signals,
            )
            parsed["feature_hypotheses"].extend(features)
            continue
        corr = _parse_correlation_hypothesis_segment(
            segment=segment,
            available_signals=available_signals,
        )
        if corr:
            parsed["user_hypotheses"].append(corr)
            continue
        features = _parse_feature_hypothesis_segment(
            segment=segment,
            available_signals=available_signals,
        )
        parsed["feature_hypotheses"].extend(features)
    return parsed


def _parse_correlation_hypothesis_segment(
    *,
    segment: str,
    available_signals: list[str],
) -> dict[str, Any] | None:
    if ":" not in segment:
        return None
    left, right = segment.split(":", 1)
    target = _resolve_signal_name(left.strip(), available_signals)
    if target is None:
        return None
    raw_predictors = [item.strip() for item in re.split(r"[,\|]", right) if item.strip()]
    predictors: list[str] = []
    seen: set[str] = set()
    for raw in raw_predictors:
        resolved = _resolve_signal_name(raw, available_signals)
        if resolved is None or resolved == target or resolved in seen:
            continue
        predictors.append(resolved)
        seen.add(resolved)
    if not predictors:
        return None
    return {
        "target_signal": target,
        "predictor_signals": predictors,
        "user_reason": "user hypothesis",
    }


def _parse_feature_hypothesis_segment(
    *,
    segment: str,
    available_signals: list[str],
) -> list[dict[str, Any]]:
    if "->" not in segment:
        return []
    target_signal = ""
    payload = segment.strip()
    if ":" in payload and payload.index(":") < payload.index("->"):
        left, right = payload.split(":", 1)
        resolved_target = _resolve_signal_name(left.strip(), available_signals)
        if resolved_target is None:
            return []
        target_signal = resolved_target
        payload = right.strip()
    if "->" not in payload:
        return []
    base_raw, transform_raw = payload.split("->", 1)
    base_signal = _resolve_signal_name(base_raw.strip(), available_signals)
    if base_signal is None:
        return []
    transforms = [
        token.strip().lower()
        for token in re.split(r"[,\|]", transform_raw)
        if token.strip()
    ]
    allowed = {
        "rate_change",
        "delta",
        "pct_change",
        "signed_log",
        "square",
        "sqrt_abs",
        "inverse",
        "lag1",
        "lag2",
        "lag3",
    }
    rows: list[dict[str, Any]] = []
    for transformation in transforms:
        if transformation not in allowed:
            continue
        rows.append(
            {
                "target_signal": target_signal,
                "base_signal": base_signal,
                "transformation": transformation,
                "user_reason": "user hypothesis",
            }
        )
    return rows


def _resolve_signal_name(raw: str, available_signals: list[str]) -> str | None:
    token = raw.strip()
    if not token:
        return None
    if token.isdigit():
        idx = int(token)
        if 1 <= idx <= len(available_signals):
            return available_signals[idx - 1]
    lookup = {name.lower(): name for name in available_signals}
    exact = lookup.get(token.lower())
    if exact:
        return exact
    fuzzy = [name for name in available_signals if token.lower() in name.lower()]
    if len(fuzzy) == 1:
        return fuzzy[0]
    return None


def _merge_hypothesis_state(
    state: dict[str, list[dict[str, Any]]],
    update: dict[str, list[dict[str, Any]]],
) -> None:
    corr_seen = {
        (
            str(item.get("target_signal", "")),
            tuple(str(v) for v in item.get("predictor_signals", [])),
        )
        for item in state.get("user_hypotheses", [])
    }
    for item in update.get("user_hypotheses", []):
        key = (
            str(item.get("target_signal", "")),
            tuple(str(v) for v in item.get("predictor_signals", [])),
        )
        if key in corr_seen:
            continue
        state.setdefault("user_hypotheses", []).append(item)
        corr_seen.add(key)

    feat_seen = {
        (
            str(item.get("target_signal", "")),
            str(item.get("base_signal", "")),
            str(item.get("transformation", "")),
        )
        for item in state.get("feature_hypotheses", [])
    }
    for item in update.get("feature_hypotheses", []):
        key = (
            str(item.get("target_signal", "")),
            str(item.get("base_signal", "")),
            str(item.get("transformation", "")),
        )
        if key in feat_seen:
            continue
        state.setdefault("feature_hypotheses", []).append(item)
        feat_seen.add(key)


def _prompt_lag_preferences(
    *,
    timestamp_column_hint: str,
    estimated_sample_period_seconds: float | None,
    chat_detour: Callable[[str, str], None] | None = None,
) -> dict[str, Any]:
    print(f"agent> Detected timestamp column hint: `{timestamp_column_hint}`.")
    while True:
        print("agent> Do you expect time-based lag effects? [Y/n]")
        answer = input("you> ").strip().lower()
        if answer in {"", "y", "yes"}:
            break
        if answer in {"n", "no"}:
            return {"enabled": False, "dimension": "none", "max_lag": 0}
        if _looks_like_small_talk(answer):
            if chat_detour is not None:
                chat_detour(
                    answer,
                    "To continue, reply Y/Enter to investigate lags, or N to skip lag search.",
                )
            else:
                print(
                    "agent> I can chat, and we are deciding lag analysis scope now. "
                    "Reply Y/Enter to investigate lags, or N to skip lag search."
                )
            continue
        print("agent> Please answer Y/Enter or N.")

    while True:
        print("agent> Lag dimension? Enter `samples` or `seconds` [samples].")
        dimension = input("you> ").strip().lower()
        if dimension in {"", "samples", "sample"}:
            max_lag_samples = _prompt_positive_int(
                prompt=(
                    "agent> Enter maximum lag in samples (positive integer). "
                    "Press Enter for default 8."
                ),
                default_value=8,
                chat_detour=chat_detour,
            )
            return {"enabled": True, "dimension": "samples", "max_lag": max_lag_samples}
        if dimension in {"seconds", "second", "sec", "s"}:
            lag_seconds = _prompt_positive_float(
                prompt=(
                    "agent> Enter maximum lag window in seconds (positive number, for example 2.5)."
                ),
                default_value=None,
                chat_detour=chat_detour,
            )
            default_period = estimated_sample_period_seconds
            if default_period is not None:
                period_prompt = (
                    "agent> Enter sample period in seconds. "
                    f"Press Enter to use estimated {default_period:.6f}s."
                )
            else:
                period_prompt = (
                    "agent> Enter sample period in seconds "
                    "(required to convert seconds to lag samples)."
                )
            sample_period = _prompt_positive_float(
                prompt=period_prompt,
                default_value=default_period,
                chat_detour=chat_detour,
            )
            max_lag_samples = max(1, int(round(lag_seconds / sample_period)))
            return {"enabled": True, "dimension": "seconds", "max_lag": max_lag_samples}
        if _looks_like_small_talk(dimension):
            if chat_detour is not None:
                chat_detour(
                    dimension,
                    "To continue, choose lag dimension: `samples` or `seconds`.",
                )
            else:
                print(
                    "agent> We need a lag dimension choice to continue. "
                    "Use `samples` or `seconds`."
                )
            continue
        print("agent> Invalid lag dimension. Use `samples` or `seconds`.")


def _prompt_sample_budget(
    *,
    row_count: int,
    chat_detour: Callable[[str, str], None] | None = None,
) -> dict[str, Any]:
    if row_count <= 0:
        return {"max_samples": None, "sample_selection": "uniform"}
    if row_count < 500:
        return {"max_samples": None, "sample_selection": "uniform"}
    print(f"agent> Dataset contains {row_count} rows.")
    while True:
        print("agent> Analyze all rows? [Y/n]")
        answer = input("you> ").strip().lower()
        if answer in {"", "y", "yes"}:
            return {"max_samples": None, "sample_selection": "uniform"}
        if answer in {"n", "no"}:
            break
        if _looks_like_small_talk(answer):
            if chat_detour is not None:
                chat_detour(
                    answer,
                    "To continue, reply Y/Enter for all rows, or N to set a subset.",
                )
            else:
                print(
                    "agent> We are choosing analysis sample count. "
                    "Reply Y/Enter for all rows, or N to set a subset."
                )
            continue
        print("agent> Please answer Y/Enter or N.")

    count = _prompt_positive_int(
        prompt=(
            "agent> Enter number of rows to analyze "
            f"(1..{row_count})."
        ),
        default_value=min(row_count, 2000),
        chat_detour=chat_detour,
    )
    count = max(1, min(count, row_count))
    while True:
        print("agent> Sampling mode? Enter `uniform`, `head`, or `tail` [uniform].")
        mode = input("you> ").strip().lower()
        if mode in {"", "uniform"}:
            return {"max_samples": count, "sample_selection": "uniform"}
        if mode in {"head", "tail"}:
            return {"max_samples": count, "sample_selection": mode}
        if _looks_like_small_talk(mode):
            if chat_detour is not None:
                chat_detour(
                    mode,
                    "To continue, choose sampling mode: uniform, head, or tail.",
                )
            else:
                print("agent> Sampling mode controls subset selection order. Use uniform/head/tail.")
            continue
        print("agent> Invalid mode. Use uniform, head, or tail.")


def _prompt_data_issue_handling(
    *,
    preflight: dict[str, Any],
    chat_detour: Callable[[str, str], None] | None = None,
) -> dict[str, Any]:
    plan: dict[str, Any] = {
        "missing_data_strategy": "keep",
        "fill_constant_value": None,
        "row_coverage_strategy": "keep",
        "sparse_row_min_fraction": 0.8,
        "row_range_start": None,
        "row_range_end": None,
    }
    missing_fraction = _safe_float_or_none(preflight.get("missing_overall_fraction")) or 0.0
    missing_cols_count = int(preflight.get("columns_with_missing_count") or 0)
    missing_cols = [str(item) for item in (preflight.get("columns_with_missing") or [])]
    if missing_fraction > 0.0:
        preview = ", ".join(missing_cols[:8]) if missing_cols else "n/a"
        print(
            "agent> Missing-data detected: "
            f"overall_fraction={missing_fraction:.3f}, "
            f"columns_with_missing={missing_cols_count} "
            f"(examples: {preview})."
        )
        print(
            "agent> Leakage note: if missing-value statistics are fit on full data before "
            "train/validation/test split, evaluation can be optimistic."
        )
        print(
            "agent> Split-safe rule for modeling: split first, fit missing-data handling on "
            "train only, then apply the same transform to validation/test."
        )
        print(
            "agent> Choose missing-data handling: "
            "`keep`, `drop_rows`, `fill_median`, `fill_constant` [keep]."
        )
        while True:
            answer = input("you> ").strip().lower()
            if answer in {"", "keep"}:
                break
            if answer in {"drop_rows", "fill_median", "fill_constant"}:
                plan["missing_data_strategy"] = answer
                if answer == "fill_constant":
                    value = _prompt_numeric_value(
                        prompt=(
                            "agent> Enter fill constant (numeric). "
                            "Use negative values if needed."
                        ),
                        default_value=0.0,
                        chat_detour=chat_detour,
                    )
                    plan["fill_constant_value"] = float(value)
                    print(
                        "agent> Leakage note: fill_constant is usually low leakage risk only "
                        "if the constant is fixed a priori."
                    )
                elif answer == "fill_median":
                    print(
                        "agent> Leakage warning: fill_median computed on full data is "
                        "data leakage for downstream train/test evaluation."
                    )
                elif answer == "drop_rows":
                    print(
                        "agent> Caution: drop_rows avoids statistic leakage, but can still bias "
                        "split distributions if done globally before splitting."
                    )
                break
            if _looks_like_small_talk(answer):
                if chat_detour is not None:
                    chat_detour(
                        answer,
                        "To continue, choose missing-data handling: keep, drop_rows, fill_median, or fill_constant.",
                    )
                else:
                    print(
                        "agent> This choice controls NaN handling before correlation. "
                        "Use keep/drop_rows/fill_median/fill_constant."
                    )
                continue
            print("agent> Invalid choice. Use keep, drop_rows, fill_median, or fill_constant.")

    mismatch = bool(preflight.get("potential_length_mismatch"))
    if mismatch:
        row_min = _safe_float_or_none(preflight.get("row_non_null_fraction_min")) or 0.0
        row_med = _safe_float_or_none(preflight.get("row_non_null_fraction_median")) or 0.0
        row_max = _safe_float_or_none(preflight.get("row_non_null_fraction_max")) or 0.0
        print(
            "agent> Uneven row coverage detected (possible different signal lengths): "
            f"min/median/max non-null fraction = {row_min:.3f}/{row_med:.3f}/{row_max:.3f}."
        )
        print(
            "agent> Choose row-coverage handling: "
            "`keep`, `drop_sparse_rows`, `trim_dense_window`, `manual_range` [keep]."
        )
        while True:
            answer = input("you> ").strip().lower()
            if answer in {"", "keep"}:
                break
            if answer in {"drop_sparse_rows", "trim_dense_window"}:
                plan["row_coverage_strategy"] = answer
                threshold = _prompt_fraction(
                    prompt=(
                        "agent> Enter non-null fraction threshold between 0 and 1 "
                        "(default 0.8)."
                    ),
                    default_value=0.8,
                    chat_detour=chat_detour,
                )
                plan["sparse_row_min_fraction"] = threshold
                break
            if answer == "manual_range":
                plan["row_coverage_strategy"] = "manual_range"
                start, end = _prompt_manual_row_range(chat_detour=chat_detour)
                plan["row_range_start"] = start
                plan["row_range_end"] = end
                break
            if _looks_like_small_talk(answer):
                if chat_detour is not None:
                    chat_detour(
                        answer,
                        "To continue, choose row-coverage handling: keep, drop_sparse_rows, trim_dense_window, or manual_range.",
                    )
                else:
                    print(
                        "agent> This choice controls how to align uneven row coverage. "
                        "Use keep/drop_sparse_rows/trim_dense_window/manual_range."
                    )
                continue
            print(
                "agent> Invalid choice. Use keep, drop_sparse_rows, "
                "trim_dense_window, or manual_range."
            )
    return plan


def _prompt_positive_int(
    *,
    prompt: str,
    default_value: int | None,
    chat_detour: Callable[[str, str], None] | None = None,
) -> int:
    while True:
        print(prompt)
        raw = input("you> ").strip()
        if not raw and default_value is not None:
            return int(default_value)
        try:
            value = int(raw)
        except ValueError:
            if _looks_like_small_talk(raw):
                if chat_detour is not None:
                    chat_detour(raw, "To continue, provide a positive integer value.")
                else:
                    print("agent> Please provide a positive integer value.")
            else:
                print("agent> Invalid number. Please provide a positive integer.")
            continue
        if value > 0:
            return value
        print("agent> Value must be > 0.")


def _prompt_positive_float(
    *,
    prompt: str,
    default_value: float | None,
    chat_detour: Callable[[str, str], None] | None = None,
) -> float:
    while True:
        print(prompt)
        raw = input("you> ").strip()
        if not raw and default_value is not None:
            return float(default_value)
        try:
            value = float(raw)
        except ValueError:
            if _looks_like_small_talk(raw):
                if chat_detour is not None:
                    chat_detour(raw, "To continue, provide a positive numeric value.")
                else:
                    print("agent> Please provide a positive numeric value.")
            else:
                print("agent> Invalid number. Please provide a positive numeric value.")
            continue
        if value > 0.0:
            return value
        print("agent> Value must be > 0.")


def _prompt_fraction(
    *,
    prompt: str,
    default_value: float,
    chat_detour: Callable[[str, str], None] | None = None,
) -> float:
    while True:
        print(prompt)
        raw = input("you> ").strip()
        if not raw:
            return float(default_value)
        try:
            value = float(raw)
        except ValueError:
            if _looks_like_small_talk(raw):
                if chat_detour is not None:
                    chat_detour(raw, "To continue, provide a number between 0 and 1.")
                else:
                    print("agent> Please provide a number between 0 and 1.")
            else:
                print("agent> Invalid value. Please provide a number between 0 and 1.")
            continue
        if 0.0 < value <= 1.0:
            return value
        print("agent> Threshold must be in (0, 1].")


def _prompt_numeric_value(
    *,
    prompt: str,
    default_value: float,
    chat_detour: Callable[[str, str], None] | None = None,
) -> float:
    while True:
        print(prompt)
        raw = input("you> ").strip()
        if not raw:
            return float(default_value)
        try:
            return float(raw)
        except ValueError:
            if _looks_like_small_talk(raw):
                if chat_detour is not None:
                    chat_detour(raw, "To continue, provide a numeric value.")
                else:
                    print("agent> Please provide a numeric value.")
            else:
                print("agent> Invalid value. Please provide a numeric value.")


def _prompt_manual_row_range(
    *,
    chat_detour: Callable[[str, str], None] | None = None,
) -> tuple[int, int]:
    while True:
        print("agent> Enter manual row range as `start,end` (0-based, inclusive).")
        raw = input("you> ").strip()
        parsed = _parse_row_range(raw)
        if parsed is not None:
            return parsed
        if _looks_like_small_talk(raw):
            if chat_detour is not None:
                chat_detour(raw, "To continue, provide a numeric row range like `100,2500`.")
            else:
                print("agent> Use numeric row range like `100,2500`.")
            continue
        print("agent> Invalid range. Use `start,end` with end >= start.")


def _parse_row_range(raw: str) -> tuple[int, int] | None:
    text = raw.strip()
    match = re.match(r"^\s*(\d+)\s*,\s*(\d+)\s*$", text)
    if not match:
        return None
    start = int(match.group(1))
    end = int(match.group(2))
    if end < start:
        return None
    return start, end


def _safe_float_or_none(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if parsed <= 0:
        return None
    return parsed


def _float_value_or_none(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed


def _print_signal_names(signals: list[str], *, query: str) -> None:
    q = query.strip().lower()
    if q:
        generic = {"signals", "signal", "signal names", "the signal names", "names"}
        if q in generic:
            q = ""
    filtered = [name for name in signals if q in name.lower()] if q else list(signals)
    if not filtered:
        print("agent> No signal names match that filter.")
        return
    print(f"agent> Available signals ({len(filtered)}):")
    for idx, name in enumerate(filtered, start=1):
        print(f"agent>   {idx}. {name}")


def _print_header_preview(preflight: dict[str, Any]) -> None:
    columns = [str(item) for item in (preflight.get("signal_columns") or [])]
    if not columns:
        return
    header_row = preflight.get("header_row")
    candidate_rows = preflight.get("candidate_header_rows") or []
    print(
        "agent> Inferred header preview "
        f"(header_row={header_row}, candidates={candidate_rows}):"
    )
    for idx, name in enumerate(columns, start=1):
        print(f"agent>   {idx}. {name}")


def _llm_chat_detour(
    *,
    agent: str,
    user_message: str,
    session_context: dict[str, Any],
    session_messages: list[dict[str, str]],
    config_path: str | None,
    record_in_history: bool = True,
) -> str:
    turn_context = dict(session_context)
    turn_context["session_messages"] = list(session_messages)
    turn_context["recent_user_prompts"] = _recent_user_prompts(
        session_messages=session_messages,
        limit=5,
    )
    turn_context["chat_only"] = True
    try:
        result = _invoke_agent_once_with_recovery(
            agent=agent,
            user_message=user_message,
            context=turn_context,
            config_path=config_path,
        )
        event = result.get("event", {})
        response = str(event.get("message", "")).strip()
        if not response:
            response = "[empty response]"
        if record_in_history:
            session_messages.append({"role": "user", "content": user_message})
            session_messages.append({"role": "assistant", "content": response})
            session_messages[:] = session_messages[-20:]
            session_context["last_event"] = _compact_event_for_context(event)
        return response
    except Exception as exc:
        response = _runtime_error_fallback_message(
            agent=agent,
            user_message=user_message,
            error=exc,
        )
        if record_in_history:
            session_messages.append({"role": "user", "content": user_message})
            session_messages.append({"role": "assistant", "content": response})
            session_messages[:] = session_messages[-20:]
            session_context["last_event"] = _compact_event_for_context(
                {"status": "respond", "message": response, "error": "runtime_error"}
            )
        return response


def _rewrite_unhelpful_response(
    *,
    agent: str,
    user_message: str,
    response: str,
    chat_detour: Callable[[str], str] | None = None,
) -> str:
    if agent != "analyst":
        return response
    lowered = response.lower()
    fallback_markers = (
        "repeated tool argument errors",
        "turn limit before a stable answer",
        "too many invalid actions",
    )
    if any(marker in lowered for marker in fallback_markers):
        no_data_path = _extract_first_data_path(user_message) is None
        if no_data_path and chat_detour is not None:
            detour = chat_detour(user_message).strip()
            if detour:
                return detour
        if _is_casual_chat_message(user_message):
            return "I hit a tool-loop issue, but I can still chat. Ask again and I will answer directly."
        if no_data_path:
            return (
                "I hit a tool-loop issue on that request. "
                "If you want analysis, paste a .csv/.xlsx path. "
                "If you want a conceptual answer, ask directly and I will respond."
            )
    return response


def _is_simple_greeting(text: str) -> bool:
    normalized = text.strip().lower()
    return normalized in {"hi", "hello", "hey", "yo", "good morning", "good evening"}


def _is_casual_chat_message(text: str) -> bool:
    normalized = text.strip().lower()
    if not normalized:
        return False
    if _is_simple_greeting(normalized):
        return True
    phrases = {
        "how are you",
        "who are you",
        "what can you do",
        "what?",
        "thanks",
        "thank you",
    }
    if any(phrase in normalized for phrase in phrases):
        return True
    if normalized.endswith("?") and _extract_first_data_path(normalized) is None:
        return True
    return False


def _casual_chat_response(user_message: str) -> str:
    normalized = user_message.strip().lower()
    if "how are you" in normalized:
        return (
            "I am ready to help. I can chat and run analysis locally. "
            "Paste a .csv/.xlsx path when you want to start."
        )
    if "who are you" in normalized or "what can you do" in normalized:
        return (
            "I am your local Corr2Surrogate analyst. I can ingest CSV/XLSX, validate headers/sheets, "
            "run quality checks, stationarity checks, correlation analysis, and generate reports."
        )
    return (
        "Hi. I can chat and help with analysis. "
        "To start dataset analysis, paste a .csv/.xlsx path. "
        "You can also ask what checks I run before correlation."
    )


def _runtime_error_fallback_message(*, agent: str, user_message: str, error: Exception) -> str:
    if _is_provider_connection_error(error):
        if agent == "analyst":
            return (
                "Local LLM runtime is not reachable at the configured endpoint. "
                "Start it with `corr2surrogate setup-local-llm` (or launch your local provider), "
                "then retry. I can still run deterministic ingestion/analysis if you paste a "
                ".csv/.xlsx path."
            )
        return (
            "Local LLM runtime is not reachable at the configured endpoint. "
            "Start it with `corr2surrogate setup-local-llm` (or launch your local provider) and retry."
        )
    if agent == "analyst":
        return (
            "I hit an internal runtime error in this step. "
            "The session is still active; you can retry, change inputs, or use /reset."
        )
    return "I hit an internal runtime error. Please retry."


def _looks_like_llm_failure_message(message: str) -> bool:
    lowered = message.lower()
    return (
        "local llm runtime is not reachable" in lowered
        or ("provider connection error" in lowered and "http" in lowered)
        or "i hit an internal runtime error in this step" in lowered
        or "session is still active; you can retry" in lowered
    )


def _invoke_agent_once_with_recovery(
    *,
    agent: str,
    user_message: str,
    context: dict[str, Any],
    config_path: str | None,
) -> dict[str, Any]:
    try:
        return run_local_agent_once(
            agent=agent,
            user_message=user_message,
            context=context,
            config_path=config_path,
        )
    except Exception as exc:
        if not _is_provider_connection_error(exc):
            raise
        # Best effort: start/check local runtime, then retry exactly once.
        try:
            setup_local_llm(
                config_path=config_path,
                provider=None,
                profile_name=None,
                model=None,
                endpoint=None,
                install_provider=False,
                start_runtime=True,
                pull_model=False,
                download_model=False,
                llama_model_path=None,
                llama_model_url=None,
                timeout_seconds=30,
            )
        except Exception:
            pass
        return run_local_agent_once(
            agent=agent,
            user_message=user_message,
            context=context,
            config_path=config_path,
        )


def _is_provider_connection_error(error: Exception) -> bool:
    lowered = str(error).lower()
    markers = (
        "provider connection error",
        "connection refused",
        "winerror 10061",
        "failed to establish a new connection",
    )
    return any(marker in lowered for marker in markers)


def _recent_user_prompts(*, session_messages: list[dict[str, str]], limit: int) -> list[str]:
    collected: list[str] = []
    for item in reversed(session_messages):
        if str(item.get("role", "")).strip().lower() != "user":
            continue
        text = str(item.get("content", "")).strip()
        if not text:
            continue
        collected.append(text)
        if len(collected) >= max(1, int(limit)):
            break
    collected.reverse()
    return collected


def _looks_like_small_talk(text: str) -> bool:
    normalized = text.strip().lower()
    if not normalized:
        return False
    if _is_simple_greeting(normalized):
        return True
    phrases = {
        "how are you",
        "who are you",
        "what can you do",
        "thank you",
        "thanks",
    }
    if any(phrase in normalized for phrase in phrases):
        return True
    if normalized.endswith("?") and not normalized.startswith("sig_"):
        return True
    return False


if __name__ == "__main__":
    sys.exit(main())
