"""Command line interface for local harness operations."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

from corr2surrogate.orchestration.default_tools import build_default_registry
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
        choices=["ollama", "llama_cpp", "llama.cpp"],
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
            print(json.dumps({"status": "error", "message": message}, indent=2))
            return 1
        print(json.dumps(result, indent=2))
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
        print(json.dumps(result, indent=2))
        return 0

    if args.command == "run-agent1-analysis":
        try:
            predictor_map = _parse_json_object(args.predictor_map_json, arg_name="--predictor-map-json")
            forced_requests = _parse_json_array(
                args.forced_requests_json, arg_name="--forced-requests-json"
            )
        except ValueError as exc:
            parser.error(str(exc))
            return 2
        registry = build_default_registry()
        tool_args: dict[str, Any] = {
            "data_path": args.data_path,
            "predictor_signals_by_target": predictor_map,
            "forced_requests": forced_requests,
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

        result = registry.execute("run_agent1_analysis", tool_args)
        print(json.dumps(result.output, indent=2))
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
        print("agent> I can chat, inspect CSV/XLSX data, run Agent 1 analysis, and save reports.")
        print(
            "agent> Useful commands: /help, /context, /reset, /exit. "
            "At target selection use: list, list <filter>, all, or comma-separated names."
        )
    else:
        print("agent> I can chat and run local modeler workflows through tool calls.")
        print("agent> Useful commands: /help, /context, /reset, /exit.")
    session_messages: list[dict[str, str]] = []
    session_context = dict(base_context)
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
                print(
                    "agent> Commands: /help, /context, /reset, /exit. "
                    "For data analysis paste a .csv/.xlsx path."
                )
                print(
                    "agent> During target selection: type list, list <filter>, "
                    "all, numeric index, or comma-separated signal names."
                )
            else:
                print("agent> Commands: /help, /context, /reset, /exit")
            continue
        if command == "/context":
            snapshot = dict(session_context)
            snapshot["session_messages"] = session_messages
            print(json.dumps(snapshot, indent=2))
            continue
        if command == "/reset":
            session_context = dict(base_context)
            session_messages = []
            print("Session state reset.")
            continue

        if agent == "analyst":
            autopilot = _run_analyst_autopilot_turn(
                user_message=user_message,
                registry=registry,
            )
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
        )
        print(f"agent> {response}")
        if show_json:
            print(json.dumps(result, indent=2))

        session_messages.append({"role": "user", "content": user_message})
        session_messages.append({"role": "assistant", "content": response})
        session_messages = session_messages[-20:]
        session_context["last_event"] = _compact_event_for_context(event)

        turns += 1
        if max_turns > 0 and turns >= max_turns:
            print(f"Reached max turns ({max_turns}). Session ended.")
            return 0


def _run_analyst_autopilot_turn(*, user_message: str, registry: Any) -> dict[str, Any] | None:
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

    print(f"agent> Detected data file: {data_path}")
    preflight_args: dict[str, Any] = {"path": data_path}
    preflight = _execute_registry_tool(registry, "prepare_ingestion_step", preflight_args)
    print(f"agent> Ingestion check: {preflight.get('message', '')}")

    if preflight.get("status") == "needs_user_input":
        options = preflight.get("options") or preflight.get("available_sheets") or []
        selected_sheet = _prompt_sheet_selection(options)
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
    row_count = int(preflight.get("row_count") or 0)
    sample_plan = _prompt_sample_budget(row_count=row_count)
    analysis_args.update(sample_plan)

    data_issue_plan = _prompt_data_issue_handling(preflight=preflight)
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
            "or press Enter to use a quick default subset."
        )
        selected_targets = _prompt_target_selection(
            available_signals=numeric_signals,
            default_count=5,
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

    timestamp_hint = str(preflight.get("timestamp_column_hint") or "").strip()
    estimated_sample_period = _safe_float_or_none(preflight.get("estimated_sample_period_seconds"))
    if timestamp_hint:
        lag_plan = _prompt_lag_preferences(
            timestamp_column_hint=timestamp_hint,
            estimated_sample_period_seconds=estimated_sample_period,
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


def _execute_registry_tool(registry: Any, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    result = registry.execute(tool_name, arguments)
    output = result.output
    if not isinstance(output, dict):
        return {"status": "error", "message": f"Tool '{tool_name}' returned non-object output."}
    return output


def _prompt_sheet_selection(options: list[str]) -> str | None:
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


def _prompt_header_confirmation(*, header_row: int, data_start_row: int) -> tuple[int, int]:
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
            )
        if _looks_like_small_talk(answer):
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
) -> list[str] | None:
    while True:
        answer = input("you> ").strip()
        lowered = answer.lower()
        if lowered == "all":
            return None
        if lowered.startswith("list"):
            query = answer[4:].strip()
            _print_signal_names(available_signals, query=query)
            print(
                "agent> Enter comma-separated target signals, "
                "'all' for full run, or press Enter for quick default subset."
            )
            continue
        selected, unknown = _parse_target_selection_with_unknowns(
            target_answer=answer,
            available_signals=available_signals,
            default_count=default_count,
        )
        if unknown and not selected:
            if _looks_like_small_talk(answer):
                print(
                    "agent> I am good and ready to continue. "
                    "We are currently selecting target signals. "
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


def _prompt_lag_preferences(
    *,
    timestamp_column_hint: str,
    estimated_sample_period_seconds: float | None,
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
            )
            return {"enabled": True, "dimension": "samples", "max_lag": max_lag_samples}
        if dimension in {"seconds", "second", "sec", "s"}:
            lag_seconds = _prompt_positive_float(
                prompt=(
                    "agent> Enter maximum lag window in seconds (positive number, for example 2.5)."
                ),
                default_value=None,
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
            )
            max_lag_samples = max(1, int(round(lag_seconds / sample_period)))
            return {"enabled": True, "dimension": "seconds", "max_lag": max_lag_samples}
        if _looks_like_small_talk(dimension):
            print(
                "agent> We need a lag dimension choice to continue. "
                "Use `samples` or `seconds`."
            )
            continue
        print("agent> Invalid lag dimension. Use `samples` or `seconds`.")


def _prompt_sample_budget(*, row_count: int) -> dict[str, Any]:
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
            print("agent> Sampling mode controls subset selection order. Use uniform/head/tail.")
            continue
        print("agent> Invalid mode. Use uniform, head, or tail.")


def _prompt_data_issue_handling(*, preflight: dict[str, Any]) -> dict[str, Any]:
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
                    )
                    plan["fill_constant_value"] = float(value)
                break
            if _looks_like_small_talk(answer):
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
                )
                plan["sparse_row_min_fraction"] = threshold
                break
            if answer == "manual_range":
                plan["row_coverage_strategy"] = "manual_range"
                start, end = _prompt_manual_row_range()
                plan["row_range_start"] = start
                plan["row_range_end"] = end
                break
            if _looks_like_small_talk(answer):
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


def _prompt_positive_int(*, prompt: str, default_value: int | None) -> int:
    while True:
        print(prompt)
        raw = input("you> ").strip()
        if not raw and default_value is not None:
            return int(default_value)
        try:
            value = int(raw)
        except ValueError:
            if _looks_like_small_talk(raw):
                print("agent> Please provide a positive integer value.")
            else:
                print("agent> Invalid number. Please provide a positive integer.")
            continue
        if value > 0:
            return value
        print("agent> Value must be > 0.")


def _prompt_positive_float(*, prompt: str, default_value: float | None) -> float:
    while True:
        print(prompt)
        raw = input("you> ").strip()
        if not raw and default_value is not None:
            return float(default_value)
        try:
            value = float(raw)
        except ValueError:
            if _looks_like_small_talk(raw):
                print("agent> Please provide a positive numeric value.")
            else:
                print("agent> Invalid number. Please provide a positive numeric value.")
            continue
        if value > 0.0:
            return value
        print("agent> Value must be > 0.")


def _prompt_fraction(*, prompt: str, default_value: float) -> float:
    while True:
        print(prompt)
        raw = input("you> ").strip()
        if not raw:
            return float(default_value)
        try:
            value = float(raw)
        except ValueError:
            if _looks_like_small_talk(raw):
                print("agent> Please provide a number between 0 and 1.")
            else:
                print("agent> Invalid value. Please provide a number between 0 and 1.")
            continue
        if 0.0 < value <= 1.0:
            return value
        print("agent> Threshold must be in (0, 1].")


def _prompt_numeric_value(*, prompt: str, default_value: float) -> float:
    while True:
        print(prompt)
        raw = input("you> ").strip()
        if not raw:
            return float(default_value)
        try:
            return float(raw)
        except ValueError:
            if _looks_like_small_talk(raw):
                print("agent> Please provide a numeric value.")
            else:
                print("agent> Invalid value. Please provide a numeric value.")


def _prompt_manual_row_range() -> tuple[int, int]:
    while True:
        print("agent> Enter manual row range as `start,end` (0-based, inclusive).")
        raw = input("you> ").strip()
        parsed = _parse_row_range(raw)
        if parsed is not None:
            return parsed
        if _looks_like_small_talk(raw):
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


def _rewrite_unhelpful_response(*, agent: str, user_message: str, response: str) -> str:
    if agent != "analyst":
        return response
    lowered = response.lower()
    fallback_markers = (
        "repeated tool argument errors",
        "turn limit before a stable answer",
        "too many invalid actions",
    )
    if any(marker in lowered for marker in fallback_markers):
        if _is_casual_chat_message(user_message):
            return _casual_chat_response(user_message)
        if _extract_first_data_path(user_message) is None:
            return (
                "I can help with that. For data analysis, paste a .csv/.xlsx path. "
                "If you want general guidance, ask directly (for example: "
                "'what checks do you run before correlation?')."
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
    return any(phrase in normalized for phrase in phrases)


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
    return f"Runtime error: {error}"


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
