"""Command line interface for local harness operations."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from corr2surrogate.orchestration.harness_runner import run_local_agent_once
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
        result = run_local_agent_once(
            agent=args.agent,
            user_message=args.message,
            context=context,
            config_path=args.config,
        )
        print(json.dumps(result, indent=2))
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


if __name__ == "__main__":
    sys.exit(main())
