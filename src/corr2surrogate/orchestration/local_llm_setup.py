"""Local runtime setup and health checks for LLM-backed agent execution."""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from dataclasses import replace
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from corr2surrogate.core.config import load_config

from .runtime_policy import (
    LOCAL_PROVIDERS,
    REMOTE_API_PROVIDERS,
    apply_environment_overrides,
    load_runtime_policy,
)

DEFAULT_LLAMA_MODEL_PATH = Path("models/Qwen_Qwen3-4B-Q4_K_M.gguf")
DEFAULT_LLAMA_MODEL_URL = (
    "https://huggingface.co/bartowski/Qwen_Qwen3-4B-GGUF/resolve/main/"
    "Qwen_Qwen3-4B-Q4_K_M.gguf?download=true"
)
DEFAULT_LLAMA_ALIAS = "c2s-4b"


def setup_local_llm(
    *,
    config_path: str | None = None,
    provider: str | None = None,
    profile_name: str | None = None,
    model: str | None = None,
    endpoint: str | None = None,
    install_provider: bool = False,
    start_runtime: bool = True,
    pull_model: bool = True,
    download_model: bool = True,
    llama_model_path: str | None = None,
    llama_model_url: str | None = None,
    timeout_seconds: int = 120,
) -> dict[str, Any]:
    """Set up configured local/remote LLM runtime and return a machine-readable report."""
    config = load_config(config_path)
    policy = apply_environment_overrides(load_runtime_policy(config))

    target_provider = (provider or policy.provider).strip().lower()
    if target_provider not in LOCAL_PROVIDERS and target_provider not in REMOTE_API_PROVIDERS:
        raise ValueError(
            "Unsupported provider "
            f"'{target_provider}'. Use one of {sorted(LOCAL_PROVIDERS | REMOTE_API_PROVIDERS)}."
        )
    policy = replace(policy, provider=target_provider)

    runtime_cfg = config.get("runtime", {})
    resolved_endpoint = endpoint or _resolve_endpoint(target_provider, runtime_cfg)
    options = policy.runtime_options(
        profile_name=profile_name,
        prefer_cpu=True,
        endpoint=resolved_endpoint,
    )
    resolved_model = model or str(options.get("model", ""))

    if target_provider in REMOTE_API_PROVIDERS:
        return _setup_remote_api(
            provider=target_provider,
            model=resolved_model,
            endpoint=resolved_endpoint,
        )

    if target_provider == "ollama":
        return _setup_ollama(
            model=resolved_model,
            endpoint=resolved_endpoint,
            install_provider=install_provider,
            start_runtime=start_runtime,
            pull_model=pull_model,
            timeout_seconds=timeout_seconds,
        )
    return _setup_llama_cpp(
        model=resolved_model,
        endpoint=resolved_endpoint,
        install_provider=install_provider,
        start_runtime=start_runtime,
        download_model=download_model,
        llama_model_path=llama_model_path,
        llama_model_url=llama_model_url,
        cpu_only=bool(options.get("cpu_only", True)),
        n_gpu_layers=options.get("n_gpu_layers", 0),
        max_context=int(options.get("max_context", 4096)),
        timeout_seconds=timeout_seconds,
    )


def _setup_remote_api(
    *,
    provider: str,
    model: str,
    endpoint: str,
) -> dict[str, Any]:
    api_key = os.getenv("C2S_API_KEY", "").strip()
    if not api_key and provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
    report: dict[str, Any] = {
        "provider": provider,
        "endpoint": endpoint,
        "model": model,
        "ready": False,
        "steps": [],
        "warnings": [],
        "errors": [],
    }
    if provider == "openai" and not api_key:
        report["errors"].append(
            "Missing API key. Set C2S_API_KEY (or OPENAI_API_KEY) to enable OpenAI calls."
        )
        report["suggested_env"] = {
            "C2S_PROVIDER": "openai",
            "C2S_MODEL": model,
            "C2S_ENDPOINT": endpoint,
            "C2S_API_KEY": "<set-your-key>",
        }
        return report

    if provider == "openai_compatible" and not api_key:
        report["warnings"].append(
            "No C2S_API_KEY provided. This is allowed for some endpoints, but many require bearer auth."
        )

    report["ready"] = True
    report["suggested_env"] = {
        "C2S_PROVIDER": provider,
        "C2S_MODEL": model,
        "C2S_ENDPOINT": endpoint,
    }
    if api_key:
        report["suggested_env"]["C2S_API_KEY"] = "<already-set>"
    return report


def _setup_ollama(
    *,
    model: str,
    endpoint: str,
    install_provider: bool,
    start_runtime: bool,
    pull_model: bool,
    timeout_seconds: int,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "provider": "ollama",
        "endpoint": endpoint,
        "model": model,
        "ready": False,
        "steps": [],
        "warnings": [],
        "errors": [],
    }
    binary = shutil.which("ollama")
    if binary is None and install_provider:
        install_result = _run_command(
            [
                "winget",
                "install",
                "--id",
                "Ollama.Ollama",
                "-e",
                "--accept-package-agreements",
                "--accept-source-agreements",
                "--disable-interactivity",
            ],
            timeout_seconds=1800,
        )
        report["steps"].append(
            {
                "action": "install_provider",
                "ok": install_result["ok"],
                "exit_code": install_result["exit_code"],
                "stderr": install_result["stderr_tail"],
            }
        )
        binary = shutil.which("ollama")

    if binary is None:
        report["errors"].append("Ollama binary not found in PATH.")
        report["suggested_env"] = {
            "C2S_PROVIDER": "llama_cpp",
            "C2S_MODEL": DEFAULT_LLAMA_ALIAS,
        }
        return report
    report["binary"] = binary

    server_up = _ollama_server_ready(endpoint, timeout_seconds=5)
    if not server_up and start_runtime:
        started = _spawn_background_process([binary, "serve"])
        report["steps"].append({"action": "start_runtime", "ok": started})
        server_up = _wait_until(
            lambda: _ollama_server_ready(endpoint, timeout_seconds=5),
            timeout_seconds=20,
        )
    if not server_up:
        report["errors"].append(
            "Ollama endpoint is unreachable. Start it with `ollama serve` and retry."
        )
        return report

    models = _list_ollama_models(endpoint, timeout_seconds=timeout_seconds)
    report["available_models"] = models
    has_model = _ollama_has_model(requested=model, available=models)
    if not has_model and pull_model:
        pull_result = _run_command(
            [binary, "pull", model],
            timeout_seconds=7200,
        )
        report["steps"].append(
            {
                "action": "pull_model",
                "ok": pull_result["ok"],
                "exit_code": pull_result["exit_code"],
                "stderr": pull_result["stderr_tail"],
            }
        )
        models = _list_ollama_models(endpoint, timeout_seconds=timeout_seconds)
        report["available_models"] = models
        has_model = _ollama_has_model(requested=model, available=models)

    if not has_model:
        report["errors"].append(
            f"Model '{model}' is not available in Ollama. Run `ollama pull {model}`."
        )
        return report

    report["ready"] = True
    report["suggested_env"] = {
        "C2S_PROVIDER": "ollama",
        "C2S_MODEL": model,
    }
    return report


def _setup_llama_cpp(
    *,
    model: str,
    endpoint: str,
    install_provider: bool,
    start_runtime: bool,
    download_model: bool,
    llama_model_path: str | None,
    llama_model_url: str | None,
    cpu_only: bool,
    n_gpu_layers: int | str,
    max_context: int,
    timeout_seconds: int,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "provider": "llama_cpp",
        "endpoint": endpoint,
        "model": model,
        "ready": False,
        "steps": [],
        "warnings": [],
        "errors": [],
    }
    binary = _find_llama_server_binary()
    if binary is None and install_provider:
        install_result = _run_command(
            [
                "winget",
                "install",
                "--id",
                "ggml.llamacpp",
                "-e",
                "--accept-package-agreements",
                "--accept-source-agreements",
                "--disable-interactivity",
            ],
            timeout_seconds=900,
        )
        report["steps"].append(
            {
                "action": "install_provider",
                "ok": install_result["ok"],
                "exit_code": install_result["exit_code"],
                "stderr": install_result["stderr_tail"],
            }
        )
        binary = _find_llama_server_binary()
    if binary is None:
        report["errors"].append("`llama-server` binary not found.")
        return report
    report["binary"] = binary

    resolved_model_path = _resolve_llama_model_path(model=model, override_path=llama_model_path)
    report["resolved_model_path"] = str(resolved_model_path)
    if not resolved_model_path.is_file() and download_model:
        source_url = llama_model_url or DEFAULT_LLAMA_MODEL_URL
        download_result = _download_file(
            url=source_url,
            destination=resolved_model_path,
            timeout_seconds=timeout_seconds,
        )
        report["steps"].append(
            {
                "action": "download_model",
                "ok": download_result["ok"],
                "bytes_written": download_result["bytes_written"],
                "error": download_result["error"],
            }
        )
    if not resolved_model_path.is_file():
        report["errors"].append(
            f"Model file missing at '{resolved_model_path}'. Provide --llama-model-path."
        )
        return report

    server_up = _llama_server_ready(endpoint, timeout_seconds=5)
    if not server_up and start_runtime:
        host, port = _extract_host_port(endpoint)
        args = [
            binary,
            "--model",
            str(resolved_model_path),
            "--alias",
            DEFAULT_LLAMA_ALIAS,
            "--host",
            host,
            "--port",
            str(port),
            "--ctx-size",
            str(max_context),
        ]
        if cpu_only or str(n_gpu_layers) == "0":
            args.extend(["--n-gpu-layers", "0"])
        started = _spawn_background_process(args)
        report["steps"].append({"action": "start_runtime", "ok": started})
        server_up = _wait_until(
            lambda: _llama_server_ready(endpoint, timeout_seconds=5),
            timeout_seconds=30,
        )
    if not server_up:
        report["errors"].append(
            "llama.cpp endpoint is unreachable. Start it with "
            f"`{binary} --model {resolved_model_path} --alias {DEFAULT_LLAMA_ALIAS}`."
        )
        return report

    report["ready"] = True
    report["suggested_env"] = {
        "C2S_PROVIDER": "llama_cpp",
        "C2S_MODEL": DEFAULT_LLAMA_ALIAS,
    }
    return report


def _resolve_endpoint(provider: str, runtime_cfg: dict[str, Any]) -> str:
    provider_key = provider.lower()
    endpoints = runtime_cfg.get("endpoints", {})
    if provider_key == "ollama":
        return str(endpoints.get("ollama", "http://127.0.0.1:11434/api/chat"))
    if provider_key in {"llama.cpp", "llama_cpp"}:
        return str(
            endpoints.get(
                "llama_cpp",
                "http://127.0.0.1:8000/v1/chat/completions",
            )
        )
    if provider_key == "openai":
        return str(endpoints.get("openai", "https://api.openai.com/v1/chat/completions"))
    if provider_key == "openai_compatible":
        return str(
            endpoints.get(
                "openai_compatible",
                "https://api.openai.com/v1/chat/completions",
            )
        )
    raise ValueError(f"Unsupported provider '{provider}'.")


def _ollama_server_ready(endpoint: str, *, timeout_seconds: int) -> bool:
    try:
        payload = _http_get_json(_endpoint_origin(endpoint) + "/api/tags", timeout_seconds)
    except RuntimeError:
        return False
    return isinstance(payload.get("models"), list)


def _list_ollama_models(endpoint: str, *, timeout_seconds: int) -> list[str]:
    try:
        payload = _http_get_json(_endpoint_origin(endpoint) + "/api/tags", timeout_seconds)
    except RuntimeError:
        return []
    models = payload.get("models") or []
    names: list[str] = []
    for item in models:
        name = str(item.get("name", "")).strip()
        if name:
            names.append(name)
    return names


def _ollama_has_model(*, requested: str, available: list[str]) -> bool:
    requested = requested.strip()
    if requested in available:
        return True
    if ":" not in requested and f"{requested}:latest" in available:
        return True
    if ":" in requested:
        base = requested.split(":", 1)[0]
        return any(name == requested or name.startswith(f"{base}:") for name in available)
    return False


def _llama_server_ready(endpoint: str, *, timeout_seconds: int) -> bool:
    health_url = _endpoint_origin(endpoint) + "/health"
    request = Request(health_url, method="GET")
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            status_code = getattr(response, "status", 200)
    except (HTTPError, URLError, TimeoutError):
        return False
    return int(status_code) < 400


def _extract_host_port(endpoint: str) -> tuple[str, int]:
    parsed = urlparse(endpoint)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port
    if port is None:
        port = 443 if parsed.scheme == "https" else 80
    return host, int(port)


def _endpoint_origin(endpoint: str) -> str:
    parsed = urlparse(endpoint)
    if parsed.scheme and parsed.netloc:
        return f"{parsed.scheme}://{parsed.netloc}"
    return endpoint.split("/api", 1)[0].rstrip("/")


def _http_get_json(url: str, timeout_seconds: int) -> dict[str, Any]:
    request = Request(url, method="GET")
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            raw = response.read().decode("utf-8")
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore") if exc.fp else str(exc)
        raise RuntimeError(f"HTTP error {exc.code}: {detail}") from exc
    except URLError as exc:
        raise RuntimeError(f"Connection error: {exc}") from exc
    except TimeoutError as exc:
        raise RuntimeError("Request timed out.") from exc

    try:
        import json

        payload = json.loads(raw)
    except Exception as exc:  # pragma: no cover - defensive guard
        raise RuntimeError("Invalid JSON response.") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("Unexpected JSON payload.")
    return payload


def _find_llama_server_binary() -> str | None:
    direct = shutil.which("llama-server")
    if direct:
        return direct
    links = Path(os.getenv("LOCALAPPDATA", "")) / "Microsoft" / "WinGet" / "Links" / "llama-server.exe"
    if links.is_file():
        return str(links)
    package_root = Path(os.getenv("LOCALAPPDATA", "")) / "Microsoft" / "WinGet" / "Packages"
    if package_root.is_dir():
        for item in package_root.rglob("llama-server.exe"):
            return str(item)
    return None


def _resolve_llama_model_path(*, model: str, override_path: str | None) -> Path:
    if override_path:
        return Path(override_path).expanduser().resolve()
    if model.endswith(".gguf") or "/" in model or "\\" in model:
        return Path(model).expanduser().resolve()
    return DEFAULT_LLAMA_MODEL_PATH.resolve()


def _download_file(url: str, destination: Path, *, timeout_seconds: int) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = Request(url, method="GET")
    bytes_written = 0
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            with destination.open("wb") as handle:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    handle.write(chunk)
                    bytes_written += len(chunk)
    except Exception as exc:  # pragma: no cover - network/system dependent
        return {"ok": False, "bytes_written": bytes_written, "error": str(exc)}
    return {"ok": True, "bytes_written": bytes_written, "error": ""}


def _run_command(args: list[str], *, timeout_seconds: int) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            args,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except Exception as exc:  # pragma: no cover - system dependent
        return {
            "ok": False,
            "exit_code": -1,
            "stdout_tail": "",
            "stderr_tail": str(exc),
        }
    return {
        "ok": completed.returncode == 0,
        "exit_code": int(completed.returncode),
        "stdout_tail": _tail(completed.stdout),
        "stderr_tail": _tail(completed.stderr),
    }


def _spawn_background_process(args: list[str]) -> bool:
    creationflags = 0
    kwargs: dict[str, Any] = {}
    if os.name == "nt":
        creationflags = (
            getattr(subprocess, "DETACHED_PROCESS", 0)
            | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        )
    else:
        kwargs["start_new_session"] = True
    try:
        subprocess.Popen(
            args,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
            creationflags=creationflags,
            **kwargs,
        )
    except Exception:
        return False
    return True


def _wait_until(check: Any, *, timeout_seconds: int, interval_seconds: float = 1.0) -> bool:
    deadline = time.monotonic() + float(timeout_seconds)
    while time.monotonic() < deadline:
        if bool(check()):
            return True
        time.sleep(interval_seconds)
    return False


def _tail(text: str | None, *, max_lines: int = 8) -> str:
    if not text:
        return ""
    lines = text.splitlines()
    return "\n".join(lines[-max_lines:])
