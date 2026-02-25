from pathlib import Path

from corr2surrogate.orchestration.local_llm_setup import setup_local_llm


def test_setup_local_llm_llama_cpp_ready(monkeypatch, tmp_path: Path) -> None:
    config = {
        "privacy": {"api_calls_allowed": False, "telemetry_allowed": False},
        "runtime": {
            "provider": "llama_cpp",
            "require_local_models": True,
            "block_remote_endpoints": True,
            "offline_mode": True,
            "endpoints": {
                "llama_cpp": "http://127.0.0.1:8000/v1/chat/completions",
            },
            "profiles": {
                "small_cpu": {
                    "model": "models/local.gguf",
                    "cpu_only": True,
                    "n_gpu_layers": 0,
                    "max_context": 2048,
                },
            },
            "default_profile": "small_cpu",
            "fallback_order": ["small_cpu"],
        },
    }
    model_path = tmp_path / "local.gguf"
    model_path.write_bytes(b"gguf")

    monkeypatch.setattr("corr2surrogate.orchestration.local_llm_setup.load_config", lambda _: config)
    monkeypatch.setattr(
        "corr2surrogate.orchestration.local_llm_setup._find_llama_server_binary",
        lambda: "llama-server",
    )
    monkeypatch.setattr(
        "corr2surrogate.orchestration.local_llm_setup._resolve_llama_model_path",
        lambda model, override_path: model_path,
    )
    monkeypatch.setattr(
        "corr2surrogate.orchestration.local_llm_setup._llama_server_ready",
        lambda endpoint, timeout_seconds: True,
    )

    result = setup_local_llm(
        provider="llama_cpp",
        start_runtime=False,
        download_model=False,
    )
    assert result["ready"] is True
    assert result["provider"] == "llama_cpp"
    assert result["suggested_env"]["C2S_PROVIDER"] == "llama_cpp"


def test_setup_local_llm_ollama_missing_binary(monkeypatch) -> None:
    config = {
        "privacy": {"api_calls_allowed": False, "telemetry_allowed": False},
        "runtime": {
            "provider": "ollama",
            "require_local_models": True,
            "block_remote_endpoints": True,
            "offline_mode": True,
            "endpoints": {
                "ollama": "http://127.0.0.1:11434/api/chat",
            },
            "profiles": {
                "small_cpu": {
                    "model": "qwen-test",
                    "cpu_only": True,
                    "n_gpu_layers": 0,
                    "max_context": 2048,
                },
            },
            "default_profile": "small_cpu",
            "fallback_order": ["small_cpu"],
        },
    }
    monkeypatch.setattr("corr2surrogate.orchestration.local_llm_setup.load_config", lambda _: config)
    monkeypatch.setattr("corr2surrogate.orchestration.local_llm_setup.shutil.which", lambda _: None)

    result = setup_local_llm(provider="ollama", install_provider=False)
    assert result["ready"] is False
    assert any("not found" in item.lower() for item in result["errors"])
