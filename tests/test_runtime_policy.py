from corr2surrogate.orchestration.runtime_policy import (
    RuntimePolicyError,
    load_runtime_policy,
)


def test_runtime_policy_selects_cpu_profile() -> None:
    config = {
        "privacy": {"api_calls_allowed": False, "telemetry_allowed": False},
        "runtime": {
            "provider": "ollama",
            "require_local_models": True,
            "block_remote_endpoints": True,
            "offline_mode": True,
            "profiles": {
                "small_cpu": {
                    "model": "qwen:3b",
                    "cpu_only": True,
                    "n_gpu_layers": 0,
                    "max_context": 4096,
                },
                "balanced": {
                    "model": "qwen:7b",
                    "cpu_only": False,
                    "n_gpu_layers": "auto",
                    "max_context": 8192,
                },
            },
            "default_profile": "balanced",
            "fallback_order": ["small_cpu"],
        },
    }
    policy = load_runtime_policy(config)
    selected = policy.select_profile(prefer_cpu=True)
    assert selected.name == "small_cpu"


def test_runtime_policy_blocks_remote_endpoint() -> None:
    config = {
        "privacy": {"api_calls_allowed": False, "telemetry_allowed": False},
        "runtime": {
            "provider": "ollama",
            "require_local_models": True,
            "block_remote_endpoints": True,
            "offline_mode": True,
            "profiles": {
                "small_cpu": {
                    "model": "qwen:3b",
                    "cpu_only": True,
                    "n_gpu_layers": 0,
                    "max_context": 4096,
                },
            },
            "default_profile": "small_cpu",
            "fallback_order": ["small_cpu"],
        },
    }
    policy = load_runtime_policy(config)
    try:
        policy.ensure_local_only(endpoint="https://api.example.com/v1/chat/completions")
    except RuntimePolicyError:
        return
    raise AssertionError("Expected RuntimePolicyError for remote endpoint.")


def test_runtime_policy_rejects_unknown_profile_name() -> None:
    config = {
        "privacy": {"api_calls_allowed": False, "telemetry_allowed": False},
        "runtime": {
            "provider": "ollama",
            "require_local_models": True,
            "block_remote_endpoints": True,
            "offline_mode": True,
            "profiles": {
                "small_cpu": {
                    "model": "qwen:3b",
                    "cpu_only": True,
                    "n_gpu_layers": 0,
                    "max_context": 4096,
                },
            },
            "default_profile": "small_cpu",
            "fallback_order": ["small_cpu"],
        },
    }
    policy = load_runtime_policy(config)
    try:
        policy.runtime_options(profile_name="missing_profile")
    except RuntimePolicyError:
        return
    raise AssertionError("Expected RuntimePolicyError for unknown profile.")
