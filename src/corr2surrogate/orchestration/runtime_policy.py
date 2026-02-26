"""Runtime policy enforcement for local-only, low-cost agent execution."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse


LOCAL_PROVIDERS = {"ollama", "llama.cpp", "llama_cpp"}
REMOTE_API_PROVIDERS = {"openai", "openai_compatible"}
SUPPORTED_PROVIDERS = LOCAL_PROVIDERS | REMOTE_API_PROVIDERS


class RuntimePolicyError(RuntimeError):
    """Raised when runtime policy checks fail."""


@dataclass(frozen=True)
class ModelProfile:
    """Runtime profile that maps to a local model configuration."""

    name: str
    model: str
    cpu_only: bool
    n_gpu_layers: int | str
    max_context: int


@dataclass(frozen=True)
class RuntimePolicy:
    """Normalized runtime policy loaded from config."""

    provider: str
    require_local_models: bool
    block_remote_endpoints: bool
    offline_mode: bool
    api_calls_allowed: bool
    telemetry_allowed: bool
    profiles: dict[str, ModelProfile]
    default_profile: str
    fallback_order: list[str] = field(default_factory=list)
    model_override: str | None = None
    remote_default_model: str = "gpt-4.1-mini"

    def ensure_local_only(self, endpoint: str | None = None) -> None:
        """Validate runtime settings against local-only policy."""
        provider = self.provider.lower()
        if provider not in SUPPORTED_PROVIDERS:
            raise RuntimePolicyError(
                f"Unsupported provider '{self.provider}'. Supported: {sorted(SUPPORTED_PROVIDERS)}."
            )
        if self.require_local_models and provider not in LOCAL_PROVIDERS:
            raise RuntimePolicyError(
                f"Provider '{self.provider}' is not local-only. Use one of {sorted(LOCAL_PROVIDERS)}."
            )
        if endpoint and self.block_remote_endpoints and not _is_local_endpoint(endpoint):
            raise RuntimePolicyError(
                f"Remote endpoint '{endpoint}' is blocked by runtime policy."
            )
        if not self.api_calls_allowed and endpoint and not _is_local_endpoint(endpoint):
            raise RuntimePolicyError(
                "API calls are disabled by policy, but a remote endpoint was requested."
            )

    def select_profile(self, *, prefer_cpu: bool = True) -> ModelProfile:
        """Choose a runtime profile with optional CPU preference."""
        ordered = [self.default_profile] + [
            name for name in self.fallback_order if name != self.default_profile
        ]
        available = [self.profiles[name] for name in ordered if name in self.profiles]
        if not available:
            raise RuntimePolicyError("No runtime profiles configured.")

        if prefer_cpu:
            for profile in available:
                if profile.cpu_only:
                    return profile
        return available[0]

    def runtime_options(
        self,
        *,
        profile_name: str | None = None,
        prefer_cpu: bool = True,
        endpoint: str | None = None,
    ) -> dict[str, Any]:
        """Return provider options validated against policy."""
        self.ensure_local_only(endpoint=endpoint)
        if profile_name is not None:
            if profile_name not in self.profiles:
                raise RuntimePolicyError(
                    f"Requested profile '{profile_name}' is not configured."
                )
            profile = self.profiles[profile_name]
        else:
            profile = self.select_profile(prefer_cpu=prefer_cpu)
        return {
            "provider": self.provider,
            "model": (
                self.model_override
                or (self.remote_default_model if self.provider.lower() in REMOTE_API_PROVIDERS else profile.model)
            ),
            "cpu_only": profile.cpu_only,
            "n_gpu_layers": profile.n_gpu_layers,
            "max_context": profile.max_context,
            "offline_mode": self.offline_mode,
            "telemetry_allowed": self.telemetry_allowed,
        }


def load_runtime_policy(config: dict[str, Any]) -> RuntimePolicy:
    """Create runtime policy from loaded project config."""
    runtime_cfg = config.get("runtime", {})
    privacy_cfg = config.get("privacy", {})
    profiles_cfg = runtime_cfg.get("profiles", {})

    profiles: dict[str, ModelProfile] = {}
    for name, raw in profiles_cfg.items():
        profiles[name] = ModelProfile(
            name=name,
            model=str(raw.get("model", "")),
            cpu_only=bool(raw.get("cpu_only", False)),
            n_gpu_layers=raw.get("n_gpu_layers", 0),
            max_context=int(raw.get("max_context", 4096)),
        )

    default_profile = str(runtime_cfg.get("default_profile", next(iter(profiles), "")))
    fallback_order = [str(item) for item in runtime_cfg.get("fallback_order", [])]

    policy = RuntimePolicy(
        provider=str(runtime_cfg.get("provider", "ollama")),
        require_local_models=bool(runtime_cfg.get("require_local_models", True)),
        block_remote_endpoints=bool(runtime_cfg.get("block_remote_endpoints", True)),
        offline_mode=bool(runtime_cfg.get("offline_mode", True)),
        api_calls_allowed=bool(privacy_cfg.get("api_calls_allowed", False)),
        telemetry_allowed=bool(privacy_cfg.get("telemetry_allowed", False)),
        profiles=profiles,
        default_profile=default_profile,
        fallback_order=fallback_order,
        model_override=None,
        remote_default_model=str(runtime_cfg.get("remote_default_model", "gpt-4.1-mini")),
    )
    _validate_policy(policy)
    return policy


def apply_environment_overrides(
    policy: RuntimePolicy, *, env: dict[str, str] | None = None
) -> RuntimePolicy:
    """Allow safe, portable profile overrides through environment variables."""
    values = env or os.environ
    provider = values.get("C2S_PROVIDER", policy.provider)
    default_profile = values.get("C2S_PROFILE", policy.default_profile)
    offline_mode = values.get("C2S_OFFLINE_MODE")
    require_local_models = values.get("C2S_REQUIRE_LOCAL_MODELS")
    block_remote_endpoints = values.get("C2S_BLOCK_REMOTE_ENDPOINTS")
    api_calls_allowed = values.get("C2S_API_CALLS_ALLOWED")
    telemetry_allowed = values.get("C2S_TELEMETRY_ALLOWED")
    model_override = values.get("C2S_MODEL")
    remote_default_model = values.get("C2S_REMOTE_DEFAULT_MODEL", policy.remote_default_model)
    resolved_offline = (
        policy.offline_mode
        if offline_mode is None
        else offline_mode.strip().lower() in {"1", "true", "yes", "on"}
    )
    resolved_require_local_models = (
        policy.require_local_models
        if require_local_models is None
        else require_local_models.strip().lower() in {"1", "true", "yes", "on"}
    )
    resolved_block_remote_endpoints = (
        policy.block_remote_endpoints
        if block_remote_endpoints is None
        else block_remote_endpoints.strip().lower() in {"1", "true", "yes", "on"}
    )
    resolved_api_calls_allowed = (
        policy.api_calls_allowed
        if api_calls_allowed is None
        else api_calls_allowed.strip().lower() in {"1", "true", "yes", "on"}
    )
    resolved_telemetry_allowed = (
        policy.telemetry_allowed
        if telemetry_allowed is None
        else telemetry_allowed.strip().lower() in {"1", "true", "yes", "on"}
    )
    updated = RuntimePolicy(
        provider=provider,
        require_local_models=resolved_require_local_models,
        block_remote_endpoints=resolved_block_remote_endpoints,
        offline_mode=resolved_offline,
        api_calls_allowed=resolved_api_calls_allowed,
        telemetry_allowed=resolved_telemetry_allowed,
        profiles=policy.profiles,
        default_profile=default_profile,
        fallback_order=policy.fallback_order,
        model_override=model_override or None,
        remote_default_model=remote_default_model,
    )
    _validate_policy(updated)
    return updated


def _validate_policy(policy: RuntimePolicy) -> None:
    provider = policy.provider.lower()
    if provider not in SUPPORTED_PROVIDERS:
        raise RuntimePolicyError(
            f"Provider '{policy.provider}' is not supported. Use one of {sorted(SUPPORTED_PROVIDERS)}."
        )
    if not policy.profiles:
        raise RuntimePolicyError("At least one runtime profile is required.")
    if policy.default_profile not in policy.profiles:
        raise RuntimePolicyError(
            f"Default profile '{policy.default_profile}' not found in configured profiles."
        )
    for name in policy.fallback_order:
        if name not in policy.profiles:
            raise RuntimePolicyError(
                f"Fallback profile '{name}' not found in configured profiles."
            )
    if policy.require_local_models and provider not in LOCAL_PROVIDERS:
        raise RuntimePolicyError(
            f"Provider '{policy.provider}' is not allowed for local-only execution."
        )
    if policy.offline_mode and policy.api_calls_allowed:
        raise RuntimePolicyError("Offline mode cannot be combined with allowed API calls.")


def _is_local_endpoint(endpoint: str) -> bool:
    parsed = urlparse(endpoint)
    if parsed.scheme == "":
        normalized = endpoint.strip().lower()
        return normalized.startswith(("/", "./", "../", "localhost", "127.0.0.1", "::1"))
    host = (parsed.hostname or "").lower()
    return host in {"localhost", "127.0.0.1", "::1"}
