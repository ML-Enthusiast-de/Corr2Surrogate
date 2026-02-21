"""Agent 2 helpers for training/inference status communication."""

from __future__ import annotations


def build_training_plan_message(
    *,
    model_name: str,
    normalization_enabled: bool,
    split_strategy: str,
) -> str:
    """Summarize the plan before training starts."""
    normalization_text = "enabled" if normalization_enabled else "disabled"
    return (
        f"Training plan: model={model_name}, split_strategy={split_strategy}, "
        f"normalization={normalization_text}."
    )


def build_artifact_saved_message(
    *,
    model_params_path: str,
    normalization_path: str | None,
) -> str:
    """Summarize artifact outputs after optimization/training."""
    parts = [f"Saved tuned model parameters to `{model_params_path}`."]
    if normalization_path is not None:
        parts.append(f"Saved normalization state to `{normalization_path}`.")
    return " ".join(parts)
