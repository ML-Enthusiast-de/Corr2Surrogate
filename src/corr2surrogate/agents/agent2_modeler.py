"""Agent 2 helpers for training, iteration, and inference communication."""

from __future__ import annotations

from corr2surrogate.orchestration.workflow import LoopEvaluation


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


def build_loop_feedback_message(loop_eval: LoopEvaluation) -> str:
    """Create a user-facing update after each optimization iteration."""
    unmet = ", ".join(loop_eval.unmet_criteria) if loop_eval.unmet_criteria else "none"
    guidance = " ".join(loop_eval.recommendations)
    return (
        f"Iteration {loop_eval.attempt}/{loop_eval.max_attempts}: {loop_eval.summary} "
        f"Unmet criteria: {unmet}. {guidance}"
    )


def build_post_test_feedback_message(feedback: dict) -> str:
    """Summarize post-test diagnostics and new data trajectory suggestions."""
    payload = feedback.get("feedback", feedback)
    summary = str(payload.get("summary", "No summary available."))
    suggestions = payload.get("trajectory_suggestions", [])
    if not suggestions:
        return f"{summary} No additional trajectory recommendations were generated."
    top_titles = ", ".join(str(item.get("title", "trajectory")) for item in suggestions[:3])
    return f"{summary} Suggested next data trajectories: {top_titles}."
