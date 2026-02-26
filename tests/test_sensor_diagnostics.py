import numpy as np
import pandas as pd

from corr2surrogate.analytics.sensor_diagnostics import run_sensor_diagnostics


def test_run_sensor_diagnostics_flags_saturation_and_stuck() -> None:
    n = 200
    signal_sat = np.concatenate([np.zeros(80), np.linspace(0, 1, 40), np.ones(80)])
    signal_stuck = np.concatenate([np.linspace(0, 1, 100), np.full(100, 2.0)])
    frame = pd.DataFrame(
        {
            "time": np.arange(n),
            "sat": signal_sat,
            "stuck": signal_stuck,
        }
    )
    summary = run_sensor_diagnostics(frame, timestamp_column="time")
    payload = summary.to_dict()
    assert payload["diagnostics"]
    by_signal = {row["signal"]: row for row in payload["diagnostics"]}
    assert "sat" in by_signal
    assert "stuck" in by_signal
    assert any(flag in by_signal["sat"]["flags"] for flag in ["saturation", "quantized"])
    assert "stuck_segments" in by_signal["stuck"]["flags"]
