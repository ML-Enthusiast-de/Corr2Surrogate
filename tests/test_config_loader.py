from pathlib import Path

from corr2surrogate.core.config import load_config


def test_load_config_supports_explicit_path(tmp_path: Path) -> None:
    cfg = tmp_path / "c2s.yaml"
    cfg.write_text("project:\n  name: test\n", encoding="utf-8")
    loaded = load_config(cfg)
    assert loaded["project"]["name"] == "test"
