"""Pytest root configuration ensuring repository packages take precedence on sys.path."""
import json
import os
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


@pytest.fixture(autouse=True)
def _protect_live_dev_config(tmp_path, monkeypatch):
    """Redirect writes targeting live REPO_ROOT config/session files so tests never clobber dev settings."""
    live_cfg = os.path.normcase(os.path.abspath(os.path.join(REPO_ROOT, "desktop_config.json")))
    live_session = os.path.normcase(os.path.abspath(os.path.join(REPO_ROOT, ".amt_last_session.json")))

    from app.core import config as app_cfg_mod
    from desktop.core import config_manager as desk_cfg_mod

    orig_app_save = app_cfg_mod.AppConfig.save

    def _safe_app_save(self, filepath: str = "desktop_config.json") -> None:
        resolved = os.path.normcase(os.path.abspath(app_cfg_mod.resolve_app_path(filepath)))
        if resolved in (live_cfg, live_session):
            filepath = str(tmp_path / os.path.basename(resolved))
        return orig_app_save(self, filepath)

    monkeypatch.setattr(app_cfg_mod.AppConfig, "save", _safe_app_save)

    def _safe_desk_save(self) -> None:
        target = desk_cfg_mod.CONFIG_FILE_PATH
        if os.path.normcase(os.path.abspath(target)) == live_cfg:
            target = str(tmp_path / "desktop_config.json")
        try:
            with open(target, "w", encoding="utf-8") as f:
                json.dump(self._data, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    monkeypatch.setattr(desk_cfg_mod.ConfigManager, "save", _safe_desk_save)

    if "app.ui.main_window" in sys.modules:
        mw_mod = sys.modules["app.ui.main_window"]
        if hasattr(mw_mod, "MainWindow"):
            monkeypatch.setattr(mw_mod.MainWindow, "_save_workspace_session", lambda self: None)
    else:
        try:
            from app.ui import main_window as mw_mod
            monkeypatch.setattr(mw_mod.MainWindow, "_save_workspace_session", lambda self: None)
        except Exception:
            pass
