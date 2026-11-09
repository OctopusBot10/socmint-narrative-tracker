"""Paths, environment loading, and collector defaults.

Values in the real process environment win over `.env`, so a shell export is
never overwritten by the example file.
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
ENV_PATH = ROOT_DIR / ".env"
RESULTS_DIR = ROOT_DIR / "results"

DEFAULT_USER_AGENT = (
    "socmint-prototype/0.1 (research assessment; public posts only)"
)
REDDIT_MIN_INTERVAL_SECONDS = 1.1
REQUEST_TIMEOUT_SECONDS = 25
MAX_REDDIT_PAGES = 4
MAX_YOUTUBE_RESULTS = 50


def load_env_file(path: Path = ENV_PATH) -> None:
    """Load KEY=VALUE lines from `.env` without overriding existing variables."""
    if not path.is_file():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, value)


def env(name: str, default: str = "") -> str:
    load_env_file()
    return os.environ.get(name, default).strip()


load_env_file()
