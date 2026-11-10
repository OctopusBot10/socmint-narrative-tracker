"""Public-profile heuristics for author authenticity.

The bot-risk percent is a weighted checklist. It is not a determination that
an account is automated. Social Analyzer is optional and is skipped when the
package is not installed.
"""

from __future__ import annotations

import re
from importlib import import_module

_TRAILING_DIGITS = re.compile(r"^@?\w+\d{5,}$")


def calculate_author_authenticity(
    author_handle: str,
    follower_count: int = 0,
    following_count: int = 0,
    account_age_days: int = 365,
    bio: str = "",
    has_default_avatar: bool = False,
) -> dict:
    """Return bot risk, authenticity, and the flags that fired.

    Authenticity is ``max(0, 100 - bot_risk_score)``.
    """
    risk = 0
    flags: list[str] = []
    if account_age_days < 30:
        risk += 30
        flags.append("Account created less than 30 days ago")
    if has_default_avatar or not str(bio or "").strip():
        risk += 20
        flags.append("Generic image or empty bio")
    handle = str(author_handle or "").strip()
    if handle and not handle.startswith("@"):
        handle = f"@{handle}"
    if handle and _TRAILING_DIGITS.match(handle):
        risk += 25
        flags.append("Auto-generated handle pattern")
    if int(following_count or 0) > 2000 and int(follower_count or 0) < 10:
        risk += 25
        flags.append("High following ratio")
    risk = min(risk, 100)
    authenticity = max(0, 100 - risk)
    return {
        "bot_risk_pct": risk,
        "authenticity_score": authenticity,
        "risk_flags": flags,
        "badge": authenticity_badge(authenticity),
    }


def authenticity_badge(score: int) -> str:
    """Map an authenticity percent to the card label."""
    if score > 75:
        return "AUTHENTIC PROXY"
    if score >= 50:
        return "UNVERIFIED FOOTPRINT"
    return "SUSPICIOUS BOT PATTERN"


def run_social_analyzer_lookup(username: str):
    """Look up a public username on YouTube, Twitter, and Telegram.

    Returns an error dict when social-analyzer is not installed or the lookup fails.
    """
    try:
        social_analyzer = import_module("social-analyzer")
        analyzer = social_analyzer.SocialAnalyzer(silent=True)
        return analyzer.run_as_object(
            username=username,
            websites="youtube twitter telegram",
            metadata=True,
            extract=True,
            silent=True,
        )
    except ImportError:
        return {"error": "social-analyzer package not installed"}
    except Exception as exc:
        return {"error": str(exc)}
