"""Flag public posting patterns that can indicate coordinated amplification.

Two indicators are implemented:

1. Near-duplicate burst. Different authors post identical or near-identical
   text inside a short time window. A group that also spans Reddit and YouTube
   is marked as cross-platform amplification.
2. Shared-link burst. Different authors post the same outbound URL inside the
   window, even when the surrounding words differ.

Usernames that show up in more than one alert are listed as repeat presence.
That is a count, not an identity finding.

These are text-and-time heuristics. Copied headlines, quotes, and ordinary
reactions to the same event produce the same signals.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone

import numpy as np
from difflib import SequenceMatcher

from models import NarrativeCluster, Post

_ANONYMOUS_AUTHORS = {
    "[deleted]",
    "[removed]",
    "automoderator",
    "auto-moderator",
    "unknown",
    "",
}
_LIMITATION = (
    "This is a text-and-time heuristic on the posts that were collected. "
    "Matching wording or a shared link can be a copied headline, a quote, "
    "or many people reacting to the same event. It is not evidence of "
    "inauthentic behavior, a shared operator, or intent."
)


def detect_coordination(
    posts: list[Post],
    clusters: list[NarrativeCluster],
    similarity: np.ndarray | None = None,
    *,
    window_hours: float = 6.0,
    text_similarity_threshold: float = 0.86,
    min_chars: int = 40,
) -> list[dict]:
    """Return JSON-ready alert dicts, highest severity first."""
    del clusters  # clustering is reported separately; alerts key off the posts
    if len(posts) < 2:
        return []

    window_seconds = float(window_hours) * 3600.0
    normalized = [_normalize(post.full_text()) for post in posts]
    times = [_as_utc(post.created_at) for post in posts]
    order = sorted(range(len(posts)), key=lambda index: times[index])

    parent = list(range(len(posts)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        root_left, root_right = find(left), find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    for position, left in enumerate(order):
        if len(normalized[left]) < min_chars:
            continue
        if _is_anonymous(posts[left].author):
            continue
        for right in order[position + 1 :]:
            delta = (times[right] - times[left]).total_seconds()
            if delta > window_seconds:
                break
            if _is_anonymous(posts[right].author):
                continue
            if posts[left].author.strip().lower() == posts[right].author.strip().lower():
                continue
            if len(normalized[right]) < min_chars:
                continue
            if _texts_match(
                normalized[left],
                normalized[right],
                left,
                right,
                similarity,
                text_similarity_threshold,
            ):
                union(left, right)

    alerts: list[dict] = []
    components: dict[int, list[int]] = {}
    for index in range(len(posts)):
        if len(normalized[index]) < min_chars or _is_anonymous(posts[index].author):
            continue
        components.setdefault(find(index), []).append(index)

    used_ids: set[str] = set()
    for members in components.values():
        authors = {_display_author(posts[index]) for index in members}
        if len(members) < 2 or len(authors) < 2:
            continue
        member_times = [times[index] for index in members]
        span_hours = (
            max(member_times) - min(member_times)
        ).total_seconds() / 3600.0
        if span_hours > window_hours:
            continue
        score, severity = _score(
            authors=len(authors),
            platforms=len({posts[index].platform for index in members}),
            similarity=_group_similarity(members, normalized, similarity),
            span_hours=span_hours,
            window_hours=window_hours,
        )
        platforms = sorted({posts[index].platform for index in members})
        indicator = (
            "cross_platform_near_duplicate"
            if len(platforms) > 1
            else "near_duplicate_burst"
        )
        post_ids = [posts[index].post_id for index in members]
        used_ids.update(post_ids)
        alerts.append(
            _alert(
                indicator=indicator,
                severity=severity,
                score=score,
                posts=[posts[index] for index in members],
                summary=_near_duplicate_summary(posts, members, platforms, span_hours),
                extra={
                    "text_similarity": round(
                        _group_similarity(members, normalized, similarity), 3
                    ),
                    "span_hours": round(span_hours, 2),
                },
            )
        )

    alerts.extend(_shared_link_alerts(posts, times, window_seconds, used_ids))
    alerts.sort(key=lambda alert: ({"high": 0, "medium": 1, "low": 2}[alert["severity"]], -alert["score"]))
    for index, alert in enumerate(alerts, start=1):
        alert["alert_id"] = f"A{index}"
    return alerts


def repeat_authors(alerts: list[dict]) -> list[dict]:
    """Authors that appear in two or more separate alerts."""
    counts: dict[str, int] = {}
    for alert in alerts:
        for author in set(alert.get("authors", [])):
            counts[author] = counts.get(author, 0) + 1
    repeats = [
        {"author": author, "alert_count": count}
        for author, count in counts.items()
        if count >= 2
    ]
    repeats.sort(key=lambda item: (-item["alert_count"], item["author"]))
    return repeats


def _shared_link_alerts(
    posts: list[Post],
    times: list[datetime],
    window_seconds: float,
    already_used: set[str],
) -> list[dict]:
    by_url: dict[str, list[int]] = {}
    for index, post in enumerate(posts):
        for url in post.outbound_urls:
            key = url.split("?")[0].rstrip("/").lower()
            if not key.startswith("http"):
                continue
            by_url.setdefault(key, []).append(index)

    alerts: list[dict] = []
    for url, indexes in by_url.items():
        unique_indexes = sorted(set(indexes), key=lambda index: times[index])
        authors_seen: list[int] = []
        named: set[str] = set()
        for index in unique_indexes:
            author = posts[index].author.strip().lower()
            if _is_anonymous(author) or author in named:
                continue
            named.add(author)
            authors_seen.append(index)
        if len(authors_seen) < 3:
            continue
        # Keep the tightest prefix of the time-sorted authors that still
        # fits in the window and has at least three accounts.
        chosen: list[int] | None = None
        for start in range(len(authors_seen)):
            for finish in range(start + 3, len(authors_seen) + 1):
                group = authors_seen[start:finish]
                span = (times[group[-1]] - times[group[0]]).total_seconds()
                if span <= window_seconds and (
                    chosen is None or len(group) > len(chosen)
                ):
                    chosen = group
        if not chosen:
            continue
        post_ids = {posts[index].post_id for index in chosen}
        if post_ids <= already_used:
            continue
        span_hours = (times[chosen[-1]] - times[chosen[0]]).total_seconds() / 3600.0
        platforms = sorted({posts[index].platform for index in chosen})
        score, severity = _score(
            authors=len(chosen),
            platforms=len(platforms),
            similarity=0.55,
            span_hours=span_hours,
            window_hours=window_seconds / 3600.0,
        )
        # Shared links are weaker than copied text, so cap severity at medium.
        if severity == "high":
            severity = "medium"
            score = min(score, 0.74)
        alerts.append(
            _alert(
                indicator="shared_link_burst",
                severity=severity,
                score=score,
                posts=[posts[index] for index in chosen],
                summary=(
                    f"{len(chosen)} accounts posted the same outbound link "
                    f"within {span_hours:.1f} hours."
                ),
                extra={"shared_url": url, "span_hours": round(span_hours, 2)},
            )
        )
    return alerts


def _alert(
    *,
    indicator: str,
    severity: str,
    score: float,
    posts: list[Post],
    summary: str,
    extra: dict,
) -> dict:
    ordered = sorted(posts, key=lambda post: post.created_at)
    return {
        "alert_id": "",
        "indicator": indicator,
        "severity": severity,
        "score": round(score, 3),
        "summary": summary,
        "authors": sorted({post.author for post in ordered}),
        "platforms": sorted({post.platform for post in ordered}),
        "post_ids": [post.post_id for post in ordered],
        "time_start": ordered[0].created_at.isoformat(),
        "time_end": ordered[-1].created_at.isoformat(),
        "evidence": extra,
        "limitation": _LIMITATION,
        "posts": [
            {
                "post_id": post.post_id,
                "platform": post.platform,
                "author": post.author,
                "created_at": post.created_at.isoformat(),
                "url": post.url,
                "title": post.title,
            }
            for post in ordered
        ],
    }


def _near_duplicate_summary(
    posts: list[Post],
    members: list[int],
    platforms: list[str],
    span_hours: float,
) -> str:
    authors = len({posts[index].author for index in members})
    platform_text = " and ".join(platforms)
    if len(platforms) > 1:
        return (
            f"{authors} accounts posted near-identical text across {platform_text} "
            f"within {span_hours:.1f} hours."
        )
    return (
        f"{authors} accounts posted near-identical text on {platform_text} "
        f"within {span_hours:.1f} hours."
    )


def _texts_match(
    left_text: str,
    right_text: str,
    left: int,
    right: int,
    similarity: np.ndarray | None,
    threshold: float,
) -> bool:
    if left_text == right_text:
        return True
    if SequenceMatcher(None, left_text, right_text).ratio() >= threshold:
        return True
    if similarity is not None and similarity.size:
        return float(similarity[left, right]) >= max(threshold, 0.9)
    return False


def _group_similarity(
    members: list[int],
    normalized: list[str],
    similarity: np.ndarray | None,
) -> float:
    scores: list[float] = []
    for offset, left in enumerate(members):
        for right in members[offset + 1 :]:
            if similarity is not None and similarity.size:
                scores.append(float(similarity[left, right]))
            else:
                scores.append(SequenceMatcher(None, normalized[left], normalized[right]).ratio())
    if not scores:
        return 1.0
    return float(sum(scores) / len(scores))


def _score(
    *,
    authors: int,
    platforms: int,
    similarity: float,
    span_hours: float,
    window_hours: float,
) -> tuple[float, str]:
    closeness = 1.0 - min(span_hours / max(window_hours, 0.1), 1.0)
    score = (
        0.5 * similarity
        + 0.25 * min(authors, 6) / 6
        + 0.15 * closeness
        + (0.1 if platforms > 1 else 0.0)
    )
    score = float(min(max(score, 0.0), 0.99))
    if authors >= 4 and similarity >= 0.8:
        severity = "high"
    elif authors >= 3 or platforms > 1:
        severity = "medium"
    else:
        severity = "low"
    return score, severity


def _normalize(text: str) -> str:
    lowered = text.lower()
    lowered = re.sub(r"https?://\S+", " ", lowered)
    lowered = re.sub(r"[^a-z0-9\s]", " ", lowered)
    return re.sub(r"\s+", " ", lowered).strip()


def _is_anonymous(author: str) -> bool:
    return author.strip().lower() in _ANONYMOUS_AUTHORS


def _display_author(post: Post) -> str:
    return post.author.strip()


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)
