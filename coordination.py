"""Flag public posting patterns that can indicate coordinated amplification.

Two indicators are implemented, and each alert lists every reason that applies:

1. Near-duplicate burst. Different authors post identical or near-identical
   text inside a short time window. A span of 15 minutes or less adds
   "posting within minutes of each other".
2. Shared-link burst. Different authors post the same outbound URL inside the
   window, even when the surrounding words differ.
3. Reused profile photos, accounts created within 48 hours, and accounts that
   reply to, repost, or quote one another. These fire only when the collected
   metadata contains the signal.

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
                reasons=_reasons_for(
                    [posts[index] for index in members],
                    ["posting the same or near-identical text"],
                ),
            )
        )

    alerts.extend(_shared_link_alerts(posts, times, window_seconds, used_ids))
    alerts.extend(_profile_photo_alerts(posts, alerts))
    alerts.extend(_account_cohort_alerts(posts, alerts))
    alerts.extend(_amplification_alerts(posts, alerts))
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
                reasons=_reasons_for(
                    [posts[index] for index in chosen],
                    ["posting the same outbound link"],
                ),
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
    reasons: list[str],
) -> dict:
    ordered = sorted(posts, key=lambda post: post.created_at)
    return {
        "alert_id": "",
        "indicator": indicator,
        "severity": severity,
        "score": round(score, 3),
        "summary": summary,
        "reasons": reasons,
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


def _reasons_for(posts: list[Post], base: list[str]) -> list[str]:
    reasons = list(base)
    if len(posts) >= 2 and _within_minutes(posts):
        reasons.append("posting within minutes of each other")
    if _created_together(posts):
        reasons.append("accounts created around the same time")
    if _shared_profile(posts):
        reasons.append("reused profile photos")
    if _amplifies_inside(posts):
        reasons.append("the same accounts amplifying each other")
    return reasons


def _within_minutes(posts: list[Post]) -> bool:
    ordered = sorted(posts, key=lambda post: _as_utc(post.created_at))
    for left_index, left in enumerate(ordered):
        for right in ordered[left_index + 1 :]:
            delta = (_as_utc(right.created_at) - _as_utc(left.created_at)).total_seconds()
            if delta > 15 * 60:
                break
            if left.author.strip().lower() != right.author.strip().lower():
                return True
    return False


def _created_together(posts: list[Post]) -> bool:
    dated = [
        post
        for post in posts
        if post.account_created is not None and not _is_anonymous(post.author)
    ]
    for left_index, left in enumerate(dated):
        for right in dated[left_index + 1 :]:
            if left.author.strip().lower() == right.author.strip().lower():
                continue
            gap = abs(
                (_as_utc(left.account_created) - _as_utc(right.account_created)).total_seconds()
            )
            if gap <= 48 * 3600:
                return True
    return False


def _profile_key(url: str) -> str:
    text = url.split("?")[0].strip().lower()
    text = re.sub(r"_(normal|bigger|mini|200x200|400x400)(?=\.)", "", text)
    if not text.startswith("http"):
        return ""
    if text.endswith("default_profile_images/default_profile.png"):
        return ""
    return text


def _shared_profile(posts: list[Post]) -> bool:
    owners: dict[str, set[str]] = {}
    for post in posts:
        key = _profile_key(post.profile_image)
        if not key or _is_anonymous(post.author):
            continue
        owners.setdefault(key, set()).add(post.author.strip().lower())
    return any(len(names) >= 2 for names in owners.values())


def _amplifies_inside(posts: list[Post]) -> bool:
    present = {post.author.strip().lower() for post in posts if not _is_anonymous(post.author)}
    for post in posts:
        for name in post.amplified_authors:
            if name.strip().lower() in present and name.strip().lower() != post.author.strip().lower():
                return True
    return False


def _covered(authors: set[str], alerts: list[dict]) -> bool:
    for alert in alerts:
        existing = {author.strip().lower() for author in alert.get("authors", [])}
        if authors <= existing:
            return True
    return False


def _profile_photo_alerts(posts: list[Post], existing: list[dict]) -> list[dict]:
    groups: dict[str, list[Post]] = {}
    for post in posts:
        key = _profile_key(post.profile_image)
        if key and not _is_anonymous(post.author):
            groups.setdefault(key, []).append(post)
    alerts: list[dict] = []
    for members in groups.values():
        authors = {post.author.strip().lower() for post in members}
        if len(authors) < 2 or _covered(authors, existing):
            continue
        chosen = []
        seen: set[str] = set()
        for post in sorted(members, key=lambda item: item.created_at):
            key = post.author.strip().lower()
            if key in seen:
                continue
            seen.add(key)
            chosen.append(post)
        alerts.append(
            _alert(
                indicator="reused_profile_photo",
                severity="medium",
                score=0.62,
                posts=chosen,
                summary=(
                    f"{len(chosen)} accounts used the same profile photo in this sample."
                ),
                extra={"profile_image": chosen[0].profile_image},
                reasons=["reused profile photos"],
            )
        )
    return alerts


def _account_cohort_alerts(posts: list[Post], existing: list[dict]) -> list[dict]:
    dated = [
        post
        for post in posts
        if post.account_created is not None and not _is_anonymous(post.author)
    ]
    dated.sort(key=lambda post: _as_utc(post.account_created))
    alerts: list[dict] = []
    index = 0
    while index < len(dated):
        window = [dated[index]]
        names = {dated[index].author.strip().lower()}
        cursor = index + 1
        while cursor < len(dated):
            gap = (
                _as_utc(dated[cursor].account_created) - _as_utc(window[0].account_created)
            ).total_seconds()
            if gap > 48 * 3600:
                break
            name = dated[cursor].author.strip().lower()
            if name not in names:
                names.add(name)
                window.append(dated[cursor])
            cursor += 1
        if len(names) >= 2 and not _covered(names, existing):
            alerts.append(
                _alert(
                    indicator="account_creation_cohort",
                    severity="low",
                    score=0.45,
                    posts=window,
                    summary=(
                        f"{len(window)} accounts in this sample were created within 48 hours of each other."
                    ),
                    extra={
                        "account_created_start": utc_account(window[0]),
                        "account_created_end": utc_account(window[-1]),
                    },
                    reasons=["accounts created around the same time"],
                )
            )
            index = cursor
        else:
            index += 1
    return alerts


def utc_account(post: Post) -> str:
    if post.account_created is None:
        return ""
    return _as_utc(post.account_created).isoformat()


def _amplification_alerts(posts: list[Post], existing: list[dict]) -> list[dict]:
    by_author: dict[str, list[Post]] = {}
    for post in posts:
        if _is_anonymous(post.author):
            continue
        by_author.setdefault(post.author.strip().lower(), []).append(post)
    parent = {author: author for author in by_author}

    def find(author: str) -> str:
        while parent[author] != author:
            parent[author] = parent[parent[author]]
            author = parent[author]
        return author

    for author, authored in by_author.items():
        for post in authored:
            for name in post.amplified_authors:
                other = name.strip().lower()
                if other in by_author and other != author:
                    parent[find(other)] = find(author)
    groups: dict[str, list[str]] = {}
    for author in by_author:
        groups.setdefault(find(author), []).append(author)
    alerts: list[dict] = []
    for members in groups.values():
        if len(members) < 2 or _covered(set(members), existing):
            continue
        chosen = [by_author[author][0] for author in members]
        alerts.append(
            _alert(
                indicator="mutual_amplification",
                severity="medium",
                score=0.58,
                posts=chosen,
                summary=(
                    f"{len(chosen)} accounts in this sample reply to, repost, or quote one another."
                ),
                extra={},
                reasons=["the same accounts amplifying each other"],
            )
        )
    return alerts


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
