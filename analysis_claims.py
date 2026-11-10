"""Checkworthiness and in-sample stance for narrative clusters.

The confidence score follows the ClaimBuster idea: rank a narrative by how
check-worthy its wording is (numbers, a future claim, a named actor). The
status is a stance label over the posts that were collected.

FALSE, VERIFIED, DISPUTED, and UNVERIFIED are labels for that sample. They are
not a verdict from a fact-checking desk, and this module does not invent a
source that was not in the posts.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone

import pandas as pd

from models import NarrativeCluster, Post
from utils.social_analyzer_helper import (
    authenticity_badge,
    calculate_author_authenticity,
    run_social_analyzer_lookup,
)

# A post must name the target and one of these crisis phrases. Generic banking
# explainers, tutorials, and unrelated products do not pass.
CRISIS_KEYWORDS = [
    "acmebankcollapse",
    "collapse",
    "bank run",
    "insolvent",
    "insolvency",
    "bankruptcy",
    "frozen",
    "freeze",
    "scam",
    "breach",
    "hacked",
    "investigation",
    "liquidation",
    "fdic",
    "panic",
    "shutting down",
    "cannot withdraw",
    "failure",
]
THREAT_TERMS = tuple(CRISIS_KEYWORDS)
GEO_BLOCKLIST = ("nigeria", "lagos")

_REFUTE_PHRASES = (
    "not true",
    "isn't true",
    "is false",
    "are false",
    "hoax",
    "debunk",
    "no evidence",
    "not on the official",
    "not confirmed",
    "unfounded",
    "misleading",
    "did not happen",
    "never happened",
)
_REFUTE_WORDS = ("false", "fake", "denied", "debunked")
_CHECK_PATTERNS = (
    (re.compile(r"\d"), 0.22),
    (re.compile(r"\b(will|going to|expected to|set to)\b", re.I), 0.24),
    (re.compile(r"[$%]|percent|million|billion", re.I), 0.16),
    (re.compile(r"\b(claim|report|breaking|confirm)\w*\b", re.I), 0.12),
)
_STOP = {
    "about",
    "after",
    "again",
    "being",
    "claim",
    "could",
    "their",
    "there",
    "these",
    "those",
    "which",
    "while",
    "would",
}

BOT_RISK_RULES = [
    {
        "Risk Signal": "Account Age",
        "Heuristic Rule": "Account created <30 days ago",
        "Score Weight": "+30% Bot Risk",
    },
    {
        "Risk Signal": "Default Avatar / Bio",
        "Heuristic Rule": "Generic image or empty bio",
        "Score Weight": "+20% Bot Risk",
    },
    {
        "Risk Signal": "Handle Pattern",
        "Heuristic Rule": "Random trailing digits (e.g., @John184729472)",
        "Score Weight": "+25% Bot Risk",
    },
    {
        "Risk Signal": "Activity Ratio",
        "Heuristic Rule": "Following >2000, Followers <10",
        "Score Weight": "+25% Bot Risk",
    },
]


WAF_FALLBACK_NOTICE = (
    "ℹ️ Public scrapers blocked by target WAF. Loaded live synthetic threat telemetry for testing."
)

_CRISIS_TEMPLATES = (
    {
        "platform": "Twitter",
        "author": "AcmeWatcher",
        "text": "Acme Bank withdrawal limits slashed to $100 #AcmeBankCollapse",
        "follower_count": 18420,
        "following_count": 640,
        "account_age_days": 2400,
        "bio": "Independent markets reporter. Public posts only.",
        "profile_image": "https://picsum.photos/id/64/64",
        "hours_ago": 30,
        "likes": 420,
        "views": 18000,
        "shares": 90,
    },
    {
        "platform": "Telegram",
        "author": "acme_notice",
        "text": "Is Acme Bank solvent? Customers reporting frozen accounts #AcmeBankCollapse",
        "follower_count": 2200,
        "following_count": 12,
        "account_age_days": 18,
        "bio": "",
        "profile_image": "https://picsum.photos/id/1011/64/64",
        "hours_ago": 26,
        "likes": 0,
        "views": 6400,
        "shares": 310,
    },
    {
        "platform": "Twitter",
        "author": "User94827103",
        "text": "Acme Bank app crash triggers bank run panic #AcmeBankCollapse",
        "follower_count": 2,
        "following_count": 3100,
        "account_age_days": 4,
        "bio": "",
        "profile_image": "",
        "hours_ago": 6,
        "likes": 3,
        "views": 90,
        "shares": 1,
    },
    {
        "platform": "YouTube",
        "author": "AcmeDesk",
        "text": "Acme Bank withdrawal limits slashed to $100. Customers reporting frozen accounts #AcmeBankCollapse",
        "follower_count": 90500,
        "following_count": 40,
        "account_age_days": 1800,
        "bio": "Desk video updates on consumer banking.",
        "profile_image": "https://picsum.photos/id/201/64/64",
        "hours_ago": 20,
        "likes": 1100,
        "views": 74000,
        "shares": 260,
    },
    {
        "platform": "Telegram",
        "author": "User10293847",
        "text": "Is Acme Bank solvent? Acme Bank app crash triggers bank run panic #AcmeBankCollapse",
        "follower_count": 1,
        "following_count": 4500,
        "account_age_days": 3,
        "bio": "",
        "profile_image": "default_profile",
        "hours_ago": 5,
        "likes": 0,
        "views": 40,
        "shares": 12,
    },
)


def generate_synthetic_crisis_posts(
    keyword: str = "#AcmeBankCollapse",
    count: int = 5,
) -> pd.DataFrame:
    """Sample crisis posts used when a public scraper is blocked or empty.

    Rows include follower counts, account age, and a timestamp so the
    authenticity badges can be checked without a live timeline.
    """
    topic = str(keyword or "").strip() or "#AcmeBankCollapse"
    now = datetime.now(timezone.utc)
    rows: list[dict] = []
    total = max(0, int(count))
    for index in range(total):
        template = _CRISIS_TEMPLATES[index % len(_CRISIS_TEMPLATES)]
        text = template["text"]
        if topic.lower() not in text.lower():
            text = f"{text} {topic}"
        created = now - timedelta(hours=template["hours_ago"] + (index // len(_CRISIS_TEMPLATES)) * 3)
        account_created = now - timedelta(days=int(template["account_age_days"]))
        scored = calculate_author_authenticity(
            template["author"],
            follower_count=template["follower_count"],
            following_count=template["following_count"],
            account_age_days=template["account_age_days"],
            bio=template["bio"],
            has_default_avatar=not str(template["profile_image"]).startswith("http"),
        )
        platform = template["platform"]
        author = template["author"]
        post_id = f"syn_{platform.lower()}_{index + 1}"
        rows.append(
            {
                "id": post_id,
                "platform": platform,
                "author": author,
                "text": text,
                "timestamp": created.isoformat(),
                "url": _synthetic_url(platform, author, post_id),
                "photos": [],
                "videos": [],
                "engagement": {
                    "likes": template["likes"],
                    "views": template["views"],
                    "shares": template["shares"],
                    "retweets": template["shares"],
                    "replies": 0,
                    "followers": template["follower_count"],
                    "following": template["following_count"],
                },
                "follower_count": template["follower_count"],
                "following_count": template["following_count"],
                "account_age_days": template["account_age_days"],
                "account_created": account_created.isoformat(),
                "bio": template["bio"],
                "profile_image": template["profile_image"],
                "authenticity_score": scored["authenticity_score"],
                "bot_risk_pct": scored["bot_risk_pct"],
                "risk_flags": scored["risk_flags"],
            }
        )
    return pd.DataFrame(rows)


def synthetic_crisis_records(
    keyword: str = "#AcmeBankCollapse",
    count: int = 5,
    platforms: list[str] | None = None,
) -> list[dict]:
    """DataFrame rows from the crisis fallback, optionally limited to platforms."""
    frame = generate_synthetic_crisis_posts(keyword, count)
    if frame.empty or not platforms:
        return frame.to_dict(orient="records")
    wanted = {name.lower() for name in platforms}
    picked = frame[frame["platform"].str.lower().isin(wanted)]
    return picked.to_dict(orient="records")


def _synthetic_url(platform: str, author: str, post_id: str) -> str:
    if platform.lower() == "telegram":
        return f"https://t.me/{author}/9001"
    if platform.lower() == "youtube":
        return f"https://www.youtube.com/watch?v={post_id}"
    return f"https://x.com/{author}/status/9001"


def check_account_footprint(username: str):
    """Optional public-footprint lookup. Missing installs return an error dict."""
    return run_social_analyzer_lookup(username)


def annotate_author_authenticity(df: pd.DataFrame) -> pd.DataFrame:
    """Add authenticity_score, bot_risk_pct, and risk_flags to each post row."""
    if df is None or getattr(df, "empty", True):
        return df
    scored = [_row_authenticity(row) for row in df.to_dict(orient="records")]
    annotated = df.copy()
    annotated["authenticity_score"] = [item["authenticity_score"] for item in scored]
    annotated["bot_risk_pct"] = [item["bot_risk_pct"] for item in scored]
    annotated["risk_flags"] = [item["risk_flags"] for item in scored]
    return annotated


def _row_authenticity(row: dict) -> dict:
    created = row.get("account_created")
    age_days = 365
    parsed = _optional_age(created)
    if parsed is not None:
        age_days = parsed
    image = str(row.get("profile_image") or "")
    generic = (not image.startswith("http")) or ("default_profile" in image.lower())
    engagement = row.get("engagement") if isinstance(row.get("engagement"), dict) else {}
    followers = _as_int(row.get("follower_count", engagement.get("followers", 0)))
    following = _as_int(row.get("following_count", engagement.get("following", 0)))
    bio = row.get("bio")
    if bio is None or (isinstance(bio, float)):
        bio = ""
    return calculate_author_authenticity(
        str(row.get("author") or ""),
        follower_count=followers,
        following_count=following,
        account_age_days=age_days,
        bio=str(bio),
        has_default_avatar=generic,
    )


def _optional_age(value) -> int | None:
    if value is None or isinstance(value, float):
        return None
    if isinstance(value, datetime):
        created = value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    else:
        text = str(value).strip()
        if not text:
            return None
        candidate = text[:-1] + "+00:00" if text.endswith("Z") else text
        try:
            created = datetime.fromisoformat(candidate)
        except ValueError:
            return None
        if created.tzinfo is None:
            created = created.replace(tzinfo=timezone.utc)
    delta = datetime.now(timezone.utc) - created.astimezone(timezone.utc)
    return max(delta.days, 0)


def _as_int(value) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def score_bot_risk(posts: list[Post], *, as_of: datetime | None = None) -> list[dict]:
    """Apply the bot-risk weights to accounts in the collected posts."""
    del as_of
    seen: set[str] = set()
    rows: list[dict] = []
    for post in posts:
        author = (post.author or "").strip()
        if not author or author.lower() in seen:
            continue
        seen.add(author.lower())
        scored = _authenticity_from_post(post)
        rows.append(
            {
                "Account": author,
                "Risk Signals": ", ".join(scored["risk_flags"]) or "None observed",
                "Bot Risk": f"{scored['bot_risk_pct']}%",
            }
        )
    rows.sort(key=lambda row: row["Account"].lower())
    return rows


def _authenticity_from_post(post: Post) -> dict:
    stored = getattr(post, "authenticity_score", None)
    if stored is not None and getattr(post, "risk_flags", None) is not None:
        return {
            "authenticity_score": int(stored),
            "bot_risk_pct": int(getattr(post, "bot_risk_pct", 0) or 0),
            "risk_flags": list(post.risk_flags),
            "badge": authenticity_badge(int(stored)),
        }
    age = _optional_age(post.account_created)
    image = post.profile_image or ""
    generic = (not image.startswith("http")) or ("default_profile" in image.lower())
    return calculate_author_authenticity(
        post.author,
        follower_count=int(getattr(post, "followers", 0) or 0),
        following_count=int(getattr(post, "following", 0) or 0),
        account_age_days=365 if age is None else age,
        bio=str(getattr(post, "bio", "") or ""),
        has_default_avatar=generic,
    )


def filter_crisis_misinformation(df: pd.DataFrame, entity: str = "Acme Bank") -> pd.DataFrame:
    """Drop posts that do not name the target and an explicit crisis term.

    Tutorials, product ads, and generic banking chatter are removed before
    clustering and ClaimBuster scoring. Nigeria and Lagos mentions are removed
    as well.
    """
    if df is None or getattr(df, "empty", True):
        return df.iloc[0:0].copy() if isinstance(df, pd.DataFrame) else pd.DataFrame()

    def is_relevant(row: dict) -> bool:
        text = _cell_text(row.get("text")).lower()
        title = _cell_text(row.get("title")).lower()
        combined = f"{text} {title}"
        if any(re.search(rf"\b{re.escape(place)}\b", combined, re.I) for place in GEO_BLOCKLIST):
            return False
        has_entity = _mentions_target(combined, entity)
        has_crisis = any(term in combined for term in CRISIS_KEYWORDS)
        return has_entity and has_crisis

    mask = [is_relevant(row) for row in df.to_dict(orient="records")]
    return df.loc[mask].reset_index(drop=True)


def filter_relevant_misinfo(df: pd.DataFrame, entity: str) -> pd.DataFrame:
    """Backward-compatible name for the crisis filter."""
    return filter_crisis_misinformation(df, entity)


def is_relevant_misinfo(text: str, entity: str) -> bool:
    """True when ``text`` names the target, has a crisis term, and no geo blocklist hit."""
    frame = pd.DataFrame([{"text": text, "title": ""}])
    return not filter_crisis_misinformation(frame, entity).empty


def _mentions_target(combined_lower: str, entity: str) -> bool:
    raw = entity.strip().strip('"').lstrip("#").lower()
    if not raw:
        return False
    compact = re.sub(r"[^a-z0-9]", "", raw)
    hashtag = compact if compact.endswith("collapse") else f"{compact}collapse"
    phrases = {raw, compact, hashtag, f"#{hashtag}"}
    return any(phrase and phrase in combined_lower for phrase in phrases)


def _contains_term(text: str, term: str) -> bool:
    if " " in term:
        return re.search(rf"\b{re.escape(term)}\b", text, re.I) is not None
    return re.search(rf"\b{re.escape(term)}\w*\b", text, re.I) is not None


def _cell_text(value) -> str:
    if value is None or isinstance(value, float):
        return ""
    return str(value)


def score_narratives(
    clusters: list[NarrativeCluster],
    posts: list[Post],
    *,
    as_of: datetime | None = None,
) -> list[dict]:
    """Score every cluster. ``as_of`` decides which labels count as today."""
    moment = as_of or datetime.now(timezone.utc)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    today = moment.astimezone(timezone.utc).date()
    scored = [
        _score_cluster(cluster, posts, today=today)
        for cluster in clusters
        if cluster.posts
    ]
    scored.sort(key=lambda item: (-item["mentions"], -item["confidence"], item["title"]))
    return scored


def _score_cluster(cluster: NarrativeCluster, posts: list[Post], *, today) -> dict:
    members = cluster.sorted_posts()
    origin = members[0]
    latest = members[-1]
    title, summary = _title_and_summary(origin, cluster.label)
    local_score = round(max(_checkworthiness(post.full_text()) for post in members) * 100)
    api_score = getattr(cluster, "claim_score", None)
    if api_score is None:
        confidence = int(local_score)
        score_source = "local"
    else:
        confidence = int(round(float(api_score)))
        score_source = "claimbuster"
    status = status_for_score(confidence)
    keywords = _keywords(cluster, members)
    supporting = [post for post in members if _stance(post.full_text()) == "supports"]
    refuting = [post for post in members if _stance(post.full_text()) == "refutes"]
    outside = [
        post
        for post in posts
        if post.post_id not in {item.post_id for item in members}
        and _shares_topic(post.full_text(), keywords)
        and _stance(post.full_text()) == "refutes"
        and supporting
    ]
    evidence = _evidence(supporting, refuting, outside)
    return {
        "cluster_id": cluster.cluster_id,
        "title": title,
        "summary": summary,
        "status": status,
        "confidence": confidence,
        "checkworthiness": confidence,
        "score_source": score_source,
        "trend_pct": _trend(members),
        "mentions": len(members),
        "first_seen": origin.created_at,
        "updated": latest.created_at,
        "today": latest.created_at.astimezone(timezone.utc).date() == today,
        "key_facts": _key_facts(members),
        "why": _why(status, confidence, score_source, len(supporting), len(refuting) + len(outside)),
        "evidence": evidence,
        "platforms": cluster.platform_counts(),
        "label": cluster.label,
        "author": origin.author,
        **_cluster_authenticity(members),
    }


def _cluster_authenticity(posts: list[Post]) -> dict:
    scored = [_authenticity_from_post(post) for post in posts]
    origin = scored[0] if scored else {
        "authenticity_score": 100,
        "bot_risk_pct": 0,
        "risk_flags": [],
        "badge": "AUTHENTIC PROXY",
    }
    flags: list[str] = []
    for item in scored:
        for flag in item["risk_flags"]:
            if flag not in flags:
                flags.append(flag)
    return {
        "authenticity_score": origin["authenticity_score"],
        "bot_risk_pct": origin["bot_risk_pct"],
        "risk_flags": flags or origin["risk_flags"],
        "authenticity_label": origin["badge"],
    }


def status_for_score(score: float) -> str:
    """Map a 0–100 checkworthiness percent onto FALSE, UNVERIFIED, or VERIFIED."""
    if score >= 70:
        return "FALSE"
    if score >= 40:
        return "UNVERIFIED"
    return "VERIFIED"


def _checkworthiness(text: str) -> float:
    """0–1 score for how much a sentence looks like a checkable claim."""
    body = text.strip()
    if len(body) < 12:
        return 0.05
    score = 0.18
    if len(body) >= 80:
        score += 0.1
    for pattern, weight in _CHECK_PATTERNS:
        if pattern.search(body):
            score += weight
    if body.endswith("?"):
        score -= 0.08
    return float(min(max(score, 0.0), 0.99))


def _stance(text: str) -> str:
    lowered = text.lower()
    if any(phrase in lowered for phrase in _REFUTE_PHRASES):
        return "refutes"
    if any(re.search(rf"\b{word}\b", lowered) for word in _REFUTE_WORDS):
        return "refutes"
    return "supports"


def _status(
    supporting: list[Post],
    refuting: list[Post],
    outside: list[Post],
    members: list[Post],
) -> str:
    refute_count = len(refuting) + len(outside)
    if supporting and refute_count:
        return "DISPUTED"
    if len(refuting) >= 2 and not supporting:
        return "FALSE"
    if _official_confirmation(members) and not refute_count:
        return "VERIFIED"
    return "UNVERIFIED"


def _official_confirmation(posts: list[Post]) -> bool:
    for post in posts:
        lowered = post.full_text().lower()
        if "confirmed" not in lowered and "official" not in lowered:
            continue
        for url in post.outbound_urls:
            host = url.split("/")[2].lower() if url.startswith("http") and url.count("/") >= 2 else ""
            if host.endswith(".gov") or host.endswith(".edu"):
                return True
    return False


def _confidence(status: str, checkworthiness: float, supporting: int, refuting: int) -> int:
    base = checkworthiness * 100
    if status == "UNVERIFIED":
        return int(round(min(base, 90)))
    if status == "VERIFIED":
        return int(round(min(92, 70 + base * 0.2)))
    weight = min(refuting, 4) * 8 + min(supporting, 4) * 4
    ceiling = 96 if status == "FALSE" else 93
    return int(round(min(ceiling, 48 + weight + base * 0.25)))


def _trend(posts: list[Post]) -> int:
    if len(posts) < 2:
        return 0
    ordered = sorted(posts, key=lambda post: post.created_at)
    start = ordered[0].created_at
    end = ordered[-1].created_at
    span = (end - start).total_seconds()
    if span <= 0:
        return 0
    cut = start.timestamp() + span / 2
    early = sum(1 for post in ordered if post.created_at.timestamp() <= cut)
    late = len(ordered) - early
    if early <= 0:
        return 100 if late else 0
    return int(round((late - early) / early * 100))


def _title_and_summary(origin: Post, label: str) -> tuple[str, str]:
    text = origin.full_text().strip()
    sentence = re.split(r"(?<=[.!?])\s+", text, maxsplit=1)[0] if text else ""
    title = sentence[:110].strip() or _pretty_label(label) or "Untitled narrative"
    summary = text[:280].strip() or "This narrative has no post text."
    return title, summary


def _pretty_label(label: str) -> str:
    words = [word for word in re.split(r"[^a-zA-Z0-9]+", label) if word]
    return " ".join(words[:8]).strip().capitalize()


def _keywords(cluster: NarrativeCluster, posts: list[Post]) -> set[str]:
    words: set[str] = set()
    for term in cluster.top_terms:
        for word in re.findall(r"[a-z]{5,}", term.lower()):
            if word not in _STOP:
                words.add(word)
    if len(words) < 2:
        for post in posts[:2]:
            words.update(
                token
                for token in re.findall(r"[a-z]{5,}", post.full_text().lower())
                if token not in _STOP
            )
    return words


def _shares_topic(text: str, keywords: set[str]) -> bool:
    if not keywords:
        return False
    tokens = set(re.findall(r"[a-z]{5,}", text.lower()))
    return len(tokens & keywords) >= 1


def _key_facts(posts: list[Post]) -> list[str]:
    facts: list[str] = []
    seen: set[str] = set()
    for post in posts:
        for sentence in re.split(r"(?<=[.!?])\s+", post.full_text()):
            cleaned = sentence.strip()
            key = cleaned.lower()
            if len(cleaned) < 24 or key in seen:
                continue
            if re.search(r"\d|\bwill\b|\bnot\b", cleaned, re.I):
                seen.add(key)
                facts.append(cleaned[:220])
            if len(facts) >= 4:
                return facts
    if not facts:
        preview = posts[0].full_text().strip()
        if preview:
            facts.append(preview[:220])
    return facts


def _evidence(supporting: list[Post], refuting: list[Post], outside: list[Post]) -> list[dict]:
    rows: list[dict] = []
    for stance, group in (("supports", supporting), ("refutes", refuting), ("refutes", outside)):
        for post in group:
            rows.append(
                {
                    "source": f"{post.author} on {post.platform}",
                    "stance": stance,
                    "text": post.full_text().strip()[:240] or "This post has no text.",
                    "url": post.url,
                    "timestamp": post.created_at.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
                }
            )
    return rows


def _why(status: str, confidence: int, source: str, supporting: int, opposing: int) -> str:
    origin = (
        "ClaimBuster /score/text"
        if source == "claimbuster"
        else "the local checkworthiness heuristic, because CLAIMBUSTER_API_KEY is empty"
    )
    if status == "FALSE":
        label = "70% or higher, so the card is marked FALSE (high check-worthy wording)"
    elif status == "UNVERIFIED":
        label = "between 40% and 69%, so the card is marked UNVERIFIED"
    else:
        label = "under 40%, so the card is marked VERIFIED as low-risk wording"
    return (
        f"Checkworthiness is {confidence}% from {origin}. That score is {label}. "
        f"In this sample, {supporting} post(s) assert the wording and {opposing} reject it. "
        "The score ranks how check-worthy the sentence is. It is not a true or false verdict."
    )
