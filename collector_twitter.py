"""Public X search through Nitter HTML. No X login and no third-party API.

``collect`` requests a public Nitter search page and reads the timeline with
BeautifulSoup. Set NITTER_INSTANCES in .env to a comma-separated list of
bases such as https://nitter.poast.org. Each tweet is mapped into the shared
post schema.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

from config import DEFAULT_USER_AGENT, env
from scrapers import CollectorError

NITTER_INSTANCES = (
    "https://nitter.poast.org",
    "https://nitter.privacydev.net",
    "https://nitter.space",
    "https://nitter.tiekoetter.com",
)

# Set by collect() when every instance refuses, times out, or returns a bot wall.
# Callers show last_warning and, when last_waf is set, the synthetic-telemetry toast.
last_warning = ""
last_offline = False
last_waf = False

_WAF_MARKERS = (
    "anubis",
    "dogwaf",
    "bot challenge",
    "verify you are human",
    "checking your browser",
    "just a moment",
)

_CONNECTION_DROPS = (
    ConnectionRefusedError,
    TimeoutError,
    requests.Timeout,
    requests.ConnectionError,
)

APIFY_ACTOR = "apidojo/tweet-scraper"
APIFY_RUN_URL = (
    "https://api.apify.com/v2/acts/apidojo~tweet-scraper/run-sync-get-dataset-items"
)
APIFY_TIMEOUT_SECONDS = 240


class MissingApiKeyError(CollectorError):
    """Raised when Twitter collection is selected but the API is not configured."""


def collect(
    query: str,
    start: datetime | None = None,
    end: datetime | None = None,
    *,
    limit: int | None = None,
    max_posts: int | None = None,
    start_date: datetime | None = None,
    end_date: datetime | None = None,
    api_key: str | None = None,
    api_url: str | None = None,
    threat_focus: bool = False,
) -> list[dict]:
    """Search public Nitter timelines for ``query`` inside the date window.

    Each base in the failover list is tried in order. A refused connection,
    timeout, bot challenge, short page, or page with no posts skips that host.
    When nothing usable comes back, this returns no rows and sets ``last_waf``.
    The dashboard decides whether every selected platform failed.
    """
    global last_warning, last_offline, last_waf
    last_warning = ""
    last_offline = False
    last_waf = False
    del api_key, api_url, threat_focus
    if not query.strip():
        raise CollectorError("Enter a keyword before searching Twitter.")
    window_start = start if start is not None else start_date
    window_end = end if end is not None else end_date
    if window_start is None or window_end is None:
        raise CollectorError("A start date and end date are required.")
    requested = max_posts if max_posts is not None else limit
    if requested is None:
        raise CollectorError("A max post count is required.")
    max_posts = max(1, int(requested))
    term = _search_term(query, window_start, window_end)
    errors: list[str] = []
    try:
        for base in _nitter_bases():
            try:
                html = _fetch_nitter(base, term)
            except CollectorError as exc:
                errors.append(str(exc))
                continue
            except (ConnectionRefusedError, TimeoutError, requests.Timeout, requests.ConnectionError) as exc:
                errors.append(f"{base} refused the connection or timed out: {exc}")
                continue
            rows = parse_nitter(html, base, start=window_start, end=window_end, limit=max_posts)
            if rows:
                return rows
            errors.append(f"{base} returned a bot challenge or no public posts.")
    except Exception as exc:
        errors.append(f"Public X search failed: {exc}")
    detail = errors[-1] if errors else "No Nitter instance is configured."
    last_offline = True
    last_waf = True
    last_warning = f"Public X search was blocked or returned no posts. {detail}"
    return []


def parse_nitter(html: str, base: str, *, start: datetime, end: datetime, limit: int) -> list[dict]:
    """Read Nitter timeline items into the shared post schema."""
    soup = BeautifulSoup(html, "html.parser")
    rows: list[dict] = []
    seen: set[str] = set()
    for item in soup.select("div.timeline-item"):
        if "unavailable" in " ".join(item.get("class") or []):
            continue
        link = item.select_one("a.tweet-link")
        href = str(link.get("href") or "") if link else ""
        match = re.search(r"/([^/]+)/status/(\d+)", href)
        if not match:
            continue
        author, post_id = match.group(1), match.group(2)
        if post_id in seen:
            continue
        content = item.select_one(".tweet-content")
        text = content.get_text(" ", strip=True) if content else ""
        when = _nitter_time(item)
        if when is not None and (when < start or when > end):
            continue
        photos, videos = _nitter_media(item, base)
        likes, shares, replies, views = _nitter_stats(item)
        rows.append(
            {
                "id": post_id,
                "platform": "twitter",
                "author": author,
                "text": text,
                "timestamp": _iso(when) if when is not None else "",
                "url": f"https://x.com/{author}/status/{post_id}",
                "photos": photos,
                "videos": videos,
                "engagement": {
                    "likes": likes,
                    "views": views,
                    "shares": shares,
                    "retweets": shares,
                    "replies": replies,
                },
            }
        )
        seen.add(post_id)
        if len(rows) >= limit:
            break
    return rows


def _nitter_bases() -> list[str]:
    """Configured hosts first, then the built-in failover list."""
    ordered: list[str] = []
    configured = env("NITTER_INSTANCES")
    if configured:
        ordered.extend(part.strip().rstrip("/") for part in configured.split(",") if part.strip())
    for base in NITTER_INSTANCES:
        if base not in ordered:
            ordered.append(base)
    return ordered


def _fetch_nitter(base: str, term: str) -> str:
    try:
        response = requests.get(
            f"{base}/search",
            params={"f": "tweets", "q": term},
            headers={"User-Agent": DEFAULT_USER_AGENT, "Accept": "text/html"},
            timeout=12,
        )
    except _CONNECTION_DROPS as exc:
        raise CollectorError(f"{base} refused the connection or timed out: {exc}") from exc
    except requests.RequestException as exc:
        raise CollectorError(f"{base} could not be reached: {exc}") from exc
    text = response.text or ""
    if response.status_code >= 400 or _waf_blocked(text):
        raise CollectorError(f"{base} returned a bot challenge or blocked the request.")
    if "timeline-item" not in text and "tweet-content" not in text:
        raise CollectorError(f"{base} did not return a public timeline.")
    return text


def _waf_blocked(html: str) -> bool:
    """True for Anubis/DogWAF interstitials and other pages too small to be a timeline."""
    if len(html or "") < 500:
        return True
    lowered = html.lower()
    return any(marker in lowered for marker in _WAF_MARKERS)


def _nitter_time(item) -> datetime | None:
    stamp = item.select_one(".tweet-date a")
    raw = ""
    if stamp is not None:
        raw = str(stamp.get("title") or stamp.get_text(" ", strip=True))
    return _parse_time(raw)


def _nitter_media(item, base: str) -> tuple[list[str], list[str]]:
    photos: list[str] = []
    videos: list[str] = []
    for image in item.select(".attachment.image img, .attachments img"):
        src = str(image.get("src") or "")
        if src:
            photos.append(urljoin(base, src))
    for video in item.select("video source, video"):
        src = str(video.get("src") or "")
        if src:
            videos.append(urljoin(base, src))
    return _unique(photos), _unique(videos)


def _nitter_stats(item) -> tuple[int, int, int, int]:
    likes = shares = replies = views = 0
    for stat in item.select(".tweet-stat"):
        icon = stat.select_one(".icon-container span, span")
        classes = " ".join(icon.get("class") or []) if icon else ""
        count = _as_int(stat.get_text(" ", strip=True))
        if "heart" in classes or "like" in classes:
            likes = count
        elif "retweet" in classes:
            shares = count
        elif "comment" in classes or "reply" in classes:
            replies = count
        elif "views" in classes or "play" in classes:
            views = count
    return likes, shares, replies, views


def normalize_items(body, *, start: datetime, end: datetime, limit: int) -> list[dict]:
    """Map a provider payload into the shared post dicts."""
    items = _item_list(body)
    rows: list[dict] = []
    seen: set[str] = set()
    for item in items:
        if not isinstance(item, dict):
            continue
        row = _normalize_item(item)
        if row is None or row["id"] in seen:
            continue
        when = _parse_time(row["timestamp"])
        if when is not None and (when < start or when > end):
            continue
        if when is not None:
            row["timestamp"] = _iso(when)
        seen.add(row["id"])
        rows.append(row)
        if len(rows) >= limit:
            break
    return rows


def demo_records(query: str, start: datetime, end: datetime, *, limit: int) -> list[dict]:
    """Synthetic posts that show text, media, a shared narrative, and copy signals."""
    topic = query.strip() or "the topic"
    span = max((end - start).total_seconds(), 1.0)
    rumor = (
        f"Customers are posting that {topic} will freeze withdrawals tomorrow. "
        "Share the notice before the branch opens."
    )
    first = start + timedelta(seconds=span * 0.30)
    created = (start - timedelta(days=2)).strftime("%Y-%m-%dT%H:%M:%SZ")
    shared_photo = "https://picsum.photos/id/64/64/64"
    samples = [
        ("demo_wire", rumor, first, ["https://picsum.photos/id/237/320/180"], [], shared_photo, created, []),
        ("demo_echo", rumor, first + timedelta(minutes=4), [], [], shared_photo, created, ["demo_wire"]),
        (
            "demo_boost",
            f"RT @demo_wire: {rumor}",
            first + timedelta(hours=6),
            [],
            ["https://interactive-examples.mdn.mozilla.net/media/cc0-videos/flower.mp4"],
            "",
            "2020-01-01T00:00:00Z",
            ["demo_wire"],
        ),
        (
            "demo_desk",
            f"{topic} fee notice looks routine. The freeze claim is not on the official page.",
            first + timedelta(hours=30),
            [],
            [],
            "",
            "2018-05-01T00:00:00Z",
            [],
        ),
    ]
    rows: list[dict] = []
    for index, (author, text, when, photos, videos, profile, account_created, amplified) in enumerate(
        samples[: max(1, limit)]
    ):
        status_id = 1800000000000000000 + index
        rows.append(
            {
                "id": str(status_id),
                "platform": "twitter",
                "author": author,
                "text": text,
                "timestamp": _iso(when),
                "url": f"https://x.com/{author}/status/{status_id}",
                "photos": photos,
                "videos": videos,
                "engagement": {
                    "likes": 40 + index * 12,
                    "views": 1200 + index * 300,
                    "shares": 8 + index * 3,
                    "retweets": 8 + index * 3,
                    "replies": 2 + index,
                },
                "account_created": account_created,
                "profile_image": profile,
                "amplified_authors": amplified,
                "outbound_urls": (
                    ["https://example.com/acme-freeze-notice"] if index < 2 else []
                ),
            }
        )
    return rows


def _item_list(body) -> list:
    if isinstance(body, list):
        return body
    if isinstance(body, dict):
        for key in ("data", "items", "tweets", "results"):
            value = body.get(key)
            if isinstance(value, list):
                return value
    return []


def _normalize_item(item: dict) -> dict | None:
    raw_id = item.get("id_str") or item.get("id")
    if raw_id in (None, ""):
        return None
    post_id = str(raw_id)
    author = _author(item)
    text = item.get("full_text") or item.get("text") or ""
    url = item.get("url") or f"https://x.com/{author}/status/{post_id}"
    timestamp = item.get("created_at") or ""
    photos, videos = _media(item)
    return {
        "id": post_id,
        "platform": "twitter",
        "author": author,
        "text": str(text).strip(),
        "timestamp": str(timestamp),
        "url": str(url),
        "photos": photos,
        "videos": videos,
        "engagement": _engagement(item),
        "account_created": _account_created(item),
        "profile_image": _profile_image(item),
        "amplified_authors": _amplified_authors(item, str(text)),
        "outbound_urls": _outbound_urls(item),
    }


def _account_created(item: dict) -> str:
    user = item.get("user") if isinstance(item.get("user"), dict) else {}
    author = item.get("author") if isinstance(item.get("author"), dict) else {}
    return str(user.get("created_at") or author.get("createdAt") or author.get("created_at") or "")


def _profile_image(item: dict) -> str:
    user = item.get("user") if isinstance(item.get("user"), dict) else {}
    author = item.get("author") if isinstance(item.get("author"), dict) else {}
    value = (
        user.get("profile_image_url_https")
        or user.get("profile_image_url")
        or author.get("profilePicture")
        or ""
    )
    return str(value) if str(value).startswith("http") else ""


def _amplified_authors(item: dict, text: str) -> list[str]:
    names: list[str] = []
    reply = item.get("in_reply_to_screen_name") or item.get("inReplyToUsername")
    if reply:
        names.append(str(reply).lstrip("@"))
    for key in ("retweeted_status", "quoted_status", "retweetedTweet", "quotedTweet"):
        nested = item.get(key)
        if isinstance(nested, dict):
            names.append(_author(nested))
    match = re.match(r"RT @([A-Za-z0-9_]{1,15})", text.strip())
    if match:
        names.append(match.group(1))
    return _unique([name for name in names if name and name != "Unknown"])


def _outbound_urls(item: dict) -> list[str]:
    found: list[str] = []
    entities = item.get("entities") if isinstance(item.get("entities"), dict) else {}
    for entry in entities.get("urls") or []:
        if isinstance(entry, dict):
            value = entry.get("expanded_url") or entry.get("url") or ""
            if str(value).startswith("http"):
                found.append(str(value))
    return _unique(found)


def _author(item: dict) -> str:
    user = item.get("user") if isinstance(item.get("user"), dict) else {}
    author = item.get("author") if isinstance(item.get("author"), dict) else {}
    name = user.get("screen_name") or author.get("userName", "Unknown")
    return str(name).lstrip("@") or "Unknown"


_VIDEO_TYPES = {"video", "animated_gif", "gif"}
_VIDEO_FILE = re.compile(r"\.(mp4|webm|mov|m4v)(\?|$)", re.IGNORECASE)


def _engagement(item: dict) -> dict:
    metrics = item.get("public_metrics") if isinstance(item.get("public_metrics"), dict) else {}
    likes = _metric(item, metrics, "favorite_count", "like_count", "likeCount", "favoriteCount")
    shares = _metric(item, metrics, "retweet_count", "retweetCount")
    return {
        "likes": likes,
        "views": _metric(
            item,
            metrics,
            "views_count",
            "impression_count",
            "view_count",
            "viewCount",
            "views",
        ),
        "shares": shares,
        "retweets": shares,
        "replies": _metric(item, metrics, "reply_count", "replyCount"),
    }


def _metric(item: dict, metrics: dict, *keys: str) -> int:
    raw = _first(metrics, *keys)
    if raw in (None, ""):
        raw = _first(item, *keys)
    return _as_int(raw)


def _media(item: dict) -> tuple[list[str], list[str]]:
    """Photos come from entities or extended entities. Videos come from extended entities."""
    photos: list[str] = []
    videos: list[str] = []
    entities = item.get("entities") if isinstance(item.get("entities"), dict) else {}
    extended = item.get("extended_entities") or item.get("extendedEntities") or {}
    if not isinstance(extended, dict):
        extended = {}
    for blob in entities.get("media") or []:
        if isinstance(blob, dict) and str(blob.get("type") or "photo").lower() not in _VIDEO_TYPES:
            photos.extend(_image_urls(blob))
    for blob in extended.get("media") or []:
        if not isinstance(blob, dict):
            continue
        kind = str(blob.get("type") or "").lower()
        if kind in _VIDEO_TYPES:
            videos.extend(_video_urls(blob))
            photos.extend(_image_urls(blob))
        else:
            photos.extend(_image_urls(blob))
    return _unique(photos), _unique(videos)


def _image_urls(blob: dict) -> list[str]:
    found: list[str] = []
    for key in ("media_url_https", "media_url", "mediaUrl", "previewUrl"):
        value = blob.get(key)
        if isinstance(value, str) and value.startswith("http") and not _VIDEO_FILE.search(value):
            found.append(value)
    return found


def _video_urls(blob: dict) -> list[str]:
    found = [url for url in _urls_from_media_object(blob) if _VIDEO_FILE.search(url)]
    return found


def _urls_from_media_object(blob: dict) -> list[str]:
    urls: list[str] = []
    for key in ("media_url_https", "media_url", "mediaUrl", "previewUrl", "url"):
        value = blob.get(key)
        if isinstance(value, str) and value.startswith("http"):
            urls.append(value)
    info = blob.get("video_info") or blob.get("videoInfo") or {}
    variants = info.get("variants") if isinstance(info, dict) else None
    if isinstance(variants, list):
        ranked = [
            variant
            for variant in variants
            if isinstance(variant, dict) and isinstance(variant.get("url"), str)
        ]
        ranked.sort(key=lambda variant: _as_int(variant.get("bitrate")), reverse=True)
        urls.extend(str(variant["url"]) for variant in ranked)
    return urls


def _assign_media_url(url: str, photos: list[str], videos: list[str]) -> None:
    if not url.startswith("http"):
        return
    if _VIDEO_FILE.search(url):
        videos.append(url)
    else:
        photos.append(url)


def _unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        unique.append(value)
    return unique


def _views(item: dict) -> int:
    for source in (item, item.get("public_metrics") or {}, item.get("views") or {}):
        if not isinstance(source, dict):
            continue
        raw = _first(source, "views", "view_count", "viewCount", "impression_count")
        if isinstance(raw, dict):
            raw = _first(raw, "count", "view_count")
        parsed = _as_int(raw)
        if parsed:
            return parsed
    return 0


def _first(item: dict, *keys: str):
    for key in keys:
        value = item.get(key)
        if value not in (None, ""):
            return value
    return None


def _as_int(value) -> int:
    if value in (None, ""):
        return 0
    if isinstance(value, bool):
        return 0
    if isinstance(value, (int, float)):
        return int(value)
    text = str(value).strip().replace(",", "")
    multiplier = 1
    if text and text[-1].lower() == "k":
        multiplier = 1000
        text = text[:-1]
    elif text and text[-1].lower() == "m":
        multiplier = 1_000_000
        text = text[:-1]
    try:
        return int(round(float(text) * multiplier))
    except ValueError:
        return 0


def _parse_time(value) -> datetime | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        seconds = value / 1000 if value > 10_000_000_000 else value
        return datetime.fromtimestamp(seconds, tz=timezone.utc)
    text = str(value or "").strip()
    if not text:
        return None
    candidate = text[:-1] + "+00:00" if text.endswith("Z") else text
    for attempt in (candidate, text):
        try:
            parsed = datetime.fromisoformat(attempt)
            break
        except ValueError:
            parsed = None
    if parsed is None:
        for fmt in (
            "%a %b %d %H:%M:%S %z %Y",
            "%Y-%m-%d %H:%M:%S",
            "%b %d, %Y · %I:%M %p UTC",
            "%b %d, %Y, %I:%M %p UTC",
        ):
            cleaned = text.replace("·", "·")
            try:
                parsed = datetime.strptime(cleaned, fmt)
                break
            except ValueError:
                parsed = None
    if parsed is None:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _iso(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _date(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).date().isoformat()


def _search_term(query: str, start: datetime | None, end: datetime | None) -> str:
    """Quote a multi-word keyword and append since/until when dates are provided."""
    text = query.strip()
    if len(text) >= 2 and text.startswith('"') and text.endswith('"'):
        quoted = text
    elif " " in text:
        quoted = f'"{text}"'
    else:
        quoted = text
    lowered = quoted.lower()
    if start is not None and "since:" not in lowered:
        quoted = f"{quoted} since:{_date(start)}"
    if end is not None and "until:" not in lowered:
        quoted = f"{quoted} until:{_date(end)}"
    return quoted


def _endpoint(api_url: str | None) -> str:
    configured = env("X_SCRAPER_API_URL") if api_url is None else api_url
    return configured.strip() or APIFY_RUN_URL
