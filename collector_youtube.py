"""Public YouTube search collector. No API key and no Google account.

Reads the public results page and the ytInitialData block embedded in the
HTML. Titles, channels, thumbnails, and view counts come from that block.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timedelta, timezone
from urllib.parse import urlencode

import requests

from scrapers import CollectorError

SEARCH_URL = "https://www.youtube.com/results"
BROWSER_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/122.0.0.0 Safari/537.36"
)
last_warning = ""
last_offline = False
last_waf = False
_RELATIVE = re.compile(r"(\d+)\s+(minute|hour|day|week|month|year)s?\s+ago", re.I)


def clean_keyword(value: str) -> str:
    """Drop commas, extra spaces, and wrapping quotes. No Boolean operators."""
    text = str(value or "").replace(",", " ")
    text = text.replace('"', " ").replace("'", " ")
    return re.sub(r"\s+", " ", text).strip()


def collect(
    query: str,
    start: datetime | None = None,
    end: datetime | None = None,
    *,
    limit: int | None = None,
    max_posts: int | None = None,
    start_date: datetime | None = None,
    end_date: datetime | None = None,
) -> list[dict]:
    """Search public YouTube results for a cleaned ``query``.

    The results HTML is parsed for ``ytInitialData`` before any fallback.
    An empty parse returns no rows and sets ``last_waf``.
    """
    global last_warning, last_offline, last_waf
    last_warning = ""
    last_offline = False
    last_waf = False
    term = clean_keyword(query)
    if not term:
        raise CollectorError("Enter a keyword before searching YouTube.")
    window_start = start if start is not None else start_date
    window_end = end if end is not None else end_date
    if window_start is None or window_end is None:
        raise CollectorError("A start date and end date are required.")
    requested = max_posts if max_posts is not None else limit
    if requested is None:
        raise CollectorError("A max post count is required.")
    url = f"{SEARCH_URL}?{urlencode({'search_query': term})}"
    try:
        response = requests.get(
            url,
            headers={
                "User-Agent": BROWSER_USER_AGENT,
                "Accept": "text/html,application/xhtml+xml",
                "Accept-Language": "en-US,en",
            },
            timeout=25,
        )
    except requests.RequestException as exc:
        last_offline = True
        last_waf = True
        last_warning = f"YouTube search failed: {exc}"
        return []
    if response.status_code >= 400:
        last_offline = True
        last_waf = True
        last_warning = f"YouTube returned HTTP {response.status_code}."
        return []
    html = response.text or ""
    if not _initial_data(html):
        last_offline = True
        last_waf = True
        last_warning = "YouTube did not return a public results page."
        return []
    rows = parse_results(
        html,
        start=window_start,
        end=window_end,
        limit=max(1, int(requested)),
    )
    if not rows:
        last_warning = "YouTube search returned no videos in that window."
        return []
    return rows


def parse_results(html: str, *, start: datetime, end: datetime, limit: int) -> list[dict]:
    payload = _initial_data(html)
    rows: list[dict] = []
    seen: set[str] = set()
    for video in _video_renderers(payload):
        video_id = str(video.get("videoId") or "")
        if not video_id or video_id in seen:
            continue
        title = _text(video.get("title"))
        description = _text(video.get("descriptionSnippet") or video.get("detailedMetadataSnippets"))
        text = " ".join(part for part in (title, description) if part).strip()
        author = _text((video.get("ownerText") or video.get("longBylineText")))
        when = _relative_time(_text(video.get("publishedTimeText")))
        if when is not None and (when < start or when > end):
            continue
        thumbs = ((video.get("thumbnail") or {}).get("thumbnails") or [])
        photo = str(thumbs[-1].get("url") or "") if thumbs else ""
        views = _views(_text(video.get("viewCountText")))
        rows.append(
            {
                "id": video_id,
                "platform": "youtube",
                "author": author or "Unknown",
                "text": text,
                "timestamp": when.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ") if when else "",
                "url": f"https://www.youtube.com/watch?v={video_id}",
                "photos": [photo] if photo.startswith("http") else [],
                "videos": [f"https://www.youtube.com/watch?v={video_id}"],
                "engagement": {"likes": 0, "views": views, "shares": 0},
            }
        )
        seen.add(video_id)
        if len(rows) >= limit:
            break
    return rows


def demo_records(query: str, start: datetime, end: datetime, *, limit: int) -> list[dict]:
    """Synthetic public videos. These were not fetched from YouTube."""
    topic = query.strip() or "the topic"
    when = start + (end - start) * 0.5
    text = (
        f"Customers are posting that {topic} will freeze withdrawals tomorrow. "
        "Share the notice before the branch opens."
    )
    return [
        {
            "id": "yt_demo_1",
            "platform": "youtube",
            "author": "demo_channel",
            "text": text,
            "timestamp": when.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "url": "https://www.youtube.com/watch?v=yt_demo_1",
            "photos": ["https://picsum.photos/id/1016/320/180"],
            "videos": ["https://www.youtube.com/watch?v=yt_demo_1"],
            "engagement": {"likes": 12, "views": 3400, "shares": 0},
        }
    ][: max(1, limit)]


def _initial_data(html: str) -> dict:
    marker = html.find("ytInitialData")
    if marker < 0:
        return {}
    start = html.find("{", marker)
    if start < 0:
        return {}
    depth = 0
    for index, char in enumerate(html[start:], start):
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(html[start : index + 1])
                except json.JSONDecodeError:
                    return {}
    return {}


def _video_renderers(node) -> list[dict]:
    found: list[dict] = []

    def walk(value) -> None:
        if isinstance(value, dict):
            renderer = value.get("videoRenderer")
            if isinstance(renderer, dict):
                found.append(renderer)
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(node)
    return found


def _text(value) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        if "simpleText" in value:
            return str(value["simpleText"]).strip()
        runs = value.get("runs")
        if isinstance(runs, list):
            return "".join(str(part.get("text") or "") for part in runs if isinstance(part, dict)).strip()
    if isinstance(value, list) and value:
        return _text(value[0])
    return ""


def _views(text: str) -> int:
    match = re.search(r"([\d,.]+)\s*([KMB])?", text.replace(",", ""), re.I)
    if not match:
        return 0
    number = float(match.group(1).replace(",", ""))
    suffix = (match.group(2) or "").upper()
    multiplier = {"": 1, "K": 1_000, "M": 1_000_000, "B": 1_000_000_000}[suffix]
    return int(number * multiplier)


def _relative_time(text: str) -> datetime | None:
    match = _RELATIVE.search(text or "")
    if not match:
        return None
    count = int(match.group(1))
    unit = match.group(2).lower()
    span = {
        "minute": timedelta(minutes=count),
        "hour": timedelta(hours=count),
        "day": timedelta(days=count),
        "week": timedelta(weeks=count),
        "month": timedelta(days=30 * count),
        "year": timedelta(days=365 * count),
    }[unit]
    return datetime.now(timezone.utc) - span
