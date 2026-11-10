"""Read a saved public Facebook HTML dump. No Facebook login.

The assignment path for Facebook is a local HTML file the researcher already
saved. This parser does not open facebook.com and does not use an account.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from bs4 import BeautifulSoup

from scrapers import CollectorError


def collect(
    path: str,
    query: str,
    start: datetime | None = None,
    end: datetime | None = None,
    *,
    limit: int | None = None,
    max_posts: int | None = None,
    start_date: datetime | None = None,
    end_date: datetime | None = None,
) -> list[dict]:
    """Parse posts from a local HTML dump that mention ``query``."""
    window_start = start if start is not None else start_date
    window_end = end if end is not None else end_date
    if window_start is None or window_end is None:
        raise CollectorError("A start date and end date are required.")
    requested = max_posts if max_posts is not None else limit
    if requested is None:
        raise CollectorError("A max post count is required.")
    file_path = Path(path).expanduser()
    if not file_path.is_file():
        raise CollectorError(f"Facebook HTML dump was not found: {file_path}")
    html = file_path.read_text(encoding="utf-8", errors="replace")
    return parse_dump(
        html,
        query=query,
        start=window_start,
        end=window_end,
        limit=max(1, int(requested)),
    )


def parse_dump(html: str, *, query: str, start: datetime, end: datetime, limit: int) -> list[dict]:
    soup = BeautifulSoup(html, "html.parser")
    keyword = query.strip().lower()
    nodes = soup.select("[role=article], .userContentWrapper, div.story_body_container")
    if not nodes:
        nodes = soup.select("article, div.post")
    rows: list[dict] = []
    for index, node in enumerate(nodes, start=1):
        text = node.get_text(" ", strip=True)
        if keyword and keyword not in text.lower():
            continue
        author_node = node.select_one("h2 a, h3 a, strong a, a.profileLink")
        author = author_node.get_text(" ", strip=True) if author_node else "Unknown"
        link = node.select_one("a[href*='facebook.com'], a[href*='story.php'], abbr a")
        url = str(link.get("href") or "") if link else ""
        time_node = node.select_one("time[datetime], abbr[data-utime]")
        when = _time(time_node)
        if when is not None and (when < start or when > end):
            continue
        photos = [
            str(image.get("src"))
            for image in node.select("img[src]")
            if str(image.get("src") or "").startswith("http")
        ]
        videos = [
            str(video.get("src"))
            for video in node.select("video[src], source[src]")
            if str(video.get("src") or "").startswith("http")
        ]
        rows.append(
            {
                "id": f"fb_{index}",
                "platform": "facebook",
                "author": author or "Unknown",
                "text": text[:2000],
                "timestamp": when.strftime("%Y-%m-%dT%H:%M:%SZ") if when else "",
                "url": url,
                "photos": photos[:4],
                "videos": videos[:2],
                "engagement": {"likes": 0, "views": 0, "shares": 0},
            }
        )
        if len(rows) >= limit:
            break
    return rows


def _time(node) -> datetime | None:
    if node is None:
        return None
    raw = str(node.get("datetime") or node.get("data-utime") or "").strip()
    if raw.isdigit():
        return datetime.fromtimestamp(int(raw), tz=timezone.utc)
    if not raw:
        return None
    candidate = raw[:-1] + "+00:00" if raw.endswith("Z") else raw
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)
