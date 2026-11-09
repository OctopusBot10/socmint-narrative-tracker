"""Collect public Reddit and YouTube posts for a keyword and date range.

Reddit uses the public search JSON endpoint by default. If `REDDIT_CLIENT_ID`
and `REDDIT_CLIENT_SECRET` are both set, the official PRAW client is used
instead. YouTube uses the official Data API v3 and requires an API key.

Requests stay on the public, documented surfaces. There is no proxy rotation,
no cookie harvesting, and no attempt to recover deleted or private content.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Iterable

import requests

from config import (
    DEFAULT_USER_AGENT,
    MAX_REDDIT_PAGES,
    MAX_YOUTUBE_RESULTS,
    REDDIT_MIN_INTERVAL_SECONDS,
    REQUEST_TIMEOUT_SECONDS,
    env,
)
from models import Post

REDDIT_SEARCH_URL = "https://www.reddit.com/search.json"
YOUTUBE_SEARCH_URL = "https://www.googleapis.com/youtube/v3/search"
YOUTUBE_VIDEOS_URL = "https://www.googleapis.com/youtube/v3/videos"


class CollectorError(RuntimeError):
    """A source could not be read. The message is safe to show in the UI."""


class MissingApiKeyError(CollectorError):
    """YouTube was requested without an API key."""


class RateLimitError(CollectorError):
    """The source asked the client to slow down, and the retry budget ran out."""


def collect_posts(
    keyword: str,
    start: datetime,
    end: datetime,
    *,
    platforms: Iterable[str],
    limit_per_platform: int,
    youtube_api_key: str = "",
) -> list[Post]:
    """Fetch each selected platform. Failures name the source and stop there."""
    selected = {platform.lower().strip() for platform in platforms}
    posts: list[Post] = []
    if "reddit" in selected:
        posts.extend(
            fetch_reddit(keyword, start, end, limit=limit_per_platform)
        )
    if "youtube" in selected:
        posts.extend(
            fetch_youtube(
                keyword,
                start,
                end,
                limit=limit_per_platform,
                api_key=youtube_api_key or env("YOUTUBE_API_KEY"),
            )
        )
    posts.sort(key=lambda post: post.created_at)
    return posts


def fetch_reddit(
    keyword: str,
    start: datetime,
    end: datetime,
    *,
    limit: int,
) -> list[Post]:
    client_id = env("REDDIT_CLIENT_ID")
    client_secret = env("REDDIT_CLIENT_SECRET")
    if client_id and client_secret:
        return _fetch_reddit_praw(keyword, start, end, limit=limit)
    return _fetch_reddit_json(keyword, start, end, limit=limit)


def fetch_youtube(
    keyword: str,
    start: datetime,
    end: datetime,
    *,
    limit: int,
    api_key: str,
) -> list[Post]:
    if not api_key:
        raise MissingApiKeyError(
            "YouTube Data API key is missing. Add YOUTUBE_API_KEY to .env "
            "or paste a key in the sidebar. Create one in Google Cloud Console "
            "by enabling YouTube Data API v3."
        )

    limit = max(1, min(limit, MAX_YOUTUBE_RESULTS))
    params = {
        "part": "snippet",
        "q": keyword,
        "type": "video",
        "maxResults": limit,
        "order": "date",
        "publishedAfter": _rfc3339(start),
        "publishedBefore": _rfc3339(end),
        "key": api_key,
        "safeSearch": "moderate",
        "relevanceLanguage": "en",
    }
    payload = _get_json(YOUTUBE_SEARCH_URL, params=params, source="YouTube")
    items = payload.get("items", [])
    video_ids = [
        item.get("id", {}).get("videoId", "")
        for item in items
        if item.get("id", {}).get("videoId")
    ]
    statistics = _youtube_statistics(video_ids, api_key)

    posts: list[Post] = []
    for item in items:
        video_id = item.get("id", {}).get("videoId", "")
        snippet = item.get("snippet", {})
        if not video_id or not snippet:
            continue
        published = _parse_datetime(snippet.get("publishedAt", ""))
        if published is None or published < start or published > end:
            continue
        stats = statistics.get(video_id, {})
        title = snippet.get("title", "") or ""
        description = snippet.get("description", "") or ""
        thumbnails = snippet.get("thumbnails", {}) or {}
        media = [
            thumb.get("url")
            for thumb in (
                thumbnails.get("high"),
                thumbnails.get("medium"),
                thumbnails.get("default"),
            )
            if isinstance(thumb, dict) and thumb.get("url")
        ][:1]
        watch_url = f"https://www.youtube.com/watch?v={video_id}"
        posts.append(
            Post(
                post_id=f"youtube-{video_id}",
                platform="youtube",
                author=snippet.get("channelTitle", "") or "unknown",
                community=snippet.get("channelTitle", "") or "",
                title=_clean_text(title),
                text=_clean_text(description),
                url=watch_url,
                created_at=published,
                media_links=media,
                outbound_urls=_extract_urls(description, exclude={watch_url}),
                ups=_as_int(stats.get("likeCount")),
                views=_as_int(stats.get("viewCount")),
                comments=_as_int(stats.get("commentCount")),
            )
        )
    return posts


def _fetch_reddit_json(
    keyword: str,
    start: datetime,
    end: datetime,
    *,
    limit: int,
) -> list[Post]:
    headers = {
        "User-Agent": env("REDDIT_USER_AGENT", DEFAULT_USER_AGENT),
        "Accept": "application/json",
    }
    session = requests.Session()
    posts: list[Post] = []
    after = None
    pages = 0
    older_than_window = 0

    while len(posts) < limit and pages < MAX_REDDIT_PAGES:
        params = {
            "q": keyword,
            "sort": "new",
            "type": "link",
            "t": "all",
            "limit": min(100, limit),
            "restrict_sr": "false",
        }
        if after:
            params["after"] = after
        payload = _get_json(
            REDDIT_SEARCH_URL,
            params=params,
            headers=headers,
            session=session,
            source="Reddit",
        )
        data = payload.get("data", {})
        children = data.get("children", [])
        if not children:
            break

        for child in children:
            raw = child.get("data", {})
            created = datetime.fromtimestamp(
                float(raw.get("created_utc", 0)), tz=timezone.utc
            )
            if created > end:
                continue
            if created < start:
                older_than_window += 1
                continue
            post = _reddit_post_from_data(raw, created)
            if post is not None:
                posts.append(post)
                if len(posts) >= limit:
                    break

        # `sort=new` is newest-first. After a full page older than the window,
        # later pages are older still.
        if older_than_window >= len(children) and children:
            break
        after = data.get("after")
        pages += 1
        if not after:
            break
        time.sleep(REDDIT_MIN_INTERVAL_SECONDS)

    return posts[:limit]


def _fetch_reddit_praw(
    keyword: str,
    start: datetime,
    end: datetime,
    *,
    limit: int,
) -> list[Post]:
    try:
        import praw
    except ImportError as exc:
        raise CollectorError(
            "PRAW is not installed. Run pip install -r requirements.txt, "
            "or clear REDDIT_CLIENT_ID to use public JSON search."
        ) from exc

    reddit = praw.Reddit(
        client_id=env("REDDIT_CLIENT_ID"),
        client_secret=env("REDDIT_CLIENT_SECRET"),
        user_agent=env("REDDIT_USER_AGENT", DEFAULT_USER_AGENT),
        check_for_async=False,
    )
    posts: list[Post] = []
    try:
        results = reddit.subreddit("all").search(
            keyword,
            sort="new",
            time_filter="all",
            limit=max(limit * 2, limit),
        )
        for submission in results:
            created = datetime.fromtimestamp(
                float(submission.created_utc), tz=timezone.utc
            )
            if created > end:
                continue
            if created < start:
                break
            raw = {
                "id": submission.id,
                "title": submission.title or "",
                "selftext": submission.selftext or "",
                "author": str(submission.author) if submission.author else "[deleted]",
                "subreddit": str(submission.subreddit),
                "permalink": submission.permalink,
                "url": submission.url or "",
                "score": submission.score,
                "num_comments": submission.num_comments,
                "over_18": submission.over_18,
            }
            post = _reddit_post_from_data(raw, created)
            if post is not None:
                posts.append(post)
            if len(posts) >= limit:
                break
    except Exception as exc:
        raise CollectorError(f"Reddit PRAW request failed: {exc}") from exc
    return posts


def _reddit_post_from_data(raw: dict, created: datetime) -> Post | None:
    if raw.get("over_18"):
        return None
    permalink = raw.get("permalink") or ""
    post_url = f"https://www.reddit.com{permalink}" if permalink else ""
    external = raw.get("url") or ""
    outbound: list[str] = []
    if external and "reddit.com" not in external and external.startswith("http"):
        outbound.append(external)
    body = raw.get("selftext") or ""
    outbound.extend(_extract_urls(body, exclude=set(outbound)))
    media: list[str] = []
    if external.startswith("http") and any(
        external.lower().split("?")[0].endswith(ext)
        for ext in (".jpg", ".jpeg", ".png", ".gif", ".webp")
    ):
        media.append(external)
    return Post(
        post_id=f"reddit-{raw.get('id', '')}",
        platform="reddit",
        author=str(raw.get("author") or "[deleted]"),
        community=str(raw.get("subreddit") or ""),
        title=_clean_text(str(raw.get("title") or "")),
        text=_clean_text(str(body)),
        url=post_url or external,
        created_at=created,
        media_links=media[:3],
        outbound_urls=outbound[:5],
        ups=_as_int(raw.get("score")),
        views=None,
        comments=_as_int(raw.get("num_comments")),
    )


def _youtube_statistics(video_ids: list[str], api_key: str) -> dict[str, dict]:
    if not video_ids:
        return {}
    params = {
        "part": "statistics",
        "id": ",".join(video_ids),
        "key": api_key,
        "maxResults": len(video_ids),
    }
    payload = _get_json(YOUTUBE_VIDEOS_URL, params=params, source="YouTube")
    stats: dict[str, dict] = {}
    for item in payload.get("items", []):
        stats[item.get("id", "")] = item.get("statistics", {}) or {}
    return stats


def _get_json(
    url: str,
    *,
    params: dict,
    source: str,
    headers: dict | None = None,
    session: requests.Session | None = None,
) -> dict:
    client = session or requests.Session()
    last_error = f"{source} request failed."
    for attempt in range(3):
        try:
            response = client.get(
                url,
                params=params,
                headers=headers,
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
        except requests.RequestException as exc:
            raise CollectorError(f"{source} could not be reached: {exc}") from exc

        if response.status_code == 429:
            retry_after = response.headers.get("Retry-After", "")
            try:
                wait = min(float(retry_after), 8.0)
            except ValueError:
                wait = float(2 ** attempt)
            last_error = (
                f"{source} rate-limited this client (HTTP 429). "
                "Wait a minute and run again. Do not send faster requests."
            )
            if attempt < 2:
                time.sleep(max(wait, 1.0))
                continue
            raise RateLimitError(last_error)

        if response.status_code in {401, 403}:
            detail = _api_error_message(response)
            raise CollectorError(
                f"{source} refused the request (HTTP {response.status_code}). {detail}"
            )
        if response.status_code >= 400:
            detail = _api_error_message(response)
            raise CollectorError(
                f"{source} returned HTTP {response.status_code}. {detail}"
            )
        try:
            return response.json()
        except ValueError as exc:
            raise CollectorError(
                f"{source} returned a non-JSON response. Public Reddit sometimes "
                "does this when it blocks datacenter addresses. Use demo mode, "
                "or add Reddit API credentials and try PRAW."
            ) from exc
    raise CollectorError(last_error)


def _api_error_message(response: requests.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        text = " ".join(response.text.split())
        return text[:240]
    if isinstance(payload, dict):
        error = payload.get("error", payload)
        if isinstance(error, dict) and error.get("message"):
            return str(error["message"])[:300]
        if payload.get("message"):
            return str(payload["message"])[:300]
    return ""


def _extract_urls(text: str, exclude: set[str] | None = None) -> list[str]:
    exclude = exclude or set()
    found: list[str] = []
    token = ""
    for chunk in (text or "").replace("\n", " ").split(" "):
        token = chunk.strip("()[]<>.,\"'")
        if token.startswith("http://") or token.startswith("https://"):
            if token not in exclude and token not in found:
                found.append(token)
    return found


def _clean_text(value: str) -> str:
    return " ".join(value.replace("\u0000", " ").split())[:5000]


def _as_int(value: object) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _parse_datetime(value: str) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _rfc3339(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
