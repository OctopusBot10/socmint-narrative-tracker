"""Public Telegram preview collector. No Telegram account and no Telegram API key.

Reads https://t.me/s/{channel} only. It does not log in, join chats, or follow
invite links. A run accepts at most 20 channels and waits 2–4 seconds between
preview requests.

New message ids are appended to a Google Sheet when GOOGLE_SHEETS_ID is set.
Otherwise they are appended to results/telegram_public.csv. Column C (post_id)
is the dedup key.

Scheduled use, every two hours via GitHub Actions, or locally:

    python collector_telegram.py channel_one channel_two
"""

from __future__ import annotations

import csv
import json
import random
import re
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests
import urllib3
from bs4 import BeautifulSoup
from urllib3.exceptions import InsecureRequestWarning

urllib3.disable_warnings(InsecureRequestWarning)

from config import (
    MAX_PUBLIC_CHANNELS,
    MAX_TELEGRAM_PAGES,
    REQUEST_TIMEOUT_SECONDS,
    RESULTS_DIR,
    env,
)
from models import Post, utc_iso

PREVIEW_URL = "https://t.me/s/{channel}"
PUBLIC_URL = "https://t.me/{channel}/{message_id}"
JSON_NAME = "telegram_public.json"
CSV_NAME = "telegram_public.csv"
SHEET_HEADERS = (
    "timestamp_utc",
    "channel_handle",
    "post_id",
    "post_text",
    "links_extracted",
    "view_count",
    "post_url",
)

# One stable browser-like header set for the public preview page.
PREVIEW_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml",
    "Accept-Language": "en-US,en;q=0.5",
    "Referer": "https://t.me/",
}

_URL_RE = re.compile(r"https?://[^\s<>()\"']+", re.IGNORECASE)
_INTERNAL_HOST = re.compile(r"https?://(?:t\.me|telegram\.me)(?:/|$)", re.IGNORECASE)

last_append_count = 0
last_store_label = f"results/{CSV_NAME}"
last_warning = ""
last_offline = False
last_waf = False
DEFAULT_CHANNEL = "telegram"
DEFAULT_CHANNELS = ("telegram", "durov")

_CONNECTION_DROPS = (
    ConnectionRefusedError,
    TimeoutError,
    requests.Timeout,
    requests.ConnectionError,
)


def pause_between_fetches() -> None:
    """Wait 2–4 seconds between preview requests."""
    time.sleep(random.uniform(2.0, 4.0))


def collect_channels(
    channels: list[str],
    start: datetime,
    end: datetime,
    *,
    limit: int,
    pause=pause_between_fetches,
) -> list[Post]:
    """Fetch each public channel, pausing between channels.

    A refused connection or timeout on one channel is recorded on
    ``last_warning`` and the remaining channels are still fetched.
    """
    global last_warning, last_offline
    from scrapers import CollectorError

    if len(channels) > MAX_PUBLIC_CHANNELS:
        raise CollectorError(
            f"A run accepts at most {MAX_PUBLIC_CHANNELS} public channels. "
            "Split the list so one batch does not trip the preview rate limit."
        )
    posts: list[Post] = []
    drops: list[str] = []
    for index, channel in enumerate(channels):
        if index:
            pause()
        try:
            posts.extend(collect_channel(channel, start, end, limit=limit, pause=pause))
        except CollectorError as exc:
            if not _connection_drop(exc):
                raise
            drops.append(f"@{channel}: {exc}")
    if drops:
        last_offline = not posts
        last_warning = (
            "A Telegram preview connection was refused or timed out. "
            + " ".join(drops)
        )
    posts.sort(key=lambda post: post.created_at)
    return posts


def _connection_drop(exc: BaseException) -> bool:
    """True when this error, or the error that caused it, is a dropped connection."""
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, _CONNECTION_DROPS):
            return True
        current = current.__cause__ or current.__context__
    return False


def to_record(post: Post) -> dict:
    """Uniform collector row shared with the Twitter collector."""
    views = int(post.views or 0)
    return {
        "id": post.post_id,
        "platform": "telegram",
        "author": post.author or post.community,
        "text": post.text or "",
        "timestamp": utc_iso(post.created_at),
        "url": post.url,
        "photos": list(post.photos),
        "videos": list(post.videos),
        "engagement": {
            "likes": int(post.likes or 0),
            "views": views,
            "shares": int(post.retweets or 0),
            "retweets": int(post.retweets or 0),
            "replies": int(post.replies or 0),
        },
        "account_created": utc_iso(post.account_created) if post.account_created else "",
        "profile_image": post.profile_image or "",
        "amplified_authors": list(post.amplified_authors),
        "outbound_urls": list(post.outbound_urls),
    }


def clean_keyword(value: str) -> str:
    """Drop commas, extra spaces, and wrapping quotes from a channel or keyword."""
    text = str(value or "").replace(",", " ")
    text = text.replace('"', " ").replace("'", " ")
    return re.sub(r"\s+", " ", text).strip()


def collect(
    channels: list[str],
    start: datetime | None = None,
    end: datetime | None = None,
    *,
    limit: int | None = None,
    max_posts: int | None = None,
    start_date: datetime | None = None,
    end_date: datetime | None = None,
    pause=pause_between_fetches,
    store: bool = True,
    query: str = "",
    threat_focus: bool = False,
) -> list[dict]:
    """Return public preview rows and, by default, append new ids to the store.

    ``max_posts`` is the same cap the sidebar sends as ``limit``. Crisis matching
    happens after collection, not in this request.
    """
    from scrapers import CollectorError

    window_start = start if start is not None else start_date
    window_end = end if end is not None else end_date
    requested = max_posts if max_posts is not None else limit
    if window_start is None or window_end is None or requested is None:
        raise CollectorError("A start date, end date, and max post count are required.")
    global last_warning, last_offline, last_waf
    last_warning = ""
    last_offline = False
    last_waf = False
    channels = [name for name in (clean_keyword(channel).lstrip("@") for channel in channels) if name]
    if not channels:
        channels = list(DEFAULT_CHANNELS)
    try:
        posts = collect_channels(
            channels, window_start, window_end, limit=max(1, int(requested)), pause=pause
        )
    except CollectorError as exc:
        if not (_connection_drop(exc) or "No public Telegram posts" in str(exc)):
            raise
        posts = []
        last_warning = str(exc)
    except (ConnectionRefusedError, TimeoutError, requests.Timeout, requests.ConnectionError) as exc:
        posts = []
        last_warning = f"Telegram preview refused the connection or timed out: {exc}"
    if not posts:
        last_offline = True
        if not last_warning:
            last_warning = "Telegram returned no messages for the selected channel."
        del query, threat_focus
        return []
    if store:
        append_new_posts(posts)
    rows = [to_record(post) for post in posts]
    del query, threat_focus
    return rows


def demo_records(
    channels: list[str],
    query: str,
    start: datetime,
    end: datetime,
    *,
    limit: int,
) -> list[dict]:
    """Synthetic public-channel posts. These are not fetched from t.me."""
    topic = clean_keyword(query) or "the topic"
    channel = channels[0] if channels else "acme_notice"
    when = start + (end - start) * 0.45
    text = (
        f"Customers are posting that {topic} will freeze withdrawals tomorrow. "
        "Share the notice before the branch opens."
    )
    post = Post(
        post_id="tg_demo_1",
        platform="telegram",
        author=channel,
        community=channel,
        title=text[:140],
        text=text,
        url=f"https://t.me/{channel}/1001",
        created_at=when,
        photos=["https://picsum.photos/id/1015/320/180"],
        videos=[],
        media_links=["https://picsum.photos/id/1015/320/180"],
        outbound_urls=["https://example.com/acme-freeze-notice"],
        views=900,
        likes=0,
        retweets=0,
        replies=0,
        profile_image="https://picsum.photos/id/1011/64/64",
    )
    return [to_record(post)][: max(1, limit)]


def collect_channel(
    channel: str,
    start: datetime,
    end: datetime,
    *,
    limit: int,
    pause=pause_between_fetches,
) -> list[Post]:
    """Page one public preview until the date window, the limit, or the page cap."""
    from scrapers import CollectorError, normalize_handle

    channel = normalize_handle("telegram", channel)
    limit = max(1, limit)
    posts: list[Post] = []
    seen_ids: set[str] = set()
    before: str | None = None
    pages = 0

    while len(posts) < limit and pages < MAX_TELEGRAM_PAGES:
        if pages:
            pause()
        url = PREVIEW_URL.format(channel=channel)
        if before:
            url = f"{url}?before={before}"
        html = fetch_preview(url)
        messages = parse_preview(html, channel)
        if not messages:
            if pages == 0:
                raise CollectorError(
                    f"No public Telegram posts were visible for @{channel}. "
                    "The preview only includes public channels. Private chats "
                    "and invite links are not requested."
                )
            break

        oldest_id = min(int(item["message_id"]) for item in messages)
        if before is not None and str(oldest_id) == before:
            break

        older_than_window = 0
        for item in messages:
            if item["post_id"] in seen_ids:
                continue
            created = item["created_at"]
            if created > end:
                continue
            if created < start:
                older_than_window += 1
                continue
            seen_ids.add(item["post_id"])
            posts.append(_to_post(channel, item))
            if len(posts) >= limit:
                break

        pages += 1
        if older_than_window:
            break
        before = str(oldest_id)

    return posts[:limit]


def fetch_preview(url: str) -> str:
    """GET one public preview page. A second 429 stops the run."""
    from scrapers import CollectorError, RateLimitError, get_public_preview

    last_error = "Telegram request failed."
    for attempt in range(2):
        try:
            response = get_public_preview(
                url,
                headers=PREVIEW_HEADERS,
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
        except _CONNECTION_DROPS as exc:
            message = f"Telegram preview refused the connection or timed out for {url}: {exc}"
            print(message, file=sys.stderr)
            raise CollectorError(message) from exc
        except requests.RequestException as exc:
            message = f"Telegram preview request failed for {url}: {exc}"
            print(message, file=sys.stderr)
            raise CollectorError(message) from exc

        if response.status_code != 200 and response.status_code != 429:
            message = (
                f"Telegram preview request failed for {url}: "
                f"HTTP {response.status_code}."
            )
            print(message, file=sys.stderr)
            raise CollectorError(message)
        if response.status_code == 429:
            last_error = (
                "Telegram rate-limited this client (HTTP 429). "
                "Wait before running again. Do not send faster requests."
            )
            if attempt == 0:
                pause_between_fetches()
                continue
            print(last_error, file=sys.stderr)
            raise RateLimitError(last_error)
        content_type = response.headers.get("Content-Type", "")
        if "html" not in content_type.lower() and "<html" not in response.text[:300].lower():
            message = f"Telegram preview request failed for {url}: response was not HTML."
            print(message, file=sys.stderr)
            raise CollectorError(message)
        return response.text
    print(last_error, file=sys.stderr)
    raise CollectorError(last_error)


def parse_preview(html: str, channel: str) -> list[dict]:
    """Read tgme_widget_message nodes into normalized public-post dicts."""
    soup = BeautifulSoup(html, "html.parser")
    messages: list[dict] = []
    for node in soup.select("div.tgme_widget_message"):
        message_id = _message_id(node.get("data-post") or "", channel)
        if not message_id:
            continue
        time_node = node.select_one("time[datetime]")
        created = _parse_datetime(time_node.get("datetime", "") if time_node else "")
        if created is None:
            continue
        text_node = node.select_one("div.tgme_widget_message_text")
        raw_text = text_node.get_text(" ", strip=True) if text_node else ""
        owner = node.select_one(".tgme_widget_message_owner_name")
        views_node = node.select_one(".tgme_widget_message_views")
        views = parse_views(views_node.get_text(" ", strip=True) if views_node else "")
        handle = _channel_handle(owner, channel)
        avatar = node.select_one(".tgme_widget_message_user_photo img[src]")
        profile_image = str(avatar.get("src") or "").strip() if avatar else ""
        forwarded = node.select_one(".tgme_widget_message_forwarded_from_name")
        forwarded_name = forwarded.get_text(" ", strip=True) if forwarded else ""
        messages.append(
            {
                "post_id": f"tg_{channel}_{message_id}",
                "message_id": message_id,
                "created_at": created,
                "text": _clean_text(raw_text),
                "author": handle,
                "views": views,
                "photos": _photo_urls(node),
                "videos": _video_urls(node),
                "url": PUBLIC_URL.format(channel=channel, message_id=message_id),
                "outbound_urls": _outgoing_links(text_node, raw_text),
                "profile_image": profile_image if profile_image.startswith("http") else "",
                "amplified_authors": [forwarded_name] if forwarded_name else [],
            }
        )
    return messages


def parse_public_channel(username: str) -> list[dict]:
    """Fetch the first public preview page and return normalized messages."""
    from scrapers import CollectorError, normalize_handle

    channel = normalize_handle("telegram", username)
    html = fetch_preview(PREVIEW_URL.format(channel=channel))
    return parse_preview(html, channel)


def parse_views(value: str) -> int | None:
    """Turn preview counts such as '1.2k' or '1.2K' into integers."""
    raw = value.strip().replace(",", "").replace(" ", "")
    if not raw:
        return 0
    multiplier = 1.0
    suffix = raw[-1:].lower()
    if suffix == "k":
        multiplier = 1_000.0
        raw = raw[:-1]
    elif suffix == "m":
        multiplier = 1_000_000.0
        raw = raw[:-1]
    try:
        return int(round(float(raw) * multiplier))
    except ValueError:
        return 0


def append_new_posts(
    posts: list[Post],
    directory: Path = RESULTS_DIR,
    sheet=None,
) -> list[Post]:
    """Append posts whose ids are not already stored.

    A configured Google Sheet is the dedup source: column C is `post_id`.
    With no sheet id, the local CSV/JSON store is used instead.
    """
    global last_append_count, last_store_label
    target = sheet if sheet is not None else _open_sheet_if_configured()
    if target is not None:
        fresh = _append_sheet(target, posts)
        last_store_label = "the configured Google Sheet"
    else:
        fresh = _append_local(posts, directory)
        last_store_label = f"results/{CSV_NAME}"
    last_append_count = len(fresh)
    return fresh


def _read_records(path: Path) -> list[dict]:
    from scrapers import CollectorError

    if not path.is_file():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CollectorError(f"Could not read {path.name}: {exc}") from exc
    if not isinstance(payload, list):
        raise CollectorError(f"{path.name} is not a list of posts.")
    return [item for item in payload if isinstance(item, dict)]


def _append_local(posts: list[Post], directory: Path) -> list[Post]:
    directory.mkdir(parents=True, exist_ok=True)
    json_path = directory / JSON_NAME
    records = _read_records(json_path)
    seen = {str(record.get("post_id", "")) for record in records}
    fresh = [post for post in posts if post.post_id not in seen]
    if fresh:
        records.extend(_record(post) for post in fresh)
        json_path.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
        _write_csv(directory / CSV_NAME, records)
    return fresh


def _append_sheet(worksheet, posts: list[Post]) -> list[Post]:
    from scrapers import CollectorError

    header = worksheet.row_values(1)
    if not header:
        worksheet.append_row(list(SHEET_HEADERS), value_input_option="RAW")
    elif tuple(header[: len(SHEET_HEADERS)]) != SHEET_HEADERS:
        raise CollectorError(
            "The Google Sheet header must be timestamp_utc, channel_handle, "
            "post_id, post_text, links_extracted, view_count, post_url."
        )
    existing = worksheet.col_values(3)
    seen = {value for value in existing[1:] if value}
    fresh = [post for post in posts if post.post_id not in seen]
    if fresh:
        worksheet.append_rows(
            [_sheet_row(post) for post in fresh],
            value_input_option="RAW",
        )
    return fresh


def _open_sheet_if_configured():
    sheet_id = env("GOOGLE_SHEETS_ID")
    if not sheet_id:
        return None
    from scrapers import CollectorError

    try:
        import gspread
    except ImportError as exc:
        raise CollectorError(
            "gspread is not installed. Run pip install -r requirements.txt."
        ) from exc
    raw = env("GOOGLE_SERVICE_ACCOUNT_JSON")
    path = env("GOOGLE_APPLICATION_CREDENTIALS")
    try:
        if raw:
            client = gspread.service_account_from_dict(json.loads(raw))
        elif path:
            client = gspread.service_account(filename=path)
        else:
            raise CollectorError(
                "GOOGLE_SHEETS_ID is set, but the service account is missing. "
                "Set GOOGLE_APPLICATION_CREDENTIALS or GOOGLE_SERVICE_ACCOUNT_JSON, "
                "and share the sheet with that account as an editor."
            )
        return client.open_by_key(sheet_id).sheet1
    except CollectorError:
        raise
    except Exception as exc:
        raise CollectorError(f"Could not open the Google Sheet: {exc}") from exc


def _write_csv(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SHEET_HEADERS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def _record(post: Post) -> dict:
    return dict(zip(SHEET_HEADERS, _sheet_row(post)))


def _sheet_row(post: Post) -> list:
    return [
        _sheet_timestamp(post.created_at),
        post.community,
        post.post_id,
        post.text,
        ", ".join(post.outbound_urls),
        0 if post.views is None else post.views,
        post.url,
    ]


def _sheet_timestamp(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _to_post(channel: str, item: dict) -> Post:
    text = item["text"]
    title = text.split(". ")[0][:140] if text else f"Telegram post {item['message_id']}"
    return Post(
        post_id=item["post_id"],
        platform="telegram",
        author=item["author"],
        community=channel,
        title=title,
        text=text,
        url=item["url"],
        created_at=item["created_at"],
        media_links=[*item.get("photos", []), *item.get("videos", [])],
        outbound_urls=item["outbound_urls"],
        photos=list(item.get("photos") or []),
        videos=list(item.get("videos") or []),
        ups=None,
        views=item["views"],
        likes=0,
        retweets=0,
        replies=0,
        comments=None,
        profile_image=str(item.get("profile_image") or ""),
        amplified_authors=list(item.get("amplified_authors") or []),
    )


_BACKGROUND_URL = re.compile(r"url\((['\"]?)(https?://[^)'\"]+)\1\)", re.IGNORECASE)
_MEDIA_FILE = re.compile(r"\.(mp4|webm|mov|m4v)(\?|$)", re.IGNORECASE)


def _channel_handle(owner, channel: str) -> str:
    """Prefer the public t.me username over the display name."""
    if owner is not None:
        link = owner.find("a", href=True)
        href = str(link.get("href") or "") if link else ""
        match = re.search(r"t\.me/([A-Za-z0-9_]{4,})", href)
        if match:
            return match.group(1)
    return channel


def _style_urls(tag) -> list[str]:
    style = str(tag.get("style") or "")
    return [match.group(2) for match in _BACKGROUND_URL.finditer(style)]


def _photo_urls(node) -> list[str]:
    found: list[str] = []
    for wrap in node.select(".tgme_widget_message_photo_wrap"):
        for image in wrap.select("img[src]"):
            src = str(image.get("src") or "").strip()
            if src.startswith("http"):
                found.append(src)
        found.extend(_style_urls(wrap))
        for styled in wrap.select("[style]"):
            found.extend(_style_urls(styled))
    return _unique(found)


def _video_urls(node) -> list[str]:
    found: list[str] = []
    for player in node.select(".tgme_widget_message_video_player"):
        for tag in player.select("video[src], source[src]"):
            src = str(tag.get("src") or "").strip()
            if src.startswith("http"):
                found.append(src)
        href = str(player.get("href") or "").strip()
        if href.startswith("http") and _MEDIA_FILE.search(href):
            found.append(href)
    return _unique(found)


def _message_id(data_post: str, channel: str) -> str:
    channel_name, separator, message_id = data_post.strip().partition("/")
    if not separator or channel_name.lower() != channel.lower():
        return ""
    message_id = message_id.strip()
    if not message_id.isdigit():
        return ""
    return message_id


def _outgoing_links(node, raw_text: str) -> list[str]:
    """External hyperlinks. t.me and telegram.me targets are not outgoing."""
    found: list[str] = []
    if node is not None:
        for anchor in node.find_all("a", href=True):
            href = str(anchor.get("href") or "").strip()
            if href.startswith("http") and not _is_internal(href):
                found.append(href)
    for match in _URL_RE.findall(raw_text or ""):
        url = match.rstrip(".,);]")
        if url.startswith("http") and not _is_internal(url):
            found.append(url)
    return _unique(found)


def _is_internal(url: str) -> bool:
    return bool(_INTERNAL_HOST.match(url))


def _unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            ordered.append(value)
    return ordered


def _clean_text(value: str) -> str:
    return " ".join(value.replace("\u0000", " ").split())[:5000]


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


def _cli(argv: list[str]) -> int:
    from scrapers import CollectorError

    channels = [arg for arg in argv if not arg.startswith("-")]
    if not channels:
        channels = [
            part.strip().lstrip("@")
            for part in re.split(r"[\s,]+", env("TELEGRAM_CHANNELS"))
            if part.strip()
        ]
    if not channels:
        print(
            "Usage: python collector_telegram.py channel_username [more_channels]",
            file=sys.stderr,
        )
        return 2
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=14)
    try:
        posts = collect_channels(channels, start, end, limit=40)
        fresh = append_new_posts(posts)
    except CollectorError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(
        f"Fetched {len(posts)} public posts in the last 14 days. "
        f"Appended {len(fresh)} new rows to {last_store_label}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli(sys.argv[1:]))
