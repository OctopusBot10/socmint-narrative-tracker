"""Collect public Telegram channel previews.

Live collection reads https://t.me/s/{handle} through collector_telegram.py and
stores each message as https://t.me/{handle}/{message_id}. It does not log in,
join private chats, or use a Telegram API key.
"""

from __future__ import annotations

import json
import re
import sys
import threading
from datetime import datetime
from typing import Iterable

import requests
import urllib3
from urllib3.exceptions import InsecureRequestWarning
from urllib3.util import connection as urllib3_connection

from models import Post

# This machine's OpenSSL build fails the t.me handshake with
# SSLError: unknown error (_ssl.c:1000). Skip certificate checks for the
# public preview only, and hide the warning that requests emits for that.
urllib3.disable_warnings(InsecureRequestWarning)

_HANDLE_RE = re.compile(r"^[A-Za-z0-9_]{5,32}$")
_URL_PREFIXES = (
    "https://t.me/s/",
    "https://t.me/",
    "http://t.me/s/",
    "http://t.me/",
)


class CollectorError(RuntimeError):
    """A source could not be read. The message is safe to show in the UI."""


class RateLimitError(CollectorError):
    """The source asked the client to slow down, and the retry budget ran out."""


_preview_lock = threading.Lock()
_resolved_telegram_ip: str | None = None


def _telegram_public_ip() -> str | None:
    """Look up t.me on the public DNS. The system resolver on this Mac does not."""
    global _resolved_telegram_ip
    if _resolved_telegram_ip:
        return _resolved_telegram_ip
    try:
        response = requests.get(
            "https://cloudflare-dns.com/dns-query",
            params={"name": "t.me", "type": "A"},
            headers={"Accept": "application/dns-json"},
            timeout=10,
        )
        response.raise_for_status()
        payload = response.json()
    except (requests.RequestException, json.JSONDecodeError, ValueError) as exc:
        print(f"Public DNS lookup for t.me failed: {exc}", file=sys.stderr)
        return None
    for answer in payload.get("Answer", []):
        address = str(answer.get("data", ""))
        if answer.get("type") == 1 and re.fullmatch(r"\d{1,3}(?:\.\d{1,3}){3}", address):
            _resolved_telegram_ip = address
            return address
    print("Public DNS lookup for t.me returned no IPv4 address.", file=sys.stderr)
    return None


def get_public_preview(url: str, *, headers: dict, timeout: int) -> requests.Response:
    """GET a public t.me preview page without verifying the TLS certificate.

    The hostname in the URL stays t.me so the preview page is the one Telegram
    publishes. The TCP connection uses the public DNS address, because the
    address from the system resolver closes the handshake.
    """
    address = _telegram_public_ip()
    original = urllib3_connection.create_connection

    def create_connection(target, *args, **kwargs):
        host, port = target
        if address and host == "t.me":
            target = (address, port)
        return original(target, *args, **kwargs)

    with _preview_lock:
        urllib3_connection.create_connection = create_connection
        try:
            return requests.get(url, headers=headers, timeout=timeout, verify=False)
        finally:
            urllib3_connection.create_connection = original


def normalize_handle(platform: str, value: str) -> str:
    """Reduce a pasted Telegram handle or t.me link to the channel username."""
    if platform != "telegram":
        raise CollectorError("Only public Telegram channels are collected right now.")
    raw = (value or "").strip()
    if raw.startswith("@"):
        raw = raw[1:]
    lowered = raw.lower()
    for prefix in _URL_PREFIXES:
        if lowered.startswith(prefix):
            raw = raw[len(prefix):]
            break
    raw = raw.split("?")[0].split("#")[0].strip("/")
    raw = raw.split("/")[0].strip()
    if not _HANDLE_RE.fullmatch(raw):
        raise CollectorError(
            "Enter a public Telegram channel handle. A t.me link is accepted."
        )
    return raw


def post_url(platform: str, handle: str, external_id: str) -> str:
    """Public permalink for one Telegram message."""
    handle = normalize_handle(platform, handle)
    external_id = str(external_id).strip().strip("/")
    if not external_id.isdigit():
        raise CollectorError("Telegram message ids must be numeric.")
    return f"https://t.me/{handle}/{external_id}"


def collect_posts(
    start: datetime,
    end: datetime,
    *,
    platforms: Iterable[str],
    limit_per_platform: int,
    handles: dict[str, str],
) -> list[Post]:
    """Fetch public Telegram posts for the channel handle."""
    selected = {platform.lower().strip() for platform in platforms}
    if "telegram" not in selected:
        raise CollectorError(
            "Live collection reads public Telegram channels. Select Telegram, "
            "or turn demo data on."
        )
    handle = normalize_handle("telegram", handles.get("telegram", ""))
    posts = fetch_telegram(handle, start, end, limit=limit_per_platform)
    from collector_telegram import append_new_posts

    append_new_posts(posts)
    posts.sort(key=lambda post: post.created_at)
    return posts


def fetch_telegram(
    handle: str,
    start: datetime,
    end: datetime,
    *,
    limit: int,
) -> list[Post]:
    """Page the public t.me/s preview. Storage happens in collect_posts."""
    from collector_telegram import collect_channel

    return collect_channel(handle, start, end, limit=limit)
