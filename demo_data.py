"""Synthetic posts so the dashboard can be demonstrated without a network call.

Every demo record is labeled `is_demo=True`. Links use the public URL shape of
the selected platform and the handle from the sidebar. They are constructed
sequence ids, not pages that were fetched.

The generator plants three reviewable patterns inside the selected dates:

- a withdrawal rumor on the Telegram handle
- a tight near-copy burst of that rumor from other demo accounts
- a shared Telegram link and a copied market headline (a realistic false positive)
"""

from __future__ import annotations

import hashlib
import random
from datetime import datetime, timedelta

from config import DEFAULT_HANDLES
from models import Post
from scrapers import normalize_handle, post_url


def generate_demo_posts(
    keyword: str,
    start: datetime,
    end: datetime,
    *,
    platforms: list[str] | None = None,
    handles: dict[str, str] | None = None,
) -> list[Post]:
    if end < start:
        start, end = end, start
    if end == start:
        end = start + timedelta(days=1)

    del platforms
    selected = ["telegram"]
    resolved = {
        "telegram": normalize_handle(
            "telegram", (handles or {}).get("telegram") or DEFAULT_HANDLES["telegram"]
        )
    }

    seed = int(hashlib.sha256(keyword.encode("utf-8")).hexdigest()[:8], 16)
    rng = random.Random(seed)
    topic = keyword.strip() or "the topic"
    counters = {"telegram": 12041}

    def at(fraction: float, minutes: int = 0) -> datetime:
        moment = start + (end - start) * fraction + timedelta(minutes=minutes)
        if moment < start:
            return start
        if moment > end:
            return end
        return moment

    def next_external_id(platform: str) -> str:
        value = counters["telegram"]
        counters["telegram"] += 1
        return str(value)

    posts: list[Post] = []

    def add(
        *,
        platform: str,
        author: str,
        title: str,
        text: str,
        when: datetime,
        ups: int,
        comments: int,
        views: int | None = None,
        outbound: list[str] | None = None,
    ) -> None:
        if platform not in resolved:
            return
        external_id = next_external_id(platform)
        handle = resolved[platform]
        posts.append(
            Post(
                post_id=f"demo-{platform}-{external_id}",
                platform=platform,
                author=author,
                community=handle,
                title=title,
                text=text,
                url=post_url(platform, handle, external_id),
                created_at=when,
                media_links=[],
                outbound_urls=outbound or [],
                ups=ups,
                views=views,
                comments=comments,
                is_demo=True,
            )
        )

    def platform_at(index: int) -> str:
        if "telegram" in resolved and index == 0:
            return "telegram"
        return selected[index % len(selected)]

    copy = (
        f"Multiple customers say {topic} will freeze withdrawals tomorrow morning. "
        "Move your money tonight and share this before it is deleted."
    )
    add(
        platform=platform_at(0),
        author="demo_branch_note",
        title=f"{topic} withdrawal freeze, please read",
        text=(
            f"Multiple customers say {topic} will freeze withdrawals tomorrow morning. "
            "A teller was filling out paperwork for that freeze when I asked at the branch counter this afternoon."
        ),
        when=at(0.34),
        ups=42,
        comments=11,
        views=1800,
    )
    burst_authors = (
        "demo_save_now",
        "demo_cash_out",
        "demo_tell_friends",
        "demo_share_fast",
    )
    for offset, author in enumerate(burst_authors):
        variant = copy
        if offset == 1:
            variant = copy.replace("your money", "your funds")
        elif offset == 2:
            variant = copy.replace("customers say", "customers report").replace(
                "it is deleted", "it gets deleted"
            )
        elif offset == 3:
            variant = copy.replace("it is deleted", "it is removed")
        add(
            platform=platform_at(offset + 1),
            author=author,
            title=f"{topic} withdrawal freeze, please read",
            text=variant,
            when=at(0.36, minutes=20 * offset),
            ups=rng.randint(80, 240),
            comments=rng.randint(15, 60),
            views=rng.randint(4000, 18000),
        )

    fee_template = (
        f"My statement from {topic} shows a new monthly maintenance fee on the checking account. "
        "The charge was not there last month and support called it a pricing update."
    )
    fee_lines = [
        fee_template,
        fee_template.replace("shows a new", "includes a new").replace(
            "The charge was not there", "That charge was absent"
        ),
        fee_template.replace("checking account", "checking plan").replace(
            "pricing update", "scheduled pricing update"
        ),
        fee_template.replace("last month", "the previous cycle").replace(
            "support called it", "the branch called it"
        ),
        fee_template.replace("monthly maintenance fee", "monthly maintenance charge"),
    ]
    for index, line in enumerate(fee_lines):
        add(
            platform=selected[index % len(selected)],
            author=f"demo_fee_user_{index + 1}",
            title=f"{topic} monthly maintenance fee",
            text=line,
            when=at(0.12 + index * 0.14),
            ups=rng.randint(3, 40),
            comments=rng.randint(1, 12),
            views=rng.randint(200, 2000),
        )

    outage_lines = [
        (
            f"The {topic} mobile app kicks me out at the login screen with error "
            "code 5042. Card purchases still work."
        ),
        (
            f"Login loop on the {topic} mobile app since breakfast. Error code "
            "5042, then a blank accounts page. Branch card reader is fine."
        ),
        (
            f"Anyone else stuck on the {topic} mobile app login? I get error "
            "code 5042 on wifi and on cellular."
        ),
        (
            f"Update: {topic} mobile app login still returns error code 5042 "
            "this evening. Bill pay from the website succeeded."
        ),
        (
            f"Reinstalled the {topic} mobile app and the login error code 5042 "
            "remains. Teller said the branch system is up."
        ),
    ]
    for index, line in enumerate(outage_lines):
        add(
            platform=selected[index % len(selected)],
            author=f"demo_app_user_{index + 1}",
            title=f"{topic} mobile app login error 5042",
            text=line,
            when=at(0.2 + index * 0.12),
            ups=rng.randint(5, 70),
            views=rng.randint(400, 5000),
            comments=rng.randint(2, 18),
        )

    clarify_template = (
        f"I checked today and there is no notice affecting transfers at {topic}. "
        "The circulating story is unverified and accounts appear to be operating normally."
    )
    clarify_lines = [
        clarify_template,
        clarify_template.replace("I checked today", "I looked this morning").replace(
            "appear to be operating normally", "still appear to be operating normally"
        ),
        clarify_template.replace("no notice affecting transfers", "no posted notice about transfers"),
        clarify_template.replace(
            "The circulating story is unverified",
            "That circulating story remains unverified",
        ),
    ]
    for index, line in enumerate(clarify_lines):
        add(
            platform=selected[index % len(selected)],
            author=f"demo_check_source_{index + 1}",
            title=f"Checking the circulating {topic} story",
            text=line,
            when=at(0.48 + index * 0.06),
            ups=rng.randint(20, 150),
            views=rng.randint(800, 8000),
            comments=rng.randint(4, 30),
        )

    shared = f"https://t.me/{resolved['telegram']}/11990"
    link_lines = [
        (
            "Posting a screenshot that is circulating in a neighborhood group. "
            "I cannot tell whether the page is genuine.",
            "Forwarded claim, neighborhood group",
        ),
        (
            "Can somebody qualified look at this page? It is being pasted into "
            "group chats with no context.",
            "Can someone review this page",
        ),
        (
            "Second time I have been sent this. I am not endorsing it. Leaving "
            "the link so others can check the wording.",
            "Same page, second time today",
        ),
    ]
    for index, (line, title) in enumerate(link_lines):
        add(
            platform=selected[index % len(selected)],
            author=f"demo_link_user_{index + 1}",
            title=title,
            text=line,
            when=at(0.57, minutes=15 * index),
            ups=rng.randint(1, 15),
            comments=rng.randint(0, 6),
            outbound=[shared],
        )

    wire = (
        f"Market wire: equity price for {topic} dips following online speculation. "
        "Traders cited social posts rather than a filing."
    )
    for index, author in enumerate(("demo_wire_desk", "demo_wire_mirror")):
        add(
            platform=selected[index % len(selected)],
            author=author,
            title=f"{topic} equity price dips",
            text=wire,
            when=at(0.72, minutes=12 * index),
            ups=rng.randint(2, 20),
            comments=rng.randint(0, 4),
            views=rng.randint(100, 900),
        )

    posts.sort(key=lambda post: post.created_at)
    return posts
