"""Synthetic posts so the dashboard can be demonstrated without API keys.

Every demo record is labeled `is_demo=True` and uses example.com URLs.
The generator plants three reviewable patterns inside the selected dates:

- a withdrawal rumor that starts on Reddit and is later repeated on YouTube
- a tight near-copy burst of that rumor, including cross-platform copies
- a shared-link burst and a copied market headline (a realistic false positive)
"""

from __future__ import annotations

import hashlib
import random
from datetime import datetime, timedelta

from models import Post


def generate_demo_posts(keyword: str, start: datetime, end: datetime) -> list[Post]:
    if end < start:
        start, end = end, start
    if end == start:
        end = start + timedelta(days=1)

    seed = int(hashlib.sha256(keyword.encode("utf-8")).hexdigest()[:8], 16)
    rng = random.Random(seed)
    topic = keyword.strip() or "the topic"

    def at(fraction: float, minutes: int = 0) -> datetime:
        moment = start + (end - start) * fraction + timedelta(minutes=minutes)
        if moment < start:
            return start
        if moment > end:
            return end
        return moment

    posts: list[Post] = []

    def add(
        *,
        platform: str,
        author: str,
        community: str,
        title: str,
        text: str,
        when: datetime,
        ups: int,
        comments: int,
        views: int | None = None,
        outbound: list[str] | None = None,
        media: list[str] | None = None,
    ) -> None:
        serial = len(posts) + 1
        post_id = f"demo-{platform}-{serial}"
        posts.append(
            Post(
                post_id=post_id,
                platform=platform,
                author=author,
                community=community,
                title=title,
                text=text,
                url=f"https://example.com/demo/{platform}/{post_id}",
                created_at=when,
                media_links=media or [],
                outbound_urls=outbound or [],
                ups=ups,
                views=views,
                comments=comments,
                is_demo=True,
            )
        )

    copy = (
        f"Multiple customers say {topic} will freeze withdrawals tomorrow morning. "
        "Move your money tonight and share this before it is deleted."
    )
    add(
        platform="reddit",
        author="demo_branch_note",
        community="personalfinance",
        title=f"{topic} withdrawal freeze, please read",
        text=(
            f"Multiple customers say {topic} will freeze withdrawals tomorrow morning. "
            "A teller was filling out paperwork for that freeze when I asked at the branch counter this afternoon."
        ),
        when=at(0.34),
        ups=42,
        comments=11,
    )
    reddit_authors = [
        "demo_save_now",
        "demo_cash_out",
        "demo_tell_friends",
        "demo_share_fast",
    ]
    for offset, author in enumerate(reddit_authors):
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
            platform="reddit",
            author=author,
            community="personalfinance",
            title=f"{topic} withdrawal freeze, please read",
            text=variant,
            when=at(0.36, minutes=20 * offset),
            ups=rng.randint(80, 240),
            comments=rng.randint(15, 60),
        )
    for offset, channel in enumerate(("Demo Desk North", "Demo Desk South")):
        add(
            platform="youtube",
            author=channel,
            community=channel,
            title=f"{topic} withdrawal freeze, please read",
            text=copy,
            when=at(0.36, minutes=90 + 25 * offset),
            ups=rng.randint(300, 900),
            views=rng.randint(4000, 18000),
            comments=rng.randint(40, 160),
            media=[f"https://example.com/demo/thumb-{offset + 1}.jpg"],
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
            platform="reddit",
            author=f"demo_fee_user_{index + 1}",
            community="banking",
            title=f"{topic} monthly maintenance fee",
            text=line,
            when=at(0.12 + index * 0.14),
            ups=rng.randint(3, 40),
            comments=rng.randint(1, 12),
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
            platform="reddit" if index < 4 else "youtube",
            author=f"demo_app_user_{index + 1}" if index < 4 else "Demo Phone Help",
            community="banking" if index < 4 else "Demo Phone Help",
            title=f"{topic} mobile app login error 5042",
            text=line,
            when=at(0.2 + index * 0.12),
            ups=rng.randint(5, 70),
            views=1200 if index == 4 else None,
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
            platform="reddit" if index < 3 else "youtube",
            author=f"demo_check_source_{index + 1}" if index < 3 else "Demo Verify Desk",
            community="personalfinance" if index < 3 else "Demo Verify Desk",
            title=f"Checking the circulating {topic} story",
            text=line,
            when=at(0.48 + index * 0.06),
            ups=rng.randint(20, 150),
            views=8000 if index == 3 else None,
            comments=rng.randint(4, 30),
        )

    shared = "https://example.com/demo/forwarded-claim"
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
            platform="reddit",
            author=f"demo_link_user_{index + 1}",
            community="localnews",
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
            platform="reddit",
            author=author,
            community="stocks",
            title=f"{topic} equity price dips",
            text=wire,
            when=at(0.72, minutes=12 * index),
            ups=rng.randint(2, 20),
            comments=rng.randint(0, 4),
        )

    posts.sort(key=lambda post: post.created_at)
    return posts
